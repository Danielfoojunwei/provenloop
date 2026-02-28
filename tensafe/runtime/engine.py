"""TenSafe Proprietary Runtime Engine.

The core execution engine that loads and runs TGSP adapters with
HE-encrypted inference.  All adapters pass through the 8-step TGSP
Load Gate before they are permitted to run.  Inference is metered
per-token for billing.  Adapters can be hot-swapped atomically
without dropping in-flight requests.

The TenSafe System Adapter (TSA) is a mandatory system-level adapter
that must be loaded before any domain adapters.  It provides:
  - Learned expert routing (replaces keyword gates)
  - System awareness (metering, privacy budget, adapter composition)
  - Throughput optimization (output length calibration)
  - Cryptographic binding (domain adapters are counter-signed by TSA key)

Architecture:
    RuntimeConfig --> TenSafeRuntime
                        |-- TGSPLoadGate   (8-step verification)
                        |-- MeteringService (per-token billing)
                        |-- TSA            (mandatory system adapter)
                        |-- Loaded adapters (thread-safe registry)
                        |-- HE engine      (CKKS encrypt/decrypt stub)
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tensafe.runtime.load_gate import (
    GateReport,
    GateVerdict,
    TGSPLoadGate,
    TGSP_HEADER_LEN,
    TGSP_MAGIC,
)
from tensafe.runtime.metering import MeteringService, UsageTier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    """Configuration for the TenSafe runtime engine."""

    # Load gate settings
    strict_moe: bool = True
    max_lora_rank: int = 256
    trusted_fingerprints: Optional[Dict[str, str]] = None

    # Metering / billing
    billing_enabled: bool = False
    stripe_api_key: str = "sk_test_stub"
    default_tier: UsageTier = UsageTier.FREE

    # Inference settings
    default_max_tokens: int = 256
    he_poly_mod_degree: int = 16384
    he_scale_bits: int = 40

    # Hot-swap settings
    hot_swap_health_check_tokens: int = 16
    hot_swap_timeout_s: float = 30.0

    # Engine limits
    max_loaded_adapters: int = 64
    inference_timeout_s: float = 120.0


# ---------------------------------------------------------------------------
# Adapter info / state
# ---------------------------------------------------------------------------

@dataclass
class AdapterInfo:
    """Public metadata for a loaded adapter."""
    adapter_id: str
    name: str
    version: str
    rank: int
    alpha: float
    target_modules: List[str]
    domain: str
    tgsp_path: str
    adapter_type: str = "domain"  # "system", "domain", or "router"
    loaded_at: float = field(default_factory=time.time)
    request_count: int = 0
    token_count: int = 0


@dataclass
class _LoadedAdapter:
    """Internal representation of a loaded adapter with its weights."""
    info: AdapterInfo
    manifest: Dict[str, Any]
    gate_report: GateReport
    # LoRA weight matrices (in production, these would be CKKS ciphertexts)
    lora_a: Optional[np.ndarray] = None  # [rank, hidden_dim]
    lora_b: Optional[np.ndarray] = None  # [hidden_dim, rank]
    # Reference count for in-flight requests
    ref_count: int = 0
    # Marked for unload after all in-flight requests complete
    draining: bool = False


# ---------------------------------------------------------------------------
# Stubbed HE engine (CKKS encrypt / decrypt)
# ---------------------------------------------------------------------------

class _CKKSStub:
    """Stubbed CKKS homomorphic encryption engine.

    In production, this delegates to tensafe-he (Rust/CUDA) for real
    CKKS operations.  The stub performs plaintext math with the same
    API shape so the orchestration logic is exercised correctly.
    """

    def __init__(self, poly_mod_degree: int = 16384, scale_bits: int = 40):
        self._n = poly_mod_degree
        self._slots = poly_mod_degree // 2
        self._scale_bits = scale_bits
        self._total_encryptions = 0
        self._total_decryptions = 0

    def encrypt(self, plaintext: np.ndarray) -> np.ndarray:
        """Encrypt a vector (stub: returns the plaintext unchanged)."""
        self._total_encryptions += 1
        return plaintext.copy().astype(np.float64)

    def decrypt(self, ciphertext: np.ndarray) -> np.ndarray:
        """Decrypt a ciphertext (stub: returns the data unchanged)."""
        self._total_decryptions += 1
        return ciphertext.copy()

    def ct_pt_matmul(
        self, ct_vector: np.ndarray, pt_matrix: np.ndarray
    ) -> np.ndarray:
        """Ciphertext-plaintext matrix multiply (stub: plaintext matmul).

        In production this is a column-packed ct-pt matmul with zero
        rotations (ZeRo-MOAI technique).
        """
        return ct_vector @ pt_matrix.T

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_encryptions": self._total_encryptions,
            "total_decryptions": self._total_decryptions,
            "slots": self._slots,
        }


# ---------------------------------------------------------------------------
# Inference result
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Result of an inference request."""
    adapter_id: str
    text: str
    tokens_generated: int
    tokens_prompt: int
    elapsed_ms: float
    encrypted: bool = True
    metered: bool = True


# ---------------------------------------------------------------------------
# TenSafeRuntime — the core engine
# ---------------------------------------------------------------------------

class TenSafeRuntime:
    """Proprietary runtime for executing TGSP adapters with HE inference.

    This is the central engine that:
      - Loads the mandatory TSA (TenSafe System Adapter) first
      - Loads domain TGSP adapters through the 8-step load gate
      - Runs HE-encrypted inference with per-token metering
      - Supports atomic hot-swap of adapters
      - Manages adapter lifecycle (load, unload, list)

    Thread safety: all mutable state is protected by a reentrant lock.

    Args:
        config: RuntimeConfig with engine settings.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._lock = threading.RLock()

        # Core components
        self.load_gate = TGSPLoadGate(
            trusted_fingerprints=config.trusted_fingerprints,
            strict_moe=config.strict_moe,
            max_rank=config.max_lora_rank,
        )
        self.metering = MeteringService(config)
        self._he = _CKKSStub(
            poly_mod_degree=config.he_poly_mod_degree,
            scale_bits=config.he_scale_bits,
        )

        # Adapter registry
        self._adapters: Dict[str, _LoadedAdapter] = {}

        # TSA (TenSafe System Adapter) state — must be loaded before
        # any domain adapter.  The TSA adapter_id is stored here once
        # a system adapter is loaded via load_adapter().
        self._tsa_adapter_id: Optional[str] = None

        logger.info(
            "TenSafeRuntime initialized (strict_moe=%s, billing=%s, "
            "max_adapters=%d)",
            config.strict_moe, config.billing_enabled, config.max_loaded_adapters,
        )

    @property
    def tsa_loaded(self) -> bool:
        """Whether the TenSafe System Adapter is loaded."""
        return self._tsa_adapter_id is not None

    @property
    def tsa_adapter_id(self) -> Optional[str]:
        """Adapter ID of the loaded TSA, or None if not loaded."""
        return self._tsa_adapter_id

    def get_tsa_info(self) -> Optional[AdapterInfo]:
        """Get info for the loaded TSA, or None."""
        if self._tsa_adapter_id is None:
            return None
        return self.get_adapter_info(self._tsa_adapter_id)

    # ------------------------------------------------------------------
    # Adapter loading
    # ------------------------------------------------------------------

    def load_adapter(self, tgsp_path: str) -> str:
        """Load a TGSP adapter through the 8-step load gate.

        The 8 steps are:
            1. Verify SHA-256 payload hash
            2. Verify dual signatures (Ed25519 + Dilithium3)
            3. Verify creator identity
            4. Validate LoraConfig
            5. Confirm sparse MoE target
            6. Run RVUv2 safety screening (3 layers)
            7. TSA binding verification (domain adapters must bind to TSA)
            8. Final decision — any failure = REJECT

        Domain adapters (adapter_type="domain") require a TSA to be loaded
        first.  System adapters (adapter_type="system") are the TSA itself
        and can be loaded without a prior TSA.

        Args:
            tgsp_path: Path to the .tgsp file.

        Returns:
            adapter_id of the loaded adapter.

        Raises:
            ValueError: If the adapter fails any load gate step.
            RuntimeError: If the adapter limit is reached or TSA not loaded.
        """
        with self._lock:
            active = sum(1 for a in self._adapters.values() if not a.draining)
            if active >= self.config.max_loaded_adapters:
                raise RuntimeError(
                    f"Adapter limit reached: {active}/{self.config.max_loaded_adapters}"
                )

        # Pre-check: peek at the manifest to determine adapter_type.
        # Domain adapters require a loaded TSA.
        pre_manifest = self._peek_manifest(tgsp_path)
        adapter_type = pre_manifest.get("adapter_type", "")

        if adapter_type == "domain" and not self.tsa_loaded:
            raise RuntimeError(
                f"Cannot load domain adapter {tgsp_path}: no TenSafe System "
                f"Adapter (TSA) is loaded.  Load a system adapter first."
            )

        # Step 1-8: Run the load gate
        report = self.load_gate.verify(tgsp_path)

        if report.verdict != GateVerdict.ACCEPT:
            failed = report.failed_steps
            reasons = [f"step {s.step} ({s.name}): {s.reason}" for s in failed]
            raise ValueError(
                f"TGSP load gate REJECTED {tgsp_path}: " + "; ".join(reasons)
            )

        # Parse the TGSP file to extract manifest and weights
        manifest, lora_a, lora_b = self._extract_adapter(tgsp_path)

        # Build adapter info
        lora_config = manifest.get("lora_config", {})
        adapter_id = manifest.get(
            "adapter_id",
            hashlib.sha256(tgsp_path.encode()).hexdigest()[:16],
        )

        info = AdapterInfo(
            adapter_id=adapter_id,
            name=manifest.get("model_name", manifest.get("name", Path(tgsp_path).stem)),
            version=manifest.get("model_version", manifest.get("version", "0.0.0")),
            rank=lora_config.get("rank", manifest.get("rank", 0)),
            alpha=lora_config.get("alpha", manifest.get("alpha", 0)),
            target_modules=lora_config.get(
                "target_modules", manifest.get("target_modules", [])
            ),
            domain=manifest.get("domain", manifest.get("metadata", {}).get("domain", "general")),
            tgsp_path=tgsp_path,
            adapter_type=adapter_type or "domain",
        )

        loaded = _LoadedAdapter(
            info=info,
            manifest=manifest,
            gate_report=report,
            lora_a=lora_a,
            lora_b=lora_b,
        )

        with self._lock:
            self._adapters[adapter_id] = loaded

            # If this is a system adapter, register it as the TSA
            if adapter_type == "system":
                self._tsa_adapter_id = adapter_id
                # Register TSA info with load gate for step 7 binding checks
                sys_caps = manifest.get("system_capabilities", {})
                self.load_gate.set_tsa({
                    "fingerprint": manifest.get("creator", {}).get(
                        "public_key_fingerprint", ""
                    ),
                    "tsa_version": sys_caps.get("tsa_version", "1.0.0"),
                    "runtime_binding_hash": sys_caps.get(
                        "runtime_binding_hash", ""
                    ),
                })
                logger.info(
                    "TSA loaded: %s (id=%s) — domain adapters can now be loaded",
                    info.name, adapter_id,
                )

        logger.info(
            "Adapter loaded: %s (id=%s, type=%s, rank=%d, gate=%s in %.1f ms)",
            info.name, adapter_id, adapter_type or "legacy",
            info.rank, report.verdict.value, report.total_elapsed_ms,
        )
        return adapter_id

    @staticmethod
    def _peek_manifest(tgsp_path: str) -> Dict[str, Any]:
        """Read only the manifest from a TGSP file (no weight loading)."""
        try:
            with open(tgsp_path, "rb") as f:
                header = f.read(TGSP_HEADER_LEN)
                if len(header) < TGSP_HEADER_LEN:
                    return {}
                manifest_len = struct.unpack_from("<I", header, 6)[0]
                manifest_bytes = f.read(manifest_len)
            return json.loads(manifest_bytes.decode("utf-8"))
        except Exception:
            return {}

    def _extract_adapter(
        self, tgsp_path: str
    ) -> tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract manifest and LoRA weights from a TGSP file.

        Returns:
            (manifest, lora_a, lora_b) — lora_a and lora_b are numpy
            arrays or None if deserialization is not available.
        """
        with open(tgsp_path, "rb") as f:
            header = f.read(TGSP_HEADER_LEN)
            manifest_len = struct.unpack_from("<I", header, 6)[0]
            manifest_bytes = f.read(manifest_len)
            payload = f.read()

        manifest = json.loads(manifest_bytes.decode("utf-8"))

        # Try to deserialize LoRA weights from payload
        lora_a = None
        lora_b = None
        try:
            import torch
            import io
            state_dict = torch.load(
                io.BytesIO(payload), map_location="cpu", weights_only=False
            )
            # Extract first A/B pair as representative weights for stub inference
            if isinstance(state_dict, dict):
                for key, tensor in state_dict.items():
                    if "lora_A" in key and lora_a is None:
                        lora_a = tensor.numpy() if hasattr(tensor, "numpy") else np.array(tensor)
                    elif "lora_B" in key and lora_b is None:
                        lora_b = tensor.numpy() if hasattr(tensor, "numpy") else np.array(tensor)
                    if lora_a is not None and lora_b is not None:
                        break
        except Exception as e:
            logger.debug(
                "Could not deserialize LoRA weights from %s: %s (stub inference will be used)",
                tgsp_path, e,
            )
            # Generate synthetic LoRA weights for stub inference
            rank = manifest.get("lora_config", manifest).get("rank", 30)
            hidden_dim = 1536  # Default for Qwen 2.5-1.5B
            rng = np.random.RandomState(42)
            lora_a = rng.randn(rank, hidden_dim).astype(np.float64) * 0.01
            lora_b = rng.randn(hidden_dim, rank).astype(np.float64) * 1e-7

        return manifest, lora_a, lora_b

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        adapter_id: str,
        query: str,
        max_tokens: int = 256,
        tenant_id: str = "default",
        session_id: Optional[str] = None,
    ) -> InferenceResult:
        """Run HE-encrypted inference with metering.

        The inference pipeline:
            1. Look up adapter and acquire reference
            2. Tokenize query (stub: word split)
            3. Encrypt hidden state via CKKS
            4. Apply LoRA delta: ct-pt matmul for A, decrypt, matmul for B
            5. Decode output tokens
            6. Meter usage
            7. Release adapter reference

        Args:
            adapter_id: ID of a loaded adapter.
            query: Input query text.
            max_tokens: Maximum tokens to generate.
            tenant_id: Tenant ID for metering.
            session_id: Session ID (auto-generated if None).

        Returns:
            InferenceResult with generated text and metering data.

        Raises:
            KeyError: If adapter_id is not loaded.
            RuntimeError: If adapter is draining (being unloaded).
        """
        t_start = time.monotonic()

        # Acquire adapter reference
        adapter = self._acquire_adapter(adapter_id)
        try:
            return self._run_inference(
                adapter, query, max_tokens, tenant_id, session_id, t_start
            )
        finally:
            self._release_adapter(adapter_id)

    def _acquire_adapter(self, adapter_id: str) -> _LoadedAdapter:
        """Acquire a reference to a loaded adapter (increment ref count)."""
        with self._lock:
            adapter = self._adapters.get(adapter_id)
            if adapter is None:
                raise KeyError(f"Adapter not loaded: {adapter_id}")
            if adapter.draining:
                raise RuntimeError(
                    f"Adapter {adapter_id} is draining (being unloaded)"
                )
            adapter.ref_count += 1
            return adapter

    def _release_adapter(self, adapter_id: str) -> None:
        """Release adapter reference (decrement ref count)."""
        with self._lock:
            adapter = self._adapters.get(adapter_id)
            if adapter and adapter.ref_count > 0:
                adapter.ref_count -= 1

    def _run_inference(
        self,
        adapter: _LoadedAdapter,
        query: str,
        max_tokens: int,
        tenant_id: str,
        session_id: Optional[str],
        t_start: float,
    ) -> InferenceResult:
        """Execute the inference pipeline."""
        info = adapter.info

        # Stub tokenization: split on whitespace
        prompt_tokens = query.split()
        num_prompt_tokens = len(prompt_tokens)

        # Generate a hidden state vector (stub: random)
        hidden_dim = adapter.lora_a.shape[1] if adapter.lora_a is not None else 1536
        hidden_state = np.random.randn(hidden_dim).astype(np.float64) * 0.1

        # CKKS encrypt hidden state
        ct_hidden = self._he.encrypt(hidden_state)

        # Apply LoRA delta under encryption
        if adapter.lora_a is not None and adapter.lora_b is not None:
            # LoRA A: ct-pt matmul (zero rotations)
            intermediate = self._he.ct_pt_matmul(ct_hidden, adapter.lora_a)
            # Decrypt intermediate
            intermediate_plain = self._he.decrypt(intermediate)
            # LoRA B: plaintext matmul
            delta = intermediate_plain @ adapter.lora_b.T
            # Scale by alpha/rank
            scale = info.alpha / max(info.rank, 1)
            delta = delta * scale
            # Add delta to hidden state
            hidden_state = hidden_state + delta

        # Stub token generation: produce placeholder tokens
        gen_tokens = min(max_tokens, self.config.default_max_tokens)
        # In production this would sample from the LM head distribution
        output_words = [f"[tok_{i}]" for i in range(gen_tokens)]
        output_text = " ".join(output_words)

        # Meter usage
        if session_id is None:
            session_id = f"sess_{uuid.uuid4().hex[:12]}"

        self.metering.start_session(session_id, tenant_id, info.adapter_id)
        allowed, reason = self.metering.meter_tokens(
            session_id,
            tokens_prompt=num_prompt_tokens,
            tokens_generated=gen_tokens,
            adapter_price_per_1k=adapter.manifest.get("price_per_1k_tokens", 0.0),
        )
        self.metering.end_session(session_id)

        if not allowed:
            logger.warning(
                "Inference rate-limited for tenant %s: %s", tenant_id, reason
            )

        # Update adapter stats
        with self._lock:
            info.request_count += 1
            info.token_count += num_prompt_tokens + gen_tokens

        elapsed_ms = (time.monotonic() - t_start) * 1000

        return InferenceResult(
            adapter_id=info.adapter_id,
            text=output_text,
            tokens_generated=gen_tokens,
            tokens_prompt=num_prompt_tokens,
            elapsed_ms=elapsed_ms,
            encrypted=True,
            metered=allowed,
        )

    # ------------------------------------------------------------------
    # Hot swap
    # ------------------------------------------------------------------

    def hot_swap(
        self,
        agent_id: str,
        from_adapter: str,
        to_adapter: str,
        verify: bool = True,
    ) -> bool:
        """Atomic adapter swap — no requests dropped.

        The swap procedure:
            1. Verify the new adapter is loaded (or load it)
            2. Run health check on new adapter
            3. Mark old adapter as draining
            4. Atomically switch the agent's adapter reference
            5. Wait for in-flight requests on old adapter to complete
            6. Unload old adapter

        Args:
            agent_id: Agent that is switching adapters.
            from_adapter: Current adapter ID.
            to_adapter: New adapter ID (must already be loaded, or a TGSP path).
            verify: If True, run health check before completing swap.

        Returns:
            True if swap succeeded, False if rolled back.
        """
        logger.info(
            "Hot-swap requested: agent=%s, from=%s, to=%s",
            agent_id, from_adapter, to_adapter,
        )

        # Ensure the new adapter is loaded
        with self._lock:
            if to_adapter not in self._adapters:
                # Try loading it as a TGSP path
                if Path(to_adapter).exists():
                    try:
                        to_adapter = self.load_adapter(to_adapter)
                    except (ValueError, RuntimeError) as e:
                        logger.error("Hot-swap: failed to load new adapter: %s", e)
                        return False
                else:
                    logger.error("Hot-swap: adapter %s not loaded", to_adapter)
                    return False

        # Health check on new adapter
        if verify:
            try:
                result = self.infer(
                    to_adapter,
                    "health check",
                    max_tokens=self.config.hot_swap_health_check_tokens,
                    tenant_id="__health__",
                )
                if result.tokens_generated == 0:
                    logger.error("Hot-swap: health check produced no tokens")
                    return False
                logger.debug(
                    "Hot-swap: health check passed (%.1f ms)", result.elapsed_ms
                )
            except Exception as e:
                logger.error("Hot-swap: health check failed: %s", e)
                return False

        # Mark old adapter as draining
        with self._lock:
            old = self._adapters.get(from_adapter)
            if old is None:
                logger.warning("Hot-swap: old adapter %s not found", from_adapter)
                # Not an error — the swap target is already active
                return True
            old.draining = True

        # Wait for in-flight requests to complete
        deadline = time.monotonic() + self.config.hot_swap_timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                old = self._adapters.get(from_adapter)
                if old is None or old.ref_count == 0:
                    break
            time.sleep(0.01)  # 10ms poll
        else:
            # Timeout — rollback
            with self._lock:
                old = self._adapters.get(from_adapter)
                if old:
                    old.draining = False
            logger.error(
                "Hot-swap: timeout waiting for in-flight requests on %s",
                from_adapter,
            )
            return False

        # Unload old adapter
        with self._lock:
            self._adapters.pop(from_adapter, None)

        logger.info(
            "Hot-swap complete: agent=%s, %s -> %s",
            agent_id, from_adapter, to_adapter,
        )
        return True

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------

    def list_adapters(self) -> List[AdapterInfo]:
        """List all loaded adapters."""
        with self._lock:
            return [a.info for a in self._adapters.values() if not a.draining]

    def get_adapter_info(self, adapter_id: str) -> Optional[AdapterInfo]:
        """Get info for a specific loaded adapter."""
        with self._lock:
            adapter = self._adapters.get(adapter_id)
            return adapter.info if adapter else None

    def unload_adapter(self, adapter_id: str) -> bool:
        """Unload an adapter (waits for in-flight requests).

        Returns:
            True if successfully unloaded, False if timeout.
        """
        with self._lock:
            adapter = self._adapters.get(adapter_id)
            if adapter is None:
                return True  # Already unloaded
            adapter.draining = True

        # Wait for in-flight requests
        deadline = time.monotonic() + self.config.hot_swap_timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                adapter = self._adapters.get(adapter_id)
                if adapter is None or adapter.ref_count == 0:
                    self._adapters.pop(adapter_id, None)
                    logger.info("Adapter unloaded: %s", adapter_id)
                    return True
            time.sleep(0.01)

        logger.error("Timeout unloading adapter %s", adapter_id)
        return False

    def get_gate_report(self, adapter_id: str) -> Optional[GateReport]:
        """Get the load gate report for a loaded adapter."""
        with self._lock:
            adapter = self._adapters.get(adapter_id)
            return adapter.gate_report if adapter else None

    @property
    def he_stats(self) -> Dict[str, int]:
        """Get HE engine statistics."""
        return self._he.stats

    @property
    def adapter_count(self) -> int:
        """Number of currently loaded (non-draining) adapters."""
        with self._lock:
            return sum(1 for a in self._adapters.values() if not a.draining)
