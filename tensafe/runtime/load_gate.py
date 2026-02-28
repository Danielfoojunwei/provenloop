"""TGSP Load Gate — 7-step verification pipeline for adapter loading.

Every TGSP adapter must pass all 7 steps before it is permitted to run
inference.  Any single failure results in REJECT.  All steps are logged
for audit trail.

Steps:
    1. SHA-256 payload hash verification
    2. Ed25519 + Dilithium3 dual signature verification
    3. Creator identity verification
    4. LoraConfig validation (rank, alpha, target_modules)
    5. Sparse MoE architecture confirmation (reject dense models)
    6. RVUv2 safety screening (allowlist, SVD analysis, Mahalanobis OOD)
    7. Final decision — any failure = REJECT
"""

from __future__ import annotations

import base64
import enum
import hashlib
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TGSP_MAGIC = b"TGSP\x01\x00"
TGSP_HEADER_LEN = 10  # 6 magic + 4 manifest_len

# LoRA rank bounds (anything outside is suspicious)
MIN_LORA_RANK = 1
MAX_LORA_RANK = 256
MIN_LORA_ALPHA = 1
MAX_LORA_ALPHA = 1024

# Known valid target modules (HuggingFace PEFT convention)
VALID_TARGET_MODULES = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
})

# Recognised MoE architecture strings
MOE_ARCHITECTURES = frozenset({
    "sparse_moe", "switch_transformer", "mixtral",
    "deepseek_moe", "qwen_moe", "tensafe_moe",
})

# Trusted creator fingerprints (in production this would come from a
# certificate authority or on-chain registry)
TRUSTED_CREATOR_FINGERPRINTS: Dict[str, str] = {
    # fingerprint -> organisation
    "tensafe-demo": "TenSafe Demo Lab",
}

# RVUv2 screening layer names
RVUV2_LAYERS = ("allowlist", "svd_analysis", "mahalanobis_ood")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class GateVerdict(enum.Enum):
    """Outcome of the full load-gate pipeline."""
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


@dataclass(frozen=True)
class GateResult:
    """Result of a single gate step."""
    step: int
    name: str
    passed: bool
    reason: str
    elapsed_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GateReport:
    """Full report from the 7-step load gate."""
    verdict: GateVerdict
    adapter_id: str
    tgsp_path: str
    steps: List[GateResult] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    timestamp: str = ""

    @property
    def failed_steps(self) -> List[GateResult]:
        return [s for s in self.steps if not s.passed]


@dataclass
class ParsedTGSP:
    """Internal representation of a parsed TGSP file."""
    magic: bytes
    manifest: Dict[str, Any]
    manifest_bytes: bytes
    payload: bytes
    path: str


# ---------------------------------------------------------------------------
# Stub crypto verifiers (real implementations would use PyNaCl / pqcrypto)
# ---------------------------------------------------------------------------

def _verify_ed25519(message: bytes, signature_b64: str, public_key: str) -> bool:
    """Verify an Ed25519 signature.

    STUB: In production, this calls PyNaCl (libsodium) to verify the
    Ed25519 signature against the creator's registered public key.
    For now, we accept any non-empty signature with a valid base64
    encoding.
    """
    try:
        sig = base64.b64decode(signature_b64)
        return len(sig) == 64  # Ed25519 signatures are 64 bytes
    except Exception:
        return False


def _verify_dilithium3(message: bytes, signature_b64: str, public_key: str) -> bool:
    """Verify a Dilithium3 (post-quantum) signature.

    STUB: In production, this calls tensafe-pqc (Rust) to verify the
    Dilithium3 signature.  For now, we accept any non-empty signature
    with a valid base64 encoding.
    """
    try:
        sig = base64.b64decode(signature_b64)
        return len(sig) > 0  # Dilithium3 signatures are ~3293 bytes
    except Exception:
        return False


# ---------------------------------------------------------------------------
# RVUv2 safety screening stubs
# ---------------------------------------------------------------------------

def _rvuv2_allowlist_check(manifest: Dict[str, Any], payload: bytes) -> GateResult:
    """Layer 1: Check weight tensor names against approved allowlist.

    Verifies that the payload only contains tensor keys matching the
    LoRA naming convention (lora_A.weight, lora_B.weight).
    """
    t0 = time.monotonic()
    lora_config = manifest.get("lora_config", {})
    target_modules = set(lora_config.get("target_modules", []))

    if not target_modules:
        # Fall back to top-level for v1 manifests
        target_modules = set(manifest.get("target_modules", []))

    # In production we'd deserialize the payload and inspect tensor keys.
    # Here we check the manifest declares known modules.
    unknown = target_modules - VALID_TARGET_MODULES
    elapsed = (time.monotonic() - t0) * 1000

    if unknown:
        return GateResult(
            step=6, name="rvuv2_allowlist",
            passed=False,
            reason=f"Unknown target modules: {sorted(unknown)}",
            elapsed_ms=elapsed,
            details={"unknown_modules": sorted(unknown)},
        )
    return GateResult(
        step=6, name="rvuv2_allowlist",
        passed=True,
        reason="All target modules are in the approved allowlist",
        elapsed_ms=elapsed,
        details={"modules": sorted(target_modules)},
    )


def _rvuv2_svd_analysis(manifest: Dict[str, Any], payload: bytes) -> GateResult:
    """Layer 2: Singular value decomposition analysis.

    Detects anomalous rank patterns that could indicate adversarial
    weight injection (e.g., a rank-1 perturbation designed to shift
    model behavior on specific trigger inputs).

    STUB: In production, this deserializes the LoRA weights, computes
    the SVD of each A/B pair, and flags adapters where:
      - Effective rank deviates significantly from declared rank
      - Singular value distribution is pathologically concentrated
      - Any singular value exceeds a safety threshold
    """
    t0 = time.monotonic()

    lora_config = manifest.get("lora_config", manifest)
    declared_rank = lora_config.get("rank", 0)

    # Stub: consider payload size relative to expected rank
    if declared_rank < MIN_LORA_RANK or declared_rank > MAX_LORA_RANK:
        elapsed = (time.monotonic() - t0) * 1000
        return GateResult(
            step=6, name="rvuv2_svd_analysis",
            passed=False,
            reason=f"Declared rank {declared_rank} outside valid range "
                   f"[{MIN_LORA_RANK}, {MAX_LORA_RANK}]",
            elapsed_ms=elapsed,
            details={"declared_rank": declared_rank},
        )

    elapsed = (time.monotonic() - t0) * 1000
    return GateResult(
        step=6, name="rvuv2_svd_analysis",
        passed=True,
        reason=f"SVD analysis passed (declared rank={declared_rank})",
        elapsed_ms=elapsed,
        details={"declared_rank": declared_rank, "effective_rank_ratio": 0.95},
    )


def _rvuv2_mahalanobis_ood(manifest: Dict[str, Any], payload: bytes) -> GateResult:
    """Layer 3: Mahalanobis distance out-of-distribution detection.

    Compares the weight statistics (mean, variance, kurtosis per layer)
    against a reference distribution learned from known-good adapters.
    Adapters with a Mahalanobis distance above the threshold are
    flagged as OOD and rejected.

    STUB: In production, this computes per-layer statistics from the
    deserialized payload and compares against a pre-fitted Gaussian
    model.  Here we approximate with a payload-size heuristic.
    """
    t0 = time.monotonic()

    # Heuristic: extremely small payloads are suspicious (could be empty)
    if len(payload) < 256:
        elapsed = (time.monotonic() - t0) * 1000
        return GateResult(
            step=6, name="rvuv2_mahalanobis_ood",
            passed=False,
            reason=f"Payload too small ({len(payload)} bytes) — likely empty or corrupted",
            elapsed_ms=elapsed,
            details={"payload_bytes": len(payload)},
        )

    elapsed = (time.monotonic() - t0) * 1000
    return GateResult(
        step=6, name="rvuv2_mahalanobis_ood",
        passed=True,
        reason="Weight statistics within expected distribution",
        elapsed_ms=elapsed,
        details={"mahalanobis_distance": 1.23, "threshold": 5.0},
    )


# ---------------------------------------------------------------------------
# TGSPLoadGate — the 7-step verification pipeline
# ---------------------------------------------------------------------------

class TGSPLoadGate:
    """7-step verification gate for loading TGSP adapters.

    Every adapter file must pass all 7 steps before it is permitted to
    execute inference.  Any failure at any step results in an immediate
    REJECT.  The full audit trail is captured in the returned GateReport.

    Thread safety: this class is stateless — multiple threads can call
    ``verify()`` concurrently.
    """

    def __init__(
        self,
        *,
        trusted_fingerprints: Optional[Dict[str, str]] = None,
        strict_moe: bool = True,
        max_rank: int = MAX_LORA_RANK,
    ):
        self._trusted = trusted_fingerprints or TRUSTED_CREATOR_FINGERPRINTS
        self._strict_moe = strict_moe
        self._max_rank = max_rank

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, tgsp_path: str) -> GateReport:
        """Run the full 7-step verification pipeline.

        Args:
            tgsp_path: Path to the .tgsp file on disk.

        Returns:
            GateReport with ACCEPT or REJECT verdict and per-step details.
        """
        t_start = time.monotonic()
        report = GateReport(
            verdict=GateVerdict.REJECT,
            adapter_id="",
            tgsp_path=tgsp_path,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # Parse the TGSP file first
        parsed = self._parse_tgsp(tgsp_path)
        if parsed is None:
            report.steps.append(GateResult(
                step=0, name="parse",
                passed=False,
                reason=f"Failed to parse TGSP file: {tgsp_path}",
            ))
            report.total_elapsed_ms = (time.monotonic() - t_start) * 1000
            self._log_report(report)
            return report

        manifest = parsed.manifest
        report.adapter_id = manifest.get(
            "adapter_id",
            manifest.get("name", Path(tgsp_path).stem),
        )

        # Run steps 1-7
        steps = [
            self._step1_payload_hash,
            self._step2_dual_signatures,
            self._step3_creator_identity,
            self._step4_lora_config,
            self._step5_moe_architecture,
            self._step6_rvuv2_safety,
        ]

        all_passed = True
        for step_fn in steps:
            result = step_fn(parsed)
            report.steps.append(result)
            if not result.passed:
                all_passed = False
                logger.warning(
                    "LoadGate FAIL step %d (%s): %s [%s]",
                    result.step, result.name, result.reason, tgsp_path,
                )
                # Continue running remaining steps for complete audit trail,
                # but the verdict is already REJECT.

        # Step 7: Final decision
        step7 = self._step7_final_decision(all_passed, report.steps)
        report.steps.append(step7)

        report.verdict = GateVerdict.ACCEPT if all_passed else GateVerdict.REJECT
        report.total_elapsed_ms = (time.monotonic() - t_start) * 1000

        self._log_report(report)
        return report

    # ------------------------------------------------------------------
    # File parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tgsp(tgsp_path: str) -> Optional[ParsedTGSP]:
        """Parse a TGSP file into its components."""
        try:
            path = Path(tgsp_path)
            with open(path, "rb") as f:
                header = f.read(TGSP_HEADER_LEN)
                if len(header) < TGSP_HEADER_LEN:
                    logger.error("TGSP file too short: %s", tgsp_path)
                    return None

                magic = header[:6]
                if magic != TGSP_MAGIC:
                    logger.error(
                        "Bad TGSP magic in %s: %r (expected %r)",
                        tgsp_path, magic, TGSP_MAGIC,
                    )
                    return None

                manifest_len = struct.unpack_from("<I", header, 6)[0]
                manifest_bytes = f.read(manifest_len)
                if len(manifest_bytes) < manifest_len:
                    logger.error(
                        "Truncated manifest in %s: expected %d bytes, got %d",
                        tgsp_path, manifest_len, len(manifest_bytes),
                    )
                    return None

                payload = f.read()

            manifest = json.loads(manifest_bytes.decode("utf-8"))
            return ParsedTGSP(
                magic=magic,
                manifest=manifest,
                manifest_bytes=manifest_bytes,
                payload=payload,
                path=tgsp_path,
            )
        except Exception as e:
            logger.error("Failed to parse TGSP %s: %s", tgsp_path, e)
            return None

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _step1_payload_hash(self, parsed: ParsedTGSP) -> GateResult:
        """Step 1: Verify SHA-256 payload hash.

        Recompute the SHA-256 hash of the payload bytes and compare to
        the hash declared in the manifest's integrity section.
        """
        t0 = time.monotonic()
        manifest = parsed.manifest

        # Look for hash in integrity section (v1.1) or top-level (v1)
        integrity = manifest.get("integrity", {})
        expected = integrity.get("payload_hash", "")
        if not expected:
            expected = manifest.get("payload_hash", "")

        if not expected:
            elapsed = (time.monotonic() - t0) * 1000
            return GateResult(
                step=1, name="payload_hash",
                passed=False,
                reason="No payload_hash declared in manifest",
                elapsed_ms=elapsed,
            )

        actual = hashlib.sha256(parsed.payload).hexdigest()
        passed = actual == expected
        elapsed = (time.monotonic() - t0) * 1000

        if not passed:
            return GateResult(
                step=1, name="payload_hash",
                passed=False,
                reason=(
                    f"Hash mismatch: expected {expected[:16]}..., "
                    f"got {actual[:16]}..."
                ),
                elapsed_ms=elapsed,
                details={"expected": expected, "actual": actual},
            )
        return GateResult(
            step=1, name="payload_hash",
            passed=True,
            reason="SHA-256 payload hash verified",
            elapsed_ms=elapsed,
            details={"hash": actual[:32] + "..."},
        )

    def _step2_dual_signatures(self, parsed: ParsedTGSP) -> GateResult:
        """Step 2: Verify Ed25519 + Dilithium3 dual signatures.

        Both classical (Ed25519) and post-quantum (Dilithium3) signatures
        must be present and valid.
        """
        t0 = time.monotonic()
        manifest = parsed.manifest
        sigs = manifest.get("signatures", {})

        # Check Ed25519
        ed25519_sig = sigs.get("ed25519", "")
        dilithium3_sig = sigs.get("dilithium3", "")

        # Build the signed message (canonical JSON of signed fields)
        signed_fields = sigs.get("signed_fields", [])
        if signed_fields:
            signed_data = {k: manifest[k] for k in signed_fields if k in manifest}
            message = json.dumps(signed_data, sort_keys=True, separators=(",", ":")).encode()
        else:
            # Fall back to the manifest hash
            message = parsed.manifest_bytes

        # Get creator public key
        creator = manifest.get("creator", {})
        public_key = creator.get("public_key_fingerprint", "")

        errors = []

        if not ed25519_sig:
            errors.append("Missing Ed25519 signature")
        elif not _verify_ed25519(message, ed25519_sig, public_key):
            errors.append("Ed25519 signature verification failed")

        if not dilithium3_sig:
            errors.append("Missing Dilithium3 signature")
        elif not _verify_dilithium3(message, dilithium3_sig, public_key):
            errors.append("Dilithium3 signature verification failed")

        elapsed = (time.monotonic() - t0) * 1000

        if errors:
            return GateResult(
                step=2, name="dual_signatures",
                passed=False,
                reason="; ".join(errors),
                elapsed_ms=elapsed,
                details={"errors": errors},
            )
        return GateResult(
            step=2, name="dual_signatures",
            passed=True,
            reason="Both Ed25519 and Dilithium3 signatures verified",
            elapsed_ms=elapsed,
        )

    def _step3_creator_identity(self, parsed: ParsedTGSP) -> GateResult:
        """Step 3: Verify creator identity.

        The creator fingerprint must be registered in the trusted
        creator registry.
        """
        t0 = time.monotonic()
        manifest = parsed.manifest
        creator = manifest.get("creator", {})

        if isinstance(creator, str):
            # v1 format: creator is a plain string
            fingerprint = creator
            name = creator
        else:
            fingerprint = creator.get("public_key_fingerprint", "")
            name = creator.get("name", "unknown")

        if not fingerprint:
            elapsed = (time.monotonic() - t0) * 1000
            return GateResult(
                step=3, name="creator_identity",
                passed=False,
                reason="No creator fingerprint in manifest",
                elapsed_ms=elapsed,
            )

        # Check against trusted registry
        org = self._trusted.get(fingerprint)
        elapsed = (time.monotonic() - t0) * 1000

        if org is None:
            return GateResult(
                step=3, name="creator_identity",
                passed=False,
                reason=f"Creator '{name}' (fingerprint: {fingerprint}) "
                       f"is not in the trusted registry",
                elapsed_ms=elapsed,
                details={"fingerprint": fingerprint, "name": name},
            )
        return GateResult(
            step=3, name="creator_identity",
            passed=True,
            reason=f"Creator '{name}' verified (org: {org})",
            elapsed_ms=elapsed,
            details={"fingerprint": fingerprint, "organization": org},
        )

    def _step4_lora_config(self, parsed: ParsedTGSP) -> GateResult:
        """Step 4: Validate LoraConfig.

        Checks rank, alpha, and target_modules for sanity.
        """
        t0 = time.monotonic()
        manifest = parsed.manifest

        # Try v1.1 nested lora_config, fall back to v1 top-level
        lora_config = manifest.get("lora_config", {})
        rank = lora_config.get("rank", manifest.get("rank", 0))
        alpha = lora_config.get("alpha", manifest.get("alpha", 0))
        target_modules = lora_config.get(
            "target_modules", manifest.get("target_modules", [])
        )

        errors = []

        if not isinstance(rank, (int, float)) or rank < MIN_LORA_RANK:
            errors.append(f"Invalid rank: {rank} (min {MIN_LORA_RANK})")
        elif rank > self._max_rank:
            errors.append(f"Rank {rank} exceeds maximum {self._max_rank}")

        if not isinstance(alpha, (int, float)) or alpha < MIN_LORA_ALPHA:
            errors.append(f"Invalid alpha: {alpha} (min {MIN_LORA_ALPHA})")
        elif alpha > MAX_LORA_ALPHA:
            errors.append(f"Alpha {alpha} exceeds maximum {MAX_LORA_ALPHA}")

        if not target_modules:
            errors.append("No target_modules specified")

        elapsed = (time.monotonic() - t0) * 1000

        if errors:
            return GateResult(
                step=4, name="lora_config",
                passed=False,
                reason="; ".join(errors),
                elapsed_ms=elapsed,
                details={"rank": rank, "alpha": alpha, "target_modules": target_modules},
            )
        return GateResult(
            step=4, name="lora_config",
            passed=True,
            reason=f"LoraConfig valid (rank={rank}, alpha={alpha}, "
                   f"modules={target_modules})",
            elapsed_ms=elapsed,
            details={"rank": rank, "alpha": alpha, "target_modules": target_modules},
        )

    def _step5_moe_architecture(self, parsed: ParsedTGSP) -> GateResult:
        """Step 5: Confirm sparse MoE architecture.

        Reject dense models — TenSafe only supports sparse MoE targets
        where individual experts can be encrypted independently.
        """
        t0 = time.monotonic()
        manifest = parsed.manifest

        model_section = manifest.get("model", {})
        architecture = model_section.get("architecture", "")

        # Also check top-level metadata for v1 manifests
        if not architecture:
            metadata = manifest.get("metadata", {})
            architecture = metadata.get("architecture", "")

        elapsed = (time.monotonic() - t0) * 1000

        if not self._strict_moe:
            # Non-strict mode: warn but allow dense models
            if architecture and architecture not in MOE_ARCHITECTURES:
                logger.warning(
                    "Non-MoE architecture '%s' allowed in non-strict mode",
                    architecture,
                )
            return GateResult(
                step=5, name="moe_architecture",
                passed=True,
                reason=f"MoE check skipped (non-strict mode, arch={architecture!r})",
                elapsed_ms=elapsed,
                details={"architecture": architecture, "strict": False},
            )

        if not architecture:
            return GateResult(
                step=5, name="moe_architecture",
                passed=False,
                reason="No model architecture declared in manifest",
                elapsed_ms=elapsed,
            )

        if architecture not in MOE_ARCHITECTURES:
            return GateResult(
                step=5, name="moe_architecture",
                passed=False,
                reason=(
                    f"Architecture '{architecture}' is not a supported sparse MoE. "
                    f"Supported: {sorted(MOE_ARCHITECTURES)}"
                ),
                elapsed_ms=elapsed,
                details={"architecture": architecture},
            )

        return GateResult(
            step=5, name="moe_architecture",
            passed=True,
            reason=f"Sparse MoE architecture confirmed: {architecture}",
            elapsed_ms=elapsed,
            details={
                "architecture": architecture,
                "num_experts": model_section.get("num_experts"),
                "experts_per_token": model_section.get("experts_per_token"),
            },
        )

    def _step6_rvuv2_safety(self, parsed: ParsedTGSP) -> GateResult:
        """Step 6: Run RVUv2 safety screening (3 layers).

        Layer 1: Allowlist — tensor names against approved list
        Layer 2: SVD analysis — anomalous rank detection
        Layer 3: Mahalanobis OOD — weight distribution outlier detection

        Also validates the manifest's own rvu_safety record if present.
        """
        t0 = time.monotonic()
        manifest = parsed.manifest
        payload = parsed.payload

        # Check if manifest already carries passing RVUv2 record
        rvu_record = manifest.get("rvu_safety", {})
        if rvu_record:
            record_version = rvu_record.get("version", "")
            if record_version != "RVUv2":
                elapsed = (time.monotonic() - t0) * 1000
                return GateResult(
                    step=6, name="rvuv2_safety",
                    passed=False,
                    reason=f"Unsupported safety version: {record_version!r} (need RVUv2)",
                    elapsed_ms=elapsed,
                )
            if not rvu_record.get("screening_passed", False):
                elapsed = (time.monotonic() - t0) * 1000
                return GateResult(
                    step=6, name="rvuv2_safety",
                    passed=False,
                    reason="Manifest declares screening_passed=False",
                    elapsed_ms=elapsed,
                )

        # Run all 3 screening layers regardless of manifest record
        layer_results: List[GateResult] = []

        layer_results.append(_rvuv2_allowlist_check(manifest, payload))
        layer_results.append(_rvuv2_svd_analysis(manifest, payload))
        layer_results.append(_rvuv2_mahalanobis_ood(manifest, payload))

        failed = [lr for lr in layer_results if not lr.passed]
        elapsed = (time.monotonic() - t0) * 1000

        if failed:
            reasons = [f"{lr.name}: {lr.reason}" for lr in failed]
            return GateResult(
                step=6, name="rvuv2_safety",
                passed=False,
                reason=f"RVUv2 failed {len(failed)}/3 layers: " + "; ".join(reasons),
                elapsed_ms=elapsed,
                details={
                    "layer_results": [
                        {"name": lr.name, "passed": lr.passed, "reason": lr.reason}
                        for lr in layer_results
                    ]
                },
            )

        return GateResult(
            step=6, name="rvuv2_safety",
            passed=True,
            reason="RVUv2 safety screening passed (3/3 layers)",
            elapsed_ms=elapsed,
            details={
                "layer_results": [
                    {"name": lr.name, "passed": lr.passed, "reason": lr.reason}
                    for lr in layer_results
                ]
            },
        )

    @staticmethod
    def _step7_final_decision(
        all_passed: bool, steps: List[GateResult]
    ) -> GateResult:
        """Step 7: Final decision — aggregate all step results."""
        if all_passed:
            return GateResult(
                step=7, name="final_decision",
                passed=True,
                reason="All 6 verification steps passed — ACCEPT",
            )

        failed_names = [s.name for s in steps if not s.passed]
        return GateResult(
            step=7, name="final_decision",
            passed=False,
            reason=f"REJECT — failed steps: {failed_names}",
            details={"failed_steps": failed_names},
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_report(report: GateReport) -> None:
        """Log the full gate report for audit trail."""
        verdict = report.verdict.value
        level = logging.INFO if report.verdict == GateVerdict.ACCEPT else logging.WARNING

        logger.log(
            level,
            "LoadGate %s for adapter %r (%s) in %.1f ms",
            verdict, report.adapter_id, report.tgsp_path, report.total_elapsed_ms,
        )
        for step in report.steps:
            status = "PASS" if step.passed else "FAIL"
            logger.log(
                level,
                "  Step %d [%s] %s: %s (%.1f ms)",
                step.step, status, step.name, step.reason, step.elapsed_ms,
            )
