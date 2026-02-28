"""TGSPAdapter — load, verify, and use .tgsp adapter files.

Provides a high-level API for working with TGSP adapters:
  - Parse and verify .tgsp binary format
  - Extract manifest metadata
  - Access LoRA weight payload
  - Verify integrity (hash, signatures)
  - Convert back to standard LoRA formats

Usage:
    adapter = TGSPAdapter.load("my-adapter.tgsp")
    print(adapter.name, adapter.rank)
    print(adapter.verify())   # True if integrity checks pass
    weights = adapter.get_payload()
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# TGSP binary format constants
TGSP_MAGIC = b"TGSP\x01\x00"
TGSP_HEADER_LEN = 10


# ---------------------------------------------------------------------------
# Parsed adapter data
# ---------------------------------------------------------------------------

@dataclass
class TGSPManifest:
    """Parsed TGSP manifest with typed access to all fields."""
    raw: Dict[str, Any]

    @property
    def format_version(self) -> str:
        return self.raw.get("version", self.raw.get("format_version", "1.0"))

    @property
    def name(self) -> str:
        return self.raw.get("name", self.raw.get("model_name", ""))

    @property
    def adapter_id(self) -> str:
        return self.raw.get("adapter_id", "")

    @property
    def model_version(self) -> str:
        return self.raw.get("model_version", self.raw.get("version", "0.0.0"))

    @property
    def domain(self) -> str:
        meta = self.raw.get("metadata", {})
        return self.raw.get("domain", meta.get("domain", "general"))

    # LoRA config
    @property
    def rank(self) -> int:
        lc = self.raw.get("lora_config", {})
        return lc.get("rank", self.raw.get("rank", 0))

    @property
    def alpha(self) -> float:
        lc = self.raw.get("lora_config", {})
        return lc.get("alpha", self.raw.get("alpha", 0.0))

    @property
    def target_modules(self) -> List[str]:
        lc = self.raw.get("lora_config", {})
        return lc.get("target_modules", self.raw.get("target_modules", []))

    # Model config
    @property
    def architecture(self) -> str:
        model = self.raw.get("model", {})
        return model.get("architecture", "")

    @property
    def base_model(self) -> str:
        model = self.raw.get("model", {})
        return model.get("base_model", "")

    # Integrity
    @property
    def payload_hash(self) -> str:
        integrity = self.raw.get("integrity", {})
        return integrity.get("payload_hash", self.raw.get("payload_hash", ""))

    # Creator
    @property
    def creator_name(self) -> str:
        creator = self.raw.get("creator", {})
        if isinstance(creator, str):
            return creator
        return creator.get("name", "")

    @property
    def creator_fingerprint(self) -> str:
        creator = self.raw.get("creator", {})
        if isinstance(creator, str):
            return creator
        return creator.get("public_key_fingerprint", "")

    # Skill
    @property
    def skill_description(self) -> str:
        skill = self.raw.get("skill", {})
        return skill.get("description", "")

    @property
    def skill_triggers(self) -> List[str]:
        skill = self.raw.get("skill", {})
        return skill.get("triggers", [])

    # RVUv2
    @property
    def rvu_passed(self) -> bool:
        rvu = self.raw.get("rvu_safety", {})
        return rvu.get("screening_passed", False)

    # Signatures
    @property
    def has_ed25519(self) -> bool:
        sigs = self.raw.get("signatures", {})
        return bool(sigs.get("ed25519", ""))

    @property
    def has_dilithium3(self) -> bool:
        sigs = self.raw.get("signatures", {})
        return bool(sigs.get("dilithium3", ""))

    # Marketplace
    @property
    def price_per_1k_tokens(self) -> float:
        return self.raw.get("price_per_1k_tokens", 0.0)

    @property
    def license(self) -> str:
        return self.raw.get("license", "unknown")


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Result of TGSP integrity verification."""
    valid: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        passed = sum(1 for v in self.checks.values() if v)
        total = len(self.checks)
        status = "VALID" if self.valid else "INVALID"
        return f"{status}: {passed}/{total} checks passed"


# ---------------------------------------------------------------------------
# TGSPAdapter
# ---------------------------------------------------------------------------

class TGSPAdapter:
    """High-level interface for loading and using .tgsp adapter files.

    Provides typed access to manifest fields, integrity verification,
    payload extraction, and format conversion.

    Use the class method ``load()`` to create an instance from a file,
    or ``from_bytes()`` to create from in-memory data.
    """

    def __init__(
        self,
        manifest: TGSPManifest,
        payload: bytes,
        source_path: Optional[str] = None,
    ):
        self._manifest = manifest
        self._payload = payload
        self._source_path = source_path

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, tgsp_path: str) -> "TGSPAdapter":
        """Load a TGSP adapter from a file on disk.

        Args:
            tgsp_path: Path to the .tgsp file.

        Returns:
            TGSPAdapter instance.

        Raises:
            ValueError: If the file is not a valid TGSP file.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(tgsp_path)
        if not path.exists():
            raise FileNotFoundError(f"TGSP file not found: {tgsp_path}")

        with open(path, "rb") as f:
            header = f.read(TGSP_HEADER_LEN)
            if len(header) < TGSP_HEADER_LEN:
                raise ValueError(f"File too short to be a TGSP file: {tgsp_path}")

            magic = header[:6]
            if magic != TGSP_MAGIC:
                raise ValueError(
                    f"Invalid TGSP magic bytes in {tgsp_path}: {magic!r}"
                )

            manifest_len = struct.unpack_from("<I", header, 6)[0]
            manifest_bytes = f.read(manifest_len)
            if len(manifest_bytes) < manifest_len:
                raise ValueError(
                    f"Truncated manifest in {tgsp_path}: expected {manifest_len} bytes"
                )

            payload = f.read()

        manifest_dict = json.loads(manifest_bytes.decode("utf-8"))
        manifest = TGSPManifest(raw=manifest_dict)

        logger.info(
            "Loaded TGSP: %s (id=%s, rank=%d, %d bytes payload)",
            manifest.name, manifest.adapter_id, manifest.rank, len(payload),
        )

        return cls(
            manifest=manifest,
            payload=payload,
            source_path=str(path.resolve()),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "TGSPAdapter":
        """Load a TGSP adapter from in-memory bytes.

        Args:
            data: Complete TGSP binary data.

        Returns:
            TGSPAdapter instance.
        """
        if len(data) < TGSP_HEADER_LEN:
            raise ValueError("Data too short to be a TGSP file")

        magic = data[:6]
        if magic != TGSP_MAGIC:
            raise ValueError(f"Invalid TGSP magic bytes: {magic!r}")

        manifest_len = struct.unpack_from("<I", data, 6)[0]
        manifest_end = TGSP_HEADER_LEN + manifest_len

        if len(data) < manifest_end:
            raise ValueError("Truncated manifest data")

        manifest_bytes = data[TGSP_HEADER_LEN:manifest_end]
        payload = data[manifest_end:]

        manifest_dict = json.loads(manifest_bytes.decode("utf-8"))
        return cls(
            manifest=TGSPManifest(raw=manifest_dict),
            payload=payload,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def manifest(self) -> TGSPManifest:
        """Access the parsed manifest."""
        return self._manifest

    @property
    def name(self) -> str:
        return self._manifest.name

    @property
    def adapter_id(self) -> str:
        return self._manifest.adapter_id

    @property
    def rank(self) -> int:
        return self._manifest.rank

    @property
    def alpha(self) -> float:
        return self._manifest.alpha

    @property
    def domain(self) -> str:
        return self._manifest.domain

    @property
    def source_path(self) -> Optional[str]:
        return self._source_path

    @property
    def payload_size(self) -> int:
        return len(self._payload)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self) -> VerificationResult:
        """Verify the integrity of the TGSP adapter.

        Checks:
            1. Payload hash matches manifest
            2. Ed25519 signature present
            3. Dilithium3 signature present
            4. RVUv2 screening passed
            5. LoRA config valid

        Returns:
            VerificationResult with per-check details.
        """
        result = VerificationResult(valid=True)

        # Check 1: Payload hash
        expected_hash = self._manifest.payload_hash
        if expected_hash:
            actual_hash = hashlib.sha256(self._payload).hexdigest()
            hash_ok = actual_hash == expected_hash
            result.checks["payload_hash"] = hash_ok
            if not hash_ok:
                result.valid = False
                result.errors.append(
                    f"Payload hash mismatch: expected {expected_hash[:16]}..., "
                    f"got {actual_hash[:16]}..."
                )
        else:
            result.checks["payload_hash"] = True  # No hash to verify

        # Check 2: Ed25519 signature
        result.checks["ed25519_signature"] = self._manifest.has_ed25519
        if not self._manifest.has_ed25519:
            result.errors.append("Missing Ed25519 signature")
            # Not fatal for verification — just a warning

        # Check 3: Dilithium3 signature
        result.checks["dilithium3_signature"] = self._manifest.has_dilithium3
        if not self._manifest.has_dilithium3:
            result.errors.append("Missing Dilithium3 signature")

        # Check 4: RVUv2 safety
        result.checks["rvu_safety"] = self._manifest.rvu_passed
        if not self._manifest.rvu_passed:
            result.valid = False
            result.errors.append("RVUv2 safety screening not passed")

        # Check 5: LoRA config
        rank_ok = 1 <= self._manifest.rank <= 256
        alpha_ok = self._manifest.alpha > 0
        modules_ok = len(self._manifest.target_modules) > 0
        lora_ok = rank_ok and alpha_ok and modules_ok
        result.checks["lora_config"] = lora_ok
        if not lora_ok:
            result.valid = False
            if not rank_ok:
                result.errors.append(f"Invalid LoRA rank: {self._manifest.rank}")
            if not alpha_ok:
                result.errors.append(f"Invalid LoRA alpha: {self._manifest.alpha}")
            if not modules_ok:
                result.errors.append("No target modules specified")

        return result

    # ------------------------------------------------------------------
    # Payload access
    # ------------------------------------------------------------------

    def get_payload_bytes(self) -> bytes:
        """Get the raw payload bytes."""
        return self._payload

    def get_payload_numpy(self) -> Dict[str, np.ndarray]:
        """Deserialize payload as numpy arrays.

        Returns:
            Dict of array name -> numpy array.

        Raises:
            RuntimeError: If the payload cannot be deserialized.
        """
        import io
        try:
            data = np.load(io.BytesIO(self._payload), allow_pickle=False)
            return dict(data)
        except Exception:
            pass

        # Try torch format
        try:
            import torch
            state_dict = torch.load(
                io.BytesIO(self._payload),
                map_location="cpu",
                weights_only=False,
            )
            if isinstance(state_dict, dict):
                result = {}
                for key, tensor in state_dict.items():
                    if hasattr(tensor, "numpy"):
                        result[key] = tensor.numpy()
                    elif isinstance(tensor, np.ndarray):
                        result[key] = tensor
                return result
        except Exception:
            pass

        raise RuntimeError(
            "Could not deserialize payload (tried numpy and torch formats)"
        )

    def get_payload_torch(self) -> Any:
        """Deserialize payload as a PyTorch state dict.

        Returns:
            Dict of tensor name -> torch.Tensor.

        Raises:
            RuntimeError: If PyTorch is not available or deserialization fails.
        """
        import io
        try:
            import torch
            return torch.load(
                io.BytesIO(self._payload),
                map_location="cpu",
                weights_only=False,
            )
        except ImportError:
            raise RuntimeError("PyTorch is required for get_payload_torch()")
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize torch payload: {e}")

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def export_payload(self, output_path: str, fmt: str = "safetensors") -> str:
        """Export the payload to a standard LoRA format.

        Args:
            output_path: Path for the output file.
            fmt: Format to export ("safetensors", "pytorch", "numpy").

        Returns:
            Absolute path to the exported file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "safetensors":
            # Write raw payload (already safetensors in TGSP v1.1)
            with open(out, "wb") as f:
                f.write(self._payload)

        elif fmt == "pytorch":
            state_dict = self.get_payload_torch()
            import torch
            torch.save(state_dict, str(out))

        elif fmt == "numpy":
            arrays = self.get_payload_numpy()
            np.savez_compressed(str(out), **arrays)

        else:
            raise ValueError(f"Unknown format: {fmt}")

        abs_path = str(out.resolve())
        logger.info("Exported payload to %s (format=%s)", abs_path, fmt)
        return abs_path

    def export_manifest(self, output_path: str) -> str:
        """Export the manifest as a standalone JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with open(out, "w") as f:
            json.dump(self._manifest.raw, f, indent=2)

        abs_path = str(out.resolve())
        logger.info("Exported manifest to %s", abs_path)
        return abs_path

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TGSPAdapter(name={self.name!r}, id={self.adapter_id!r}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"domain={self.domain!r}, payload={self.payload_size} bytes)"
        )
