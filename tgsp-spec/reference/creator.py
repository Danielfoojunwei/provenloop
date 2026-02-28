"""
TGSP Reference Creator

Creates .tgsp files from LoRA weights (safetensors) and metadata.
Handles hashing, signing (Ed25519 + Dilithium3 stub), and binary assembly.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

from .parser import (
    TGSP_MAGIC,
    MAGIC_LEN,
    MANIFEST_LEN_SIZE,
    Dilithium3Stub,
    canonical_json,
)


# ---------------------------------------------------------------------------
# Data classes for structured input
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Describes the base model."""

    architecture: str = "sparse_moe"
    base_model: str = ""
    num_experts: int = 8
    experts_per_token: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "base_model": self.base_model,
            "num_experts": self.num_experts,
            "experts_per_token": self.experts_per_token,
        }


@dataclass
class LoraConfig:
    """LoRA hyperparameters and FHE configuration."""

    rank: int = 30
    alpha: float = 64
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    simd_slots: int = 8192
    cols_per_ct: int = 5
    batches: int = 6

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "simd_slots": self.simd_slots,
            "cols_per_ct": self.cols_per_ct,
            "batches": self.batches,
        }


@dataclass
class SkillInfo:
    """Adapter skill description."""

    description: str = ""
    triggers: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    input_format: str = "text"
    output_format: str = "text"
    quality_score: float = 0.0
    compliance: list[str] = field(default_factory=list)
    composable_with: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "triggers": self.triggers,
            "capabilities": self.capabilities,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "quality_score": self.quality_score,
            "compliance": self.compliance,
            "composable_with": self.composable_with,
        }


@dataclass
class RVUSafetyResult:
    """RVUv2 safety screening results."""

    layers: list[str] = field(
        default_factory=lambda: ["allowlist", "svd_analysis", "mahalanobis_ood"]
    )
    screening_passed: bool = True
    screening_timestamp: str = ""
    screening_hash: str = ""

    def __post_init__(self) -> None:
        if not self.screening_timestamp:
            self.screening_timestamp = datetime.now(timezone.utc).isoformat()
        if not self.screening_hash:
            # Generate a hash based on the screening results
            content = json.dumps(
                {
                    "layers": self.layers,
                    "passed": self.screening_passed,
                    "timestamp": self.screening_timestamp,
                },
                sort_keys=True,
            )
            self.screening_hash = hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "RVUv2",
            "layers": self.layers,
            "screening_passed": self.screening_passed,
            "screening_timestamp": self.screening_timestamp,
            "screening_hash": self.screening_hash,
        }


@dataclass
class CreatorIdentity:
    """Creator provenance information."""

    name: str = ""
    organization: str = ""
    email: str = ""
    public_key_fingerprint: str = ""
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "organization": self.organization,
            "email": self.email,
            "public_key_fingerprint": self.public_key_fingerprint,
            "verified": self.verified,
        }


# ---------------------------------------------------------------------------
# Key fingerprint helper
# ---------------------------------------------------------------------------


def compute_public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    """
    Compute the SHA-256 fingerprint of an Ed25519 public key.

    Returns the hex digest of SHA-256(raw_public_key_bytes).
    """
    raw = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Creator
# ---------------------------------------------------------------------------


class TGSPCreator:
    """
    Creates .tgsp files from LoRA weights and metadata.

    Usage::

        creator = TGSPCreator(
            name="my-adapter",
            domain="medical",
            model=ModelInfo(base_model="mistralai/Mixtral-8x7B-v0.1"),
            lora_config=LoraConfig(rank=30, alpha=64),
            skill=SkillInfo(description="Medical Q&A adapter"),
            rvu_safety=RVUSafetyResult(),
            creator_identity=CreatorIdentity(name="Alice", organization="TenSafe"),
        )
        creator.create_from_safetensors(
            weights_path="adapter.safetensors",
            output_path="adapter.tgsp",
            private_key=ed25519_private_key,
        )
    """

    def __init__(
        self,
        name: str,
        domain: str,
        model: ModelInfo,
        lora_config: LoraConfig,
        skill: SkillInfo,
        rvu_safety: RVUSafetyResult,
        creator_identity: CreatorIdentity,
    ) -> None:
        self.name = name
        self.domain = domain
        self.model = model
        self.lora_config = lora_config
        self.skill = skill
        self.rvu_safety = rvu_safety
        self.creator_identity = creator_identity

    # ------------------------------------------------------------------
    # Payload loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_payload(path: str | Path) -> bytes:
        """
        Load a LoRA payload from a safetensors file or a directory
        containing a single safetensors file.

        Returns the raw bytes.
        """
        path = Path(path)
        if path.is_dir():
            st_files = list(path.glob("*.safetensors"))
            if not st_files:
                raise FileNotFoundError(
                    f"No .safetensors files found in {path}"
                )
            if len(st_files) > 1:
                raise ValueError(
                    f"Multiple .safetensors files in {path}. "
                    "Please specify a single file."
                )
            path = st_files[0]

        if not path.exists():
            raise FileNotFoundError(f"Payload file not found: {path}")

        return path.read_bytes()

    # ------------------------------------------------------------------
    # Manifest building
    # ------------------------------------------------------------------

    def _build_manifest_without_signatures(
        self,
        payload_hash: str,
    ) -> dict[str, Any]:
        """
        Build the manifest dict with everything except the signatures
        block.
        """
        manifest = {
            "format": "TGSP",
            "version": "1.1",
            "name": self.name,
            "domain": self.domain,
            "model": self.model.to_dict(),
            "lora_config": self.lora_config.to_dict(),
            "skill": self.skill.to_dict(),
            "rvu_safety": self.rvu_safety.to_dict(),
            "creator": self.creator_identity.to_dict(),
            "integrity": {
                "payload_hash": payload_hash,
                "manifest_hash": "",  # placeholder, filled below
                "hash_algorithm": "SHA-256",
            },
        }

        # Compute manifest hash over the manifest with manifest_hash set
        # to empty string (both creator and verifier use this convention).
        manifest_hash = hashlib.sha256(canonical_json(manifest)).hexdigest()
        manifest["integrity"]["manifest_hash"] = manifest_hash

        return manifest

    # ------------------------------------------------------------------
    # Signing
    # ------------------------------------------------------------------

    @staticmethod
    def _sign_ed25519(
        private_key: Ed25519PrivateKey,
        message: bytes,
    ) -> str:
        """Sign *message* with Ed25519 and return base64-encoded signature."""
        sig = private_key.sign(message)
        return base64.b64encode(sig).decode("ascii")

    @staticmethod
    def _sign_dilithium3_stub(
        private_key_bytes: bytes,
        message: bytes,
    ) -> str:
        """
        Create a stub Dilithium3 signature.

        In production, replace with real Dilithium3 signing via tensafe-pqc.
        The stub creates: MARKER + SHA-256(public_key_bytes || message).
        For the stub, we treat private_key_bytes as the "public key" as well.
        """
        sig = (
            Dilithium3Stub.STUB_MARKER
            + hashlib.sha256(private_key_bytes + message).digest()
        )
        return base64.b64encode(sig).decode("ascii")

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def create(
        self,
        payload: bytes,
        private_key: Ed25519PrivateKey,
        dilithium3_key_bytes: Optional[bytes] = None,
        signed_fields: Optional[list[str]] = None,
    ) -> bytes:
        """
        Assemble a complete .tgsp file from *payload* bytes.

        Parameters
        ----------
        payload : bytes
            Raw LoRA weight bytes (safetensors format).
        private_key : Ed25519PrivateKey
            Ed25519 private key for signing.
        dilithium3_key_bytes : bytes, optional
            Bytes used as the Dilithium3 stub key.  If ``None``, the
            Ed25519 public key bytes are used.
        signed_fields : list[str], optional
            Manifest fields to sign.  Defaults to
            ``["integrity", "creator", "rvu_safety", "lora_config"]``.

        Returns
        -------
        bytes
            The assembled .tgsp file.
        """
        if signed_fields is None:
            signed_fields = ["integrity", "creator", "rvu_safety", "lora_config"]

        # Compute payload hash
        payload_hash = hashlib.sha256(payload).hexdigest()

        # Set creator fingerprint from the key
        public_key = private_key.public_key()
        self.creator_identity.public_key_fingerprint = (
            compute_public_key_fingerprint(public_key)
        )

        # Build manifest (no signatures yet)
        manifest = self._build_manifest_without_signatures(payload_hash)

        # Build the data to sign
        signing_data_obj = {f: manifest[f] for f in sorted(signed_fields)}
        signing_data = canonical_json(signing_data_obj)

        # Ed25519 signature
        ed25519_sig = self._sign_ed25519(private_key, signing_data)

        # Dilithium3 stub signature
        if dilithium3_key_bytes is None:
            dilithium3_key_bytes = public_key.public_bytes(
                Encoding.Raw, PublicFormat.Raw
            )
        dilithium3_sig = self._sign_dilithium3_stub(
            dilithium3_key_bytes, signing_data
        )

        # Add signatures to manifest
        manifest["signatures"] = {
            "ed25519": ed25519_sig,
            "dilithium3": dilithium3_sig,
            "signed_fields": signed_fields,
        }

        # Encode manifest as JSON
        manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")

        # Assemble binary
        manifest_len = struct.pack("<I", len(manifest_bytes))
        return TGSP_MAGIC + manifest_len + manifest_bytes + payload

    def create_from_safetensors(
        self,
        weights_path: str | Path,
        output_path: str | Path,
        private_key: Ed25519PrivateKey,
        dilithium3_key_bytes: Optional[bytes] = None,
        signed_fields: Optional[list[str]] = None,
    ) -> Path:
        """
        Create a .tgsp file from a safetensors file or directory.

        Returns the path to the written .tgsp file.
        """
        payload = self.load_payload(weights_path)
        tgsp_bytes = self.create(
            payload, private_key, dilithium3_key_bytes, signed_fields
        )
        output_path = Path(output_path)
        output_path.write_bytes(tgsp_bytes)
        return output_path

    def wrap_lora(
        self,
        lora_bytes: bytes,
        private_key: Ed25519PrivateKey,
        dilithium3_key_bytes: Optional[bytes] = None,
    ) -> bytes:
        """
        Import existing LoRA weights (as raw bytes) and wrap them in a
        TGSP trust chain.

        This is the simplest entry point for wrapping an already-loaded
        LoRA payload.

        Returns the assembled .tgsp bytes.
        """
        return self.create(lora_bytes, private_key, dilithium3_key_bytes)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Simple CLI: create a .tgsp file from a safetensors payload.

    Usage:
        python creator.py <weights.safetensors> <output.tgsp> \\
            --name "my-adapter" --domain "general"

    A fresh Ed25519 key is generated for demonstration purposes.
    """
    if len(sys.argv) < 3:
        print(
            "Usage: python creator.py <weights.safetensors> <output.tgsp> "
            "[--name NAME] [--domain DOMAIN]"
        )
        sys.exit(1)

    weights_path = sys.argv[1]
    output_path = sys.argv[2]

    # Parse optional flags
    name = "unnamed-adapter"
    domain = "general"
    for i, arg in enumerate(sys.argv):
        if arg == "--name" and i + 1 < len(sys.argv):
            name = sys.argv[i + 1]
        elif arg == "--domain" and i + 1 < len(sys.argv):
            domain = sys.argv[i + 1]

    # Generate a fresh key for demo
    private_key = Ed25519PrivateKey.generate()

    creator = TGSPCreator(
        name=name,
        domain=domain,
        model=ModelInfo(
            architecture="sparse_moe",
            base_model="mistralai/Mixtral-8x7B-v0.1",
            num_experts=8,
            experts_per_token=2,
        ),
        lora_config=LoraConfig(),
        skill=SkillInfo(description=f"{name} adapter"),
        rvu_safety=RVUSafetyResult(),
        creator_identity=CreatorIdentity(
            name="CLI User",
            organization="Unknown",
            email="user@example.com",
            verified=False,
        ),
    )

    out = creator.create_from_safetensors(weights_path, output_path, private_key)
    print(f"Created TGSP file: {out}")
    print(f"  Ed25519 public key fingerprint: "
          f"{compute_public_key_fingerprint(private_key.public_key())}")


if __name__ == "__main__":
    main()
