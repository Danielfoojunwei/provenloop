"""TGSPBuilder — programmatic TGSP file creation.

Provides a fluent builder API for creating .tgsp files from LoRA
weights with full manifest, signatures, and RVUv2 safety records.

Usage:
    builder = TGSPBuilder("my-adapter")
    builder.set_lora_config(rank=30, alpha=64, target_modules=["q_proj", "v_proj"])
    builder.set_model(architecture="sparse_moe", base_model="Qwen/Qwen2.5-1.5B")
    builder.set_creator(name="Alice", organization="TenSafe", email="alice@tensafe.io")
    builder.set_skill(description="Finance Q&A", triggers=["banking", "investment"])
    builder.set_payload(weights_path="adapter.safetensors")
    tgsp_path = builder.build("output/my-adapter.tgsp")
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# TGSP binary format constants
TGSP_MAGIC = b"TGSP\x01\x00"
TGSP_HEADER_LEN = 10


# ---------------------------------------------------------------------------
# Builder configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LoraConfig:
    """LoRA configuration for the adapter."""
    rank: int = 30
    alpha: float = 64.0
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.0
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    simd_slots: int = 8192
    cols_per_ct: int = 5
    batches: int = 6


@dataclass
class ModelConfig:
    """Target model configuration."""
    architecture: str = "sparse_moe"
    base_model: str = ""
    num_experts: int = 1
    experts_per_token: int = 1


@dataclass
class CreatorConfig:
    """Creator identity."""
    name: str = ""
    organization: str = ""
    email: str = ""
    public_key_fingerprint: str = ""
    verified: bool = False


@dataclass
class SkillConfig:
    """Skill description for the adapter."""
    description: str = ""
    triggers: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    input_format: str = "text"
    output_format: str = "text"
    quality_score: float = 0.0
    compliance: List[str] = field(default_factory=list)
    composable_with: List[str] = field(default_factory=list)


@dataclass
class RVUSafetyRecord:
    """RVUv2 safety screening record."""
    version: str = "RVUv2"
    layers: List[str] = field(
        default_factory=lambda: ["allowlist", "svd_analysis", "mahalanobis_ood"]
    )
    screening_passed: bool = False
    screening_timestamp: str = ""
    screening_hash: str = ""


@dataclass
class SignatureConfig:
    """Cryptographic signatures."""
    ed25519: str = ""  # Base64-encoded
    dilithium3: str = ""  # Base64-encoded
    signed_fields: List[str] = field(
        default_factory=lambda: ["integrity", "creator", "rvu_safety", "lora_config"]
    )


# ---------------------------------------------------------------------------
# TGSPBuilder
# ---------------------------------------------------------------------------

class TGSPBuilder:
    """Programmatic builder for .tgsp files.

    Constructs a TGSP package with proper binary format, manifest,
    and optional cryptographic signatures.

    The builder validates all inputs before writing the file and
    raises ValueError for invalid configurations.
    """

    def __init__(self, adapter_name: str, adapter_id: Optional[str] = None):
        self._name = adapter_name
        self._adapter_id = adapter_id or hashlib.sha256(
            adapter_name.encode()
        ).hexdigest()[:16]
        self._version = "1.0.0"
        self._format_version = "1.1"
        self._domain = "general"
        self._license = "commercial"
        self._price_per_1k_tokens = 0.0
        self._usage_metering = True

        # Configs
        self._lora = LoraConfig()
        self._model = ModelConfig()
        self._creator = CreatorConfig()
        self._skill = SkillConfig()
        self._rvu = RVUSafetyRecord()
        self._signatures = SignatureConfig()

        # Payload
        self._payload: Optional[bytes] = None
        self._metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Fluent setters
    # ------------------------------------------------------------------

    def set_version(self, version: str) -> "TGSPBuilder":
        """Set the adapter version string."""
        self._version = version
        return self

    def set_domain(self, domain: str) -> "TGSPBuilder":
        """Set the domain (e.g. 'finance', 'healthcare', 'general')."""
        self._domain = domain
        return self

    def set_license(self, license_type: str) -> "TGSPBuilder":
        """Set the license type."""
        self._license = license_type
        return self

    def set_pricing(
        self, price_per_1k_tokens: float, metering: bool = True
    ) -> "TGSPBuilder":
        """Set per-token pricing for marketplace."""
        self._price_per_1k_tokens = price_per_1k_tokens
        self._usage_metering = metering
        return self

    def set_lora_config(
        self,
        rank: int = 30,
        alpha: float = 64.0,
        target_modules: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "TGSPBuilder":
        """Set LoRA configuration."""
        self._lora.rank = rank
        self._lora.alpha = alpha
        if target_modules:
            self._lora.target_modules = target_modules
        for k, v in kwargs.items():
            if hasattr(self._lora, k):
                setattr(self._lora, k, v)
        return self

    def set_model(
        self,
        architecture: str = "sparse_moe",
        base_model: str = "",
        num_experts: int = 1,
        experts_per_token: int = 1,
    ) -> "TGSPBuilder":
        """Set target model configuration."""
        self._model = ModelConfig(
            architecture=architecture,
            base_model=base_model,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
        )
        return self

    def set_creator(
        self,
        name: str,
        organization: str = "",
        email: str = "",
        public_key_fingerprint: str = "",
        verified: bool = False,
    ) -> "TGSPBuilder":
        """Set creator identity."""
        self._creator = CreatorConfig(
            name=name,
            organization=organization,
            email=email,
            public_key_fingerprint=public_key_fingerprint,
            verified=verified,
        )
        return self

    def set_skill(
        self,
        description: str,
        triggers: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "TGSPBuilder":
        """Set skill description."""
        self._skill.description = description
        if triggers:
            self._skill.triggers = triggers
        if capabilities:
            self._skill.capabilities = capabilities
        for k, v in kwargs.items():
            if hasattr(self._skill, k):
                setattr(self._skill, k, v)
        return self

    def set_rvu_safety(
        self,
        screening_passed: bool = True,
        screening_hash: str = "",
    ) -> "TGSPBuilder":
        """Set RVUv2 safety screening record."""
        self._rvu.screening_passed = screening_passed
        self._rvu.screening_timestamp = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        self._rvu.screening_hash = screening_hash
        return self

    def set_signatures(
        self,
        ed25519: str = "",
        dilithium3: str = "",
        signed_fields: Optional[List[str]] = None,
    ) -> "TGSPBuilder":
        """Set cryptographic signatures.

        In production, these would be computed by the signing tool.
        For testing, you can pass pre-computed base64 signatures.
        """
        if ed25519:
            self._signatures.ed25519 = ed25519
        if dilithium3:
            self._signatures.dilithium3 = dilithium3
        if signed_fields:
            self._signatures.signed_fields = signed_fields
        return self

    def set_metadata(self, **kwargs: Any) -> "TGSPBuilder":
        """Set additional metadata fields."""
        self._metadata.update(kwargs)
        return self

    # ------------------------------------------------------------------
    # Payload
    # ------------------------------------------------------------------

    def set_payload_bytes(self, payload: bytes) -> "TGSPBuilder":
        """Set the adapter payload from raw bytes."""
        self._payload = payload
        return self

    def set_payload_file(self, path: str) -> "TGSPBuilder":
        """Set the adapter payload from a file on disk."""
        with open(path, "rb") as f:
            self._payload = f.read()
        return self

    def set_payload_numpy(
        self, lora_a: np.ndarray, lora_b: np.ndarray
    ) -> "TGSPBuilder":
        """Set the adapter payload from numpy LoRA weight matrices.

        Creates a simple serialized format with the A and B matrices.
        """
        buf = io.BytesIO()
        np.savez_compressed(buf, lora_a=lora_a, lora_b=lora_b)
        self._payload = buf.getvalue()
        return self

    def set_payload_torch(self, state_dict: Any) -> "TGSPBuilder":
        """Set payload from a PyTorch state dict.

        Args:
            state_dict: dict of tensor name -> tensor value.
        """
        try:
            import torch
            buf = io.BytesIO()
            torch.save(state_dict, buf)
            self._payload = buf.getvalue()
        except ImportError:
            raise RuntimeError("PyTorch is required for set_payload_torch()")
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _validate(self) -> List[str]:
        """Validate the builder configuration. Returns list of errors."""
        errors = []

        if not self._name:
            errors.append("Adapter name is required")

        if self._payload is None:
            errors.append("Payload is required (call set_payload_*)")

        if self._lora.rank < 1 or self._lora.rank > 256:
            errors.append(f"LoRA rank must be 1-256, got {self._lora.rank}")

        if not self._lora.target_modules:
            errors.append("At least one target module is required")

        if not self._creator.name:
            errors.append("Creator name is required")

        return errors

    def _build_manifest(self, payload_hash: str) -> Dict[str, Any]:
        """Build the JSON manifest."""
        manifest: Dict[str, Any] = {
            "format": "TGSP",
            "version": self._format_version,
            "name": self._name,
            "adapter_id": self._adapter_id,
            "model_name": self._name,
            "model_version": self._version,
            "domain": self._domain,
            "model": {
                "architecture": self._model.architecture,
                "base_model": self._model.base_model,
                "num_experts": self._model.num_experts,
                "experts_per_token": self._model.experts_per_token,
            },
            "lora_config": {
                "rank": self._lora.rank,
                "alpha": self._lora.alpha,
                "target_modules": self._lora.target_modules,
                "lora_dropout": self._lora.lora_dropout,
                "bias": self._lora.bias,
                "task_type": self._lora.task_type,
                "simd_slots": self._lora.simd_slots,
                "cols_per_ct": self._lora.cols_per_ct,
                "batches": self._lora.batches,
            },
            "skill": {
                "description": self._skill.description,
                "triggers": self._skill.triggers,
                "capabilities": self._skill.capabilities,
                "input_format": self._skill.input_format,
                "output_format": self._skill.output_format,
                "quality_score": self._skill.quality_score,
                "compliance": self._skill.compliance,
                "composable_with": self._skill.composable_with,
            },
            "rvu_safety": {
                "version": self._rvu.version,
                "layers": self._rvu.layers,
                "screening_passed": self._rvu.screening_passed,
                "screening_timestamp": self._rvu.screening_timestamp,
                "screening_hash": self._rvu.screening_hash,
            },
            "creator": {
                "name": self._creator.name,
                "organization": self._creator.organization,
                "email": self._creator.email,
                "public_key_fingerprint": self._creator.public_key_fingerprint,
                "verified": self._creator.verified,
            },
            "integrity": {
                "payload_hash": payload_hash,
                "manifest_hash": "",  # Filled after serialization
                "hash_algorithm": "SHA-256",
            },
            "signatures": {
                "ed25519": self._signatures.ed25519,
                "dilithium3": self._signatures.dilithium3,
                "signed_fields": self._signatures.signed_fields,
            },
            # Marketplace fields
            "license": self._license,
            "price_per_1k_tokens": self._price_per_1k_tokens,
            "usage_metering": self._usage_metering,
            "payload_size": len(self._payload) if self._payload else 0,
            "payload_hash": payload_hash,
            "metadata": {
                "domain": self._domain,
                "description": self._skill.description,
                "tags": self._skill.triggers[:5],
                **self._metadata,
            },
        }

        return manifest

    def build(self, output_path: str) -> str:
        """Build the .tgsp file and write to disk.

        Args:
            output_path: Path for the output .tgsp file.

        Returns:
            Absolute path to the created file.

        Raises:
            ValueError: If validation fails.
        """
        errors = self._validate()
        if errors:
            raise ValueError(
                f"TGSP build validation failed: {'; '.join(errors)}"
            )

        assert self._payload is not None  # Guaranteed by _validate

        # Compute payload hash
        payload_hash = hashlib.sha256(self._payload).hexdigest()

        # Build manifest
        manifest = self._build_manifest(payload_hash)

        # Compute manifest hash (before adding it to the manifest)
        manifest_for_hash = {k: v for k, v in manifest.items() if k != "integrity"}
        manifest_hash = hashlib.sha256(
            json.dumps(manifest_for_hash, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest["integrity"]["manifest_hash"] = manifest_hash

        # Serialize manifest
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

        # Assemble TGSP binary
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "wb") as f:
            f.write(TGSP_MAGIC)                                  # 6 bytes
            f.write(struct.pack("<I", len(manifest_bytes)))      # 4 bytes
            f.write(manifest_bytes)                               # variable
            f.write(self._payload)                                # remainder

        abs_path = str(out_path.resolve())
        logger.info(
            "TGSP built: %s (%d bytes manifest, %d bytes payload, hash=%s)",
            abs_path, len(manifest_bytes), len(self._payload), payload_hash[:16],
        )
        return abs_path

    def build_bytes(self) -> bytes:
        """Build the .tgsp file in memory and return as bytes.

        Returns:
            Complete TGSP binary data.
        """
        errors = self._validate()
        if errors:
            raise ValueError(
                f"TGSP build validation failed: {'; '.join(errors)}"
            )

        assert self._payload is not None

        payload_hash = hashlib.sha256(self._payload).hexdigest()
        manifest = self._build_manifest(payload_hash)

        manifest_for_hash = {k: v for k, v in manifest.items() if k != "integrity"}
        manifest_hash = hashlib.sha256(
            json.dumps(manifest_for_hash, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest["integrity"]["manifest_hash"] = manifest_hash

        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

        buf = io.BytesIO()
        buf.write(TGSP_MAGIC)
        buf.write(struct.pack("<I", len(manifest_bytes)))
        buf.write(manifest_bytes)
        buf.write(self._payload)
        return buf.getvalue()
