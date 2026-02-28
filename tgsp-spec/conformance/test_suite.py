"""
TGSP Conformance Test Suite

Pytest-based tests covering:
  - Magic bytes parsing
  - Manifest schema validation
  - Hash verification (valid + tampered)
  - Signature verification (valid + invalid)
  - LoraConfig validation
  - Creator identity verification
  - RVUv2 screening status
  - Round-trip: create -> parse -> verify -> export -> reimport

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

import sys
import os

# Ensure the parent package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reference.parser import (
    TGSP_MAGIC,
    MAGIC_LEN,
    MANIFEST_LEN_SIZE,
    Dilithium3Stub,
    TGSPFile,
    TGSPParser,
    canonical_json,
    build_signing_data,
)
from reference.creator import (
    TGSPCreator,
    ModelInfo,
    LoraConfig,
    SkillInfo,
    RVUSafetyResult,
    CreatorIdentity,
    compute_public_key_fingerprint,
)
from reference.verifier import (
    TGSPVerifier,
    CheckStatus,
    VerificationReport,
    validate_lora_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ed25519_keypair():
    """Generate a fresh Ed25519 key pair."""
    private = Ed25519PrivateKey.generate()
    public = private.public_key()
    return private, public


@pytest.fixture
def sample_payload() -> bytes:
    """
    A minimal safetensors-like payload for testing.

    This is not a valid safetensors file, but has enough structure
    to serve as a payload blob for TGSP format testing.
    """
    # Minimal safetensors-like header + dummy weight data
    header = {"__metadata__": {"format": "pt"}, "weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}}
    header_json = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<Q", len(header_json))
    weight_data = b"\x00" * 64  # 4x4 float32 zeros
    return header_len + header_json + weight_data


@pytest.fixture
def default_creator_identity() -> CreatorIdentity:
    return CreatorIdentity(
        name="Test Creator",
        organization="TenSafe Labs",
        email="test@tensafe.dev",
        verified=True,
    )


@pytest.fixture
def default_lora_config() -> LoraConfig:
    return LoraConfig(
        rank=30,
        alpha=64,
        target_modules=["q_proj", "v_proj", "gate"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        simd_slots=8192,
        cols_per_ct=5,
        batches=6,
    )


@pytest.fixture
def default_model_info() -> ModelInfo:
    return ModelInfo(
        architecture="sparse_moe",
        base_model="mistralai/Mixtral-8x7B-v0.1",
        num_experts=8,
        experts_per_token=2,
    )


@pytest.fixture
def default_skill_info() -> SkillInfo:
    return SkillInfo(
        description="Test adapter for conformance",
        triggers=["test", "conformance"],
        capabilities=["qa", "summarization"],
        input_format="text",
        output_format="text",
        quality_score=0.95,
        compliance=["SOC2"],
        composable_with=["base-adapter"],
    )


@pytest.fixture
def default_rvu_safety() -> RVUSafetyResult:
    return RVUSafetyResult(
        layers=["allowlist", "svd_analysis", "mahalanobis_ood"],
        screening_passed=True,
    )


@pytest.fixture
def tgsp_creator(
    default_model_info,
    default_lora_config,
    default_skill_info,
    default_rvu_safety,
    default_creator_identity,
) -> TGSPCreator:
    """Create a fully configured TGSPCreator."""
    return TGSPCreator(
        name="test-adapter",
        domain="testing",
        model=default_model_info,
        lora_config=default_lora_config,
        skill=default_skill_info,
        rvu_safety=default_rvu_safety,
        creator_identity=default_creator_identity,
    )


@pytest.fixture
def sample_tgsp_bytes(tgsp_creator, sample_payload, ed25519_keypair) -> bytes:
    """Create a valid .tgsp file as bytes."""
    private_key, _ = ed25519_keypair
    return tgsp_creator.create(sample_payload, private_key)


@pytest.fixture
def sample_tgsp(sample_tgsp_bytes) -> TGSPFile:
    """Parse a valid .tgsp file."""
    return TGSPParser.read_bytes(sample_tgsp_bytes)


# ===================================================================
# 1. Magic Bytes Parsing
# ===================================================================


class TestMagicBytes:
    """Tests for TGSP magic byte validation."""

    def test_valid_magic_bytes(self, sample_tgsp_bytes):
        """Valid TGSP file should parse without error."""
        tgsp = TGSPParser.read_bytes(sample_tgsp_bytes)
        assert tgsp.magic == TGSP_MAGIC

    def test_magic_bytes_content(self):
        """Magic bytes must be exactly TGSP\\x01\\x00."""
        assert TGSP_MAGIC == b"TGSP\x01\x00"
        assert len(TGSP_MAGIC) == 6
        assert TGSP_MAGIC[0:4] == b"TGSP"
        assert TGSP_MAGIC[4] == 0x01  # major version
        assert TGSP_MAGIC[5] == 0x00  # minor version

    def test_invalid_magic_raises(self):
        """Wrong magic bytes should raise ValueError."""
        bad_data = b"XXXX\x01\x00" + b"\x02\x00\x00\x00" + b"{}"
        with pytest.raises(ValueError, match="Invalid magic bytes"):
            TGSPParser.read_bytes(bad_data)

    def test_truncated_file_raises(self):
        """File too small for header should raise ValueError."""
        with pytest.raises(ValueError, match="File too small"):
            TGSPParser.read_bytes(b"TGSP")

    def test_empty_file_raises(self):
        """Empty file should raise ValueError."""
        with pytest.raises(ValueError, match="File too small"):
            TGSPParser.read_bytes(b"")

    def test_wrong_version_raises(self):
        """Wrong version bytes should raise ValueError."""
        bad_data = b"TGSP\x02\x00" + b"\x02\x00\x00\x00" + b"{}"
        with pytest.raises(ValueError, match="Invalid magic bytes"):
            TGSPParser.read_bytes(bad_data)


# ===================================================================
# 2. Manifest Schema Validation
# ===================================================================


class TestManifestSchema:
    """Tests for manifest JSON schema validation."""

    def test_manifest_is_valid_json(self, sample_tgsp):
        """Manifest must parse as valid JSON."""
        assert isinstance(sample_tgsp.manifest, dict)

    def test_manifest_has_required_fields(self, sample_tgsp):
        """Manifest must contain all required top-level fields."""
        required = [
            "format", "version", "name", "domain", "model",
            "lora_config", "skill", "rvu_safety", "creator",
            "integrity", "signatures",
        ]
        for field in required:
            assert field in sample_tgsp.manifest, f"Missing field: {field}"

    def test_format_is_tgsp(self, sample_tgsp):
        """format field must be 'TGSP'."""
        assert sample_tgsp.manifest["format"] == "TGSP"

    def test_version_is_1_1(self, sample_tgsp):
        """version field must be '1.1'."""
        assert sample_tgsp.manifest["version"] == "1.1"

    def test_model_fields(self, sample_tgsp):
        """model block must contain required fields."""
        model = sample_tgsp.manifest["model"]
        assert "architecture" in model
        assert "base_model" in model
        assert "num_experts" in model
        assert "experts_per_token" in model

    def test_integrity_fields(self, sample_tgsp):
        """integrity block must contain required fields."""
        integrity = sample_tgsp.manifest["integrity"]
        assert "payload_hash" in integrity
        assert "manifest_hash" in integrity
        assert integrity["hash_algorithm"] == "SHA-256"

    def test_signatures_fields(self, sample_tgsp):
        """signatures block must contain required fields."""
        sigs = sample_tgsp.manifest["signatures"]
        assert "ed25519" in sigs
        assert "dilithium3" in sigs
        assert "signed_fields" in sigs
        assert isinstance(sigs["signed_fields"], list)

    def test_invalid_manifest_json_raises(self):
        """Invalid JSON in manifest position should raise ValueError."""
        manifest_bytes = b"not valid json{{"
        manifest_len = struct.pack("<I", len(manifest_bytes))
        data = TGSP_MAGIC + manifest_len + manifest_bytes
        with pytest.raises(ValueError, match="Invalid manifest JSON"):
            TGSPParser.read_bytes(data)

    def test_manifest_length_overflow_raises(self):
        """Manifest length exceeding file size should raise ValueError."""
        manifest_bytes = b'{"format":"TGSP"}'
        # Claim manifest is much longer than it actually is
        manifest_len = struct.pack("<I", 999999)
        data = TGSP_MAGIC + manifest_len + manifest_bytes
        with pytest.raises(ValueError, match="Manifest length.*exceeds file size"):
            TGSPParser.read_bytes(data)


# ===================================================================
# 3. Hash Verification
# ===================================================================


class TestHashVerification:
    """Tests for SHA-256 hash verification (valid and tampered)."""

    def test_valid_payload_hash(self, sample_tgsp):
        """Payload hash should match for an untampered file."""
        assert TGSPParser.verify_payload_hash(sample_tgsp) is True

    def test_valid_manifest_hash(self, sample_tgsp):
        """Manifest hash should match for an untampered file."""
        assert TGSPParser.verify_manifest_hash(sample_tgsp) is True

    def test_tampered_payload_detected(self, sample_tgsp_bytes):
        """Modifying the payload should cause hash verification to fail."""
        # Parse, tamper the payload, re-check
        tgsp = TGSPParser.read_bytes(sample_tgsp_bytes)
        # Replace payload with different data
        tampered = TGSPFile(
            magic=tgsp.magic,
            manifest=tgsp.manifest,
            payload=b"TAMPERED_PAYLOAD_DATA",
            raw_manifest_bytes=tgsp.raw_manifest_bytes,
        )
        assert TGSPParser.verify_payload_hash(tampered) is False

    def test_tampered_manifest_detected(self, sample_tgsp):
        """Modifying a manifest field should cause manifest hash to fail."""
        import copy
        tampered_manifest = copy.deepcopy(sample_tgsp.manifest)
        tampered_manifest["name"] = "TAMPERED_NAME"
        tampered = TGSPFile(
            magic=sample_tgsp.magic,
            manifest=tampered_manifest,
            payload=sample_tgsp.payload,
            raw_manifest_bytes=sample_tgsp.raw_manifest_bytes,
        )
        assert TGSPParser.verify_manifest_hash(tampered) is False

    def test_payload_hash_is_sha256_hex(self, sample_tgsp):
        """Payload hash should be a 64-char hex string."""
        h = sample_tgsp.manifest["integrity"]["payload_hash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_manifest_hash_is_sha256_hex(self, sample_tgsp):
        """Manifest hash should be a 64-char hex string."""
        h = sample_tgsp.manifest["integrity"]["manifest_hash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ===================================================================
# 4. Signature Verification
# ===================================================================


class TestSignatureVerification:
    """Tests for Ed25519 and Dilithium3 signature verification."""

    def test_valid_ed25519_signature(self, sample_tgsp, ed25519_keypair):
        """Valid Ed25519 signature should verify."""
        _, public_key = ed25519_keypair
        assert TGSPParser.verify_ed25519(sample_tgsp, public_key) is True

    def test_invalid_ed25519_with_wrong_key(self, sample_tgsp):
        """Ed25519 signature should fail with a different key."""
        wrong_key = Ed25519PrivateKey.generate().public_key()
        assert TGSPParser.verify_ed25519(sample_tgsp, wrong_key) is False

    def test_invalid_ed25519_tampered_signature(
        self, sample_tgsp_bytes, ed25519_keypair
    ):
        """Tampered Ed25519 signature should fail verification."""
        tgsp = TGSPParser.read_bytes(sample_tgsp_bytes)
        # Corrupt the signature
        import copy
        tampered_manifest = copy.deepcopy(tgsp.manifest)
        tampered_manifest["signatures"]["ed25519"] = base64.b64encode(
            b"INVALID_SIG_DATA_THAT_IS_TOTALLY_WRONG_AND_64_BYTES_LONG!!!!!!!!"
        ).decode()
        tampered = TGSPFile(
            magic=tgsp.magic,
            manifest=tampered_manifest,
            payload=tgsp.payload,
            raw_manifest_bytes=tgsp.raw_manifest_bytes,
        )
        _, public_key = ed25519_keypair
        assert TGSPParser.verify_ed25519(tampered, public_key) is False

    def test_valid_dilithium3_stub(self, sample_tgsp, ed25519_keypair):
        """Valid Dilithium3 stub signature should verify."""
        _, public_key = ed25519_keypair
        pk_bytes = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        assert TGSPParser.verify_dilithium3(sample_tgsp, pk_bytes) is True

    def test_invalid_dilithium3_wrong_key(self, sample_tgsp):
        """Dilithium3 stub should fail with wrong key bytes."""
        wrong_bytes = b"totally_wrong_key_bytes_32_chars!"
        assert TGSPParser.verify_dilithium3(sample_tgsp, wrong_bytes) is False

    def test_signed_fields_include_required(self, sample_tgsp):
        """signed_fields must include integrity, creator, rvu_safety, lora_config."""
        signed = set(sample_tgsp.manifest["signatures"]["signed_fields"])
        required = {"integrity", "creator", "rvu_safety", "lora_config"}
        assert required.issubset(signed)

    def test_ed25519_signature_is_base64(self, sample_tgsp):
        """Ed25519 signature should be valid base64."""
        sig = sample_tgsp.manifest["signatures"]["ed25519"]
        decoded = base64.b64decode(sig)
        assert len(decoded) == 64  # Ed25519 signatures are 64 bytes

    def test_dilithium3_signature_is_base64(self, sample_tgsp):
        """Dilithium3 stub signature should be valid base64."""
        sig = sample_tgsp.manifest["signatures"]["dilithium3"]
        decoded = base64.b64decode(sig)
        assert len(decoded) > 0


# ===================================================================
# 5. LoraConfig Validation
# ===================================================================


class TestLoraConfigValidation:
    """Tests for LoRA configuration schema validation."""

    def test_valid_config(self, sample_tgsp):
        """Default LoraConfig should validate."""
        config = sample_tgsp.manifest["lora_config"]
        errors = validate_lora_config(config)
        assert errors == []

    def test_missing_required_field(self):
        """Missing a required field should produce an error."""
        config = {"rank": 30}  # Missing most fields
        errors = validate_lora_config(config)
        assert len(errors) > 0
        assert any("Missing required field" in e for e in errors)

    def test_invalid_rank_zero(self):
        """rank=0 should be rejected."""
        config = {
            "rank": 0, "alpha": 64, "target_modules": ["q_proj"],
            "lora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM",
            "simd_slots": 8192, "cols_per_ct": 5, "batches": 6,
        }
        errors = validate_lora_config(config)
        assert any("rank" in e for e in errors)

    def test_invalid_rank_negative(self):
        """Negative rank should be rejected."""
        config = {
            "rank": -5, "alpha": 64, "target_modules": ["q_proj"],
            "lora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM",
            "simd_slots": 8192, "cols_per_ct": 5, "batches": 6,
        }
        errors = validate_lora_config(config)
        assert any("rank" in e for e in errors)

    def test_invalid_rank_too_large(self):
        """rank > 1024 should be rejected."""
        config = {
            "rank": 2048, "alpha": 64, "target_modules": ["q_proj"],
            "lora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM",
            "simd_slots": 8192, "cols_per_ct": 5, "batches": 6,
        }
        errors = validate_lora_config(config)
        assert any("rank" in e for e in errors)

    def test_invalid_alpha_zero(self):
        """alpha=0 should be rejected."""
        config = {
            "rank": 30, "alpha": 0, "target_modules": ["q_proj"],
            "lora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM",
            "simd_slots": 8192, "cols_per_ct": 5, "batches": 6,
        }
        errors = validate_lora_config(config)
        assert any("alpha" in e for e in errors)

    def test_invalid_bias_value(self):
        """Invalid bias value should be rejected."""
        config = {
            "rank": 30, "alpha": 64, "target_modules": ["q_proj"],
            "lora_dropout": 0.0, "bias": "invalid_value", "task_type": "CAUSAL_LM",
            "simd_slots": 8192, "cols_per_ct": 5, "batches": 6,
        }
        errors = validate_lora_config(config)
        assert any("bias" in e for e in errors)

    def test_invalid_task_type(self):
        """Invalid task_type should be rejected."""
        config = {
            "rank": 30, "alpha": 64, "target_modules": ["q_proj"],
            "lora_dropout": 0.0, "bias": "none", "task_type": "INVALID",
            "simd_slots": 8192, "cols_per_ct": 5, "batches": 6,
        }
        errors = validate_lora_config(config)
        assert any("task_type" in e for e in errors)

    def test_empty_target_modules(self):
        """Empty target_modules should be rejected."""
        config = {
            "rank": 30, "alpha": 64, "target_modules": [],
            "lora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM",
            "simd_slots": 8192, "cols_per_ct": 5, "batches": 6,
        }
        errors = validate_lora_config(config)
        assert any("target_modules" in e for e in errors)

    def test_dropout_out_of_range(self):
        """lora_dropout > 1.0 should be rejected."""
        config = {
            "rank": 30, "alpha": 64, "target_modules": ["q_proj"],
            "lora_dropout": 1.5, "bias": "none", "task_type": "CAUSAL_LM",
            "simd_slots": 8192, "cols_per_ct": 5, "batches": 6,
        }
        errors = validate_lora_config(config)
        assert any("lora_dropout" in e for e in errors)

    def test_valid_all_task_types(self):
        """All valid task types should be accepted."""
        for tt in ["CAUSAL_LM", "SEQ_2_SEQ_LM", "TOKEN_CLS",
                    "SEQ_CLS", "QUESTION_ANS", "FEATURE_EXTRACTION"]:
            config = {
                "rank": 30, "alpha": 64, "target_modules": ["q_proj"],
                "lora_dropout": 0.0, "bias": "none", "task_type": tt,
                "simd_slots": 8192, "cols_per_ct": 5, "batches": 6,
            }
            errors = validate_lora_config(config)
            assert errors == [], f"task_type '{tt}' should be valid"

    def test_config_defaults(self, default_lora_config):
        """Default LoraConfig values should produce valid config."""
        config = default_lora_config.to_dict()
        errors = validate_lora_config(config)
        assert errors == []
        assert config["rank"] == 30
        assert config["alpha"] == 64
        assert config["simd_slots"] == 8192
        assert config["cols_per_ct"] == 5
        assert config["batches"] == 6


# ===================================================================
# 6. Creator Identity Verification
# ===================================================================


class TestCreatorIdentity:
    """Tests for creator identity verification."""

    def test_creator_fields_present(self, sample_tgsp):
        """Creator block must have all required fields."""
        creator = sample_tgsp.manifest["creator"]
        assert "name" in creator
        assert "organization" in creator
        assert "email" in creator
        assert "public_key_fingerprint" in creator
        assert "verified" in creator

    def test_fingerprint_matches_key(self, sample_tgsp, ed25519_keypair):
        """Creator fingerprint should match the signing key."""
        _, public_key = ed25519_keypair
        expected = compute_public_key_fingerprint(public_key)
        actual = sample_tgsp.manifest["creator"]["public_key_fingerprint"]
        assert actual == expected

    def test_fingerprint_is_sha256_hex(self, sample_tgsp):
        """Fingerprint should be a 64-char hex string."""
        fp = sample_tgsp.manifest["creator"]["public_key_fingerprint"]
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_wrong_fingerprint_detected(self, sample_tgsp, ed25519_keypair):
        """Verifier should detect mismatched key fingerprint."""
        _, public_key = ed25519_keypair
        import copy
        tampered = copy.deepcopy(sample_tgsp.manifest)
        tampered["creator"]["public_key_fingerprint"] = "a" * 64
        tampered_tgsp = TGSPFile(
            magic=sample_tgsp.magic,
            manifest=tampered,
            payload=sample_tgsp.payload,
            raw_manifest_bytes=sample_tgsp.raw_manifest_bytes,
        )
        verifier = TGSPVerifier(tampered_tgsp, ed25519_public_key=public_key)
        report = VerificationReport()
        verifier.check_creator_identity(report)
        assert any(
            c.status == CheckStatus.FAIL and "fingerprint" in c.message.lower()
            for c in report.checks
        )

    def test_verified_flag(self, sample_tgsp):
        """The verified flag should be a boolean."""
        assert isinstance(sample_tgsp.manifest["creator"]["verified"], bool)


# ===================================================================
# 7. RVUv2 Screening Status
# ===================================================================


class TestRVUSafety:
    """Tests for RVUv2 safety screening status."""

    def test_rvu_version(self, sample_tgsp):
        """RVU version must be 'RVUv2'."""
        assert sample_tgsp.manifest["rvu_safety"]["version"] == "RVUv2"

    def test_rvu_layers(self, sample_tgsp):
        """All three standard screening layers should be present."""
        layers = sample_tgsp.manifest["rvu_safety"]["layers"]
        assert "allowlist" in layers
        assert "svd_analysis" in layers
        assert "mahalanobis_ood" in layers

    def test_screening_passed(self, sample_tgsp):
        """screening_passed should be a boolean."""
        assert isinstance(
            sample_tgsp.manifest["rvu_safety"]["screening_passed"], bool
        )

    def test_screening_timestamp_present(self, sample_tgsp):
        """Screening timestamp should be a non-empty string."""
        ts = sample_tgsp.manifest["rvu_safety"]["screening_timestamp"]
        assert isinstance(ts, str)
        assert len(ts) > 0

    def test_screening_hash_format(self, sample_tgsp):
        """Screening hash should be a 64-char hex string."""
        h = sample_tgsp.manifest["rvu_safety"]["screening_hash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_failed_screening_detected(self):
        """Verifier should detect failed screening."""
        rvu = RVUSafetyResult(
            layers=["allowlist", "svd_analysis", "mahalanobis_ood"],
            screening_passed=False,
        )
        manifest = {
            "rvu_safety": rvu.to_dict(),
            "format": "TGSP",
            "version": "1.1",
        }
        tgsp = TGSPFile(
            magic=TGSP_MAGIC,
            manifest=manifest,
            payload=b"",
            raw_manifest_bytes=b"{}",
        )
        verifier = TGSPVerifier(tgsp)
        report = VerificationReport()
        verifier.check_rvu_safety(report)
        assert any(
            c.status == CheckStatus.FAIL and "NOT pass" in c.message
            for c in report.checks
        )

    def test_missing_layers_warning(self):
        """Missing standard layers should produce a warning."""
        manifest = {
            "rvu_safety": {
                "version": "RVUv2",
                "layers": ["allowlist"],  # Missing svd_analysis, mahalanobis_ood
                "screening_passed": True,
                "screening_timestamp": datetime.now(timezone.utc).isoformat(),
                "screening_hash": "a" * 64,
            },
            "format": "TGSP",
            "version": "1.1",
        }
        tgsp = TGSPFile(
            magic=TGSP_MAGIC, manifest=manifest, payload=b"",
            raw_manifest_bytes=b"{}",
        )
        verifier = TGSPVerifier(tgsp)
        report = VerificationReport()
        verifier.check_rvu_safety(report)
        assert any(c.status == CheckStatus.WARN for c in report.checks)


# ===================================================================
# 8. Full Verifier Report
# ===================================================================


class TestFullVerification:
    """Tests for the full verification pipeline."""

    def test_valid_file_passes(self, sample_tgsp, ed25519_keypair):
        """A valid TGSP file should pass all checks."""
        _, public_key = ed25519_keypair
        pk_bytes = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        verifier = TGSPVerifier(
            sample_tgsp,
            ed25519_public_key=public_key,
            dilithium3_public_key_bytes=pk_bytes,
        )
        report = verifier.verify()
        assert report.passed, f"Verification failed:\n{report}"
        assert report.num_failed == 0

    def test_report_string_format(self, sample_tgsp, ed25519_keypair):
        """Report string should contain key sections."""
        _, public_key = ed25519_keypair
        verifier = TGSPVerifier(sample_tgsp, ed25519_public_key=public_key)
        report = verifier.verify()
        report_str = str(report)
        assert "TGSP Verification Report" in report_str
        assert "Verdict:" in report_str

    def test_no_keys_still_checks_hashes(self, sample_tgsp):
        """Without keys, hash checks should still run."""
        verifier = TGSPVerifier(sample_tgsp)
        report = verifier.verify()
        # Hash checks should pass
        hash_checks = [
            c for c in report.checks
            if "Hash" in c.name
        ]
        assert all(c.status == CheckStatus.PASS for c in hash_checks)
        # Signature checks should be skipped
        sig_checks = [
            c for c in report.checks
            if "Signature" in c.name
        ]
        assert all(c.status == CheckStatus.SKIP for c in sig_checks)


# ===================================================================
# 9. Round-Trip: Create -> Parse -> Verify -> Export -> Reimport
# ===================================================================


class TestRoundTrip:
    """End-to-end round-trip tests."""

    def test_create_parse_verify(
        self, tgsp_creator, sample_payload, ed25519_keypair
    ):
        """Create a TGSP, parse it, and verify it."""
        private_key, public_key = ed25519_keypair

        # Create
        tgsp_bytes = tgsp_creator.create(sample_payload, private_key)

        # Parse
        tgsp = TGSPParser.read_bytes(tgsp_bytes)
        assert tgsp.manifest["format"] == "TGSP"
        assert tgsp.manifest["name"] == "test-adapter"

        # Verify
        pk_bytes = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        verifier = TGSPVerifier(
            tgsp,
            ed25519_public_key=public_key,
            dilithium3_public_key_bytes=pk_bytes,
        )
        report = verifier.verify()
        assert report.passed, f"Verification failed:\n{report}"

    def test_export_reimport(
        self, tgsp_creator, sample_payload, ed25519_keypair
    ):
        """Export payload to safetensors then reimport into a new TGSP."""
        private_key, public_key = ed25519_keypair

        # Create original
        tgsp_bytes = tgsp_creator.create(sample_payload, private_key)
        tgsp = TGSPParser.read_bytes(tgsp_bytes)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Export payload
            export_path = Path(tmpdir) / "exported.safetensors"
            TGSPParser.export_safetensors(tgsp, export_path)
            assert export_path.exists()

            # Verify exported bytes match original payload
            exported_bytes = export_path.read_bytes()
            assert exported_bytes == sample_payload

            # Reimport: create a new TGSP from exported weights
            new_key = Ed25519PrivateKey.generate()
            tgsp_creator2 = TGSPCreator(
                name="reimported-adapter",
                domain="testing",
                model=ModelInfo(
                    architecture="sparse_moe",
                    base_model="mistralai/Mixtral-8x7B-v0.1",
                ),
                lora_config=LoraConfig(),
                skill=SkillInfo(description="Reimported adapter"),
                rvu_safety=RVUSafetyResult(),
                creator_identity=CreatorIdentity(
                    name="Re-importer",
                    organization="Test",
                    email="re@test.com",
                ),
            )
            reimport_path = Path(tmpdir) / "reimported.tgsp"
            tgsp_creator2.create_from_safetensors(
                export_path, reimport_path, new_key
            )
            assert reimport_path.exists()

            # Parse and verify reimported file
            reimported = TGSPParser.read(reimport_path)
            assert reimported.manifest["name"] == "reimported-adapter"
            assert reimported.payload == sample_payload

            # Verify reimported
            new_public = new_key.public_key()
            new_pk_bytes = new_public.public_bytes(
                Encoding.Raw, PublicFormat.Raw
            )
            verifier = TGSPVerifier(
                reimported,
                ed25519_public_key=new_public,
                dilithium3_public_key_bytes=new_pk_bytes,
            )
            report = verifier.verify()
            assert report.passed, f"Reimport verification failed:\n{report}"

    def test_file_round_trip(
        self, tgsp_creator, sample_payload, ed25519_keypair
    ):
        """Write to disk and read back, ensuring byte-level fidelity."""
        private_key, public_key = ed25519_keypair

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write payload to disk as safetensors
            payload_path = Path(tmpdir) / "adapter.safetensors"
            payload_path.write_bytes(sample_payload)

            # Create TGSP from file
            tgsp_path = Path(tmpdir) / "adapter.tgsp"
            tgsp_creator.create_from_safetensors(
                payload_path, tgsp_path, private_key
            )
            assert tgsp_path.exists()

            # Read back
            tgsp = TGSPParser.read(tgsp_path)
            assert tgsp.magic == TGSP_MAGIC
            assert tgsp.payload == sample_payload
            assert TGSPParser.verify_payload_hash(tgsp) is True
            assert TGSPParser.verify_manifest_hash(tgsp) is True

    def test_wrap_lora_convenience(
        self, tgsp_creator, sample_payload, ed25519_keypair
    ):
        """Test the wrap_lora convenience method."""
        private_key, public_key = ed25519_keypair

        tgsp_bytes = tgsp_creator.wrap_lora(sample_payload, private_key)
        tgsp = TGSPParser.read_bytes(tgsp_bytes)

        assert tgsp.manifest["format"] == "TGSP"
        assert tgsp.payload == sample_payload
        assert TGSPParser.verify_payload_hash(tgsp) is True

    def test_display_info(self, sample_tgsp):
        """display_info should return a non-empty string with key info."""
        info = TGSPParser.display_info(sample_tgsp)
        assert "TGSP File Summary" in info
        assert "test-adapter" in info
        assert "sparse_moe" in info
        assert "RVUv2" in info


# ===================================================================
# 10. Dilithium3 Stub
# ===================================================================


class TestDilithium3Stub:
    """Tests for the Dilithium3 stub implementation."""

    def test_stub_sign_verify_roundtrip(self):
        """Stub signature should verify with the same key."""
        key = b"test_key_for_dilithium3_stub_000"
        message = b"hello world"
        sig = (
            Dilithium3Stub.STUB_MARKER
            + hashlib.sha256(key + message).digest()
        )
        assert Dilithium3Stub.verify(key, sig, message) is True

    def test_stub_wrong_key_fails(self):
        """Stub verification should fail with a different key."""
        key = b"correct_key_bytes_for_testing_00"
        wrong = b"wrong_key_bytes_for_testing_0000"
        message = b"hello world"
        sig = (
            Dilithium3Stub.STUB_MARKER
            + hashlib.sha256(key + message).digest()
        )
        assert Dilithium3Stub.verify(wrong, sig, message) is False

    def test_stub_wrong_message_fails(self):
        """Stub verification should fail with a different message."""
        key = b"test_key_for_dilithium3_stub_000"
        message = b"hello world"
        sig = (
            Dilithium3Stub.STUB_MARKER
            + hashlib.sha256(key + message).digest()
        )
        assert Dilithium3Stub.verify(key, sig, b"different message") is False

    def test_non_stub_signature_fails(self):
        """Non-stub signatures should fail verification."""
        assert Dilithium3Stub.verify(b"key", b"random_data", b"msg") is False


# ===================================================================
# 11. Canonical JSON
# ===================================================================


class TestCanonicalJSON:
    """Tests for canonical JSON serialization."""

    def test_sorted_keys(self):
        """Keys should be sorted."""
        obj = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(obj)
        assert result == b'{"a":2,"m":3,"z":1}'

    def test_no_whitespace(self):
        """No extra whitespace should be present."""
        obj = {"key": "value", "num": 42}
        result = canonical_json(obj)
        assert b" " not in result
        assert b"\n" not in result

    def test_deterministic(self):
        """Same input should always produce same output."""
        obj = {"b": [1, 2, 3], "a": {"x": True}}
        r1 = canonical_json(obj)
        r2 = canonical_json(obj)
        assert r1 == r2

    def test_utf8_encoding(self):
        """Output should be UTF-8 bytes."""
        obj = {"name": "test"}
        result = canonical_json(obj)
        assert isinstance(result, bytes)
        result.decode("utf-8")  # Should not raise


# ===================================================================
# 12. Edge Cases
# ===================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_payload(self, tgsp_creator, ed25519_keypair):
        """TGSP file with zero-length payload should be valid."""
        private_key, public_key = ed25519_keypair
        tgsp_bytes = tgsp_creator.create(b"", private_key)
        tgsp = TGSPParser.read_bytes(tgsp_bytes)
        assert tgsp.payload == b""
        assert TGSPParser.verify_payload_hash(tgsp) is True

    def test_large_name(self, ed25519_keypair):
        """Adapter name at maximum length should work."""
        private_key, _ = ed25519_keypair
        creator = TGSPCreator(
            name="x" * 256,
            domain="test",
            model=ModelInfo(),
            lora_config=LoraConfig(),
            skill=SkillInfo(description="Long name test"),
            rvu_safety=RVUSafetyResult(),
            creator_identity=CreatorIdentity(
                name="Test", organization="Test", email="t@t.com"
            ),
        )
        tgsp_bytes = creator.create(b"payload", private_key)
        tgsp = TGSPParser.read_bytes(tgsp_bytes)
        assert tgsp.manifest["name"] == "x" * 256

    def test_multiple_target_modules(self, ed25519_keypair):
        """LoRA config with many target modules should work."""
        private_key, _ = ed25519_keypair
        modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate", "up_proj", "down_proj"]
        creator = TGSPCreator(
            name="multi-module",
            domain="test",
            model=ModelInfo(),
            lora_config=LoraConfig(target_modules=modules),
            skill=SkillInfo(description="Multi-module test"),
            rvu_safety=RVUSafetyResult(),
            creator_identity=CreatorIdentity(
                name="Test", organization="Test", email="t@t.com"
            ),
        )
        tgsp_bytes = creator.create(b"payload", private_key)
        tgsp = TGSPParser.read_bytes(tgsp_bytes)
        assert tgsp.manifest["lora_config"]["target_modules"] == modules
