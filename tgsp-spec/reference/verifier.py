"""
TGSP Reference Verifier

Performs comprehensive verification of .tgsp files:
  - Payload hash (SHA-256)
  - Manifest hash (SHA-256)
  - Ed25519 signature
  - Dilithium3 stub signature
  - Creator identity
  - LoraConfig schema validation
  - RVUv2 screening status

Prints a detailed verification report.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

from .parser import (
    REQUIRED_SIGNED_FIELDS,
    TGSPFile,
    TGSPParser,
    canonical_json,
)


# ---------------------------------------------------------------------------
# Verification result types
# ---------------------------------------------------------------------------


class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARN = "WARN"


@dataclass
class CheckResult:
    """Result of a single verification check."""

    name: str
    status: CheckStatus
    message: str

    def __str__(self) -> str:
        symbol = {
            CheckStatus.PASS: "[PASS]",
            CheckStatus.FAIL: "[FAIL]",
            CheckStatus.SKIP: "[SKIP]",
            CheckStatus.WARN: "[WARN]",
        }[self.status]
        return f"  {symbol} {self.name}: {self.message}"


@dataclass
class VerificationReport:
    """Aggregated verification report."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no checks failed."""
        return all(c.status != CheckStatus.FAIL for c in self.checks)

    @property
    def num_passed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.PASS)

    @property
    def num_failed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def num_warnings(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.WARN)

    @property
    def num_skipped(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.SKIP)

    def add(self, name: str, status: CheckStatus, message: str) -> None:
        self.checks.append(CheckResult(name, status, message))

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "TGSP Verification Report",
            "=" * 60,
        ]
        for check in self.checks:
            lines.append(str(check))
        lines.append("-" * 60)
        lines.append(
            f"  Total: {len(self.checks)} checks | "
            f"{self.num_passed} passed | "
            f"{self.num_failed} failed | "
            f"{self.num_warnings} warnings | "
            f"{self.num_skipped} skipped"
        )
        verdict = "VERIFIED" if self.passed else "VERIFICATION FAILED"
        lines.append(f"  Verdict: {verdict}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema validation helpers
# ---------------------------------------------------------------------------

# Inline LoraConfig validation rules (avoids dependency on jsonschema library)
LORA_CONFIG_REQUIRED_FIELDS = [
    "rank", "alpha", "target_modules", "lora_dropout",
    "bias", "task_type", "simd_slots", "cols_per_ct", "batches",
]

VALID_BIAS_VALUES = {"none", "all", "lora_only"}
VALID_TASK_TYPES = {
    "CAUSAL_LM", "SEQ_2_SEQ_LM", "TOKEN_CLS",
    "SEQ_CLS", "QUESTION_ANS", "FEATURE_EXTRACTION",
}


def validate_lora_config(config: dict[str, Any]) -> list[str]:
    """
    Validate a lora_config dict against the schema rules.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []

    # Check required fields
    for f in LORA_CONFIG_REQUIRED_FIELDS:
        if f not in config:
            errors.append(f"Missing required field: {f}")

    # Type and range checks
    if "rank" in config:
        if not isinstance(config["rank"], int) or config["rank"] < 1:
            errors.append(f"rank must be a positive integer, got {config['rank']}")
        elif config["rank"] > 1024:
            errors.append(f"rank must be <= 1024, got {config['rank']}")

    if "alpha" in config:
        if not isinstance(config["alpha"], (int, float)) or config["alpha"] <= 0:
            errors.append(f"alpha must be a positive number, got {config['alpha']}")

    if "target_modules" in config:
        tm = config["target_modules"]
        if not isinstance(tm, list) or len(tm) < 1:
            errors.append("target_modules must be a non-empty array")
        elif not all(isinstance(m, str) and len(m) > 0 for m in tm):
            errors.append("target_modules must contain non-empty strings")

    if "lora_dropout" in config:
        d = config["lora_dropout"]
        if not isinstance(d, (int, float)) or d < 0.0 or d > 1.0:
            errors.append(f"lora_dropout must be in [0.0, 1.0], got {d}")

    if "bias" in config:
        if config["bias"] not in VALID_BIAS_VALUES:
            errors.append(
                f"bias must be one of {VALID_BIAS_VALUES}, got '{config['bias']}'"
            )

    if "task_type" in config:
        if config["task_type"] not in VALID_TASK_TYPES:
            errors.append(
                f"task_type must be one of {VALID_TASK_TYPES}, "
                f"got '{config['task_type']}'"
            )

    if "simd_slots" in config:
        if not isinstance(config["simd_slots"], int) or config["simd_slots"] < 1:
            errors.append(f"simd_slots must be a positive integer, got {config['simd_slots']}")

    if "cols_per_ct" in config:
        if not isinstance(config["cols_per_ct"], int) or config["cols_per_ct"] < 1:
            errors.append(f"cols_per_ct must be a positive integer, got {config['cols_per_ct']}")

    if "batches" in config:
        if not isinstance(config["batches"], int) or config["batches"] < 1:
            errors.append(f"batches must be a positive integer, got {config['batches']}")

    return errors


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class TGSPVerifier:
    """
    Comprehensive verifier for .tgsp files.

    Verifies integrity, signatures, creator identity, LoRA config,
    and RVUv2 safety screening.
    """

    def __init__(
        self,
        tgsp: TGSPFile,
        ed25519_public_key: Optional[Ed25519PublicKey] = None,
        dilithium3_public_key_bytes: Optional[bytes] = None,
    ) -> None:
        self.tgsp = tgsp
        self.ed25519_public_key = ed25519_public_key
        self.dilithium3_public_key_bytes = dilithium3_public_key_bytes

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_payload_hash(self, report: VerificationReport) -> None:
        """Verify SHA-256 payload hash."""
        ok = TGSPParser.verify_payload_hash(self.tgsp)
        if ok:
            report.add(
                "Payload Hash (SHA-256)",
                CheckStatus.PASS,
                "Payload hash matches integrity.payload_hash",
            )
        else:
            expected = self.tgsp.manifest.get("integrity", {}).get("payload_hash", "")
            actual = self.tgsp.payload_hash
            report.add(
                "Payload Hash (SHA-256)",
                CheckStatus.FAIL,
                f"Hash mismatch: expected {expected}, got {actual}",
            )

    def check_manifest_hash(self, report: VerificationReport) -> None:
        """Verify SHA-256 manifest hash."""
        ok = TGSPParser.verify_manifest_hash(self.tgsp)
        if ok:
            report.add(
                "Manifest Hash (SHA-256)",
                CheckStatus.PASS,
                "Manifest hash matches integrity.manifest_hash",
            )
        else:
            report.add(
                "Manifest Hash (SHA-256)",
                CheckStatus.FAIL,
                "Manifest hash mismatch (manifest may have been tampered with)",
            )

    def check_ed25519_signature(self, report: VerificationReport) -> None:
        """Verify Ed25519 signature."""
        if self.ed25519_public_key is None:
            report.add(
                "Ed25519 Signature",
                CheckStatus.SKIP,
                "No public key provided",
            )
            return

        ok = TGSPParser.verify_ed25519(self.tgsp, self.ed25519_public_key)
        if ok:
            report.add(
                "Ed25519 Signature",
                CheckStatus.PASS,
                "Signature is valid",
            )
        else:
            report.add(
                "Ed25519 Signature",
                CheckStatus.FAIL,
                "Signature verification failed",
            )

    def check_dilithium3_signature(self, report: VerificationReport) -> None:
        """Verify Dilithium3 stub signature."""
        if self.dilithium3_public_key_bytes is None:
            report.add(
                "Dilithium3 Signature (stub)",
                CheckStatus.SKIP,
                "No public key provided",
            )
            return

        ok = TGSPParser.verify_dilithium3(
            self.tgsp, self.dilithium3_public_key_bytes
        )
        if ok:
            report.add(
                "Dilithium3 Signature (stub)",
                CheckStatus.PASS,
                "Stub signature is valid",
            )
        else:
            report.add(
                "Dilithium3 Signature (stub)",
                CheckStatus.FAIL,
                "Stub signature verification failed",
            )

    def check_signed_fields(self, report: VerificationReport) -> None:
        """Verify that signed_fields covers all required fields."""
        sigs = self.tgsp.manifest.get("signatures", {})
        signed = set(sigs.get("signed_fields", []))
        missing = REQUIRED_SIGNED_FIELDS - signed
        if missing:
            report.add(
                "Signed Fields Coverage",
                CheckStatus.FAIL,
                f"Missing required signed fields: {missing}",
            )
        else:
            report.add(
                "Signed Fields Coverage",
                CheckStatus.PASS,
                f"All required fields are signed: {REQUIRED_SIGNED_FIELDS}",
            )

    def check_creator_identity(self, report: VerificationReport) -> None:
        """Verify creator identity fields and key fingerprint."""
        creator = self.tgsp.manifest.get("creator", {})

        # Check required fields exist
        required = ["name", "organization", "email", "public_key_fingerprint", "verified"]
        missing = [f for f in required if f not in creator]
        if missing:
            report.add(
                "Creator Identity",
                CheckStatus.FAIL,
                f"Missing fields: {missing}",
            )
            return

        # If we have a public key, verify the fingerprint
        if self.ed25519_public_key is not None:
            raw = self.ed25519_public_key.public_bytes(
                Encoding.Raw, PublicFormat.Raw
            )
            expected_fp = hashlib.sha256(raw).hexdigest()
            actual_fp = creator.get("public_key_fingerprint", "")
            if expected_fp == actual_fp:
                report.add(
                    "Creator Identity",
                    CheckStatus.PASS,
                    f"Key fingerprint matches: {actual_fp[:16]}...",
                )
            else:
                report.add(
                    "Creator Identity",
                    CheckStatus.FAIL,
                    f"Key fingerprint mismatch: manifest has {actual_fp[:16]}..., "
                    f"expected {expected_fp[:16]}...",
                )
        else:
            report.add(
                "Creator Identity",
                CheckStatus.WARN,
                "Creator fields present but key fingerprint not verified "
                "(no public key provided)",
            )

    def check_lora_config(self, report: VerificationReport) -> None:
        """Validate LoraConfig against schema rules."""
        config = self.tgsp.manifest.get("lora_config", {})
        errors = validate_lora_config(config)
        if errors:
            report.add(
                "LoRA Config Validation",
                CheckStatus.FAIL,
                "; ".join(errors),
            )
        else:
            report.add(
                "LoRA Config Validation",
                CheckStatus.PASS,
                f"Valid (rank={config.get('rank')}, alpha={config.get('alpha')}, "
                f"task={config.get('task_type')})",
            )

    def check_rvu_safety(self, report: VerificationReport) -> None:
        """Check RVUv2 screening status."""
        safety = self.tgsp.manifest.get("rvu_safety", {})

        # Check version
        version = safety.get("version", "")
        if version != "RVUv2":
            report.add(
                "RVU Safety Version",
                CheckStatus.FAIL,
                f"Expected 'RVUv2', got '{version}'",
            )
        else:
            report.add(
                "RVU Safety Version",
                CheckStatus.PASS,
                "Version is RVUv2",
            )

        # Check layers
        layers = safety.get("layers", [])
        expected_layers = {"allowlist", "svd_analysis", "mahalanobis_ood"}
        provided = set(layers)
        missing_layers = expected_layers - provided
        if missing_layers:
            report.add(
                "RVU Safety Layers",
                CheckStatus.WARN,
                f"Missing recommended layers: {missing_layers}",
            )
        else:
            report.add(
                "RVU Safety Layers",
                CheckStatus.PASS,
                f"All standard layers present: {sorted(provided)}",
            )

        # Check screening result
        passed = safety.get("screening_passed")
        if passed is True:
            report.add(
                "RVU Screening Result",
                CheckStatus.PASS,
                "Screening passed",
            )
        elif passed is False:
            report.add(
                "RVU Screening Result",
                CheckStatus.FAIL,
                "Screening did NOT pass",
            )
        else:
            report.add(
                "RVU Screening Result",
                CheckStatus.FAIL,
                "screening_passed field is missing or not boolean",
            )

        # Check timestamp
        ts = safety.get("screening_timestamp", "")
        if ts:
            report.add(
                "RVU Screening Timestamp",
                CheckStatus.PASS,
                f"Timestamp: {ts}",
            )
        else:
            report.add(
                "RVU Screening Timestamp",
                CheckStatus.FAIL,
                "Missing screening timestamp",
            )

        # Check screening hash
        sh = safety.get("screening_hash", "")
        if sh and len(sh) == 64:
            report.add(
                "RVU Screening Hash",
                CheckStatus.PASS,
                f"Hash: {sh[:16]}...",
            )
        else:
            report.add(
                "RVU Screening Hash",
                CheckStatus.FAIL,
                "Missing or invalid screening hash (expected 64-char hex)",
            )

    def check_format_fields(self, report: VerificationReport) -> None:
        """Verify top-level format and version fields."""
        fmt = self.tgsp.manifest.get("format", "")
        ver = self.tgsp.manifest.get("version", "")

        if fmt == "TGSP":
            report.add("Format Field", CheckStatus.PASS, "format = 'TGSP'")
        else:
            report.add(
                "Format Field",
                CheckStatus.FAIL,
                f"Expected 'TGSP', got '{fmt}'",
            )

        if ver == "1.1":
            report.add("Version Field", CheckStatus.PASS, "version = '1.1'")
        else:
            report.add(
                "Version Field",
                CheckStatus.WARN,
                f"Expected '1.1', got '{ver}'",
            )

    # ------------------------------------------------------------------
    # Full verification
    # ------------------------------------------------------------------

    def verify(self) -> VerificationReport:
        """
        Run all verification checks and return a report.

        The checks are run in this order:
        1. Format fields
        2. Payload hash
        3. Manifest hash
        4. Signed fields coverage
        5. Ed25519 signature
        6. Dilithium3 signature
        7. Creator identity
        8. LoRA config validation
        9. RVU safety screening
        """
        report = VerificationReport()

        self.check_format_fields(report)
        self.check_payload_hash(report)
        self.check_manifest_hash(report)
        self.check_signed_fields(report)
        self.check_ed25519_signature(report)
        self.check_dilithium3_signature(report)
        self.check_creator_identity(report)
        self.check_lora_config(report)
        self.check_rvu_safety(report)

        return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Simple CLI: verify a .tgsp file (without signature verification,
    since no public key is provided on the CLI).
    """
    if len(sys.argv) < 2:
        print("Usage: python verifier.py <file.tgsp>")
        sys.exit(1)

    tgsp = TGSPParser.read(sys.argv[1])
    verifier = TGSPVerifier(tgsp)
    report = verifier.verify()
    print(report)
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
