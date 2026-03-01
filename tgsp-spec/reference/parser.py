"""
TGSP Reference Parser

Reads .tgsp files, validates magic bytes, parses the manifest,
verifies hash integrity, verifies signatures, and exports the
LoRA payload to safetensors format.

SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TGSP_MAGIC = b"TGSP\x01\x00"
MAGIC_LEN = 6
MANIFEST_LEN_SIZE = 4  # uint32 little-endian

REQUIRED_SIGNED_FIELDS = {"integrity", "creator", "rvu_safety", "lora_config"}


# ---------------------------------------------------------------------------
# Dilithium3 stub
# ---------------------------------------------------------------------------


class Dilithium3Stub:
    """
    Stub for Dilithium3 (CRYSTALS-Dilithium) post-quantum signatures.

    Production implementations should replace this with the tensafe-pqc
    Rust crate exposed via PyO3 bindings.
    """

    STUB_MARKER = b"DILITHIUM3_STUB_SIG_v1:"

    @staticmethod
    def verify(public_key_bytes: bytes, signature: bytes, message: bytes) -> bool:
        """
        Verify a Dilithium3 signature.

        In stub mode, accepts any signature that starts with the stub marker
        and whose remaining bytes are the SHA-256 of (public_key || message).
        """
        if not signature.startswith(Dilithium3Stub.STUB_MARKER):
            return False
        expected_hash = hashlib.sha256(public_key_bytes + message).digest()
        return signature[len(Dilithium3Stub.STUB_MARKER):] == expected_hash


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TGSPFile:
    """Parsed representation of a .tgsp file."""

    magic: bytes
    manifest: dict[str, Any]
    payload: bytes
    raw_manifest_bytes: bytes = field(repr=False)

    @property
    def name(self) -> str:
        return self.manifest.get("name", "<unnamed>")

    @property
    def version(self) -> str:
        return self.manifest.get("version", "<unknown>")

    @property
    def payload_hash(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


# ---------------------------------------------------------------------------
# Canonical JSON helper
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> bytes:
    """
    Return the canonical JSON serialization of *obj*: sorted keys, no extra
    whitespace, encoded as UTF-8.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Signing-data helper
# ---------------------------------------------------------------------------


def build_signing_data(manifest: dict[str, Any], signed_fields: list[str]) -> bytes:
    """
    Build the byte string that was signed.

    This is the canonical JSON of ``{field: manifest[field] for field in
    signed_fields}``, sorted by key.
    """
    data = {f: manifest[f] for f in sorted(signed_fields)}
    return canonical_json(data)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TGSPParser:
    """Reference parser for .tgsp files."""

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @staticmethod
    def read(path: str | Path) -> TGSPFile:
        """
        Read and parse a .tgsp file from *path*.

        Raises ``ValueError`` on any structural error (bad magic, truncated
        data, invalid JSON manifest).
        """
        path = Path(path)
        data = path.read_bytes()
        return TGSPParser.read_bytes(data)

    @staticmethod
    def read_bytes(data: bytes) -> TGSPFile:
        """Parse a .tgsp file from an in-memory byte string."""
        if len(data) < MAGIC_LEN + MANIFEST_LEN_SIZE:
            raise ValueError(
                f"File too small: expected at least {MAGIC_LEN + MANIFEST_LEN_SIZE} "
                f"bytes, got {len(data)}"
            )

        # --- Magic ---
        magic = data[:MAGIC_LEN]
        if magic != TGSP_MAGIC:
            raise ValueError(
                f"Invalid magic bytes: expected {TGSP_MAGIC!r}, got {magic!r}"
            )

        # --- Manifest length ---
        manifest_len = struct.unpack_from("<I", data, MAGIC_LEN)[0]
        manifest_start = MAGIC_LEN + MANIFEST_LEN_SIZE
        manifest_end = manifest_start + manifest_len

        if manifest_end > len(data):
            raise ValueError(
                f"Manifest length ({manifest_len}) exceeds file size "
                f"({len(data)} bytes)"
            )

        # --- Manifest ---
        raw_manifest = data[manifest_start:manifest_end]
        try:
            manifest = json.loads(raw_manifest.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid manifest JSON: {exc}") from exc

        # --- Payload ---
        payload = data[manifest_end:]

        return TGSPFile(
            magic=magic,
            manifest=manifest,
            payload=payload,
            raw_manifest_bytes=raw_manifest,
        )

    # ------------------------------------------------------------------
    # Hash verification
    # ------------------------------------------------------------------

    @staticmethod
    def verify_payload_hash(tgsp: TGSPFile) -> bool:
        """
        Return ``True`` if the SHA-256 of the payload matches
        ``manifest.integrity.payload_hash``.
        """
        expected = tgsp.manifest.get("integrity", {}).get("payload_hash", "")
        actual = tgsp.payload_hash
        return actual == expected

    @staticmethod
    def verify_manifest_hash(tgsp: TGSPFile) -> bool:
        """
        Return ``True`` if the SHA-256 of the manifest (excluding the
        ``signatures`` block, with ``manifest_hash`` set to empty string)
        matches ``manifest.integrity.manifest_hash``.

        Both creator and verifier use the convention of hashing the
        manifest with ``manifest_hash`` set to ``""`` so the hash is
        deterministic and self-consistent.
        """
        import copy

        expected = tgsp.manifest.get("integrity", {}).get("manifest_hash", "")
        # Build manifest without signatures, with manifest_hash blanked
        manifest_for_hash = {
            k: v for k, v in tgsp.manifest.items() if k != "signatures"
        }
        manifest_for_hash = copy.deepcopy(manifest_for_hash)
        manifest_for_hash["integrity"]["manifest_hash"] = ""
        actual = hashlib.sha256(canonical_json(manifest_for_hash)).hexdigest()
        return actual == expected

    # ------------------------------------------------------------------
    # Signature verification
    # ------------------------------------------------------------------

    @staticmethod
    def verify_ed25519(
        tgsp: TGSPFile,
        public_key: Ed25519PublicKey,
    ) -> bool:
        """
        Verify the Ed25519 signature in the manifest.

        Returns ``True`` if the signature is valid, ``False`` otherwise.
        """
        sigs = tgsp.manifest.get("signatures", {})
        sig_b64 = sigs.get("ed25519", "")
        signed_fields = sigs.get("signed_fields", [])

        try:
            sig_bytes = base64.b64decode(sig_b64)
        except Exception:
            return False

        message = build_signing_data(tgsp.manifest, signed_fields)

        try:
            public_key.verify(sig_bytes, message)
            return True
        except InvalidSignature:
            return False

    @staticmethod
    def verify_dilithium3(
        tgsp: TGSPFile,
        public_key_bytes: bytes,
    ) -> bool:
        """
        Verify the Dilithium3 stub signature in the manifest.

        In production, replace with real Dilithium3 verification via
        tensafe-pqc.
        """
        sigs = tgsp.manifest.get("signatures", {})
        sig_b64 = sigs.get("dilithium3", "")
        signed_fields = sigs.get("signed_fields", [])

        try:
            sig_bytes = base64.b64decode(sig_b64)
        except Exception:
            return False

        message = build_signing_data(tgsp.manifest, signed_fields)
        return Dilithium3Stub.verify(public_key_bytes, sig_bytes, message)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def export_safetensors(tgsp: TGSPFile, output_path: str | Path) -> Path:
        """
        Export the LoRA payload to a .safetensors file.

        Returns the path to the written file.
        """
        output_path = Path(output_path)
        output_path.write_bytes(tgsp.payload)
        return output_path

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    @staticmethod
    def display_info(tgsp: TGSPFile) -> str:
        """Return a human-readable summary of the TGSP file."""
        m = tgsp.manifest
        lines = [
            "=" * 60,
            "TGSP File Summary",
            "=" * 60,
            f"  Name:      {m.get('name', 'N/A')}",
            f"  Version:   {m.get('version', 'N/A')}",
            f"  Domain:    {m.get('domain', 'N/A')}",
            "",
            "Model:",
            f"  Architecture:      {m.get('model', {}).get('architecture', 'N/A')}",
            f"  Base model:        {m.get('model', {}).get('base_model', 'N/A')}",
            f"  Num experts:       {m.get('model', {}).get('num_experts', 'N/A')}",
            f"  Experts per token: {m.get('model', {}).get('experts_per_token', 'N/A')}",
            "",
            "LoRA Config:",
            f"  Rank:           {m.get('lora_config', {}).get('rank', 'N/A')}",
            f"  Alpha:          {m.get('lora_config', {}).get('alpha', 'N/A')}",
            f"  Target modules: {m.get('lora_config', {}).get('target_modules', 'N/A')}",
            f"  Dropout:        {m.get('lora_config', {}).get('lora_dropout', 'N/A')}",
            f"  Bias:           {m.get('lora_config', {}).get('bias', 'N/A')}",
            f"  Task type:      {m.get('lora_config', {}).get('task_type', 'N/A')}",
            f"  SIMD slots:     {m.get('lora_config', {}).get('simd_slots', 'N/A')}",
            f"  Cols per CT:    {m.get('lora_config', {}).get('cols_per_ct', 'N/A')}",
            f"  Batches:        {m.get('lora_config', {}).get('batches', 'N/A')}",
            "",
            "Skill:",
            f"  Description:  {m.get('skill', {}).get('description', 'N/A')}",
            f"  Quality:      {m.get('skill', {}).get('quality_score', 'N/A')}",
            f"  Compliance:   {m.get('skill', {}).get('compliance', 'N/A')}",
            "",
            "RVU Safety:",
            f"  Version:     {m.get('rvu_safety', {}).get('version', 'N/A')}",
            f"  Layers:      {m.get('rvu_safety', {}).get('layers', 'N/A')}",
            f"  Passed:      {m.get('rvu_safety', {}).get('screening_passed', 'N/A')}",
            f"  Timestamp:   {m.get('rvu_safety', {}).get('screening_timestamp', 'N/A')}",
            "",
            "Creator:",
            f"  Name:         {m.get('creator', {}).get('name', 'N/A')}",
            f"  Organization: {m.get('creator', {}).get('organization', 'N/A')}",
            f"  Email:        {m.get('creator', {}).get('email', 'N/A')}",
            f"  Verified:     {m.get('creator', {}).get('verified', 'N/A')}",
            "",
            "Integrity:",
            f"  Payload hash:  {m.get('integrity', {}).get('payload_hash', 'N/A')}",
            f"  Manifest hash: {m.get('integrity', {}).get('manifest_hash', 'N/A')}",
            f"  Algorithm:     {m.get('integrity', {}).get('hash_algorithm', 'N/A')}",
            "",
            f"Payload size: {len(tgsp.payload):,} bytes",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Simple CLI: parse and display a .tgsp file."""
    if len(sys.argv) < 2:
        print("Usage: python parser.py <file.tgsp> [--export <output.safetensors>]")
        sys.exit(1)

    tgsp = TGSPParser.read(sys.argv[1])
    print(TGSPParser.display_info(tgsp))

    # Quick integrity check
    ph_ok = TGSPParser.verify_payload_hash(tgsp)
    mh_ok = TGSPParser.verify_manifest_hash(tgsp)
    print(f"\nPayload hash valid:  {ph_ok}")
    print(f"Manifest hash valid: {mh_ok}")

    # Optional export
    if "--export" in sys.argv:
        idx = sys.argv.index("--export")
        if idx + 1 < len(sys.argv):
            out = TGSPParser.export_safetensors(tgsp, sys.argv[idx + 1])
            print(f"\nExported payload to: {out}")
        else:
            print("Error: --export requires an output path", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
