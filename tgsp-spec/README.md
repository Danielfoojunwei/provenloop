# TGSP: TenSafe Gated Secure Package Specification

**Version 1.1** | **License: Apache 2.0**

---

## 1. Overview

TGSP (TenSafe Gated Secure Package) is an open adapter format that wraps LoRA
(Low-Rank Adaptation) weights with a cryptographic trust chain. Every TGSP file
carries:

| Layer | Purpose |
|-------|---------|
| **Creator identity** | Who produced the adapter, verified via public key |
| **RVUv2 safety screening** | Multi-layer safety analysis results |
| **Hash integrity** | SHA-256 over payload and manifest |
| **Dual signatures** | Ed25519 (classical) + Dilithium3 (post-quantum) |

TGSP is a **trust wrapper** around standard LoRA. You can always export back to
plain LoRA (safetensors, PyTorch, or GGUF).

---

## 2. Binary Format

A `.tgsp` file is a single binary blob with three contiguous sections:

```
+------+------------------+-----------+-------------------------+
| Byte | Field            | Size      | Description             |
+------+------------------+-----------+-------------------------+
| 0    | Magic            | 6 bytes   | TGSP\x01\x00           |
| 6    | manifest_length  | 4 bytes   | Little-endian uint32    |
| 10   | manifest         | variable  | UTF-8 JSON              |
| 10+N | payload          | remainder | LoRA weights (safetensors) |
+------+------------------+-----------+-------------------------+
```

### 2.1 Magic Bytes

The first 6 bytes of every TGSP file **MUST** be:

```
0x54 0x47 0x53 0x50 0x01 0x00
```

- Bytes 0-3: ASCII `TGSP` (format identifier)
- Byte 4: Major version (`0x01`)
- Byte 5: Minor version (`0x00`)

### 2.2 Manifest Length

Bytes 6-9 encode the manifest length as a **little-endian unsigned 32-bit
integer**. This gives a maximum manifest size of ~4 GiB, which is more than
sufficient for any metadata payload.

### 2.3 Manifest

The manifest is a UTF-8 encoded JSON object conforming to the
[Manifest v1.1 JSON Schema](schema/manifest-v1.1.json). It contains all
metadata, safety records, creator identity, and signatures.

### 2.4 Payload

Everything after the manifest is the LoRA weight payload in
**safetensors** format. The payload is stored verbatim -- no additional
framing or compression is applied.

---

## 3. Manifest v1.1 Schema

The manifest is a JSON object with the following top-level fields:

```json
{
  "format": "TGSP",
  "version": "1.1",
  "name": "<adapter name>",
  "domain": "<domain identifier>",
  "model": { ... },
  "lora_config": { ... },
  "skill": { ... },
  "rvu_safety": { ... },
  "creator": { ... },
  "integrity": { ... },
  "signatures": { ... }
}
```

### 3.1 `model`

Describes the base model this adapter targets.

| Field | Type | Description |
|-------|------|-------------|
| `architecture` | string | Model architecture (e.g. `"sparse_moe"`) |
| `base_model` | string | HuggingFace model ID or path |
| `num_experts` | integer | Number of MoE experts |
| `experts_per_token` | integer | Experts activated per token |

### 3.2 `lora_config`

LoRA hyperparameters and FHE-related configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rank` | integer | 30 | LoRA rank (r) |
| `alpha` | number | 64 | LoRA scaling alpha |
| `target_modules` | array[string] | | Module names to adapt |
| `lora_dropout` | number | 0.0 | Dropout rate |
| `bias` | string | `"none"` | Bias handling: none, all, lora_only |
| `task_type` | string | | PEFT task type |
| `simd_slots` | integer | 8192 | FHE SIMD slot count |
| `cols_per_ct` | integer | 5 | Columns per ciphertext |
| `batches` | integer | 6 | Number of FHE batches |

See [lora_config.json](schema/lora_config.json) for the full JSON Schema.

### 3.3 `skill`

Describes what the adapter does.

| Field | Type | Description |
|-------|------|-------------|
| `description` | string | Human-readable description |
| `triggers` | array[string] | Activation triggers or keywords |
| `capabilities` | array[string] | List of capabilities |
| `input_format` | string | Expected input format |
| `output_format` | string | Expected output format |
| `quality_score` | number | Quality metric (0.0 - 1.0) |
| `compliance` | array[string] | Compliance standards met |
| `composable_with` | array[string] | Compatible adapter names |

### 3.4 `rvu_safety`

RVUv2 safety screening results.

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Must be `"RVUv2"` |
| `layers` | array[string] | Screening layers applied |
| `screening_passed` | boolean | Overall pass/fail |
| `screening_timestamp` | string | ISO 8601 timestamp |
| `screening_hash` | string | SHA-256 hash of screening report |

The standard RVUv2 screening layers are:

1. **allowlist** -- Weight tensor names checked against an approved allowlist
2. **svd_analysis** -- Singular value decomposition analysis for anomalous rank patterns
3. **mahalanobis_ood** -- Mahalanobis distance out-of-distribution detection on weight statistics

### 3.5 `creator`

Creator identity and provenance.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Creator display name |
| `organization` | string | Organization name |
| `email` | string | Contact email |
| `public_key_fingerprint` | string | SHA-256 fingerprint of public key |
| `verified` | boolean | Whether identity has been verified |

### 3.6 `integrity`

Cryptographic integrity records.

| Field | Type | Description |
|-------|------|-------------|
| `payload_hash` | string | SHA-256 hex digest of payload bytes |
| `manifest_hash` | string | SHA-256 hex digest of manifest JSON (before signatures are added) |
| `hash_algorithm` | string | Must be `"SHA-256"` |

### 3.7 `signatures`

Dual cryptographic signatures over the integrity fields.

| Field | Type | Description |
|-------|------|-------------|
| `ed25519` | string | Base64-encoded Ed25519 signature |
| `dilithium3` | string | Base64-encoded Dilithium3 signature |
| `signed_fields` | array[string] | List of manifest fields that were signed |

The `signed_fields` array specifies exactly which top-level manifest keys are
covered by the signatures. At minimum it **MUST** include:
`["integrity", "creator", "rvu_safety", "lora_config"]`.

The signature is computed over the canonical JSON serialization (sorted keys, no
extra whitespace) of the object `{ field: manifest[field] for field in
signed_fields }`.

---

## 4. Conversion

TGSP supports bidirectional conversion with standard LoRA formats.

### 4.1 TGSP to LoRA

```
.tgsp --> parse magic + manifest --> extract payload --> .safetensors
                                                     --> .pt (PyTorch)
                                                     --> .gguf (GGUF)
```

The payload is already in safetensors format. Conversion to PyTorch or GGUF
requires the appropriate libraries.

### 4.2 LoRA to TGSP

```
.safetensors --> hash payload --> build manifest --> sign --> assemble .tgsp
```

The creator tool handles:
1. Reading the LoRA weight file(s)
2. Computing SHA-256 payload hash
3. Building the manifest with creator identity, LoraConfig, and RVUv2 results
4. Signing with Ed25519 (and optionally Dilithium3)
5. Assembling the binary TGSP file

---

## 5. Security Model

### 5.1 Threat Model

TGSP protects against:
- **Tampering**: Hash integrity detects any modification to weights or metadata
- **Impersonation**: Ed25519 signatures bind the adapter to a creator identity
- **Quantum threats**: Dilithium3 provides post-quantum signature security
- **Unsafe adapters**: RVUv2 screening records provide auditable safety evidence

### 5.2 Signature Verification

Verifiers **MUST**:
1. Recompute the payload hash and compare to `integrity.payload_hash`
2. Recompute the manifest hash (excluding `signatures`) and compare to `integrity.manifest_hash`
3. Verify the Ed25519 signature against the creator's public key
4. Optionally verify the Dilithium3 signature (when PQC libraries are available)
5. Check that `signed_fields` covers all required fields

### 5.3 Trust Chain

```
Creator Key --> signs --> Manifest (integrity + metadata)
                              |
                              +--> payload_hash --> covers --> LoRA weights
                              +--> rvu_safety   --> covers --> Safety screening
                              +--> creator      --> covers --> Provenance
```

---

## 6. Reference Implementation

This repository includes reference Python implementations:

| File | Purpose |
|------|---------|
| [`reference/parser.py`](reference/parser.py) | Parse .tgsp files, validate, export |
| [`reference/creator.py`](reference/creator.py) | Create .tgsp files from LoRA weights |
| [`reference/verifier.py`](reference/verifier.py) | Verify integrity, signatures, safety |
| [`conformance/test_suite.py`](conformance/test_suite.py) | Pytest conformance test suite |

### Dependencies

- Python 3.9+
- `cryptography` (Ed25519 signatures)
- `safetensors` (weight I/O)
- `pytest` (conformance tests)

Dilithium3 is stubbed in the reference implementation. Production use should
link to the `tensafe-pqc` Rust crate via PyO3 bindings.

---

## 7. MIME Type and File Extension

| Property | Value |
|----------|-------|
| File extension | `.tgsp` |
| MIME type | `application/x-tgsp` |

---

## 8. Versioning

The format version is encoded in the magic bytes (bytes 4-5). The manifest
`version` field carries the schema version. These are independent:

- **Format version** (magic bytes): `1.0` -- binary layout
- **Schema version** (manifest): `1.1` -- manifest JSON structure

Future versions **MUST** increment the minor version for backward-compatible
changes and the major version for breaking changes.

---

## 9. License

Copyright 2025-2026 TenSafe Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
