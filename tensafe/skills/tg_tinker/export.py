"""TGSP <-> LoRA format conversion.

Handles export from TGSP to plain LoRA formats (safetensors, PyTorch, GGUF)
and import from existing LoRA adapters into signed TGSP packages.

TGSP binary format (v1):
    Bytes 0-3:   Magic b"TGSP"
    Bytes 4-5:   Version (uint16 LE) = 0x0100
    Bytes 6-9:   Manifest length (uint32 LE)
    Bytes 10..:  JSON manifest (UTF-8)
    After manifest: payload (serialized weights)
"""

import hashlib
import io
import json
import logging
import os
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# TGSP format constants
TGSP_MAGIC = b"TGSP"
TGSP_VERSION = 0x0100  # v1.0
TGSP_HEADER_SIZE = 10  # 4 (magic) + 2 (version) + 4 (manifest_len)


# ---------------------------------------------------------------------------
# TGSP reading / writing primitives
# ---------------------------------------------------------------------------

def read_tgsp(tgsp_path: str) -> Tuple[dict, bytes]:
    """Read a TGSP file and return (manifest, payload_bytes).

    Args:
        tgsp_path: Path to .tgsp file.

    Returns:
        Tuple of (manifest_dict, raw_payload_bytes).

    Raises:
        ValueError: If file is not a valid TGSP package.
    """
    path = Path(tgsp_path)
    if not path.exists():
        raise FileNotFoundError(f"TGSP file not found: {tgsp_path}")

    with open(path, "rb") as f:
        header = f.read(TGSP_HEADER_SIZE)
        if len(header) < TGSP_HEADER_SIZE:
            raise ValueError(f"File too small to be TGSP: {tgsp_path}")

        magic = header[:4]
        if magic != TGSP_MAGIC:
            raise ValueError(
                f"Invalid TGSP magic bytes: {magic!r} (expected {TGSP_MAGIC!r})"
            )

        version = struct.unpack_from("<H", header, 4)[0]
        manifest_len = struct.unpack_from("<I", header, 6)[0]

        manifest_bytes = f.read(manifest_len)
        if len(manifest_bytes) < manifest_len:
            raise ValueError(
                f"Truncated manifest: expected {manifest_len} bytes, "
                f"got {len(manifest_bytes)}"
            )

        payload = f.read()

    manifest = json.loads(manifest_bytes.decode("utf-8"))
    return manifest, payload


def write_tgsp(
    manifest: dict,
    payload: bytes,
    output_path: str,
) -> str:
    """Write a TGSP file from manifest and payload.

    Args:
        manifest: JSON-serializable manifest dict.
        payload: Raw weight payload bytes.
        output_path: Destination path for .tgsp file.

    Returns:
        The absolute path of the written file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Compute payload hash and update manifest
    payload_hash = hashlib.sha256(payload).hexdigest()
    manifest["payload_hash"] = payload_hash
    manifest["payload_size"] = len(payload)

    manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    manifest_len = len(manifest_bytes)

    # Write atomically via temp file
    tmp_path = path.with_suffix(".tgsp.tmp")
    with open(tmp_path, "wb") as f:
        # Header: magic + version + manifest_len
        f.write(TGSP_MAGIC)
        f.write(struct.pack("<H", TGSP_VERSION))
        f.write(struct.pack("<I", manifest_len))
        # Manifest
        f.write(manifest_bytes)
        # Payload
        f.write(payload)

    tmp_path.rename(path)
    logger.info(
        f"Wrote TGSP: {path.name} "
        f"(manifest={manifest_len}B, payload={len(payload)}B, "
        f"hash={payload_hash[:16]}...)"
    )
    return str(path.resolve())


def verify_tgsp(tgsp_path: str) -> Tuple[bool, dict]:
    """Verify a TGSP file's integrity.

    Checks magic bytes, parses manifest, and validates payload hash.

    Args:
        tgsp_path: Path to .tgsp file.

    Returns:
        Tuple of (is_valid, details_dict).
    """
    details: Dict[str, Any] = {
        "path": tgsp_path,
        "valid": False,
        "magic_ok": False,
        "manifest_ok": False,
        "hash_ok": False,
        "manifest": None,
        "errors": [],
    }

    try:
        manifest, payload = read_tgsp(tgsp_path)
        details["magic_ok"] = True
        details["manifest_ok"] = True
        details["manifest"] = manifest

        # Verify payload hash
        expected_hash = manifest.get("payload_hash", "")
        if expected_hash:
            actual_hash = hashlib.sha256(payload).hexdigest()
            details["hash_ok"] = actual_hash == expected_hash
            if not details["hash_ok"]:
                details["errors"].append(
                    f"Hash mismatch: expected {expected_hash[:16]}..., "
                    f"got {actual_hash[:16]}..."
                )
        else:
            # No hash in manifest (v1 compat) -- skip check
            details["hash_ok"] = True

        # Check for signature
        details["signed"] = "signature" in manifest
        details["creator"] = manifest.get("creator", "unknown")
        details["format_version"] = manifest.get("format_version", "1.0")

        details["valid"] = (
            details["magic_ok"]
            and details["manifest_ok"]
            and details["hash_ok"]
        )

    except FileNotFoundError as e:
        details["errors"].append(str(e))
    except ValueError as e:
        details["errors"].append(str(e))
    except json.JSONDecodeError as e:
        details["manifest_ok"] = False
        details["errors"].append(f"Invalid manifest JSON: {e}")

    return details["valid"], details


# ---------------------------------------------------------------------------
# Weight serialization helpers
# ---------------------------------------------------------------------------

def _serialize_state_dict(state_dict: dict) -> bytes:
    """Serialize a state dict to bytes using PyTorch if available,
    falling back to a simple JSON+binary format."""
    try:
        import torch
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        return buf.getvalue()
    except ImportError:
        # Fallback: JSON keys + numpy arrays
        import numpy as np
        buf = io.BytesIO()
        arrays = {}
        for key, val in state_dict.items():
            if hasattr(val, "numpy"):
                arrays[key] = val.numpy()
            elif isinstance(val, np.ndarray):
                arrays[key] = val
            else:
                arrays[key] = np.array(val)
        np.savez_compressed(buf, **arrays)
        return buf.getvalue()


def _deserialize_state_dict(data: bytes) -> dict:
    """Deserialize a state dict from bytes."""
    try:
        import torch
        buf = io.BytesIO(data)
        return torch.load(buf, map_location="cpu", weights_only=True)
    except ImportError:
        import numpy as np
        buf = io.BytesIO(data)
        npz = np.load(buf, allow_pickle=False)
        return dict(npz)


def _extract_lora_weights(state_dict: dict) -> dict:
    """Extract only LoRA weight keys from a full state dict.

    Handles both flat dicts (lora_A.weight, lora_B.weight) and nested
    dicts (model_state_dict -> lora keys).
    """
    # Check for nested model_state_dict
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    lora_weights = {}
    for key, val in state_dict.items():
        if "lora_" in key.lower() or "lora_A" in key or "lora_B" in key:
            lora_weights[key] = val

    return lora_weights


# ---------------------------------------------------------------------------
# Export: TGSP -> plain LoRA formats
# ---------------------------------------------------------------------------

def export_tgsp_to_safetensors(tgsp_path: str, output_dir: str) -> str:
    """Export a TGSP package to safetensors format.

    Creates output_dir with:
      - adapter_model.safetensors (weights)
      - adapter_config.json (LoRA configuration)

    Args:
        tgsp_path: Path to source .tgsp file.
        output_dir: Directory to write safetensors output.

    Returns:
        Path to the output directory.
    """
    manifest, payload = read_tgsp(tgsp_path)
    state_dict = _deserialize_state_dict(payload)
    lora_weights = _extract_lora_weights(state_dict)

    if not lora_weights:
        logger.warning("No LoRA weights found in TGSP payload. Using full state dict.")
        lora_weights = state_dict

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write safetensors
    safetensors_path = out_path / "adapter_model.safetensors"
    try:
        from safetensors.torch import save_file as st_save
        import torch

        # Ensure all values are tensors
        tensor_dict = {}
        for k, v in lora_weights.items():
            if isinstance(v, torch.Tensor):
                tensor_dict[k] = v
            else:
                import numpy as np
                if isinstance(v, np.ndarray):
                    tensor_dict[k] = torch.from_numpy(v)
                else:
                    tensor_dict[k] = torch.tensor(v)

        st_save(tensor_dict, str(safetensors_path))
        logger.info(f"Wrote safetensors: {safetensors_path}")

    except ImportError:
        logger.warning(
            "safetensors library not available. Writing manual safetensors format."
        )
        _write_manual_safetensors(lora_weights, str(safetensors_path))

    # Write adapter_config.json
    config = _manifest_to_adapter_config(manifest)
    config_path = out_path / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Wrote adapter config: {config_path}")

    return str(out_path.resolve())


def export_tgsp_to_pytorch(tgsp_path: str, output_path: str) -> str:
    """Export a TGSP package to PyTorch format (.pt or .bin).

    Args:
        tgsp_path: Path to source .tgsp file.
        output_path: Destination path for .pt/.bin file.

    Returns:
        Path to the output file.
    """
    manifest, payload = read_tgsp(tgsp_path)
    state_dict = _deserialize_state_dict(payload)
    lora_weights = _extract_lora_weights(state_dict)

    if not lora_weights:
        lora_weights = state_dict

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        torch.save(lora_weights, str(out))
    except ImportError:
        # Fallback: write raw bytes
        serialized = _serialize_state_dict(lora_weights)
        with open(out, "wb") as f:
            f.write(serialized)

    logger.info(f"Wrote PyTorch weights: {out} ({out.stat().st_size} bytes)")

    # Also write adapter_config.json alongside
    config = _manifest_to_adapter_config(manifest)
    config_path = out.parent / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return str(out.resolve())


def export_tgsp_to_gguf(tgsp_path: str, output_path: str) -> str:
    """Export a TGSP package to GGUF format.

    GGUF is the format used by llama.cpp and related tools. This
    performs a best-effort conversion by writing LoRA weights into
    the GGUF container format.

    Note: Full GGUF quantization requires ggml tools. This function
    creates an f16 GGUF file that can be further quantized with
    llama.cpp's quantize tool.

    Args:
        tgsp_path: Path to source .tgsp file.
        output_path: Destination path for .gguf file.

    Returns:
        Path to the output file.
    """
    manifest, payload = read_tgsp(tgsp_path)
    state_dict = _deserialize_state_dict(payload)
    lora_weights = _extract_lora_weights(state_dict)

    if not lora_weights:
        lora_weights = state_dict

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        _write_gguf(lora_weights, manifest, str(out))
        logger.info(f"Wrote GGUF: {out} ({out.stat().st_size} bytes)")
    except Exception as e:
        logger.error(f"GGUF export failed: {e}")
        raise

    return str(out.resolve())


# ---------------------------------------------------------------------------
# Import: plain LoRA -> TGSP
# ---------------------------------------------------------------------------

def import_lora_to_tgsp(
    lora_dir: str,
    name: str,
    domain: str,
    signing_key: Optional[str] = None,
    output_path: Optional[str] = None,
    creator: str = "unknown",
    description: str = "",
    license_type: str = "apache-2.0",
    base_model: str = "",
    tags: Optional[List[str]] = None,
) -> str:
    """Import an existing LoRA adapter directory into a signed TGSP package.

    Supports:
      - HuggingFace PEFT format (adapter_model.safetensors + adapter_config.json)
      - Raw PyTorch format (adapter_model.bin / pytorch_model.bin)
      - Single .pt file

    Args:
        lora_dir: Path to LoRA adapter directory.
        name: Adapter name for the TGSP manifest.
        domain: Domain classification (healthcare, finance, legal, etc.).
        signing_key: Optional hex-encoded signing key for TGSP signature.
        output_path: Output .tgsp path. Defaults to <lora_dir>/<name>.tgsp.
        creator: Creator name/email for manifest.
        description: Description for manifest.
        license_type: License string.
        base_model: Base model name (e.g., "Qwen/Qwen2.5-1.5B").
        tags: List of tags for marketplace.

    Returns:
        Path to the created .tgsp file.
    """
    lora_path = Path(lora_dir)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA directory not found: {lora_dir}")

    # Detect and load weights
    state_dict, adapter_config = _load_lora_from_dir(lora_path)

    if not state_dict:
        raise ValueError(f"No LoRA weights found in {lora_dir}")

    # Build manifest
    manifest = _build_import_manifest(
        name=name,
        domain=domain,
        creator=creator,
        description=description,
        license_type=license_type,
        base_model=base_model or adapter_config.get("base_model_name_or_path", ""),
        adapter_config=adapter_config,
        signing_key=signing_key,
        tags=tags or [],
    )

    # Serialize weights
    payload = _serialize_state_dict(state_dict)

    # Determine output path
    if output_path is None:
        output_path = str(lora_path / f"{name}.tgsp")

    return write_tgsp(manifest, payload, output_path)


def _load_lora_from_dir(lora_path: Path) -> Tuple[dict, dict]:
    """Load LoRA weights and config from a directory.

    Returns:
        Tuple of (state_dict, adapter_config).
    """
    adapter_config = {}
    state_dict = {}

    # Load adapter config if present
    config_path = lora_path / "adapter_config.json"
    if config_path.exists():
        with open(config_path) as f:
            adapter_config = json.load(f)

    # Try safetensors first
    safetensors_path = lora_path / "adapter_model.safetensors"
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file as st_load
            state_dict = st_load(str(safetensors_path))
            logger.info(f"Loaded safetensors: {len(state_dict)} tensors")
            return state_dict, adapter_config
        except ImportError:
            state_dict = _read_manual_safetensors(str(safetensors_path))
            if state_dict:
                logger.info(
                    f"Loaded safetensors (manual): {len(state_dict)} tensors"
                )
                return state_dict, adapter_config

    # Try PyTorch .bin format
    for bin_name in ("adapter_model.bin", "pytorch_model.bin"):
        bin_path = lora_path / bin_name
        if bin_path.exists():
            try:
                import torch
                state_dict = torch.load(
                    str(bin_path), map_location="cpu", weights_only=True
                )
                logger.info(f"Loaded {bin_name}: {len(state_dict)} tensors")
                return state_dict, adapter_config
            except ImportError:
                logger.warning("torch not available for loading .bin files")

    # Try single .pt file
    pt_files = list(lora_path.glob("*.pt"))
    if pt_files:
        try:
            import torch
            state_dict = torch.load(
                str(pt_files[0]), map_location="cpu", weights_only=True
            )
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            logger.info(f"Loaded {pt_files[0].name}: {len(state_dict)} tensors")
            return state_dict, adapter_config
        except ImportError:
            logger.warning("torch not available for loading .pt files")

    return state_dict, adapter_config


# ---------------------------------------------------------------------------
# TGSP manifest helpers
# ---------------------------------------------------------------------------

def _manifest_to_adapter_config(manifest: dict) -> dict:
    """Convert a TGSP manifest to HuggingFace PEFT adapter_config.json."""
    meta = manifest.get("metadata", {})
    lora_config = manifest.get("lora_config", {})

    return {
        "base_model_name_or_path": manifest.get("base_model", ""),
        "bias": lora_config.get("bias", "none"),
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": manifest.get("alpha", lora_config.get("alpha", 64)),
        "lora_dropout": lora_config.get("dropout", 0.0),
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": manifest.get("rank", lora_config.get("rank", 30)),
        "target_modules": lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        "task_type": lora_config.get("task_type", "CAUSAL_LM"),
        # TGSP provenance
        "tgsp_adapter_id": manifest.get("adapter_id", ""),
        "tgsp_creator": manifest.get("creator", ""),
        "tgsp_domain": meta.get("domain", "general"),
    }


def _build_import_manifest(
    name: str,
    domain: str,
    creator: str,
    description: str,
    license_type: str,
    base_model: str,
    adapter_config: dict,
    signing_key: Optional[str],
    tags: List[str],
) -> dict:
    """Build a TGSP manifest for an imported LoRA adapter."""
    import uuid

    adapter_id = str(uuid.uuid4())
    rank = adapter_config.get("r", adapter_config.get("rank", 30))
    alpha = adapter_config.get("lora_alpha", adapter_config.get("alpha", 64))

    manifest = {
        "format_version": "1.0",
        "adapter_id": adapter_id,
        "model_name": name,
        "model_version": "1.0.0",
        "base_model": base_model,
        "rank": rank,
        "alpha": alpha,
        "creator": creator,
        "license": license_type,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "domain": domain,
            "description": description,
            "tags": tags,
            "source": "lora_import",
            "expert_type": domain,
        },
        "lora_config": {
            "rank": rank,
            "alpha": alpha,
            "dropout": adapter_config.get("lora_dropout", 0.0),
            "target_modules": adapter_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            "bias": adapter_config.get("bias", "none"),
            "task_type": adapter_config.get("task_type", "CAUSAL_LM"),
        },
        "screening": {
            "rvu_v2_passed": False,
            "qa_verify_score": None,
            "dp_budget_verified": False,
        },
        "usage_metering": True,
        "price_per_1k_tokens": 0.0,
    }

    # Sign if key provided
    if signing_key:
        manifest["signature"] = _sign_manifest(manifest, signing_key)
        manifest["signed"] = True
    else:
        manifest["signed"] = False

    return manifest


def _sign_manifest(manifest: dict, signing_key: str) -> str:
    """Create HMAC-SHA256 signature of the manifest.

    Uses the manifest JSON (without the signature field) as the message.

    Args:
        manifest: Manifest dict (signature field will be excluded).
        signing_key: Hex-encoded signing key.

    Returns:
        Hex-encoded HMAC-SHA256 signature.
    """
    import hmac

    # Remove signature field if present for signing
    signable = {k: v for k, v in manifest.items() if k != "signature"}
    message = json.dumps(signable, sort_keys=True, ensure_ascii=False).encode("utf-8")

    try:
        key_bytes = bytes.fromhex(signing_key)
    except ValueError:
        key_bytes = signing_key.encode("utf-8")

    sig = hmac.new(key_bytes, message, hashlib.sha256).hexdigest()
    return sig


# ---------------------------------------------------------------------------
# Manual safetensors implementation (fallback when library not installed)
# ---------------------------------------------------------------------------

def _write_manual_safetensors(state_dict: dict, output_path: str) -> None:
    """Write a safetensors file without the safetensors library.

    safetensors format:
      - 8 bytes: header size (uint64 LE)
      - header_size bytes: JSON header mapping tensor names to
        {dtype, shape, data_offsets: [start, end]}
      - remaining bytes: raw tensor data
    """
    import numpy as np

    # Convert all values to numpy arrays
    tensors = {}
    for k, v in state_dict.items():
        if hasattr(v, "numpy"):
            arr = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            arr = v
        else:
            arr = np.array(v)
        tensors[k] = arr

    # Map numpy dtype to safetensors dtype string
    dtype_map = {
        np.float16: "F16",
        np.float32: "F32",
        np.float64: "F64",
        np.int32: "I32",
        np.int64: "I64",
        np.uint8: "U8",
        np.bool_: "BOOL",
    }

    # Build header and data buffer
    header: Dict[str, Any] = {}
    data_parts = []
    offset = 0

    for name in sorted(tensors.keys()):
        arr = tensors[name]
        raw = arr.tobytes()
        dtype_str = dtype_map.get(arr.dtype.type, "F32")
        shape = list(arr.shape)

        header[name] = {
            "dtype": dtype_str,
            "shape": shape,
            "data_offsets": [offset, offset + len(raw)],
        }
        data_parts.append(raw)
        offset += len(raw)

    # Add metadata
    header["__metadata__"] = {"format": "tgsp_export"}

    header_json = json.dumps(header, ensure_ascii=False).encode("utf-8")
    header_size = len(header_json)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(header_json)
        for part in data_parts:
            f.write(part)


def _read_manual_safetensors(safetensors_path: str) -> dict:
    """Read a safetensors file without the safetensors library.

    Returns a dict of name -> numpy array.
    """
    import numpy as np

    dtype_map = {
        "F16": np.float16,
        "F32": np.float32,
        "F64": np.float64,
        "BF16": np.float16,  # approximate: read as f16
        "I32": np.int32,
        "I64": np.int64,
        "U8": np.uint8,
        "BOOL": np.bool_,
    }

    try:
        with open(safetensors_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size)
            header = json.loads(header_json)
            data_start = 8 + header_size
            # Read all data
            data = f.read()

        result = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = dtype_map.get(info["dtype"], np.float32)
            shape = info["shape"]
            start, end = info["data_offsets"]
            raw = data[start:end]
            arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
            result[name] = arr.copy()

        return result
    except Exception as e:
        logger.warning(f"Failed to read safetensors manually: {e}")
        return {}


# ---------------------------------------------------------------------------
# GGUF writing (minimal implementation)
# ---------------------------------------------------------------------------

# GGUF magic and version constants
GGUF_MAGIC = 0x46475547  # "GGUF" in LE
GGUF_VERSION = 3

# GGUF value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10

# GGUF tensor types
GGUF_TENSOR_F16 = 1
GGUF_TENSOR_F32 = 0


def _write_gguf(state_dict: dict, manifest: dict, output_path: str) -> None:
    """Write LoRA weights in GGUF format.

    This is a minimal GGUF writer for LoRA adapter weights. The output
    can be used with llama.cpp's LoRA adapter loading.

    Args:
        state_dict: Dictionary of weight tensors.
        manifest: TGSP manifest for metadata.
        output_path: Destination .gguf path.
    """
    import numpy as np

    # Convert all weights to numpy float16
    tensors = {}
    for k, v in state_dict.items():
        if hasattr(v, "numpy"):
            arr = v.detach().cpu().float().numpy().astype(np.float16)
        elif isinstance(v, np.ndarray):
            arr = v.astype(np.float16)
        else:
            arr = np.array(v, dtype=np.float16)
        tensors[k] = arr

    # Build GGUF metadata key-value pairs
    metadata = {
        "general.architecture": "llama",
        "general.name": manifest.get("model_name", "tgsp_lora"),
        "general.type": "adapter",
        "adapter.type": "lora",
        "adapter.lora.alpha": float(manifest.get("alpha", 64)),
    }

    n_tensors = len(tensors)
    n_kv = len(metadata)

    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", n_tensors))
        f.write(struct.pack("<Q", n_kv))

        # Write metadata KV pairs
        for key, val in metadata.items():
            _gguf_write_string(f, key)
            if isinstance(val, str):
                f.write(struct.pack("<I", GGUF_TYPE_STRING))
                _gguf_write_string(f, val)
            elif isinstance(val, float):
                f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
                f.write(struct.pack("<f", val))
            elif isinstance(val, int):
                f.write(struct.pack("<I", GGUF_TYPE_UINT32))
                f.write(struct.pack("<I", val))

        # Tensor info section
        # Calculate data offset (align to 32 bytes)
        tensor_data_parts = []
        data_offset = 0
        tensor_infos = []

        for name in sorted(tensors.keys()):
            arr = tensors[name]
            raw = arr.tobytes()

            # Align offset to 32 bytes
            aligned_offset = (data_offset + 31) & ~31
            padding = aligned_offset - data_offset

            tensor_infos.append({
                "name": name,
                "n_dims": len(arr.shape),
                "dims": list(arr.shape),
                "type": GGUF_TENSOR_F16,
                "offset": aligned_offset,
            })
            tensor_data_parts.append((padding, raw))
            data_offset = aligned_offset + len(raw)

        # Write tensor infos
        for info in tensor_infos:
            _gguf_write_string(f, info["name"])
            f.write(struct.pack("<I", info["n_dims"]))
            for dim in info["dims"]:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", info["type"]))
            f.write(struct.pack("<Q", info["offset"]))

        # Align to 32 bytes before tensor data
        current_pos = f.tell()
        align_padding = (32 - (current_pos % 32)) % 32
        f.write(b"\x00" * align_padding)

        # Write tensor data
        for padding, raw in tensor_data_parts:
            f.write(b"\x00" * padding)
            f.write(raw)


def _gguf_write_string(f, s: str) -> None:
    """Write a GGUF string (length-prefixed UTF-8)."""
    encoded = s.encode("utf-8")
    f.write(struct.pack("<Q", len(encoded)))
    f.write(encoded)
