"""TenSafe SDK — TGSP Builder and Adapter APIs.

High-level API for creating, loading, and using TGSP adapter files.

Quick start:
    import tensafe.sdk as sdk

    # Build a TGSP adapter
    builder = sdk.TGSPBuilder("my-adapter")
    builder.set_lora_config(rank=30, alpha=64)
    builder.set_creator(name="Alice", organization="TenSafe")
    builder.set_payload_file("weights.safetensors")
    builder.build("my-adapter.tgsp")

    # Load and verify
    adapter = sdk.TGSPAdapter.load("my-adapter.tgsp")
    result = adapter.verify()
    print(result.summary)

Convenience functions:
    sdk.encrypt(data)     -- CKKS encrypt a vector (stub)
    sdk.decrypt(ct)       -- CKKS decrypt a ciphertext (stub)
    sdk.infer(adapter_id, query) -- Run inference through the runtime
"""

from tensafe.sdk.tgsp_builder import (
    CreatorConfig,
    LoraConfig,
    ModelConfig,
    RVUSafetyRecord,
    SkillConfig,
    TGSPBuilder,
)
from tensafe.sdk.tgsp_adapter import (
    TGSPAdapter,
    TGSPManifest,
    VerificationResult,
)

import numpy as np
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def encrypt(data: np.ndarray) -> np.ndarray:
    """CKKS-encrypt a vector (stub: returns copy).

    In production, this delegates to tensafe-he (Rust/CUDA) for real
    CKKS encryption. The stub returns an unmodified copy.

    Args:
        data: 1-D numpy array to encrypt.

    Returns:
        Encrypted ciphertext (stub: plaintext copy).
    """
    return data.copy().astype(np.float64)


def decrypt(ciphertext: np.ndarray) -> np.ndarray:
    """CKKS-decrypt a ciphertext (stub: returns copy).

    Args:
        ciphertext: Encrypted data from encrypt().

    Returns:
        Decrypted plaintext vector.
    """
    return ciphertext.copy()


def infer(
    adapter_id: str,
    query: str,
    max_tokens: int = 256,
    runtime: Optional[Any] = None,
) -> Any:
    """Run HE-encrypted inference through the TenSafe runtime.

    Convenience wrapper that uses the default runtime if none is
    provided.

    Args:
        adapter_id: ID of a loaded adapter.
        query: Input query text.
        max_tokens: Maximum tokens to generate.
        runtime: TenSafeRuntime instance (uses default if None).

    Returns:
        InferenceResult from the runtime.

    Raises:
        RuntimeError: If no runtime is available.
    """
    if runtime is None:
        raise RuntimeError(
            "No runtime provided. Create a TenSafeRuntime and pass it, "
            "or use runtime.infer() directly."
        )
    return runtime.infer(adapter_id, query, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Adapter convenience
# ---------------------------------------------------------------------------

class adapter:
    """Namespace for adapter convenience functions."""

    @staticmethod
    def load(tgsp_path: str) -> TGSPAdapter:
        """Load a TGSP adapter from file.

        Shorthand for ``TGSPAdapter.load(path)``.
        """
        return TGSPAdapter.load(tgsp_path)

    @staticmethod
    def verify(tgsp_path: str) -> VerificationResult:
        """Load and verify a TGSP adapter.

        Shorthand for ``TGSPAdapter.load(path).verify()``.
        """
        return TGSPAdapter.load(tgsp_path).verify()

    @staticmethod
    def build(name: str) -> TGSPBuilder:
        """Create a new TGSPBuilder.

        Shorthand for ``TGSPBuilder(name)``.
        """
        return TGSPBuilder(name)


__all__ = [
    "TGSPBuilder",
    "TGSPAdapter",
    "TGSPManifest",
    "VerificationResult",
    "LoraConfig",
    "ModelConfig",
    "CreatorConfig",
    "SkillConfig",
    "RVUSafetyRecord",
    "encrypt",
    "decrypt",
    "infer",
    "adapter",
]
