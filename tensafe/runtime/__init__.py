"""TenSafe Runtime — proprietary execution engine for TGSP adapters.

Exports:
    TenSafeRuntime: Core runtime that loads and runs TGSP adapters with
                    HE-encrypted inference, 7-step load gate verification,
                    per-token metering, and atomic hot-swap.
    RuntimeConfig:  Configuration dataclass for the runtime.
    TGSPLoadGate:   7-step TGSP verification gate.
    MeteringService: Per-token inference metering with billing integration.
"""

from tensafe.runtime.engine import RuntimeConfig, TenSafeRuntime
from tensafe.runtime.load_gate import GateResult, GateVerdict, TGSPLoadGate
from tensafe.runtime.metering import MeteringService, UsageTier

__all__ = [
    "TenSafeRuntime",
    "RuntimeConfig",
    "TGSPLoadGate",
    "GateResult",
    "GateVerdict",
    "MeteringService",
    "UsageTier",
]
