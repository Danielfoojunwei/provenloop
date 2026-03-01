"""TenSafe Agents — meta-agent orchestrator with self-improvement.

Exports:
    MetaAgent:        Orchestrates specialized agents with TGSP adapters.
    AgentResult:      Result of an agent task execution.
    AtomicSwap:       Low-level atomic adapter swap with rollback.
    HotSwapManager:   High-level hot-swap orchestrator with A/B routing.
    SelfImproveLoop:  Quality evaluation and automatic adapter improvement.
"""

from tensafe.agents.meta_agent import (
    AgentInstance,
    AgentResult,
    MetaAgent,
    SwapApprovalMode,
    SwapRedTeamCheck,
    SwapRedTeamResult,
)
from tensafe.agents.hot_swap import (
    AtomicSwap,
    HotSwapManager,
    SwapRecord,
    SwapResult,
    SwapState,
)
from tensafe.agents.self_improve import (
    ImprovementResult,
    QualityMetrics,
    SelfImproveLoop,
)

__all__ = [
    "MetaAgent",
    "AgentInstance",
    "AgentResult",
    "SwapApprovalMode",
    "SwapRedTeamCheck",
    "SwapRedTeamResult",
    "AtomicSwap",
    "HotSwapManager",
    "SwapRecord",
    "SwapResult",
    "SwapState",
    "SelfImproveLoop",
    "ImprovementResult",
    "QualityMetrics",
]
