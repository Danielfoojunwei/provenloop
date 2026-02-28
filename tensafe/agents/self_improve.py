"""
Self-improvement loop engine for TenSafe agents.

Agents that detect quality degradation can trigger a self-improvement
cycle: analyze gaps -> train improved adapter -> red-team the swap ->
deploy if approved.

All self-improvement generates metered inference (training is 100-1000x
more tokens than inference), creating the revenue flywheel.

Components:
    SelfImprovementEngine: Original cycle-based improvement pipeline.
    SelfImproveLoop:       Continuous background loop with A/B testing.
    QualityEvaluator:      Heuristic quality scoring for responses.
    QualityMetrics:        Per-adapter quality statistics.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class ImprovementCycle:
    """Record of a self-improvement cycle."""
    cycle_id: str
    adapter_id: str
    timestamp: str
    trigger: str  # "quality_drop", "scheduled", "manual"
    quality_before: float
    quality_after: float = 0.0
    improvement_percent: float = 0.0
    new_adapter_id: str = ""
    swap_approved: bool = False
    status: str = "pending"  # pending, training, evaluating, red_teaming, complete, failed
    tokens_consumed: int = 0
    error: str = ""


@dataclass
class ImprovementConfig:
    """Configuration for self-improvement loops."""
    min_quality_threshold: float = 0.80
    min_improvement_percent: float = 2.0
    max_dp_epsilon_per_cycle: float = 1.0
    max_training_examples: int = 10000
    ab_test_traffic_percent: float = 10.0
    auto_promote_threshold: float = 0.85


class SelfImprovementEngine:
    """
    Manages self-improvement cycles for agents.

    When an agent's quality drops below threshold:
    1. Analyze interaction logs for gaps
    2. Curate targeted training data
    3. Train improved adapter via TG Tinker
    4. Evaluate improvement on held-out set
    5. Red-team the swap (MANDATORY)
    6. A/B test with subset of traffic
    7. Promote if improved
    """

    def __init__(self, config: ImprovementConfig | None = None):
        self.config = config or ImprovementConfig()
        self._cycles: list[ImprovementCycle] = []
        self._cycle_counter = 0

    async def trigger_improvement(
        self,
        adapter_id: str,
        interaction_log: list[dict],
        current_quality: float,
        trigger: str = "quality_drop",
    ) -> ImprovementCycle:
        """
        Trigger a self-improvement cycle for an adapter.

        Returns the cycle record. The actual swap is handled by MetaAgent
        after red-teaming.
        """
        self._cycle_counter += 1
        cycle = ImprovementCycle(
            cycle_id=f"cycle-{self._cycle_counter:06d}",
            adapter_id=adapter_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            trigger=trigger,
            quality_before=current_quality,
        )
        self._cycles.append(cycle)

        try:
            # Step 1: Analyze gaps
            cycle.status = "analyzing"
            gaps = self._analyze_gaps(interaction_log)
            logger.info(f"Cycle {cycle.cycle_id}: found {len(gaps)} knowledge gaps")

            # Step 2: Curate training data from gaps
            training_data = self._curate_from_gaps(gaps, interaction_log)
            logger.info(f"Cycle {cycle.cycle_id}: curated {len(training_data)} training examples")

            # Step 3: Train improved adapter (via TG Tinker)
            cycle.status = "training"
            new_adapter_id = await self._train_improved(
                adapter_id, training_data, cycle
            )
            cycle.new_adapter_id = new_adapter_id

            # Step 4: Evaluate
            cycle.status = "evaluating"
            new_quality = await self._evaluate(new_adapter_id)
            cycle.quality_after = new_quality
            cycle.improvement_percent = (
                (new_quality - current_quality) / max(current_quality, 0.01) * 100
            )

            # Step 5: Quality gate
            if cycle.improvement_percent < self.config.min_improvement_percent:
                cycle.status = "failed"
                cycle.error = (
                    f"Improvement {cycle.improvement_percent:.1f}% < "
                    f"minimum {self.config.min_improvement_percent}%"
                )
                logger.warning(f"Cycle {cycle.cycle_id}: {cycle.error}")
                return cycle

            # Step 6: Ready for red-teaming (MetaAgent handles this)
            cycle.status = "red_teaming"
            logger.info(
                f"Cycle {cycle.cycle_id}: improvement {cycle.improvement_percent:.1f}%, "
                f"ready for red-team approval"
            )

        except Exception as e:
            cycle.status = "failed"
            cycle.error = str(e)
            logger.error(f"Cycle {cycle.cycle_id} failed: {e}")

        return cycle

    def mark_swap_approved(self, cycle_id: str, approved: bool) -> None:
        """Mark a cycle's swap as approved or rejected after red-teaming."""
        for cycle in self._cycles:
            if cycle.cycle_id == cycle_id:
                cycle.swap_approved = approved
                cycle.status = "complete" if approved else "failed"
                if not approved:
                    cycle.error = "Swap rejected by red-team"
                break

    def _analyze_gaps(self, interaction_log: list[dict]) -> list[dict]:
        """Analyze interaction logs to identify knowledge gaps."""
        gaps = []
        for entry in interaction_log:
            quality = entry.get("quality_score", 1.0)
            if quality < self.config.min_quality_threshold:
                gaps.append({
                    "query": entry.get("query", ""),
                    "quality": quality,
                    "weakness": entry.get("weakness", "low_quality_response"),
                })
        return gaps

    def _curate_from_gaps(
        self, gaps: list[dict], interaction_log: list[dict]
    ) -> list[dict]:
        """Create targeted training examples from identified gaps."""
        examples = []
        for gap in gaps[:self.config.max_training_examples]:
            examples.append({
                "instruction": gap["query"],
                "response": "",  # To be filled by improved training
                "weakness": gap["weakness"],
            })
        return examples

    async def _train_improved(
        self,
        adapter_id: str,
        training_data: list[dict],
        cycle: ImprovementCycle,
    ) -> str:
        """Train an improved adapter using TG Tinker (stub)."""
        # In production: invoke TG Tinker with training data
        new_id = f"{adapter_id}-v{self._cycle_counter}"
        cycle.tokens_consumed = len(training_data) * 500  # Estimate
        return new_id

    async def _evaluate(self, adapter_id: str) -> float:
        """Evaluate adapter quality (stub)."""
        # In production: run qa_verify against held-out test set
        return 0.85

    def get_cycles(self, adapter_id: str | None = None) -> list[ImprovementCycle]:
        """Get improvement cycle history."""
        if adapter_id:
            return [c for c in self._cycles if c.adapter_id == adapter_id]
        return list(self._cycles)

    def get_stats(self) -> dict:
        """Get self-improvement statistics."""
        total = len(self._cycles)
        successful = sum(1 for c in self._cycles if c.status == "complete")
        failed = sum(1 for c in self._cycles if c.status == "failed")
        avg_improvement = (
            sum(c.improvement_percent for c in self._cycles if c.status == "complete")
            / max(successful, 1)
        )
        return {
            "total_cycles": total,
            "successful": successful,
            "failed": failed,
            "avg_improvement_percent": round(avg_improvement, 2),
            "total_tokens_consumed": sum(c.tokens_consumed for c in self._cycles),
        }
