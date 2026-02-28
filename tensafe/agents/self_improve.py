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


# =========================================================================
# Quality metrics and evaluation
# =========================================================================

@dataclass
class QualityMetrics:
    """Per-adapter quality statistics with percentile tracking."""
    adapter_id: str
    sample_count: int = 0
    mean_score: float = 0.0
    median_score: float = 0.0
    p10_score: float = 0.0   # 10th percentile (worst-case)
    p90_score: float = 0.0   # 90th percentile (best-case)
    scores: List[float] = field(default_factory=list, repr=False)
    category_scores: Dict[str, List[float]] = field(default_factory=dict, repr=False)

    def update(self, score: float, category: str = "general") -> None:
        """Add a quality observation and recompute statistics."""
        self.scores.append(score)
        self.sample_count = len(self.scores)
        self.mean_score = statistics.mean(self.scores)

        if self.sample_count >= 2:
            self.median_score = statistics.median(self.scores)
            sorted_s = sorted(self.scores)
            idx_10 = max(0, int(self.sample_count * 0.1))
            idx_90 = min(self.sample_count - 1, int(self.sample_count * 0.9))
            self.p10_score = sorted_s[idx_10]
            self.p90_score = sorted_s[idx_90]
        else:
            self.median_score = self.mean_score
            self.p10_score = self.mean_score
            self.p90_score = self.mean_score

        if category not in self.category_scores:
            self.category_scores[category] = []
        self.category_scores[category].append(score)


@dataclass
class InteractionRecord:
    """Interaction log entry for gap analysis."""
    interaction_id: str
    adapter_id: str
    query: str
    response: str
    quality_score: float
    category: str = "general"
    timestamp: float = field(default_factory=time.time)
    feedback: Optional[str] = None


@dataclass
class GapReport:
    """Identified quality gap for a category."""
    category: str
    mean_score: float
    sample_count: int
    below_threshold: bool
    gap_severity: float  # 0.0 to 1.0
    sample_queries: List[str] = field(default_factory=list)
    recommended_action: str = ""


class ImprovementStatus(enum.Enum):
    """Status of an improvement cycle."""
    PENDING = "pending"
    TRAINING = "training"
    TRAINED = "trained"
    AB_TESTING = "ab_testing"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class ImprovementResult:
    """Result of a self-improvement cycle with A/B test data."""
    improvement_id: str
    adapter_id: str
    candidate_id: str
    status: ImprovementStatus
    gap_analysis: List[GapReport] = field(default_factory=list)
    training_started_at: float = 0.0
    training_completed_at: float = 0.0
    training_loss: float = 0.0
    ab_test_started_at: float = 0.0
    ab_test_completed_at: float = 0.0
    ab_traffic_pct: float = 0.0
    ab_original_score: float = 0.0
    ab_candidate_score: float = 0.0
    ab_improvement_pct: float = 0.0
    promoted: bool = False
    promotion_reason: str = ""
    error: str = ""


# =========================================================================
# Quality evaluator
# =========================================================================

class QualityEvaluator:
    """Evaluate agent response quality.

    In production this would use LLM-as-judge or task-specific metrics.
    The stub uses heuristic scoring.
    """

    def __init__(self, custom_scorer: Optional[Callable] = None):
        self._custom_scorer = custom_scorer

    def score(self, query: str, response: str, category: str = "general") -> float:
        """Score a response (0.0 - 1.0)."""
        if self._custom_scorer:
            return self._custom_scorer(query, response, category)

        if not response or not response.strip():
            return 0.0

        score = 0.0
        query_words = len(query.split())
        response_words = len(response.split())

        if response_words == 0:
            return 0.0

        # Length ratio scoring
        ratio = response_words / max(query_words, 1)
        if ratio < 0.5:
            score += 0.1
        elif ratio < 2.0:
            score += 0.3
        elif ratio < 20.0:
            score += 0.5
        else:
            score += 0.3

        # Uniqueness penalty
        unique_words = len(set(response.lower().split()))
        uniqueness = unique_words / max(response_words, 1)
        score += uniqueness * 0.3

        # Structured content bonus
        if any(m in response for m in ["\n-", "\n*", "```", "1.", "2."]):
            score += 0.1

        # Bad response penalty
        if any(m.lower() in response.lower() for m in
               ["I don't know", "I cannot", "error", "undefined"]):
            score -= 0.2

        return max(0.0, min(1.0, score))


# =========================================================================
# TG Tinker training stub
# =========================================================================

class _TGTinkerStub:
    """Stub for the TG Tinker LoRA fine-tuning service."""

    async def request_training(
        self,
        adapter_id: str,
        weak_interactions: List[InteractionRecord],
        gap_analysis: List[GapReport],
        target_threshold: float,
    ) -> tuple[str, float]:
        """Request improved adapter training.

        Returns (candidate_adapter_id, training_loss).
        """
        logger.info(
            "TG Tinker: training for %s (%d weak, %d gaps)",
            adapter_id, len(weak_interactions), len(gap_analysis),
        )
        await asyncio.sleep(0.1)  # Simulate training
        candidate_id = f"{adapter_id}-improved-{uuid.uuid4().hex[:8]}"
        return candidate_id, 0.05


# =========================================================================
# SelfImproveLoop — continuous quality monitoring with A/B testing
# =========================================================================

class SelfImproveLoop:
    """Continuous self-improvement loop with A/B testing and promotion.

    Monitors adapter quality, identifies gaps, triggers TG Tinker
    training, runs A/B tests, and promotes improved adapters.

    Args:
        runtime: TenSafeRuntime instance.
        hot_swap_manager: HotSwapManager for traffic splitting.
        quality_threshold: Minimum acceptable quality score.
        ab_traffic_pct: Traffic percentage routed to candidate in A/B test.
        ab_min_samples: Minimum samples before concluding A/B test.
        ab_improvement_threshold: Minimum improvement to promote (e.g. 0.02 = 2%).
    """

    def __init__(
        self,
        runtime: Any = None,
        hot_swap_manager: Any = None,
        quality_threshold: float = 0.80,
        ab_traffic_pct: float = 0.10,
        ab_min_samples: int = 50,
        ab_improvement_threshold: float = 0.02,
    ):
        self._runtime = runtime
        self._hot_swap = hot_swap_manager
        self._quality_threshold = quality_threshold
        self._ab_traffic_pct = ab_traffic_pct
        self._ab_min_samples = ab_min_samples
        self._ab_improvement_threshold = ab_improvement_threshold

        self._metrics: Dict[str, QualityMetrics] = {}
        self._interactions: List[InteractionRecord] = []
        self._improvements: List[ImprovementResult] = []
        self._active_ab_tests: Dict[str, ImprovementResult] = {}

        self._evaluator = QualityEvaluator()
        self._tinker = _TGTinkerStub()
        self._running = False
        self._loop_interval_s = 60.0

    # ------------------------------------------------------------------
    # Quality tracking (called per inference)
    # ------------------------------------------------------------------

    def record_interaction(
        self,
        adapter_id: str,
        query: str,
        response: str,
        category: str = "general",
        feedback: Optional[str] = None,
    ) -> float:
        """Record an interaction and return quality score."""
        score = self._evaluator.score(query, response, category)

        if adapter_id not in self._metrics:
            self._metrics[adapter_id] = QualityMetrics(adapter_id=adapter_id)
        self._metrics[adapter_id].update(score, category)

        self._interactions.append(InteractionRecord(
            interaction_id=uuid.uuid4().hex[:12],
            adapter_id=adapter_id,
            query=query,
            response=response,
            quality_score=score,
            category=category,
            feedback=feedback,
        ))

        # Keep last 10k interactions
        if len(self._interactions) > 10_000:
            self._interactions = self._interactions[-10_000:]

        return score

    def get_metrics(self, adapter_id: str) -> Optional[QualityMetrics]:
        """Get quality metrics for an adapter."""
        return self._metrics.get(adapter_id)

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------

    def analyze_gaps(self, adapter_id: str) -> List[GapReport]:
        """Identify quality gaps by category."""
        metrics = self._metrics.get(adapter_id)
        if metrics is None or metrics.sample_count < 10:
            return []

        gaps = []
        for category, scores in metrics.category_scores.items():
            if len(scores) < 5:
                continue

            mean = statistics.mean(scores)
            below = mean < self._quality_threshold
            severity = min(1.0, max(0.0,
                (self._quality_threshold - mean) / self._quality_threshold
            )) if below else 0.0

            weak = [
                i for i in self._interactions
                if i.adapter_id == adapter_id
                and i.category == category
                and i.quality_score < self._quality_threshold
            ]

            action = ""
            if severity > 0.5:
                action = "Urgent: retrain with focused data"
            elif severity > 0.2:
                action = "Recommended: add training examples"
            elif below:
                action = "Monitor: slightly below threshold"

            gaps.append(GapReport(
                category=category,
                mean_score=mean,
                sample_count=len(scores),
                below_threshold=below,
                gap_severity=severity,
                sample_queries=[i.query for i in weak[:5]],
                recommended_action=action,
            ))

        gaps.sort(key=lambda g: g.gap_severity, reverse=True)
        return gaps

    # ------------------------------------------------------------------
    # Improvement pipeline
    # ------------------------------------------------------------------

    async def evaluate_and_improve(self, adapter_id: str) -> Optional[ImprovementResult]:
        """Run one improvement cycle: evaluate, train, start A/B test."""
        metrics = self._metrics.get(adapter_id)
        if metrics is None or metrics.sample_count < 10:
            return None

        if adapter_id in self._active_ab_tests:
            return self._active_ab_tests[adapter_id]

        if metrics.mean_score >= self._quality_threshold:
            return None

        logger.info(
            "Quality %.2f < %.2f for %s, starting improvement",
            metrics.mean_score, self._quality_threshold, adapter_id,
        )

        gaps = self.analyze_gaps(adapter_id)
        weak = [
            i for i in self._interactions
            if i.adapter_id == adapter_id
            and i.quality_score < self._quality_threshold
        ]

        improvement = ImprovementResult(
            improvement_id=uuid.uuid4().hex[:12],
            adapter_id=adapter_id,
            candidate_id="",
            status=ImprovementStatus.TRAINING,
            gap_analysis=gaps,
            training_started_at=time.time(),
        )

        try:
            candidate_id, loss = await self._tinker.request_training(
                adapter_id=adapter_id,
                weak_interactions=weak[-100:],
                gap_analysis=gaps,
                target_threshold=self._quality_threshold,
            )
            improvement.candidate_id = candidate_id
            improvement.training_loss = loss
            improvement.training_completed_at = time.time()
            improvement.status = ImprovementStatus.TRAINED
        except Exception as e:
            improvement.status = ImprovementStatus.FAILED
            improvement.error = str(e)
            self._improvements.append(improvement)
            return improvement

        # Start A/B test
        improvement.status = ImprovementStatus.AB_TESTING
        improvement.ab_test_started_at = time.time()
        improvement.ab_traffic_pct = self._ab_traffic_pct
        improvement.ab_original_score = metrics.mean_score

        if self._hot_swap:
            self._hot_swap.set_traffic_split(
                agent_id=adapter_id,
                splits={
                    adapter_id: 1.0 - self._ab_traffic_pct,
                    candidate_id: self._ab_traffic_pct,
                },
            )

        self._active_ab_tests[adapter_id] = improvement
        self._improvements.append(improvement)

        logger.info(
            "A/B test started: %s vs %s (%.0f%% candidate)",
            adapter_id, candidate_id, self._ab_traffic_pct * 100,
        )
        return improvement

    async def check_ab_test(self, adapter_id: str) -> Optional[ImprovementResult]:
        """Check A/B test and promote/reject candidate."""
        improvement = self._active_ab_tests.get(adapter_id)
        if improvement is None:
            return None

        candidate_metrics = self._metrics.get(improvement.candidate_id)
        original_metrics = self._metrics.get(adapter_id)

        if (candidate_metrics is None
                or candidate_metrics.sample_count < self._ab_min_samples):
            return improvement

        orig = original_metrics.mean_score if original_metrics else 0.0
        cand = candidate_metrics.mean_score
        improvement.ab_original_score = orig
        improvement.ab_candidate_score = cand
        improvement.ab_improvement_pct = (
            (cand - orig) / orig if orig > 0 else (1.0 if cand > 0 else 0.0)
        )

        if improvement.ab_improvement_pct >= self._ab_improvement_threshold:
            improvement.status = ImprovementStatus.PROMOTED
            improvement.promoted = True
            improvement.promotion_reason = (
                f"Candidate {cand:.2f} vs original {orig:.2f} "
                f"(+{improvement.ab_improvement_pct:.1%})"
            )
            if self._hot_swap:
                self._hot_swap.promote_adapter(adapter_id, improvement.candidate_id)
            logger.info("A/B PROMOTED: %s", improvement.promotion_reason)
        else:
            improvement.status = ImprovementStatus.REJECTED
            improvement.promotion_reason = (
                f"Insufficient: {improvement.ab_improvement_pct:.1%} "
                f"(need >= {self._ab_improvement_threshold:.1%})"
            )
            if self._hot_swap:
                self._hot_swap.promote_adapter(adapter_id, adapter_id)
            logger.info("A/B REJECTED: %s", improvement.promotion_reason)

        improvement.ab_test_completed_at = time.time()
        self._active_ab_tests.pop(adapter_id, None)
        return improvement

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def run_loop(self, adapter_ids: Sequence[str]) -> None:
        """Run continuous self-improvement as a background task."""
        self._running = True
        logger.info(
            "Self-improve loop started (adapters=%d, interval=%.0fs)",
            len(adapter_ids), self._loop_interval_s,
        )

        while self._running:
            for adapter_id in adapter_ids:
                try:
                    if adapter_id in self._active_ab_tests:
                        await self.check_ab_test(adapter_id)
                    else:
                        await self.evaluate_and_improve(adapter_id)
                except Exception as e:
                    logger.error("Self-improve error for %s: %s", adapter_id, e)

            await asyncio.sleep(self._loop_interval_s)

    def stop_loop(self) -> None:
        """Signal the background loop to stop."""
        self._running = False

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_improvement_history(self) -> List[ImprovementResult]:
        """Get full improvement history."""
        return list(self._improvements)

    def get_active_ab_tests(self) -> Dict[str, ImprovementResult]:
        """Get active A/B tests."""
        return dict(self._active_ab_tests)

    def get_all_metrics(self) -> Dict[str, QualityMetrics]:
        """Get quality metrics for all tracked adapters."""
        return dict(self._metrics)
