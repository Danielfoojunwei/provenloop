"""Differential privacy budget tracker using advanced composition.

Tracks cumulative epsilon spend per session using the advanced composition
theorem (Dwork, Rothblum & Vadhan 2010):

    epsilon_total(T) = sqrt(2T * ln(1/delta')) * epsilon + T * epsilon * (e^epsilon - 1)

where delta' = delta / (2T).
"""

import math
import threading
from dataclasses import dataclass, field


@dataclass
class PrivacyState:
    """Per-session privacy accounting state."""
    total_epsilon: float = 0.0
    total_requests: int = 0
    budget_exhausted: bool = False


class PrivacyBudgetTracker:
    """Track differential privacy budget across sessions.

    Thread-safe: uses a lock to protect concurrent read-modify-write
    on per-session state (multiple async requests may call consume()
    concurrently for the same session).

    Args:
        max_epsilon: Maximum cumulative epsilon before refusing queries.
        delta: DP delta parameter (probability of privacy breach).
    """

    def __init__(self, max_epsilon: float = 10.0, delta: float = 1e-5):
        self.max_epsilon = max_epsilon
        self.delta = delta
        self._states: dict[str, PrivacyState] = {}
        self._lock = threading.Lock()

    def _advanced_composition(self, per_query_eps: float, num_queries: int) -> float:
        """Advanced composition theorem for (eps, delta)-DP."""
        if num_queries == 0:
            return 0.0
        t = num_queries
        eps = per_query_eps
        delta_prime = self.delta / (2 * t) if t > 0 else self.delta
        if delta_prime <= 0 or delta_prime >= 1:
            return t * eps
        return math.sqrt(2 * t * math.log(1.0 / delta_prime)) * eps + t * eps * (math.exp(eps) - 1)

    def consume(self, epsilon: float, session_id: str = "default") -> tuple:
        """Consume privacy budget for a query.

        Returns:
            (budget_ok, state): budget_ok is True if within budget.
        """
        with self._lock:
            state = self._states.setdefault(session_id, PrivacyState())

            # Compute what epsilon_total would be after this query
            projected = self._advanced_composition(epsilon, state.total_requests + 1)

            if projected > self.max_epsilon:
                state.budget_exhausted = True
                return False, state

            state.total_requests += 1
            state.total_epsilon = projected
            return True, state

    def get_state(self, session_id: str = "default") -> PrivacyState:
        """Get current privacy state for a session."""
        with self._lock:
            return self._states.get(session_id, PrivacyState())
