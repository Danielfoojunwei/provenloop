"""Per-token inference metering with billing integration.

Tracks tokens per adapter, per tenant, per session.  Provides usage
reports, rate limiting for free-tier users, and Stripe billing stubs.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

class UsageTier(enum.Enum):
    """Pricing tiers for inference metering."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TierLimits:
    """Rate limits and pricing for a usage tier."""
    tier: UsageTier
    tokens_per_minute: int
    tokens_per_day: int
    max_concurrent_sessions: int
    price_per_1k_tokens: float  # USD
    overage_price_per_1k_tokens: float  # USD (0 = hard limit)

    @staticmethod
    def defaults() -> Dict[UsageTier, "TierLimits"]:
        return {
            UsageTier.FREE: TierLimits(
                tier=UsageTier.FREE,
                tokens_per_minute=100,
                tokens_per_day=10_000,
                max_concurrent_sessions=1,
                price_per_1k_tokens=0.0,
                overage_price_per_1k_tokens=0.0,  # hard cap
            ),
            UsageTier.STARTER: TierLimits(
                tier=UsageTier.STARTER,
                tokens_per_minute=1_000,
                tokens_per_day=500_000,
                max_concurrent_sessions=5,
                price_per_1k_tokens=0.001,
                overage_price_per_1k_tokens=0.002,
            ),
            UsageTier.PROFESSIONAL: TierLimits(
                tier=UsageTier.PROFESSIONAL,
                tokens_per_minute=10_000,
                tokens_per_day=5_000_000,
                max_concurrent_sessions=25,
                price_per_1k_tokens=0.0008,
                overage_price_per_1k_tokens=0.0012,
            ),
            UsageTier.ENTERPRISE: TierLimits(
                tier=UsageTier.ENTERPRISE,
                tokens_per_minute=100_000,
                tokens_per_day=100_000_000,
                max_concurrent_sessions=500,
                price_per_1k_tokens=0.0005,
                overage_price_per_1k_tokens=0.0008,
            ),
        }


# ---------------------------------------------------------------------------
# Usage tracking records
# ---------------------------------------------------------------------------

@dataclass
class SessionUsage:
    """Per-session token usage."""
    session_id: str
    tenant_id: str
    adapter_id: str
    tokens_generated: int = 0
    tokens_prompt: int = 0
    requests: int = 0
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


@dataclass
class TenantUsage:
    """Aggregate usage for a tenant across all sessions."""
    tenant_id: str
    tier: UsageTier = UsageTier.FREE
    total_tokens: int = 0
    total_requests: int = 0
    tokens_today: int = 0
    tokens_this_minute: int = 0
    minute_window_start: float = field(default_factory=time.time)
    day_window_start: float = field(default_factory=time.time)
    active_sessions: int = 0
    stripe_customer_id: Optional[str] = None


@dataclass
class AdapterUsage:
    """Aggregate usage for a single adapter across all tenants."""
    adapter_id: str
    total_tokens: int = 0
    total_requests: int = 0
    unique_tenants: int = 0
    revenue_usd: float = 0.0


@dataclass
class MeteringEvent:
    """A single metering event for the audit log."""
    timestamp: float
    tenant_id: str
    session_id: str
    adapter_id: str
    tokens_prompt: int
    tokens_generated: int
    cost_usd: float
    rate_limited: bool = False


@dataclass
class UsageReport:
    """Usage report for a tenant."""
    tenant_id: str
    tier: UsageTier
    period_start: float
    period_end: float
    total_tokens: int
    total_requests: int
    total_cost_usd: float
    by_adapter: Dict[str, AdapterUsage] = field(default_factory=dict)
    events: List[MeteringEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stripe billing stubs
# ---------------------------------------------------------------------------

class StripeBillingStub:
    """Stub for Stripe billing integration.

    In production, this would use the Stripe Python SDK to create
    metered usage records and manage subscriptions.
    """

    def __init__(self, api_key: str = "sk_test_stub"):
        self._api_key = api_key
        self._pending_usage: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def record_usage(
        self,
        customer_id: str,
        tokens: int,
        adapter_id: str,
        price_per_1k: float,
    ) -> str:
        """Record metered usage for billing.

        Returns:
            Usage record ID (stub: sequential counter).
        """
        with self._lock:
            record = {
                "id": f"ur_{len(self._pending_usage) + 1:06d}",
                "customer_id": customer_id,
                "tokens": tokens,
                "adapter_id": adapter_id,
                "amount_usd": (tokens / 1000.0) * price_per_1k,
                "timestamp": time.time(),
                "status": "pending",
            }
            self._pending_usage.append(record)
            logger.debug(
                "Stripe stub: recorded %d tokens for customer %s ($%.6f)",
                tokens, customer_id, record["amount_usd"],
            )
            return record["id"]

    def flush(self) -> List[Dict[str, Any]]:
        """Flush pending usage records (would send to Stripe API).

        Returns:
            List of flushed records.
        """
        with self._lock:
            flushed = self._pending_usage[:]
            for rec in flushed:
                rec["status"] = "flushed"
            self._pending_usage.clear()
            logger.info("Stripe stub: flushed %d usage records", len(flushed))
            return flushed

    def create_customer(self, tenant_id: str, email: str) -> str:
        """Create a Stripe customer (stub).

        Returns:
            Stub customer ID.
        """
        customer_id = f"cus_{tenant_id[:16]}"
        logger.info("Stripe stub: created customer %s for %s", customer_id, email)
        return customer_id


# ---------------------------------------------------------------------------
# MeteringService — the main metering engine
# ---------------------------------------------------------------------------

class MeteringService:
    """Per-token inference metering with rate limiting and billing.

    Thread-safe: all mutable state is protected by a lock.

    Args:
        config: RuntimeConfig (must have ``tier_limits`` and ``billing_enabled``).
    """

    def __init__(self, config: Any):
        self._config = config
        self._lock = threading.Lock()

        # State
        self._tenants: Dict[str, TenantUsage] = {}
        self._sessions: Dict[str, SessionUsage] = {}
        self._adapters: Dict[str, AdapterUsage] = {}
        self._events: List[MeteringEvent] = []

        # Tier limits
        self._tier_limits = TierLimits.defaults()

        # Billing
        billing_enabled = getattr(config, "billing_enabled", False)
        stripe_key = getattr(config, "stripe_api_key", "sk_test_stub")
        self._billing = StripeBillingStub(stripe_key) if billing_enabled else None

        logger.info(
            "MeteringService initialized (billing=%s)",
            "enabled" if self._billing else "disabled",
        )

    # ------------------------------------------------------------------
    # Tenant management
    # ------------------------------------------------------------------

    def register_tenant(
        self,
        tenant_id: str,
        tier: UsageTier = UsageTier.FREE,
        stripe_customer_id: Optional[str] = None,
    ) -> TenantUsage:
        """Register or update a tenant."""
        with self._lock:
            if tenant_id in self._tenants:
                tenant = self._tenants[tenant_id]
                tenant.tier = tier
                if stripe_customer_id:
                    tenant.stripe_customer_id = stripe_customer_id
            else:
                tenant = TenantUsage(
                    tenant_id=tenant_id,
                    tier=tier,
                    stripe_customer_id=stripe_customer_id,
                )
                self._tenants[tenant_id] = tenant
            return tenant

    def get_tenant(self, tenant_id: str) -> Optional[TenantUsage]:
        """Get tenant usage data."""
        with self._lock:
            return self._tenants.get(tenant_id)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(
        self, session_id: str, tenant_id: str, adapter_id: str
    ) -> Optional[str]:
        """Start a metering session.

        Returns:
            session_id on success, None if rate-limited (too many sessions).
        """
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                # Auto-register as free tier
                tenant = TenantUsage(tenant_id=tenant_id, tier=UsageTier.FREE)
                self._tenants[tenant_id] = tenant

            limits = self._tier_limits[tenant.tier]
            if tenant.active_sessions >= limits.max_concurrent_sessions:
                logger.warning(
                    "Rate limit: tenant %s has %d/%d concurrent sessions",
                    tenant_id, tenant.active_sessions, limits.max_concurrent_sessions,
                )
                return None

            session = SessionUsage(
                session_id=session_id,
                tenant_id=tenant_id,
                adapter_id=adapter_id,
            )
            self._sessions[session_id] = session
            tenant.active_sessions += 1

            logger.debug(
                "Session started: %s (tenant=%s, adapter=%s)",
                session_id, tenant_id, adapter_id,
            )
            return session_id

    def end_session(self, session_id: str) -> Optional[SessionUsage]:
        """End a metering session and return final usage."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is None:
                return None

            tenant = self._tenants.get(session.tenant_id)
            if tenant and tenant.active_sessions > 0:
                tenant.active_sessions -= 1

            logger.debug(
                "Session ended: %s (tokens=%d, requests=%d)",
                session_id, session.tokens_generated, session.requests,
            )
            return session

    # ------------------------------------------------------------------
    # Token metering (the hot path)
    # ------------------------------------------------------------------

    def meter_tokens(
        self,
        session_id: str,
        tokens_prompt: int = 0,
        tokens_generated: int = 0,
        adapter_price_per_1k: float = 0.0,
    ) -> tuple[bool, Optional[str]]:
        """Record token usage for a session.

        This is the hot path — called for every inference request.

        Args:
            session_id: Active session ID.
            tokens_prompt: Number of prompt tokens consumed.
            tokens_generated: Number of tokens generated.
            adapter_price_per_1k: Per-adapter price override (0 = use tier price).

        Returns:
            (allowed, reason): allowed is True if within rate limits.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False, "Unknown session"

            tenant = self._tenants.get(session.tenant_id)
            if tenant is None:
                return False, "Unknown tenant"

            limits = self._tier_limits[tenant.tier]
            total_tokens = tokens_prompt + tokens_generated
            now = time.time()

            # Reset minute window if expired
            if now - tenant.minute_window_start >= 60.0:
                tenant.tokens_this_minute = 0
                tenant.minute_window_start = now

            # Reset day window if expired (86400 seconds)
            if now - tenant.day_window_start >= 86400.0:
                tenant.tokens_today = 0
                tenant.day_window_start = now

            # Check per-minute rate limit
            if tenant.tokens_this_minute + total_tokens > limits.tokens_per_minute:
                event = MeteringEvent(
                    timestamp=now,
                    tenant_id=session.tenant_id,
                    session_id=session_id,
                    adapter_id=session.adapter_id,
                    tokens_prompt=tokens_prompt,
                    tokens_generated=tokens_generated,
                    cost_usd=0.0,
                    rate_limited=True,
                )
                self._events.append(event)
                return False, (
                    f"Per-minute rate limit exceeded: "
                    f"{tenant.tokens_this_minute + total_tokens}/{limits.tokens_per_minute}"
                )

            # Check per-day rate limit
            if tenant.tokens_today + total_tokens > limits.tokens_per_day:
                if limits.overage_price_per_1k_tokens == 0.0:
                    # Hard cap (free tier)
                    event = MeteringEvent(
                        timestamp=now,
                        tenant_id=session.tenant_id,
                        session_id=session_id,
                        adapter_id=session.adapter_id,
                        tokens_prompt=tokens_prompt,
                        tokens_generated=tokens_generated,
                        cost_usd=0.0,
                        rate_limited=True,
                    )
                    self._events.append(event)
                    return False, (
                        f"Daily token limit exceeded: "
                        f"{tenant.tokens_today + total_tokens}/{limits.tokens_per_day}"
                    )
                # Overage pricing applies
                logger.info(
                    "Tenant %s in overage (day tokens: %d/%d)",
                    session.tenant_id, tenant.tokens_today + total_tokens,
                    limits.tokens_per_day,
                )

            # Determine price
            if adapter_price_per_1k > 0:
                price_per_1k = adapter_price_per_1k
            elif tenant.tokens_today + total_tokens > limits.tokens_per_day:
                price_per_1k = limits.overage_price_per_1k_tokens
            else:
                price_per_1k = limits.price_per_1k_tokens

            cost = (total_tokens / 1000.0) * price_per_1k

            # Update session
            session.tokens_prompt += tokens_prompt
            session.tokens_generated += tokens_generated
            session.requests += 1
            session.last_activity = now

            # Update tenant
            tenant.total_tokens += total_tokens
            tenant.total_requests += 1
            tenant.tokens_today += total_tokens
            tenant.tokens_this_minute += total_tokens

            # Update adapter stats
            adapter_usage = self._adapters.get(session.adapter_id)
            if adapter_usage is None:
                adapter_usage = AdapterUsage(adapter_id=session.adapter_id)
                self._adapters[session.adapter_id] = adapter_usage
            adapter_usage.total_tokens += total_tokens
            adapter_usage.total_requests += 1
            adapter_usage.revenue_usd += cost

            # Record event
            event = MeteringEvent(
                timestamp=now,
                tenant_id=session.tenant_id,
                session_id=session_id,
                adapter_id=session.adapter_id,
                tokens_prompt=tokens_prompt,
                tokens_generated=tokens_generated,
                cost_usd=cost,
            )
            self._events.append(event)

            # Send to billing
            if self._billing and cost > 0 and tenant.stripe_customer_id:
                self._billing.record_usage(
                    customer_id=tenant.stripe_customer_id,
                    tokens=total_tokens,
                    adapter_id=session.adapter_id,
                    price_per_1k=price_per_1k,
                )

            return True, None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def usage_report(
        self,
        tenant_id: str,
        period_start: Optional[float] = None,
        period_end: Optional[float] = None,
    ) -> Optional[UsageReport]:
        """Generate a usage report for a tenant.

        Args:
            tenant_id: Tenant to report on.
            period_start: Start of reporting period (epoch). Default: 24h ago.
            period_end: End of reporting period (epoch). Default: now.

        Returns:
            UsageReport or None if tenant not found.
        """
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                return None

            now = time.time()
            start = period_start if period_start else now - 86400.0
            end = period_end if period_end else now

            # Filter events for this tenant and period
            tenant_events = [
                e for e in self._events
                if e.tenant_id == tenant_id
                and start <= e.timestamp <= end
            ]

            total_tokens = sum(e.tokens_prompt + e.tokens_generated for e in tenant_events)
            total_requests = len(tenant_events)
            total_cost = sum(e.cost_usd for e in tenant_events)

            # Group by adapter
            by_adapter: Dict[str, AdapterUsage] = {}
            for e in tenant_events:
                if e.adapter_id not in by_adapter:
                    by_adapter[e.adapter_id] = AdapterUsage(adapter_id=e.adapter_id)
                au = by_adapter[e.adapter_id]
                au.total_tokens += e.tokens_prompt + e.tokens_generated
                au.total_requests += 1
                au.revenue_usd += e.cost_usd

            return UsageReport(
                tenant_id=tenant_id,
                tier=tenant.tier,
                period_start=start,
                period_end=end,
                total_tokens=total_tokens,
                total_requests=total_requests,
                total_cost_usd=total_cost,
                by_adapter=by_adapter,
                events=tenant_events,
            )

    def adapter_dashboard(self) -> Dict[str, AdapterUsage]:
        """Return aggregate usage statistics for all adapters."""
        with self._lock:
            return {k: v for k, v in self._adapters.items()}

    def flush_billing(self) -> int:
        """Flush pending billing records to Stripe.

        Returns:
            Number of records flushed.
        """
        if self._billing is None:
            return 0
        flushed = self._billing.flush()
        return len(flushed)
