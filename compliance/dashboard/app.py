"""
TenSafe Compliance Dashboard Service.

Provides real-time compliance posture, audit evidence collection,
and automated reporting for SOC 2, HIPAA, ISO 27001, NIST AI RMF, etc.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ComplianceStandard(str, Enum):
    SOC2 = "soc2"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"
    NIST_AI_RMF = "nist_ai_rmf"
    GDPR = "gdpr"
    CCPA = "ccpa"
    PCI_DSS = "pci_dss"
    FEDRAMP = "fedramp"
    CMMC = "cmmc"
    EU_AI_ACT = "eu_ai_act"
    DORA = "dora"
    NIS2 = "nis2"


class EvidenceCategory(str, Enum):
    ACCESS_CONTROL = "access_control"
    ENCRYPTION = "encryption"
    ADAPTER_SCREENING = "adapter_screening"
    DP_BUDGET = "dp_budget"
    AUDIT_TRAIL = "audit_trail"
    INCIDENT_RESPONSE = "incident_response"
    METERING = "metering"
    SWAP_REDTEAM = "swap_redteam"


@dataclass
class ComplianceEvent:
    timestamp: str
    category: str
    standard: str
    description: str
    evidence: dict = field(default_factory=dict)
    tenant_id: str = ""
    adapter_id: str = ""
    severity: str = "info"  # info, warning, critical


@dataclass
class SwapRedTeamResult:
    """Result of red-teaming an adapter swap."""
    timestamp: str
    agent_id: str
    from_adapter: str
    to_adapter: str
    red_team_checks: list = field(default_factory=list)
    approved: bool = False
    approved_by: str = ""  # "meta_agent" or "human:<user_id>"
    rejection_reason: str = ""


@dataclass
class CompliancePosture:
    """Current compliance posture for a tenant."""
    tenant_id: str
    timestamp: str
    standards: dict = field(default_factory=dict)
    overall_status: str = "compliant"  # compliant, warning, non_compliant
    active_adapters: int = 0
    dp_budget_usage: float = 0.0
    rvu_screenings_passed: int = 0
    rvu_screenings_failed: int = 0
    swap_redteams_passed: int = 0
    swap_redteams_failed: int = 0
    open_incidents: int = 0


class ComplianceDashboard:
    """
    Real-time compliance posture and audit evidence for TenSafe deployments.

    Collects evidence from:
    - TGSP Load Gate (adapter verification)
    - RVUv2 safety screening (adapter safety)
    - DP budget tracking (privacy enforcement)
    - Metering service (usage accounting)
    - Adapter swap red-teaming (change management safety)
    - Access control logs (authentication/authorization)
    """

    def __init__(self, storage_path: str = "./compliance_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._events: list[ComplianceEvent] = []
        self._swap_redteams: list[SwapRedTeamResult] = []

    def record_event(self, event: ComplianceEvent) -> None:
        """Record a compliance-relevant event."""
        self._events.append(event)
        logger.info(
            f"Compliance event: [{event.category}] {event.description} "
            f"(tenant={event.tenant_id}, severity={event.severity})"
        )

    def record_adapter_load(
        self,
        tenant_id: str,
        adapter_id: str,
        load_gate_result: dict,
        passed: bool,
    ) -> None:
        """Record adapter load gate result (7-step verification)."""
        self.record_event(ComplianceEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            category=EvidenceCategory.ADAPTER_SCREENING,
            standard=ComplianceStandard.SOC2,
            description=f"Adapter {'loaded' if passed else 'REJECTED'}: {adapter_id}",
            evidence=load_gate_result,
            tenant_id=tenant_id,
            adapter_id=adapter_id,
            severity="info" if passed else "warning",
        ))

    def record_dp_budget(
        self,
        tenant_id: str,
        session_id: str,
        epsilon_used: float,
        epsilon_max: float,
        exhausted: bool,
    ) -> None:
        """Record differential privacy budget consumption."""
        self.record_event(ComplianceEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            category=EvidenceCategory.DP_BUDGET,
            standard=ComplianceStandard.HIPAA,
            description=f"DP budget: ε={epsilon_used:.4f}/{epsilon_max:.4f}"
                        f"{' EXHAUSTED' if exhausted else ''}",
            evidence={
                "session_id": session_id,
                "epsilon_used": epsilon_used,
                "epsilon_max": epsilon_max,
                "exhausted": exhausted,
            },
            tenant_id=tenant_id,
            severity="critical" if exhausted else "info",
        ))

    def record_swap_redteam(self, result: SwapRedTeamResult) -> None:
        """
        Record the result of red-teaming an adapter swap.

        Every adapter swap — linear or non-linear — must be red-teamed
        and confirmed by the meta-agent or a human-in-the-loop before
        the swap takes effect.
        """
        self._swap_redteams.append(result)
        severity = "info" if result.approved else "critical"
        self.record_event(ComplianceEvent(
            timestamp=result.timestamp,
            category=EvidenceCategory.SWAP_REDTEAM,
            standard=ComplianceStandard.SOC2,
            description=(
                f"Adapter swap {'APPROVED' if result.approved else 'REJECTED'}: "
                f"{result.from_adapter} → {result.to_adapter} "
                f"(approved_by={result.approved_by})"
            ),
            evidence={
                "agent_id": result.agent_id,
                "from_adapter": result.from_adapter,
                "to_adapter": result.to_adapter,
                "red_team_checks": result.red_team_checks,
                "approved": result.approved,
                "approved_by": result.approved_by,
                "rejection_reason": result.rejection_reason,
            },
            severity=severity,
        ))

    def get_posture(self, tenant_id: str) -> CompliancePosture:
        """Get current compliance posture for a tenant."""
        tenant_events = [e for e in self._events if e.tenant_id == tenant_id]
        tenant_swaps = [s for s in self._swap_redteams if True]  # filter by tenant if needed

        screening_passed = sum(
            1 for e in tenant_events
            if e.category == EvidenceCategory.ADAPTER_SCREENING
            and e.severity == "info"
        )
        screening_failed = sum(
            1 for e in tenant_events
            if e.category == EvidenceCategory.ADAPTER_SCREENING
            and e.severity != "info"
        )
        swap_approved = sum(1 for s in tenant_swaps if s.approved)
        swap_rejected = sum(1 for s in tenant_swaps if not s.approved)

        status = "compliant"
        if screening_failed > 0 or swap_rejected > 0:
            status = "warning"

        critical_events = [e for e in tenant_events if e.severity == "critical"]
        if critical_events:
            status = "non_compliant"

        return CompliancePosture(
            tenant_id=tenant_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            standards={s.value: "active" for s in ComplianceStandard},
            overall_status=status,
            rvu_screenings_passed=screening_passed,
            rvu_screenings_failed=screening_failed,
            swap_redteams_passed=swap_approved,
            swap_redteams_failed=swap_rejected,
        )

    def export_evidence(
        self,
        tenant_id: str,
        standard: ComplianceStandard | None = None,
        period_start: str | None = None,
        period_end: str | None = None,
        format: str = "json",
    ) -> str:
        """
        Export compliance evidence for auditors.

        Formats: json, csv
        """
        events = [e for e in self._events if e.tenant_id == tenant_id]

        if standard:
            events = [e for e in events if e.standard == standard]

        if period_start:
            events = [e for e in events if e.timestamp >= period_start]
        if period_end:
            events = [e for e in events if e.timestamp <= period_end]

        if format == "json":
            return json.dumps([asdict(e) for e in events], indent=2)
        elif format == "csv":
            lines = ["timestamp,category,standard,description,severity,tenant_id,adapter_id"]
            for e in events:
                lines.append(
                    f"{e.timestamp},{e.category},{e.standard},"
                    f'"{e.description}",{e.severity},{e.tenant_id},{e.adapter_id}'
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_swap_redteam_report(self, agent_id: str | None = None) -> list[dict]:
        """Get all adapter swap red-team results, optionally filtered by agent."""
        results = self._swap_redteams
        if agent_id:
            results = [r for r in results if r.agent_id == agent_id]
        return [asdict(r) for r in results]
