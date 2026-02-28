# NIST AI Risk Management Framework (AI RMF) Alignment Report

## TenSafe Privacy-Preserving AI Inference Platform

**Framework Version:** NIST AI 100-1 (January 2023)
**Assessment Date:** 2026-02-28
**Platform Version:** TenSafe Runtime v1.0, TGSP Spec v1.1

---

## Executive Summary

TenSafe aligns with the NIST AI RMF across all four core functions: GOVERN, MAP,
MEASURE, and MANAGE. The platform's homomorphic encryption architecture provides
unique architectural guarantees for AI trustworthiness — particularly in privacy,
security, and safety — that most AI platforms cannot achieve.

---

## GOVERN: Establish AI Risk Management Culture

### GOV-1: Policies and Procedures

| Sub-category | TenSafe Implementation |
|-------------|----------------------|
| GOV-1.1 Legal compliance | HIPAA, SOC 2, ISO 27001, GDPR compliance packages |
| GOV-1.2 Accountability | Per-tenant audit trails, creator identity on every adapter |
| GOV-1.3 Risk tolerance | Configurable DP budget (epsilon), RVUv2 screening thresholds |
| GOV-1.4 Org policies | TGSP validation policy (mandatory for marketplace), compliance mode flags |

### GOV-2: Accountability Structures

| Sub-category | TenSafe Implementation |
|-------------|----------------------|
| GOV-2.1 Roles | RBAC: Admin, Agent Manager, Developer, Auditor |
| GOV-2.2 Responsibilities | Compliance dashboard assigns monitoring per role |
| GOV-2.3 Human oversight | Meta-agent quality thresholds + human review for marketplace |

### GOV-3: Workforce Diversity and Competence

| Sub-category | TenSafe Implementation |
|-------------|----------------------|
| GOV-3.1 AI expertise | HE/PQC cryptography team, ML safety researchers |
| GOV-3.2 Interdisciplinary | Creator program spans healthcare, finance, legal, government |

---

## MAP: Context and Risk Identification

### MAP-1: AI System Context

| Sub-category | TenSafe Implementation |
|-------------|----------------------|
| MAP-1.1 Intended purpose | Privacy-preserving AI inference; domain-specific expertise via TGSP adapters |
| MAP-1.2 Intended users | Enterprise (healthcare, finance, government, legal), prosumers, SMBs |
| MAP-1.3 Assumptions | Users encrypt data before transmission; adapters are TGSP-validated |
| MAP-1.4 Technology | CKKS HE, sparse MoE models, LoRA fine-tuning, post-quantum crypto |

### MAP-2: Risk Categorization

| Risk Category | Risk Level | Rationale |
|--------------|-----------|-----------|
| Privacy | **Very Low** | HE ensures server never sees plaintext; DP bounds leakage |
| Security | **Low** | PQC signatures, SHA-256 hash, TGSP Load Gate 7-step verification |
| Safety | **Low** | RVUv2 3-layer screening (allowlist + SVD + Mahalanobis OOD) |
| Fairness/Bias | **Medium** | Adapter quality depends on training data; qa_verify checks quality |
| Transparency | **Low** | TGSP manifest declares all properties; embedded SKILL.md |
| Accountability | **Very Low** | Cryptographic creator identity; immutable audit trail |

### MAP-3: Benefits and Costs

| Benefit | Evidence |
|---------|---------|
| Privacy preservation | CKKS HE: mathematically proven privacy |
| Regulatory compliance | HIPAA, SOC 2, GDPR architectural compliance |
| Quantum resistance | ML-KEM-768 + ML-DSA-65 (NIST PQC finalized) |
| Adapter trust | TGSP: creator identity + RVUv2 + hash + dual signatures |

---

## MEASURE: Analyze, Assess, and Track AI Risks

### MEASURE-1: AI Risk Metrics

| Metric | Measurement Method | Target |
|--------|-------------------|--------|
| **Privacy leakage** | Differential privacy epsilon per session | ε < configured max |
| **Adapter safety** | RVUv2 screening pass rate | 100% of loaded adapters |
| **Adapter quality** | qa_verify score | ≥ 0.80 for all validated adapters |
| **Inference integrity** | SHA-256 hash verification | 100% pass rate |
| **Creator authenticity** | Dual-signature verification rate | 100% pass rate |
| **OOD detection** | Mahalanobis distance flagging rate | Track and review all flags |

### MEASURE-2: Trustworthiness Characteristics

| Characteristic | TenSafe Implementation | Measurement |
|---------------|----------------------|-------------|
| **Valid & Reliable** | qa_verify quality benchmarks; A/B testing for self-improved adapters | Score ≥ 0.80 |
| **Safe** | RVUv2 3-layer screening at creation + every load | Pass all 3 layers |
| **Secure & Resilient** | CKKS HE + PQC + TGSP Load Gate 7-step | 0 plaintext exposures |
| **Accountable & Transparent** | Creator identity, SKILL.md in every adapter, audit trail | 100% traceable |
| **Explainable & Interpretable** | SKILL.md describes what each adapter does, domain, capabilities | Human-readable |
| **Privacy-Enhanced** | CKKS HE + DP budget + session isolation | ε tracking |
| **Fair** | Quality gates + domain-appropriate validation | qa_verify per vertical |

### MEASURE-3: Risk Tracking

- Compliance Dashboard provides real-time risk metrics
- Automated alerting for: RVUv2 failures, DP budget exhaustion, signature failures, hash mismatches
- Quarterly risk assessment review (GOVERN cycle)
- Annual penetration testing of TGSP Load Gate and RVUv2 screening

---

## MANAGE: Prioritize, Respond, and Communicate AI Risks

### MANAGE-1: Risk Response

| Risk | Response | Owner |
|------|----------|-------|
| Adapter poisoning | RVUv2 screening rejects; adapter revoked from marketplace | Security team |
| DP budget exhaustion | Blocking response (inference stops); alert to tenant admin | Runtime engine |
| Signature verification failure | Adapter rejected; incident logged; creator notified | Load gate |
| Hash mismatch | Adapter rejected; integrity alert; investigation triggered | Load gate |
| Quality degradation | A/B test gates promotion; rollback to previous version | Self-improvement engine |

### MANAGE-2: Risk Prioritization

1. **Critical:** Any plaintext data exposure (mitigated: HE makes this impossible)
2. **High:** Adapter poisoning/adversarial outputs (mitigated: RVUv2 3-layer)
3. **Medium:** Bias/fairness in adapter outputs (mitigated: qa_verify + human review)
4. **Low:** Performance degradation (mitigated: monitoring + auto-scaling)

### MANAGE-3: Communication

| Stakeholder | Communication | Frequency |
|------------|---------------|-----------|
| Enterprise customers | Compliance dashboard + quarterly report | Real-time + quarterly |
| Creators | Validation results + quality feedback | Per-submission |
| Regulators | SOC 2 report + NIST AI RMF alignment | Annual |
| Public | TGSP spec + security architecture docs | Continuous (open source) |

---

## Appendix: NIST AI RMF Profile for TenSafe

This profile identifies the most relevant subcategories for TenSafe's deployment context:

**High Priority (Implemented):**
- GOV-1.1, GOV-2.1, MAP-1.1, MAP-2, MEASURE-1, MEASURE-2, MANAGE-1, MANAGE-2

**Medium Priority (In Progress):**
- GOV-3.1, MAP-3, MEASURE-3, MANAGE-3

**Lower Priority (Planned):**
- GOV-1.4, GOV-3.2 (expanded as ecosystem grows)
