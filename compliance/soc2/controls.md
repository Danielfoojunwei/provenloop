# SOC 2 Type II Control Mapping — TenSafe Platform

## Overview

This document maps TenSafe's security controls to the AICPA Trust Services Criteria
(TSC) for SOC 2 Type II certification. TenSafe's homomorphic encryption (HE)
architecture provides architectural guarantees that satisfy many controls by design.

## CC1: Control Environment

| Criteria | TenSafe Control | Evidence |
|----------|----------------|----------|
| CC1.1 CISO commitment | Security-first architecture; HE ensures data never leaves ciphertext | Architecture docs, CKKS parameter selection |
| CC1.2 Board oversight | Quarterly security review, compliance dashboard | Dashboard export, meeting minutes |
| CC1.3 Org structure | RBAC: Admin, Agent Manager, Developer, Auditor roles | Role definitions, access control logs |
| CC1.4 Competence | Engineering team with HE/PQC expertise | Team credentials, code review process |
| CC1.5 Accountability | Per-tenant audit trail, DP budget tracking per session | Audit log exports, compliance dashboard |

## CC2: Communication and Information

| Criteria | TenSafe Control | Evidence |
|----------|----------------|----------|
| CC2.1 Internal communication | TGSP manifest documents all adapter properties | Manifest schema, SKILL.md embedded in each adapter |
| CC2.2 External communication | Public TGSP spec (Apache 2.0), API documentation | tgsp-spec repo, API reference docs |
| CC2.3 Relevant information | Compliance dashboard with real-time posture | Dashboard screenshots, SIEM integration |

## CC3: Risk Assessment

| Criteria | TenSafe Control | Evidence |
|----------|----------------|----------|
| CC3.1 Risk objectives | Protect inference data confidentiality via CKKS HE | Threat model document |
| CC3.2 Risk identification | RVUv2 3-layer safety screening on every adapter load | RVUv2 screening logs, rejection reports |
| CC3.3 Fraud risk | Creator identity verification (Ed25519 + Dilithium3 dual-sig) | Signature verification logs |
| CC3.4 Change risk | TGSP versioning + re-validation required per version | Validation pipeline logs |

## CC4: Monitoring Activities

| Criteria | TenSafe Control | Evidence |
|----------|----------------|----------|
| CC4.1 Ongoing monitoring | Per-token metering, DP budget tracking, inference latency monitoring | Metering service logs, Prometheus metrics |
| CC4.2 Deficiency evaluation | RVUv2 screening rejects malicious adapters; automated alerting | Rejection logs, alert configuration |

## CC5: Control Activities

| Criteria | TenSafe Control | Evidence |
|----------|----------------|----------|
| CC5.1 Risk mitigation | CKKS HE: server never sees plaintext user data | HE parameter configuration, encryption proofs |
| CC5.2 Technology controls | TGSP Load Gate: 7-step verification before adapter executes | Load gate audit trail |
| CC5.3 Policy deployment | Compliance mode flags in Helm chart deployment | Helm values.yaml, deployment configuration |

## CC6: Logical and Physical Access Controls

| Criteria | TenSafe Control | Evidence |
|----------|----------------|----------|
| CC6.1 Access control | Per-tenant CKKS keys, RBAC, SSO (SAML/OIDC) | Access control configuration, SSO integration logs |
| CC6.2 Authentication | API key + tenant isolation; optional MFA via SSO provider | Authentication logs |
| CC6.3 Access management | Tenant-scoped adapter registries, role-based permissions | RBAC configuration, tenant isolation audit |
| CC6.4 Physical security | Cloud provider responsibility (AWS/Azure/GCP); air-gapped option | Cloud compliance certs, air-gap deployment docs |
| CC6.5 Logical access | Network segmentation; GPU contexts isolated per tenant | Network architecture diagram |
| CC6.6 Access review | Quarterly access review + automated stale IP pruning | Access review logs, rate limiter config |
| CC6.7 System changes | TGSP re-validation required for any adapter update | Validation pipeline logs |
| CC6.8 Data handling | All inference data encrypted via CKKS HE; DP budget limits data exposure | HE parameters, DP budget logs |

## CC7: System Operations

| Criteria | TenSafe Control | Evidence |
|----------|----------------|----------|
| CC7.1 Infrastructure monitoring | Prometheus metrics, health checks, Docker healthcheck with start_period | Monitoring configuration, docker-compose.yml |
| CC7.2 Anomaly detection | RVUv2 Mahalanobis OOD detector for adapter inputs; rate limiting | OOD detection logs, rate limiter logs |
| CC7.3 Incident response | Automated flagging of RVUv2 failures; DP budget exhaustion blocking | Incident response playbook, alert configuration |
| CC7.4 Recovery | Docker container restart policies; Kubernetes self-healing | Deployment configuration |

## CC8: Change Management

| Criteria | TenSafe Control | Evidence |
|----------|----------------|----------|
| CC8.1 Change management | TGSP versioning (v1.1 manifest); re-validation per version bump | TGSP spec version history, validation logs |

## CC9: Risk Mitigation

| Criteria | TenSafe Control | Evidence |
|----------|----------------|----------|
| CC9.1 Risk mitigation | HE eliminates data breach risk; PQC (ML-KEM-768 + ML-DSA-65) future-proofs | Cryptographic parameter documentation |
| CC9.2 Vendor management | Pinned dependencies with upper bounds; TGSP signature verification | requirements.txt, TGSP verification logs |

## Availability (A1)

| Criteria | TenSafe Control | Evidence |
|----------|----------------|----------|
| A1.1 Processing capacity | Auto-scaling via Kubernetes; GPU-aware scheduling | Helm chart, HPA configuration |
| A1.2 Recovery objectives | Docker healthcheck + restart; Kubernetes pod recreation | Deployment configuration, healthcheck config |
| A1.3 Recovery testing | Automated selftest endpoint (`/api/v1/split/selftest`) | Selftest results, CI pipeline |

## Confidentiality (C1)

| Criteria | TenSafe Control | Evidence |
|----------|----------------|----------|
| C1.1 Confidential info | **ARCHITECTURAL GUARANTEE:** CKKS HE ensures server NEVER sees plaintext data | HE parameter proof, CKKS security analysis |
| C1.2 Disposal | DP budget tracking; right-to-erasure via budget reset | DP budget management documentation |

## Privacy (P1-P8)

| Criteria | TenSafe Control | Evidence |
|----------|----------------|----------|
| P1.1 Privacy notice | Transparency: users informed of HE processing, DP budget | Privacy policy, user documentation |
| P4.1 Data collection | Minimal collection: only encrypted activations processed | Data flow diagram |
| P5.1 Data use | Inference only; DP ensures bounded information leakage | DP budget configuration, epsilon tracking |
| P6.1 Data retention | Session-scoped; DP budget resets per session | Session management documentation |
| P8.1 Data quality | TGSP quality gates (qa_verify ≥ 0.80) ensure adapter quality | Validation pipeline logs |

## TenSafe-Specific Architectural Advantages

### Why HE Makes SOC 2 Easier

Most AI platforms struggle with CC6.8 (data handling) and C1.1 (confidentiality) because
the inference server processes plaintext user data. **TenSafe eliminates this concern entirely:**

- The server performs inference on **encrypted data** (CKKS ciphertexts)
- Even a complete server compromise reveals **zero plaintext**
- Data breach notification requirements are moot — there's no plaintext to breach

### Why TGSP Makes CC8 (Change Management) Easier

Every adapter change is:
1. Version-bumped in the TGSP manifest
2. Re-validated (RVUv2 + quality + security)
3. Re-signed by the creator
4. Hash-verified before loading

This creates an immutable audit trail of every adapter version ever deployed.

## Evidence Collection Automation

The TenSafe Compliance Dashboard automatically collects evidence for auditors:

```
GET /api/v1/compliance/evidence?standard=soc2&period=2026-Q1
```

Returns:
- Access control logs (CC6)
- Adapter load/reject logs (CC5, CC7)
- DP budget consumption reports (C1, P5)
- RVUv2 screening audit trail (CC3, CC7)
- Metering and usage reports (CC4, A1)

Export formats: PDF, CSV, JSON, SIEM (Splunk/Datadog/Sentinel)
