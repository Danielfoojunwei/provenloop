# Business Associate Agreement (BAA) Template — TenSafe Platform

## HIPAA Business Associate Agreement

**Between:**
- **Covered Entity:** [Customer Name] ("Covered Entity")
- **Business Associate:** TenSafe Inc. ("Business Associate")

**Effective Date:** [Date]

---

## 1. Definitions

**1.1** "Protected Health Information" (PHI) means individually identifiable health
information as defined under 45 CFR § 160.103.

**1.2** "Electronic Protected Health Information" (ePHI) means PHI transmitted or
maintained in electronic media as defined under 45 CFR § 160.103.

**1.3** "TenSafe Platform" means Business Associate's homomorphic encryption-based
AI inference platform, including the TGSP adapter ecosystem.

**1.4** "CKKS Homomorphic Encryption" means the Cheon-Kim-Kim-Song encryption scheme
used by the TenSafe Platform to process encrypted data without decryption.

---

## 2. Obligations of Business Associate

### 2.1 Use and Disclosure Restrictions

Business Associate shall:
- Not use or disclose PHI other than as permitted by this Agreement or as required by law
- Use CKKS homomorphic encryption for all PHI processing, ensuring the TenSafe
  inference server **never accesses plaintext PHI**
- Process PHI only in encrypted form (CKKS ciphertexts) for AI inference

### 2.2 Safeguards

Business Associate shall implement:

**Administrative Safeguards:**
- Role-based access control (RBAC) with Admin, Agent Manager, Developer, Auditor roles
- Per-tenant isolation for multi-tenant deployments
- Workforce training on PHI handling procedures

**Technical Safeguards:**
- CKKS Homomorphic Encryption: all inference performed on encrypted data
- Post-quantum cryptography: ML-KEM-768 for key exchange, ML-DSA-65 for signatures
- Differential Privacy (DP) budget tracking with hard limits per session
- RVUv2 safety screening on all adapters before PHI processing
- TLS 1.3 for data in transit
- SHA-256 hash verification on all TGSP adapters

**Physical Safeguards:**
- Cloud infrastructure compliant with HIPAA (AWS GovCloud, Azure Government, or
  customer-hosted air-gapped deployment)
- GPU-level tenant isolation

### 2.3 Reporting

Business Associate shall report to Covered Entity:
- Any use or disclosure of PHI not provided for by this Agreement within 30 days
- Any Security Incident (as defined in 45 CFR § 164.304) within 60 days
- **Note:** Due to homomorphic encryption, traditional data breaches are
  architecturally infeasible — the server never holds plaintext PHI

### 2.4 Subcontractors

Business Associate shall ensure that any subcontractor that creates, receives,
maintains, or transmits PHI agrees to the same restrictions and conditions.

### 2.5 Access to PHI

Business Associate shall:
- Make PHI available to Covered Entity as required under 45 CFR § 164.524
- Provide DP budget consumption reports and audit logs via the Compliance Dashboard
- Support right-to-erasure requests via DP budget reset

### 2.6 Amendments to PHI

Business Associate shall make amendments to PHI as directed by Covered Entity
in accordance with 45 CFR § 164.526.

### 2.7 Accounting of Disclosures

Business Associate shall maintain an accounting of disclosures of PHI as required
under 45 CFR § 164.528, including:
- Adapter access logs (which adapters processed which tenant's data)
- DP budget consumption reports (bounded information leakage per session)
- Inference metering logs (volume of PHI processed)

### 2.8 HHS Access

Business Associate shall make its internal practices, books, and records relating
to the use and disclosure of PHI available to the Secretary of HHS for purposes of
determining compliance.

---

## 3. Obligations of Covered Entity

Covered Entity shall:
- Notify Business Associate of any restrictions on use/disclosure of PHI
- Notify Business Associate of any changes to permissions or authorizations
- Not request Business Associate to use or disclose PHI in a manner inconsistent
  with HIPAA

---

## 4. TenSafe Architectural Guarantee

### 4.1 Homomorphic Encryption Processing

The TenSafe Platform processes all PHI exclusively through CKKS homomorphic encryption.
This means:

1. **PHI is encrypted before transmission** to the TenSafe inference server
2. **The server performs AI inference on encrypted data** (CKKS ciphertexts)
3. **Results are returned encrypted** to the Covered Entity for decryption
4. **The server NEVER accesses, stores, or processes plaintext PHI**

This architectural guarantee means that even in the event of a complete server
compromise, no PHI is exposed.

### 4.2 Differential Privacy Budget

Each inference session operates under a differential privacy budget (epsilon).
Once the budget is exhausted, no further inference is permitted for that session.
This provides a mathematical bound on information leakage.

### 4.3 Adapter Safety

All TGSP adapters used for PHI processing must:
- Pass RVUv2 3-layer safety screening (allowlist + SVD + Mahalanobis OOD)
- Be TenSafe Validated (security, quality, and compliance verified)
- Carry verified creator identity (Ed25519 + Dilithium3 dual signatures)
- Include SHA-256 payload hash for tamper detection

---

## 5. Term and Termination

### 5.1 Term
This Agreement is effective as of the Effective Date and continues until terminated.

### 5.2 Termination for Cause
Either party may terminate this Agreement if the other party materially breaches
any provision and fails to cure within 30 days of written notice.

### 5.3 Effect of Termination
Upon termination, Business Associate shall:
- Return or destroy all PHI received from Covered Entity
- Reset all DP budgets associated with Covered Entity's tenant
- Provide confirmation of destruction within 30 days

---

## 6. Miscellaneous

### 6.1 Amendments
This Agreement may be amended only by written agreement of both parties.

### 6.2 Survival
The obligations of Business Associate under Sections 2.5, 2.6, 2.7, and 5.3
shall survive termination.

### 6.3 Interpretation
This Agreement shall be interpreted consistently with the HIPAA Rules.

---

**IN WITNESS WHEREOF:**

Covered Entity: _____________________________ Date: ___________

Business Associate: _________________________ Date: ___________

---

*This BAA template is provided for reference. Legal counsel should review and
customize this agreement for each specific engagement.*
