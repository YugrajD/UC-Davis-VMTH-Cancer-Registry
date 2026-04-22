# Future Plans — Scaling to a Multi-Clinic Registry

**Document type:** Planning  
**Status:** Proposed  
**Context:** The current system serves UC Davis VMTH exclusively. This document outlines the architectural, infrastructure, and operational changes needed to expand the registry to additional veterinary clinics across California and beyond.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Multi-Tenancy Architecture](#2-multi-tenancy-architecture)
3. [Data Format Standardization](#3-data-format-standardization)
4. [Authentication and Access Control](#4-authentication-and-access-control)
5. [NLP Model Generalization](#5-nlp-model-generalization)
6. [Infrastructure Scaling](#6-infrastructure-scaling)
7. [Geographic Expansion](#7-geographic-expansion)
8. [Clinic Onboarding Process](#8-clinic-onboarding-process)
9. [Data Privacy and Governance](#9-data-privacy-and-governance)
10. [EHR System Integrations](#10-ehr-system-integrations)
11. [Implementation Priorities](#11-implementation-priorities)

---

## 1. Overview

The current system is built around a single data source: UC Davis VMTH pathology reports processed through PetBERT and ingested via a two-file CSV upload (Dataset A + Dataset B). Every design decision — from the anon ID format (`"ID_37"`) to the column names (`Clinical Diagnoses`, `DtOfRq`) — is tailored to VMTH's specific export format.

Scaling to multiple clinics requires addressing five core problems:

| Problem | Impact |
|---|---|
| Hard-coded VMTH data format | Other clinics' exports won't parse correctly |
| No clinic-level data isolation | All records mix in the same tables |
| PetBERT fine-tuned on VMTH writing style | Accuracy drops on reports from other clinics |
| Single admin queue with no clinic context | Admins can't distinguish which clinic submitted what |
| No self-service onboarding | Each new clinic requires developer intervention |

---

## 2. Multi-Tenancy Architecture

### Current State

All patient, case, and diagnosis records share the same tables with no clinic-level partitioning. The only source tag is `patients.data_source = 'petbert'`, which was designed to distinguish real data from seed/mock data — not to identify which clinic a record came from.

### Required Changes

#### 2.1 Add a `clinics` Table

```sql
CREATE TABLE clinics (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(255) NOT NULL UNIQUE,
    slug        VARCHAR(100) NOT NULL UNIQUE,  -- e.g., 'uc-davis-vmth', 'cornell-vet'
    state       VARCHAR(2),                    -- e.g., 'CA', 'NY'
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMP NOT NULL DEFAULT NOW()
);
```

#### 2.2 Add `clinic_id` to Core Tables

```sql
ALTER TABLE patients      ADD COLUMN clinic_id INTEGER REFERENCES clinics(id);
ALTER TABLE ingestion_jobs ADD COLUMN clinic_id INTEGER REFERENCES clinics(id);
```

`clinic_id` on `patients` flows through to `cancer_cases` and `case_diagnoses` via the existing FK chain — no changes needed to those tables.

#### 2.3 Filter All Queries by `clinic_id`

Every backend query that currently filters on `data_source = 'petbert'` should be extended to also filter by `clinic_id` when a clinic context is set. The dashboard, geo, trends, and incidence endpoints all need a `clinic_id` (or `clinic_slug`) query parameter.

#### 2.4 Cross-Clinic Aggregate View (Optional)

For researchers who want to query across all clinics (e.g., statewide cancer prevalence), add a query mode that aggregates across `clinic_id` values the user has access to. This requires role-based scoping — see Section 4.

### Multi-Tenancy Pattern: Shared Database, Clinic-Scoped Rows

This approach (one database, `clinic_id` on every row) is the right choice for this registry because:
- Case counts per clinic are small (hundreds to thousands, not millions)
- Cross-clinic research queries are a core use case — separate databases would make this harder
- Supabase's RLS can enforce clinic-level row isolation at the database layer

The alternative — one Supabase project per clinic — would make cross-clinic research queries impossible without a data warehouse layer and is not recommended.

---

## 3. Data Format Standardization

### Current State

The ingestion pipeline hard-codes VMTH-specific column names: `anon_id`, `Clinical Diagnoses`, `DtOfRq`, `Sex`, `Species`, `Breed`, `Zipcode`. Any clinic with different column names will fail to ingest.

### Required Changes

#### 3.1 Field Mapping Configuration per Clinic

Introduce a per-clinic field mapping that translates a clinic's column names to the canonical internal schema:

```python
# Example: VMTH mapping
VMTH_FIELD_MAP = {
    "patient_id":       "anon_id",
    "report_text":      "Clinical Diagnoses",
    "request_date":     "DtOfRq",
    "sex":              "Sex",
    "species":          "Species",
    "breed":            "Breed",
    "zip_code":         "Zipcode",
}

# Example: Cornell mapping (hypothetical)
CORNELL_FIELD_MAP = {
    "patient_id":       "PatientID",
    "report_text":      "PathologyNarrative",
    "request_date":     "DiagnosisDate",
    "sex":              "PatientSex",
    "species":          "AnimalSpecies",
    "breed":            "AnimalBreed",
    "zip_code":         "OwnerZip",
}
```

Store these mappings in a `clinic_field_maps` table or as JSON in the `clinics` table. During ingestion, apply the clinic's mapping before the standard validation logic runs.

#### 3.2 Flexible Column Detection

For clinics that can't provide an exact field map upfront, add a column-detection step that uses fuzzy matching (e.g., Levenshtein distance) to suggest mappings during upload and ask for confirmation before processing.

#### 3.3 Canonical Internal Schema

Define a single canonical column set that all clinic data maps to before hitting the database. This is the contract between the field mapping layer and the ingestion logic:

| Canonical Field | Type | Required | Source |
|---|---|---|---|
| `patient_id` | string | Yes | Clinic-specific ID |
| `report_text` | string | Yes | Free-text pathology narrative (PetBERT input) |
| `request_date` | date | Yes | Date of report |
| `sex` | string | Yes | Mapped to: Male, Female, Neutered Male, Spayed Female |
| `species` | string | Yes | Must match `species.name` |
| `breed` | string | Yes | Must match `breeds.name` for species |
| `zip_code` | string | No | Owner ZIP → county mapping |
| `human_diagnosis` | string | No | Human-assigned label (not NLP input) |

#### 3.4 Single-File Upload Option

The current system requires two separate files (Dataset A and Dataset B). Many clinics will have a single export that contains all columns. The upload UI should accept either a single file (all columns) or the existing two-file flow.

---

## 4. Authentication and Access Control

### Current State

There is one admin role and one researcher role. Admins can approve all upload jobs. All authenticated users see all data.

### Required Changes

#### 4.1 Clinic-Scoped User Roles

Extend the user model with a `clinic_id` so that a user belongs to a specific clinic:

```sql
ALTER TABLE users ADD COLUMN clinic_id INTEGER REFERENCES clinics(id);
-- NULL = system-wide admin (can see all clinics)
-- NOT NULL = clinic-scoped user (can only see their clinic's data)
```

Roles per clinic:

| Role | Permissions |
|---|---|
| `system_admin` | Manage all clinics, approve all jobs, see all data |
| `clinic_admin` | Approve jobs for their clinic, manage their clinic's users |
| `researcher` | Upload data, view their clinic's data and cross-clinic aggregates |
| `viewer` | Read-only access to their clinic's data |

#### 4.2 RLS Policies for Clinic Isolation

Currently, RLS is enabled with no permissive policies — all access goes through FastAPI as postgres superuser. For multi-clinic use, add clinic-scoped RLS policies so that even if a direct Supabase connection is used, users cannot read another clinic's data:

```sql
-- Patients: users can only see their clinic's patients
CREATE POLICY clinic_isolation ON patients
    USING (clinic_id = current_setting('app.clinic_id')::integer);
```

FastAPI sets `app.clinic_id` as a session variable based on the authenticated user's `clinic_id` before executing queries.

#### 4.3 Cross-Clinic Read for Researchers

Researchers with `clinic_id = NULL` (system-wide) can query across all clinics. Researchers scoped to a clinic can view their own data plus any anonymized aggregate data from other clinics (no row-level access to other clinics' records).

---

## 5. NLP Model Generalization

### Current State

PetBERT is used with embeddings tuned against a label catalog built from VMTH training data. The fine-tuning scripts in `ml/scripts/` have only ever seen VMTH-style pathology reports. Accuracy on reports from other clinics will likely be lower due to:
- Different diagnostic writing styles and abbreviations per institution
- Regional vocabulary variation
- Different species mix (VMTH is almost entirely canine; other clinics may have feline, equine, etc.)

### Strategies

#### 5.1 Zero-Shot Baseline First

Before any fine-tuning, test the current PetBERT pipeline on a sample from the new clinic. Cosine similarity against the Vet-ICD-O label embeddings is relatively vocabulary-agnostic. Measure the `low_confidence` rate — if it's under 20%, the model generalizes well enough to start.

#### 5.2 Per-Clinic Fine-Tuning

As labeled data accumulates from each clinic (via the manual review workflow), trigger a fine-tuning run specific to that clinic:

```
clinic_A_labeled_data  →  fine-tune PetBERT  →  petbert-clinic-A:v1
clinic_B_labeled_data  →  fine-tune PetBERT  →  petbert-clinic-B:v1
```

Store per-clinic model versions in Artifact Registry using a naming convention:

```
us-central1-docker.pkg.dev/PROJECT/vmth/petbert-{clinic_slug}:{version}
```

The `ingestion_jobs` table already has a `clinic_id` field (once added per Section 2) — the GCP Batch job submission can use this to select the correct model image.

#### 5.3 Federated Fine-Tuning (Long-Term)

Once multiple clinics have contributed labeled data, consider training a shared multi-clinic model that benefits from the combined dataset. This improves accuracy for clinics with small label sets while preserving privacy — raw report text never leaves each clinic's upload flow.

#### 5.4 Species Expansion

VMTH is primarily canine. Other clinics may submit feline, equine, or exotic animal records. The Vet-ICD-O-canine-1 taxonomy is canine-specific. For multi-species expansion:
- Feline: Vet-ICD-O-feline taxonomy exists and could be integrated with a second label catalog
- Equine / exotic: Would require a separate label catalog and potentially separate model fine-tuning

---

## 6. Infrastructure Scaling

### Current State

The system is designed for a single clinic with ~395 patients and ~2,348 diagnoses. Supabase free tier, single GCP Batch job at a time, Vercel hobby plan.

### Scaling Considerations

#### 6.1 Database (Supabase)

| Trigger | Action |
|---|---|
| > 50,000 rows in `case_diagnoses` | Move to Supabase Pro plan (8 GB database, no row limits) |
| > 500,000 rows | Consider Supabase Enterprise or self-hosted PostgreSQL on GCP Cloud SQL |
| Cross-clinic aggregate queries become slow | Add materialized views per clinic and a `mv_cross_clinic_summary` view refreshed nightly |

The existing materialized view pattern (`mv_county_cancer_incidence`, `mv_yearly_trends`) should be extended to include `clinic_id` partitioning so per-clinic dashboards remain fast.

#### 6.2 GCP Batch (Inference)

The current setup runs one Batch job at a time. If multiple clinics submit uploads simultaneously:
- GCP Batch handles parallelism natively — multiple jobs can run concurrently on separate VMs
- No application-level queuing changes needed
- Cost scales linearly: each job is ~$1–2 per run regardless of concurrency

For a fine-tuning job (A100 GPU, several hours), GCP Batch also supports this — see `docs/GCP_BATCH_SETUP.md` and the handoff doc for details.

#### 6.3 Backend (Cloud Run)

FastAPI is deployed on GCP Cloud Run, which auto-scales to zero when idle and scales up under load. No changes needed until request volume exceeds ~1,000 concurrent users. At that point, set minimum instance count to 1 to eliminate cold starts.

#### 6.4 Frontend (Vercel)

Vercel scales automatically. The only limitation at scale is the free tier's bandwidth cap. Move to Vercel Pro when monthly bandwidth exceeds the free limit (~100 GB/month).

#### 6.5 Artifact Registry (Model Storage)

Each per-clinic model image is ~7.7 GB. At 10 clinics with 3 versions each, storage is ~230 GB — roughly $5/month in Artifact Registry storage costs. This is acceptable. Add a retention policy to automatically delete images older than the 3 most recent versions per clinic.

---

## 7. Geographic Expansion

### Current State

The dashboard is hard-coded to California counties. The `counties` table contains all 58 CA counties with PostGIS geometries. The map uses a California-specific GeoJSON source.

### Required Changes for National or Multi-State Expansion

#### 7.1 County Boundaries

The `counties` table and the PostGIS geometry data need to be extended to cover additional states. US county boundaries are available from the US Census Bureau TIGER/Line Shapefiles and can be loaded the same way California boundaries were loaded via `database/seed/county_boundaries.py`.

Add a `state` column to `counties`:
```sql
ALTER TABLE counties ADD COLUMN state VARCHAR(2);  -- e.g., 'CA', 'NY', 'TX'
```

#### 7.2 Map Component

`ChoroplethMap.tsx` currently fetches a California-specific GeoJSON URL. For multi-state support:
- Replace the hard-coded URL with a dynamic source filtered by the user's selected state(s)
- Or load a national county GeoJSON and filter client-side by `clinic.state`

#### 7.3 Dashboard Filtering

Add a `state` filter to the dashboard so researchers can view a single state's data or aggregate across states. This is a straightforward addition to the existing filter system.

---

## 8. Clinic Onboarding Process

### Current State

There is no self-service onboarding. Adding a new clinic requires a developer to manually set up the database, adjust ingestion scripts, and configure credentials.

### Target Onboarding Flow

#### Step 1: Clinic Registration (System Admin)
A system admin creates a clinic record in the `clinics` table and generates an initial `clinic_admin` account for the new clinic.

#### Step 2: Field Mapping Configuration
The clinic admin uploads a sample export file. The system detects column names and presents a mapping UI where the clinic admin maps their columns to the canonical schema. The confirmed mapping is saved to `clinic_field_maps`.

#### Step 3: Test Upload
The clinic admin uploads a small test dataset (10–20 records). The system runs the full pipeline (ingestion → PetBERT → database write) and presents results for review. If accuracy is acceptable, the clinic proceeds to production use.

#### Step 4: User Provisioning
The clinic admin creates researcher and viewer accounts for their staff via `POST /api/v1/auth/register` (already scoped to admin-only).

#### Step 5: First Full Upload
The clinic uploads their full historical dataset. The system processes it as a standard ingestion job. Low-confidence predictions go to the review queue for the clinic's own reviewers.

### Developer Work Required for Each Step

| Step | Developer work needed now | Developer work needed at scale |
|---|---|---|
| Clinic registration | Manual DB insert | Admin UI for system admins |
| Field mapping | Manual config | Self-service mapping UI |
| Test upload | Manual verification | Automated accuracy report |
| User provisioning | Already implemented | Already implemented |
| Full upload | Already implemented | Already implemented |

---

## 9. Data Privacy and Governance

### Current State

All data is de-identified at source (VMTH strips PII before export). The system stores no owner names or addresses. ZIP codes are stored for county mapping and then used as a geographic proxy — they are not PII by themselves but narrow the geographic resolution.

### Multi-Clinic Considerations

#### 9.1 Data Ownership

Each clinic owns its data. The registry aggregates and analyzes it but should not expose one clinic's raw records to another clinic's users. The RLS clinic isolation policy (Section 4.2) enforces this at the database layer.

#### 9.2 Data Use Agreements

Before onboarding a new clinic, a data use agreement (DUA) should be signed specifying:
- What data is collected and how it is stored
- Who can access the data (clinic's own users, system admins, aggregate-only for cross-clinic queries)
- Data retention policy (how long records are kept)
- Right to withdraw (clinic can request deletion of their records)

#### 9.3 De-identification Requirements

Enforce de-identification at the upload layer, not just at the source:
- Reject any upload where a `patient_id` column contains names (regex check for alphabetic strings in numeric ID fields)
- Reject any upload where free-text fields contain email addresses or phone numbers (simple regex scan before NLP processing)
- Log all rejection reasons for audit

#### 9.4 GDPR / CCPA Considerations

If any clinic is located in a jurisdiction covered by GDPR (EU) or CCPA (California residents' data), the system's data retention and deletion capabilities need to be formalized. Currently, there is no record deletion endpoint. Add:
- `DELETE /api/v1/clinic/{clinic_id}/data` — system admin only, deletes all records for a clinic
- Audit log table to record all data access and deletions

---

## 10. EHR System Integrations

### Current State

Data entry is entirely manual: a clinic exports a CSV from their EHR system and uploads it through the dashboard. This requires staff time and creates a lag between a diagnosis being recorded and it appearing in the registry.

### Future Integration Options

#### 10.1 FHIR API Integration (Long-Term)

Modern veterinary EHR systems (e.g., Cornerstone, AVImark, eVetPractice) are beginning to expose FHIR-compatible APIs. A FHIR integration would allow the registry to pull new records automatically without manual exports.

Key FHIR resources for veterinary oncology:
- `Patient` — animal demographics
- `DiagnosticReport` — pathology report text
- `Condition` — structured diagnosis

This is a significant engineering effort and requires each clinic's EHR vendor to support FHIR. Treat as a long-term goal (2–3 years out).

#### 10.2 Scheduled SFTP / S3 Drop

A simpler near-term alternative: clinics configure an automated export from their EHR to a dedicated SFTP server or GCS bucket on a nightly or weekly schedule. The registry backend polls the bucket for new files and triggers ingestion automatically.

This eliminates the manual upload step while avoiding the complexity of FHIR:
```
Clinic EHR  →  (nightly export)  →  GCS bucket  →  Cloud Scheduler  →  POST /api/v1/ingest/auto
```

#### 10.3 Webhook / Push Integration

For clinics that can configure outbound webhooks from their EHR, add a `POST /api/v1/ingest/webhook` endpoint that accepts a single record payload and ingests it in real time. This is the lowest-latency option but requires the EHR to support webhooks and the payload format to be standardized.

---

## 11. Implementation Priorities

The following table ranks the changes by value and effort, assuming a team of 2–3 developers continuing after the initial handoff:

| Priority | Change | Effort | Value |
|---|---|---|---|
| 1 | Add `clinics` table and `clinic_id` to patients + jobs | Small | Enables everything else |
| 2 | Per-clinic field mapping configuration | Medium | Unblocks non-VMTH clinics |
| 3 | Clinic-scoped user roles and RLS policies | Medium | Required for data isolation |
| 4 | Single-file upload option | Small | Most clinics have one export file |
| 5 | Admin UI for clinic registration and user management | Medium | Removes developer dependency for onboarding |
| 6 | Per-clinic PetBERT fine-tuning via GCP Batch | Medium | Improves accuracy for new clinics |
| 7 | Automated SFTP / GCS drop ingestion | Medium | Eliminates manual upload burden |
| 8 | Multi-state county boundaries and map | Medium | Enables national expansion |
| 9 | Cross-clinic aggregate dashboard view | Medium | Core research value at scale |
| 10 | FHIR API integration | Large | Long-term; EHR vendor dependent |

### Suggested Phasing

**Phase 1 — Foundation (enables 2nd clinic)**
- Items 1, 2, 3, 4 above
- Sign DUA with one partner clinic
- Run a pilot with their data using the existing upload flow + field mapping config

**Phase 2 — Self-Service (enables 5–10 clinics)**
- Items 5, 6, 7
- Build admin onboarding UI so developers are not needed for each new clinic
- Implement per-clinic model versioning in Artifact Registry

**Phase 3 — Scale (10+ clinics, national)**
- Items 8, 9
- Expand county data to cover additional states
- Build cross-clinic aggregate research views

**Phase 4 — Integration (long-term)**
- Item 10
- Evaluate FHIR adoption among target clinic EHR vendors
- Prototype with one willing clinic
