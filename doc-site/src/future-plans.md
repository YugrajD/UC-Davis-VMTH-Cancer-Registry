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

The system has three roles: **Uploader** (can submit files and audit their own diagnoses), **Reviewer** (can approve/reject jobs and review individual predictions), and **Admin** (full access including user management). Roles are stored in the `user_roles` table and enforced in `backend/app/auth.py`. Admins cannot edit their own roles or another admin's roles (self-lockout prevention). Uploaders can browse confirmed/corrected/rejected diagnoses for jobs they submitted.

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

The application is provisioned for tens-to-hundreds of active users. The May 2026 scalability pass added gunicorn, configurable connection pooling, an admin refresh-views endpoint, and a Cloud Run service template (`backend/service.yaml`). What remains for thousands of concurrent users is mostly **migrating per-process in-memory state to a shared store** and tuning the scaling envelope.

### 6.1 Where the System Bottlenecks First

Ordered by what hits the wall first as the user base grows from hundreds → thousands:

| Bottleneck | Why it bites first | Symptom |
|---|---|---|
| Per-process rate limiting | `slowapi` and the auth-failure tracker both live in memory. Each Cloud Run instance has its own counters. Effective limit = `instance_count × per_instance_limit` | Brute-force protection weakens proportional to instance count |
| Per-process response cache | `cachetools.TTLCache` is per-worker. Hit rate falls as `instance_count` rises because the same query lands on a different instance each time | DB read load grows linearly with instance count for "cached" endpoints |
| PostgreSQL connection ceiling | Supabase pooler limits and PG `max_connections`. With `DB_POOL_SIZE=5`, `MAX_OVERFLOW=10`, `WORKERS=1` and 10 Cloud Run instances → 150 connections | Connection refused errors during traffic spikes |
| Cold starts | Cloud Run scales to zero by default; cold starts are 2–5 s for this image | Sporadic slow first-request latency for low-traffic periods |
| Materialized view staleness | Views are only refreshed at ingest-time. Manual `/api/v1/admin/refresh-views` exists but nothing schedules it | Dashboards show stale aggregates if no upload that day |

### 6.2 Configuration Tunables Already in Place

These are env-driven knobs added during the May 2026 pass — they exist but should be reviewed at each scaling milestone.

| Variable | Default | When to raise |
|---|---|---|
| `WORKERS` | 1 | When the Cloud Run instance has ≥ 2 vCPUs **and** Redis is wired up for distributed cache/rate-limiting (otherwise multiple workers fragment per-process state further) |
| `DB_POOL_SIZE` | 5 | When `pool exhausted` warnings appear; raise gradually and watch PG `pg_stat_activity` |
| `DB_MAX_OVERFLOW` | 10 | Same trigger as above |
| `FORWARDED_ALLOW_IPS` | `""` | Must be set to `0.0.0.0/0` on Cloud Run, otherwise SlowAPI sees the load balancer IP for every request |
| `RATE_LIMIT_DEFAULT` | `120/minute` | Per-instance, per-IP. Raise when legitimate users are hitting it |
| `CACHE_MAX_SIZE` | 256 entries | Raise for endpoints with high cardinality of filter combinations |

The connection-budget rule that ties three of these together:

```
DB_POOL_SIZE × WORKERS × max_instances ≤ supabase_pooler_connections
```

For Supabase free tier (15-connection pooler), the safe envelope is `5 × 1 × 3 = 15`. To scale beyond 3 Cloud Run instances, either move to Supabase Pro (60 connections) or switch the URL from the direct connection to the transaction-mode pooler which can multiplex many more clients.

### 6.3 Database (Supabase)

| Trigger | Action |
|---|---|
| > 50,000 rows in `case_diagnoses` | Move to Supabase Pro plan (8 GB database, no row limits) |
| > 500,000 rows | Consider Supabase Enterprise or self-hosted PostgreSQL on GCP Cloud SQL with read replicas |
| Cross-clinic aggregate queries become slow | Add materialized views per clinic and a `mv_cross_clinic_summary` view refreshed nightly |
| Hundreds of concurrent users | Switch `DATABASE_URL` to Supabase's **transaction-mode pooler** URL (port 6543) — PgBouncer multiplexes many clients onto few PG connections, so `DB_POOL_SIZE` can be raised safely |
| Read-heavy traffic dominates | Add a read replica and route dashboard/incidence/trends/geo endpoints to it (writes still go to the primary) |

The materialized views (`mv_county_cancer_incidence`, `mv_yearly_trends`) are already in place but only refreshed inside `ingest_upload`. Schedule `POST /api/v1/admin/refresh-views` via **Cloud Scheduler** to run nightly so dashboards stay fresh even on no-upload days. Cloud Scheduler → HTTPS target → service account → bearer JWT in header.

### 6.4 GCP Batch (Inference)

The current setup runs one Batch job at a time. If multiple clinics submit uploads simultaneously:
- GCP Batch handles parallelism natively — multiple jobs can run concurrently on separate VMs
- No application-level queuing changes needed
- Cost scales linearly: each job is ~$1–2 per run regardless of concurrency

For a fine-tuning job (A100 GPU, several hours), GCP Batch also supports this — see `docs/GCP_BATCH_SETUP.md` and the handoff doc for details.

### 6.5 Backend (Cloud Run)

FastAPI runs on Cloud Run with the template at `backend/service.yaml`. The current defaults target hundreds of users:

| Setting | Current | Scaling to thousands |
|---|---|---|
| `cpu` | 1 vCPU | 2 vCPU once `WORKERS=2` and Redis is wired |
| `memory` | 512 Mi | 1 Gi if memory pressure shows in Cloud Logging |
| `containerConcurrency` | 80 | 100–200 once async paths are confirmed I/O-bound (each await yields the loop) |
| `maxScale` | 10 | 50+ as traffic grows; watch the DB connection budget rule above |
| `minScale` | 0 | **1** when low-latency first-request is required (kills cold starts at ~$3/month) |
| Cold start mitigation | None | Set `minScale: 1` OR enable **CPU always allocated** for the service |

`WORKERS` should remain at `1` until **all three** of these are true:
1. The Cloud Run instance has ≥ 2 vCPUs allocated
2. Redis (or another shared store) is hosting rate-limit and cache state
3. A load test shows the single worker is CPU-bound (rare for async FastAPI)

Without #2, adding workers makes the per-process state fragmentation worse, not better.

### 6.6 Redis Migration (Required for True Horizontal Scaling)

This is the single biggest improvement available. Today, two pieces of state are per-process: rate-limiting (`slowapi` + `_failed_attempts` dict in `app/auth.py`) and response caching (`cachetools.TTLCache` in `app/cache.py`).

**Migration plan:**

1. **Provision Redis** — GCP Memorystore (Basic Tier, 1 GB ≈ $35/month) or Upstash (serverless, pay-per-request, often cheaper at this scale). Connect over the Cloud Run VPC connector.

2. **Migrate rate limiting** — `slowapi` already supports Redis natively:
   ```python
   from slowapi.util import get_remote_address
   limiter = Limiter(
       key_func=get_client_ip,
       storage_uri=settings.REDIS_URL,   # was: in-memory
       strategy="fixed-window",
   )
   ```
   Also migrate `_failed_attempts` in `app/auth.py` to a Redis sorted-set keyed by IP, with a TTL on each entry.

3. **Migrate response cache** — Replace `cachetools.TTLCache` with `redis-py` calls keyed by the same hash we compute today in `_make_cache_key`. Keep the decorator API the same so call sites don't change. Use `SETEX` for TTL and `DEL` patterns for `clear_all_caches`.

4. **Watch the cardinality** — The dashboard's filter combinations are the highest-cardinality cache namespace. Set per-namespace `maxsize` Redis-side using a `LRU` eviction policy at the instance level.

After this migration, raising `WORKERS` and `maxScale` becomes safe — every instance shares state.

### 6.7 Frontend (Vercel)

Vercel scales automatically. Limitations to watch:
- Free tier bandwidth cap (~100 GB/month). Move to Pro when this is hit; Pro is $20/month and lifts the cap to 1 TB.
- Function execution time (10 s on Hobby, 60 s on Pro) — the frontend has no server-side functions today, so this is not a current concern.

For the production deploy, ensure `VITE_API_URL` points to the Cloud Run service URL and that the Cloud Run service has `CORS_ORIGINS` set to the Vercel production URL.

### 6.8 Artifact Registry (Model Storage)

Each per-clinic model image is ~7.7 GB. At 10 clinics with 3 versions each, storage is ~230 GB — roughly $5/month in Artifact Registry storage costs. This is acceptable. Add a retention policy to automatically delete images older than the 3 most recent versions per clinic.

### 6.9 Observability and Load Testing

Before declaring "ready for thousands of users," set up:

| Tool | Purpose | Cost |
|---|---|---|
| Cloud Logging (structured) | Request logs, error traces | Free up to 50 GiB/month |
| Cloud Monitoring dashboards | p50/p95/p99 latency, error rate, instance count, DB pool usage | Free up to a generous quota |
| Cloud Monitoring alert policies | Alert when p99 latency > 2 s or error rate > 1% for 5 min | Free |
| k6 or Artillery load test | Confirm capacity before traffic spikes (e.g. before a press release or paper publication) | Self-hosted, free |

A good load-test target before claiming "thousands of users": 500 RPS sustained for 10 min against the cached dashboard endpoints with p95 < 500 ms.

### 6.10 Scale Checkpoints

Concrete actions tied to user growth milestones:

| User base | Action |
|---|---|
| Up to 200 active | No changes from current defaults. Monitor metrics. |
| 200 – 1,000 active | Set `minScale: 1` to kill cold starts. Switch Supabase URL to the transaction-mode pooler. |
| 1,000 – 5,000 active | Migrate rate limiting and cache to Redis (§6.6). Raise `maxScale` to 50. Schedule nightly materialized view refresh. |
| 5,000+ active | Add a Supabase read replica and route dashboard reads there. Move to Supabase Pro or self-host on Cloud SQL. Provision Memorystore Standard Tier (HA) for Redis. |

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
| 0 | **Redis migration for rate limiting + response cache** (§6.6) | Medium | Required to safely scale beyond 3 Cloud Run instances |
| 0 | **Cloud Scheduler nightly refresh-views** (§6.3) | Small | Keeps dashboards fresh without manual intervention |
| 0 | **Observability dashboards and alerts** (§6.9) | Small–Medium | Required before declaring "production-ready at scale" |
| 1 | Add `clinics` table and `clinic_id` to patients + jobs | Small | Enables everything else multi-tenant |
| 2 | Per-clinic field mapping configuration | Medium | Unblocks non-VMTH clinics |
| 3 | Clinic-scoped user roles and RLS policies | Medium | Required for data isolation |
| 4 | Single-file upload option | Small | Most clinics have one export file |
| 5 | Admin UI for clinic registration and user management | Medium | Removes developer dependency for onboarding |
| 6 | Per-clinic PetBERT fine-tuning via GCP Batch | Medium | Improves accuracy for new clinics |
| 7 | Automated SFTP / GCS drop ingestion | Medium | Eliminates manual upload burden |
| 8 | Multi-state county boundaries and map | Medium | Enables national expansion |
| 9 | Cross-clinic aggregate dashboard view | Medium | Core research value at scale |
| 10 | FHIR API integration | Large | Long-term; EHR vendor dependent |

Priority 0 items are infrastructure prerequisites that should be tackled before — or in parallel with — the multi-tenant work in priorities 1–3.

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

---

## 12. Deferred Product Features

These were planned and scoped in May 2026 but consciously deferred. Each has an existing backend surface (so re-activating it is mostly frontend work) and a documented reason for the deferral.

### 12.1 Plain-Language Search UI

**User stories:** US-15, US-NTH-4 from `requirements_doc_v2.md` — *"query with little to no knowledge of the coding system or SQL"*.

**Status:** Deferred.

**Why:** The existing dashboard filters (cancer type, breed, sex, county, year range) already satisfy the spirit of these user stories by providing structured, taxonomy-aligned navigation that does not require knowledge of ICD codes or SQL. Building a free-text search UI on top of filters would create two ways to do the same thing without enough additional research value to justify the maintenance cost.

**What's already in place:**
- `POST /api/v1/search/classify` — takes free text and returns suggested Vet-ICD-O codes via PetBERT. Auth-required, rate-limited (`RATE_LIMIT_EXPENSIVE`).
- `GET /api/v1/search/reports?keyword=...` — searches `case_diagnoses.original_text` for a keyword. Auth-required, LIKE-escaped to prevent injection.

**The narrow use case search would unlock:** finding cases by clinical features that exist in raw report text but aren't captured in any structured field (e.g., "lymph node metastasis," "myxoid," specific histological terms). This is a power-user feature for researchers studying histological detail rather than population-level patterns.

**Trigger to revisit:** When a researcher specifically asks for it. Until then, the backend endpoints stay in place (they cost nothing sitting idle) but the UI is not built.

**Estimated effort if reactivated:** 1–2 days. All frontend — `SearchBar.tsx`, `SearchResults.tsx`, API client helpers, and a new tab or inline result view.

### 12.2 Automated PetBERT Fine-Tuning Trigger

**User story:** US-PL-2 — *"model quality to improve as more labeled data becomes available."*

**Status:** Deferred. The acceptance test (fine-tuning produces equal-or-better held-out scores than the previous version) cannot be satisfied without significant infrastructure work.

**Why:**
- A single A100 fine-tuning run is **$10–25 in GCP compute** (A100 at ~$2–3/hour × 4–8 hours). At the current labeled-data accumulation rate, training runs would be infrequent but each one is non-trivial.
- The full plan requires three phases (training pipeline → validation comparison + model versioning → held-out set governance), each ~1 week of work. Total ~3 weeks.
- The team chose to redirect engineering time toward features with broader user impact (Plans 1 and 2 in this same review). Fine-tuning improves accuracy at the margins; the existing zero-shot PetBERT pipeline is already producing usable predictions.

**What's already in place:**
- Local training pipeline: `ml/scripts/run_training_cycle.py` works end-to-end on a development machine.
- Labeled data accumulates naturally — `case_diagnoses.review_status IN ('confirmed', 'corrected')` rows are training-ready pairs of `(original_text, predicted_term)`.
- The validation CSV `ml/data/validation/review_threshold_validation.csv` is a starting held-out set.

**Three-phase implementation when reactivated:**

| Phase | Scope | Effort |
|---|---|---|
| 4A: One-shot batch training | New `ml/production/finetune/` package; new `Dockerfile.finetune` and Artifact Registry image; new `training_jobs` table; `POST /api/v1/training/submit` admin endpoint; GCP Batch A100 job submission service mirroring `gcp_batch_service.py` | ~1 week |
| 4B: Validation comparison + model versioning | Training writes precision/recall/F1 JSON to GCS; new model tagged `petbert:v{N}` in Artifact Registry; `POST /api/v1/training/jobs/{id}/promote` flips `petbert:current` tag; `POST .../rollback` reverts; frontend admin page with side-by-side metrics | ~1 week |
| 4C: Held-out validation set governance | New `case_diagnoses.held_out: bool` column; stratified-by-cancer-type validation set selection script; training pipeline excludes held-out rows; validation comparison uses only held-out rows | ~3–5 days |

**Open decisions deferred to reactivation:**
- Trigger model: manual button only, or auto-trigger when N new reviewed labels accumulate? (Recommend manual to start.)
- Promotion model: manual click required, or auto-promote when validation F1 improves? (Recommend manual — research registry should not silently swap models.)
- GCP A100 quota: must be requested from GCP support before the first run; new GCP projects start with zero A100 quota.

**Trigger to revisit:**
- When ≥ 500 reviewed-and-confirmed labels have accumulated in `case_diagnoses` (current count under 100), AND
- When the team has bandwidth for a multi-week ML-infrastructure project.

**Until then:** zero-shot PetBERT continues to handle inference; the local `run_training_cycle.py` is available for ad-hoc experiments on a dev machine without involving GCP.
