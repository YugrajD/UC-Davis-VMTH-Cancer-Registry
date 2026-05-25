# UC Davis VMTH Cancer Registry — Project Handoff

**Project:** UC Davis VMTH Canine Cancer Registry  
**Team:** ECS 193A Team 14  
**Authors:** Yugraj Dhillon, David Estrella, Chun Ho Li, Justin Pak  
**Handoff Date:** April 15, 2026  
**Last Updated:** May 18, 2026  
**Repository:** https://github.com/ECS-193A-Team-14/UC-Davis-VMTH-Cancer-Registry  

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Tech Stack](#2-tech-stack)
3. [Repository Structure](#3-repository-structure)
4. [What Is Implemented](#4-what-is-implemented)
5. [What Is Not Implemented](#5-what-is-not-implemented)
6. [Running the Project Locally](#6-running-the-project-locally)
7. [GCP Setup](#7-gcp-setup)
8. [Database Schema](#8-database-schema)
9. [Data Pipeline](#9-data-pipeline)
10. [Key Architectural Decisions](#10-key-architectural-decisions)
11. [Known Issues and Gotchas](#11-known-issues-and-gotchas)
12. [Remaining Work](#12-remaining-work)
13. [Credentials and Secrets](#13-credentials-and-secrets)

---

## 1. Project Summary

The VMTH Cancer Registry is a web application for UC Davis veterinary researchers to upload, process, and visualize canine cancer case data across California. The system accepts de-identified pathology reports from the VMTH, runs them through the PetBERT NLP model to extract structured cancer labels aligned to the Vet-ICD-O-canine-1 taxonomy, and displays the results on an interactive California county choropleth map.

**Primary users:** Veterinary researchers and VMTH staff  
**Data:** De-identified canine pathology reports from UC Davis VMTH  
**Current dataset:** 395 patients, ~2,348 case diagnoses (PetBERT predictions)  

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, Vite, Tailwind CSS v4, react-simple-maps, d3-scale |
| Backend | FastAPI, Python 3.11, SQLAlchemy async, Pydantic v2 |
| ASGI server | gunicorn + uvicorn workers (production), uvicorn (dev) |
| Database | PostgreSQL 16 + PostGIS 3.4, hosted on Supabase |
| NLP Model | PetBERT (110M param BERT pretrained on veterinary EHR data) |
| ML Inference | GCP Batch (production), local ml-worker container (development) |
| Model Storage | GCP Artifact Registry (Docker image, ~7.7 GB compressed) |
| Raw File Storage | GCP Cloud Storage |
| Frontend Hosting | Vercel |
| Backend Hosting | GCP Cloud Run |
| Auth | Supabase Auth (JWT) |
| CI/CD | GitHub Actions |

---

## 3. Repository Structure

```
UC-Davis-VMTH-Cancer-Registry/
├── frontend/                  # React 19 + Vite frontend
│   └── src/
│       ├── components/        # UI components (see below)
│       ├── contexts/          # AuthContext (Supabase auth)
│       ├── api/               # API client functions
│       └── types/             # TypeScript types
├── backend/                   # FastAPI backend
│   └── app/
│       ├── routers/           # API route handlers
│       ├── services/          # Business logic (ingestion, GCP Batch)
│       ├── models/            # SQLAlchemy ORM models
│       └── auth.py            # JWT validation middleware
├── ml/                        # PetBERT inference pipeline
│   ├── petbert_scan/          # Core pipeline (embedding, categorization)
│   ├── labels/                # Vet-ICD-O-canine-1 taxonomy + embeddings
│   └── scripts/               # Fine-tuning and training scripts
├── ml-worker/                 # Dockerized ML worker for local dev
│   └── Dockerfile.batch       # Production GCP Batch container
├── database/
│   └── migrations/            # 013 SQL migrations (run in order)
└── docs/                      # Architecture docs, workstream plans
```

### Key Frontend Components

| Component | Location | Status |
|---|---|---|
| `DataUpload` | `components/DataUpload/` | Implemented |
| `AdminQueue` | `components/AdminQueue/` | Implemented |
| `ChoroplethMap` | `components/ChoroplethMap/` | Implemented |
| `Filters` | `components/Filters/` | Partial (no year/species in UI) |
| `TrendChart` | `components/TrendChart/` | **Not implemented** |
| Case detail view | — | **Not implemented** |
| CSV export button | — | **Not implemented** |
| Plain-language search bar | — | **Not implemented** |

### Key Backend Routers

| Router | File | Purpose |
|---|---|---|
| `ingest` | `routers/ingest.py` | File upload, job management, admin approval |
| `geo` | `routers/geo.py` | County GeoJSON + case counts |
| `dashboard` | `routers/dashboard.py` | Summary stats |
| `trends` | `routers/trends.py` | Yearly time-series data |
| `incidence` | `routers/incidence.py` | Case counts by type/breed/species |
| `search` | `routers/search.py` | PetBERT classification endpoint |
| `auth` | `routers/auth.py` | `/auth/me` for role checking |

---

## 4. What Is Implemented

### Upload & Ingestion
- Two-file upload UI (Dataset A: clinical notes, Dataset B: demographics)
- Admin approval queue — all jobs require admin approval before processing
- Job status tracking with auto-polling (pending_review → processing → completed/failed)
- GCP Batch integration (`USE_GCP_BATCH=true` in `.env`) with local fallback
- Anon ID normalization (`"ID_37"`, `"37"`, `"37.0"` all resolve to `"ID_37"`)
- Idempotent re-runs (case_diagnoses deleted before re-insert)

### NLP Pipeline
- PetBERT embedding-based cosine similarity matching against Vet-ICD-O-canine-1 labels
- Dual execution paths: GCP Batch (production) and local ml-worker (development)
- `low_confidence` method flag for predictions below `embedding_min_sim` threshold
- Non-cancer / uncategorized case handling (empty text → `method = "empty"`)
- Full Vet-ICD-O-canine-1 taxonomy: `icd_o_code`, `predicted_term`, `predicted_group`

### Map Dashboard
- California county choropleth map (react-simple-maps + d3-scale)
- Filters: sex, cancer type, breed (wired end-to-end)
- Sortable county table with hover-to-highlight
- Hierarchical summary table (California → UC Davis Catchment → Regions → Counties)
- Case counts only (no incidence rate — dog population by county unavailable)

### Authentication & Security
- Supabase Auth with email/password sign-in
- JWT validation in FastAPI (`auth.py`)
- Admin role enforcement on upload and review endpoints
- RLS enabled on all Supabase tables (`012_enable_rls.sql`) — direct anon/authenticated key access is blocked; all queries go through FastAPI as the postgres superuser

### Database
- 13 migrations applied in order (001–013); migration 023 (sex column on materialized views) auto-applies on startup
- PostGIS county boundaries for all 58 CA counties
- `data_source = 'petbert'` distinguishes real ingested data from seed/mock data
- One case per patient (dog) model — prevents double-counting
- Materialized views (`mv_county_cancer_incidence`, `mv_yearly_trends`) refreshed after each ingest and via `POST /api/v1/admin/refresh-views`

### Security Hardening (May 2026)
- Three full security review rounds completed; see `docs/current-architecture.md` for the layer-by-layer summary
- JWT auth hardened: algorithm pinning (HS256/ES256/RS256/EdDSA only), generic error messages, in-memory failure rate limiting (5/15-minute IP lockout)
- Per-IP brute-force tracking has a 10 k IP cap with oldest-half eviction to bound memory
- Request body size enforced on **actual received bytes** (not just `Content-Length`) — 10 MB default, 50 MB for uploads
- CSV formula injection sanitization on every export cell
- Path-traversal defense via `pathlib.relative_to()` on every file system operation
- Race conditions fixed: atomic `UPDATE WHERE status='pending' RETURNING` for export approvals, `SELECT FOR UPDATE` for diagnosis review
- Supabase PKCE flow for password reset: link arrives at the app as `?token_hash=...&type=recovery`, verified client-side so Gmail/Outlook link previews can't consume the OTP
- Admin gating on every `/admin/*` and `/diagnoses/*` mutation; reviewer role for read-only review queue access
- API docs (`/docs`, `/redoc`, `/openapi.json`) are 404 in production (`DEBUG=false`)
- Strict security response headers on every response: `x-frame-options: DENY`, HSTS, nosniff, referrer-policy
- `.claudeignore` blocks Claude from reading `.env` or `secrets/`; pre-commit hook blocks committing secrets

### Test Suite (May 2026)
- Backend: **63 pytest tests** across 6 files (admin users, admin refresh-views, dashboard, incidence, security, review threshold analysis)
- Frontend: **279 vitest tests** across 13 files (filtered data, cancer types, CalEnviroScreen, password reset PKCE URL parsing, export requests, user management, pipeline stages, etc.)
- CI: `.github/workflows/ci.yml` runs both suites + TypeScript check + ESLint on every push to `main`/`database` and every PR
- DB is fully mocked in backend tests; no PostgreSQL container is needed for CI

### Scalability (May 2026)
- Backend container uses gunicorn with configurable `WORKERS` env var (defaults to 1 for single-vCPU Cloud Run)
- PostgreSQL connection pool is env-tunable via `DB_POOL_SIZE` (default 5) and `DB_MAX_OVERFLOW` (default 10)
- Cloud Run service template at `backend/service.yaml` with recommended limits, autoscaling, and Secret Manager refs
- See `docs/handoff/future_plans.md §6` for the scale-to-thousands-of-users plan

---

## 5. What Is Not Implemented

These features have backend support but no frontend UI, or are entirely missing:

| Feature | Backend Status | Frontend Status | Notes |
|---|---|---|---|
| **Fine-tuning trigger** | Local scripts in `ml/scripts/` | No trigger | `run_training_cycle.py` works locally; no GCP Batch integration for training |
| **Multi-clinic source tagging** | `data_source` column exists | No filter/display | Currently hardcoded to `'petbert'` |
| **Distributed rate limiting** | In-memory per-process | N/A | SlowAPI + auth-failure tracking are per-instance. See future_plans.md §6 for the Redis migration plan |
| **Distributed response cache** | `cachetools` per-process | N/A | TTLCache lives in each Cloud Run instance. Cache hit rate drops as instance count grows. Redis migration is documented |
| **Recently shipped** (April–May 2026) | — | — | CSV export with admin approval, case-level review queue, year/species filters, plain-language search, trend chart (recharts), Google OAuth, password reset PKCE, role-request workflow |

---

## 6. Running the Project Locally

### Prerequisites
- Docker Desktop
- Python 3.11+
- Node.js 20+

### Environment Setup

```bash
cp .env.example .env
# Fill in Supabase credentials and GCP config (see section 13)
```

### Start All Services

```bash
# Start database, backend, and frontend
docker compose up

# With seed data (mock records for testing)
docker compose --profile seed up
```

Services:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- Database: localhost:5432

### Run Database Migrations

```bash
# Run migrations in order
for i in $(seq -w 1 13); do
  docker compose exec db psql -U postgres -d vmth_cancer \
    -f /database/migrations/0${i}_*.sql 2>/dev/null || true
done
```

### Run the Ingestion Pipeline (Local)

Requires:
- `petbert_scan_predictions.csv` — PetBERT model output
- `All_deidentified_K9.xlsx` — Dataset A (clinical notes + demographics)
- Dataset B CSV — supplementary demographics (ZIP codes)

```bash
docker compose run --rm ingest
```

Expected output: 395 patients, 395 cancer_cases, ~2,348 case_diagnoses.

### Use GCP Batch for Inference (Production)

Set `USE_GCP_BATCH=true` in `.env`. See `docs/GCP_BATCH_SETUP.md` for full setup. In development, leave `USE_GCP_BATCH=false` to use the local ml-worker container.

---

## 7. GCP Setup

Full setup instructions: `docs/GCP_BATCH_SETUP.md`

### Summary of Required GCP Resources

| Resource | Purpose |
|---|---|
| Cloud Batch API | Run PetBERT inference jobs |
| Cloud Storage bucket | Store uploaded files + PetBERT model weights |
| Artifact Registry repo (`vmth`) | Store PetBERT Docker image (~7.7 GB compressed) |
| Service account (`vmth-batch`) | Batch job execution permissions |
| Secret Manager | Store Supabase service role key (injected into Batch container at runtime) |

### Required IAM Roles for `vmth-batch` Service Account
- `roles/batch.jobsEditor`
- `roles/storage.objectAdmin`
- `roles/artifactregistry.reader`

### Environment Variables for GCP

```env
USE_GCP_BATCH=true
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1
GCS_BUCKET=your-gcs-bucket-name
GCP_BATCH_IMAGE_URI=us-central1-docker.pkg.dev/your-project/vmth/petbert-batch:latest
GCP_BATCH_MACHINE_TYPE=n1-standard-4
GCP_BATCH_POLL_INTERVAL=60
GCP_BATCH_TIMEOUT_HOURS=12
GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/gcp-sa-key.json
```

The service account key (`gcp-sa-key.json`) lives in `secrets/` which is git-ignored. **Never commit it.**

---

## 8. Database Schema

### Core Tables

```
patients
  id, anon_id (UNIQUE), species_id, breed_id, sex, age_years,
  weight_kg, county_id, zip_code, data_source ('petbert' for real data)

cancer_cases
  id, patient_id → patients, cancer_type_id (nullable),
  county_id, diagnosis_date, stage, outcome
  NOTE: One row per patient (dog). Types live in case_diagnoses.

case_diagnoses
  id, case_id → cancer_cases, cancer_type_id → cancer_types,
  icd_o_code, predicted_term, predicted_group, original_text,
  confidence, prediction_method ('low_confidence' if below threshold)

cancer_types
  id, name (Vet-ICD-O group name)

counties
  id, name, fips_code, geom (PostGIS MULTIPOLYGON), is_catchment

ingestion_jobs
  id, status (pending_review/processing/completed/failed/rejected),
  batch_job_name, user_id, created_at, updated_at

species, breeds, calenviroscreen, ingestion_logs
```

### Important Design Decisions

1. **One case per patient:** A dog with multiple diagnoses has one `cancer_cases` row and multiple `case_diagnoses` rows. This prevents double-counting.
2. **`cancer_cases.cancer_type_id` is nullable:** Types live in `case_diagnoses`, not on the case itself.
3. **`data_source = 'petbert'`:** All dashboard queries filter on this to exclude seed/mock data.
4. **RLS with no permissive policies:** All table access goes through FastAPI (postgres superuser). The Supabase anon/authenticated keys cannot read or write any table directly.

---

## 9. Data Pipeline

### Input Files

| File | Columns | Purpose |
|---|---|---|
| Dataset A | `anon_id`, `DtOfRq`, `Sex`, `Species`, `Breed`, `Clinical Diagnoses`, `Text` | Demographics + pathology report text (PetBERT input) |
| Dataset B | `anon_id`, `Sex`, `Zipcode` | Supplementary ZIP code for county mapping |

### Anon ID Normalization (Critical)

CSV uses `"ID_37"`, Excel may export `"37"` or `"37.0"`. All formats normalize to `"ID_<number>"` via `normalize_anon_id()`. Without this, only ~269 of 395 patients match.

### Pipeline Steps

1. PetBERT processes `Text` column from Dataset A → predictions CSV
2. Ingestion job is submitted via upload UI
3. Admin approves job in AdminQueue
4. GCP Batch (or local ml-worker) runs `batch_predict.py`
5. Results written to `patients`, `cancer_cases`, `case_diagnoses` in Supabase
6. Map dashboard queries update automatically

### PetBERT Inference

- Uses cosine similarity between report embeddings and Vet-ICD-O label embeddings
- Threshold: `embedding_min_sim` (configurable) — below threshold → `method = "low_confidence"`
- Output per prediction: `predicted_term`, `predicted_group`, `icd_o_code`, `confidence`, `method`

---

## 10. Key Architectural Decisions

| Decision | Rationale |
|---|---|
| PetBERT over BERT Base / BioBERT | Only BERT variant pretrained on veterinary EHR data — best domain fit for VMTH pathology terminology |
| Supabase over self-hosted Postgres | Managed hosting, built-in auth, RLS — reduces operational burden |
| GCP Batch over Vertex AI | Inference is infrequent and doesn't need MLOps tooling; Batch reuses existing infrastructure and costs ~$1–2/run vs persistent GPU |
| FastAPI as middleman for all writes | Validates files, enforces auth, abstracts GCP from frontend; all data writes go through backend, never direct to Supabase |
| Frontend reads Supabase directly | Not applicable — RLS blocks direct access. All reads also go through FastAPI. |
| One case per patient model | Prevents inflated case counts when a dog has multiple visit records or predictions |
| Vercel for frontend | Zero-config React/Vite deploys, automatic PR previews, works well with GitHub Actions CI/CD |

---

## 11. Known Issues and Gotchas

### Anon ID Format Mismatch
CSV files may export anon IDs in different formats (`"ID_37"` vs `"37"` vs `"37.0"`). The `normalize_anon_id()` function in the ingestion pipeline handles this, but if you see unexpectedly low patient match counts, check that normalization is running.

### Re-run Idempotency
Re-running the ingestion pipeline on the same dataset deletes and re-inserts `case_diagnoses` for affected patients. This is intentional. `cancer_cases` rows are upserted (not deleted), so case IDs are stable across re-runs.

### GCP Batch Cold Start
Job startup takes ~60–90 seconds (container pull + model load) before inference begins. This is normal. The frontend polls job status every 10 seconds.

### `cancer_cases.cancer_type_id` is Nullable
Do not query `cancer_cases.cancer_type_id` for cancer type analysis — it is intentionally NULL for PetBERT cases. Always query `case_diagnoses` for type-level data.

### Dog Population Data Unavailable
The dashboard shows case counts, not incidence rates. Dog population by California county is not available, so rates cannot be calculated. Do not add population or rate fields to the UI without sourcing this data.

### Target Architecture Doc is Outdated
`docs/target-architecture.md` describes an earlier design using Redis + Celery for the NLP worker. The actual implementation uses GCP Batch instead. Refer to `docs/GCP_BATCH_SETUP.md` and `backend/app/services/gcp_batch_service.py` for the current design.

---

## 12. Remaining Work

Listed in priority order. Most of the original handoff backlog has shipped (test suite, CSV export, trend chart, case-level review queue, year/species filters, plain-language search). What remains:

### High Priority — Scale & Reliability

1. **Distributed rate limiting and response cache** — Both are per-process today (`slowapi` + `cachetools`). With multiple Cloud Run instances, rate limits are effectively `N × WORKERS × limit` and cache hit rate falls off. Migrate both to Redis (Memorystore on GCP, or Upstash). See `docs/handoff/future_plans.md §6.6` for the concrete plan.

2. **Materialized view refresh cron** — `POST /api/v1/admin/refresh-views` exists for manual refresh, but the view is otherwise only refreshed at the end of each ingestion. Set up Cloud Scheduler to hit the refresh endpoint nightly so data stays fresh even when no upload happened that day.

3. **Observability** — Add structured logging (already partially in place with `logger.info` calls) and ship logs/metrics to Cloud Logging + Cloud Monitoring. Build a dashboard for: request latency p50/p95/p99, error rate per endpoint, DB pool exhaustion events, auth-failure rate.

### Medium Priority — Product

4. **Fine-tuning via GCP Batch** — Local training pipeline exists in `ml/scripts/run_training_cycle.py`. Add a GCP Batch job definition and an API endpoint (`POST /api/v1/training/submit`) to trigger fine-tuning when enough labeled data has accumulated.

5. **Multi-clinic source tagging** — `patients.data_source` column is in place but hardcoded to `'petbert'`. Extend the ingestion pipeline to tag records by clinic and add a source filter to the dashboard. (Prerequisite for any multi-tenant work — see `future_plans.md` for the full multi-clinic plan.)

6. **Audit log** — Add an `audit_log` table that records every privileged action (role change, export approval, diagnosis review, refresh-views). Currently `diagnosis_review_events` is the only audit table.

7. **Account deletion / data retention** — No `DELETE /api/v1/users/me` exists. For GDPR/CCPA compliance, add user-initiated account deletion and a clinic-data deletion endpoint for system admins.

### Low Priority

8. **Federated cross-clinic queries** — Once multi-clinic tagging ships, expose anonymized aggregate-only queries across clinics for state-wide research.

9. **FHIR integration** — Long-term, replace manual CSV upload with a FHIR pull from clinic EHR systems. Treat as 2–3 year horizon.

---

## 13. Credentials and Secrets

**Never commit any of the following to git.**

| Secret | Where It Lives | Used By |
|---|---|---|
| Supabase URL | `.env` → `SUPABASE_URL` | Backend |
| Supabase anon key | `.env` → `SUPABASE_ANON_KEY` | Backend auth validation |
| Supabase service role key | GCP Secret Manager | PetBERT Batch container (write to DB) |
| GCP service account key | `secrets/gcp-sa-key.json` (git-ignored) | FastAPI → GCP Batch + GCS |
| Database URL | `.env` → `DATABASE_URL` | Backend SQLAlchemy |

The `secrets/` directory is in `.gitignore`. The Supabase service role key is **not** stored locally — it is injected into the GCP Batch container at runtime via GCP Secret Manager.

To rotate the GCP service account key:
```bash
gcloud iam service-accounts keys create secrets/gcp-sa-key.json \
  --iam-account=vmth-batch@YOUR_PROJECT_ID.iam.gserviceaccount.com
```
