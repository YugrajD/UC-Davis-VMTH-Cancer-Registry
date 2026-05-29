# Current Architecture Snapshot

[Back to Overview](../IMPLEMENTATION_PLAN.md)

---

```
+---------------------------------------------------------------------------+
|                            Docker Compose                                 |
|                                                                           |
|  +-----------+    +---------------+    +----------------------+           |
|  |  Frontend  |--->|   Backend     |--->|  PostgreSQL + PostGIS|           |
|  |  React 19  |    |  FastAPI      |    |  postgis:16-3.4      |           |
|  |  Vite      |    |  Python 3.11  |    |                      |           |
|  |  Port 5173 |    |  Port 8000    |    |  Port 5432           |           |
|  +-----------+    +-------+-------+    +----------------------+           |
|                           |                                               |
|                     +-----+------+                                        |
|                     |  ml/ volume | (PetBERT models + training code)      |
|                     +-----+------+                                        |
|                           |                                               |
|                    +------+-------+                                       |
|                    |  ML Worker   | (PetBERT inference, internal only)    |
|                    |  Port 8001*  | * not exposed to host                 |
|                    +--------------+                                       |
|                                                                           |
|  +-----------+  +-----------+  +-----------+  +-----------+              |
|  |   Seed    |  |  Ingest   |  | Geo-Seed  |  |  GCP Batch| (optional)  |
|  | --profile |  | --profile |  | --profile |  |  (remote) |              |
|  |   seed    |  |  ingest   |  |  geo-seed |  |           |              |
|  +-----------+  +-----------+  +-----------+  +-----------+              |
+---------------------------------------------------------------------------+
```

## Security Layers

The backend enforces multiple layers of security:

| Layer | Mechanism | Details |
|-------|-----------|---------|
| **Authentication** | Supabase JWT (HS256/ES256) | All write endpoints + search require auth |
| **Authorization** | Role-based (admin, reviewer, uploader) | DB-backed `user_roles` table with env-var fallback |
| **Rate limiting** | slowapi (60 req/min global) | Per-IP with trusted proxy support |
| **Auth brute-force** | In-memory IP tracking | 5 failures per 15 min window |
| **Body size limit** | ASGI middleware | 10 MB default, 50 MB for uploads |
| **Input validation** | Pydantic `Field()` + FastAPI `Query()` | `max_length`, `Literal` types, range bounds |
| **LIKE injection** | `_escape_like()` helper | Escapes `%`, `_`, `\` in search keywords |
| **Error sanitization** | `_safe_error_message()` | Only RuntimeError messages passed through; others show class name only |
| **Docker hardening** | Non-root user, no `--reload` in CMD | ML worker port not exposed to host |
| **SMTP** | TLS with `ssl.create_default_context()` | 30-second connection timeout |
| **CSP** | `<meta>` Content Security Policy | Restricts script/style/connect sources |
| **Security response headers** | ASGI middleware | `x-content-type-options`, `x-frame-options: DENY`, HSTS, `referrer-policy` |
| **HTTP cache headers** | ASGI middleware | `Cache-Control` + `Vary` on public read endpoints |
| **Cache invalidation** | `clear_all_caches()` | Cleared after every ingestion and admin refresh-views call |
| **API docs gating** | `DEBUG` env flag (default false) | `/docs`, `/redoc`, `/openapi.json` are 404 in production |
| **Race condition protection** | Atomic UPDATE WHERE + SELECT FOR UPDATE | Export approvals, diagnosis review, job review |
| **Path traversal** | `pathlib.relative_to()` | All file-system operations checked against `UPLOAD_DIR` |
| **PKCE auth flow** | Supabase `flowType: 'pkce'` | Password reset uses `verifyOtp(token_hash)` so email scanners can't consume OTPs by pre-fetching |

## Data Flow

1. Users upload CSV/XLSX datasets via the ingestion workflow.
2. An admin or reviewer approves the upload job.
3. The backend sends the dataset to the ML worker (or GCP Batch) for PetBERT categorization.
4. Predictions are ingested into PostgreSQL (patients, cancer cases, case diagnoses).
5. Materialized views are refreshed for fast aggregation queries.
6. Frontend fetches from backend REST endpoints, renders choropleth maps, trend charts, and tables.
7. Reviewers can manually review and correct individual diagnoses via the review workflow.

## Database Schema

```
species (id, name)
breeds (id, species_id, name)
cancer_types (id, name, description, confirmed, icd_o_morphology_code)
counties (id, name, fips_code, geom, population, area_sq_miles, is_catchment)
patients (id, species_id, breed_id, sex, birth_date, county_id, zip_code, anon_id,
          diagnosis_date, outcome, data_source)
  -- Note: no cancer_cases table; diagnosis_date and outcome are on patients directly
case_diagnoses (id, patient_id, cancer_type_id, predicted_term, predicted_group,
                icd_o_code, confidence, prediction_method, review_status,
                pathology_report_id, ingestion_job_id, ...)
pathology_reports (id, patient_id, gcs_path, report_date, created_at)
ingestion_jobs (id, status, uploaded_by_sub, storage_path, batch_job_name, ...)
user_roles (email, is_admin, is_uploader, is_reviewer)
role_requests (id, email, requested_role, status, reason, ...)
export_requests (id, email, status, reason, ...)
diagnosis_review_events (id, case_diagnosis_id, action, ...)
calenviroscreen (county_id, ces_score, pollution_burden, ...)

Materialized Views:
  mv_county_cancer_incidence  -- top-1 prediction per patient, Non-Cancer excluded
  mv_yearly_trends
```

## API Endpoints (11 routers)

```
auth:             GET  /api/v1/auth/me
dashboard:        GET  /api/v1/dashboard/summary
                  GET  /api/v1/dashboard/filters
incidence:        GET  /api/v1/incidence
                  GET  /api/v1/incidence/by-cancer-type
                  GET  /api/v1/incidence/by-species
                  GET  /api/v1/incidence/by-breed
                  GET  /api/v1/incidence/breed-detail
geo:              GET  /api/v1/geo/counties
                  GET  /api/v1/geo/counties/{county_id}
trends:           GET  /api/v1/trends/yearly
                  GET  /api/v1/trends/by-cancer-type
search:           POST /api/v1/search/classify          (auth required)
                  GET  /api/v1/search/reports            (auth required)
ingest:           GET  /api/v1/ingest/status
                  POST /api/v1/ingest/upload             (auth required)
                  GET  /api/v1/ingest/jobs
                  GET  /api/v1/ingest/jobs/{id}
                  GET  /api/v1/ingest/jobs/{id}/preview  (reviewer)
                  POST /api/v1/ingest/jobs/{id}/review   (reviewer)
                  POST /api/v1/ingest/jobs/{id}/cancel   (reviewer)
diagnoses:        GET  /api/v1/diagnoses/pending         (reviewer)
                  GET  /api/v1/diagnoses/pending/count   (reviewer)
                  GET  /api/v1/diagnoses/{id}            (reviewer)
                  POST /api/v1/diagnoses/{id}/review     (reviewer)
admin:            GET  /api/v1/admin/users/{email}/roles (admin)
                  PUT  /api/v1/admin/users/{email}/roles (admin)
                  GET  /api/v1/admin/users/roles         (admin)
                  POST /api/v1/admin/refresh-views       (admin)
role-requests:    POST /api/v1/role-requests/            (auth required)
                  GET  /api/v1/role-requests/            (auth required)
                  GET  /api/v1/role-requests/pending/count (admin)
                  POST /api/v1/role-requests/{id}/resolve  (admin)
export-requests:  POST /api/v1/export-requests/          (auth required)
                  GET  /api/v1/export-requests/           (auth required)
                  GET  /api/v1/export-requests/pending/count (admin)
                  POST /api/v1/export-requests/{id}/resolve  (admin)
                  GET  /api/v1/export-requests/download      (approved user)
health:           GET  /health
root:             GET  /
```
