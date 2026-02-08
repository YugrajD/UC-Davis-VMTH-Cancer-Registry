# Target Architecture

[Back to Overview](../IMPLEMENTATION_PLAN.md)

---

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            Docker Compose                                │
│                                                                          │
│  ┌────────────┐    ┌──────────────────┐    ┌──────────────────────┐     │
│  │  Frontend   │───▶│    Backend       │───▶│  PostgreSQL + PostGIS│     │
│  │  React 19   │    │  FastAPI         │    │  postgis:16-3.4      │     │
│  │  + recharts │    │  Python 3.11     │    │                      │     │
│  │  Port 5173  │    │  Port 8000       │    │  Port 5432           │     │
│  └────────────┘    │                  │    └──────────────────────┘     │
│                     │  Routers:        │              ▲                  │
│                     │   dashboard      │              │                  │
│                     │   incidence      │    ┌─────────┴────────┐        │
│                     │   geo            │    │   Redis           │        │
│                     │   trends         │    │   (task broker)   │        │
│                     │   search         │    │   Port 6379       │        │
│                     │   upload  [NEW]  │    └─────────┬────────┘        │
│                     │   auth   [NEW]   │              │                  │
│                     │   review [NEW]   │    ┌─────────┴────────┐        │
│                     └──────────────────┘    │   NLP Worker      │        │
│                                             │   (Celery)        │        │
│                                             │   BERT model      │        │
│                                             └──────────────────┘        │
│                                                                          │
│  ┌───────────┐                                                           │
│  │   Seed    │ (one-shot, --profile seed)                               │
│  └───────────┘                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

**Target data flow:**

```
CSV File ──▶ POST /api/v1/upload/csv ──▶ ingestion_service.py
                                              │
                           ┌──────────────────┴──────────────────┐
                           │                                      │
                     Structured rows                        Free-text fields
                     (species, breed,                       (pathology notes)
                      sex, age, county,                          │
                      diagnosis_date,                     ┌──────▼──────┐
                      cancer_type)                        │ Redis Queue  │
                           │                              └──────┬──────┘
                           ▼                                     │
                    ┌──────────────┐                    ┌────────▼────────┐
                    │  patients    │                    │  NLP Worker     │
                    │  cancer_cases│                    │  (BERT model)   │
                    │  raw_uploads │                    │                 │
                    └──────────────┘                    │  confidence ≥ 0.7 → auto_accepted
                                                       │  confidence < 0.7 → flagged
                                                       └────────┬────────┘
                                                                │
                                                       ┌────────▼────────┐
                                                       │ pathology_reports│
                                                       │ (classification, │
                                                       │  confidence,     │
                                                       │  review_status)  │
                                                       └─────────────────┘
                                                                │
                                                       ┌────────▼────────┐
                                                       │  Review Queue   │
                                                       │  (flagged items)│
                                                       │  Manual review  │
                                                       │  by auth'd user │
                                                       └─────────────────┘
```

**Target database schema (additions marked with [+]):**

```
species (id, name)
breeds (id, species_id, name)
cancer_types (id, name, description, [+icd_o_morphology_code], [+icd_o_topography_code], [+icd_o_label])
counties (id, name, fips_code, geom, population, area_sq_miles)
patients (id, species_id, breed_id, sex, age_years, weight_kg, county_id, registered_date)
cancer_cases (id, patient_id, cancer_type_id, diagnosis_date, stage, outcome, county_id)
pathology_reports (id, case_id, report_text, classification, confidence_score, report_date,
                   [+review_status], [+reviewed_by], [+reviewed_at])

[+users] (id, username, email, hashed_password, role, is_active, created_at, updated_at)
[+raw_uploads] (id, user_id, filename, uploaded_at, row_count, accepted_count, rejected_count, status)
[+upload_records] (id, upload_id, cancer_case_id)
[+nlp_jobs] (id, report_id, status, queued_at, started_at, completed_at, error_message)
```

**Target API surface (additions marked with [NEW]):**
```
dashboard:  GET  /api/v1/dashboard/summary
            GET  /api/v1/dashboard/filters
incidence:  GET  /api/v1/incidence
            GET  /api/v1/incidence/by-cancer-type
            GET  /api/v1/incidence/by-species
            GET  /api/v1/incidence/by-breed
geo:        GET  /api/v1/geo/counties
            GET  /api/v1/geo/counties/{county_id}
trends:     GET  /api/v1/trends/yearly
            GET  /api/v1/trends/by-cancer-type
search:     POST /api/v1/search/classify
            GET  /api/v1/search/reports
upload:     POST /api/v1/upload/csv              [NEW]
            POST /api/v1/upload/text             [NEW]
            GET  /api/v1/upload/history           [NEW]
            GET  /api/v1/upload/status/{job_id}   [NEW]
auth:       POST /api/v1/auth/login              [NEW]
            POST /api/v1/auth/register           [NEW]
            GET  /api/v1/auth/me                 [NEW]
review:     GET  /api/v1/review/queue            [NEW]
            GET  /api/v1/review/stats            [NEW]
            PUT  /api/v1/review/{report_id}      [NEW]
health:     GET  /health
root:       GET  /
```

**Target frontend component tree:**
```
AuthProvider (useAuth context)
└── App.tsx
    ├── LoginPage (shown when not authenticated and accessing protected routes)
    ├── Navigation (tabs: overview, breed-disparities, cancer-types, trends,
    │               regional-comparison, upload*, review*)    [* = auth-only]
    ├── [overview tab]
    │   ├── Filters
    │   ├── ChoroplethMap
    │   ├── SummaryTable
    │   ├── CountyTable
    │   └── TrendChart (yearly overview)         [NEW]
    ├── [breed-disparities tab]
    │   └── BreedDisparities (real API data)     [NEW]
    ├── [cancer-types tab]
    │   └── CancerTypesChart (real API data)     [NEW]
    ├── [trends tab]                             [NEW]
    │   └── TrendChart (multi-series by cancer type)
    ├── [regional-comparison tab]
    │   └── RegionalComparison (real API data)   [NEW]
    ├── [upload tab] (auth-required)             [NEW]
    │   └── UploadPage (CSV + text paste)
    ├── [review tab] (auth-required)             [NEW]
    │   └── ReviewQueue (flagged reports)
    └── Footer
```
