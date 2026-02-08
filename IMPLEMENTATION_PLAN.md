# Implementation Plan: Closing Requirements Gaps

**Project:** UC Davis VMTH Cancer Registry
**Date:** 2026-02-08
**Reference:** `Requirements_Doc.md`

This document outlines every gap between the current codebase and the requirements, organized into workstreams with architecture, API contracts, database schemas, component designs, and step-by-step implementation tasks.

---

## Table of Contents

- [Gap Summary](#gap-summary)
- [Current Architecture Snapshot](#current-architecture-snapshot)
- [Target Architecture](#target-architecture)
- [Workstream 1: CSV Upload & Data Ingestion Pipeline](#workstream-1-csv-upload--data-ingestion-pipeline)
- [Workstream 2: Real BERT Integration & Async NLP Worker](#workstream-2-real-bert-integration--async-nlp-worker)
- [Workstream 3: Vet-ICD-O-canine-1 Coding System](#workstream-3-vet-icd-o-canine-1-coding-system)
- [Workstream 4: Authentication & Access Control](#workstream-4-authentication--access-control)
- [Workstream 5: Trend Line Visualization](#workstream-5-trend-line-visualization)
- [Workstream 6: Ambiguous Diagnosis Flagging & Review](#workstream-6-ambiguous-diagnosis-flagging--review)
- [Workstream 7: Fix Frontend Tabs (Real Data)](#workstream-7-fix-frontend-tabs-real-data)
- [Workstream 8: Tests](#workstream-8-tests)
- [Implementation Order](#implementation-order)
- [File Change Summary](#file-change-summary)

---

## Gap Summary

| # | Gap | Severity | User Stories |
|---|-----|----------|--------------|
| 1 | No CSV upload / data ingestion pipeline | Critical | US #2, #9 |
| 2 | No real BERT integration (mock keyword matcher only) | Critical | US #3 |
| 3 | No Vet-ICD-O-canine-1 coding system | Critical | US #1 |
| 4 | No authentication / access control | Critical | Security req |
| 5 | No trend line visualization in the frontend | Major | US #6 |
| 6 | No ambiguous diagnosis flagging / review workflow | Major | US #8 |
| 7 | Frontend tabs rendering fake data instead of real API data | Major | US #11 |
| 8 | NLP worker is inline, not async/separate | Moderate | Architecture |
| 9 | No tests | Moderate | Maintainability |

---

## Current Architecture Snapshot

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Docker Compose                              │
│                                                                     │
│  ┌───────────┐    ┌───────────────┐    ┌──────────────────────┐    │
│  │  Frontend  │───▶│   Backend     │───▶│  PostgreSQL + PostGIS│    │
│  │  React 19  │    │  FastAPI      │    │  postgis:16-3.4      │    │
│  │  Vite      │    │  Python 3.11  │    │                      │    │
│  │  Port 5173 │    │  Port 8000    │    │  Port 5432           │    │
│  └───────────┘    └───────────────┘    └──────────────────────┘    │
│                          │                                          │
│                    ┌─────┴──────┐                                   │
│                    │  ml/ volume │ (mounted, not integrated)        │
│                    └────────────┘                                   │
│                                                                     │
│  ┌───────────┐                                                      │
│  │   Seed    │ (one-shot, --profile seed)                          │
│  └───────────┘                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Current data flow:**
1. Mock seed script inserts ~5,000 random records into PostgreSQL.
2. Frontend fetches from backend REST endpoints, renders choropleth map + tables.
3. 3 of 4 frontend tabs use hardcoded/random data (`Math.random()`).
4. Classification endpoint (`POST /api/v1/search/classify`) uses inline keyword matcher — NOT the `ml/model/classifier.py` VetBERT mock.
5. No file upload, no auth, no ICD-O codes, no review workflow.

**Current database schema (6 migrations):**
```
species (id, name)
breeds (id, species_id, name)
cancer_types (id, name, description)
counties (id, name, fips_code, geom, population, area_sq_miles)
patients (id, species_id, breed_id, sex, age_years, weight_kg, county_id, registered_date)
cancer_cases (id, patient_id, cancer_type_id, diagnosis_date, stage, outcome, county_id)
pathology_reports (id, case_id, report_text, classification, confidence_score, report_date)

Materialized Views:
  mv_county_cancer_incidence
  mv_yearly_trends
```

**Current API endpoints (5 routers):**
```
dashboard:  GET /api/v1/dashboard/summary
            GET /api/v1/dashboard/filters
incidence:  GET /api/v1/incidence
            GET /api/v1/incidence/by-cancer-type
            GET /api/v1/incidence/by-species
            GET /api/v1/incidence/by-breed
geo:        GET /api/v1/geo/counties
            GET /api/v1/geo/counties/{county_id}
trends:     GET /api/v1/trends/yearly
            GET /api/v1/trends/by-cancer-type
search:     POST /api/v1/search/classify
            GET  /api/v1/search/reports
health:     GET /health
root:       GET /
```

**Current frontend component tree:**
```
App.tsx
├── Navigation (tabs: overview, breed-disparities, cancer-types, regional-comparison)
├── Filters (rateType, sex, cancerType, breed)
├── ChoroplethMap (D3 + react-simple-maps)
├── SummaryTable (region hierarchy)
├── CountyTable (sortable county list)
├── [breed-disparities tab] — HARDCODED DATA
├── [cancer-types tab] — HARDCODED DATA
├── [regional-comparison tab] — HARDCODED DATA
└── Footer
```

---

## Target Architecture

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

**Target database schema (additions in bold):**

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

---

## Workstream 1: CSV Upload & Data Ingestion Pipeline

**Gaps addressed:** #1 (US #2, #9)

### 1.1 Database — New Migration: `database/migrations/007_raw_uploads.sql`

This migration creates tables to track file uploads and link them to the records they produce.

**Exact SQL:**

```sql
-- 007_raw_uploads.sql
-- Track CSV uploads and link to created records

CREATE TABLE IF NOT EXISTS raw_uploads (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),  -- NULL if uploaded before auth exists
    filename VARCHAR(255) NOT NULL,
    uploaded_at TIMESTAMP NOT NULL DEFAULT NOW(),
    row_count INTEGER NOT NULL DEFAULT 0,
    accepted_count INTEGER NOT NULL DEFAULT 0,
    rejected_count INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    error_log TEXT  -- JSON array of row-level errors
);

CREATE TABLE IF NOT EXISTS upload_records (
    id SERIAL PRIMARY KEY,
    upload_id INTEGER NOT NULL REFERENCES raw_uploads(id) ON DELETE CASCADE,
    cancer_case_id INTEGER NOT NULL REFERENCES cancer_cases(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_raw_uploads_status ON raw_uploads (status);
CREATE INDEX IF NOT EXISTS idx_raw_uploads_user ON raw_uploads (user_id);
CREATE INDEX IF NOT EXISTS idx_upload_records_upload ON upload_records (upload_id);
```

**Docker Compose change** — add volume mount in `db` service:
```yaml
- ./database/migrations/007_raw_uploads.sql:/docker-entrypoint-initdb.d/007_raw_uploads.sql
```

> **Note:** This migration depends on the `users` table from Workstream 4 (migration 009). If implementing before auth, make `user_id` nullable with no FK constraint initially, then add the FK via ALTER TABLE in migration 009.

---

### 1.2 Backend — Ingestion Service: `backend/app/services/ingestion_service.py`

**Purpose:** Validate and map CSV rows to database records.

**Expected CSV format (minimum required columns):**

| Column | Type | Maps To | Required |
|--------|------|---------|----------|
| `species` | string | `species.name` | Yes |
| `breed` | string | `breeds.name` | Yes |
| `sex` | string | `patients.sex` | Yes |
| `age_years` | float | `patients.age_years` | Yes |
| `weight_kg` | float | `patients.weight_kg` | No |
| `county` | string | `counties.name` | Yes |
| `registered_date` | date | `patients.registered_date` | Yes |
| `cancer_type` | string | `cancer_types.name` | Yes |
| `diagnosis_date` | date | `cancer_cases.diagnosis_date` | Yes |
| `stage` | string | `cancer_cases.stage` | No |
| `outcome` | string | `cancer_cases.outcome` | No |
| `pathology_notes` | text | `pathology_reports.report_text` | No |

**Validation rules:**
1. `species` must be case-insensitive match to existing `species.name` ("dog", "Dog", "DOG" all match "Dog").
2. `breed` must be case-insensitive match to existing `breeds.name` for the given species. If no match, reject the row with error `"Unknown breed '{value}' for species '{species}'"`.
3. `sex` must be one of: `Male`, `Female`, `Neutered Male`, `Spayed Female` (case-insensitive). Map common aliases: `M` → `Male`, `F` → `Female`, `MN` → `Neutered Male`, `SF` → `Spayed Female`.
4. `county` must be case-insensitive match to existing `counties.name`. If no match, reject.
5. `cancer_type` must be case-insensitive match to existing `cancer_types.name`. If no match, reject.
6. `stage` must be one of `I`, `II`, `III`, `IV` or empty.
7. `outcome` must be one of `alive`, `deceased`, `unknown` or empty.
8. Date fields must parse as valid dates.
9. Numeric fields (`age_years`, `weight_kg`) must be non-negative numbers.

**Architecture:**

```python
# backend/app/services/ingestion_service.py

class IngestionError:
    row_number: int
    column: str
    value: str
    message: str

class IngestionResult:
    upload_id: int
    rows_parsed: int
    rows_accepted: int
    rows_rejected: int
    errors: list[IngestionError]

class IngestionService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self._species_cache: dict[str, int] = {}   # name_lower -> id
        self._breed_cache: dict[str, int] = {}      # "species_id:name_lower" -> id
        self._cancer_cache: dict[str, int] = {}     # name_lower -> id
        self._county_cache: dict[str, int] = {}     # name_lower -> id

    async def _load_lookup_caches(self):
        """Pre-load all lookup tables into memory for fast matching."""

    async def ingest_csv(self, file_content: bytes, filename: str, user_id: int | None) -> IngestionResult:
        """
        1. Parse CSV with pandas.read_csv()
        2. Create raw_uploads record with status='processing'
        3. Validate each row, collect errors
        4. Bulk insert valid patients + cancer_cases
        5. Create upload_records links
        6. If pathology_notes present, create pathology_reports
           and queue NLP jobs (Workstream 2)
        7. Update raw_uploads status to 'completed'
        8. Return IngestionResult
        """

    def _validate_row(self, row: pd.Series, row_num: int) -> tuple[dict | None, list[IngestionError]]:
        """Validate a single CSV row. Returns (mapped_data, errors)."""

    def _map_sex(self, value: str) -> str | None:
        """Map sex aliases to canonical values."""
```

---

### 1.3 Backend — Upload Router: `backend/app/routers/upload.py`

**Endpoints:**

#### `POST /api/v1/upload/csv`

Accepts multipart file upload, invokes ingestion service.

```
Request:
  Content-Type: multipart/form-data
  Body: file (binary, .csv)

Response 200:
{
  "upload_id": 42,
  "filename": "vmth_cases_2024.csv",
  "rows_parsed": 150,
  "rows_accepted": 143,
  "rows_rejected": 7,
  "errors": [
    {"row_number": 12, "column": "breed", "value": "Goldenn Retriver", "message": "Unknown breed 'Goldenn Retriver' for species 'Dog'"},
    {"row_number": 45, "column": "stage", "value": "V", "message": "Invalid stage 'V'. Must be one of: I, II, III, IV"}
  ]
}

Response 400:
{
  "detail": "File must be a CSV. Received: application/pdf"
}

Response 422:
{
  "detail": "CSV is missing required columns: ['species', 'breed']"
}
```

**Implementation pattern** (matching existing codebase style):

```python
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.services.ingestion_service import IngestionService
from app.schemas.schemas import UploadResponse

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])

REQUIRED_COLUMNS = {"species", "breed", "sex", "age_years", "county",
                    "registered_date", "cancer_type", "diagnosis_date"}

@router.post("/csv", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    # current_user: User = Depends(get_current_user),  # Add after Workstream 4
):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    content = await file.read()
    service = IngestionService(db)
    result = await service.ingest_csv(content, file.filename, user_id=None)
    return result
```

#### `POST /api/v1/upload/text`

Accepts raw text for classification (free-text pathology notes).

```
Request:
{
  "text": "Histopathology reveals diffuse large B-cell lymphoma...",
  "save": true   // optional: persist classification to DB
}

Response 200:
{
  "predicted_cancer_type": "Lymphoma",
  "icd_o_code": "9590/3",           // after Workstream 3
  "confidence": 0.92,
  "review_status": "auto_accepted",  // after Workstream 6
  "report_id": 1234,                 // if save=true
  "top_predictions": [
    {"cancer_type": "Lymphoma", "confidence": 0.92},
    {"cancer_type": "Hemangiosarcoma", "confidence": 0.04}
  ]
}
```

#### `GET /api/v1/upload/history`

Returns paginated list of past uploads.

```
Query params: ?limit=20&offset=0

Response 200:
{
  "uploads": [
    {
      "id": 42,
      "filename": "vmth_cases_2024.csv",
      "uploaded_at": "2026-02-08T10:30:00Z",
      "row_count": 150,
      "accepted_count": 143,
      "rejected_count": 7,
      "status": "completed"
    }
  ],
  "total": 5
}
```

#### `GET /api/v1/upload/status/{job_id}`

Check the status of an async NLP processing job (Workstream 2).

```
Response 200:
{
  "job_id": "abc-123",
  "status": "processing",    // pending | processing | completed | failed
  "reports_total": 10,
  "reports_completed": 7,
  "reports_flagged": 2
}
```

---

### 1.4 Backend — Pydantic Schemas

Add to `backend/app/schemas/schemas.py`:

```python
# --- Upload ---

class UploadError(BaseModel):
    row_number: int
    column: str
    value: str = ""
    message: str

class UploadResponse(BaseModel):
    upload_id: int
    filename: str
    rows_parsed: int
    rows_accepted: int
    rows_rejected: int
    errors: List[UploadError]

class UploadHistoryItem(BaseModel):
    id: int
    filename: str
    uploaded_at: str
    row_count: int
    accepted_count: int
    rejected_count: int
    status: str
    model_config = {"from_attributes": True}

class UploadHistoryResponse(BaseModel):
    uploads: List[UploadHistoryItem]
    total: int

class TextClassifyRequest(BaseModel):
    text: str
    save: bool = False
```

---

### 1.5 Frontend — Upload Page Component

**File:** `frontend/src/components/UploadPage/UploadPage.tsx`

**Component design:**

```
┌──────────────────────────────────────────────────┐
│  Upload Data                                      │
│                                                    │
│  ┌──────────────────────────────────────────────┐│
│  │  [Tab: CSV File]  [Tab: Free Text]           ││
│  └──────────────────────────────────────────────┘│
│                                                    │
│  CSV File tab:                                     │
│  ┌──────────────────────────────────────────────┐│
│  │  ┌─────────────────────────────────────┐     ││
│  │  │  📁 Drag & drop CSV here            │     ││
│  │  │     or click to browse              │     ││
│  │  └─────────────────────────────────────┘     ││
│  │                                               ││
│  │  Preview (first 5 rows):                      ││
│  │  ┌────────┬───────┬──────┬─────┐             ││
│  │  │species │breed  │sex   │age  │ ...         ││
│  │  ├────────┼───────┼──────┼─────┤             ││
│  │  │Dog     │Golden │Male  │7.5  │             ││
│  │  └────────┴───────┴──────┴─────┘             ││
│  │                                               ││
│  │  [Upload]                                     ││
│  │                                               ││
│  │  Results:                                     ││
│  │  ✅ 143 rows accepted                         ││
│  │  ❌ 7 rows rejected                           ││
│  │  Row 12: breed "Goldenn Retriver" not found   ││
│  └──────────────────────────────────────────────┘│
│                                                    │
│  Free Text tab:                                    │
│  ┌──────────────────────────────────────────────┐│
│  │  Paste pathology report:                      ││
│  │  ┌─────────────────────────────────────┐     ││
│  │  │                                     │     ││
│  │  │  (textarea)                         │     ││
│  │  │                                     │     ││
│  │  └─────────────────────────────────────┘     ││
│  │  [Classify]                                   ││
│  │                                               ││
│  │  Result:                                      ││
│  │  Predicted: Lymphoma (9590/3)  Confidence: 92%││
│  │  [Save to Registry]                           ││
│  └──────────────────────────────────────────────┘│
│                                                    │
│  Upload History:                                   │
│  ┌──────────┬──────────┬─────┬─────┬──────────┐ │
│  │ Filename │ Date     │ OK  │ Err │ Status   │ │
│  ├──────────┼──────────┼─────┼─────┼──────────┤ │
│  │ data.csv │ 2/8/2026 │ 143 │  7  │ complete │ │
│  └──────────┴──────────┴─────┴─────┴──────────┘ │
└──────────────────────────────────────────────────┘
```

**Props and state:**
```typescript
// No external props — self-contained page component
// Internal state:
//   activeUploadTab: 'csv' | 'text'
//   file: File | null
//   preview: string[][] (first 5 rows parsed client-side)
//   uploading: boolean
//   uploadResult: UploadResponse | null
//   textInput: string
//   classifyResult: ClassifyResult | null
//   uploadHistory: UploadHistoryItem[]
```

**API client additions in `frontend/src/api/client.ts`:**

```typescript
export async function uploadCSV(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch('/api/v1/upload/csv', {
    method: 'POST',
    body: formData,
    headers: getAuthHeaders(),  // After Workstream 4
  });
  if (!response.ok) throw new Error(`Upload failed: ${response.status}`);
  return response.json();
}

export async function classifyText(text: string, save = false): Promise<ClassifyResult> {
  const response = await fetch('/api/v1/upload/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
    body: JSON.stringify({ text, save }),
  });
  if (!response.ok) throw new Error(`Classification failed: ${response.status}`);
  return response.json();
}

export async function fetchUploadHistory(limit = 20, offset = 0): Promise<UploadHistoryResponse> {
  return fetchJson(`/api/v1/upload/history?limit=${limit}&offset=${offset}`);
}
```

**Type additions in `frontend/src/types/index.ts`:**

```typescript
export interface UploadError {
  row_number: number;
  column: string;
  value: string;
  message: string;
}

export interface UploadResponse {
  upload_id: number;
  filename: string;
  rows_parsed: number;
  rows_accepted: number;
  rows_rejected: number;
  errors: UploadError[];
}

export interface UploadHistoryItem {
  id: number;
  filename: string;
  uploaded_at: string;
  row_count: number;
  accepted_count: number;
  rejected_count: number;
  status: string;
}
```

---

### 1.6 Navigation & App Integration

**`frontend/src/types/index.ts`** — extend `TabType`:
```typescript
// Before:
export type TabType = 'overview' | 'breed-disparities' | 'cancer-types' | 'regional-comparison';

// After:
export type TabType = 'overview' | 'breed-disparities' | 'cancer-types' | 'trends' | 'regional-comparison' | 'upload' | 'review';
```

**`frontend/src/types/index.ts`** — extend `TABS`:
```typescript
export const TABS: Tab[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'breed-disparities', label: 'Breed Disparities' },
  { id: 'cancer-types', label: 'Cancer Types' },
  { id: 'trends', label: 'Trends' },
  { id: 'regional-comparison', label: 'Regional Comparison' },
  // upload and review tabs rendered conditionally when authenticated
];
```

**`frontend/src/App.tsx`** — add conditional rendering:
```tsx
{activeTab === 'upload' && <UploadPage />}
{activeTab === 'review' && <ReviewQueue />}
```

---

### 1.7 Files Summary for Workstream 1

**Files to create:**
| File | Purpose |
|------|---------|
| `database/migrations/007_raw_uploads.sql` | Upload tracking tables |
| `backend/app/routers/upload.py` | Upload endpoints |
| `backend/app/services/ingestion_service.py` | CSV validation + DB insertion |
| `frontend/src/components/UploadPage/UploadPage.tsx` | Upload UI |

**Files to modify:**
| File | Change |
|------|--------|
| `backend/app/main.py` | Register upload router |
| `backend/app/schemas/schemas.py` | Add Upload* schemas |
| `frontend/src/api/client.ts` | Add `uploadCSV()`, `classifyText()`, `fetchUploadHistory()` |
| `frontend/src/types/index.ts` | Add Upload types, extend TabType |
| `frontend/src/components/Navigation/Navigation.tsx` | Show Upload tab (conditionally after auth) |
| `frontend/src/components/index.ts` | Export UploadPage |
| `frontend/src/App.tsx` | Render UploadPage for upload tab |
| `docker-compose.yml` | Mount migration 007 |

---

## Workstream 2: Real BERT Integration & Async NLP Worker

**Gaps addressed:** #2, #8 (US #3, Architecture)

### 2.1 Consolidate Classifier Code

**Problem:** Two separate keyword matchers exist:
- `ml/model/classifier.py` → `VetBERTClassifier` with weighted keywords
- `backend/app/services/bert_service.py` → `BertClassifier` with unweighted keywords

**Solution:** Delete `backend/app/services/bert_service.py` inline matcher. Make the backend import from `ml/model/classifier.py` (already volume-mounted in Docker).

**Update `backend/app/services/bert_service.py`:**

```python
"""
BERT classification service.
Wraps the ml/model/classifier.py module, which provides either
a real BERT model (production) or keyword-based fallback (development).
"""

import sys
import os

# ml/ is mounted at /ml in the Docker container
sys.path.insert(0, "/ml")
from model.classifier import VetBERTClassifier

from app.config import settings
from app.schemas.schemas import ClassifyResult


class BertService:
    def __init__(self):
        self.classifier = VetBERTClassifier(
            use_real_model=settings.USE_REAL_BERT,
            model_path=settings.BERT_MODEL_PATH,
        )

    def classify(self, text: str) -> ClassifyResult:
        result = self.classifier.predict(text)
        return ClassifyResult(
            predicted_cancer_type=result["predicted_label"],
            confidence=result["confidence"],
            top_predictions=[
                {"cancer_type": ct, "confidence": conf}
                for ct, conf in list(result["all_probabilities"].items())[:5]
            ],
        )

# Singleton instance
bert_service = BertService()
```

### 2.2 Upgrade `ml/model/classifier.py` to Support Real BERT

**Architecture:**

```python
class VetBERTClassifier:
    def __init__(self, use_real_model: bool = False, model_path: str = "./vetbert-finetuned"):
        self.use_real_model = use_real_model
        if use_real_model:
            self._load_bert(model_path)
        else:
            self._load_keyword_fallback()

    def _load_bert(self, model_path: str):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _load_keyword_fallback(self):
        # Existing weighted keyword matching (current code)
        ...

    def predict(self, text: str) -> dict:
        if self.use_real_model:
            return self._predict_bert(text)
        return self._predict_keywords(text)

    def _predict_bert(self, text: str) -> dict:
        import torch

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        sorted_indices = torch.argsort(probs, descending=True)
        all_probs = {
            CANCER_LABELS[i.item()]: round(probs[i.item()].item(), 4)
            for i in sorted_indices
        }
        predicted_idx = sorted_indices[0].item()

        return {
            "predicted_label": CANCER_LABELS[predicted_idx],
            "confidence": round(probs[predicted_idx].item(), 4),
            "all_probabilities": all_probs,
        }

    def _predict_keywords(self, text: str) -> dict:
        # ... existing weighted keyword code from current classifier.py ...
```

### 2.3 Configuration Additions

**`backend/app/config.py`** — add:

```python
USE_REAL_BERT: bool = False   # Toggle real BERT vs keyword fallback
BERT_MODEL_PATH: str = "/ml/model/weights/vetbert-finetuned"
CONFIDENCE_THRESHOLD: float = 0.7   # For flagging (Workstream 6)
REDIS_URL: str = "redis://redis:6379/0"
```

### 2.4 Async NLP Worker with Celery + Redis

**Architecture decision:** Use Celery + Redis for a decoupled worker that can process pathology reports without blocking the API, matching the architecture diagram's "separate NLP processing worker."

**New file: `backend/app/services/nlp_worker.py`**

```python
"""
Async NLP worker using Celery for background BERT classification.
Processes pathology reports queued by the upload pipeline.
"""

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings

celery_app = Celery("nlp_worker", broker=settings.REDIS_URL)

# Sync DB session (Celery tasks are synchronous)
sync_engine = create_engine(settings.DATABASE_URL_SYNC)
SyncSession = sessionmaker(bind=sync_engine)


@celery_app.task(name="classify_report")
def classify_report(report_id: int):
    """
    1. Load pathology_report by ID
    2. Run BERT classification
    3. Write classification + confidence_score back to DB
    4. Set review_status based on confidence threshold
    5. Update nlp_jobs status
    """
    from ml.model.classifier import VetBERTClassifier
    classifier = VetBERTClassifier(
        use_real_model=settings.USE_REAL_BERT,
        model_path=settings.BERT_MODEL_PATH,
    )

    with SyncSession() as db:
        report = db.query(PathologyReport).get(report_id)
        if not report:
            return

        result = classifier.predict(report.report_text)

        report.classification = result["predicted_label"]
        report.confidence_score = result["confidence"]
        report.review_status = (
            "auto_accepted" if result["confidence"] >= settings.CONFIDENCE_THRESHOLD
            else "flagged"
        )
        db.commit()


@celery_app.task(name="classify_batch")
def classify_batch(report_ids: list[int]):
    """Classify multiple reports in sequence."""
    for report_id in report_ids:
        classify_report(report_id)
```

### 2.5 Docker Compose Additions

Add to `docker-compose.yml`:

```yaml
  redis:
    image: redis:7-alpine
    container_name: vmth_cancer_redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  nlp_worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: vmth_cancer_nlp_worker
    environment:
      DATABASE_URL: postgresql+asyncpg://postgres:postgres@db:5432/vmth_cancer
      DATABASE_URL_SYNC: postgresql://postgres:postgres@db:5432/vmth_cancer
      REDIS_URL: redis://redis:6379/0
      USE_REAL_BERT: "false"
      BERT_MODEL_PATH: /ml/model/weights/vetbert-finetuned
    volumes:
      - ./backend:/app
      - ./ml:/ml
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: celery -A app.services.nlp_worker.celery_app worker --loglevel=info --concurrency=2
```

Also add `REDIS_URL` to the backend service environment.

### 2.6 Backend Requirements Additions

Add to `backend/requirements.txt`:

```
# NLP / ML
transformers>=4.38.0
torch>=2.2.0
# Task queue
celery>=5.3.0
redis>=5.0.0
```

### 2.7 NLP Jobs Table

**New migration: `database/migrations/011_nlp_jobs.sql`**

```sql
CREATE TABLE IF NOT EXISTS nlp_jobs (
    id SERIAL PRIMARY KEY,
    report_id INTEGER REFERENCES pathology_reports(id),
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    queued_at TIMESTAMP NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_nlp_jobs_status ON nlp_jobs (status);
CREATE INDEX IF NOT EXISTS idx_nlp_jobs_report ON nlp_jobs (report_id);
```

### 2.8 Integration with Upload Pipeline

When `ingestion_service.py` encounters a `pathology_notes` column:
1. Create a `pathology_reports` row with `classification=NULL`, `confidence_score=NULL`.
2. Create an `nlp_jobs` row with `status='pending'`.
3. Dispatch `classify_report.delay(report_id)` to the Celery queue.

The upload endpoint returns immediately; the NLP worker processes asynchronously.

### 2.9 Files Summary for Workstream 2

**Files to create:**
| File | Purpose |
|------|---------|
| `backend/app/services/nlp_worker.py` | Celery task definitions |
| `database/migrations/011_nlp_jobs.sql` | Job tracking table |

**Files to modify:**
| File | Change |
|------|--------|
| `ml/model/classifier.py` | Add real BERT mode, keep keyword fallback |
| `backend/app/services/bert_service.py` | Rewrite to import from ml/, use config toggle |
| `backend/app/routers/search.py` | Use updated bert_service |
| `backend/app/config.py` | Add `USE_REAL_BERT`, `BERT_MODEL_PATH`, `REDIS_URL`, `CONFIDENCE_THRESHOLD` |
| `backend/requirements.txt` | Add transformers, torch, celery, redis |
| `backend/Dockerfile` | Install ml dependencies |
| `docker-compose.yml` | Add redis + nlp_worker services, add env vars to backend |
| `backend/app/services/ingestion_service.py` | Queue NLP jobs after inserting reports |

---

## Workstream 3: Vet-ICD-O-canine-1 Coding System

**Gaps addressed:** #3 (US #1)

### 3.1 ICD-O Code Reference

The Vet-ICD-O-canine-1 system uses the same morphology code structure as human ICD-O-3 (format: `XXXX/B` where XXXX = morphology, B = behavior).

**Mapping for existing cancer types:**

| Cancer Type | ICD-O Morphology Code | ICD-O Label |
|---|---|---|
| Lymphoma | 9590/3 | Malignant lymphoma, NOS |
| Mast Cell Tumor | 9740/3 | Mast cell sarcoma |
| Osteosarcoma | 9180/3 | Osteosarcoma, NOS |
| Hemangiosarcoma | 9120/3 | Hemangiosarcoma |
| Melanoma | 8720/3 | Malignant melanoma, NOS |
| Squamous Cell Carcinoma | 8070/3 | Squamous cell carcinoma, NOS |
| Fibrosarcoma | 8810/3 | Fibrosarcoma, NOS |
| Transitional Cell Carcinoma | 8120/3 | Transitional cell carcinoma, NOS |

### 3.2 Database Migration: `database/migrations/008_icd_codes.sql`

```sql
-- 008_icd_codes.sql
-- Add Vet-ICD-O-canine-1 coding to cancer types

ALTER TABLE cancer_types
    ADD COLUMN IF NOT EXISTS icd_o_morphology_code VARCHAR(10),
    ADD COLUMN IF NOT EXISTS icd_o_topography_code VARCHAR(10),
    ADD COLUMN IF NOT EXISTS icd_o_label VARCHAR(200);

UPDATE cancer_types SET icd_o_morphology_code = '9590/3', icd_o_label = 'Malignant lymphoma, NOS' WHERE name = 'Lymphoma';
UPDATE cancer_types SET icd_o_morphology_code = '9740/3', icd_o_label = 'Mast cell sarcoma' WHERE name = 'Mast Cell Tumor';
UPDATE cancer_types SET icd_o_morphology_code = '9180/3', icd_o_label = 'Osteosarcoma, NOS' WHERE name = 'Osteosarcoma';
UPDATE cancer_types SET icd_o_morphology_code = '9120/3', icd_o_label = 'Hemangiosarcoma' WHERE name = 'Hemangiosarcoma';
UPDATE cancer_types SET icd_o_morphology_code = '8720/3', icd_o_label = 'Malignant melanoma, NOS' WHERE name = 'Melanoma';
UPDATE cancer_types SET icd_o_morphology_code = '8070/3', icd_o_label = 'Squamous cell carcinoma, NOS' WHERE name = 'Squamous Cell Carcinoma';
UPDATE cancer_types SET icd_o_morphology_code = '8810/3', icd_o_label = 'Fibrosarcoma, NOS' WHERE name = 'Fibrosarcoma';
UPDATE cancer_types SET icd_o_morphology_code = '8120/3', icd_o_label = 'Transitional cell carcinoma, NOS' WHERE name = 'Transitional Cell Carcinoma';
```

### 3.3 Backend Model Update

**`backend/app/models/models.py`** — add to `CancerType`:

```python
class CancerType(Base):
    __tablename__ = "cancer_types"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    icd_o_morphology_code = Column(String(10))      # NEW
    icd_o_topography_code = Column(String(10))       # NEW
    icd_o_label = Column(String(200))                # NEW

    cases = relationship("CancerCase", back_populates="cancer_type")
```

### 3.4 Schema Updates

**`backend/app/schemas/schemas.py`** — update `CancerTypeOut`:

```python
class CancerTypeOut(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    icd_o_morphology_code: Optional[str] = None    # NEW
    icd_o_topography_code: Optional[str] = None    # NEW
    icd_o_label: Optional[str] = None              # NEW
    model_config = {"from_attributes": True}
```

Also update `IncidenceRecord` to include `icd_o_code`:
```python
class IncidenceRecord(BaseModel):
    cancer_type: str
    icd_o_code: Optional[str] = None   # NEW
    county: Optional[str] = None
    species: Optional[str] = None
    breed: Optional[str] = None
    year: Optional[int] = None
    count: int
```

And update `ClassifyResult`:
```python
class ClassifyResult(BaseModel):
    predicted_cancer_type: str
    icd_o_code: Optional[str] = None   # NEW
    confidence: float
    top_predictions: List[dict]
```

### 3.5 Router Updates

**`backend/app/routers/incidence.py`** — add ICD-O code to the SELECT clause wherever `CancerType.name` is selected. Example for `/by-cancer-type`:

```python
# Before:
select(CancerType.name.label("cancer_type"), func.count(CancerCase.id).label("count"))

# After:
select(
    CancerType.name.label("cancer_type"),
    CancerType.icd_o_morphology_code.label("icd_o_code"),
    func.count(CancerCase.id).label("count"),
)
```

**`backend/app/routers/search.py`** — after classification, look up the ICD-O code:

```python
# After getting classification result:
if result.predicted_cancer_type != "Unknown":
    ct = await db.execute(
        select(CancerType.icd_o_morphology_code)
        .where(CancerType.name == result.predicted_cancer_type)
    )
    code = ct.scalar_one_or_none()
    result.icd_o_code = code
```

### 3.6 Frontend Display

**`frontend/src/types/index.ts`** — update `CANCER_TYPES` to include codes:

```typescript
export interface CancerTypeOption {
  name: string;
  icd_o_code?: string;
}
```

**`frontend/src/components/Filters/Filters.tsx`** — display ICD-O codes in dropdown:
```tsx
// Show: "Lymphoma (9590/3)" in the cancer type dropdown
```

**`frontend/src/components/CountyTable/CountyTable.tsx`** and **`SummaryTable/SummaryTable.tsx`** — add ICD-O code column where cancer types are displayed.

### 3.7 Files Summary for Workstream 3

**Files to create:**
| File | Purpose |
|------|---------|
| `database/migrations/008_icd_codes.sql` | Add ICD-O columns and data |

**Files to modify:**
| File | Change |
|------|--------|
| `backend/app/models/models.py` | Add 3 columns to CancerType |
| `backend/app/schemas/schemas.py` | Add ICD-O fields to CancerTypeOut, IncidenceRecord, ClassifyResult |
| `backend/app/routers/dashboard.py` | Include ICD-O in filter options response |
| `backend/app/routers/incidence.py` | Include ICD-O code in all incidence queries |
| `backend/app/routers/search.py` | Look up and return ICD-O code for predictions |
| `frontend/src/types/index.ts` | Add CancerTypeOption interface |
| `frontend/src/components/Filters/Filters.tsx` | Display ICD-O codes in dropdown |
| `frontend/src/components/CountyTable/CountyTable.tsx` | Add ICD-O column |
| `frontend/src/components/SummaryTable/SummaryTable.tsx` | Add ICD-O column |
| `docker-compose.yml` | Mount migration 008 |

---

## Workstream 4: Authentication & Access Control

**Gaps addressed:** #4 (Security requirement)

### 4.1 Database Migration: `database/migrations/009_users.sql`

```sql
-- 009_users.sql
-- User accounts for authentication

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'researcher'
        CHECK (role IN ('admin', 'researcher', 'viewer')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Seed a default admin account (password: 'changeme' — MUST be changed on first login)
-- The hashed_password below is bcrypt hash of 'changeme'
-- In production, this should be set via environment variable or initial setup script
```

### 4.2 Backend — User Model

Add to `backend/app/models/models.py`:

```python
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), nullable=False, unique=True)
    email = Column(String(255), nullable=False, unique=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="researcher")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
```

### 4.3 Auth Service: `backend/app/services/auth_service.py`

**Architecture:**

```python
"""
JWT-based authentication service.
Uses python-jose for JWT tokens and passlib for password hashing.
"""

from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
from app.database import get_db
from app.models.models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def verify_password(plain: str, hashed: str) -> bool: ...
def hash_password(password: str) -> str: ...
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str: ...

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """FastAPI dependency — extract and validate JWT, return User object.
    Raise 401 if token is invalid or user not found."""

async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency — require admin role."""
```

**JWT token payload:**
```json
{
  "sub": "username",
  "role": "researcher",
  "exp": 1707400000
}
```

### 4.4 Auth Router: `backend/app/routers/auth.py`

**Endpoints:**

#### `POST /api/v1/auth/login`

```
Request (form data, per OAuth2 spec):
  username: string
  password: string

Response 200:
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}

Response 401:
{
  "detail": "Incorrect username or password"
}
```

#### `POST /api/v1/auth/register` (admin-only)

```
Request:
{
  "username": "jdoe",
  "email": "jdoe@ucdavis.edu",
  "password": "securepassword123",
  "role": "researcher"
}

Response 201:
{
  "id": 2,
  "username": "jdoe",
  "email": "jdoe@ucdavis.edu",
  "role": "researcher"
}

Response 403:
{
  "detail": "Only administrators can register new users"
}
```

#### `GET /api/v1/auth/me`

```
Headers: Authorization: Bearer <token>

Response 200:
{
  "id": 1,
  "username": "admin",
  "email": "admin@ucdavis.edu",
  "role": "admin"
}
```

### 4.5 Protecting Existing Endpoints

Apply `Depends(get_current_user)` to these routes:

| Router | Endpoint | Protection |
|--------|----------|------------|
| `upload` | `POST /csv`, `POST /text`, `GET /history` | `get_current_user` |
| `review` | `GET /queue`, `PUT /{id}`, `GET /stats` | `get_current_user` |
| `auth` | `POST /register` | `require_admin` |
| All others | Dashboard, incidence, geo, trends, search | **Public** (read-only visualization) |

### 4.6 Configuration Additions

**`backend/app/config.py`** — add:

```python
SECRET_KEY: str = "CHANGE-ME-IN-PRODUCTION-use-openssl-rand-hex-32"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 480   # 8 hours
JWT_ALGORITHM: str = "HS256"
```

**`docker-compose.yml`** — add to backend environment:
```yaml
SECRET_KEY: "${SECRET_KEY:-dev-secret-key-change-in-production}"
```

### 4.7 Backend Requirements Additions

Add to `backend/requirements.txt`:

```
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
```

### 4.8 Frontend — Auth Context: `frontend/src/hooks/useAuth.ts`

```typescript
interface AuthContextType {
  user: UserInfo | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

// Store token in React state (memory), NOT localStorage
// This means token is lost on page refresh — acceptable for security
// Alternative: use httpOnly cookies (requires backend cookie support)
```

### 4.9 Frontend — Login Page: `frontend/src/components/LoginPage/LoginPage.tsx`

```
┌──────────────────────────────────────┐
│                                      │
│   UC Davis VMTH Cancer Registry      │
│                                      │
│   ┌──────────────────────────────┐   │
│   │  Username                    │   │
│   │  ┌────────────────────────┐  │   │
│   │  │                        │  │   │
│   │  └────────────────────────┘  │   │
│   │  Password                    │   │
│   │  ┌────────────────────────┐  │   │
│   │  │                        │  │   │
│   │  └────────────────────────┘  │   │
│   │                              │   │
│   │  [Sign In]                   │   │
│   │                              │   │
│   │  ⚠ Invalid credentials      │   │
│   └──────────────────────────────┘   │
│                                      │
└──────────────────────────────────────┘
```

### 4.10 Frontend — API Client Auth Integration

**`frontend/src/api/client.ts`** — modify `fetchJson`:

```typescript
let authToken: string | null = null;

export function setAuthToken(token: string | null) {
  authToken = token;
}

export function getAuthHeaders(): Record<string, string> {
  return authToken ? { 'Authorization': `Bearer ${authToken}` } : {};
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url, {
    headers: getAuthHeaders(),
  });
  if (response.status === 401) {
    // Trigger logout via event or callback
    throw new Error('Unauthorized');
  }
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json();
}

export async function login(username: string, password: string): Promise<{ access_token: string }> {
  const formData = new URLSearchParams();
  formData.append('username', username);
  formData.append('password', password);

  const response = await fetch('/api/v1/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: formData,
  });
  if (!response.ok) throw new Error('Invalid credentials');
  return response.json();
}
```

### 4.11 Navigation — Conditional Tabs

**`frontend/src/components/Navigation/Navigation.tsx`:**

The Upload and Review tabs should only appear when the user is authenticated. Pass `isAuthenticated` as a prop and conditionally render those tabs.

```tsx
interface NavigationProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  isAuthenticated: boolean;
  onLoginClick: () => void;
}

// Filter TABS to exclude 'upload' and 'review' when not authenticated
// Add a "Sign In" / "Sign Out" button in the top-right corner of the banner
```

### 4.12 Files Summary for Workstream 4

**Files to create:**
| File | Purpose |
|------|---------|
| `database/migrations/009_users.sql` | Users table |
| `backend/app/routers/auth.py` | Auth endpoints |
| `backend/app/services/auth_service.py` | JWT + password logic |
| `frontend/src/components/LoginPage/LoginPage.tsx` | Login form |
| `frontend/src/hooks/useAuth.ts` | Auth React context |

**Files to modify:**
| File | Change |
|------|--------|
| `backend/app/main.py` | Register auth router |
| `backend/app/models/models.py` | Add User model |
| `backend/app/schemas/schemas.py` | Add User/Token schemas |
| `backend/app/config.py` | Add SECRET_KEY, JWT settings |
| `backend/requirements.txt` | Add python-jose, passlib |
| `backend/app/routers/upload.py` | Add `Depends(get_current_user)` |
| `frontend/src/api/client.ts` | Add auth headers, login function |
| `frontend/src/App.tsx` | Wrap in AuthProvider, render LoginPage |
| `frontend/src/components/Navigation/Navigation.tsx` | Conditional tabs, sign in/out |
| `frontend/src/components/index.ts` | Export LoginPage |
| `docker-compose.yml` | Mount migration 009, add SECRET_KEY env |

---

## Workstream 5: Trend Line Visualization

**Gaps addressed:** #5 (US #6)

### 5.1 Install Charting Library

Add `recharts` to `frontend/package.json`:

```bash
npm install recharts
```

`recharts` is chosen because:
- Pure React components (no D3 DOM manipulation conflicts)
- Built-in responsive container
- Lightweight (~40KB gzipped)
- Good TypeScript support

### 5.2 Frontend — TrendChart Component

**File:** `frontend/src/components/TrendChart/TrendChart.tsx`

**Component design:**

```
┌──────────────────────────────────────────────────┐
│  Cancer Case Trends Over Time                     │
│                                                    │
│  [All Cases]  [By Cancer Type]  (toggle buttons)   │
│                                                    │
│  Count                                             │
│  ▲                                                 │
│  │     ╱╲                                          │
│  │    ╱  ╲    ╱╲                                   │
│  │   ╱    ╲  ╱  ╲  ╱──╲                           │
│  │  ╱      ╲╱    ╲╱    ╲                           │
│  │ ╱                     ╲                          │
│  └────────────────────────────────▶ Year           │
│   1995  2000  2005  2010  2015  2020  2025         │
│                                                    │
│  Legend: ── All Cases  ── Lymphoma  ── MCT          │
│                                                    │
│  Tooltip: Year 2020: 342 cases                     │
└──────────────────────────────────────────────────┘
```

**Props:**

```typescript
interface TrendChartProps {
  filters: FilterState;  // Current filter state to pass to API
  mode?: 'overview' | 'detailed';  // overview shows single line, detailed shows by cancer type
}
```

**Data fetching:**

```typescript
// Uses existing backend endpoints:
// GET /api/v1/trends/yearly         → single "All Cases" line
// GET /api/v1/trends/by-cancer-type → one line per cancer type

// These endpoints already support filter params:
//   species[], cancer_type[], county[], sex
```

**Recharts implementation outline:**

```tsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Transform API response (TrendsResponse) into recharts format:
// [{ year: 2020, "All Cases": 342, "Lymphoma": 45, "MCT": 38, ... }]

// Color palette matching UC Davis branding:
const CANCER_COLORS: Record<string, string> = {
  'All Cases': '#022851',        // UC Davis blue
  'Lymphoma': '#B4D3B2',
  'Mast Cell Tumor': '#F2A900',  // UC Davis gold
  'Osteosarcoma': '#6B7280',
  'Hemangiosarcoma': '#DC2626',
  'Melanoma': '#7C3AED',
  'Squamous Cell Carcinoma': '#059669',
  'Fibrosarcoma': '#D97706',
  'Transitional Cell Carcinoma': '#2563EB',
};
```

### 5.3 API Client Additions

Add to `frontend/src/api/client.ts`:

```typescript
export interface TrendPoint {
  year: number;
  count: number;
  deceased?: number;
  alive?: number;
}

export interface TrendSeries {
  name: string;
  data: TrendPoint[];
}

export interface TrendsResponse {
  series: TrendSeries[];
}

export async function fetchYearlyTrends(filters: FilterParams = {}): Promise<TrendsResponse> {
  const params = filtersToParams(filters);
  const url = params.toString() ? `/api/v1/trends/yearly?${params}` : '/api/v1/trends/yearly';
  return fetchJson(url);
}

export async function fetchTrendsByCancerType(filters: FilterParams = {}): Promise<TrendsResponse> {
  const params = filtersToParams(filters);
  const url = params.toString() ? `/api/v1/trends/by-cancer-type?${params}` : '/api/v1/trends/by-cancer-type';
  return fetchJson(url);
}
```

### 5.4 Integration into App

1. **Overview tab:** Add `<TrendChart mode="overview" />` below the map/tables grid.
2. **New "Trends" tab:** Add dedicated `TabType = 'trends'` with `<TrendChart mode="detailed" />` showing by-cancer-type multi-series view.

### 5.5 Files Summary for Workstream 5

**Files to create:**
| File | Purpose |
|------|---------|
| `frontend/src/components/TrendChart/TrendChart.tsx` | Recharts line chart |

**Files to modify:**
| File | Change |
|------|--------|
| `frontend/package.json` | Add `recharts` |
| `frontend/src/api/client.ts` | Add `fetchYearlyTrends`, `fetchTrendsByCancerType`, trend types |
| `frontend/src/types/index.ts` | Add `TrendPoint`, `TrendSeries`, `TrendsResponse`, extend TabType with 'trends' |
| `frontend/src/App.tsx` | Add TrendChart to overview + trends tab |
| `frontend/src/components/index.ts` | Export TrendChart |

---

## Workstream 6: Ambiguous Diagnosis Flagging & Review

**Gaps addressed:** #6 (US #8)

### 6.1 Database Migration: `database/migrations/010_review_status.sql`

```sql
-- 010_review_status.sql
-- Add review workflow to pathology reports

ALTER TABLE pathology_reports
    ADD COLUMN IF NOT EXISTS review_status VARCHAR(20)
        DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'auto_accepted', 'flagged', 'manually_reviewed', 'rejected')),
    ADD COLUMN IF NOT EXISTS reviewed_by INTEGER REFERENCES users(id),
    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP;

-- Set existing reports with classification to auto_accepted
UPDATE pathology_reports SET review_status = 'auto_accepted' WHERE classification IS NOT NULL;
UPDATE pathology_reports SET review_status = 'pending' WHERE classification IS NULL;

CREATE INDEX IF NOT EXISTS idx_reports_review_status ON pathology_reports (review_status);
```

### 6.2 Backend Model Update

**`backend/app/models/models.py`** — update `PathologyReport`:

```python
class PathologyReport(Base):
    __tablename__ = "pathology_reports"

    id = Column(Integer, primary_key=True)
    case_id = Column(Integer, ForeignKey("cancer_cases.id"), nullable=False)
    report_text = Column(Text, nullable=False)
    classification = Column(String(100))
    confidence_score = Column(Numeric(5, 4))
    report_date = Column(Date, nullable=False)
    review_status = Column(String(20), default="pending")       # NEW
    reviewed_by = Column(Integer, ForeignKey("users.id"))       # NEW
    reviewed_at = Column(DateTime)                              # NEW

    case = relationship("CancerCase", back_populates="reports")
    reviewer = relationship("User")                             # NEW
```

### 6.3 Flagging Logic

The confidence threshold is configured in `backend/app/config.py` as `CONFIDENCE_THRESHOLD = 0.7`.

**In `backend/app/services/nlp_worker.py` (Workstream 2) or `bert_service.py`:**

```python
def determine_review_status(confidence: float) -> str:
    if confidence >= settings.CONFIDENCE_THRESHOLD:
        return "auto_accepted"
    return "flagged"
```

### 6.4 Review Router: `backend/app/routers/review.py`

**Endpoints:**

#### `GET /api/v1/review/queue`

```
Query params: ?status=flagged&limit=20&offset=0&cancer_type=Lymphoma&min_confidence=0.3&max_confidence=0.7

Response 200:
{
  "reports": [
    {
      "id": 456,
      "case_id": 123,
      "report_text": "Histopathology reveals...",
      "classification": "Lymphoma",
      "confidence_score": 0.55,
      "report_date": "2024-03-15",
      "review_status": "flagged",
      "patient_info": {            // Join data for context
        "species": "Dog",
        "breed": "Golden Retriever",
        "age_years": 8.5,
        "county": "Sacramento"
      }
    }
  ],
  "total": 42,
  "stats": {
    "flagged": 42,
    "pending": 5,
    "auto_accepted": 893,
    "manually_reviewed": 31,
    "rejected": 3
  }
}
```

#### `PUT /api/v1/review/{report_id}`

```
Request:
{
  "action": "approve",                    // approve | reclassify | reject
  "new_classification": "Mast Cell Tumor" // required if action=reclassify
}

Response 200:
{
  "id": 456,
  "review_status": "manually_reviewed",
  "classification": "Mast Cell Tumor",
  "reviewed_by": "admin",
  "reviewed_at": "2026-02-08T15:30:00Z"
}
```

- `approve`: Keep the BERT classification, set `review_status = 'manually_reviewed'`
- `reclassify`: Override classification with `new_classification`, set `review_status = 'manually_reviewed'`
- `reject`: Set `review_status = 'rejected'`, mark case as needing further investigation

#### `GET /api/v1/review/stats`

```
Response 200:
{
  "pending": 5,
  "flagged": 42,
  "auto_accepted": 893,
  "manually_reviewed": 31,
  "rejected": 3,
  "total": 974,
  "average_confidence": 0.78
}
```

### 6.5 Frontend — Review Queue Component

**File:** `frontend/src/components/ReviewQueue/ReviewQueue.tsx`

```
┌────────────────────────────────────────────────────────────────┐
│  Review Queue                                  Stats: 42 flagged│
│                                                                 │
│  Filters: [Confidence: 0.3-0.7] [Cancer Type: All] [Search...] │
│                                                                 │
│  ┌─────┬──────────────────┬──────────┬──────┬─────────────────┐│
│  │  ID │ Report Excerpt    │ Predicted│ Conf │ Actions         ││
│  ├─────┼──────────────────┼──────────┼──────┼─────────────────┤│
│  │ 456 │ Histopathology   │ Lymphoma │ 55%  │ [✓] [✎] [✗]    ││
│  │     │ reveals diffuse  │          │      │                 ││
│  │     │ large B-cell...  │          │      │                 ││
│  ├─────┼──────────────────┼──────────┼──────┼─────────────────┤│
│  │ 457 │ Excisional biopsy│ MCT      │ 48%  │ [✓] [✎] [✗]    ││
│  │     │ reveals dermal...│          │      │                 ││
│  └─────┴──────────────────┴──────────┴──────┴─────────────────┘│
│                                                                 │
│  Showing 1-20 of 42  [< Prev] [Next >]                         │
│                                                                 │
│  Expanded report view (on row click):                           │
│  ┌──────────────────────────────────────────────────┐          │
│  │ Full report text...                               │          │
│  │ Patient: Dog, Golden Retriever, 8.5y, Sacramento  │          │
│  │ BERT prediction: Lymphoma (55%)                   │          │
│  │ Alternatives: MCT (20%), Hemangiosarcoma (15%)    │          │
│  │                                                    │          │
│  │ Reclassify as: [dropdown of cancer types]         │          │
│  │ [Approve] [Reclassify] [Reject]                   │          │
│  └──────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────┘
```

### 6.6 Files Summary for Workstream 6

**Files to create:**
| File | Purpose |
|------|---------|
| `database/migrations/010_review_status.sql` | Review workflow columns |
| `backend/app/routers/review.py` | Review queue endpoints |
| `frontend/src/components/ReviewQueue/ReviewQueue.tsx` | Review UI |

**Files to modify:**
| File | Change |
|------|--------|
| `backend/app/models/models.py` | Add review columns to PathologyReport |
| `backend/app/schemas/schemas.py` | Add Review* schemas, update ReportOut |
| `backend/app/services/bert_service.py` | Add `determine_review_status()` |
| `backend/app/config.py` | `CONFIDENCE_THRESHOLD` (already added in WS2) |
| `backend/app/main.py` | Register review router |
| `frontend/src/api/client.ts` | Add `fetchReviewQueue`, `updateReview`, `fetchReviewStats` |
| `frontend/src/types/index.ts` | Add ReviewItem, ReviewQueueResponse types |
| `frontend/src/components/Navigation/Navigation.tsx` | Add Review tab (auth-only) |
| `frontend/src/components/index.ts` | Export ReviewQueue |
| `frontend/src/App.tsx` | Render ReviewQueue for review tab |
| `docker-compose.yml` | Mount migration 010 |

---

## Workstream 7: Fix Frontend Tabs (Real Data)

**Gaps addressed:** #7 (US #11)

### 7.1 Current Problem

In `frontend/src/App.tsx` lines 65-153, three tabs render fake data:

- **`breed-disparities`** (lines 65-86): Hardcoded 4 breeds with `Math.random() * 50 + 30`
- **`cancer-types`** (lines 88-116): Hardcoded 6 cancer types with `50 - i * 6 + Math.random() * 5`
- **`regional-comparison`** (lines 118-153): Hardcoded 5 regions with static values

### 7.2 BreedDisparities Component

**File:** `frontend/src/components/BreedDisparities/BreedDisparities.tsx`

**Data source:** `GET /api/v1/incidence/by-breed` (already exists, `fetchIncidenceByBreed` already defined in client.ts)

**Implementation:**
```typescript
interface BreedDisparitiesProps {
  filters: FilterState;
}

// 1. Call fetchIncidenceByBreed with current filters
// 2. Render a horizontal bar chart (can use recharts BarChart or the existing
//    CSS bar style from the current cancer-types tab)
// 3. Show: breed name, case count, rate per 10k (if population data available)
// 4. Sort by count descending
```

### 7.3 CancerTypesChart Component

**File:** `frontend/src/components/CancerTypesChart/CancerTypesChart.tsx`

**Data source:** `GET /api/v1/incidence/by-cancer-type` (already exists, `fetchIncidenceByCancerType` already defined in client.ts)

**Implementation:**
```typescript
interface CancerTypesChartProps {
  filters: FilterState;
}

// 1. Call fetchIncidenceByCancerType with current filters
// 2. Render horizontal bars (keep existing visual style)
// 3. Show ICD-O code next to cancer type name (after Workstream 3)
// 4. Replace the Math.random() values with actual counts
```

### 7.4 RegionalComparison Component

**File:** `frontend/src/components/RegionalComparison/RegionalComparison.tsx`

**Data source:** The `regionSummary` object already computed in `useFilteredData.ts` contains real region-level aggregations.

**Implementation:**
```typescript
interface RegionalComparisonProps {
  regionSummary: RegionSummary;
  filters: FilterState;
}

// 1. Use the regionSummary data (already contains real counts + rates per region)
// 2. Fetch trend data (GET /api/v1/trends/yearly with county filter per region)
//    to determine actual trend direction (up/down/stable)
// 3. Replace hardcoded regions with actual regions from the data
// 4. Calculate trend: compare last 2 years of data, if increasing → "up", etc.
```

**Trend direction calculation:**
```typescript
function getTrendDirection(trendData: TrendPoint[]): 'up' | 'down' | 'stable' {
  if (trendData.length < 2) return 'stable';
  const recent = trendData.slice(-3);  // Last 3 years
  const earlier = trendData.slice(-6, -3);  // 3 years before that
  const recentAvg = recent.reduce((s, p) => s + p.count, 0) / recent.length;
  const earlierAvg = earlier.reduce((s, p) => s + p.count, 0) / earlier.length;
  const change = (recentAvg - earlierAvg) / earlierAvg;
  if (change > 0.05) return 'up';
  if (change < -0.05) return 'down';
  return 'stable';
}
```

### 7.5 Refactor App.tsx

Remove the inline hardcoded tab content from `App.tsx` and replace with component imports:

```tsx
// Before (App.tsx lines 65-153): inline JSX with Math.random()
// After:
{activeTab === 'breed-disparities' && <BreedDisparities filters={filters} />}
{activeTab === 'cancer-types' && <CancerTypesChart filters={filters} />}
{activeTab === 'regional-comparison' && (
  <RegionalComparison regionSummary={regionSummary} filters={filters} />
)}
```

### 7.6 Files Summary for Workstream 7

**Files to create:**
| File | Purpose |
|------|---------|
| `frontend/src/components/BreedDisparities/BreedDisparities.tsx` | Real breed data |
| `frontend/src/components/CancerTypesChart/CancerTypesChart.tsx` | Real cancer type data |
| `frontend/src/components/RegionalComparison/RegionalComparison.tsx` | Real regional data |

**Files to modify:**
| File | Change |
|------|--------|
| `frontend/src/App.tsx` | Replace inline hardcoded tabs with component imports |
| `frontend/src/api/client.ts` | Wire up existing `fetchIncidenceByBreed`, `fetchIncidenceByCancerType` |
| `frontend/src/components/index.ts` | Export new components |

---

## Workstream 8: Tests

**Gaps addressed:** #9 (Maintainability requirement)

### 8.1 Backend Test Architecture

**Directory structure:**
```
backend/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── test_dashboard.py        # Dashboard endpoint tests
│   ├── test_incidence.py        # Incidence endpoint tests
│   ├── test_trends.py           # Trends endpoint tests
│   ├── test_geo.py              # Geo endpoint tests
│   ├── test_search.py           # Search/classify endpoint tests
│   ├── test_upload.py           # CSV upload tests
│   ├── test_auth.py             # Authentication tests
│   ├── test_review.py           # Review queue tests
│   └── test_ingestion.py        # Ingestion service unit tests
```

**`conftest.py` architecture:**

```python
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.main import app
from app.database import get_db, Base

# Use a test database
TEST_DB_URL = "postgresql+asyncpg://postgres:postgres@db:5432/vmth_cancer_test"

@pytest_asyncio.fixture
async def db_session():
    """Create a fresh test database session."""
    engine = create_async_engine(TEST_DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession)
    async with session_factory() as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture
async def client(db_session):
    """Create test HTTP client with DB override."""
    app.dependency_overrides[get_db] = lambda: db_session
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()

@pytest.fixture
def sample_csv():
    """Generate a valid test CSV file."""
    return b"species,breed,sex,age_years,county,registered_date,cancer_type,diagnosis_date\n" \
           b"Dog,Golden Retriever,Male,7.5,Sacramento,2024-01-15,Lymphoma,2024-02-01\n"
```

**Key test cases per file:**

| File | Test Cases |
|------|-----------|
| `test_dashboard.py` | Summary returns correct totals, filters return valid options, empty DB returns zeros |
| `test_incidence.py` | Incidence with no filters, with species filter, with date range, by-cancer-type grouping |
| `test_trends.py` | Yearly aggregation correct, by-cancer-type returns multiple series, filter narrows results |
| `test_geo.py` | GeoJSON has valid structure, counties have geometry, filter reduces case counts |
| `test_search.py` | Classify returns valid cancer type, empty text returns 400, confidence between 0-1 |
| `test_upload.py` | Valid CSV accepted, missing columns return 422, invalid breed rejected with error, file type validation |
| `test_auth.py` | Login returns token, invalid password returns 401, protected endpoint returns 403 without token |
| `test_review.py` | Queue returns flagged reports, approve updates status, reclassify changes classification |
| `test_ingestion.py` | Sex aliases mapped correctly, case-insensitive breed matching, date parsing |

**Dependencies to add to `backend/requirements.txt`:**
```
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

### 8.2 Frontend Test Architecture

**Directory structure:**
```
frontend/
├── src/
│   ├── components/
│   │   └── __tests__/
│   │       ├── Filters.test.tsx
│   │       ├── ChoroplethMap.test.tsx
│   │       ├── TrendChart.test.tsx
│   │       ├── UploadPage.test.tsx
│   │       └── ReviewQueue.test.tsx
│   ├── hooks/
│   │   └── __tests__/
│   │       ├── useFilteredData.test.ts
│   │       └── useAuth.test.ts
│   └── api/
│       └── __tests__/
│           └── client.test.ts
```

**Dependencies to add to `frontend/package.json` (devDependencies):**
```json
"vitest": "^3.0.0",
"@testing-library/react": "^16.0.0",
"@testing-library/jest-dom": "^6.0.0",
"jsdom": "^25.0.0"
```

**Add test script to `frontend/package.json`:**
```json
"test": "vitest",
"test:coverage": "vitest --coverage"
```

**Vitest configuration in `frontend/vite.config.ts`:**
```typescript
export default defineConfig({
  // ... existing config ...
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test-setup.ts',
  },
});
```

### 8.3 Files Summary for Workstream 8

**Files to create:**
| File | Purpose |
|------|---------|
| `backend/tests/__init__.py` | Package init |
| `backend/tests/conftest.py` | Shared fixtures |
| `backend/tests/test_dashboard.py` | Dashboard tests |
| `backend/tests/test_incidence.py` | Incidence tests |
| `backend/tests/test_trends.py` | Trends tests |
| `backend/tests/test_geo.py` | Geo tests |
| `backend/tests/test_search.py` | Search tests |
| `backend/tests/test_upload.py` | Upload tests |
| `backend/tests/test_auth.py` | Auth tests |
| `backend/tests/test_review.py` | Review tests |
| `backend/tests/test_ingestion.py` | Ingestion unit tests |
| `frontend/src/test-setup.ts` | Test environment setup |
| `frontend/src/components/__tests__/*.test.tsx` | Component tests |
| `frontend/src/hooks/__tests__/*.test.ts` | Hook tests |
| `frontend/src/api/__tests__/client.test.ts` | API client tests |

**Files to modify:**
| File | Change |
|------|--------|
| `backend/requirements.txt` | Add pytest, pytest-asyncio |
| `frontend/package.json` | Add vitest, testing-library, jsdom |
| `frontend/vite.config.ts` | Add test configuration |

---

## Implementation Order

The workstreams have dependencies. Here is the recommended execution order with rationale:

```
Phase 1 — Foundation (no cross-dependencies, can be parallelized)
│
├── Workstream 7: Fix frontend tabs (real data)
│   Rationale: Quick win. Only frontend changes. No backend work.
│   Effort: Small (3 new components, refactor App.tsx)
│   Can be done by: Frontend developer
│
├── Workstream 3: Vet-ICD-O-canine-1 codes
│   Rationale: Database schema change — do early before other migrations.
│   Effort: Small-Medium (1 migration, model + schema + router updates)
│   Can be done by: Backend developer
│
└── Workstream 5: Trend line visualization
    Rationale: No backend changes needed (endpoints exist). Frontend only.
    Effort: Medium (install recharts, build TrendChart, wire to API)
    Can be done by: Frontend developer


Phase 2 — Core Features (build on Phase 1)
│
├── Workstream 4: Authentication
│   Rationale: Must exist before upload & review (they require auth).
│   Effort: Medium (JWT service, auth router, login page, auth context)
│   Depends on: Nothing (but needed by WS1 and WS6)
│
├── Workstream 1: CSV upload & data ingestion
│   Rationale: Core feature — data entry into the registry.
│   Effort: Large (ingestion service, upload router, upload page, validation)
│   Depends on: WS4 for auth on upload endpoints
│
└── Workstream 2: Real BERT integration
    Rationale: Needed for free-text processing in upload pipeline + review.
    Effort: Large (model integration, Celery worker, Redis, Docker changes)
    Depends on: Nothing directly, but integrates with WS1


Phase 3 — Advanced Features (depend on Phase 2)
│
└── Workstream 6: Ambiguous diagnosis flagging & review
    Rationale: Requires BERT (for confidence scores) + auth (for reviewers).
    Effort: Medium (review router, flagging logic, review queue UI)
    Depends on: WS2 (BERT confidence), WS4 (authenticated reviewers)


Phase 4 — Quality
│
└── Workstream 8: Tests
    Rationale: Test all implemented features. Write tests after code is stable.
    Effort: Medium-Large (backend + frontend test suites)
    Depends on: All other workstreams (to test their functionality)
```

**Gantt-style timeline (if 2 developers working in parallel):**

```
                   Week 1        Week 2        Week 3        Week 4        Week 5
Dev A (Frontend):  [WS7: Tabs]   [WS5: Trends] [WS1.5: Upload UI] [WS6.5: Review UI] [WS8: FE Tests]
Dev B (Backend):   [WS3: ICD-O]  [WS4: Auth]   [WS1: Upload API]  [WS2: BERT+Worker]  [WS6: Review]
                                                                     [WS8: BE Tests]
```

---

## File Change Summary

### New Files to Create (26+ files)

| File | Workstream | Layer |
|------|------------|-------|
| `database/migrations/007_raw_uploads.sql` | 1 | DB |
| `database/migrations/008_icd_codes.sql` | 3 | DB |
| `database/migrations/009_users.sql` | 4 | DB |
| `database/migrations/010_review_status.sql` | 6 | DB |
| `database/migrations/011_nlp_jobs.sql` | 2 | DB |
| `backend/app/routers/upload.py` | 1 | Backend |
| `backend/app/routers/auth.py` | 4 | Backend |
| `backend/app/routers/review.py` | 6 | Backend |
| `backend/app/services/ingestion_service.py` | 1 | Backend |
| `backend/app/services/auth_service.py` | 4 | Backend |
| `backend/app/services/nlp_worker.py` | 2 | Backend |
| `backend/tests/` (11 test files) | 8 | Backend |
| `frontend/src/components/UploadPage/UploadPage.tsx` | 1 | Frontend |
| `frontend/src/components/LoginPage/LoginPage.tsx` | 4 | Frontend |
| `frontend/src/components/TrendChart/TrendChart.tsx` | 5 | Frontend |
| `frontend/src/components/ReviewQueue/ReviewQueue.tsx` | 6 | Frontend |
| `frontend/src/components/BreedDisparities/BreedDisparities.tsx` | 7 | Frontend |
| `frontend/src/components/CancerTypesChart/CancerTypesChart.tsx` | 7 | Frontend |
| `frontend/src/components/RegionalComparison/RegionalComparison.tsx` | 7 | Frontend |
| `frontend/src/hooks/useAuth.ts` | 4 | Frontend |
| `frontend/src/components/__tests__/` (5+ test files) | 8 | Frontend |

### Existing Files to Modify (18 files)

| File | Workstreams | Changes |
|------|-------------|---------|
| `backend/app/main.py` | 1, 4, 6 | Register upload, auth, review routers |
| `backend/app/config.py` | 2, 4, 6 | Add USE_REAL_BERT, SECRET_KEY, REDIS_URL, CONFIDENCE_THRESHOLD |
| `backend/app/models/models.py` | 3, 4, 6 | Add ICD-O cols to CancerType, add User model, add review cols to PathologyReport |
| `backend/app/schemas/schemas.py` | 1, 3, 4, 6 | Add Upload*, ICD-O, User/Token, Review* schemas |
| `backend/app/services/bert_service.py` | 2, 6 | Rewrite to use ml/ module, add flagging |
| `backend/app/routers/dashboard.py` | 3 | Include ICD-O codes in filter options |
| `backend/app/routers/incidence.py` | 3 | Include ICD-O code in incidence queries |
| `backend/app/routers/search.py` | 2, 3 | Use updated bert_service, return ICD-O code |
| `backend/requirements.txt` | 2, 4, 8 | Add transformers, torch, celery, redis, jose, passlib, pytest |
| `backend/Dockerfile` | 2 | Install ml/ dependencies |
| `ml/model/classifier.py` | 2 | Add real BERT mode alongside keyword fallback |
| `docker-compose.yml` | 1, 2, 3, 4, 6 | Mount new migrations, add redis + nlp_worker services, add env vars |
| `frontend/package.json` | 5, 8 | Add recharts, vitest, testing-library |
| `frontend/vite.config.ts` | 8 | Add test configuration |
| `frontend/src/App.tsx` | 1, 4, 5, 6, 7 | AuthProvider wrapper, replace hardcoded tabs, add TrendChart + new tabs |
| `frontend/src/api/client.ts` | 1, 4, 5, 6, 7 | Auth headers, upload functions, trends functions, review functions |
| `frontend/src/types/index.ts` | 1, 3, 4, 5, 6, 7 | Extend TabType, add Upload/Trend/Review/Auth types, CancerTypeOption |
| `frontend/src/components/Navigation/Navigation.tsx` | 1, 4, 6 | Conditional auth tabs, sign in/out button |
| `frontend/src/components/index.ts` | 1, 4, 5, 6, 7 | Export all new components |
