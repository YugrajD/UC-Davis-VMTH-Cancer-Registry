# Workstream 1: CSV Upload & Data Ingestion Pipeline

[Back to Overview](../IMPLEMENTATION_PLAN.md)

---

**Gaps addressed:** #1 (US #2, #9)

## 1.1 Database — New Migration: `database/migrations/007_raw_uploads.sql`

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

## 1.2 Backend — Ingestion Service: `backend/app/services/ingestion_service.py`

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

## 1.3 Backend — Upload Router: `backend/app/routers/upload.py`

**Endpoints:**

### `POST /api/v1/upload/csv`

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

### `POST /api/v1/upload/text`

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

### `GET /api/v1/upload/history`

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

### `GET /api/v1/upload/status/{job_id}`

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

## 1.4 Backend — Pydantic Schemas

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

## 1.5 Frontend — Upload Page Component

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
│  │  │  Drag & drop CSV here               │     ││
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
│  │  143 rows accepted                            ││
│  │  7 rows rejected                              ││
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

## 1.6 Navigation & App Integration

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

## 1.7 Files Summary

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
