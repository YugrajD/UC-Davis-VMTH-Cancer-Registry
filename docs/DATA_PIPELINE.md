# UC Davis VMTH Cancer Registry — Data Pipeline

## Overview

The registry ingests PetBERT cancer predictions and patient demographics to populate a unified PostgreSQL database for visualizing canine cancer incidence across California. The model is **one patient per dog**: a single `patients` row with multiple cancer predictions stored as rows in `case_diagnoses`.

There is **no `cancer_cases` table** — it was removed in an earlier refactor. `diagnosis_date` and `outcome` live directly on `patients`. All cancer type data is in `case_diagnoses`.

---

## Input Files

### Dataset A — Demographics

A CSV with one row per patient visit (may have duplicate `case_id` rows; the parser takes the first non-empty value per field).

**Accepted column names (two conventions supported):**

| Field | Long-form column name | Short-form column name |
|---|---|---|
| Patient ID | `anon_id` | `case_id` |
| Request date | `DtOfRq` | `DtOfRq` |
| Sex | `Sex` | `Sex` |
| Species | `Species` | `Species` |
| Breed | `Breed` | `Breed` |
| Primary zip | `Zipcode Zipcode` | `Zipcode` |
| Referral zip | `RfrrVtrn Zipcode Zipcode` | `RfrrVtrnZipcode` |

Sex codes: `M` → Male, `F` → Female, `FS` → Spayed Female, `MC` → Neutered Male. Unknown codes (`U`, `X`, blank) → `NULL`.

Zip preference: primary zip first; falls back to referral zip when the primary is blank or NA.

### PetBERT Predictions CSV

ML worker output, one file per ingestion run.

**Two output formats are accepted:**

**Per-row format** (one row per diagnosis rank):

| Column | Description |
|---|---|
| `anon_id` or `case_id` | Patient identifier |
| `diagnosis_index` | Rank (1 = top prediction, 2 = second, …) |
| `predicted_term` | Vet-ICD-O term |
| `predicted_group` | Vet-ICD-O group (stored as `cancer_type.name`) |
| `predicted_code` | ICD-O code |
| `confidence` | Float 0–1 |
| `method` | `embedding`, `low_confidence`, or `empty` |
| `original_text` | Pathology report text (optional; enables GCS upload) |

**Numbered-string format** (one row per patient, multiple predictions joined as `"1) X 2) Y"`):

Same columns, but `predicted_term`, `predicted_group`, `predicted_code`, `confidence`, and `method` may be numbered strings. The parser splits them automatically.

Rows where `method == "empty"` are skipped entirely.

---

## Ingestion Pipeline

### Step 1: Parse

**`parse_predictions(predictions: list[dict])`**
- Accepts `case_id` as alias for `anon_id`
- Detects format automatically:
  - Per-row: `diagnosis_index` is a plain integer AND no `"1) ..."` pattern in term/group fields
  - Numbered-string: splits `"1) foo 2) bar"` fields into individual diagnosis records
- Returns `{anon_id: [{"diagnosis_index": int, "predicted_group": str, ...}, ...]}`

**`parse_dataset_a_demographics(csv_bytes: bytes)`**
- Accepts `case_id` or `anon_id` as patient ID column
- Accepts short (`Zipcode`, `RfrrVtrnZipcode`) and long (`Zipcode Zipcode`, `RfrrVtrn Zipcode Zipcode`) zip column names
- Takes first non-empty value per patient across duplicate rows
- Returns `{anon_id: {"sex": str|None, "breed": str|None, "diagnosis_date": date|None, "species": str|None, "zip": str|None}}`

**`normalize_anon_id(raw: str)`**
- `"37"` / `"37.0"` → `"ID_37"` (integer and Excel float formats)
- `"ID_37"` → `"ID_37"` (already canonical)
- `"CASE-0001"` → `"CASE-0001"` (non-numeric prefixes pass through unchanged)
- `""` / `"nan"` → `""` (skipped)

### Step 2: Upsert Patients

For every `anon_id` in the predictions, `INSERT ... ON CONFLICT (anon_id) DO UPDATE` writes the patient row with demographics from Dataset A. `data_source = 'petbert'` is always set, distinguishing real data from seed/mock records.

### Step 3: Upload Pathology Reports to GCS (optional)

When `original_text` is present in prediction rows and a GCS bucket is configured, the text is uploaded to `gs://{GCS_BUCKET}/reports/{job_id}/{anon_id}.txt`. A `pathology_reports` row is created pointing to that path. `case_diagnoses.pathology_report_id` links to this row so the Diagnosis Review UI can show the source text.

When `original_text` is absent (e.g., the predictions CSV has no text column), `pathology_report_id` is `NULL` on all diagnoses and no GCS uploads occur.

### Step 4: Delete and Re-insert Diagnoses (idempotent)

Before inserting new `case_diagnoses`, the pipeline deletes all existing `case_diagnoses` (and `pathology_reports`) for the affected patients. This makes re-runs safe — the final state always reflects the most recent predictions for each patient.

### Step 5: Upsert Cancer Types and Insert Diagnoses

`predicted_group` values are upserted into `cancer_types` (ON CONFLICT DO NOTHING). Each prediction becomes one `case_diagnoses` row.

**Review auto-flagging:** A diagnosis is set to `review_status = 'pending'` when any of:
- `confidence < REVIEW_AUTO_ACCEPT_CONFIDENCE` (env var, default 0.7)
- The margin between rank-1 and rank-2 confidence < `REVIEW_AUTO_ACCEPT_MARGIN`
- `method == "low_confidence"`

Otherwise, `review_status = 'confirmed'`. Every auto-flag also writes a `diagnosis_review_events` row for the audit trail.

### Step 6: Refresh and Log

Materialized views (`mv_county_cancer_incidence`, `mv_yearly_trends`) are refreshed. An `ingestion_logs` row records the run summary. The `ingestion_jobs` status is updated to `completed`.

---

## Database Schema (Relevant Tables)

```
patients
  id (PK), anon_id (UNIQUE, indexed)
  species_id → species, breed_id → breeds
  sex ('Male'|'Female'|'Neutered Male'|'Spayed Female')
  county_id → counties, zip_code
  data_source ('petbert' for ingested data, 'mock' for seed)
  diagnosis_date, outcome

case_diagnoses                          ← one row per prediction rank
  id (PK), patient_id → patients        ← direct FK, no intermediate table
  cancer_type_id → cancer_types
  icd_o_code, predicted_term
  confidence, prediction_method
  source_row_index, diagnosis_index (rank)
  pathology_report_id → pathology_reports (nullable — NULL when no source text)
  review_status ('pending'|'confirmed'|'corrected'|'rejected')
  ingestion_job_id → ingestion_jobs

pathology_reports
  id (PK), patient_id → patients
  gcs_path (e.g. 'reports/42/CASE-0001.txt')
  report_date, created_at

cancer_types
  id (PK), name (Vet-ICD-O group), confirmed (bool)

counties
  id (PK), name, fips_code
  geom (PostGIS MULTIPOLYGON, SRID 4326)
  is_catchment (true for the 16 UC Davis catchment counties)

ingestion_jobs
  id (PK), status, uploaded_by_sub, uploaded_by_email
  dataset_a_filename, storage_path, batch_job_name
  created_at, updated_at, result_summary (JSONB)
```

---

## API Endpoints (Data Flow)

| Endpoint | What it queries |
|---|---|
| `GET /api/v1/dashboard/summary` | `case_diagnoses` grouped by `cancer_type`; `patients` for species breakdown |
| `GET /api/v1/geo/counties` | PostGIS `ST_AsGeoJSON`, case counts per county |
| `GET /api/v1/trends/yearly` | `patients.diagnosis_date` grouped by year |
| `GET /api/v1/incidence/by-type` | `case_diagnoses` joined to `patients` |
| `GET /api/v1/diagnoses` | Diagnosis review list with filter by `review_status` |

All dashboard queries filter `patients.data_source = 'petbert'` to exclude seed/mock data.

---

## Known Constraints

- **No dog population data by county** — the dashboard shows case counts, not incidence rates.
- **No `cancer_cases` table** — always query `case_diagnoses JOIN patients`, not `cancer_cases`.
- **Pathology text in GCS, not the DB** — `case_diagnoses` does not store text inline; the text pointer is `pathology_reports.gcs_path`. If no `original_text` was in the predictions, this is `NULL`.
- **Per-run idempotency** — re-running on the same dataset overwrites prior diagnoses for affected patients.
