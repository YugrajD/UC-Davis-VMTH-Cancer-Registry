# UC Davis VMTH Cancer Registry - Data Pipeline Documentation

## Overview

The UC Davis VMTH Cancer Registry ingests PetBERT cancer predictions and patient demographics to create a unified database for visualizing canine cancer incidence across California. The system follows a **one dog = one registry case** model, where each dog has a single `cancer_cases` record with multiple cancer diagnoses stored in `case_diagnoses`.

---

## Data Sources

### 1. PetBERT Predictions (`petbert_scan_predictions.csv`)
- **Format**: CSV with columns: `anon_id`, `original_text`, `predicted_term`, `predicted_group`, `predicted_code`, `confidence`, `method`
- **Content**: AI-generated cancer predictions from pathology reports
- **Key Features**:
  - Multiple predictions per row (numbered format: `"1) Term 2) Term"`)
  - Each row may represent a different visit or report for the same dog
  - `anon_id` format: `"ID_3"`, `"ID_37"`, etc.

### 2. Dog Demographics (`All_deidentified_K9.xlsx`)
- **Format**: Excel file with columns: `anon_id`, `Sex`, `Owner Zipcode Zipcode`
- **Content**: Patient demographic information (sex, ZIP code)
- **Key Features**:
  - Multiple rows per `anon_id` (different visits)
  - Takes first non-empty sex/ZIP per `anon_id`
  - `anon_id` may be stored as numeric (`37`) or string (`"37.0"`, `"ID_37"`)

---

## Ingestion Pipeline

### Step 1: Data Parsing

**PetBERT CSV (`parse_petbert`)**
- Reads CSV and splits numbered predictions (e.g., `"1) Lymphoma 2) Mast Cell"`)
- Groups by `anon_id` → `{anon_id: [diagnosis1, diagnosis2, ...]}`
- Each diagnosis includes: `row_index`, `diagnosis_index`, `predicted_group`, `predicted_term`, `icd_o_code`, `confidence`, `original_text`, `method`

**K9 Excel (`parse_visits`)**
- Reads Excel using pandas
- Groups by `anon_id` → `{anon_id: {sex, zip}}`
- Takes first non-empty sex/ZIP per `anon_id`

**Critical: Anon ID Normalization**
- **Problem**: CSV uses `"ID_37"`, Excel may use `"37"` or `"37.0"` → same dog appears twice
- **Solution**: `normalize_anon_id()` converts all formats to canonical `"ID_<number>"`
  - `"37"` → `"ID_37"`
  - `"37.0"` → `"ID_37"`
  - `"ID_37"` → `"ID_37"` (already canonical)

### Step 2: Matching Patients

```python
matched_ids = set(petbert.keys()) & set(visits.keys())
```

- Only processes `anon_id`s that appear in **both** datasets
- Skips PetBERT-only (no demographics) and visits-only (no diagnoses)
- Result: List of `anon_id`s ready for ingestion

### Step 3: Database Upsert

**Patients Table**
- Upserts patients with `ON CONFLICT (anon_id) DO UPDATE`
- Sets `data_source = 'petbert'` for all ingested patients
- Explicit `UPDATE` ensures all matched patients are marked as petbert (handles edge cases)

**One Case Per Dog**
- **Problem**: Original design created one `cancer_cases` row per prediction → same dog counted hundreds of times
- **Solution**: Get-or-create **one** `cancer_cases` row per patient
  ```python
  # Get existing cases
  SELECT id, patient_id FROM cancer_cases WHERE patient_id = ANY(patient_ids)
  
  # Create missing cases
  INSERT INTO cancer_cases (id, patient_id, county_id, ...) 
  VALUES ... ON CONFLICT DO NOTHING
  ```
- Result: 395 dogs → 395 cases (not 2,348 cases)

**Case Diagnoses**
- All PetBERT predictions → `case_diagnoses` table
- Links to the single `cancer_cases` row via `case_id`
- Before insert: `DELETE FROM case_diagnoses WHERE case_id IN (...)` (idempotent re-run)
- Bulk insert using `execute_values()` for performance

**Cancer Types**
- Creates missing `cancer_types` from `predicted_group` values
- Uses Vet-ICD-O taxonomy groups

---

## Database Schema

### Core Tables

**`patients`**
- `id` (PK)
- `anon_id` (UNIQUE, indexed)
- `species_id`, `breed_id`, `sex`, `age_years`, `weight_kg`
- `county_id` (derived from ZIP)
- `zip_code`
- `data_source` (`'petbert'` for ingested data)

**`cancer_cases`**
- `id` (PK)
- `patient_id` (FK → patients)
- `cancer_type_id` (**nullable** - types live in `case_diagnoses`)
- `county_id`, `diagnosis_date`, `stage`, `outcome`
- One row per patient (dog)

**`case_diagnoses`** (Migration 009)
- `id` (PK)
- `case_id` (FK → cancer_cases)
- `cancer_type_id` (FK → cancer_types)
- `icd_o_code`, `predicted_term`, `original_text`
- `confidence`, `prediction_method`
- `source_row_index`, `diagnosis_index`
- Multiple rows per case (one per PetBERT prediction)

**`cancer_types`**
- `id` (PK)
- `name` (Vet-ICD-O group, e.g., "Mature B-cell lymphomas")
- Relationships: `cases`, `case_diagnoses`

**`counties`**
- `id` (PK)
- `name`, `fips_code`
- `geom` (PostGIS MULTIPOLYGON)
- `is_catchment` (boolean for UC Davis catchment area)

### Key Design Decisions

1. **`cancer_cases.cancer_type_id` is nullable**: PetBERT cases have multiple diagnoses, so no single type. Types live in `case_diagnoses`.
2. **One case per patient**: Prevents double-counting dogs with multiple visits/predictions.
3. **`case_diagnoses` for type-level queries**: Dashboard "top cancers" counts `case_diagnoses`, not `cancer_cases`.

---

## API Endpoints

### Dashboard (`/api/v1/dashboard/summary`)
- **Total cases**: `COUNT(cancer_cases)` where `patient.data_source = 'petbert'`
- **Top cancers**: `COUNT(case_diagnoses)` grouped by `cancer_type`
- **Species breakdown**: Joins `cancer_cases` → `patients` → `species`

### Geo (`/api/v1/geo/counties`)
- Returns GeoJSON FeatureCollection
- Uses PostGIS: `ST_AsGeoJSON(c.geom)`
- Aggregates case counts per county
- Filters by species, cancer_type, year, sex
- **Note**: No population/rate (dog population by county not available)

### Trends (`/api/v1/trends/yearly`)
- Groups `cancer_cases` by `EXTRACT(YEAR FROM diagnosis_date)`
- When filtering by cancer_type: joins to `case_diagnoses`

### Incidence (`/api/v1/incidence/by-type`)
- Uses `case_diagnoses` for type-level counts
- Groups by cancer_type, species, breed, county, year

---

## Frontend Visualization

### Data Flow
1. Frontend calls `/api/v1/geo/counties` with filters
2. Backend returns GeoJSON with `total_cases` per county
3. Frontend maps counties to regions (Bay Area, Central Valley, etc.)
4. Generates hierarchical summary (State → Regions → Counties)

### Components

**ChoroplethMap**
- Colors counties by **case count** (not rate)
- Uses `react-simple-maps` with external GeoJSON URL
- Tooltip shows: County, Region, Cases

**CountyTable**
- Sortable table: County, Count
- Color-coded by count (light → dark teal)
- Hover highlights county on map

**SummaryTable**
- Hierarchical: California → UC Davis Catchment → Regions → Counties
- Shows case counts only (no population/rate)

---

## Problems Solved

### 1. Anon ID Mismatch
- **Problem**: CSV `"ID_37"` vs Excel `"37"` → only 269 matches instead of 395
- **Solution**: `normalize_anon_id()` canonicalizes all formats to `"ID_<number>"`

### 2. Multiple Cases Per Dog
- **Problem**: One `cancer_cases` row per prediction → inflated counts
- **Solution**: One case per patient, all predictions in `case_diagnoses`

### 3. Re-run Idempotency
- **Problem**: Re-running ingest duplicates `case_diagnoses`
- **Solution**: `DELETE FROM case_diagnoses WHERE case_id IN (...)` before insert

### 4. INSERT Column Mismatch
- **Problem**: `cancer_cases` INSERT had 14 columns, tuple had 13 values
- **Solution**: Added missing `None` value to tuple

### 5. Population/Rate Without Data
- **Problem**: UI showed rates but only had human county population
- **Solution**: Removed population/rate from API and UI; show case counts only

---

## Running the Pipeline

### Prerequisites
- PostgreSQL with PostGIS extension
- Python dependencies: `psycopg2-binary`, `pandas`, `openpyxl`
- Data files: `petbert_scan_predictions.csv`, `All_deidentified_K9.xlsx`

### Steps

1. **Run migrations** (in order: 001 → 009)
   ```bash
   docker compose exec db psql -U postgres -d vmth_cancer -f /database/migrations/001_*.sql
   # ... through 009_one_case_per_patient.sql
   ```

2. **Load county boundaries** (optional, for accurate geometries)
   ```bash
   docker compose run --rm ingest python /database/seed/county_boundaries.py
   ```

3. **Run ingestion**
   ```bash
   docker compose run --rm ingest
   ```

### Expected Output
- **395 patients** (matched anon_ids from both datasets)
- **395 cancer_cases** (one per dog)
- **~2,348 case_diagnoses** (all PetBERT predictions)
- Dashboard shows: 395 total cases, top cancers by diagnosis count

---

## Data Quality Notes

- **Only matched patients**: Skips PetBERT-only and visits-only records
- **ZIP → County**: Uses lookup service; invalid ZIPs result in `county_id = NULL`
- **Confidence scores**: Stored as `NUMERIC(4,2)` from PetBERT predictions
- **Source tracking**: `data_source = 'petbert'` distinguishes ingested from mock/seed data

---

## Future Enhancements

- Add dog population by county to enable incidence rate calculations
- Support multiple data sources (not just PetBERT)
- Add temporal analysis (diagnosis_date trends)
- Implement data validation rules (confidence thresholds, required fields)
