# Cancer Rates (HCIP) Implementation Plan

## Background

The **Hospital-based Cancer Incidence Proportion (HCIP)** is the cancer rate metric used in:

> Rupasinghe, R. et al. "Descriptive epidemiology of canine and feline cancer in California, United States from 2000 to 2019." *The Veterinary Journal* (2026). https://doi.org/10.1016/j.tvjl.2026.106612

Formula (Eq. 1 from the paper):

```
HCIP = (incident cancer patients in year Y / total patients seen in year Y) × 100
```

Reported as **cases per 100 animals**.

**"Incident" definition:** Only the first-ever cancer diagnosis per patient counts toward the numerator. Subsequent visits for the same or a new cancer type in later years are excluded, even if a new diagnostic keyword was added.

---

## What We Already Have (Numerator)

| Data | Source |
|---|---|
| Incident cancer patients | `patients` + `case_diagnoses` tables (PetBERT pipeline) |
| Cancer type | `case_diagnoses.cancer_type` |
| County | `patients.county_id` |
| Diagnosis year | `patients.diagnosis_date` |
| Species / breed / sex | `patients.*` |

The numerator is fully available once we add logic to deduplicate to first-diagnosis-per-patient per year.

---

## What Is Still Needed (Denominator)

**The denominator — total patients seen per year — will be provided separately.**

Once it arrives, it goes into a new table:

```sql
-- Migration: add after existing migrations
CREATE TABLE patient_population (
  year            INT  NOT NULL,
  county_id       INT  REFERENCES counties(id),  -- NULL = statewide
  species         TEXT,                           -- 'Dog', 'Cat', or NULL = all
  total_patients  INT  NOT NULL,
  PRIMARY KEY (year, county_id, species)
);
```

---

## Planned Changes

### 1. Database

- [ ] Write migration (`0XX_patient_population.sql`) to create `patient_population` table
- [ ] Write seeder / admin ingestion endpoint to load denominator data when provided

### 2. Backend

- [ ] New endpoint: `GET /api/v1/incidence/rates`
  - Accepts same filters as `GET /api/v1/incidence` (species, cancer_type, county, sex, year_start, year_end)
  - Returns HCIP per county/year/cancer type alongside raw count
  - SQL: `COUNT(DISTINCT patient_id) AS incident_cases` joined to `patient_population` for the denominator
- [ ] Add optional `hcip` field to existing incidence response schemas so count and rate travel together

### 3. Frontend

- [ ] Wire the existing **Rate Type** filter (`incidence` / `mortality` dropdown in `Filters.tsx`) — currently renders but does nothing
  - `incidence` → show raw **Count** (current behavior)
  - `mortality` → **blocked** (no mortality data yet; keep disabled or relabel)
  - Consider renaming to **Display Mode**: `Count` vs `Rate (per 100 animals)`
- [ ] Choropleth map: switch color scale domain from raw count to HCIP when rate mode is active
- [ ] County table: add HCIP column (hide when count mode)
- [ ] Map tooltip: show both count and HCIP
- [ ] Dashboard summary cards: show average HCIP alongside total case count

---

## What Is Blocked

| Task | Blocked on |
|---|---|
| `patient_population` migration | Denominator data format confirmation |
| `/api/v1/incidence/rates` endpoint | `patient_population` table |
| Frontend rate toggle (data) | Backend endpoint |
| Frontend rate toggle (UI wiring) | Can be done now — just needs the API call swapped |

---

## References

- Paper methodology: HCIP defined in Eq. 1, Materials & Methods section
- Study area: 145-mile buffer around VMTH, 40 counties, 2707 census tracts
- Cancer categorization: 9 major types (see Table 1 in paper) — already matches our `case_diagnoses` schema
