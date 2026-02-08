# Current Architecture Snapshot

[Back to Overview](../IMPLEMENTATION_PLAN.md)

---

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
