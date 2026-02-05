# Project Summary: UC Davis VMTH Cancer Registry

70 files, ~4,300 lines of code

## Database Layer (6 SQL migrations + 2 Python seed scripts)
- PostGIS extensions, lookup tables (species, breeds, cancer types), counties with geometry, core tables (patients, cancer_cases, pathology_reports), materialized views
- Seed script generates ~5,000 mock cancer cases with realistic distributions across 16 Northern CA counties, 5 species, 8 cancer types, and ~500 pathology reports

## Backend (FastAPI - 13 Python files)
- `app/main.py` - FastAPI entry point with CORS middleware
- `app/models/models.py` - 7 SQLAlchemy + GeoAlchemy2 models
- `app/schemas/schemas.py` - Pydantic request/response models
- 5 routers: `dashboard`, `incidence`, `geo`, `trends`, `search`
- 3 services: `bert_service` (keyword classifier), `geo_service` (PostGIS queries), `stats_service` (aggregations)

## Frontend (React 18 + TypeScript + Tailwind - 19 source files)
- **Layout**: Header (UC Davis branding), Footer, TabNavigation (7 tabs)
- **Context**: FilterContext with global filter state
- **Components**: FilterPanel, StatCard, ChoroplethMap (Leaflet), 4 chart types (Recharts), DataTable, BertSearchPanel
- **Pages**: Overview, Map, Incidence, Trends, Species & Breed, County Data, Report Search
- **API Client**: Axios with typed endpoints

## ML/NLP (3 files)
- VetBERT mock classifier using weighted keyword patterns
- Mock pathology report generator with realistic veterinary terminology

## Infrastructure
- Docker Compose with 4 services (db, backend, frontend, seed)
- PostgreSQL 16 + PostGIS 3.4 with health checks
- Vite dev server with API proxy to backend

## To Run
```bash
docker compose up -d
docker compose --profile seed run seed
# Frontend: http://localhost:5173
# API docs: http://localhost:8000/docs
```
