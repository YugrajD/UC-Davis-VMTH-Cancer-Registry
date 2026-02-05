# UC Davis VMTH Cancer Registry

A full-stack veterinary cancer catchment area dashboard modeled after the UCSF Helen Diller Comprehensive Cancer Center dashboard, adapted for UC Davis VMTH pet cancer data with mock data.

## Tech Stack

- **Frontend**: React 18 + TypeScript (Vite), Tailwind CSS, React Leaflet, Recharts
- **Backend**: Python 3.11 + FastAPI, SQLAlchemy + GeoAlchemy2
- **Database**: PostgreSQL 16 + PostGIS 3.4
- **ML/NLP**: VetBERT (keyword-based mock classifier for development)
- **Orchestration**: Docker Compose

## Quick Start

```bash
# Start all services
docker compose up -d

# Run database migrations (automatic on first start via init scripts)
# Seed mock data (~5000 cases)
docker compose --profile seed run seed

# Access the application
open http://localhost:5173    # Frontend
open http://localhost:8000/docs  # API docs (Swagger)
```

## Features (7 Tabs)

| Tab | Description |
|-----|-------------|
| **Overview** | Summary stats cards, top-level metrics, species breakdown |
| **Map** | Interactive choropleth of Northern CA catchment area (16 counties) |
| **Incidence** | Cancer incidence rates by type with bar charts |
| **Trends** | Time series 2015-2024 with multi-line charts |
| **Species & Breed** | Pie charts and horizontal bar charts for demographic breakdowns |
| **County Data** | County-level stacked bar comparisons and data tables |
| **Report Search** | VetBERT-powered pathology report classifier and search |

## Mock Data

- ~5,000 cancer cases across 30 years (1995-2024)
- 16 Northern CA counties with realistic population weighting
- 2 species: Dogs (65%), Cats (35%)
- 8 cancer types with species-appropriate distributions
- ~500 mock pathology reports with veterinary oncology terminology

## API Endpoints

- `GET /api/v1/dashboard/summary` - Dashboard summary stats
- `GET /api/v1/dashboard/filters` - Available filter options
- `GET /api/v1/incidence` - Cancer incidence with filters
- `GET /api/v1/incidence/by-cancer-type` - Grouped by cancer type
- `GET /api/v1/incidence/by-species` - Grouped by species
- `GET /api/v1/incidence/by-breed` - Grouped by breed
- `GET /api/v1/geo/counties` - GeoJSON FeatureCollection with case counts
- `GET /api/v1/geo/counties/{id}` - Single county detail
- `GET /api/v1/trends/yearly` - Yearly case trends
- `GET /api/v1/trends/by-cancer-type` - Trends by cancer type
- `POST /api/v1/search/classify` - Classify pathology report text
- `GET /api/v1/search/reports` - Search pathology reports

## Verification

```bash
# Verify database
docker compose exec db psql -U postgres -d vmth_cancer -c "SELECT count(*) FROM cancer_cases;"

# Verify PostGIS
docker compose exec db psql -U postgres -d vmth_cancer -c "SELECT name, ST_AsText(ST_Centroid(geom)) FROM counties LIMIT 3;"

# Verify API
curl http://localhost:8000/api/v1/dashboard/summary
curl http://localhost:8000/api/v1/geo/counties

# Verify BERT classifier
curl -X POST http://localhost:8000/api/v1/search/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Histopathology reveals diffuse large B-cell lymphoma in the submandibular lymph node"}'
```
