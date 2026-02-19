# PostGIS geospatial analysis

The app uses **PostGIS** for county-level maps and choropleths. County boundaries are stored in the `counties.geom` column (WGS84). The API serves GeoJSON at `GET /api/v1/geo/counties` for the frontend map.

## How it works

1. **Database**: PostGIS is enabled in `001_extensions.sql`. The `counties` table has a `geom` column (`GEOMETRY(MULTIPOLYGON, 4326)`). Only rows with non-null `geom` are returned by the geo API.
2. **Boundaries**: County polygons must be loaded into `counties.geom`. Two options:
   - **16 UCD catchment counties**: Embedded in `database/seed/county_boundaries.py`. Loaded automatically when you run ingest if no county has geometry yet.
   - **All 58 California counties**: Generate `geo/data/ca_counties.geojson` from Census TIGER data, then load it (see below).
3. **API**: `backend/app/routers/geo.py` uses `ST_AsGeoJSON(c.geom)` and joins with `cancer_cases` to return county boundaries plus case counts, so the frontend can draw a choropleth.

## Quick start (16 counties)

After bringing up the DB and running ingest, the **16 catchment counties** get their boundaries loaded automatically:

```bash
docker compose up -d
docker compose run --rm ingest
```

Then start the backend and hit `GET /api/v1/geo/counties` — you’ll get 16 GeoJSON features with `total_cases` and `geometry`.

## All 58 California counties

To show every CA county (including those with cases outside the catchment), load boundaries from Census data:

1. **One-time: download shapefile and export GeoJSON** (on your machine, with Python + GeoPandas):

   ```bash
   cd geo
   pip install geopandas requests
   python download_boundaries.py
   python process_counties.py --all-ca
   ```

   This creates `geo/data/ca_counties.geojson`.

2. **Load into PostGIS** (with DB and geo folder available):

   ```bash
   docker compose run --rm geo-seed
   ```

   Or run ingest again (it runs boundary loading when no county has geometry; with `geo` mounted and `GEO_DATA_DIR=/geo/data`, it will use `ca_counties.geojson` if present).

   To load only from the file (e.g. after adding new counties), run:

   ```bash
   docker compose run --rm geo-seed
   ```

   The `geo-seed` service uses `GEO_DATA_DIR=/geo/data` and reads `geo/data/ca_counties.geojson` when it exists; otherwise it loads the 16 embedded catchment counties.

## Re-loading boundaries

- **Only boundaries (no ingest):**  
  `docker compose run --rm geo-seed`
- **Ingest (and boundaries if missing):**  
  `docker compose run --rm ingest`

## Year filters and ingested data

Ingested PetBERT data has `diagnosis_date = NULL`. The geo endpoint still counts those cases when **no** `year_start` / `year_end` query params are used. If you filter by year, only cases with a non-null diagnosis date are included, so ingested-only data won’t appear in year-filtered geo requests until you add diagnosis dates.
