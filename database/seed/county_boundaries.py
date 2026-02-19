#!/usr/bin/env python3
"""
Load county boundary geometries into PostGIS for geospatial (choropleth) maps.

- If geo/data/ca_counties.geojson exists: loads all CA counties from that file
  (generate it with: cd geo && python process_counties.py --all-ca).
- Otherwise: loads the 16 UCD catchment counties from embedded COUNTY_GEOMETRIES.
"""

import os
import json
import sys
from pathlib import Path

import psycopg2

DATABASE_URL = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql://postgres:postgres@localhost:5432/vmth_cancer",
)

# Path to optional GeoJSON (all 58 CA counties). Set GEO_DATA_DIR in Docker to /geo/data.
GEO_DATA_DIR = Path(os.getenv("GEO_DATA_DIR", ""))
if not GEO_DATA_DIR:
    script_dir = Path(__file__).resolve().parent
    # Repo root: from database/seed go up to database, then to repo
    repo_root = script_dir.parent.parent
    GEO_DATA_DIR = repo_root / "geo" / "data"
CA_COUNTIES_GEOJSON = GEO_DATA_DIR / "ca_counties.geojson"

# Simplified bounding polygons for the 16 Northern CA counties in the UCD catchment area.
# These are approximate but sufficient for a choropleth visualization.
COUNTY_GEOMETRIES = {
    "Sacramento": {
        "type": "MultiPolygon",
        "coordinates": [[[[-121.862, 38.924], [-121.862, 38.018], [-121.027, 38.018],
                          [-121.027, 38.500], [-121.141, 38.711], [-121.501, 38.924],
                          [-121.862, 38.924]]]]
    },
    "Yolo": {
        "type": "MultiPolygon",
        "coordinates": [[[[-122.395, 38.924], [-122.395, 38.317], [-121.588, 38.317],
                          [-121.588, 38.734], [-121.862, 38.924], [-122.395, 38.924]]]]
    },
    "Solano": {
        "type": "MultiPolygon",
        "coordinates": [[[[-122.406, 38.427], [-122.406, 38.065], [-121.593, 38.065],
                          [-121.593, 38.317], [-121.862, 38.317], [-122.078, 38.427],
                          [-122.406, 38.427]]]]
    },
    "Placer": {
        "type": "MultiPolygon",
        "coordinates": [[[[-121.141, 39.316], [-121.141, 38.711], [-120.072, 38.711],
                          [-120.072, 39.067], [-120.310, 39.316], [-121.141, 39.316]]]]
    },
    "El Dorado": {
        "type": "MultiPolygon",
        "coordinates": [[[[-121.141, 38.711], [-121.141, 38.500], [-120.072, 38.500],
                          [-119.889, 38.711], [-120.072, 38.711], [-121.141, 38.711]]]]
    },
    "San Joaquin": {
        "type": "MultiPolygon",
        "coordinates": [[[[-121.593, 38.065], [-121.593, 37.632], [-120.920, 37.632],
                          [-120.920, 37.837], [-121.027, 38.018], [-121.027, 38.065],
                          [-121.593, 38.065]]]]
    },
    "Contra Costa": {
        "type": "MultiPolygon",
        "coordinates": [[[[-122.406, 38.065], [-122.406, 37.832], [-121.780, 37.832],
                          [-121.593, 37.837], [-121.593, 38.065], [-122.406, 38.065]]]]
    },
    "Alameda": {
        "type": "MultiPolygon",
        "coordinates": [[[[-122.373, 37.832], [-122.373, 37.454], [-121.582, 37.454],
                          [-121.582, 37.632], [-121.780, 37.832], [-122.373, 37.832]]]]
    },
    "Stanislaus": {
        "type": "MultiPolygon",
        "coordinates": [[[[-121.226, 37.837], [-121.226, 37.284], [-120.387, 37.284],
                          [-120.387, 37.632], [-120.920, 37.837], [-121.226, 37.837]]]]
    },
    "Sutter": {
        "type": "MultiPolygon",
        "coordinates": [[[[-121.862, 39.304], [-121.862, 38.924], [-121.501, 38.924],
                          [-121.279, 39.024], [-121.279, 39.304], [-121.862, 39.304]]]]
    },
    "Yuba": {
        "type": "MultiPolygon",
        "coordinates": [[[[-121.279, 39.438], [-121.279, 39.024], [-120.901, 39.024],
                          [-120.901, 39.316], [-121.067, 39.438], [-121.279, 39.438]]]]
    },
    "Nevada": {
        "type": "MultiPolygon",
        "coordinates": [[[[-121.141, 39.438], [-121.141, 39.067], [-120.072, 39.067],
                          [-120.072, 39.316], [-120.310, 39.438], [-121.141, 39.438]]]]
    },
    "Amador": {
        "type": "MultiPolygon",
        "coordinates": [[[[-121.027, 38.500], [-121.027, 38.247], [-120.296, 38.247],
                          [-120.296, 38.431], [-120.653, 38.500], [-121.027, 38.500]]]]
    },
    "Butte": {
        "type": "MultiPolygon",
        "coordinates": [[[[-122.058, 39.978], [-122.058, 39.438], [-121.279, 39.438],
                          [-121.067, 39.580], [-121.067, 39.829], [-121.436, 39.978],
                          [-122.058, 39.978]]]]
    },
    "Colusa": {
        "type": "MultiPolygon",
        "coordinates": [[[[-122.775, 39.383], [-122.775, 38.924], [-122.058, 38.924],
                          [-121.862, 39.050], [-121.862, 39.304], [-122.058, 39.383],
                          [-122.775, 39.383]]]]
    },
    "Glenn": {
        "type": "MultiPolygon",
        "coordinates": [[[[-122.912, 39.831], [-122.912, 39.383], [-122.058, 39.383],
                          [-122.058, 39.580], [-122.058, 39.831], [-122.912, 39.831]]]]
    },
}


def _geom_to_multipolygon_json(geom_dict):
    """Ensure geometry is MultiPolygon for PostGIS column."""
    if geom_dict.get("type") == "Polygon":
        return {"type": "MultiPolygon", "coordinates": [geom_dict["coordinates"]]}
    return geom_dict


def load_boundaries():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    loaded = 0

    # Prefer full CA GeoJSON if present (all 58 counties)
    if CA_COUNTIES_GEOJSON.exists():
        with open(CA_COUNTIES_GEOJSON, encoding="utf-8") as f:
            data = json.load(f)
        features = data.get("features", data) if isinstance(data, dict) else data
        for feat in features:
            props = feat.get("properties", {})
            fips = props.get("GEOID") or props.get("fips_code") or props.get("FIPS")
            name = props.get("NAME") or props.get("name")
            geom = feat.get("geometry")
            if not fips or not geom:
                continue
            geom = _geom_to_multipolygon_json(geom)
            geom_json = json.dumps(geom)
            cur.execute(
                """UPDATE counties
                   SET geom = ST_SetSRID(ST_Multi(ST_GeomFromGeoJSON(%s)), 4326)
                   WHERE fips_code = %s""",
                (geom_json, fips),
            )
            if cur.rowcount:
                loaded += 1
                print(f"  Loaded geometry for {name or fips}")
        print(f"Loaded {loaded} county boundaries from {CA_COUNTIES_GEOJSON}.")
    else:
        for county_name, geojson in COUNTY_GEOMETRIES.items():
            geom_json = json.dumps(geojson)
            cur.execute(
                """UPDATE counties
                   SET geom = ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)
                   WHERE name = %s""",
                (geom_json, county_name),
            )
            if cur.rowcount:
                loaded += 1
            print(f"  Loaded geometry for {county_name}")
        print(f"Loaded {loaded} catchment county boundaries (embedded).")
        if loaded < 58:
            print("  Tip: generate geo/data/ca_counties.geojson for all 58 CA counties:")
            print("    cd geo && pip install geopandas && python download_boundaries.py")
            print("    python process_counties.py --all-ca")

    conn.commit()
    cur.close()
    conn.close()
    print("County boundaries loaded successfully.")


if __name__ == "__main__":
    load_boundaries()
