#!/usr/bin/env python3
"""Download Census ZCTA shapefile, filter to California, add county names, export GeoJSON.

Usage:
    python scripts/generate_zcta_geojson.py

Output:
    frontend/public/california-zctas.geojson
"""

import json
import sys
import tempfile
import zipfile
from pathlib import Path

try:
    import geopandas as gpd
    import requests
except ImportError:
    print("Install dependencies: pip install geopandas requests", file=sys.stderr)
    sys.exit(1)

# Census Bureau 500k cartographic boundary files (2020)
ZCTA_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_zcta520_500k.zip"
STATE_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_state_500k.zip"
COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_500k.zip"

CA_FIPS = "06"
OUTPUT = Path(__file__).resolve().parent.parent / "frontend" / "public" / "california-zctas.geojson"
COORD_PRECISION = 5


def download_and_read(url: str, tmp: Path) -> gpd.GeoDataFrame:
    """Download a Census zip file and read the shapefile inside."""
    zip_path = tmp / url.split("/")[-1]
    print(f"  Downloading {url.split('/')[-1]}...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp / zip_path.stem)
    shp_files = list((tmp / zip_path.stem).glob("*.shp"))
    if not shp_files:
        raise FileNotFoundError(f"No .shp found in {zip_path}")
    return gpd.read_file(shp_files[0])


def round_coords(geom, precision: int):
    """Round coordinates in a geometry to `precision` decimal places."""
    import shapely

    return shapely.wkt.loads(
        shapely.wkt.dumps(geom, rounding_precision=precision)
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Download all three shapefiles
        print("Step 1: Downloading shapefiles...")
        zcta_gdf = download_and_read(ZCTA_URL, tmp)
        state_gdf = download_and_read(STATE_URL, tmp)
        county_gdf = download_and_read(COUNTY_URL, tmp)

        # 2. Filter to California
        print("Step 2: Filtering to California...")
        ca_boundary = state_gdf[state_gdf["STATEFP"] == CA_FIPS].geometry.union_all()
        ca_counties = county_gdf[county_gdf["STATEFP"] == CA_FIPS].copy()
        ca_counties = ca_counties[["NAME", "COUNTYFP", "geometry"]].rename(
            columns={"NAME": "COUNTY_NAME"}
        )

        # Spatial filter: keep ZCTAs whose centroid intersects CA boundary
        zcta_gdf = zcta_gdf.to_crs(ca_counties.crs)
        zcta_gdf["centroid"] = zcta_gdf.geometry.centroid
        ca_zctas = zcta_gdf[zcta_gdf["centroid"].intersects(ca_boundary)].copy()
        ca_zctas = ca_zctas.drop(columns=["centroid"])
        print(f"  Found {len(ca_zctas)} California ZCTAs")

        # 3. Spatial-join ZCTA centroids to county polygons for COUNTY_NAME
        print("Step 3: Assigning county names via spatial join...")
        centroids = ca_zctas.copy()
        centroids["geometry"] = ca_zctas.geometry.centroid
        joined = gpd.sjoin(centroids, ca_counties, how="left", predicate="within")

        # Assign county info back to the original ZCTA geometries
        ca_zctas["COUNTY_NAME"] = joined["COUNTY_NAME"].values
        ca_zctas["COUNTYFP"] = joined["COUNTYFP"].values

        # Fill any NaN (centroid fell outside county polygons — rare edge cases)
        ca_zctas["COUNTY_NAME"] = ca_zctas["COUNTY_NAME"].fillna("Unknown")
        ca_zctas["COUNTYFP"] = ca_zctas["COUNTYFP"].fillna("000")

        # 4. Keep only needed columns
        keep_cols = ["ZCTA5CE20", "GEOID20", "COUNTY_NAME", "COUNTYFP", "ALAND20", "AWATER20", "geometry"]
        ca_zctas = ca_zctas[keep_cols]

        # 5. Round coordinates for smaller file size
        print("Step 4: Rounding coordinates...")
        ca_zctas["geometry"] = ca_zctas["geometry"].apply(
            lambda g: round_coords(g, COORD_PRECISION)
        )

        # 6. Write GeoJSON
        print(f"Step 5: Writing to {OUTPUT}...")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        ca_zctas.to_file(OUTPUT, driver="GeoJSON")

        # Verify
        with open(OUTPUT) as f:
            data = json.load(f)
        n = len(data.get("features", []))
        size_mb = OUTPUT.stat().st_size / (1024 * 1024)
        print(f"Done! {n} features, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
