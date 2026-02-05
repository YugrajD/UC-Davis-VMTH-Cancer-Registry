#!/usr/bin/env python3
"""
GeoPandas processing pipeline: filter US county shapefile to the 16 Northern CA
counties in the UC Davis VMTH catchment area and export as GeoJSON.
"""

import os
import json
import geopandas as gpd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SHAPEFILE = os.path.join(DATA_DIR, "tl_2023_us_county.shp")
OUTPUT_GEOJSON = os.path.join(DATA_DIR, "catchment_counties.geojson")

# 16 Northern CA counties in the UCD catchment area (FIPS codes)
CATCHMENT_FIPS = [
    "06067",  # Sacramento
    "06113",  # Yolo
    "06095",  # Solano
    "06061",  # Placer
    "06017",  # El Dorado
    "06077",  # San Joaquin
    "06013",  # Contra Costa
    "06001",  # Alameda
    "06099",  # Stanislaus
    "06101",  # Sutter
    "06115",  # Yuba
    "06057",  # Nevada
    "06005",  # Amador
    "06007",  # Butte
    "06011",  # Colusa
    "06021",  # Glenn
]


def process_counties():
    if not os.path.exists(SHAPEFILE):
        print(f"Shapefile not found: {SHAPEFILE}")
        print("Run download_boundaries.py first.")
        return

    print(f"Reading shapefile: {SHAPEFILE}")
    gdf = gpd.read_file(SHAPEFILE)

    # Filter to catchment area counties
    # GEOID column contains the full FIPS code
    catchment = gdf[gdf["GEOID"].isin(CATCHMENT_FIPS)].copy()
    print(f"Filtered to {len(catchment)} catchment area counties")

    # Simplify geometry for web display (tolerance in degrees)
    catchment["geometry"] = catchment["geometry"].simplify(tolerance=0.005)

    # Reproject to WGS84 (EPSG:4326) if needed
    if catchment.crs and catchment.crs.to_epsg() != 4326:
        catchment = catchment.to_crs(epsg=4326)

    # Select and rename columns
    catchment = catchment[["GEOID", "NAME", "geometry"]].rename(
        columns={"GEOID": "fips_code", "NAME": "name"}
    )

    # Export as GeoJSON
    catchment.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"Exported to {OUTPUT_GEOJSON}")

    # Print summary
    for _, row in catchment.iterrows():
        centroid = row.geometry.centroid
        print(f"  {row['name']} ({row['fips_code']}): centroid ({centroid.y:.3f}, {centroid.x:.3f})")


if __name__ == "__main__":
    process_counties()
