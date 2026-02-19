#!/usr/bin/env python3
"""
GeoPandas processing pipeline: filter US county shapefile to California counties
and export as GeoJSON for PostGIS.

  python process_counties.py          # 16 UCD catchment counties → catchment_counties.geojson
  python process_counties.py --all-ca  # All 58 CA counties → ca_counties.geojson (for geo maps)
"""

import argparse
import os
import geopandas as gpd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SHAPEFILE = os.path.join(DATA_DIR, "tl_2023_us_county.shp")

# 16 Northern CA counties in the UCD catchment area (FIPS codes)
CATCHMENT_FIPS = [
    "06067", "06113", "06095", "06061", "06017", "06077",
    "06013", "06001", "06099", "06101", "06115", "06057",
    "06005", "06007", "06011", "06021",
]


def process_counties(all_ca: bool = False):
    if not os.path.exists(SHAPEFILE):
        print(f"Shapefile not found: {SHAPEFILE}")
        print("Run download_boundaries.py first.")
        return

    print(f"Reading shapefile: {SHAPEFILE}")
    gdf = gpd.read_file(SHAPEFILE)

    if all_ca:
        # All California counties (state FIPS 06)
        subset = gdf[gdf["GEOID"].str.startswith("06")].copy()
        output_path = os.path.join(DATA_DIR, "ca_counties.geojson")
        print(f"Filtered to {len(subset)} California counties")
    else:
        subset = gdf[gdf["GEOID"].isin(CATCHMENT_FIPS)].copy()
        output_path = os.path.join(DATA_DIR, "catchment_counties.geojson")
        print(f"Filtered to {len(subset)} catchment area counties")

    subset["geometry"] = subset["geometry"].simplify(tolerance=0.005)
    if subset.crs and subset.crs.to_epsg() != 4326:
        subset = subset.to_crs(epsg=4326)

    # Keep GEOID and NAME for PostGIS loader (county_boundaries.py matches on fips_code)
    subset = subset[["GEOID", "NAME", "geometry"]]
    subset.to_file(output_path, driver="GeoJSON")
    print(f"Exported to {output_path}")

    for _, row in subset.iterrows():
        c = row.geometry.centroid
        print(f"  {row['NAME']} ({row['GEOID']}): centroid ({c.y:.3f}, {c.x:.3f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export county boundaries to GeoJSON")
    p.add_argument("--all-ca", action="store_true", help="Export all 58 CA counties to ca_counties.geojson")
    args = p.parse_args()
    process_counties(all_ca=args.all_ca)
