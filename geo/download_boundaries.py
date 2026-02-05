#!/usr/bin/env python3
"""
Download California county boundary shapefiles from the US Census Bureau.
These are used to populate the PostGIS counties table with accurate geometries.
"""

import os
import requests
import zipfile
import io

TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_county_boundaries():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "tl_2023_us_county.zip")

    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return output_path

    print(f"Downloading county boundaries from {TIGER_URL}...")
    response = requests.get(TIGER_URL, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded to {output_path}")

    # Extract
    with zipfile.ZipFile(output_path, "r") as z:
        z.extractall(OUTPUT_DIR)
    print(f"Extracted to {OUTPUT_DIR}")

    return output_path


if __name__ == "__main__":
    download_county_boundaries()
