#!/usr/bin/env python3
"""
Ingest CalEnviroScreen 4.0 data into the VMTH Cancer Registry database.

Downloads the CalEnviroScreen 4.0 XLSX from California Open Data, aggregates
census-tract-level indicators to county level using population-weighted means,
and inserts into the calenviroscreen table.

Usage:
  python database/seed/ingest_calenviroscreen.py

Or via Docker Compose:
  docker compose run --rm ingest python /database/seed/ingest_calenviroscreen.py
"""

import os
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import psycopg2

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql://postgres:postgres@localhost:5432/vmth_cancer",
)

DATA_DIR = Path(os.getenv("DATA_DIR", "/database/data"))

CES_XLSX_URL = (
    "https://data.ca.gov/dataset/11eb2b90-f3c1-46b4-bdf2-ba1dab939dac"
    "/resource/9a90474a-3912-4955-9552-457ee41cea51/download/"
    "calenviroscreen40resultsdatadictionary_f_2021.xlsx"
)

# CalEnviroScreen 4.0 column name -> our DB column name
INDICATOR_COLUMNS = {
    "CES 4.0 Score":              "ces_score",
    "CES 4.0 Percentile":         "ces_score",          # prefer percentile
    "Pollution Burden Score":      "pollution_burden",
    "Pollution Burden Pctl":       "pollution_burden",   # prefer percentile
    "Ozone Pctl":                  "ozone",
    "PM2.5 Pctl":                  "pm25",
    "Diesel PM Pctl":              "diesel_pm",
    "Pesticides Pctl":             "pesticides",
    "Tox. Release Pctl":           "toxic_releases",
    "Traffic Pctl":                "traffic",
    "Drinking Water Pctl":         "drinking_water",
    "Lead Pctl":                   "lead",
    "Cleanup Sites Pctl":          "cleanup_sites",
    "Groundwater Threats Pctl":    "groundwater_threats",
    "Haz. Waste Pctl":             "hazardous_waste",
    "Solid Waste Pctl":            "solid_waste",
    "Imp. Water Bodies Pctl":      "impaired_water",
    "Pop. Char. Score":            "pop_characteristics",
    "Pop. Char. Pctl":             "pop_characteristics", # prefer percentile
    "Asthma Pctl":                 "asthma",
    "Low Birth Weight Pctl":       "low_birth_weight",
    "Cardiovascular Disease Pctl": "cardiovascular",
    "Poverty Pctl":                "poverty",
    "Unemployment Pctl":           "unemployment",
    "Housing Burden Pctl":         "housing_burden",
    "Educ. Attainment Pctl":       "education",
    "Education Pctl":              "education",
    "Ling. Isolation Pctl":        "linguistic_isolation",
    "Linguistic Isolation Pctl":   "linguistic_isolation",
}

# We prefer percentile columns where both score and percentile exist.
# Build the final mapping: only keep percentile version when both exist.
PREFERRED_COLUMNS = {
    "CES 4.0 Percentile":         "ces_score",
    "Pollution Burden Pctl":       "pollution_burden",
    "Ozone Pctl":                  "ozone",
    "PM2.5 Pctl":                  "pm25",
    "Diesel PM Pctl":              "diesel_pm",
    "Pesticides Pctl":             "pesticides",
    "Tox. Release Pctl":           "toxic_releases",
    "Traffic Pctl":                "traffic",
    "Drinking Water Pctl":         "drinking_water",
    "Lead Pctl":                   "lead",
    "Cleanup Sites Pctl":          "cleanup_sites",
    "Groundwater Threats Pctl":    "groundwater_threats",
    "Haz. Waste Pctl":             "hazardous_waste",
    "Solid Waste Pctl":            "solid_waste",
    "Imp. Water Bodies Pctl":      "impaired_water",
    "Pop. Char. Pctl":             "pop_characteristics",
    "Asthma Pctl":                 "asthma",
    "Low Birth Weight Pctl":       "low_birth_weight",
    "Cardiovascular Disease Pctl": "cardiovascular",
    "Poverty Pctl":                "poverty",
    "Unemployment Pctl":           "unemployment",
    "Housing Burden Pctl":         "housing_burden",
    "Educ. Attainment Pctl":       "education",
    "Education Pctl":              "education",
    "Ling. Isolation Pctl":        "linguistic_isolation",
    "Linguistic Isolation Pctl":   "linguistic_isolation",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def download_ces_data() -> pd.DataFrame:
    """Download or read cached CES 4.0 XLSX and return the results sheet."""
    local_path = DATA_DIR / "calenviroscreen40.xlsx"

    if local_path.exists():
        print(f"Using cached CES file: {local_path}")
    else:
        print(f"Downloading CalEnviroScreen 4.0 from California Open Data...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(CES_XLSX_URL, local_path)
        print(f"  Saved to {local_path}")

    # The results sheet is typically the first sheet
    xls = pd.ExcelFile(local_path, engine="openpyxl")
    # Find the results sheet (not the data dictionary)
    results_sheet = None
    for sheet in xls.sheet_names:
        if "dictionary" not in sheet.lower() and "dict" not in sheet.lower():
            results_sheet = sheet
            break
    if results_sheet is None:
        results_sheet = xls.sheet_names[0]

    print(f"  Reading sheet: '{results_sheet}'")
    df = pd.read_excel(xls, sheet_name=results_sheet, dtype=str)
    df.columns = df.columns.str.strip()
    return df


def aggregate_to_county(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate census-tract indicators to county level using population-weighted means."""

    # Identify the county column
    county_col = None
    for col in df.columns:
        if "county" in col.lower() and "california" in col.lower():
            county_col = col
            break
    if county_col is None:
        for col in df.columns:
            if "county" in col.lower():
                county_col = col
                break
    if county_col is None:
        print("ERROR: Could not find county column in CES data.")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"  County column: '{county_col}'")

    # Identify population column
    pop_col = None
    for col in df.columns:
        if col.lower().strip() in ("total population", "totalpopulation", "pop"):
            pop_col = col
            break
    if pop_col is None:
        for col in df.columns:
            if "population" in col.lower() and "total" in col.lower():
                pop_col = col
                break

    if pop_col:
        print(f"  Population column: '{pop_col}'")
    else:
        print("  WARNING: No population column found, using unweighted means")

    # Find which preferred columns actually exist in the data
    available_mappings = {}
    for ces_col, db_col in PREFERRED_COLUMNS.items():
        if ces_col in df.columns:
            available_mappings[ces_col] = db_col
        else:
            # Try fuzzy match
            for actual_col in df.columns:
                if ces_col.lower().replace(" ", "") == actual_col.lower().replace(" ", ""):
                    available_mappings[actual_col] = db_col
                    break

    print(f"  Matched {len(available_mappings)} of {len(PREFERRED_COLUMNS)} indicator columns")

    # Convert numeric columns
    numeric_cols = list(available_mappings.keys())
    if pop_col:
        numeric_cols.append(pop_col)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no county
    df = df[df[county_col].notna() & (df[county_col].str.strip() != "")]

    # Aggregate: population-weighted mean per county
    results = []
    for county_name, group in df.groupby(county_col):
        row = {"county_name": county_name.strip()}

        for ces_col, db_col in available_mappings.items():
            values = group[ces_col].dropna()
            if pop_col and pop_col in group.columns:
                weights = group.loc[values.index, pop_col].fillna(0)
                total_weight = weights.sum()
                if total_weight > 0:
                    row[db_col] = round((values * weights).sum() / total_weight, 2)
                elif len(values) > 0:
                    row[db_col] = round(values.mean(), 2)
                else:
                    row[db_col] = None
            elif len(values) > 0:
                row[db_col] = round(values.mean(), 2)
            else:
                row[db_col] = None

        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run():
    df = download_ces_data()
    print(f"  {len(df)} census tracts loaded")
    print(f"  Columns: {list(df.columns)[:10]}...")

    county_df = aggregate_to_county(df)
    print(f"\n  Aggregated to {len(county_df)} counties")

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    # Load county name -> id mapping
    cur.execute("SELECT id, name FROM counties")
    county_map = {name.lower(): id_ for id_, name in cur.fetchall()}

    # Ensure calenviroscreen table exists
    cur.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'calenviroscreen'
        )
    """)
    if not cur.fetchone()[0]:
        print("ERROR: calenviroscreen table does not exist. Run migration 010 first.")
        cur.close()
        conn.close()
        sys.exit(1)

    db_columns = [
        "ces_score", "pollution_burden", "ozone", "pm25", "diesel_pm",
        "pesticides", "toxic_releases", "traffic", "drinking_water", "lead",
        "cleanup_sites", "groundwater_threats", "hazardous_waste", "solid_waste",
        "impaired_water", "pop_characteristics", "asthma", "low_birth_weight",
        "cardiovascular", "poverty", "unemployment", "housing_burden",
        "education", "linguistic_isolation",
    ]

    inserted = 0
    skipped = 0
    not_found = []

    for _, row in county_df.iterrows():
        county_name = row["county_name"]
        county_id = county_map.get(county_name.lower())

        if county_id is None:
            not_found.append(county_name)
            skipped += 1
            continue

        values = [row.get(col) for col in db_columns]
        # Convert NaN to None
        values = [None if pd.isna(v) else v for v in values]

        placeholders = ", ".join(["%s"] * (1 + len(db_columns)))
        col_names = ", ".join(["county_id"] + db_columns)
        update_set = ", ".join(f"{col} = EXCLUDED.{col}" for col in db_columns)

        cur.execute(
            f"INSERT INTO calenviroscreen ({col_names}) VALUES ({placeholders}) "
            f"ON CONFLICT (county_id) DO UPDATE SET {update_set}",
            [county_id] + values,
        )
        inserted += 1

    conn.commit()

    print(f"\nDone!")
    print(f"  Inserted/updated: {inserted} counties")
    if skipped:
        print(f"  Skipped: {skipped} (county not found in DB)")
        for name in not_found[:10]:
            print(f"    - {name}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    run()
