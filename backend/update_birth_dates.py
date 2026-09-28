"""
One-shot script: reads birth dates from the source CSV/XLSX and
updates ONLY the birth_date column for existing patients.

No ML, no diagnoses touched. Safe to run multiple times (idempotent).
"""

import asyncio
import csv
import io
import os
import sys
from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# ---------------------------------------------------------------------------
# Normalise anon_id exactly as ingestion_service.py does
# ---------------------------------------------------------------------------
def normalize_anon_id(raw: str) -> str:
    s = raw.strip().upper()
    if not s or s in ("NA", "NAN", "NONE", ""):
        return ""
    return s

# ---------------------------------------------------------------------------
# Parse a date / year string into a date
# Handles: year-only "1980", full dates like "1980-01-02", "01/02/1980"
# Age calculation only uses the year, so year-only → Jan 1 of that year.
# ---------------------------------------------------------------------------
def parse_date(raw: str) -> date | None:
    if not raw:
        return None
    s = raw.strip()
    # Year-only (4-digit integer)
    if s.isdigit() and len(s) == 4:
        return date(int(s), 1, 1)
    from datetime import datetime
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None

# ---------------------------------------------------------------------------
# Load anon_id → birth_date from the source file
# Accepts column names: "DateOfBirth" or "Date of Birth"
# Accepts id columns: "anon_id" or "case_id"
# ---------------------------------------------------------------------------
def load_birth_dates(filepath: str) -> dict[str, date]:
    if filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath, engine="openpyxl", dtype=str)
        rows = df.to_dict("records")
    else:
        with open(filepath, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))

    result: dict[str, date] = {}
    for row in rows:
        raw_id = (row.get("anon_id") or row.get("case_id") or "").strip()
        anon_id = normalize_anon_id(raw_id)
        if not anon_id:
            continue

        # Accept both column-name conventions
        raw_dob = str(row.get("DateOfBirth") or row.get("Date of Birth") or "").strip()
        if raw_dob.lower() in ("nan", "none", ""):
            continue

        parsed = parse_date(raw_dob)
        if parsed:
            result.setdefault(anon_id, parsed)

    return result

# ---------------------------------------------------------------------------
# Update patients
# ---------------------------------------------------------------------------
async def update_birth_dates(filepath: str) -> None:
    birth_dates = load_birth_dates(filepath)
    print(f"Found {len(birth_dates)} distinct anon_ids with a birth date in the source file")

    DB_URL = os.environ.get("DATABASE_URL")
    if not DB_URL:
        sys.exit("DATABASE_URL env var not set")

    engine = create_async_engine(DB_URL)

    updated = 0
    not_found = 0
    already_set = 0

    async with engine.begin() as conn:
        # Only update patients that exist and don't already have birth_date
        existing = await conn.execute(
            text("SELECT anon_id, birth_date FROM patients WHERE data_source = 'petbert'")
        )
        db_map = {row.anon_id.upper(): row.birth_date for row in existing}
        print(f"{len(db_map)} petbert patients found in DB")

        to_update = []
        for anon_id, dob in birth_dates.items():
            if anon_id not in db_map:
                not_found += 1
                continue
            if db_map[anon_id] is not None:
                already_set += 1
                continue
            to_update.append({"anon_id": anon_id, "birth_date": dob})

        if to_update:
            await conn.execute(
                text("UPDATE patients SET birth_date = :birth_date WHERE anon_id = :anon_id"),
                to_update,
            )
            updated = len(to_update)

    await engine.dispose()

    print(f"\nResult:")
    print(f"  Updated : {updated} patients")
    print(f"  Skipped (already had birth_date): {already_set}")
    print(f"  Not in DB: {not_found}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_birth_dates.py <path-to-csv-or-xlsx>")
        sys.exit(1)
    asyncio.run(update_birth_dates(sys.argv[1]))
