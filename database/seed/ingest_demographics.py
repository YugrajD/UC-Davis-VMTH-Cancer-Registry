#!/usr/bin/env python3
"""
Ingest demographics.csv into the VMTH Cancer Registry database.

CSV columns: case_id, DtOfRq, Sex, Species, Breed

Creates:
  - New breed rows (insert-if-missing) for breeds not yet in the DB
  - New patient rows with breed_id, sex, species_id, data_source='demographics'
  - One cancer_case per patient (with diagnosis_date from DtOfRq)

Usage:
  export $(grep -v '^#' .env | xargs)
  python database/seed/ingest_demographics.py
"""

import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values, Json

DATABASE_URL = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql://postgres:postgres@localhost:5432/vmth_cancer",
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEMOGRAPHICS_FILE = DATA_DIR / "demographics.csv"

SEX_MAP = {
    "M": "Male",
    "F": "Female",
    "FS": "Spayed Female",
    "MC": "Neutered Male",
}


def parse_date(raw: str):
    """Parse '8-Jan-25' style dates into a Python date."""
    if not raw or not raw.strip():
        return None
    try:
        return datetime.strptime(raw.strip(), "%d-%b-%y").date()
    except ValueError:
        return None


def title_case_breed(raw: str) -> str:
    """Normalize 'GERMAN SHEPHERD DOG' → 'German Shepherd Dog'."""
    return raw.strip().title()


def run():
    if not DEMOGRAPHICS_FILE.exists():
        print(f"ERROR: demographics.csv not found at {DEMOGRAPHICS_FILE}")
        sys.exit(1)

    # Parse CSV
    rows = []
    with open(DEMOGRAPHICS_FILE, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row["case_id"].strip()
            if not case_id:
                continue
            rows.append({
                "case_id": case_id,
                "date": parse_date(row.get("DtOfRq", "")),
                "sex": SEX_MAP.get(row.get("Sex", "").strip().upper()),
                "breed_raw": row.get("Breed", "").strip(),
            })

    print(f"Parsed {len(rows)} rows from demographics.csv")

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    # Load lookups
    cur.execute("SELECT id, name FROM species")
    species_map = {name: id_ for id_, name in cur.fetchall()}
    dog_species_id = species_map.get("Dog")
    if not dog_species_id:
        print("ERROR: 'Dog' species not found. Run migrations first.")
        sys.exit(1)

    cur.execute("SELECT id, name FROM breeds WHERE species_id = %s", (dog_species_id,))
    breed_map = {name.lower(): id_ for id_, name in cur.fetchall()}

    cur.execute("SELECT COALESCE(MAX(id), 0) FROM patients")
    next_patient_id = cur.fetchone()[0] + 1
    cur.execute("SELECT COALESCE(MAX(id), 0) FROM cancer_cases")
    next_case_id = cur.fetchone()[0] + 1

    # Insert missing breeds
    unique_breeds = set()
    for r in rows:
        if r["breed_raw"]:
            unique_breeds.add(title_case_breed(r["breed_raw"]))

    new_breeds = []
    for breed_name in sorted(unique_breeds):
        if breed_name.lower() not in breed_map:
            new_breeds.append(breed_name)

    if new_breeds:
        print(f"Inserting {len(new_breeds)} new breeds...")
        for breed_name in new_breeds:
            cur.execute(
                "INSERT INTO breeds (species_id, name) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING RETURNING id",
                (dog_species_id, breed_name),
            )
            result = cur.fetchone()
            if result:
                breed_map[breed_name.lower()] = result[0]
            else:
                cur.execute(
                    "SELECT id FROM breeds WHERE species_id = %s AND name = %s",
                    (dog_species_id, breed_name),
                )
                breed_map[breed_name.lower()] = cur.fetchone()[0]

    # Build patient rows
    patient_rows = []
    warnings = []
    for r in rows:
        breed_name = title_case_breed(r["breed_raw"]) if r["breed_raw"] else None
        breed_id = breed_map.get(breed_name.lower()) if breed_name else None

        if breed_name and not breed_id:
            warnings.append(f"{r['case_id']}: breed '{breed_name}' not resolved")

        patient_id = next_patient_id
        next_patient_id += 1

        patient_rows.append((
            patient_id,
            dog_species_id,
            breed_id,
            r["sex"],
            None,           # age_years
            None,           # weight_kg
            None,           # county_id (no ZIP in this CSV)
            r["date"],      # registered_date
            r["case_id"],   # anon_id
            None,           # zip_code
            "demographics",
        ))

    print(f"Inserting {len(patient_rows)} patients...")
    execute_values(
        cur,
        """INSERT INTO patients
               (id, species_id, breed_id, sex, age_years, weight_kg,
                county_id, registered_date, anon_id, zip_code, data_source)
           VALUES %s
           ON CONFLICT (anon_id) WHERE anon_id IS NOT NULL DO UPDATE SET
               species_id = EXCLUDED.species_id,
               breed_id = EXCLUDED.breed_id,
               sex = EXCLUDED.sex,
               registered_date = EXCLUDED.registered_date,
               data_source = EXCLUDED.data_source""",
        patient_rows,
    )

    # Resolve patient IDs
    case_ids = [r["case_id"] for r in rows]
    cur.execute(
        "SELECT id, anon_id FROM patients WHERE anon_id = ANY(%s)",
        (case_ids,),
    )
    anon_to_patient_id = {row[1]: row[0] for row in cur.fetchall()}
    patient_ids = list(anon_to_patient_id.values())

    # Get existing cases for these patients
    cur.execute(
        "SELECT id, patient_id FROM cancer_cases WHERE patient_id = ANY(%s)",
        (patient_ids,),
    )
    patient_id_to_case_id = {pid: cid for cid, pid in cur.fetchall()}

    # Create one cancer_case per patient (if not already present)
    case_rows = []
    for r in rows:
        patient_id = anon_to_patient_id.get(r["case_id"])
        if not patient_id or patient_id in patient_id_to_case_id:
            continue
        case_rows.append((
            next_case_id,
            patient_id,
            None,           # cancer_type_id
            r["date"],      # diagnosis_date
            None,           # stage
            None,           # outcome
            None,           # county_id
            None, None, None, None, None, None, None,
        ))
        patient_id_to_case_id[patient_id] = next_case_id
        next_case_id += 1

    if case_rows:
        print(f"Inserting {len(case_rows)} cancer cases...")
        execute_values(
            cur,
            """INSERT INTO cancer_cases
                   (id, patient_id, cancer_type_id, diagnosis_date, stage, outcome,
                    county_id, source_row_index, diagnosis_index,
                    icd_o_code, predicted_term, original_text, confidence,
                    prediction_method)
               VALUES %s ON CONFLICT DO NOTHING""",
            case_rows,
        )

    # Reset sequences
    cur.execute("SELECT setval('patients_id_seq', (SELECT COALESCE(MAX(id), 1) FROM patients))")
    cur.execute("SELECT setval('cancer_cases_id_seq', (SELECT COALESCE(MAX(id), 1) FROM cancer_cases))")

    # Refresh materialized views (may not exist in Supabase)
    print("Refreshing materialized views...")
    for view in ("mv_county_cancer_incidence", "mv_yearly_trends"):
        try:
            # Use savepoint so a missing view doesn't kill the transaction
            cur.execute("SAVEPOINT mv_refresh")
            cur.execute(f"REFRESH MATERIALIZED VIEW {view}")
            cur.execute("RELEASE SAVEPOINT mv_refresh")
            print(f"  {view} refreshed.")
        except Exception as e:
            cur.execute("ROLLBACK TO SAVEPOINT mv_refresh")
            print(f"  Skipped {view} (not found)")

    # Log ingestion (table may not exist in Supabase)
    now = datetime.now(timezone.utc)
    try:
        cur.execute("SAVEPOINT log_insert")
        cur.execute(
            """INSERT INTO ingestion_logs
                   (dataset_a_filename, started_at, completed_at,
                    rows_processed, rows_inserted, rows_skipped, rows_errored, warnings)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)""",
            (
                DEMOGRAPHICS_FILE.name,
                now, now,
                len(rows),
                len(patient_rows) + len(case_rows),
                0, 0,
                Json(warnings),
            ),
        )
        cur.execute("RELEASE SAVEPOINT log_insert")
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT log_insert")
        print("  Skipped ingestion_logs (table not found)")

    conn.commit()

    print(f"\nDone!")
    print(f"  Patients inserted/updated: {len(patient_rows)}")
    print(f"  Cancer cases created: {len(case_rows)}")
    print(f"  New breeds added: {len(new_breeds)}")
    if warnings:
        print(f"  Warnings: {len(warnings)}")
        for w in warnings[:10]:
            print(f"    - {w}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    run()
