#!/usr/bin/env python3
"""
Ingest pre-classified PetBERT predictions + dog visit demographics into the
VMTH Cancer Registry database.

Expected files in database/data/:
  - petbert_scan_predictions.csv  (columns: anon_id, original_text,
        predicted_term, predicted_group, predicted_code, confidence, method)
  - All_deidentified_K9.xlsx      (columns: anon_id, Sex, Owner Zipcode Zipcode)

Usage:
  python database/seed/ingest_petbert.py

Or via Docker Compose:
  docker compose run --rm ingest
"""

import csv
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql://postgres:postgres@localhost:5432/vmth_cancer",
)

DATA_DIR = Path(os.getenv("DATA_DIR", "/database/data"))
PETBERT_FILE = DATA_DIR / "petbert_scan_predictions.csv"
VISITS_FILE = DATA_DIR / "All_deidentified_K9.xlsx"

SEX_MAP = {
    "M": "Male",
    "F": "Female",
    "FS": "Spayed Female",
    "MC": "Neutered Male",
}

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend" / "app" / "services"))
try:
    from zip_county_service import lookup_county
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend" / "app" / "services"))
    try:
        from zip_county_service import lookup_county
    except ImportError:
        sys.path.insert(0, "/app/app/services")
        from zip_county_service import lookup_county


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_NUMBERED_RE = re.compile(r"\d+\)\s*")


def normalize_anon_id(raw: str) -> str:
    """Canonicalize anon_id so CSV 'ID_37' and Excel '37' or '37.0' match."""
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.isdigit():
        return f"ID_{s}"
    try:
        n = float(s)
        if n == int(n):
            return f"ID_{int(n)}"
        return f"ID_{int(n)}"
    except ValueError:
        pass
    if s.upper().startswith("ID_"):
        suffix = s[3:].strip()
        try:
            return f"ID_{int(float(suffix))}"
        except ValueError:
            return s
    return s


def split_numbered(text: str) -> list[str]:
    """Split '1) foo 2) bar' into ['foo', 'bar'].

    If text has no numbering, returns [text] as a single-item list.
    """
    if not text or not text.strip():
        return []
    if not _NUMBERED_RE.search(text):
        return [text.strip()]
    parts = _NUMBERED_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Parse PetBERT predictions
# ---------------------------------------------------------------------------

def parse_petbert(path: Path) -> dict[str, list[dict]]:
    """Parse PetBERT predictions CSV.

    Each row may contain multiple numbered diagnoses (e.g. "1) ... 2) ...").
    These are split into individual diagnosis records.

    Returns: {anon_id: [
        {"row_index": int, "diagnosis_index": int, "predicted_group": str,
         "predicted_term": str, "icd_o_code": str, "confidence": float,
         "original_text": str, "method": str},
        ...
    ]}
    """
    result: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row_idx, row in enumerate(reader):
            raw_id = row.get("anon_id", "").strip()
            anon_id = normalize_anon_id(raw_id)
            if not anon_id:
                continue

            method = row.get("method", "").strip()
            if method == "empty":
                continue

            original_text = row.get("original_text", "").strip()
            terms = split_numbered(row.get("predicted_term", ""))
            groups = split_numbered(row.get("predicted_group", ""))
            codes = split_numbered(row.get("predicted_code", ""))
            confidences = split_numbered(row.get("confidence", ""))
            methods = split_numbered(method)

            n_diag = max(len(terms), len(groups), len(codes), 1)

            for i in range(n_diag):
                term = terms[i] if i < len(terms) else ""
                group = groups[i] if i < len(groups) else ""
                code = codes[i] if i < len(codes) else ""
                conf_str = confidences[i] if i < len(confidences) else "0"
                meth = methods[i] if i < len(methods) else method

                try:
                    conf = float(conf_str)
                except ValueError:
                    conf = 0.0

                result[anon_id].append({
                    "row_index": row_idx,
                    "diagnosis_index": i + 1,
                    "predicted_group": group,
                    "predicted_term": term,
                    "icd_o_code": code,
                    "confidence": conf,
                    "original_text": original_text,
                    "method": meth,
                })

    return dict(result)


# ---------------------------------------------------------------------------
# Parse dog visits
# ---------------------------------------------------------------------------

def parse_visits(path: Path) -> dict[str, dict]:
    """Parse dog visits xlsx, taking first non-empty sex/zip per anon_id.

    Returns: {anon_id: {"sex": str|None, "zip": str|None}}
    """
    import pandas as pd

    df = pd.read_excel(path, engine="openpyxl", dtype=str)
    df.columns = df.columns.str.strip()

    zip_col = None
    for col in df.columns:
        if "zip" in col.lower():
            zip_col = col
            break

    result: dict[str, dict] = {}
    for _, row in df.iterrows():
        anon_id = normalize_anon_id(row.get("anon_id", ""))
        if not anon_id:
            continue

        raw_sex = str(row.get("Sex", "")).strip().upper()
        if raw_sex == "NAN":
            raw_sex = ""

        raw_zip = str(row.get(zip_col, "")).strip() if zip_col else ""
        if raw_zip == "NAN":
            raw_zip = ""
        raw_zip = raw_zip.split(".")[0]

        if anon_id not in result:
            result[anon_id] = {"sex": None, "zip": None}

        if result[anon_id]["sex"] is None and raw_sex:
            result[anon_id]["sex"] = SEX_MAP.get(raw_sex)

        if result[anon_id]["zip"] is None and raw_zip:
            result[anon_id]["zip"] = raw_zip

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    if not PETBERT_FILE.exists():
        print(f"ERROR: PetBERT file not found at {PETBERT_FILE}")
        sys.exit(1)
    if not VISITS_FILE.exists():
        print(f"ERROR: Dog visits file not found at {VISITS_FILE}")
        sys.exit(1)

    # Ensure PostGIS county geometries are loaded (for geo map)
    conn_check = psycopg2.connect(DATABASE_URL)
    cur_check = conn_check.cursor()
    cur_check.execute("SELECT 1 FROM counties WHERE geom IS NOT NULL LIMIT 1")
    has_geom = cur_check.fetchone() is not None
    cur_check.close()
    conn_check.close()
    if not has_geom:
        print("Loading county boundaries for geospatial maps...")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from county_boundaries import load_boundaries
        load_boundaries()
        print("")

    print("Parsing PetBERT predictions...")
    petbert = parse_petbert(PETBERT_FILE)
    total_diag = sum(len(v) for v in petbert.values())
    print(f"  {total_diag} diagnoses across {len(petbert)} patients")

    print("Parsing dog visits...")
    visits = parse_visits(VISITS_FILE)
    print(f"  {len(visits)} unique patients with demographics")

    matched_ids = sorted(set(petbert.keys()) & set(visits.keys()))
    petbert_only = sorted(set(petbert.keys()) - set(visits.keys()))
    visits_only = sorted(set(visits.keys()) - set(petbert.keys()))
    print(f"\n  Matched (both datasets): {len(matched_ids)}")
    print(f"  PetBERT only (no demographics, skipped): {len(petbert_only)}")
    print(f"  Visits only (no diagnoses, skipped): {len(visits_only)}")

    # Only ingest IDs that appear in both: PetBERT predictions + K9 demographics
    ids_to_process = matched_ids
    print(f"\n  Total patients to ingest: {len(ids_to_process)}")

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    # ------------------------------------------------------------------
    # Load lookups
    # ------------------------------------------------------------------
    cur.execute("SELECT id, name FROM species")
    species_map = {name: id_ for id_, name in cur.fetchall()}
    dog_species_id = species_map.get("Dog")
    if not dog_species_id:
        print("ERROR: 'Dog' species not found. Run migrations first.")
        sys.exit(1)

    cur.execute("SELECT id, name FROM counties")
    county_map = {name: id_ for id_, name in cur.fetchall()}

    cur.execute("SELECT id, name FROM cancer_types")
    cancer_type_map = {name: id_ for id_, name in cur.fetchall()}

    cur.execute("SELECT COALESCE(MAX(id), 0) FROM patients")
    next_patient_id = cur.fetchone()[0] + 1
    cur.execute("SELECT COALESCE(MAX(id), 0) FROM cancer_cases")
    next_case_id = cur.fetchone()[0] + 1

    # ------------------------------------------------------------------
    # Build and insert patients (unchanged)
    # ------------------------------------------------------------------
    patient_rows = []
    warnings = []
    new_cancer_types = set()

    for anon_id in ids_to_process:
        demo = visits.get(anon_id, {})
        sex = demo.get("sex")
        raw_zip = demo.get("zip", "")

        county_name = lookup_county(raw_zip) if raw_zip else None
        county_id = county_map.get(county_name) if county_name else None
        if raw_zip and not county_name:
            warnings.append(f"{anon_id}: zip '{raw_zip}' not in California")

        patient_id = next_patient_id
        next_patient_id += 1

        patient_rows.append((
            patient_id,
            dog_species_id,
            None,           # breed_id
            sex,
            None,           # age_years
            None,           # weight_kg
            county_id,
            None,           # registered_date
            anon_id,
            raw_zip or None,
            "petbert",
        ))

    print(f"\nInserting {len(patient_rows)} patients...")
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
               age_years = EXCLUDED.age_years,
               weight_kg = EXCLUDED.weight_kg,
               county_id = EXCLUDED.county_id,
               registered_date = EXCLUDED.registered_date,
               zip_code = EXCLUDED.zip_code,
               data_source = EXCLUDED.data_source""",
        patient_rows,
    )
    # Ensure every ingested patient is marked petbert (covers any upsert edge cases)
    cur.execute(
        "UPDATE patients SET data_source = 'petbert' WHERE anon_id = ANY(%s)",
        (list(ids_to_process),),
    )
    updated = cur.rowcount
    if updated != len(ids_to_process):
        print(f"  Note: marked data_source=petbert for {updated} patients (expected {len(ids_to_process)})")

    # Resolve actual patient IDs
    cur.execute(
        "SELECT id, anon_id FROM patients WHERE anon_id = ANY(%s)",
        (list(ids_to_process),),
    )
    anon_to_patient_id = {row[1]: row[0] for row in cur.fetchall()}
    patient_ids = [anon_to_patient_id[a] for a in ids_to_process if anon_to_patient_id.get(a)]

    # Get-or-create one registry case per patient (dog)
    cur.execute(
        "SELECT id, patient_id FROM cancer_cases WHERE patient_id = ANY(%s)",
        (patient_ids,),
    )
    # One case per patient: keep the one with highest id (most recent)
    patient_id_to_case_id = {}
    for case_id, pid in cur.fetchall():
        patient_id_to_case_id[pid] = max(patient_id_to_case_id.get(pid, case_id), case_id)

    case_rows = []
    for anon_id in ids_to_process:
        patient_id = anon_to_patient_id.get(anon_id)
        if not patient_id or patient_id in patient_id_to_case_id:
            continue
        demo = visits.get(anon_id, {})
        raw_zip = demo.get("zip", "")
        county_name = lookup_county(raw_zip) if raw_zip else None
        county_id = county_map.get(county_name) if county_name else None
        case_rows.append((
            next_case_id,
            patient_id,
            None, None, None, None, county_id,
            None, None, None, None, None, None, None,
        ))
        patient_id_to_case_id[patient_id] = next_case_id
        next_case_id += 1

    if case_rows:
        print(f"Inserting {len(case_rows)} new registry cases (one per dog)...")
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
        # Resolve case_id for newly inserted (in case of conflict we already have patient_id_to_case_id)
        cur.execute(
            "SELECT id, patient_id FROM cancer_cases WHERE patient_id = ANY(%s)",
            (patient_ids,),
        )
        for case_id, pid in cur.fetchall():
            patient_id_to_case_id[pid] = max(patient_id_to_case_id.get(pid, case_id), case_id)

    # ------------------------------------------------------------------
    # All PetBERT predictions as case_diagnoses (many per case)
    # ------------------------------------------------------------------
    # Replace any existing diagnoses for these cases (idempotent re-run)
    if patient_ids:
        cur.execute(
            "DELETE FROM case_diagnoses WHERE case_id IN "
            "(SELECT id FROM cancer_cases WHERE patient_id = ANY(%s))",
            (patient_ids,),
        )
    diagnosis_rows = []
    for anon_id in ids_to_process:
        patient_id = anon_to_patient_id.get(anon_id)
        case_id = patient_id_to_case_id.get(patient_id) if patient_id else None
        if not case_id:
            continue
        for diag in petbert[anon_id]:
            group = diag["predicted_group"] or "Unknown"
            if group not in cancer_type_map:
                new_cancer_types.add(group)
                cur.execute(
                    "INSERT INTO cancer_types (name) VALUES (%s) "
                    "ON CONFLICT (name) DO NOTHING RETURNING id",
                    (group,),
                )
                result = cur.fetchone()
                if result:
                    cancer_type_map[group] = result[0]
                else:
                    cur.execute("SELECT id FROM cancer_types WHERE name = %s", (group,))
                    cancer_type_map[group] = cur.fetchone()[0]
            cancer_type_id = cancer_type_map[group]
            diagnosis_rows.append((
                case_id,
                cancer_type_id,
                diag["icd_o_code"] or None,
                diag["predicted_term"] or None,
                diag["original_text"] or None,
                round(diag["confidence"], 2) if diag["confidence"] else None,
                diag["method"] or None,
                diag["row_index"],
                diag["diagnosis_index"],
            ))

    if new_cancer_types:
        print(f"\n  Created {len(new_cancer_types)} new cancer types (Vet-ICD-O groups):")
        for ct in sorted(new_cancer_types):
            print(f"    - {ct}")

    print(f"Inserting {len(diagnosis_rows)} cancer diagnoses (predictions per dog)...")
    execute_values(
        cur,
        """INSERT INTO case_diagnoses
               (case_id, cancer_type_id, icd_o_code, predicted_term, original_text,
                confidence, prediction_method, source_row_index, diagnosis_index)
           VALUES %s""",
        diagnosis_rows,
    )

    cur.execute("SELECT setval('patients_id_seq', (SELECT COALESCE(MAX(id), 1) FROM patients))")
    cur.execute("SELECT setval('cancer_cases_id_seq', (SELECT COALESCE(MAX(id), 1) FROM cancer_cases))")
    cur.execute("SELECT setval('case_diagnoses_id_seq', (SELECT COALESCE(MAX(id), 1) FROM case_diagnoses))")

    # ------------------------------------------------------------------
    # Refresh materialized views
    # ------------------------------------------------------------------
    print("Refreshing materialized views...")
    for view in ("mv_county_cancer_incidence", "mv_yearly_trends"):
        try:
            cur.execute(f"REFRESH MATERIALIZED VIEW {view}")
            print(f"  {view} refreshed.")
        except Exception as e:
            print(f"  WARNING: could not refresh {view}: {e}")
            conn.rollback()

    # ------------------------------------------------------------------
    # Log the ingestion
    # ------------------------------------------------------------------
    now = datetime.now(timezone.utc)
    cur.execute(
        """INSERT INTO ingestion_logs
               (dataset_a_filename, dataset_b_filename, started_at, completed_at,
                rows_processed, rows_inserted, rows_skipped, rows_errored, warnings)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)""",
        (
            PETBERT_FILE.name,
            VISITS_FILE.name,
            now,
            now,
            len(diagnosis_rows),
            len(case_rows) + len(diagnosis_rows),
            0,
            0,
            psycopg2.extras.Json(warnings),
        ),
    )

    conn.commit()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    unique_codes = set(r[2] for r in diagnosis_rows if r[2])  # icd_o_code
    print(f"\nDone!")
    print(f"  Patients: {len(patient_rows)}")
    print(f"  Registry cases (one per dog): {len(case_rows)}")
    print(f"  Cancer diagnoses (predictions): {len(diagnosis_rows)}")
    print(f"  Unique ICD-O codes: {len(unique_codes)}")
    if warnings:
        print(f"  Warnings: {len(warnings)}")
        for w in warnings[:10]:
            print(f"    - {w}")
        if len(warnings) > 10:
            print(f"    ... and {len(warnings) - 10} more")

    cur.close()
    conn.close()


if __name__ == "__main__":
    run()
