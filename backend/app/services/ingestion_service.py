"""Ingestion service — async SQLAlchemy implementation for API-based upload.

Ported from database/seed/ingest_petbert.py to work with async sessions.
Handles: parsing ML worker predictions, parsing demographics CSV,
upserting patients, upserting cancer_types, inserting case_diagnoses,
and logging the ingestion run.
"""

import csv
import io
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    CancerType,
    CaseDiagnosis,
    County,
    IngestionLog,
    Patient,
    Species,
)
from app.schemas.schemas import IngestionResponse, IngestionRowResult
from app.services.zip_county_service import lookup_county

# ---------------------------------------------------------------------------
# Parsing helpers (ported from ingest_petbert.py)
# ---------------------------------------------------------------------------

SEX_MAP = {
    "M": "Male",
    "F": "Female",
    "FS": "Spayed Female",
    "MC": "Neutered Male",
}

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


def split_numbered(text_val: str) -> list[str]:
    """Split '1) foo 2) bar' into ['foo', 'bar']."""
    if not text_val or not text_val.strip():
        return []
    if not _NUMBERED_RE.search(text_val):
        return [text_val.strip()]
    parts = _NUMBERED_RE.split(text_val.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Parse ML worker predictions
# ---------------------------------------------------------------------------

def parse_predictions(predictions: list[dict]) -> dict[str, list[dict]]:
    """Parse ML worker prediction rows into per-patient diagnosis lists.

    Each prediction row may contain numbered multi-diagnoses like
    '1) term_a 2) term_b'. These are split into individual records.

    Returns: {anon_id: [{"row_index": ..., "diagnosis_index": ..., ...}, ...]}
    """
    result: dict[str, list[dict]] = defaultdict(list)

    for row_idx, row in enumerate(predictions):
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
# Parse demographics CSV
# ---------------------------------------------------------------------------

def parse_demographics_csv(csv_bytes: bytes) -> dict[str, dict]:
    """Parse Dataset B (demographics) CSV.

    Expects columns: anon_id, Sex, and a zip column (any column with 'zip' in name).
    Takes first non-empty sex/zip per anon_id.

    Returns: {anon_id: {"sex": str|None, "zip": str|None}}
    """
    text = csv_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))

    # Find zip column
    fieldnames = reader.fieldnames or []
    zip_col = None
    for col in fieldnames:
        if "zip" in col.lower():
            zip_col = col
            break

    result: dict[str, dict] = {}
    for row in reader:
        anon_id = normalize_anon_id(row.get("anon_id", ""))
        if not anon_id:
            continue

        raw_sex = str(row.get("Sex", "")).strip().upper()
        if raw_sex == "NAN" or raw_sex == "":
            raw_sex = ""

        raw_zip = str(row.get(zip_col, "")).strip() if zip_col else ""
        if raw_zip.lower() == "nan":
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
# Main ingestion
# ---------------------------------------------------------------------------

async def ingest_upload(
    db: AsyncSession,
    predictions: list[dict],
    demographics_csv: Optional[bytes],
    dataset_a_filename: str,
    dataset_b_filename: Optional[str],
) -> IngestionResponse:
    """Ingest ML worker predictions + optional demographics into the database.

    1. Parse predictions into per-patient diagnosis lists
    2. Parse demographics CSV if provided
    3. Match patients across datasets by anon_id
    4. Upsert patients (INSERT ... ON CONFLICT)
    5. Delete existing case_diagnoses for these patients (idempotent re-run)
    6. Upsert cancer_types, insert case_diagnoses
    7. Reset sequences, refresh materialized views
    8. Log to ingestion_logs table
    9. Return IngestionResponse
    """
    started_at = datetime.now(timezone.utc)
    warnings: list[str] = []
    row_results: list[IngestionRowResult] = []

    # --- Parse predictions ---
    petbert = parse_predictions(predictions)
    total_diag = sum(len(v) for v in petbert.values())

    # --- Parse demographics ---
    demographics: dict[str, dict] = {}
    if demographics_csv:
        demographics = parse_demographics_csv(demographics_csv)

    # --- Determine which IDs to process ---
    if demographics:
        ids_to_process = sorted(set(petbert.keys()) & set(demographics.keys()))
        petbert_only = set(petbert.keys()) - set(demographics.keys())
        if petbert_only:
            warnings.append(
                f"{len(petbert_only)} patients in Dataset A have no demographics match (skipped)"
            )
    else:
        # No demographics provided — ingest all prediction IDs
        ids_to_process = sorted(petbert.keys())

    if not ids_to_process:
        return IngestionResponse(
            total_rows=len(predictions),
            inserted=0,
            skipped=len(predictions),
            errors=0,
            warnings=warnings + ["No matching patients found between datasets"],
            row_results=[],
        )

    # --- Load lookups ---
    species_result = await db.execute(select(Species.id, Species.name))
    species_map = {name: id_ for id_, name in species_result.fetchall()}
    dog_species_id = species_map.get("Dog")
    if not dog_species_id:
        # Try to create the Dog species
        dog = Species(name="Dog")
        db.add(dog)
        await db.flush()
        dog_species_id = dog.id

    county_result = await db.execute(select(County.id, County.name))
    county_map = {name: id_ for id_, name in county_result.fetchall()}

    cancer_type_result = await db.execute(select(CancerType.id, CancerType.name))
    cancer_type_map = {name: id_ for id_, name in cancer_type_result.fetchall()}

    # --- Upsert patients ---
    patients_inserted = 0
    for anon_id in ids_to_process:
        demo = demographics.get(anon_id, {})
        sex = demo.get("sex")
        raw_zip = demo.get("zip", "")

        county_name = lookup_county(raw_zip) if raw_zip else None
        county_id = county_map.get(county_name) if county_name else None
        if raw_zip and not county_name:
            warnings.append(f"{anon_id}: zip '{raw_zip}' not in California")

        stmt = pg_insert(Patient.__table__).values(
            species_id=dog_species_id,
            breed_id=None,
            sex=sex,
            county_id=county_id,
            anon_id=anon_id,
            zip_code=raw_zip or None,
            data_source="petbert",
            diagnosis_date=None,
            outcome=None,
        ).on_conflict_do_update(
            index_elements=["anon_id"],
            set_={
                "species_id": dog_species_id,
                "sex": sex,
                "county_id": county_id,
                "zip_code": raw_zip or None,
                "data_source": "petbert",
            },
        )
        await db.execute(stmt)
        patients_inserted += 1

    await db.flush()

    # Resolve actual patient IDs
    patient_result = await db.execute(
        select(Patient.id, Patient.anon_id).where(Patient.anon_id.in_(ids_to_process))
    )
    anon_to_patient_id = {row.anon_id: row.id for row in patient_result.fetchall()}

    patient_ids = [anon_to_patient_id[a] for a in ids_to_process if a in anon_to_patient_id]

    # --- Delete existing case_diagnoses for these patients (idempotent) ---
    if patient_ids:
        await db.execute(
            delete(CaseDiagnosis).where(CaseDiagnosis.patient_id.in_(patient_ids))
        )

    # --- Upsert cancer_types & insert case_diagnoses ---
    diagnoses_inserted = 0
    for anon_id in ids_to_process:
        patient_id = anon_to_patient_id.get(anon_id)
        if not patient_id:
            continue

        for diag in petbert[anon_id]:
            group = diag["predicted_group"] or "Unknown"

            if group not in cancer_type_map:
                # Upsert cancer type
                ct_stmt = pg_insert(CancerType.__table__).values(
                    name=group,
                ).on_conflict_do_nothing(index_elements=["name"])
                await db.execute(ct_stmt)
                await db.flush()

                ct_result = await db.execute(
                    select(CancerType.id).where(CancerType.name == group)
                )
                cancer_type_map[group] = ct_result.scalar_one()

            cancer_type_id = cancer_type_map[group]

            diagnosis = CaseDiagnosis(
                patient_id=patient_id,
                cancer_type_id=cancer_type_id,
                icd_o_code=diag["icd_o_code"] or None,
                predicted_term=diag["predicted_term"] or None,
                confidence=round(diag["confidence"], 2) if diag["confidence"] else None,
                prediction_method=diag["method"] or None,
                source_row_index=diag["row_index"],
                diagnosis_index=diag["diagnosis_index"],
            )
            db.add(diagnosis)
            diagnoses_inserted += 1

            row_results.append(IngestionRowResult(
                row_number=diag["row_index"],
                anon_id=anon_id,
                status="inserted",
                cancer_type=group,
                confidence=round(diag["confidence"], 2) if diag["confidence"] else None,
            ))

    await db.flush()

    # --- Reset sequences ---
    await db.execute(text(
        "SELECT setval('patients_id_seq', (SELECT COALESCE(MAX(id), 1) FROM patients))"
    ))
    await db.execute(text(
        "SELECT setval('case_diagnoses_id_seq', (SELECT COALESCE(MAX(id), 1) FROM case_diagnoses))"
    ))

    # --- Refresh materialized views ---
    for view in ("mv_county_cancer_incidence", "mv_yearly_trends"):
        try:
            await db.execute(text(f"REFRESH MATERIALIZED VIEW {view}"))
        except Exception as e:
            warnings.append(f"Could not refresh {view}: {e}")

    # --- Log the ingestion ---
    log = IngestionLog(
        dataset_a_filename=dataset_a_filename,
        dataset_b_filename=dataset_b_filename,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        rows_processed=total_diag,
        rows_inserted=patients_inserted + diagnoses_inserted,
        rows_skipped=0,
        rows_errored=0,
        warnings=warnings,
    )
    db.add(log)
    await db.flush()

    log_id = log.id

    await db.commit()

    return IngestionResponse(
        total_rows=len(predictions),
        inserted=diagnoses_inserted,
        skipped=len(predictions) - len(ids_to_process),
        errors=0,
        warnings=warnings,
        row_results=row_results,
        ingestion_log_id=log_id,
    )
