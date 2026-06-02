"""Ingestion service — async SQLAlchemy implementation for API-based upload.

Ported from database/seed/ingest_petbert.py to work with async sessions.
Handles: parsing ML worker predictions, parsing demographics CSV,
upserting patients, upserting cancer_types, inserting case_diagnoses,
and logging the ingestion run.
"""

import asyncio
import csv
import io
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.models import (
    Breed,
    CancerType,
    CaseDiagnosis,
    County,
    DiagnosisReviewEvent,
    IngestionLog,
    PathologyReport,
    Patient,
    Species,
)
from app.schemas.schemas import IngestionResponse, IngestionRowResult
from app.services.zip_county_service import lookup_county

# asyncpg hard limit: 32 767 bind parameters per statement.
# Use 1 000 per chunk to stay well under the cap.
_IN_CHUNK = 1_000

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
        return f"ID_{int(float(s))}"
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

    Accepts two formats:
    - Per-row format: one row per diagnosis rank with an explicit numeric
      ``diagnosis_index`` column (e.g. CSV output from the pipeline).
    - Numbered-string format: one row per patient with values like
      ``"1) Lymphoma 2) MCT"`` split across fields (ml-worker JSON output).

    Accepts ``case_id`` as an alias for ``anon_id``.

    Returns: {anon_id: [{"row_index": ..., "diagnosis_index": ..., ...}, ...]}
    """
    result: dict[str, list[dict]] = defaultdict(list)

    for row_idx, row in enumerate(predictions):
        raw_id = (row.get("anon_id") or row.get("case_id") or "").strip()
        anon_id = normalize_anon_id(raw_id)
        if not anon_id:
            continue

        method = row.get("method", "").strip()
        if method == "empty":
            continue

        original_text = row.get("original_text", "").strip()

        # Detect per-row format: explicit integer diagnosis_index AND no
        # numbered strings in predicted_term/predicted_group.
        raw_di = str(row.get("diagnosis_index", "")).strip()
        has_numbered = bool(
            _NUMBERED_RE.search(row.get("predicted_term", "") or "")
            or _NUMBERED_RE.search(row.get("predicted_group", "") or "")
        )
        if raw_di.isdigit() and not has_numbered:
            conf_str = str(row.get("confidence", "0")).strip()
            try:
                conf = float(conf_str)
            except ValueError:
                conf = 0.0
            result[anon_id].append({
                "row_index": row_idx,
                "diagnosis_index": int(raw_di),
                "predicted_group": (row.get("predicted_group") or "").strip(),
                "predicted_term": (row.get("predicted_term") or "").strip(),
                "icd_o_code": (row.get("predicted_code") or "").strip(),
                "confidence": conf,
                "original_text": original_text,
                "method": method,
            })
            continue

        # Numbered-string format: split "1) foo 2) bar" fields per rank.
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

# ---------------------------------------------------------------------------
# Parse Dataset A demographics
# ---------------------------------------------------------------------------

def _parse_date(raw: str):
    """Parse dates into a Python date. Supports YYYY-MM-DD and 8-Jan-25 formats."""
    if not raw or not raw.strip():
        return None
    s = raw.strip()
    for fmt in ("%Y-%m-%d", "%d-%b-%y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _clean_zip(raw: str) -> str:
    """Normalize a raw zip cell: strip NA/NaN, drop trailing '.0', keep empty string on miss."""
    s = str(raw or "").strip()
    if s.lower() in ("", "na", "nan"):
        return ""
    return s.split(".")[0]


def parse_dataset_a_demographics(csv_bytes: bytes) -> dict[str, dict]:
    """Parse Dataset A CSV for demographic columns.

    Accepts two column-name conventions for the same fields:
      - Patient ID:  ``anon_id``  or  ``case_id``
      - Primary zip: ``Zipcode Zipcode``  or  ``Zipcode``
      - Referral zip: ``RfrrVtrn Zipcode Zipcode``  or  ``RfrrVtrnZipcode``

    Extracts: sex, breed, diagnosis_date, species, zip per patient.
    Takes first non-empty value per patient (idempotent on duplicate rows).

    Zip preference: primary zip first; falls back to referral zip when missing.

    Returns: {anon_id: {"sex": str|None, "breed": str|None,
                         "diagnosis_date": date|None, "birth_date": date|None,
                         "species": str|None, "zip": str|None}}
    """
    text = csv_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))

    result: dict[str, dict] = {}
    for row in reader:
        raw_id = (row.get("anon_id") or row.get("case_id") or "").strip()
        anon_id = normalize_anon_id(raw_id)
        if not anon_id:
            continue

        raw_sex = str(row.get("Sex", "")).strip().upper()
        if raw_sex == "NAN" or raw_sex == "":
            raw_sex = ""

        raw_breed = str(row.get("Breed", "")).strip()
        if raw_breed.lower() == "nan":
            raw_breed = ""

        raw_date = str(row.get("DtOfRq", "")).strip()
        if raw_date.lower() == "nan":
            raw_date = ""

        raw_birth_date = str(row.get("Date of Birth", "")).strip()
        if raw_birth_date.lower() in ("nan", ""):
            raw_birth_date = ""

        raw_species = str(row.get("Species", "")).strip()
        if raw_species.lower() == "nan":
            raw_species = ""

        primary_zip = _clean_zip(row.get("Zipcode Zipcode") or row.get("Zipcode") or "")
        referral_zip = _clean_zip(
            row.get("RfrrVtrn Zipcode Zipcode") or row.get("RfrrVtrnZipcode") or ""
        )
        raw_zip = primary_zip or referral_zip

        if anon_id not in result:
            result[anon_id] = {
                "sex": None,
                "breed": None,
                "diagnosis_date": None,
                "birth_date": None,
                "species": None,
                "zip": None,
            }

        if result[anon_id]["sex"] is None and raw_sex:
            result[anon_id]["sex"] = SEX_MAP.get(raw_sex)

        if result[anon_id]["breed"] is None and raw_breed:
            result[anon_id]["breed"] = raw_breed

        if result[anon_id]["diagnosis_date"] is None and raw_date:
            result[anon_id]["diagnosis_date"] = _parse_date(raw_date)

        if result[anon_id]["birth_date"] is None and raw_birth_date:
            result[anon_id]["birth_date"] = _parse_date(raw_birth_date)

        if result[anon_id]["species"] is None and raw_species:
            result[anon_id]["species"] = raw_species

        if result[anon_id]["zip"] is None and raw_zip:
            result[anon_id]["zip"] = raw_zip

    return result


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

async def ingest_upload(
    db: AsyncSession,
    predictions: list[dict],
    dataset_a_filename: str,
    dataset_a_csv: Optional[bytes] = None,
    ingestion_job_id: int | None = None,
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

    # --- Parse Dataset A demographics ---
    demographics: dict[str, dict] = {}
    if dataset_a_csv:
        demographics = parse_dataset_a_demographics(dataset_a_csv)

    # --- Determine which IDs to process ---
    # Process all patients with predictions; demographics are optional enrichment
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
        dog = Species(name="Dog")
        db.add(dog)
        await db.flush()
        dog_species_id = dog.id
        species_map["Dog"] = dog_species_id

    county_result = await db.execute(select(County.id, County.name))
    county_map = {name: id_ for id_, name in county_result.fetchall()}

    cancer_type_result = await db.execute(select(CancerType.id, CancerType.name))
    cancer_type_map = {name: id_ for id_, name in cancer_type_result.fetchall()}

    # Pre-load breed lookup: (species_id, name) → id
    breed_result = await db.execute(select(Breed.id, Breed.species_id, Breed.name))
    breed_map: dict[tuple[int, str], int] = {
        (row.species_id, row.name): row.id for row in breed_result.fetchall()
    }

    # --- Pre-create all missing species in bulk ---
    unique_species_names: set[str] = {
        (demographics.get(a) or {}).get("species") or ""
        for a in ids_to_process
    } - {""}
    new_species = unique_species_names - set(species_map)
    if new_species:
        for sp_name in new_species:
            db.add(Species(name=sp_name))
        await db.flush()
        sp_result = await db.execute(select(Species.id, Species.name))
        species_map = {name: id_ for id_, name in sp_result.fetchall()}
        dog_species_id = species_map.get("Dog", dog_species_id)

    # --- Pre-create all missing breeds in bulk ---
    unique_breed_keys: set[tuple[int, str]] = set()
    for anon_id in ids_to_process:
        demo = demographics.get(anon_id) or {}
        sp_name = demo.get("species") or ""
        breed_name = demo.get("breed") or ""
        if breed_name:
            sp_id = species_map.get(sp_name, dog_species_id) if sp_name else dog_species_id
            unique_breed_keys.add((sp_id, breed_name))
    new_breeds = unique_breed_keys - set(breed_map)
    if new_breeds:
        for sp_id, breed_name in new_breeds:
            db.add(Breed(species_id=sp_id, name=breed_name))
        await db.flush()
        breed_result2 = await db.execute(select(Breed.id, Breed.species_id, Breed.name))
        breed_map = {(row.species_id, row.name): row.id for row in breed_result2.fetchall()}

    # --- Bulk upsert patients in chunks ---
    # Build one row per patient; deduplicate zip warnings by zip code.
    warned_zips: set[str] = set()
    patient_values: list[dict] = []
    for anon_id in ids_to_process:
        demo = demographics.get(anon_id) or {}
        sex = demo.get("sex")
        raw_zip = demo.get("zip") or ""
        breed_name = demo.get("breed") or ""
        diagnosis_date = demo.get("diagnosis_date")
        birth_date = demo.get("birth_date")
        sp_name = demo.get("species") or ""

        sp_id = species_map.get(sp_name, dog_species_id) if sp_name else dog_species_id
        breed_id = breed_map.get((sp_id, breed_name)) if breed_name else None
        county_name = lookup_county(raw_zip) if raw_zip else None
        county_id = county_map.get(county_name) if county_name else None
        if raw_zip and not county_name and raw_zip not in warned_zips:
            warned_zips.add(raw_zip)
            warnings.append(f"zip '{raw_zip}' not in California (first patient: {anon_id})")

        patient_values.append({
            "anon_id": anon_id,
            "species_id": sp_id,
            "breed_id": breed_id,
            "sex": sex,
            "county_id": county_id,
            "zip_code": raw_zip or None,
            "data_source": "petbert",
            "birth_date": birth_date,
            "diagnosis_date": diagnosis_date,
            "outcome": None,
        })

    for i in range(0, len(patient_values), _IN_CHUNK):
        chunk = patient_values[i : i + _IN_CHUNK]
        ins_stmt = pg_insert(Patient.__table__).values(chunk)
        await db.execute(ins_stmt.on_conflict_do_update(
            index_elements=["anon_id"],
            set_={col: ins_stmt.excluded[col] for col in [
                "species_id", "breed_id", "sex", "county_id",
                "zip_code", "data_source", "birth_date", "diagnosis_date",
            ]},
        ))
    patients_inserted = len(patient_values)

    await db.flush()

    # Resolve actual patient IDs (chunked to stay under asyncpg 32 767 param limit)
    all_patient_rows = []
    for i in range(0, len(ids_to_process), _IN_CHUNK):
        chunk = ids_to_process[i : i + _IN_CHUNK]
        result = await db.execute(
            select(Patient.id, Patient.anon_id).where(Patient.anon_id.in_(chunk))
        )
        all_patient_rows.extend(result.fetchall())
    anon_to_patient_id = {row.anon_id: row.id for row in all_patient_rows}

    patient_ids = [anon_to_patient_id[a] for a in ids_to_process if a in anon_to_patient_id]

    # --- Delete existing diagnoses for these patients from THIS JOB ONLY ---
    # Scoping to ingestion_job_id preserves data from other jobs when the same
    # patient appears in multiple uploads, making each job independently idempotent.
    if patient_ids:
        for i in range(0, len(patient_ids), _IN_CHUNK):
            chunk = patient_ids[i : i + _IN_CHUNK]
            if ingestion_job_id is not None:
                await db.execute(
                    delete(CaseDiagnosis).where(
                        CaseDiagnosis.patient_id.in_(chunk),
                        CaseDiagnosis.ingestion_job_id == ingestion_job_id,
                    )
                )
            else:
                await db.execute(
                    delete(CaseDiagnosis).where(CaseDiagnosis.patient_id.in_(chunk))
                )
        # Clean up PathologyReport rows that no CaseDiagnosis references anymore.
        referenced_report_ids = (
            select(CaseDiagnosis.pathology_report_id)
            .where(CaseDiagnosis.pathology_report_id.is_not(None))
            .scalar_subquery()
        )
        for i in range(0, len(patient_ids), _IN_CHUNK):
            chunk = patient_ids[i : i + _IN_CHUNK]
            await db.execute(
                delete(PathologyReport).where(
                    PathologyReport.patient_id.in_(chunk),
                    PathologyReport.id.not_in(referenced_report_ids),
                )
            )

    # --- Upload report texts to GCS, then create one PathologyReport per patient ---
    # Collect anon_id → text pairs for patients that have text.
    text_by_anon_id: dict[str, str] = {}
    for anon_id in ids_to_process:
        report_text = next(
            (d["original_text"] for d in petbert.get(anon_id, []) if d.get("original_text")),
            None,
        )
        if report_text:
            text_by_anon_id[anon_id] = report_text

    # Upload to GCS whenever a bucket is configured — works for both GCP Batch
    # and local ml-worker runs as long as GCS credentials are available.
    # Falls back to None (no GCS path) only when GCS is not configured at all.
    gcs_path_by_anon_id: dict[str, str] = {}
    if settings.GCS_BUCKET and ingestion_job_id and text_by_anon_id:
        from app.services.gcp_batch_service import upload_report_text_to_gcs
        loop = asyncio.get_running_loop()
        _UPLOAD_CHUNK = 50  # limit concurrent GCS connections

        items = list(text_by_anon_id.items())
        for i in range(0, len(items), _UPLOAD_CHUNK):
            chunk = items[i : i + _UPLOAD_CHUNK]
            results = await asyncio.gather(*[
                loop.run_in_executor(
                    None, upload_report_text_to_gcs, ingestion_job_id, anon_id, txt
                )
                for anon_id, txt in chunk
            ], return_exceptions=True)
            for (anon_id, _), gcs_path in zip(chunk, results):
                if isinstance(gcs_path, Exception):
                    logger.warning("GCS upload failed for %s: %s", anon_id, gcs_path)
                else:
                    gcs_path_by_anon_id[anon_id] = gcs_path

    report_by_anon_id: dict[str, PathologyReport] = {}
    for anon_id in ids_to_process:
        patient_id = anon_to_patient_id.get(anon_id)
        if not patient_id or anon_id not in text_by_anon_id:
            continue
        demo = demographics.get(anon_id, {})
        report = PathologyReport(
            patient_id=patient_id,
            gcs_path=gcs_path_by_anon_id.get(anon_id),
            report_date=demo.get("diagnosis_date"),
        )
        db.add(report)
        report_by_anon_id[anon_id] = report

    await db.flush()  # populates report IDs before diagnoses reference them

    # --- Upsert cancer_types & insert case_diagnoses ---
    diagnoses_inserted = 0
    for anon_id in ids_to_process:
        patient_id = anon_to_patient_id.get(anon_id)
        if not patient_id:
            continue

        # Top-1 vs top-2 margin is computed once per case_id and only
        # attached to the rank-1 row (it is not meaningful for lower ranks).
        conf_by_rank = {d["diagnosis_index"]: (d["confidence"] or 0.0) for d in petbert[anon_id]}
        top1_conf = conf_by_rank.get(1, 0.0)
        top2_conf = conf_by_rank.get(2, 0.0)
        margin = top1_conf - top2_conf if top1_conf and top2_conf else None

        for diag in petbert[anon_id]:
            group = diag["predicted_group"] or "Unknown"

            if group not in cancer_type_map:
                # Upsert cancer type. PetBERT-introduced types are accepted
                # as confirmed; reviewer-introduced types go in unconfirmed
                # (see review router).
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

            conf = diag["confidence"] or 0.0
            method = diag["method"]
            rank = diag["diagnosis_index"]

            # Review gate: flag pending when confidence is below the auto-
            # accept threshold OR (rank-1 only) the margin to the next
            # candidate is too tight OR the pipeline already labelled the
            # row low_confidence.
            row_margin = margin if rank == 1 else None
            margin_too_tight = (
                row_margin is not None
                and row_margin < settings.REVIEW_AUTO_ACCEPT_MARGIN
            )
            needs_review = (
                method == "low_confidence"
                or conf < settings.REVIEW_AUTO_ACCEPT_CONFIDENCE
                or margin_too_tight
            )
            review_status = "pending" if needs_review else "confirmed"

            report = report_by_anon_id.get(anon_id)
            diagnosis = CaseDiagnosis(
                patient_id=patient_id,
                cancer_type_id=cancer_type_id,
                icd_o_code=diag["icd_o_code"] or None,
                predicted_term=diag["predicted_term"] or None,
                pathology_report_id=report.id if report else None,
                confidence=round(conf, 2) if conf is not None else None,
                prediction_method=method or None,
                source_row_index=diag["row_index"],
                diagnosis_index=rank,
                review_status=review_status,
                top2_margin=round(row_margin, 2) if row_margin is not None else None,
                ingestion_job_id=ingestion_job_id,
            )
            db.add(diagnosis)
            diagnoses_inserted += 1

            if needs_review:
                # Audit-log the auto-flag so the queue can show "flagged at
                # ingest" alongside subsequent reviewer actions.
                db.add(DiagnosisReviewEvent(
                    case_diagnosis=diagnosis,
                    actor_email="system",
                    action="flagged",
                    from_status=None,
                    to_status="pending",
                    cancer_type_id_after=cancer_type_id,
                    icd_o_code_after=diag["icd_o_code"] or None,
                ))

            row_results.append(IngestionRowResult(
                row_number=diag["row_index"],
                anon_id=anon_id,
                status="inserted",
                cancer_type=group,
                confidence=round(conf, 2) if conf is not None else None,
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
    _ALLOWED_VIEWS = frozenset({"mv_county_cancer_incidence", "mv_yearly_trends"})
    for view in _ALLOWED_VIEWS:
        try:
            await db.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}"))
        except Exception as e:
            warnings.append(f"Could not refresh {view}: {e}")

    # --- Log the ingestion ---
    log = IngestionLog(
        dataset_a_filename=dataset_a_filename,
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

    # --- Compute confidence summary ---
    confidences = [r.confidence for r in row_results if r.confidence is not None]
    cancer_counts: dict[str, int] = defaultdict(int)
    for r in row_results:
        if r.cancer_type:
            cancer_counts[r.cancer_type] += 1

    result_summary = {
        "patients": len(ids_to_process),
        "diagnoses": diagnoses_inserted,
        "avg_confidence": round(sum(confidences) / len(confidences) * 100, 1) if confidences else None,
        "high_confidence": sum(1 for c in confidences if c >= 0.8),
        "medium_confidence": sum(1 for c in confidences if 0.5 <= c < 0.8),
        "low_confidence": sum(1 for c in confidences if c < 0.5),
        "top_cancer_types": [
            {"name": name, "count": count}
            for name, count in sorted(cancer_counts.items(), key=lambda x: -x[1])[:5]
        ],
    }

    return IngestionResponse(
        total_rows=len(predictions),
        inserted=diagnoses_inserted,
        skipped=len(predictions) - len(ids_to_process),
        errors=0,
        warnings=warnings,
        row_results=row_results,
        ingestion_log_id=log_id,
        result_summary=result_summary,
    )
