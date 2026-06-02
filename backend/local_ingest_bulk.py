#!/usr/bin/env python3
"""Bulk ingestion — same structure as local_ingest.py but replaces all ORM
round-trips with chunked multi-row INSERTs.

Accepts an optional diagnoses CSV (e.g. diagnoses.csv) as a third argument.
When provided, its diagnosis text is concatenated per case_id and stored on
pathology_reports.source_diagnosis so reviewers can see the original clinic
diagnoses alongside the PetBERT prediction.

This script preserves existing patients and pathology_reports rows (GCS paths
are kept intact). It only clears case_diagnoses and diagnosis_review_events
before re-inserting predictions.

Usage:
  # Full ingest (patients + source_diagnosis + predictions):
  docker exec vmth_cancer_backend python -u /app/local_ingest_bulk.py \
      /app/petbert_predictions.csv \
      /app/demographics.csv \
      /app/diagnoses.csv          # optional — populates source_diagnosis

  # Fast re-ingest (predictions only — patients/reports already in DB):
  docker exec vmth_cancer_backend python -u /app/local_ingest_bulk.py \
      /app/petbert_predictions.csv --predictions-only
"""

import asyncio
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/app")

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import settings
from app.models.models import (
    Breed, CancerType, CaseDiagnosis, County, DiagnosisReviewEvent,
    Patient, PathologyReport, Species,
)
from app.services.ingestion_service import (
    parse_predictions,
    parse_dataset_a_demographics,
)
from app.services.zip_county_service import lookup_county

PREDICTIONS_ONLY = "--predictions-only" in sys.argv

# Strip flag from positional args for clean indexing
_args = [a for a in sys.argv[1:] if not a.startswith("--")]

PREDICTIONS_PATH = Path(_args[0]) if len(_args) > 0 else Path("/app/petbert_predictions.csv")
DEMOGRAPHICS_PATH = Path(_args[1]) if len(_args) > 1 else Path("/app/demographics.csv")
DIAGNOSES_PATH    = Path(_args[2]) if len(_args) > 2 else None

_CHUNK = 1_000   # rows per bulk INSERT / IN clause


def load_csv_dicts(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_source_diagnosis_map(diagnoses_rows: list[dict]) -> dict[str, str]:
    """Group and concatenate diagnosis text per case_id."""
    by_case: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in diagnoses_rows:
        case_id = (row.get("case_id") or "").strip()
        diagnosis = (row.get("diagnosis") or "").strip()
        if not case_id or not diagnosis:
            continue
        try:
            num = int(row.get("diagnosis_number") or 0)
        except (ValueError, TypeError):
            num = 0
        by_case[case_id].append((num, diagnosis))

    result: dict[str, str] = {}
    for case_id, entries in by_case.items():
        entries.sort(key=lambda x: x[0])
        result[case_id] = "\n".join(text for _, text in entries)
    return result


def build_diag_rows(
    petbert: dict,
    ids_to_process: list[str],
    anon_to_patient_id: dict[str, int],
    anon_to_report_id: dict[str, int],
    cancer_type_map: dict[str, int],
) -> list[dict]:
    conf_threshold   = settings.REVIEW_AUTO_ACCEPT_CONFIDENCE
    margin_threshold = settings.REVIEW_AUTO_ACCEPT_MARGIN

    diag_rows: list[dict] = []
    for anon_id in ids_to_process:
        patient_id = anon_to_patient_id.get(anon_id)
        if not patient_id:
            continue
        preds = petbert[anon_id]
        conf_by_rank = {d["diagnosis_index"]: (d["confidence"] or 0.0) for d in preds}
        top1 = conf_by_rank.get(1, 0.0)
        top2 = conf_by_rank.get(2, 0.0)
        margin = top1 - top2 if top1 and top2 else None
        report_id = anon_to_report_id.get(anon_id)

        for diag in preds:
            group = diag["predicted_group"] or "Unknown"
            cancer_type_id = cancer_type_map.get(group)
            if not cancer_type_id:
                continue
            conf = diag["confidence"] or 0.0
            method = diag["method"]
            rank = diag["diagnosis_index"]
            row_margin = margin if rank == 1 else None

            is_non_cancer = group == "Non-Cancer"
            needs_review = not is_non_cancer and method == "unidentified_cancer"

            diag_rows.append({
                "patient_id": patient_id,
                "cancer_type_id": cancer_type_id,
                "icd_o_code": diag["icd_o_code"] or None,
                "predicted_term": diag["predicted_term"] or None,
                "pathology_report_id": report_id,
                "confidence": round(conf, 2) if conf is not None else None,
                "prediction_method": method or None,
                "source_row_index": diag["row_index"],
                "diagnosis_index": rank,
                "review_status": "pending" if needs_review else "confirmed",
                "top2_margin": round(row_margin, 2) if row_margin is not None else None,
                "ingestion_job_id": None,
                "_needs_review": needs_review,
                "_cancer_type_id": cancer_type_id,
                "_icd_o_code": diag["icd_o_code"] or None,
            })
    return diag_rows


async def insert_diagnoses(db, diag_rows: list[dict]) -> list[tuple]:
    """Bulk insert case_diagnoses, return (id, needs_review, ct_id, icd) tuples."""
    diag_meta = []
    clean_keys = [k for k in diag_rows[0].keys() if not k.startswith("_")] if diag_rows else []
    for i in range(0, len(diag_rows), _CHUNK):
        chunk = diag_rows[i : i + _CHUNK]
        clean_chunk = [{k: row[k] for k in clean_keys} for row in chunk]
        result = await db.execute(
            pg_insert(CaseDiagnosis.__table__)
            .values(clean_chunk)
            .returning(CaseDiagnosis.__table__.c.id)
        )
        for idx, (diag_id,) in enumerate(result.fetchall()):
            r = chunk[idx]
            diag_meta.append((diag_id, r["_needs_review"], r["_cancer_type_id"], r["_icd_o_code"]))
    await db.commit()
    await db.execute(text("SET statement_timeout = 0"))
    return diag_meta


async def insert_review_events(db, diag_meta: list[tuple]) -> list[dict]:
    review_rows = [
        {
            "case_diagnosis_id": diag_id,
            "actor_email": "system",
            "action": "flagged",
            "from_status": None,
            "to_status": "pending",
            "cancer_type_id_after": ct_id,
            "icd_o_code_after": icd,
        }
        for diag_id, needs_review, ct_id, icd in diag_meta
        if needs_review
    ]
    if review_rows:
        for i in range(0, len(review_rows), _CHUNK):
            await db.execute(
                pg_insert(DiagnosisReviewEvent.__table__).values(review_rows[i : i + _CHUNK])
            )
        await db.commit()
        await db.execute(text("SET statement_timeout = 0"))
    return review_rows


async def refresh_views(db, warnings: list[str]):
    for view in ("mv_county_cancer_incidence", "mv_yearly_trends"):
        try:
            await db.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}"))
            print(f"  Refreshed {view}.")
        except Exception as e:
            warnings.append(f"Could not refresh {view}: {e}")
    await db.commit()


async def predictions_only_ingest(db, petbert, ids_to_process, warnings):
    """Fast path: load existing IDs from DB, skip all patient/demo work."""
    pr_table = PathologyReport.__table__

    # Load cancer type map
    cancer_type_map = {
        name: id_
        for id_, name in (await db.execute(select(CancerType.id, CancerType.name))).fetchall()
    }

    # Ensure Non-Cancer type exists
    if "Non-Cancer" not in cancer_type_map:
        r = await db.execute(
            pg_insert(CancerType.__table__).values(name="Non-Cancer")
            .on_conflict_do_nothing(index_elements=["name"])
            .returning(CancerType.__table__.c.id)
        )
        row = r.fetchone()
        if row:
            cancer_type_map["Non-Cancer"] = row[0]
        await db.commit()

    # Ensure Unidentified Group type exists
    if "Unidentified Group" not in cancer_type_map:
        r = await db.execute(
            pg_insert(CancerType.__table__).values(name="Unidentified Group")
            .on_conflict_do_nothing(index_elements=["name"])
            .returning(CancerType.__table__.c.id)
        )
        row = r.fetchone()
        if row:
            cancer_type_map["Unidentified Group"] = row[0]
        await db.commit()

    # Resolve anon_id → patient.id from existing patients
    print("Loading existing patient IDs...")
    anon_to_patient_id: dict[str, int] = {}
    for i in range(0, len(ids_to_process), _CHUNK):
        chunk = ids_to_process[i : i + _CHUNK]
        rows = (await db.execute(
            select(Patient.id, Patient.anon_id).where(Patient.anon_id.in_(chunk))
        )).fetchall()
        anon_to_patient_id.update({r.anon_id: r.id for r in rows})
    print(f"  {len(anon_to_patient_id)} patients found.")

    missing = len(ids_to_process) - len(anon_to_patient_id)
    if missing:
        warnings.append(f"{missing} anon_ids not found in patients table — run full ingest first")

    # Resolve patient_id → pathology_report.id
    print("Loading existing pathology report IDs...")
    patient_to_anon = {v: k for k, v in anon_to_patient_id.items()}
    anon_to_report_id: dict[str, int] = {}
    all_pids = list(anon_to_patient_id.values())
    for i in range(0, len(all_pids), _CHUNK):
        chunk_pids = all_pids[i : i + _CHUNK]
        rows = (await db.execute(
            select(pr_table.c.id, pr_table.c.patient_id)
            .where(pr_table.c.patient_id.in_(chunk_pids))
        )).fetchall()
        for report_id, patient_id in rows:
            anon_id = patient_to_anon.get(patient_id)
            if anon_id:
                anon_to_report_id[anon_id] = report_id
    print(f"  {len(anon_to_report_id)} pathology reports found.")

    return anon_to_patient_id, anon_to_report_id, cancer_type_map


async def main():
    print(f"Predictions : {PREDICTIONS_PATH}")
    if PREDICTIONS_ONLY:
        print("Mode        : predictions-only (fast path)")
    else:
        print(f"Demographics: {DEMOGRAPHICS_PATH}")

    if not PREDICTIONS_PATH.exists():
        sys.exit(f"ERROR: {PREDICTIONS_PATH} not found")
    if not PREDICTIONS_ONLY and not DEMOGRAPHICS_PATH.exists():
        sys.exit(f"ERROR: {DEMOGRAPHICS_PATH} not found")

    use_diagnoses = not PREDICTIONS_ONLY and DIAGNOSES_PATH is not None and DIAGNOSES_PATH.exists()
    if not PREDICTIONS_ONLY:
        if use_diagnoses:
            print(f"Diagnoses   : {DIAGNOSES_PATH}")
        else:
            print(f"Diagnoses   : (not provided, source_diagnosis will not be updated)")

    raw_predictions = load_csv_dicts(PREDICTIONS_PATH)
    print(f"Loaded {len(raw_predictions)} prediction rows")

    source_diagnosis_map: dict[str, str] = {}
    if use_diagnoses:
        source_diagnosis_map = build_source_diagnosis_map(load_csv_dicts(DIAGNOSES_PATH))
        print(f"Loaded source diagnoses for {len(source_diagnosis_map)} cases")

    petbert = parse_predictions(raw_predictions)
    ids_to_process = sorted(petbert.keys())
    print(f"Unique patients: {len(ids_to_process)}")

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    warnings: list[str] = []

    async with Session() as db:
        await db.execute(text("SET statement_timeout = 0"))

        # --- Clear diagnosis data ---
        print("Clearing diagnosis data...")
        await db.execute(text("DELETE FROM diagnosis_review_events"))
        await db.execute(text("DELETE FROM case_diagnoses"))
        await db.commit()
        await db.execute(text("SET statement_timeout = 0"))
        print("Diagnosis data cleared.")

        if PREDICTIONS_ONLY:
            # --- Fast path ---
            anon_to_patient_id, anon_to_report_id, cancer_type_map = \
                await predictions_only_ingest(db, petbert, ids_to_process, warnings)
            patient_values_len = len(anon_to_patient_id)
            source_diag_updated = 0
        else:
            # --- Full path ---
            demographics_bytes = DEMOGRAPHICS_PATH.read_bytes()
            demographics = parse_dataset_a_demographics(demographics_bytes)

            sp_rows = (await db.execute(select(Species.id, Species.name))).fetchall()
            species_map = {name: id_ for id_, name in sp_rows}
            dog_id = species_map.get("Dog")
            if not dog_id:
                r = await db.execute(
                    pg_insert(Species.__table__).values(name="Dog").returning(Species.__table__.c.id)
                )
                dog_id = r.scalar()
                species_map["Dog"] = dog_id
                await db.commit()
                await db.execute(text("SET statement_timeout = 0"))

            county_map = {
                name: id_
                for id_, name in (await db.execute(select(County.id, County.name))).fetchall()
            }
            cancer_type_map = {
                name: id_
                for id_, name in (await db.execute(select(CancerType.id, CancerType.name))).fetchall()
            }
            breed_rows = (await db.execute(select(Breed.id, Breed.species_id, Breed.name))).fetchall()
            breed_map: dict[tuple[int, str], int] = {(r.species_id, r.name): r.id for r in breed_rows}

            unique_sp = {(demographics.get(a) or {}).get("species") or "" for a in ids_to_process} - {""}
            for name in unique_sp - set(species_map):
                await db.execute(pg_insert(Species.__table__).values(name=name).on_conflict_do_nothing(index_elements=["name"]))
            await db.commit()
            await db.execute(text("SET statement_timeout = 0"))
            sp_rows = (await db.execute(select(Species.id, Species.name))).fetchall()
            species_map = {name: id_ for id_, name in sp_rows}
            dog_id = species_map.get("Dog", dog_id)

            unique_breed_keys: set[tuple[int, str]] = set()
            for anon_id in ids_to_process:
                demo = demographics.get(anon_id) or {}
                sp_name = demo.get("species") or ""
                breed_name = demo.get("breed") or ""
                if breed_name:
                    sp_id = species_map.get(sp_name, dog_id) if sp_name else dog_id
                    unique_breed_keys.add((sp_id, breed_name))
            for sp_id, name in unique_breed_keys - set(breed_map):
                await db.execute(pg_insert(Breed.__table__).values(species_id=sp_id, name=name).on_conflict_do_nothing())
            await db.commit()
            await db.execute(text("SET statement_timeout = 0"))
            breed_rows = (await db.execute(select(Breed.id, Breed.species_id, Breed.name))).fetchall()
            breed_map = {(r.species_id, r.name): r.id for r in breed_rows}

            unique_groups = {d["predicted_group"] or "Unknown" for diags in petbert.values() for d in diags}
            for name in unique_groups - set(cancer_type_map):
                await db.execute(pg_insert(CancerType.__table__).values(name=name).on_conflict_do_nothing(index_elements=["name"]))
            await db.commit()
            await db.execute(text("SET statement_timeout = 0"))
            cancer_type_map = {
                name: id_
                for id_, name in (await db.execute(select(CancerType.id, CancerType.name))).fetchall()
            }

            warned_zips: set[str] = set()
            patient_values: list[dict] = []
            for anon_id in ids_to_process:
                demo = demographics.get(anon_id) or {}
                sp_name = demo.get("species") or ""
                sp_id = species_map.get(sp_name, dog_id) if sp_name else dog_id
                breed_name = demo.get("breed") or ""
                breed_id = breed_map.get((sp_id, breed_name)) if breed_name else None
                raw_zip = demo.get("zip") or ""
                county_name = lookup_county(raw_zip) if raw_zip else None
                county_id = county_map.get(county_name) if county_name else None
                if raw_zip and not county_name and raw_zip not in warned_zips:
                    warned_zips.add(raw_zip)
                    warnings.append(f"zip '{raw_zip}' not in California (first: {anon_id})")
                patient_values.append({
                    "anon_id": anon_id, "species_id": sp_id, "breed_id": breed_id,
                    "sex": demo.get("sex"), "county_id": county_id,
                    "zip_code": raw_zip or None, "data_source": "petbert",
                    "diagnosis_date": demo.get("diagnosis_date"), "outcome": None,
                })

            print(f"Upserting {len(patient_values)} patients...")
            for i in range(0, len(patient_values), _CHUNK):
                chunk = patient_values[i : i + _CHUNK]
                ins = pg_insert(Patient.__table__).values(chunk)
                await db.execute(ins.on_conflict_do_update(
                    index_elements=["anon_id"],
                    set_={col: ins.excluded[col] for col in [
                        "species_id", "breed_id", "sex", "county_id",
                        "zip_code", "data_source", "diagnosis_date",
                    ]},
                ))
            await db.commit()
            await db.execute(text("SET statement_timeout = 0"))
            print(f"  {len(patient_values)} patients upserted.")

            anon_to_patient_id: dict[str, int] = {}
            for i in range(0, len(ids_to_process), _CHUNK):
                chunk = ids_to_process[i : i + _CHUNK]
                rows = (await db.execute(
                    select(Patient.id, Patient.anon_id).where(Patient.anon_id.in_(chunk))
                )).fetchall()
                anon_to_patient_id.update({r.anon_id: r.id for r in rows})

            pr_table = PathologyReport.__table__
            patient_to_anon = {v: k for k, v in anon_to_patient_id.items()}
            anon_to_report_id: dict[str, int] = {}
            for i in range(0, len(list(anon_to_patient_id.values())), _CHUNK):
                chunk_pids = list(anon_to_patient_id.values())[i : i + _CHUNK]
                rows = (await db.execute(
                    select(pr_table.c.id, pr_table.c.patient_id)
                    .where(pr_table.c.patient_id.in_(chunk_pids))
                )).fetchall()
                for report_id, patient_id in rows:
                    anon_id = patient_to_anon.get(patient_id)
                    if anon_id:
                        anon_to_report_id[anon_id] = report_id
            print(f"  {len(anon_to_report_id)} existing pathology_reports found.")

            source_diag_updated = 0
            if source_diagnosis_map:
                print("Updating source_diagnosis on pathology_reports...")
                update_pairs = [
                    (anon_to_report_id[anon_id], src)
                    for anon_id, src in source_diagnosis_map.items()
                    if anon_id in anon_to_report_id
                ]
                for i in range(0, len(update_pairs), _CHUNK):
                    for report_id, src in update_pairs[i : i + _CHUNK]:
                        await db.execute(
                            update(PathologyReport).where(PathologyReport.id == report_id).values(source_diagnosis=src)
                        )
                if update_pairs:
                    await db.commit()
                    await db.execute(text("SET statement_timeout = 0"))
                source_diag_updated = len(update_pairs)
                print(f"  {source_diag_updated} pathology_reports updated with source_diagnosis.")

            patient_values_len = len(patient_values)

        # --- Build + insert diagnoses (shared by both paths) ---
        print("Building diagnoses...")
        diag_rows = build_diag_rows(petbert, ids_to_process, anon_to_patient_id, anon_to_report_id, cancer_type_map)
        print(f"  {len(diag_rows)} diagnosis rows built.")

        print("Inserting diagnoses...")
        diag_meta = await insert_diagnoses(db, diag_rows)
        print(f"  {len(diag_meta)} diagnoses inserted.")

        review_rows = await insert_review_events(db, diag_meta)
        if review_rows:
            print(f"  {len(review_rows)} review events inserted.")

        await db.execute(text(
            "SELECT setval('patients_id_seq', (SELECT COALESCE(MAX(id), 1) FROM patients))"
        ))
        await db.execute(text(
            "SELECT setval('case_diagnoses_id_seq', (SELECT COALESCE(MAX(id), 1) FROM case_diagnoses))"
        ))

        await refresh_views(db, warnings)

    await engine.dispose()

    print(f"\nDone.")
    if not PREDICTIONS_ONLY:
        print(f"  Patients             : {patient_values_len}")
        print(f"  Source diag updated  : {source_diag_updated}")
    print(f"  Diagnoses            : {len(diag_rows)}")
    print(f"  Flagged              : {len(review_rows)}")
    print(f"  Warnings             : {len(warnings)}")
    for w in warnings[:20]:
        print(f"    - {w}")
    if len(warnings) > 20:
        print(f"    ... and {len(warnings) - 20} more")


if __name__ == "__main__":
    asyncio.run(main())
