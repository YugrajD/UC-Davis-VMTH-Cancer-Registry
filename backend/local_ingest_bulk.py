#!/usr/bin/env python3
"""Bulk ingestion — same structure as local_ingest.py but replaces all ORM
round-trips with chunked multi-row INSERTs.

Optionally accepts a report CSV (e.g. report.csv) as a third argument.
When provided and GCS is configured, each patient's pathology report text
is uploaded to GCS and linked via pathology_reports.gcs_path — matching
exactly how the UI pipeline stores reports.

Usage:
  docker exec vmth_cancer_backend python -u /app/local_ingest_bulk.py \
      /app/petbert_predictions.csv \
      /app/demographics.csv \
      /app/report.csv          # optional
"""

import asyncio
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, text
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

PREDICTIONS_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/petbert_predictions.csv")
DEMOGRAPHICS_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/app/demographics.csv")
REPORT_PATH      = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/app/report.csv")

_CHUNK        = 1_000   # rows per bulk INSERT / IN clause
_GCS_CHUNK    = 50      # concurrent GCS uploads per batch


def load_csv_dicts(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_report_text(row: dict) -> str:
    """Concatenate all non-empty report sections into a single text blob."""
    sections = [
        "CLINICAL ABSTRACT",
        "GROSS DESCRIPTION",
        "HISTOPATHOLOGICAL SUMMARY",
        "COMMENT",
        "FINAL COMMENT",
        "ANCILLARY TESTS",
        "ADDENDUM",
        "ADDITIONAL INFORMATION",
    ]
    parts = []
    for col in sections:
        val = (row.get(col) or "").strip()
        if val and val.upper() not in ("NA", "N/A"):
            parts.append(f"[{col}]\n{val}")
    return "\n\n".join(parts)


async def upload_reports_to_gcs(
    report_text_by_anon_id: dict[str, str],
    gcs_job_id: int,
) -> dict[str, str]:
    """Upload all reports to GCS in parallel chunks.

    Returns {anon_id: gcs_path} for every successful upload.
    Uses the same path format as the UI pipeline: reports/{job_id}/{anon_id}.txt
    """
    from app.services.gcp_batch_service import upload_report_text_to_gcs

    loop = asyncio.get_running_loop()
    gcs_path_by_anon_id: dict[str, str] = {}
    items = list(report_text_by_anon_id.items())
    total = len(items)

    for i in range(0, total, _GCS_CHUNK):
        chunk = items[i : i + _GCS_CHUNK]
        results = await asyncio.gather(
            *[
                loop.run_in_executor(
                    None, upload_report_text_to_gcs, gcs_job_id, anon_id, txt
                )
                for anon_id, txt in chunk
            ],
            return_exceptions=True,
        )
        for (anon_id, _), result in zip(chunk, results):
            if isinstance(result, Exception):
                print(f"  WARNING: GCS upload failed for {anon_id}: {result}")
            else:
                gcs_path_by_anon_id[anon_id] = result
        done = min(i + _GCS_CHUNK, total)
        print(f"  GCS upload: {done}/{total}", end="\r", flush=True)

    print()  # newline after progress
    return gcs_path_by_anon_id


async def main():
    print(f"Predictions : {PREDICTIONS_PATH}")
    print(f"Demographics: {DEMOGRAPHICS_PATH}")

    if not PREDICTIONS_PATH.exists():
        sys.exit(f"ERROR: {PREDICTIONS_PATH} not found")
    if not DEMOGRAPHICS_PATH.exists():
        sys.exit(f"ERROR: {DEMOGRAPHICS_PATH} not found")

    use_reports = REPORT_PATH.exists()
    if use_reports:
        print(f"Reports     : {REPORT_PATH}")
    else:
        print(f"Reports     : (not found, skipping)")

    raw_predictions = load_csv_dicts(PREDICTIONS_PATH)
    demographics_bytes = DEMOGRAPHICS_PATH.read_bytes()
    print(f"Loaded {len(raw_predictions)} prediction rows")

    # Parse report CSV → {case_id: text} (only if file exists)
    report_text_by_case_id: dict[str, str] = {}
    if use_reports:
        for row in load_csv_dicts(REPORT_PATH):
            case_id = row.get("case_id", "").strip()
            text_blob = build_report_text(row)
            if case_id and text_blob:
                report_text_by_case_id[case_id] = text_blob
        print(f"Loaded {len(report_text_by_case_id)} report texts")

    petbert = parse_predictions(raw_predictions)
    demographics = parse_dataset_a_demographics(demographics_bytes)
    ids_to_process = sorted(petbert.keys())
    print(f"Unique patients: {len(ids_to_process)}")

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with Session() as db:
        await db.execute(text("SET statement_timeout = 0"))

        # --- Clear existing patient data ---
        print("Clearing existing data...")
        await db.execute(text("DELETE FROM diagnosis_review_events"))
        await db.execute(text("DELETE FROM case_diagnoses"))
        await db.execute(text("DELETE FROM pathology_reports"))
        await db.execute(text("DELETE FROM patients"))
        await db.execute(text("DELETE FROM ingestion_jobs"))
        await db.execute(text("DELETE FROM ingestion_logs"))
        await db.commit()
        await db.execute(text("SET statement_timeout = 0"))
        print("Database cleared.")

        # --- Load reference lookups ---
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
        breed_map: dict[tuple[int, str], int] = {
            (r.species_id, r.name): r.id for r in breed_rows
        }

        # --- Pre-create missing species in bulk ---
        unique_sp = {
            (demographics.get(a) or {}).get("species") or ""
            for a in ids_to_process
        } - {""}
        new_sp = unique_sp - set(species_map)
        if new_sp:
            for name in new_sp:
                await db.execute(
                    pg_insert(Species.__table__).values(name=name)
                    .on_conflict_do_nothing(index_elements=["name"])
                )
            await db.commit()
            await db.execute(text("SET statement_timeout = 0"))
            sp_rows = (await db.execute(select(Species.id, Species.name))).fetchall()
            species_map = {name: id_ for id_, name in sp_rows}
            dog_id = species_map.get("Dog", dog_id)

        # --- Pre-create missing breeds in bulk ---
        unique_breed_keys: set[tuple[int, str]] = set()
        for anon_id in ids_to_process:
            demo = demographics.get(anon_id) or {}
            sp_name = demo.get("species") or ""
            breed_name = demo.get("breed") or ""
            if breed_name:
                sp_id = species_map.get(sp_name, dog_id) if sp_name else dog_id
                unique_breed_keys.add((sp_id, breed_name))
        new_breeds = unique_breed_keys - set(breed_map)
        if new_breeds:
            for sp_id, name in new_breeds:
                await db.execute(
                    pg_insert(Breed.__table__).values(species_id=sp_id, name=name)
                    .on_conflict_do_nothing()
                )
            await db.commit()
            await db.execute(text("SET statement_timeout = 0"))
            breed_rows = (await db.execute(select(Breed.id, Breed.species_id, Breed.name))).fetchall()
            breed_map = {(r.species_id, r.name): r.id for r in breed_rows}

        # --- Pre-create missing cancer types in bulk ---
        unique_groups = {
            d["predicted_group"] or "Unknown"
            for diags in petbert.values()
            for d in diags
        }
        new_groups = unique_groups - set(cancer_type_map)
        if new_groups:
            for name in new_groups:
                await db.execute(
                    pg_insert(CancerType.__table__).values(name=name)
                    .on_conflict_do_nothing(index_elements=["name"])
                )
            await db.commit()
            await db.execute(text("SET statement_timeout = 0"))
            cancer_type_map = {
                name: id_
                for id_, name in (await db.execute(select(CancerType.id, CancerType.name))).fetchall()
            }

        # --- Bulk upsert patients ---
        print(f"Upserting {len(ids_to_process)} patients...")
        warnings: list[str] = []
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
                "anon_id": anon_id,
                "species_id": sp_id,
                "breed_id": breed_id,
                "sex": demo.get("sex"),
                "county_id": county_id,
                "zip_code": raw_zip or None,
                "data_source": "petbert",
                "diagnosis_date": demo.get("diagnosis_date"),
                "outcome": None,
            })

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

        # --- Resolve anon_id → patient.id ---
        anon_to_patient_id: dict[str, int] = {}
        for i in range(0, len(ids_to_process), _CHUNK):
            chunk = ids_to_process[i : i + _CHUNK]
            rows = (await db.execute(
                select(Patient.id, Patient.anon_id).where(Patient.anon_id.in_(chunk))
            )).fetchall()
            anon_to_patient_id.update({r.anon_id: r.id for r in rows})

        # --- Upload reports to GCS and insert pathology_reports rows ---
        # GCS job ID: use a timestamp so each local run gets a unique path prefix,
        # matching the reports/{job_id}/{anon_id}.txt convention from the UI pipeline.
        anon_to_report_id: dict[str, int] = {}
        reports_inserted = 0

        if use_reports and report_text_by_case_id:
            # Only upload for patients we actually have in the DB
            report_text_by_anon_id = {
                anon_id: report_text_by_case_id[anon_id]
                for anon_id in ids_to_process
                if anon_id in report_text_by_case_id and anon_id in anon_to_patient_id
            }

            gcs_path_by_anon_id: dict[str, str] = {}
            if settings.GCS_BUCKET:
                gcs_job_id = int(time.time())
                print(f"Uploading {len(report_text_by_anon_id)} reports to GCS (job_id={gcs_job_id})...")
                gcs_path_by_anon_id = await upload_reports_to_gcs(report_text_by_anon_id, gcs_job_id)
                print(f"  {len(gcs_path_by_anon_id)} reports uploaded successfully.")
            else:
                print("GCS not configured — storing pathology_reports without gcs_path.")

            # Build pathology_report rows (one per patient that has report text)
            demo_by_anon = {a: (demographics.get(a) or {}) for a in report_text_by_anon_id}
            pr_values = [
                {
                    "patient_id": anon_to_patient_id[anon_id],
                    "gcs_path": gcs_path_by_anon_id.get(anon_id),
                    "report_date": demo_by_anon[anon_id].get("diagnosis_date"),
                }
                for anon_id in report_text_by_anon_id
            ]

            print(f"Inserting {len(pr_values)} pathology_report rows...")
            pr_table = PathologyReport.__table__
            for i in range(0, len(pr_values), _CHUNK):
                chunk = pr_values[i : i + _CHUNK]
                result = await db.execute(
                    pg_insert(pr_table)
                    .values(chunk)
                    .returning(pr_table.c.id, pr_table.c.patient_id)
                )
                for report_id, patient_id in result.fetchall():
                    # Reverse-lookup anon_id from patient_id to build the map
                    pass  # populated below via a separate query
            await db.commit()
            await db.execute(text("SET statement_timeout = 0"))
            reports_inserted = len(pr_values)

            # Resolve patient_id → pathology_report.id for the FK on case_diagnoses
            patient_to_report_id: dict[int, int] = {}
            for i in range(0, len(ids_to_process), _CHUNK):
                chunk_anon = ids_to_process[i : i + _CHUNK]
                chunk_pids = [anon_to_patient_id[a] for a in chunk_anon if a in anon_to_patient_id]
                if not chunk_pids:
                    continue
                rows = (await db.execute(
                    select(pr_table.c.id, pr_table.c.patient_id)
                    .where(pr_table.c.patient_id.in_(chunk_pids))
                )).fetchall()
                patient_to_report_id.update({pid: rid for rid, pid in rows})

            anon_to_report_id = {
                anon_id: patient_to_report_id[anon_to_patient_id[anon_id]]
                for anon_id in report_text_by_anon_id
                if anon_to_patient_id.get(anon_id) in patient_to_report_id
            }
            print(f"  {len(anon_to_report_id)} pathology_reports linked.")

        # --- Build diagnosis rows ---
        print("Building diagnoses...")
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
                needs_review = (
                    method == "low_confidence"
                    or conf < conf_threshold
                    or (row_margin is not None and row_margin < margin_threshold)
                )
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
                    # carried for review event creation, stripped before INSERT
                    "_needs_review": needs_review,
                    "_cancer_type_id": cancer_type_id,
                    "_icd_o_code": diag["icd_o_code"] or None,
                })
        print(f"  {len(diag_rows)} diagnosis rows built.")

        # --- Bulk insert diagnoses, capture returned IDs ---
        print("Inserting diagnoses...")
        diag_meta: list[tuple[int, bool, int, str | None]] = []  # (id, needs_review, ct_id, icd)
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
        print(f"  {len(diag_meta)} diagnoses inserted.")

        # --- Bulk insert review events for flagged diagnoses ---
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
            print(f"Inserting {len(review_rows)} review events...")
            for i in range(0, len(review_rows), _CHUNK):
                await db.execute(
                    pg_insert(DiagnosisReviewEvent.__table__).values(review_rows[i : i + _CHUNK])
                )
            await db.commit()
            await db.execute(text("SET statement_timeout = 0"))

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
                await db.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {view}"))
                print(f"  Refreshed {view}.")
            except Exception as e:
                warnings.append(f"Could not refresh {view}: {e}")

        await db.commit()

    await engine.dispose()

    print(f"\nDone.")
    print(f"  Patients          : {len(patient_values)}")
    print(f"  Pathology reports : {reports_inserted}")
    print(f"  Diagnoses         : {len(diag_rows)}")
    print(f"  Flagged           : {len(review_rows)}")
    print(f"  Warnings          : {len(warnings)}")
    for w in warnings[:20]:
        print(f"    - {w}")
    if len(warnings) > 20:
        print(f"    ... and {len(warnings) - 20} more")


if __name__ == "__main__":
    asyncio.run(main())
