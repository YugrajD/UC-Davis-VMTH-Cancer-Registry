#!/usr/bin/env python3
"""Run ingestion directly against the database using two local CSVs.

Usage (from project root):
  docker exec vmth_cancer_backend python /app/scripts/local_ingest.py \
      /app/scripts/petbert_predictions.csv \
      /app/scripts/demographics.csv
"""

import asyncio
import csv
import io
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make sure the app package is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, "/app")

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from app.config import settings
from app.services.ingestion_service import ingest_upload

PREDICTIONS_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/scripts/petbert_predictions.csv")
DEMOGRAPHICS_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/app/scripts/demographics.csv")


def load_csv_dicts(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


async def main():
    print(f"Predictions : {PREDICTIONS_PATH}")
    print(f"Demographics: {DEMOGRAPHICS_PATH}")

    if not PREDICTIONS_PATH.exists():
        sys.exit(f"ERROR: {PREDICTIONS_PATH} not found")
    if not DEMOGRAPHICS_PATH.exists():
        sys.exit(f"ERROR: {DEMOGRAPHICS_PATH} not found")

    predictions = load_csv_dicts(PREDICTIONS_PATH)
    demographics_bytes = DEMOGRAPHICS_PATH.read_bytes()
    print(f"Loaded {len(predictions)} prediction rows")

    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with Session() as db:
        # Remove per-statement timeout so large batch flushes don't get killed.
        await db.execute(text("SET statement_timeout = 0"))

        # --- Clear existing patient data ---
        print("Clearing existing data...")
        await db.execute(text("DELETE FROM case_diagnoses"))
        await db.execute(text("DELETE FROM pathology_reports"))
        await db.execute(text("DELETE FROM patients"))
        await db.execute(text("DELETE FROM ingestion_jobs"))
        await db.execute(text("DELETE FROM ingestion_logs"))
        await db.commit()
        print("Database cleared.")

        # --- Run ingestion ---
        print("Running ingestion...")
        result = await ingest_upload(
            db=db,
            predictions=predictions,
            dataset_a_filename=DEMOGRAPHICS_PATH.name,
            dataset_a_csv=demographics_bytes,
            ingestion_job_id=None,
        )
        await db.commit()

    await engine.dispose()

    print(f"\nDone.")
    summary = result.result_summary or {}
    print(f"  Patients upserted : {summary.get('patients', '?')}")
    print(f"  Diagnoses inserted: {result.inserted}")
    print(f"  Total rows        : {result.total_rows}")
    print(f"  Warnings          : {len(result.warnings)}")
    for w in result.warnings[:20]:
        print(f"    - {w}")
    if len(result.warnings) > 20:
        print(f"    ... and {len(result.warnings) - 20} more")


if __name__ == "__main__":
    asyncio.run(main())
