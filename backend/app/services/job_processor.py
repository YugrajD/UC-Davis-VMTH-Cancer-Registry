"""Background job processor — runs approved ingestion jobs asynchronously."""

import logging
from datetime import datetime, timezone

import httpx
from sqlalchemy import select

from app.config import settings
from app.database import async_session
from app.models.models import IngestionJob
from app.services.ingestion_service import ingest_upload

logger = logging.getLogger(__name__)

ML_WORKER_TIMEOUT = 600.0


async def process_approved_job(job_id: int) -> None:
    """Process an approved ingestion job in the background.

    Opens its own DB session (outside request lifecycle), reads files from
    storage_path, forwards to ML worker, then calls ingest_upload().
    """
    async with async_session() as db:
        result = await db.execute(
            select(IngestionJob).where(IngestionJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        if not job:
            logger.error("Job %d not found", job_id)
            return

        job.status = "processing"
        job.updated_at = datetime.now(timezone.utc)
        await db.commit()

        try:
            storage_path = job.storage_path
            dataset_a_path = f"{storage_path}/dataset_a.csv"
            dataset_b_path = f"{storage_path}/dataset_b.csv"

            with open(dataset_a_path, "rb") as f:
                dataset_a_bytes = f.read()
            with open(dataset_b_path, "rb") as f:
                dataset_b_bytes = f.read()

            # Forward Dataset A to ML worker
            ml_worker_url = f"{settings.ML_WORKER_URL}/predict"
            async with httpx.AsyncClient(timeout=ML_WORKER_TIMEOUT) as client:
                response = await client.post(
                    ml_worker_url,
                    files={"file": (job.dataset_a_filename, dataset_a_bytes, "text/csv")},
                )

            if response.status_code != 200:
                detail = "ML worker error"
                try:
                    err_body = response.json()
                    detail = err_body.get("detail", detail)
                except Exception:
                    pass
                raise RuntimeError(f"ML worker returned {response.status_code}: {detail}")

            ml_result = response.json()
            predictions = ml_result.get("predictions", [])

            if not predictions:
                raise RuntimeError("ML worker returned no predictions")

            # Ingest into database
            ingestion_result = await ingest_upload(
                db=db,
                predictions=predictions,
                demographics_csv=dataset_b_bytes if dataset_b_bytes else None,
                dataset_a_filename=job.dataset_a_filename,
                dataset_b_filename=job.dataset_b_filename,
            )

            job.status = "completed"
            job.ingestion_log_id = ingestion_result.ingestion_log_id
            job.updated_at = datetime.now(timezone.utc)
            await db.commit()

            logger.info("Job %d completed: %d inserted", job_id, ingestion_result.inserted)

        except Exception as e:
            logger.exception("Job %d failed", job_id)
            # Re-fetch job in case the session state is stale after error
            await db.rollback()
            result = await db.execute(
                select(IngestionJob).where(IngestionJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = "failed"
                job.processing_error = str(e)[:2000]
                job.updated_at = datetime.now(timezone.utc)
                await db.commit()
