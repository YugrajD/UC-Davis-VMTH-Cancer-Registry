"""Background job processor — runs approved ingestion jobs asynchronously."""

import asyncio
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
    """Route to the correct processing backend based on config."""
    if settings.USE_GCP_BATCH:
        await _process_via_gcp_batch(job_id)
    else:
        await _process_via_local_ml_worker(job_id)


async def _process_via_local_ml_worker(job_id: int) -> None:
    """Process an approved ingestion job via the local ml-worker container.

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


async def _process_via_gcp_batch(job_id: int) -> None:
    """Process an approved ingestion job via GCP Batch.

    1. Upload dataset_a.csv to GCS
    2. Submit a Batch job
    3. Poll until SUCCEEDED/FAILED
    4. Download predictions.json from GCS
    5. Ingest into database
    6. Cleanup GCS files
    """
    from app.services.gcp_batch_service import (
        cleanup_gcs_job_files,
        download_predictions_from_gcs,
        get_batch_job_status,
        submit_batch_job,
        upload_csv_to_gcs,
    )

    loop = asyncio.get_running_loop()

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

            # 1. Upload CSV to GCS
            logger.info("Job %d: uploading dataset_a.csv to GCS", job_id)
            await loop.run_in_executor(
                None, upload_csv_to_gcs, job_id, "dataset_a.csv", dataset_a_bytes
            )

            # 2. Submit Batch job
            logger.info("Job %d: submitting GCP Batch job", job_id)
            batch_job_name = await loop.run_in_executor(
                None, submit_batch_job, job_id
            )

            # Store the Batch job name so it can be found in GCP Console
            job.batch_job_name = batch_job_name
            job.updated_at = datetime.now(timezone.utc)
            await db.commit()

            # 3. Poll until terminal state
            logger.info("Job %d: polling Batch job %s", job_id, batch_job_name)
            terminal_states = {"SUCCEEDED", "FAILED", "DELETION_IN_PROGRESS"}
            poll_interval = settings.GCP_BATCH_POLL_INTERVAL

            while True:
                await asyncio.sleep(poll_interval)
                state, error_msg = await loop.run_in_executor(
                    None, get_batch_job_status, batch_job_name
                )
                logger.info("Job %d: Batch status = %s", job_id, state)

                if state in terminal_states:
                    break

            if state != "SUCCEEDED":
                raise RuntimeError(
                    f"GCP Batch job {batch_job_name} ended with state {state}: "
                    f"{error_msg or 'no details'}"
                )

            # 4. Download predictions from GCS
            logger.info("Job %d: downloading predictions from GCS", job_id)
            predictions = await loop.run_in_executor(
                None, download_predictions_from_gcs, job_id
            )

            if not predictions:
                raise RuntimeError("Batch job produced no predictions")

            # 5. Ingest into database (same as local path)
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

            logger.info("Job %d completed via GCP Batch: %d inserted", job_id, ingestion_result.inserted)

            # 6. Cleanup GCS files (best-effort)
            try:
                await loop.run_in_executor(None, cleanup_gcs_job_files, job_id)
            except Exception:
                logger.warning("Job %d: GCS cleanup failed (non-fatal)", job_id, exc_info=True)

        except Exception as e:
            logger.exception("Job %d failed (GCP Batch path)", job_id)
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
