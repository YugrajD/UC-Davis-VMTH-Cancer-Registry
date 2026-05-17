"""Background job processor — runs approved ingestion jobs asynchronously."""

import asyncio
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import select

from app.cache import clear_all_caches
from app.config import settings
from app.database import async_session
from app.models.models import IngestionJob
from app.services.ingestion_service import ingest_upload

logger = logging.getLogger(__name__)

ML_WORKER_TIMEOUT = 600.0

# Maps GCP Batch state names to our processing_stage values
_GCP_BATCH_STATE_TO_STAGE: dict[str, str] = {
    "QUEUED":     "batch_queued",
    "SCHEDULED":  "batch_scheduled",
    "RUNNING":    "batch_running",
}


# ---------------------------------------------------------------------------
# Helpers — each opens and closes its own short-lived DB session so callers
# never hold a connection across long I/O (ML worker calls, GCP polling, etc.)
# ---------------------------------------------------------------------------

async def _update_job(job_id: int, **fields) -> None:
    """Open a fresh session, update the given fields on the job, and commit."""
    async with async_session() as db:
        result = await db.execute(
            select(IngestionJob).where(IngestionJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        if not job:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = datetime.now(timezone.utc)
        await db.commit()


async def _is_cancelled(job_id: int) -> bool:
    """Check if a job was cancelled (uses its own short-lived session)."""
    async with async_session() as db:
        result = await db.execute(
            select(IngestionJob.status).where(IngestionJob.id == job_id)
        )
        status = result.scalar_one_or_none()
        return status == "cancelled"


def _safe_error_message(e: Exception) -> str:
    """Return an API-safe error message from an exception.

    RuntimeError messages are ones we control (e.g. "ML worker returned 500").
    For all other exception types, only expose the class name to avoid leaking
    file paths, connection strings, or stack details.
    """
    if isinstance(e, RuntimeError):
        return str(e)[:500]
    return type(e).__name__



async def _mark_failed(job_id: int, error_msg: str) -> None:
    """Mark a job as failed with the given error message."""
    async with async_session() as db:
        result = await db.execute(
            select(IngestionJob).where(IngestionJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        if job and job.status == "processing":
            job.status = "failed"
            job.processing_stage = None
            job.processing_error = error_msg[:500]
            job.updated_at = datetime.now(timezone.utc)
            await db.commit()


def _delete_upload_dir(storage_path: str) -> None:
    """Remove the per-job upload directory after processing completes or fails.

    The upload directory is no longer needed once ML processing finishes and
    raw files should not persist indefinitely on disk.  Rejections are NOT
    deleted here — the file stays for the audit trail until an admin reviews.
    """
    try:
        path = Path(storage_path).resolve()
        upload_root = Path(settings.UPLOAD_DIR).resolve()
        path.relative_to(upload_root)  # raises ValueError if path escapes root
        if path.is_dir():
            shutil.rmtree(path)
            logger.info("Deleted upload directory: %s", path)
    except (ValueError, OSError):
        logger.warning("Could not delete upload directory: %s", storage_path, exc_info=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def process_approved_job(job_id: int) -> None:
    """Route to the correct processing backend based on config."""
    try:
        if settings.USE_GCP_BATCH:
            await _process_via_gcp_batch(job_id)
        else:
            await _process_via_local_ml_worker(job_id)
    except Exception as e:
        # Last-resort handler: catches crashes that happen before the inner
        # error handlers run (e.g. import errors, missing config).
        logger.exception("Job %d crashed before inner error handler", job_id)
        try:
            await _mark_failed(job_id, _safe_error_message(e))
        except Exception:
            logger.exception("Job %d: failed to mark job as failed in last-resort handler", job_id)


# ---------------------------------------------------------------------------
# Local ML-worker path
# ---------------------------------------------------------------------------

async def _process_via_local_ml_worker(job_id: int) -> None:
    """Process an approved ingestion job via the local ml-worker container.

    Uses short-lived DB sessions so no connection is held open during
    the (potentially minutes-long) ML worker HTTP call.
    """

    # --- Phase 1: fetch job metadata, mark as processing ----------------
    async with async_session() as db:
        result = await db.execute(
            select(IngestionJob).where(IngestionJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        if not job:
            logger.error("Job %d not found", job_id)
            return

        storage_path = job.storage_path
        dataset_a_filename = job.dataset_a_filename

        job.status = "processing"
        job.processing_stage = "reading_files"
        job.updated_at = datetime.now(timezone.utc)
        await db.commit()

    try:
        # --- Phase 2: read file from disk (no DB needed) ----------------
        dataset_a_path = f"{storage_path}/dataset_a.csv"

        with open(dataset_a_path, "rb") as f:
            dataset_a_bytes = f.read()

        # --- Phase 3: call ML worker (long-running, no DB session) ------
        await _update_job(job_id, processing_stage="running_ml_worker")

        ml_worker_url = f"{settings.ML_WORKER_URL}/predict"
        async with httpx.AsyncClient(timeout=ML_WORKER_TIMEOUT) as client:
            response = await client.post(
                ml_worker_url,
                files={"file": ("dataset_a.csv", dataset_a_bytes, "text/csv")},
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

        # --- Phase 4: fresh session for cancellation check + ingestion --
        if await _is_cancelled(job_id):
            logger.info("Job %d was cancelled before ingestion", job_id)
            return

        await _update_job(job_id, processing_stage="ingesting")

        async with async_session() as db:
            ingestion_result = await ingest_upload(
                db=db,
                predictions=predictions,
                dataset_a_filename=dataset_a_filename,
                dataset_a_csv=dataset_a_bytes,
                ingestion_job_id=job_id,
            )

            result = await db.execute(
                select(IngestionJob).where(IngestionJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = "completed"
                job.processing_stage = None
                job.ingestion_log_id = ingestion_result.ingestion_log_id
                job.result_summary = ingestion_result.result_summary
                job.updated_at = datetime.now(timezone.utc)
                await db.commit()

        logger.info("Job %d completed: %d inserted", job_id, ingestion_result.inserted)
        clear_all_caches()
        _delete_upload_dir(storage_path)

    except Exception as e:
        logger.exception("Job %d failed", job_id)
        await _mark_failed(job_id, _safe_error_message(e))
        _delete_upload_dir(storage_path)


# ---------------------------------------------------------------------------
# GCP Batch path
# ---------------------------------------------------------------------------

async def _process_via_gcp_batch(job_id: int) -> None:
    """Process an approved ingestion job via GCP Batch.

    1. Upload dataset_a.csv to GCS
    2. Submit a Batch job
    3. Poll until SUCCEEDED/FAILED, updating processing_stage from GCP Batch state
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

    # --- Phase 1: fetch job metadata, mark as processing ----------------
    async with async_session() as db:
        result = await db.execute(
            select(IngestionJob).where(IngestionJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        if not job:
            logger.error("Job %d not found", job_id)
            return

        storage_path = job.storage_path
        dataset_a_filename = job.dataset_a_filename

        job.status = "processing"
        job.processing_stage = "uploading_to_gcs"
        job.updated_at = datetime.now(timezone.utc)
        await db.commit()

    try:
        # --- Phase 2: read file and upload to GCS (no long DB hold) -----
        dataset_a_path = f"{storage_path}/dataset_a.csv"

        with open(dataset_a_path, "rb") as f:
            dataset_a_bytes = f.read()

        logger.info("Job %d: uploading dataset_a.csv to GCS", job_id)
        await loop.run_in_executor(
            None, upload_csv_to_gcs, job_id, "dataset_a.csv", dataset_a_bytes
        )

        # --- Phase 3: submit Batch job ----------------------------------
        await _update_job(job_id, processing_stage="submitting_batch_job")
        logger.info("Job %d: submitting GCP Batch job", job_id)
        batch_job_name = await loop.run_in_executor(
            None, submit_batch_job, job_id
        )

        await _update_job(
            job_id,
            processing_stage="batch_queued",
            batch_job_name=batch_job_name,
        )

        # --- Phase 4: poll until terminal state -------------------------
        logger.info("Job %d: polling Batch job %s", job_id, batch_job_name)
        terminal_states = {"SUCCEEDED", "FAILED", "DELETION_IN_PROGRESS"}
        poll_interval = settings.GCP_BATCH_POLL_INTERVAL

        while True:
            await asyncio.sleep(poll_interval)
            state, error_msg = await loop.run_in_executor(
                None, get_batch_job_status, batch_job_name
            )
            logger.info("Job %d: Batch status = %s", job_id, state)

            mapped_stage = _GCP_BATCH_STATE_TO_STAGE.get(state)
            if mapped_stage:
                await _update_job(job_id, processing_stage=mapped_stage)

            if state in terminal_states:
                break

            if await _is_cancelled(job_id):
                logger.info("Job %d was cancelled during Batch polling", job_id)
                return

        if state != "SUCCEEDED":
            raise RuntimeError(
                f"GCP Batch job {batch_job_name} ended with state {state}: "
                f"{error_msg or 'no details'}"
            )

        # --- Phase 5: download predictions ------------------------------
        await _update_job(job_id, processing_stage="downloading_predictions")
        logger.info("Job %d: downloading predictions from GCS", job_id)
        predictions = await loop.run_in_executor(
            None, download_predictions_from_gcs, job_id
        )

        if not predictions:
            raise RuntimeError("Batch job produced no predictions")

        # --- Phase 6: ingest into database (fresh session) --------------
        await _update_job(job_id, processing_stage="ingesting")

        async with async_session() as db:
            ingestion_result = await ingest_upload(
                db=db,
                predictions=predictions,
                dataset_a_filename=dataset_a_filename,
                dataset_a_csv=dataset_a_bytes,
                ingestion_job_id=job_id,
            )

            result = await db.execute(
                select(IngestionJob).where(IngestionJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = "completed"
                job.processing_stage = None
                job.ingestion_log_id = ingestion_result.ingestion_log_id
                job.result_summary = ingestion_result.result_summary
                job.updated_at = datetime.now(timezone.utc)
                await db.commit()

        logger.info("Job %d completed via GCP Batch: %d inserted", job_id, ingestion_result.inserted)
        clear_all_caches()
        _delete_upload_dir(storage_path)

        # Cleanup GCS files (best-effort)
        try:
            await loop.run_in_executor(None, cleanup_gcs_job_files, job_id)
        except Exception:
            logger.warning("Job %d: GCS cleanup failed (non-fatal)", job_id, exc_info=True)

    except Exception as e:
        logger.exception("Job %d failed (GCP Batch path)", job_id)
        await _mark_failed(job_id, _safe_error_message(e))
        _delete_upload_dir(storage_path)
