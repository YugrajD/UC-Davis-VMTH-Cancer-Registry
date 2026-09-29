"""Background job processor — runs approved ingestion jobs asynchronously."""

import asyncio
import logging
import time
from datetime import datetime, timezone

import httpx
from sqlalchemy import select

from app.cache import clear_all_caches
from app.config import settings
from app.database import async_session
from app.models.models import IngestionJob
from app.services.ingestion_service import ingest_upload
from app.services.s3_service import download_csv

logger = logging.getLogger(__name__)

ML_WORKER_TIMEOUT = 600.0

# Maps normalised ECS task states (see ml_task_service) to processing_stage values
_ML_TASK_STATE_TO_STAGE: dict[str, str] = {
    "PENDING": "batch_scheduled",
    "RUNNING": "batch_running",
}


# ---------------------------------------------------------------------------
# Helpers — each opens and closes its own short-lived DB session so callers
# never hold a connection across long I/O (ML worker calls, ECS polling, etc.)
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


def _compact_petbert_summary(summary: dict) -> dict:
    """Keep only diagnostic PetBERT fields worth storing on ingestion_jobs."""
    if not summary:
        return {}
    return {
        "input_rows": summary.get("input_rows"),
        "prediction_method_counts": summary.get("prediction_method_counts", {}),
        "predicted_group_counts": summary.get("predicted_group_counts", {}),
        "thresholds": summary.get("thresholds", {}),
    }


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def process_approved_job(job_id: int) -> None:
    """Route to the correct processing backend based on config."""
    try:
        if settings.USE_ECS_ML:
            await _process_via_ecs_task(job_id)
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

        dataset_a_filename = job.dataset_a_filename

        job.status = "processing"
        job.processing_stage = "reading_files"
        job.updated_at = datetime.now(timezone.utc)
        await db.commit()

    try:
        t_job_start = time.perf_counter()
        timings: dict[str, float] = {}

        # --- Phase 2: read file from S3 (no DB needed) ------------------
        loop = asyncio.get_running_loop()
        dataset_a_bytes = await loop.run_in_executor(None, download_csv, job_id)

        # --- Phase 3: call ML worker (long-running, no DB session) ------
        await _update_job(job_id, processing_stage="running_ml_worker")

        ml_worker_url = f"{settings.ML_WORKER_URL}/predict"
        _t = time.perf_counter()
        async with httpx.AsyncClient(timeout=ML_WORKER_TIMEOUT) as client:
            response = await client.post(
                ml_worker_url,
                files={"file": ("dataset_a.csv", dataset_a_bytes, "text/csv")},
            )
        timings["ml_worker_s"] = round(time.perf_counter() - _t, 2)

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
        petbert_summary = _compact_petbert_summary(ml_result.get("petbert_summary") or {})

        if not predictions:
            raise RuntimeError("ML worker returned no predictions")

        # --- Phase 4: fresh session for cancellation check + ingestion --
        if await _is_cancelled(job_id):
            logger.info("Job %d was cancelled before ingestion", job_id)
            return

        await _update_job(job_id, processing_stage="ingesting")
        _t = time.perf_counter()

        async with async_session() as db:
            ingestion_result = await ingest_upload(
                db=db,
                predictions=predictions,
                dataset_a_filename=dataset_a_filename,
                dataset_a_csv=dataset_a_bytes,
                ingestion_job_id=job_id,
            )

            timings["db_ingest_s"] = round(time.perf_counter() - _t, 2)
            timings["total_s"] = round(time.perf_counter() - t_job_start, 2)
            logger.info("Job %d timings: %s", job_id, timings)

            summary = ingestion_result.result_summary or {}
            summary["timings_seconds"] = timings
            if petbert_summary:
                summary["petbert"] = petbert_summary

            result = await db.execute(
                select(IngestionJob).where(IngestionJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = "completed"
                job.processing_stage = None
                job.ingestion_log_id = ingestion_result.ingestion_log_id
                job.result_summary = summary
                job.updated_at = datetime.now(timezone.utc)
                await db.commit()

        logger.info("Job %d completed: %d inserted", job_id, ingestion_result.inserted)
        clear_all_caches()

    except Exception as e:
        logger.exception("Job %d failed", job_id)
        await _mark_failed(job_id, _safe_error_message(e))




# ---------------------------------------------------------------------------
# ECS Fargate ML task path
# ---------------------------------------------------------------------------

async def _process_via_ecs_task(job_id: int) -> None:
    """Process an approved ingestion job via an on-demand ECS Fargate task.

    1. Start the ML task (it reads uploads/{job_id}/dataset_a.csv from S3)
    2. Poll until the task stops, updating processing_stage from ECS state
    3. Download predictions.json and PetBERT diagnostics from S3
    4. Ingest into database
    5. Optionally cleanup S3 job files
    """
    from app.services.ml_task_service import (
        STATE_SUCCEEDED,
        get_task_status,
        stop_ml_task,
        submit_ml_task,
    )
    from app.services.s3_service import (
        cleanup_job_files,
        download_petbert_summary,
        download_predictions,
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

        dataset_a_filename = job.dataset_a_filename
        model_folder = job.model_folder or "production"

        job.status = "processing"
        job.processing_stage = "submitting_batch_job"
        job.updated_at = datetime.now(timezone.utc)
        await db.commit()

    try:
        t_job_start = time.perf_counter()
        timings: dict[str, float] = {}

        # --- Phase 2: start the ML task ---------------------------------
        logger.info("Job %d: starting ML task (model_folder=%s)", job_id, model_folder)
        _t = time.perf_counter()
        task_arn = await loop.run_in_executor(None, submit_ml_task, job_id, model_folder)
        timings["task_submit_s"] = round(time.perf_counter() - _t, 2)

        await _update_job(
            job_id,
            processing_stage="batch_queued",
            batch_job_name=task_arn,
        )

        # --- Phase 3: poll until the task stops -------------------------
        logger.info("Job %d: polling ML task %s", job_id, task_arn)
        poll_interval = settings.ML_POLL_INTERVAL
        timeout_s = settings.ML_TIMEOUT_HOURS * 3600
        _t = time.perf_counter()

        while True:
            await asyncio.sleep(poll_interval)
            state, error_msg = await loop.run_in_executor(None, get_task_status, task_arn)
            logger.info("Job %d: ML task state = %s", job_id, state)

            mapped_stage = _ML_TASK_STATE_TO_STAGE.get(state)
            if mapped_stage:
                await _update_job(job_id, processing_stage=mapped_stage)

            if state in ("SUCCEEDED", "FAILED"):
                break

            if await _is_cancelled(job_id):
                logger.info("Job %d was cancelled during ML task polling", job_id)
                return

            # ECS has no built-in max run time, so enforce it here.
            if time.perf_counter() - _t > timeout_s:
                await loop.run_in_executor(None, stop_ml_task, task_arn, "Timed out")
                raise RuntimeError(
                    f"ML task {task_arn} exceeded {settings.ML_TIMEOUT_HOURS}h timeout"
                )

        timings["ml_task_run_s"] = round(time.perf_counter() - _t, 2)

        if state != STATE_SUCCEEDED:
            raise RuntimeError(
                f"ML task {task_arn} ended with state {state}: {error_msg or 'no details'}"
            )

        # --- Phase 4: download predictions ------------------------------
        await _update_job(job_id, processing_stage="downloading_predictions")
        logger.info("Job %d: downloading predictions from S3", job_id)
        _t = time.perf_counter()
        predictions = await loop.run_in_executor(None, download_predictions, job_id)
        petbert_summary = _compact_petbert_summary(
            await loop.run_in_executor(None, download_petbert_summary, job_id)
        )
        dataset_a_bytes = await loop.run_in_executor(None, download_csv, job_id)
        timings["s3_download_s"] = round(time.perf_counter() - _t, 2)

        if not predictions:
            raise RuntimeError("ML task produced no predictions")

        # --- Phase 5: ingest into database (fresh session) --------------
        await _update_job(job_id, processing_stage="ingesting")
        _t = time.perf_counter()

        async with async_session() as db:
            ingestion_result = await ingest_upload(
                db=db,
                predictions=predictions,
                dataset_a_filename=dataset_a_filename,
                dataset_a_csv=dataset_a_bytes,
                ingestion_job_id=job_id,
            )

            timings["db_ingest_s"] = round(time.perf_counter() - _t, 2)
            timings["total_s"] = round(time.perf_counter() - t_job_start, 2)
            logger.info("Job %d timings: %s", job_id, timings)

            summary = ingestion_result.result_summary or {}
            summary["timings_seconds"] = timings
            if petbert_summary:
                summary["petbert"] = petbert_summary

            result = await db.execute(
                select(IngestionJob).where(IngestionJob.id == job_id)
            )
            job = result.scalar_one_or_none()
            if job:
                job.status = "completed"
                job.processing_stage = None
                job.ingestion_log_id = ingestion_result.ingestion_log_id
                job.result_summary = summary
                job.updated_at = datetime.now(timezone.utc)
                await db.commit()

        logger.info("Job %d completed via ECS task: %d inserted", job_id, ingestion_result.inserted)
        clear_all_caches()

        # Cleanup S3 files (best-effort). Disabled by default so scan_output
        # diagnostics remain available after a suspicious successful run.
        if settings.ML_CLEANUP_JOB_FILES:
            try:
                await loop.run_in_executor(None, cleanup_job_files, job_id)
            except Exception:
                logger.warning("Job %d: S3 cleanup failed (non-fatal)", job_id, exc_info=True)
        else:
            logger.info("Job %d: preserving S3 job files for diagnostics", job_id)

    except Exception as e:
        logger.exception("Job %d failed (ECS task path)", job_id)
        await _mark_failed(job_id, _safe_error_message(e))
