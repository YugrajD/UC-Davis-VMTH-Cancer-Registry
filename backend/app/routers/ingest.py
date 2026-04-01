"""Router for data ingestion — job-based upload with admin approval workflow."""

import asyncio
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from typing import Optional
from app.auth import CurrentUser, get_current_user, get_optional_user, require_admin
from app.config import settings
from app.database import get_db
from app.models.models import IngestionJob
from app.schemas.schemas import IngestionJobOut, IngestionJobReview
from app.services.job_processor import process_approved_job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])


@router.get("/status")
async def ingestion_status():
    """Health check for the ingestion module."""
    return {"status": "ok"}


@router.post("/upload")
async def upload_datasets(
    dataset_a: UploadFile = File(...),
    dataset_b: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: Optional[CurrentUser] = Depends(get_optional_user),
):
    """Upload Dataset A and Dataset B. Creates a job in pending_review status.

    Authentication is optional — anonymous uploads are allowed.
    """
    if not dataset_a.filename:
        raise HTTPException(status_code=400, detail="Dataset A file is required")
    if not dataset_b.filename:
        raise HTTPException(status_code=400, detail="Dataset B file is required")

    dataset_a_bytes = await dataset_a.read()
    dataset_b_bytes = await dataset_b.read()

    if not dataset_a_bytes:
        raise HTTPException(status_code=400, detail="Dataset A file is empty")
    if not dataset_b_bytes:
        raise HTTPException(status_code=400, detail="Dataset B file is empty")

    # Rate limit: 3 uploads per day for non-admin users
    if not user or not user.is_admin:
        uploader_id = user.email if user else "anonymous"
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        count_result = await db.execute(
            select(func.count(IngestionJob.id)).where(
                IngestionJob.uploaded_by_email == uploader_id,
                IngestionJob.created_at >= today_start,
            )
        )
        today_count = count_result.scalar() or 0
        if today_count >= 3:
            raise HTTPException(
                status_code=429,
                detail="Upload limit reached (3 per day). Please try again tomorrow.",
            )

    # Create job record first to get an ID
    job = IngestionJob(
        uploaded_by_email=user.email if user else "anonymous",
        uploaded_by_sub=user.sub if user else "anonymous",
        dataset_a_filename=dataset_a.filename,
        dataset_b_filename=dataset_b.filename,
        storage_path="",  # will update after we know the ID
        status="pending_review",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.flush()

    # Save files to disk
    storage_path = os.path.join(settings.UPLOAD_DIR, str(job.id))
    os.makedirs(storage_path, exist_ok=True)

    with open(os.path.join(storage_path, "dataset_a.csv"), "wb") as f:
        f.write(dataset_a_bytes)
    with open(os.path.join(storage_path, "dataset_b.csv"), "wb") as f:
        f.write(dataset_b_bytes)

    job.storage_path = storage_path
    await db.commit()
    await db.refresh(job)

    return _job_to_dict(job)


@router.get("/jobs")
async def list_jobs(
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
    mine: bool = Query(False),
):
    """List jobs. Admins see all (unless mine=true); regular users always see only their own."""
    if user.is_admin and not mine:
        result = await db.execute(
            select(IngestionJob).order_by(IngestionJob.created_at.desc())
        )
    else:
        result = await db.execute(
            select(IngestionJob)
            .where(IngestionJob.uploaded_by_email == user.email)
            .order_by(IngestionJob.created_at.desc())
        )
    jobs = result.scalars().all()
    return [_job_to_dict(j) for j in jobs]


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
):
    """Get a single job by ID (for polling)."""
    job = await _get_job_or_404(db, job_id)

    # Non-admin can only see their own jobs
    if not user.is_admin and job.uploaded_by_email != user.email:
        raise HTTPException(status_code=403, detail="Access denied")

    return _job_to_dict(job)


@router.get("/jobs/{job_id}/preview/{dataset}")
async def preview_job_dataset(
    job_id: int,
    dataset: str,
    db: AsyncSession = Depends(get_db),
    _admin: CurrentUser = Depends(require_admin),
):
    """Stream a stored CSV file for admin preview."""
    if dataset not in ("a", "b"):
        raise HTTPException(status_code=400, detail="dataset must be 'a' or 'b'")

    job = await _get_job_or_404(db, job_id)
    filename = "dataset_a.csv" if dataset == "a" else "dataset_b.csv"
    filepath = os.path.join(job.storage_path, filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    def iter_file():
        with open(filepath, "rb") as f:
            yield from f

    return StreamingResponse(
        iter_file(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/jobs/{job_id}/review")
async def review_job(
    job_id: int,
    review: IngestionJobReview,
    db: AsyncSession = Depends(get_db),
    admin: CurrentUser = Depends(require_admin),
):
    """Approve or reject a pending job."""
    job = await _get_job_or_404(db, job_id)

    if job.status != "pending_review":
        raise HTTPException(
            status_code=400,
            detail=f"Job is '{job.status}', can only review 'pending_review' jobs",
        )

    if review.action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="action must be 'approve' or 'reject'")

    now = datetime.now(timezone.utc)
    job.reviewed_by_email = admin.email
    job.reviewed_at = now
    job.updated_at = now

    if review.action == "reject":
        job.status = "rejected"
        job.rejection_reason = review.rejection_reason
        await db.commit()
        await db.refresh(job)
        return _job_to_dict(job)

    # Approve → kick off background processing
    job.status = "processing"
    await db.commit()
    await db.refresh(job)

    asyncio.create_task(process_approved_job(job.id))

    return _job_to_dict(job)


# --- Helpers ---

async def _get_job_or_404(db: AsyncSession, job_id: int) -> IngestionJob:
    result = await db.execute(
        select(IngestionJob).where(IngestionJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _job_to_dict(job: IngestionJob) -> dict:
    return {
        "id": job.id,
        "uploaded_by_email": job.uploaded_by_email,
        "dataset_a_filename": job.dataset_a_filename,
        "dataset_b_filename": job.dataset_b_filename,
        "status": job.status,
        "reviewed_by_email": job.reviewed_by_email,
        "reviewed_at": job.reviewed_at.isoformat() if job.reviewed_at else None,
        "rejection_reason": job.rejection_reason,
        "ingestion_log_id": job.ingestion_log_id,
        "processing_error": job.processing_error,
        "batch_job_name": job.batch_job_name,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }
