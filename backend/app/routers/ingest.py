"""Router for data ingestion — job-based upload with admin approval workflow."""

import asyncio
import io
import logging
import os
import re
from datetime import datetime, timezone

import pandas as pd
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


def _ensure_csv(raw_bytes: bytes, filename: str) -> bytes:
    """Convert XLSX to CSV bytes; pass CSV through unchanged."""
    lower = filename.lower()
    if lower.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(raw_bytes), engine="openpyxl")
        return df.to_csv(index=False).encode("utf-8")
    if lower.endswith(".csv"):
        return raw_bytes
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file format: {filename}. Use .csv or .xlsx",
    )


# Column name mapping: uploaded name → name expected by GCP Batch image
_COLUMN_RENAMES = {
    "diagnoses (labels)": "Clinical Diagnoses",
    "diagnoses": "Clinical Diagnoses",
    "clinical diagnoses": "Clinical Diagnoses",
    "text (pathology report)": "Text",
    "text": "Text",
}


# Fields that must have real values (not NA/empty) for a row to be a patient header.
_HEADER_SENTINEL_COLS = ["Date of Birth", "Sex", "Species", "Breed"]

# Columns concatenated from continuation rows into the preceding header.
_CONTINUATION_COLS = ["Text", "Clinical Diagnoses"]

# HTML tag pattern for stripping markup from pathology text.
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _is_na(value) -> bool:
    """Return True if the value is missing, empty, or the string 'NA'."""
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() == "na" or s.lower() == "nan"


def _merge_continuation_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Merge HTML-like continuation rows into single patient records.

    The uploaded file may have one 'header row' per patient followed by zero or
    more 'continuation rows' where all demographic fields are NA and only the
    ``Text`` and/or ``Clinical Diagnoses`` columns carry additional content.
    This function collapses each group into a single row with the fragments
    concatenated.

    A row is treated as a *patient header* when at least one sentinel column
    (Date of Birth, Sex, Species, Breed) contains a real value.  Otherwise it
    is a continuation row appended to the preceding patient.

    Rows that appear before the first header are silently skipped.
    """
    # Identify which sentinel columns are actually present in the dataframe.
    sentinel_cols = [c for c in _HEADER_SENTINEL_COLS if c in df.columns]
    cont_cols = [c for c in _CONTINUATION_COLS if c in df.columns]

    # Fast path: no sentinel columns found → file is already flat, return as-is.
    if not sentinel_cols:
        return df

    patients: list[dict] = []
    current: dict | None = None

    for _, row in df.iterrows():
        is_header = any(not _is_na(row.get(c)) for c in sentinel_cols)

        if is_header:
            if current is not None:
                patients.append(current)
            current = row.to_dict()
            # Strip HTML from the initial fragments.
            for col in cont_cols:
                if not _is_na(current.get(col)):
                    current[col] = _HTML_TAG_RE.sub("", str(current[col])).strip()
        else:
            # Continuation row — append its fragments to the current patient.
            if current is None:
                continue  # orphaned continuation before any header; skip
            for col in cont_cols:
                if _is_na(row.get(col)):
                    continue
                fragment = _HTML_TAG_RE.sub("", str(row[col])).strip()
                if not fragment:
                    continue
                existing = str(current.get(col, "") or "").strip()
                current[col] = f"{existing} {fragment}".strip() if existing else fragment

    # Flush the last patient.
    if current is not None:
        patients.append(current)

    if not patients:
        return df

    merged = pd.DataFrame(patients, columns=df.columns)
    logger.info(
        "_merge_continuation_rows: %d raw rows → %d patient records",
        len(df),
        len(merged),
    )
    return merged


def _normalize_columns(csv_bytes: bytes) -> bytes:
    """Normalize CSV columns to match the deployed GCP Batch image expectations.

    - Merges HTML-like continuation rows into single patient records
    - Renames known variant column names to canonical forms
    - Adds ``anon_id`` column (sequential per patient) if missing
    """
    df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str)

    # Rename columns to canonical names first so sentinel detection works.
    rename_map = {}
    for col in df.columns:
        canonical = _COLUMN_RENAMES.get(col.strip().lower())
        if canonical and canonical not in df.columns:
            rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)

    # Merge continuation rows into one record per patient.
    df = _merge_continuation_rows(df)

    # Add anon_id if missing — one sequential ID per merged patient record.
    if "anon_id" not in df.columns:
        df.insert(0, "anon_id", [f"VMTH_{i}" for i in range(len(df))])

    return df.to_csv(index=False).encode("utf-8")


@router.get("/status")
async def ingestion_status():
    """Health check for the ingestion module."""
    return {"status": "ok"}


@router.post("/upload")
async def upload_datasets(
    dataset_a: UploadFile = File(...),
    dataset_b: Optional[UploadFile] = None,
    db: AsyncSession = Depends(get_db),
    user: Optional[CurrentUser] = Depends(get_optional_user),
):
    """Upload Dataset A and optionally Dataset B. Creates a job in pending_review status.

    Authentication is optional — anonymous uploads are allowed.
    """
    if not dataset_a.filename:
        raise HTTPException(status_code=400, detail="Dataset A file is required")

    dataset_a_bytes = await dataset_a.read()
    dataset_b_bytes = (await dataset_b.read()) if dataset_b and dataset_b.filename else b""

    if not dataset_a_bytes:
        raise HTTPException(status_code=400, detail="Dataset A file is empty")

    # Convert XLSX → CSV if needed, then normalize columns for GCP Batch image
    dataset_a_bytes = _normalize_columns(_ensure_csv(dataset_a_bytes, dataset_a.filename))
    if dataset_b_bytes:
        dataset_b_bytes = _normalize_columns(_ensure_csv(dataset_b_bytes, dataset_b.filename))

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
    dataset_b_filename = dataset_b.filename if dataset_b and dataset_b.filename else None
    job = IngestionJob(
        uploaded_by_email=user.email if user else "anonymous",
        uploaded_by_sub=user.sub if user else "anonymous",
        dataset_a_filename=dataset_a.filename,
        dataset_b_filename=dataset_b_filename,
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
    if dataset_b_bytes:
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
    status: list[str] = Query(default=[]),
):
    """List jobs. Admins see all (unless mine=true); regular users always see only their own.

    Optionally filter by one or more statuses via ?status=pending_review&status=processing.
    """
    valid_statuses = {"pending_review", "processing", "completed", "failed", "rejected"}
    filtered_statuses = [s for s in status if s in valid_statuses]

    if user.is_admin and not mine:
        query = select(IngestionJob)
    else:
        query = select(IngestionJob).where(
            IngestionJob.uploaded_by_email == user.email
        )

    if filtered_statuses:
        query = query.where(IngestionJob.status.in_(filtered_statuses))

    query = query.order_by(IngestionJob.created_at.desc())
    result = await db.execute(query)
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


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    admin: CurrentUser = Depends(require_admin),
):
    """Cancel a job that is currently processing."""
    job = await _get_job_or_404(db, job_id)

    if job.status != "processing":
        raise HTTPException(
            status_code=400,
            detail=f"Job is '{job.status}', only 'processing' jobs can be cancelled",
        )

    now = datetime.now(timezone.utc)
    job.status = "cancelled"
    job.processing_stage = None
    job.processing_error = "Cancelled by admin"
    job.reviewed_by_email = admin.email
    job.reviewed_at = now
    job.updated_at = now
    await db.commit()
    await db.refresh(job)

    # Best-effort: cancel the GCP Batch job if one was submitted
    if job.batch_job_name:
        try:
            from app.services.gcp_batch_service import cancel_batch_job
            import asyncio
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, cancel_batch_job, job.batch_job_name)
        except Exception:
            logger.warning("Failed to cancel GCP Batch job for job %d", job_id, exc_info=True)

    return _job_to_dict(job)


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
    job.processing_stage = "queued"
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
        "processing_stage": job.processing_stage,
        "reviewed_by_email": job.reviewed_by_email,
        "reviewed_at": job.reviewed_at.isoformat() if job.reviewed_at else None,
        "rejection_reason": job.rejection_reason,
        "ingestion_log_id": job.ingestion_log_id,
        "processing_error": job.processing_error,
        "batch_job_name": job.batch_job_name,
        "result_summary": job.result_summary,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None,
    }
