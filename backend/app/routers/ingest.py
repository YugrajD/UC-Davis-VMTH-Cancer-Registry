"""Router for data ingestion — CSV upload → ML worker → Supabase pipeline."""

import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.schemas.schemas import IngestionResponse
from app.services.ingestion_service import ingest_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])

ML_WORKER_TIMEOUT = 600.0  # 10 minutes for PetBERT processing


@router.get("/status")
async def ingestion_status():
    """Health check for the ingestion module."""
    return {"status": "ok"}


@router.post("/upload", response_model=IngestionResponse)
async def upload_datasets(
    dataset_a: UploadFile = File(...),
    dataset_b: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
):
    """Upload Dataset A (clinical notes) and optional Dataset B (demographics).

    Dataset A is forwarded to the ML worker for PetBERT classification.
    Dataset B is parsed locally for sex/zip demographics.
    Results are ingested into Supabase via the ingestion service.
    """
    # --- Validate Dataset A ---
    if not dataset_a.filename:
        raise HTTPException(status_code=400, detail="Dataset A file is required")

    dataset_a_bytes = await dataset_a.read()
    if not dataset_a_bytes:
        raise HTTPException(status_code=400, detail="Dataset A file is empty")

    # --- Read Dataset B if provided ---
    dataset_b_bytes: Optional[bytes] = None
    dataset_b_filename: Optional[str] = None
    if dataset_b and dataset_b.filename:
        dataset_b_bytes = await dataset_b.read()
        dataset_b_filename = dataset_b.filename
        if not dataset_b_bytes:
            dataset_b_bytes = None

    # --- Forward Dataset A to ML worker ---
    ml_worker_url = f"{settings.ML_WORKER_URL}/predict"

    try:
        async with httpx.AsyncClient(timeout=ML_WORKER_TIMEOUT) as client:
            response = await client.post(
                ml_worker_url,
                files={"file": (dataset_a.filename, dataset_a_bytes, "text/csv")},
            )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="ML worker service is unavailable. Ensure ml-worker container is running.",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="ML worker timed out. The dataset may be too large or the model is still loading.",
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"ML worker communication error: {e}",
        )

    if response.status_code != 200:
        detail = "ML worker error"
        try:
            err_body = response.json()
            detail = err_body.get("detail", detail)
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=detail)

    ml_result = response.json()
    predictions = ml_result.get("predictions", [])

    if not predictions:
        raise HTTPException(
            status_code=400,
            detail="ML worker returned no predictions. Check that Dataset A has valid data.",
        )

    # --- Ingest into database ---
    try:
        result = await ingest_upload(
            db=db,
            predictions=predictions,
            demographics_csv=dataset_b_bytes,
            dataset_a_filename=dataset_a.filename or "unknown",
            dataset_b_filename=dataset_b_filename,
        )
    except Exception as e:
        logger.exception("Ingestion error")
        raise HTTPException(
            status_code=500,
            detail=f"Database ingestion error: {e}",
        )

    return result
