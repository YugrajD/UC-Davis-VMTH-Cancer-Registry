"""FastAPI application entry point for the VMTH Cancer Registry."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, update

from app.config import settings
from app.database import async_session
from app.models.models import IngestionJob
from app.routers import dashboard, incidence, geo, trends, search, ingest
from app.routers import auth as auth_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle handler."""
    # Mark any stale 'processing' jobs as 'failed' on startup
    try:
        async with async_session() as db:
            # Log any stale jobs that had a GCP Batch job running so the
            # admin can check GCP Console manually.
            stale_result = await db.execute(
                select(IngestionJob).where(IngestionJob.status == "processing")
            )
            stale_jobs = stale_result.scalars().all()
            for sj in stale_jobs:
                if sj.batch_job_name:
                    logger.warning(
                        "Stale job %d had GCP Batch job %s — check GCP Console",
                        sj.id,
                        sj.batch_job_name,
                    )

            result = await db.execute(
                update(IngestionJob)
                .where(IngestionJob.status == "processing")
                .values(
                    status="failed",
                    processing_error="Server restarted during processing",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            if result.rowcount > 0:
                logger.warning("Marked %d stale processing jobs as failed", result.rowcount)
            await db.commit()
    except Exception as e:
        logger.warning("Could not check stale jobs on startup: %s", e)

    yield


app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="UC Davis Veterinary Medical Teaching Hospital Cancer Registry API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard.router)
app.include_router(incidence.router)
app.include_router(geo.router)
app.include_router(trends.router)
app.include_router(search.router)
app.include_router(ingest.router)
app.include_router(auth_router.router)


@app.get("/")
async def root():
    return {
        "name": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
