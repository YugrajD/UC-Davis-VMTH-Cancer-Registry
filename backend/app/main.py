"""FastAPI application entry point for the VMTH Cancer Registry."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, text, update

from app.config import settings
from app.database import async_session, engine, Base
from app.models.models import ExportRequest, IngestionJob, UserRole
from app.routers import dashboard, incidence, geo, trends, search, ingest, diagnoses_review, admin_users, role_requests, export
from app.routers import auth as auth_router
from app.services.role_seed import seed_user_roles_from_env

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

    # Seed user_roles from env-var allow lists. Idempotent — only inserts
    # rows that don't already exist; never overwrites UI-managed values.
    try:
        async with async_session() as db:
            inserted = await seed_user_roles_from_env(db)
            if inserted:
                logger.info("Seeded %d user_roles rows from env vars", inserted)
            await db.commit()
    except Exception as e:
        logger.warning("Could not seed user_roles from env: %s", e)

    # Ensure export_requests table exists (idempotent).
    try:
        async with engine.begin() as conn:
            await conn.run_sync(
                Base.metadata.create_all,
                tables=[ExportRequest.__table__],
            )
    except Exception as e:
        logger.warning("Could not ensure export_requests table: %s", e)

    # Drop the overly-restrictive UNIQUE(email, requested_role, status)
    # constraint on role_requests.  It prevents denying a second request
    # for the same user+role because (email, role, 'denied') already exists.
    try:
        async with async_session() as db:
            await db.execute(text(
                "ALTER TABLE role_requests "
                "DROP CONSTRAINT IF EXISTS role_requests_email_requested_role_status_key"
            ))
            await db.commit()
    except Exception as e:
        logger.warning("Could not fix role_requests constraint: %s", e)

    # Widen the export_requests status CHECK to include 'downloaded'.
    try:
        async with async_session() as db:
            await db.execute(text(
                "ALTER TABLE export_requests "
                "DROP CONSTRAINT IF EXISTS export_requests_status_check"
            ))
            await db.execute(text(
                "ALTER TABLE export_requests "
                "ADD CONSTRAINT export_requests_status_check "
                "CHECK (status IN ('pending', 'approved', 'denied', 'downloaded'))"
            ))
            await db.commit()
    except Exception as e:
        logger.warning("Could not update export_requests status constraint: %s", e)

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
app.include_router(diagnoses_review.router)
app.include_router(admin_users.router)
app.include_router(auth_router.router)
app.include_router(role_requests.router)
app.include_router(export.router)


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
