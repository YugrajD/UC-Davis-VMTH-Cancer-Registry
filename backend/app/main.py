"""FastAPI application entry point for the VMTH Cancer Registry."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import select, text, update

from app.config import settings
from app.rate_limit import limiter
from app.database import async_session, engine, Base
from app.models.models import ExportRequest, IngestionJob, UserRole
from app.routers import dashboard, incidence, geo, trends, search, ingest, diagnoses_review, admin_users, role_requests, export
from app.routers import auth as auth_router
from app.services.role_seed import seed_user_roles_from_env

logger = logging.getLogger(__name__)


class RequestBodySizeLimitMiddleware:
    """Reject requests whose Content-Length exceeds a fixed limit.

    The upload endpoint gets a larger allowance (50 MB); everything else
    is capped at 10 MB.  Returns 413 Payload Too Large on violation.
    """

    DEFAULT_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
    UPLOAD_MAX_BYTES = 50 * 1024 * 1024    # 50 MB
    UPLOAD_PATH = "/api/v1/ingest/upload"

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        max_bytes = (
            self.UPLOAD_MAX_BYTES
            if path == self.UPLOAD_PATH
            else self.DEFAULT_MAX_BYTES
        )

        # Check Content-Length header if present
        headers = dict(
            (k.lower(), v)
            for k, v in scope.get("headers", [])
        )
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > max_bytes:
                    response = Response(
                        "Request body too large",
                        status_code=413,
                    )
                    await response(scope, receive, send)
                    return
            except (ValueError, TypeError):
                pass

        await self.app(scope, receive, send)


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

# --- Rate limiting ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request body size limit ---
app.add_middleware(RequestBodySizeLimitMiddleware)

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
