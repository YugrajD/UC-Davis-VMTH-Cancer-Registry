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


class CacheHeaderMiddleware:
    """Set Cache-Control headers on public read-only GET endpoints.

    Pure ASGI middleware (not BaseHTTPMiddleware) to avoid the internal
    response-queue mechanism that can deadlock when downstream handlers
    perform blocking I/O.
    """

    CACHE_RULES: list[tuple[str, int]] = [
        ("/api/v1/dashboard/summary", 60),
        ("/api/v1/dashboard/filters", 3600),
        ("/api/v1/incidence", 60),
        ("/api/v1/trends", 60),
        ("/api/v1/geo/calenviroscreen", 3600),
        ("/api/v1/geo/counties", 300),
    ]

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "")
        path = scope.get("path", "")

        # Only inject headers for GET requests matching a cache rule.
        max_age = None
        if method == "GET":
            for prefix, age in self.CACHE_RULES:
                if path.startswith(prefix):
                    max_age = age
                    break

        if max_age is None:
            await self.app(scope, receive, send)
            return

        swr = max(max_age // 2, 30)
        cache_value = f"public, max-age={max_age}, stale-while-revalidate={swr}".encode()

        async def send_with_cache_headers(message):
            if message["type"] == "http.response.start":
                status = message.get("status", 200)
                if status < 400:
                    headers = list(message.get("headers", []))
                    headers.append((b"cache-control", cache_value))
                    headers.append((b"vary", b"Accept-Encoding"))
                    message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_cache_headers)


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

    # Ensure materialized views have the sex column (migration 023).
    # If the column is absent the sex filter returns a 500; this block
    # detects that and rebuilds both views idempotently on startup.
    try:
        async with async_session() as db:
            row = await db.execute(text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = 'public' "
                "AND table_name = 'mv_county_cancer_incidence' "
                "AND column_name = 'sex'"
            ))
            if row.scalar() is None:
                logger.info("Applying migration 023: rebuilding materialized views with sex column")
                await db.execute(text(
                    "DROP MATERIALIZED VIEW IF EXISTS mv_county_cancer_incidence CASCADE"
                ))
                await db.execute(text("""
                    CREATE MATERIALIZED VIEW mv_county_cancer_incidence AS
                    SELECT
                        p.county_id,
                        co.name        AS county_name,
                        ct.id          AS cancer_type_id,
                        ct.name        AS cancer_type_name,
                        s.id           AS species_id,
                        s.name         AS species_name,
                        COALESCE(p.sex, 'Unknown') AS sex,
                        EXTRACT(YEAR FROM p.diagnosis_date)::INTEGER AS year,
                        COUNT(*)       AS case_count
                    FROM case_diagnoses cd
                    JOIN patients     p  ON cd.patient_id     = p.id
                    JOIN counties     co ON p.county_id       = co.id
                    JOIN cancer_types ct ON cd.cancer_type_id  = ct.id
                    JOIN species      s  ON p.species_id      = s.id
                    WHERE p.data_source = 'petbert'
                      AND cd.review_status IN ('confirmed', 'corrected')
                    GROUP BY p.county_id, co.name, ct.id, ct.name,
                             s.id, s.name, COALESCE(p.sex, 'Unknown'),
                             EXTRACT(YEAR FROM p.diagnosis_date)
                """))
                await db.execute(text(
                    "CREATE UNIQUE INDEX idx_mv_county_cancer "
                    "ON mv_county_cancer_incidence (county_id, cancer_type_id, species_id, sex, year)"
                ))
                await db.execute(text(
                    "DROP MATERIALIZED VIEW IF EXISTS mv_yearly_trends CASCADE"
                ))
                await db.execute(text("""
                    CREATE MATERIALIZED VIEW mv_yearly_trends AS
                    SELECT
                        EXTRACT(YEAR FROM p.diagnosis_date)::INTEGER AS year,
                        ct.id          AS cancer_type_id,
                        ct.name        AS cancer_type_name,
                        s.id           AS species_id,
                        s.name         AS species_name,
                        p.county_id,
                        co.name        AS county_name,
                        COALESCE(p.sex, 'Unknown') AS sex,
                        COUNT(*)       AS case_count,
                        COUNT(*) FILTER (WHERE p.outcome = 'deceased') AS deceased_count,
                        COUNT(*) FILTER (WHERE p.outcome = 'alive')    AS alive_count
                    FROM case_diagnoses cd
                    JOIN patients     p  ON cd.patient_id     = p.id
                    JOIN cancer_types ct ON cd.cancer_type_id  = ct.id
                    JOIN species      s  ON p.species_id      = s.id
                    JOIN counties     co ON p.county_id       = co.id
                    WHERE p.data_source = 'petbert'
                      AND cd.review_status IN ('confirmed', 'corrected')
                    GROUP BY EXTRACT(YEAR FROM p.diagnosis_date),
                             ct.id, ct.name, s.id, s.name,
                             p.county_id, co.name, COALESCE(p.sex, 'Unknown')
                """))
                await db.execute(text(
                    "CREATE UNIQUE INDEX idx_mv_yearly_trends "
                    "ON mv_yearly_trends (year, cancer_type_id, species_id, county_id, sex)"
                ))
                await db.commit()
                logger.info("Migration 023 applied: materialized views rebuilt with sex column")
    except Exception as e:
        logger.warning("Could not apply migration 023: %s", e)

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

# --- HTTP cache headers for public read-only endpoints ---
app.add_middleware(CacheHeaderMiddleware)

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
