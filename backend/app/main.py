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
from app.routers import dashboard, incidence, geo, trends, search, ingest, diagnoses_review, admin_users, role_requests, export, admin as admin_router_module
from app.routers import auth as auth_router
from app.services.role_seed import seed_user_roles_from_env

logger = logging.getLogger(__name__)


class RequestBodySizeLimitMiddleware:
    """Reject requests whose body exceeds a fixed size limit.

    The upload endpoint gets a larger allowance (50 MB); everything else
    is capped at 10 MB.  Returns 413 Payload Too Large on violation.

    Enforces the limit on actual received bytes, not just the Content-Length
    header, to catch chunked requests that omit or lie about Content-Length.
    Cloud Run also enforces a 32 MB hard cap at the infrastructure level.
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

        # Fast-reject on Content-Length header when present.
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        content_length_raw = headers.get(b"content-length")
        if content_length_raw is not None:
            try:
                if int(content_length_raw) > max_bytes:
                    response = Response("Request body too large", status_code=413)
                    await response(scope, receive, send)
                    return
            except (ValueError, TypeError):
                pass

        # Wrap receive to count actual bytes for chunked or headerless bodies.
        total_received = 0
        oversized = False
        app_response_started = False

        async def limited_receive():
            nonlocal total_received, oversized
            message = await receive()
            if message.get("type") == "http.request" and not oversized:
                total_received += len(message.get("body", b""))
                if total_received > max_bytes:
                    oversized = True
                    # Return an empty terminal chunk so FastAPI stops reading.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def intercepting_send(message):
            nonlocal app_response_started
            if not oversized:
                await send(message)
                return
            # Replace the first response from the app with 413.
            if message["type"] == "http.response.start" and not app_response_started:
                app_response_started = True
                await send({
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [(b"content-type", b"text/plain; charset=utf-8")],
                })
                await send({
                    "type": "http.response.body",
                    "body": b"Request body too large",
                    "more_body": False,
                })
            # Drop the app's body chunks — we already sent our complete response.

        await self.app(scope, limited_receive, intercepting_send)


class SecurityHeaderMiddleware:
    """Inject security-related HTTP response headers on every response.

    Pure ASGI middleware to avoid BaseHTTPMiddleware deadlock risk.
    """

    _HEADERS: list[tuple[bytes, bytes]] = [
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"x-xss-protection", b"1; mode=block"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
        (b"strict-transport-security", b"max-age=31536000; includeSubDomains"),
    ]

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(self._HEADERS)
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


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
                    LEFT JOIN counties co ON p.county_id       = co.id
                    JOIN cancer_types ct ON cd.cancer_type_id  = ct.id
                    JOIN species      s  ON p.species_id      = s.id
                    WHERE p.data_source = 'petbert'
                      AND cd.review_status IN ('confirmed', 'corrected')
                      AND ct.name != 'Non-Cancer'
                    GROUP BY p.county_id, co.name, ct.id, ct.name,
                             s.id, s.name, COALESCE(p.sex, 'Unknown'),
                             EXTRACT(YEAR FROM p.diagnosis_date)
                """))
                await db.execute(text(
                    "CREATE UNIQUE INDEX idx_mv_county_cancer "
                    "ON mv_county_cancer_incidence (county_id, cancer_type_id, species_id, sex, year) "
                    "NULLS NOT DISTINCT"
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
                    LEFT JOIN counties co ON p.county_id       = co.id
                    WHERE p.data_source = 'petbert'
                      AND cd.review_status IN ('confirmed', 'corrected')
                      AND ct.name != 'Non-Cancer'
                    GROUP BY EXTRACT(YEAR FROM p.diagnosis_date),
                             ct.id, ct.name, s.id, s.name,
                             p.county_id, co.name, COALESCE(p.sex, 'Unknown')
                """))
                await db.execute(text(
                    "CREATE UNIQUE INDEX idx_mv_yearly_trends "
                    "ON mv_yearly_trends (year, cancer_type_id, species_id, county_id, sex) "
                    "NULLS NOT DISTINCT"
                ))
                await db.commit()
                logger.info("Migration 023 applied: materialized views rebuilt with sex column")
    except Exception as e:
        logger.warning("Could not apply migration 023: %s", e)

    # Add clinic_name column to ingestion_jobs (migration 024).
    try:
        async with async_session() as db:
            await db.execute(text(
                "ALTER TABLE ingestion_jobs "
                "ADD COLUMN IF NOT EXISTS clinic_name VARCHAR(255)"
            ))
            await db.commit()
    except Exception as e:
        logger.warning("Could not apply migration 024 (clinic_name): %s", e)

    # Add source_diagnosis column to pathology_reports (migration 025).
    try:
        async with async_session() as db:
            await db.execute(text(
                "ALTER TABLE pathology_reports "
                "ADD COLUMN IF NOT EXISTS source_diagnosis TEXT"
            ))
            await db.commit()
    except Exception as e:
        logger.warning("Could not apply migration 025 (source_diagnosis): %s", e)

    # Widen prediction_method from VARCHAR(20) to VARCHAR(50) (migration 026).
    try:
        async with async_session() as db:
            await db.execute(text(
                "ALTER TABLE case_diagnoses "
                "ALTER COLUMN prediction_method TYPE VARCHAR(50)"
            ))
            await db.commit()
    except Exception as e:
        logger.warning("Could not apply migration 026 (prediction_method width): %s", e)

    # Rebuild materialized views to exclude Non-Cancer rows (migration 027).
    try:
        async with async_session() as db:
            row = await db.execute(text(
                "SELECT 1 FROM pg_matviews "
                "WHERE matviewname = 'mv_county_cancer_incidence' "
                "AND definition LIKE '%Non-Cancer%'"
            ))
            if row.scalar() is None:
                logger.info("Applying migration 027: rebuilding materialized views to exclude Non-Cancer")
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
                    LEFT JOIN counties co ON p.county_id       = co.id
                    JOIN cancer_types ct ON cd.cancer_type_id  = ct.id
                    JOIN species      s  ON p.species_id      = s.id
                    WHERE p.data_source = 'petbert'
                      AND cd.review_status IN ('confirmed', 'corrected')
                      AND ct.name != 'Non-Cancer'
                    GROUP BY p.county_id, co.name, ct.id, ct.name,
                             s.id, s.name, COALESCE(p.sex, 'Unknown'),
                             EXTRACT(YEAR FROM p.diagnosis_date)
                """))
                await db.execute(text(
                    "CREATE UNIQUE INDEX idx_mv_county_cancer "
                    "ON mv_county_cancer_incidence (county_id, cancer_type_id, species_id, sex, year) "
                    "NULLS NOT DISTINCT"
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
                    LEFT JOIN counties co ON p.county_id       = co.id
                    WHERE p.data_source = 'petbert'
                      AND cd.review_status IN ('confirmed', 'corrected')
                      AND ct.name != 'Non-Cancer'
                    GROUP BY EXTRACT(YEAR FROM p.diagnosis_date),
                             ct.id, ct.name, s.id, s.name,
                             p.county_id, co.name, COALESCE(p.sex, 'Unknown')
                """))
                await db.execute(text(
                    "CREATE UNIQUE INDEX idx_mv_yearly_trends "
                    "ON mv_yearly_trends (year, cancer_type_id, species_id, county_id, sex) "
                    "NULLS NOT DISTINCT"
                ))
                await db.commit()
                logger.info("Migration 027 applied: materialized views rebuilt without Non-Cancer")
    except Exception as e:
        logger.warning("Could not apply migration 027: %s", e)

    # Switch materialized views from INNER JOIN to LEFT JOIN on counties (migration 028).
    # Patients without a county_id were silently excluded from all counts; they should
    # be included in yearly totals even if they can't be placed on the map.
    try:
        async with async_session() as db:
            row = await db.execute(text(
                "SELECT 1 FROM pg_matviews "
                "WHERE matviewname = 'mv_county_cancer_incidence' "
                "AND definition ILIKE '%left join counties%'"
            ))
            if row.scalar() is None:
                logger.info("Applying migration 028: rebuilding materialized views with LEFT JOIN on counties")
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
                    LEFT JOIN counties co ON p.county_id       = co.id
                    JOIN cancer_types ct ON cd.cancer_type_id  = ct.id
                    JOIN species      s  ON p.species_id      = s.id
                    WHERE p.data_source = 'petbert'
                      AND cd.review_status IN ('confirmed', 'corrected')
                      AND ct.name != 'Non-Cancer'
                    GROUP BY p.county_id, co.name, ct.id, ct.name,
                             s.id, s.name, COALESCE(p.sex, 'Unknown'),
                             EXTRACT(YEAR FROM p.diagnosis_date)
                """))
                await db.execute(text(
                    "CREATE UNIQUE INDEX idx_mv_county_cancer "
                    "ON mv_county_cancer_incidence (county_id, cancer_type_id, species_id, sex, year) "
                    "NULLS NOT DISTINCT"
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
                    LEFT JOIN counties co ON p.county_id       = co.id
                    WHERE p.data_source = 'petbert'
                      AND cd.review_status IN ('confirmed', 'corrected')
                      AND ct.name != 'Non-Cancer'
                    GROUP BY EXTRACT(YEAR FROM p.diagnosis_date),
                             ct.id, ct.name, s.id, s.name,
                             p.county_id, co.name, COALESCE(p.sex, 'Unknown')
                """))
                await db.execute(text(
                    "CREATE UNIQUE INDEX idx_mv_yearly_trends "
                    "ON mv_yearly_trends (year, cancer_type_id, species_id, county_id, sex) "
                    "NULLS NOT DISTINCT"
                ))
                await db.commit()
                logger.info("Migration 028 applied: materialized views rebuilt with LEFT JOIN on counties")
    except Exception as e:
        logger.warning("Could not apply migration 028: %s", e)

    # Rebuild idx_mv_county_cancer with NULLS NOT DISTINCT so concurrent refresh
    # correctly handles patients with no county assignment (migration 029).
    try:
        async with async_session() as db:
            row = await db.execute(text(
                "SELECT indnullsnotdistinct FROM pg_index "
                "WHERE indexrelid = 'idx_mv_county_cancer'::regclass"
            ))
            if not row.scalar():
                logger.info("Applying migration 029: rebuilding idx_mv_county_cancer with NULLS NOT DISTINCT")
                await db.execute(text("DROP INDEX IF EXISTS idx_mv_county_cancer"))
                await db.execute(text(
                    "CREATE UNIQUE INDEX idx_mv_county_cancer "
                    "ON mv_county_cancer_incidence (county_id, cancer_type_id, species_id, sex, year) "
                    "NULLS NOT DISTINCT"
                ))
                await db.execute(text("DROP INDEX IF EXISTS idx_mv_yearly_trends"))
                await db.execute(text(
                    "CREATE UNIQUE INDEX idx_mv_yearly_trends "
                    "ON mv_yearly_trends (year, cancer_type_id, species_id, county_id, sex) "
                    "NULLS NOT DISTINCT"
                ))
                await db.commit()
                logger.info("Migration 029 applied: MV indexes rebuilt with NULLS NOT DISTINCT")
    except Exception as e:
        logger.warning("Could not apply migration 029: %s", e)

    yield


_docs_url = "/docs" if settings.DEBUG else None
_redoc_url = "/redoc" if settings.DEBUG else None
_openapi_url = "/openapi.json" if settings.DEBUG else None

app = FastAPI(
    title=settings.APP_TITLE,
    version=settings.APP_VERSION,
    description="UC Davis Veterinary Medical Teaching Hospital Cancer Registry API",
    lifespan=lifespan,
    docs_url=_docs_url,
    redoc_url=_redoc_url,
    openapi_url=_openapi_url,
)

# --- Rate limiting ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["authorization", "content-type", "x-requested-with"],
)

# --- Security headers on every response ---
app.add_middleware(SecurityHeaderMiddleware)

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
app.include_router(admin_router_module.router)


@app.get("/")
async def root():
    return {
        "name": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        **({"docs": "/docs"} if settings.DEBUG else {}),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
