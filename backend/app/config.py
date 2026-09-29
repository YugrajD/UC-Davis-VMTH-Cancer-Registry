"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import json


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required — no defaults.  The app will refuse to start if these are
    # unset, preventing silent fallback to hardcoded credentials.
    DATABASE_URL: str
    DATABASE_URL_SYNC: str
    COGNITO_USER_POOL_ID: str
    COGNITO_CLIENT_ID: str

    DEBUG: bool = False

    CORS_ORIGINS: str = '["http://localhost:5173"]'
    APP_TITLE: str = "UC Davis VMTH Cancer Registry API"
    APP_VERSION: str = "1.0.0"
    ML_WORKER_URL: str = "http://localhost:8001"
    # Base URL of the Cognito IdP-compatible token issuer used for JWKS
    # lookups — e.g. https://cognito-idp.{region}.amazonaws.com in production,
    # or the local Floci endpoint in dev.
    COGNITO_ISSUER_URL: str = "https://cognito-idp.us-east-1.amazonaws.com"
    # Comma-separated email lists. Admins implicitly hold uploader and
    # reviewer privileges, so these env vars only need users who don't
    # also appear in ADMIN_EMAILS.
    ADMIN_EMAILS: str = ""
    UPLOADER_EMAILS: str = ""
    REVIEWER_EMAILS: str = ""

    # S3 blob storage — uploaded CSVs (uploads/), report text (reports/) and
    # model bundles (models/). Set AWS_S3_ENDPOINT_URL to point at Floci in
    # local dev; leave empty in AWS.
    S3_BUCKET: str = ""
    AWS_REGION: str = "us-east-1"
    AWS_S3_ENDPOINT_URL: str = ""

    # ECS Fargate ML task — set USE_ECS_ML=true to run PetBERT inference as an
    # on-demand Fargate task instead of the local ml-worker container.
    USE_ECS_ML: bool = False
    ECS_CLUSTER_ARN: str = ""
    ML_TASK_DEFINITION_ARN: str = ""
    ML_TASK_CONTAINER_NAME: str = "ml-worker"
    # Comma-separated subnet IDs for the ML task's network configuration.
    ML_TASK_SUBNET_IDS: str = ""
    ML_TASK_SECURITY_GROUP_ID: str = ""
    ML_POLL_INTERVAL: int = 60
    ML_TIMEOUT_HOURS: int = 12
    ML_CLEANUP_JOB_FILES: bool = False

    # PetBERT runtime thresholds. Case presence is Stage 1: rows below this
    # become method=low_confidence / Uncategorized before review thresholds run.
    CASE_PRESENCE_THRESHOLD: float = 0.5
    GROUP_CLASSIFIER_THRESHOLD: float = 0.3

    # SMTP — email notifications for role requests (all default to empty = disabled)
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = ""

    # Per-diagnosis manual review thresholds. These defaults are calibrated
    # from ml/data/validation/review_threshold_validation.csv; re-run
    # scripts/analyze_review_thresholds.py before changing the model or labels.
    # A row is auto-confirmed at ingest only when *both* gates pass.
    REVIEW_AUTO_ACCEPT_CONFIDENCE: float = 0.23
    REVIEW_AUTO_ACCEPT_MARGIN: float = 0.15  # top1 - top2 spread

    # Rate limiting
    RATE_LIMIT_DEFAULT: str = "120/minute"            # authenticated users
    RATE_LIMIT_ANONYMOUS: str = "30/minute"            # unauthenticated (IP-keyed)
    RATE_LIMIT_WRITE: str = "10/minute"                # write/upload endpoints
    RATE_LIMIT_EXPENSIVE: str = "10/minute"            # ML classify

    # Response cache TTLs (seconds)
    CACHE_TTL_DASHBOARD: int = 60       # 1 minute — summary/filters
    CACHE_TTL_INCIDENCE: int = 60       # 1 minute — aggregation queries
    CACHE_TTL_GEO: int = 300            # 5 minutes — GeoJSON (large, slow)
    CACHE_TTL_TRENDS: int = 60          # 1 minute — time series
    CACHE_TTL_CALENVIRO: int = 3600     # 1 hour — static reference data
    CACHE_MAX_SIZE: int = 256           # max entries per cache namespace

    # Gunicorn workers per container.  Keep at 1 on Cloud Run — the service
    # scales horizontally (more instances), not vertically (more workers).
    # Raise only when also running Redis for distributed rate-limiting/caching,
    # and only to match the container's vCPU count.
    WORKERS: int = 1

    # PostgreSQL connection pool per worker process.
    # Capacity rule: DB_POOL_SIZE × WORKERS × max_instances ≤ PG max_connections.
    # Supabase Pooler users can relax this (PgBouncer multiplexes connections).
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # Reverse-proxy trust — comma-separated IPs whose X-Forwarded-For header
    # is trusted for rate-limiting.  Empty (default) means only the TCP peer
    # address is used, which is safe.  On Cloud Run, set this to the internal
    # Google Front End IP range (e.g. "0.0.0.0/0" or the specific GFE range)
    # so client IPs are not all seen as the same load-balancer address.
    FORWARDED_ALLOW_IPS: str = ""

    @property
    def forwarded_allow_ips_set(self) -> set:
        if not self.FORWARDED_ALLOW_IPS:
            return set()
        return {ip.strip() for ip in self.FORWARDED_ALLOW_IPS.split(",") if ip.strip()}

    @property
    def ml_task_subnet_ids_list(self) -> List[str]:
        return [s.strip() for s in self.ML_TASK_SUBNET_IDS.split(",") if s.strip()]

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.CORS_ORIGINS)

    @property
    def admin_emails_list(self) -> List[str]:
        if not self.ADMIN_EMAILS:
            return []
        return [e.strip() for e in self.ADMIN_EMAILS.split(",") if e.strip()]

    @property
    def uploader_emails_list(self) -> List[str]:
        if not self.UPLOADER_EMAILS:
            return []
        return [e.strip() for e in self.UPLOADER_EMAILS.split(",") if e.strip()]

    @property
    def reviewer_emails_list(self) -> List[str]:
        if not self.REVIEWER_EMAILS:
            return []
        return [e.strip() for e in self.REVIEWER_EMAILS.split(",") if e.strip()]


settings = Settings()
