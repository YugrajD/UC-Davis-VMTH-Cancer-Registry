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
    SUPABASE_JWT_SECRET: str

    CORS_ORIGINS: str = '["http://localhost:5173"]'
    APP_TITLE: str = "UC Davis VMTH Cancer Registry API"
    APP_VERSION: str = "1.0.0"
    ML_WORKER_URL: str = "http://localhost:8001"
    SUPABASE_URL: str = ""
    # Comma-separated email lists. Admins implicitly hold uploader and
    # reviewer privileges, so these env vars only need users who don't
    # also appear in ADMIN_EMAILS.
    ADMIN_EMAILS: str = ""
    UPLOADER_EMAILS: str = ""
    REVIEWER_EMAILS: str = ""
    UPLOAD_DIR: str = "/app/uploads"

    # GCP Batch — set USE_GCP_BATCH=true to route ML inference through GCP
    USE_GCP_BATCH: bool = False
    GCP_PROJECT_ID: str = ""
    GCP_REGION: str = "us-central1"
    GCS_BUCKET: str = ""
    GCP_BATCH_IMAGE_URI: str = ""
    GCP_BATCH_MACHINE_TYPE: str = "n1-standard-4"
    GCP_BATCH_POLL_INTERVAL: int = 60
    GCP_BATCH_TIMEOUT_HOURS: int = 12
    GCP_BATCH_SERVICE_ACCOUNT: str = ""

    # SMTP — email notifications for role requests (all default to empty = disabled)
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM: str = ""

    # Per-diagnosis manual review thresholds (conservative defaults; tune
    # against a labeled validation set before relying on the auto-accept
    # band). A row is auto-confirmed at ingest only when *both* gates pass.
    REVIEW_AUTO_ACCEPT_CONFIDENCE: float = 0.85
    REVIEW_AUTO_ACCEPT_MARGIN: float = 0.10  # top1 - top2 spread

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

    # Reverse-proxy trust — comma-separated IPs whose X-Forwarded-For
    # header is trusted for rate-limiting.  Empty (default) means only
    # the TCP peer address is used, which is the safe default.
    FORWARDED_ALLOW_IPS: str = ""

    @property
    def forwarded_allow_ips_set(self) -> set:
        if not self.FORWARDED_ALLOW_IPS:
            return set()
        return {ip.strip() for ip in self.FORWARDED_ALLOW_IPS.split(",") if ip.strip()}

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
