"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@db:5432/vmth_cancer"
    DATABASE_URL_SYNC: str = "postgresql://postgres:postgres@db:5432/vmth_cancer"
    CORS_ORIGINS: str = '["http://localhost:5173"]'
    APP_TITLE: str = "UC Davis VMTH Cancer Registry API"
    APP_VERSION: str = "1.0.0"
    ML_WORKER_URL: str = "http://localhost:8001"
    SUPABASE_JWT_SECRET: str = ""
    SUPABASE_URL: str = ""
    ADMIN_EMAILS: str = ""
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

    # Per-diagnosis manual review thresholds (conservative defaults; tune
    # against a labeled validation set before relying on the auto-accept
    # band). A row is auto-confirmed at ingest only when *both* gates pass.
    REVIEW_AUTO_ACCEPT_CONFIDENCE: float = 0.85
    REVIEW_AUTO_ACCEPT_MARGIN: float = 0.10  # top1 - top2 spread

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.CORS_ORIGINS)

    @property
    def admin_emails_list(self) -> List[str]:
        if not self.ADMIN_EMAILS:
            return []
        return [e.strip() for e in self.ADMIN_EMAILS.split(",") if e.strip()]

    class Config:
        env_file = ".env"


settings = Settings()
