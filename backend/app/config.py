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
