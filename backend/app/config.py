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

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.CORS_ORIGINS)

    class Config:
        env_file = ".env"


settings = Settings()
