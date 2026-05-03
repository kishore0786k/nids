"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.project_paths import PROJECT_ROOT


class Settings(BaseSettings):  # type: ignore[misc]
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_prefix="NIDS_",
        extra="ignore",
    )

    app_name: str = "Neuro-Symbolic NIDS"
    app_version: str = "1.0.0"
    environment: str = "production"
    host: str = "0.0.0.0"
    port: int = 5000
    log_level: str = "INFO"
    log_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "logs")
    log_file: str = "nids.log"
    max_upload_mb: int = 16
    allowed_upload_extensions: str = ".pcap,.pcapng,.csv"
    rate_limit_default: str = "120 per minute"
    rate_limit_run_all: str = "10 per minute"
    runs_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "runs")
    frontend_dir: Path = Field(default_factory=lambda: PROJECT_ROOT / "frontend")

    @property
    def allowed_upload_suffixes(self) -> set[str]:
        return {
            item.strip().lower()
            for item in self.allowed_upload_extensions.split(",")
            if item.strip()
        }

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb) * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
