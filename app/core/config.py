from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")

    project_root: Path = PROJECT_ROOT
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    data_dir: Path = PROJECT_ROOT / "data"
    app_data_dir: Path = PROJECT_ROOT / "app_data"
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'app_data' / 'app.db').as_posix()}"

    api_key: str = "change-me-dev-key"

    spatial_index_k: int = 30
    drift_min_samples: int = 30

    log_level: str = "INFO"
    cors_origins: list[str] = ["*"]


settings = Settings()
settings.app_data_dir.mkdir(parents=True, exist_ok=True)
