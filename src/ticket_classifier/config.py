"""Shared configuration for training, inference, and monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

try:  # pydantic-settings is a runtime dependency, but keep CLI imports helpful.
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal bootstrap envs.
    BaseSettings = object  # type: ignore[assignment,misc]
    Field = None  # type: ignore[assignment]


DEFAULT_BASE_MODEL = "microsoft/deberta-v3-small"
DEFAULT_SEED = 42
DEFAULT_MAX_LENGTH = 160
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 3
DEFAULT_MODEL_VERSION = "1.2.0"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_DRIFT_WINDOW = 1000
DEFAULT_DRIFT_THRESHOLD = 0.1


if Field is not None:

    class Settings(BaseSettings):
        """Environment-driven settings; no secrets or paths are hardcoded."""

        model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

        environment: Literal["dev", "staging", "prod"] = "dev"
        model_dir: Path = Field(default=Path("output/model"), validation_alias="MODEL_DIR")
        model_version: str = Field(default=DEFAULT_MODEL_VERSION, validation_alias="MODEL_VERSION")
        base_model: str = Field(default=DEFAULT_BASE_MODEL, validation_alias="BASE_MODEL")
        max_input_chars: int = Field(default=2000, validation_alias="MAX_INPUT_CHARS")
        confidence_threshold: float = Field(default=DEFAULT_CONFIDENCE_THRESHOLD, validation_alias="CONFIDENCE_THRESHOLD")
        low_confidence_threshold: float = Field(
            default=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
            validation_alias="LOW_CONFIDENCE_THRESHOLD",
        )
        drift_window: int = Field(default=DEFAULT_DRIFT_WINDOW, validation_alias="DRIFT_WINDOW")
        drift_threshold: float = Field(default=DEFAULT_DRIFT_THRESHOLD, validation_alias="DRIFT_THRESHOLD")
        host: str = Field(default="0.0.0.0", validation_alias="HOST")
        port: int = Field(default=8080, validation_alias="PORT")
        log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")


else:

    class Settings:  # type: ignore[no-redef]
        """Fallback used only before dependencies are installed."""

        environment = "dev"
        model_dir = Path("output/model")
        model_version = DEFAULT_MODEL_VERSION
        base_model = DEFAULT_BASE_MODEL
        max_input_chars = 2000
        confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        low_confidence_threshold = DEFAULT_LOW_CONFIDENCE_THRESHOLD
        drift_window = DEFAULT_DRIFT_WINDOW
        drift_threshold = DEFAULT_DRIFT_THRESHOLD
        host = "0.0.0.0"
        port = 8080
        log_level = "INFO"


def get_settings() -> Settings:
    """Return settings from environment variables."""
    return Settings()
