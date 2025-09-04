from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project configuration loaded from environment with safe defaults.

    Reads from `.env` if present and environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Sliding window
    WINDOW_SIZE: int = Field(default=100, ge=1, description="Number of logs kept in the sliding window")

    # Detection
    THRESHOLD: float = Field(default=0.1, ge=0.0, le=1.0, description="Probability threshold for anomaly")
    ALERT_ANOMALY_COUNT: int = Field(default=2, ge=1, description="Minimum anomalies in window to alert")

    # Stream source
    STREAM_SOURCE: Literal["file", "stdin"] = Field(default="file", description="Log input source")
    LOG_FILE_PATH: str = Field(default="data/sample_logs.txt", description="Path to log file when STREAM_SOURCE='file'")

    # LogBERT real model (optional). If paths are provided, runner can use real mode.
    LOGBERT_MODEL_PATH: Optional[str] = Field(default=None, description="Path to external LogBERT checkpoint (best_bert.pth)")
    LOGBERT_VOCAB_PATH: Optional[str] = Field(default=None, description="Path to external LogBERT vocab.pkl")
    LOGBERT_DEVICE: Optional[str] = Field(default=None, description="Device for LogBERT real mode: cpu or cuda")


# Singleton-style convenient accessor
settings = Settings()

__all__ = ["Settings", "settings"]
