"""Environment-loaded settings for the Sentinel-1 pipeline.

Settings are read from the process environment only. Loading a ``.env`` file is
left to the application: Call ``dotenv.load_dotenv()`` before :func:`get_settings`
or export the variables (e.g. ``uv run --env-file .env``).
"""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class CDSES3Settings(BaseSettings):
    """CDSE S3 credentials for the burst extraction script.

    Loaded from ``CDSE_S3_*`` environment variables. Obtain credentials at
    https://eodata-s3keysmanager.dataspace.copernicus.eu/
    """

    model_config = SettingsConfigDict(
        env_prefix="CDSE_S3_",
        case_sensitive=False,
        extra="ignore",
    )

    endpoint_url: str = "https://eodata.dataspace.copernicus.eu"
    access_key_id: str | None = None
    secret_access_key: SecretStr | None = None


class CDSESettings(BaseSettings):
    """Copernicus Data Space Ecosystem credentials.

    Loaded from ``CDSE_*`` environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="CDSE_",
        case_sensitive=False,
        extra="ignore",
    )

    username: str | None = None
    password: SecretStr | None = None
    s3: CDSES3Settings = Field(default_factory=CDSES3Settings)


class SNAPSettings(BaseSettings):
    """SNAP GPT executable location and resource budget.

    Loaded from ``SNAP_*`` environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="SNAP_",
        case_sensitive=False,
        extra="ignore",
    )

    bin: Path | None = None
    auxdata_path: Path = Path.home() / ".snap/auxdata"
    dem_cache: Path = Path.home() / ".s1bursts/dem"
    max_workers: int = 1
    max_ram_gb: float = 4.0
    cache_ratio: float = 0.70
    threads_per_job: int | None = None


class Settings(BaseSettings):
    """Top-level settings container."""

    model_config = SettingsConfigDict(extra="ignore")

    cdse: CDSESettings = Field(default_factory=CDSESettings)
    snap: SNAPSettings = Field(default_factory=SNAPSettings)


def get_settings() -> Settings:
    """Load settings from the process environment.

    Settings are re-read on every call so that changes to the environment
    made after import (e.g. a late ``load_dotenv()``) are picked up.

    Returns
    -------
    Settings
        Freshly resolved settings.
    """
    return Settings()
