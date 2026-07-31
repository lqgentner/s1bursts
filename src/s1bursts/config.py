"""Environment-loaded settings for the Sentinel-1 pipeline."""

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
        env_file=".env",
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
        env_file=".env",
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
        env_file=".env",
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
    """Load settings from the environment (and a ``.env`` file, if present).

    Settings are re-read on every call so that changes to the environment
    made after import (e.g. a late ``load_dotenv()``) are picked up.

    Returns
    -------
    Settings
        Freshly resolved settings.
    """
    return Settings()
