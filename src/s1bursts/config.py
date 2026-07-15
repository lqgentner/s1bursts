"""Environment-loaded settings for the Sentinel-1 pipeline."""

from collections.abc import Iterator
from contextlib import contextmanager
import os
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

    @contextmanager
    def s3_env(self) -> Iterator[None]:
        """Temporarily set ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``.

        The burst extraction script reads standard AWS env vars.  This context
        manager injects the CDSE S3 credentials and restores the previous
        values on exit.

        Raises
        ------
        ValueError
            If credentials are not configured.
        """
        if self.access_key_id is None or self.secret_access_key is None:
            msg = (
                "CDSE S3 credentials not configured. "
                "Set CDSE_S3_ACCESS_KEY_ID and CDSE_S3_SECRET_ACCESS_KEY."
            )
            raise ValueError(msg)

        prev_key = os.environ.get("AWS_ACCESS_KEY_ID")
        prev_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
        try:
            os.environ["AWS_ACCESS_KEY_ID"] = self.access_key_id
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_access_key.get_secret_value()
            yield
        finally:
            for var, prev in [
                ("AWS_ACCESS_KEY_ID", prev_key),
                ("AWS_SECRET_ACCESS_KEY", prev_secret),
            ]:
                if prev is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = prev


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

    gpt_path: Path | None = None
    auxdata_path: Path = Path.home() / ".snap/auxdata"
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
