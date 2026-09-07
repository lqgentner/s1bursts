"""Environment-loaded settings for the Sentinel-1 pipeline.

Settings are read from the process environment only. Loading a ``.env`` file is
left to the application: Call ``dotenv.load_dotenv()`` before :func:`get_settings`
or export the variables (e.g. ``uv run --env-file .env``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class CDSES3Settings(BaseSettings):
    """CDSE S3 credentials for extracting bursts from parent products.

    Loaded from ``CDSE_S3_*`` environment variables. Obtain credentials at
    https://eodata-s3keysmanager.dataspace.copernicus.eu/

    Authenticate either with a named AWS profile (``CDSE_S3_PROFILE``, resolved
    from ``~/.aws/credentials``, which keeps the keys out of the environment) or
    with an explicit key pair (``CDSE_S3_ACCESS_KEY_ID`` and
    ``CDSE_S3_SECRET_ACCESS_KEY``). The profile takes precedence if both are set.
    """

    model_config = SettingsConfigDict(
        env_prefix="CDSE_S3_",
        case_sensitive=False,
        extra="ignore",
        # An unconfigured GitHub Actions secret interpolates as an empty
        # string rather than being omitted; without this, an empty string
        # env var would be treated as set, defeating the `is None` skip
        # guards in tests/test_integration.py.
        env_ignore_empty=True,
    )

    endpoint_url: str = "https://eodata.dataspace.copernicus.eu"
    profile: str | None = None
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
        env_ignore_empty=True,
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
        env_ignore_empty=True,
    )

    bin: Path | None = None
    auxdata_path: Path = Path.home() / ".snap/auxdata"
    dem_cache: Path = Path.home() / ".s1bursts/dem"
    max_workers: int = Field(default=1, ge=1)
    max_ram_gb: float = Field(default=4.0, gt=0)
    # A ratio above 1 would size the GPT tile cache larger than the heap it has
    # to fit inside; a ratio of 0 or below would disable caching entirely.
    cache_ratio: float = Field(default=0.70, gt=0, le=1)
    threads_per_job: int | None = Field(default=None, ge=1)


class Settings(BaseSettings):
    """Top-level settings container."""

    model_config = SettingsConfigDict(extra="ignore")

    cdse: CDSESettings = Field(default_factory=CDSESettings)
    snap: SNAPSettings = Field(default_factory=SNAPSettings)


def warn_if_tmpdir_unusable() -> None:
    """Warn when ``TMPDIR`` is set but cannot be written to.

    ``tempfile`` silently skips an unusable ``TMPDIR`` and lands on ``/tmp``,
    which on many hosts is a RAM-backed tmpfs — usually the very thing an
    explicit ``TMPDIR`` was set to avoid.
    """
    configured = os.environ.get("TMPDIR")
    if not configured or os.access(configured, os.W_OK | os.X_OK):
        return
    logger.warning(
        "TMPDIR is set to '%s', which is not a writable directory; staging will "
        "fall back to '%s'. Create it, or unset TMPDIR to silence this.",
        configured,
        tempfile.gettempdir(),
    )


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
