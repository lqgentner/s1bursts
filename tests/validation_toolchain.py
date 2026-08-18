"""Provenance of a cross-validation campaign: what produced the numbers.

The measurements a campaign records are only meaningful against a stated SNAP,
Java and package version, so every regeneration script writes those alongside
its results and keys its on-disk product cache to them.
"""

from __future__ import annotations

from contextlib import suppress
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

import numpy as np

import s1bursts
from s1bursts.config import get_settings
from s1bursts.errors import ConfigurationError
from s1bursts.snap_client import _resolve_gpt_path

if TYPE_CHECKING:
    import pytest

    from s1bursts.snap_graph import GeocodingConfig

WORKDIR_ENV = "S1BURSTS_VALIDATION_WORKDIR"


def workdir(tmp_path_factory: pytest.TempPathFactory, name: str) -> Path:
    """Directory a campaign downloads and processes into.

    A temporary one per session by default. Set ``S1BURSTS_VALIDATION_WORKDIR``
    -- the same variable the regeneration scripts take -- to reuse what a
    regeneration already downloaded and processed, which turns a validation run
    from tens of minutes of GPT into a few seconds of arithmetic.
    """
    configured = os.environ.get(WORKDIR_ENV)
    if configured:
        path = Path(configured)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path_factory.mktemp(name)


def snap_home() -> Path | None:
    """Directory the resolved ``gpt`` lives under, if one resolves at all."""
    with suppress(ConfigurationError, OSError):
        return Path(_resolve_gpt_path(None, get_settings().snap)).parent.parent
    return None


def snap_version() -> str:
    """SNAP's own version string, or ``"unknown"`` if it cannot be read."""
    home = snap_home()
    if home is not None:
        with suppress(OSError):
            return (home / "VERSION.txt").read_text().strip()
    return "unknown"


def processing_fingerprint(config: GeocodingConfig, crs: str) -> str:
    """Short hash of everything that determines the raster a campaign produces.

    Products are cached on disk so that a regeneration interrupted part-way
    through gigabytes of downloads can resume. That cache must not survive a
    change to the processing itself, because the reason to regenerate is
    usually a SNAP upgrade or a graph change -- reusing an old raster would
    record the new toolchain beside measurements from the old one.

    The SNAP version and the graph XML are hashed rather than the package
    version, so that unrelated commits do not needlessly discard hours of work.

    Parameters
    ----------
    config : GeocodingConfig
        Graph configuration the campaign processes with.
    crs : str
        Target CRS of the processed product.

    Returns
    -------
    str
        Twelve hexadecimal characters.

    Raises
    ------
    RuntimeError
        If the SNAP version cannot be read. Provenance may degrade to
        ``"unknown"``, but a cache key must not: two different SNAP versions
        would then hash alike and the stale products this key exists to discard
        would be reused.
    """
    version = snap_version()
    if version == "unknown":
        msg = (
            "cannot read the SNAP version, so cached products cannot be keyed "
            "safely; set SNAP_BIN (or run with `--env-file .env`) before regenerating"
        )
        raise RuntimeError(msg)

    material = f"{version}\n{crs}\n{config.to_xml()}"
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def toolchain(**extra: Any) -> dict[str, Any]:
    """Record the versions a campaign's numbers are only meaningful against.

    Provenance is best-effort: a missing version string must degrade to
    ``"unknown"`` rather than abort a regeneration that already cost tens of
    minutes of downloads and GPT runs.

    Parameters
    ----------
    **extra
        Campaign-specific entries to record beside the versions, such as the
        grid spacing the comparison ran at.

    Returns
    -------
    dict
        The versions in play, plus ``extra``.
    """
    home = snap_home()

    # SNAP bundles its own JRE, and that is the one GPT actually runs on.
    java = shutil.which("java")
    if home is not None and (home / "jre" / "bin" / "java").exists():
        java = str(home / "jre" / "bin" / "java")

    java_version = "unknown"
    if java:
        with suppress(OSError, subprocess.SubprocessError, IndexError):
            result = subprocess.run(  # noqa: S603
                [java, "-version"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            java_version = (result.stderr or result.stdout).splitlines()[0].strip()

    return {
        "s1bursts": getattr(s1bursts, "__version__", "unknown"),
        "snap": snap_version(),
        "java": java_version,
        "python": platform.python_version(),
        "numpy": np.__version__,
        **extra,
    }
