"""Make an external DEM sample correctly under SNAP."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from secrets import token_hex
import time
from typing import Any

import numpy as np
import rasterio
from rasterio.errors import RasterioError
from rasterio.transform import Affine

from s1bursts.config import get_settings

logger = logging.getLogger(__name__)

# Marks a copy as deliberately offset, so it cannot be mistaken for a canonical
# DEM if it turns up outside the cache.
COMPENSATED_TAG = "SNAP_HALF_PIXEL_COMPENSATED"

# Bumping this invalidates every cache entry. Raise it whenever the conversion
# changes in a way that alters the written raster (offset, dtype handling,
# compression, block layout), so that stale entries are never reused.
CACHE_SCHEMA_VERSION = 1

# Suffix of the JSON provenance sidecar written next to each cache entry.
PROVENANCE_SUFFIX = ".provenance.json"

_HASH_CHUNK_BYTES = 4 * 1024 * 1024


def _file_digest(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_entry(dem: Path, cache_dir: Path, source_digest: str) -> Path:
    """Return the cache path for a source content digest.

    The name carries the schema version and a prefix of the source's content
    digest, so two different DEMs never share an entry no matter how their
    files are named.
    """
    stem = Path(dem.name).stem
    suffix = "".join(Path(dem.name).suffixes) or ".tif"
    return cache_dir / f"{stem}-v{CACHE_SCHEMA_VERSION}-{source_digest[:16]}{suffix}"


def _read_provenance(sidecar: Path) -> dict[str, Any] | None:
    """Return the parsed provenance record, or None if it is absent or unusable."""
    try:
        record = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _entry_is_intact(dest: Path) -> bool:
    """Check the written raster still opens and is still marked as compensated."""
    try:
        with rasterio.open(dest) as src:
            tagged = src.tags().get(COMPENSATED_TAG) == "1"
    except (RasterioError, OSError):
        logger.warning("Compensated DEM '%s' could not be opened; rewriting", dest.name)
        return False

    if not tagged:
        logger.warning("Compensated DEM '%s' lost its compensation tag; rewriting", dest.name)
        return False
    return True


def _cache_entry_is_valid(dest: Path, sidecar: Path, source_digest: str) -> bool:
    """Check that a cache entry matches its provenance and is readable.

    Verifies the recorded source identity and schema version, then confirms the
    written raster is intact: the byte size must match what was recorded, the
    file must open, and it must carry the compensation tag. A truncated or
    overwritten entry therefore fails rather than being handed to SNAP.
    """
    if not dest.exists():
        return False

    record = _read_provenance(sidecar)
    if record is None:
        logger.debug("Cache entry '%s' has no readable provenance record", dest.name)
        return False

    matches_source = (
        record.get("source_sha256") == source_digest
        and record.get("schema_version") == CACHE_SCHEMA_VERSION
        and record.get("compensated_tag") == COMPENSATED_TAG
    )
    if not matches_source:
        logger.debug("Cache entry '%s' does not match the requested source", dest.name)
        return False

    if dest.stat().st_size != record.get("output_size"):
        logger.warning("Compensated DEM '%s' has an unexpected size; rewriting", dest.name)
        return False

    return _entry_is_intact(dest)


def _compensate(dem: Path, dest: Path) -> None:
    """Write the half-pixel-shifted copy of ``dem`` to ``dest``.

    A unique suffix keeps concurrent writers from clobbering one another's
    partial file; the rename that publishes it is atomic.
    """
    partial = dest.with_name(f"{dest.name}.{token_hex(8)}.partial")
    try:
        with rasterio.open(dem) as src:
            profile = src.profile.copy()
            profile.update(
                transform=src.transform * Affine.translation(0.5, 0.5),
                driver="GTiff",
                tiled=True,
                blockxsize=512,
                blockysize=512,
                compress="deflate",
                # Horizontal differencing is only defined for integers; floating
                # point elevations need the dedicated float predictor.
                predictor=3 if np.issubdtype(np.dtype(src.dtypes[0]), np.floating) else 2,
                BIGTIFF="IF_SAFER",
            )
            with rasterio.open(partial, "w", **profile) as dst:
                dst.update_tags(**{COMPENSATED_TAG: "1"})
                for _, window in src.block_windows(1):
                    dst.write(src.read(window=window), window=window)
        partial.replace(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _write_provenance(sidecar: Path, record: dict[str, Any]) -> None:
    """Write a provenance record atomically next to its cache entry."""
    partial = sidecar.with_name(f"{sidecar.name}.{token_hex(8)}.partial")
    try:
        partial.write_text(json.dumps(record, indent=2, sort_keys=True))
        partial.replace(sidecar)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def compensate_dem_for_snap(dem: Path, *, cache_dir: Path | None = None) -> Path:
    """Return a copy of ``dem`` that SNAP samples at the right position.

    Writes a copy whose geotransform origin sits on the centre of the first pixel
    rather than on its outer corner, which is where SNAP expects it.

    The result is correct **only** when read with SNAP's convention: GDAL, rasterio
    and QGIS place it half a pixel off. It carries a ``SNAP_HALF_PIXEL_COMPENSATED``
    tag and must not be used for anything but ``externalDEMFile``.

    The copy is cached so that a pipeline processing many bursts against one DEM
    pays for the conversion once. The cache is keyed by the source's content
    digest. Each entry is written with a JSON provenance sidecar that is checked,
    along with the entry's own integrity, before the entry is reused.
    Writes are atomic, so concurrent workers may call this on the same DEM.

    Parameters
    ----------
    dem : Path
        Local DEM to convert. A DEM held on object storage has to be copied to
        disk first, since SNAP takes a filesystem path.
    cache_dir : Path, optional
        Directory to write the copy to. Defaults to ``SNAP_DEM_CACHE``
        (``~/.s1bursts/dem``).

    Returns
    -------
    Path
        Path to the compensated copy.

    Raises
    ------
    FileNotFoundError
        If ``dem`` does not exist.
    ValueError
        If ``dem`` resolves into ``cache_dir``.
    """
    if not dem.exists():
        msg = f"DEM not found: '{dem}'"
        raise FileNotFoundError(msg)

    cache_dir = cache_dir or get_settings().snap.dem_cache
    cache_dir.mkdir(parents=True, exist_ok=True)

    if dem.resolve().parent == cache_dir.resolve():
        msg = f"DEM is already inside the compensated-DEM cache: '{dem}'"
        raise ValueError(msg)

    source_digest = _file_digest(dem)
    dest = _cache_entry(dem, cache_dir, source_digest)
    sidecar = dest.with_name(dest.name + PROVENANCE_SUFFIX)

    if _cache_entry_is_valid(dest, sidecar, source_digest):
        logger.debug("Reusing compensated DEM '%s'", dest)
        return dest

    logger.info("Writing SNAP-compensated DEM '%s' -> '%s'", dem, dest)
    start = time.perf_counter()
    _compensate(dem, dest)
    _write_provenance(
        sidecar,
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "compensated_tag": COMPENSATED_TAG,
            "source_path": str(dem.resolve()),
            "source_sha256": source_digest,
            "source_size": dem.stat().st_size,
            "output_name": dest.name,
            "output_size": dest.stat().st_size,
            "pixel_offset": [0.5, 0.5],
        },
    )
    logger.info("Wrote '%s' in %.1f s", dest.name, time.perf_counter() - start)
    return dest
