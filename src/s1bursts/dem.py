"""Prepare an external DEM for use by SNAP."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from secrets import token_hex
from tempfile import TemporaryDirectory
import time
from typing import Any
import warnings

import numpy as np
from pyproj import CRS, Transformer, datadir, proj_version_str
from pyproj.transformer import TransformerGroup
import rasterio
from rasterio.errors import RasterioError
from rasterio.transform import Affine

from s1bursts.config import get_settings

logger = logging.getLogger(__name__)

# Marks a copy as deliberately offset, so it cannot be mistaken for a canonical
# DEM if it turns up outside the cache.
COMPENSATED_TAG = "SNAP_HALF_PIXEL_COMPENSATED"
VERTICAL_CONVERSION_TAG = "SNAP_VERTICAL_DATUM_CONVERTED"

TARGET_CRS = "EPSG:4979"

# Bumping this invalidates every cache entry. Raise it whenever preparation
# changes in a way that alters the written raster or its cache identity.
CACHE_SCHEMA_VERSION = 2

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


def _preparation_digest(source_crs: str | None) -> str:
    """Return a stable digest of options that affect the prepared raster."""
    payload = json.dumps(
        {"source_crs": source_crs, "target_crs": TARGET_CRS if source_crs else None},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_entry(
    dem: Path,
    cache_dir: Path,
    source_digest: str,
    preparation_digest: str,
) -> Path:
    """Return the cache path for source content and preparation options."""
    stem = Path(dem.name).stem
    suffix = "".join(Path(dem.name).suffixes) or ".tif"
    return cache_dir / (
        f"{stem}-v{CACHE_SCHEMA_VERSION}-{source_digest[:16]}-{preparation_digest[:12]}{suffix}"
    )


def _read_provenance(sidecar: Path) -> dict[str, Any] | None:
    """Return the parsed provenance record, or None if it is absent or unusable."""
    try:
        record = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _entry_is_intact(dest: Path, *, converted: bool) -> bool:
    """Check that the prepared raster opens and carries the expected tags."""
    try:
        with rasterio.open(dest) as src:
            tags = src.tags()
            tagged = tags.get(COMPENSATED_TAG) == "1"
            conversion_matches = tags.get(VERTICAL_CONVERSION_TAG) == str(int(converted))
    except (RasterioError, OSError):
        logger.warning("Prepared DEM '%s' could not be opened; rewriting", dest.name)
        return False

    if not tagged or not conversion_matches:
        logger.warning("Prepared DEM '%s' has unexpected preparation tags; rewriting", dest.name)
        return False
    return True


def _cache_entry_is_valid(
    dest: Path,
    sidecar: Path,
    source_digest: str,
    preparation_digest: str,
    *,
    converted: bool,
) -> bool:
    """Check that a cache entry matches its provenance and is readable."""
    if not dest.exists():
        return False

    record = _read_provenance(sidecar)
    if record is None:
        logger.debug("Cache entry '%s' has no readable provenance record", dest.name)
        return False

    matches_source = (
        record.get("source_sha256") == source_digest
        and record.get("preparation_sha256") == preparation_digest
        and record.get("schema_version") == CACHE_SCHEMA_VERSION
        and record.get("compensated_tag") == COMPENSATED_TAG
    )
    if not matches_source:
        logger.debug("Cache entry '%s' does not match the requested source", dest.name)
        return False

    if dest.stat().st_size != record.get("output_size"):
        logger.warning("Prepared DEM '%s' has an unexpected size; rewriting", dest.name)
        return False

    return _entry_is_intact(dest, converted=converted)


def _normalise_source_crs(dem: Path, source_crs: str | None) -> tuple[CRS | None, str | None]:
    """Validate and canonicalise the optional compound source CRS."""
    if source_crs is None:
        return None, None

    try:
        compound = CRS.from_user_input(source_crs)
    except Exception as err:
        msg = f"Invalid source CRS '{source_crs}'"
        raise ValueError(msg) from err

    horizontal = [
        part for part in compound.sub_crs_list if part.is_geographic or part.is_projected
    ]
    vertical = [part for part in compound.sub_crs_list if part.is_vertical]
    if not compound.is_compound or len(horizontal) != 1 or len(vertical) != 1:
        msg = (
            "source_crs must be a compound horizontal and vertical CRS, "
            "for example 'EPSG:4326+3855'"
        )
        raise ValueError(msg)

    with rasterio.open(dem) as src:
        if src.crs is None:
            msg = f"DEM has no horizontal CRS: '{dem}'"
            raise ValueError(msg)
        raster_crs = CRS.from_wkt(src.crs.to_wkt())

    if not horizontal[0].equals(raster_crs, ignore_axis_order=True):
        msg = (
            f"source_crs horizontal component '{horizontal[0].to_string()}' does not match "
            f"DEM CRS '{raster_crs.to_string()}'"
        )
        raise ValueError(msg)

    authority = compound.to_authority()
    canonical = (
        f"{authority[0]}:{authority[1]}"
        if authority is not None
        else compound.to_wkt("WKT2_2019", pretty=False)
    )
    return compound, canonical


def _new_transformer_group(source_crs: CRS) -> TransformerGroup:
    """Build a strict transformer group without leaking pyproj's missing-grid warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Best transformation is not available due to missing Grid.*",
            category=UserWarning,
        )
        return TransformerGroup(
            source_crs,
            TARGET_CRS,
            always_xy=True,
            allow_ballpark=False,
        )


def _missing_grid_names(group: TransformerGroup, *, fetchable: bool | None = None) -> list[str]:
    """Return the unique names of grids unavailable to a transformer group.

    Parameters
    ----------
    group : TransformerGroup
        Group whose unavailable operations are inspected.
    fetchable : bool, optional
        Restrict the result to grids PROJ can download itself (``True``) or to
        those it cannot (``False``). By default every missing grid is returned.
    """
    return sorted(
        {
            grid.short_name
            for operation in group.unavailable_operations
            for grid in operation.grids
            if not grid.available
            and (
                fetchable is None or bool(grid.open_license and grid.direct_download) is fetchable
            )
        },
    )


def _refresh_grid_availability() -> None:
    """Make the current PROJ context notice grids that arrived after it started.

    PROJ caches grid availability per context and pyproj keeps one context per
    thread for the lifetime of the process, so a grid downloaded moments ago
    stays invisible to this thread until the context is rebuilt. Re-pointing it
    at the data directory it already uses does that without changing which
    directory that is.
    """
    datadir.set_data_dir(datadir.get_data_dir())


def _download_grids(group: TransformerGroup) -> None:
    """Download open transformation grids and publish each file atomically."""
    names = _missing_grid_names(group, fetchable=True)
    if not names:
        msg = (
            f"No directly downloadable open-license PROJ grid is available for "
            f"{', '.join(_missing_grid_names(group))}. Install the required grid manually."
        )
        raise RuntimeError(msg)
    logger.info("Downloading PROJ transformation grid(s): %s", ", ".join(names))

    user_dir: Path | None = None
    published: list[str] = []
    try:
        user_dir = Path(datadir.get_user_data_dir(create=True))
        with TemporaryDirectory(prefix="s1bursts-proj-", dir=user_dir) as temporary:
            temporary_dir = Path(temporary)
            group.download_grids(directory=temporary_dir, open_license=True)
            for downloaded in temporary_dir.iterdir():
                if downloaded.is_file():
                    downloaded.replace(user_dir / downloaded.name)
                    published.append(downloaded.name)
    except (OSError, RuntimeError) as err:
        location = f" to '{user_dir}'" if user_dir is not None else ""
        msg = f"Could not download PROJ transformation grid(s) {', '.join(names)}{location}"
        raise RuntimeError(msg) from err

    if not published:
        msg = (
            f"PROJ reported {', '.join(names)} as downloadable but produced no file. "
            f"Install the required grid manually."
        )
        raise RuntimeError(msg)


def _unavailable_transformation_error(
    source_crs: CRS,
    group: TransformerGroup,
    *,
    download_missing_grids: bool,
) -> RuntimeError:
    """Build the error explaining why no exact transformation could be built.

    Missing grids are reported by what the caller can do about them: PROJ can
    fetch some itself, while others are only distributed by their producer.
    """
    fetchable = _missing_grid_names(group, fetchable=True)
    blocked = _missing_grid_names(group, fetchable=False)

    parts = [f"No exact PROJ transformation from '{source_crs.to_string()}' to {TARGET_CRS}."]
    if fetchable:
        parts.append(f"Missing downloadable grid(s): {', '.join(fetchable)}.")
    if blocked:
        parts.append(f"Missing grid(s) PROJ cannot download: {', '.join(blocked)}.")

    if fetchable and download_missing_grids:
        parts.append(
            "Downloading was enabled but did not make them available; "
            "install them with pyproj sync.",
        )
    elif fetchable:
        parts.append("Pass download_missing_grids=True or install them with pyproj sync.")
    elif blocked:
        parts.append("Install them manually into the PROJ user data directory.")

    return RuntimeError(" ".join(parts))


def _get_transformer(source_crs: CRS, *, download_missing_grids: bool) -> Transformer:
    """Return the best exact vertical transformer, downloading grids if requested."""
    group = _new_transformer_group(source_crs)
    if (
        not group.best_available
        and download_missing_grids
        and _missing_grid_names(group, fetchable=True)
    ):
        _download_grids(group)
        # Without this the rebuilt group reads a PROJ cache that still predates
        # the download, and the grid just installed is reported as missing.
        _refresh_grid_availability()
        group = _new_transformer_group(source_crs)

    if not group.best_available or not group.transformers:
        raise _unavailable_transformation_error(
            source_crs,
            group,
            download_missing_grids=download_missing_grids,
        )
    return group.transformers[0]


def _pixel_centres(
    transform: Affine,
    window: rasterio.windows.Window,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two arrays containing every pixel centre in a raster window."""
    columns, rows = np.meshgrid(
        np.arange(window.col_off, window.col_off + window.width, dtype="float64") + 0.5,
        np.arange(window.row_off, window.row_off + window.height, dtype="float64") + 0.5,
    )
    x = transform.a * columns + transform.b * rows + transform.c
    y = transform.d * columns + transform.e * rows + transform.f
    return x, y


def _convert_elevations(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    transformer: Transformer,
    nodata: float | None,
) -> None:
    """Convert valid elevation samples in an array in place."""
    for band in data:
        valid = np.isfinite(band)
        if nodata is not None and np.isfinite(nodata):
            valid &= band != nodata
        if not valid.any():
            continue
        _, _, transformed = transformer.transform(
            x[valid],
            y[valid],
            band[valid],
            errcheck=True,
        )
        band[valid] = transformed


def _prepare(dem: Path, dest: Path, transformer: Transformer | None) -> None:
    """Write a SNAP-prepared copy of ``dem`` to ``dest`` atomically."""
    partial = dest.with_name(f"{dest.name}.{token_hex(8)}.partial")
    try:
        with rasterio.open(dem) as src:
            source_dtype = np.dtype(src.dtypes[0])
            output_dtype = (
                np.dtype("float32")
                if transformer is not None and np.issubdtype(source_dtype, np.integer)
                else source_dtype
            )
            profile = src.profile.copy()
            profile.update(
                transform=src.transform * Affine.translation(0.5, 0.5),
                dtype=output_dtype.name,
                driver="GTiff",
                tiled=True,
                blockxsize=512,
                blockysize=512,
                compress="deflate",
                predictor=3 if np.issubdtype(output_dtype, np.floating) else 2,
                BIGTIFF="IF_SAFER",
            )
            with rasterio.open(partial, "w", **profile) as dst:
                dst.update_tags(
                    **{
                        COMPENSATED_TAG: "1",
                        VERTICAL_CONVERSION_TAG: str(int(transformer is not None)),
                    },
                )
                for _, window in src.block_windows(1):
                    data = src.read(window=window, out_dtype=output_dtype)
                    if transformer is not None:
                        x, y = _pixel_centres(src.transform, window)
                        _convert_elevations(data, x, y, transformer, src.nodata)
                    dst.write(data, window=window)
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


def prepare_dem_for_snap(
    dem: Path,
    *,
    source_crs: str | None = None,
    download_missing_grids: bool = False,
    cache_dir: Path | None = None,
) -> Path:
    """Return an external DEM prepared for SNAP's sampling and height conventions.

    The output geotransform is shifted by half a pixel because SNAP interprets an
    external DEM's origin as the centre of its first pixel. When ``source_crs`` is
    supplied, elevations are also transformed from that compound horizontal and
    vertical CRS to WGS 84 ellipsoidal heights (EPSG:4979).

    The result is correct **only** when read by SNAP and must only be passed as an
    ``externalDEMFile``. Set SNAP's ``externalDEMApplyEGM`` parameter to ``False``.
    Prepared copies are cached by source content and conversion settings.

    Parameters
    ----------
    dem : Path
        Local DEM to prepare.
    source_crs : str, optional
        Compound CRS describing the DEM's horizontal and orthometric references,
        for example ``"EPSG:4326+3855"`` for Copernicus GLO-30. If omitted,
        elevations pass through unchanged.
    download_missing_grids : bool, default False
        Download required open-license PROJ grids when they are not installed.
    cache_dir : Path, optional
        Directory for prepared copies. Defaults to ``SNAP_DEM_CACHE``
        (``~/.s1bursts/dem``).

    Returns
    -------
    Path
        Path to the prepared copy.

    Raises
    ------
    FileNotFoundError
        If ``dem`` does not exist.
    ValueError
        If the source CRS is invalid or inconsistent with the DEM.
    RuntimeError
        If no exact vertical transformation is available.
    pyproj.exceptions.ProjError
        If any DEM pixel falls outside the selected transformation's valid area.
    """
    if not dem.exists():
        msg = f"DEM not found: '{dem}'"
        raise FileNotFoundError(msg)

    compound_crs, canonical_source_crs = _normalise_source_crs(dem, source_crs)
    preparation_digest = _preparation_digest(canonical_source_crs)

    cache_dir = cache_dir or get_settings().snap.dem_cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    if dem.resolve().parent == cache_dir.resolve():
        msg = f"DEM is already inside the prepared-DEM cache: '{dem}'"
        raise ValueError(msg)

    source_digest = _file_digest(dem)
    dest = _cache_entry(dem, cache_dir, source_digest, preparation_digest)
    sidecar = dest.with_name(dest.name + PROVENANCE_SUFFIX)
    converted = compound_crs is not None

    if _cache_entry_is_valid(
        dest,
        sidecar,
        source_digest,
        preparation_digest,
        converted=converted,
    ):
        logger.debug("Reusing prepared DEM '%s'", dest)
        return dest

    transformer = (
        _get_transformer(compound_crs, download_missing_grids=download_missing_grids)
        if compound_crs is not None
        else None
    )
    logger.info("Writing SNAP-prepared DEM '%s' -> '%s'", dem, dest)
    start = time.perf_counter()
    _prepare(dem, dest, transformer)
    _write_provenance(
        sidecar,
        {
            "schema_version": CACHE_SCHEMA_VERSION,
            "compensated_tag": COMPENSATED_TAG,
            "source_path": str(dem.resolve()),
            "source_sha256": source_digest,
            "source_size": dem.stat().st_size,
            "source_crs": canonical_source_crs,
            "target_crs": TARGET_CRS if converted else None,
            "transformer_description": transformer.description
            if transformer is not None
            else None,
            "proj_version": proj_version_str,
            "preparation_sha256": preparation_digest,
            "output_name": dest.name,
            "output_size": dest.stat().st_size,
            "pixel_offset": [0.5, 0.5],
        },
    )
    logger.info("Wrote '%s' in %.1f s", dest.name, time.perf_counter() - start)
    return dest
