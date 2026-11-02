"""Compare an existing product's recorded settings with a request before skipping it.

File names carry no processing settings, so a product made with other settings
has the same name as the requested one. Before a ``create_*`` or ``compose_*``
call returns an existing product unchanged, it reads the settings the file
records in its tags and raises ``ProductSettingsMismatchError`` where they
differ from the request. A file written before the tags existed, or a product
whose DEM is recorded as unknown, cannot be checked and is skipped as before;
the skipped result says so.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING
import warnings

from freezebase.raster import rasterio_open
from rasterio.errors import RasterioError

from s1bursts.errors import ProductSettingsMismatchError
from s1bursts.product_tags import (
    UNKNOWN,
    _burst_settings,
    _composite_settings,
    _software_version,
)
from s1bursts.summary import OkResult

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from affine import Affine
    from freezebase.mgrs import MGRSGeoBox
    from rasterio.crs import CRS
    from upath import UPath

    from s1bursts.filemanager import FileManager
    from s1bursts.product import (
        CoherenceProduct,
        GTCProduct,
        MGRSTileProduct,
        RTCProduct,
    )
    from s1bursts.product_tags import Aggregation
    from s1bursts.snap_graph import BackscatterConfig, CoherenceConfig
    from s1bursts.summary import UncheckedReason

logger = logging.getLogger(__name__)

type _Differences = dict[str, tuple[str | None, str | None]]

# Every tagged file has it: the tags arrived together in s1bursts 0.3.0.
_TAGGED_MARKER = "PIPELINE_SOFTWARE_VERSION"
_SNAP_VERSION = "PROCESSING_SOFTWARE_VERSION"
_DECIBEL = "decibel"

# The argument behind each compared tag, for the message.
_ARGUMENTS = {
    "GRID_SPACING": "grid_spacing",
    "DEM": "dem",
    "MULTILOOK_RANGE": "multilook",
    "MULTILOOK_AZIMUTH": "multilook",
    "COHERENCE_WINDOW_RANGE": "window",
    "COHERENCE_WINDOW_AZIMUTH": "window",
    "COMPOSITING_PARAMETERS": "aggregation, clip_max_resolution",
    "CRS": "geobox",
    "TRANSFORM": "geobox",
    "SHAPE": "geobox",
    "BACKSCATTER_SCALING": "decibel, now removed",
    "CQM_SCALING": "cqm_decibel, now removed",
}


@dataclass(frozen=True)
class _BurstSkip:
    """What a skipped per-burst product's check could not confirm."""

    unchecked: UncheckedReason | None = None
    other_snap_version: str | None = None


@dataclass(frozen=True)
class _Header:
    """The tags and grid of an existing file."""

    tags: dict[str, str]
    crs: CRS | None
    transform: Affine
    shape: tuple[int, int]


def _read_header(path: Path | UPath) -> _Header:
    """Return a file's tags and grid, noting on a read error which file it was."""
    try:
        with rasterio_open(path) as src:
            return _Header(dict(src.tags()), src.crs, src.transform, (src.height, src.width))
    except (OSError, RasterioError) as err:
        err.add_note(
            f"Reading '{path}', an existing product, to check its settings failed. "
            "Pass overwrite=True to rebuild it."
        )
        raise


def _shown(value: str | None) -> str:
    """Render a compared value for the message; ``None`` is an absent tag."""
    return "none" if value is None else repr(value)


def _mismatch(path: Path | UPath, differences: _Differences) -> ProductSettingsMismatchError:
    """Build the error naming each differing setting with its recorded and requested value."""
    listed = "; ".join(
        f"{key} ({_ARGUMENTS.get(key, key)}): recorded {_shown(recorded)}, "
        f"requested {_shown(requested)}"
        for key, (recorded, requested) in differences.items()
    )
    msg = (
        f"'{path.name}' exists but was made with other settings: {listed}. Pass "
        "overwrite=True to replace it, or give these settings their own FileManager "
        "directory to keep both."
    )
    return ProductSettingsMismatchError(msg, path=path, differences=differences)


def _check_burst_product(
    path: Path | UPath,
    product: RTCProduct | GTCProduct | CoherenceProduct,
    config: BackscatterConfig | CoherenceConfig,
    *,
    snap_version: tuple[int, ...] | None,
) -> _BurstSkip:
    """Check an existing per-burst product, returning what its skip could not confirm.

    Raises ``ProductSettingsMismatchError`` if the file records other settings
    than ``config`` would write, and the read error if it cannot be read.
    """
    recorded = _read_header(path).tags
    if _TAGGED_MARKER not in recorded:
        return _BurstSkip(unchecked="untagged")
    requested = _burst_settings(product, config)
    differences: _Differences = {
        key: (recorded.get(key), value)
        for key, value in requested.items()
        if recorded.get(key) != value
    }
    # An external DEM without dem_name is recorded as unknown, so it cannot be compared.
    dem_unknown = UNKNOWN in (recorded.get("DEM"), requested["DEM"])
    if dem_unknown:
        differences.pop("DEM", None)
    if differences:
        raise _mismatch(path, differences)

    running = _software_version(snap_version)
    made_with = recorded.get(_SNAP_VERSION, UNKNOWN)
    other = made_with if UNKNOWN not in (running, made_with) and made_with != running else None
    return _BurstSkip(unchecked="unknown_dem" if dem_unknown else None, other_snap_version=other)


def _skip_existing[P: RTCProduct | GTCProduct | CoherenceProduct](
    file_manager: FileManager,
    source: object,
    product: P,
    config: BackscatterConfig | CoherenceConfig,
    *,
    snap_version: tuple[int, ...] | None,
    log_level: int = logging.WARNING,
) -> OkResult[P] | None:
    """Return the skipped result of an existing per-burst product, or None if there is none.

    Logs a skip whose settings could not be checked, or that another SNAP
    version made, at ``log_level``; a batch that warns once about them all
    passes DEBUG. Raises as ``_check_burst_product`` does.
    """
    if not file_manager.product_exists(product):
        return None
    path = file_manager.get_product_path(product)
    skip = _check_burst_product(path, product, config, snap_version=snap_version)
    if skip.unchecked is not None:
        logger.log(
            log_level,
            "Skipped '%s' without checking its settings (%s)",
            path.name,
            skip.unchecked,
        )
    if skip.other_snap_version is not None:
        logger.log(
            log_level,
            "Skipped '%s', made by SNAP %s rather than the running %s",
            path.name,
            skip.other_snap_version,
            _software_version(snap_version),
        )
    logger.debug("%s product already exists: '%s'", product.product_type, path.name)
    return OkResult(
        source=source,
        result=product,
        skipped=True,
        unchecked=skip.unchecked,
        other_snap_version=skip.other_snap_version,
    )


def _grid_differences(header: _Header, geobox: MGRSGeoBox) -> _Differences:
    """Return how a file's grid differs from the target cell's."""
    differences: _Differences = {}
    if header.crs != geobox.crs:
        differences["CRS"] = (str(header.crs), str(geobox.crs))
    if not header.transform.almost_equals(geobox.transform):
        differences["TRANSFORM"] = (
            str(tuple(header.transform)[:6]),
            str(tuple(geobox.transform)[:6]),
        )
    shape = (geobox.height, geobox.width)
    if header.shape != shape:
        differences["SHAPE"] = ("x".join(map(str, header.shape)), "x".join(map(str, shape)))
    return differences


def _tag_differences(recorded: Mapping[str, str], expected: Mapping[str, str]) -> _Differences:
    """Return the expected tags a composite layer records otherwise, and any decibel scaling."""
    differences: _Differences = {
        key: (recorded.get(key), value)
        for key, value in expected.items()
        if recorded.get(key) != value
    }
    # Composites are written in linear units only.
    for key in ("BACKSCATTER_SCALING", "CQM_SCALING"):
        if recorded.get(key) == _DECIBEL:
            differences[key] = (_DECIBEL, None)
    return differences


def _check_composite(
    file_manager: FileManager,
    product: MGRSTileProduct,
    geobox: MGRSGeoBox,
    *,
    aggregation: Aggregation,
    clip_max_resolution: float | None = None,
) -> bool:
    """Check every requested layer of an existing composite; False if one could not be checked.

    Only the arguments of the ``compose_*`` call are compared, not what the
    composite inherits from its inputs, so no input needs to be read. Raises
    ``ProductSettingsMismatchError`` if a layer records another grid, other
    compositing parameters or decibel values, and the read error if a layer
    cannot be read.
    """
    expected = _composite_settings(
        product, aggregation=aggregation, clip_max_resolution=clip_max_resolution
    )
    checked = True
    for layer in sorted(product.layers):
        path = file_manager.get_product_path(product, layer)
        header = _read_header(path)
        if _TAGGED_MARKER not in header.tags:
            checked = False
            continue
        differences = _grid_differences(header, geobox) | _tag_differences(header.tags, expected)
        if differences:
            raise _mismatch(path, differences)
    return checked


def _warn_unchecked(count: int, *, stacklevel: int, per_burst: bool = True) -> None:
    """Warn that ``count`` existing products were skipped without checking their settings.

    Only a per-burst product can go unchecked for an external DEM without
    ``dem_name``; a composite only for being written before s1bursts 0.3.0.
    """
    if not count:
        return
    if per_burst:
        reason = (
            "they were written before s1bursts 0.3.0, or made with an external DEM that "
            "has no dem_name. Pass overwrite=True to rebuild them, and set dem_name for "
            "an external DEM."
        )
    else:
        reason = "they were written before s1bursts 0.3.0. Pass overwrite=True to rebuild them."
    msg = f"Skipped {count} existing product(s) without checking their settings: {reason}"

    warnings.warn(msg, stacklevel=stacklevel + 1)


def _warn_other_snap(count: int, running: str, *, stacklevel: int) -> None:
    """Warn that ``count`` existing products were made by another SNAP version."""
    if not count:
        return
    msg = (
        f"Skipped {count} existing product(s) made by another SNAP version than the "
        f"running {running}. SNAP 13 and 14 outputs differ, so do not mix them in one "
        "time series; pass overwrite=True to rebuild them."
    )
    warnings.warn(msg, stacklevel=stacklevel + 1)
