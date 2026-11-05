"""Compare an existing product's recorded settings with a request before skipping it.

File names carry no processing settings, so a product made with other settings
has the same name as the requested one. Before a ``create_*`` or ``compose_*``
call returns an existing product unchanged, it reads the settings the file
records in its tags and raises ``ProductSettingsMismatchError`` where they
differ from the request or are not recorded at all.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

from freezebase.raster import rasterio_open
from rasterio.errors import RasterioError

from s1bursts.errors import ProductSettingsMismatchError
from s1bursts.product import Layer
from s1bursts.product_tags import (
    _SETTING_TAGS,
    _SNAP_OPTION_KEYS,
    _SNAP_VERSION_KEY,
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

logger = logging.getLogger(__name__)

type _Differences = dict[str, tuple[str | None, str | None]]

_DECIBEL = "decibel"

# The argument behind each compared tag, for the message.
_ARGUMENTS = {
    **{key: name for name, keys in _SETTING_TAGS.items() for key in keys},
    "DEM": "dem, dem_name",
    _SNAP_VERSION_KEY: "the SNAP installation",
    "COMPOSITING_PARAMETERS": (
        "aggregation, clip_max_resolution, or the burst mosaicking of another s1bursts version"
    ),
    "CRS": "geobox",
    "TRANSFORM": "geobox",
    "SHAPE": "geobox",
    "BACKSCATTER_SCALING": "decibel, now removed",
    "CQM_SCALING": "cqm_decibel, now removed",
}


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
    msg = f"'{path.name}' exists but records other settings than requested: {listed}."
    if _SNAP_VERSION_KEY in differences:
        msg += " SNAP 13 and 14 outputs differ, so do not mix them in one time series."
    msg += (
        " Pass overwrite=True to replace it, or give these settings their own FileManager "
        "directory to keep both."
    )
    return ProductSettingsMismatchError(msg, path=path, differences=differences)


def _check_burst_product(
    path: Path | UPath,
    product: RTCProduct | GTCProduct | CoherenceProduct,
    config: BackscatterConfig | CoherenceConfig,
    *,
    snap_version: tuple[int, ...] | None,
) -> None:
    """Check an existing per-burst product against ``config`` and the running SNAP version.

    An undetermined SNAP version, recorded or running, matches any other, and
    so do the coherence tags that depend on it. Raises
    ``ProductSettingsMismatchError`` if the file records other settings than
    ``config`` would write, or does not record one of them, and the read
    error if it cannot be read.
    """
    recorded = _read_header(path).tags
    requested = _burst_settings(product, config, snap_version=snap_version)
    requested[_SNAP_VERSION_KEY] = _software_version(snap_version)
    if UNKNOWN in (requested[_SNAP_VERSION_KEY], recorded.get(_SNAP_VERSION_KEY)):
        for key in (_SNAP_VERSION_KEY, *_SNAP_OPTION_KEYS):
            requested.pop(key, None)
    differences: _Differences = {
        key: (recorded.get(key), value)
        for key, value in requested.items()
        if recorded.get(key) != value
    }
    if differences:
        raise _mismatch(path, differences)


def _skip_existing[P: RTCProduct | GTCProduct | CoherenceProduct](
    file_manager: FileManager,
    source: object,
    product: P,
    config: BackscatterConfig | CoherenceConfig,
    *,
    snap_version: tuple[int, ...] | None,
) -> OkResult[P] | None:
    """Return the skipped result of an existing per-burst product, or None if there is none.

    Raises as ``_check_burst_product`` does.
    """
    if not file_manager.product_exists(product):
        return None
    path = file_manager.get_product_path(product)
    _check_burst_product(path, product, config, snap_version=snap_version)
    logger.debug("%s product already exists: '%s'", product.product_type, path.name)
    return OkResult(source=source, result=product, skipped=True)


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
) -> dict[str, str]:
    """Check every requested layer of an existing composite, returning its measurement's tags.

    Only the arguments of the ``compose_*`` call are compared, not what the
    composite inherits from its inputs, so no input needs to be read. Raises
    ``ProductSettingsMismatchError`` if a layer records another grid, other
    or no compositing parameters, or decibel values, and the read error if a
    layer cannot be read.
    """
    expected = _composite_settings(
        product, aggregation=aggregation, clip_max_resolution=clip_max_resolution
    )
    measurement: dict[str, str] = {}
    for layer in sorted(product.layers):
        path = file_manager.get_product_path(product, layer)
        header = _read_header(path)
        differences = _grid_differences(header, geobox) | _tag_differences(header.tags, expected)
        if differences:
            raise _mismatch(path, differences)
        if layer == Layer.MEASUREMENT:
            measurement = header.tags
    return measurement
