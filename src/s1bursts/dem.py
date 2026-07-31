"""Make an external DEM sample correctly under SNAP."""

from __future__ import annotations

import logging
from secrets import token_hex
import time
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.transform import Affine

from s1bursts.config import get_settings

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Marks a copy as deliberately offset, so it cannot be mistaken for a canonical
# DEM if it turns up outside the cache.
COMPENSATED_TAG = "SNAP_HALF_PIXEL_COMPENSATED"


def compensate_dem_for_snap(dem: Path, *, cache_dir: Path | None = None) -> Path:
    """Return a copy of ``dem`` that SNAP samples at the right position.

    Writes a copy whose geotransform origin sits on the centre of the first pixel
    rather than on its outer corner, which is where SNAP expects it.

    The result is correct **only** when read with SNAP's convention: GDAL, rasterio
    and QGIS place it half a pixel off. It carries a ``SNAP_HALF_PIXEL_COMPENSATED``
    tag and must not be used for anything but ``externalDEMFile``.

    The copy is cached and reused until ``dem`` changes, so a pipeline that
    processes many bursts against one DEM pays for the conversion once. Writes are
    atomic, so concurrent workers may call this on the same DEM.


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
        If ``dem`` already resolves into ``cache_dir``.
    """
    if not dem.exists():
        msg = f"DEM not found: '{dem}'"
        raise FileNotFoundError(msg)

    cache_dir = cache_dir or get_settings().snap.dem_cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / dem.name

    if dest.resolve() == dem.resolve():
        msg = f"DEM is already inside the compensated-DEM cache: '{dem}'"
        raise ValueError(msg)

    if dest.exists() and dest.stat().st_mtime >= dem.stat().st_mtime:
        logger.debug("Reusing compensated DEM '%s'", dest)
        return dest

    logger.info("Writing SNAP-compensated DEM '%s' -> '%s'", dem, dest)
    start = time.perf_counter()
    # A unique suffix keeps concurrent writers from clobbering one another's
    # partial file; the rename that publishes it is atomic.
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
    logger.info("Wrote '%s' in %.1f s", dest.name, time.perf_counter() - start)
    return dest
