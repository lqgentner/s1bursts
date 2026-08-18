"""Raster grid handling shared by the scientific cross-validation campaigns.

Each campaign compares a SNAP-derived product against a raster produced by an
independent processor, so each needs the same two operations: a window over the
shared geographic extent that involves no resampling, and averaging over blocks
of valid pixels. The metrics themselves belong to the quantity under
comparison and live with the campaign that defines them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.windows import Window

if TYPE_CHECKING:
    from pathlib import Path


def read_aligned(
    path_a: Path,
    path_b: Path,
    band_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a band from two rasters over their shared geographic extent.

    Products of the same burst from different processors have different extents
    *and* different origins, so cropping both from ``(0, 0)`` compares pixels
    hundreds of columns apart. The window is computed in world coordinates
    instead; the origins differ by whole pixels, so it is exact and involves no
    resampling.

    Parameters
    ----------
    path_a, path_b : pathlib.Path
        Rasters to compare. They must share a CRS and resolution.
    band_name : str
        Band description to read. Falls back to the first band for writers that
        set no descriptions, such as the single-band GeoTIFFs both reference
        products ship.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, dict)
        The two bands on the shared window, and the alignment it used.

    Raises
    ------
    ValueError
        If the rasters differ in CRS or resolution, sit on incompatible grids,
        or do not overlap.
    """
    with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:
        if src_a.crs != src_b.crs:
            msg = f"CRS mismatch: {src_a.crs} vs {src_b.crs}"
            raise ValueError(msg)
        if src_a.res != src_b.res:
            msg = f"resolution mismatch: {src_a.res} vs {src_b.res}"
            raise ValueError(msg)
        res_x, res_y = src_a.res

        offset_x = (src_b.transform.c - src_a.transform.c) / res_x
        offset_y = (src_a.transform.f - src_b.transform.f) / res_y
        if abs(offset_x - round(offset_x)) > 1e-6 or abs(offset_y - round(offset_y)) > 1e-6:
            msg = (
                "rasters are not on a common grid: origins differ by "
                f"({offset_x}, {offset_y}) pixels, which is not whole"
            )
            raise ValueError(msg)

        left = max(src_a.bounds.left, src_b.bounds.left)
        right = min(src_a.bounds.right, src_b.bounds.right)
        bottom = max(src_a.bounds.bottom, src_b.bounds.bottom)
        top = min(src_a.bounds.top, src_b.bounds.top)
        if right <= left or top <= bottom:
            msg = "rasters do not overlap"
            raise ValueError(msg)

        def shared_window(src: rasterio.DatasetReader) -> Window:
            return Window(
                round((left - src.transform.c) / res_x),
                round((src.transform.f - top) / res_y),
                round((right - left) / res_x),
                round((top - bottom) / res_y),
            )

        def band_index(src: rasterio.DatasetReader) -> int:
            names = list(src.descriptions)
            return names.index(band_name) + 1 if band_name in names else 1

        array_a = src_a.read(band_index(src_a), window=shared_window(src_a)).astype("float64")
        array_b = src_b.read(band_index(src_b), window=shared_window(src_b)).astype("float64")
        alignment = {
            "x_offset_px": round(offset_x),
            "y_offset_px": round(offset_y),
            "shape": list(array_a.shape),
        }

    return array_a, array_b, alignment


def block_mean(values: np.ndarray, valid: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Average ``values`` over ``k`` x ``k`` blocks of valid pixels.

    The averaging is arithmetic in whatever units it is handed, so a caller
    comparing backscatter must pass linear power rather than decibels.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The block means, ``NaN`` where a block holds no valid pixel, and the
        number of valid pixels each block averaged.
    """
    rows = (values.shape[0] // k) * k
    cols = (values.shape[1] // k) * k
    blocks = np.where(valid, values, 0.0)[:rows, :cols].reshape(rows // k, k, cols // k, k)
    counts = valid[:rows, :cols].reshape(rows // k, k, cols // k, k)
    total = blocks.sum(axis=(1, 3))
    count = counts.sum(axis=(1, 3))
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0), count
