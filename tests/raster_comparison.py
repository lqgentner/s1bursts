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
    from collections.abc import Sequence
    from pathlib import Path


def _grid_offsets(
    reference: rasterio.DatasetReader,
    other: rasterio.DatasetReader,
) -> tuple[float, float]:
    """Origin offset of ``other`` from ``reference``, in pixels.

    Raises
    ------
    ValueError
        If the two rasters differ in CRS or resolution, or their origins differ
        by a fraction of a pixel, which would make any window a resampling.
    """
    if reference.crs != other.crs:
        msg = f"CRS mismatch: {reference.crs} vs {other.crs}"
        raise ValueError(msg)
    if reference.res != other.res:
        msg = f"resolution mismatch: {reference.res} vs {other.res}"
        raise ValueError(msg)

    res_x, res_y = reference.res
    offset_x = (other.transform.c - reference.transform.c) / res_x
    offset_y = (reference.transform.f - other.transform.f) / res_y
    if abs(offset_x - round(offset_x)) > 1e-6 or abs(offset_y - round(offset_y)) > 1e-6:
        msg = (
            "rasters are not on a common grid: origins differ by "
            f"({offset_x}, {offset_y}) pixels, which is not whole"
        )
        raise ValueError(msg)
    return offset_x, offset_y


def read_aligned(
    path_a: Path,
    path_b: Path,
    band_name: str,
    companions: Sequence[Path] = (),
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], dict]:
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
    companions : sequence of pathlib.Path, optional
        Further rasters to read -- a layover/shadow mask, say -- over the window
        ``path_a`` and ``path_b`` share rather than over one of their own. A
        companion windowed on its own extent only lines up with the compared
        arrays while its bounds match theirs exactly, and a mask applied one
        pixel out is precisely the error these comparisons exist to detect. Each
        must be on the same grid and cover the shared window.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, list of numpy.ndarray, dict)
        The two bands on the shared window, the companion bands on that same
        window, and the alignment it used.

    Raises
    ------
    ValueError
        If the rasters differ in CRS or resolution, sit on incompatible grids,
        do not overlap, or a companion does not cover the shared window.
    """
    with rasterio.open(path_a) as src_a, rasterio.open(path_b) as src_b:
        offset_x, offset_y = _grid_offsets(src_a, src_b)
        res_x, res_y = src_a.res

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

        def read_shared(src: rasterio.DatasetReader) -> np.ndarray:
            return src.read(band_index(src), window=shared_window(src)).astype("float64")

        array_a = read_shared(src_a)
        array_b = read_shared(src_b)
        alignment = {
            "x_offset_px": round(offset_x),
            "y_offset_px": round(offset_y),
            "shape": list(array_a.shape),
        }

        extras: list[np.ndarray] = []
        for path in companions:
            with rasterio.open(path) as src:
                _grid_offsets(src_a, src)
                # `read` clips a window to the dataset rather than refusing it,
                # so a companion that falls short of the shared extent would
                # come back quietly shorter than the arrays it describes.
                window = shared_window(src)
                if (
                    window.col_off < 0
                    or window.row_off < 0
                    or window.col_off + window.width > src.width
                    or window.row_off + window.height > src.height
                ):
                    msg = (
                        f"{path.name} does not cover the extent shared by "
                        f"{path_a.name} and {path_b.name}"
                    )
                    raise ValueError(msg)
                extras.append(read_shared(src))

    return array_a, array_b, extras, alignment


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
