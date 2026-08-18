"""Cross-validation of the SNAP-derived products against OPERA RTC-S1.

SNAP is the reference implementation for this package's processing, so its
output cannot be validated against itself. OPERA RTC-S1, produced by NASA/JPL
with ISCE3, provides an independent reference: it is generated per burst under
the same ESA burst-ID scheme, geocoded to 30 m in UTM using the Copernicus
GLO-30 DEM, and reports gamma-0 as linear power. The processing chain is the
only substantial difference between the two products.

Two layers are compared, each subject to a condition that must be met for the
result to be meaningful.

``gamma0``
    Compared after averaging in power over blocks. Per-pixel differences are
    dominated by speckle and by the difference in effective resolution between
    this package's multilooked product and OPERA's native 30 m posting; only
    the block bias supports a statement about radiometry. Averaging is
    performed in power, not in decibels.

``simulatedImage`` and ``rtc_anf_gamma0_to_beta0``
    The area normalization factor *A*. Since gamma-0 = beta-0 / *A*, the factor
    converting gamma-0 back to beta-0 is *A*, which SNAP writes as
    ``simulatedImage``. The layer depends only on geometry and the DEM, so
    comparing it isolates terrain flattening and geocoding from calibration and
    speckle. OPERA ships it only in the static granule, which is derived from
    the single acquisition named in the granule's ``InputGranules`` field and
    is not time-invariant: another acquisition of the same burst differs by
    about 2% at the median and 9% at the 90th percentile, exceeding the
    difference between processors. Process the acquisition recorded in
    ``Scene.static_date``.

See ``fixtures/REFERENCE.md`` for the full description of the reference and its
conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.windows import Window

if TYPE_CHECKING:
    from s1bursts.query import Polarization, SubSwath

# OPERA's posting. This package's standard grid places pixel edges on multiples
# of the spacing and OPERA's origins are whole multiples of 30 m, so processing
# at this spacing makes the two grids coincide and removes resampling from the
# comparison entirely.
GRID_SPACING = 30.0

# 150 m blocks: past the speckle knee (the bias is flat from 90 m outward) while
# still local enough that terrain variation is not averaged away.
BLOCK = 5

# Percentile at which the low-backscatter offset is reported. The difference
# between the two products grows towards the dark tail, and the 1st percentile
# tracks it; at the 10th it is already largely washed out.
NOISE_FLOOR_PERCENTILE = 1.0


@dataclass(frozen=True)
class Scene:
    """A burst under test and the OPERA granules it is compared against."""

    name: str
    burst_id: int
    sub_swath: SubSwath
    crs: str
    track: int
    gamma0_date: str
    """Acquisition compared for gamma0; must match an OPERA per-acquisition granule."""
    static_date: str
    """Acquisition OPERA derived the STATIC layers from (its ``InputGranules``)."""
    polarizations: tuple[Polarization, ...]
    description: str

    @property
    def burst_tag(self) -> str:
        """OPERA's burst identifier, e.g. ``T015-030346-IW1``."""
        return f"T{self.track:03d}-{self.burst_id:06d}-{self.sub_swath}"

    def search_window(self, acquisition: str) -> tuple[str, str]:
        """Return a +/-1 day window around an acquisition date, for ``SearchQuery``."""
        day = _date.fromisoformat(acquisition)
        return (str(day - timedelta(days=1)), str(day + timedelta(days=1)))


SCENES: tuple[Scene, ...] = (
    Scene(
        name="alps-iw1",
        burst_id=30346,
        sub_swath="IW1",
        crs="EPSG:32632",
        track=15,
        gamma0_date="2022-08-25",
        static_date="2016-05-04",
        polarizations=("VV", "VH"),
        description="Aletsch: extreme relief, exercising terrain flattening at its limit",
    ),
    Scene(
        name="iceland-iw3",
        burst_id=32602,
        sub_swath="IW3",
        crs="EPSG:32627",
        track=16,
        gamma0_date="2016-05-16",
        static_date="2016-05-16",
        polarizations=("VV", "VH"),
        description="Iceland ice caps: low relief and low backscatter, exercising the noise floor",
    ),
)


def read_aligned(
    path_a: Path,
    path_b: Path,
    band_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a band from two rasters over their shared geographic extent.

    Products of the same burst from different acquisitions have different
    extents *and* different origins, so cropping both from ``(0, 0)`` compares
    pixels hundreds of columns apart. The window is computed in world
    coordinates instead; the origins differ by whole pixels, so it is exact and
    involves no resampling.

    Parameters
    ----------
    path_a, path_b : pathlib.Path
        Rasters to compare. They must share a CRS and resolution.
    band_name : str
        Band description to read. Falls back to the first band for writers that
        set no descriptions, such as OPERA's single-band GeoTIFFs.

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
    """Average ``values`` in power over ``k`` x ``k`` blocks of valid pixels."""
    rows = (values.shape[0] // k) * k
    cols = (values.shape[1] // k) * k
    blocks = np.where(valid, values, 0.0)[:rows, :cols].reshape(rows // k, k, cols // k, k)
    counts = valid[:rows, :cols].reshape(rows // k, k, cols // k, k)
    total = blocks.sum(axis=(1, 3))
    count = counts.sum(axis=(1, 3))
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0), count


def agreement(ours: np.ndarray, theirs: np.ndarray, valid: np.ndarray, k: int = 1) -> dict:
    """Summarize the agreement between two rasters as bias and scatter.

    Parameters
    ----------
    ours, theirs : numpy.ndarray
        Co-registered arrays on the same grid.
    valid : numpy.ndarray
        Boolean mask of pixels to include.
    k : int, default 1
        Block size for power averaging; ``1`` compares per pixel.

    Returns
    -------
    dict
        Sample count, median ratio and its decibel equivalent, and the
        correlation of the logarithms.
    """
    if k > 1:
        blocked_ours, counts = block_mean(ours, valid, k)
        blocked_theirs, _ = block_mean(theirs, valid, k)
        keep = (
            np.isfinite(blocked_ours)
            & np.isfinite(blocked_theirs)
            & (counts >= 0.9 * k * k)
            & (blocked_ours > 0)
            & (blocked_theirs > 0)
        )
        x, y = blocked_ours[keep], blocked_theirs[keep]
    else:
        keep = valid & np.isfinite(ours) & np.isfinite(theirs) & (ours > 0) & (theirs > 0)
        x, y = ours[keep], theirs[keep]

    if x.size < 100:
        return {"n": int(x.size), "insufficient_samples": True}

    ratio = x / y
    return {
        "n": int(x.size),
        "median_ratio": round(float(np.median(ratio)), 5),
        "median_dB_bias": round(float(10 * np.log10(np.median(ratio))), 4),
        "corr_log": round(float(np.corrcoef(np.log10(x), np.log10(y))[0, 1]), 5),
    }


def percentile_offset_db(
    ours: np.ndarray,
    theirs: np.ndarray,
    valid: np.ndarray,
    percentile: float = NOISE_FLOOR_PERCENTILE,
) -> float:
    """Difference between the two products' backscatter levels at a percentile.

    Compares the two distributions at the same percentile rather than comparing
    selected pixels pairwise, so no pixels are chosen on either image and the
    result cannot be distorted by regression towards the mean. Reported in the
    same direction as the ``median_dB_bias`` of :func:`agreement`: a negative
    value means OPERA is the brighter of the two.

    Parameters
    ----------
    ours, theirs : numpy.ndarray
        Co-registered arrays on the same grid, in linear power.
    valid : numpy.ndarray
        Boolean mask of pixels to include.
    percentile : float, default 1.0
        Percentile at which to compare, in the range 0 to 100.

    Returns
    -------
    float
        The level difference in decibels.
    """
    keep = valid & np.isfinite(ours) & np.isfinite(theirs) & (ours > 0) & (theirs > 0)
    level_ours = np.percentile(ours[keep], percentile)
    level_theirs = np.percentile(theirs[keep], percentile)
    return round(float(10 * np.log10(level_ours / level_theirs)), 4)
