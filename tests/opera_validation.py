"""Helpers and scene definitions for OPERA validation tests.

See ``fixtures/REFERENCE.md`` for the comparison design and requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from s1bursts.time_range import TimeRange
from tests.raster_comparison import block_mean

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
        """OPERA's burst identifier as its granule names spell it, ``T015-030346-IW1``."""
        return f"T{self.track:03d}-{self.burst_id:06d}-{self.sub_swath}"

    @property
    def opera_burst_id(self) -> str:
        """The same identifier as the catalog indexes it, ``T015_030346_IW1``.

        CMR's ``operaBurstID`` field separates the parts with underscores where
        the granule names use hyphens. Searching with the hyphenated form
        matches nothing rather than failing, so the two spellings are kept
        apart here.
        """
        return self.burst_tag.replace("-", "_")

    def search_window(self, acquisition: str) -> TimeRange:
        """Return the day before an acquisition through the day after, for ``SearchQuery``."""
        day = _date.fromisoformat(acquisition)
        return TimeRange(day - timedelta(days=1), day + timedelta(days=1))


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


def agreement(
    ours: np.ndarray, theirs: np.ndarray, valid: np.ndarray, k: int = 1
) -> dict[str, Any]:
    """Summarize the agreement between two rasters as bias and scatter.

    Block averaging is performed in power, so ``ours`` and ``theirs`` must be
    linear power rather than decibels.

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
    """Compare backscatter levels at the same percentile.

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
        Difference in decibels; negative means OPERA is brighter.
    """
    keep = valid & np.isfinite(ours) & np.isfinite(theirs) & (ours > 0) & (theirs > 0)
    level_ours = np.percentile(ours[keep], percentile)
    level_theirs = np.percentile(theirs[keep], percentile)
    return round(float(10 * np.log10(level_ours / level_theirs)), 4)
