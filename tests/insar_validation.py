"""Helpers and scene definitions for HyP3 coherence validation tests.

See ``fixtures/REFERENCE.md`` for the comparison design and requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from tests.raster_comparison import block_mean

if TYPE_CHECKING:
    from s1bursts.query import Polarization, SubSwath

# HyP3's looks option, and the posting it implies: its pixel size is
# ``max(range_looks, azimuth_looks * 5) * 4`` metres. Of the three options
# offered, the finest is the one to compare against. Its 5 x 5 phase-sigma box
# then covers exactly the 25 x 5 single-look samples of this package's default
# coherence window, so the two estimators are measured over the same ground,
# and its 20 m posting resolves the geolocation check four times as finely as
# the 20 x 4 default would. A single-burst job costs the same either way.
LOOKS = "5x1"
GRID_SPACING = 20.0

# 60 m blocks: about one resolution cell of either estimator, which is several
# pixels at this posting. Averaging to that scale leaves neighbouring samples
# roughly independent and takes estimator noise out of the correlation, without
# averaging away the land-cover pattern that carries the signal.
BLOCK = 3

# Pixels this close to the edge of either product's valid area are dropped.
# HyP3's 5 x 5 phase-sigma box and its Goldstein filter both reach across the
# boundary, and the two products' valid areas do not end in the same place.
# Five pixels is 100 m, which covers the ground either estimator draws on.
EDGE_EROSION_PX = 5

# Quantiles at which the two distributions are compared. Comparing them
# quantile by quantile, rather than conditioning on one product's value, keeps
# regression towards the mean out of the result: no pixel is selected on either
# image.
QUANTILES = (10, 25, 50, 75, 90)

# The upper quantile at which the two estimators are expected to converge:
# where coherence is high there is little phase noise for HyP3's filter to
# remove, so the estimators have little left to disagree about.
HIGH_QUANTILE = 90

# Whole-pixel shifts searched when checking the two products against each other
# on the ground. Three pixels is 60 m, about one resolution cell: far enough
# that a real offset is bracketed, near enough that the search stays over the
# same terrain.
MAX_SHIFT_PX = 3


@dataclass(frozen=True)
class Scene:
    """A burst pair under test and the HyP3 job it is compared against."""

    name: str
    burst_id: int
    sub_swath: SubSwath
    crs: str
    """UTM zone HyP3 geocodes this burst into, derived from its upper-left corner."""
    track: int
    reference_date: str
    secondary_date: str
    polarization: Polarization
    """HyP3 processes co-polarized channels only, so ``VV`` or ``HH``."""
    reference_has_pattern: bool
    """Whether the reference varies enough for ranking and alignment checks."""
    description: str

    @property
    def full_burst_id(self) -> str:
        """ASF's burst identifier, e.g. ``015_030346_IW1``."""
        return f"{self.track:03d}_{self.burst_id:06d}_{self.sub_swath}"

    @property
    def job_name(self) -> str:
        """Name the HyP3 job is submitted and looked up under.

        Carries the burst and both acquisitions, so that editing a scene asks
        for a new job instead of silently reusing the previous one.
        """
        return (
            f"s1bursts-{self.name}-"
            f"{self.reference_date.replace('-', '')}-{self.secondary_date.replace('-', '')}"
        )

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
        reference_date="2022-08-25",
        secondary_date="2022-09-06",
        polarization="VV",
        reference_has_pattern=True,
        description=(
            "Aletsch: extreme relief and land cover from bare rock to forest, "
            "spanning the full coherence range within one burst"
        ),
    ),
    Scene(
        name="mojave-iw1",
        burst_id=151219,
        sub_swath="IW1",
        crs="EPSG:32611",
        track=71,
        reference_date="2023-06-01",
        secondary_date="2023-06-13",
        polarization="VV",
        reference_has_pattern=False,
        description=(
            "Mojave Desert: arid and stable, holding high coherence over "
            "almost the whole burst where the two estimators must converge"
        ),
    ),
)


PATTERNED_SCENES = tuple(scene for scene in SCENES if scene.reference_has_pattern)
"""Scenes whose reference varies enough to compare spatial pattern against."""

SATURATED_SCENES = tuple(scene for scene in SCENES if not scene.reference_has_pattern)
"""Scenes whose reference reports the whole burst as coherent."""


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Drop pixels within ``iterations`` steps of the boundary of ``mask``."""
    for _ in range(iterations):
        shrunk = mask.copy()
        shrunk[1:, :] &= mask[:-1, :]
        shrunk[:-1, :] &= mask[1:, :]
        shrunk[:, 1:] &= mask[:, :-1]
        shrunk[:, :-1] &= mask[:, 1:]
        # The array border is the edge of the data too.
        shrunk[0, :] = shrunk[-1, :] = False
        shrunk[:, 0] = shrunk[:, -1] = False
        mask = shrunk
    return mask


def valid_mask(
    ours: np.ndarray,
    theirs: np.ndarray,
    erosion_px: int = EDGE_EROSION_PX,
) -> np.ndarray:
    """Find pixels where both products carry a usable coherence estimate.

    Parameters
    ----------
    ours, theirs : numpy.ndarray
        Co-registered coherence arrays on the same grid.
    erosion_px : int, default 5
        Pixels to drop at the boundary of the shared valid area.

    Returns
    -------
    numpy.ndarray
        Boolean mask of pixels the comparison may use.
    """
    inside = (
        np.isfinite(ours)
        & np.isfinite(theirs)
        & (ours > 0)
        & (theirs > 0)
        & (ours <= 1)
        & (theirs <= 1)
    )
    return _erode(inside, erosion_px)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank ``values`` from 1 upwards, giving tied values their average rank."""
    order = np.argsort(values, kind="stable")
    _, inverse, counts = np.unique(values[order], return_inverse=True, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ranks = np.empty(values.size, dtype="float64")
    ranks[order] = (starts + (counts + 1) / 2)[inverse]
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _paired_samples(
    ours: np.ndarray,
    theirs: np.ndarray,
    valid: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten both arrays to the pixels, or blocks of pixels, worth comparing."""
    if k <= 1:
        return ours[valid], theirs[valid]

    blocked_ours, counts = block_mean(ours, valid, k)
    blocked_theirs, _ = block_mean(theirs, valid, k)
    keep = np.isfinite(blocked_ours) & np.isfinite(blocked_theirs) & (counts >= 0.9 * k * k)
    return blocked_ours[keep], blocked_theirs[keep]


def agreement(
    ours: np.ndarray, theirs: np.ndarray, valid: np.ndarray, k: int = 1
) -> dict[str, Any]:
    """Summarize the agreement between two coherence fields.

    Parameters
    ----------
    ours, theirs : numpy.ndarray
        Co-registered coherence arrays on the same grid.
    valid : numpy.ndarray
        Boolean mask of pixels to include.
    k : int, default 1
        Block size for averaging; ``1`` compares per pixel.

    Returns
    -------
    dict
        Sample count, the median of each product, the median difference in the
        direction HyP3 minus ours, its robust scatter, and the linear and rank
        correlations.
    """
    x, y = _paired_samples(ours, theirs, valid, k)
    if x.size < 100:
        return {"n": x.size, "insufficient_samples": True}

    difference = y - x
    median_difference = float(np.median(difference))
    return {
        "n": x.size,
        "median_ours": round(float(np.median(x)), 5),
        "median_hyp3": round(float(np.median(y)), 5),
        "median_diff_hyp3_minus_ours": round(median_difference, 5),
        # Median absolute deviation, scaled to a standard deviation for a
        # normal distribution, so the scatter is not driven by the tails.
        "robust_scatter": round(
            float(1.4826 * np.median(np.abs(difference - median_difference))),
            5,
        ),
        "pearson": round(_pearson(x, y), 5),
        "spearman": round(_pearson(_average_ranks(x), _average_ranks(y)), 5),
    }


def quantile_map(
    ours: np.ndarray,
    theirs: np.ndarray,
    valid: np.ndarray,
    quantiles: tuple[int, ...] = QUANTILES,
) -> dict[str, Any]:
    """Compare the two coherence distributions by quantile.

    Parameters
    ----------
    ours, theirs : numpy.ndarray
        Co-registered coherence arrays on the same grid.
    valid : numpy.ndarray
        Boolean mask of pixels to include.
    quantiles : tuple of int, default (10, 25, 50, 75, 90)
        Percentiles to report, in the range 0 to 100.

    Returns
    -------
    dict
        Keyed ``"q10"``, ``"q25"``, ... with each product's coherence at that
        quantile and the difference, HyP3 minus ours.
    """
    x, y = ours[valid], theirs[valid]
    levels_ours = np.percentile(x, quantiles)
    levels_theirs = np.percentile(y, quantiles)
    return {
        f"q{quantile:02d}": {
            "ours": round(float(level_ours), 5),
            "hyp3": round(float(level_theirs), 5),
            "diff_hyp3_minus_ours": round(float(level_theirs - level_ours), 5),
        }
        for quantile, level_ours, level_theirs in zip(
            quantiles,
            levels_ours,
            levels_theirs,
            strict=True,
        )
    }


def reference_spread(theirs: np.ndarray, valid: np.ndarray) -> float:
    """Measure the coherence range spanned by the reference.

    Parameters
    ----------
    theirs : numpy.ndarray
        The reference coherence.
    valid : numpy.ndarray
        Boolean mask of pixels to include.

    Returns
    -------
    float
        The distance between the 10th and 90th percentiles of the reference.
    """
    low, high = np.percentile(theirs[valid], (10, 90))
    return round(float(high - low), 5)


def _parabolic_offset(minus: float, centre: float, plus: float) -> float:
    """Sub-sample position of a peak sampled at three consecutive points."""
    curvature = minus - 2 * centre + plus
    if curvature == 0:
        return 0.0
    return 0.5 * (minus - plus) / curvature


def alignment_scan(
    ours: np.ndarray,
    theirs: np.ndarray,
    valid: np.ndarray,
    max_shift: int = MAX_SHIFT_PX,
) -> dict[str, Any]:
    """Locate the two products against each other on the ground.

    Parameters
    ----------
    ours, theirs : numpy.ndarray
        Co-registered coherence arrays on the same grid.
    valid : numpy.ndarray
        Boolean mask of pixels to include.
    max_shift : int, default 3
        Largest whole-pixel shift searched, in each direction along each axis.

    Returns
    -------
    dict
        The correlation at every shift searched, the best whole-pixel shift as
        ``(row, column)``, the peak refined to sub-pixel precision by fitting a
        parabola through it, in pixels and in metres, and how far that peak
        stands above the edge of the search -- which is what says whether the
        position it reports means anything.
    """
    rows, cols = theirs.shape
    core = (slice(max_shift, rows - max_shift), slice(max_shift, cols - max_shift))
    reference, reference_valid = theirs[core], valid[core]

    surface: dict[str, float] = {}
    for row_shift in range(-max_shift, max_shift + 1):
        for col_shift in range(-max_shift, max_shift + 1):
            window = (
                slice(max_shift + row_shift, rows - max_shift + row_shift),
                slice(max_shift + col_shift, cols - max_shift + col_shift),
            )
            keep = reference_valid & valid[window]
            surface[f"{row_shift},{col_shift}"] = round(
                _pearson(ours[window][keep], reference[keep]),
                5,
            )

    best_key = max(surface, key=lambda key: surface[key])
    best_row, best_col = (int(part) for part in best_key.split(","))

    # Against the largest correlation on the boundary of the search rather than
    # the smallest anywhere in it: a peak that barely rises above the shifts
    # furthest from it is not a peak, and neither is one sitting on the edge.
    border = [
        value
        for key, value in surface.items()
        if max(abs(int(part)) for part in key.split(",")) == max_shift
    ]
    peak_contrast = surface[best_key] - max(border)
    # A maximum sitting on the edge means the true peak is outside the search:
    # the products are offset by more than `max_shift` pixels. That is the
    # opposite of "no peak to speak of", and the contrast alone cannot tell the
    # two apart, because it is not positive in either case.
    peak_on_boundary = max(abs(best_row), abs(best_col)) == max_shift

    def refine(row_shift: int, col_shift: int, axis: int) -> float:
        """Refine the peak along one axis, if the neighbouring shifts exist."""
        offsets = [(row_shift, col_shift - 1), (row_shift, col_shift + 1)]
        if axis == 0:
            offsets = [(row_shift - 1, col_shift), (row_shift + 1, col_shift)]
        keys = [f"{row},{col}" for row, col in offsets]
        if not all(key in surface for key in keys):
            return 0.0
        return _parabolic_offset(surface[keys[0]], surface[best_key], surface[keys[1]])

    subpixel = (
        best_row + refine(best_row, best_col, axis=0),
        best_col + refine(best_row, best_col, axis=1),
    )
    return {
        "max_shift_px": max_shift,
        "peak_contrast": round(peak_contrast, 5),
        "peak_on_boundary": peak_on_boundary,
        "best_shift_px": [best_row, best_col],
        "subpixel_shift_px": [round(value, 4) for value in subpixel],
        "subpixel_shift_m": [round(value * GRID_SPACING, 2) for value in subpixel],
        "correlation": surface,
    }
