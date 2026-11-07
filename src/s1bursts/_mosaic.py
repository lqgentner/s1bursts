"""The mosaic of the bursts of one relative orbit and date on a cell grid.

Every burst loses the pixels along its data edge that multilooking or the
coherence window has darkened, and each remaining pixel comes from the burst
whose data edge is farthest from it, which puts the seam on the midline of
every overlap.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from s1bursts.errors import CompositeInputError, EmptyCompositeError
from s1bursts.product import CoherenceProduct

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from numpy.typing import NDArray
    from pandas import Timestamp

    from s1bursts.resolution_weighter import ComposableProduct

logger = logging.getLogger(__name__)

type OrbitDateKey = tuple[int, Timestamp]
type BurstKey = tuple[int, Timestamp, int, str]
type LoadedBand = tuple[ComposableProduct, NDArray[np.float32]]

# How far beyond the cell a burst's data edge is looked for, in metres. The
# overlap rule is exact for overlaps up to twice as wide; the widest measured
# across the Alps is 4.8 km.
EDGE_SEARCH_MARGIN_M = 3000.0

# Ground size of one slant-range sample (2.33 m) at the far edge of IW1, the
# widest of the far edges, where terrain correction moves the darkened pixels.
GROUND_RANGE_SAMPLE_M = 3.9
# Azimuth line spacing of IW SLCs.
AZIMUTH_LINE_M = 14.0


@dataclass(frozen=True)
class TrimmedBurst:
    """One burst product on the cell grid, its outermost pixels already dropped."""

    product: ComposableProduct
    data: NDArray[np.float32]
    # Distance to the product's own data edge along the pixel's row or column,
    # whichever is shorter, in pixels; 1 at the outermost pixel, and capped at
    # ``deep``, one more than the search margin.
    depth: NDArray[np.integer[Any]]
    deep: int


@dataclass(frozen=True)
class AcquisitionMosaic:
    """The bursts of one orbit and date, mosaicked at the midline of each overlap."""

    data: NDArray[np.float32]
    # The group's bursts, in tie-break order.
    sources: tuple[ComposableProduct, ...]
    # Position in ``sources`` of the burst each pixel is taken from; -1 where none.
    index: NDArray[np.intp]


def acquisition_date(product: ComposableProduct) -> Timestamp:
    """Return the acquisition date used for orbit-date grouping.

    ``CoherenceProduct`` uses ``date_primary``; ``RTCProduct`` uses ``date``.
    """
    return product.date_primary if isinstance(product, CoherenceProduct) else product.date


def orbit_date_key(product: ComposableProduct) -> OrbitDateKey:
    """Return the orbit/date group a product is mosaicked into."""
    return product.rel_orbit_number, acquisition_date(product).normalize()


def burst_key(product: ComposableProduct) -> BurstKey:
    """Return the burst and acquisition a product holds, matching coherence to RTC."""
    orbit, day = orbit_date_key(product)
    return orbit, day, product.burst_id, product.sub_swath


def auto_edge_trim_px(tags: Mapping[str, str], *, coherence: bool, grid_spacing: float) -> int:
    """Return how many pixels along a burst's data edge its settings darken.

    Multilooking divides the look block that straddles the data edge by the
    full number of looks, and terrain correction moves that block about one
    radar pixel outward, so a backscatter edge is darkened over two
    multilooked pixels. The coherence window reaches degraded samples
    outside the valid area over half its size, plus the SLC's dark first
    line. Both take the worst case, since how much of the edge block is
    valid differs from burst to burst and is not recorded.

    Raises
    ------
    KeyError
        If ``coherence`` is true and ``tags`` do not record the estimation
        window, naming the missing tag.
    """
    if coherence:
        win_rg = int(tags["COHERENCE_WINDOW_RANGE"])
        win_az = int(tags["COHERENCE_WINDOW_AZIMUTH"])
        fade_m = max(win_rg / 2 * GROUND_RANGE_SAMPLE_M, (win_az / 2 + 1) * AZIMUTH_LINE_M)
    else:
        # A product made without multilooking records no looks.
        looks_rg = int(tags.get("MULTILOOK_RANGE", "1"))
        looks_az = int(tags.get("MULTILOOK_AZIMUTH", "1"))
        fade_m = 2 * max(looks_rg * GROUND_RANGE_SAMPLE_M, looks_az * AZIMUTH_LINE_M)
    # Rounded first, so float error cannot add a pixel to an exact multiple.
    return math.ceil(round(fade_m / grid_spacing, 6))


def trim(
    product: ComposableProduct, padded: NDArray[np.float32], margin: int, trim_px: int
) -> TrimmedBurst:
    """Crop a widened read to the cell, without the ``trim_px`` pixels along its data edge.

    Holes inside the footprint are not edges and keep their neighbors.
    """
    depth = edge_depth(np.isfinite(padded), margin)
    height, width = padded.shape
    data = padded[margin : height - margin, margin : width - margin].copy()
    data[depth <= trim_px] = np.nan
    return TrimmedBurst(product, data, depth, margin + 1)


def edge_depth(finite: NDArray[np.bool_], margin: int) -> NDArray[np.integer[Any]]:
    """Distance of each cell pixel to the data edge along its row and column.

    ``finite`` covers the cell and ``margin`` pixels around it. The edges
    are the first and last finite pixel of a row or column, so holes
    inside the footprint are not edges. The depth is the shortest of the
    four distances, 1 at the outermost pixel, capped at ``margin + 1``: a
    run reaching the border of the read is at least that deep, and its
    edge lies outside the read. Meaningless at non-finite pixels.
    """
    height, width = finite.shape
    inner_rows = slice(margin, height - margin)
    inner_cols = slice(margin, width - margin)
    rows = np.arange(height, dtype=np.int32)[inner_rows, np.newaxis]
    cols = np.arange(width, dtype=np.int32)[np.newaxis, inner_cols]
    # argmax returns the first True; on the reversed axis, the last.
    row_first = finite.argmax(axis=1).astype(np.int32)[inner_rows, np.newaxis]
    row_last = (width - 1 - finite[:, ::-1].argmax(axis=1)).astype(np.int32)[
        inner_rows, np.newaxis
    ]
    col_first = finite.argmax(axis=0).astype(np.int32)[np.newaxis, inner_cols]
    col_last = (height - 1 - finite[::-1].argmax(axis=0)).astype(np.int32)[np.newaxis, inner_cols]
    deep = margin + 1
    dtype = np.int16 if deep <= np.iinfo(np.int16).max else np.int32
    depth = np.full((height - 2 * margin, width - 2 * margin), deep, dtype=dtype)
    # One side at a time, so only one full-size temporary exists.
    np.minimum(depth, cols - row_first + 1, out=depth, casting="unsafe")
    np.minimum(depth, row_last - cols + 1, out=depth, casting="unsafe")
    np.minimum(depth, rows - col_first + 1, out=depth, casting="unsafe")
    np.minimum(depth, col_last - rows + 1, out=depth, casting="unsafe")
    return depth


def mosaic_by_orbit_date(
    bursts: Sequence[TrimmedBurst],
    cell: str,
) -> dict[OrbitDateKey, AcquisitionMosaic]:
    """Group bursts by orbit/date and mosaic each group at the midline of its overlaps.

    The grouping key is ``(rel_orbit_number, acquisition_date)``, where
    ``acquisition_date`` is ``date_primary`` for coherence products and
    ``date`` for RTC products. Each pixel is taken from the burst whose
    data edge is farthest from it, which puts the seam on the midline of
    every overlap, as SNAP's TOPSAR-Deburst and TOPSAR-Merge do in radar
    geometry. Seams are never averaged. A tie goes to the lowest
    ``(burst_id, sub_swath)``, whatever order ``bursts`` is in.
    """
    groups: defaultdict[OrbitDateKey, list[TrimmedBurst]] = defaultdict(list)
    # The caller's list order is arbitrary, so it must not decide a tie.
    for burst in sorted(bursts, key=lambda b: (b.product.burst_id, b.product.sub_swath)):
        groups[orbit_date_key(burst.product)].append(burst)
    return {key: mosaic_at_midline(members, key, cell) for key, members in sorted(groups.items())}


def mosaic_at_midline(
    members: Sequence[TrimmedBurst],
    key: OrbitDateKey,
    cell: str,
) -> AcquisitionMosaic:
    """Take each pixel from the finite burst it lies deepest in."""
    sources = tuple(member.product for member in members)
    if len(members) == 1:
        data = members[0].data
        index = np.where(np.isfinite(data), 0, -1).astype(np.intp)
        return AcquisitionMosaic(data, sources, index)
    values = np.stack([member.data for member in members])  # (n, H, W)
    finite = np.isfinite(values)
    depth = np.where(finite, np.stack([member.depth for member in members]), -1)
    # argmax takes the first maximum, so a tie goes to the lowest (burst_id, sub_swath).
    index = depth.argmax(axis=0)
    mosaic = np.take_along_axis(values, index[np.newaxis], axis=0)[0]
    index[~finite.any(axis=0)] = -1

    # Beyond the margin every depth is the same cap, so the tie-break decides.
    capped = finite & (depth == members[0].deep)
    undecided = np.count_nonzero(capped.sum(axis=0) > 1)
    if undecided:
        orbit, day = key
        logger.warning(
            "cell %s, orbit %d on %s: %d pixels lie more than %g km inside two "
            "overlapping bursts, so their burst is chosen by burst ID, not the midline",
            cell,
            orbit,
            day.date(),
            undecided,
            EDGE_SEARCH_MARGIN_M / 1000,
        )
    return AcquisitionMosaic(mosaic, sources, index)


def scattering_areas(
    mosaics: Mapping[OrbitDateKey, AcquisitionMosaic],
    area_bands: Iterable[LoadedBand],
) -> dict[OrbitDateKey, NDArray[np.float32]]:
    """Take each pixel's scattering area from the RTC product of the burst it came from.

    Raises
    ------
    CompositeInputError
        If a burst some pixel is taken from has no RTC product of the same
        burst and date.
    """
    by_burst: dict[BurstKey, NDArray[np.float32]] = {}
    for prod, data in area_bands:
        by_burst.setdefault(burst_key(prod), data)
    areas: dict[OrbitDateKey, NDArray[np.float32]] = {}
    missing: list[BurstKey] = []
    for key, mosaic in mosaics.items():
        area = np.full(mosaic.data.shape, np.nan, dtype=np.float32)
        for position, source in enumerate(mosaic.sources):
            chosen = mosaic.index == position
            if not chosen.any():
                continue
            band = by_burst.get(burst_key(source))
            if band is None:
                missing.append(burst_key(source))
                continue
            area[chosen] = band[chosen]
        areas[key] = area
    if missing:
        formatted = ", ".join(
            f"{burst_id} {sub_swath} (orbit {orbit}, {day.date()})"
            for orbit, day, burst_id, sub_swath in missing
        )
        msg = f"No matching RTC scattering area for burst(s): {formatted}"
        raise CompositeInputError(msg)
    return areas


def contributing(
    mosaics: dict[OrbitDateKey, AcquisitionMosaic],
    cell: str,
) -> dict[OrbitDateKey, AcquisitionMosaic]:
    """Keep the orbit/date groups with at least one valid pixel in the cell.

    Bursts are matched to a cell by footprint, and a burst may touch a
    cell without reaching any of its pixels. Such a group adds nothing to
    the composite, so it is not counted as an observation and its date is
    not part of the acquisition span.

    Raises
    ------
    EmptyCompositeError
        If no group has a valid pixel in the cell.
    """
    kept = reaching(mosaics)
    if not kept:
        msg = f"no input observation has a valid pixel in cell {cell}"
        raise EmptyCompositeError(msg)
    return kept


def reaching(
    mosaics: dict[OrbitDateKey, AcquisitionMosaic],
) -> dict[OrbitDateKey, AcquisitionMosaic]:
    """Keep the orbit/date groups with at least one valid pixel in the cell."""
    return {key: mosaic for key, mosaic in mosaics.items() if np.isfinite(mosaic.data).any()}
