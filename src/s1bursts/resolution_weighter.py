"""Locally Resolution Weighted (LRW) compositing for Sentinel-1 products.

Implements the LRW backscatter composite and its composite quality map (CQM)
of [Small et al. (2022)](https://doi.org/10.1109/TGRS.2021.3055562), which the CEOS-ARD
Product Family Specification for Synthetic Aperture Radar adopts as its
Composite Backscatter (CB) product.

The weights come from the local scattering area ``A``, the area terrain
flattening divides beta nought by, so that gamma nought is ``beta0 / A``. That
is the CEOS-ARD term; the same quantity is the local contributing area of Small
et al., the area normalization factor (ANF) of OPERA, and the
``simulatedImage`` band SNAP's Terrain-Flattening writes. This package stores
it as the ``scattering_area`` band of every RTC product. ``A = 1`` is a facet
at 45° local incidence; flat terrain has ``A = cot(θ)`` at incidence angle
``θ``, about 1.0 to 1.7 across the IW swath.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import math
from typing import TYPE_CHECKING, Any, Literal

from freezebase.raster import build_rasterio_profile, rasterio_open, write_cog
import numpy as np
from pandas import Timestamp
from rasterio.windows import from_bounds

from s1bursts import _mosaic, product_tags
from s1bursts.errors import (
    CompositeInputError,
    EmptyCompositeError,
    InconsistentInputsError,
)
from s1bursts.product import (
    CoherenceProduct,
    CohLRWComposite,
    CohPassComposite,
    GTCPassComposite,
    GTCProduct,
    Layer,
    RTCLRWComposite,
    RTCPassComposite,
    RTCProduct,
    _layer_product_type,
)
from s1bursts.product_tags import (
    _BURST_SETTING_KEYS,
    _SNAP_OPTION_KEYS,
    _SNAP_VERSION_KEY,
    UNKNOWN,
)
from s1bursts.query import _PASS_DIRECTIONS
from s1bursts.settings_check import _check_composite
from s1bursts.time_range import TimeRange, calendar_day

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterable, Sequence
    from pathlib import Path

    from freezebase.mgrs import MGRSGeoBox
    from numpy.typing import NDArray
    from upath import UPath

    from s1bursts.filemanager import FileManager
    from s1bursts.product_tags import RasterMetadata
    from s1bursts.query import PassDirection, Polarization

logger = logging.getLogger(__name__)


type ComposableProduct = CoherenceProduct | RTCProduct | GTCProduct
type CompositeProduct = (
    CohLRWComposite | RTCLRWComposite | RTCPassComposite | CohPassComposite | GTCPassComposite
)

# The layers a composite holds without ``write_qa``; with it, all it supports.
_MEASUREMENT_ONLY = frozenset({Layer.MEASUREMENT})


def _calendar_day(value: Timestamp) -> Timestamp:
    """``value`` as a naive midnight timestamp of its UTC calendar day."""
    return Timestamp(calendar_day(value))


@dataclass(frozen=True)
class _PassObservations:
    """What the metadata of one direction's pass composite is built from."""

    composite: NDArray[np.floating[Any]]
    n_observations: int
    source_tags: dict[str, str]
    acquisitions: product_tags.Acquisitions
    burst_edge_trim_px: int


_DEFAULT_PROFILE_KEY = "default"

# Largest sub-pixel offset, in pixels, still read as grid-aligned.
_ALIGNMENT_TOLERANCE = 1e-3


def _split_profiles(
    rasterio_profile: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Normalize a flat or per-product rasterio profile mapping."""
    if not rasterio_profile:
        return {}
    if all(isinstance(value, Mapping) for value in rasterio_profile.values()):
        return {key: dict(value) for key, value in rasterio_profile.items()}
    return {_DEFAULT_PROFILE_KEY: dict(rasterio_profile)}


def _snap_version_of(tags: Mapping[str, str]) -> str | None:
    """Return the SNAP version an input records, None if it records none."""
    return tags.get(_SNAP_VERSION_KEY)


def _shown(value: str | None) -> str:
    """Render a tag value for a message; ``None`` is an absent tag."""
    return "none" if value is None else repr(value)


def _agreed_snap_version(inputs: Sequence[Mapping[str, str]]) -> str | None:
    """Return the SNAP version the inputs share, raising if two recorded versions differ.

    ``UNKNOWN`` matches any version, and is returned where it is all that
    differs; an absent tag is a difference like another version.
    """
    versions = {_snap_version_of(tags) for tags in inputs}
    known = versions - {UNKNOWN}
    if len(known) > 1:
        listed = ", ".join(sorted("none" if version is None else version for version in known))
        msg = (
            f"All inputs must be made by the same SNAP version; got {listed}. "
            "SNAP 13 and 14 outputs differ, so do not mix them in one composite."
        )
        raise InconsistentInputsError(msg)
    return versions.pop() if len(versions) == 1 else UNKNOWN


def _agreed_settings(inputs: Sequence[Mapping[str, str]], *, what: str) -> dict[str, str]:
    """Return the setting tags the inputs record, raising if they differ.

    The DEM is compared separately, across every input of a composite. An
    input of an undetermined SNAP version matches any value of the tags that
    depend on the version; where only such an input differs, the tag is
    ``UNKNOWN``.
    """
    agreed: dict[str, str] = {}
    for key in _BURST_SETTING_KEYS:
        if key == "DEM":
            continue
        compared = inputs
        if key in _SNAP_OPTION_KEYS:
            compared = [tags for tags in inputs if _snap_version_of(tags) != UNKNOWN]
        values = {tags.get(key) for tags in compared}
        if len(values) > 1:
            listed = ", ".join(sorted(_shown(value) for value in values))
            msg = f"{what} must be made with the same settings; they differ on {key}: {listed}."
            raise InconsistentInputsError(msg)
        values = {tags.get(key) for tags in inputs}
        value = values.pop() if len(values) == 1 else UNKNOWN
        if value is not None:
            agreed[key] = value
    return agreed


def _agreed_tags(
    inputs: Sequence[Mapping[str, str]],
    rtc_weights: Sequence[Mapping[str, str]] = (),
) -> dict[str, str]:
    """Return the tags of the per-burst inputs, which must agree on how they were made.

    The SNAP version and the processing settings exist nowhere at
    compositing time except in the input files. The inputs must record the
    same settings, the same SNAP version and the same DEM, or
    ``InconsistentInputsError`` is raised; an input of an undetermined SNAP
    version matches any version, and where it is all that differs, the
    composite records ``UNKNOWN``. Any other tag is carried forward only
    where every input states the same value.

    ``rtc_weights`` are the tags of the RTC products that supply a coherence
    LRW composite's scattering area. They are compared among themselves on
    their settings, being made with another configuration than the coherence
    inputs, and with the coherence inputs on the SNAP version and the DEM.
    """
    if not inputs:
        return {}
    every_input = [*inputs, *rtc_weights]
    dem_values = {tags.get("DEM", UNKNOWN) for tags in every_input}
    if len(dem_values) > 1:
        listed = ", ".join(sorted(dem_values))
        msg = f"All inputs must use the same DEM; got {listed}."
        raise InconsistentInputsError(msg)
    snap_version = _agreed_snap_version(every_input)
    settings = _agreed_settings(inputs, what="All inputs")
    _agreed_settings(rtc_weights, what="The RTC products supplying the scattering area")

    first, *rest = inputs
    agreed = {k: v for k, v in first.items() if all(other.get(k) == v for other in rest)}
    agreed |= settings
    agreed["DEM"] = dem_values.pop()
    if snap_version is not None:
        agreed[_SNAP_VERSION_KEY] = snap_version
    return agreed


def _check_existing_composite(
    name: str,
    recorded: Mapping[str, str],
    inputs: Sequence[Mapping[str, str]],
) -> None:
    """Raise if an existing composite records other settings or another SNAP version.

    ``inputs`` are the tags of the per-burst inputs, which already agree. The
    rules are those of ``_agreed_tags``: an undetermined SNAP version, on
    either side, matches any version and any value of the tags that depend
    on it, and an absent tag is a difference.
    """
    composite_version = _snap_version_of(recorded)
    known = [tags for tags in inputs if _snap_version_of(tags) != UNKNOWN]
    for key in ("DEM", *(k for k in _BURST_SETTING_KEYS if k != "DEM"), _SNAP_VERSION_KEY):
        compared = inputs
        if key == _SNAP_VERSION_KEY or key in _SNAP_OPTION_KEYS:
            compared = known if composite_version != UNKNOWN else []
        default = UNKNOWN if key == "DEM" else None
        values = {tags.get(key, default) for tags in compared}
        value = recorded.get(key, default)
        if values and value not in values:
            msg = (
                f"The existing composite '{name}' records {key} {_shown(value)}, but the "
                f"inputs record {_shown(values.pop())}. Pass overwrite=True with inputs "
                "that agree to rebuild it."
            )
            raise InconsistentInputsError(msg)


class ResolutionWeighter:
    """Compute LRW and per-pass composites for Sentinel-1 products.

    Each compose call receives its target ``geobox``, so one instance can be
    reused across MGRS grid cells.

    Parameters
    ----------
    file_manager
        Used to resolve input file paths and output product paths.
    max_workers
        Number of threads for parallel file I/O during product loading.
    rasterio_profile
        GeoTIFF profile applied to every product, or profiles keyed by product
        type with an optional ``"default"`` entry. Product-specific settings
        override the default. If omitted, uses the library defaults with
        float32 data and NaN no-data.
    checksum
        Compute a SHA-256 multihash of every written file's exact bytes and
        record it in ``checksums``, keyed by output path. Costs holding
        the encoded COG in memory a second time per write.

    Attributes
    ----------
    checksums : dict[str, str]
        SHA-256 multihash of every file written so far, keyed by output path
        as a string. Populated only when ``checksum=True``.

    References
    ----------
    .. [1] Small, D., Rohner, C., Miranda, N., Ruetschi, M., & Schaepman, M. E.
       (2022). Wide-Area Analysis-Ready Radar Backscatter Composites. IEEE
       Transactions on Geoscience and Remote Sensing, 60, 5201814.
       https://doi.org/10.1109/TGRS.2021.3055562
    .. [2] CEOS-ARD Product Family Specification: Synthetic Aperture Radar,
       version 1.3.1 (2026). Committee on Earth Observation Satellites.
       https://ceos.org/ard/files/PFS/SAR/v1.3.1/CEOS-ARD_PFS_SAR_v1.3.1.pdf
    """

    def __init__(
        self,
        file_manager: FileManager,
        *,
        rasterio_profile: dict[str, Any] | None = None,
        max_workers: int = 8,
        checksum: bool = False,
    ) -> None:
        self._fm = file_manager
        self._max_workers = max_workers
        self._profiles = _split_profiles(rasterio_profile)
        self._checksum = checksum
        self.checksums: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose_rtc_lrw(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "mean",
        clip_max_resolution: float | None = None,
        write_qa: bool = False,
        overwrite: bool = False,
        time_range: TimeRange | None = None,
        burst_edge_trim: bool = True,
    ) -> RTCLRWComposite:
        r"""Compute a locally resolution weighted (LRW) RTC intensity composite.

        Band 1 supplies intensity and band 2 the local scattering area $A$.
        Observations are weighted by normalized $1/A$ ([1] eq. 3-4, [2]
        eq. A5.1-A5.3) after mosaicking each orbit and acquisition date,
        without the darkened edge of each burst and splitting each overlap at
        its midline.

        Parameters
        ----------
        rtc_products
            RTC products to composite.  All products must share the same
            polarization.
        geobox
            Target MGRS grid cell.
        aggregation
            Temporal aggregation applied after weighting.  Pass ``"mean"`` or
            ``"median"`` for the corresponding statistic, or a float in
            ``(0, 1)`` to compute a weighted quantile at that level.
        clip_max_resolution
            Upper limit for $W = 1/A$, used to suppress extreme weights in
            radar shadow. ``1.0`` is the weight of a facet at 45° local
            incidence; flat terrain weighs ``tan(θ)``, about 0.6 to 1.0.
        write_qa
            If ``True``, also write the two QA layers: a
            contributing-observations map (``Layer.QA_NUM``) and a composite
            quality map (``Layer.QA_CQM``).
            ``QA-CQM`` is $1 / \sum_i W_i A_i$ under every ``aggregation``:
            under ``median`` or a quantile it describes the weighting, not the
            observations the composite's value was taken from.
        overwrite
            If ``False`` and every requested layer's file already exists, skip
            computation and return the existing product, after checking that
            each file records the requested grid and compositing parameters
            and linear units; any other difference, such as the DEM of the
            inputs, is not checked. Otherwise every
            requested layer is written, replacing an existing file, so the
            measurement and its QA layers always come from the same run.
            If no input reaches the cell, an earlier composite of the same
            identity is deleted, all its layers, before the error is raised;
            with ``overwrite=False`` it is left alone.
        time_range
            The compositing window, as calendar days; pass the same
            :class:`~s1bursts.time_range.TimeRange` the search was made with.
            When given, the product is named and tagged with its days rather
            than with the first and last acquisition, so every tile composited
            from one search shares them; the observed span is kept in the
            ``FIRST_ACQUISITION_DATE`` and ``LAST_ACQUISITION_DATE`` tags. An
            input outside the window is a ``ValueError``.
        burst_edge_trim
            Drop the edge of each burst before mosaicking, where multilooking
            has darkened it. Its width follows the inputs' looks and the grid
            spacing: 4 pixels at the default looks on a 10 m grid. ``False``
            keeps every pixel.

        Returns
        -------
        RTCLRWComposite
            The composite, whose ``layers`` are the measurement and, with
            ``write_qa=True``, the QA layers. Pass a layer to
            ``FileManager.get_product_path`` for its file.

        Raises
        ------
        TypeError
            If ``time_range`` is not a ``TimeRange``.
        ValueError
            If ``rtc_products`` is empty, ``aggregation`` is neither ``"mean"``,
            ``"median"`` nor a quantile level in ``(0, 1)``, or an input lies
            outside ``time_range``.
        EmptyCompositeError
            If no input has a valid pixel in ``geobox``; nothing is written.
        InconsistentInputsError
            If the inputs differ in polarization, an input is on another
            CRS, pixel size or grid than ``geobox``, or the inputs were made
            with different DEMs, processing settings or SNAP versions, where
            an undetermined SNAP version matches any. Not a
            ``CompositeError``: other cells given these inputs would meet it
            too.
        CompositeInputError
            If an input product is missing or cannot be read.
        ProductSettingsMismatchError
            If ``overwrite`` is ``False`` and an existing layer file records
            another grid than ``geobox``, other or no compositing parameters,
            or values in decibels. Not a ``CompositeError``: other cells would
            meet it too.
        OSError or RasterioError
            If ``overwrite`` is ``False`` and an existing layer file cannot be
            read to check its settings.
        """
        rtc_products = list(rtc_products)
        if not rtc_products:
            msg = "rtc_products must not be empty."
            raise ValueError(msg)
        self._check_aggregation(aggregation)

        polarization = self._check_polarization(rtc_products)
        acquisitions = self._acquisition_span(rtc_products)
        start_date, end_date = self._composite_dates(acquisitions, time_range)

        product = RTCLRWComposite(
            mgrs_code=geobox.mgrs_code,
            start_date=start_date,
            end_date=end_date,
            polarization=polarization,
            layers=RTCLRWComposite.supported_layers if write_qa else _MEASUREMENT_ONLY,
        )
        if not overwrite and self._all_exist(product):
            _check_composite(
                self._fm,
                product,
                geobox,
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                burst_edge_trim=burst_edge_trim,
            )
            return product

        source_tags = self._input_tags(rtc_products)
        burst_edge_trim_px = self._edge_trim_px(
            source_tags, geobox, coherence=False, enabled=burst_edge_trim
        )
        try:
            rtc_groups = _mosaic.contributing(
                _mosaic.mosaic_by_orbit_date(
                    self._load_bursts(rtc_products, geobox, burst_edge_trim_px), geobox.mgrs_code
                ),
                geobox.mgrs_code,
            )
        except EmptyCompositeError:
            if overwrite:
                self._remove(product)
            raise
        weighting = [prod for prod in rtc_products if _mosaic.orbit_date_key(prod) in rtc_groups]
        area_groups = _mosaic.scattering_areas(
            rtc_groups, self._load_band(weighting, geobox, band=2)
        )
        # The tags name the acquisitions that reach the cell, not every input.
        observed = self._contributing_span(rtc_products, rtc_groups)

        rtc_obs = np.stack([mosaic.data for mosaic in rtc_groups.values()])  # (N, H, W)
        area_obs = np.stack(list(area_groups.values()))  # (N, H, W)

        weights = self._calc_weights(area_obs, clip_max_resolution)
        composite = self._aggregate_weighted(rtc_obs, weights, aggregation)
        n_observations = len(rtc_obs)
        self._save(
            composite,
            product,
            Layer.MEASUREMENT,
            geobox,
            product_tags.rtc_lrw_metadata(
                product,
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                burst_edge_trim_px=burst_edge_trim_px,
                n_observations=n_observations,
                source_tags=source_tags,
                acquisitions=observed,
            ),
        )
        if write_qa:
            self._write_lrw_qa(
                rtc_obs,
                area_obs,
                weights,
                product,
                geobox,
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                burst_edge_trim_px=burst_edge_trim_px,
                source_tags=source_tags,
                acquisitions=observed,
            )
        return product

    def compose_coherence_lrw(
        self,
        coh_products: Collection[CoherenceProduct],
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "mean",
        clip_max_resolution: float | None = None,
        write_qa: bool = False,
        overwrite: bool = False,
        time_range: TimeRange | None = None,
        burst_edge_trim: bool = True,
    ) -> CohLRWComposite:
        r"""Compute a locally resolution weighted (LRW) coherence composite.

        Coherence observations are weighted by normalized $1/A$ ([1] eq. 3-4)
        from matching RTC products. The bursts of one relative orbit and
        primary acquisition date are mosaicked without the darkened edge of
        each burst, splitting each overlap at its midline, and each
        pixel is weighted by the RTC product of its own burst.

        Parameters
        ----------
        coh_products
            Coherence products to composite.  All must share the same
            polarization and ``baseline_days``.
        rtc_products
            RTC products providing the local scattering area (band 2) used as
            weights. Each coherence burst needs the RTC product of the same
            burst and primary date.
        geobox
            Target MGRS grid cell.
        aggregation
            Temporal aggregation applied after weighting.  Pass ``"mean"`` or
            ``"median"`` for the corresponding statistic, or a float in
            ``(0, 1)`` to compute a weighted quantile at that level.
        clip_max_resolution
            Upper limit for $W = 1/A$, used to suppress extreme weights in
            radar shadow. ``1.0`` is the weight of a facet at 45° local
            incidence; flat terrain weighs ``tan(θ)``, about 0.6 to 1.0.
        write_qa
            If ``True``, also write the two QA layers: a
            contributing-observations map (``Layer.QA_NUM``) and a composite
            quality map (``Layer.QA_CQM``).
            ``QA-CQM`` is $1 / \sum_i W_i A_i$ under every ``aggregation``:
            under ``median`` or a quantile it describes the weighting, not the
            observations the composite's value was taken from.
        overwrite
            If ``False`` and every requested layer's file already exists, skip
            computation and return the existing product, after checking that
            each file records the requested grid and compositing parameters
            and linear units; any other difference, such as the DEM of the
            inputs, is not checked. Otherwise every
            requested layer is written, replacing an existing file, so the
            measurement and its QA layers always come from the same run.
            If no input reaches the cell, an earlier composite of the same
            identity is deleted, all its layers, before the error is raised;
            with ``overwrite=False`` it is left alone.
        time_range
            The compositing window, as calendar days; pass the same
            :class:`~s1bursts.time_range.TimeRange` the search was made with.
            When given, the product is named and tagged with its days rather
            than with the first and last acquisition, so every tile composited
            from one search shares them; the observed span is kept in the
            ``FIRST_ACQUISITION_DATE`` and ``LAST_ACQUISITION_DATE`` tags. An
            input outside the window is a ``ValueError``.
        burst_edge_trim
            Drop the edge of each burst before mosaicking, where the coherence
            window has darkened it. Its width follows the inputs' estimation
            window and the grid spacing: 5 pixels at the default window on a
            10 m grid. ``False`` keeps every pixel.

        Returns
        -------
        CohLRWComposite
            The composite, whose ``layers`` are the measurement and, with
            ``write_qa=True``, the QA layers. Pass a layer to
            ``FileManager.get_product_path`` for its file.

        Raises
        ------
        TypeError
            If ``time_range`` is not a ``TimeRange``.
        ValueError
            If ``coh_products`` or ``rtc_products`` is empty, ``aggregation``
            is neither ``"mean"``, ``"median"`` nor a quantile level in
            ``(0, 1)``, an input lies outside ``time_range``, or
            ``burst_edge_trim`` is ``True`` and the coherence inputs, or with
            ``overwrite=False`` the existing composite, do not record the
            estimation window.
        EmptyCompositeError
            If no input has a valid pixel in ``geobox``; nothing is written.
        InconsistentInputsError
            If the inputs differ in polarization or temporal baseline, an
            input is on another CRS, pixel size or grid than ``geobox``, the
            inputs were made with different DEMs or SNAP versions, where an
            undetermined SNAP version matches any, or the coherence products,
            or the RTC products that weight them, among themselves with
            different processing settings. RTC products that weight no
            observation are not compared. Not a ``CompositeError``: other
            cells given these inputs would meet it too.
        CompositeInputError
            If an input product is missing or cannot be read, or a burst some
            pixel is taken from has no RTC product of the same burst and
            primary date to supply its scattering area.
        ProductSettingsMismatchError
            If ``overwrite`` is ``False`` and an existing layer file records
            another grid than ``geobox``, other or no compositing parameters,
            or values in decibels. Not a ``CompositeError``: other cells would
            meet it too.
        OSError or RasterioError
            If ``overwrite`` is ``False`` and an existing layer file cannot be
            read to check its settings.
        """
        coh_products = list(coh_products)
        if not coh_products:
            msg = "coh_products must not be empty."
            raise ValueError(msg)
        rtc_products = list(rtc_products)
        # Without any, every observation would lack its area: a wrong call, not bad data.
        if not rtc_products:
            msg = "rtc_products must not be empty."
            raise ValueError(msg)
        self._check_aggregation(aggregation)

        polarization = self._check_polarization(coh_products)
        baseline_days = self._check_baseline_days(coh_products)
        acquisitions = self._acquisition_span(coh_products)
        start_date, end_date = self._composite_dates(acquisitions, time_range)

        product = CohLRWComposite(
            mgrs_code=geobox.mgrs_code,
            start_date=start_date,
            end_date=end_date,
            polarization=polarization,
            baseline_days=baseline_days,
            layers=CohLRWComposite.supported_layers if write_qa else _MEASUREMENT_ONLY,
        )
        if not overwrite and self._all_exist(product):
            _check_composite(
                self._fm,
                product,
                geobox,
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                burst_edge_trim=burst_edge_trim,
            )
            return product

        coh_tags = self._read_tags(coh_products)
        burst_edge_trim_px = self._edge_trim_px(
            _agreed_tags(coh_tags), geobox, coherence=True, enabled=burst_edge_trim
        )
        try:
            coh_groups = _mosaic.contributing(
                _mosaic.mosaic_by_orbit_date(
                    self._load_bursts(coh_products, geobox, burst_edge_trim_px), geobox.mgrs_code
                ),
                geobox.mgrs_code,
            )
        except EmptyCompositeError:
            if overwrite:
                self._remove(product)
            raise
        # Only the RTC products that weight an observation are read and compared.
        weighting = [prod for prod in rtc_products if _mosaic.orbit_date_key(prod) in coh_groups]
        area_groups = _mosaic.scattering_areas(
            coh_groups, self._load_band(weighting, geobox, band=2)
        )
        # The tags name the acquisitions that reach the cell, not every input.
        observed = self._contributing_span(coh_products, coh_groups)

        coh_obs = np.stack([mosaic.data for mosaic in coh_groups.values()])  # (N, H, W)
        area_obs = np.stack(list(area_groups.values()))  # (N, H, W)

        weights = self._calc_weights(area_obs, clip_max_resolution)
        composite = self._aggregate_weighted(coh_obs, weights, aggregation)
        source_tags = _agreed_tags(coh_tags, self._read_tags(weighting))
        n_observations = len(coh_obs)
        self._save(
            composite,
            product,
            Layer.MEASUREMENT,
            geobox,
            product_tags.coh_lrw_metadata(
                product,
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                burst_edge_trim_px=burst_edge_trim_px,
                n_observations=n_observations,
                statistics=product_tags.coherence_statistics(composite),
                source_tags=source_tags,
                acquisitions=observed,
            ),
        )
        if write_qa:
            self._write_lrw_qa(
                coh_obs,
                area_obs,
                weights,
                product,
                geobox,
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                burst_edge_trim_px=burst_edge_trim_px,
                source_tags=source_tags,
                acquisitions=observed,
            )
        return product

    def compose_rtc_by_pass(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "mean",
        write_qa: bool = False,
        overwrite: bool = False,
        time_range: TimeRange | None = None,
        burst_edge_trim: bool = True,
    ) -> dict[PassDirection, RTCPassComposite]:
        """Compute unweighted RTC composites stratified by pass direction.

        Bursts from the same orbit and acquisition date are mosaicked without
        the darkened edge of each burst, splitting each overlap at its
        midline.  The resulting orbit-date observations are then aggregated
        temporally (mean or median) per pass direction (ascending /
        descending).

        Parameters
        ----------
        rtc_products
            RTC products to composite.  Products from both pass directions may
            be mixed; they are split by their ``pass_direction``.
        geobox
            Target MGRS grid cell.
        aggregation
            Temporal aggregation method.  Pass ``"mean"`` or ``"median"`` for
            the corresponding statistic, or a float in ``(0, 1)`` for an
            interpolated quantile at that level.
        write_qa
            If ``True``, also write the contributing-observations QA layer
            (``Layer.QA_NUM``) of each pass direction.
        overwrite
            If ``False`` and every requested layer's file already exists, skip
            computation and return the existing product, after checking that
            each file records the requested grid and compositing parameters
            and linear units; any other difference, such as the DEM of the
            inputs, is not checked. Otherwise every
            requested layer is written, replacing an existing file, so the
            measurement and its QA layers always come from the same run.
            A direction that no input reaches has the files of an earlier
            composite of the same identity deleted; with ``overwrite=False``
            they are left alone.
        time_range
            The compositing window, as calendar days; pass the same
            :class:`~s1bursts.time_range.TimeRange` the search was made with.
            When given, the product is named and tagged with its days rather
            than with the first and last acquisition, so every tile composited
            from one search shares them; the observed span is kept in the
            ``FIRST_ACQUISITION_DATE`` and ``LAST_ACQUISITION_DATE`` tags. An
            input outside the window is a ``ValueError``.
        burst_edge_trim
            Drop the edge of each burst before mosaicking, where multilooking
            has darkened it. Its width follows the inputs' looks and the grid
            spacing: 4 pixels at the default looks on a 10 m grid. ``False``
            keeps every pixel.

        Returns
        -------
        dict[PassDirection, RTCPassComposite]
            One composite per pass direction among the inputs, whose
            ``layers`` are the measurement and, with ``write_qa=True``, the
            QA layer. A direction whose inputs have no valid pixel in
            ``geobox`` is left out and gets no file, as if it had no inputs.

        Raises
        ------
        TypeError
            If ``time_range`` is not a ``TimeRange``.
        ValueError
            If ``rtc_products`` is empty, ``aggregation`` is neither ``"mean"``,
            ``"median"`` nor a quantile level in ``(0, 1)``, or an input lies
            outside ``time_range``.
        EmptyCompositeError
            If no input of any direction has a valid pixel in ``geobox``;
            nothing is written.
        InconsistentInputsError
            If the inputs differ in polarization, an input is on another
            CRS, pixel size or grid than ``geobox``, or the inputs were made
            with different DEMs, processing settings or SNAP versions, where
            an undetermined SNAP version matches any. When a direction is
            built, the composite of a direction skipped as existing must agree
            with its inputs too. Not a ``CompositeError``: other cells given
            these inputs would meet it too.
        CompositeInputError
            If an input product is missing or cannot be read.
        ProductSettingsMismatchError
            If ``overwrite`` is ``False`` and an existing layer file records
            another grid than ``geobox``, other or no compositing parameters,
            or values in decibels. Not a ``CompositeError``: other cells would
            meet it too.
        OSError or RasterioError
            If ``overwrite`` is ``False`` and an existing layer file cannot be
            read to check its settings.
        """
        return self._compose_backscatter_by_pass(
            rtc_products,
            geobox,
            name="rtc_products",
            composite_cls=RTCPassComposite,
            metadata_fn=product_tags.rtc_pass_metadata,
            aggregation=aggregation,
            write_qa=write_qa,
            overwrite=overwrite,
            time_range=time_range,
            burst_edge_trim=burst_edge_trim,
        )

    def compose_gtc_by_pass(
        self,
        gtc_products: Collection[GTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "mean",
        write_qa: bool = False,
        overwrite: bool = False,
        time_range: TimeRange | None = None,
        burst_edge_trim: bool = True,
    ) -> dict[PassDirection, GTCPassComposite]:
        """Compute unweighted GTC beta0 composites stratified by pass direction.

        Bursts from the same orbit and acquisition date are mosaicked without
        the darkened edge of each burst, splitting each overlap at its
        midline.  The resulting orbit-date observations are then aggregated
        temporally (mean or median) per pass direction (ascending /
        descending).  The stored value is beta-0 power.

        Parameters
        ----------
        gtc_products
            GTC products to composite.  Products from both pass directions may
            be mixed; they are split by their ``pass_direction``.
        geobox
            Target MGRS grid cell.
        aggregation
            Temporal aggregation method.  Pass ``"mean"`` or ``"median"`` for
            the corresponding statistic, or a float in ``(0, 1)`` for an
            interpolated quantile at that level.
        write_qa
            If ``True``, also write the contributing-observations QA layer
            (``Layer.QA_NUM``) of each pass direction.
        overwrite
            If ``False`` and every requested layer's file already exists, skip
            computation and return the existing product, after checking that
            each file records the requested grid and compositing parameters
            and linear units; any other difference, such as the DEM of the
            inputs, is not checked. Otherwise every
            requested layer is written, replacing an existing file, so the
            measurement and its QA layers always come from the same run.
            A direction that no input reaches has the files of an earlier
            composite of the same identity deleted; with ``overwrite=False``
            they are left alone.
        time_range
            The compositing window, as calendar days; pass the same
            :class:`~s1bursts.time_range.TimeRange` the search was made with.
            When given, the product is named and tagged with its days rather
            than with the first and last acquisition; the observed span is
            kept in the ``FIRST_ACQUISITION_DATE`` and ``LAST_ACQUISITION_DATE``
            tags. An input outside the window is a ``ValueError``.
        burst_edge_trim
            Drop the edge of each burst before mosaicking, where multilooking
            has darkened it. Its width follows the inputs' looks and the grid
            spacing: 4 pixels at the default looks on a 10 m grid. ``False``
            keeps every pixel.

        Returns
        -------
        dict[PassDirection, GTCPassComposite]
            One composite per pass direction among the inputs, whose
            ``layers`` are the measurement and, with ``write_qa=True``, the
            QA layer. A direction whose inputs have no valid pixel in
            ``geobox`` is left out and gets no file, as if it had no inputs.

        Raises
        ------
        TypeError
            If ``time_range`` is not a ``TimeRange``.
        ValueError
            If ``gtc_products`` is empty, ``aggregation`` is neither ``"mean"``,
            ``"median"`` nor a quantile level in ``(0, 1)``, or an input lies
            outside ``time_range``.
        EmptyCompositeError
            If no input of any direction has a valid pixel in ``geobox``;
            nothing is written.
        InconsistentInputsError
            If the inputs differ in polarization, an input is on another
            CRS, pixel size or grid than ``geobox``, or the inputs were made
            with different DEMs, processing settings or SNAP versions, where
            an undetermined SNAP version matches any. When a direction is
            built, the composite of a direction skipped as existing must agree
            with its inputs too. Not a ``CompositeError``: other cells given
            these inputs would meet it too.
        CompositeInputError
            If an input product is missing or cannot be read.
        ProductSettingsMismatchError
            If ``overwrite`` is ``False`` and an existing layer file records
            another grid than ``geobox``, other or no compositing parameters,
            or values in decibels. Not a ``CompositeError``: other cells would
            meet it too.
        OSError or RasterioError
            If ``overwrite`` is ``False`` and an existing layer file cannot be
            read to check its settings.
        """
        return self._compose_backscatter_by_pass(
            gtc_products,
            geobox,
            name="gtc_products",
            composite_cls=GTCPassComposite,
            metadata_fn=product_tags.gtc_pass_metadata,
            aggregation=aggregation,
            write_qa=write_qa,
            overwrite=overwrite,
            time_range=time_range,
            burst_edge_trim=burst_edge_trim,
        )

    def _compose_backscatter_by_pass[C: (RTCPassComposite, GTCPassComposite)](
        self,
        products: Collection[RTCProduct | GTCProduct],
        geobox: MGRSGeoBox,
        *,
        name: str,
        composite_cls: type[C],
        metadata_fn: Callable[..., RasterMetadata],
        aggregation: Literal["mean", "median"] | float,
        write_qa: bool,
        overwrite: bool,
        time_range: TimeRange | None,
        burst_edge_trim: bool,
    ) -> dict[PassDirection, C]:
        """Per-pass unweighted backscatter composite shared by the RTC and GTC entry points."""
        inputs = list(products)
        if not inputs:
            msg = f"{name} must not be empty."
            raise ValueError(msg)
        self._check_aggregation(aggregation)

        polarization = self._check_polarization(inputs)
        start_date, end_date = self._composite_dates(self._acquisition_span(inputs), time_range)

        def make(direction: PassDirection) -> C:
            return composite_cls(
                mgrs_code=geobox.mgrs_code,
                start_date=start_date,
                end_date=end_date,
                polarization=polarization,
                pass_direction=direction,
                layers=composite_cls.supported_layers if write_qa else _MEASUREMENT_ONLY,
            )

        def metadata(product: C, entry: _PassObservations) -> RasterMetadata:
            return metadata_fn(
                product,
                aggregation=aggregation,
                burst_edge_trim_px=entry.burst_edge_trim_px,
                n_observations=entry.n_observations,
                source_tags=entry.source_tags,
                acquisitions=entry.acquisitions,
            )

        return self._compose_by_pass(
            inputs,
            geobox,
            make=make,
            metadata=metadata,
            aggregation=aggregation,
            overwrite=overwrite,
            burst_edge_trim=burst_edge_trim,
            coherence=False,
        )

    def _compose_by_pass[C: (RTCPassComposite, GTCPassComposite, CohPassComposite)](
        self,
        products: Sequence[ComposableProduct],
        geobox: MGRSGeoBox,
        *,
        make: Callable[[PassDirection], C],
        metadata: Callable[[C, _PassObservations], RasterMetadata],
        aggregation: Literal["mean", "median"] | float,
        overwrite: bool,
        burst_edge_trim: bool,
        coherence: bool,
    ) -> dict[PassDirection, C]:
        """Composite each pass direction on its own, skipping finished ones before any read.

        Every existing direction is checked before any is built. The inputs
        of every direction to be built are checked against each other and
        against the existing composite of every other direction, also one
        this call has no inputs for, whose own inputs are not read, so a
        mismatch writes nothing.

        Raises
        ------
        EmptyCompositeError
            If no direction has an input with a valid pixel in the cell.
        ProductSettingsMismatchError
            If a direction's composite exists with other settings.
        InconsistentInputsError
            If the inputs of the directions to be built differ in DEM,
            processing settings or SNAP version, from each other or from the
            composite of an existing direction.
        """
        by_pass: defaultdict[PassDirection, list[ComposableProduct]] = defaultdict(list)
        for prod in products:
            by_pass[prod.pass_direction].append(prod)

        result: dict[PassDirection, C] = {}
        existing_tags: dict[str, dict[str, str]] = {}
        if not overwrite:
            result, existing_tags = self._existing_by_pass(
                by_pass, make, geobox, aggregation, burst_edge_trim=burst_edge_trim
            )
        building = {d: subset for d, subset in sorted(by_pass.items()) if d not in result}
        if building:
            existing_tags |= self._other_direction_tags(by_pass, make)
        source_tags = self._source_tags_by_pass(building, existing_tags)
        for direction, subset in building.items():
            product = make(direction)
            burst_edge_trim_px = self._edge_trim_px(
                source_tags[direction], geobox, coherence=coherence, enabled=burst_edge_trim
            )
            groups = _mosaic.reaching(
                _mosaic.mosaic_by_orbit_date(
                    self._load_bursts(subset, geobox, burst_edge_trim_px), geobox.mgrs_code
                )
            )
            if not groups:
                # Like a direction without inputs: no composite, rather than an all-NaN file.
                logger.debug("no %s input reaches cell %s", direction.lower(), geobox.mgrs_code)
                if overwrite:
                    self._remove(product)
                continue
            observations = np.stack([mosaic.data for mosaic in groups.values()])  # (N, H, W)
            composite = self._aggregate_unweighted(observations, aggregation)
            entry = _PassObservations(
                composite=composite,
                n_observations=len(observations),
                source_tags=source_tags[direction],
                # This file holds one direction, so its span is that direction's.
                acquisitions=self._contributing_span(subset, groups),
                burst_edge_trim_px=burst_edge_trim_px,
            )
            self._save(
                composite,
                product,
                Layer.MEASUREMENT,
                geobox,
                metadata(product, entry),
            )
            if Layer.QA_NUM in product.layers:
                self._write_pass_qa(
                    observations,
                    product,
                    geobox,
                    aggregation=aggregation,
                    burst_edge_trim_px=entry.burst_edge_trim_px,
                    source_tags=entry.source_tags,
                    acquisitions=entry.acquisitions,
                )
            result[direction] = product

        if not result:
            msg = f"no input observation has a valid pixel in cell {geobox.mgrs_code}"
            raise EmptyCompositeError(msg)
        # Sorted, as before the existing directions were checked first.
        return dict(sorted(result.items()))

    def _source_tags_by_pass(
        self,
        by_pass: Mapping[PassDirection, Sequence[ComposableProduct]],
        existing: Mapping[str, Mapping[str, str]],
    ) -> dict[PassDirection, dict[str, str]]:
        """Check that every input agrees across the directions, then return each one's tags.

        ``existing`` maps the file name of each existing direction composite
        to its tags, which the inputs must agree with too.
        """
        tags = iter(self._read_tags([prod for subset in by_pass.values() for prod in subset]))
        per_direction = {
            direction: [next(tags) for _ in subset] for direction, subset in by_pass.items()
        }
        every_input = [entry for entries in per_direction.values() for entry in entries]
        _agreed_tags(every_input)
        if every_input:
            for name, recorded in existing.items():
                _check_existing_composite(name, recorded, every_input)
        return {direction: _agreed_tags(entries) for direction, entries in per_direction.items()}

    def _other_direction_tags(
        self,
        by_pass: Mapping[PassDirection, Sequence[ComposableProduct]],
        make: Callable[[PassDirection], CompositeProduct],
    ) -> dict[str, dict[str, str]]:
        """Return the tags of each input-less direction's existing composite, by file name.

        A call with the inputs of one direction must not add a composite that
        disagrees with the other direction's already on disk. Only its tags
        are read; it is neither checked against this call nor rebuilt.
        """
        tags: dict[str, dict[str, str]] = {}
        for direction in _PASS_DIRECTIONS:
            if direction in by_pass:
                continue
            product = make(direction)
            if self._fm.product_exists(product, Layer.MEASUREMENT):
                name = self._fm.get_product_path(product).name
                tags[name] = self._read_tags([product])[0]
        return tags

    def _existing_by_pass[C: (RTCPassComposite, GTCPassComposite, CohPassComposite)](
        self,
        directions: Iterable[PassDirection],
        make: Callable[[PassDirection], C],
        geobox: MGRSGeoBox,
        aggregation: Literal["mean", "median"] | float,
        *,
        burst_edge_trim: bool,
    ) -> tuple[dict[PassDirection, C], dict[str, dict[str, str]]]:
        """Check the existing direction composites, returning them and their tags by file name."""
        existing: dict[PassDirection, C] = {}
        tags: dict[str, dict[str, str]] = {}
        for direction in sorted(directions):
            product = make(direction)
            if self._all_exist(product):
                name = self._fm.get_product_path(product).name
                tags[name] = _check_composite(
                    self._fm,
                    product,
                    geobox,
                    aggregation=aggregation,
                    burst_edge_trim=burst_edge_trim,
                )
                existing[direction] = product
        return existing, tags

    def compose_coherence_by_pass(
        self,
        coh_products: Collection[CoherenceProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "mean",
        write_qa: bool = False,
        overwrite: bool = False,
        time_range: TimeRange | None = None,
        burst_edge_trim: bool = True,
    ) -> dict[PassDirection, CohPassComposite]:
        """Compute unweighted coherence composites stratified by pass direction.

        Bursts from the same orbit and primary acquisition date are mosaicked
        without the darkened edge of each burst, splitting each overlap at its
        midline.  The resulting orbit-date observations are then
        aggregated temporally (mean or median) per pass direction (ascending /
        descending).

        Parameters
        ----------
        coh_products
            Coherence products to composite.  All must share the same
            ``baseline_days``.
        geobox
            Target MGRS grid cell.
        aggregation
            Temporal aggregation method.  Pass ``"mean"`` or ``"median"`` for
            the corresponding statistic, or a float in ``(0, 1)`` for an
            interpolated quantile at that level.
        write_qa
            If ``True``, also write the contributing-observations QA layer
            (``Layer.QA_NUM``) of each pass direction.
        overwrite
            If ``False`` and every requested layer's file already exists, skip
            computation and return the existing product, after checking that
            each file records the requested grid and compositing parameters
            and linear units; any other difference, such as the DEM of the
            inputs, is not checked. Otherwise every
            requested layer is written, replacing an existing file, so the
            measurement and its QA layers always come from the same run.
            A direction that no input reaches has the files of an earlier
            composite of the same identity deleted; with ``overwrite=False``
            they are left alone.
        time_range
            The compositing window, as calendar days; pass the same
            :class:`~s1bursts.time_range.TimeRange` the search was made with.
            When given, the product is named and tagged with its days rather
            than with the first and last acquisition, so every tile composited
            from one search shares them; the observed span is kept in the
            ``FIRST_ACQUISITION_DATE`` and ``LAST_ACQUISITION_DATE`` tags. An
            input outside the window is a ``ValueError``.
        burst_edge_trim
            Drop the edge of each burst before mosaicking, where the coherence
            window has darkened it. Its width follows the inputs' estimation
            window and the grid spacing: 5 pixels at the default window on a
            10 m grid. ``False`` keeps every pixel.

        Returns
        -------
        dict[PassDirection, CohPassComposite]
            One composite per pass direction among the inputs, whose
            ``layers`` are the measurement and, with ``write_qa=True``, the
            QA layer. A direction whose inputs have no valid pixel in
            ``geobox`` is left out and gets no file, as if it had no inputs.

        Raises
        ------
        TypeError
            If ``time_range`` is not a ``TimeRange``.
        ValueError
            If ``coh_products`` is empty, ``aggregation`` is neither ``"mean"``,
            ``"median"`` nor a quantile level in ``(0, 1)``, an input lies
            outside ``time_range``, or ``burst_edge_trim`` is ``True`` and the
            coherence inputs, or with ``overwrite=False`` the existing
            composite, do not record the estimation window.
        EmptyCompositeError
            If no input of any direction has a valid pixel in ``geobox``;
            nothing is written.
        InconsistentInputsError
            If the inputs differ in polarization or temporal baseline, an
            input is on another CRS, pixel size or grid than ``geobox``, or
            the inputs were made with different DEMs, processing settings or
            SNAP versions, where an undetermined SNAP version matches any.
            When a direction is built, the composite of a direction skipped
            as existing must agree with its inputs too. Not a
            ``CompositeError``: other cells given these inputs would meet it
            too.
        CompositeInputError
            If an input product is missing or cannot be read.
        ProductSettingsMismatchError
            If ``overwrite`` is ``False`` and an existing layer file records
            another grid than ``geobox``, other or no compositing parameters,
            or values in decibels. Not a ``CompositeError``: other cells would
            meet it too.
        OSError or RasterioError
            If ``overwrite`` is ``False`` and an existing layer file cannot be
            read to check its settings.
        """
        coh_products = list(coh_products)
        if not coh_products:
            msg = "coh_products must not be empty."
            raise ValueError(msg)
        self._check_aggregation(aggregation)

        polarization = self._check_polarization(coh_products)
        baseline_days = self._check_baseline_days(coh_products)
        acquisitions = self._acquisition_span(coh_products)
        start_date, end_date = self._composite_dates(acquisitions, time_range)

        def make(direction: PassDirection) -> CohPassComposite:
            return CohPassComposite(
                mgrs_code=geobox.mgrs_code,
                start_date=start_date,
                end_date=end_date,
                polarization=polarization,
                pass_direction=direction,
                baseline_days=baseline_days,
                layers=CohPassComposite.supported_layers if write_qa else _MEASUREMENT_ONLY,
            )

        def metadata(product: CohPassComposite, entry: _PassObservations) -> RasterMetadata:
            return product_tags.coh_pass_metadata(
                product,
                aggregation=aggregation,
                burst_edge_trim_px=entry.burst_edge_trim_px,
                n_observations=entry.n_observations,
                statistics=product_tags.coherence_statistics(entry.composite),
                source_tags=entry.source_tags,
                acquisitions=entry.acquisitions,
            )

        return self._compose_by_pass(
            coh_products,
            geobox,
            make=make,
            metadata=metadata,
            aggregation=aggregation,
            overwrite=overwrite,
            burst_edge_trim=burst_edge_trim,
            coherence=True,
        )

    # ------------------------------------------------------------------
    # Private helpers — I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _check_baseline_days(products: Sequence[CoherenceProduct]) -> int:
        """Return the common temporal baseline, or raise InconsistentInputsError if they differ.

        Coherence falls off with the baseline, so mixing 12-day and 24-day
        pairs averages two different quantities — and the composite carries a
        single ``baseline_days`` in its name and its tags.
        """
        baselines = {p.baseline_days for p in products}
        if len(baselines) != 1:
            listed = ", ".join(str(b) for b in sorted(baselines))
            msg = f"All coherence products must share a temporal baseline; got {listed} days."
            raise InconsistentInputsError(msg)
        return baselines.pop()

    @staticmethod
    def _check_aggregation(aggregation: Literal["mean", "median"] | float) -> None:
        """Raise ValueError unless ``aggregation`` names a statistic or a quantile in (0, 1)."""
        if aggregation in ("mean", "median"):
            return
        if isinstance(aggregation, float) and 0.0 < aggregation < 1.0:
            return
        msg = (
            f"aggregation must be 'mean', 'median' or a quantile level in the open "
            f"interval (0, 1); got {aggregation!r}."
        )
        raise ValueError(msg)

    @staticmethod
    def _check_polarization(products: Sequence[ComposableProduct]) -> Polarization:
        """Return the common polarization or raise InconsistentInputsError if products differ."""
        polarization = products[0].polarization
        if not all(p.polarization == polarization for p in products):
            listed = ", ".join(sorted({p.polarization for p in products}))
            msg = f"All products must have the same polarization; got {listed}."
            raise InconsistentInputsError(msg)
        return polarization

    def _load_band(
        self,
        products: Collection[ComposableProduct],
        geobox: MGRSGeoBox,
        band: int = 1,
    ) -> list[_mosaic.LoadedBand]:
        """Parallel windowed rasterio reads of a single band clipped to the geobox extent.

        Each product is read into a ``(H, W)`` float32 array using a boundless
        windowed read aligned to ``geobox``.  Out-of-bounds pixels are filled
        with NaN.
        """

        def load(prod: ComposableProduct) -> _mosaic.LoadedBand:
            return self._load_one(prod, geobox, band, 0)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(load, products))

    def _load_one(
        self,
        product: ComposableProduct,
        geobox: MGRSGeoBox,
        band: int,
        margin: int,
    ) -> _mosaic.LoadedBand:
        """Read one band of one product, widened by ``margin`` pixels on every side."""
        path = self._fm.get_product_path(product)
        if not path.exists():
            msg = f"product file not found: {path}"
            raise CompositeInputError(msg)
        try:
            return product, self._read_window(path, geobox, band, margin)
        except OSError as err:
            msg = f"cannot read '{path.name}': {err}"
            raise CompositeInputError(msg) from err

    @staticmethod
    def _read_window(
        path: Path | UPath,
        geobox: MGRSGeoBox,
        band: int,
        margin: int = 0,
    ) -> NDArray[np.float32]:
        """Read one band of ``path`` on the grid of ``geobox`` widened by ``margin`` pixels."""
        left, bottom, right, top = geobox.boundingbox
        pad_x = margin * geobox.resolution.x
        pad_y = margin * abs(geobox.resolution.y)
        with rasterio_open(path) as src:
            if src.crs != geobox.crs:
                msg = (
                    f"'{path.name}' is in {src.crs}, but the target grid is "
                    f"{geobox.crs}; reproject it or compose onto its own grid."
                )
                raise InconsistentInputsError(msg)
            if src.res != (geobox.resolution.x, abs(geobox.resolution.y)):
                msg = (
                    f"'{path.name}' has {src.res[0]} m pixels, but the target grid has "
                    f"{geobox.resolution.x} m; all inputs must share the output grid."
                )
                raise InconsistentInputsError(msg)
            win = from_bounds(
                left - pad_x, bottom - pad_y, right + pad_x, top + pad_y, transform=src.transform
            )
            # Rounding the window would silently shift an input whose pixel
            # corners do not coincide with the target grid's.
            offsets = (win.col_off, win.row_off)
            if any(abs(o - round(o)) > _ALIGNMENT_TOLERANCE for o in offsets):
                msg = (
                    f"'{path.name}' is offset from the target grid by a fraction of a "
                    f"pixel ({offsets[0]:.3f}, {offsets[1]:.3f}); all inputs must share "
                    "the output grid."
                )
                raise InconsistentInputsError(msg)
            win = win.round_lengths().round_offsets()
            data = src.read(band, window=win, boundless=True, fill_value=np.nan)
        return data.astype(np.float32)

    def _remove(self, product: CompositeProduct) -> None:
        """Delete every layer file of ``product``'s identity, so no earlier composite remains."""
        for layer in product.supported_layers:
            path = self._fm.get_product_path(product, layer)
            if path.exists():
                path.unlink()
                logger.debug("removed %s, which no input reaches", path.name)

    def _all_exist(self, product: CompositeProduct) -> bool:
        """Whether every requested layer of ``product`` has its file already, logging the skip."""
        if not all(self._fm.product_exists(product, layer) for layer in product.layers):
            return False
        logger.debug("skipping existing %s", self._fm.get_product_path(product).name)
        return True

    def _profile_for(self, product: CompositeProduct, layer: Layer) -> dict[str, Any]:
        """Merge the default and layer-specific rasterio profiles."""
        override = {
            **self._profiles.get(_DEFAULT_PROFILE_KEY, {}),
            **self._profiles.get(_layer_product_type(product, layer), {}),
        }
        base_profile = {"dtype": np.float32, "nodata": np.nan}
        return build_rasterio_profile(base_profile, override)

    def _save(
        self,
        data: NDArray[np.number[Any]],
        product: CompositeProduct,
        layer: Layer,
        geobox: MGRSGeoBox,
        metadata: RasterMetadata,
        *,
        dtype: type[np.generic] = np.float32,
        nodata: float = np.nan,
    ) -> None:
        """Write a ``(H, W)`` array as one layer's single-band GeoTIFF, replacing any file."""
        path = self._fm.get_product_path(product, layer)
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = self._profile_for(product, layer)
        profile.update(
            count=1,
            dtype=dtype,
            nodata=nodata,
            crs=geobox.crs,
            transform=geobox.transform,
            height=geobox.height,
            width=geobox.width,
        )
        # Every product _save writes is single-band (count=1 above), so exact
        # statistics for that one band merge into whatever band_tags the
        # product's own metadata already carries.
        band_tags = [
            {
                **(metadata.band_tags[0] if metadata.band_tags else {}),
                **product_tags.band_statistics(data, nodata),
            },
        ]
        digest = write_cog(
            data,
            path,
            profile,
            band_names=metadata.band_names,
            tags=metadata.tags,
            band_tags=band_tags if band_tags[0] else None,
            units=metadata.units,
            checksum=self._checksum,
        )
        if digest is not None:
            self.checksums[str(path)] = digest
        logger.debug("saved %s", path.name)

    def _input_tags(
        self,
        products: Collection[ComposableProduct],
        *,
        rtc_weights: Collection[ComposableProduct] = (),
    ) -> dict[str, str]:
        """Read the per-burst inputs' tags and return those they agree on, as ``_agreed_tags``."""
        tags = self._read_tags([*products, *rtc_weights])
        return _agreed_tags(tags[: len(products)], tags[len(products) :])

    def _read_tags(
        self, products: Sequence[ComposableProduct | CompositeProduct]
    ) -> list[dict[str, str]]:
        """Read the tags of each product, in parallel and in order."""

        def read(product: ComposableProduct | CompositeProduct) -> dict[str, str]:
            path = self._fm.get_product_path(product)
            try:
                with rasterio_open(path) as src:
                    return dict(src.tags())
            except OSError as err:
                msg = f"cannot read '{path.name}': {err}"
                raise CompositeInputError(msg) from err

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(read, products))

    def _write_lrw_qa(
        self,
        values: NDArray[np.floating[Any]],
        scattering_area: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
        product: RTCLRWComposite | CohLRWComposite,
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float,
        clip_max_resolution: float | None,
        burst_edge_trim_px: int,
        source_tags: Mapping[str, str],
        acquisitions: product_tags.Acquisitions,
    ) -> None:
        """Compute and write the contributing-observations and composite quality layers."""
        n_observations = len(values)
        self._save(
            self._count_contributions(values, weights),
            product,
            Layer.QA_NUM,
            geobox,
            product_tags.qa_metadata(
                product,
                Layer.QA_NUM,
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                burst_edge_trim_px=burst_edge_trim_px,
                n_observations=n_observations,
                source_tags=source_tags,
                acquisitions=acquisitions,
            ),
            dtype=np.uint16,
            nodata=0,
        )
        cqm = self._calc_cqm(values, scattering_area, weights)
        self._save(
            cqm,
            product,
            Layer.QA_CQM,
            geobox,
            product_tags.qa_metadata(
                product,
                Layer.QA_CQM,
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                burst_edge_trim_px=burst_edge_trim_px,
                n_observations=n_observations,
                source_tags=source_tags,
                acquisitions=acquisitions,
            ),
            dtype=np.float32,
            nodata=np.nan,
        )

    def _write_pass_qa(
        self,
        observations: NDArray[np.floating[Any]],
        product: RTCPassComposite | GTCPassComposite | CohPassComposite,
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float,
        burst_edge_trim_px: int,
        source_tags: Mapping[str, str],
        acquisitions: product_tags.Acquisitions,
    ) -> None:
        """Compute and write the contributing-observations layer of a pass composite."""
        self._save(
            self._count_observations(observations),
            product,
            Layer.QA_NUM,
            geobox,
            product_tags.qa_metadata(
                product,
                Layer.QA_NUM,
                aggregation=aggregation,
                burst_edge_trim_px=burst_edge_trim_px,
                n_observations=len(observations),
                source_tags=source_tags,
                acquisitions=acquisitions,
            ),
            dtype=np.uint16,
            nodata=0,
        )

    # ------------------------------------------------------------------
    # Private helpers — acquisitions
    # ------------------------------------------------------------------

    @staticmethod
    def _acquisition_span(products: Collection[ComposableProduct]) -> product_tags.Acquisitions:
        """First and last acquisition date among ``products``, as calendar days.

        A burst found through CDSE carries its UTC acquisition time; one indexed
        from a filename carries midnight. Both name the same day, which is the
        resolution every date tag and every window is expressed at.
        """
        days = [_calendar_day(_mosaic.acquisition_date(p)) for p in products]
        return min(days), max(days)

    @staticmethod
    def _composite_dates(
        acquisitions: product_tags.Acquisitions,
        time_range: TimeRange | None,
    ) -> tuple[Timestamp, Timestamp]:
        """Pick the dates a composite is named and tagged with.

        The compositing window when the caller gave one — the same for every
        tile composited from the same search — otherwise the observed span.

        Raises
        ------
        ValueError
            If an input lies outside ``time_range``: the window is a claim about
            what the composite was built from, and an input beyond it would make
            that claim false.
        """
        if time_range is None:
            return acquisitions
        if not isinstance(time_range, TimeRange):
            # A ``(start, stop)`` pair of instants, such as a search range, ends
            # one day later than the window it was made from.
            msg = (
                "time_range must be a TimeRange of calendar days, "
                f"received {type(time_range).__name__}"
            )
            raise TypeError(msg)
        first, last = acquisitions
        if first not in time_range or last not in time_range:
            msg = f"inputs span {first.date()}..{last.date()}, outside the time range {time_range}"
            raise ValueError(msg)
        return Timestamp(time_range.start), Timestamp(time_range.end)

    @staticmethod
    def _edge_trim_px(
        tags: Mapping[str, str], geobox: MGRSGeoBox, *, coherence: bool, enabled: bool
    ) -> int:
        """Pixels to drop along each burst's data edge, as ``burst_edge_trim`` asks."""
        if not enabled:
            return 0
        try:
            return _mosaic.auto_edge_trim_px(
                tags, coherence=coherence, grid_spacing=geobox.resolution.x
            )
        except KeyError as err:
            msg = (
                f"the coherence inputs do not record {err.args[0]}, which the burst edge "
                "trim is derived from; pass burst_edge_trim=False"
            )
            raise ValueError(msg) from err

    def _load_bursts(
        self,
        products: Collection[ComposableProduct],
        geobox: MGRSGeoBox,
        trim_px: int,
    ) -> list[_mosaic.TrimmedBurst]:
        """Read the measurement band of each product, trimmed and with its depth.

        The band is read ``EDGE_SEARCH_MARGIN_M`` beyond the cell, so that a
        data edge outside the cell still counts. Each widened read is cropped
        as soon as it arrives, so only one per worker is held at a time.
        """
        margin = math.ceil(_mosaic.EDGE_SEARCH_MARGIN_M / geobox.resolution.x)

        def load(prod: ComposableProduct) -> _mosaic.TrimmedBurst:
            return _mosaic.trim(*self._load_one(prod, geobox, 1, margin), margin, trim_px)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(load, products))

    @classmethod
    def _contributing_span(
        cls,
        products: Collection[ComposableProduct],
        groups: Mapping[_mosaic.OrbitDateKey, _mosaic.AcquisitionMosaic],
    ) -> product_tags.Acquisitions:
        """First and last acquisition day among the products of the contributing groups."""
        return cls._acquisition_span([p for p in products if _mosaic.orbit_date_key(p) in groups])

    # ------------------------------------------------------------------
    # Private helpers — weighting
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_weights(
        scattering_area: NDArray[np.floating[Any]],
        clip_max_resolution: float | None,
    ) -> NDArray[np.floating[Any]]:
        """Compute resolution weights from the local scattering area.

        The local area is inversely proportional to the local resolution, so
        the unnormalized resolution weight is ``w = 1 / A`` (Small et al. 2022,
        eq. 3-4; CEOS-ARD SAR v1.3.1, eq. A5.1).  Normalization to sum one
        (eq. A5.2) happens in the aggregation, which needs the clipped weights.

        Invalid pixels (``scattering_area <= 0`` or non-finite) receive weight
        zero.  An optional upper clip on ``1/A`` suppresses extreme weights in
        shadow areas.
        """
        is_valid = (scattering_area > 0) & np.isfinite(scattering_area)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(is_valid, 1.0 / scattering_area, 0.0)
        if clip_max_resolution is not None:
            weights = weights.clip(max=clip_max_resolution)
        return weights

    @staticmethod
    def _count_observations(values: NDArray[np.floating[Any]]) -> NDArray[np.uint16]:
        """Per-pixel count of finite observations over the first (N) axis."""
        return np.count_nonzero(np.isfinite(values), axis=0).astype(np.uint16)

    @staticmethod
    def _count_contributions(
        values: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
    ) -> NDArray[np.uint16]:
        """Per-pixel count of observations that carry weight into the composite.

        An observation contributes when it is finite and its resolution weight
        is non-zero, which is the pairing ``_weighted_mean`` sums over.  The
        CEOS-ARD Contributing Observations Image.
        """
        contributes = np.isfinite(values) & (weights > 0)
        return np.count_nonzero(contributes, axis=0).astype(np.uint16)

    @classmethod
    def _calc_cqm(
        cls,
        values: NDArray[np.floating[Any]],
        scattering_area: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
    ) -> NDArray[np.float32]:
        """Composite quality map (CQM) over the first (N) axis.

        ``1 / Σ(Wᵢ · Aᵢ)``, with ``Wᵢ`` normalized to sum to one, the published
        form (Small et al. 2022, eq. 6; CEOS-ARD SAR v1.3.1, eq. A5.2), whatever
        the composite's aggregation.  A weighted quantile picks its observations
        by rank in the measured value, which changes with speckle from pixel to
        pixel, so the area of the observations it picked is noise rather than a
        quality summary; the weighted mean of the area describes the weighting
        those picks are drawn under.

        An observation whose value is non-finite did not enter the composite,
        so its weight is zeroed here too and its area does not count.
        """
        contributing = np.where(np.isfinite(values), weights, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cqm = (1.0 / cls._weighted_mean(scattering_area, contributing)).astype(np.float32)
        cqm[contributing.sum(axis=0) == 0] = np.nan
        return cqm

    # ------------------------------------------------------------------
    # Private helpers — aggregation
    # ------------------------------------------------------------------

    @classmethod
    def _aggregate_weighted(
        cls,
        values: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
        aggregation: Literal["mean", "median"] | float,
    ) -> NDArray[np.floating[Any]]:
        """Dispatch to the aggregation function."""
        cls._check_aggregation(aggregation)
        if aggregation == "mean":
            return cls._weighted_mean(values, weights)
        if aggregation == "median":
            return cls._weighted_quantile(values, weights, q=0.5)
        return cls._weighted_quantile(values, weights, q=aggregation)

    @staticmethod
    def _weighted_mean(
        values: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        """Weighted mean over the first (N) axis.

        NaN values are excluded by zeroing their weights before summation.
        """
        masked_weights = np.where(np.isfinite(values), weights, 0.0)
        weight_sum = masked_weights.sum(axis=0)
        with np.errstate(invalid="ignore"):
            result = np.nansum(values * masked_weights, axis=0) / weight_sum
        result[weight_sum == 0] = np.nan
        return result

    @staticmethod
    def _weighted_quantile(
        values: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
        q: float,
    ) -> NDArray[np.floating[Any]]:
        """Vectorised, linearly interpolated weighted quantile over the first (N) axis.

        Each value is placed at the plotting position
        ``(cumulative_weight - 0.5 * weight) / total`` and the target quantile
        ``q`` is linearly interpolated between the two bracketing order
        statistics. Pixels whose total weight is zero return NaN.

        An observation is excluded by being made NaN, which sorts it to the end
        where the forward-fill below removes it from the interpolation. Zeroing
        its weight is not enough on its own: a zero-weight observation still
        sorts by its own value, and interpolating across it blends that value
        into the result.
        """
        values = np.where(weights > 0, values, np.nan)
        masked_weights = np.where(np.isfinite(values), weights, 0.0)

        sort_idx = np.argsort(values, axis=0)
        sorted_vals = np.take_along_axis(values, sort_idx, axis=0)
        sorted_wts = np.take_along_axis(masked_weights, sort_idx, axis=0)

        cum_wts = np.cumsum(sorted_wts, axis=0)
        weight_sum = cum_wts[-1]  # (H, W)
        valid = weight_sum > 0

        # Hazen plotting position of each sorted value, normalized to [0, 1].
        with np.errstate(invalid="ignore", divide="ignore"):
            positions = (cum_wts - 0.5 * sorted_wts) / weight_sum[np.newaxis]

        # Forward-fill NaN order statistics (which carry zero weight and sort to
        # the end) so interpolation near q = 1 never blends into a NaN.
        n = values.shape[0]
        finite = np.isfinite(sorted_vals)
        indices = np.broadcast_to(np.arange(n)[:, np.newaxis, np.newaxis], sorted_vals.shape)
        fill_idx = np.where(finite, indices, 0)
        np.maximum.accumulate(fill_idx, axis=0, out=fill_idx)
        filled_vals = np.take_along_axis(sorted_vals, fill_idx, axis=0)

        # Bracket q between the lower and upper plotting positions.
        at_or_above = positions >= q
        hi = np.where(at_or_above.any(axis=0), np.argmax(at_or_above, axis=0), n - 1)
        lo = np.clip(hi - 1, 0, None)

        pos_hi = np.take_along_axis(positions, hi[np.newaxis], axis=0).squeeze(0)
        pos_lo = np.take_along_axis(positions, lo[np.newaxis], axis=0).squeeze(0)
        val_hi = np.take_along_axis(filled_vals, hi[np.newaxis], axis=0).squeeze(0)
        val_lo = np.take_along_axis(filled_vals, lo[np.newaxis], axis=0).squeeze(0)

        span = pos_hi - pos_lo
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(span > 0, (q - pos_lo) / span, 0.0)
        frac = np.clip(frac, 0.0, 1.0)

        result = val_lo + frac * (val_hi - val_lo)
        result[~valid] = np.nan
        return result

    def _aggregate_unweighted(
        self,
        values: NDArray[np.floating[Any]],
        aggregation: Literal["mean", "median"] | float,
    ) -> NDArray[np.floating[Any]]:
        """Unweighted temporal aggregate over the first (N) axis.

        Equivalent to ``_aggregate_weighted`` with uniform weights, so the
        mean and the interpolated ``median`` / quantile use the same definition
        as the LRW composites.
        """
        return self._aggregate_weighted(values, np.ones_like(values), aggregation)
