"""Locally Resolution Weighted (LRW) compositing for Sentinel-1 products.

Implements the LRW backscatter composite and its composite quality map (CQM)
of [Small et al. (2022)](https://doi.org/10.1109/TGRS.2021.3055562), which the CEOS-ARD
Product Family Specification for Synthetic Aperture Radar adopts as its
Composite Backscatter (CB) product.

The weights come from the local scattering area ``A``, the area terrain
flattening divides beta nought by, so ``A = 1`` is a flat reference facet.
That is the CEOS-ARD term; the same quantity is the local contributing area
of Small et al., the area normalization factor (ANF) of OPERA, and the
``simulatedImage`` band SNAP's Terrain-Flattening writes. This package stores
it as the ``scattering_area`` band of every RTC product.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import TYPE_CHECKING, Any, Literal, overload

from freezebase.raster import build_rasterio_profile, rasterio_open, write_cog
import numpy as np
from numpy.typing import NDArray
from pandas import Timestamp
from rasterio.windows import from_bounds

from s1bursts import product_tags
from s1bursts.errors import EmptyCompositeError
from s1bursts.product import (
    RTCLRWQA,
    CoherenceProduct,
    CohLRWComposite,
    CohLRWQA,
    CohPassComposite,
    CohPassQA,
    GTCPassComposite,
    GTCPassQA,
    GTCProduct,
    QaMetric,
    RTCLRWComposite,
    RTCPassComposite,
    RTCPassQA,
    RTCProduct,
)
from s1bursts.time_range import TimeRange, calendar_day

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from freezebase.mgrs import MGRSGeoBox

    from s1bursts.filemanager import FileManager
    from s1bursts.product_tags import RasterMetadata
    from s1bursts.query import PassDirection, Polarization

logger = logging.getLogger(__name__)


type ComposableProduct = CoherenceProduct | RTCProduct | GTCProduct
type CompositeProduct = (
    CohLRWComposite | RTCLRWComposite | RTCPassComposite | CohPassComposite | GTCPassComposite
)
type LRWQAProduct = RTCLRWQA | CohLRWQA
type QAProduct = RTCLRWQA | CohLRWQA | RTCPassQA | CohPassQA | GTCPassQA

# Return shapes when ``write_qa=True``: the composite paired with its QA-product bundle.
type RTCLRWResult = tuple[RTCLRWComposite, tuple[RTCLRWQA, RTCLRWQA]]
type CohLRWResult = tuple[CohLRWComposite, tuple[CohLRWQA, CohLRWQA]]
type RTCPassResult = tuple[RTCPassComposite, RTCPassQA]
type GTCPassResult = tuple[GTCPassComposite, GTCPassQA]
type CohPassResult = tuple[CohPassComposite, CohPassQA]

_OrbitDateKey = tuple[int, Timestamp]


def _calendar_day(value: Timestamp) -> Timestamp:
    """``value`` as a naive midnight timestamp of its UTC calendar day."""
    return Timestamp(calendar_day(value))


_LoadedEntry = tuple[ComposableProduct, NDArray[np.float32]]

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


class ResolutionWeighter:
    """Compute LRW and per-pass composites for Sentinel-1 products.

    Each compose call receives its target ``geobox``, so one instance can be
    reused across MGRS grid cells.

    Parameters
    ----------
    file_manager
        Used to resolve input file paths and output product paths.
    n_workers
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
        n_workers: int = 8,
        checksum: bool = False,
    ) -> None:
        self._fm = file_manager
        self._n_workers = n_workers
        self._profiles = _split_profiles(rasterio_profile)
        self._checksum = checksum
        self.checksums: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @overload
    def compose_rtc_lrw(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        clip_max_resolution: float | None = ...,
        decibel: bool = ...,
        write_qa: Literal[False] = False,
        cqm_decibel: bool = ...,
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> RTCLRWComposite: ...
    @overload
    def compose_rtc_lrw(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        clip_max_resolution: float | None = ...,
        decibel: bool = ...,
        write_qa: Literal[True],
        cqm_decibel: bool = ...,
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> RTCLRWResult: ...
    @overload
    def compose_rtc_lrw(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        clip_max_resolution: float | None = ...,
        decibel: bool = ...,
        write_qa: bool,
        cqm_decibel: bool = ...,
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> RTCLRWComposite | RTCLRWResult: ...
    def compose_rtc_lrw(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "median",
        clip_max_resolution: float | None = None,
        decibel: bool = False,
        write_qa: bool = False,
        cqm_decibel: bool = False,
        overwrite: bool = False,
        time_range: TimeRange | None = None,
    ) -> RTCLRWComposite | RTCLRWResult:
        r"""Compute a locally resolution weighted (LRW) RTC intensity composite.

        Band 1 supplies intensity and band 2 the local scattering area $A$.
        Observations are weighted by normalized $1/A$ ([1] eq. 3-4, [2]
        eq. A5.1-A5.3) after mosaicking each orbit and acquisition date.

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
            radar shadow. ``1.0`` is the weight of a flat reference facet.
        decibel
            Convert the result from power to decibels after aggregation.
            Non-positive values become NaN.
        write_qa
            If ``True``, also write two auxiliary QA products alongside the
            composite: a contributing-observations map (``QA-NUM``) and a
            composite quality map (``QA-CQM``); see [2], items 2.21 and 2.22.
        cqm_decibel
            Write ``QA-CQM`` as $-10 \log_{10}(\sum_i W_i A_i)$ instead of
            linear units. Independent of ``decibel``.
        overwrite
            If ``False`` and the output file already exists, skip computation
            and return the existing product metadata.
        time_range
            The compositing window, as calendar days; pass the same
            :class:`~s1bursts.time_range.TimeRange` the search was made with.
            When given, the product is named and tagged with its days rather
            than with the first and last acquisition, so every tile composited
            from one search shares them; the observed span is kept in the
            ``FIRST_ACQUISITION_DATE`` and ``LAST_ACQUISITION_DATE`` tags. An
            input outside the window is a ``ValueError``.

        Returns
        -------
        RTCLRWComposite or tuple[RTCLRWComposite, tuple[RTCLRWQA, RTCLRWQA]]
            The composite product metadata, or ``(composite, (qa_num, qa_cqm))``
            when ``write_qa=True``.

        Raises
        ------
        ValueError
            If ``rtc_products`` is empty.
        EmptyCompositeError
            If no input has a valid pixel in ``geobox``; nothing is written.
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
        )
        qa: dict[QaMetric, RTCLRWQA] = (
            {
                metric: RTCLRWQA(
                    mgrs_code=geobox.mgrs_code,
                    start_date=start_date,
                    end_date=end_date,
                    polarization=polarization,
                    qa_metric=metric,
                )
                for metric in QaMetric
            }
            if write_qa
            else {}
        )
        if not overwrite and all(map(self._fm.product_exists, [product, *qa.values()])):
            logger.debug("skipping existing %s", self._fm.get_product_path(product).name)
            return (product, (qa[QaMetric.NUM], qa[QaMetric.CQM])) if qa else product

        rtc_groups = self._contributing(
            self._stitch_by_orbit_date(self._load_band(rtc_products, geobox, band=1)),
            geobox,
        )
        area_groups = self._stitch_by_orbit_date(self._load_band(rtc_products, geobox, band=2))
        # The tags name the acquisitions that reach the cell, not every input.
        observed = self._contributing_span(rtc_products, rtc_groups)

        self._check_area_coverage(rtc_groups, area_groups)
        rtc_obs = np.stack(list(rtc_groups.values()))  # (N, H, W)
        area_obs = np.stack([area_groups[key] for key in rtc_groups])  # (N, H, W)

        weights = self._calc_weights(area_obs, clip_max_resolution)
        composite = self._aggregate_weighted(rtc_obs, weights, aggregation)
        if decibel:
            composite = self._to_decibel(composite)
        source_tags = self._input_tags(rtc_products)
        n_observations = len(rtc_obs)
        self._save(
            composite,
            product,
            geobox,
            product_tags.rtc_lrw_metadata(
                product,
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                n_observations=n_observations,
                decibel=decibel,
                source_tags=source_tags,
                acquisitions=observed,
            ),
            overwrite=overwrite,
        )
        if qa:
            self._write_lrw_qa(
                rtc_obs,
                area_obs,
                weights,
                qa,
                geobox,
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                cqm_decibel=cqm_decibel,
                source_tags=source_tags,
                acquisitions=observed,
                overwrite=overwrite,
            )
        return (product, (qa[QaMetric.NUM], qa[QaMetric.CQM])) if qa else product

    @overload
    def compose_coherence_lrw(
        self,
        coh_products: Collection[CoherenceProduct],
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        clip_max_resolution: float | None = ...,
        write_qa: Literal[False] = False,
        cqm_decibel: bool = ...,
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> CohLRWComposite: ...
    @overload
    def compose_coherence_lrw(
        self,
        coh_products: Collection[CoherenceProduct],
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        clip_max_resolution: float | None = ...,
        write_qa: Literal[True],
        cqm_decibel: bool = ...,
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> CohLRWResult: ...
    @overload
    def compose_coherence_lrw(
        self,
        coh_products: Collection[CoherenceProduct],
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        clip_max_resolution: float | None = ...,
        write_qa: bool,
        cqm_decibel: bool = ...,
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> CohLRWComposite | CohLRWResult: ...
    def compose_coherence_lrw(
        self,
        coh_products: Collection[CoherenceProduct],
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "median",
        clip_max_resolution: float | None = None,
        write_qa: bool = False,
        cqm_decibel: bool = False,
        overwrite: bool = False,
        time_range: TimeRange | None = None,
    ) -> CohLRWComposite | CohLRWResult:
        r"""Compute a locally resolution weighted (LRW) coherence composite.

        Coherence observations are weighted by normalized $1/A$ from matching
        RTC products. Products are matched and mosaicked by relative orbit and
        primary acquisition date.

        Parameters
        ----------
        coh_products
            Coherence products to composite.  All must share the same
            polarization and ``baseline_days``.
        rtc_products
            RTC products providing the local scattering area (band 2) used as
            weights. They must cover the coherence orbit/date combinations.
        geobox
            Target MGRS grid cell.
        aggregation
            Temporal aggregation applied after weighting.  Pass ``"mean"`` or
            ``"median"`` for the corresponding statistic, or a float in
            ``(0, 1)`` to compute a weighted quantile at that level.
        clip_max_resolution
            Upper limit for $W = 1/A$, used to suppress extreme weights in
            radar shadow. ``1.0`` is the weight of a flat reference facet.
        write_qa
            If ``True``, also write two auxiliary QA products alongside the
            composite: a contributing-observations map (``QA-NUM``) and a
            composite quality map (``QA-CQM``); see [2], items 2.21 and 2.22.
        cqm_decibel
            Write ``QA-CQM`` as $-10 \log_{10}(\sum_i W_i A_i)$ instead of
            linear units.
        overwrite
            If ``False`` and the output file already exists, skip computation
            and return the existing product metadata.
        time_range
            The compositing window, as calendar days; pass the same
            :class:`~s1bursts.time_range.TimeRange` the search was made with.
            When given, the product is named and tagged with its days rather
            than with the first and last acquisition, so every tile composited
            from one search shares them; the observed span is kept in the
            ``FIRST_ACQUISITION_DATE`` and ``LAST_ACQUISITION_DATE`` tags. An
            input outside the window is a ``ValueError``.

        Returns
        -------
        CohLRWComposite or tuple[CohLRWComposite, tuple[CohLRWQA, CohLRWQA]]
            The composite product metadata, or ``(composite, (qa_num, qa_cqm))``
            when ``write_qa=True``.

        Raises
        ------
        ValueError
            If ``coh_products`` is empty.
        EmptyCompositeError
            If no input has a valid pixel in ``geobox``; nothing is written.
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

        product = CohLRWComposite(
            mgrs_code=geobox.mgrs_code,
            start_date=start_date,
            end_date=end_date,
            polarization=polarization,
            baseline_days=baseline_days,
        )
        qa: dict[QaMetric, CohLRWQA] = (
            {
                metric: CohLRWQA(
                    mgrs_code=geobox.mgrs_code,
                    start_date=start_date,
                    end_date=end_date,
                    polarization=polarization,
                    baseline_days=baseline_days,
                    qa_metric=metric,
                )
                for metric in QaMetric
            }
            if write_qa
            else {}
        )
        if not overwrite and all(map(self._fm.product_exists, [product, *qa.values()])):
            logger.debug("skipping existing %s", self._fm.get_product_path(product).name)
            return (product, (qa[QaMetric.NUM], qa[QaMetric.CQM])) if qa else product

        coh_groups = self._contributing(
            self._stitch_by_orbit_date(self._load_band(coh_products, geobox, band=1)),
            geobox,
        )
        area_groups = self._stitch_by_orbit_date(self._load_band(rtc_products, geobox, band=2))
        # The tags name the acquisitions that reach the cell, not every input.
        observed = self._contributing_span(coh_products, coh_groups)

        self._check_area_coverage(coh_groups, area_groups)
        coh_obs = np.stack(list(coh_groups.values()))  # (N, H, W)
        area_obs = np.stack([area_groups[key] for key in coh_groups])  # (N, H, W)

        weights = self._calc_weights(area_obs, clip_max_resolution)
        composite = self._aggregate_weighted(coh_obs, weights, aggregation)
        source_tags = self._input_tags(coh_products, additional_dem_products=rtc_products)
        n_observations = len(coh_obs)
        self._save(
            composite,
            product,
            geobox,
            product_tags.coh_lrw_metadata(
                product,
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                n_observations=n_observations,
                statistics=product_tags.coherence_statistics(composite),
                source_tags=source_tags,
                acquisitions=observed,
            ),
            overwrite=overwrite,
        )
        if qa:
            self._write_lrw_qa(
                coh_obs,
                area_obs,
                weights,
                qa,
                geobox,
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                cqm_decibel=cqm_decibel,
                source_tags=source_tags,
                acquisitions=observed,
                overwrite=overwrite,
            )
        return (product, (qa[QaMetric.NUM], qa[QaMetric.CQM])) if qa else product

    @overload
    def compose_rtc_by_pass(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        decibel: bool = ...,
        write_qa: Literal[False] = False,
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> dict[PassDirection, RTCPassComposite]: ...
    @overload
    def compose_rtc_by_pass(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        decibel: bool = ...,
        write_qa: Literal[True],
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> dict[PassDirection, RTCPassResult]: ...
    @overload
    def compose_rtc_by_pass(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        decibel: bool = ...,
        write_qa: bool,
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> dict[PassDirection, RTCPassComposite | RTCPassResult]: ...
    def compose_rtc_by_pass(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "median",
        decibel: bool = False,
        write_qa: bool = False,
        overwrite: bool = False,
        time_range: TimeRange | None = None,
    ) -> Mapping[PassDirection, RTCPassComposite | RTCPassResult]:
        """Compute unweighted RTC composites stratified by pass direction.

        Bursts from the same orbit and acquisition date are mosaicked using the
        first non-NaN pixel strategy.  The resulting orbit-date observations are
        then aggregated temporally (mean or median) per pass direction
        (ascending / descending).

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
        decibel
            If ``True``, convert the aggregated composite from intensity
            (power) to decibels before writing; non-positive intensities
            become NaN.  Aggregation always happens in the linear domain.
        write_qa
            If ``True``, also write a contributing-observations QA product
            (``QA-NUM``) per pass direction.
        overwrite
            If ``False`` and the output file already exists, skip computation
            and return the existing product metadata.
        time_range
            The compositing window, as calendar days; pass the same
            :class:`~s1bursts.time_range.TimeRange` the search was made with.
            When given, the product is named and tagged with its days rather
            than with the first and last acquisition, so every tile composited
            from one search shares them; the observed span is kept in the
            ``FIRST_ACQUISITION_DATE`` and ``LAST_ACQUISITION_DATE`` tags. An
            input outside the window is a ``ValueError``.

        Returns
        -------
        dict[PassDirection, RTCPassComposite or tuple[RTCPassComposite, RTCPassQA]]
            One entry per pass direction.  Each value is the composite, or
            ``(composite, qa_num)`` when ``write_qa=True``.
        """
        return self._compose_backscatter_by_pass(
            rtc_products,
            geobox,
            composite_cls=RTCPassComposite,
            qa_cls=RTCPassQA,
            metadata_fn=product_tags.rtc_pass_metadata,
            aggregation=aggregation,
            decibel=decibel,
            write_qa=write_qa,
            overwrite=overwrite,
            time_range=time_range,
        )

    @overload
    def compose_gtc_by_pass(
        self,
        gtc_products: Collection[GTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        decibel: bool = ...,
        write_qa: Literal[False] = False,
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> dict[PassDirection, GTCPassComposite]: ...
    @overload
    def compose_gtc_by_pass(
        self,
        gtc_products: Collection[GTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        decibel: bool = ...,
        write_qa: Literal[True],
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> dict[PassDirection, GTCPassResult]: ...
    @overload
    def compose_gtc_by_pass(
        self,
        gtc_products: Collection[GTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        decibel: bool = ...,
        write_qa: bool,
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> dict[PassDirection, GTCPassComposite | GTCPassResult]: ...
    def compose_gtc_by_pass(
        self,
        gtc_products: Collection[GTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "median",
        decibel: bool = False,
        write_qa: bool = False,
        overwrite: bool = False,
        time_range: TimeRange | None = None,
    ) -> Mapping[PassDirection, GTCPassComposite | GTCPassResult]:
        """Compute unweighted GTC beta0 composites stratified by pass direction.

        Bursts from the same orbit and acquisition date are mosaicked using the
        first non-NaN pixel strategy.  The resulting orbit-date observations are
        then aggregated temporally (mean or median) per pass direction
        (ascending / descending).  The stored value is beta-0 power, or
        decibels with ``decibel=True``.

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
        decibel
            If ``True``, convert the aggregated composite from intensity
            (power) to decibels before writing; non-positive intensities
            become NaN.  Aggregation always happens in the linear domain.
        write_qa
            If ``True``, also write a contributing-observations QA product
            (``QA-NUM``) per pass direction.
        overwrite
            If ``False`` and the output file already exists, skip computation
            and return the existing product metadata.
        time_range
            The compositing window, as calendar days; pass the same
            :class:`~s1bursts.time_range.TimeRange` the search was made with.
            When given, the product is named and tagged with its days rather
            than with the first and last acquisition; the observed span is
            kept in the ``FIRST_ACQUISITION_DATE`` and ``LAST_ACQUISITION_DATE``
            tags. An input outside the window is a ``ValueError``.

        Returns
        -------
        dict[PassDirection, GTCPassComposite or tuple[GTCPassComposite, GTCPassQA]]
            One entry per pass direction.  Each value is the composite, or
            ``(composite, qa_num)`` when ``write_qa=True``.
        """
        return self._compose_backscatter_by_pass(
            gtc_products,
            geobox,
            composite_cls=GTCPassComposite,
            qa_cls=GTCPassQA,
            metadata_fn=product_tags.gtc_pass_metadata,
            aggregation=aggregation,
            decibel=decibel,
            write_qa=write_qa,
            overwrite=overwrite,
            time_range=time_range,
        )

    def _compose_backscatter_by_pass[
        C: (RTCPassComposite, GTCPassComposite),
        Q: (RTCPassQA, GTCPassQA),
    ](
        self,
        products: Collection[RTCProduct | GTCProduct],
        geobox: MGRSGeoBox,
        *,
        composite_cls: type[C],
        qa_cls: type[Q],
        metadata_fn: Callable[..., RasterMetadata],
        aggregation: Literal["mean", "median"] | float,
        decibel: bool,
        write_qa: bool,
        overwrite: bool,
        time_range: TimeRange | None,
    ) -> dict[PassDirection, C | tuple[C, Q]]:
        """Per-pass unweighted composite shared by the RTC and GTC entry points."""
        inputs = list(products)
        if not inputs:
            return {}
        self._check_aggregation(aggregation)

        polarization = self._check_polarization(inputs)
        acquisitions = self._acquisition_span(inputs)
        start_date, end_date = self._composite_dates(acquisitions, time_range)

        loaded = self._load_band(inputs, geobox, band=1)

        subsets: defaultdict[PassDirection, list[_LoadedEntry]] = defaultdict(list)
        for prod, data in loaded:
            subsets[prod.pass_direction].append((prod, data))

        result: dict[PassDirection, C | tuple[C, Q]] = {}
        for direction, subset in subsets.items():
            product = composite_cls(
                mgrs_code=geobox.mgrs_code,
                start_date=start_date,
                end_date=end_date,
                polarization=polarization,
                pass_direction=direction,
            )
            qa_num: list[Q] = (
                [
                    qa_cls(
                        mgrs_code=geobox.mgrs_code,
                        start_date=start_date,
                        end_date=end_date,
                        polarization=polarization,
                        pass_direction=direction,
                    ),
                ]
                if write_qa
                else []
            )
            if not overwrite and all(map(self._fm.product_exists, [product, *qa_num])):
                logger.debug("skipping existing %s", self._fm.get_product_path(product).name)
                result[direction] = (product, qa_num[0]) if qa_num else product
                continue

            groups = self._stitch_by_orbit_date(subset)
            observations = np.stack(list(groups.values()))  # (N, H, W)
            composite = self._aggregate_unweighted(observations, aggregation)
            if decibel:
                composite = self._to_decibel(composite)
            source_tags = self._input_tags([prod for prod, _ in subset])
            # This file holds one direction, so its span is that direction's.
            acquisitions = self._acquisition_span([prod for prod, _ in subset])
            self._save(
                composite,
                product,
                geobox,
                metadata_fn(
                    product,
                    aggregation=aggregation,
                    n_observations=len(observations),
                    decibel=decibel,
                    source_tags=source_tags,
                    acquisitions=acquisitions,
                ),
                overwrite=overwrite,
            )
            for qa_product in qa_num:
                self._save(
                    self._count_observations(observations),
                    qa_product,
                    geobox,
                    product_tags.qa_metadata(
                        qa_product,
                        aggregation=aggregation,
                        n_observations=len(observations),
                        source_tags=source_tags,
                        acquisitions=acquisitions,
                    ),
                    dtype=np.uint16,
                    nodata=0,
                    overwrite=overwrite,
                )
            result[direction] = (product, qa_num[0]) if qa_num else product

        return result

    @overload
    def compose_coherence_by_pass(
        self,
        coh_products: Collection[CoherenceProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        write_qa: Literal[False] = False,
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> dict[PassDirection, CohPassComposite]: ...
    @overload
    def compose_coherence_by_pass(
        self,
        coh_products: Collection[CoherenceProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        write_qa: Literal[True],
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> dict[PassDirection, CohPassResult]: ...
    @overload
    def compose_coherence_by_pass(
        self,
        coh_products: Collection[CoherenceProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        write_qa: bool,
        overwrite: bool = ...,
        time_range: TimeRange | None = ...,
    ) -> dict[PassDirection, CohPassComposite | CohPassResult]: ...
    def compose_coherence_by_pass(
        self,
        coh_products: Collection[CoherenceProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "median",
        write_qa: bool = False,
        overwrite: bool = False,
        time_range: TimeRange | None = None,
    ) -> Mapping[PassDirection, CohPassComposite | CohPassResult]:
        """Compute unweighted coherence composites stratified by pass direction.

        Bursts from the same orbit and primary acquisition date are mosaicked
        using the first non-NaN pixel strategy.  The resulting orbit-date
        observations are then aggregated temporally (mean or median) per pass
        direction (ascending / descending).

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
            If ``True``, also write a contributing-observations QA product
            (``QA-NUM``) per pass direction.
        overwrite
            If ``False`` and the output file already exists, skip computation
            and return the existing product metadata.
        time_range
            The compositing window, as calendar days; pass the same
            :class:`~s1bursts.time_range.TimeRange` the search was made with.
            When given, the product is named and tagged with its days rather
            than with the first and last acquisition, so every tile composited
            from one search shares them; the observed span is kept in the
            ``FIRST_ACQUISITION_DATE`` and ``LAST_ACQUISITION_DATE`` tags. An
            input outside the window is a ``ValueError``.

        Returns
        -------
        dict[PassDirection, CohPassComposite or tuple[CohPassComposite, CohPassQA]]
            One entry per pass direction.  Each value is the composite, or
            ``(composite, qa_num)`` when ``write_qa=True``.
        """
        coh_products = list(coh_products)
        if not coh_products:
            return {}
        self._check_aggregation(aggregation)

        polarization = self._check_polarization(coh_products)
        baseline_days = self._check_baseline_days(coh_products)
        acquisitions = self._acquisition_span(coh_products)
        start_date, end_date = self._composite_dates(acquisitions, time_range)

        loaded = self._load_band(coh_products, geobox, band=1)

        subsets: defaultdict[PassDirection, list[_LoadedEntry]] = defaultdict(list)
        for prod, data in loaded:
            subsets[prod.pass_direction].append((prod, data))

        result: dict[PassDirection, CohPassComposite | CohPassResult] = {}
        for direction, subset in subsets.items():
            product = CohPassComposite(
                mgrs_code=geobox.mgrs_code,
                start_date=start_date,
                end_date=end_date,
                polarization=polarization,
                pass_direction=direction,
                baseline_days=baseline_days,
            )
            qa_num: list[CohPassQA] = (
                [
                    CohPassQA(
                        mgrs_code=geobox.mgrs_code,
                        start_date=start_date,
                        end_date=end_date,
                        polarization=polarization,
                        pass_direction=direction,
                        baseline_days=baseline_days,
                    ),
                ]
                if write_qa
                else []
            )
            if not overwrite and all(map(self._fm.product_exists, [product, *qa_num])):
                logger.debug("skipping existing %s", self._fm.get_product_path(product).name)
                result[direction] = (product, qa_num[0]) if qa_num else product
                continue

            groups = self._stitch_by_orbit_date(subset)
            coh_obs = np.stack(list(groups.values()))  # (N, H, W)
            composite = self._aggregate_unweighted(coh_obs, aggregation)
            source_tags = self._input_tags([prod for prod, _ in subset])
            acquisitions = self._acquisition_span([prod for prod, _ in subset])
            self._save(
                composite,
                product,
                geobox,
                product_tags.coh_pass_metadata(
                    product,
                    aggregation=aggregation,
                    n_observations=len(coh_obs),
                    statistics=product_tags.coherence_statistics(composite),
                    source_tags=source_tags,
                    acquisitions=acquisitions,
                ),
                overwrite=overwrite,
            )
            for qa_product in qa_num:
                self._save(
                    self._count_observations(coh_obs),
                    qa_product,
                    geobox,
                    product_tags.qa_metadata(
                        qa_product,
                        aggregation=aggregation,
                        n_observations=len(coh_obs),
                        source_tags=source_tags,
                        acquisitions=acquisitions,
                    ),
                    dtype=np.uint16,
                    nodata=0,
                    overwrite=overwrite,
                )
            result[direction] = (product, qa_num[0]) if qa_num else product

        return result

    # ------------------------------------------------------------------
    # Private helpers — I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _check_baseline_days(products: Sequence[CoherenceProduct]) -> int:
        """Return the common temporal baseline, or raise ValueError if products differ.

        Coherence falls off with the baseline, so mixing 12-day and 24-day
        pairs averages two different quantities — and the composite carries a
        single ``baseline_days`` in its name and its tags.
        """
        baselines = {p.baseline_days for p in products}
        if len(baselines) != 1:
            listed = ", ".join(str(b) for b in sorted(baselines))
            msg = f"All coherence products must share a temporal baseline; got {listed} days."
            raise ValueError(msg)
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
        """Return the common polarization or raise ValueError if products differ."""
        polarization = products[0].polarization
        if not all(p.polarization == polarization for p in products):
            msg = "All products must have the same polarization."
            raise ValueError(msg)
        return polarization

    def _load_band(
        self,
        products: Collection[ComposableProduct],
        geobox: MGRSGeoBox,
        band: int = 1,
    ) -> list[_LoadedEntry]:
        """Parallel windowed rasterio reads of a single band clipped to the geobox extent.

        Each product is read into a ``(H, W)`` float32 array using a boundless
        windowed read aligned to ``geobox``.  Out-of-bounds pixels are filled
        with NaN.
        """
        left, bottom, right, top = geobox.boundingbox

        def _load_one(prod: ComposableProduct) -> _LoadedEntry:
            path = self._fm.get_product_path(prod)
            if not path.exists():
                msg = f"product file not found: {path}"
                raise FileNotFoundError(msg)
            with rasterio_open(path) as src:
                if src.crs != geobox.crs:
                    msg = (
                        f"'{path.name}' is in {src.crs}, but the target grid is "
                        f"{geobox.crs}; reproject it or compose onto its own grid."
                    )
                    raise ValueError(msg)
                if src.res != (geobox.resolution.x, abs(geobox.resolution.y)):
                    msg = (
                        f"'{path.name}' has {src.res[0]} m pixels, but the target grid has "
                        f"{geobox.resolution.x} m; all inputs must share the output grid."
                    )
                    raise ValueError(msg)
                win = from_bounds(left, bottom, right, top, transform=src.transform)
                # Rounding the window would silently shift an input whose pixel
                # corners do not coincide with the target grid's.
                offsets = (win.col_off, win.row_off)
                if any(abs(o - round(o)) > _ALIGNMENT_TOLERANCE for o in offsets):
                    msg = (
                        f"'{path.name}' is offset from the target grid by a fraction of a "
                        f"pixel ({offsets[0]:.3f}, {offsets[1]:.3f}); all inputs must share "
                        "the output grid."
                    )
                    raise ValueError(msg)
                win = win.round_lengths().round_offsets()
                data = src.read(band, window=win, boundless=True, fill_value=np.nan)
            return prod, data.astype(np.float32)

        with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            return list(executor.map(_load_one, products))

    def _profile_for(self, product: CompositeProduct | QAProduct) -> dict[str, Any]:
        """Merge the default and product-specific rasterio profiles."""
        override = {
            **self._profiles.get(_DEFAULT_PROFILE_KEY, {}),
            **self._profiles.get(product.product_type, {}),
        }
        base_profile = {"dtype": np.float32, "nodata": np.nan}
        return build_rasterio_profile(base_profile, override)

    def _save(
        self,
        data: NDArray[np.number[Any]],
        product: CompositeProduct | QAProduct,
        geobox: MGRSGeoBox,
        metadata: RasterMetadata,
        *,
        dtype: type[np.generic] = np.float32,
        nodata: float = np.nan,
        overwrite: bool = False,
    ) -> None:
        """Write a ``(H, W)`` array as a single-band GeoTIFF, skipping existing files."""
        path = self._fm.get_product_path(product)
        if not overwrite and path.exists():
            logger.debug("skipping existing %s", path.name)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = self._profile_for(product)
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
        additional_dem_products: Collection[ComposableProduct] = (),
    ) -> dict[str, str]:
        """Tags that every per-burst input agrees on.

        The SNAP version and the coherence estimation window exist nowhere at
        compositing time except in the input files. A key is carried forward
        only where every input states the same value, so a composite spanning
        two SNAP versions stays silent instead of claiming whichever its first
        input happened to use.

        All inputs must identify the same DEM. ``additional_dem_products``
        supports coherence LRW composites, whose RTC contributing-area rasters
        also depend on a DEM but do not supply the other coherence provenance.

        Called only after ``_load_band`` has read the same products, so the
        files are there.
        """

        def read(product: ComposableProduct) -> dict[str, str]:
            with rasterio_open(self._fm.get_product_path(product)) as src:
                return dict(src.tags())

        primary_count = len(products)
        all_products = [*products, *additional_dem_products]
        with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            all_input_tags = list(executor.map(read, all_products))
        per_input = all_input_tags[:primary_count]
        if not per_input:
            return {}

        dem_values = {tags.get("DEM", product_tags.UNKNOWN) for tags in all_input_tags}
        if len(dem_values) > 1:
            listed = ", ".join(sorted(dem_values))
            msg = f"All inputs must use the same DEM; got {listed}."
            raise ValueError(msg)

        first, *rest = per_input
        agreed = {k: v for k, v in first.items() if all(other.get(k) == v for other in rest)}
        agreed["DEM"] = dem_values.pop()

        # Mixing SNAP versions is legitimate, but it is worth saying out loud
        for key in ("PROCESSING_SOFTWARE_VERSION",):
            values = {tags[key] for tags in per_input if key in tags}
            # Absent from every input is agreement, not disagreement: an archive
            # written before these tags existed simply has nothing to say.
            if key in agreed or len(values) < 2:  # noqa: PLR2004
                continue
            logger.warning(
                "inputs disagree on %s (%s); the composite will record it as unknown",
                key,
                ", ".join(sorted(values)),
            )
        return agreed

    def _write_lrw_qa(
        self,
        values: NDArray[np.floating[Any]],
        scattering_area: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
        qa: Mapping[QaMetric, LRWQAProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float,
        clip_max_resolution: float | None,
        cqm_decibel: bool,
        source_tags: Mapping[str, str],
        acquisitions: product_tags.Acquisitions,
        overwrite: bool,
    ) -> None:
        """Compute and write the contributing-observations and composite quality QA products."""
        n_observations = len(values)
        self._save(
            self._count_contributions(values, weights),
            qa[QaMetric.NUM],
            geobox,
            product_tags.qa_metadata(
                qa[QaMetric.NUM],
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                n_observations=n_observations,
                source_tags=source_tags,
                acquisitions=acquisitions,
            ),
            dtype=np.uint16,
            nodata=0,
            overwrite=overwrite,
        )
        cqm = self._calc_cqm(values, scattering_area, weights, aggregation)
        self._save(
            self._to_decibel(cqm) if cqm_decibel else cqm,
            qa[QaMetric.CQM],
            geobox,
            product_tags.qa_metadata(
                qa[QaMetric.CQM],
                aggregation=aggregation,
                clip_max_resolution=clip_max_resolution,
                n_observations=n_observations,
                cqm_decibel=cqm_decibel,
                source_tags=source_tags,
                acquisitions=acquisitions,
            ),
            dtype=np.float32,
            nodata=np.nan,
            overwrite=overwrite,
        )

    # ------------------------------------------------------------------
    # Private helpers — spatial mosaicking
    # ------------------------------------------------------------------

    @classmethod
    def _acquisition_span(
        cls, products: Collection[ComposableProduct]
    ) -> product_tags.Acquisitions:
        """First and last acquisition date among ``products``, as calendar days.

        A burst found through CDSE carries its UTC acquisition time; one indexed
        from a filename carries midnight. Both name the same day, which is the
        resolution every date tag and every window is expressed at.
        """
        days = [_calendar_day(cls._acquisition_date(p)) for p in products]
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
    def _acquisition_date(product: ComposableProduct) -> Timestamp:
        """Return the acquisition date used for orbit-date grouping.

        ``CoherenceProduct`` uses ``date_primary``; ``RTCProduct`` uses ``date``.
        """
        return product.date_primary if isinstance(product, CoherenceProduct) else product.date

    def _stitch_by_orbit_date(
        self,
        loaded: Sequence[_LoadedEntry],
    ) -> dict[_OrbitDateKey, NDArray[np.floating[Any]]]:
        """Group loaded arrays by orbit/date and mosaic within each group.

        The grouping key is ``(rel_orbit_number, acquisition_date)``, where
        ``acquisition_date`` is ``date_primary`` for coherence products and
        ``date`` for RTC products.  Within each group the arrays are combined
        using pixel-wise first non-NaN selection, which avoids averaging along
        burst seams.
        """
        groups: defaultdict[_OrbitDateKey, list[NDArray[np.floating[Any]]]] = defaultdict(list)
        for prod, data in loaded:
            groups[self._group_key(prod)].append(data)
        return {key: self._first_notna(arrays) for key, arrays in sorted(groups.items())}

    @staticmethod
    def _contributing(
        groups: dict[_OrbitDateKey, NDArray[np.floating[Any]]],
        geobox: MGRSGeoBox,
    ) -> dict[_OrbitDateKey, NDArray[np.floating[Any]]]:
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
        kept = {key: data for key, data in groups.items() if np.isfinite(data).any()}
        if not kept:
            msg = f"no input observation has a valid pixel in cell {geobox.mgrs_code}"
            raise EmptyCompositeError(msg)
        return kept

    @classmethod
    def _group_key(cls, product: ComposableProduct) -> _OrbitDateKey:
        """Return the orbit/date group a product is stitched into."""
        return product.rel_orbit_number, cls._acquisition_date(product).normalize()

    @classmethod
    def _contributing_span(
        cls,
        products: Collection[ComposableProduct],
        groups: dict[_OrbitDateKey, NDArray[np.floating[Any]]],
    ) -> product_tags.Acquisitions:
        """First and last acquisition day among the products of the contributing groups."""
        return cls._acquisition_span([p for p in products if cls._group_key(p) in groups])

    @staticmethod
    def _first_notna(arrays: Sequence[NDArray[np.floating[Any]]]) -> NDArray[np.floating[Any]]:
        """Pixel-wise first non-NaN value across a list of ``(H, W)`` arrays.

        For each ``(row, col)`` position the value from the first array in the
        list that is finite is used.  Positions that are NaN in all arrays
        remain NaN.
        """
        stacked = np.stack(arrays, axis=0)  # (N, H, W)
        finite = np.isfinite(stacked)
        idx = finite.argmax(axis=0, keepdims=True)  # (1, H, W)
        result = np.take_along_axis(stacked, idx, axis=0).squeeze(0)  # (H, W)
        result[~finite.any(axis=0)] = np.nan
        return result

    # ------------------------------------------------------------------
    # Private helpers — weighting
    # ------------------------------------------------------------------

    @staticmethod
    def _check_area_coverage(
        obs_groups: dict[_OrbitDateKey, NDArray[np.floating[Any]]],
        area_groups: dict[_OrbitDateKey, NDArray[np.floating[Any]]],
    ) -> None:
        """Verify every observation group has a matching scattering-area group."""
        missing = [key for key in obs_groups if key not in area_groups]
        if missing:
            formatted = ", ".join(f"(orbit {orbit}, {date.date()})" for orbit, date in missing)
            msg = f"No matching RTC scattering area for orbit/date group(s): {formatted}"
            raise ValueError(msg)

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
    def _to_decibel(data: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        """Convert intensity (power) to decibels; non-positive values become NaN."""
        decibel = np.full_like(data, np.nan, dtype=np.float32)
        valid = data > 0
        decibel[valid] = 10.0 * np.log10(data[valid])
        return decibel

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
        CEOS-ARD Contributing Observations Image (SAR PFS v1.3.1, item 2.21).
        """
        contributes = np.isfinite(values) & (weights > 0)
        return np.count_nonzero(contributes, axis=0).astype(np.uint16)

    @classmethod
    def _calc_cqm(
        cls,
        values: NDArray[np.floating[Any]],
        scattering_area: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
        aggregation: Literal["mean", "median"] | float,
    ) -> NDArray[np.float32]:
        """Composite quality map (CQM) over the first (N) axis.

        The reciprocal of the local scattering area ``Aᵢ`` combined under the
        same weights and the same aggregation as the composite itself, so the
        map describes the composite that was actually written.  Under ``mean``
        this is ``1 / Σ(Wᵢ · Aᵢ)`` with ``Wᵢ`` normalized to sum to one, the
        published form (Small et al. 2022, eq. 6; CEOS-ARD SAR v1.3.1,
        eq. A5.2).  The area is aggregated and then inverted, rather than the
        resolutions being aggregated, which is what keeps the ``mean`` case
        equal to that form.

        An observation whose value is non-finite did not enter the composite,
        so its weight is zeroed here too and its area does not count.
        """
        contributing = np.where(np.isfinite(values), weights, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            aggregated_area = cls._aggregate_weighted(scattering_area, contributing, aggregation)
            cqm = (1.0 / aggregated_area).astype(np.float32)
        invalid = contributing.sum(axis=0) == 0
        cqm[invalid] = np.nan
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
