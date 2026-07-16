"""Locally Resolution Weighted (LRW) compositing for Sentinel-1 products."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import TYPE_CHECKING, Any, Literal, overload

from freezecore.raster import build_rasterio_profile, rasterio_open
import numpy as np
from numpy.typing import NDArray
from pandas import Timestamp
from rasterio.windows import from_bounds

from s1bursts.product import (
    RTCLRWQA,
    CoherenceProduct,
    CohLRWComposite,
    CohLRWQA,
    CohPassComposite,
    CohPassQA,
    QaMetric,
    RTCLRWComposite,
    RTCPassComposite,
    RTCPassQA,
    RTCProduct,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from freezecore.mgrs import MGRSGeoBox

    from s1bursts.filemanager import FileManager
    from s1bursts.query import OrbitDirection, Polarization

logger = logging.getLogger(__name__)


type ComposableProduct = CoherenceProduct | RTCProduct
type CompositeProduct = CohLRWComposite | RTCLRWComposite | RTCPassComposite | CohPassComposite
type LRWQAProduct = RTCLRWQA | CohLRWQA
type QAProduct = RTCLRWQA | CohLRWQA | RTCPassQA | CohPassQA

# Return shapes when ``write_qa=True``: the composite paired with its QA-product bundle.
type RTCLRWResult = tuple[RTCLRWComposite, tuple[RTCLRWQA, RTCLRWQA]]
type CohLRWResult = tuple[CohLRWComposite, tuple[CohLRWQA, CohLRWQA]]
type RTCPassResult = tuple[RTCPassComposite, RTCPassQA]
type CohPassResult = tuple[CohPassComposite, CohPassQA]

_OrbitDateKey = tuple[int, Timestamp]
_LoadedEntry = tuple[ComposableProduct, NDArray[np.float32]]


class ResolutionWeighter:
    """Compute LRW and per-orbit-pass composites for Sentinel-1 products.

    The class is stateless with respect to the MGRS grid cell: the ``geobox``
    that defines the spatial extent, CRS, and pixel grid is passed to each
    compose call.  This allows one ``ResolutionWeighter`` instance to be
    reused across many cells in a grid loop.

    Parameters
    ----------
    file_manager : FileManager
        Used to resolve input file paths and output product paths.
    n_workers : int, default 8
        Number of threads for parallel file I/O during product loading.
    rasterio_profile : dict[str, Any] or None, optional
        rasterio profile dictionary for GeoTIFF output. If None, uses
        float32 as dtype, nan as nodata, pixel interleave,
        deflate compression, and 512x512 tiling.
    """

    def __init__(
        self,
        file_manager: FileManager,
        *,
        rasterio_profile: dict[str, Any] | None = None,
        n_workers: int = 8,
    ) -> None:
        self._fm = file_manager
        self._n_workers = n_workers

        # Overwrite profile defaults with user-specified profile
        base_profile = {"dtype": np.float32, "nodata": np.nan}
        self.rasterio_profile = build_rasterio_profile(base_profile, rasterio_profile)

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
        write_qa: Literal[False] = False,
        overwrite: bool = ...,
    ) -> RTCLRWComposite: ...
    @overload
    def compose_rtc_lrw(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        clip_max_resolution: float | None = ...,
        write_qa: Literal[True],
        overwrite: bool = ...,
    ) -> RTCLRWResult: ...
    def compose_rtc_lrw(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "median",
        clip_max_resolution: float | None = None,
        write_qa: bool = False,
        overwrite: bool = False,
    ) -> RTCLRWComposite | RTCLRWResult:
        """Compute a locally resolution weighted (LRW) RTC intensity composite.

        Each RTC file provides two bands: band 1 (intensity / power) and band 2
        (simulated illuminated area used as the inverse weight ``w = 1/A``).
        Bursts from the same orbit and acquisition date are mosaicked using the
        first non-NaN pixel strategy before temporal aggregation.

        Parameters
        ----------
        rtc_products : Collection[RTCProduct]
            RTC products to composite.  All products must share the same
            polarization.
        geobox : MGRSGeoBox
            Target MGRS grid cell defining the output CRS, spatial extent, and
            pixel grid.
        aggregation : {"mean", "median"} or float, default "median"
            Temporal aggregation applied after weighting.  Pass ``"mean"`` or
            ``"median"`` for the corresponding statistic, or a float in
            ``(0, 1)`` to compute a weighted quantile at that level.
        clip_max_resolution : float or None, default None
            Upper limit applied to the resolution weight ``1/area`` before
            further processing.  Suppresses extreme weights in radar shadow
            areas.
        write_qa : bool, default False
            If ``True``, also write two auxiliary QA products alongside the
            composite: a number-of-observations map (``QA-NUM``) and a composite
            quality map (``QA-CQM``).
        overwrite : bool, default False
            If ``False`` and the output file already exists, skip computation
            and return the existing product metadata.

        Returns
        -------
        RTCLRWComposite or tuple[RTCLRWComposite, tuple[RTCLRWQA, RTCLRWQA]]
            The composite product metadata, or ``(composite, (qa_num, qa_cqm))``
            when ``write_qa=True``.  GeoTIFFs are written to the paths returned
            by ``file_manager.get_product_path()``.

        Raises
        ------
        ValueError
            If ``rtc_products`` is empty.
        """
        rtc_products = list(rtc_products)
        if not rtc_products:
            msg = "rtc_products must not be empty."
            raise ValueError(msg)

        polarization = self._check_polarization(rtc_products)
        start_date = min(self._acquisition_date(p) for p in rtc_products)
        end_date = max(self._acquisition_date(p) for p in rtc_products)

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

        rtc_groups = self._stitch_by_orbit_date(self._load_band(rtc_products, geobox, band=1))
        sim_groups = self._stitch_by_orbit_date(self._load_band(rtc_products, geobox, band=2))

        self._check_sim_coverage(rtc_groups, sim_groups)
        rtc_obs = np.stack(list(rtc_groups.values()))  # (N, H, W)
        sim_obs = np.stack([sim_groups[key] for key in rtc_groups])  # (N, H, W)

        weights = self._calc_weights(sim_obs, clip_max_resolution)
        self._save(
            self._aggregate_weighted(rtc_obs, weights, aggregation),
            product,
            geobox,
            polarization,
            overwrite=overwrite,
        )
        if qa:
            self._write_lrw_qa(rtc_obs, sim_obs, weights, qa, geobox, overwrite=overwrite)
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
        overwrite: bool = ...,
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
        overwrite: bool = ...,
    ) -> CohLRWResult: ...
    def compose_coherence_lrw(
        self,
        coh_products: Collection[CoherenceProduct],
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "median",
        clip_max_resolution: float | None = None,
        write_qa: bool = False,
        overwrite: bool = False,
    ) -> CohLRWComposite | CohLRWResult:
        """Compute a locally resolution weighted (LRW) coherence composite.

        Coherence observations are weighted by the inverse simulated area
        ``1/A`` taken from band 2 of the matched RTC files.  The match is made
        per ``(rel_orbit_number, date_primary)`` group: for each coherence
        group the sim-area arrays are looked up from the corresponding RTC
        products (matched by ``burst_id``, ``sub_swath``, and
        ``date == date_primary``), then stitched in the same first-non-NaN
        fashion.

        Parameters
        ----------
        coh_products : Collection[CoherenceProduct]
            Coherence magnitude products to composite.  All must share the same
            polarization and ``baseline_days``.
        rtc_products : Collection[RTCProduct]
            RTC products providing the simulated area (band 2) used as weights.
            Must cover the same orbit/date combinations as ``coh_products``.
        geobox : MGRSGeoBox
            Target MGRS grid cell.
        aggregation : {"mean", "median"} or float, default "median"
            Temporal aggregation applied after weighting.  Pass ``"mean"`` or
            ``"median"`` for the corresponding statistic, or a float in
            ``(0, 1)`` to compute a weighted quantile at that level.
        clip_max_resolution : float or None, default None
            Upper limit applied to the resolution weight ``1/area`` before
            further processing.  Suppresses extreme weights in radar shadow
            areas.
        write_qa : bool, default False
            If ``True``, also write two auxiliary QA products alongside the
            composite: a number-of-observations map (``QA-NUM``) and a composite
            quality map (``QA-CQM``).
        overwrite : bool, default False
            Skip writing if the output file already exists.

        Returns
        -------
        CohLRWComposite or tuple[CohLRWComposite, tuple[CohLRWQA, CohLRWQA]]
            The composite product metadata, or ``(composite, (qa_num, qa_cqm))``
            when ``write_qa=True``.

        Raises
        ------
        ValueError
            If ``coh_products`` is empty.
        """
        coh_products = list(coh_products)
        if not coh_products:
            msg = "coh_products must not be empty."
            raise ValueError(msg)

        polarization = self._check_polarization(coh_products)
        baseline_days = coh_products[0].baseline_days
        start_date = min(self._acquisition_date(p) for p in coh_products)
        end_date = max(self._acquisition_date(p) for p in coh_products)

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

        coh_groups = self._stitch_by_orbit_date(self._load_band(coh_products, geobox, band=1))
        sim_groups = self._stitch_by_orbit_date(self._load_band(rtc_products, geobox, band=2))

        self._check_sim_coverage(coh_groups, sim_groups)
        coh_obs = np.stack(list(coh_groups.values()))  # (N, H, W)
        sim_obs = np.stack([sim_groups[key] for key in coh_groups])  # (N, H, W)

        weights = self._calc_weights(sim_obs, clip_max_resolution)
        self._save(
            self._aggregate_weighted(coh_obs, weights, aggregation),
            product,
            geobox,
            polarization,
            overwrite=overwrite,
        )
        if qa:
            self._write_lrw_qa(coh_obs, sim_obs, weights, qa, geobox, overwrite=overwrite)
        return (product, (qa[QaMetric.NUM], qa[QaMetric.CQM])) if qa else product

    @overload
    def compose_rtc_by_pass(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        write_qa: Literal[False] = False,
        overwrite: bool = ...,
    ) -> dict[OrbitDirection, RTCPassComposite]: ...
    @overload
    def compose_rtc_by_pass(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        write_qa: Literal[True],
        overwrite: bool = ...,
    ) -> dict[OrbitDirection, RTCPassResult]: ...
    def compose_rtc_by_pass(
        self,
        rtc_products: Collection[RTCProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "median",
        write_qa: bool = False,
        overwrite: bool = False,
    ) -> Mapping[OrbitDirection, RTCPassComposite | RTCPassResult]:
        """Compute unweighted RTC composites stratified by orbit direction.

        Bursts from the same orbit and acquisition date are mosaicked using the
        first non-NaN pixel strategy.  The resulting orbit-date observations are
        then aggregated temporally (mean or median) per orbit direction
        (ascending / descending).

        Parameters
        ----------
        rtc_products : Collection[RTCProduct]
            RTC products to composite.  Products from both orbit directions may
            be mixed; they are split automatically via the ESA burst map.
        geobox : MGRSGeoBox
            Target MGRS grid cell.
        aggregation : {"mean", "median"} or float, default "median"
            Temporal aggregation method.  Pass ``"mean"`` or ``"median"`` for
            the corresponding statistic, or a float in ``(0, 1)`` for an
            interpolated quantile at that level.
        write_qa : bool, default False
            If ``True``, also write a number-of-observations QA product
            (``QA-NUM``) per orbit direction.
        overwrite : bool, default False
            Skip writing if the output file already exists.

        Returns
        -------
        dict[OrbitDirection, RTCPassComposite or tuple[RTCPassComposite, RTCPassQA]]
            One entry per orbit direction.  Each value is the composite, or
            ``(composite, qa_num)`` when ``write_qa=True``.
        """
        rtc_products = list(rtc_products)
        if not rtc_products:
            return {}

        polarization = self._check_polarization(rtc_products)
        start_date = min(self._acquisition_date(p) for p in rtc_products)
        end_date = max(self._acquisition_date(p) for p in rtc_products)

        loaded = self._load_band(rtc_products, geobox, band=1)

        subsets: defaultdict[OrbitDirection, list[_LoadedEntry]] = defaultdict(list)
        for prod, data in loaded:
            subsets[prod.orbit_direction].append((prod, data))

        result: dict[OrbitDirection, RTCPassComposite | RTCPassResult] = {}
        for direction, subset in subsets.items():
            product = RTCPassComposite(
                mgrs_code=geobox.mgrs_code,
                start_date=start_date,
                end_date=end_date,
                polarization=polarization,
                orbit_direction=direction,
            )
            qa_num: list[RTCPassQA] = (
                [
                    RTCPassQA(
                        mgrs_code=geobox.mgrs_code,
                        start_date=start_date,
                        end_date=end_date,
                        polarization=polarization,
                        orbit_direction=direction,
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
            rtc_obs = np.stack(list(groups.values()))  # (N, H, W)
            self._save(
                self._aggregate_unweighted(rtc_obs, aggregation),
                product,
                geobox,
                polarization,
                overwrite=overwrite,
            )
            for qa_product in qa_num:
                self._save(
                    self._count_observations(rtc_obs),
                    qa_product,
                    geobox,
                    "num_obs",
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
    ) -> dict[OrbitDirection, CohPassComposite]: ...
    @overload
    def compose_coherence_by_pass(
        self,
        coh_products: Collection[CoherenceProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = ...,
        write_qa: Literal[True],
        overwrite: bool = ...,
    ) -> dict[OrbitDirection, CohPassResult]: ...
    def compose_coherence_by_pass(
        self,
        coh_products: Collection[CoherenceProduct],
        geobox: MGRSGeoBox,
        *,
        aggregation: Literal["mean", "median"] | float = "median",
        write_qa: bool = False,
        overwrite: bool = False,
    ) -> Mapping[OrbitDirection, CohPassComposite | CohPassResult]:
        """Compute unweighted coherence composites stratified by orbit direction.

        Bursts from the same orbit and primary acquisition date are mosaicked
        using the first non-NaN pixel strategy.  The resulting orbit-date
        observations are then aggregated temporally (mean or median) per orbit
        direction (ascending / descending).

        Parameters
        ----------
        coh_products : Collection[CoherenceProduct]
            Coherence magnitude products to composite.  All must share the same
            ``baseline_days``.
        geobox : MGRSGeoBox
            Target MGRS grid cell.
        aggregation : {"mean", "median"} or float, default "median"
            Temporal aggregation method.  Pass ``"mean"`` or ``"median"`` for
            the corresponding statistic, or a float in ``(0, 1)`` for an
            interpolated quantile at that level.
        write_qa : bool, default False
            If ``True``, also write a number-of-observations QA product
            (``QA-NUM``) per orbit direction.
        overwrite : bool, default False
            Skip writing if the output file already exists.

        Returns
        -------
        dict[OrbitDirection, CohPassComposite or tuple[CohPassComposite, CohPassQA]]
            One entry per orbit direction.  Each value is the composite, or
            ``(composite, qa_num)`` when ``write_qa=True``.
        """
        coh_products = list(coh_products)
        if not coh_products:
            return {}

        polarization = self._check_polarization(coh_products)
        baseline_days = coh_products[0].baseline_days
        start_date = min(self._acquisition_date(p) for p in coh_products)
        end_date = max(self._acquisition_date(p) for p in coh_products)

        loaded = self._load_band(coh_products, geobox, band=1)

        subsets: defaultdict[OrbitDirection, list[_LoadedEntry]] = defaultdict(list)
        for prod, data in loaded:
            subsets[prod.orbit_direction].append((prod, data))

        result: dict[OrbitDirection, CohPassComposite | CohPassResult] = {}
        for direction, subset in subsets.items():
            product = CohPassComposite(
                mgrs_code=geobox.mgrs_code,
                start_date=start_date,
                end_date=end_date,
                polarization=polarization,
                orbit_direction=direction,
                baseline_days=baseline_days,
            )
            qa_num: list[CohPassQA] = (
                [
                    CohPassQA(
                        mgrs_code=geobox.mgrs_code,
                        start_date=start_date,
                        end_date=end_date,
                        polarization=polarization,
                        orbit_direction=direction,
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
            self._save(
                self._aggregate_unweighted(coh_obs, aggregation),
                product,
                geobox,
                polarization,
                overwrite=overwrite,
            )
            for qa_product in qa_num:
                self._save(
                    self._count_observations(coh_obs),
                    qa_product,
                    geobox,
                    "num_obs",
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
        bounds = tuple(geobox.boundingbox)

        def _load_one(prod: ComposableProduct) -> _LoadedEntry:
            path = self._fm.get_product_path(prod)
            if not path.exists():
                msg = f"product file not found: {path}"
                raise FileNotFoundError(msg)
            with rasterio_open(path) as src:
                win = from_bounds(*bounds, src.transform).round_lengths().round_offsets()
                data = src.read(band, window=win, boundless=True, fill_value=np.nan)
            return prod, data.astype(np.float32)

        with ThreadPoolExecutor(max_workers=self._n_workers) as executor:
            return list(executor.map(_load_one, products))

    def _save(
        self,
        data: NDArray,
        product: CompositeProduct | QAProduct,
        geobox: MGRSGeoBox,
        band_name: str,
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
        profile = self.rasterio_profile.copy()
        profile.update(
            count=1,
            dtype=dtype,
            nodata=nodata,
            crs=geobox.crs,
            transform=geobox.transform,
            height=geobox.height,
            width=geobox.width,
        )
        with rasterio_open(path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, band_name)
        logger.debug("saved %s", path.name)

    def _write_lrw_qa(
        self,
        values: NDArray[np.floating[Any]],
        sim_area: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
        qa: Mapping[QaMetric, LRWQAProduct],
        geobox: MGRSGeoBox,
        *,
        overwrite: bool,
    ) -> None:
        """Compute and write the number-of-observations and composite quality map QA products."""
        self._save(
            self._count_observations(values),
            qa[QaMetric.NUM],
            geobox,
            "num_obs",
            dtype=np.uint16,
            nodata=0,
            overwrite=overwrite,
        )
        self._save(
            self._calc_cqm(values, sim_area, weights),
            qa[QaMetric.CQM],
            geobox,
            "cqm",
            dtype=np.float32,
            nodata=np.nan,
            overwrite=overwrite,
        )

    # ------------------------------------------------------------------
    # Private helpers — spatial mosaicking
    # ------------------------------------------------------------------

    @staticmethod
    def _acquisition_date(product: ComposableProduct) -> Timestamp:
        """Return the acquisition date used for orbit-date grouping.

        ``CoherenceProduct`` uses ``date_primary``; ``RTCProduct`` uses ``date``.
        """
        return product.date_primary if isinstance(product, CoherenceProduct) else product.date

    def _stitch_by_orbit_date(
        self,
        loaded: Sequence[_LoadedEntry],
    ) -> dict[_OrbitDateKey, NDArray]:
        """Group loaded arrays by orbit/date and mosaic within each group.

        The grouping key is ``(rel_orbit_number, acquisition_date)``, where
        ``acquisition_date`` is ``date_primary`` for coherence products and
        ``date`` for RTC products.  Within each group the arrays are combined
        using pixel-wise first non-NaN selection, which avoids averaging along
        burst seams.
        """
        groups: defaultdict[_OrbitDateKey, list[NDArray]] = defaultdict(list)
        for prod, data in loaded:
            key: _OrbitDateKey = (
                prod.rel_orbit_number,
                self._acquisition_date(prod).normalize(),
            )
            groups[key].append(data)
        return {key: self._first_notna(arrays) for key, arrays in sorted(groups.items())}

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
    def _check_sim_coverage(
        obs_groups: dict[_OrbitDateKey, NDArray],
        sim_groups: dict[_OrbitDateKey, NDArray],
    ) -> None:
        """Verify every observation group has a matching simulated-area group.

        Raises
        ------
        ValueError
            If any ``(rel_orbit_number, date)`` group present in the
            observations has no corresponding RTC simulated-area group.
        """
        missing = [key for key in obs_groups if key not in sim_groups]
        if missing:
            formatted = ", ".join(f"(orbit {orbit}, {date.date()})" for orbit, date in missing)
            msg = f"No matching RTC simulated area for orbit/date group(s): {formatted}"
            raise ValueError(msg)

    @staticmethod
    def _calc_weights(
        sim_area: NDArray[np.floating[Any]],
        clip_max_resolution: float | None,
    ) -> NDArray[np.floating[Any]]:
        """Compute resolution weights from simulated illuminated area.

        The resolution weight is ``w = 1 / sim_area``.  Invalid pixels
        (``sim_area <= 0`` or non-finite) receive weight zero.  An optional
        upper clip on the resolution weight ``1/area`` suppresses extreme
        weights in shadow areas.
        """
        is_valid = (sim_area > 0) & np.isfinite(sim_area)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(is_valid, 1.0 / sim_area, 0.0)
        if clip_max_resolution is not None:
            weights = weights.clip(max=clip_max_resolution)
        return weights

    @staticmethod
    def _count_observations(values: NDArray[np.floating[Any]]) -> NDArray[np.uint16]:
        """Per-pixel count of finite observations over the first (N) axis."""
        return np.count_nonzero(np.isfinite(values), axis=0).astype(np.uint16)

    @staticmethod
    def _calc_cqm(
        values: NDArray[np.floating[Any]],
        sim_area: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
    ) -> NDArray[np.float32]:
        """Composite quality map (CQM) over the first (N) axis.

        Computed as ``1 / Σ(wᵢ · Aᵢ)`` where ``wᵢ`` are the resolution weights
        normalized to sum to one and ``Aᵢ`` is the contributing area.  Uses the
        same (clipped) weights as the composite.
        """
        weight_sum = weights.sum(axis=0)  # (H, W)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.where(weight_sum[np.newaxis] > 0, weights / weight_sum[np.newaxis], 0.0)
            weighted_area = 1.0 / np.nansum(sim_area * norm, axis=0)
        cqm = weighted_area.astype(np.float32)
        invalid = (weight_sum == 0) | ~np.isfinite(values).any(axis=0)
        cqm[invalid] = np.nan
        return cqm

    # ------------------------------------------------------------------
    # Private helpers — aggregation
    # ------------------------------------------------------------------

    def _aggregate_weighted(
        self,
        values: NDArray[np.floating[Any]],
        weights: NDArray[np.floating[Any]],
        aggregation: Literal["mean", "median"] | float,
    ) -> NDArray[np.floating[Any]]:
        """Dispatch to the aggregation function.

        Parameters
        ----------
        values : NDArray[np.floating[Any]]
            Shape ``(N, H, W)`` observations.
        weights : NDArray[np.floating[Any]]
            Shape ``(N, H, W)`` non-negative resolution weights.
        aggregation : {"mean", "median"} or float
            Determines which aggregation function is called.

        Returns
        -------
        NDArray[np.floating[Any]]
            Shape ``(H, W)`` composite.
        """
        if aggregation == "mean":
            return self._weighted_mean(values, weights)
        if aggregation == "median":
            return self._weighted_quantile(values, weights, q=0.5)
        if isinstance(aggregation, float):
            return self._weighted_quantile(values, weights, q=aggregation)
        msg = f"Unknown aggregation: {aggregation!r}"
        raise ValueError(msg)

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

        NaN values are excluded by zeroing their weights.  Each value is placed
        at the plotting position ``(cumulative_weight - 0.5 * weight) / total``
        and the target quantile ``q`` is linearly interpolated between the two
        bracketing order statistics. Pixels whose total weight is zero return NaN.
        """
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

        Equivalent to :meth:`_aggregate_weighted` with uniform weights, so the
        mean and the interpolated ``median`` / quantile use the same definition
        as the LRW composites.
        """
        return self._aggregate_weighted(values, np.ones_like(values), aggregation)
