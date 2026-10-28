"""High-level batch processing for Sentinel-1 bursts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import logging
import time
from typing import TYPE_CHECKING, overload

from freezebase.progress import create_progress

from s1bursts.product import _utm_zone_crs
from s1bursts.snap_client import SNAPClient, SNAPConfig
from s1bursts.snap_graph import (
    CoherenceConfig,
    GTCConfig,
    InterferogramConfig,
    RTCConfig,
)
from s1bursts.summary import ErrorResult, OperationResult, OperationSummary

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from s1bursts.burst import Burst, BurstPair
    from s1bursts.filemanager import FileManager
    from s1bursts.product import (
        CoherenceProduct,
        CRSInput,
        GTCProduct,
        InterferogramProduct,
        RTCProduct,
    )

logger = logging.getLogger(__name__)


class BurstProcessor:
    """Batch processor for Sentinel-1 bursts and burst pairs.

    Runs one SNAP GPT job per burst or pair with progress tracking, skips
    products that already exist, records failures per item, and returns an
    ``OperationSummary``. Takes bursts and burst pairs, as collections or
    lists, rather than product classes.

    Parameters
    ----------
    file_manager
        File manager resolving input SLC and output product paths.
    config
        Runtime configuration for SNAP GPT processing; defaults to
        ``SNAPConfig()``.
    show_progress
        Whether to show progress bars.
    continue_on_error
        Whether to continue processing after an error. ``True``, the default,
        attempts every item and records each failure in the summary, so one bad
        burst does not end a long job; ``False`` raises at the first error.
        Follow a run with ``OperationSummary.raise_on_failure()`` to fail
        once all items have been attempted.
    """

    def __init__(
        self,
        file_manager: FileManager,
        config: SNAPConfig | None = None,
        *,
        show_progress: bool = True,
        continue_on_error: bool = True,
    ) -> None:
        self.config = config or SNAPConfig()
        self.snap_client = SNAPClient(file_manager, self.config)
        self.manager = file_manager
        self.show_progress = show_progress
        self.continue_on_error = continue_on_error

    @overload
    def create_backscatter(
        self,
        bursts: Iterable[Burst],
        crs: CRSInput,
        *,
        config: GTCConfig,
        overwrite: bool = False,
    ) -> OperationSummary[GTCProduct]: ...

    @overload
    def create_backscatter(
        self,
        bursts: Iterable[Burst],
        crs: CRSInput,
        *,
        config: RTCConfig | None = None,
        overwrite: bool = False,
    ) -> OperationSummary[RTCProduct]: ...

    def create_backscatter(
        self,
        bursts: Iterable[Burst],
        crs: CRSInput,
        *,
        config: RTCConfig | GTCConfig | None = None,
        overwrite: bool = False,
    ) -> OperationSummary[RTCProduct] | OperationSummary[GTCProduct]:
        """Create backscatter products from bursts, terrain-flattened (RTC) or geocoded (GTC).

        The type of ``config`` selects the product. ``RTCConfig``, the default,
        gives terrain-flattened gamma-0 backscatter (``RTCProduct``), called
        Normalised Radar Backscatter (NRB) in CEOS-ARD and RTC-S1 by OPERA.
        ``GTCConfig`` gives beta-0 backscatter that is geocoded only, not
        terrain-flattened (``GTCProduct``).

        Parameters
        ----------
        bursts
            Bursts to process, such as a ``BurstCollection`` or a list of
            ``Burst`` objects.
        crs
            The coordinate reference system of the geocoded output, a WGS 84 /
            UTM zone, north or south, which product file names record. Given as
            a CRS object, an authority string such as ``"EPSG:32632"`` or an
            EPSG code. For composites, pass the target cell's ``crs``; for
            single bursts, ``utm_zone_of(burst)`` gives the zone of a burst.
        config
            Graph-shape configuration, ``RTCConfig`` or ``GTCConfig``; defaults
            to ``RTCConfig()`` (Copernicus 30m DEM, multilook ``(5, 1)``).
        overwrite
            Whether to overwrite existing products.

        Returns
        -------
        OperationSummary
            Summary of processing results, holding ``RTCProduct`` or
            ``GTCProduct`` as ``config`` selects.

        Raises
        ------
        TypeError
            If ``config`` is neither an ``RTCConfig`` nor a ``GTCConfig``.
        ValueError
            If ``crs`` is not a WGS 84 / UTM zone. Raised before any item is
            processed.
        """
        # Checked once, so an invalid CRS fails before any burst is processed.
        utm_crs = _utm_zone_crs(crs)
        if config is None or isinstance(config, RTCConfig):
            rtc_config = config or RTCConfig()

            def create_rtc(burst: Burst) -> OperationResult[RTCProduct]:
                return self.snap_client.create_backscatter(
                    burst, utm_crs, config=rtc_config, overwrite=overwrite
                )

            return self._process_collection(bursts, create_rtc, item_description="RTC")
        if isinstance(config, GTCConfig):
            gtc_config = config

            def create_gtc(burst: Burst) -> OperationResult[GTCProduct]:
                return self.snap_client.create_backscatter(
                    burst, utm_crs, config=gtc_config, overwrite=overwrite
                )

            return self._process_collection(bursts, create_gtc, item_description="GTC")
        msg = f"`config` must be an RTCConfig or a GTCConfig, received '{type(config).__name__}'."
        raise TypeError(msg)

    def create_coherences(
        self,
        pairs: Iterable[BurstPair],
        crs: CRSInput,
        *,
        config: CoherenceConfig | None = None,
        overwrite: bool = False,
    ) -> OperationSummary[CoherenceProduct]:
        """Create coherence products from burst pairs.

        Parameters
        ----------
        pairs
            Burst pairs to process, such as a ``BurstPairCollection`` or a
            list of ``BurstPair`` objects.
        crs
            The coordinate reference system of the geocoded output, a WGS 84 /
            UTM zone, north or south, which product file names record. Given as
            a CRS object, an authority string such as ``"EPSG:32632"`` or an
            EPSG code. For composites, pass the target cell's ``crs``; for
            single bursts, ``utm_zone_of(burst)`` gives the zone of a burst.
        config
            Graph-shape configuration; defaults to ``CoherenceConfig()``
            (25x5 window, Copernicus 30m DEM).
        overwrite
            Whether to overwrite existing products.

        Returns
        -------
        OperationSummary
            Summary of processing results.

        Raises
        ------
        ValueError
            If ``crs`` is not a WGS 84 / UTM zone. Raised before any item is
            processed.
        """
        cfg = config or CoherenceConfig()
        return self._process_collection(
            pairs,
            partial(
                self.snap_client.create_coherence,
                crs=_utm_zone_crs(crs),
                config=cfg,
                overwrite=overwrite,
            ),
            item_description="Coherence",
        )

    def create_interferograms(
        self,
        pairs: Iterable[BurstPair],
        crs: CRSInput,
        *,
        config: InterferogramConfig | None = None,
        overwrite: bool = False,
    ) -> OperationSummary[InterferogramProduct]:
        """Create interferogram products from burst pairs.

        Parameters
        ----------
        pairs
            Burst pairs to process, such as a ``BurstPairCollection`` or a
            list of ``BurstPair`` objects.
        crs
            The coordinate reference system of the geocoded output, a WGS 84 /
            UTM zone, north or south, which product file names record. Given as
            a CRS object, an authority string such as ``"EPSG:32632"`` or an
            EPSG code. For composites, pass the target cell's ``crs``; for
            single bursts, ``utm_zone_of(burst)`` gives the zone of a burst.
        config
            Graph-shape configuration; defaults to ``InterferogramConfig()``.
        overwrite
            Whether to overwrite existing products.

        Returns
        -------
        OperationSummary
            Summary of processing results.

        Raises
        ------
        ValueError
            If ``crs`` is not a WGS 84 / UTM zone. Raised before any item is
            processed.
        """
        cfg = config or InterferogramConfig()
        return self._process_collection(
            pairs,
            partial(
                self.snap_client.create_interferogram,
                crs=_utm_zone_crs(crs),
                config=cfg,
                overwrite=overwrite,
            ),
            item_description="Interferogram",
        )

    def _process_collection[I, T](
        self,
        collection: Iterable[I],
        process_fn: Callable[[I], OperationResult[T]],
        item_description: str,
    ) -> OperationSummary[T]:
        """Process bursts with progress tracking."""
        start_time = time.time()
        results: list[OperationResult[T]] = []
        items = list(collection)
        n_workers = self.config.resolved_max_workers

        progress = create_progress(show_progress=self.show_progress)
        task_id = progress.add_task(
            f"Processing {item_description.lower()}",
            total=len(items),
        )
        with progress:
            if n_workers <= 1:
                for item in items:
                    result = process_fn(item)
                    if isinstance(result, ErrorResult) and not self.continue_on_error:
                        raise result.error
                    results.append(result)
                    progress.advance(task_id)
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(process_fn, item) for item in items]
                    for future in as_completed(futures):
                        result = future.result()
                        if isinstance(result, ErrorResult) and not self.continue_on_error:
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise result.error
                        progress.advance(task_id)
                # Submission order, so the summary does not depend on thread timing.
                results.extend(future.result() for future in futures)
            progress.update(task_id, refresh=True)

        return OperationSummary.from_results(
            results,
            start_time=start_time,
            end_time=time.time(),
        )
