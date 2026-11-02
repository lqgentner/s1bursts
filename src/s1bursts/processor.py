"""High-level batch processing for Sentinel-1 bursts."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import logging
import time
from typing import TYPE_CHECKING, Literal, overload

from freezebase.progress import create_progress

from s1bursts.errors import NoUTMZoneError, ProductSettingsMismatchError
from s1bursts.product import (
    CoherenceProduct,
    GTCProduct,
    InterferogramProduct,
    RTCProduct,
    _check_output_crs,
)
from s1bursts.product_tags import _software_version
from s1bursts.settings_check import _skip_existing, _warn_other_snap, _warn_unchecked
from s1bursts.snap_client import SNAPClient, SNAPConfig, _item_failure
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
    from s1bursts.product import CRSInput

logger = logging.getLogger(__name__)


class BurstProcessor:
    """Batch processor for Sentinel-1 bursts and burst pairs.

    Runs one SNAP GPT job per burst or pair with progress tracking, skips
    products that already exist with the requested settings, records failures
    per item, and returns an ``OperationSummary``. Takes bursts and burst
    pairs, as collections or lists, rather than product classes.

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
        burst does not end a long job; ``False`` raises at the first error,
        once the items already running have finished, without starting the
        queued ones.
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
        crs: CRSInput | Literal["auto"],
        *,
        config: GTCConfig,
        overwrite: bool = False,
    ) -> OperationSummary[GTCProduct]: ...

    @overload
    def create_backscatter(
        self,
        bursts: Iterable[Burst],
        crs: CRSInput | Literal["auto"],
        *,
        config: RTCConfig | None = None,
        overwrite: bool = False,
    ) -> OperationSummary[RTCProduct]: ...

    def create_backscatter(
        self,
        bursts: Iterable[Burst],
        crs: CRSInput | Literal["auto"],
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
            EPSG code, or ``"auto"`` to process each burst in the UTM zone
            that holds its footprint (``utm_zone``). Use ``"auto"`` for
            bursts processed on their own, such as a time series, and the
            target cell's ``crs`` for products meant for a composite: an input
            in another zone than the cell raises ``InconsistentInputsError``
            when composited.
        config
            Graph-shape configuration, ``RTCConfig`` or ``GTCConfig``; defaults
            to ``RTCConfig()`` (Copernicus 30m DEM, multilook ``(5, 1)``).
        overwrite
            Whether to overwrite existing products. Without it, every existing
            product is checked against ``config`` before any item is
            processed, and skipped if its recorded settings match.

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
            If ``crs`` is not a WGS 84 / UTM zone, raised before any item is
            processed. Under ``"auto"``, a footprint outside the UTM zones
            fails its item alone, recorded as a ``NoUTMZoneError``.
        BurstMapUnavailableError
            With ``"auto"``, if an item carries no footprint and the burst
            map has not been downloaded, which every such item would meet.
        OSError, SnapProcessingError or RasterWriteError
            If the disk is full, over quota or read-only, which would fail
            every remaining item alike: an ``OSError`` saying so, or an error
            caused by one or whose message or GPT output says so.
        ProductSettingsMismatchError
            If a product exists with other settings than ``config`` and
            ``overwrite`` is ``False``. Every product existing when the call
            starts is checked before any item is processed, so for those it is
            raised before any GPT run; a product written by another process
            meanwhile is checked when its item is reached.
        OSError or RasterioError
            If a product exists, ``overwrite`` is ``False``, and it cannot be
            read to check its settings; checked up front in the same way.

        Warns
        -----
        UserWarning
            Once per call, counting the existing products skipped without
            checking their settings: written before s1bursts 0.3.0, or made
            with an external DEM without ``dem_name``. Once more if another
            SNAP version made some of them. The summary counts both, and each
            skipped result records why.
        """
        # Checked once, so an invalid CRS fails before any burst is processed.
        utm_crs = _check_output_crs(crs)
        client = self.snap_client
        if config is None or isinstance(config, RTCConfig):
            rtc_config = config or RTCConfig()
            rtc_summary = self._run(
                bursts,
                lambda burst: RTCProduct.from_burst(burst, utm_crs),
                lambda burst: client.create_backscatter(
                    burst, utm_crs, config=rtc_config, overwrite=overwrite
                ),
                config=rtc_config,
                overwrite=overwrite,
                item_description="RTC",
            )
            self._warn_about_skips(rtc_summary)
            return rtc_summary
        if isinstance(config, GTCConfig):
            gtc_config = config
            gtc_summary = self._run(
                bursts,
                lambda burst: GTCProduct.from_burst(burst, utm_crs),
                lambda burst: client.create_backscatter(
                    burst, utm_crs, config=gtc_config, overwrite=overwrite
                ),
                config=gtc_config,
                overwrite=overwrite,
                item_description="GTC",
            )
            self._warn_about_skips(gtc_summary)
            return gtc_summary
        msg = f"`config` must be an RTCConfig or a GTCConfig, received '{type(config).__name__}'."
        raise TypeError(msg)

    def create_coherences(
        self,
        pairs: Iterable[BurstPair],
        crs: CRSInput | Literal["auto"],
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
            EPSG code, or ``"auto"`` to process each pair in the UTM zone
            that holds its footprint (``utm_zone``). Use ``"auto"`` for
            pairs processed on their own, such as a time series, and the
            target cell's ``crs`` for products meant for a composite: an input
            in another zone than the cell raises ``InconsistentInputsError``
            when composited.
        config
            Graph-shape configuration; defaults to ``CoherenceConfig()``
            (25x5 window, Copernicus 30m DEM).
        overwrite
            Whether to overwrite existing products. Without it, every existing
            product is checked against ``config`` before any item is
            processed, and skipped if its recorded settings match.

        Returns
        -------
        OperationSummary
            Summary of processing results.

        Raises
        ------
        ValueError
            If ``crs`` is not a WGS 84 / UTM zone, raised before any item is
            processed. Under ``"auto"``, a footprint outside the UTM zones
            fails its item alone, recorded as a ``NoUTMZoneError``.
        BurstMapUnavailableError
            With ``"auto"``, if an item carries no footprint and the burst
            map has not been downloaded, which every such item would meet.
        OSError, SnapProcessingError or RasterWriteError
            If the disk is full, over quota or read-only, which would fail
            every remaining item alike: an ``OSError`` saying so, or an error
            caused by one or whose message or GPT output says so.
        ProductSettingsMismatchError
            If a product exists with other settings than ``config`` and
            ``overwrite`` is ``False``. Every product existing when the call
            starts is checked before any item is processed, so for those it is
            raised before any GPT run; a product written by another process
            meanwhile is checked when its item is reached.
        OSError or RasterioError
            If a product exists, ``overwrite`` is ``False``, and it cannot be
            read to check its settings; checked up front in the same way.

        Warns
        -----
        UserWarning
            Once per call, counting the existing products skipped without
            checking their settings: written before s1bursts 0.3.0, or made
            with an external DEM without ``dem_name``. Once more if another
            SNAP version made some of them. The summary counts both, and each
            skipped result records why.
        """
        cfg = config or CoherenceConfig()
        utm_crs = _check_output_crs(crs)
        client = self.snap_client
        summary = self._run(
            pairs,
            lambda pair: CoherenceProduct.from_burst_pair(pair, utm_crs),
            lambda pair: client.create_coherence(pair, utm_crs, config=cfg, overwrite=overwrite),
            config=cfg,
            overwrite=overwrite,
            item_description="Coherence",
        )
        self._warn_about_skips(summary)
        return summary

    def create_interferograms(
        self,
        pairs: Iterable[BurstPair],
        crs: CRSInput | Literal["auto"],
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
            EPSG code, or ``"auto"`` to process each pair in the UTM zone
            that holds its footprint (``utm_zone``). Use ``"auto"`` for
            pairs processed on their own, such as a time series, and the
            target cell's ``crs`` for products meant for a composite: an input
            in another zone than the cell raises ``InconsistentInputsError``
            when composited.
        config
            Graph-shape configuration; defaults to ``InterferogramConfig()``.
        overwrite
            Whether to overwrite existing products. Without it, every existing
            product is checked against ``config`` before any item is
            processed, and skipped if its recorded settings match.

        Returns
        -------
        OperationSummary
            Summary of processing results.

        Raises
        ------
        ValueError
            If ``crs`` is not a WGS 84 / UTM zone, raised before any item is
            processed. Under ``"auto"``, a footprint outside the UTM zones
            fails its item alone, recorded as a ``NoUTMZoneError``.
        BurstMapUnavailableError
            With ``"auto"``, if an item carries no footprint and the burst
            map has not been downloaded, which every such item would meet.
        OSError, SnapProcessingError or RasterWriteError
            If the disk is full, over quota or read-only, which would fail
            every remaining item alike: an ``OSError`` saying so, or an error
            caused by one or whose message or GPT output says so.
        ProductSettingsMismatchError
            If a product exists with other settings than ``config`` and
            ``overwrite`` is ``False``. Every product existing when the call
            starts is checked before any item is processed, so for those it is
            raised before any GPT run; a product written by another process
            meanwhile is checked when its item is reached.
        OSError or RasterioError
            If a product exists, ``overwrite`` is ``False``, and it cannot be
            read to check its settings; checked up front in the same way.

        Warns
        -----
        UserWarning
            Once per call, counting the existing products skipped without
            checking their settings: written before s1bursts 0.3.0, or made
            with an external DEM without ``dem_name``. Once more if another
            SNAP version made some of them. The summary counts both, and each
            skipped result records why.
        """
        cfg = config or InterferogramConfig()
        utm_crs = _check_output_crs(crs)
        client = self.snap_client
        summary = self._run(
            pairs,
            lambda pair: InterferogramProduct.from_burst_pair(pair, utm_crs),
            lambda pair: client.create_interferogram(
                pair, utm_crs, config=cfg, overwrite=overwrite
            ),
            config=cfg,
            overwrite=overwrite,
            item_description="Interferogram",
        )
        self._warn_about_skips(summary)
        return summary

    def _run[I, P: RTCProduct | GTCProduct | CoherenceProduct | InterferogramProduct](
        self,
        items: Iterable[I],
        product_of: Callable[[I], P],
        create: Callable[[I], OperationResult[P]],
        *,
        config: RTCConfig | GTCConfig | CoherenceConfig | InterferogramConfig,
        overwrite: bool,
        item_description: str,
    ) -> OperationSummary[P]:
        """Check every existing product, then create only the missing ones."""
        start_time = time.time()
        items = list(items)
        existing = [None] * len(items) if overwrite else self._existing(items, product_of, config)

        def process(entry: tuple[I, OperationResult[P] | None]) -> OperationResult[P]:
            item, skipped = entry
            return skipped if skipped is not None else create(item)

        return self._process_collection(
            list(zip(items, existing, strict=True)),
            process,
            item_description,
            start_time=start_time,
        )

    def _existing[I, P: RTCProduct | GTCProduct | CoherenceProduct | InterferogramProduct](
        self,
        items: list[I],
        product_of: Callable[[I], P],
        config: RTCConfig | GTCConfig | CoherenceConfig | InterferogramConfig,
    ) -> list[OperationResult[P] | None]:
        """Return each item's skipped result if its product exists, before any item runs.

        An item without a UTM zone under ``crs="auto"`` gets its failure
        instead, so it is recorded without being handed to the client.

        Raises on the first product whose settings differ or that cannot be
        read, so a batch with changed settings writes nothing.
        """
        snap_version = self.config.resolved_snap_version

        def check(item: I) -> OperationResult[P] | None:
            try:
                product = product_of(item)
            except NoUTMZoneError as err:
                # Under "auto", this item alone has no zone; it fails without being run.
                return _item_failure(item, err)
            # The batch warns once about these skips, so each is logged at DEBUG only.
            return _skip_existing(
                self.manager,
                item,
                product,
                config,
                snap_version=snap_version,
                log_level=logging.DEBUG,
            )

        max_workers = self.config.resolved_max_workers
        if max_workers <= 1 or len(items) <= 1:
            return [check(item) for item in items]
        # Each check is a request on S3, so they run in parallel.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                return list(executor.map(check, items))
            except BaseException:
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    def _warn_about_skips[T](self, summary: OperationSummary[T]) -> None:
        """Warn once about the skips the summary counts as unchecked or of another SNAP."""
        # Points at the caller of the public create_* method.
        _warn_unchecked(summary.n_unchecked, stacklevel=3)
        _warn_other_snap(
            summary.n_other_snap_version,
            _software_version(self.config.resolved_snap_version),
            stacklevel=3,
        )

    def _raise_unless_continuing[T](self, result: OperationResult[T]) -> None:
        """Raise the error of a failed item unless the batch continues on error."""
        if isinstance(result, ErrorResult) and not self.continue_on_error:
            raise result.error

    def _process_collection[I, T](
        self,
        collection: Iterable[I],
        process_fn: Callable[[I], OperationResult[T]],
        item_description: str,
        *,
        start_time: float | None = None,
    ) -> OperationSummary[T]:
        """Process bursts with progress tracking, timed from ``start_time`` if given."""
        start_time = time.time() if start_time is None else start_time
        results: list[OperationResult[T]] = []
        items = list(collection)
        max_workers = self.config.resolved_max_workers

        progress = create_progress(show_progress=self.show_progress)
        task_id = progress.add_task(
            f"Processing {item_description.lower()}",
            total=len(items),
        )
        futures: list[Future[OperationResult[T]]] = []
        try:
            with progress:
                if max_workers <= 1:
                    for item in items:
                        result = process_fn(item)
                        results.append(result)
                        self._raise_unless_continuing(result)
                        progress.advance(task_id)
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submitting inside the try, so that an interrupt while the
                        # queue is filled also cancels what was queued.
                        try:
                            # extend keeps the futures submitted before an interrupt.
                            futures.extend(executor.submit(process_fn, item) for item in items)
                            for future in as_completed(futures):
                                result = future.result()
                                self._raise_unless_continuing(result)
                                progress.advance(task_id)
                        except BaseException:
                            # Leaving the block joins the workers, which would run
                            # the whole queue first; only the running items finish.
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
                    # Submission order, so the summary does not depend on thread timing.
                    results.extend(future.result() for future in futures)
                progress.update(task_id, refresh=True)
        except BaseException as err:
            # Counted once the running items have finished, so it includes them.
            n_finished = (
                sum(future.done() and not future.cancelled() for future in futures)
                if futures
                else len(results)
            )
            note = (
                f"Stopped with {n_finished} of {len(items)} items finished; "
                "products already written are kept"
            )
            # A rerun would meet the same existing product again.
            if not isinstance(err, ProductSettingsMismatchError):
                note += ", and a rerun without overwrite skips them"
            err.add_note(f"{note}.")
            raise

        return OperationSummary.from_results(
            results,
            start_time=start_time,
            end_time=time.time(),
        )
