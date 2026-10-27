"""High-level batch processing for Sentinel-1 bursts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import logging
import time
from typing import TYPE_CHECKING, Any, overload

from freezebase.progress import create_progress

from s1bursts.snap_client import SNAPClient, SNAPConfig
from s1bursts.snap_graph import (
    BackscatterConfig,
    CoherenceConfig,
    InterferogramConfig,
)
from s1bursts.summary import ErrorResult, OperationResult, OperationSummary

if TYPE_CHECKING:
    from collections.abc import Callable

    from s1bursts.burst import (
        Burst,
        BurstCollection,
        BurstPair,
        BurstPairCollection,
    )
    from s1bursts.filemanager import FileManager
    from s1bursts.product import (
        CoherenceProduct,
        CRSLike,
        GTCProduct,
        InterferogramProduct,
        RTCProduct,
    )

logger = logging.getLogger(__name__)


class BurstProcessor:
    """Batch processor for Sentinel-1 bursts and burst pairs.

    Runs one SNAP GPT job per burst or pair with progress tracking, skips
    products that already exist, records failures per item, and returns an
    ``OperationSummary``. Takes ``BurstCollection`` and
    ``BurstPairCollection`` rather than product classes.

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
        Whether to continue processing after an error.
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

    def create_backscatter(
        self,
        bursts: BurstCollection,
        crs: CRSLike,
        *,
        config: BackscatterConfig | None = None,
        overwrite: bool = False,
    ) -> OperationSummary[RTCProduct | GTCProduct]:
        """Create backscatter products from burst collection.

        Parameters
        ----------
        bursts
            Collection of bursts to process.
        crs
            The coordinate reference system of the geocoded output.
        config
            Graph-shape configuration; defaults to ``BackscatterConfig()``
            (RTC, Copernicus 30m DEM, multilook ``(5, 1)``).
        overwrite
            Whether to overwrite existing products.

        Returns
        -------
        OperationSummary
            Summary of processing results.
        """
        cfg = config or BackscatterConfig()
        return self._process_collection(
            bursts,
            partial(
                self.snap_client.create_backscatter,
                crs=crs,
                config=cfg,
                overwrite=overwrite,
            ),
            item_description="RTC" if cfg.flatten_terrain else "GTC",
        )

    def create_coherences(
        self,
        pairs: BurstPairCollection,
        crs: CRSLike,
        *,
        config: CoherenceConfig | None = None,
        overwrite: bool = False,
    ) -> OperationSummary[CoherenceProduct]:
        """Create coherence products from burst pair collection.

        Parameters
        ----------
        pairs
            Collection of burst pairs to process.
        crs
            The coordinate reference system of the geocoded output.
        config
            Graph-shape configuration; defaults to ``CoherenceConfig()``
            (25x5 window, Copernicus 30m DEM).
        overwrite
            Whether to overwrite existing products.

        Returns
        -------
        OperationSummary
            Summary of processing results.
        """
        cfg = config or CoherenceConfig()
        return self._process_collection(
            pairs,
            partial(
                self.snap_client.create_coherence,
                crs=crs,
                config=cfg,
                overwrite=overwrite,
            ),
            item_description="Coherence",
        )

    def create_interferograms(
        self,
        pairs: BurstPairCollection,
        crs: CRSLike,
        *,
        config: InterferogramConfig | None = None,
        overwrite: bool = False,
    ) -> OperationSummary[InterferogramProduct]:
        """Create interferogram products from burst pair collection.

        Parameters
        ----------
        pairs
            Collection of burst pairs to process.
        crs
            The coordinate reference system of the geocoded output.
        config
            Graph-shape configuration; defaults to ``InterferogramConfig()``.
        overwrite
            Whether to overwrite existing products.

        Returns
        -------
        OperationSummary
            Summary of processing results.
        """
        cfg = config or InterferogramConfig()
        return self._process_collection(
            pairs,
            partial(
                self.snap_client.create_interferogram,
                crs=crs,
                config=cfg,
                overwrite=overwrite,
            ),
            item_description="Interferogram",
        )

    @overload
    def _process_collection[T](
        self,
        collection: BurstCollection,
        process_fn: Callable[[Burst], OperationResult[T]],
        item_description: str,
    ) -> OperationSummary[T]: ...
    @overload
    def _process_collection[T](
        self,
        collection: BurstPairCollection,
        process_fn: Callable[[BurstPair], OperationResult[T]],
        item_description: str,
    ) -> OperationSummary[T]: ...
    def _process_collection(
        self,
        collection: BurstCollection | BurstPairCollection,
        process_fn: Callable[[Any], OperationResult[Any]],
        item_description: str,
    ) -> OperationSummary[Any]:
        """Process bursts with progress tracking."""
        start_time = time.time()
        results: list[OperationResult[Any]] = []
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
