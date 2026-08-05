"""High-level batch processing for Sentinel-1 bursts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """High-level batch processor for Sentinel-1 bursts.

    Provides user-friendly interface for batch processing with:
    - Progress tracking using rich progress bars
    - Error handling with continue-on-failure
    - Summary reporting
    - Skip existing products

    Does not expose product classes to users - accepts BurstCollection
    and BurstPairCollection instead.
    """

    def __init__(
        self,
        file_manager: FileManager,
        config: SNAPConfig | None = None,
        *,
        show_progress: bool = True,
        continue_on_error: bool = True,
    ) -> None:
        """Initialize BurstProcessor.

        Parameters
        ----------
        file_manager : FileManager
            File manager resolving input SLC and output product paths.
        config : SNAPConfig | None, default None
            Runtime configuration for SNAP GPT processing; defaults to
            ``SNAPConfig()``.
        show_progress : bool, default True
            Whether to show progress bars.
        continue_on_error : bool, default True
            Whether to continue processing after an error.
        """
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
        bursts : BurstCollection
            Collection of bursts to process.
        crs : CRSLike
            The coordinate reference system of the geocoded output.
        config : BackscatterConfig | None, default None
            Graph-shape configuration; defaults to ``BackscatterConfig()``
            (RTC, Copernicus 30m DEM, multilook ``(5, 1)``).
        overwrite : bool, default False
            Whether to overwrite existing products.

        Returns
        -------
        OperationSummary
            Summary of processing results.
        """
        cfg = config or BackscatterConfig()
        return self._process_collection(
            bursts,
            lambda burst: self.snap_client.create_backscatter(
                burst,
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
        pairs : BurstPairCollection
            Collection of burst pairs to process.
        crs : CRSLike
            The coordinate reference system of the geocoded output.
        config : CoherenceConfig | None, default None
            Graph-shape configuration; defaults to ``CoherenceConfig()``
            (25x5 window, Copernicus 30m DEM).
        overwrite : bool, default False
            Whether to overwrite existing products.

        Returns
        -------
        OperationSummary
            Summary of processing results.
        """
        cfg = config or CoherenceConfig()
        return self._process_collection(
            pairs,
            lambda pair: self.snap_client.create_coherence(
                pair,
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
        pairs : BurstPairCollection
            Collection of burst pairs to process.
        crs : CRSLike
            The coordinate reference system of the geocoded output.
        config : InterferogramConfig | None, default None
            Graph-shape configuration; defaults to ``InterferogramConfig()``.
        overwrite : bool, default False
            Whether to overwrite existing products.

        Returns
        -------
        OperationSummary
            Summary of processing results.
        """
        cfg = config or InterferogramConfig()
        return self._process_collection(
            pairs,
            lambda pair: self.snap_client.create_interferogram(
                pair,
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
        """Process bursts with progress tracking.

        Parameters
        ----------
        collection : BurstCollection | BurstPairCollection
            Collection of bursts or burst pairs to process.
        process_fn : callable
            Processing function that takes a Burst or BurstPair and returns a OperationResult.
            If collection is a BurstCollection, process_fn must take Burst objects.
            If collection is a BurstPairCollection, process_fn must take BurstPair objects.
        item_description : str
            Description for progress bar.

        Returns
        -------
        OperationSummary
            Processing summary with results.
        """
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
                    result = process_fn(item)  # type: ignore[arg-type]
                    if isinstance(result, ErrorResult) and not self.continue_on_error:
                        raise result.error
                    results.append(result)
                    progress.advance(task_id)
            else:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    future_to_item = {
                        executor.submit(process_fn, item): item  # type: ignore[arg-type]
                        for item in items
                    }
                    for future in as_completed(future_to_item):
                        result = future.result()
                        if isinstance(result, ErrorResult) and not self.continue_on_error:
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise result.error
                        results.append(result)
                        progress.advance(task_id)
            progress.update(task_id, refresh=True)

        return OperationSummary.from_results(
            results,
            start_time=start_time,
            end_time=time.time(),
        )
