"""Burst database interface for searching and downloading Sentinel-1 bursts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading
import time
from typing import TYPE_CHECKING, Literal, TypeGuard

from freezebase.progress import create_progress
from requests import HTTPError

from s1bursts.burst import (
    Burst,
    BurstCollection,
    BurstPair,
    BurstPairCollection,
)
from s1bursts.cdse import CDSEAuth, CDSEBurstClient
from s1bursts.errors import (
    BurstDownloadError,
    BurstError,
    BurstNotFoundError,
    ConfigurationError,
)
from s1bursts.summary import ErrorResult, OkResult, OperationResult, OperationSummary

if TYPE_CHECKING:
    from s1bursts.cdse import CDSEResponse
    from s1bursts.filemanager import FileManager
    from s1bursts.query import SearchQuery

logger = logging.getLogger(__name__)

type Endpoint = Literal["odata", "s3"]


def _cdse_id_present(cdse_id: str | None) -> TypeGuard[str]:
    """Type guard narrowing a burst's ``cdse_id`` to ``str`` when present."""
    return cdse_id is not None


class BurstDatabase:
    """Search and download Sentinel-1 bursts from the CDSE catalog.

    Provides search via the CDSE API and parallel download with progress
    tracking, skip-existing logic, and automatic endpoint fallback.

    Parameters
    ----------
    file_manager
        File manager for determining SLC download paths.
    auth
        Authentication credentials for the CDSE API. If ``None``, a
        ``CDSEAuth`` instance is created lazily from the ``CDSE_USERNAME`` /
        ``CDSE_PASSWORD`` settings on the first download. Searching the
        catalog requires no authentication.
    show_progress
        Whether to show progress bars during downloads.
    continue_on_error
        Whether to continue downloading after individual failures.
    default_endpoint
        Default download endpoint. If a download fails, the other endpoint
        is tried automatically.
    """

    def __init__(
        self,
        file_manager: FileManager,
        auth: CDSEAuth | None = None,
        *,
        show_progress: bool = True,
        continue_on_error: bool = True,
        default_endpoint: Endpoint = "odata",
    ) -> None:
        self.client = CDSEBurstClient(auth=auth)
        self.file_manager = file_manager
        self.show_progress = show_progress
        self.continue_on_error = continue_on_error
        self.default_endpoint: Endpoint = default_endpoint
        # Cache source-confirmed absences to avoid repeating both endpoint attempts.
        self._unavailable_bursts: dict[Burst, BurstNotFoundError] = {}
        self._unavailable_lock = threading.Lock()
        self._auth_lock = threading.Lock()

    def _ensure_auth(self) -> None:
        """Create the CDSE OData authentication handler on first use.

        Deferred so that searching (which needs no auth) and S3-only downloads
        work without OData credentials, and so that credentials loaded after
        import (e.g. via ``load_dotenv()``) are picked up. Called only
        immediately before an OData request, so a user who configured only
        ``CDSE_S3_*`` credentials is never asked for OData credentials unless an
        OData attempt actually happens (as primary endpoint or as S3 fallback).
        """
        with self._auth_lock:
            if self.client.auth is None:
                try:
                    self.client.auth = CDSEAuth()
                except ValueError as e:
                    msg = (
                        "OData download requires CDSE credentials, but none are "
                        "configured. Set CDSE_USERNAME and CDSE_PASSWORD, or download "
                        "with endpoint='s3' using CDSE_S3_* credentials only."
                    )
                    raise BurstDownloadError(msg) from e

    def search(self, query: SearchQuery, max_items: int = 100) -> BurstCollection:
        """Search for Sentinel-1 bursts matching the query.

        Parameters
        ----------
        query
            Search parameters.
        max_items
            Maximum number of results to return.

        Returns
        -------
        BurstCollection
            Collection of matching bursts.

        Raises
        ------
        RuntimeError
            If the response is truncated and max_items is too small.
        """
        filters = self._convert_query_to_filters(query)
        response = self.client.search_bursts(filters, top=max_items, count=True)
        self._validate_completeness(response)
        return BurstCollection.from_cdse_response(response)

    def download(
        self,
        bursts: BurstCollection | BurstPairCollection | Burst | BurstPair,
        *,
        overwrite: bool = False,
        max_workers: int = 5,
        endpoint: Endpoint | None = None,
    ) -> OperationSummary[Burst]:
        """Download bursts with progress tracking and summary reporting.

        Parameters
        ----------
        bursts
            Bursts to download.
        overwrite
            Whether to overwrite existing files.
        max_workers
            Number of parallel download threads.
        endpoint
            Download endpoint to use. If ``None``, uses the
            ``default_endpoint`` specified during initialization. If a
            download fails, the other endpoint is tried automatically.

        Returns
        -------
        OperationSummary
            Summary of download results.
        """
        start_time = time.time()

        # Convert input to burst list
        match bursts:
            case BurstCollection():
                burst_list = list(bursts)
            case BurstPairCollection():
                burst_list = list(bursts.unpair())
            case Burst():
                burst_list = [bursts]
            case BurstPair():
                burst_list = [bursts.primary, bursts.secondary]
            case _:
                msg = (
                    "Expected `bursts` of type "
                    "BurstCollection | BurstPairCollection | Burst | BurstPair, "
                    f"received '{type(bursts).__name__}'."
                )
                raise TypeError(msg)

        results: list[OperationResult[Burst]] = []

        # Separate bursts into skipped and to-download
        bursts_to_download = []
        for burst in burst_list:
            if not overwrite and self.file_manager.get_slc_path(burst).exists():
                # Skip download
                results.append(OkResult(source=burst, result=burst, skipped=True))
            else:
                bursts_to_download.append(burst)

        # Determine which endpoint to use
        endpoint_to_use = endpoint if endpoint is not None else self.default_endpoint

        # Download remaining bursts if any. Authentication is deferred to the
        # OData code path (see `_ensure_auth`) so that `endpoint="s3"` works with
        # only CDSE_S3_* credentials configured.
        if bursts_to_download:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._download_single_burst, burst, endpoint_to_use)
                    for burst in bursts_to_download
                ]

                # Setup iterator with or without progress tracking
                progress = create_progress(show_progress=self.show_progress)
                with progress:
                    task_id = progress.add_task(
                        "Downloading bursts",
                        total=len(bursts_to_download),
                    )
                    for future in as_completed(futures):
                        results.append(future.result())
                        progress.advance(task_id)
                    # Ensure the progress bar shows 100% completion
                    progress.update(task_id, refresh=True)

            # Invalidate fsspec directory cache so that subsequent glob/ls
            # calls see the newly uploaded files (relevant for S3-backed UPath
            # directories where listings are cached per-session).
            slc_dir = self.file_manager.slc_dir
            if hasattr(slc_dir, "fs"):
                slc_dir.fs.invalidate_cache(str(slc_dir))

        return OperationSummary.from_results(
            results,
            start_time=start_time,
            end_time=time.time(),
        )

    def _download_single_burst(self, burst: Burst, endpoint: Endpoint) -> OperationResult[Burst]:
        """Download a single burst using specified endpoint with fallback."""
        if self.file_manager.get_slc_path(burst).exists():
            return OkResult(source=burst, result=burst, skipped=True)

        with self._unavailable_lock:
            cached_error = self._unavailable_bursts.get(burst)
        if cached_error is not None:
            logger.debug(
                "Skipping burst %s: already confirmed unavailable (%s).",
                str(burst),
                cached_error,
            )
            return ErrorResult(source=burst, error=cached_error)

        # Determine primary and fallback endpoints
        if endpoint.lower() == "odata":
            primary_method = self._download_burst_odata
            fallback_method = self._download_burst_s3
        elif endpoint.lower() == "s3":
            primary_method = self._download_burst_s3
            fallback_method = self._download_burst_odata
        else:
            msg = f"`endpoint` must be 'odata' or 's3', received '{endpoint}'"
            raise ValueError(msg)

        # Try primary endpoint
        primary_result = primary_method(burst)
        if isinstance(primary_result, OkResult):
            logger.info("Successfully downloaded burst %s using primary endpoint.", str(burst))
            return primary_result

        # Log the primary error (not just that it failed) before trying the fallback.
        # Without the error detail an environment-wide failure (e.g. a broken PROJ
        # database breaking CRS lookups) looks like a per-burst data gap.
        logger.info(
            "Primary endpoint failed for burst %s (%s: %s). Attempting fallback endpoint.",
            str(burst),
            type(primary_result.error).__name__,
            primary_result.error,
        )

        # Try fallback endpoint
        fallback_result = fallback_method(burst)
        if isinstance(fallback_result, OkResult):
            logger.info(
                "Successfully downloaded burst %s using fallback endpoint.",
                str(burst),
            )
            return fallback_result

        # Both endpoints failed: surface both root causes, otherwise the downstream
        # FileNotFoundError (missing SLC) hides why the SLC was never produced.
        logger.error(
            "Both endpoints failed for burst %s. Primary %s: %s. Fallback %s: %s.",
            str(burst),
            type(primary_result.error).__name__,
            primary_result.error,
            type(fallback_result.error).__name__,
            fallback_result.error,
        )

        # Cache only definitive absences, not transient transport failures.
        not_found_error = next(
            (
                err
                for err in (fallback_result.error, primary_result.error)
                if isinstance(err, BurstNotFoundError)
            ),
            None,
        )
        if not_found_error is not None:
            with self._unavailable_lock:
                self._unavailable_bursts[burst] = not_found_error

        if not self.continue_on_error:
            raise fallback_result.error
        return fallback_result

    def _download_burst_odata(self, burst: Burst) -> OperationResult[Burst]:
        """Download burst using OData API.

        Writes the result to the deterministic SLC path resolved by
        ``FileManager`` (local or S3 via UPath).
        """
        if not _cdse_id_present(burst.cdse_id):
            msg = "ODATA download requires `cdse_id`, received None."
            return ErrorResult(source=burst, error=BurstDownloadError(msg))
        try:
            self._ensure_auth()
            self.client.download_burst(
                burst.cdse_id,
                save_path=self.file_manager.get_slc_path(burst),
                progress=False,
            )
            return OkResult(source=burst, result=burst)
        except (BurstError, HTTPError, OSError) as e:
            return ErrorResult(source=burst, error=e)

    def _download_burst_s3(self, burst: Burst) -> OperationResult[Burst]:
        """Download a burst by extracting it from the parent product in CDSE S3.

        Writes the result to the deterministic SLC path resolved by
        ``FileManager`` (local or S3 via UPath).
        """
        if burst.parent_product_name is None:
            msg = "S3 download requires `parent_product_name`, received None."
            return ErrorResult(source=burst, error=BurstDownloadError(msg))
        try:
            self.client.download_burst_from_s3(
                save_path=self.file_manager.get_slc_path(burst),
                parent_product_name=burst.parent_product_name,
                polarization=burst.polarization,
                sub_swath=burst.sub_swath,
                burst_id=burst.burst_id,
            )
            return OkResult(source=burst, result=burst)
        except ConfigurationError as e:
            # Recorded per burst, as `_ensure_auth` does for OData, so that the
            # other endpoint is still tried with the credentials that are set.
            msg = (
                f"{e} Set them, or download with endpoint='odata' using "
                "CDSE_USERNAME and CDSE_PASSWORD only."
            )
            return ErrorResult(source=burst, error=BurstDownloadError(msg))
        except (BurstError, OSError) as e:
            return ErrorResult(source=burst, error=e)

    @staticmethod
    def _convert_query_to_filters(query: SearchQuery) -> list[str]:
        """Convert a SearchQuery to a list of CDSE OData filter strings."""
        # CDSE spells the pass direction upper case and calls it OrbitDirection.
        pass_direction = query.pass_direction.upper() if query.pass_direction else None
        simple_filters: list[tuple[str | int | None, str]] = [
            (query.burst_id, f"BurstId eq {query.burst_id}"),
            (query.abs_burst_id, f"AbsoluteBurstId eq {query.abs_burst_id}"),
            (query.polarization, f"PolarisationChannels eq '{query.polarization}'"),
            (query.mode, f"OperationalMode eq '{query.mode}'"),
            (query.sub_swath, f"SwathIdentifier eq '{query.sub_swath}'"),
            (pass_direction, f"OrbitDirection eq '{pass_direction}'"),
            (
                query.platform_serial_id,
                f"PlatformSerialIdentifier eq '{query.platform_serial_id}'",
            ),
            (
                query.rel_orbit_number,
                f"RelativeOrbitNumber eq {query.rel_orbit_number}",
            ),
        ]

        filters = [expr for value, expr in simple_filters if value is not None]

        if query.time_range:
            start, end = query.time_range
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            # ContentDate/Start maps to BeginningDateTime (sensingTime),
            # which matches Burst.date.
            filters.append(
                f"ContentDate/Start ge {start.strftime(fmt)} and "
                f"ContentDate/Start le {end.strftime(fmt)}",
            )
        if query.intersects:
            filters.append(
                f"OData.CSC.Intersects(area=geography'SRID=4326;{query.intersects.wkt}')",
            )

        return filters

    @staticmethod
    def _validate_completeness(response: CDSEResponse) -> None:
        """Raise if the CDSE response is truncated."""
        if response.is_complete():
            return
        msg = (
            f"{response.count} bursts match the query, but response only contains "
            f"{len(response)} elements. Increase `max_items` parameter or narrow down "
            "the query."
        )
        raise RuntimeError(msg)
