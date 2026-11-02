"""Client for the CDSE burst catalog: search and download Sentinel-1 bursts."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import partial
from http import HTTPStatus
import logging
import threading
import time
from typing import TYPE_CHECKING, Literal, TypeGuard

from freezebase.errors import DownloadError
from freezebase.progress import create_progress
from requests import HTTPError

from s1bursts.burst import Burst, BurstCollection, BurstPair
from s1bursts.cdse import MAX_PAGE_SIZE, CDSEAuth, CDSEBurstClient
from s1bursts.config import get_settings
from s1bursts.errors import (
    BurstDownloadError,
    BurstError,
    BurstNotFoundError,
    CatalogError,
    ConfigurationError,
    S1BurstsError,
    TooManyResultsError,
    _copy_error,
    _is_environment_wide,
)
from s1bursts.summary import ErrorResult, OkResult, OperationResult, OperationSummary

if TYPE_CHECKING:
    from s1bursts.cdse import CDSEResponse
    from s1bursts.filemanager import FileManager
    from s1bursts.query import SearchQuery

logger = logging.getLogger(__name__)

type Endpoint = Literal["odata", "s3"]


_ENDPOINT_LABELS: dict[Endpoint, str] = {"odata": "OData", "s3": "S3"}


def _describe(error: Exception) -> str:
    """Describe an error by its message, prefixed with its type unless it is one of ours."""
    if isinstance(error, S1BurstsError):
        return str(error)
    return f"{type(error).__name__}: {error}"


def _cdse_id_present(cdse_id: str | None) -> TypeGuard[str]:
    """Type guard narrowing a burst's ``cdse_id`` to ``str`` when present."""
    return cdse_id is not None


class BurstCatalog:
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
        Whether to continue downloading after individual failures. ``True``,
        the default, attempts every burst and records each failure in the
        summary, so one missing burst does not end a long download; ``False``
        raises at the first burst that fails on both endpoints, once the
        downloads already running have finished, without starting the queued
        ones. A burst this
        catalog already found missing at the source is not requested again:
        it is recorded as failed without raising, whichever the setting, and a
        rerun on the same catalog reports it again rather than retrying it.
        Follow a run with ``OperationSummary.raise_on_failure()`` to fail once
        all bursts have been attempted.
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
        self._auth_rejected: ConfigurationError | None = None

    def _ensure_auth(self, *, primary: bool = True) -> None:
        """Create the CDSE OData authentication handler on first use.

        Deferred so that searching (which needs no auth) and S3-only downloads
        work without OData credentials, and so that credentials loaded after
        import (e.g. via ``load_dotenv()``) are picked up. Called only
        immediately before an OData request, so a user who configured only
        ``CDSE_S3_*`` credentials is never asked for OData credentials unless an
        OData attempt actually happens (as primary endpoint or as S3 fallback).
        Credentials CDSE rejects raise ``ConfigurationError`` where OData is the
        ``primary`` endpoint, and ``BurstDownloadError`` where it is the fallback.
        """
        with self._auth_lock:
            if self._auth_rejected is None and self.client.auth is None:
                cdse = get_settings().cdse
                if cdse.username is None or cdse.password is None:
                    msg = (
                        "OData download requires CDSE credentials, but none are "
                        "configured. Set CDSE_USERNAME and CDSE_PASSWORD, or download "
                        "with endpoint='s3' using CDSE_S3_* credentials only."
                    )
                    raise BurstDownloadError(msg)
                try:
                    self.client.auth = CDSEAuth()
                except ConfigurationError as e:
                    # Logging in again for every burst could lock the account.
                    self._auth_rejected = e
            rejected = self._auth_rejected
        if rejected is None:
            return
        # Copies, so that a note added where one is raised does not reach the others.
        if primary:
            raise _copy_error(rejected)
        # The endpoint the caller chose may still work, so it fails this burst only.
        raise BurstDownloadError(str(rejected)) from _copy_error(rejected)

    def search(self, query: SearchQuery) -> BurstCollection:
        """Search for Sentinel-1 bursts matching the query.

        Parameters
        ----------
        query
            Search parameters.

        Returns
        -------
        BurstCollection
            Every burst that matches the query.

        Raises
        ------
        TooManyResultsError
            If more than 1000 bursts match, the most one catalog request
            returns. Narrow the query, or combine several searches with ``+``.
        CatalogError
            If the catalog's response is malformed or does not hold the
            number of bursts it reports as matching.
        """
        filters = self._convert_query_to_filters(query)
        response = self.client.search_bursts(filters, count=True)
        self._validate_completeness(response)
        return BurstCollection.from_cdse_response(response)

    def download(
        self,
        bursts: Burst | BurstPair | Iterable[Burst | BurstPair],
        *,
        overwrite: bool = False,
        max_workers: int = 5,
        endpoint: Endpoint | None = None,
    ) -> OperationSummary[Burst]:
        """Download bursts with progress tracking and summary reporting.

        Parameters
        ----------
        bursts
            Bursts to download: a single ``Burst`` or ``BurstPair``, or any
            iterable of them, such as a ``BurstCollection``, a
            ``BurstPairCollection``, a list or a generator. A pair downloads
            both of its bursts, and a burst listed more than once, for example
            one shared by two pairs, is downloaded once.
        overwrite
            Whether to overwrite existing files. SLC file names record the pass
            direction, so a burst downloaded under an earlier name is
            downloaded again; ``FileManager`` warns once when ``slc_dir``
            holds such names.
        max_workers
            Number of parallel download threads.
        endpoint
            Download endpoint to use. If ``None``, uses the
            ``default_endpoint`` specified during initialization. If a
            download fails, the other endpoint is tried automatically.

        Returns
        -------
        OperationSummary
            Summary of download results. A burst that neither endpoint could
            download is recorded with one error naming the failure of each,
            such as ``OData: HTTPError: 503 ...; S3: ...``: a
            ``BurstNotFoundError`` if both found it missing, a
            ``BurstDownloadError`` otherwise.

        Raises
        ------
        TypeError
            If an item is neither a ``Burst`` nor a ``BurstPair``.
        ConfigurationError
            If an environment variable holds an invalid value, or CDSE rejects
            the username and password with OData as ``endpoint``, which the
            next call tries again. Missing credentials of an endpoint, and
            rejected ones of the fallback endpoint, are recorded per burst
            instead.
        OSError
            If the disk is full, over quota or read-only, which would fail
            every remaining burst alike.
        """
        start_time = time.time()
        burst_list = self._unique_bursts(bursts)
        # The credentials may have been corrected since a previous call.
        self._auth_rejected = None

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
            futures: list[Future[OperationResult[Burst]]] = []
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    progress = create_progress(show_progress=self.show_progress)
                    with progress:
                        task_id = progress.add_task(
                            "Downloading bursts",
                            total=len(bursts_to_download),
                        )
                        # Submitting inside the try, so that an interrupt while the
                        # queue is filled also cancels what was queued.
                        try:
                            # extend keeps the futures submitted before an interrupt.
                            futures.extend(
                                executor.submit(
                                    self._download_single_burst, burst, endpoint_to_use
                                )
                                for burst in bursts_to_download
                            )
                            for future in as_completed(futures):
                                results.append(future.result())
                                progress.advance(task_id)
                        except BaseException:
                            # Leaving the block joins the workers, which would run
                            # the whole queue first; only the running downloads finish.
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise
                        # Ensure the progress bar shows 100% completion
                        progress.update(task_id, refresh=True)
            except BaseException as err:
                # Counted once the running downloads have finished, so it includes them.
                n_finished = sum(future.done() and not future.cancelled() for future in futures)
                err.add_note(
                    f"Stopped with {n_finished} of {len(bursts_to_download)} bursts finished; "
                    "SLCs already downloaded are kept, and a rerun without overwrite skips them."
                )
                raise

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

    @staticmethod
    def _unique_bursts(bursts: Burst | BurstPair | Iterable[Burst | BurstPair]) -> list[Burst]:
        """Flatten the input of ``download`` to its distinct bursts, in input order."""
        expected = "Expected `bursts` to be a Burst, a BurstPair or an iterable of them"
        if isinstance(bursts, (Burst, BurstPair)):
            items: Iterable[Burst | BurstPair] = [bursts]
        elif isinstance(bursts, Iterable):
            items = bursts
        else:
            msg = f"{expected}, received '{type(bursts).__name__}'."
            raise TypeError(msg)
        # Keyed by burst equality, so concurrent downloads never write one file twice.
        unique: dict[Burst, None] = {}
        for item in items:
            match item:
                case Burst():
                    unique[item] = None
                case BurstPair():
                    unique[item.primary] = None
                    unique[item.secondary] = None
                case _:
                    msg = f"{expected}, received an item of type '{type(item).__name__}'."
                    raise TypeError(msg)
        return list(unique)

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
            return ErrorResult(source=burst, error=_copy_error(cached_error))

        # Determine primary and fallback endpoints
        primary_name: Endpoint
        fallback_name: Endpoint
        if endpoint.lower() == "odata":
            primary_name, fallback_name = "odata", "s3"
            primary_method = self._download_burst_odata
            fallback_method = self._download_burst_s3
        elif endpoint.lower() == "s3":
            primary_name, fallback_name = "s3", "odata"
            primary_method = self._download_burst_s3
            fallback_method = partial(self._download_burst_odata, primary=False)
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

        error = self._combine_endpoint_errors(
            {primary_name: primary_result.error, fallback_name: fallback_result.error},
        )
        logger.error("Both endpoints failed for burst %s: %s", str(burst), error)
        if isinstance(error, BurstNotFoundError):
            with self._unavailable_lock:
                # A copy, so that a note added where this one is raised stays here.
                self._unavailable_bursts[burst] = _copy_error(error)

        if not self.continue_on_error:
            raise error
        return ErrorResult(source=burst, error=error)

    @staticmethod
    def _combine_endpoint_errors(errors: dict[Endpoint, Exception]) -> BurstError:
        """Return one error naming the failure of each endpoint, caused by the first one's."""
        detail = "; ".join(
            f"{_ENDPOINT_LABELS[name]}: {_describe(error)}" for name, error in errors.items()
        )
        # Absent only if every endpoint says so: one transient failure can hide a
        # burst that is there.
        combined: BurstError = (
            BurstNotFoundError(detail)
            if all(isinstance(error, BurstNotFoundError) for error in errors.values())
            else BurstDownloadError(detail)
        )
        combined.__cause__ = next(iter(errors.values()))
        return combined

    def _download_burst_odata(
        self,
        burst: Burst,
        *,
        primary: bool = True,
    ) -> OperationResult[Burst]:
        """Download burst using OData API.

        Writes the result to the deterministic SLC path resolved by
        ``FileManager`` (local or S3 via UPath). ``primary`` is whether OData is
        the endpoint the caller chose rather than the fallback.
        """
        if not _cdse_id_present(burst.cdse_id):
            msg = "ODATA download requires `cdse_id`, received None."
            return ErrorResult(source=burst, error=BurstDownloadError(msg))
        try:
            self._ensure_auth(primary=primary)
            self.client.download_burst(
                burst.cdse_id,
                save_path=self.file_manager.get_slc_path(burst),
                progress=False,
            )
            return OkResult(source=burst, result=burst)
        except HTTPError as e:
            if e.response is None or e.response.status_code != HTTPStatus.NOT_FOUND:
                return ErrorResult(source=burst, error=e)
            msg = f"CDSE does not serve the burst '{burst.cdse_id}' (HTTP 404)."
            not_found = BurstNotFoundError(msg)
            not_found.__cause__ = e
            return ErrorResult(source=burst, error=not_found)
        except (BurstError, DownloadError, OSError) as e:
            if _is_environment_wide(e):
                raise
            return ErrorResult(source=burst, error=e)

    def _download_burst_s3(self, burst: Burst) -> OperationResult[Burst]:
        """Download a burst by extracting it from the parent product in CDSE S3.

        Writes the result to the deterministic SLC path resolved by
        ``FileManager`` (local or S3 via UPath).
        """
        if burst.parent_product_name is None:
            msg = "S3 download requires `parent_product_name`, received None."
            return ErrorResult(source=burst, error=BurstDownloadError(msg))
        # Outside the try: a missing burst map or an invalid setting is not a
        # credentials problem.
        save_path = self.file_manager.get_slc_path(burst)
        get_settings()
        try:
            self.client.download_burst_from_s3(
                save_path=save_path,
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
            if _is_environment_wide(e):
                raise
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

        if query.time_range is not None:
            start, end = query.time_range.half_open()
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            # ContentDate/Start maps to BeginningDateTime (sensingTime),
            # which matches Burst.date. The stop is exclusive, so the midnight
            # after the window's last day keeps that whole day.
            filters.append(
                f"ContentDate/Start ge {start.strftime(fmt)} and "
                f"ContentDate/Start lt {end.strftime(fmt)}",
            )
        if query.intersects:
            filters.append(
                f"OData.CSC.Intersects(area=geography'SRID=4326;{query.intersects.wkt}')",
            )

        return filters

    @staticmethod
    def _validate_completeness(response: CDSEResponse) -> None:
        """Raise unless the response holds every burst that matches."""
        if response.count is None:
            msg = "The catalog did not report how many bursts match the query."
            raise CatalogError(msg)
        if response.count > MAX_PAGE_SIZE:
            msg = (
                f"{response.count} bursts match the query, but one search returns at "
                f"most {MAX_PAGE_SIZE}. Narrow the TimeRange or the AOI, or split the "
                "query into several searches and combine their results with `+`."
            )
            raise TooManyResultsError(msg)
        if not response.is_complete():
            msg = (
                f"The catalog reported {response.count} matching bursts but returned "
                f"{len(response)}. Repeat the search."
            )
            raise CatalogError(msg)
