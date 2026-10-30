"""Tests for BurstCatalog (search filters, download dispatch, negative cache)."""

from __future__ import annotations

from collections import Counter
import errno
import os
from typing import TYPE_CHECKING, Literal, cast
from unittest.mock import MagicMock, patch

from freezebase.errors import DownloadError
from pandas import Timestamp
import pytest
from requests import HTTPError, Request
from shapely.geometry import box

from s1bursts.burst import Burst, BurstCollection, BurstPair
from s1bursts.catalog import BurstCatalog
from s1bursts.cdse import MAX_PAGE_SIZE, CDSEResponse
from s1bursts.errors import (
    BurstDownloadError,
    BurstMapUnavailableError,
    BurstNotFoundError,
    CatalogError,
    ConfigurationError,
    TooManyResultsError,
)
from s1bursts.query import SearchQuery
from s1bursts.summary import ErrorResult, OkResult, OperationResult
from s1bursts.time_range import TimeRange
from tests.gated_pool import Gate

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def _make_catalog() -> BurstCatalog:
    file_manager = MagicMock()
    file_manager.get_slc_path.return_value.exists.return_value = False
    return BurstCatalog(file_manager=file_manager, auth=MagicMock())


def _make_burst(date: str = "2016-07-24") -> Burst:
    return Burst(
        burst_id=297227,
        sub_swath="IW3",
        polarization="VV",
        date=Timestamp(date),
    )


class TestQueryToFilters:
    def test_simple_filters(self) -> None:
        query = SearchQuery(
            burst_id=30346,
            polarization="VV",
            mode="IW",
            sub_swath="IW1",
            pass_direction="Ascending",
            platform_serial_id="A",
            rel_orbit_number=15,
        )

        filters = BurstCatalog._convert_query_to_filters(query)

        assert "BurstId eq 30346" in filters
        assert "PolarisationChannels eq 'VV'" in filters
        assert "OperationalMode eq 'IW'" in filters
        assert "SwathIdentifier eq 'IW1'" in filters
        # CDSE's own spelling of the field and its value, not the package's.
        assert "OrbitDirection eq 'ASCENDING'" in filters
        assert "PlatformSerialIdentifier eq 'A'" in filters
        assert "RelativeOrbitNumber eq 15" in filters

    def test_empty_query_produces_no_filters(self) -> None:
        assert BurstCatalog._convert_query_to_filters(SearchQuery()) == []

    def test_time_range_window_filter_keeps_its_last_day(self) -> None:
        query = SearchQuery(time_range=TimeRange("2022-01-01", "2022-01-31"))

        filters = BurstCatalog._convert_query_to_filters(query)

        assert filters == [
            (
                "ContentDate/Start ge 2022-01-01T00:00:00Z and "
                "ContentDate/Start lt 2022-02-01T00:00:00Z"
            ),
        ]

    def test_intersects_filter(self) -> None:
        query = SearchQuery(intersects=box(7, 46, 8, 47))

        (filter_,) = BurstCatalog._convert_query_to_filters(query)

        assert filter_.startswith("OData.CSC.Intersects(area=geography'SRID=4326;POLYGON")


class TestDownloadDispatch:
    def test_rejects_an_item_that_is_not_a_burst(self) -> None:
        catalog = _make_catalog()

        with pytest.raises(TypeError, match="received an item of type 'str'"):
            # The wrong type is what is under test.
            # pyrefly: ignore [bad-argument-type]
            catalog.download(["not-a-burst"])

    def test_rejects_a_non_iterable(self) -> None:
        catalog = _make_catalog()

        with pytest.raises(TypeError, match="iterable of them, received 'int'"):
            # The wrong type is what is under test.
            # pyrefly: ignore [bad-argument-type]
            catalog.download(30346)

    @pytest.mark.parametrize("container", [list, tuple, iter, BurstCollection.from_bursts])
    def test_accepts_any_iterable_of_bursts(
        self,
        container: Callable[[list[Burst]], Iterable[Burst]],
    ) -> None:
        catalog = _make_catalog()
        bursts = [_make_burst(), _make_burst(date="2016-08-05")]

        with patch.object(
            catalog,
            "_download_single_burst",
            side_effect=lambda b, _e: OkResult(source=b, result=b),
        ) as mock_download:
            summary = catalog.download(container(bursts))

        # Downloads run in a thread pool, so the call order is not fixed.
        assert Counter(c.args[0] for c in mock_download.call_args_list) == Counter(bursts)
        assert summary.n_processed == 2

    def test_downloads_a_burst_shared_by_two_pairs_once(self) -> None:
        catalog = _make_catalog()
        first, second, third = (
            _make_burst(date=d) for d in ("2016-07-24", "2016-08-05", "2016-08-17")
        )
        # Equal to `second` but a separate object, as two pairs built from one
        # search would hold.
        second_again = _make_burst(date="2016-08-05")
        assert second_again == second
        assert second_again is not second
        pairs = [BurstPair(first, second), BurstPair(second_again, third)]

        with patch.object(
            catalog,
            "_download_single_burst",
            side_effect=lambda b, _e: OkResult(source=b, result=b),
        ) as mock_download:
            summary = catalog.download(pair for pair in pairs)

        downloaded = Counter(c.args[0] for c in mock_download.call_args_list)
        assert downloaded == Counter([first, second, third])
        assert summary.n_total == 3

    def test_accepts_a_mixed_list(self) -> None:
        catalog = _make_catalog()
        pair = BurstPair(_make_burst(), _make_burst(date="2016-08-05"))
        single = _make_burst(date="2016-08-17")

        with patch.object(
            catalog,
            "_download_single_burst",
            side_effect=lambda b, _e: OkResult(source=b, result=b),
        ):
            summary = catalog.download([pair, single])

        assert summary.n_processed == 3

    def test_rejects_unknown_endpoint(self) -> None:
        catalog = _make_catalog()

        with pytest.raises(ValueError, match="must be 'odata' or 's3'"):
            # pyrefly: ignore [bad-argument-type]
            catalog._download_single_burst(_make_burst(), "ftp")

    def test_skips_existing_files_without_auth(self) -> None:
        file_manager = MagicMock()
        file_manager.get_slc_path.return_value.exists.return_value = True
        catalog = BurstCatalog(file_manager=file_manager)

        summary = catalog.download(_make_burst())

        assert summary.n_skipped == 1
        assert summary.n_processed == 0
        # No download happened, so no CDSEAuth may have been created
        assert catalog.client.auth is None

    def test_creates_auth_lazily_on_odata_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OData downloads build a CDSEAuth on first use, guarded to run once."""
        monkeypatch.setenv("CDSE_USERNAME", "user")
        monkeypatch.setenv("CDSE_PASSWORD", "pass")
        catalog = _make_catalog()
        catalog.client = MagicMock()
        catalog.client.auth = None
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            cdse_id="cdse-id",
        )

        with patch("s1bursts.catalog.CDSEAuth") as mock_auth_cls:
            result = catalog._download_burst_odata(burst)

        assert isinstance(result, OkResult)
        mock_auth_cls.assert_called_once()

    def test_s3_download_needs_no_odata_auth(self) -> None:
        """endpoint='s3' must not construct a CDSEAuth (S1-03)."""
        file_manager = MagicMock()
        file_manager.get_slc_path.return_value.exists.return_value = False
        catalog = BurstCatalog(file_manager=file_manager, default_endpoint="s3")
        catalog.client = MagicMock()
        catalog.client.auth = None
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            parent_product_name="S1A_IW_SLC__1SDV",
        )

        with patch("s1bursts.catalog.CDSEAuth") as mock_auth_cls:
            summary = catalog.download(burst, endpoint="s3")

        assert summary.n_processed == 1
        mock_auth_cls.assert_not_called()

    def test_burst_pair_downloads_both_bursts(self) -> None:
        catalog = _make_catalog()
        primary = _make_burst()
        secondary = Burst(
            burst_id=primary.burst_id,
            sub_swath=primary.sub_swath,
            polarization=primary.polarization,
            date=Timestamp("2016-08-05"),
        )
        pair = BurstPair(primary, secondary)

        with (
            patch.object(catalog, "_ensure_auth"),
            patch.object(
                catalog,
                "_download_single_burst",
                side_effect=lambda b, _e: OkResult(source=b, result=b),
            ) as mock_download,
        ):
            summary = catalog.download(pair)

        assert mock_download.call_count == 2
        assert summary.n_processed == 2


class TestStoppingADownload:
    """A download that stops early cancels its queue and says what it kept."""

    @staticmethod
    def _bursts(n: int) -> list[Burst]:
        return [_make_burst(f"2016-07-{day:02d}") for day in range(1, n + 1)]

    def test_ctrl_c_cancels_the_queued_downloads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Previously the queue was drained before the KeyboardInterrupt surfaced."""
        catalog = _make_catalog()
        catalog.show_progress = False
        gate = Gate(n_running=2)
        monkeypatch.setattr("s1bursts.catalog.ThreadPoolExecutor", gate.executor())
        monkeypatch.setattr("s1bursts.catalog.as_completed", gate.interrupt)
        bursts = self._bursts(6)

        def download(burst: Burst, _endpoint: str) -> OkResult[Burst]:
            gate.enter(burst)
            return OkResult(source=burst, result=burst)

        with (
            patch.object(catalog, "_download_single_burst", side_effect=download),
            pytest.raises(KeyboardInterrupt) as excinfo,
        ):
            catalog.download(bursts, max_workers=2)

        assert set(gate.started) == set(bursts[:2])
        assert excinfo.value.__notes__ == [
            (
                "Stopped with 2 of 6 bursts finished; SLCs already downloaded are kept, "
                "and a rerun without overwrite skips them."
            )
        ]

    def test_ctrl_c_while_submitting_cancels_what_was_queued(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        catalog = _make_catalog()
        catalog.show_progress = False
        gate = Gate(n_running=2)
        monkeypatch.setattr(
            "s1bursts.catalog.ThreadPoolExecutor",
            gate.executor(interrupt_at_submit=4),
        )
        bursts = self._bursts(6)

        def download(burst: Burst, _endpoint: str) -> OkResult[Burst]:
            gate.enter(burst)
            return OkResult(source=burst, result=burst)

        with (
            patch.object(catalog, "_download_single_burst", side_effect=download),
            pytest.raises(KeyboardInterrupt) as excinfo,
        ):
            catalog.download(bursts, max_workers=2)

        # The third and fourth were queued and cancelled; the first two may have
        # been cancelled too if no worker had taken them yet.
        assert set(gate.started) <= set(bursts[:2])
        n_finished = len(gate.started)
        assert f"Stopped with {n_finished} of 6 bursts finished" in excinfo.value.__notes__[0]

    def test_a_failure_without_continue_on_error_cancels_the_queue(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Previously the queue was drained before the first failure was raised."""
        catalog = _make_catalog()
        catalog.show_progress = False
        catalog.continue_on_error = False
        gate = Gate(n_running=2)
        monkeypatch.setattr("s1bursts.catalog.ThreadPoolExecutor", gate.executor())
        bursts = self._bursts(8)

        def odata(burst: Burst) -> OperationResult[Burst]:
            if burst == bursts[0]:
                return ErrorResult(source=burst, error=HTTPError("503 Service Unavailable"))
            gate.enter(burst)
            return OkResult(source=burst, result=burst)

        def s3(burst: Burst) -> ErrorResult:
            return ErrorResult(source=burst, error=HTTPError("502 Bad Gateway"))

        with (
            patch.object(catalog, "_download_burst_odata", side_effect=odata),
            patch.object(catalog, "_download_burst_s3", side_effect=s3),
            pytest.raises(BurstDownloadError) as excinfo,
        ):
            catalog.download(bursts, max_workers=2)

        # The worker freed by the failure may have taken the third burst first.
        assert set(gate.started) <= set(bursts[1:3])
        n_finished = 1 + len(gate.started)
        assert f"Stopped with {n_finished} of 8 bursts finished" in excinfo.value.__notes__[0]


class TestEndpointWrappers:
    def test_odata_requires_cdse_id(self) -> None:
        catalog = _make_catalog()
        burst = _make_burst()  # has no cdse_id

        result = catalog._download_burst_odata(burst)

        assert isinstance(result, ErrorResult)
        assert "requires `cdse_id`" in str(result.error)

    def test_odata_success(self) -> None:
        catalog = _make_catalog()
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            cdse_id="cdse-id",
        )
        catalog.client = MagicMock()

        result = catalog._download_burst_odata(burst)

        assert isinstance(result, OkResult)
        catalog.client.download_burst.assert_called_once()

    def test_odata_http_error_becomes_error_result(self) -> None:
        catalog = _make_catalog()
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            cdse_id="cdse-id",
        )
        catalog.client = MagicMock()
        catalog.client.download_burst.side_effect = HTTPError("503 Service Unavailable")

        result = catalog._download_burst_odata(burst)

        assert isinstance(result, ErrorResult)
        assert isinstance(result.error, HTTPError)

    def test_odata_non_file_response_becomes_error_result(self) -> None:
        catalog = _make_catalog()
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            cdse_id="cdse-id",
        )
        catalog.client = MagicMock()
        # Previously escaped the handler and aborted the whole batch.
        catalog.client.download_burst.side_effect = DownloadError("No downloadable file found")

        result = catalog._download_burst_odata(burst)

        assert isinstance(result, ErrorResult)
        assert isinstance(result.error, DownloadError)

    def test_odata_unrelated_runtime_error_propagates(self) -> None:
        """Only freezebase's download failures become a per-burst error."""
        catalog = _make_catalog()
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            cdse_id="cdse-id",
        )
        catalog.client = MagicMock()
        catalog.client.download_burst.side_effect = RuntimeError("bug")

        with pytest.raises(RuntimeError, match="bug"):
            catalog._download_burst_odata(burst)

    def test_odata_without_credentials_returns_clear_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A missing-credentials failure surfaces as a per-burst BurstDownloadError."""
        monkeypatch.delenv("CDSE_USERNAME", raising=False)
        monkeypatch.delenv("CDSE_PASSWORD", raising=False)
        file_manager = MagicMock()
        file_manager.get_slc_path.return_value.exists.return_value = False
        catalog = BurstCatalog(file_manager=file_manager)
        catalog.client.auth = None
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            cdse_id="cdse-id",
        )

        with patch("s1bursts.catalog.CDSEAuth") as mock_auth_cls:
            result = catalog._download_burst_odata(burst)

        assert isinstance(result, ErrorResult)
        assert isinstance(result.error, BurstDownloadError)
        assert "endpoint='s3'" in str(result.error)
        mock_auth_cls.assert_not_called()

    def test_s3_without_credentials_returns_clear_error(self) -> None:
        """Missing S3 credentials surface per burst, so the OData fallback still runs.

        Previously the ConfigurationError propagated and aborted the whole batch.
        """
        catalog = _make_catalog()
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            parent_product_name="S1A_IW_SLC__1SDV",
        )
        catalog.client = MagicMock()
        catalog.client.download_burst_from_s3.side_effect = ConfigurationError(
            "CDSE S3 credentials not configured",
        )

        result = catalog._download_burst_s3(burst)

        assert isinstance(result, ErrorResult)
        assert isinstance(result.error, BurstDownloadError)
        assert "endpoint='odata'" in str(result.error)

    def test_s3_missing_burst_map_is_not_blamed_on_credentials(self) -> None:
        """A missing burst map propagates instead of becoming a credentials error."""
        file_manager = MagicMock()
        file_manager.get_slc_path.side_effect = BurstMapUnavailableError("not downloaded")
        catalog = BurstCatalog(file_manager=file_manager, auth=MagicMock())
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            parent_product_name="S1A_IW_SLC__1SDV",
        )
        catalog.client = MagicMock()

        with pytest.raises(BurstMapUnavailableError, match="not downloaded"):
            catalog._download_burst_s3(burst)

        catalog.client.download_burst_from_s3.assert_not_called()

    @pytest.mark.parametrize("endpoint", ["odata", "s3"])
    def test_an_invalid_setting_stops_the_download(
        self,
        monkeypatch: pytest.MonkeyPatch,
        endpoint: Literal["odata", "s3"],
    ) -> None:
        """An invalid environment value fails fast instead of passing as missing credentials."""
        monkeypatch.setenv("SNAP_MAX_WORKERS", "0")
        catalog = _make_catalog()
        catalog.client = MagicMock()
        catalog.client.auth = None
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            cdse_id="cdse-id",
            parent_product_name="S1A_IW_SLC__1SDV",
        )

        with pytest.raises(ConfigurationError, match="SNAP_MAX_WORKERS"):
            catalog.download(burst, endpoint=endpoint)

        catalog.client.download_burst.assert_not_called()
        catalog.client.download_burst_from_s3.assert_not_called()

    def test_s3_requires_parent_product_name(self) -> None:
        catalog = _make_catalog()
        burst = _make_burst()  # has no parent_product_name

        result = catalog._download_burst_s3(burst)

        assert isinstance(result, ErrorResult)
        assert "requires `parent_product_name`" in str(result.error)

    def test_s3_success(self) -> None:
        catalog = _make_catalog()
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            parent_product_name="S1A_IW_SLC__1SDV",
        )
        catalog.client = MagicMock()

        result = catalog._download_burst_s3(burst)

        assert isinstance(result, OkResult)
        catalog.client.download_burst_from_s3.assert_called_once()


class TestSearch:
    @staticmethod
    def _catalog(n_values: int, count: int) -> tuple[BurstCatalog, MagicMock]:
        """Return a catalog whose client returns ``n_values`` of ``count`` matches."""
        catalog = _make_catalog()
        client = MagicMock()
        client.search_bursts.return_value = CDSEResponse(
            context="$metadata#Bursts",
            value=[{} for _ in range(n_values)],
            count=count,
        )
        catalog.client = client
        return catalog, client

    def test_returns_every_match_from_one_counted_request(self) -> None:
        catalog, client = self._catalog(n_values=2, count=2)

        with patch("s1bursts.catalog.BurstCollection.from_cdse_response") as from_response:
            bursts = catalog.search(SearchQuery(burst_id=30346))

        assert bursts is from_response.return_value
        client.search_bursts.assert_called_once_with(["BurstId eq 30346"], count=True)
        assert from_response.call_args.args[0] is client.search_bursts.return_value

    def test_more_matches_than_one_page_raises(self) -> None:
        catalog, _ = self._catalog(n_values=MAX_PAGE_SIZE, count=1226)

        with pytest.raises(TooManyResultsError, match=r"1226 bursts match.*at most 1000"):
            catalog.search(SearchQuery())

    @pytest.mark.parametrize("count", [0, MAX_PAGE_SIZE])
    def test_a_full_or_empty_page_is_complete(self, count: int) -> None:
        catalog, _ = self._catalog(n_values=count, count=count)

        with patch("s1bursts.catalog.BurstCollection.from_cdse_response") as from_response:
            bursts = catalog.search(SearchQuery())

        assert bursts is from_response.return_value

    def test_a_short_page_is_not_reported_as_too_many(self) -> None:
        catalog, _ = self._catalog(n_values=3, count=5)

        with pytest.raises(CatalogError, match=r"reported 5 matching bursts but returned 3"):
            catalog.search(SearchQuery())

    def test_a_response_without_count_raises(self) -> None:
        catalog, client = self._catalog(n_values=2, count=2)
        client.search_bursts.return_value.count = None

        with pytest.raises(CatalogError, match=r"did not report how many bursts match"):
            catalog.search(SearchQuery())


def _fail_both(
    monkeypatch: pytest.MonkeyPatch,
    catalog: BurstCatalog,
    odata: Exception,
    s3: Exception,
) -> tuple[MagicMock, MagicMock]:
    """Patch both endpoints of ``catalog`` to fail with the given errors."""
    mock_odata = MagicMock(side_effect=lambda b, **_: ErrorResult(source=b, error=odata))
    mock_s3 = MagicMock(side_effect=lambda b: ErrorResult(source=b, error=s3))
    monkeypatch.setattr(catalog, "_download_burst_odata", mock_odata)
    monkeypatch.setattr(catalog, "_download_burst_s3", mock_s3)
    return mock_odata, mock_s3


class TestBothEndpointsFailing:
    def test_the_error_names_both_endpoints(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Previously only the fallback's error was kept, blaming S3 for an OData outage."""
        catalog = _make_catalog()
        odata_error = HTTPError("503 Service Unavailable")
        _fail_both(
            monkeypatch,
            catalog,
            odata_error,
            BurstDownloadError("CDSE S3 credentials not configured."),
        )

        result = catalog._download_single_burst(_make_burst(), "odata")

        assert isinstance(result, ErrorResult)
        assert type(result.error) is BurstDownloadError
        assert str(result.error) == (
            "OData: HTTPError: 503 Service Unavailable; S3: CDSE S3 credentials not configured."
        )
        assert result.error.__cause__ is odata_error

    def test_the_primary_endpoint_is_named_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        catalog = _make_catalog()
        s3_error = OSError("connection reset")
        _fail_both(monkeypatch, catalog, HTTPError("503 Service Unavailable"), s3_error)

        result = catalog._download_single_burst(_make_burst(), "s3")

        assert isinstance(result, ErrorResult)
        assert str(result.error).startswith("S3: OSError: connection reset; OData: HTTPError")
        assert result.error.__cause__ is s3_error

    def test_without_continue_on_error_the_combined_error_is_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        catalog = _make_catalog()
        catalog.continue_on_error = False
        _fail_both(
            monkeypatch,
            catalog,
            HTTPError("503 Service Unavailable"),
            HTTPError("502 Bad Gateway"),
        )

        with pytest.raises(BurstDownloadError, match=r"^OData: .*503.*; S3: .*502"):
            catalog._download_single_burst(_make_burst(), "odata")

    def test_a_burst_both_endpoints_miss_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A confirmed absence is not requested again from the same catalog."""
        catalog = _make_catalog()
        burst = _make_burst()
        mock_odata, mock_s3 = _fail_both(
            monkeypatch,
            catalog,
            BurstNotFoundError("no such burst"),
            BurstNotFoundError("no such product"),
        )

        first = catalog._download_single_burst(burst, "odata")
        second = catalog._download_single_burst(burst, "odata")

        assert isinstance(first, ErrorResult)
        assert isinstance(first.error, BurstNotFoundError)
        assert str(first.error) == "OData: no such burst; S3: no such product"
        assert isinstance(second, ErrorResult)
        assert type(second.error) is type(first.error)
        assert str(second.error) == str(first.error)
        mock_odata.assert_called_once()
        mock_s3.assert_called_once()

    @pytest.mark.parametrize("endpoint", ["odata", "s3"])
    def test_a_burst_missing_from_one_endpoint_only_is_not_cached(
        self,
        monkeypatch: pytest.MonkeyPatch,
        endpoint: Literal["odata", "s3"],
    ) -> None:
        """Previously cached on one BurstNotFoundError, though the other failure was transient."""
        catalog = _make_catalog()
        burst = _make_burst()
        mock_odata, mock_s3 = _fail_both(
            monkeypatch,
            catalog,
            HTTPError("503 Service Unavailable"),
            BurstNotFoundError("not yet replicated"),
        )

        first = catalog._download_single_burst(burst, endpoint)
        catalog._download_single_burst(burst, endpoint)

        assert isinstance(first, ErrorResult)
        assert type(first.error) is BurstDownloadError
        assert mock_odata.call_count == 2
        assert mock_s3.call_count == 2

    def test_a_transient_failure_is_not_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        catalog = _make_catalog()
        burst = _make_burst()
        mock_odata, mock_s3 = _fail_both(
            monkeypatch,
            catalog,
            HTTPError("500 Server Error"),
            HTTPError("connection reset"),
        )

        catalog._download_single_burst(burst, "odata")
        catalog._download_single_burst(burst, "odata")

        assert mock_odata.call_count == 2
        assert mock_s3.call_count == 2


def _odata_burst(day: int = 24) -> Burst:
    return Burst(
        burst_id=297227,
        sub_swath="IW3",
        polarization="VV",
        date=Timestamp(f"2016-07-{day:02d}"),
        cdse_id=f"cdse-{day}",
        parent_product_name="S1A_IW_SLC__1SDV",
    )


def _http_error(status: int) -> HTTPError:
    response = MagicMock()
    response.status_code = status
    return HTTPError(f"{status} Client Error", response=response)


class TestErrorClassification:
    def test_odata_404_is_burst_not_found(self) -> None:
        """Previously a plain HTTPError, so an absence both endpoints confirmed was not cached."""
        catalog = _make_catalog()
        catalog.client = MagicMock()
        http_error = _http_error(404)
        catalog.client.download_burst.side_effect = http_error

        result = catalog._download_burst_odata(_odata_burst())

        assert isinstance(result, ErrorResult)
        assert isinstance(result.error, BurstNotFoundError)
        assert "cdse-24" in str(result.error)
        assert result.error.__cause__ is http_error

    def test_another_odata_http_error_is_kept(self) -> None:
        catalog = _make_catalog()
        catalog.client = MagicMock()
        http_error = _http_error(403)
        catalog.client.download_burst.side_effect = http_error

        result = catalog._download_burst_odata(_odata_burst())

        assert isinstance(result, ErrorResult)
        assert result.error is http_error

    def test_a_malformed_product_is_recorded_per_burst(self) -> None:
        """The extractor reports a malformed SAFE file as a BurstDownloadError."""
        catalog = _make_catalog()
        catalog.client = MagicMock()
        malformed = BurstDownloadError("Malformed annotation in 'S1A_IW_SLC__1SDV': ...")
        catalog.client.download_burst_from_s3.side_effect = malformed

        result = catalog._download_burst_s3(_odata_burst())

        assert isinstance(result, ErrorResult)
        assert result.error is malformed

    def test_a_type_error_outside_parsing_is_not_recorded(self) -> None:
        """Only the extractor's parsing turns a TypeError into a per-burst failure."""
        catalog = _make_catalog()
        catalog.client = MagicMock()
        catalog.client.download_burst_from_s3.side_effect = TypeError("bug")

        with pytest.raises(TypeError, match="bug"):
            catalog._download_burst_s3(_odata_burst())

    @pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT, errno.EROFS])
    @pytest.mark.parametrize("endpoint", ["odata", "s3"])
    def test_a_full_or_read_only_disk_fails_the_download(
        self,
        code: int,
        endpoint: Literal["odata", "s3"],
    ) -> None:
        """Every remaining burst would fail alike, so it is not recorded per burst."""
        catalog = _make_catalog()
        catalog.show_progress = False
        catalog.client = MagicMock()
        disk_error = OSError(code, os.strerror(code))
        catalog.client.download_burst.side_effect = disk_error
        catalog.client.download_burst_from_s3.side_effect = disk_error

        with pytest.raises(OSError, match=os.strerror(code)) as excinfo:
            catalog.download(_odata_burst(), endpoint=endpoint)

        assert excinfo.value is disk_error
        assert "Stopped with 1 of 1 bursts finished" in excinfo.value.__notes__[0]

    @pytest.mark.parametrize("endpoint", ["odata", "s3"])
    def test_another_os_error_is_recorded_per_burst(
        self,
        endpoint: Literal["odata", "s3"],
    ) -> None:
        catalog = _make_catalog()
        catalog.show_progress = False
        catalog.client = MagicMock()
        missing = FileNotFoundError(errno.ENOENT, "no such file")
        catalog.client.download_burst.side_effect = missing
        catalog.client.download_burst_from_s3.side_effect = missing

        summary = catalog.download(_odata_burst(), endpoint=endpoint)

        assert summary.n_failed == 1


class TestRejectedCredentials:
    @pytest.fixture
    def catalog(self, monkeypatch: pytest.MonkeyPatch) -> BurstCatalog:
        monkeypatch.setenv("CDSE_USERNAME", "user")
        monkeypatch.setenv("CDSE_PASSWORD", "pass")
        catalog = _make_catalog()
        catalog.show_progress = False
        catalog.client = MagicMock()
        catalog.client.auth = None
        return catalog

    def test_a_rejected_login_fails_the_download_once(self, catalog: BurstCatalog) -> None:
        """Previously recorded per burst as an HTTPError, logging in again for every burst."""
        rejected = ConfigurationError("CDSE rejected the username and password (HTTP 401).")

        with (
            patch("s1bursts.catalog.CDSEAuth", side_effect=rejected) as mock_auth_cls,
            pytest.raises(ConfigurationError, match="rejected") as excinfo,
        ):
            catalog.download([_odata_burst(day) for day in range(1, 7)], max_workers=3)

        assert str(excinfo.value) == str(rejected)
        mock_auth_cls.assert_called_once()
        mock_client = cast("MagicMock", catalog.client)
        mock_client.download_burst.assert_not_called()
        mock_client.download_burst_from_s3.assert_not_called()

    def test_the_next_download_logs_in_again(self, catalog: BurstCatalog) -> None:
        """Credentials corrected after a rejection are used by the next call."""
        with patch(
            "s1bursts.catalog.CDSEAuth",
            side_effect=[ConfigurationError("rejected"), MagicMock()],
        ) as mock_auth_cls:
            with pytest.raises(ConfigurationError, match="rejected"):
                catalog.download(_odata_burst())
            summary = catalog.download(_odata_burst())

        assert summary.n_processed == 1
        assert mock_auth_cls.call_count == 2

    def test_a_rejected_fallback_login_is_recorded_per_burst(self, catalog: BurstCatalog) -> None:
        """With S3 chosen, OData's rejected credentials must not stop S3 from working."""
        mock_client = cast("MagicMock", catalog.client)
        s3_error = OSError("connection reset")
        mock_client.download_burst_from_s3.side_effect = [None, s3_error, s3_error]
        rejected = ConfigurationError("CDSE rejected the username and password (HTTP 401).")

        with patch("s1bursts.catalog.CDSEAuth", side_effect=rejected) as mock_auth_cls:
            summary = catalog.download(
                [_odata_burst(day) for day in range(1, 4)],
                endpoint="s3",
                max_workers=1,
            )

        assert summary.n_processed == 1
        assert summary.n_failed == 2
        for failure in summary.failed_items:
            assert isinstance(failure.error, BurstDownloadError)
            assert str(failure.error) == (
                "S3: OSError: connection reset; OData: CDSE rejected the username and "
                "password (HTTP 401)."
            )
        # One login attempt, however many bursts fall back to OData.
        mock_auth_cls.assert_called_once()
        mock_client.download_burst.assert_not_called()

    def test_the_kept_rejection_is_never_the_raised_one(self, catalog: BurstCatalog) -> None:
        """A note added to the raised error must not reach the copy kept for later bursts."""
        with (
            patch("s1bursts.catalog.CDSEAuth", side_effect=ConfigurationError("rejected")),
            pytest.raises(ConfigurationError) as excinfo,
        ):
            catalog.download(_odata_burst())

        assert excinfo.value.__notes__
        kept = catalog._auth_rejected
        assert kept is not None
        assert kept is not excinfo.value
        assert not hasattr(kept, "__notes__")


class TestCachedAbsence:
    def test_a_second_download_reports_the_absence_without_the_first_note(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Previously the cached error was the one raised, so its note came back."""
        catalog = _make_catalog()
        catalog.show_progress = False
        catalog.continue_on_error = False
        _fail_both(
            monkeypatch,
            catalog,
            BurstNotFoundError("no such burst"),
            BurstNotFoundError("no such product"),
        )
        burst = _make_burst()

        with pytest.raises(BurstNotFoundError) as excinfo:
            catalog.download(burst)
        summary = catalog.download(burst)

        assert "Stopped with 1 of 1 bursts finished" in excinfo.value.__notes__[0]
        error = summary.failed_items[0].error
        assert isinstance(error, BurstNotFoundError)
        assert str(error) == str(excinfo.value)
        assert not hasattr(error, "__notes__")
        assert error.__cause__ is excinfo.value.__cause__


def _error_chain(error: BaseException) -> list[BaseException]:
    """Return ``error`` and every error it was raised from or during."""
    chain: list[BaseException] = []
    pending: list[BaseException | None] = [error]
    while pending:
        current = pending.pop()
        if current is None or any(current is seen for seen in chain):
            continue
        chain.append(current)
        pending += [current.__cause__, current.__context__]
    return chain


class TestTokenRequestFailure:
    def test_the_password_reaches_no_recorded_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Previously the token request's HTTPError, holding the request body, was recorded."""
        password = "hunter2-secret"
        monkeypatch.setenv("CDSE_USERNAME", "someone")
        monkeypatch.setenv("CDSE_PASSWORD", password)
        catalog = _make_catalog()
        catalog.show_progress = False
        catalog.client.auth = None
        request = Request(
            "POST",
            "https://identity.example/token",
            data={"username": "someone", "password": password},
        ).prepare()
        response = MagicMock()
        response.status_code = 500
        response.request = request
        response.raise_for_status.side_effect = HTTPError(
            "500 Server Error: Internal Server Error for url: https://identity.example/token",
            response=response,
        )
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            cdse_id="cdse-id",
        )

        with patch("s1bursts.cdse.requests.post", return_value=response):
            summary = catalog.download(burst)

        error = summary.failed_items[0].error
        assert "500 Server Error" in str(error)
        for link in _error_chain(error):
            assert getattr(link, "request", None) is None
            assert getattr(link, "response", None) is None
            texts = [str(link), repr(link.args), *getattr(link, "__notes__", [])]
            assert not any(password in text for text in texts)
