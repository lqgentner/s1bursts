"""Tests for BurstCatalog (search filters, download dispatch, negative cache)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Literal
from unittest.mock import MagicMock, patch

from freezebase.errors import DownloadError
from pandas import Timestamp
import pytest
from requests import HTTPError
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
from s1bursts.summary import ErrorResult, OkResult
from s1bursts.time_range import TimeRange

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

    def test_creates_auth_lazily_on_odata_download(self) -> None:
        """OData downloads build a CDSEAuth on first use, guarded to run once."""
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

    def test_odata_without_credentials_returns_clear_error(self) -> None:
        """A missing-credentials failure surfaces as a per-burst BurstDownloadError."""
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

        with patch(
            "s1bursts.catalog.CDSEAuth",
            side_effect=ConfigurationError("CDSE credentials not configured"),
        ):
            result = catalog._download_burst_odata(burst)

        assert isinstance(result, ErrorResult)
        assert isinstance(result.error, BurstDownloadError)
        assert "endpoint='s3'" in str(result.error)

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


def test_burst_not_found_is_cached_and_short_circuits() -> None:
    """A deterministic BurstNotFoundError should not be re-fetched on the next request."""
    catalog = _make_catalog()
    burst = _make_burst()

    with (
        patch.object(
            catalog,
            "_download_burst_odata",
            return_value=ErrorResult(source=burst, error=HTTPError("500 Server Error")),
        ) as mock_primary,
        patch.object(
            catalog,
            "_download_burst_s3",
            return_value=ErrorResult(source=burst, error=BurstNotFoundError("not found")),
        ) as mock_fallback,
    ):
        first = catalog._download_single_burst(burst, "odata")
        second = catalog._download_single_burst(burst, "odata")

    assert isinstance(first, ErrorResult)
    assert isinstance(first.error, BurstNotFoundError)
    assert isinstance(second, ErrorResult)
    assert isinstance(second.error, BurstNotFoundError)

    # Second call must hit the cache, not the download methods again.
    mock_primary.assert_called_once()
    mock_fallback.assert_called_once()


def test_transient_failure_is_not_cached() -> None:
    """A failure with no BurstNotFoundError anywhere should keep retrying."""
    catalog = _make_catalog()
    burst = _make_burst()

    with (
        patch.object(
            catalog,
            "_download_burst_odata",
            return_value=ErrorResult(source=burst, error=HTTPError("500 Server Error")),
        ) as mock_primary,
        patch.object(
            catalog,
            "_download_burst_s3",
            return_value=ErrorResult(source=burst, error=HTTPError("connection reset")),
        ) as mock_fallback,
    ):
        catalog._download_single_burst(burst, "odata")
        catalog._download_single_burst(burst, "odata")

    # No BurstNotFoundError was ever raised, so nothing should have been cached.
    assert mock_primary.call_count == 2
    assert mock_fallback.call_count == 2


def test_burst_not_found_from_primary_endpoint_is_cached() -> None:
    """BurstNotFoundError from the primary attempt (not just the fallback) must cache too."""
    catalog = _make_catalog()
    burst = _make_burst()

    with (
        patch.object(
            catalog,
            "_download_burst_s3",
            return_value=ErrorResult(source=burst, error=BurstNotFoundError("not found")),
        ) as mock_primary,
        patch.object(
            catalog,
            "_download_burst_odata",
            return_value=ErrorResult(source=burst, error=HTTPError("500 Server Error")),
        ) as mock_fallback,
    ):
        first = catalog._download_single_burst(burst, "s3")
        second = catalog._download_single_burst(burst, "s3")

    # The first call still returns the terminal (fallback) result, matching
    # existing behavior — but the BurstNotFoundError surfaced by the primary
    # attempt is what gets cached.
    assert isinstance(first, ErrorResult)
    assert isinstance(first.error, HTTPError)
    assert isinstance(second, ErrorResult)
    assert isinstance(second.error, BurstNotFoundError)
    mock_primary.assert_called_once()
    mock_fallback.assert_called_once()
