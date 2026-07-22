"""Tests for BurstDatabase (search filters, download dispatch, negative cache)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pandas import Timestamp
import pytest
from requests import HTTPError
from shapely.geometry import box

from s1bursts.burst import Burst, BurstPair
from s1bursts.database import BurstDatabase
from s1bursts.errors import BurstDownloadError, BurstNotFoundError
from s1bursts.query import SearchQuery
from s1bursts.summary import ErrorResult, OkResult


def _make_database() -> BurstDatabase:
    file_manager = MagicMock()
    file_manager.get_slc_path.return_value.exists.return_value = False
    return BurstDatabase(file_manager=file_manager, auth=MagicMock())


def _make_burst() -> Burst:
    return Burst(
        burst_id=297227,
        sub_swath="IW3",
        polarization="VV",
        date=Timestamp("2016-07-24"),
    )


class TestQueryToFilters:
    def test_simple_filters(self) -> None:
        query = SearchQuery(
            burst_id=30346,
            polarization="VV",
            mode="IW",
            sub_swath="IW1",
            orbit_direction="ASCENDING",
            platform_serial_id="A",
            rel_orbit_number=15,
        )

        filters = BurstDatabase._convert_query_to_filters(query)

        assert "BurstId eq 30346" in filters
        assert "PolarisationChannels eq 'VV'" in filters
        assert "OperationalMode eq 'IW'" in filters
        assert "SwathIdentifier eq 'IW1'" in filters
        assert "OrbitDirection eq 'ASCENDING'" in filters
        assert "PlatformSerialIdentifier eq 'A'" in filters
        assert "RelativeOrbitNumber eq 15" in filters

    def test_empty_query_produces_no_filters(self) -> None:
        assert BurstDatabase._convert_query_to_filters(SearchQuery()) == []

    def test_time_range_filter(self) -> None:
        query = SearchQuery(time_range=("2022-01-01", "2022-02-01"))

        filters = BurstDatabase._convert_query_to_filters(query)

        assert filters == [
            "ContentDate/Start ge 2022-01-01T00:00:00Z and "
            "ContentDate/Start le 2022-02-01T00:00:00Z",
        ]

    def test_intersects_filter(self) -> None:
        query = SearchQuery(intersects=box(7, 46, 8, 47))

        (filter_,) = BurstDatabase._convert_query_to_filters(query)

        assert filter_.startswith("OData.CSC.Intersects(area=geography'SRID=4326;POLYGON")


class TestDownloadDispatch:
    def test_rejects_unknown_type(self) -> None:
        db = _make_database()

        with pytest.raises(TypeError, match="Expected `bursts` of type"):
            db.download(["not-a-burst"])  # type: ignore[arg-type]

    def test_rejects_unknown_endpoint(self) -> None:
        db = _make_database()

        with pytest.raises(ValueError, match="must be 'odata' or 's3'"):
            db._download_single_burst(_make_burst(), "ftp")  # type: ignore[arg-type]

    def test_skips_existing_files_without_auth(self) -> None:
        file_manager = MagicMock()
        file_manager.get_slc_path.return_value.exists.return_value = True
        db = BurstDatabase(file_manager=file_manager)

        summary = db.download(_make_burst())

        assert summary.n_skipped == 1
        assert summary.n_processed == 0
        # No download happened, so no CDSEAuth may have been created
        assert db.client.auth is None

    def test_creates_auth_lazily_on_odata_download(self) -> None:
        """OData downloads build a CDSEAuth on first use, guarded to run once."""
        db = _make_database()
        db.client = MagicMock()
        db.client.auth = None
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            cdse_id="cdse-id",
        )

        with patch("s1bursts.database.CDSEAuth") as mock_auth_cls:
            result = db._download_burst_odata(burst)

        assert isinstance(result, OkResult)
        mock_auth_cls.assert_called_once()

    def test_s3_download_needs_no_odata_auth(self) -> None:
        """endpoint='s3' must not construct a CDSEAuth (S1-03)."""
        file_manager = MagicMock()
        file_manager.get_slc_path.return_value.exists.return_value = False
        db = BurstDatabase(file_manager=file_manager, default_endpoint="s3")
        db.client = MagicMock()
        db.client.auth = None
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            parent_product_name="S1A_IW_SLC__1SDV",
        )

        with patch("s1bursts.database.CDSEAuth") as mock_auth_cls:
            summary = db.download(burst, endpoint="s3")

        assert summary.n_processed == 1
        mock_auth_cls.assert_not_called()

    def test_burst_pair_downloads_both_bursts(self) -> None:
        db = _make_database()
        primary = _make_burst()
        secondary = Burst(
            burst_id=primary.burst_id,
            sub_swath=primary.sub_swath,
            polarization=primary.polarization,
            date=Timestamp("2016-08-05"),
        )
        pair = BurstPair(primary, secondary)

        with (
            patch.object(db, "_ensure_auth"),
            patch.object(
                db,
                "_download_single_burst",
                side_effect=lambda b, _e: OkResult(source=b, result=b),
            ) as mock_download,
        ):
            summary = db.download(pair)

        assert mock_download.call_count == 2
        assert summary.n_processed == 2


class TestEndpointWrappers:
    def test_odata_requires_cdse_id(self) -> None:
        db = _make_database()
        burst = _make_burst()  # has no cdse_id

        result = db._download_burst_odata(burst)

        assert isinstance(result, ErrorResult)
        assert "requires `cdse_id`" in str(result.error)

    def test_odata_success(self) -> None:
        db = _make_database()
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            cdse_id="cdse-id",
        )
        db.client = MagicMock()

        result = db._download_burst_odata(burst)

        assert isinstance(result, OkResult)
        db.client.download_burst.assert_called_once()

    def test_odata_http_error_becomes_error_result(self) -> None:
        db = _make_database()
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            cdse_id="cdse-id",
        )
        db.client = MagicMock()
        db.client.download_burst.side_effect = HTTPError("503 Service Unavailable")

        result = db._download_burst_odata(burst)

        assert isinstance(result, ErrorResult)
        assert isinstance(result.error, HTTPError)

    def test_odata_without_credentials_returns_clear_error(self) -> None:
        """A missing-credentials failure surfaces as a per-burst BurstDownloadError."""
        file_manager = MagicMock()
        file_manager.get_slc_path.return_value.exists.return_value = False
        db = BurstDatabase(file_manager=file_manager)
        db.client.auth = None
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            cdse_id="cdse-id",
        )

        with patch(
            "s1bursts.database.CDSEAuth",
            side_effect=ValueError("CDSE credentials not configured"),
        ):
            result = db._download_burst_odata(burst)

        assert isinstance(result, ErrorResult)
        assert isinstance(result.error, BurstDownloadError)
        assert "endpoint='s3'" in str(result.error)

    def test_s3_requires_parent_product_name(self) -> None:
        db = _make_database()
        burst = _make_burst()  # has no parent_product_name

        result = db._download_burst_s3(burst)

        assert isinstance(result, ErrorResult)
        assert "requires `parent_product_name`" in str(result.error)

    def test_s3_success(self) -> None:
        db = _make_database()
        burst = Burst(
            burst_id=297227,
            sub_swath="IW3",
            polarization="VV",
            date=Timestamp("2016-07-24"),
            parent_product_name="S1A_IW_SLC__1SDV",
        )
        db.client = MagicMock()

        result = db._download_burst_s3(burst)

        assert isinstance(result, OkResult)
        db.client.download_burst_from_s3.assert_called_once()


class TestValidateCompleteness:
    def test_complete_response_passes(self) -> None:
        response = MagicMock()
        response.is_complete.return_value = True

        BurstDatabase._validate_completeness(response)

    def test_truncated_response_raises(self) -> None:
        response = MagicMock()
        response.is_complete.return_value = False
        response.count = 500
        response.__len__.return_value = 100

        with pytest.raises(RuntimeError, match="Increase `max_items`"):
            BurstDatabase._validate_completeness(response)


def test_burst_not_found_is_cached_and_short_circuits() -> None:
    """A deterministic BurstNotFoundError should not be re-fetched on the next request."""
    db = _make_database()
    burst = _make_burst()

    with (
        patch.object(
            db,
            "_download_burst_odata",
            return_value=ErrorResult(source=burst, error=HTTPError("500 Server Error")),
        ) as mock_primary,
        patch.object(
            db,
            "_download_burst_s3",
            return_value=ErrorResult(source=burst, error=BurstNotFoundError("not found")),
        ) as mock_fallback,
    ):
        first = db._download_single_burst(burst, "odata")
        second = db._download_single_burst(burst, "odata")

    assert isinstance(first, ErrorResult)
    assert isinstance(first.error, BurstNotFoundError)
    assert isinstance(second, ErrorResult)
    assert isinstance(second.error, BurstNotFoundError)

    # Second call must hit the cache, not the download methods again.
    mock_primary.assert_called_once()
    mock_fallback.assert_called_once()


def test_transient_failure_is_not_cached() -> None:
    """A failure with no BurstNotFoundError anywhere should keep retrying."""
    db = _make_database()
    burst = _make_burst()

    with (
        patch.object(
            db,
            "_download_burst_odata",
            return_value=ErrorResult(source=burst, error=HTTPError("500 Server Error")),
        ) as mock_primary,
        patch.object(
            db,
            "_download_burst_s3",
            return_value=ErrorResult(source=burst, error=HTTPError("connection reset")),
        ) as mock_fallback,
    ):
        db._download_single_burst(burst, "odata")
        db._download_single_burst(burst, "odata")

    # No BurstNotFoundError was ever raised, so nothing should have been cached.
    assert mock_primary.call_count == 2
    assert mock_fallback.call_count == 2


def test_burst_not_found_from_primary_endpoint_is_cached() -> None:
    """BurstNotFoundError from the primary attempt (not just the fallback) must cache too."""
    db = _make_database()
    burst = _make_burst()

    with (
        patch.object(
            db,
            "_download_burst_s3",
            return_value=ErrorResult(source=burst, error=BurstNotFoundError("not found")),
        ) as mock_primary,
        patch.object(
            db,
            "_download_burst_odata",
            return_value=ErrorResult(source=burst, error=HTTPError("500 Server Error")),
        ) as mock_fallback,
    ):
        first = db._download_single_burst(burst, "s3")
        second = db._download_single_burst(burst, "s3")

    # The first call still returns the terminal (fallback) result, matching
    # existing behavior — but the BurstNotFoundError surfaced by the primary
    # attempt is what gets cached.
    assert isinstance(first, ErrorResult)
    assert isinstance(first.error, HTTPError)
    assert isinstance(second, ErrorResult)
    assert isinstance(second.error, BurstNotFoundError)
    mock_primary.assert_called_once()
    mock_fallback.assert_called_once()
