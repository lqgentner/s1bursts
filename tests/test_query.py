"""Tests for s1bursts.query.SearchQuery."""

from __future__ import annotations

from typing import Any

from pandas import Timestamp
import pytest
from shapely.geometry import box

from s1bursts.query import SearchQuery
from s1bursts.time_range import TimeRange


class TestTimeRange:
    def test_stores_the_window_as_given(self) -> None:
        window = TimeRange("2022-01-01", "2022-01-31")

        query = SearchQuery(time_range=window)

        assert query.time_range is window

    def test_none_stays_none(self) -> None:
        assert SearchQuery().time_range is None

    @pytest.mark.parametrize(
        "time_range",
        [
            ("2022-01-01", "2022-02-01"),
            (Timestamp("2022-01-01"), Timestamp("2022-02-01")),
            Timestamp("2022-01-01"),
        ],
    )
    def test_rejects_anything_but_a_time_range(self, time_range: object) -> None:
        # Previously a tuple was accepted as a half-open pair of instants, which
        # dropped every acquisition on the day written as its end.
        with pytest.raises(TypeError, match=r"must be a TimeRange.*inclusive"):
            # The test passes a non-TimeRange on purpose.
            # pyrefly: ignore [bad-argument-type]
            SearchQuery(time_range=time_range)

    def test_repr_shows_the_window_days(self) -> None:
        query = SearchQuery(time_range=TimeRange("2022-01-01", "2022-01-31"))

        assert repr(query) == "SearchQuery(time_range=2022-01-01..2022-01-31)"


class TestGeometry:
    def test_accepts_shapely_geometry(self) -> None:
        geom = box(7.0, 46.0, 8.0, 47.0)
        query = SearchQuery(intersects=geom)

        assert query.intersects is geom

    def test_rejects_object_without_wkt(self) -> None:
        with pytest.raises(ValueError, match="implements `wkt`"):
            # pyrefly: ignore [bad-argument-type]
            SearchQuery(intersects="POINT (1 2)")


class TestIntValidation:
    @pytest.mark.parametrize("field", ["burst_id", "abs_burst_id", "rel_orbit_number"])
    @pytest.mark.parametrize("value", [-1, 0])
    def test_rejects_non_positive(self, field: str, value: int) -> None:
        """Burst and orbit identifiers are 1-based, so zero is as invalid as a negative."""
        kwargs: dict[str, Any] = {field: value}
        with pytest.raises(ValueError, match=f"{field} must be"):
            SearchQuery(**kwargs)

    def test_accepts_the_full_relative_orbit_range(self) -> None:
        assert SearchQuery(rel_orbit_number=1).rel_orbit_number == 1
        assert SearchQuery(rel_orbit_number=175).rel_orbit_number == 175

    def test_rejects_relative_orbit_above_the_repeat_cycle(self) -> None:
        """Sentinel-1's repeat cycle has 175 relative orbits; 176 cannot exist."""
        with pytest.raises(ValueError, match="rel_orbit_number must be 1-175"):
            SearchQuery(rel_orbit_number=176)

    def test_accepts_positive_identifiers(self) -> None:
        query = SearchQuery(burst_id=30346, abs_burst_id=5, rel_orbit_number=15)

        assert query.burst_id == 30346
        assert query.abs_burst_id == 5
        assert query.rel_orbit_number == 15


class TestChoiceValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("polarization", "VVV"),
            ("mode", "SM"),
            ("sub_swath", "IW4"),
            ("pass_direction", "UP"),
            ("platform_serial_id", "Z"),
        ],
    )
    def test_rejects_unknown_identifier(self, field: str, value: str) -> None:
        """A typo must fail here, not silently return an empty result set from CDSE."""
        kwargs: dict[str, Any] = {field: value}
        with pytest.raises(ValueError, match=f"{field} must be one of"):
            SearchQuery(**kwargs)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("polarization", "HV"),
            ("mode", "EW"),
            ("sub_swath", "EW5"),
            ("pass_direction", "Descending"),
            ("platform_serial_id", "C"),
        ],
    )
    def test_accepts_every_documented_identifier(self, field: str, value: str) -> None:
        kwargs: dict[str, Any] = {field: value}
        assert getattr(SearchQuery(**kwargs), field) == value

    def test_is_case_sensitive(self) -> None:
        """CDSE's OData filters are case-sensitive, so lower case cannot be accepted."""
        with pytest.raises(ValueError, match="polarization must be one of"):
            # Deliberately lower case.
            # pyrefly: ignore [bad-argument-type]
            SearchQuery(polarization="vv")


class TestRepr:
    def test_only_shows_non_none_parameters(self) -> None:
        query = SearchQuery(burst_id=30346, polarization="VV")

        assert repr(query) == "SearchQuery(burst_id=30346, polarization=VV)"

    def test_empty_query(self) -> None:
        assert repr(SearchQuery()) == "SearchQuery()"
