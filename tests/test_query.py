"""Tests for s1bursts.query.SearchQuery."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas import Timestamp
import pytest
from shapely.geometry import box

from s1bursts.query import SearchQuery
from s1bursts.time_range import TimeRange


class TestTimeRange:
    def test_converts_strings_to_timestamps(self) -> None:
        query = SearchQuery(time_range=("2022-01-01", "2022-02-01"))

        assert query.time_range == (Timestamp("2022-01-01"), Timestamp("2022-02-01"))

    def test_accepts_timestamps(self) -> None:
        start, end = Timestamp("2022-01-01"), Timestamp("2022-02-01")
        query = SearchQuery(time_range=(start, end))

        assert query.time_range == (start, end)

    def test_a_window_becomes_the_half_open_instants_covering_its_days(self) -> None:
        query = SearchQuery(time_range=TimeRange("2022-01-01", "2022-01-31"))

        assert query.time_range == (Timestamp("2022-01-01"), Timestamp("2022-02-01"))

    def test_none_stays_none(self) -> None:
        assert SearchQuery().time_range is None

    def test_rejects_non_sequence(self) -> None:
        with pytest.raises(TypeError, match="must be of type tuple"):
            # pyrefly: ignore [bad-argument-type]
            SearchQuery(time_range=pd.Timestamp("2022-01-01"))

    @pytest.mark.parametrize(
        "time_range",
        [("2022-01-01",), ("2022-01-01", "2022-02-01", "2022-03-01")],
    )
    def test_rejects_wrong_length(self, time_range: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match="exactly 2 elements"):
            # pyrefly: ignore [bad-argument-type]
            SearchQuery(time_range=time_range)

    def test_rejects_start_after_end(self) -> None:
        with pytest.raises(ValueError, match="must be before end time"):
            SearchQuery(time_range=("2022-02-01", "2022-01-01"))

    def test_rejects_start_equal_end(self) -> None:
        with pytest.raises(ValueError, match="must be before end time"):
            SearchQuery(time_range=("2022-01-01", "2022-01-01"))


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
