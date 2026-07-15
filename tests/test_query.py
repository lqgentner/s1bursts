"""Tests for s1bursts.query.SearchQuery."""

from __future__ import annotations

import pandas as pd
from pandas import Timestamp
import pytest
from shapely.geometry import box

from s1bursts.query import SearchQuery


class TestTimeRange:
    def test_converts_strings_to_timestamps(self) -> None:
        query = SearchQuery(time_range=("2022-01-01", "2022-02-01"))

        assert query.time_range == (Timestamp("2022-01-01"), Timestamp("2022-02-01"))

    def test_accepts_timestamps(self) -> None:
        start, end = Timestamp("2022-01-01"), Timestamp("2022-02-01")
        query = SearchQuery(time_range=(start, end))

        assert query.time_range == (start, end)

    def test_none_stays_none(self) -> None:
        assert SearchQuery().time_range is None

    def test_rejects_non_sequence(self) -> None:
        with pytest.raises(TypeError, match="must be of type tuple"):
            SearchQuery(time_range=pd.Timestamp("2022-01-01"))  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "time_range",
        [("2022-01-01",), ("2022-01-01", "2022-02-01", "2022-03-01")],
    )
    def test_rejects_wrong_length(self, time_range: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match="exactly 2 elements"):
            SearchQuery(time_range=time_range)  # type: ignore[arg-type]

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
            SearchQuery(intersects="POINT (1 2)")  # type: ignore[arg-type]


class TestIntValidation:
    @pytest.mark.parametrize("field", ["burst_id", "abs_burst_id", "rel_orbit_number"])
    def test_rejects_negative(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be non-negative"):
            SearchQuery(**{field: -1})

    def test_accepts_zero_and_positive(self) -> None:
        query = SearchQuery(burst_id=0, abs_burst_id=5, rel_orbit_number=175)

        assert query.burst_id == 0
        assert query.abs_burst_id == 5
        assert query.rel_orbit_number == 175


class TestRepr:
    def test_only_shows_non_none_parameters(self) -> None:
        query = SearchQuery(burst_id=30346, polarization="VV")

        assert repr(query) == "SearchQuery(burst_id=30346, polarization=VV)"

    def test_empty_query(self) -> None:
        assert repr(SearchQuery()) == "SearchQuery()"
