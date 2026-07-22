"""Tests for s1bursts.burst data structures and collections."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import geopandas as gpd
from pandas import Timedelta, Timestamp
import pytest
from shapely.geometry import box

from s1bursts.burst import (
    Burst,
    BurstCollection,
    BurstPair,
    BurstPairCollection,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

BURST_ID = 30346
SUB_SWATH = "IW1"
GEOMETRY = box(7.0, 46.0, 8.0, 47.0)


@pytest.fixture(autouse=True)
def _patched_luts() -> Iterator[None]:
    """Replace the burst-map lookup tables so no download is required."""
    key = (BURST_ID, SUB_SWATH)
    with (
        patch(
            "s1bursts.burst.create_s1_geometry_lut",
            return_value={key: GEOMETRY},
        ),
        patch(
            "s1bursts.burst.create_s1_orbit_number_lut",
            return_value={key: 15},
        ),
        patch(
            "s1bursts.burst.create_s1_orbit_direction_lut",
            return_value={key: "ASCENDING"},
        ),
    ):
        yield


def make_burst(
    date: str = "2022-01-01",
    burst_id: int = BURST_ID,
    sub_swath: str = SUB_SWATH,
    polarization: str = "VV",
) -> Burst:
    return Burst(
        burst_id=burst_id,
        sub_swath=sub_swath,  # type: ignore[arg-type]
        polarization=polarization,  # type: ignore[arg-type]
        date=Timestamp(date),
    )


class TestBurst:
    def test_footprint_properties_use_lut(self) -> None:
        burst = make_burst()

        assert burst.rel_orbit_number == 15
        assert burst.orbit_direction == "ASCENDING"
        assert burst.geometry is GEOMETRY

    def test_full_burst_id(self) -> None:
        assert make_burst().full_burst_id == "015_030346_IW1"

    @pytest.mark.parametrize(
        ("sub_swath", "expected_mode"),
        [
            ("IW1", "IW"),
            ("IW3", "IW"),
            ("EW1", "EW"),
            ("EW5", "EW"),
        ],
    )
    def test_mode_derived_from_subswath(self, sub_swath: str, expected_mode: str) -> None:
        assert make_burst(sub_swath=sub_swath).mode == expected_mode

    def test_ew_burst_queries_ew_lut(self) -> None:
        """EW bursts must query the EW burst map, not the IW default (S1-01)."""
        ew_key = (12345, "EW2")
        ew_geometry = box(10.0, 70.0, 12.0, 71.0)
        with (
            patch("s1bursts.burst.create_s1_geometry_lut") as geom_lut,
            patch("s1bursts.burst.create_s1_orbit_number_lut") as orbit_num_lut,
            patch("s1bursts.burst.create_s1_orbit_direction_lut") as orbit_dir_lut,
        ):
            geom_lut.return_value = {ew_key: ew_geometry}
            orbit_num_lut.return_value = {ew_key: 42}
            orbit_dir_lut.return_value = {ew_key: "DESCENDING"}
            burst = make_burst(burst_id=12345, sub_swath="EW2")

            assert burst.geometry is ew_geometry
            assert burst.rel_orbit_number == 42
            assert burst.orbit_direction == "DESCENDING"
            assert burst.full_burst_id == "042_012345_EW2"

        geom_lut.assert_called_once_with(mode="EW")
        orbit_num_lut.assert_called_with(mode="EW")
        orbit_dir_lut.assert_called_once_with(mode="EW")

    def test_unknown_burst_raises_key_error(self) -> None:
        burst = make_burst(burst_id=99999)

        with pytest.raises(KeyError, match="not found in the Sentinel-1 burst map"):
            _ = burst.rel_orbit_number

    def test_from_dict_drops_unknown_keys(self) -> None:
        burst = Burst.from_dict(
            {
                "burst_id": BURST_ID,
                "sub_swath": SUB_SWATH,
                "polarization": "VV",
                "date": Timestamp("2022-01-01"),
                "unknown_key": "ignored",
            },
        )

        assert burst.burst_id == BURST_ID

    def test_repr(self) -> None:
        expected = (
            "Burst(burst_id='030346', sub_swath='IW1', polarization='VV', date='2022-01-01')"
        )
        assert repr(make_burst()) == expected

    def test_equality_ignores_provenance_fields(self) -> None:
        one = make_burst()
        other = Burst(
            burst_id=BURST_ID,
            sub_swath=SUB_SWATH,
            polarization="VV",
            date=Timestamp("2022-01-01"),
            cdse_id="some-id",
            parent_product_name="some-product",
        )

        assert one == other


class TestBurstPair:
    def test_valid_pair(self) -> None:
        pair = BurstPair(make_burst("2022-01-01"), make_burst("2022-01-13"))

        assert pair.baseline_days == 12
        assert pair.burst_id == BURST_ID
        assert pair.sub_swath == SUB_SWATH
        assert pair.polarization == "VV"
        assert pair.rel_orbit_number == 15
        assert pair.orbit_direction == "ASCENDING"
        assert pair.geometry is GEOMETRY
        assert pair.full_burst_id == "015_030346_IW1"

    def test_mismatched_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="Burst parameters don't match"):
            BurstPair(make_burst(polarization="VV"), make_burst("2022-01-13", polarization="VH"))

    def test_non_positive_baseline_raises(self) -> None:
        with pytest.raises(ValueError, match="positive non-zero baseline"):
            BurstPair(make_burst("2022-01-13"), make_burst("2022-01-01"))


def make_collection(dates: list[str], polarization: str = "VV") -> BurstCollection:
    return BurstCollection.from_bursts([make_burst(d, polarization=polarization) for d in dates])


class TestBurstCollection:
    def test_empty_collection(self) -> None:
        collection = BurstCollection()

        assert len(collection) == 0
        assert list(collection) == []

    def test_from_bursts_roundtrip(self) -> None:
        collection = make_collection(["2022-01-01", "2022-01-13"])

        assert len(collection) == 2
        assert [b.date for b in collection] == [Timestamp("2022-01-01"), Timestamp("2022-01-13")]

    def test_rejects_plain_dataframe(self) -> None:
        with pytest.raises(TypeError, match="must be a GeoDataFrame"):
            BurstCollection(gpd.pd.DataFrame({"burst_id": [1]}))  # type: ignore[arg-type]

    def test_rejects_missing_columns(self) -> None:
        df = gpd.GeoDataFrame({"burst_id": [1], "geometry": [GEOMETRY]})

        with pytest.raises(ValueError, match="missing required columns"):
            BurstCollection(df)

    def test_warns_on_unexpected_columns(self) -> None:
        df = make_collection(["2022-01-01"]).df.assign(extra=1)

        with pytest.warns(UserWarning, match="unexpected columns"):
            collection = BurstCollection(gpd.GeoDataFrame(df))

        assert "extra" not in collection.df.columns

    def test_deduplicates_entries(self) -> None:
        collection = make_collection(["2022-01-01", "2022-01-01"])

        assert len(collection) == 1

    def test_add_concatenates_and_deduplicates(self) -> None:
        first = make_collection(["2022-01-01", "2022-01-13"])
        second = make_collection(["2022-01-13", "2022-01-25"])

        combined = first + second

        assert len(combined) == 3

    def test_getitem_int_and_slice(self) -> None:
        collection = make_collection(["2022-01-01", "2022-01-13", "2022-01-25"])

        assert isinstance(collection[0], Burst)
        assert collection[0].date == Timestamp("2022-01-01")
        sliced = collection[1:]
        assert isinstance(sliced, BurstCollection)
        assert len(sliced) == 2

    def test_removes_near_identical_timestamps(self) -> None:
        bursts = [
            make_burst("2022-01-01 05:00:00"),
            make_burst("2022-01-01 05:00:01"),
            make_burst("2022-01-13 05:00:00"),
        ]

        with pytest.warns(UserWarning, match="nearly identical timestamp"):
            collection = BurstCollection.from_bursts(bursts)

        assert len(collection) == 2
        # The more recent of the two near-duplicates is kept
        assert Timestamp("2022-01-01 05:00:01") in list(collection.df.date)

    def test_find_pairs_with_12_day_baseline(self) -> None:
        collection = make_collection(["2022-01-01", "2022-01-13", "2022-01-25"])

        pairs = collection.find_pairs(baseline=12)

        assert len(pairs) == 2
        for pair in pairs:
            assert pair.baseline_days == 12

    def test_find_pairs_spanning_gaps(self) -> None:
        # 24-day baseline must pair bursts two acquisitions apart
        collection = make_collection(["2022-01-01", "2022-01-13", "2022-01-25"])

        pairs = collection.find_pairs(baseline=Timedelta(days=24))

        assert len(pairs) == 1
        assert pairs[0].baseline_days == 24

    def test_find_pairs_does_not_mix_polarizations(self) -> None:
        vv = make_collection(["2022-01-01"], polarization="VV")
        vh = make_collection(["2022-01-13"], polarization="VH")

        pairs = (vv + vh).find_pairs(baseline=12)

        assert len(pairs) == 0

    def test_find_pairs_no_match_returns_empty(self) -> None:
        collection = make_collection(["2022-01-01", "2022-01-13"])

        pairs = collection.find_pairs(baseline=6)

        assert isinstance(pairs, BurstPairCollection)
        assert len(pairs) == 0

    def test_repr_html_smoke(self) -> None:
        html = make_collection(["2022-01-01"])._repr_html_()

        assert "BurstCollection" in html
        assert "<svg" in html


class TestBurstPairCollection:
    def make_pairs(self) -> BurstPairCollection:
        return BurstPairCollection.from_pairs(
            [
                BurstPair(make_burst("2022-01-01"), make_burst("2022-01-13")),
                BurstPair(make_burst("2022-01-13"), make_burst("2022-01-25")),
            ],
        )

    def test_from_pairs_roundtrip(self) -> None:
        pairs = self.make_pairs()

        assert len(pairs) == 2
        assert all(isinstance(p, BurstPair) for p in pairs)

    def test_empty(self) -> None:
        assert len(BurstPairCollection.from_pairs([])) == 0

    def test_unpair_returns_unique_bursts(self) -> None:
        bursts = self.make_pairs().unpair()

        assert isinstance(bursts, BurstCollection)
        # 2 pairs share one burst -> 3 unique bursts
        assert len(bursts) == 3
