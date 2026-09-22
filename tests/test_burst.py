"""Tests for s1bursts.burst data structures and collections."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import geopandas as gpd
from pandas import Timedelta, Timestamp
from pyproj import CRS
import pytest
from shapely.geometry import box

from s1bursts.burst import (
    Burst,
    BurstCollection,
    BurstPair,
    BurstPairCollection,
)
from s1bursts.cdse import CDSEResponse
from s1bursts.product import CoherenceProduct

if TYPE_CHECKING:
    from collections.abc import Iterator

    from shapely.geometry import MultiPolygon, Polygon

    from s1bursts.query import Polarization, SubSwath

BURST_ID = 30346
SUB_SWATH: SubSwath = "IW1"
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
            "s1bursts.burst.create_s1_pass_direction_lut",
            return_value={key: "Ascending"},
        ),
    ):
        yield


def make_burst(
    date: str = "2022-01-01",
    burst_id: int = BURST_ID,
    sub_swath: SubSwath = SUB_SWATH,
    polarization: Polarization = "VV",
    geometry: Polygon | MultiPolygon | None = None,
) -> Burst:
    return Burst(
        burst_id=burst_id,
        sub_swath=sub_swath,
        polarization=polarization,
        date=Timestamp(date),
        _geometry=geometry,
    )


class TestBurst:
    def test_footprint_properties_use_lut(self) -> None:
        burst = make_burst()

        assert burst.rel_orbit_number == 15
        assert burst.pass_direction == "Ascending"
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
    def test_mode_derived_from_subswath(self, sub_swath: SubSwath, expected_mode: str) -> None:
        assert make_burst(sub_swath=sub_swath).mode == expected_mode

    def test_ew_burst_queries_ew_lut(self) -> None:
        """EW bursts must query the EW burst map, not the IW default (S1-01)."""
        ew_key = (12345, "EW2")
        ew_geometry = box(10.0, 70.0, 12.0, 71.0)
        with (
            patch("s1bursts.burst.create_s1_geometry_lut") as geom_lut,
            patch("s1bursts.burst.create_s1_orbit_number_lut") as orbit_num_lut,
            patch("s1bursts.burst.create_s1_pass_direction_lut") as orbit_dir_lut,
        ):
            geom_lut.return_value = {ew_key: ew_geometry}
            orbit_num_lut.return_value = {ew_key: 42}
            orbit_dir_lut.return_value = {ew_key: "Descending"}
            burst = make_burst(burst_id=12345, sub_swath="EW2")

            assert burst.geometry is ew_geometry
            assert burst.rel_orbit_number == 42
            assert burst.pass_direction == "Descending"
            assert burst.full_burst_id == "042_012345_EW2"

        geom_lut.assert_called_once_with(mode="EW")
        orbit_num_lut.assert_called_with(mode="EW")
        orbit_dir_lut.assert_called_once_with(mode="EW")

    def test_unknown_burst_raises_key_error(self) -> None:
        burst = make_burst(burst_id=99999)

        with pytest.raises(KeyError, match="not found in the Sentinel-1 burst map"):
            _ = burst.rel_orbit_number

    def test_propagated_values_bypass_the_lut(self) -> None:
        """Stored footprint overrides are returned without consulting the LUT."""
        burst = Burst(
            burst_id=BURST_ID,
            sub_swath="IW1",
            polarization="VV",
            date=Timestamp("2022-01-01"),
            _rel_orbit_number=88,
            _pass_direction="Descending",
            _geometry=GEOMETRY,
        )
        with (
            patch("s1bursts.burst.create_s1_orbit_number_lut", side_effect=AssertionError),
            patch("s1bursts.burst.create_s1_pass_direction_lut", side_effect=AssertionError),
            patch("s1bursts.burst.create_s1_geometry_lut", side_effect=AssertionError),
        ):
            assert burst.rel_orbit_number == 88
            assert burst.pass_direction == "Descending"
            assert burst.geometry is GEOMETRY
            assert burst.full_burst_id == "088_030346_IW1"

    def test_from_dict_propagates_rel_orbit_from_filename(self) -> None:
        """A burst reconstructed from a parsed SLC filename keeps its relative orbit."""
        burst = Burst.from_dict(
            {
                "burst_id": BURST_ID,
                "sub_swath": "IW1",
                "polarization": "VV",
                "date": Timestamp("2022-01-01"),
                "rel_orbit_number": 123,
            },
        )
        with patch("s1bursts.burst.create_s1_orbit_number_lut", side_effect=AssertionError):
            assert burst.rel_orbit_number == 123

    def test_equality_normalizes_date_to_day(self) -> None:
        """A CDSE burst (full time) equals its filesystem twin (midnight)."""
        with_time = make_burst("2022-01-01 05:00:00")
        midnight = make_burst("2022-01-01")

        assert with_time == midnight
        assert hash(with_time) == hash(midnight)

    def test_inequality_across_days(self) -> None:
        assert make_burst("2022-01-01") != make_burst("2022-01-02")

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


class TestFromCdseResponse:
    @staticmethod
    def _response(rel_orbit: int = 88, direction: str = "DESCENDING") -> CDSEResponse:
        value = {
            "Id": "some-id",
            "Name": "some-name",
            "Footprint": "SRID=4326;POLYGON ((7 46, 8 46, 8 47, 7 47, 7 46))",
            "BurstId": BURST_ID,
            "AbsoluteBurstId": 123456,
            "SwathIdentifier": "IW1",
            "PolarisationChannels": "VV",
            "ParentProductName": "S1A_IW_SLC__1SDV",
            "BeginningDateTime": "2022-01-01T05:00:00.000Z",
            "EndingDateTime": "2022-01-01T05:00:03.000Z",
            "AzimuthTime": "2022-01-01T05:00:01.000Z",
            "RelativeOrbitNumber": rel_orbit,
            "OrbitDirection": direction,
        }
        return CDSEResponse(context="$metadata#Bursts", value=[value], count=1)

    def test_propagates_cdse_metadata_without_lut(self) -> None:
        """Orbit, pass direction and geometry from a CDSE search never hit the burst-map LUT."""
        collection = BurstCollection.from_cdse_response(self._response())
        (burst,) = list(collection)
        with (
            patch("s1bursts.burst.create_s1_orbit_number_lut", side_effect=AssertionError),
            patch("s1bursts.burst.create_s1_pass_direction_lut", side_effect=AssertionError),
            patch("s1bursts.burst.create_s1_geometry_lut", side_effect=AssertionError),
        ):
            assert burst.rel_orbit_number == 88
            assert burst.pass_direction == "Descending"
            assert burst.geometry.bounds == (7.0, 46.0, 8.0, 47.0)


class TestBurstPair:
    def test_valid_pair(self) -> None:
        pair = BurstPair(make_burst("2022-01-01"), make_burst("2022-01-13"))

        assert pair.baseline_days == 12
        assert pair.burst_id == BURST_ID
        assert pair.sub_swath == SUB_SWATH
        assert pair.polarization == "VV"
        assert pair.rel_orbit_number == 15
        assert pair.pass_direction == "Ascending"
        assert pair.geometry is GEOMETRY
        assert pair.full_burst_id == "015_030346_IW1"

    def test_mismatched_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="Burst parameters don't match"):
            BurstPair(make_burst(polarization="VV"), make_burst("2022-01-13", polarization="VH"))

    def test_non_positive_baseline_raises(self) -> None:
        with pytest.raises(ValueError, match="positive non-zero baseline"):
            BurstPair(make_burst("2022-01-13"), make_burst("2022-01-01"))


def make_collection(dates: list[str], polarization: Polarization = "VV") -> BurstCollection:
    return BurstCollection.from_bursts(
        [make_burst(d, polarization=polarization, geometry=GEOMETRY) for d in dates],
    )


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
            # pyrefly: ignore [bad-argument-type]
            BurstCollection(gpd.pd.DataFrame({"burst_id": [1]}))

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
        # Without dimensions, the more recent of the two near-duplicates is kept.
        assert Timestamp("2022-01-01 05:00:01") in list(collection.df.date)

    def test_dedup_keeps_most_complete_over_most_recent(self) -> None:
        """The fuller burst wins even when it has the earlier timestamp."""
        base = {
            "burst_id": BURST_ID,
            "sub_swath": SUB_SWATH,
            "polarization": "VV",
            "abs_burst_id": 1,
            "parent_product_name": "P",
            "geometry": GEOMETRY,
        }
        fuller = {
            **base,
            "date": Timestamp("2018-07-25 05:51:06.894074", tz="UTC"),
            "cdse_id": "fuller",
            "_lines_per_burst": 1508,
            "_samples_per_burst": 23281,
        }
        cropped = {
            **base,
            "date": Timestamp("2018-07-25 05:51:06.894080", tz="UTC"),
            "cdse_id": "cropped",
            "_lines_per_burst": 1504,
            "_samples_per_burst": 22424,
        }
        df = gpd.GeoDataFrame([fuller, cropped], geometry="geometry")

        with pytest.warns(UserWarning, match="most complete"):
            collection = BurstCollection(df)

        assert len(collection) == 1
        assert list(collection.df.cdse_id) == ["fuller"]

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


class TestFindPairsValidation:
    """`find_pairs` must reject baselines that cannot describe a pair."""

    @staticmethod
    def _collection() -> BurstCollection:
        return BurstCollection.from_bursts(
            [
                Burst(burst_id=30346, sub_swath="IW1", polarization="VV", date=Timestamp(date))
                for date in ("2022-08-01", "2022-08-07", "2022-08-13")
            ],
        )

    @pytest.mark.parametrize("baseline", [0, -6])
    def test_rejects_non_positive_baseline(self, baseline: int) -> None:
        with pytest.raises(ValueError, match="`baseline` must be a positive duration"):
            self._collection().find_pairs(baseline)

    def test_rejects_negative_max_delta(self) -> None:
        with pytest.raises(ValueError, match="`max_delta` must not be negative"):
            self._collection().find_pairs(6, max_delta=-1)

    def test_returns_an_empty_collection_when_nothing_matches(self) -> None:
        """Documented contract: no match yields an empty collection, not an exception."""
        pairs = self._collection().find_pairs(baseline=99, max_delta=1)

        assert len(pairs) == 0

    def test_accepts_a_zero_max_delta(self) -> None:
        """An exact baseline is a legitimate request, so zero tolerance must pass."""
        pairs = self._collection().find_pairs(baseline=6, max_delta=0)

        assert len(pairs) == 2

    @pytest.mark.parametrize(
        "max_delta",
        [12, Timedelta(hours=12), Timedelta(hours=12, microseconds=1), 13, 24, 48],
    )
    def test_rejects_max_delta_of_half_a_day_or_more(self, max_delta: Timedelta | int) -> None:
        """From half a day on, an accepted pair can round to a neighbouring whole day."""
        with pytest.raises(ValueError, match="`max_delta` must be less than 12 hours"):
            self._collection().find_pairs(6, max_delta=max_delta)

    def test_accepts_max_delta_just_below_half_a_day(self) -> None:
        pairs = self._collection().find_pairs(6, max_delta=Timedelta(hours=11.9))

        assert len(pairs) == 2

    @pytest.mark.parametrize(
        "baseline",
        [Timedelta(days=12, hours=12), Timedelta(hours=6), Timedelta(days=6, seconds=1)],
    )
    def test_rejects_a_baseline_that_is_not_whole_days(self, baseline: Timedelta) -> None:
        """Products are named by whole days, so a fractional baseline has no name."""
        with pytest.raises(ValueError, match="`baseline` must be a whole number of days"):
            self._collection().find_pairs(baseline)

    def test_default_max_delta_is_six_hours(self) -> None:
        """A pair 7 h off the baseline is outside the default tolerance but not a wider one."""
        collection = make_collection(["2022-01-01 00:00", "2022-01-13 07:00"])

        assert len(collection.find_pairs(12)) == 0
        assert len(collection.find_pairs(12, max_delta=8)) == 1

    def test_default_max_delta_includes_a_pair_five_hours_off(self) -> None:
        collection = make_collection(["2022-01-01 00:00", "2022-01-13 05:00"])

        assert len(collection.find_pairs(12)) == 1


# S1A on a 12-day repeat with jitter of seconds, one burst two hours off its
# nominal time (as CDSE reported on orbit 088 in 2019) and one five hours off,
# plus S1C a day out of phase, which forms 13-day pairs with S1A.
JITTERED_DATES = [
    "2026-06-01 05:50:00",
    "2026-06-13 05:50:03",
    "2026-06-25 05:49:58",
    "2026-07-07 07:50:00",
    "2026-07-19 05:50:01",
    "2026-07-31 10:49:59",
    "2026-06-14 05:50:02",
    "2026-06-26 05:49:57",
    "2026-07-08 05:50:04",
]


class TestFindPairsBaselineNaming:
    """Every pair `find_pairs` selects is named by the baseline it was selected for."""

    @pytest.mark.parametrize(
        "max_delta",
        [
            None,
            Timedelta(seconds=10),
            1,
            3,
            6,
            8,
            11,
            Timedelta(hours=11, minutes=59, seconds=59),
        ],
    )
    @pytest.mark.parametrize("baseline", [12, 13, 24])
    def test_every_pair_is_named_by_the_requested_baseline(
        self,
        baseline: int,
        max_delta: Timedelta | int | None,
    ) -> None:
        collection = make_collection(JITTERED_DATES)

        pairs = (
            collection.find_pairs(baseline)
            if max_delta is None
            else collection.find_pairs(baseline, max_delta=max_delta)
        )

        assert len(pairs) > 0
        for pair in pairs:
            product = CoherenceProduct.from_burst_pair(pair, crs=CRS("EPSG:32632"))
            assert product.baseline_days == baseline
            assert pair.baseline_days == baseline
