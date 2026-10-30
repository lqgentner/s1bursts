"""Tests for s1bursts.burst data structures and collections."""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING
from unittest.mock import patch

from freezebase.mgrs import MGRSGeoBox
import geopandas as gpd
import numpy as np
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from odc.geo.geom import box as odc_box
import pandas as pd
from pandas import Timedelta, Timestamp
from pyproj import CRS, Transformer
import pytest
from shapely.geometry import MultiPolygon as ShapelyMultiPolygon
from shapely.geometry import Point, box

import s1bursts
from s1bursts.burst import (
    Burst,
    BurstCollection,
    BurstPair,
    BurstPairCollection,
    utm_zone_of,
)
from s1bursts.cdse import CDSEResponse
from s1bursts.errors import BurstNotInMapError
from s1bursts.product import CoherenceProduct
from s1bursts.time_range import TimeRange

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from shapely.geometry import MultiPolygon, Polygon

    from s1bursts.query import PassDirection, Polarization, SubSwath

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

    def test_unknown_burst_raises_burst_not_in_map_error(self) -> None:
        burst = make_burst(burst_id=99999)

        with pytest.raises(BurstNotInMapError, match="not found in the Sentinel-1 burst map"):
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

    @pytest.mark.parametrize(
        "baseline",
        [dt.timedelta(days=12), np.timedelta64(12, "D")],
    )
    def test_find_pairs_accepts_any_duration_as_baseline(
        self,
        baseline: dt.timedelta | np.timedelta64,
    ) -> None:
        """Previously a dt.timedelta baseline raised pandas' unit UserWarning."""
        collection = make_collection(["2022-01-01", "2022-01-13", "2022-01-25"])

        pairs = collection.find_pairs(baseline=baseline)

        assert len(pairs) == 2

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


WEST = box(7.0, 46.0, 8.0, 47.0)
EAST = box(9.0, 46.0, 10.0, 47.0)


def make_located_burst(
    date: str,
    burst_id: int = BURST_ID,
    geometry: Polygon | MultiPolygon | None = WEST,
    rel_orbit: int | None = 66,
    direction: PassDirection | None = "Ascending",
    sub_swath: SubSwath = SUB_SWATH,
) -> Burst:
    return Burst(
        burst_id=burst_id,
        sub_swath=sub_swath,
        polarization="VV",
        date=Timestamp(date),
        _geometry=geometry,
        _rel_orbit_number=rel_orbit,
        _pass_direction=direction,
    )


class TestPublicColumns:
    def test_footprint_columns_have_public_names(self) -> None:
        collection = BurstCollection.from_bursts([make_located_burst("2022-01-01")])

        assert {"rel_orbit_number", "pass_direction"} <= set(collection.df.columns)
        assert not [c for c in collection.df.columns if c.startswith("_")]
        assert collection.df.rel_orbit_number.tolist() == [66]

    def test_pair_columns_have_public_names(self) -> None:
        bursts = BurstCollection.from_bursts(
            [make_located_burst("2022-01-01"), make_located_burst("2022-01-13")],
        )

        pairs = bursts.find_pairs(baseline=12)

        assert pairs.df.pass_direction.tolist() == ["Ascending"]
        assert not [c for c in pairs.df.columns if c.startswith("_")]

    def test_cdse_columns_round_trip_to_bursts(self) -> None:
        collection = BurstCollection.from_cdse_response(TestFromCdseResponse._response())

        assert collection.df.rel_orbit_number.tolist() == [88]
        assert collection.df.pass_direction.tolist() == ["Descending"]


class TestWithin:
    def test_keeps_every_day_of_the_window(self) -> None:
        collection = BurstCollection.from_bursts(
            [
                make_located_burst("2022-01-01 00:00"),
                make_located_burst("2022-01-13 23:59"),
                # Another sub-swath, else it would merge with the burst a minute before.
                make_located_burst("2022-01-14 00:00", sub_swath="IW2"),
            ],
        )

        selected = collection.within(TimeRange("2022-01-01", "2022-01-13"))

        assert isinstance(selected, BurstCollection)
        assert [b.date for b in selected] == [
            Timestamp("2022-01-01 00:00"),
            Timestamp("2022-01-13 23:59"),
        ]
        assert len(collection) == 3

    def test_zoned_dates_are_compared_by_their_utc_day(self) -> None:
        collection = BurstCollection.from_cdse_response(TestFromCdseResponse._response())

        assert len(collection.within(TimeRange("2022-01-01", "2022-01-01"))) == 1
        assert len(collection.within(TimeRange("2022-01-02", "2022-01-31"))) == 0

    def test_empty_collection_stays_empty(self) -> None:
        selected = BurstCollection().within(TimeRange("2022-01-01", "2022-01-31"))

        assert isinstance(selected, BurstCollection)
        assert len(selected) == 0

    def test_selections_chain_without_copying(self) -> None:
        collection = BurstCollection.from_bursts(
            [
                make_located_burst("2022-01-01"),
                make_located_burst("2022-01-13", geometry=EAST, burst_id=BURST_ID + 1),
                make_located_burst("2022-03-01"),
            ],
        )

        selected = collection.within(TimeRange("2022-01-01", "2022-01-31")).intersecting(WEST)
        selected.df["note"] = "kept"

        assert len(selected) == 1
        assert "note" not in collection.df.columns


class TestIntersecting:
    def test_keeps_footprints_that_intersect(self) -> None:
        collection = BurstCollection.from_bursts(
            [
                make_located_burst("2022-01-01"),
                make_located_burst("2022-01-01", geometry=EAST, burst_id=BURST_ID + 1),
            ],
        )

        selected = collection.intersecting(box(7.5, 46.5, 7.6, 46.6))

        assert [b.burst_id for b in selected] == [BURST_ID]

    def test_a_touching_footprint_intersects(self) -> None:
        collection = BurstCollection.from_bursts([make_located_burst("2022-01-01")])

        assert len(collection.intersecting(box(8.0, 46.0, 9.0, 47.0))) == 1

    def test_missing_footprint_is_looked_up_in_the_burst_map(self) -> None:
        collection = BurstCollection.from_bursts(
            [make_located_burst("2022-01-01", geometry=None)],
        )

        assert len(collection.intersecting(box(7.5, 46.5, 7.6, 46.6))) == 1
        assert len(collection.intersecting(EAST)) == 0

    def test_selects_pairs(self) -> None:
        bursts = BurstCollection.from_bursts(
            [
                make_located_burst(date, burst_id=burst_id, geometry=geometry)
                for date in ("2022-01-01", "2022-01-13")
                for burst_id, geometry in ((BURST_ID, WEST), (BURST_ID + 1, EAST))
            ],
        )

        selected = bursts.find_pairs(baseline=12).intersecting(EAST)

        assert isinstance(selected, BurstPairCollection)
        assert [p.burst_id for p in selected] == [BURST_ID + 1]


def _utm_box(lon: float, lat: float, crs: str, half_width: float = 1000.0) -> Geometry:
    """Return a square in a projected CRS around a WGS 84 point."""
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    return odc_box(x - half_width, y - half_width, x + half_width, y + half_width, crs)


def _projected_box(west: float, south: float, east: float, north: float, crs: str) -> Geometry:
    """Return the box in ``crs`` whose corners are the given WGS 84 corners."""
    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = to_crs.transform(west, south)
    x1, y1 = to_crs.transform(east, north)
    return odc_box(x0, y0, x1, y1, crs)


def _collection_of(*footprints: Polygon | MultiPolygon) -> BurstCollection:
    """Return one burst per footprint, its burst ID offset by its position."""
    return BurstCollection.from_bursts(
        [
            make_located_burst("2022-01-01", geometry=footprint, burst_id=BURST_ID + i)
            for i, footprint in enumerate(footprints)
        ],
    )


def _ids(collection: BurstCollection) -> list[int]:
    """Return the positions passed to ``_collection_of`` of the bursts kept."""
    return sorted(b.burst_id - BURST_ID for b in collection)


class TestIntersectingOdcGeometry:
    @staticmethod
    def make_west_and_east() -> BurstCollection:
        return BurstCollection.from_bursts(
            [
                make_located_burst("2022-01-01"),
                make_located_burst("2022-01-01", geometry=EAST, burst_id=BURST_ID + 1),
            ],
        )

    def test_utm_geometry_is_reprojected(self) -> None:
        selected = self.make_west_and_east().intersecting(_utm_box(7.5, 46.5, "EPSG:32632"))

        assert [b.burst_id for b in selected] == [BURST_ID]

    def test_geographic_geometry_is_used_as_it_is(self) -> None:
        lon_lat = Geometry(box(9.5, 46.5, 9.6, 46.6), "EPSG:4326")

        selected = self.make_west_and_east().intersecting(lon_lat)

        assert [b.burst_id for b in selected] == [BURST_ID + 1]

    def test_geometry_without_crs_is_taken_as_wgs84(self) -> None:
        unlocated = Geometry(box(9.5, 46.5, 9.6, 46.6))

        selected = self.make_west_and_east().intersecting(unlocated)

        assert [b.burst_id for b in selected] == [BURST_ID + 1]

    def test_geobox_stands_for_its_extent(self) -> None:
        cell = GeoBox.from_bbox(_utm_box(9.5, 46.5, "EPSG:32632").boundingbox, resolution=10)

        selected = self.make_west_and_east().intersecting(cell)

        assert [b.burst_id for b in selected] == [BURST_ID + 1]

    def test_mgrs_cell_and_its_extent_select_alike(self) -> None:
        cell = MGRSGeoBox.from_mgrs("32TMT64", resolution=10.0)  # near 8.5 E, 47.35 N
        collection = BurstCollection.from_bursts(
            [
                make_located_burst("2022-01-01", geometry=box(8.0, 47.0, 9.0, 48.0)),
                make_located_burst("2022-01-01", geometry=EAST, burst_id=BURST_ID + 1),
            ],
        )

        by_cell = collection.intersecting(cell)
        by_extent = collection.intersecting(cell.extent)

        assert [b.burst_id for b in by_cell] == [BURST_ID]
        assert [b.burst_id for b in by_extent] == [BURST_ID]

    def test_utm_area_across_the_antimeridian(self) -> None:
        collection = _collection_of(
            box(-179.99, -17.05, -179.9, -16.95),
            # Split at the antimeridian, as CDSE delivers such a footprint.
            ShapelyMultiPolygon(
                [box(179.95, -17.02, 180.0, -16.98), box(-180.0, -17.02, -179.98, -16.98)],
            ),
            # Inside the longitude span of the area's corners, outside the area.
            box(-1.0, -17.05, 1.0, -16.95),
        )

        selected = collection.intersecting(_utm_box(180.0, -17.0, "EPSG:32760", 5000.0))

        assert _ids(selected) == [0, 1]

    def test_wide_web_mercator_area(self) -> None:
        collection = _collection_of(
            box(10.0, 0.0, 11.0, 1.0),
            box(-60.0, 0.0, -59.0, 1.0),
            box(170.0, 0.0, 171.0, 1.0),
        )

        selected = collection.intersecting(_projected_box(-120.0, -60.0, 120.0, 60.0, "EPSG:3857"))

        assert _ids(selected) == [0, 1]

    def test_near_global_web_mercator_area(self) -> None:
        collection = _collection_of(
            box(10.0, 0.0, 11.0, 1.0),
            box(179.5, 0.0, 179.9, 1.0),
        )

        selected = collection.intersecting(_projected_box(-179.0, -80.0, 179.0, 80.0, "EPSG:3857"))

        assert _ids(selected) == [0]

    def test_polar_area_containing_the_pole(self) -> None:
        collection = _collection_of(
            box(30.0, -85.0, 40.0, -84.0),
            box(30.0, -60.0, 31.0, -59.0),
            # Northern hemisphere: far outside the area, and far from where the
            # projection is usable.
            box(30.0, 60.0, 31.0, 61.0),
            box(30.0, 89.5, 31.0, 89.9),
        )

        selected = collection.intersecting(odc_box(-2e6, -2e6, 2e6, 2e6, "EPSG:3031"))

        assert _ids(selected) == [0]

    def test_ring_around_the_pole(self) -> None:
        to_polar = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
        inner = to_polar.transform(0.0, -85.0)[1]
        outer = to_polar.transform(0.0, -75.0)[1]
        ring = Geometry(
            Point(0.0, 0.0).buffer(abs(outer)).difference(Point(0.0, 0.0).buffer(abs(inner))),
            "EPSG:3031",
        )
        collection = _collection_of(
            box(120.0, -81.0, 125.0, -80.0),
            box(120.0, -89.0, 125.0, -88.0),
            box(120.0, -70.0, 125.0, -69.0),
        )

        assert _ids(collection.intersecting(ring)) == [0]

    def test_north_polar_area(self) -> None:
        collection = _collection_of(
            box(-40.0, 75.0, -38.0, 76.0),
            box(-40.0, -75.0, -38.0, -74.0),
        )

        selected = collection.intersecting(odc_box(-2e6, -3e6, 1e6, 0.0, "EPSG:3413"))

        assert _ids(selected) == [0]

    def test_footprint_edges_follow_their_course_in_the_area_crs(self) -> None:
        # The footprint's edges run along parallels, which are circles around the
        # pole in polar stereographic. Joined straight from corner to corner, the
        # outer edge would cut the circle short and miss the area.
        to_polar = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
        x, y = to_polar.transform(45.0, -79.6)
        collection = _collection_of(box(0.0, -80.0, 90.0, -79.0))

        selected = collection.intersecting(
            odc_box(x - 1e3, y - 1e3, x + 1e3, y + 1e3, "EPSG:3031")
        )

        assert _ids(selected) == [0]

    @pytest.mark.parametrize(
        "crs",
        [
            pytest.param("EPSG:3832", id="metres"),
            pytest.param("+proj=merc +lon_0=150 +datum=WGS84 +units=km", id="kilometres"),
        ],
    )
    def test_footprint_torn_by_the_crs_does_not_intersect(self, crs: str) -> None:
        # Pacific-centred Mercator cuts the globe at 30 W. A footprint across the
        # cut would otherwise span the whole map at its latitudes.
        collection = _collection_of(
            box(-30.5, -1.0, -29.5, 1.0),
            box(150.0, 0.0, 151.0, 1.0),
        )

        selected = collection.intersecting(_projected_box(140.0, -10.0, 160.0, 10.0, crs))

        assert _ids(selected) == [1]

    @pytest.mark.parametrize(
        ("crs", "torn"),
        [
            # Paris meridian in grads: the cut is at 177.66 W.
            pytest.param("EPSG:4807", box(-177.9, 10.0, -177.4, 10.8), id="paris-grads"),
            pytest.param(
                "+proj=longlat +datum=WGS84 +pm=90 +no_defs",
                box(-90.3, 10.0, -89.7, 10.8),
                id="prime-meridian-90",
            ),
        ],
    )
    def test_footprint_torn_by_a_shifted_prime_meridian_does_not_intersect(
        self,
        crs: str,
        torn: Polygon,
    ) -> None:
        # The CRS's antimeridian is not 180 degrees, so a footprint across it is
        # torn into a band around the globe at its latitudes.
        collection = _collection_of(torn, box(24.0, 10.0, 25.0, 11.0))

        selected = collection.intersecting(_projected_box(20.0, 9.0, 30.0, 12.0, crs))

        assert _ids(selected) == [1]

    def test_footprint_torn_near_the_far_pole_of_a_cone_does_not_intersect(self) -> None:
        # Near the pole, Alaska Albers's cut at 26 E jumps less than 1000 km. The
        # area sits between the two halves the cut leaves of the footprint.
        to_alaska = Transformer.from_crs("EPSG:4326", "EPSG:3338", always_xy=True)
        x, y = np.mean(
            [to_alaska.transform(25.8, 84.65), to_alaska.transform(26.2, 84.65)], axis=0
        )
        collection = _collection_of(box(25.6, 84.5, 26.4, 84.8))

        selected = collection.intersecting(Geometry(Point(x, y).buffer(5000.0), "EPSG:3338"))

        assert _ids(selected) == []

    def test_footprint_in_alaska_albers(self) -> None:
        collection = _collection_of(box(-150.0, 61.0, -149.0, 62.0), box(25.0, 61.0, 26.0, 62.0))

        selected = collection.intersecting(_projected_box(-155.0, 58.0, -145.0, 65.0, "EPSG:3338"))

        assert _ids(selected) == [0]

    def test_footprint_the_crs_cannot_project_does_not_intersect(self) -> None:
        # On the equator 90 degrees from the zone's meridian, UTM has no coordinates.
        collection = _collection_of(box(8.0, 0.0, 9.0, 1.0), box(98.5, -0.5, 99.5, 0.5))

        selected = collection.intersecting(_utm_box(9.0, 0.0, "EPSG:32632", 5e6))

        assert _ids(selected) == [0]

    def test_area_in_another_geographic_crs(self) -> None:
        collection = _collection_of(box(7.0, 46.0, 8.0, 47.0), box(9.0, 46.0, 10.0, 47.0))

        selected = collection.intersecting(Geometry(box(7.4, 46.4, 7.6, 46.6), "EPSG:4258"))

        assert _ids(selected) == [0]

    def test_empty_collection_in_a_projected_area(self) -> None:
        assert len(BurstCollection().intersecting(odc_box(-1e6, -1e6, 1e6, 1e6, "EPSG:3031"))) == 0


class TestFilter:
    @staticmethod
    def make_mixed() -> BurstCollection:
        return BurstCollection.from_bursts(
            [
                make_located_burst("2022-01-01", rel_orbit=66),
                make_located_burst("2022-01-02", rel_orbit=139, direction="Descending"),
                make_located_burst("2022-01-03", rel_orbit=15, sub_swath="IW2"),
            ],
        )

    def test_keeps_matching_value(self) -> None:
        selected = self.make_mixed().filter(rel_orbit_number=66)

        assert isinstance(selected, BurstCollection)
        assert [b.rel_orbit_number for b in selected] == [66]

    def test_a_list_keeps_any_of_its_values(self) -> None:
        selected = self.make_mixed().filter(rel_orbit_number=[66, 139])

        assert sorted(b.rel_orbit_number for b in selected) == [66, 139]

    @pytest.mark.parametrize(
        "orbits",
        [
            pytest.param(np.array([66, 139]), id="ndarray"),
            pytest.param(pd.Series([66, 139]), id="series"),
            pytest.param(pd.Index([66, 139]), id="index"),
            pytest.param(range(66, 140, 73), id="range"),
            pytest.param((o for o in (66, 139)), id="generator"),
        ],
    )
    def test_any_iterable_keeps_any_of_its_values(self, orbits: Iterable[int]) -> None:
        selected = self.make_mixed().filter(rel_orbit_number=orbits)

        assert sorted(b.rel_orbit_number for b in selected) == [66, 139]

    def test_unique_values_of_another_collection_select_its_orbits(self) -> None:
        other = self.make_mixed().filter(pass_direction="Ascending")

        selected = self.make_mixed().filter(rel_orbit_number=other.df.rel_orbit_number.unique())

        assert sorted(b.rel_orbit_number for b in selected) == [15, 66]

    def test_a_string_is_one_value(self) -> None:
        selected = self.make_mixed().filter(sub_swath="IW2")

        assert [b.sub_swath for b in selected] == ["IW2"]

    def test_every_keyword_must_match(self) -> None:
        selected = self.make_mixed().filter(
            rel_orbit_number=(66, 15),
            pass_direction="Ascending",
            sub_swath="IW2",
        )

        assert [b.rel_orbit_number for b in selected] == [15]

    def test_no_keywords_keeps_everything(self) -> None:
        assert len(self.make_mixed().filter()) == 3

    def test_missing_footprint_values_are_looked_up_in_the_burst_map(self) -> None:
        collection = BurstCollection.from_bursts(
            [
                make_located_burst("2022-01-01", rel_orbit=None, direction=None),
                make_located_burst("2022-01-13", rel_orbit=66),
            ],
        )

        assert len(collection.filter(rel_orbit_number=15, pass_direction="Ascending")) == 1
        assert len(collection.filter(rel_orbit_number={15, 66})) == 2

    @pytest.mark.parametrize("field", ["rel_orbit", "date", "geometry", "_lines_per_burst"])
    def test_rejects_unknown_and_unfilterable_fields(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"Cannot filter on '{field}'"):
            self.make_mixed().filter(**{field: 1})

    def test_selects_pairs(self) -> None:
        bursts = BurstCollection.from_bursts(
            [
                make_located_burst(date, rel_orbit=orbit, sub_swath=sub_swath)
                for date in ("2022-01-01", "2022-01-13")
                for orbit, sub_swath in ((66, "IW1"), (139, "IW2"))
            ],
        )

        selected = bursts.find_pairs(baseline=12).filter(rel_orbit_number=139)

        assert isinstance(selected, BurstPairCollection)
        assert [p.sub_swath for p in selected] == ["IW2"]


class TestStartingWithin:
    def test_selects_pairs_by_primary_date(self) -> None:
        bursts = BurstCollection.from_bursts(
            [make_located_burst(d) for d in ("2022-01-19", "2022-01-31", "2022-02-12")],
        )

        selected = bursts.find_pairs(baseline=12).starting_within(
            TimeRange("2022-01-01", "2022-01-31")
        )

        assert isinstance(selected, BurstPairCollection)
        # The pair starting on the last day is kept though it ends after the window.
        assert [(p.primary.date.day, p.secondary.date.day) for p in selected] == [
            (19, 31),
            (31, 12),
        ]


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
            self._collection().find_pairs(baseline=baseline)

    def test_rejects_negative_max_delta(self) -> None:
        with pytest.raises(ValueError, match="`max_delta` must not be negative"):
            self._collection().find_pairs(baseline=6, max_delta=Timedelta(hours=-1))

    def test_returns_an_empty_collection_when_nothing_matches(self) -> None:
        """Documented contract: no match yields an empty collection, not an exception."""
        pairs = self._collection().find_pairs(baseline=99, max_delta=Timedelta(hours=1))

        assert len(pairs) == 0

    def test_accepts_a_zero_max_delta(self) -> None:
        """An exact baseline is a legitimate request, so zero tolerance must pass."""
        pairs = self._collection().find_pairs(baseline=6, max_delta=Timedelta(hours=0))

        assert len(pairs) == 2

    @pytest.mark.parametrize(
        "max_delta",
        [
            Timedelta(hours=12),
            Timedelta(hours=12, microseconds=1),
            Timedelta(hours=13),
            Timedelta(days=1),
            Timedelta(days=2),
        ],
    )
    def test_rejects_max_delta_of_half_a_day_or_more(self, max_delta: Timedelta) -> None:
        """From half a day on, an accepted pair can round to a neighbouring whole day."""
        with pytest.raises(ValueError, match="`max_delta` must be less than 12 hours"):
            self._collection().find_pairs(baseline=6, max_delta=max_delta)

    def test_rejects_half_a_day_for_an_odd_baseline(self) -> None:
        """The bound is ``>=`` because half-to-even rounding breaks odd baselines at 12 h.

        An even baseline would survive exactly 12 h (12 d +- 12 h both round to
        12), but an odd one does not: 13 d + 12 h rounds to 14, 13 d - 12 h to 12.
        """
        assert Timedelta(days=12, hours=12).round("D") == Timedelta(days=12)
        assert Timedelta(days=11, hours=12).round("D") == Timedelta(days=12)
        assert Timedelta(days=13, hours=12).round("D") == Timedelta(days=14)

        with pytest.raises(ValueError, match="`max_delta` must be less than 12 hours"):
            self._collection().find_pairs(baseline=13, max_delta=Timedelta(hours=12))

    def test_accepts_max_delta_just_below_half_a_day(self) -> None:
        pairs = self._collection().find_pairs(baseline=6, max_delta=Timedelta(hours=11.9))

        assert len(pairs) == 2

    @pytest.mark.parametrize(
        "baseline",
        [Timedelta(days=12, hours=12), Timedelta(hours=6), Timedelta(days=6, seconds=1)],
    )
    def test_rejects_a_baseline_that_is_not_whole_days(self, baseline: Timedelta) -> None:
        """Products are named by whole days, so a fractional baseline has no name."""
        with pytest.raises(ValueError, match="`baseline` must be a whole number of days"):
            self._collection().find_pairs(baseline=baseline)

    def test_rejects_positional_arguments(self) -> None:
        """Both arguments are keyword-only, so a bare ``find_pairs(12, 6)`` cannot be misread."""
        with pytest.raises(TypeError):
            # The positional call is the misuse under test.
            self._collection().find_pairs(6)  # pyrefly: ignore[unexpected-positional-argument]

    @pytest.mark.parametrize("baseline", [12.0, "12 days"])
    def test_rejects_a_baseline_that_is_neither_an_integer_nor_a_duration(
        self,
        baseline: object,
    ) -> None:
        with pytest.raises(TypeError, match=r"baseline=pd\.Timedelta\(days=12\)"):
            # The wrong type is the misuse under test.
            self._collection().find_pairs(baseline=baseline)  # pyrefly: ignore[bad-argument-type]

    @pytest.mark.parametrize("argument", ["baseline", "max_delta"])
    def test_rejects_not_a_time(self, argument: str) -> None:
        durations: dict[str, Timedelta | np.timedelta64] = {
            "baseline": Timedelta(days=6),
            "max_delta": Timedelta(hours=6),
        }
        durations[argument] = np.timedelta64("NaT", "ns")

        with pytest.raises(ValueError, match=f"`{argument}` must be a duration"):
            self._collection().find_pairs(**durations)

    @pytest.mark.parametrize(
        "max_delta",
        [Timedelta(hours=1), dt.timedelta(hours=1), np.timedelta64(1, "h")],
    )
    def test_accepts_any_duration_as_max_delta(
        self,
        max_delta: dt.timedelta | np.timedelta64,
    ) -> None:
        """A pair two hours off the baseline is outside a one-hour tolerance of any type."""
        collection = make_collection(["2022-01-01 00:00", "2022-01-13 02:00"])

        assert len(collection.find_pairs(baseline=12, max_delta=max_delta)) == 0
        assert len(collection.find_pairs(baseline=12)) == 1

    @pytest.mark.parametrize("max_delta", [6, 6.0, "6h"])
    def test_rejects_a_max_delta_that_is_not_a_timedelta(self, max_delta: object) -> None:
        """Previously an integer was read as hours; now the unit must be explicit."""
        with pytest.raises(TypeError, match=r"max_delta=pd\.Timedelta\(hours=6\)"):
            self._collection().find_pairs(
                baseline=6,
                # The wrong type is the misuse under test.
                max_delta=max_delta,  # pyrefly: ignore[bad-argument-type]
            )

    def test_default_max_delta_is_six_hours(self) -> None:
        """A pair 7 h off the baseline is outside the default tolerance but not a wider one."""
        collection = make_collection(["2022-01-01 00:00", "2022-01-13 07:00"])

        assert len(collection.find_pairs(baseline=12)) == 0
        assert len(collection.find_pairs(baseline=12, max_delta=Timedelta(hours=8))) == 1

    def test_default_max_delta_includes_a_pair_five_hours_off(self) -> None:
        collection = make_collection(["2022-01-01 00:00", "2022-01-13 05:00"])

        assert len(collection.find_pairs(baseline=12)) == 1

    def test_default_max_delta_includes_a_pair_exactly_six_hours_off(self) -> None:
        """The default tolerance is inclusive at its upper bound."""
        collection = make_collection(["2022-01-01 00:00", "2022-01-13 06:00"])

        assert len(collection.find_pairs(baseline=12)) == 1


# S1A on a 12-day repeat with jitter of seconds, one burst two hours off its
# nominal time (as CDSE reported on orbit 088 in 2019) and one five hours off,
# plus S1C a day out of phase, which forms 13-day pairs with S1A. The last entry
# is not observed jitter but a synthetic boundary probe: 11 h 30 m late, so it
# pairs with its predecessor close to 12 d + 12 h, just inside the widest
# tolerance a caller may ask for.
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
    "2026-07-20 17:20:00",
]
EDGE_PAIR = (Timestamp("2026-07-08 05:50:04"), Timestamp("2026-07-20 17:20:00"))


class TestFindPairsBaselineNaming:
    """Every pair `find_pairs` selects is named by the baseline it was selected for."""

    @pytest.mark.parametrize(
        "max_delta",
        [
            None,
            Timedelta(seconds=10),
            Timedelta(hours=1),
            Timedelta(hours=3),
            Timedelta(hours=6),
            Timedelta(hours=8),
            Timedelta(hours=11),
            Timedelta(hours=11, minutes=59, seconds=59),
        ],
    )
    @pytest.mark.parametrize("baseline", [12, 13, 24])
    def test_every_pair_is_named_by_the_requested_baseline(
        self,
        baseline: int,
        max_delta: Timedelta | None,
    ) -> None:
        collection = make_collection(JITTERED_DATES)

        pairs = (
            collection.find_pairs(baseline=baseline)
            if max_delta is None
            else collection.find_pairs(baseline=baseline, max_delta=max_delta)
        )

        assert len(pairs) > 0
        for pair in pairs:
            product = CoherenceProduct.from_burst_pair(pair, crs=CRS("EPSG:32632"))
            assert product.baseline_days == baseline
            assert pair.baseline_days == baseline

    def test_the_edge_pair_is_found_at_the_widest_tolerance(self) -> None:
        """A pair 11 h 30 m off the baseline is still accepted and named by it."""
        collection = make_collection(JITTERED_DATES)

        pairs = collection.find_pairs(
            baseline=12, max_delta=Timedelta(hours=11, minutes=59, seconds=59)
        )

        found = [(pair.primary.date, pair.secondary.date) for pair in pairs]
        assert EDGE_PAIR in found
        assert EDGE_PAIR[1] - EDGE_PAIR[0] == Timedelta(days=12, hours=11, minutes=29, seconds=56)

    def test_finds_a_pair_at_a_lag_past_the_first_over_long_one(self) -> None:
        """Widening the lag must not stop at the first lag whose spacings all exceed the baseline.

        Spacings grow with the lag, so a lag whose shortest spacing is just past
        the baseline can still be followed by one inside the tolerance.
        """
        # Two acquisitions per day, two hours apart: a synthetic probe of the
        # lag arithmetic, and the gap has to exceed the one hour at which
        # ``_remove_timestamp_duplicates`` would collapse them.
        dates = [
            "2026-01-01 00:00",
            "2026-01-01 02:00",
            "2026-01-13 01:00",
            "2026-01-13 05:00",
        ]
        collection = make_collection(dates)

        pairs = collection.find_pairs(baseline=12, max_delta=Timedelta(hours=6))

        found = [(pair.primary.date, pair.secondary.date) for pair in pairs]
        # Lag 3, reached only after lag 2, whose spacings are 12 d 1 h and 12 d 3 h.
        assert (Timestamp(dates[0]), Timestamp(dates[3])) in found
        for pair in pairs:
            assert pair.baseline_days == 12


class TestUtmZoneOf:
    @pytest.mark.parametrize(
        ("footprint", "epsg"),
        [
            pytest.param(box(7.0, 46.0, 8.0, 47.0), 32632, id="north"),
            pytest.param(box(-70.0, -34.0, -69.0, -33.0), 32719, id="south"),
            # Centred on the 12 degree border between zones 32 and 33.
            pytest.param(box(11.0, 46.0, 13.0, 47.0), 32633, id="border-takes-eastern-zone"),
            pytest.param(
                ShapelyMultiPolygon(
                    [box(179.6, -17.0, 180.0, -16.0), box(-180.0, -17.0, -179.8, -16.0)],
                ),
                32760,
                id="antimeridian-west-of-the-line",
            ),
            pytest.param(
                ShapelyMultiPolygon(
                    [box(179.9, -17.0, 180.0, -16.0), box(-180.0, -17.0, -179.5, -16.0)],
                ),
                32701,
                id="antimeridian-east-of-the-line",
            ),
        ],
    )
    def test_zone_holds_the_footprint_centre(
        self,
        footprint: Polygon | MultiPolygon,
        epsg: int,
    ) -> None:
        assert utm_zone_of(make_burst(geometry=footprint)) == CRS.from_epsg(epsg)

    def test_centre_outside_the_utm_zones_raises(self) -> None:
        with pytest.raises(ValueError, match="No UTM zone covers"):
            utm_zone_of(make_burst(geometry=box(10.0, 84.5, 11.0, 85.5)))

    def test_burst_pair_takes_the_zone_of_its_footprint(self) -> None:
        pair = BurstPair(make_burst("2022-01-01"), make_burst("2022-01-13"))

        assert utm_zone_of(pair) == CRS.from_epsg(32632)

    def test_is_exported_from_the_package(self) -> None:
        assert s1bursts.utm_zone_of is utm_zone_of
