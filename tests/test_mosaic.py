"""Tests for the per-acquisition burst mosaic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

import numpy as np
from pandas import Timestamp
from pyproj import CRS
import pytest

from s1bursts._mosaic import (
    AcquisitionMosaic,
    TrimmedBurst,
    acquisition_date,
    auto_edge_trim_px,
    mosaic_by_orbit_date,
    scattering_areas,
    trim,
)
from s1bursts.errors import CompositeInputError
from s1bursts.product import CoherenceProduct
from tests.product_stubs import RTCStub

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# trim, mosaic_by_orbit_date
# ---------------------------------------------------------------------------

# Pixels read around the cell in these tests, as ``EDGE_SEARCH_MARGIN_M`` does.
_MARGIN = 4
# Pixels trimmed from each burst edge in these tests.
_TRIM = 2


def _burst(
    product: object,
    padded: NDArray[np.floating[Any]],
    margin: int = _MARGIN,
    trim_px: int = _TRIM,
) -> TrimmedBurst:
    """Measure and trim a widened read, as ``ResolutionWeighter._load_bursts`` does."""
    # Stubs carry only the fields the mosaicking reads.
    # pyrefly: ignore [bad-argument-type]
    return trim(product, padded.astype(np.float32), margin, trim_px)


def _strip(first: int, last: int, value: float, width: int = 24) -> NDArray[np.float32]:
    """Return a read of three cell rows with burst data in columns ``first`` to ``last``."""
    padded = np.full((3 + 2 * _MARGIN, width), np.nan, dtype=np.float32)
    padded[:, first : last + 1] = value
    return padded


def _row(mosaic: AcquisitionMosaic) -> NDArray[np.float32]:
    """Return the middle cell row of a mosaic of ``_strip`` reads, which all rows repeat."""
    return mosaic.data[1]


class TestTrim:
    """Tests for ``trim``, the edge trim and depth."""

    _P = RTCStub(10, Timestamp("2023-01-10"), "VV")

    def test_outermost_pixels_are_dropped(self) -> None:
        """The two outermost pixels of the burst become NaN, in rows and columns."""
        padded = np.full((20, 20), np.nan)
        padded[6:15, 7:15] = 1.0
        burst = _burst(self._P, padded)
        cell = padded[_MARGIN:-_MARGIN, _MARGIN:-_MARGIN].copy()
        cell[:, [3, 4, 9, 10]] = np.nan  # columns 7, 8, 13, 14 of the read
        cell[[2, 3, 9, 10], :] = np.nan  # rows 6, 7, 13, 14 of the read
        np.testing.assert_array_equal(burst.data, cell)

    def test_holes_are_not_edges(self) -> None:
        """A NaN hole inside the footprint keeps its neighbors, which stay deep."""
        padded = np.ones((20, 20))
        padded[10, 10] = np.nan
        burst = _burst(self._P, padded)
        assert np.count_nonzero(np.isnan(burst.data)) == 1
        assert burst.depth[10 - _MARGIN, 11 - _MARGIN] > _TRIM

    def test_data_reaching_the_read_border_is_deep_and_kept(self) -> None:
        """A run reaching the border of the read continues beyond it, so it has no edge there."""
        burst = _burst(self._P, np.ones((11, 11)))
        assert np.isfinite(burst.data).all()
        np.testing.assert_array_equal(burst.depth, _MARGIN + 1)

    def test_depth_counts_from_one_at_the_outermost_pixel(self) -> None:
        """The depth is the distance to the nearer data edge along the row."""
        burst = _burst(self._P, _strip(5, 18, 1.0))
        np.testing.assert_array_equal(burst.depth[1, 2:6], [2, 3, 4, 5])  # read columns 6-9

    @pytest.mark.parametrize("trim_px", [0, 3])
    def test_drops_the_given_number_of_edge_pixels(self, trim_px: int) -> None:
        """Pixels up to depth ``trim_px`` become NaN; the next one is kept."""
        burst = _burst(self._P, _strip(5, 18, 1.0), trim_px=trim_px)
        finite = np.isfinite(burst.data[1])
        # Cell column 1 is read column 5, the burst's outermost pixel (depth 1).
        assert not finite[1 : 1 + trim_px].any()
        assert finite[1 + trim_px]


class TestAutoEdgeTrim:
    """Tests for ``auto_edge_trim_px``, the trim derived from the settings."""

    @pytest.mark.parametrize(
        ("tags", "spacing", "expected"),
        [
            ({"MULTILOOK_RANGE": "5", "MULTILOOK_AZIMUTH": "1"}, 10.0, 4),
            ({"MULTILOOK_RANGE": "10", "MULTILOOK_AZIMUTH": "2"}, 10.0, 8),
            ({"MULTILOOK_RANGE": "5", "MULTILOOK_AZIMUTH": "1"}, 20.0, 2),
            ({"MULTILOOK_RANGE": "5", "MULTILOOK_AZIMUTH": "1"}, 30.0, 2),
            # multilook=None writes no MULTILOOK tags.
            ({}, 10.0, 3),
        ],
    )
    def test_backscatter_follows_the_looks_and_grid_spacing(
        self, tags: dict[str, str], spacing: float, expected: int
    ) -> None:
        assert auto_edge_trim_px(tags, coherence=False, grid_spacing=spacing) == expected

    @pytest.mark.parametrize(("window", "expected"), [((25, 5), 5), ((10, 2), 3), ((35, 7), 7)])
    def test_coherence_follows_the_estimation_window(
        self, window: tuple[int, int], expected: int
    ) -> None:
        tags = {
            "COHERENCE_WINDOW_RANGE": str(window[0]),
            "COHERENCE_WINDOW_AZIMUTH": str(window[1]),
        }
        assert auto_edge_trim_px(tags, coherence=True, grid_spacing=10.0) == expected

    def test_coherence_needs_the_window(self) -> None:
        with pytest.raises(ValueError, match="COHERENCE_WINDOW"):
            auto_edge_trim_px({}, coherence=True, grid_spacing=10.0)


class TestMosaicByOrbitDate:
    """Tests for ``mosaic_by_orbit_date``."""

    _D1 = Timestamp("2023-01-10")
    _D2 = Timestamp("2023-01-22")

    def _mosaic(self, *bursts: TrimmedBurst) -> dict[tuple[int, Timestamp], AcquisitionMosaic]:
        return mosaic_by_orbit_date(bursts, "32TMS03")

    def test_overlap_is_split_at_its_midline(self) -> None:
        """Each pixel comes from the burst whose data edge is farther away.

        The overlap spans read columns 8 to 13, so its midline lies between
        columns 10 and 11.
        """
        a = RTCStub(10, self._D1, "VV", burst_id=1)
        b = RTCStub(10, self._D1, "VV", burst_id=2)
        [mosaic] = self._mosaic(
            _burst(a, _strip(0, 13, 1.0)), _burst(b, _strip(8, 23, 2.0))
        ).values()
        # Cell columns 0-15 are read columns 4-19.
        np.testing.assert_array_equal(_row(mosaic), [1.0] * 7 + [2.0] * 9)

    def test_an_azimuth_overlap_is_split_at_its_midline(self) -> None:
        """Columns find the midline of an overlap between bursts above each other."""
        a = RTCStub(10, self._D1, "VV", burst_id=1)
        b = RTCStub(10, self._D1, "VV", burst_id=2)
        [mosaic] = self._mosaic(
            _burst(a, _strip(0, 13, 1.0).T), _burst(b, _strip(8, 23, 2.0).T)
        ).values()
        np.testing.assert_array_equal(mosaic.data[:, 1], [1.0] * 7 + [2.0] * 9)

    def test_midline_follows_a_tilted_overlap(self) -> None:
        """Rows and columns find the midline of a strip at an angle to the grid too."""
        size = 60
        rows, cols = np.mgrid[0:size, 0:size]
        a = np.where(cols < 34 + 0.25 * rows, 1.0, np.nan)
        b = np.where(cols >= 18 + 0.25 * rows, 2.0, np.nan)
        pa = RTCStub(10, self._D1, "VV", burst_id=1)
        pb = RTCStub(10, self._D1, "VV", burst_id=2)
        [mosaic] = self._mosaic(_burst(pa, a, 10), _burst(pb, b, 10)).values()
        for cell_row, values in enumerate(mosaic.data):
            switch = 10 + int(np.argmax(values == 2.0))  # first read column from b
            midline = 26 + 0.25 * (cell_row + 10)
            assert abs(switch - midline) <= 1

    def test_a_tie_goes_to_the_lowest_burst_in_any_input_order(self) -> None:
        """On the midline pixel itself, the lower (burst_id, sub_swath) wins.

        The caller's list order is arbitrary, as a processing run lists its
        products in completion order.
        """
        low = RTCStub(10, self._D1, "VV", burst_id=1, sub_swath="IW2")
        high = RTCStub(10, self._D1, "VV", burst_id=2, sub_swath="IW1")
        # Overlap in read columns 8-12; column 10 lies 3 pixels inside both.
        first = _burst(high, _strip(0, 12, 1.0))
        second = _burst(low, _strip(8, 23, 2.0))
        for order in ((first, second), (second, first)):
            [mosaic] = self._mosaic(*order).values()
            assert _row(mosaic)[10 - _MARGIN] == 2.0

    def test_an_edge_outside_the_cell_still_counts(self) -> None:
        """A data edge in the margin around the cell places the midline correctly."""
        a = RTCStub(10, self._D1, "VV", burst_id=1)
        b = RTCStub(10, self._D1, "VV", burst_id=2)
        # b starts at read column 1, left of the cell; the overlap is columns 1-9.
        [mosaic] = self._mosaic(
            _burst(a, _strip(0, 9, 1.0)), _burst(b, _strip(1, 23, 2.0))
        ).values()
        np.testing.assert_array_equal(_row(mosaic)[:4], [1.0, 1.0, 2.0, 2.0])

    def test_a_hole_in_the_deeper_burst_takes_the_other(self) -> None:
        """Where the deeper burst has no data, the other burst's value is used."""
        a = RTCStub(10, self._D1, "VV", burst_id=1)
        b = RTCStub(10, self._D1, "VV", burst_id=2)
        holed = _strip(0, 13, 1.0)
        holed[:, 10] = np.nan
        [mosaic] = self._mosaic(_burst(a, holed), _burst(b, _strip(8, 23, 2.0))).values()
        assert _row(mosaic)[10 - _MARGIN] == 2.0
        assert _row(mosaic)[9 - _MARGIN] == 1.0

    def test_index_names_the_source_burst(self) -> None:
        """``index`` points into ``sources``, and is -1 where no burst has data."""
        a = RTCStub(10, self._D1, "VV", burst_id=1)
        b = RTCStub(10, self._D1, "VV", burst_id=2)
        [mosaic] = self._mosaic(
            _burst(a, _strip(0, 13, 1.0)), _burst(b, _strip(8, 18, 2.0))
        ).values()
        assert mosaic.sources == (a, b)
        np.testing.assert_array_equal(mosaic.index[1], [0] * 7 + [1] * 6 + [-1] * 3)

    def test_two_bursts_deeper_than_the_margin_are_logged(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Where two bursts cover the whole read, their depths cannot be ranked."""
        a = RTCStub(10, self._D1, "VV", burst_id=1)
        b = RTCStub(10, self._D1, "VV", burst_id=2)
        with caplog.at_level(logging.WARNING, logger="s1bursts._mosaic"):
            [mosaic] = self._mosaic(
                _burst(a, np.ones((11, 11))), _burst(b, np.full((11, 11), 2.0))
            ).values()
        assert "32TMS03, orbit 10 on 2023-01-10: 9 pixels" in caplog.text
        np.testing.assert_array_equal(mosaic.data, 1.0)

    def test_the_warning_reaches_the_package_logger(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The warning comes from a logger under ``s1bursts``, where users configure logging."""
        a = RTCStub(10, self._D1, "VV", burst_id=1)
        b = RTCStub(10, self._D1, "VV", burst_id=2)
        with caplog.at_level(logging.WARNING, logger="s1bursts"):
            self._mosaic(_burst(a, np.ones((11, 11))), _burst(b, np.full((11, 11), 2.0)))
        # caplog listens on the root logger, so only the record's name shows where it came from.
        assert [record.name for record in caplog.records] == ["s1bursts._mosaic"]

    def test_a_ranked_overlap_is_not_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """An overlap narrower than twice the margin is decided without a warning."""
        a = RTCStub(10, self._D1, "VV", burst_id=1)
        b = RTCStub(10, self._D1, "VV", burst_id=2)
        with caplog.at_level(logging.WARNING, logger="s1bursts._mosaic"):
            self._mosaic(_burst(a, _strip(0, 13, 1.0)), _burst(b, _strip(8, 23, 2.0)))
        assert not caplog.records

    def test_different_dates_produce_separate_groups(self) -> None:
        """Products with different acquisition dates must produce separate keys."""
        p1 = RTCStub(rel_orbit_number=10, date=self._D1, polarization="VV")
        p2 = RTCStub(rel_orbit_number=10, date=self._D2, polarization="VV")
        result = self._mosaic(_burst(p1, np.ones((9, 9))), _burst(p2, np.ones((9, 9))))
        assert len(result) == 2

    def test_different_orbits_produce_separate_groups(self) -> None:
        """Products with different orbit numbers must produce separate keys."""
        p1 = RTCStub(rel_orbit_number=10, date=self._D1, polarization="VV")
        p2 = RTCStub(rel_orbit_number=20, date=self._D1, polarization="VV")
        result = self._mosaic(_burst(p1, np.ones((9, 9))), _burst(p2, np.ones((9, 9))))
        assert len(result) == 2

    def test_result_is_sorted_by_key(self) -> None:
        """Output dict must be sorted by (orbit_number, date) key."""
        p_late = RTCStub(rel_orbit_number=10, date=self._D2, polarization="VV")
        p_early = RTCStub(rel_orbit_number=10, date=self._D1, polarization="VV")
        result = self._mosaic(_burst(p_late, np.ones((9, 9))), _burst(p_early, np.ones((9, 9))))
        keys = list(result.keys())
        assert keys == sorted(keys)
        assert keys[0][1] < keys[1][1]

    def test_coherence_product_uses_date_primary_for_key(self) -> None:
        """CoherenceProduct-like objects group by date_primary, not a 'date' attr."""
        crs = CRS.from_epsg(32632)

        # Since CoherenceProduct is frozen we cannot setattr safely for a
        # property backed by LUT.  Use a trivial subclass instead.
        class _CohNoLUT(CoherenceProduct):
            @property
            @override
            # Frozen fields are read-only; the base declares it read-write.
            # pyrefly: ignore [bad-override]
            def rel_orbit_number(self) -> int:
                return 99

        def _make_coh_nolut(date_primary: Timestamp, burst_id: int) -> _CohNoLUT:
            prod = object.__new__(_CohNoLUT)
            object.__setattr__(prod, "burst_id", burst_id)
            object.__setattr__(prod, "sub_swath", "IW1")
            object.__setattr__(prod, "crs", crs)
            object.__setattr__(prod, "date_primary", date_primary)
            object.__setattr__(prod, "date_secondary", Timestamp("2023-07-13"))
            object.__setattr__(prod, "polarization", "VV")
            return prod

        d = Timestamp("2023-07-01")
        result = self._mosaic(
            _burst(_make_coh_nolut(d, 1), _strip(0, 13, 1.0)),
            _burst(_make_coh_nolut(d, 2), _strip(8, 23, 2.0)),
        )
        # Both products share (rel_orbit_number=99, date_primary), so one group.
        assert list(result) == [(99, d)]


# ---------------------------------------------------------------------------
# acquisition_date
# ---------------------------------------------------------------------------


class TestAcquisitionDate:
    """Tests for ``acquisition_date``."""

    def test_rtc_product_returns_date(self) -> None:
        """For an RTCProduct-like object (not CoherenceProduct), returns .date."""

        class _FakeRTC:
            date = Timestamp("2023-06-15")

        # Stub carries only the date.
        # pyrefly: ignore [bad-argument-type]
        result = acquisition_date(_FakeRTC())
        assert result == Timestamp("2023-06-15")

    def test_coherence_product_returns_date_primary(self) -> None:
        """For a real CoherenceProduct instance, returns .date_primary."""
        # CoherenceProduct is a frozen dataclass; supply all required fields.
        # burst_id / sub_swath are needed but rel_orbit_number is a property
        # backed by LUT.  We bypass it by setting fields via object.__setattr__.
        crs = CRS.from_epsg(32632)
        prod = object.__new__(CoherenceProduct)
        object.__setattr__(prod, "burst_id", 0)
        object.__setattr__(prod, "sub_swath", "IW1")
        object.__setattr__(prod, "crs", crs)
        object.__setattr__(prod, "date_primary", Timestamp("2023-07-01"))
        object.__setattr__(prod, "date_secondary", Timestamp("2023-07-13"))
        object.__setattr__(prod, "polarization", "VV")

        result = acquisition_date(prod)
        assert result == Timestamp("2023-07-01")


# ---------------------------------------------------------------------------
# scattering_areas
# ---------------------------------------------------------------------------


class TestScatteringAreas:
    """Tests for ``scattering_areas``."""

    _D1 = Timestamp("2023-01-10")

    def _mosaics(self) -> tuple[dict[tuple[int, Timestamp], AcquisitionMosaic], RTCStub, RTCStub]:
        a = RTCStub(10, self._D1, "VV", burst_id=1)
        b = RTCStub(10, self._D1, "VV", burst_id=2)
        mosaics = mosaic_by_orbit_date(
            [_burst(a, _strip(0, 13, 1.0)), _burst(b, _strip(8, 23, 2.0))], "32TMS03"
        )
        return mosaics, a, b

    def test_area_comes_from_the_burst_of_each_value(self) -> None:
        """Each pixel's area is read from the RTC product of the burst its value came from."""
        mosaics, a, b = self._mosaics()
        cell = np.ones((3, 16), dtype=np.float32)
        # Stubs carry only the fields the lookup reads.
        # pyrefly: ignore [bad-argument-type]
        areas = scattering_areas(mosaics, [(b, cell * 20), (a, cell * 10)])
        [area] = areas.values()
        np.testing.assert_array_equal(area[1], [10.0] * 7 + [20.0] * 9)

    def test_a_missing_burst_raises(self) -> None:
        """A burst some pixel comes from without its RTC product is an input error."""
        mosaics, a, _ = self._mosaics()
        cell = np.ones((3, 16), dtype=np.float32)
        with pytest.raises(CompositeInputError, match=r"2 IW1 \(orbit 10, 2023-01-10\)"):
            # Stubs carry only the fields the lookup reads.
            # pyrefly: ignore [bad-argument-type]
            scattering_areas(mosaics, [(a, cell)])

    def test_a_burst_no_pixel_comes_from_needs_no_area(self) -> None:
        """An RTC product is needed only for the bursts the mosaic takes pixels from."""
        a = RTCStub(10, self._D1, "VV", burst_id=1)
        b = RTCStub(10, self._D1, "VV", burst_id=2)
        # b has data only beyond the cell.
        mosaics = mosaic_by_orbit_date(
            [_burst(a, np.ones((11, 24))), _burst(b, _strip(20, 23, 2.0))], "32TMS03"
        )
        cell = np.ones((3, 16), dtype=np.float32)
        # pyrefly: ignore [bad-argument-type]
        [area] = scattering_areas(mosaics, [(a, cell)]).values()
        np.testing.assert_array_equal(area, 1.0)
