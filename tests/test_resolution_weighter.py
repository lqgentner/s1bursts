"""Tests for ResolutionWeighter pure-array helpers."""

from __future__ import annotations

import hashlib
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, override
import warnings as _warnings

from affine import Affine
from freezebase.mgrs import MGRSGeoBox
import numpy as np
from pandas import Timestamp
from pyproj import CRS
import pytest
import rasterio

from s1bursts import product_tags
from s1bursts.errors import EmptyCompositeError
from s1bursts.filemanager import FileManager
from s1bursts.product import (
    RTCLRWQA,
    CoherenceProduct,
    GTCPassComposite,
    GTCProduct,
    QaMetric,
    RTCLRWComposite,
    RTCPassComposite,
    RTCPassQA,
    RTCProduct,
)
from s1bursts.resolution_weighter import ResolutionWeighter
from s1bursts.time_range import TimeRange

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from upath import UPath

# ---------------------------------------------------------------------------
# Lightweight product stubs for _stitch_by_orbit_date
#
# RTCProduct and CoherenceProduct derive rel_orbit_number via a LUT lookup
# (BurstFootprintMixin._lut_lookup → create_s1_orbit_number_lut).  Accessing
# it on a real product requires a valid (burst_id, sub_swath) pair in the ESA
# burst map, which we cannot guarantee in unit tests.  Instead we use small
# NamedTuple stubs that expose only the attributes consumed by
# _stitch_by_orbit_date / _acquisition_date.
# ---------------------------------------------------------------------------


class _RTCStub(NamedTuple):
    """Minimal RTCProduct-like stub."""

    rel_orbit_number: int
    date: Timestamp
    polarization: str


class _CohStub(NamedTuple):
    """Minimal CoherenceProduct-like stub (isinstance check uses the real class)."""

    rel_orbit_number: int
    date_primary: Timestamp
    polarization: str


def _make_weighter() -> ResolutionWeighter:
    """Return a ResolutionWeighter with a None file_manager.

    Safe to use for methods that never touch ``self._fm``.
    """
    w: ResolutionWeighter = object.__new__(ResolutionWeighter)
    # `_fm` is unused by the methods under test.
    # pyrefly: ignore [bad-assignment]
    w._fm = None
    w._n_workers = 1
    return w


# ---------------------------------------------------------------------------
# _calc_weights
# ---------------------------------------------------------------------------


class TestCalcWeights:
    """Tests for ``ResolutionWeighter._calc_weights``."""

    def test_normal_area(self) -> None:
        """Weight equals 1/area for a positive finite input."""
        sim = np.array([[[2.0]]], dtype=np.float32)
        result = ResolutionWeighter._calc_weights(sim, clip_max_resolution=None)
        np.testing.assert_allclose(result, [[[0.5]]])

    def test_zero_area_gives_zero_weight(self) -> None:
        """scattering_area == 0 must produce weight 0 (not inf)."""
        sim = np.array([[[0.0]]], dtype=np.float32)
        result = ResolutionWeighter._calc_weights(sim, clip_max_resolution=None)
        np.testing.assert_allclose(result, [[[0.0]]])

    def test_negative_area_gives_zero_weight(self) -> None:
        """scattering_area < 0 must produce weight 0."""
        sim = np.array([[[-1.0]]], dtype=np.float32)
        result = ResolutionWeighter._calc_weights(sim, clip_max_resolution=None)
        np.testing.assert_allclose(result, [[[0.0]]])

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_nonfinite_area_gives_zero_weight(self, bad: float) -> None:
        """Non-finite scattering_area values must produce weight 0."""
        sim = np.array([[[bad]]], dtype=np.float64)
        result = ResolutionWeighter._calc_weights(sim, clip_max_resolution=None)
        np.testing.assert_allclose(result, [[[0.0]]])

    def test_clip_max_resolution_clamps_large_weight(self) -> None:
        """A small area (high 1/area) must be clamped to clip_max_resolution."""
        # area=0.1 → 1/area=10.0; clip at 5.0 → weight=5.0
        sim = np.array([[[0.1]]], dtype=np.float64)
        result = ResolutionWeighter._calc_weights(sim, clip_max_resolution=5.0)
        np.testing.assert_allclose(result, [[[5.0]]])

    def test_clip_max_resolution_does_not_affect_small_weight(self) -> None:
        """A weight already below clip_max_resolution must be unchanged."""
        # area=4.0 → 1/area=0.25; clip at 5.0 → weight=0.25 (unchanged)
        sim = np.array([[[4.0]]], dtype=np.float64)
        result = ResolutionWeighter._calc_weights(sim, clip_max_resolution=5.0)
        np.testing.assert_allclose(result, [[[0.25]]])

    def test_clip_max_resolution_none_leaves_weights_unclipped(self) -> None:
        """clip_max_resolution=None must not limit any weight."""
        sim = np.array([[[0.01]]], dtype=np.float64)
        result = ResolutionWeighter._calc_weights(sim, clip_max_resolution=None)
        np.testing.assert_allclose(result, [[[100.0]]])


# ---------------------------------------------------------------------------
# _weighted_mean
# ---------------------------------------------------------------------------


class TestWeightedMean:
    """Tests for ``ResolutionWeighter._weighted_mean``."""

    def test_basic_weighted_mean(self) -> None:
        """Hand-computed weighted average over two pixels, two observations."""
        # shape (N=2, H=1, W=2)
        # pixel 0: (1*2 + 3*1) / (2+1) = 5/3
        # pixel 1: (2*1 + 4*3) / (1+3) = 14/4 = 3.5
        values = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        weights = np.array([[[2.0, 1.0]], [[1.0, 3.0]]])
        result = ResolutionWeighter._weighted_mean(values, weights)
        np.testing.assert_allclose(result, [[5.0 / 3.0, 3.5]])

    def test_nan_observation_excluded(self) -> None:
        """A NaN value must not contribute to the weighted average."""
        # pixel 0: NaN masked out → only obs2 with weight 1 → result=3.0
        # pixel 1: only finite obs1 with weight 2 → result=2.0
        values = np.array([[[np.nan, 2.0]], [[3.0, np.nan]]])
        weights = np.array([[[1.0, 2.0]], [[1.0, 1.0]]])
        result = ResolutionWeighter._weighted_mean(values, weights)
        np.testing.assert_allclose(result[0, 0], 3.0)
        np.testing.assert_allclose(result[0, 1], 2.0)

    def test_all_zero_weights_gives_nan(self) -> None:
        """A pixel with all-zero weights must produce NaN."""
        values = np.array([[[1.0]], [[2.0]]])
        weights = np.zeros((2, 1, 1))
        result = ResolutionWeighter._weighted_mean(values, weights)
        assert np.isnan(result[0, 0])

    def test_all_nan_values_gives_nan(self) -> None:
        """A pixel that is NaN in every observation must produce NaN."""
        values = np.array([[[np.nan]], [[np.nan]]])
        weights = np.ones((2, 1, 1))
        result = ResolutionWeighter._weighted_mean(values, weights)
        assert np.isnan(result[0, 0])


# ---------------------------------------------------------------------------
# _weighted_quantile
# ---------------------------------------------------------------------------


class TestWeightedQuantile:
    """Tests for ``ResolutionWeighter._weighted_quantile``."""

    def test_equal_weights_median_matches_numpy(self) -> None:
        """With equal weights the weighted median equals the interpolated median.

        values=[10, 20, 30, 40], equal weights.  Hazen positions are
        [0.125, 0.375, 0.625, 0.875]; q=0.5 falls halfway between the 0.375
        and 0.625 nodes → 0.5*(20+30) = 25, matching ``np.median``.
        """
        values = np.array([[[10.0]], [[20.0]], [[30.0]], [[40.0]]])
        weights = np.ones((4, 1, 1))
        result = ResolutionWeighter._weighted_quantile(values, weights, q=0.5)
        np.testing.assert_allclose(result, [[np.median([10.0, 20.0, 30.0, 40.0])]])

    def test_equal_weights_odd_median(self) -> None:
        """Odd count median returns the middle value (position 0.5 lands on it)."""
        values = np.array([[[10.0]], [[20.0]], [[30.0]]])
        weights = np.ones((3, 1, 1))
        result = ResolutionWeighter._weighted_quantile(values, weights, q=0.5)
        np.testing.assert_allclose(result, [[20.0]])

    def test_asymmetric_weights_pull_median(self) -> None:
        """A heavier central observation pulls the median onto its value.

        values=[1, 3, 5], weights=[1, 2, 1], total=4.  Hazen positions are
        [0.125, 0.5, 0.875]; q=0.5 lands exactly on the central node → 3.0.
        """
        values = np.array([[[1.0]], [[3.0]], [[5.0]]])
        weights = np.array([[[1.0]], [[2.0]], [[1.0]]])
        result = ResolutionWeighter._weighted_quantile(values, weights, q=0.5)
        np.testing.assert_allclose(result, [[3.0]])

    def test_interpolates_between_nodes(self) -> None:
        """Q between two plotting positions interpolates linearly.

        values=[10, 20, 30, 40], equal weights, positions as above.  q=0.25
        falls one quarter of the way past the 0.125 node toward 0.375:
        frac=(0.25-0.125)/0.25=0.5 → 0.5*(10+20)=15.
        """
        values = np.array([[[10.0]], [[20.0]], [[30.0]], [[40.0]]])
        weights = np.ones((4, 1, 1))
        result = ResolutionWeighter._weighted_quantile(values, weights, q=0.25)
        np.testing.assert_allclose(result, [[15.0]])

    def test_q0_returns_min_q1_returns_max(self) -> None:
        """q=0 clamps to the smallest value and q=1 to the largest."""
        values = np.array([[[2.0]], [[7.0]], [[4.0]]])
        weights = np.ones((3, 1, 1))
        res_min = ResolutionWeighter._weighted_quantile(values, weights, q=0.0)
        res_max = ResolutionWeighter._weighted_quantile(values, weights, q=1.0)
        np.testing.assert_allclose(res_min, [[2.0]])
        np.testing.assert_allclose(res_max, [[7.0]])

    def test_all_zero_weights_gives_nan(self) -> None:
        """A pixel with all-zero weights must produce NaN."""
        values = np.array([[[1.0]], [[2.0]]])
        weights = np.zeros((2, 1, 1))
        result = ResolutionWeighter._weighted_quantile(values, weights, q=0.5)
        assert np.isnan(result[0, 0])

    def test_nan_excluded_from_quantile(self) -> None:
        """NaN observations are masked out (zero weight) before interpolation.

        values=[nan, 3.0, 5.0], equal weights.  The NaN carries zero weight and
        sorts to the end, leaving two effective observations [3, 5] with Hazen
        positions [0.25, 0.75]; q=0.5 interpolates halfway → 4.0.
        """
        values = np.array([[[np.nan]], [[3.0]], [[5.0]]])
        weights = np.ones((3, 1, 1))
        result = ResolutionWeighter._weighted_quantile(values, weights, q=0.5)
        np.testing.assert_allclose(result, [[4.0]])

    def test_high_quantile_with_nan_does_not_return_nan(self) -> None:
        """A high q must clamp to the top finite value, never the masked NaN."""
        values = np.array([[[3.0]], [[5.0]], [[np.nan]]])
        weights = np.ones((3, 1, 1))
        result = ResolutionWeighter._weighted_quantile(values, weights, q=0.95)
        np.testing.assert_allclose(result, [[5.0]])

    def test_zero_weight_observation_is_excluded(self) -> None:
        """A finite value carrying zero weight must not reach the result.

        Previously it sorted by its own value rather than to the end, so the
        interpolation blended it in: values=[0, 2] with weights=[0, 0.5] at
        q=0.25 returned 1.0 instead of 2.0.
        """
        values = np.array([[[0.0]], [[2.0]]])
        weights = np.array([[[0.0]], [[0.5]]])
        result = ResolutionWeighter._weighted_quantile(values, weights, q=0.25)
        np.testing.assert_allclose(result, [[2.0]])

    def test_zero_weight_matches_dropping_the_observation(self) -> None:
        """Zero weight is equivalent to the observation not being there at all."""
        values = np.array([[[1.0]], [[5.0]], [[9.0]]])
        weights = np.array([[[0.0]], [[1.0]], [[1.0]]])
        for q in (0.2, 0.5, 0.8):
            with_zero = ResolutionWeighter._weighted_quantile(values, weights, q=q)
            without = ResolutionWeighter._weighted_quantile(values[1:], weights[1:], q=q)
            np.testing.assert_allclose(with_zero, without, err_msg=f"q={q}")


# ---------------------------------------------------------------------------
# _aggregate_unweighted
# ---------------------------------------------------------------------------


class TestAggregateUnweighted:
    """Tests for ``ResolutionWeighter._aggregate_unweighted``.

    The unweighted path delegates to the weighted path with uniform weights, so
    its ``median`` / quantile use the same interpolated definition as LRW.
    """

    def test_mean(self) -> None:
        """'mean' returns the arithmetic mean over axis 0."""
        values = np.array([[[1.0, 3.0]], [[3.0, 5.0]]])
        result = _make_weighter()._aggregate_unweighted(values, "mean")
        np.testing.assert_allclose(result, [[2.0, 4.0]])

    def test_median(self) -> None:
        """'median' returns the median over axis 0."""
        values = np.array([[[1.0]], [[3.0]], [[5.0]]])
        result = _make_weighter()._aggregate_unweighted(values, "median")
        np.testing.assert_allclose(result, [[3.0]])

    def test_float_quantile_uses_interpolated_definition(self) -> None:
        """A float aggregation selects the interpolated (Hazen) quantile.

        values=[0, 1, 2, 3], uniform weights → Hazen positions
        [0.125, 0.375, 0.625, 0.875]; q=0.75 interpolates halfway between the
        0.625 and 0.875 nodes → 0.5*(2+3) = 2.5.  (numpy's type-7 quantile would
        give 2.25; the LRW-consistent definition gives 2.5.)
        """
        values = np.array([[[0.0]], [[1.0]], [[2.0]], [[3.0]]])
        result = _make_weighter()._aggregate_unweighted(values, 0.75)
        np.testing.assert_allclose(result, [[2.5]])

    def test_matches_weighted_with_uniform_weights(self) -> None:
        """The unweighted result equals the weighted result with uniform weights."""
        values = np.array([[[1.0]], [[4.0]], [[8.0]]])
        weighter = _make_weighter()
        for aggregation in ("mean", "median", 0.25, 0.9):
            unweighted = weighter._aggregate_unweighted(values, aggregation)
            weighted = weighter._aggregate_weighted(values, np.ones_like(values), aggregation)
            np.testing.assert_allclose(unweighted, weighted)

    def test_unknown_string_raises(self) -> None:
        """An unknown string aggregation must raise ValueError."""
        values = np.array([[[1.0]]])
        with pytest.raises(ValueError, match="aggregation must be"):
            # pyrefly: ignore [bad-argument-type]
            _make_weighter()._aggregate_unweighted(values, "mode")

    def test_all_nan_column_yields_nan_without_warning(self) -> None:
        """All-NaN slices must produce NaN and must not raise RuntimeWarning."""
        values = np.array([[[np.nan]], [[np.nan]]])
        with _warnings.catch_warnings(record=True) as record:
            _warnings.simplefilter("always")
            result = _make_weighter()._aggregate_unweighted(values, "mean")
        runtime_warns = [w for w in record if issubclass(w.category, RuntimeWarning)]
        assert not runtime_warns, f"Unexpected RuntimeWarning(s): {runtime_warns}"
        assert np.isnan(result[0, 0])


# ---------------------------------------------------------------------------
# _first_notna
# ---------------------------------------------------------------------------


class TestFirstNotna:
    """Tests for ``ResolutionWeighter._first_notna``."""

    def test_first_finite_wins(self) -> None:
        """The first array's value is used wherever it is finite."""
        a = np.array([[1.0, np.nan]])
        b = np.array([[9.0, 7.0]])
        result = ResolutionWeighter._first_notna([a, b])
        np.testing.assert_allclose(result, [[1.0, 7.0]])

    def test_all_nan_stays_nan(self) -> None:
        """A position that is NaN in all arrays must remain NaN."""
        a = np.array([[np.nan, 1.0]])
        b = np.array([[np.nan, 2.0]])
        result = ResolutionWeighter._first_notna([a, b])
        assert np.isnan(result[0, 0])
        np.testing.assert_allclose(result[0, 1], 1.0)

    def test_three_arrays_first_wins(self) -> None:
        """With three arrays the first finite value across all layers is used."""
        a = np.array([[np.nan]])
        b = np.array([[np.nan]])
        c = np.array([[42.0]])
        result = ResolutionWeighter._first_notna([a, b, c])
        np.testing.assert_allclose(result, [[42.0]])

    def test_single_array_passthrough(self) -> None:
        """A single-element list must return the array unchanged (modulo NaN mask)."""
        a = np.array([[1.0, np.nan]])
        result = ResolutionWeighter._first_notna([a])
        np.testing.assert_allclose(result[0, 0], 1.0)
        assert np.isnan(result[0, 1])


# ---------------------------------------------------------------------------
# _stitch_by_orbit_date
# ---------------------------------------------------------------------------


class TestStitchByOrbitDate:
    """Tests for ``ResolutionWeighter._stitch_by_orbit_date``."""

    _D1 = Timestamp("2023-01-10")
    _D2 = Timestamp("2023-01-22")

    def _weighter(self) -> ResolutionWeighter:
        return _make_weighter()

    def test_two_products_same_group_are_mosaicked(self) -> None:
        """Two products with the same orbit/date must be merged into one array."""
        p1 = _RTCStub(rel_orbit_number=10, date=self._D1, polarization="VV")
        p2 = _RTCStub(rel_orbit_number=10, date=self._D1, polarization="VV")
        arr1 = np.array([[1.0, np.nan]])
        arr2 = np.array([[9.0, 5.0]])
        loaded = [(p1, arr1), (p2, arr2)]

        # Stub carries only the fields read here.
        # pyrefly: ignore [bad-argument-type]
        result = self._weighter()._stitch_by_orbit_date(loaded)

        assert len(result) == 1
        (orbit, _date), mosaic = next(iter(result.items()))
        assert orbit == 10
        # First non-NaN: pixel0 → 1.0 (from arr1), pixel1 → 5.0 (from arr2)
        np.testing.assert_allclose(mosaic, [[1.0, 5.0]])

    def test_different_dates_produce_separate_groups(self) -> None:
        """Products with different acquisition dates must produce separate keys."""
        p1 = _RTCStub(rel_orbit_number=10, date=self._D1, polarization="VV")
        p2 = _RTCStub(rel_orbit_number=10, date=self._D2, polarization="VV")
        arr1 = np.array([[1.0]])
        arr2 = np.array([[2.0]])
        loaded = [(p1, arr1), (p2, arr2)]

        # Stub carries only the fields read here.
        # pyrefly: ignore [bad-argument-type]
        result = self._weighter()._stitch_by_orbit_date(loaded)

        assert len(result) == 2

    def test_different_orbits_produce_separate_groups(self) -> None:
        """Products with different orbit numbers must produce separate keys."""
        p1 = _RTCStub(rel_orbit_number=10, date=self._D1, polarization="VV")
        p2 = _RTCStub(rel_orbit_number=20, date=self._D1, polarization="VV")
        arr1 = np.array([[1.0]])
        arr2 = np.array([[2.0]])
        loaded = [(p1, arr1), (p2, arr2)]

        # Stub carries only the fields read here.
        # pyrefly: ignore [bad-argument-type]
        result = self._weighter()._stitch_by_orbit_date(loaded)

        assert len(result) == 2

    def test_result_is_sorted_by_key(self) -> None:
        """Output dict must be sorted by (orbit_number, date) key."""
        p_late = _RTCStub(rel_orbit_number=10, date=self._D2, polarization="VV")
        p_early = _RTCStub(rel_orbit_number=10, date=self._D1, polarization="VV")
        arr = np.array([[1.0]])
        loaded = [(p_late, arr), (p_early, arr)]

        # Stub carries only the fields read here.
        # pyrefly: ignore [bad-argument-type]
        result = self._weighter()._stitch_by_orbit_date(loaded)

        keys = list(result.keys())
        assert keys == sorted(keys)
        # Earlier date must come first
        assert keys[0][1] < keys[1][1]

    def test_coherence_product_uses_date_primary_for_key(self) -> None:
        """CoherenceProduct-like objects group by date_primary, not a 'date' attr.

        Construct a frozen CoherenceProduct via object.__new__ (bypassing the
        LUT), give it date_primary='2023-07-01', then feed two copies with the
        same (rel_orbit_number=99, date_primary) into _stitch_by_orbit_date.
        They must collapse into a single group.
        """
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

        def _make_coh_nolut(date_primary: Timestamp) -> _CohNoLUT:
            prod = object.__new__(_CohNoLUT)
            object.__setattr__(prod, "burst_id", 0)
            object.__setattr__(prod, "sub_swath", "IW1")
            object.__setattr__(prod, "crs", crs)
            object.__setattr__(prod, "date_primary", date_primary)
            object.__setattr__(prod, "date_secondary", Timestamp("2023-07-13"))
            object.__setattr__(prod, "polarization", "VV")
            return prod

        d = Timestamp("2023-07-01")
        p1 = _make_coh_nolut(d)
        p2 = _make_coh_nolut(d)
        arr1 = np.array([[1.0, np.nan]])
        arr2 = np.array([[9.0, 5.0]])
        loaded = [(p1, arr1), (p2, arr2)]

        result = self._weighter()._stitch_by_orbit_date(loaded)

        # Both products have the same (rel_orbit_number=99, date_primary) →
        # they must be merged into a single key.
        assert len(result) == 1
        _, mosaic = next(iter(result.items()))
        np.testing.assert_allclose(mosaic, [[1.0, 5.0]])


# ---------------------------------------------------------------------------
# _acquisition_date
# ---------------------------------------------------------------------------


class TestAcquisitionDate:
    """Tests for ``ResolutionWeighter._acquisition_date``."""

    def test_rtc_product_returns_date(self) -> None:
        """For an RTCProduct-like object (not CoherenceProduct), returns .date."""

        class _FakeRTC:
            date = Timestamp("2023-06-15")

        # Stub carries only the date.
        # pyrefly: ignore [bad-argument-type]
        result = ResolutionWeighter._acquisition_date(_FakeRTC())
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

        result = ResolutionWeighter._acquisition_date(prod)
        assert result == Timestamp("2023-07-01")


# ---------------------------------------------------------------------------
# _check_polarization
# ---------------------------------------------------------------------------


class TestCheckPolarization:
    """Tests for ``ResolutionWeighter._check_polarization``."""

    def test_returns_shared_polarization(self) -> None:
        """Returns the common polarization when all products agree."""
        p1 = _RTCStub(rel_orbit_number=1, date=Timestamp("2023-01-01"), polarization="VV")
        p2 = _RTCStub(rel_orbit_number=2, date=Timestamp("2023-01-12"), polarization="VV")
        # Stub carries only the polarization.
        # pyrefly: ignore [bad-argument-type]
        result = ResolutionWeighter._check_polarization([p1, p2])
        assert result == "VV"

    def test_single_product_returns_its_polarization(self) -> None:
        """A list with one product must return that product's polarization."""
        p = _RTCStub(rel_orbit_number=1, date=Timestamp("2023-01-01"), polarization="HH")
        # Stub carries only the polarization.
        # pyrefly: ignore [bad-argument-type]
        result = ResolutionWeighter._check_polarization([p])
        assert result == "HH"

    def test_mismatched_polarizations_raise(self) -> None:
        """Differing polarizations must raise ValueError."""
        p1 = _RTCStub(rel_orbit_number=1, date=Timestamp("2023-01-01"), polarization="VV")
        p2 = _RTCStub(rel_orbit_number=2, date=Timestamp("2023-01-12"), polarization="VH")
        with pytest.raises(ValueError, match="polarization"):
            # pyrefly: ignore [bad-argument-type]
            ResolutionWeighter._check_polarization([p1, p2])


# ---------------------------------------------------------------------------
# _aggregate_weighted (dispatcher)
# ---------------------------------------------------------------------------


class TestAggregateWeighted:
    """Brief smoke-tests for the ``_aggregate_weighted`` dispatcher."""

    _values = np.array([[[1.0]], [[3.0]], [[5.0]]])
    _weights = np.ones((3, 1, 1))

    def _weighter(self) -> ResolutionWeighter:
        return _make_weighter()

    def test_mean_routes_to_weighted_mean(self) -> None:
        """'mean' must call _weighted_mean, not a quantile."""
        expected = ResolutionWeighter._weighted_mean(self._values, self._weights)
        result = self._weighter()._aggregate_weighted(self._values, self._weights, "mean")
        np.testing.assert_allclose(result, expected)

    def test_median_routes_to_q05_quantile(self) -> None:
        """'median' must produce the same result as _weighted_quantile(q=0.5)."""
        expected = ResolutionWeighter._weighted_quantile(self._values, self._weights, q=0.5)
        result = self._weighter()._aggregate_weighted(self._values, self._weights, "median")
        np.testing.assert_allclose(result, expected)

    def test_float_routes_to_quantile(self) -> None:
        """A float argument must route to _weighted_quantile at that level."""
        q = 0.75
        expected = ResolutionWeighter._weighted_quantile(self._values, self._weights, q=q)
        result = self._weighter()._aggregate_weighted(self._values, self._weights, q)
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("q", [0.0, 1.0, 1.5, -0.2])
    def test_quantile_outside_the_open_unit_interval_raises(self, q: float) -> None:
        # Previously accepted silently and returned the minimum or maximum.
        with pytest.raises(ValueError, match="quantile"):
            self._weighter()._aggregate_weighted(self._values, self._weights, q)

    def test_unknown_string_raises(self) -> None:
        """An unrecognised string must raise ValueError."""
        with pytest.raises(ValueError, match="aggregation must be"):
            self._weighter()._aggregate_weighted(
                self._values,
                self._weights,
                # Deliberately invalid aggregation.
                # pyrefly: ignore [bad-argument-type]
                "mode",
            )


# ---------------------------------------------------------------------------
# _check_area_coverage
# ---------------------------------------------------------------------------


class TestCheckSimCoverage:
    """Tests for ``ResolutionWeighter._check_area_coverage``."""

    def test_full_coverage_passes(self) -> None:
        """No error is raised when every observation group has a sim group."""
        arr: NDArray[np.floating[Any]] = np.zeros((1, 1))
        key_a = (10, Timestamp("2024-01-01"))
        key_b = (10, Timestamp("2024-01-13"))
        obs = {key_a: arr, key_b: arr}
        sim = {key_a: arr, key_b: arr}
        # Must not raise.
        ResolutionWeighter._check_area_coverage(obs, sim)

    def test_missing_group_raises(self) -> None:
        """A missing sim group raises ValueError naming the orbit/date."""
        arr: NDArray[np.floating[Any]] = np.zeros((1, 1))
        key_a = (10, Timestamp("2024-01-01"))
        key_b = (10, Timestamp("2024-01-13"))
        obs = {key_a: arr, key_b: arr}
        sim = {key_a: arr}  # key_b missing
        with pytest.raises(ValueError, match=r"orbit 10, 2024-01-13"):
            ResolutionWeighter._check_area_coverage(obs, sim)


# ---------------------------------------------------------------------------
# _count_observations
# ---------------------------------------------------------------------------


class TestCountObservations:
    """Tests for ``ResolutionWeighter._count_observations``."""

    def test_counts_finite_per_pixel(self) -> None:
        """Each pixel counts only its finite observations across the N axis."""
        # shape (N=3, H=1, W=2)
        # pixel 0: finite at obs 0,1 -> 2 ; pixel 1: finite at obs 2 -> 1
        values = np.array([[[1.0, np.nan]], [[2.0, np.nan]], [[np.nan, 5.0]]])
        result = ResolutionWeighter._count_observations(values)
        np.testing.assert_array_equal(result, [[2, 1]])

    def test_all_nan_pixel_is_zero(self) -> None:
        """A pixel that is NaN in every observation counts zero."""
        values = np.array([[[np.nan]], [[np.nan]]])
        result = ResolutionWeighter._count_observations(values)
        np.testing.assert_array_equal(result, [[0]])

    def test_returns_uint16(self) -> None:
        """The count is returned as uint16 for compact storage."""
        values = np.array([[[1.0]], [[2.0]]])
        assert ResolutionWeighter._count_observations(values).dtype == np.uint16


# ---------------------------------------------------------------------------
# _count_contributions
# ---------------------------------------------------------------------------


class TestCountContributions:
    """Tests for ``ResolutionWeighter._count_contributions``."""

    def test_counts_finite_observations_with_weight(self) -> None:
        """An observation counts when it is both finite and weighted."""
        # shape (N=3, H=1, W=2)
        values = np.array([[[1.0, np.nan]], [[2.0, np.nan]], [[np.nan, 5.0]]])
        weights = np.ones_like(values)
        result = ResolutionWeighter._count_contributions(values, weights)
        np.testing.assert_array_equal(result, [[2, 1]])

    def test_zero_weight_observation_is_not_counted(self) -> None:
        """A finite value whose scattering area was invalid must not count."""
        values = np.array([[[1.0]], [[2.0]]])
        weights = np.array([[[0.0]], [[1.0]]])
        result = ResolutionWeighter._count_contributions(values, weights)
        np.testing.assert_array_equal(result, [[1]])

    def test_weighted_nan_observation_is_not_counted(self) -> None:
        """A weight without a value contributes nothing and must not count."""
        values = np.array([[[np.nan]], [[2.0]]])
        weights = np.array([[[1.0]], [[1.0]]])
        result = ResolutionWeighter._count_contributions(values, weights)
        np.testing.assert_array_equal(result, [[1]])

    def test_matches_the_weights_the_mean_sums_over(self) -> None:
        """The count equals the observations ``_weighted_mean`` keeps."""
        values = np.array([[[1.0]], [[np.nan]], [[3.0]]])
        weights = np.array([[[2.0]], [[2.0]], [[0.0]]])
        masked = np.where(np.isfinite(values), weights, 0.0)
        expected = np.count_nonzero(masked > 0, axis=0)
        result = ResolutionWeighter._count_contributions(values, weights)
        np.testing.assert_array_equal(result, expected)

    def test_returns_uint16(self) -> None:
        """The count is returned as uint16 for compact storage."""
        values = np.array([[[1.0]], [[2.0]]])
        weights = np.ones_like(values)
        assert ResolutionWeighter._count_contributions(values, weights).dtype == np.uint16


# ---------------------------------------------------------------------------
# _calc_cqm
# ---------------------------------------------------------------------------


class TestCalcCqm:
    """Tests for ``ResolutionWeighter._calc_cqm``."""

    def test_matches_the_published_equation(self) -> None:
        """CQM is Small et al. eq. (6): 1 / sum(normalized_weight * area).

        areas=[2, 4, 10], weights=1/area=[0.5, 0.25, 0.1].  With unclipped
        weights this reduces to ``sum(w) / N`` = (0.5+0.25+0.1)/3.
        """
        scattering_area = np.array([[[2.0]], [[4.0]], [[10.0]]])
        weights = 1.0 / scattering_area
        values = np.ones((3, 1, 1))
        result = ResolutionWeighter._calc_cqm(values, scattering_area, weights)
        np.testing.assert_allclose(result, [[(0.5 + 0.25 + 0.1) / 3.0]], rtol=1e-6)

    def test_clipped_weights_change_cqm(self) -> None:
        """CQM uses the supplied (possibly clipped) weights verbatim.

        areas=[0.25, 10] give 1/area=[4, 0.1]; clipped at 2 the weights become
        [2, 0.1] (sum 2.1).  CQM = 1 / Σ(wgt·A) with normalized weights:
        Σ(w·A) = (2·0.25 + 0.1·10) / 2.1 = 1.5 / 2.1, so CQM = 2.1 / 1.5 = 1.4.
        """
        scattering_area = np.array([[[0.25]], [[10.0]]])
        weights = np.array([[[2.0]], [[0.1]]])  # 1/area=[4, 0.1] clipped at 2
        values = np.ones((2, 1, 1))
        result = ResolutionWeighter._calc_cqm(values, scattering_area, weights)
        np.testing.assert_allclose(result, [[1.4]], rtol=1e-6)

    def test_observations_that_do_not_contribute_are_excluded(self) -> None:
        """An observation with a non-finite value contributes no area either.

        Previously its area entered the CQM even though the composite and the
        QA-NUM count both excluded it.
        """
        scattering_area = np.array([[[1.0]], [[4.0]]])
        weights = 1.0 / scattering_area
        values = np.array([[[0.8]], [[np.nan]]])
        result = ResolutionWeighter._calc_cqm(values, scattering_area, weights)
        np.testing.assert_allclose(result, [[1.0]], rtol=1e-6)

    def test_exclusion_agrees_with_the_contributing_observation_count(self) -> None:
        """CQM and QA-NUM describe the same set of observations."""
        scattering_area = np.array([[[1.0]], [[4.0]]])
        weights = 1.0 / scattering_area
        values = np.array([[[0.8]], [[np.nan]]])
        counted = ResolutionWeighter._count_contributions(values, weights)
        only_contributor = ResolutionWeighter._calc_cqm(
            values[:1], scattering_area[:1], weights[:1]
        )
        result = ResolutionWeighter._calc_cqm(values, scattering_area, weights)
        assert counted[0, 0] == 1
        np.testing.assert_allclose(result, only_contributor, rtol=1e-6)

    def test_zero_total_weight_is_nan(self) -> None:
        """A pixel whose weights are all zero yields NaN."""
        scattering_area = np.array([[[2.0]], [[4.0]]])
        weights = np.zeros((2, 1, 1))
        values = np.ones((2, 1, 1))
        result = ResolutionWeighter._calc_cqm(values, scattering_area, weights)
        assert np.isnan(result[0, 0])

    def test_all_nan_observations_is_nan(self) -> None:
        """A pixel with valid area but no finite observation yields NaN.

        This keeps CQM's invalid footprint aligned with the composite, which is
        also NaN where every observation is non-finite.
        """
        scattering_area = np.array([[[2.0]], [[4.0]]])
        weights = 1.0 / scattering_area  # valid geometry, nonzero weight
        values = np.array([[[np.nan]], [[np.nan]]])  # but no finite observation
        result = ResolutionWeighter._calc_cqm(values, scattering_area, weights)
        assert np.isnan(result[0, 0])

    def test_returns_float32(self) -> None:
        """CQM is returned as float32."""
        scattering_area = np.array([[[2.0]]])
        weights = np.array([[[0.5]]])
        values = np.ones((1, 1, 1))
        result = ResolutionWeighter._calc_cqm(values, scattering_area, weights)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# compose_* return shapes (integration)
# ---------------------------------------------------------------------------


def _rtc_stub(orbit: int, day: int, direction: str = "Ascending") -> SimpleNamespace:
    """Minimal RTCProduct-like stub consumed by the compose orchestration."""
    return SimpleNamespace(
        polarization="VV",
        date=Timestamp(f"2020-01-{day:02d}"),
        rel_orbit_number=orbit,
        pass_direction=direction,
    )


class TestComposeReturns:
    """Integration tests for the ``compose_*`` return shapes (``write_qa`` on/off).

    ``_load_band`` is monkeypatched to feed synthetic band stacks, so no real
    input GeoTIFFs are needed; ``_save`` writes tiny real GeoTIFFs into a
    ``tmp_path`` ``FileManager``.
    """

    @staticmethod
    def _setup(tmp_path: Path) -> tuple[FileManager, ResolutionWeighter, SimpleNamespace]:
        fm = FileManager(tmp_path)
        weighter = ResolutionWeighter(fm, n_workers=1)
        geobox = SimpleNamespace(
            mgrs_code="32TMS03",
            crs=CRS.from_epsg(32632),
            transform=Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5100000.0),
            height=2,
            width=2,
        )
        return fm, weighter, geobox

    @staticmethod
    def _patch_load(
        weighter: ResolutionWeighter,
        power: dict[int, np.ndarray],
        area: dict[int, np.ndarray] | None = None,
    ) -> None:
        def fake_load(
            products: list[Any],
            geobox: object,  # noqa: ARG001
            band: int = 1,
        ) -> list[tuple[Any, np.ndarray]]:
            src = power if band == 1 else area
            assert src is not None
            return [(p, src[id(p)]) for p in products]

        # Instance attributes shadow the real windowed reader and the provenance
        # lookup, which the stubs have no file for. Written through ``vars`` because
        # Pyrefly 1.4 rejects assigning to a method, and 1.3 reports it under a
        # different code, so no single ignore comment fits both.
        vars(weighter).update(_load_band=fake_load, _input_tags=lambda _product: {})

    def test_rtc_lrw_without_qa_returns_bare_composite(self, tmp_path: Path) -> None:
        """``write_qa=False`` returns just the composite (unchanged)."""
        fm, weighter, geobox = self._setup(tmp_path)
        p1, p2 = _rtc_stub(10, 1), _rtc_stub(20, 13)
        power = {
            id(p1): np.full((2, 2), 1.0, np.float32),
            id(p2): np.full((2, 2), 2.0, np.float32),
        }
        area = {id(p1): np.full((2, 2), 2.0, np.float32), id(p2): np.full((2, 2), 4.0, np.float32)}
        self._patch_load(weighter, power, area)

        # Stubs stand in for real RTC products.
        # pyrefly: ignore [no-matching-overload]
        result = weighter.compose_rtc_lrw([p1, p2], geobox, aggregation="mean")
        assert isinstance(result, RTCLRWComposite)
        assert fm.get_product_path(result).exists()

    def test_rtc_lrw_with_qa_returns_nested_tuple(self, tmp_path: Path) -> None:
        """``write_qa=True`` returns ``(composite, (qa_num, qa_cqm))`` and writes the files."""
        fm, weighter, geobox = self._setup(tmp_path)
        p1, p2 = _rtc_stub(10, 1), _rtc_stub(20, 13)
        power = {
            id(p1): np.full((2, 2), 1.0, np.float32),
            id(p2): np.full((2, 2), 2.0, np.float32),
        }
        area = {id(p1): np.full((2, 2), 2.0, np.float32), id(p2): np.full((2, 2), 4.0, np.float32)}
        self._patch_load(weighter, power, area)

        # Stubs stand in for real RTC products.
        # pyrefly: ignore [no-matching-overload]
        composite, (qa_num, qa_cqm) = weighter.compose_rtc_lrw(
            [p1, p2],
            geobox,
            aggregation="mean",
            write_qa=True,
        )
        assert isinstance(composite, RTCLRWComposite)
        assert qa_num.qa_metric is QaMetric.NUM
        assert qa_cqm.qa_metric is QaMetric.CQM
        assert fm.get_product_path(qa_num).exists()
        assert fm.get_product_path(qa_cqm).exists()

    def test_rtc_pass_without_qa_returns_dict_of_composites(self, tmp_path: Path) -> None:
        """``write_qa=False`` returns a dict of bare composites per direction (unchanged)."""
        _fm, weighter, geobox = self._setup(tmp_path)
        asc, desc = _rtc_stub(10, 1, "Ascending"), _rtc_stub(11, 2, "Descending")
        power = {
            id(asc): np.full((2, 2), 1.0, np.float32),
            id(desc): np.full((2, 2), 2.0, np.float32),
        }
        self._patch_load(weighter, power)

        # Stubs stand in for real RTC products.
        # pyrefly: ignore [bad-argument-type]
        result = weighter.compose_rtc_by_pass([asc, desc], geobox)
        assert set(result) == {"Ascending", "Descending"}
        assert all(isinstance(v, RTCPassComposite) for v in result.values())

    def test_rtc_pass_with_qa_returns_dict_of_tuples(self, tmp_path: Path) -> None:
        """``write_qa=True`` returns ``dict[dir, (composite, qa_num)]`` and writes the files."""
        fm, weighter, geobox = self._setup(tmp_path)
        asc, desc = _rtc_stub(10, 1, "Ascending"), _rtc_stub(11, 2, "Descending")
        power = {
            id(asc): np.full((2, 2), 1.0, np.float32),
            id(desc): np.full((2, 2), 2.0, np.float32),
        }
        self._patch_load(weighter, power)

        # Stubs stand in for real RTC products.
        # pyrefly: ignore [no-matching-overload]
        result = weighter.compose_rtc_by_pass([asc, desc], geobox, write_qa=True)
        composite, qa_num = result["Ascending"]
        assert isinstance(composite, RTCPassComposite)
        assert isinstance(qa_num, RTCPassQA)
        assert qa_num.pass_direction == "Ascending"
        assert fm.get_product_path(qa_num).exists()


class TestSaveStatisticsAndChecksum:
    """``_save`` embeds exact band statistics, and can record a checksum."""

    def test_composite_carries_exact_band_statistics(self, tmp_path: Path) -> None:
        fm, weighter, geobox = TestComposeReturns._setup(tmp_path)
        p1, p2 = _rtc_stub(10, 1), _rtc_stub(20, 13)
        power = {
            id(p1): np.full((2, 2), 1.0, np.float32),
            id(p2): np.full((2, 2), 3.0, np.float32),
        }
        area = {id(p1): np.full((2, 2), 2.0, np.float32), id(p2): np.full((2, 2), 4.0, np.float32)}
        TestComposeReturns._patch_load(weighter, power, area)

        # Stubs stand in for real RTC products.
        # pyrefly: ignore [no-matching-overload]
        result = weighter.compose_rtc_lrw([p1, p2], geobox, aggregation="mean")

        with rasterio.open(fm.get_product_path(result)) as ds:
            tags = ds.tags(1)
        minimum, maximum = float(tags["STATISTICS_MINIMUM"]), float(tags["STATISTICS_MAXIMUM"])
        assert minimum == pytest.approx(maximum)
        assert float(tags["STATISTICS_VALID_PERCENT"]) == pytest.approx(100.0)
        assert "STATISTICS_APPROXIMATE" not in tags

    def test_qa_num_statistics_treat_zero_as_nodata_not_finite(self, tmp_path: Path) -> None:
        fm, weighter, geobox = TestComposeReturns._setup(tmp_path)
        p1, p2 = _rtc_stub(10, 1), _rtc_stub(20, 13)
        power = {
            id(p1): np.full((2, 2), 1.0, np.float32),
            id(p2): np.array([[2.0, np.nan], [2.0, np.nan]], np.float32),
        }
        area = {id(p1): np.full((2, 2), 2.0, np.float32), id(p2): np.full((2, 2), 4.0, np.float32)}
        TestComposeReturns._patch_load(weighter, power, area)

        # Stubs stand in for real RTC products.
        # pyrefly: ignore [no-matching-overload]
        _composite, (qa_num, _qa_cqm) = weighter.compose_rtc_lrw(
            [p1, p2],
            geobox,
            aggregation="mean",
            write_qa=True,
        )

        with rasterio.open(fm.get_product_path(qa_num)) as ds:
            tags = ds.tags(1)
        # Every pixel has p1's contribution, so num_obs is never the uint16
        # nodata sentinel 0 — every one of the 4 pixels is "valid".
        assert float(tags["STATISTICS_VALID_PERCENT"]) == pytest.approx(100.0)

    def test_checksum_defaults_to_off_and_records_nothing(self, tmp_path: Path) -> None:
        _fm, weighter, geobox = TestComposeReturns._setup(tmp_path)
        p1, p2 = _rtc_stub(10, 1), _rtc_stub(20, 13)
        power = {
            id(p1): np.full((2, 2), 1.0, np.float32),
            id(p2): np.full((2, 2), 2.0, np.float32),
        }
        area = {id(p1): np.full((2, 2), 2.0, np.float32), id(p2): np.full((2, 2), 4.0, np.float32)}
        TestComposeReturns._patch_load(weighter, power, area)

        # Stubs stand in for real RTC products.
        # pyrefly: ignore [no-matching-overload]
        weighter.compose_rtc_lrw([p1, p2], geobox, aggregation="mean")

        assert weighter.checksums == {}

    def test_checksum_true_records_a_multihash_matching_the_file(self, tmp_path: Path) -> None:
        fm = FileManager(tmp_path)
        weighter = ResolutionWeighter(fm, n_workers=1, checksum=True)
        geobox = SimpleNamespace(
            mgrs_code="32TMS03",
            crs=CRS.from_epsg(32632),
            transform=Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5100000.0),
            height=2,
            width=2,
        )
        p1, p2 = _rtc_stub(10, 1), _rtc_stub(20, 13)
        power = {
            id(p1): np.full((2, 2), 1.0, np.float32),
            id(p2): np.full((2, 2), 2.0, np.float32),
        }
        area = {id(p1): np.full((2, 2), 2.0, np.float32), id(p2): np.full((2, 2), 4.0, np.float32)}
        TestComposeReturns._patch_load(weighter, power, area)

        # Stubs stand in for real RTC products.
        # pyrefly: ignore [no-matching-overload]
        result = weighter.compose_rtc_lrw([p1, p2], geobox, aggregation="mean")

        path = fm.get_product_path(result)
        digest = weighter.checksums[str(path)]
        assert digest.startswith("1220")
        assert digest == "1220" + hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# End-to-end composites
#
# These drive the public compose_* methods over real GeoTIFFs on disk, so the
# windowed reads, weighting, aggregation, COG writing, QA layers and
# skip-existing logic all execute.
# ---------------------------------------------------------------------------

_UTM32 = CRS.from_epsg(32632)
_MGRS_CODE = "32TMS"
_TILE = 8


def _geobox() -> MGRSGeoBox:
    return MGRSGeoBox(
        _MGRS_CODE,
        (_TILE, _TILE),
        Affine(10.0, 0.0, 400000.0, 0.0, -10.0, 5200000.0),
        _UTM32,
    )


def _write_raster(
    path: Path | UPath,
    bands: list[NDArray[np.floating[Any]]],
    tags: dict[str, str] | None = None,
) -> None:
    """Write a small georeferenced GeoTIFF aligned to the test geobox."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=_TILE,
        height=_TILE,
        count=len(bands),
        dtype="float32",
        crs=_UTM32,
        transform=Affine(10.0, 0.0, 400000.0, 0.0, -10.0, 5200000.0),
        nodata=np.nan,
    ) as dst:
        for index, band in enumerate(bands, 1):
            dst.write(band.astype("float32"), index)
        if tags:
            dst.update_tags(**tags)


def _rtc(date: str, orbit: int = 15, burst_id: int = 30346) -> RTCProduct:
    return RTCProduct(
        burst_id=burst_id,
        sub_swath="IW1",
        crs=_UTM32,
        date=Timestamp(date),
        polarization="VV",
        _rel_orbit_number=orbit,
    )


def _coh(
    primary: str,
    secondary: str,
    orbit: int = 15,
    burst_id: int = 30346,
) -> CoherenceProduct:
    return CoherenceProduct(
        burst_id=burst_id,
        sub_swath="IW1",
        crs=_UTM32,
        date_primary=Timestamp(primary),
        date_secondary=Timestamp(secondary),
        polarization="VV",
        _rel_orbit_number=orbit,
    )


class _Scene(NamedTuple):
    weighter: ResolutionWeighter
    manager: FileManager
    geobox: MGRSGeoBox


# Burst IDs used by the composite tests, and the pass direction each stands for.
# Products resolve `pass_direction` through the burst-map LUT, so the table is
# stubbed to keep these tests offline and independent of the cached map.
_ASCENDING_BURST = 30346
_DESCENDING_BURST = 30347
_DIRECTION_LUT = {
    (_ASCENDING_BURST, "IW1"): "Ascending",
    (_DESCENDING_BURST, "IW1"): "Descending",
}


@pytest.fixture
def scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Scene:
    monkeypatch.setattr(
        "s1bursts.burst.create_s1_pass_direction_lut",
        lambda mode="IW": _DIRECTION_LUT,  # noqa: ARG005
    )
    manager = FileManager(base_dir=tmp_path)
    return _Scene(ResolutionWeighter(manager, n_workers=2), manager, _geobox())


def _full(value: float) -> NDArray[np.float32]:
    return np.full((_TILE, _TILE), value, dtype="float32")


class TestComposeRtcLrw:
    def test_weights_by_scattering_area_and_writes_a_readable_composite(
        self,
        scene: _Scene,
    ) -> None:
        """The scene with the smaller scattering area must dominate the result.

        A small scattering area means fine local resolution, so Small et al.'s
        weighting gives it the larger weight.
        """
        coarse, fine = _rtc("2022-08-13"), _rtc("2022-08-25")
        _write_raster(scene.manager.get_product_path(coarse), [_full(1.0), _full(100.0)])
        _write_raster(scene.manager.get_product_path(fine), [_full(9.0), _full(1.0)])

        product = scene.weighter.compose_rtc_lrw([coarse, fine], scene.geobox)

        with rasterio.open(scene.manager.get_product_path(product)) as src:
            values = src.read(1)
            assert src.crs == _UTM32
            assert src.shape == (_TILE, _TILE)
        # weights 1/100 and 1/1 → (1*0.01 + 9*1) / 1.01
        assert values == pytest.approx(np.full((_TILE, _TILE), (1 * 0.01 + 9) / 1.01), rel=1e-5)

    def test_rejects_an_out_of_range_quantile_before_reading_inputs(
        self,
        scene: _Scene,
    ) -> None:
        product = _rtc("2022-08-13")
        # No raster is written for it, so a read attempt would fail differently.
        with pytest.raises(ValueError, match="quantile"):
            scene.weighter.compose_rtc_lrw([product], scene.geobox, aggregation=1.5)

    def test_writes_the_requested_qa_layers(self, scene: _Scene) -> None:
        products = [_rtc("2022-08-13"), _rtc("2022-08-25")]
        for product in products:
            _write_raster(scene.manager.get_product_path(product), [_full(2.0), _full(4.0)])

        composite, (num_obs, cqm) = scene.weighter.compose_rtc_lrw(
            products,
            scene.geobox,
            write_qa=True,
        )

        assert scene.manager.get_product_path(composite).exists()
        with rasterio.open(scene.manager.get_product_path(num_obs)) as src:
            assert src.dtypes[0] == "uint16"
            assert (src.read(1) == 2).all()
        assert scene.manager.get_product_path(cqm).exists()

    @pytest.mark.parametrize("aggregation", ["mean", "median", 0.25, 0.75])
    def test_cqm_is_the_weighted_mean_form_under_every_aggregation(
        self,
        scene: _Scene,
        aggregation: Literal["mean", "median"] | float,
    ) -> None:
        """QA-CQM is 1 / sum(W * A) whatever statistic the composite uses.

        areas=[1, 2, 10] give weights [1, 0.5, 0.1] (sum 1.6), so sum(W * A) =
        3 / 1.6 and CQM = 1.6 / 3.  The brightest observation has the largest
        area and the smallest weight, so the value order differs from the area
        order.  Previously ``median`` wrote 1 / wquantile(A; W) = 1 / 1.4.
        """
        products = [_rtc("2022-08-13"), _rtc("2022-08-25"), _rtc("2022-09-06")]
        for product, value, area in zip(products, [2.0, 1.0, 9.0], [1.0, 2.0, 10.0], strict=True):
            _write_raster(scene.manager.get_product_path(product), [_full(value), _full(area)])

        _, (_, cqm) = scene.weighter.compose_rtc_lrw(
            products,
            scene.geobox,
            aggregation=aggregation,
            write_qa=True,
        )

        with rasterio.open(scene.manager.get_product_path(cqm)) as src:
            assert src.read(1) == pytest.approx(_full(1.6 / 3.0), rel=1e-6)

    def test_counts_only_observed_pixels_in_the_qa_layer(self, scene: _Scene) -> None:
        """A NaN pixel in one scene must lower that pixel's observation count."""
        first, second = _rtc("2022-08-13"), _rtc("2022-08-25")
        partial = _full(2.0)
        partial[0, 0] = np.nan
        _write_raster(scene.manager.get_product_path(first), [partial, _full(4.0)])
        _write_raster(scene.manager.get_product_path(second), [_full(2.0), _full(4.0)])

        _, (num_obs, _) = scene.weighter.compose_rtc_lrw(
            [first, second],
            scene.geobox,
            write_qa=True,
        )

        with rasterio.open(scene.manager.get_product_path(num_obs)) as src:
            counts = src.read(1)
        assert counts[0, 0] == 1
        assert counts[1, 1] == 2

    def test_unweighted_pixels_do_not_count_in_the_qa_layer(self, scene: _Scene) -> None:
        """A pixel with a value but no scattering area carries no weight.

        SNAP masks the intensity and the area band independently, so a finite
        intensity can meet a missing area.  Such a pixel contributes nothing to
        the composite and must not be counted (CEOS-ARD SAR v1.3.1, item 2.21).
        """
        first, second = _rtc("2022-08-13"), _rtc("2022-08-25")
        no_area = _full(4.0)
        no_area[0, 0] = np.nan
        _write_raster(scene.manager.get_product_path(first), [_full(2.0), no_area])
        _write_raster(scene.manager.get_product_path(second), [_full(2.0), _full(4.0)])

        _, (num_obs, _) = scene.weighter.compose_rtc_lrw(
            [first, second],
            scene.geobox,
            write_qa=True,
        )

        with rasterio.open(scene.manager.get_product_path(num_obs)) as src:
            counts = src.read(1)
        assert counts[0, 0] == 1
        assert counts[1, 1] == 2

    def test_skips_an_existing_composite_unless_overwritten(self, scene: _Scene) -> None:
        products = [_rtc("2022-08-13"), _rtc("2022-08-25")]
        for product in products:
            _write_raster(scene.manager.get_product_path(product), [_full(3.0), _full(1.0)])
        composite = scene.weighter.compose_rtc_lrw(products, scene.geobox)
        path = scene.manager.get_product_path(composite)
        path.write_bytes(b"sentinel: not a raster")

        scene.weighter.compose_rtc_lrw(products, scene.geobox)

        assert path.read_bytes() == b"sentinel: not a raster"

    def test_overwrite_replaces_an_existing_composite(self, scene: _Scene) -> None:
        products = [_rtc("2022-08-13"), _rtc("2022-08-25")]
        for product in products:
            _write_raster(scene.manager.get_product_path(product), [_full(3.0), _full(1.0)])
        composite = scene.weighter.compose_rtc_lrw(products, scene.geobox)
        path = scene.manager.get_product_path(composite)
        path.write_bytes(b"stale")

        scene.weighter.compose_rtc_lrw(products, scene.geobox, overwrite=True)

        with rasterio.open(path) as src:
            assert src.read(1) == pytest.approx(_full(3.0))

    def test_rejects_an_empty_product_collection(self, scene: _Scene) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            scene.weighter.compose_rtc_lrw([], scene.geobox)

    def test_reports_a_missing_product_file(self, scene: _Scene) -> None:
        """A catalogued product whose raster was never written must fail loudly."""
        with pytest.raises(FileNotFoundError, match="product file not found"):
            scene.weighter.compose_rtc_lrw([_rtc("2022-08-25")], scene.geobox)


class TestEmptyObservations:
    """A burst can touch a cell's footprint without reaching any of its pixels."""

    def test_an_rtc_cell_no_input_reaches_writes_nothing(
        self, scene: _Scene, tmp_path: Path
    ) -> None:
        product = _rtc("2022-08-13")
        _write_raster(scene.manager.get_product_path(product), [_full(np.nan), _full(np.nan)])
        before = set(tmp_path.rglob("*.tif"))

        with pytest.raises(EmptyCompositeError, match="no input observation"):
            scene.weighter.compose_rtc_lrw([product], scene.geobox, write_qa=True)

        assert set(tmp_path.rglob("*.tif")) == before

    def test_a_coherence_cell_no_input_reaches_writes_nothing(
        self, scene: _Scene, tmp_path: Path
    ) -> None:
        coh, rtc = _coh("2022-08-01", "2022-08-13"), _rtc("2022-08-01")
        _write_raster(scene.manager.get_product_path(coh), [_full(np.nan)])
        _write_raster(scene.manager.get_product_path(rtc), [_full(1.0), _full(1.0)])
        before = set(tmp_path.rglob("*.tif"))

        with pytest.raises(EmptyCompositeError):
            scene.weighter.compose_coherence_lrw([coh], [rtc], scene.geobox, write_qa=True)

        assert set(tmp_path.rglob("*.tif")) == before

    def test_the_tags_count_and_date_only_inputs_that_reach_the_cell(self, scene: _Scene) -> None:
        empty, valid = _rtc("2022-08-01", orbit=66), _rtc("2022-08-13")
        _write_raster(scene.manager.get_product_path(empty), [_full(np.nan), _full(np.nan)])
        _write_raster(scene.manager.get_product_path(valid), [_full(2.0), _full(4.0)])

        composite = scene.weighter.compose_rtc_lrw([empty, valid], scene.geobox)

        with rasterio.open(scene.manager.get_product_path(composite)) as src:
            tags = src.tags()
        assert tags["CONTRIBUTING_OBSERVATIONS"] == "1"
        assert tags["FIRST_ACQUISITION_DATE"] == "2022-08-13"
        assert tags["LAST_ACQUISITION_DATE"] == "2022-08-13"

    def test_the_tagged_span_matches_a_composite_of_the_reaching_inputs_alone(
        self, scene: _Scene, tmp_path: Path
    ) -> None:
        """Dropping an empty input must date the rest as if it had never been passed.

        The acquisition carries a UTC offset, so a span taken from the group
        keys (local midnight) would differ from one taken from the products.
        """
        empty = _rtc("2022-08-01T10:00+02:00", orbit=66)
        valid = _rtc("2022-08-13T12:00+02:00")
        _write_raster(scene.manager.get_product_path(empty), [_full(np.nan), _full(np.nan)])
        _write_raster(scene.manager.get_product_path(valid), [_full(2.0), _full(4.0)])
        alone = _Scene(
            ResolutionWeighter(FileManager(base_dir=tmp_path / "alone"), n_workers=2),
            FileManager(base_dir=tmp_path / "alone"),
            scene.geobox,
        )
        _write_raster(alone.manager.get_product_path(valid), [_full(2.0), _full(4.0)])

        mixed = scene.weighter.compose_rtc_lrw([empty, valid], scene.geobox)
        single = alone.weighter.compose_rtc_lrw([valid], alone.geobox)

        tags = {}
        for name, (manager, product) in {
            "mixed": (scene.manager, mixed),
            "single": (alone.manager, single),
        }.items():
            with rasterio.open(manager.get_product_path(product)) as src:
                tags[name] = src.tags()
        for key in ("FIRST_ACQUISITION_DATE", "LAST_ACQUISITION_DATE"):
            assert tags["mixed"][key] == tags["single"][key]

    def test_a_coherence_composite_counts_only_pairs_that_reach_the_cell(
        self, scene: _Scene
    ) -> None:
        empty, valid = _coh("2022-08-01", "2022-08-13", orbit=66), _coh("2022-08-13", "2022-08-25")
        rtcs = [_rtc("2022-08-01", orbit=66), _rtc("2022-08-13")]
        _write_raster(scene.manager.get_product_path(empty), [_full(np.nan)])
        _write_raster(scene.manager.get_product_path(valid), [_full(0.5)])
        for rtc in rtcs:
            _write_raster(scene.manager.get_product_path(rtc), [_full(1.0), _full(1.0)])

        composite = scene.weighter.compose_coherence_lrw([empty, valid], rtcs, scene.geobox)

        with rasterio.open(scene.manager.get_product_path(composite)) as src:
            assert src.tags()["CONTRIBUTING_OBSERVATIONS"] == "1"


class TestComposeCohLrw:
    def test_weights_coherence_by_the_matching_rtc_simulated_area(
        self,
        scene: _Scene,
    ) -> None:
        """Coherence carries no area band, so weights come from the RTC of the same pass."""
        coh_a, coh_b = _coh("2022-08-01", "2022-08-13"), _coh("2022-08-13", "2022-08-25")
        rtc_a, rtc_b = _rtc("2022-08-01"), _rtc("2022-08-13")
        _write_raster(scene.manager.get_product_path(coh_a), [_full(0.2)])
        _write_raster(scene.manager.get_product_path(coh_b), [_full(0.8)])
        _write_raster(scene.manager.get_product_path(rtc_a), [_full(1.0), _full(100.0)])
        _write_raster(scene.manager.get_product_path(rtc_b), [_full(1.0), _full(1.0)])

        product = scene.weighter.compose_coherence_lrw(
            [coh_a, coh_b],
            [rtc_a, rtc_b],
            scene.geobox,
        )

        with rasterio.open(scene.manager.get_product_path(product)) as src:
            values = src.read(1)
        assert values == pytest.approx(
            np.full((_TILE, _TILE), (0.2 * 0.01 + 0.8) / 1.01),
            rel=1e-5,
        )

    def test_rejects_an_empty_product_collection(self, scene: _Scene) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            scene.weighter.compose_coherence_lrw([], [], scene.geobox)

    def test_reports_coherence_without_a_matching_rtc(self, scene: _Scene) -> None:
        """Every coherence group needs a simulated-area group from the same date/orbit."""
        coh = _coh("2022-08-01", "2022-08-13")
        _write_raster(scene.manager.get_product_path(coh), [_full(0.5)])

        with pytest.raises(ValueError, match="No matching RTC scattering area"):
            scene.weighter.compose_coherence_lrw([coh], [], scene.geobox)

    def test_writes_the_requested_qa_layers(self, scene: _Scene) -> None:
        coh = _coh("2022-08-01", "2022-08-13")
        rtc = _rtc("2022-08-01")
        _write_raster(scene.manager.get_product_path(coh), [_full(0.5)])
        _write_raster(scene.manager.get_product_path(rtc), [_full(1.0), _full(4.0)])

        composite, (num_obs, cqm) = scene.weighter.compose_coherence_lrw(
            [coh],
            [rtc],
            scene.geobox,
            write_qa=True,
        )

        for product in (composite, num_obs, cqm):
            assert scene.manager.get_product_path(product).exists()


class TestComposeByPass:
    def test_rtc_composites_are_split_by_pass_direction(self, scene: _Scene) -> None:
        """Ascending and descending passes must never be averaged together."""
        ascending = _rtc("2022-08-13", orbit=15, burst_id=_ASCENDING_BURST)
        descending = _rtc("2022-08-25", orbit=88, burst_id=_DESCENDING_BURST)
        _write_raster(scene.manager.get_product_path(ascending), [_full(1.0), _full(1.0)])
        _write_raster(scene.manager.get_product_path(descending), [_full(5.0), _full(1.0)])

        result = scene.weighter.compose_rtc_by_pass([ascending, descending], scene.geobox)

        assert set(result) == {"Ascending", "Descending"}
        with rasterio.open(
            scene.manager.get_product_path(result[ascending.pass_direction]),
        ) as src:
            assert src.read(1) == pytest.approx(_full(1.0))
        with rasterio.open(
            scene.manager.get_product_path(result[descending.pass_direction]),
        ) as src:
            assert src.read(1) == pytest.approx(_full(5.0))

    def test_coherence_composites_are_split_by_pass_direction(self, scene: _Scene) -> None:
        first = _coh("2022-08-01", "2022-08-13", orbit=15, burst_id=_ASCENDING_BURST)
        second = _coh("2022-08-01", "2022-08-13", orbit=88, burst_id=_DESCENDING_BURST)
        _write_raster(scene.manager.get_product_path(first), [_full(0.3)])
        _write_raster(scene.manager.get_product_path(second), [_full(0.9)])

        result = scene.weighter.compose_coherence_by_pass([first, second], scene.geobox)

        assert set(result) == {"Ascending", "Descending"}

    def test_empty_input_yields_no_composites(self, scene: _Scene) -> None:
        assert scene.weighter.compose_coherence_by_pass([], scene.geobox) == {}
        assert scene.weighter.compose_rtc_by_pass([], scene.geobox) == {}

    def test_writes_per_pass_qa(self, scene: _Scene) -> None:
        product = _coh("2022-08-01", "2022-08-13")
        _write_raster(scene.manager.get_product_path(product), [_full(0.4)])

        result = scene.weighter.compose_coherence_by_pass([product], scene.geobox, write_qa=True)

        composite, num_obs = result[product.pass_direction]
        assert scene.manager.get_product_path(composite).exists()
        with rasterio.open(scene.manager.get_product_path(num_obs)) as src:
            assert (src.read(1) == 1).all()

    def test_coherence_pass_composite_carries_window_and_span(self, scene: _Scene) -> None:
        product = _coh("2022-08-01", "2022-08-13")
        _write_raster(scene.manager.get_product_path(product), [_full(0.4)])
        window = TimeRange("2022-07-15", "2022-10-12")

        result = scene.weighter.compose_coherence_by_pass(
            [product], scene.geobox, write_qa=True, time_range=window
        )

        for written in result[product.pass_direction]:
            with rasterio.open(scene.manager.get_product_path(written)) as src:
                tags = src.tags()
            assert (tags["COMPOSITE_START_DATE"], tags["COMPOSITE_END_DATE"]) == (
                "2022-07-15",
                "2022-10-12",
            )
            assert (tags["FIRST_ACQUISITION_DATE"], tags["LAST_ACQUISITION_DATE"]) == (
                "2022-08-01",
                "2022-08-01",
            )

    def test_skips_existing_per_pass_composites(self, scene: _Scene) -> None:
        product = _rtc("2022-08-13")
        _write_raster(scene.manager.get_product_path(product), [_full(7.0), _full(1.0)])
        first = scene.weighter.compose_rtc_by_pass([product], scene.geobox)
        path = scene.manager.get_product_path(first[product.pass_direction])
        path.write_bytes(b"kept")

        scene.weighter.compose_rtc_by_pass([product], scene.geobox)

        assert path.read_bytes() == b"kept"


def _gtc(date: str, orbit: int = 15, burst_id: int = _ASCENDING_BURST) -> GTCProduct:
    return GTCProduct(
        burst_id=burst_id,
        sub_swath="IW1",
        crs=_UTM32,
        date=Timestamp(date),
        polarization="VV",
        _rel_orbit_number=orbit,
    )


_SOURCE_TAGS = {
    "DEM": "glo30",
    "PROCESSING_SOFTWARE": "SNAP",
    "PROCESSING_SOFTWARE_VERSION": "14.0",
}


class TestComposeGtcByPass:
    def test_aggregates_per_pass_and_writes_qa(self, scene: _Scene) -> None:
        """Ascending dates 3.0 and 5.0 average to 4.0; descending keeps its NaN pixel."""
        a1, a2 = _gtc("2019-01-05"), _gtc("2019-01-17")
        _write_raster(scene.manager.get_product_path(a1), [_full(3.0)], _SOURCE_TAGS)
        _write_raster(scene.manager.get_product_path(a2), [_full(5.0)], _SOURCE_TAGS)
        d1 = _gtc("2019-01-08", orbit=66, burst_id=_DESCENDING_BURST)
        partial = _full(9.0)
        partial[0, 0] = np.nan
        _write_raster(scene.manager.get_product_path(d1), [partial], _SOURCE_TAGS)
        window = TimeRange("2019-01-01", "2019-01-31")

        result = scene.weighter.compose_gtc_by_pass(
            [a1, a2, d1], scene.geobox, aggregation="mean", write_qa=True, time_range=window
        )

        asc, asc_qa = result["Ascending"]
        desc, desc_qa = result["Descending"]
        assert isinstance(asc, GTCPassComposite)
        assert (
            scene.manager.get_product_path(asc).name
            == "S1_GTC_COMP_A_32TMS_VV_20190101_20190131.tif"
        )
        with rasterio.open(scene.manager.get_product_path(asc)) as src:
            assert src.read(1) == pytest.approx(np.full((_TILE, _TILE), 4.0))
            assert src.tags()["BACKSCATTER_MEASUREMENT"] == "Beta-0"
            assert src.tags()["CONTRIBUTING_OBSERVATIONS"] == "2"
            assert src.tags()["FIRST_ACQUISITION_DATE"].startswith("2019-01-05")
        with rasterio.open(scene.manager.get_product_path(asc_qa)) as src:
            assert (src.read(1) == 2).all()
        with rasterio.open(scene.manager.get_product_path(desc)) as src:
            values = src.read(1)
            assert np.isnan(values[0, 0])
            assert values[1, 1] == 9.0
        with rasterio.open(scene.manager.get_product_path(desc_qa)) as src:
            counts = src.read(1)
            assert counts[0, 0] == 0
            assert counts[1, 1] == 1

    def test_single_direction_input_yields_one_entry(self, scene: _Scene) -> None:
        a1 = _gtc("2019-01-05")
        _write_raster(scene.manager.get_product_path(a1), [_full(2.0)], _SOURCE_TAGS)

        result = scene.weighter.compose_gtc_by_pass([a1], scene.geobox, aggregation="mean")

        assert list(result) == ["Ascending"]

    def test_rtc_products_still_write_rtc_names(self, scene: _Scene) -> None:
        r = _rtc("2019-01-05")
        _write_raster(scene.manager.get_product_path(r), [_full(2.0), _full(1.0)], _SOURCE_TAGS)

        result = scene.weighter.compose_rtc_by_pass([r], scene.geobox, aggregation="mean")

        name = scene.manager.get_product_path(result["Ascending"]).name
        assert name.startswith("S1_RTC_COMP_A_")


class TestCompositeTags:
    """The tags written into the composites, exercised through the real write path.

    The builders are unit-tested in ``test_product_tags``; what these cover is
    that ``_save`` actually carries them into the file, and that the arguments
    reaching them are the ones the call was made with.
    """

    def test_rtc_composite_declares_how_to_read_its_pixels(self, scene: _Scene) -> None:
        products = [_rtc("2022-08-13"), _rtc("2022-08-25")]
        for product in products:
            _write_raster(scene.manager.get_product_path(product), [_full(2.0), _full(4.0)])

        composite = scene.weighter.compose_rtc_lrw(
            products,
            scene.geobox,
            aggregation="mean",
            clip_max_resolution=2.0,
        )

        with rasterio.open(scene.manager.get_product_path(composite)) as src:
            tags = src.tags()
            assert tags["PRODUCT_TYPE"] == "RTC_LRW"
            assert tags["BACKSCATTER_CONVENTION"] == "Power"
            assert tags["CONTRIBUTING_OBSERVATIONS"] == "2"
            # The parameters must be the ones this call actually used.
            assert tags["COMPOSITING_PARAMETERS"] == "aggregation=mean;clip_max_resolution=2"
            assert tags["METADATA_VOCABULARY"] == product_tags.METADATA_VOCABULARY
            assert src.units == ("linear power",)
            assert src.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"

    def test_decibel_composite_says_so_in_the_file(self, scene: _Scene) -> None:
        products = [_rtc("2022-08-13")]
        _write_raster(scene.manager.get_product_path(products[0]), [_full(2.0), _full(4.0)])

        composite = scene.weighter.compose_rtc_lrw(products, scene.geobox, decibel=True)

        with rasterio.open(scene.manager.get_product_path(composite)) as src:
            assert "BACKSCATTER_CONVENTION" not in src.tags()
            assert src.tags()["BACKSCATTER_SCALING"] == "decibel"
            assert src.units == ("dB",)

    def test_qa_layers_carry_their_own_vocabulary(self, scene: _Scene) -> None:
        products = [_rtc("2022-08-13"), _rtc("2022-08-25")]
        for product in products:
            _write_raster(scene.manager.get_product_path(product), [_full(2.0), _full(4.0)])

        _, (num_obs, cqm) = scene.weighter.compose_rtc_lrw(
            products,
            scene.geobox,
            write_qa=True,
        )

        with rasterio.open(scene.manager.get_product_path(num_obs)) as src:
            assert src.tags()["QA_LAYER"] == "Contributing Observations Image"
            assert src.descriptions == ("num_obs",)
        with rasterio.open(scene.manager.get_product_path(cqm)) as src:
            assert src.tags()["QA_LAYER"] == "Composite Quality Map Image"
            assert src.tags()["CQM_CONVERSION_EQ"] == "dB = 10*log10(DN)"

    def test_cqm_in_decibels_is_tagged_as_converted(self, scene: _Scene) -> None:
        products = [_rtc("2022-08-13")]
        _write_raster(scene.manager.get_product_path(products[0]), [_full(2.0), _full(4.0)])

        _, (num_obs, cqm) = scene.weighter.compose_rtc_lrw(
            products,
            scene.geobox,
            write_qa=True,
            cqm_decibel=True,
        )

        with rasterio.open(scene.manager.get_product_path(cqm)) as src:
            assert src.tags()["CQM_SCALING"] == "decibel"
            assert "CQM_CONVERSION_EQ" not in src.tags()
        with rasterio.open(scene.manager.get_product_path(num_obs)) as src:
            assert "CQM_SCALING" not in src.tags()

    def test_coherence_composite_inherits_the_estimation_window(self, scene: _Scene) -> None:
        """The window exists only in ``CoherenceConfig``; the input files are its only route."""
        coh = _coh("2022-08-01", "2022-08-13")
        rtc = _rtc("2022-08-01")
        _write_raster(
            scene.manager.get_product_path(coh),
            [_full(0.5)],
            tags={
                "COHERENCE_WINDOW_RANGE": "5",
                "COHERENCE_WINDOW_AZIMUTH": "1",
                "COHERENCE_WEIGHTING_SHAPE": "Rectangular",
                "PROCESSING_SOFTWARE": "ESA SNAP",
            },
        )
        _write_raster(scene.manager.get_product_path(rtc), [_full(1.0), _full(4.0)])

        composite = scene.weighter.compose_coherence_lrw([coh], [rtc], scene.geobox)

        with rasterio.open(scene.manager.get_product_path(composite)) as src:
            tags = src.tags()
            assert tags["COHERENCE_WINDOW_RANGE"] == "5"
            assert tags["COHERENCE_WINDOW_AZIMUTH"] == "1"
            assert tags["PROCESSING_SOFTWARE"] == "ESA SNAP"
            assert tags["MEASUREMENT_TYPE"] == "Coherence"
            assert tags["TEMPORAL_BASELINE_DAYS"] == "12"
            # Coherence is not a power quantity, so it gets no conversion.
            assert not [k for k in tags if k.startswith("BACKSCATTER_")]
            assert float(tags["COHERENCE_AVERAGE"]) == pytest.approx(0.5)
            assert src.descriptions == ("VV",)

    def test_provenance_is_only_inherited_where_the_inputs_agree(
        self,
        scene: _Scene,
    ) -> None:
        """A composite spanning two SNAP versions must not claim either of them."""
        agree, differ = _rtc("2022-08-13"), _rtc("2022-08-25")
        _write_raster(
            scene.manager.get_product_path(agree),
            [_full(2.0), _full(4.0)],
            tags={"PROCESSING_SOFTWARE": "ESA SNAP", "PROCESSING_SOFTWARE_VERSION": "13.0.0"},
        )
        _write_raster(
            scene.manager.get_product_path(differ),
            [_full(2.0), _full(4.0)],
            tags={"PROCESSING_SOFTWARE": "ESA SNAP", "PROCESSING_SOFTWARE_VERSION": "14.0.0"},
        )

        composite = scene.weighter.compose_rtc_lrw([agree, differ], scene.geobox)

        with rasterio.open(scene.manager.get_product_path(composite)) as src:
            tags = src.tags()
            # Unanimous, so it carries forward.
            assert tags["PROCESSING_SOFTWARE"] == "ESA SNAP"
            # Contested, so the composite states neither rather than picking one.
            assert tags["PROCESSING_SOFTWARE_VERSION"] == product_tags.UNKNOWN

    def test_unanimous_provenance_is_inherited(self, scene: _Scene) -> None:
        products = [_rtc("2022-08-13"), _rtc("2022-08-25")]
        for product in products:
            _write_raster(
                scene.manager.get_product_path(product),
                [_full(2.0), _full(4.0)],
                tags={"PROCESSING_SOFTWARE_VERSION": "14.0.0"},
            )

        composite = scene.weighter.compose_rtc_lrw(products, scene.geobox)

        with rasterio.open(scene.manager.get_product_path(composite)) as src:
            assert src.tags()["PROCESSING_SOFTWARE_VERSION"] == "14.0.0"

    def test_composite_inherits_the_dem(self, scene: _Scene) -> None:
        products = [_rtc("2022-08-13"), _rtc("2022-08-25")]
        for product in products:
            _write_raster(
                scene.manager.get_product_path(product),
                [_full(2.0), _full(4.0)],
                tags={"DEM": "Copernicus 30m Global DEM"},
            )

        composite = scene.weighter.compose_rtc_lrw(products, scene.geobox)

        with rasterio.open(scene.manager.get_product_path(composite)) as src:
            assert src.tags()["DEM"] == "Copernicus 30m Global DEM"

    def test_pass_composite_records_its_direction_and_claims_no_weighting(
        self,
        scene: _Scene,
    ) -> None:
        ascending = _rtc("2022-08-13", orbit=15, burst_id=_ASCENDING_BURST)
        _write_raster(scene.manager.get_product_path(ascending), [_full(1.0), _full(1.0)])

        result = scene.weighter.compose_rtc_by_pass([ascending], scene.geobox, write_qa=True)
        composite, num_obs = result["Ascending"]

        with rasterio.open(scene.manager.get_product_path(composite)) as src:
            tags = src.tags()
            assert tags["PASS_DIRECTION"] == "Ascending"
            assert tags["COMPOSITING_METHOD"] == "Unweighted temporal aggregation"
            assert "COMPOSITING_METHOD_REFERENCE" not in tags
        with rasterio.open(scene.manager.get_product_path(num_obs)) as src:
            assert src.tags()["QA_LAYER"] == "Contributing Observations Image"


class TestInputConsistency:
    """Guards on inputs a composite must not silently average together."""

    def test_a_product_in_another_crs_is_refused(self, scene: _Scene) -> None:
        """Another CRS reads pixels from elsewhere on Earth.

        The window is computed in the geobox's coordinates against the source's
        own transform, so nothing downstream notices.
        """
        ours, foreign = _rtc("2022-08-13"), _rtc("2022-08-25")
        _write_raster(scene.manager.get_product_path(ours), [_full(1.0), _full(1.0)])
        path = scene.manager.get_product_path(foreign)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=_TILE,
            height=_TILE,
            count=2,
            dtype="float32",
            crs=CRS.from_epsg(32633),
            transform=Affine(10.0, 0.0, 400000.0, 0.0, -10.0, 5200000.0),
            nodata=np.nan,
        ) as dst:
            dst.write(_full(1.0), 1)
            dst.write(_full(1.0), 2)

        with pytest.raises(ValueError, match="target grid is"):
            scene.weighter.compose_rtc_lrw([ours, foreign], scene.geobox)

    def test_a_product_at_another_pixel_size_is_refused(self, scene: _Scene) -> None:
        ours, coarse = _rtc("2022-08-13"), _rtc("2022-08-25")
        _write_raster(scene.manager.get_product_path(ours), [_full(1.0), _full(1.0)])
        path = scene.manager.get_product_path(coarse)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=_TILE,
            height=_TILE,
            count=2,
            dtype="float32",
            crs=_UTM32,
            transform=Affine(20.0, 0.0, 400000.0, 0.0, -20.0, 5200000.0),
            nodata=np.nan,
        ) as dst:
            dst.write(_full(1.0), 1)
            dst.write(_full(1.0), 2)

        with pytest.raises(ValueError, match="must share the output grid"):
            scene.weighter.compose_rtc_lrw([ours, coarse], scene.geobox)

    def test_a_product_offset_by_a_fraction_of_a_pixel_is_refused(self, scene: _Scene) -> None:
        # Previously the read window was rounded to whole pixels, shifting the
        # product silently by up to half a pixel.
        ours, shifted = _rtc("2022-08-13"), _rtc("2022-08-25")
        _write_raster(scene.manager.get_product_path(ours), [_full(1.0), _full(1.0)])
        path = scene.manager.get_product_path(shifted)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=_TILE,
            height=_TILE,
            count=2,
            dtype="float32",
            crs=_UTM32,
            transform=Affine(10.0, 0.0, 400004.0, 0.0, -10.0, 5200000.0),
            nodata=np.nan,
        ) as dst:
            dst.write(_full(1.0), 1)
            dst.write(_full(1.0), 2)

        with pytest.raises(ValueError, match="must share the output grid"):
            scene.weighter.compose_rtc_lrw([ours, shifted], scene.geobox)

    def test_a_product_at_a_whole_pixel_offset_is_accepted(self, scene: _Scene) -> None:
        """A neighbouring burst starts elsewhere on the same grid; that is the normal case."""
        ours, neighbour = _rtc("2022-08-13"), _rtc("2022-08-25")
        _write_raster(scene.manager.get_product_path(ours), [_full(1.0), _full(1.0)])
        path = scene.manager.get_product_path(neighbour)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=_TILE,
            height=_TILE,
            count=2,
            dtype="float32",
            crs=_UTM32,
            transform=Affine(10.0, 0.0, 400030.0, 0.0, -10.0, 5200020.0),
            nodata=np.nan,
        ) as dst:
            dst.write(_full(1.0), 1)
            dst.write(_full(1.0), 2)

        scene.weighter.compose_rtc_lrw([ours, neighbour], scene.geobox)

    def test_mixed_temporal_baselines_are_refused(self, scene: _Scene) -> None:
        """Coherence falls off with the baseline, and the composite carries one."""
        twelve = _coh("2022-08-01", "2022-08-13")
        twenty_four = _coh("2022-08-13", "2022-09-06")
        assert (twelve.baseline_days, twenty_four.baseline_days) == (12, 24)
        for product in (twelve, twenty_four):
            _write_raster(scene.manager.get_product_path(product), [_full(0.5)])

        with pytest.raises(ValueError, match="share a temporal baseline"):
            scene.weighter.compose_coherence_lrw([twelve, twenty_four], [], scene.geobox)

    def test_mixed_dems_are_refused(self, scene: _Scene) -> None:
        coh = _coh("2022-08-01", "2022-08-13")
        rtc = _rtc("2022-08-01")
        _write_raster(
            scene.manager.get_product_path(coh),
            [_full(0.5)],
            tags={"DEM": "Copernicus 30m Global DEM"},
        )
        _write_raster(
            scene.manager.get_product_path(rtc),
            [_full(1.0), _full(4.0)],
            tags={"DEM": "GETASSE30"},
        )

        with pytest.raises(ValueError, match="same DEM"):
            scene.weighter.compose_coherence_lrw([coh], [rtc], scene.geobox)

    def test_mixed_snap_versions_are_allowed_but_warned_about(
        self,
        scene: _Scene,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An archive built over several campaigns will legitimately mix these."""
        first, second = _rtc("2022-08-13"), _rtc("2022-08-25")
        for product, version in ((first, "13.0.0"), (second, "14.0.0")):
            _write_raster(
                scene.manager.get_product_path(product),
                [_full(2.0), _full(4.0)],
                tags={"PROCESSING_SOFTWARE_VERSION": version},
            )

        with caplog.at_level(logging.WARNING, logger="s1bursts.resolution_weighter"):
            composite = scene.weighter.compose_rtc_lrw([first, second], scene.geobox)

        assert any(
            "disagree on PROCESSING_SOFTWARE_VERSION" in r.getMessage() for r in caplog.records
        )
        with rasterio.open(scene.manager.get_product_path(composite)) as src:
            assert src.tags()["PROCESSING_SOFTWARE_VERSION"] == product_tags.UNKNOWN

    def test_inputs_without_provenance_tags_warn_about_nothing(
        self,
        scene: _Scene,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Absent from every input is agreement, not disagreement."""
        products = [_rtc("2022-08-13"), _rtc("2022-08-25")]
        for product in products:
            _write_raster(scene.manager.get_product_path(product), [_full(2.0), _full(4.0)])

        with caplog.at_level(logging.WARNING, logger="s1bursts.resolution_weighter"):
            scene.weighter.compose_rtc_lrw(products, scene.geobox)

        assert not [r for r in caplog.records if "disagree" in r.message]


class TestPerProductProfiles:
    """Composites and their QA auxiliaries can take different codec settings."""

    @staticmethod
    def _weighter(tmp_path: Path, profile: dict[str, Any] | None) -> ResolutionWeighter:
        return ResolutionWeighter(
            FileManager(base_dir=tmp_path),
            rasterio_profile=profile,
        )

    @staticmethod
    def _product(qa_metric: QaMetric | None = None) -> RTCLRWComposite | RTCLRWQA:
        common: dict[str, Any] = {
            "mgrs_code": "32TMS03",
            "start_date": Timestamp("2023-07-18"),
            "end_date": Timestamp("2023-10-15"),
            "polarization": "VV",
        }
        if qa_metric is None:
            return RTCLRWComposite(**common)
        return RTCLRWQA(**common, qa_metric=qa_metric)

    def test_a_flat_profile_applies_to_every_product(self, tmp_path: Path) -> None:
        w = self._weighter(tmp_path, {"compress": "zstd"})

        for product in (self._product(), self._product(QaMetric.NUM)):
            assert w._profile_for(product)["compress"] == "zstd"

    def test_no_profile_keeps_the_library_default(self, tmp_path: Path) -> None:
        w = self._weighter(tmp_path, None)

        assert w._profile_for(self._product())["compress"] == "deflate"

    def test_per_product_entry_overrides_the_default(self, tmp_path: Path) -> None:
        w = self._weighter(
            tmp_path,
            {
                "default": {"compress": "lerc_zstd", "max_z_error": 1e-5},
                "RTC_LRW_QA-NUM": {"max_z_error": 0},
            },
        )

        assert w._profile_for(self._product())["max_z_error"] == 1e-5
        assert w._profile_for(self._product(QaMetric.NUM))["max_z_error"] == 0

    def test_a_per_product_entry_inherits_the_default_codec(self, tmp_path: Path) -> None:
        """A partial override must not restore the library's default codec."""
        w = self._weighter(
            tmp_path,
            {
                "default": {"compress": "lerc_zstd", "max_z_error": 1e-5},
                "RTC_LRW_QA-NUM": {"max_z_error": 0},
            },
        )

        assert w._profile_for(self._product(QaMetric.NUM))["compress"] == "lerc_zstd"

    def test_a_product_without_an_entry_falls_back_to_the_default(self, tmp_path: Path) -> None:
        w = self._weighter(tmp_path, {"default": {"compress": "lerc_zstd"}})

        assert w._profile_for(self._product(QaMetric.CQM))["compress"] == "lerc_zstd"

    def test_a_mapping_without_a_default_leaves_others_on_the_library_default(
        self,
        tmp_path: Path,
    ) -> None:
        w = self._weighter(tmp_path, {"RTC_LRW": {"compress": "lerc_zstd"}})

        assert w._profile_for(self._product())["compress"] == "lerc_zstd"
        assert w._profile_for(self._product(QaMetric.NUM))["compress"] == "deflate"

    def test_per_product_codec_reaches_the_written_cog(self, tmp_path: Path) -> None:
        """Exercise the freezebase boundary, not only the profile lookup."""
        manager = FileManager(base_dir=tmp_path)
        weighter = ResolutionWeighter(
            manager,
            rasterio_profile={
                "default": {"compress": "lerc_zstd", "max_z_error": 1e-5},
                "RTC_LRW_QA-NUM": {"max_z_error": 0},
            },
        )
        product = self._product()
        weighter._save(
            np.full((_TILE, _TILE), 0.25, dtype=np.float32),
            product,
            _geobox(),
            product_tags.RasterMetadata(tags={}, band_names=["VV"], units=["natural"]),
        )

        with rasterio.open(manager.get_product_path(product)) as src:
            structure = src.tags(ns="IMAGE_STRUCTURE")
        assert structure["COMPRESSION"] == "LERC_ZSTD"
        assert float(structure["MAX_Z_ERROR"]) == pytest.approx(1e-5)


class TestComposeTimeRange:
    """``time_range`` names the compositing window instead of the observed span.

    The window is the sampling frame a caller chose, identical for every tile of
    a year; the first and last acquisition are what actually contributed and
    vary per tile. Without ``time_range`` the two coincide, as before.
    """

    @staticmethod
    def _compose(tmp_path: Path, **kwargs: Any) -> tuple[FileManager, RTCLRWComposite]:
        fm, weighter, geobox = TestComposeReturns._setup(tmp_path)
        p1, p2 = _rtc_stub(10, 12), _rtc_stub(20, 24)  # observed 2020-01-12 .. 2020-01-24
        power = {
            id(p1): np.full((2, 2), 1.0, np.float32),
            id(p2): np.full((2, 2), 2.0, np.float32),
        }
        area = {id(p1): np.full((2, 2), 2.0, np.float32), id(p2): np.full((2, 2), 4.0, np.float32)}
        TestComposeReturns._patch_load(weighter, power, area)
        # pyrefly: ignore [no-matching-overload]
        result = weighter.compose_rtc_lrw([p1, p2], geobox, aggregation="mean", **kwargs)
        assert isinstance(result, RTCLRWComposite)
        return fm, result

    def test_window_names_the_product_and_the_file(self, tmp_path: Path) -> None:
        window = TimeRange("2020-01-01", "2020-01-31")
        fm, product = self._compose(tmp_path, time_range=window)
        assert (product.start_date, product.end_date) == (
            Timestamp("2020-01-01"),
            Timestamp("2020-01-31"),
        )
        assert fm.get_product_path(product).name.endswith("_20200101_20200131.tif")

    def test_window_is_the_composite_date_and_the_span_is_kept_as_acquisitions(
        self, tmp_path: Path
    ) -> None:
        window = TimeRange("2020-01-01", "2020-01-31")
        fm, product = self._compose(tmp_path, time_range=window)
        with rasterio.open(fm.get_product_path(product)) as src:
            tags = src.tags()
        assert (tags["COMPOSITE_START_DATE"], tags["COMPOSITE_END_DATE"]) == (
            "2020-01-01",
            "2020-01-31",
        )
        assert (tags["FIRST_ACQUISITION_DATE"], tags["LAST_ACQUISITION_DATE"]) == (
            "2020-01-12",
            "2020-01-24",
        )

    def test_without_a_window_the_span_is_the_composite_date(self, tmp_path: Path) -> None:
        fm, product = self._compose(tmp_path)
        assert (product.start_date, product.end_date) == (
            Timestamp("2020-01-12"),
            Timestamp("2020-01-24"),
        )
        with rasterio.open(fm.get_product_path(product)) as src:
            tags = src.tags()
        assert tags["FIRST_ACQUISITION_DATE"] == tags["COMPOSITE_START_DATE"] == "2020-01-12"
        assert tags["LAST_ACQUISITION_DATE"] == tags["COMPOSITE_END_DATE"] == "2020-01-24"

    def test_a_pair_of_instants_is_refused(self, tmp_path: Path) -> None:
        """A search range stops the midnight after the window, a day past the composite's end."""
        search_range = TimeRange("2020-01-01", "2020-01-31").half_open()
        with pytest.raises(TypeError, match="must be a TimeRange"):
            self._compose(tmp_path, time_range=search_range)

    def test_an_input_outside_the_window_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="outside the time range"):
            self._compose(tmp_path, time_range=TimeRange("2020-01-01", "2020-01-20"))

    def test_timezone_aware_acquisitions_are_compared_by_calendar_day(
        self, tmp_path: Path
    ) -> None:
        """CDSE bursts carry a UTC time of day; an acquisition on the window's last day counts."""
        fm, weighter, geobox = TestComposeReturns._setup(tmp_path)
        p1, p2 = _rtc_stub(10, 12), _rtc_stub(20, 24)
        p1.date = Timestamp("2020-01-12T05:31:07", tz="UTC")
        p2.date = Timestamp("2020-01-24T17:12:44", tz="UTC")
        power = {
            id(p1): np.full((2, 2), 1.0, np.float32),
            id(p2): np.full((2, 2), 2.0, np.float32),
        }
        area = {id(p1): np.full((2, 2), 2.0, np.float32), id(p2): np.full((2, 2), 4.0, np.float32)}
        TestComposeReturns._patch_load(weighter, power, area)
        window = TimeRange("2020-01-12", "2020-01-24")
        # pyrefly: ignore [no-matching-overload]
        product = weighter.compose_rtc_lrw([p1, p2], geobox, aggregation="mean", time_range=window)
        assert isinstance(product, RTCLRWComposite)
        with rasterio.open(fm.get_product_path(product)) as src:
            tags = src.tags()
        assert (tags["FIRST_ACQUISITION_DATE"], tags["LAST_ACQUISITION_DATE"]) == (
            "2020-01-12",
            "2020-01-24",
        )

    def test_pass_composites_tag_each_direction_with_its_own_span(self, tmp_path: Path) -> None:
        fm, weighter, geobox = TestComposeReturns._setup(tmp_path)
        asc = [_rtc_stub(10, 3, "Ascending"), _rtc_stub(10, 15, "Ascending")]
        desc = [_rtc_stub(20, 9, "Descending"), _rtc_stub(20, 27, "Descending")]
        power = {id(p): np.full((2, 2), 1.0, np.float32) for p in [*asc, *desc]}
        TestComposeReturns._patch_load(weighter, power)
        window = TimeRange("2020-01-01", "2020-01-31")
        # pyrefly: ignore [no-matching-overload]
        result = weighter.compose_rtc_by_pass(
            [*asc, *desc],
            geobox,
            aggregation="mean",
            write_qa=True,
            time_range=window,
        )
        spans = {}
        for direction, (composite, qa_num) in result.items():
            for product in (composite, qa_num):
                with rasterio.open(fm.get_product_path(product)) as src:
                    tags = src.tags()
                spans[(direction, type(product).__name__)] = (
                    tags["COMPOSITE_START_DATE"],
                    tags["COMPOSITE_END_DATE"],
                    tags["FIRST_ACQUISITION_DATE"],
                    tags["LAST_ACQUISITION_DATE"],
                )
        for key, (start, end, first, last) in spans.items():
            assert (start, end) == ("2020-01-01", "2020-01-31"), key
            expected = (
                ("2020-01-03", "2020-01-15")
                if key[0] == "Ascending"
                else ("2020-01-09", "2020-01-27")
            )
            assert (first, last) == expected, key
