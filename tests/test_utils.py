"""Tests for s1bursts.utils."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pytest

from s1bursts.utils import sliding_window_mean


def _reference_mean(
    img: np.ndarray,
    win_row: int,
    win_col: int,
    min_coverage: float | None = None,
) -> np.ndarray:
    """Brute-force sliding window mean with uniform_filter window placement."""
    left_r, left_c = win_row // 2, win_col // 2
    pad = ((left_r, win_row - 1 - left_r), (left_c, win_col - 1 - left_c))
    valid = np.isfinite(img)
    acc = np.complex128 if np.iscomplexobj(img) else np.float64
    img_pad = np.pad(np.where(valid, img, 0.0).astype(acc), pad)
    valid_pad = np.pad(valid.astype(np.float64), pad)
    win_sum = sliding_window_view(img_pad, (win_row, win_col)).sum(axis=(-2, -1))
    win_cnt = sliding_window_view(valid_pad, (win_row, win_col)).sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = win_sum / win_cnt
    if min_coverage is not None:
        out[win_cnt / (win_row * win_col) < min_coverage] = np.nan
    return out.astype(img.dtype)


class TestSlidingWindowMean:
    def test_uniform_input_stays_uniform(self) -> None:
        img = np.full((8, 8), 3.0)

        out = sliding_window_mean(img, win_row=3, win_col=3)

        np.testing.assert_allclose(out, 3.0)

    def test_window_of_one_returns_input(self) -> None:
        img = np.random.default_rng(0).random((4, 4))

        out = sliding_window_mean(img, win_row=1, win_col=1)

        assert out is img

    def test_nan_pixels_are_excluded(self) -> None:
        img = np.full((5, 5), 2.0)
        img[2, 2] = np.nan

        out = sliding_window_mean(img, win_row=3, win_col=3)

        # NaN neighbours are excluded from the mean, not propagated
        np.testing.assert_allclose(out, 2.0)

    def test_min_coverage_masks_sparse_windows(self) -> None:
        img = np.full((6, 6), 1.0)
        img[:, :3] = np.nan

        out = sliding_window_mean(img, win_row=3, win_col=3, min_coverage=0.75)

        assert np.isnan(out[3, 1])  # inside the NaN block
        assert out[3, 4] == 1.0  # interior pixel with a fully valid window

    @pytest.mark.parametrize("min_coverage", [0.0, -0.1, 1.5])
    def test_invalid_min_coverage_raises(self, min_coverage: float) -> None:
        with pytest.raises(ValueError, match="min_coverage"):
            sliding_window_mean(np.ones((4, 4)), win_row=3, win_col=3, min_coverage=min_coverage)

    def test_complex_input(self) -> None:
        img = np.full((5, 5), 1.0 + 1.0j)

        out = sliding_window_mean(img, win_row=3, win_col=3)

        np.testing.assert_allclose(out, 1.0 + 1.0j)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
    @pytest.mark.parametrize(("win_row", "win_col"), [(3, 3), (4, 19), (2, 5), (1, 6)])
    def test_matches_brute_force_reference(
        self,
        dtype: type[np.inexact],
        win_row: int,
        win_col: int,
    ) -> None:
        rng = np.random.default_rng(42)
        img = rng.random((40, 60))
        if np.issubdtype(dtype, np.complexfloating):
            img = img + 1j * rng.random((40, 60))
        img = img.astype(dtype)
        img[rng.random((40, 60)) < 0.3] = np.nan

        out = sliding_window_mean(img, win_row=win_row, win_col=win_col, min_coverage=0.6)
        ref = _reference_mean(img, win_row, win_col, min_coverage=0.6)

        assert out.dtype == img.dtype
        np.testing.assert_array_equal(np.isnan(out), np.isnan(ref))
        np.testing.assert_allclose(
            np.where(np.isnan(out), 0, out),
            np.where(np.isnan(ref), 0, ref),
            rtol=1e-6,
        )
