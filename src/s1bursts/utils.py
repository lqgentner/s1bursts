"""Numerical helpers for SAR image processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _box_sum_2d(
    a: NDArray[Any],
    win_row: int,
    win_col: int,
    dtype: type[np.number[Any]],
) -> NDArray[Any]:
    """Compute a 2-D sliding-window box sum with zero padding outside the array.

    Uses a padded integral image so that the window sums are obtained from
    pure slice arithmetic. The window placement matches
    ``scipy.ndimage.uniform_filter(mode="constant", cval=0)``: the window for
    output pixel ``i`` spans input rows ``[i - win_row // 2,
    i - win_row // 2 + win_row - 1]`` (and likewise for columns).
    """
    lr, lc = win_row // 2, win_col // 2
    nr, nc = a.shape
    ii = np.zeros((nr + win_row, nc + win_col), dtype=dtype)
    ii[1 + lr : 1 + lr + nr, 1 + lc : 1 + lc + nc] = a
    np.cumsum(ii, axis=0, out=ii)
    np.cumsum(ii, axis=1, out=ii)
    return (
        ii[win_row:, win_col:]
        - ii[:-win_row, win_col:]
        - ii[win_row:, :-win_col]
        + ii[:-win_row, :-win_col]
    )


def sliding_window_mean[T: np.inexact[Any]](
    img: NDArray[T],
    win_row: int,
    win_col: int,
    min_coverage: float | None = None,
) -> NDArray[T]:
    """Compute a sliding window mean with explicit handling of non-finite values.

    Sums are accumulated over an integral image with zero boundary padding
    and an explicit finite-pixel count, so that non-finite pixels are
    excluded from the calculation. Accumulation happens in double precision.

    Parameters
    ----------
    img : NDArray[T]
        2-D input array, shape ``(nrows, ncols)``. Real or complex.
    win_row : int
        Window size in the row direction.
    win_col : int
        Window size in the column direction.
    min_coverage : float | None, default None
        Minimum fraction of finite pixels required within a window for the
        result to be finite. Windows whose finite-pixel fraction falls below
        this threshold are set to ``nan``.  Must be in ``(0, 1]``.
        If ``None``, coverage masking is disabled and every window that
        contains at least one finite pixel produces a finite output.

    Returns
    -------
    NDArray[T]
        Array of the same shape and dtype as *img* containing the sliding window means.

    Raises
    ------
    ValueError
        If *min_coverage* is not within ``(0, 1]``.
    """
    if win_row <= 1 and win_col <= 1:
        return img

    if min_coverage is not None and not 0 < min_coverage <= 1:
        msg = f"`min_coverage` must be within (0, 1] or None, got '{min_coverage}'."
        raise ValueError(msg)

    valid = np.isfinite(img)
    img_ = np.where(valid, img, 0.0)

    acc_dtype = np.complex128 if np.iscomplexobj(img) else np.float64
    sum_out = _box_sum_2d(img_, win_row, win_col, acc_dtype)
    count = _box_sum_2d(valid, win_row, win_col, np.int64)

    with np.errstate(invalid="ignore", divide="ignore"):
        out = sum_out / count

    if min_coverage is not None:
        out[count < min_coverage * (win_row * win_col)] = np.nan

    return cast("NDArray[T]", out.astype(img.dtype))
