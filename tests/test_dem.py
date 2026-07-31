"""Tests for the SNAP half-pixel DEM compensation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from s1bursts.dem import COMPENSATED_TAG, compensate_dem_for_snap

if TYPE_CHECKING:
    from collections.abc import Callable

RES = 1.0 / 3600.0
ORIGIN_X = 7.0 - RES / 2
ORIGIN_Y = 47.0 + RES / 2


@pytest.fixture
def make_dem(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory writing a small georeferenced DEM."""

    def _make(name: str = "dem.tif", dtype: str = "float32", count: int = 1) -> Path:
        path = tmp_path / name
        data = np.arange(20 * 30, dtype=dtype).reshape(1, 20, 30).repeat(count, axis=0)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=30,
            height=20,
            count=count,
            dtype=dtype,
            crs="EPSG:4326",
            transform=from_origin(ORIGIN_X, ORIGIN_Y, RES, RES),
            nodata=np.nan if dtype.startswith("float") else 0,
        ) as dst:
            dst.write(data)
        return path

    return _make


class TestCompensateForSnap:
    def test_moves_origin_onto_first_pixel_centre(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        dem = make_dem()
        out = compensate_dem_for_snap(dem, cache_dir=tmp_path / "cache")

        with rasterio.open(dem) as src, rasterio.open(out) as dst:
            assert dst.transform.c == pytest.approx(src.transform.c + RES / 2)
            assert dst.transform.f == pytest.approx(src.transform.f - RES / 2)
            # The origin now sits exactly on the first pixel's centre.
            assert dst.transform.c == pytest.approx(7.0)
            assert dst.transform.f == pytest.approx(47.0)

    def test_preserves_grid_and_elevations(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        dem = make_dem()
        out = compensate_dem_for_snap(dem, cache_dir=tmp_path / "cache")

        with rasterio.open(dem) as src, rasterio.open(out) as dst:
            assert dst.res == src.res
            assert dst.shape == src.shape
            assert dst.crs == src.crs
            assert np.array_equal(src.read(), dst.read(), equal_nan=True)

    def test_tags_the_copy(self, make_dem: Callable[..., Path], tmp_path: Path) -> None:
        out = compensate_dem_for_snap(make_dem(), cache_dir=tmp_path / "cache")

        with rasterio.open(out) as dst:
            assert dst.tags()[COMPENSATED_TAG] == "1"

    def test_handles_integer_dtype(self, make_dem: Callable[..., Path], tmp_path: Path) -> None:
        """The float predictor is invalid for integers, so the dtype must pick it."""
        dem = make_dem(dtype="int16")
        out = compensate_dem_for_snap(dem, cache_dir=tmp_path / "cache")

        with rasterio.open(dem) as src, rasterio.open(out) as dst:
            assert dst.dtypes == src.dtypes
            assert np.array_equal(src.read(), dst.read())

    def test_handles_multiple_bands(self, make_dem: Callable[..., Path], tmp_path: Path) -> None:
        dem = make_dem(count=3)
        out = compensate_dem_for_snap(dem, cache_dir=tmp_path / "cache")

        with rasterio.open(dem) as src, rasterio.open(out) as dst:
            assert dst.count == 3
            assert np.array_equal(src.read(), dst.read(), equal_nan=True)

    def test_reuses_cached_copy(self, make_dem: Callable[..., Path], tmp_path: Path) -> None:
        dem = make_dem()
        cache = tmp_path / "cache"
        first = compensate_dem_for_snap(dem, cache_dir=cache)
        stamp = first.stat().st_mtime_ns

        second = compensate_dem_for_snap(dem, cache_dir=cache)

        assert second == first
        assert second.stat().st_mtime_ns == stamp

    def test_rewrites_when_source_changes(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        dem = make_dem()
        cache = tmp_path / "cache"
        out = compensate_dem_for_snap(dem, cache_dir=cache)
        with rasterio.open(out) as dst:
            before = dst.read(1)[0, 0]

        # Rewrite the source with different values and a newer mtime.
        with rasterio.open(dem, "r+") as src:
            src.write(np.full((1, 20, 30), 42.0, dtype="float32"))
        dem.touch()

        with rasterio.open(compensate_dem_for_snap(dem, cache_dir=cache)) as dst:
            assert dst.read(1)[0, 0] != before
            assert dst.read(1)[0, 0] == pytest.approx(42.0)

    def test_leaves_no_partial_file_behind(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        compensate_dem_for_snap(make_dem(), cache_dir=cache)

        assert not list(cache.glob("*.partial"))

    def test_missing_dem_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="DEM not found"):
            compensate_dem_for_snap(tmp_path / "absent.tif", cache_dir=tmp_path / "cache")

    def test_rejects_dem_already_in_cache(self, make_dem: Callable[..., Path]) -> None:
        dem = make_dem()

        with pytest.raises(ValueError, match="already inside"):
            compensate_dem_for_snap(dem, cache_dir=dem.parent)

    def test_defaults_cache_dir_to_settings(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache = tmp_path / "from-env"
        monkeypatch.setenv("SNAP_DEM_CACHE", str(cache))

        out = compensate_dem_for_snap(make_dem())

        assert out.parent == cache
