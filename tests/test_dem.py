"""Tests for the SNAP half-pixel DEM compensation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from s1bursts.dem import (
    CACHE_SCHEMA_VERSION,
    COMPENSATED_TAG,
    PROVENANCE_SUFFIX,
    compensate_dem_for_snap,
)

if TYPE_CHECKING:
    from collections.abc import Callable

RES = 1.0 / 3600.0
ORIGIN_X = 7.0 - RES / 2
ORIGIN_Y = 47.0 + RES / 2


def _write_dem(path: Path, *, dtype: str = "float32", count: int = 1, fill: float | None) -> Path:
    """Write a small georeferenced DEM, optionally filled with a constant."""
    data = (
        np.full((count, 20, 30), fill, dtype=dtype)
        if fill is not None
        else np.arange(20 * 30, dtype=dtype).reshape(1, 20, 30).repeat(count, axis=0)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
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


@pytest.fixture
def make_dem(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory writing a small georeferenced DEM."""

    def _make(
        name: str = "dem.tif",
        dtype: str = "float32",
        count: int = 1,
        fill: float | None = None,
    ) -> Path:
        return _write_dem(tmp_path / name, dtype=dtype, count=count, fill=fill)

    return _make


def _sidecar_of(entry: Path) -> Path:
    return entry.with_name(entry.name + PROVENANCE_SUFFIX)


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


class TestCacheIdentity:
    """The cache must key on source content, never on the source's basename."""

    def test_same_basename_different_directories_do_not_collide(self, tmp_path: Path) -> None:
        """Two distinct DEMs called `dem.tif` must not be served from one entry."""
        cache = tmp_path / "cache"
        first = _write_dem(tmp_path / "a" / "dem.tif", fill=1.0)
        second = _write_dem(tmp_path / "b" / "dem.tif", fill=2.0)
        # The colliding source is deliberately the older file: an mtime-based
        # cache would consider the entry written for `a/dem.tif` fresh for it.
        old = first.stat().st_mtime - 3600
        os.utime(second, (old, old))

        out_first = compensate_dem_for_snap(first, cache_dir=cache)
        out_second = compensate_dem_for_snap(second, cache_dir=cache)

        assert out_first != out_second
        with rasterio.open(out_first) as src:
            assert src.read(1)[0, 0] == pytest.approx(1.0)
        with rasterio.open(out_second) as src:
            assert src.read(1)[0, 0] == pytest.approx(2.0)

    def test_content_change_with_preserved_mtime_is_not_reused(self, tmp_path: Path) -> None:
        """Rewriting a DEM in place without advancing its mtime must invalidate."""
        cache = tmp_path / "cache"
        dem = _write_dem(tmp_path / "dem.tif", fill=1.0)
        stat = dem.stat()
        compensate_dem_for_snap(dem, cache_dir=cache)

        _write_dem(dem, fill=2.0)
        os.utime(dem, (stat.st_atime, stat.st_mtime))

        with rasterio.open(compensate_dem_for_snap(dem, cache_dir=cache)) as src:
            assert src.read(1)[0, 0] == pytest.approx(2.0)

    def test_identical_content_under_different_names_shares_one_entry(
        self,
        tmp_path: Path,
    ) -> None:
        """Byte-identical sources converge on the same entry, so the work happens once."""
        cache = tmp_path / "cache"
        first = _write_dem(tmp_path / "a" / "dem.tif", fill=3.0)
        second = _write_dem(tmp_path / "b" / "dem.tif", fill=3.0)

        assert compensate_dem_for_snap(first, cache_dir=cache) == compensate_dem_for_snap(
            second,
            cache_dir=cache,
        )

    def test_records_source_provenance(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        dem = make_dem()
        out = compensate_dem_for_snap(dem, cache_dir=tmp_path / "cache")

        record = json.loads(_sidecar_of(out).read_text())
        assert record["source_path"] == str(dem.resolve())
        assert record["schema_version"] == CACHE_SCHEMA_VERSION
        assert record["source_size"] == dem.stat().st_size
        assert record["output_size"] == out.stat().st_size
        assert len(record["source_sha256"]) == 64
        assert record["pixel_offset"] == [0.5, 0.5]


class TestCacheIntegrity:
    """A damaged or unattributable entry must be rewritten, never handed to SNAP."""

    def test_truncated_entry_is_rewritten(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        dem = make_dem(fill=5.0)
        out = compensate_dem_for_snap(dem, cache_dir=cache)
        out.write_bytes(out.read_bytes()[:100])

        again = compensate_dem_for_snap(dem, cache_dir=cache)

        assert again == out
        with rasterio.open(again) as src:
            assert src.read(1)[0, 0] == pytest.approx(5.0)

    def test_entry_of_the_right_size_but_wrong_content_is_rewritten(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        """Same-size garbage must fail the structural check, not slip through."""
        cache = tmp_path / "cache"
        dem = make_dem(fill=5.0)
        out = compensate_dem_for_snap(dem, cache_dir=cache)
        out.write_bytes(b"\x00" * out.stat().st_size)

        with rasterio.open(compensate_dem_for_snap(dem, cache_dir=cache)) as src:
            assert src.read(1)[0, 0] == pytest.approx(5.0)

    def test_missing_provenance_forces_a_rewrite(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        dem = make_dem(fill=5.0)
        out = compensate_dem_for_snap(dem, cache_dir=cache)
        _sidecar_of(out).unlink()

        assert compensate_dem_for_snap(dem, cache_dir=cache) == out
        assert _sidecar_of(out).exists()

    def test_unreadable_provenance_forces_a_rewrite(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        dem = make_dem()
        out = compensate_dem_for_snap(dem, cache_dir=cache)
        _sidecar_of(out).write_text("{not json")

        assert compensate_dem_for_snap(dem, cache_dir=cache) == out
        assert json.loads(_sidecar_of(out).read_text())["schema_version"] == CACHE_SCHEMA_VERSION

    def test_stale_schema_version_forces_a_rewrite(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        dem = make_dem()
        out = compensate_dem_for_snap(dem, cache_dir=cache)
        sidecar = _sidecar_of(out)
        record = json.loads(sidecar.read_text())
        record["schema_version"] = CACHE_SCHEMA_VERSION - 1
        sidecar.write_text(json.dumps(record))

        compensate_dem_for_snap(dem, cache_dir=cache)

        assert json.loads(sidecar.read_text())["schema_version"] == CACHE_SCHEMA_VERSION

    def test_entry_missing_the_compensation_tag_is_rewritten(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        dem = make_dem()
        out = compensate_dem_for_snap(dem, cache_dir=cache)
        with rasterio.open(out, "r+") as dst:
            dst.update_tags(**{COMPENSATED_TAG: "0"})
        # Keep the recorded size in step so the tag check, not the size check, decides.
        sidecar = _sidecar_of(out)
        record = json.loads(sidecar.read_text())
        record["output_size"] = out.stat().st_size
        sidecar.write_text(json.dumps(record))

        with rasterio.open(compensate_dem_for_snap(dem, cache_dir=cache)) as src:
            assert src.tags()[COMPENSATED_TAG] == "1"


class TestConcurrency:
    def test_concurrent_callers_agree_on_one_valid_entry(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        dem = make_dem(fill=7.0)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(lambda _: compensate_dem_for_snap(dem, cache_dir=cache), range(8)),
            )

        assert len(set(results)) == 1
        assert not list(cache.glob("*.partial"))
        with rasterio.open(results[0]) as src:
            assert src.read(1)[0, 0] == pytest.approx(7.0)
            assert src.tags()[COMPENSATED_TAG] == "1"
