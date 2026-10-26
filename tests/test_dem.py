"""Tests for the SNAP DEM preparation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import s1bursts.dem as dem_module
from s1bursts.dem import (
    CACHE_SCHEMA_VERSION,
    COMPENSATED_TAG,
    PROVENANCE_SUFFIX,
    TARGET_CRS,
    VERTICAL_CONVERSION_TAG,
    prepare_dem_for_snap,
)
from s1bursts.snap_client import external_dem_needs_half_pixel_shift

if TYPE_CHECKING:
    from collections.abc import Callable

RES = 1.0 / 3600.0
ORIGIN_X = 7.0 - RES / 2
ORIGIN_Y = 47.0 + RES / 2


def _write_dem(
    path: Path,
    *,
    dtype: str = "float32",
    count: int = 1,
    fill: float | None,
    width: int = 30,
    height: int = 20,
    tiled: bool = False,
) -> Path:
    """Write a small georeferenced DEM, optionally filled with a constant."""
    data = (
        np.full((count, height, width), fill, dtype=dtype)
        if fill is not None
        else np.arange(height * width, dtype=dtype).reshape(1, height, width).repeat(count, axis=0)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    block_options = {"tiled": True, "blockxsize": 16, "blockysize": 16} if tiled else {}
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=count,
        dtype=dtype,
        crs="EPSG:4326",
        transform=from_origin(ORIGIN_X, ORIGIN_Y, RES, RES),
        nodata=np.nan if dtype.startswith("float") else 0,
        **block_options,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture(autouse=True)
def snap_needs_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the SNAP-dependent shift, so no test depends on the SNAP installed here."""
    monkeypatch.setattr(dem_module, "external_dem_needs_half_pixel_shift", lambda _bin=None: True)


@pytest.fixture
def make_dem(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory writing a small georeferenced DEM."""

    def _make(
        name: str = "dem.tif",
        dtype: str = "float32",
        count: int = 1,
        fill: float | None = None,
        width: int = 30,
        height: int = 20,
        tiled: bool = False,
    ) -> Path:
        return _write_dem(
            tmp_path / name,
            dtype=dtype,
            count=count,
            fill=fill,
            width=width,
            height=height,
            tiled=tiled,
        )

    return _make


def _sidecar_of(entry: Path) -> Path:
    return entry.with_name(entry.name + PROVENANCE_SUFFIX)


class TestPrepareForSnap:
    def test_moves_origin_onto_first_pixel_centre(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        dem = make_dem()
        out = prepare_dem_for_snap(dem, cache_dir=tmp_path / "cache")

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
        out = prepare_dem_for_snap(dem, cache_dir=tmp_path / "cache")

        with rasterio.open(dem) as src, rasterio.open(out) as dst:
            assert dst.res == src.res
            assert dst.shape == src.shape
            assert dst.crs == src.crs
            assert np.array_equal(src.read(), dst.read(), equal_nan=True)

    def test_tags_the_copy(self, make_dem: Callable[..., Path], tmp_path: Path) -> None:
        out = prepare_dem_for_snap(make_dem(), cache_dir=tmp_path / "cache")

        with rasterio.open(out) as dst:
            assert dst.tags()[COMPENSATED_TAG] == "1"

    def test_handles_integer_dtype(self, make_dem: Callable[..., Path], tmp_path: Path) -> None:
        """The float predictor is invalid for integers, so the dtype must pick it."""
        dem = make_dem(dtype="int16")
        out = prepare_dem_for_snap(dem, cache_dir=tmp_path / "cache")

        with rasterio.open(dem) as src, rasterio.open(out) as dst:
            assert dst.dtypes == src.dtypes
            assert np.array_equal(src.read(), dst.read())

    def test_handles_multiple_bands(self, make_dem: Callable[..., Path], tmp_path: Path) -> None:
        dem = make_dem(count=3)
        out = prepare_dem_for_snap(dem, cache_dir=tmp_path / "cache")

        with rasterio.open(dem) as src, rasterio.open(out) as dst:
            assert dst.count == 3
            assert np.array_equal(src.read(), dst.read(), equal_nan=True)

    def test_reuses_cached_copy(self, make_dem: Callable[..., Path], tmp_path: Path) -> None:
        dem = make_dem()
        cache = tmp_path / "cache"
        first = prepare_dem_for_snap(dem, cache_dir=cache)
        stamp = first.stat().st_mtime_ns

        second = prepare_dem_for_snap(dem, cache_dir=cache)

        assert second == first
        assert second.stat().st_mtime_ns == stamp

    def test_rewrites_when_source_changes(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        dem = make_dem()
        cache = tmp_path / "cache"
        out = prepare_dem_for_snap(dem, cache_dir=cache)
        with rasterio.open(out) as dst:
            before = dst.read(1)[0, 0]

        # Rewrite the source with different values and a newer mtime.
        with rasterio.open(dem, "r+") as src:
            src.write(np.full((1, 20, 30), 42.0, dtype="float32"))
        dem.touch()

        with rasterio.open(prepare_dem_for_snap(dem, cache_dir=cache)) as dst:
            assert dst.read(1)[0, 0] != before
            assert dst.read(1)[0, 0] == pytest.approx(42.0)

    def test_leaves_no_partial_file_behind(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        prepare_dem_for_snap(make_dem(), cache_dir=cache)

        assert not list(cache.glob("*.partial"))

    def test_missing_dem_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="DEM not found"):
            prepare_dem_for_snap(tmp_path / "absent.tif", cache_dir=tmp_path / "cache")

    def test_rejects_dem_already_in_cache(self, make_dem: Callable[..., Path]) -> None:
        dem = make_dem()

        with pytest.raises(ValueError, match="already inside"):
            prepare_dem_for_snap(dem, cache_dir=dem.parent)

    def test_defaults_cache_dir_to_settings(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cache = tmp_path / "from-env"
        monkeypatch.setenv("SNAP_DEM_CACHE", str(cache))

        out = prepare_dem_for_snap(make_dem())

        assert out.parent == cache


class TestSnapThatSamplesInPlace:
    """A SNAP whose ``snap-dem`` no longer needs the half-pixel shift."""

    @pytest.fixture(autouse=True)
    def fixed_snap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            dem_module,
            "external_dem_needs_half_pixel_shift",
            lambda _bin=None: False,
        )

    def test_leaves_the_origin_where_it_was(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        dem = make_dem()
        out = prepare_dem_for_snap(dem, cache_dir=tmp_path / "cache")

        with rasterio.open(dem) as src, rasterio.open(out) as dst:
            assert dst.transform == src.transform
            assert np.array_equal(src.read(), dst.read(), equal_nan=True)

    def test_tags_and_records_the_copy_as_unshifted(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        out = prepare_dem_for_snap(make_dem(), cache_dir=tmp_path / "cache")

        with rasterio.open(out) as dst:
            assert dst.tags()[COMPENSATED_TAG] == "0"
        assert json.loads(_sidecar_of(out).read_text())["pixel_offset"] == [0.0, 0.0]

    def test_reuses_its_cached_copy(self, make_dem: Callable[..., Path], tmp_path: Path) -> None:
        dem = make_dem()
        cache = tmp_path / "cache"
        first = prepare_dem_for_snap(dem, cache_dir=cache)
        stamp = first.stat().st_mtime_ns

        assert prepare_dem_for_snap(dem, cache_dir=cache) == first
        assert first.stat().st_mtime_ns == stamp

    def test_shifted_copy_is_not_reused(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A copy prepared for an older SNAP stays in the cache, but is never handed out."""
        dem = make_dem()
        cache = tmp_path / "cache"
        with monkeypatch.context() as older:
            older.setattr(
                dem_module, "external_dem_needs_half_pixel_shift", lambda _bin=None: True
            )
            shifted = prepare_dem_for_snap(dem, cache_dir=cache)

        unshifted = prepare_dem_for_snap(dem, cache_dir=cache)

        assert unshifted != shifted
        with rasterio.open(shifted) as old, rasterio.open(unshifted) as new:
            assert old.transform.c == pytest.approx(new.transform.c + RES / 2)

    def test_entry_tagged_as_shifted_is_rewritten(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        dem = make_dem()
        out = prepare_dem_for_snap(dem, cache_dir=cache)
        with rasterio.open(out, "r+") as dst:
            dst.update_tags(**{COMPENSATED_TAG: "1"})
        # Keep the recorded size in step so the tag check, not the size check, decides.
        sidecar = _sidecar_of(out)
        record = json.loads(sidecar.read_text())
        record["output_size"] = out.stat().st_size
        sidecar.write_text(json.dumps(record))

        with rasterio.open(prepare_dem_for_snap(dem, cache_dir=cache)) as src:
            assert src.tags()[COMPENSATED_TAG] == "0"


class TestSnapSelection:
    @pytest.mark.parametrize(("version", "shifted"), [("14.0.1", True), ("14.0.2", False)])
    def test_decides_from_the_installation_unmocked(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        version: str,
        shifted: bool,
    ) -> None:
        """The version check and `snap_bin` forwarding work together, with no stub between."""
        monkeypatch.setattr(
            dem_module,
            "external_dem_needs_half_pixel_shift",
            external_dem_needs_half_pixel_shift,
        )
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
        bin_dir = tmp_path / "snap" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "gpt").touch()
        (bin_dir / "gpt").chmod(0o755)
        jar = tmp_path / "snap" / "snap" / "modules" / "org-esa-snap-snap-dem.jar"
        jar.parent.mkdir(parents=True)
        with zipfile.ZipFile(jar, "w") as archive:
            archive.writestr(
                "META-INF/MANIFEST.MF",
                f"Manifest-Version: 1.0\nOpenIDE-Module-Specification-Version: {version}\n",
            )
        dem = make_dem()

        out = prepare_dem_for_snap(dem, cache_dir=tmp_path / "cache", snap_bin=bin_dir)

        with rasterio.open(dem) as src, rasterio.open(out) as dst:
            expected = src.transform.c + RES / 2 if shifted else src.transform.c
            assert dst.transform.c == pytest.approx(expected)

    def test_asks_the_requested_installation(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        asked: list[Path | None] = []

        def _record(bin_dir: Path | None = None) -> bool:
            asked.append(bin_dir)
            return True

        monkeypatch.setattr(dem_module, "external_dem_needs_half_pixel_shift", _record)
        snap_bin = tmp_path / "snap" / "bin"

        prepare_dem_for_snap(make_dem(), cache_dir=tmp_path / "cache", snap_bin=snap_bin)

        assert asked == [snap_bin]


class _OffsetTransformer:
    """Small deterministic stand-in for a PROJ vertical transformer."""

    def __init__(self, offset: float = 10.0) -> None:
        self.offset = offset
        self.description = f"Test vertical offset of {offset} metres"
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = []

    def transform(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        *,
        errcheck: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.calls.append((x.copy(), y.copy(), z.copy(), errcheck))
        return x, y, z + self.offset


class TestVerticalConversion:
    def test_converts_at_original_pixel_centres(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dem = make_dem(fill=100.0)
        transformer = _OffsetTransformer()
        monkeypatch.setattr(
            dem_module,
            "_get_transformer",
            lambda *_args, **_kwargs: transformer,
        )

        out = prepare_dem_for_snap(
            dem,
            source_crs="EPSG:4326+3855",
            cache_dir=tmp_path / "cache",
        )

        with rasterio.open(out) as src:
            assert np.all(src.read(1) == pytest.approx(110.0))
            assert src.tags()[VERTICAL_CONVERSION_TAG] == "1"
        x, y, z, errcheck = transformer.calls[0]
        assert x[0] == pytest.approx(7.0)
        assert y[0] == pytest.approx(47.0)
        assert np.all(z == pytest.approx(100.0))
        assert errcheck is True

    def test_uses_offsets_for_every_tiled_window(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dem = make_dem(fill=100.0, width=32, height=32, tiled=True)
        transformer = _OffsetTransformer()
        monkeypatch.setattr(
            dem_module,
            "_get_transformer",
            lambda *_args, **_kwargs: transformer,
        )

        prepare_dem_for_snap(
            dem,
            source_crs="EPSG:4326+3855",
            cache_dir=tmp_path / "cache",
        )

        assert len(transformer.calls) == 4
        last_x, last_y, _, _ = transformer.calls[-1]
        assert last_x[0] == pytest.approx(7.0 + 16 * RES)
        assert last_y[0] == pytest.approx(47.0 - 16 * RES)
        assert last_x[-1] == pytest.approx(7.0 + 31 * RES)
        assert last_y[-1] == pytest.approx(47.0 - 31 * RES)

    def test_preserves_nodata_and_promotes_integer_dem(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dem = make_dem(dtype="int16", fill=2.0)
        with rasterio.open(dem, "r+") as src:
            data = src.read(1)
            data[0, 0] = 0
            src.write(data, 1)
        monkeypatch.setattr(
            dem_module,
            "_get_transformer",
            lambda *_args, **_kwargs: _OffsetTransformer(),
        )

        out = prepare_dem_for_snap(
            dem,
            source_crs="EPSG:4326+3855",
            cache_dir=tmp_path / "cache",
        )

        with rasterio.open(out) as src:
            assert src.dtypes == ("float32",)
            assert src.read(1)[0, 0] == pytest.approx(0.0)
            assert src.read(1)[0, 1] == pytest.approx(12.0)

    def test_preserves_nan_and_converts_every_band(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dem = make_dem(count=3, fill=5.0)
        with rasterio.open(dem, "r+") as src:
            data = src.read()
            data[:, 0, 0] = np.nan
            src.write(data)
        transformer = _OffsetTransformer()
        monkeypatch.setattr(
            dem_module,
            "_get_transformer",
            lambda *_args, **_kwargs: transformer,
        )

        out = prepare_dem_for_snap(
            dem,
            source_crs="EPSG:4326+3855",
            cache_dir=tmp_path / "cache",
        )

        with rasterio.open(out) as src:
            data = src.read()
            assert np.isnan(data[:, 0, 0]).all()
            assert np.all(data[:, 0, 1] == pytest.approx(15.0))
        assert len(transformer.calls) == 3
        assert all(len(call[2]) == 599 for call in transformer.calls)

    @pytest.mark.parametrize(
        ("source_crs", "message"),
        [
            ("EPSG:4326", "source_crs must be a compound"),
            ("not-a-crs", "Invalid source CRS"),
        ],
    )
    def test_rejects_invalid_source_crs(
        self,
        source_crs: str,
        message: str,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            prepare_dem_for_snap(
                make_dem(),
                source_crs=source_crs,
                cache_dir=tmp_path / "cache",
            )

    def test_rejects_horizontal_crs_mismatch(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="does not match"):
            prepare_dem_for_snap(
                make_dem(),
                source_crs="EPSG:2056+5728",
                cache_dir=tmp_path / "cache",
            )

    def test_conversion_settings_have_distinct_cache_entries(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dem = make_dem(fill=3.0)
        cache = tmp_path / "cache"
        monkeypatch.setattr(
            dem_module,
            "_get_transformer",
            lambda *_args, **_kwargs: _OffsetTransformer(),
        )

        unchanged = prepare_dem_for_snap(dem, cache_dir=cache)
        converted = prepare_dem_for_snap(
            dem,
            source_crs="EPSG:4326+3855",
            cache_dir=cache,
        )

        assert unchanged != converted
        record = json.loads(_sidecar_of(converted).read_text())
        assert record["target_crs"] == TARGET_CRS
        assert record["source_crs"] == "EPSG:9518"
        assert record["transformer_description"] == "Test vertical offset of 10.0 metres"
        assert record["proj_version"] == dem_module.proj_version_str

    def test_cached_conversion_does_not_need_the_grid_again(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dem = make_dem(fill=3.0)
        cache = tmp_path / "cache"
        monkeypatch.setattr(
            dem_module,
            "_get_transformer",
            lambda *_args, **_kwargs: _OffsetTransformer(),
        )
        first = prepare_dem_for_snap(
            dem,
            source_crs="EPSG:4326+3855",
            cache_dir=cache,
        )
        monkeypatch.setattr(
            dem_module,
            "_get_transformer",
            lambda *_args, **_kwargs: pytest.fail("cache hit requested a transformer"),
        )

        second = prepare_dem_for_snap(
            dem,
            source_crs="EPSG:4326+3855",
            cache_dir=cache,
        )

        assert second == first


def _fake_grid(short_name: str, *, fetchable: bool = True) -> SimpleNamespace:
    """Build a stand-in for a pyproj grid record.

    ``fetchable`` mirrors the two flags that decide whether PROJ can install a
    grid itself; a grid such as EGM2008's 1'x1' NGA file has both unset and can
    only ever be obtained from its producer.
    """
    return SimpleNamespace(
        short_name=short_name,
        available=False,
        open_license=fetchable,
        direct_download=fetchable,
    )


class TestTransformationGrids:
    @staticmethod
    def _group(
        *,
        best_available: bool,
        transformer: object | None = None,
        fetchable: bool = True,
    ) -> SimpleNamespace:
        operation = SimpleNamespace(grids=[_fake_grid("test-grid.tif", fetchable=fetchable)])
        return SimpleNamespace(
            best_available=best_available,
            transformers=[] if transformer is None else [transformer],
            unavailable_operations=[] if best_available else [operation],
        )

    def test_missing_grid_error_is_actionable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        group = self._group(best_available=False)
        monkeypatch.setattr(dem_module, "_new_transformer_group", lambda _source_crs: group)

        with pytest.raises(RuntimeError, match=r"test-grid\.tif.*download_missing_grids"):
            dem_module._get_transformer(
                dem_module.CRS("EPSG:4326+3855"),
                download_missing_grids=False,
            )

    def test_opt_in_download_rebuilds_transformer(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        transformer = _OffsetTransformer()
        groups = iter(
            [
                self._group(best_available=False),
                self._group(best_available=True, transformer=transformer),
            ],
        )
        downloaded: list[object] = []
        monkeypatch.setattr(
            dem_module,
            "_new_transformer_group",
            lambda _source_crs: next(groups),
        )
        monkeypatch.setattr(dem_module, "_download_grids", downloaded.append)

        result = dem_module._get_transformer(
            dem_module.CRS("EPSG:4326+3855"),
            download_missing_grids=True,
        )

        assert result is transformer
        assert len(downloaded) == 1

    def test_failed_opt_in_does_not_recommend_the_flag_again(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        group = self._group(best_available=False)
        monkeypatch.setattr(dem_module, "_new_transformer_group", lambda _source_crs: group)
        monkeypatch.setattr(dem_module, "_download_grids", lambda _group: None)

        with pytest.raises(RuntimeError, match="Downloading was enabled") as error:
            dem_module._get_transformer(
                dem_module.CRS("EPSG:4326+3855"),
                download_missing_grids=True,
            )

        assert "Pass download_missing_grids=True" not in str(error.value)

    def test_download_refreshes_proj_grid_cache_before_rebuilding(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rebuild that skips the refresh reads a stale PROJ cache.

        PROJ answers grid availability from a cache held by the context, and
        pyproj keeps one context per thread for the life of the process. Without
        an explicit refresh the group rebuilt after a download still reports the
        grid that was just installed as missing.
        """
        transformer = _OffsetTransformer()
        groups = iter(
            [
                self._group(best_available=False),
                self._group(best_available=True, transformer=transformer),
            ],
        )
        events: list[str] = []

        def _build(_source_crs: object) -> SimpleNamespace:
            events.append("build")
            return next(groups)

        monkeypatch.setattr(dem_module, "_new_transformer_group", _build)
        monkeypatch.setattr(
            dem_module,
            "_download_grids",
            lambda _group: events.append("download"),
        )
        monkeypatch.setattr(
            dem_module,
            "_refresh_grid_availability",
            lambda: events.append("refresh"),
        )

        result = dem_module._get_transformer(
            dem_module.CRS("EPSG:4326+3855"),
            download_missing_grids=True,
        )

        assert result is transformer
        assert events == ["build", "download", "refresh", "build"]

    def test_refresh_keeps_the_proj_data_directory_unchanged(self) -> None:
        """The refresh must rebuild the context without repointing it."""
        before = dem_module.datadir.get_data_dir()

        dem_module._refresh_grid_availability()

        assert dem_module.datadir.get_data_dir() == before

    def test_grid_proj_cannot_download_is_not_attempted(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Opting in must not promise a download PROJ is unable to perform."""
        group = self._group(best_available=False, fetchable=False)
        downloaded: list[object] = []
        monkeypatch.setattr(dem_module, "_new_transformer_group", lambda _source_crs: group)
        monkeypatch.setattr(dem_module, "_download_grids", downloaded.append)

        with pytest.raises(RuntimeError, match=r"cannot download: test-grid\.tif") as error:
            dem_module._get_transformer(
                dem_module.CRS("EPSG:4326+3855"),
                download_missing_grids=True,
            )

        assert downloaded == []
        assert "pyproj sync" not in str(error.value)
        assert "manually" in str(error.value)

    def test_error_reports_downloadable_and_restricted_grids_apart(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A grid PROJ can never fetch must not be presented as the thing to install.

        EGM2008 reaches EPSG:4979 through either of two operations. Only the one
        needing `us_nga_egm08_25.tif` is usable; the alternative needs an NGA file
        that is not redistributable and stays unavailable forever, so naming it
        alongside sends the reader after the wrong grid.
        """
        group = SimpleNamespace(
            best_available=False,
            transformers=[],
            unavailable_operations=[
                SimpleNamespace(grids=[_fake_grid("us_nga_egm08_25.tif")]),
                SimpleNamespace(grids=[_fake_grid("Und_min1x1_egm2008.gz", fetchable=False)]),
            ],
        )
        monkeypatch.setattr(dem_module, "_new_transformer_group", lambda _source_crs: group)

        with pytest.raises(RuntimeError) as error:
            dem_module._get_transformer(
                dem_module.CRS("EPSG:4326+3855"),
                download_missing_grids=False,
            )

        message = str(error.value)
        assert "Missing downloadable grid(s): us_nga_egm08_25.tif." in message
        assert "cannot download: Und_min1x1_egm2008.gz." in message
        assert "Pass download_missing_grids=True" in message

    def test_downloads_open_grids_and_publishes_atomically(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        user_dir = tmp_path / "proj"
        calls: list[tuple[Path, bool]] = []
        grid = _fake_grid("test-grid.tif")

        class DownloadGroup:
            def __init__(self) -> None:
                self.unavailable_operations = [SimpleNamespace(grids=[grid])]

            @staticmethod
            def download_grids(*, directory: Path, open_license: bool) -> None:
                calls.append((directory, open_license))
                (directory / "test-grid.tif").write_bytes(b"complete-grid")

        monkeypatch.setattr(
            dem_module.datadir,
            "get_user_data_dir",
            lambda *, create: str(user_dir),  # noqa: ARG005
        )
        user_dir.mkdir()

        # Local double with just the attributes read here.
        # pyrefly: ignore [bad-argument-type]
        dem_module._download_grids(DownloadGroup())

        assert calls[0][1] is True
        assert (user_dir / "test-grid.tif").read_bytes() == b"complete-grid"
        assert not list(user_dir.glob("s1bursts-proj-*"))

    def test_download_raises_when_no_open_grid_is_available(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        user_dir = tmp_path / "proj"
        grid = _fake_grid("restricted-grid.gtx", fetchable=False)
        attempts: list[Path] = []

        class DownloadGroup:
            def __init__(self) -> None:
                self.unavailable_operations = [SimpleNamespace(grids=[grid])]

            @staticmethod
            def download_grids(*, directory: Path, open_license: bool) -> None:  # noqa: ARG004
                attempts.append(directory)

        monkeypatch.setattr(
            dem_module.datadir,
            "get_user_data_dir",
            lambda *, create: str(user_dir),  # noqa: ARG005
        )
        user_dir.mkdir()

        with pytest.raises(RuntimeError, match=r"No directly downloadable open-license.*manually"):
            # pyrefly: ignore [bad-argument-type]
            dem_module._download_grids(DownloadGroup())

        assert attempts == []


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

        out_first = prepare_dem_for_snap(first, cache_dir=cache)
        out_second = prepare_dem_for_snap(second, cache_dir=cache)

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
        prepare_dem_for_snap(dem, cache_dir=cache)

        _write_dem(dem, fill=2.0)
        os.utime(dem, (stat.st_atime, stat.st_mtime))

        with rasterio.open(prepare_dem_for_snap(dem, cache_dir=cache)) as src:
            assert src.read(1)[0, 0] == pytest.approx(2.0)

    def test_identical_content_under_different_names_shares_one_entry(
        self,
        tmp_path: Path,
    ) -> None:
        """Byte-identical sources converge on the same entry, so the work happens once."""
        cache = tmp_path / "cache"
        first = _write_dem(tmp_path / "a" / "dem.tif", fill=3.0)
        second = _write_dem(tmp_path / "b" / "dem.tif", fill=3.0)

        assert prepare_dem_for_snap(first, cache_dir=cache) == prepare_dem_for_snap(
            second,
            cache_dir=cache,
        )

    def test_records_source_provenance(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        dem = make_dem()
        out = prepare_dem_for_snap(dem, cache_dir=tmp_path / "cache")

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
        out = prepare_dem_for_snap(dem, cache_dir=cache)
        out.write_bytes(out.read_bytes()[:100])

        again = prepare_dem_for_snap(dem, cache_dir=cache)

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
        out = prepare_dem_for_snap(dem, cache_dir=cache)
        out.write_bytes(b"\x00" * out.stat().st_size)

        with rasterio.open(prepare_dem_for_snap(dem, cache_dir=cache)) as src:
            assert src.read(1)[0, 0] == pytest.approx(5.0)

    def test_missing_provenance_forces_a_rewrite(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        dem = make_dem(fill=5.0)
        out = prepare_dem_for_snap(dem, cache_dir=cache)
        _sidecar_of(out).unlink()

        assert prepare_dem_for_snap(dem, cache_dir=cache) == out
        assert _sidecar_of(out).exists()

    def test_unreadable_provenance_forces_a_rewrite(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        dem = make_dem()
        out = prepare_dem_for_snap(dem, cache_dir=cache)
        _sidecar_of(out).write_text("{not json")

        assert prepare_dem_for_snap(dem, cache_dir=cache) == out
        assert json.loads(_sidecar_of(out).read_text())["schema_version"] == CACHE_SCHEMA_VERSION

    def test_stale_schema_version_forces_a_rewrite(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        dem = make_dem()
        out = prepare_dem_for_snap(dem, cache_dir=cache)
        sidecar = _sidecar_of(out)
        record = json.loads(sidecar.read_text())
        record["schema_version"] = CACHE_SCHEMA_VERSION - 1
        sidecar.write_text(json.dumps(record))

        prepare_dem_for_snap(dem, cache_dir=cache)

        assert json.loads(sidecar.read_text())["schema_version"] == CACHE_SCHEMA_VERSION

    def test_entry_missing_the_compensation_tag_is_rewritten(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        dem = make_dem()
        out = prepare_dem_for_snap(dem, cache_dir=cache)
        with rasterio.open(out, "r+") as dst:
            dst.update_tags(**{COMPENSATED_TAG: "0"})
        # Keep the recorded size in step so the tag check, not the size check, decides.
        sidecar = _sidecar_of(out)
        record = json.loads(sidecar.read_text())
        record["output_size"] = out.stat().st_size
        sidecar.write_text(json.dumps(record))

        with rasterio.open(prepare_dem_for_snap(dem, cache_dir=cache)) as src:
            assert src.tags()[COMPENSATED_TAG] == "1"

    def test_entry_with_wrong_vertical_conversion_tag_is_rewritten(
        self,
        make_dem: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        cache = tmp_path / "cache"
        dem = make_dem()
        out = prepare_dem_for_snap(dem, cache_dir=cache)
        with rasterio.open(out, "r+") as dst:
            dst.update_tags(**{VERTICAL_CONVERSION_TAG: "1"})
        sidecar = _sidecar_of(out)
        record = json.loads(sidecar.read_text())
        record["output_size"] = out.stat().st_size
        sidecar.write_text(json.dumps(record))

        with rasterio.open(prepare_dem_for_snap(dem, cache_dir=cache)) as src:
            assert src.tags()[VERTICAL_CONVERSION_TAG] == "0"


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
                pool.map(lambda _: prepare_dem_for_snap(dem, cache_dir=cache), range(8)),
            )

        assert len(set(results)) == 1
        assert not list(cache.glob("*.partial"))
        with rasterio.open(results[0]) as src:
            assert src.read(1)[0, 0] == pytest.approx(7.0)
            assert src.tags()[COMPENSATED_TAG] == "1"
