"""Tests for checking an existing per-burst product's settings before it is skipped."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import warnings

from affine import Affine
import numpy as np
from pandas import Timestamp
from pyproj import CRS
import pytest
import rasterio
import rasterio.errors

from s1bursts import product_tags
from s1bursts.errors import (
    BurstError,
    CompositeError,
    ProductSettingsMismatchError,
    S1BurstsError,
)
from s1bursts.product import CoherenceProduct, InterferogramProduct, RTCProduct
from s1bursts.settings_check import (
    _BurstSkip,
    _check_burst_product,
    _warn_other_snap,
    _warn_unchecked,
)
from s1bursts.snap_graph import CoherenceConfig, InterferogramConfig, RTCConfig

if TYPE_CHECKING:
    from s1bursts.snap_graph import BackscatterConfig

_UTM32 = CRS.from_epsg(32632)
_SNAP_14 = (14, 0, 0)


def _rtc() -> RTCProduct:
    return RTCProduct(
        burst_id=30346,
        sub_swath="IW1",
        crs=_UTM32,
        date=Timestamp("2022-01-05"),
        polarization="VV",
        pass_direction="Ascending",
        _rel_orbit_number=66,
    )


def _pair[P: (CoherenceProduct, InterferogramProduct)](cls: type[P]) -> P:
    return cls(
        burst_id=30346,
        sub_swath="IW1",
        crs=_UTM32,
        date_primary=Timestamp("2022-01-05"),
        date_secondary=Timestamp("2022-01-17"),
        polarization="VV",
        pass_direction="Ascending",
        _rel_orbit_number=66,
    )


def _write(path: Path, tags: dict[str, str] | None) -> Path:
    """Write a one-pixel GeoTIFF carrying ``tags``."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1,
        height=1,
        count=1,
        dtype="float32",
        crs=_UTM32,
        transform=Affine(10.0, 0.0, 400000.0, 0.0, -10.0, 5200000.0),
    ) as dst:
        dst.write(np.ones((1, 1, 1), dtype="float32"))
        if tags:
            dst.update_tags(**tags)
    return path


def _made_with(
    tmp_path: Path,
    product: RTCProduct | CoherenceProduct | InterferogramProduct,
    config: BackscatterConfig | CoherenceConfig | InterferogramConfig,
    snap_version: tuple[int, ...] | None = _SNAP_14,
) -> Path:
    """Write the file ``config`` would have produced, tags included."""
    metadata = product_tags.burst_metadata(product, config, snap_version=snap_version)
    return _write(tmp_path / "product.tif", metadata.tags)


def _check(
    path: Path,
    product: RTCProduct | CoherenceProduct | InterferogramProduct,
    config: BackscatterConfig | CoherenceConfig | InterferogramConfig,
    snap_version: tuple[int, ...] | None = _SNAP_14,
) -> _BurstSkip:
    return _check_burst_product(path, product, config, snap_version=snap_version)


class TestBurstProducts:
    def test_matching_settings_pass(self, tmp_path: Path) -> None:
        path = _made_with(tmp_path, _rtc(), RTCConfig())

        assert _check(path, _rtc(), RTCConfig()) == _BurstSkip()

    def test_another_grid_spacing_raises_naming_both_values(self, tmp_path: Path) -> None:
        path = _made_with(tmp_path, _rtc(), RTCConfig(grid_spacing=20.0))

        with pytest.raises(ProductSettingsMismatchError) as excinfo:
            _check(path, _rtc(), RTCConfig())

        assert excinfo.value.path == path
        assert excinfo.value.differences == {"GRID_SPACING": ("20 m", "10 m")}
        assert "GRID_SPACING (grid_spacing): recorded '20 m', requested '10 m'" in str(
            excinfo.value
        )

    @pytest.mark.parametrize(
        ("made", "requested", "differences"),
        [
            pytest.param(
                RTCConfig(multilook=(3, 3)),
                RTCConfig(),
                {"MULTILOOK_RANGE": ("3", "5"), "MULTILOOK_AZIMUTH": ("3", "1")},
                id="looks",
            ),
            pytest.param(
                RTCConfig(multilook=None),
                RTCConfig(),
                {"MULTILOOK_RANGE": (None, "5"), "MULTILOOK_AZIMUTH": (None, "1")},
                id="none-recorded",
            ),
            pytest.param(
                RTCConfig(),
                RTCConfig(multilook=None),
                {"MULTILOOK_RANGE": ("5", None), "MULTILOOK_AZIMUTH": ("1", None)},
                id="none-requested",
            ),
            pytest.param(
                RTCConfig(dem="SRTM 1Sec HGT"),
                RTCConfig(),
                {"DEM": ("SRTM 1Sec HGT", "Copernicus 30m Global DEM")},
                id="dem",
            ),
        ],
    )
    def test_backscatter_settings_are_compared(
        self,
        tmp_path: Path,
        made: RTCConfig,
        requested: RTCConfig,
        differences: dict[str, tuple[str | None, str | None]],
    ) -> None:
        path = _made_with(tmp_path, _rtc(), made)

        with pytest.raises(ProductSettingsMismatchError) as excinfo:
            _check(path, _rtc(), requested)

        assert excinfo.value.differences == differences

    def test_the_coherence_window_is_compared(self, tmp_path: Path) -> None:
        product = _pair(CoherenceProduct)
        path = _made_with(tmp_path, product, CoherenceConfig(window=(10, 2)))

        with pytest.raises(ProductSettingsMismatchError, match=r"\(window\)") as excinfo:
            _check(path, product, CoherenceConfig())

        assert excinfo.value.differences == {
            "COHERENCE_WINDOW_RANGE": ("10", "25"),
            "COHERENCE_WINDOW_AZIMUTH": ("2", "5"),
        }

    def test_the_interferogram_phase_removal_is_compared(self, tmp_path: Path) -> None:
        product = _pair(InterferogramProduct)
        path = _made_with(tmp_path, product, InterferogramConfig(subtract_topographic_phase=False))

        with pytest.raises(ProductSettingsMismatchError) as excinfo:
            _check(path, product, InterferogramConfig())

        assert excinfo.value.differences == {"TOPOGRAPHIC_PHASE_REMOVED": ("false", "true")}

    def test_settings_the_tags_do_not_record_are_not_compared(self, tmp_path: Path) -> None:
        path = _made_with(tmp_path, _rtc(), RTCConfig(dem_oversampling=2.0))

        assert _check(path, _rtc(), RTCConfig()) == _BurstSkip()


class TestUnverifiable:
    """A product whose settings are not recorded is skipped and says why."""

    def test_a_file_written_before_the_tags_existed(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "old.tif", {"AREA_OR_POINT": "Area"})

        assert _check(path, _rtc(), RTCConfig()).unchecked == "untagged"

    def test_an_unreadable_file_raises_naming_it(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.tif"
        path.write_bytes(b"not a raster")

        with pytest.raises(rasterio.errors.RasterioIOError) as excinfo:
            _check(path, _rtc(), RTCConfig())

        assert excinfo.value.__notes__ == [
            (
                f"Reading '{path}', an existing product, to check its settings failed. "
                "Pass overwrite=True to rebuild it."
            )
        ]

    @pytest.mark.parametrize("side", ["recorded", "requested"])
    def test_an_external_dem_without_a_name(self, tmp_path: Path, side: str) -> None:
        external = RTCConfig(dem=tmp_path / "dem.tif")
        made, requested = (
            (external, RTCConfig()) if side == "recorded" else (RTCConfig(), external)
        )
        path = _made_with(tmp_path, _rtc(), made)

        assert _check(path, _rtc(), requested).unchecked == "unknown_dem"

    def test_a_named_external_dem_is_compared(self, tmp_path: Path) -> None:
        named = RTCConfig(dem=tmp_path / "dem.tif", dem_name="GLO-30 prepared")
        path = _made_with(tmp_path, _rtc(), named)

        assert _check(path, _rtc(), named).unchecked is None
        with pytest.raises(ProductSettingsMismatchError, match="DEM"):
            _check(path, _rtc(), RTCConfig())

    def test_an_unknown_dem_does_not_hide_another_difference(self, tmp_path: Path) -> None:
        path = _made_with(tmp_path, _rtc(), RTCConfig(dem=tmp_path / "dem.tif", grid_spacing=20))

        with pytest.raises(ProductSettingsMismatchError) as excinfo:
            _check(path, _rtc(), RTCConfig(dem=tmp_path / "dem.tif"))

        assert set(excinfo.value.differences) == {"GRID_SPACING"}


class TestSnapVersion:
    def test_another_snap_version_is_counted_not_raised(self, tmp_path: Path) -> None:
        path = _made_with(tmp_path, _rtc(), RTCConfig(), snap_version=(13, 0, 0))

        skip = _check(path, _rtc(), RTCConfig(), snap_version=_SNAP_14)

        assert skip == _BurstSkip(other_snap_version="13.0.0")

    @pytest.mark.parametrize(
        ("made", "running"), [((13, 0, 0), None), (None, _SNAP_14)], ids=["running", "recorded"]
    )
    def test_an_unknown_version_is_not_compared(
        self,
        tmp_path: Path,
        made: tuple[int, ...] | None,
        running: tuple[int, ...] | None,
    ) -> None:
        path = _made_with(tmp_path, _rtc(), RTCConfig(), snap_version=made)

        assert _check(path, _rtc(), RTCConfig(), snap_version=running).other_snap_version is None


class TestWarnings:
    def test_each_kind_names_its_count(self) -> None:
        with pytest.warns(UserWarning, match="without checking") as unchecked:
            _warn_unchecked(3, stacklevel=1)
        with pytest.warns(UserWarning, match="another SNAP version") as other:
            _warn_other_snap(2, "14.0.0", stacklevel=1)

        messages = [str(record[0].message) for record in (unchecked, other)]
        assert "Skipped 3 existing product(s) without checking" in messages[0]
        assert "dem_name" in messages[0]
        assert "Skipped 2 existing product(s) made by another SNAP version" in messages[1]
        assert "running 14.0.0" in messages[1]

    def test_no_skips_stay_quiet(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_unchecked(0, stacklevel=1)
            _warn_other_snap(0, "14.0.0", stacklevel=1)


class TestError:
    def test_is_neither_item_local_nor_per_cell(self) -> None:
        error = ProductSettingsMismatchError("differs", path=Path("x.tif"), differences={})

        assert isinstance(error, S1BurstsError)
        assert not isinstance(error, BurstError | CompositeError | OSError | ValueError)

    def test_defaults_to_no_path_and_no_differences(self) -> None:
        error = ProductSettingsMismatchError("differs")

        assert (error.path, error.differences) == (None, {})
