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
    ConfigurationError,
    ProductSettingsMismatchError,
    S1BurstsError,
)
from s1bursts.product import CoherenceProduct, GTCProduct, RTCProduct
from s1bursts.product_tags import _SETTING_TAGS
from s1bursts.settings_check import (
    _check_burst_product,
    _warn_other_snap,
)
from s1bursts.snap_graph import CoherenceConfig, GTCConfig, RTCConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from s1bursts.snap_graph import BackscatterConfig

type _BurstProduct = RTCProduct | GTCProduct | CoherenceProduct
type _ConfigClass = type[RTCConfig | GTCConfig | CoherenceConfig]

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


def _gtc() -> GTCProduct:
    return GTCProduct(
        burst_id=30346,
        sub_swath="IW1",
        crs=_UTM32,
        date=Timestamp("2022-01-05"),
        polarization="VV",
        pass_direction="Ascending",
        _rel_orbit_number=66,
    )


def _coherence() -> CoherenceProduct:
    return CoherenceProduct(
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
    product: _BurstProduct,
    config: BackscatterConfig | CoherenceConfig,
    snap_version: tuple[int, ...] | None = _SNAP_14,
) -> Path:
    """Write the file ``config`` would have produced, tags included."""
    metadata = product_tags.burst_metadata(product, config, snap_version=snap_version)
    return _write(tmp_path / "product.tif", metadata.tags)


def _check(
    path: Path,
    product: _BurstProduct,
    config: BackscatterConfig | CoherenceConfig,
    snap_version: tuple[int, ...] | None = _SNAP_14,
) -> str | None:
    return _check_burst_product(path, product, config, snap_version=snap_version)


# Another value for every config field that a tag records, so that a new field
# fails `test_every_field_is_recorded_and_compared` until it is given a tag.
_OTHER_VALUES: dict[str, object] = {
    "grid_spacing": 20.0,
    "dem": "SRTM 1Sec HGT",
    "img_resampling": "NEAREST_NEIGHBOUR",
    "dem_resampling": "CUBIC_CONVOLUTION",
    "align_to_standard_grid": False,
    "external_dem_nodata": -9999.0,
    "external_dem_apply_egm": True,
    "replace_zeros_with_nan": False,
    "multilook": (3, 3),
    "dem_oversampling": 2.0,
    "additional_overlap": 0.5,
    "window": (10, 2),
    "subtract_flat_earth_phase": False,
    "subtract_topographic_phase": False,
    "correct_coherence_bias": True,
    "remove_local_phase_ramp": True,
}
# Recorded through DEM, and computed from grid_spacing.
_NOT_TAGGED = {"dem_name", "standard_grid_origin"}
# Written only for an external DEM, so compared against one.
_EXTERNAL_DEM_ONLY = {"external_dem_nodata", "external_dem_apply_egm"}
_EXTERNAL_DEM: dict[str, object] = {"dem": Path("/data/dem.tif"), "dem_name": "GLO-30 prepared"}

_CONFIGS: list[tuple[_ConfigClass, Callable[[], _BurstProduct]]] = [
    (RTCConfig, _rtc),
    (GTCConfig, _gtc),
    (CoherenceConfig, _coherence),
]


def _fields(config_cls: _ConfigClass) -> set[str]:
    return {*config_cls.model_fields, *config_cls.model_computed_fields} - _NOT_TAGGED


class TestEveryField:
    @pytest.mark.parametrize("config_cls", [cls for cls, _ in _CONFIGS], ids=lambda c: c.__name__)
    def test_every_field_has_a_tag(self, config_cls: _ConfigClass) -> None:
        missing = _fields(config_cls) - set(_SETTING_TAGS)

        assert not missing, f"{config_cls.__name__} fields without a tag: {sorted(missing)}"

    @pytest.mark.parametrize(
        ("config_cls", "product", "field"),
        [
            pytest.param(cls, product, field, id=f"{cls.__name__}.{field}")
            for cls, product in _CONFIGS
            for field in sorted(_fields(cls))
        ],
    )
    def test_every_field_is_recorded_and_compared(
        self,
        tmp_path: Path,
        config_cls: _ConfigClass,
        product: Callable[[], _BurstProduct],
        field: str,
    ) -> None:
        burst_product = product()
        base = _EXTERNAL_DEM if field in _EXTERNAL_DEM_ONLY else {}
        path = _made_with(tmp_path, burst_product, config_cls.model_validate(base))

        with pytest.raises(ProductSettingsMismatchError) as excinfo:
            _check(
                path,
                burst_product,
                config_cls.model_validate({**base, field: _OTHER_VALUES[field]}),
            )

        assert set(excinfo.value.differences) == set(_SETTING_TAGS[field])


class TestBurstProducts:
    def test_matching_settings_pass(self, tmp_path: Path) -> None:
        path = _made_with(tmp_path, _rtc(), RTCConfig())

        assert _check(path, _rtc(), RTCConfig()) is None

    def test_another_grid_spacing_raises_naming_both_values(self, tmp_path: Path) -> None:
        path = _made_with(tmp_path, _rtc(), RTCConfig(grid_spacing=20.0))

        with pytest.raises(ProductSettingsMismatchError) as excinfo:
            _check(path, _rtc(), RTCConfig())

        assert excinfo.value.path == path
        assert excinfo.value.differences == {"GRID_SPACING": ("20 m", "10 m")}
        message = str(excinfo.value)
        assert "GRID_SPACING (grid_spacing): recorded '20 m', requested '10 m'" in message
        assert "overwrite=True" in message

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
            pytest.param(
                RTCConfig(dem_oversampling=2.0),
                RTCConfig(),
                {"TERRAIN_FLATTENING_OVERSAMPLING": ("2", "1")},
                id="oversampling",
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
        product = _coherence()
        path = _made_with(tmp_path, product, CoherenceConfig(window=(10, 2)))

        with pytest.raises(ProductSettingsMismatchError, match=r"\(window\)") as excinfo:
            _check(path, product, CoherenceConfig())

        assert excinfo.value.differences == {
            "COHERENCE_WINDOW_RANGE": ("10", "25"),
            "COHERENCE_WINDOW_AZIMUTH": ("2", "5"),
        }

    def test_a_named_external_dem_is_compared(self, tmp_path: Path) -> None:
        named = RTCConfig(dem=tmp_path / "dem.tif", dem_name="GLO-30 prepared")
        path = _made_with(tmp_path, _rtc(), named)

        assert _check(path, _rtc(), named) is None
        with pytest.raises(ProductSettingsMismatchError, match=r"\(dem, dem_name\)") as excinfo:
            _check(path, _rtc(), RTCConfig())

        assert excinfo.value.differences["DEM"] == (
            "GLO-30 prepared",
            "Copernicus 30m Global DEM",
        )


class TestMissingTags:
    """A product that does not record a requested setting raises."""

    # Previously skipped as "untagged" without comparing anything.
    def test_a_file_written_before_the_tags_existed(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "old.tif", {"AREA_OR_POINT": "Area"})

        with pytest.raises(ProductSettingsMismatchError, match="overwrite=True") as excinfo:
            _check(path, _rtc(), RTCConfig())

        differences = excinfo.value.differences
        assert set(differences) >= {"GRID_SPACING", "DEM", "IMAGE_RESAMPLING"}
        assert all(recorded is None for recorded, _ in differences.values())

    # Previously skipped as matching: only GRID_SPACING, DEM, multilook and
    # window were compared.
    def test_a_file_recording_only_some_settings(self, tmp_path: Path) -> None:
        tags = product_tags.burst_metadata(_rtc(), RTCConfig(), snap_version=_SNAP_14).tags
        del tags["IMAGE_RESAMPLING"]
        path = _write(tmp_path / "older.tif", tags)

        with pytest.raises(ProductSettingsMismatchError) as excinfo:
            _check(path, _rtc(), RTCConfig())

        assert excinfo.value.differences == {"IMAGE_RESAMPLING": (None, "BILINEAR_INTERPOLATION")}

    # Previously skipped as "unknown_dem" without comparing the DEM.
    def test_a_dem_recorded_as_unknown(self, tmp_path: Path) -> None:
        tags = product_tags.burst_metadata(_rtc(), RTCConfig(), snap_version=_SNAP_14).tags
        tags["DEM"] = product_tags.UNKNOWN
        path = _write(tmp_path / "unknown.tif", tags)

        with pytest.raises(ProductSettingsMismatchError) as excinfo:
            _check(path, _rtc(), RTCConfig())

        assert excinfo.value.differences == {"DEM": ("N/A", "Copernicus 30m Global DEM")}

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


class TestSnapVersion:
    def test_another_snap_version_is_counted_not_raised(self, tmp_path: Path) -> None:
        path = _made_with(tmp_path, _rtc(), RTCConfig(), snap_version=(13, 0, 0))

        assert _check(path, _rtc(), RTCConfig(), snap_version=_SNAP_14) == "13.0.0"

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

        assert _check(path, _rtc(), RTCConfig(), snap_version=running) is None

    def test_snap_13_coherence_matches_snap_14_with_the_options_off(self, tmp_path: Path) -> None:
        product = _coherence()
        path = _made_with(tmp_path, product, CoherenceConfig(), snap_version=(13, 0, 0))

        assert _check(path, product, CoherenceConfig(), snap_version=_SNAP_14) == "13.0.0"

    def test_snap_13_coherence_differs_from_snap_14_with_an_option_on(
        self, tmp_path: Path
    ) -> None:
        product = _coherence()
        path = _made_with(tmp_path, product, CoherenceConfig(), snap_version=(13, 0, 0))

        with pytest.raises(ProductSettingsMismatchError) as excinfo:
            _check(path, product, CoherenceConfig(correct_coherence_bias=True))

        assert excinfo.value.differences == {"COHERENCE_BIAS_CORRECTED": ("false", "true")}

    def test_an_unknown_version_cannot_confirm_the_snap_14_options(self, tmp_path: Path) -> None:
        product = _coherence()
        path = _made_with(tmp_path, product, CoherenceConfig())

        with pytest.raises(ProductSettingsMismatchError) as excinfo:
            _check(path, product, CoherenceConfig(), snap_version=None)

        assert excinfo.value.differences == {
            "COHERENCE_BIAS_CORRECTED": ("false", "N/A"),
            "LOCAL_PHASE_RAMP_REMOVED": ("false", "N/A"),
        }

    def test_an_option_snap_cannot_apply_raises_as_the_graph_would(self, tmp_path: Path) -> None:
        product = _coherence()
        path = _made_with(tmp_path, product, CoherenceConfig(), snap_version=None)

        with pytest.raises(ConfigurationError, match="SNAP 14"):
            _check(path, product, CoherenceConfig(remove_local_phase_ramp=True), snap_version=None)


class TestWarnings:
    def test_names_its_count(self) -> None:
        with pytest.warns(UserWarning, match="another SNAP version") as other:
            _warn_other_snap(2, "14.0.0", stacklevel=1)

        message = str(other[0].message)
        assert "Skipped 2 existing product(s) made by another SNAP version" in message
        assert "running 14.0.0" in message

    def test_no_skips_stay_quiet(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_other_snap(0, "14.0.0", stacklevel=1)


class TestError:
    def test_is_neither_item_local_nor_per_cell(self) -> None:
        error = ProductSettingsMismatchError("differs", path=Path("x.tif"), differences={})

        assert isinstance(error, S1BurstsError)
        assert not isinstance(error, BurstError | CompositeError | OSError | ValueError)

    def test_defaults_to_no_path_and_no_differences(self) -> None:
        error = ProductSettingsMismatchError("differs")

        assert (error.path, error.differences) == (None, {})
