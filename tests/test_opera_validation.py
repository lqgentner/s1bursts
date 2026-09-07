"""Validate SNAP products against OPERA RTC-S1.

Deselected by default; run with::

    uv sync --group validation
    uv run pytest -m validation

See ``fixtures/REFERENCE.md`` for reference details and run requirements.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from s1bursts.config import get_settings
from s1bursts.errors import ConfigurationError
from s1bursts.snap_client import _resolve_gpt_path
from tests.opera_validation import (
    BLOCK,
    GRID_SPACING,
    NOISE_FLOOR_PERCENTILE,
    SCENES,
    Scene,
    agreement,
)
from tests.raster_comparison import read_aligned
from tests.validation_toolchain import workdir

if TYPE_CHECKING:
    from collections.abc import Callable

# Contract bounds. Deliberately looser than the measured values: they state what
# must be true of any correct build, not what this particular SNAP produced.
MAX_ABS_GAMMA0_BIAS_DB = 0.25
MIN_GAMMA0_LOG_CORRELATION = 0.95
MAX_ABS_LOW_BACKSCATTER_OFFSET_DB = 4.0
MAX_ANF_RATIO_DEVIATION = 0.05
MIN_ANF_LOG_CORRELATION = 0.95


def _settings_have_cdse() -> bool:
    settings = get_settings()
    return settings.cdse.username is not None and settings.cdse.password is not None


def _gpt_available() -> bool:
    try:
        _resolve_gpt_path(None, get_settings().snap)
    except ConfigurationError:
        return False
    return True


requires_prerequisites = pytest.mark.skipif(
    not (_settings_have_cdse() and _gpt_available()),
    reason="needs CDSE credentials and a resolvable SNAP `gpt`",
)


@pytest.fixture(scope="session")
def campaign(tmp_path_factory: pytest.TempPathFactory) -> Callable[[Scene], dict[str, Any]]:
    """Run each scene comparison once per test session."""
    pytest.importorskip("asf_search", reason="install the `validation` dependency group")
    # Imported lazily so that collecting this module does not require the
    # optional `validation` dependency group.
    from tests.fixtures.regenerate_opera_reference import compare_scene, login  # noqa: PLC0415

    work = workdir(tmp_path_factory, "opera-validation")
    session = login()
    cache: dict[str, dict[str, Any]] = {}

    def run(scene: Scene) -> dict[str, Any]:
        if scene.name not in cache:
            cache[scene.name] = compare_scene(scene, work, session)
        return cache[scene.name]

    return run


def _scene_ids(scenes: tuple[Scene, ...]) -> list[str]:
    return [scene.name for scene in scenes]


def _write_raster(
    path: Path,
    origin: tuple[float, float],
    size: int,
    values: np.ndarray | None = None,
) -> Path:
    """Write a square 10 m raster on the UTM grid, for the helper tests below."""
    profile = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32632",
        "transform": from_origin(origin[0], origin[1], 10.0, 10.0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.ones((size, size), dtype="float32") if values is None else values, 1)
    return path


@pytest.mark.validation
@requires_prerequisites
@pytest.mark.parametrize("scene", SCENES, ids=_scene_ids(SCENES))
class TestRadiometricContract:
    """Claims about gamma0 that any correct build must satisfy."""

    def test_gamma0_bias_is_within_a_quarter_db_of_opera(
        self,
        scene: Scene,
        campaign: Callable[[Scene], dict[str, Any]],
    ) -> None:
        """Check block-averaged gamma-0 bias against OPERA."""
        results = campaign(scene)
        block_key = f"block_{BLOCK}x{BLOCK}_{int(BLOCK * GRID_SPACING)}m"

        assert results["gamma0"], f"{scene.name}: no polarization was compared"
        offenders = {
            polarization: entry[block_key]["median_dB_bias"]
            for polarization, entry in results["gamma0"].items()
            if abs(entry[block_key]["median_dB_bias"]) > MAX_ABS_GAMMA0_BIAS_DB
        }
        assert not offenders, (
            f"{scene.name}: gamma0 bias exceeds {MAX_ABS_GAMMA0_BIAS_DB} dB: {offenders}"
        )

    def test_low_backscatter_offset_stays_bounded(
        self,
        scene: Scene,
        campaign: Callable[[Scene], dict[str, Any]],
    ) -> None:
        """Check the low-backscatter offset against its limit."""
        results = campaign(scene)
        key = f"p{int(NOISE_FLOOR_PERCENTILE):02d}_dB_offset"

        assert results["gamma0"], f"{scene.name}: no polarization was compared"
        offenders = {
            polarization: entry[key]
            for polarization, entry in results["gamma0"].items()
            if abs(entry[key]) > MAX_ABS_LOW_BACKSCATTER_OFFSET_DB
        }
        assert not offenders, (
            f"{scene.name}: low-backscatter offset exceeds "
            f"{MAX_ABS_LOW_BACKSCATTER_OFFSET_DB} dB: {offenders}"
        )

    def test_gamma0_tracks_opera_spatially(
        self,
        scene: Scene,
        campaign: Callable[[Scene], dict[str, Any]],
    ) -> None:
        """A correct geocoding reproduces OPERA's spatial pattern, not just its mean."""
        results = campaign(scene)
        block_key = f"block_{BLOCK}x{BLOCK}_{int(BLOCK * GRID_SPACING)}m"

        assert results["gamma0"], f"{scene.name}: no polarization was compared"
        offenders = {
            polarization: entry[block_key]["corr_log"]
            for polarization, entry in results["gamma0"].items()
            if entry[block_key]["corr_log"] < MIN_GAMMA0_LOG_CORRELATION
        }
        assert not offenders, (
            f"{scene.name}: log-correlation below {MIN_GAMMA0_LOG_CORRELATION}: {offenders}"
        )


@pytest.mark.validation
@requires_prerequisites
@pytest.mark.parametrize("scene", SCENES, ids=_scene_ids(SCENES))
class TestGeometryContract:
    """Claims about the area normalization factor, independent of radiometry."""

    def test_simulated_image_matches_operas_area_normalization_factor(
        self,
        scene: Scene,
        campaign: Callable[[Scene], dict[str, Any]],
    ) -> None:
        """Compare the local scattering area (SNAP's ``simulatedImage``) with OPERA's ANF."""
        stats = campaign(scene)["anf"]["layover_shadow_excluded"]

        assert abs(stats["median_ratio"] - 1.0) < MAX_ANF_RATIO_DEVIATION, (
            f"{scene.name}: ANF median ratio {stats['median_ratio']} deviates from unity"
        )
        assert stats["corr_log"] > MIN_ANF_LOG_CORRELATION, (
            f"{scene.name}: ANF log-correlation {stats['corr_log']} too low"
        )


class TestComparisonHelpers:
    """The comparison machinery itself, which the claims above depend on."""

    def test_misaligned_grids_are_rejected_rather_than_silently_compared(
        self,
        tmp_path: Path,
    ) -> None:
        """Half-pixel offsets must raise: this is the bug that invalidates a campaign."""

        def write(path: Path, origin_x: float) -> None:
            profile = {
                "driver": "GTiff",
                "height": 8,
                "width": 8,
                "count": 1,
                "dtype": "float32",
                "crs": "EPSG:32632",
                "transform": from_origin(origin_x, 100.0, 10.0, 10.0),
            }
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(np.ones((8, 8), dtype="float32"), 1)

        aligned, offset = tmp_path / "a.tif", tmp_path / "b.tif"
        write(aligned, 0.0)
        write(offset, 5.0)  # half a pixel

        with pytest.raises(ValueError, match="not on a common grid"):
            read_aligned(aligned, offset, "whatever")

    def test_a_companion_is_windowed_with_the_pair_it_describes(self, tmp_path: Path) -> None:
        """A mask on its own extent would index different ground than the data.

        OPERA's layover/shadow mask need not share bounds with the gamma0 layer
        it masks, and a mask applied one pixel out is the very error these
        comparisons exist to detect.
        """
        ours = _write_raster(tmp_path / "ours.tif", origin=(0.0, 100.0), size=8)
        theirs = _write_raster(tmp_path / "theirs.tif", origin=(20.0, 80.0), size=8)
        # Deliberately a third extent, offset from both, with a single flagged
        # pixel at the north-west corner of the window the pair shares.
        flag = np.zeros((9, 9), dtype="float32")
        flag[1, 1] = 1.0
        mask = _write_raster(tmp_path / "mask.tif", origin=(10.0, 90.0), size=9, values=flag)

        array_a, array_b, (companion,), _ = read_aligned(ours, theirs, "whatever", (mask,))

        assert array_a.shape == array_b.shape == companion.shape == (6, 6)
        assert companion[0, 0] == 1.0
        assert companion.sum() == 1.0

    def test_a_companion_that_falls_short_of_the_window_is_rejected(self, tmp_path: Path) -> None:
        """Rasterio clips a window to the dataset instead of refusing it."""
        ours = _write_raster(tmp_path / "ours.tif", origin=(0.0, 100.0), size=8)
        theirs = _write_raster(tmp_path / "theirs.tif", origin=(0.0, 100.0), size=8)
        short = _write_raster(tmp_path / "short.tif", origin=(0.0, 100.0), size=4)

        with pytest.raises(ValueError, match="does not cover the extent"):
            read_aligned(ours, theirs, "whatever", (short,))

    def test_block_averaging_suppresses_speckle_without_moving_the_bias(self) -> None:
        """Check that block averaging preserves a known bias."""
        rng = np.random.default_rng(0)
        patches = rng.uniform(0.05, 0.5, size=(10, 10))
        truth = np.kron(patches, np.ones((20, 20)))
        theirs = truth * rng.gamma(shape=1.0, scale=1.0, size=truth.shape)
        ours = truth * 1.05 * rng.gamma(shape=1.0, scale=1.0, size=truth.shape)
        valid = np.ones(truth.shape, dtype=bool)

        per_pixel = agreement(ours, theirs, valid, k=1)
        blocked = agreement(ours, theirs, valid, k=10)

        # Speckle dominates per pixel and averages away over blocks.
        assert per_pixel["corr_log"] < 0.5
        assert blocked["corr_log"] > 0.95
        # The offset survives the averaging.
        assert abs(blocked["median_ratio"] - 1.05) < 0.05
