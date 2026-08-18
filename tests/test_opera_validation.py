"""Scientific cross-validation against OPERA RTC-S1 (ISCE3).

Deselected by default; run with::

    uv sync --group validation
    uv run pytest -m validation

These tests are slow and heavy: each scene downloads Sentinel-1 bursts from
CDSE and OPERA granules from ASF DAAC, then runs SNAP GPT. Run them per
release, not per commit.

They assert only absolute bounds -- what must be true of any correct build --
rather than reproduction of previously measured numbers. Because SNAP is
bit-reproducible, the product goldens already detect any change to the output,
offline and immediately; repeating that check here would cost two downloads and
two GPT runs to learn nothing new. The measured values live in
``fixtures/opera_reference.json`` as evidence, not as a test input; regenerate
them with ``fixtures/regenerate_opera_reference.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
    read_aligned,
)

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
def campaign(tmp_path_factory: pytest.TempPathFactory) -> Callable[[Scene], dict]:
    """Run (and cache) the comparison for a scene.

    Session-scoped because a scene costs several burst downloads and SNAP
    runs; every test for that scene must share the one result.
    """
    earthaccess = pytest.importorskip(
        "earthaccess",
        reason="install the `validation` dependency group",
    )
    # Imported lazily so that collecting this module does not require the
    # optional `validation` dependency group.
    from tests.fixtures.regenerate_opera_reference import compare_scene  # noqa: PLC0415

    work = tmp_path_factory.mktemp("opera-validation")
    cache: dict[str, dict] = {}

    def run(scene: Scene) -> dict:
        if scene.name not in cache:
            cache[scene.name] = compare_scene(scene, work, earthaccess=earthaccess)
        return cache[scene.name]

    return run


def _scene_ids(scenes: tuple[Scene, ...]) -> list[str]:
    return [scene.name for scene in scenes]


@pytest.mark.validation
@requires_prerequisites
@pytest.mark.parametrize("scene", SCENES, ids=_scene_ids(SCENES))
class TestRadiometricContract:
    """Claims about gamma0 that any correct build must satisfy."""

    def test_gamma0_bias_is_within_a_quarter_db_of_opera(
        self,
        scene: Scene,
        campaign: Callable[[Scene], dict],
    ) -> None:
        """Block-averaged gamma0 must agree with ISCE3 to well under calibration accuracy.

        Compared on 150 m blocks: per-pixel scatter is speckle and differing
        effective resolution, and says nothing about radiometry.
        """
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
        campaign: Callable[[Scene], dict],
    ) -> None:
        """The two products diverge in the dark tail, but only by so much.

        OPERA clips its noise-corrected signal at zero while still single-looked,
        which biases its multilooked mean upward where the signal approaches the
        noise level. The offset therefore grows as a scene darkens, and this
        bounds how far. Levels are compared at a percentile rather than over
        selected pixels, so nothing is chosen on either image.
        """
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
        campaign: Callable[[Scene], dict],
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
        campaign: Callable[[Scene], dict],
    ) -> None:
        """SNAP's ``simulatedImage`` is the same quantity as OPERA's gamma0-to-beta0 ANF.

        This isolates terrain flattening and geocoding: the factor is pure
        geometry and DEM, with no calibration or speckle in it.
        """
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

    def test_block_averaging_suppresses_speckle_without_moving_the_bias(self) -> None:
        """The premise of the gamma0 comparison, on synthetic data.

        Two views of one scene differing by a known 5% offset, each with its own
        speckle. Averaging in power must recover the offset and the underlying
        spatial pattern. The scene is built from patches rather than white noise
        because real terrain is spatially correlated: averaging a white field
        would remove the signal along with the speckle.
        """
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
