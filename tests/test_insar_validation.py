"""Validate SNAP coherence against ASF HyP3 burst InSAR.

See ``fixtures/REFERENCE.md`` for reference details and run requirements.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

from s1bursts.config import get_settings
from s1bursts.errors import ConfigurationError
from s1bursts.snap_client import _resolve_gpt_path
from tests.insar_validation import (
    BLOCK,
    GRID_SPACING,
    PATTERNED_SCENES,
    SATURATED_SCENES,
    SCENES,
    Scene,
    _erode,
    agreement,
    alignment_scan,
)
from tests.validation_toolchain import workdir

if TYPE_CHECKING:
    from collections.abc import Callable

# Contract bounds. Deliberately looser than the measured values: they state what
# must be true of any correct build, not what this particular SNAP produced.
# They are loose in a second sense too, because the two products do not estimate
# coherence the same way -- see `tests/insar_validation` for what differs.
MIN_RANK_CORRELATION = 0.60
MAX_QUANTILE_OFFSET = 0.45
MAX_ABS_MEDIAN_OFFSET = 0.30
MIN_COHERENT_MEDIAN = 0.60
# Stated in metres, not pixels, because it is a claim about the ground: half a
# resolution cell of either estimator, at a posting of 20 m.
MAX_GEOLOCATION_SHIFT_M = 30.0

# What each scene's declared role commits it to. ISCE2's estimator saturates
# near 1 over uniformly coherent ground, so which claims a burst can carry
# depends on the reference; `Scene.reference_has_pattern` records that, and
# these bounds hold the declaration to the measurement. A scene that stops
# behaving as declared fails, rather than excusing itself from its claims.
MIN_REFERENCE_SPREAD = 0.20
MIN_PEAK_CONTRAST = 0.01
HIGH_REFERENCE_MEDIAN = 0.90

BLOCK_KEY = f"block_{BLOCK}x{BLOCK}_{int(BLOCK * GRID_SPACING)}m"


def _settings_have_cdse() -> bool:
    settings = get_settings()
    return settings.cdse.username is not None and settings.cdse.password is not None


def _gpt_available() -> bool:
    try:
        _resolve_gpt_path(None, get_settings().snap)
    except ConfigurationError:
        return False
    return True


def _earthdata_credentials() -> bool:
    return bool(os.environ.get("EOSDIS_USERNAME") and os.environ.get("EOSDIS_PASSWORD"))


requires_prerequisites = pytest.mark.skipif(
    not (_settings_have_cdse() and _gpt_available() and _earthdata_credentials()),
    reason="needs CDSE credentials, Earthdata credentials, and a resolvable SNAP `gpt`",
)


@pytest.fixture(scope="session")
def campaign(tmp_path_factory: pytest.TempPathFactory) -> Callable[[Scene], dict]:
    """Run each scene comparison once per test session."""
    for module in ("hyp3_sdk", "asf_search"):
        pytest.importorskip(module, reason="install the `validation` dependency group")
    # Imported lazily so that collecting this module does not require the
    # optional `validation` dependency group.
    from tests.fixtures.regenerate_insar_reference import (  # noqa: PLC0415
        NoReferenceJobError,
        compare_scene,
        login,
    )

    work = workdir(tmp_path_factory, "insar-validation")
    hyp3 = login()
    allow_submit = os.environ.get("S1BURSTS_HYP3_SUBMIT") == "1"
    cache: dict[str, dict] = {}

    def run(scene: Scene) -> dict:
        if scene.name not in cache:
            try:
                cache[scene.name] = compare_scene(
                    scene,
                    work,
                    hyp3,
                    allow_submit=allow_submit,
                )
            except NoReferenceJobError as error:
                pytest.skip(str(error))
        return cache[scene.name]

    return run


def _scene_ids(scenes: tuple[Scene, ...]) -> list[str]:
    return [scene.name for scene in scenes]


@pytest.mark.validation
@requires_prerequisites
@pytest.mark.parametrize("scene", SCENES, ids=_scene_ids(SCENES))
class TestCoherenceContract:
    """Claims about coherence that hold on any burst, whatever the reference sees."""

    def test_the_offset_between_the_estimators_has_the_expected_sign_and_size(
        self,
        scene: Scene,
        campaign: Callable[[Scene], dict],
    ) -> None:
        """Check that HyP3 reads higher at each quantile, within the limit."""
        quantiles = campaign(scene)["quantiles"]

        offenders = {
            quantile: entry["diff_hyp3_minus_ours"]
            for quantile, entry in quantiles.items()
            if not 0.0 < entry["diff_hyp3_minus_ours"] < MAX_QUANTILE_OFFSET
        }
        assert not offenders, (
            f"{scene.name}: offsets outside (0, {MAX_QUANTILE_OFFSET}): {offenders}"
        )

    def test_the_median_offset_stays_bounded(
        self,
        scene: Scene,
        campaign: Callable[[Scene], dict],
    ) -> None:
        """Scene-wide, the two products must describe the same burst."""
        stats = campaign(scene)[BLOCK_KEY]

        assert abs(stats["median_diff_hyp3_minus_ours"]) < MAX_ABS_MEDIAN_OFFSET, (
            f"{scene.name}: median coherence {stats['median_ours']} against "
            f"HyP3's {stats['median_hyp3']}"
        )


@pytest.mark.validation
@requires_prerequisites
@pytest.mark.parametrize("scene", PATTERNED_SCENES, ids=_scene_ids(PATTERNED_SCENES))
class TestPatternContract:
    """Claims for scenes with a varying HyP3 reference."""

    def test_the_reference_varies_across_the_burst(
        self,
        scene: Scene,
        campaign: Callable[[Scene], dict],
    ) -> None:
        """The premise the other claims in this class rest on."""
        spread = campaign(scene)["reference_spread"]

        assert spread > MIN_REFERENCE_SPREAD, (
            f"{scene.name} is declared to have a varying reference, but HyP3's "
            f"coherence spans only {spread} between its 10th and 90th percentiles; "
            "the ranking and geolocation claims below cannot be made on it"
        )

    def test_coherence_ranks_the_scene_like_isce2(
        self,
        scene: Scene,
        campaign: Callable[[Scene], dict],
    ) -> None:
        """Check that both processors rank the burst the same way."""
        stats = campaign(scene)[BLOCK_KEY]

        assert stats["spearman"] > MIN_RANK_CORRELATION, (
            f"{scene.name}: rank correlation {stats['spearman']} below "
            f"{MIN_RANK_CORRELATION}; the two processors disagree about where "
            "the scene is coherent"
        )

    def test_the_two_products_sit_on_the_same_ground(
        self,
        scene: Scene,
        campaign: Callable[[Scene], dict],
    ) -> None:
        """Check that the products align on the ground."""
        geolocation = campaign(scene)["geolocation"]

        assert not geolocation["peak_on_boundary"], (
            f"{scene.name}: the correlation peaks at {geolocation['best_shift_px']} px, on the "
            f"edge of a +/-{geolocation['max_shift_px']} px search, so the two products are "
            "offset by more than the search is wide"
        )
        assert geolocation["peak_contrast"] > MIN_PEAK_CONTRAST, (
            f"{scene.name}: the correlation peak stands only "
            f"{geolocation['peak_contrast']} above the edge of the search, on a burst whose "
            "reference does carry a pattern to lock onto"
        )

        rows, cols = geolocation["subpixel_shift_m"]
        assert max(abs(rows), abs(cols)) < MAX_GEOLOCATION_SHIFT_M, (
            f"{scene.name}: the products are offset by "
            f"{geolocation['subpixel_shift_m']} m (row, column)"
        )


@pytest.mark.validation
@requires_prerequisites
@pytest.mark.parametrize("scene", SATURATED_SCENES, ids=_scene_ids(SATURATED_SCENES))
class TestSaturatedReferenceContract:
    """The claim a burst can carry when its reference reports coherence everywhere."""

    def test_a_scene_the_reference_calls_coherent_reads_coherent(
        self,
        scene: Scene,
        campaign: Callable[[Scene], dict],
    ) -> None:
        """Check that this package also finds the burst coherent."""
        results = campaign(scene)
        reference_median = results["per_pixel"]["median_hyp3"]

        assert reference_median > HIGH_REFERENCE_MEDIAN, (
            f"{scene.name} is declared to have a saturated reference, but HyP3's median "
            f"coherence is {reference_median}; this burst no longer exercises the coherent end"
        )
        assert results["per_pixel"]["median_ours"] > MIN_COHERENT_MEDIAN, (
            f"{scene.name}: median coherence {results['per_pixel']['median_ours']} "
            f"where HyP3 reports {reference_median}"
        )


class TestComparisonHelpers:
    """The comparison machinery itself, which the claims above depend on."""

    @staticmethod
    def _coherence_field(rng: np.random.Generator, noise: float = 0.02) -> np.ndarray:
        """Build a coherence-like field: patches of land cover, plus estimator noise."""
        patches = rng.uniform(0.15, 0.9, size=(12, 12))
        field = np.kron(patches, np.ones((8, 8)))
        return np.clip(field + rng.normal(0.0, noise, field.shape), 0.01, 1.0)

    def test_alignment_scan_finds_no_shift_between_aligned_fields(self) -> None:
        """Two noisy views of one scene, in place: the peak must sit at zero."""
        rng = np.random.default_rng(0)
        truth = self._coherence_field(rng, noise=0.0)
        ours = np.clip(truth + rng.normal(0.0, 0.03, truth.shape), 0.01, 1.0)
        theirs = np.clip(truth + rng.normal(0.0, 0.03, truth.shape), 0.01, 1.0)

        result = alignment_scan(ours, theirs, np.ones(truth.shape, dtype=bool))

        assert result["best_shift_px"] == [0, 0]
        assert max(abs(value) for value in result["subpixel_shift_px"]) < 0.2

    def test_alignment_scan_recovers_a_known_shift(self) -> None:
        """Check that the scan finds a one-pixel offset."""
        rng = np.random.default_rng(1)
        truth = self._coherence_field(rng, noise=0.0)
        ours = np.clip(truth + rng.normal(0.0, 0.03, truth.shape), 0.01, 1.0)
        # theirs[row] holds the ground ours has at row - 1.
        theirs = np.roll(ours, shift=1, axis=0)

        result = alignment_scan(ours, theirs, np.ones(truth.shape, dtype=bool))

        assert result["best_shift_px"] == [-1, 0]
        assert abs(result["subpixel_shift_px"][0] + 1.0) < 0.2
        assert abs(result["subpixel_shift_m"][0] + GRID_SPACING) < 0.2 * GRID_SPACING

    def test_rank_correlation_survives_a_change_of_estimator(self) -> None:
        """Check that rank survives a monotonic change in values."""
        rng = np.random.default_rng(2)
        ours = self._coherence_field(rng, noise=0.0)
        theirs = ours**3
        valid = np.ones(ours.shape, dtype=bool)

        stats = agreement(ours, theirs, valid)

        assert stats["spearman"] > 0.99
        assert stats["pearson"] < stats["spearman"]

    def test_erosion_drops_the_edge_of_the_valid_area(self) -> None:
        """Edge pixels are contaminated on both sides and must not be compared."""
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True

        eroded = _erode(mask, 2)

        assert eroded[7:13, 7:13].all()
        assert eroded.sum() == 6 * 6
