"""Tests for the GeoTIFF metadata builders.

Fully offline: the builders take a product and the call arguments and return a
mapping, so nothing here touches a file or the burst-map LUT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pandas import Timestamp
from pyproj import CRS
import pytest

import s1bursts
from s1bursts import product_tags
from s1bursts.product import (
    RTCLRWQA,
    CoherenceProduct,
    CohLRWComposite,
    CohLRWQA,
    CohPassComposite,
    CohPassQA,
    GTCProduct,
    InterferogramProduct,
    QaMetric,
    RTCLRWComposite,
    RTCPassComposite,
    RTCPassQA,
    RTCProduct,
)
from s1bursts.product_tags import RasterMetadata
from s1bursts.snap_graph import BackscatterConfig, CoherenceConfig, InterferogramConfig

_UTM32 = CRS.from_epsg(32632)
_START = Timestamp("2020-01-01")
_END = Timestamp("2020-03-31")


def _rtc_lrw() -> RTCLRWComposite:
    return RTCLRWComposite(
        mgrs_code="32TMS03",
        start_date=_START,
        end_date=_END,
        polarization="VV",
    )


def _coh_lrw() -> CohLRWComposite:
    return CohLRWComposite(
        mgrs_code="32TMS03",
        start_date=_START,
        end_date=_END,
        polarization="VV",
        baseline_days=12,
    )


def _qa(metric: QaMetric) -> RTCLRWQA:
    return RTCLRWQA(
        mgrs_code="32TMS03",
        start_date=_START,
        end_date=_END,
        polarization="VV",
        qa_metric=metric,
    )


def _coh_qa(metric: QaMetric) -> CohLRWQA:
    return CohLRWQA(
        mgrs_code="32TMS03",
        start_date=_START,
        end_date=_END,
        polarization="VV",
        baseline_days=12,
        qa_metric=metric,
    )


def _rtc_pass() -> RTCPassComposite:
    return RTCPassComposite(
        mgrs_code="32TMS03",
        start_date=_START,
        end_date=_END,
        polarization="VV",
        orbit_direction="ASCENDING",
    )


def _coh_pass() -> CohPassComposite:
    return CohPassComposite(
        mgrs_code="32TMS03",
        start_date=_START,
        end_date=_END,
        polarization="VV",
        orbit_direction="DESCENDING",
        baseline_days=12,
    )


def _rtc_burst() -> RTCProduct:
    return RTCProduct(
        burst_id=30346,
        sub_swath="IW1",
        crs=_UTM32,
        date=Timestamp("2020-02-05"),
        polarization="VV",
        _rel_orbit_number=15,
    )


def _gtc_burst() -> GTCProduct:
    return GTCProduct(
        burst_id=30346,
        sub_swath="IW1",
        crs=_UTM32,
        date=Timestamp("2020-02-05"),
        polarization="VH",
        _rel_orbit_number=15,
    )


def _coh_burst() -> CoherenceProduct:
    return CoherenceProduct(
        burst_id=30346,
        sub_swath="IW1",
        crs=_UTM32,
        date_primary=Timestamp("2020-02-05"),
        date_secondary=Timestamp("2020-02-17"),
        polarization="VV",
        _rel_orbit_number=15,
    )


def _ifg_burst() -> InterferogramProduct:
    return InterferogramProduct(
        burst_id=30346,
        sub_swath="IW1",
        crs=_UTM32,
        date_primary=Timestamp("2020-02-05"),
        date_secondary=Timestamp("2020-02-17"),
        polarization="VV",
        _rel_orbit_number=15,
    )


def _lrw(**kwargs: object) -> dict[str, str]:
    defaults: dict = {
        "aggregation": "median",
        "clip_max_resolution": None,
        "n_observations": 7,
        "decibel": False,
    }
    return product_tags.rtc_lrw_metadata(_rtc_lrw(), **(defaults | kwargs)).tags  # type: ignore[arg-type]


def _every_builder() -> dict[str, dict[str, str]]:
    """Tags from one product of every kind this module builds metadata for."""
    composite = {"aggregation": "median", "clip_max_resolution": None, "n_observations": 3}
    unweighted = {"aggregation": "median", "n_observations": 3}
    return {
        "rtc_lrw": _lrw(),
        "coh_lrw": product_tags.coh_lrw_metadata(_coh_lrw(), **composite).tags,  # type: ignore[arg-type]
        "rtc_pass": product_tags.rtc_pass_metadata(_rtc_pass(), decibel=False, **unweighted).tags,  # type: ignore[arg-type]
        "coh_pass": product_tags.coh_pass_metadata(_coh_pass(), **unweighted).tags,  # type: ignore[arg-type]
        "qa_num": product_tags.qa_metadata(_qa(QaMetric.NUM), **composite).tags,  # type: ignore[arg-type]
        "qa_cqm": product_tags.qa_metadata(_qa(QaMetric.CQM), **composite).tags,  # type: ignore[arg-type]
        "coh_qa": product_tags.qa_metadata(_coh_qa(QaMetric.NUM), **composite).tags,  # type: ignore[arg-type]
        "rtc_burst": product_tags.backscatter_metadata(_rtc_burst(), BackscatterConfig()).tags,
        "gtc_burst": product_tags.backscatter_metadata(
            _gtc_burst(), BackscatterConfig(flatten_terrain=False)
        ).tags,
        "coh_burst": product_tags.coherence_metadata(_coh_burst(), CoherenceConfig()).tags,
        "ifg_burst": product_tags.interferogram_metadata(_ifg_burst(), InterferogramConfig()).tags,
    }


class TestMetadataVocabulary:
    """One pointer at the specification, identical everywhere, claiming nothing."""

    def test_every_product_carries_it(self) -> None:
        for name, tags in _every_builder().items():
            assert tags["METADATA_VOCABULARY"] == product_tags.METADATA_VOCABULARY, name

    def test_it_names_the_specification_and_its_url(self) -> None:
        assert "CEOS-ARD PFS SAR v1.3.1" in product_tags.METADATA_VOCABULARY
        assert product_tags.SPECIFICATION_URL in product_tags.METADATA_VOCABULARY

    def test_no_tag_key_claims_the_namespace(self) -> None:
        # The vocabulary is borrowed, not authored by CEOS, so no key may look
        # like a CEOS-defined field. The value is the only place it is named.
        for name, tags in _every_builder().items():
            assert [key for key in tags if "CEOS" in key] == [], name

    def test_the_pointer_does_not_vary_by_scaling(self) -> None:
        # Scaling is stated by BACKSCATTER_SCALING, not by qualifying the
        # reference; a pointer has nothing to scope.
        assert (
            _lrw(decibel=True)["METADATA_VOCABULARY"] == _lrw(decibel=False)["METADATA_VOCABULARY"]
        )


class TestBackscatterScaling:
    """Linear and decibel rasters must never carry each other's tags."""

    def test_linear_declares_the_conversion(self) -> None:
        tags = _lrw(decibel=False)
        assert tags["BACKSCATTER_CONVENTION"] == "Power"
        assert tags["BACKSCATTER_CONVERSION_EQ"] == "dB = 10*log10(DN)"
        assert "BACKSCATTER_SCALING" not in tags

    def test_decibel_omits_the_convention_and_the_equation(self) -> None:
        # BACKSCATTER_CONVENTION is enumerated [Amplitude, Power] and describes
        # the stored values, so neither value fits dB-stored data; emitting the
        # equation alongside would invite a second conversion.
        tags = _lrw(decibel=True)
        assert "BACKSCATTER_CONVENTION" not in tags
        assert "BACKSCATTER_CONVERSION_EQ" not in tags
        assert tags["BACKSCATTER_SCALING"] == "decibel"
        assert tags["BACKSCATTER_CONVERSION_APPLIED"] == "10*log10(power)"

    def test_unit_follows_the_scaling(self) -> None:
        common: dict = {
            "aggregation": "median",
            "clip_max_resolution": None,
            "n_observations": 1,
        }
        linear = product_tags.rtc_lrw_metadata(_rtc_lrw(), decibel=False, **common)
        decibel = product_tags.rtc_lrw_metadata(_rtc_lrw(), decibel=True, **common)
        assert linear.units == ["linear power"]
        assert decibel.units == ["dB"]

    def test_gtc_measures_beta_nought(self) -> None:
        config = BackscatterConfig(flatten_terrain=False)
        tags = product_tags.backscatter_metadata(_gtc_burst(), config).tags
        assert tags["BACKSCATTER_MEASUREMENT"] == "Beta-0"
        assert tags["TERRAIN_CORRECTION"] == "Geometric"

    def test_a_product_and_config_that_disagree_on_flattening_raise(self) -> None:
        # The band count comes from the config and the units from the product;
        # pairing them across a mismatch would silently misdescribe the file.
        with pytest.raises(ValueError, match="disagree on terrain flattening"):
            product_tags.backscatter_metadata(_gtc_burst(), BackscatterConfig())

    def test_rtc_measures_gamma_nought(self) -> None:
        tags = product_tags.backscatter_metadata(_rtc_burst(), BackscatterConfig()).tags
        assert tags["BACKSCATTER_MEASUREMENT"] == "Gamma-0"
        assert tags["TERRAIN_CORRECTION"] == "Radiometric and geometric"


class TestCompositingParameters:
    """Built from the call arguments, so they cannot describe a run that did not happen."""

    def test_records_the_aggregation_and_the_absent_clip(self) -> None:
        assert _lrw(aggregation="mean")["COMPOSITING_PARAMETERS"] == (
            "aggregation=mean;clip_max_resolution=none"
        )

    def test_records_a_clip_that_was_applied(self) -> None:
        assert _lrw(clip_max_resolution=2.5)["COMPOSITING_PARAMETERS"] == (
            "aggregation=median;clip_max_resolution=2.5"
        )

    def test_a_float_aggregation_is_named_as_a_quantile(self) -> None:
        assert "aggregation=quantile=0.9" in _lrw(aggregation=0.9)["COMPOSITING_PARAMETERS"]

    def test_lrw_cites_small_et_al(self) -> None:
        tags = _lrw()
        assert tags["COMPOSITING_METHOD"] == "Local resolution weighting"
        assert tags["COMPOSITING_METHOD_REFERENCE"] == product_tags.LRW_REFERENCE

    def test_pass_composite_claims_no_weighting_and_no_reference(self) -> None:
        # It applies none, so citing the LRW paper would misdescribe it.
        tags = product_tags.rtc_pass_metadata(
            _rtc_pass(),
            aggregation="median",
            n_observations=4,
            decibel=False,
        ).tags
        assert tags["COMPOSITING_METHOD"] == "Unweighted temporal aggregation"
        assert "COMPOSITING_METHOD_REFERENCE" not in tags
        assert "clip_max_resolution" not in tags["COMPOSITING_PARAMETERS"]
        assert tags["ORBIT_DIRECTION"] == "ASCENDING"


class TestTileIdentity:
    def test_carries_the_date_window_and_polarization(self) -> None:
        tags = _lrw()
        assert tags["COMPOSITE_START_DATE"] == "2020-01-01"
        assert tags["COMPOSITE_END_DATE"] == "2020-03-31"
        assert tags["POLARIZATION"] == "VV"
        assert tags["MGRS_TILE"] == "32TMS03"

    def test_product_type_matches_the_dataclass(self) -> None:
        product = _rtc_lrw()
        assert _lrw()["PRODUCT_TYPE"] == product.product_type == "RTC_LRW"

    def test_product_family_uses_the_ceos_wording(self) -> None:
        assert _lrw()["PRODUCT_FAMILY"] == "Composite Backscatter"


class TestCoherenceComposite:
    def _tags(self, **kwargs: object) -> dict[str, str]:
        defaults: dict = {
            "aggregation": "median",
            "clip_max_resolution": None,
            "n_observations": 5,
        }
        return product_tags.coh_lrw_metadata(_coh_lrw(), **(defaults | kwargs)).tags  # type: ignore[arg-type]

    def test_emits_no_backscatter_tags(self) -> None:
        # Coherence is a dimensionless magnitude on [0, 1], not a power
        # quantity, so dB = 10*log10(DN) would be actively wrong for it.
        tags = self._tags()
        assert not [k for k in tags if k.startswith("BACKSCATTER_")]
        assert "CONVERSION_EQ" not in tags
        assert tags["MEASUREMENT_TYPE"] == "Coherence"

    def test_compositing_method_says_lrw_is_applied_to_coherence(self) -> None:
        # Small et al. define LRW for backscatter power; the reference alone
        # would imply a method the paper does not cover.
        method = self._tags()["COMPOSITING_METHOD"]
        assert "applied to 12-day repeat-pass coherence" in method

    def test_carries_the_temporal_baseline(self) -> None:
        assert self._tags()["TEMPORAL_BASELINE_DAYS"] == "12"

    def test_inherits_the_estimation_window_from_the_inputs(self) -> None:
        # The window originates in CoherenceConfig at graph time and reaches the
        # composite by no other route.
        source = product_tags.coherence_metadata(_coh_burst(), CoherenceConfig(window=(5, 1))).tags
        tags = self._tags(source_tags=source)
        assert tags["COHERENCE_WINDOW_RANGE"] == "5"
        assert tags["COHERENCE_WINDOW_AZIMUTH"] == "1"
        assert tags["COHERENCE_WEIGHTING_SHAPE"] == "Rectangular"
        assert tags["PHASE_QUALITY_METHOD"] == "Coherence"

    def test_an_estimation_window_the_inputs_never_stated_stays_absent(self) -> None:
        assert "COHERENCE_WINDOW_RANGE" not in self._tags()

    def test_inherits_only_the_named_keys(self) -> None:
        tags = self._tags(source_tags={"PROCESSING_SOFTWARE": "ESA SNAP", "BURST_ID": "30346"})
        assert tags["PROCESSING_SOFTWARE"] == "ESA SNAP"
        assert "BURST_ID" not in tags

    def test_band_is_named_for_the_polarization(self) -> None:
        # Coherence composites ship VV and VH like the backscatter ones, so the
        # band carries the polarization rather than the quantity.
        metadata = product_tags.coh_lrw_metadata(
            _coh_lrw(),
            aggregation="median",
            clip_max_resolution=None,
            n_observations=5,
        )
        assert metadata.band_names == ["VV"]
        assert metadata.units == [""]

    def test_pass_composite_carries_direction_and_baseline(self) -> None:
        tags = product_tags.coh_pass_metadata(
            _coh_pass(),
            aggregation="mean",
            n_observations=3,
        ).tags
        assert tags["ORBIT_DIRECTION"] == "DESCENDING"
        assert tags["TEMPORAL_BASELINE_DAYS"] == "12"
        assert tags["PRODUCT_FAMILY"] == "Composite Coherence"


class TestCoherenceStatistics:
    def test_reports_mean_and_standard_deviation(self) -> None:
        stats = product_tags.coherence_statistics(np.array([0.2, 0.4, 0.6], dtype=np.float32))
        assert float(stats["COHERENCE_AVERAGE"]) == pytest.approx(0.4, abs=1e-6)
        assert float(stats["COHERENCE_STD"]) == pytest.approx(np.std([0.2, 0.4, 0.6]), abs=1e-6)

    def test_ignores_non_finite_pixels(self) -> None:
        stats = product_tags.coherence_statistics(
            np.array([0.5, np.nan, np.inf], dtype=np.float32)
        )
        assert float(stats["COHERENCE_AVERAGE"]) == pytest.approx(0.5)

    def test_an_empty_tile_carries_no_statistics_rather_than_a_nan(self) -> None:
        assert product_tags.coherence_statistics(np.full((2, 2), np.nan, np.float32)) == {}


class TestQaLayers:
    def _num(self, **kwargs: object) -> dict[str, str]:
        defaults: dict = {"aggregation": "median", "n_observations": 6}
        return product_tags.qa_metadata(_qa(QaMetric.NUM), **(defaults | kwargs)).tags  # type: ignore[arg-type]

    def _cqm(self, **kwargs: object) -> dict[str, str]:
        defaults: dict = {"aggregation": "median", "n_observations": 6}
        return product_tags.qa_metadata(_qa(QaMetric.CQM), **(defaults | kwargs)).tags  # type: ignore[arg-type]

    def test_num_carries_no_scaling_tags(self) -> None:
        # The PFS prose for 2.21 lists a dB-scaling expression, but the metadata
        # spec's parameter list does not: a per-pixel integer count has none.
        tags = self._num()
        assert not [k for k in tags if "CONVERSION" in k or "SCALING" in k]
        assert not [k for k in tags if k.startswith("BACKSCATTER_")]

    def test_num_is_not_described_as_a_data_mask(self) -> None:
        # PFS item 2.2 asks for three distinguishable states; a zero count
        # collapses "invalid data" and "no data" into one.
        tags = self._num()
        assert tags["QA_LAYER"] == "Contributing Observations Image"
        assert "mask" not in tags["LAYER_DESCRIPTION"].lower()

    def test_num_records_the_contributing_observation_count(self) -> None:
        assert self._num(n_observations=6)["CONTRIBUTING_OBSERVATIONS"] == "6"

    def test_num_band_is_dimensionless(self) -> None:
        metadata = product_tags.qa_metadata(
            _qa(QaMetric.NUM),
            aggregation="median",
            n_observations=6,
        )
        assert metadata.band_names == ["num_obs"]
        assert metadata.units == [""]

    def test_linear_cqm_declares_the_conversion(self) -> None:
        tags = self._cqm(cqm_decibel=False)
        assert tags["CQM_CONVERSION_EQ"] == "dB = 10*log10(DN)"
        assert "CQM_SCALING" not in tags
        assert tags["QA_LAYER"] == "Composite Quality Map Image"

    def test_decibel_cqm_states_that_it_was_applied(self) -> None:
        tags = self._cqm(cqm_decibel=True)
        assert "CQM_CONVERSION_EQ" not in tags
        assert tags["CQM_SCALING"] == "decibel"
        assert tags["CQM_CONVERSION_APPLIED"] == "10*log10(cqm)"

    def test_cqm_decibel_does_not_leak_onto_the_num_layer(self) -> None:
        # Only the CQM is written in dB; the count is unaffected by the flag.
        tags = self._num(cqm_decibel=True)
        assert "CQM_SCALING" not in tags
        assert "CQM_CONVERSION_APPLIED" not in tags

    def test_pass_qa_is_unweighted_and_counts_finite_values(self) -> None:
        tags = product_tags.qa_metadata(
            RTCPassQA(
                mgrs_code="32TMS03",
                start_date=_START,
                end_date=_END,
                polarization="VV",
                orbit_direction="ASCENDING",
            ),
            aggregation="mean",
            n_observations=2,
        ).tags
        assert tags["COMPOSITING_METHOD"] == "Unweighted temporal aggregation"
        assert tags["QA_LAYER"] == "Contributing Observations Image"
        assert "holding a finite value" in tags["LAYER_DESCRIPTION"]
        assert tags["ORBIT_DIRECTION"] == "ASCENDING"

    def test_lrw_qa_counts_weighted_contributions(self) -> None:
        assert "non-zero resolution weight" in self._num()["LAYER_DESCRIPTION"]

    def test_coherence_qa_takes_the_coherence_family(self) -> None:
        tags = product_tags.qa_metadata(
            _coh_qa(QaMetric.CQM),
            aggregation="median",
            n_observations=4,
        ).tags
        assert tags["PRODUCT_FAMILY"] == "Composite Coherence"

    def test_coherence_pass_qa_carries_the_direction(self) -> None:
        tags = product_tags.qa_metadata(
            CohPassQA(
                mgrs_code="32TMS03",
                start_date=_START,
                end_date=_END,
                polarization="VV",
                orbit_direction="DESCENDING",
                baseline_days=12,
            ),
            aggregation="mean",
            n_observations=2,
        ).tags
        assert tags["ORBIT_DIRECTION"] == "DESCENDING"
        assert tags["PRODUCT_FAMILY"] == "Composite Coherence"


class TestBurstProducts:
    def test_rtc_names_the_scattering_area_band(self) -> None:
        metadata = product_tags.backscatter_metadata(_rtc_burst(), BackscatterConfig())
        assert metadata.band_names == ["Gamma0", "simulatedImage"]
        assert metadata.units == ["linear power", ""]
        assert metadata.band_tags[1]["LAYER_NAME"] == "scatteringArea"

    def test_gtc_has_a_single_band(self) -> None:
        metadata = product_tags.backscatter_metadata(
            _gtc_burst(),
            BackscatterConfig(flatten_terrain=False),
        )
        assert metadata.band_names == ["Beta0"]
        assert metadata.units == ["linear power"]
        assert metadata.band_tags == []

    def test_multilook_is_recorded_when_applied(self) -> None:
        tags = product_tags.backscatter_metadata(
            _rtc_burst(),
            BackscatterConfig(multilook=(5, 1)),
        ).tags
        assert tags["MULTILOOK_RANGE"] == "5"
        assert tags["MULTILOOK_AZIMUTH"] == "1"

    def test_no_multilook_emits_no_looks(self) -> None:
        tags = product_tags.backscatter_metadata(
            _rtc_burst(),
            BackscatterConfig(multilook=None),
        ).tags
        assert "MULTILOOK_RANGE" not in tags

    def test_burst_identity_is_recorded(self) -> None:
        tags = product_tags.backscatter_metadata(_rtc_burst(), BackscatterConfig()).tags
        assert tags["BURST_ID"] == "30346"
        assert tags["SUB_SWATH"] == "IW1"
        assert tags["ACQUISITION_DATE"] == "2020-02-05"
        assert tags["GRID_SPACING"] == "10 m"
        assert tags["PROCESSING_SOFTWARE"] == "ESA SNAP"

    def test_a_builtin_dem_is_identified_by_its_snap_name(self) -> None:
        config = BackscatterConfig(
            dem="Copernicus 30m Global DEM",
            dem_name="ignored for built-in DEMs",
        )
        tags = product_tags.backscatter_metadata(_rtc_burst(), config).tags
        assert tags["DEM"] == "Copernicus 30m Global DEM"

    def test_an_external_dem_uses_its_configured_name_without_leaking_its_path(self) -> None:
        config = CoherenceConfig(
            dem=Path("/data/dem/glo30_ellipsoid.tif"),
            dem_name="Copernicus GLO-30, EGM2008",
        )
        tags = product_tags.coherence_metadata(_coh_burst(), config).tags
        assert tags["DEM"] == "Copernicus GLO-30, EGM2008"
        assert str(config.dem) not in tags.values()

    def test_an_unnamed_external_dem_is_unknown(self) -> None:
        config = InterferogramConfig(dem="/data/dem/glo30_ellipsoid.tif")
        tags = product_tags.interferogram_metadata(_ifg_burst(), config).tags
        assert tags["DEM"] == product_tags.UNKNOWN

    def test_a_composite_inherits_the_dem(self) -> None:
        tags = _lrw(source_tags={"DEM": "Copernicus 30m Global DEM"})
        assert tags["DEM"] == "Copernicus 30m Global DEM"

    def test_coherence_burst_records_the_window_and_the_baseline(self) -> None:
        tags = product_tags.coherence_metadata(_coh_burst(), CoherenceConfig(window=(25, 5))).tags
        assert tags["COHERENCE_WINDOW_RANGE"] == "25"
        assert tags["COHERENCE_WINDOW_AZIMUTH"] == "5"
        assert tags["TEMPORAL_BASELINE_DAYS"] == "12"
        assert tags["DATE_PRIMARY"] == "2020-02-05"
        assert tags["DATE_SECONDARY"] == "2020-02-17"

    def test_interferogram_marks_only_the_intensities_as_power(self) -> None:
        metadata = product_tags.interferogram_metadata(_ifg_burst(), InterferogramConfig())
        assert metadata.band_names == ["ifg_i", "ifg_q", "intensity_1", "intensity_2"]
        assert metadata.units == ["", "", "linear power", "linear power"]

    def test_interferogram_records_the_phase_removals(self) -> None:
        tags = product_tags.interferogram_metadata(
            _ifg_burst(),
            InterferogramConfig(subtract_topographic_phase=False),
        ).tags
        assert tags["FLAT_EARTH_PHASE_REMOVED"] == "true"
        assert tags["TOPOGRAPHIC_PHASE_REMOVED"] == "false"


class TestBurstDispatch:
    @pytest.mark.parametrize(
        ("product", "config", "product_type"),
        [
            (_rtc_burst(), BackscatterConfig(), "RTC"),
            (_gtc_burst(), BackscatterConfig(flatten_terrain=False), "GTC"),
            (_coh_burst(), CoherenceConfig(), "COH"),
            (_ifg_burst(), InterferogramConfig(), "IFG"),
        ],
    )
    def test_dispatches_on_the_config(
        self,
        product: object,
        config: object,
        product_type: str,
    ) -> None:
        metadata = product_tags.burst_metadata(product, config)  # type: ignore[arg-type]
        assert metadata.tags["PRODUCT_TYPE"] == product_type

    def test_a_mismatched_pair_raises(self) -> None:
        with pytest.raises(TypeError, match="cannot build metadata"):
            product_tags.burst_metadata(_rtc_burst(), CoherenceConfig())  # type: ignore[arg-type]


class TestRasterMetadata:
    def test_rejects_a_band_count_the_lists_disagree_on(self) -> None:
        with pytest.raises(ValueError, match="same number of bands"):
            RasterMetadata(tags={}, band_names=["a", "b"], units=["linear power"])

    def test_band_tags_are_optional(self) -> None:
        metadata = RasterMetadata(tags={}, band_names=["a"], units=[""])
        assert metadata.band_tags == []

    def test_every_tag_value_is_a_string(self) -> None:
        # GDAL stores tag values as strings; a stray int or Timestamp would be
        # coerced silently and could format differently across versions.
        for tags in (
            _lrw(),
            _lrw(decibel=True),
            product_tags.coh_lrw_metadata(
                _coh_lrw(),
                aggregation=0.5,
                clip_max_resolution=1.0,
                n_observations=2,
            ).tags,
            product_tags.qa_metadata(_qa(QaMetric.CQM), aggregation="mean", n_observations=1).tags,
            product_tags.backscatter_metadata(_rtc_burst(), BackscatterConfig()).tags,
            product_tags.coherence_metadata(_coh_burst(), CoherenceConfig()).tags,
            product_tags.interferogram_metadata(_ifg_burst(), InterferogramConfig()).tags,
        ):
            assert all(isinstance(v, str) for v in tags.values())
            assert all(isinstance(k, str) for k in tags)


class TestProvenance:
    """Who wrote the file, with what, and at which processing level."""

    def test_composite_stamps_the_writing_package(self) -> None:
        # The vocabulary in these tags is defined by this package, and the
        # meaning of QA-NUM has already changed once between releases.
        tags = _lrw()
        assert tags["PIPELINE_SOFTWARE"] == "s1bursts"
        assert tags["PIPELINE_SOFTWARE_VERSION"] == s1bursts.__version__
        assert tags["PIPELINE_SOFTWARE_URL"].startswith("https://github.com/")

    def test_burst_stamps_the_writing_package_too(self) -> None:
        tags = product_tags.backscatter_metadata(_rtc_burst(), BackscatterConfig()).tags
        assert tags["PIPELINE_SOFTWARE_VERSION"] == s1bursts.__version__

    def test_burst_records_the_snap_version(self) -> None:
        tags = product_tags.backscatter_metadata(
            _rtc_burst(),
            BackscatterConfig(),
            snap_version=(14, 0, 0),
        ).tags
        assert tags["PROCESSING_SOFTWARE"] == "ESA SNAP"
        assert tags["PROCESSING_SOFTWARE_VERSION"] == "14.0.0"

    def test_an_undetermined_snap_version_is_stated_as_unknown(self) -> None:
        # Present but empty, so a reader can tell it was considered rather than
        # written by a version of this package that predates the tag.
        tags = product_tags.backscatter_metadata(_rtc_burst(), BackscatterConfig()).tags
        assert tags["PROCESSING_SOFTWARE_VERSION"] == product_tags.UNKNOWN

    def test_provenance_a_composite_cannot_establish_is_stated_as_unknown(self) -> None:
        tags = _lrw(source_tags={})
        assert tags["DEM"] == product_tags.UNKNOWN
        assert tags["PROCESSING_SOFTWARE"] == product_tags.UNKNOWN
        assert tags["PROCESSING_SOFTWARE_VERSION"] == product_tags.UNKNOWN

    def test_an_inapplicable_coherence_key_stays_absent(self) -> None:
        # An RTC composite has no estimation window, so N/A would be nonsense.
        assert "COHERENCE_WINDOW_RANGE" not in _lrw(source_tags={})

    def test_the_snap_version_reaches_every_burst_product(self) -> None:
        for product, config in (
            (_rtc_burst(), BackscatterConfig()),
            (_coh_burst(), CoherenceConfig()),
            (_ifg_burst(), InterferogramConfig()),
        ):
            tags = product_tags.burst_metadata(product, config, snap_version=(13, 0, 0)).tags  # type: ignore[arg-type]
            assert tags["PROCESSING_SOFTWARE_VERSION"] == "13.0.0"

    def test_a_burst_is_level_two_over_an_l1_slc(self) -> None:
        tags = product_tags.backscatter_metadata(_rtc_burst(), BackscatterConfig()).tags
        assert tags["PRODUCT_LEVEL"] == "L2"
        assert tags["SOURCE_DATA_PRODUCT_LEVEL"] == "L1"

    def test_a_composite_is_level_three_over_l2_bursts(self) -> None:
        tags = _lrw()
        assert tags["PRODUCT_LEVEL"] == "L3"
        assert tags["SOURCE_DATA_PRODUCT_LEVEL"] == "L2"

    def test_a_composite_inherits_the_snap_version_from_its_inputs(self) -> None:
        source = product_tags.backscatter_metadata(
            _rtc_burst(),
            BackscatterConfig(),
            snap_version=(14, 0, 0),
        ).tags
        assert _lrw(source_tags=source)["PROCESSING_SOFTWARE_VERSION"] == "14.0.0"

    def test_the_pipeline_version_is_never_inherited(self) -> None:
        # The composite is written by the running package, not by whatever
        # wrote its inputs, so a stale version must not travel forward.
        tags = _lrw(source_tags={"PIPELINE_SOFTWARE_VERSION": "0.0.1-ancient"})
        assert tags["PIPELINE_SOFTWARE_VERSION"] == s1bursts.__version__
