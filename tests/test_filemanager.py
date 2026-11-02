"""Tests for FileManager filename parsing, round-trips, and product discovery."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch
import warnings

import pandas as pd
from pyproj import CRS
import pytest

if TYPE_CHECKING:
    from typing import ClassVar

from s1bursts.filemanager import _PARSERS, FileManager
from s1bursts.product import (
    BaseProduct,
    CoherenceProduct,
    CohLRWComposite,
    CohPassComposite,
    GTCPassComposite,
    GTCProduct,
    InterferogramProduct,
    Layer,
    MGRSTileProduct,
    RTCLRWComposite,
    RTCPassComposite,
    RTCProduct,
)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

_CRS_32N = CRS.from_epsg(32632)

_PARSE_CASES = [
    pytest.param(
        RTCProduct,
        "S1_RTC_015A_030343_IW1_VH_20150709_32N.tif",
        {
            "burst_id": 30343,
            "sub_swath": "IW1",
            "polarization": "VH",
            "date": pd.Timestamp("20150709"),
            "crs": _CRS_32N,
            "rel_orbit_number": 15,
            "pass_direction": "Ascending",
        },
        id="rtc",
    ),
    pytest.param(
        GTCProduct,
        "S1_GTC_015D_030343_IW1_VH_20150709_32N.tif",
        {
            "burst_id": 30343,
            "sub_swath": "IW1",
            "polarization": "VH",
            "date": pd.Timestamp("20150709"),
            "crs": _CRS_32N,
            "rel_orbit_number": 15,
            "pass_direction": "Descending",
        },
        id="gtc",
    ),
    pytest.param(
        CoherenceProduct,
        "S1_COH12_015A_030345_IW1_VH_20150709_20150721_32N.tif",
        {
            "burst_id": 30345,
            "sub_swath": "IW1",
            "polarization": "VH",
            "date_primary": pd.Timestamp("20150709"),
            "date_secondary": pd.Timestamp("20150721"),
            "crs": _CRS_32N,
            "rel_orbit_number": 15,
            "pass_direction": "Ascending",
        },
        id="coherence",
    ),
    pytest.param(
        InterferogramProduct,
        "S1_IFG12_CMPLX_015D_030345_IW1_VH_20150709_20150721_32N.tif",
        {
            "burst_id": 30345,
            "sub_swath": "IW1",
            "polarization": "VH",
            "date_primary": pd.Timestamp("20150709"),
            "date_secondary": pd.Timestamp("20150721"),
            "crs": _CRS_32N,
            "rel_orbit_number": 15,
            "pass_direction": "Descending",
        },
        id="interferogram",
    ),
    pytest.param(
        RTCLRWComposite,
        "S1_RTC_LRW_32TMS03_VH_20150706_20151001.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20151001"),
            "layers": frozenset({Layer.MEASUREMENT}),
        },
        id="rtc-lrw",
    ),
    pytest.param(
        CohLRWComposite,
        "S1_COH12_LRW_32TMS03_VH_20150706_20150919.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20150919"),
            "baseline_days": 12,
        },
        id="coh-lrw",
    ),
    pytest.param(
        RTCPassComposite,
        "S1_RTC_COMP_A_32TMS03_VH_20150706_20151001.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20151001"),
            "pass_direction": "Ascending",
        },
        id="rtc-pass",
    ),
    pytest.param(
        CohPassComposite,
        "S1_COH12_COMP_D_32TMS03_VH_20150706_20150919.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20150919"),
            "baseline_days": 12,
            "pass_direction": "Descending",
        },
        id="coh-pass",
    ),
    pytest.param(
        RTCLRWComposite,
        "S1_RTC_LRW_QA-NUM_32TMS03_VH_20150706_20151001.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20151001"),
            "layers": frozenset({Layer.QA_NUM}),
        },
        id="rtc-lrw-qa-num",
    ),
    pytest.param(
        RTCLRWComposite,
        "S1_RTC_LRW_QA-CQM_32TMS03_VH_20150706_20151001.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20151001"),
            "layers": frozenset({Layer.QA_CQM}),
        },
        id="rtc-lrw-qa-cqm",
    ),
    pytest.param(
        CohLRWComposite,
        "S1_COH12_LRW_QA-NUM_32TMS03_VH_20150706_20150919.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20150919"),
            "baseline_days": 12,
            "layers": frozenset({Layer.QA_NUM}),
        },
        id="coh-lrw-qa-num",
    ),
    pytest.param(
        RTCPassComposite,
        "S1_RTC_COMP_A_QA-NUM_32TMS03_VH_20150706_20151001.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20151001"),
            "pass_direction": "Ascending",
            "layers": frozenset({Layer.QA_NUM}),
        },
        id="rtc-pass-qa-num",
    ),
    pytest.param(
        CohPassComposite,
        "S1_COH12_COMP_D_QA-NUM_32TMS03_VH_20150706_20150919.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20150919"),
            "baseline_days": 12,
            "pass_direction": "Descending",
            "layers": frozenset({Layer.QA_NUM}),
        },
        id="coh-pass-qa-num",
    ),
]

_ROUNDTRIP_CASES = [
    pytest.param(RTCProduct, "S1_RTC_066A_030346_IW1_VV_20220105_32N.tif", id="rtc"),
    pytest.param(GTCProduct, "S1_GTC_139D_030346_IW1_VV_20220105_32N.tif", id="gtc"),
    pytest.param(
        CoherenceProduct,
        "S1_COH12_066A_030346_IW1_VV_20220105_20220117_32N.tif",
        id="coherence",
    ),
    pytest.param(
        InterferogramProduct,
        "S1_IFG12_CMPLX_066D_030346_IW1_VV_20220105_20220117_32S.tif",
        id="interferogram",
    ),
    pytest.param(RTCLRWComposite, "S1_RTC_LRW_32TMS03_VH_20150706_20151001.tif", id="rtc-lrw"),
    pytest.param(CohLRWComposite, "S1_COH12_LRW_32TMS03_VH_20150706_20150919.tif", id="coh-lrw"),
    pytest.param(
        RTCLRWComposite,
        "S1_RTC_LRW_QA-NUM_32TMS03_VH_20150706_20151001.tif",
        id="rtc-lrw-qa-num",
    ),
    pytest.param(
        RTCLRWComposite,
        "S1_RTC_LRW_QA-CQM_32TMS03_VH_20150706_20151001.tif",
        id="rtc-lrw-qa-cqm",
    ),
    pytest.param(
        CohLRWComposite,
        "S1_COH12_LRW_QA-NUM_32TMS03_VH_20150706_20150919.tif",
        id="coh-lrw-qa-num",
    ),
    pytest.param(
        RTCPassComposite,
        "S1_RTC_COMP_A_QA-NUM_32TMS03_VH_20150706_20151001.tif",
        id="rtc-pass-qa-num",
    ),
    pytest.param(
        CohPassComposite,
        "S1_COH12_COMP_D_QA-NUM_32TMS03_VH_20150706_20150919.tif",
        id="coh-pass-qa-num",
    ),
    pytest.param(
        RTCPassComposite,
        "S1_RTC_COMP_A_32TMS03_VH_20150706_20151001.tif",
        id="rtc-pass",
    ),
    pytest.param(
        CohPassComposite,
        "S1_COH12_COMP_D_32TMS03_VH_20150706_20150919.tif",
        id="coh-pass",
    ),
]


# ---------------------------------------------------------------------------
# TestFromDict
# ---------------------------------------------------------------------------


class TestFromDict:
    """Tests for ``BaseProduct.from_dict``."""

    def test_exact_keys(self) -> None:
        """Providing exactly the init fields constructs the product correctly."""
        product = RTCLRWComposite.from_dict(
            {
                "mgrs_code": "32TMS03",
                "polarization": "VH",
                "start_date": pd.Timestamp("20150706"),
                "end_date": pd.Timestamp("20151001"),
            },
        )
        assert product.mgrs_code == "32TMS03"
        assert product.polarization == "VH"
        assert product.start_date == pd.Timestamp("20150706")
        assert product.end_date == pd.Timestamp("20151001")

    def test_extra_keys_dropped(self) -> None:
        """Unknown keys are silently ignored — no TypeError is raised."""
        product = RTCLRWComposite.from_dict(
            {
                "mgrs_code": "32TMS03",
                "polarization": "VH",
                "start_date": pd.Timestamp("20150706"),
                "end_date": pd.Timestamp("20151001"),
                "unknown_field": "ignored",
                "another_extra": 99,
            },
        )
        assert isinstance(product, RTCLRWComposite)

    def test_init_false_keys_dropped(self) -> None:
        """Keys that correspond to ``init=False`` fields are filtered before construction."""
        product = CohLRWComposite.from_dict(
            {
                "mgrs_code": "32TMS03",
                "polarization": "VH",
                "start_date": pd.Timestamp("20150706"),
                "end_date": pd.Timestamp("20150919"),
                "baseline_days": 12,
                "rel_orbit_number": 999,  # not an init field on any MGRSTileProduct
            },
        )
        assert isinstance(product, CohLRWComposite)
        assert product.baseline_days == 12


# ---------------------------------------------------------------------------
# TestParseFilename
# ---------------------------------------------------------------------------


class TestParseFilename:
    """Tests for filename parsing via the ``_PARSERS`` registry."""

    @pytest.fixture
    def fm(self, tmp_path: Path) -> FileManager:
        """FileManager backed by a temporary directory."""
        return FileManager(tmp_path)

    @pytest.mark.parametrize(("cls", "filename", "expected"), _PARSE_CASES)
    def test_parse_returns_correct_fields(
        self,
        fm: FileManager,
        cls: type[BaseProduct],
        filename: str,
        expected: dict[str, Any],
    ) -> None:
        """Each expected field value must appear in the parsed result."""
        prefix_re, spec, prefix_captures = _PARSERS[cls]
        result = fm._parse_filename(prefix_re, spec, filename, prefix_captures)
        assert result is not None, f"Failed to parse {filename!r} for {cls.__name__}"
        for key, value in expected.items():
            assert result[key] == value, f"Field {key!r}: got {result[key]!r}, expected {value!r}"

    def test_name_without_pass_direction_returns_none(self, fm: FileManager) -> None:
        """A per-burst name in the scheme without the pass direction is not parsed."""
        prefix_re, spec, prefix_captures = _PARSERS[RTCProduct]
        result = fm._parse_filename(
            prefix_re,
            spec,
            "S1_RTC_015_030343_IW1_VH_20150709_32N.tif",
            prefix_captures,
        )
        assert result is None

    def test_wrong_type_returns_none(self, fm: FileManager) -> None:
        """An RTC filename does not parse under the CoherenceProduct spec."""
        prefix_re, spec, prefix_captures = _PARSERS[CoherenceProduct]
        result = fm._parse_filename(
            prefix_re,
            spec,
            "S1_RTC_015A_030343_IW1_VH_20150709_32N.tif",
            prefix_captures,
        )
        assert result is None


# ---------------------------------------------------------------------------
# TestRoundTrip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Parse filename → ``from_dict`` → ``get_product_path().name`` equality."""

    @pytest.fixture
    def fm(self, tmp_path: Path) -> FileManager:
        """FileManager backed by a temporary directory."""
        return FileManager(tmp_path)

    @pytest.mark.parametrize(("cls", "filename"), _ROUNDTRIP_CASES)
    def test_roundtrip(
        self,
        fm: FileManager,
        cls: type[BaseProduct],
        filename: str,
    ) -> None:
        """Parsing then reconstructing a path reproduces the original filename."""
        prefix_re, spec, prefix_captures = _PARSERS[cls]
        parsed = fm._parse_filename(prefix_re, spec, filename, prefix_captures)
        assert parsed is not None, f"Failed to parse {filename!r}"
        product = cls.from_dict(parsed)
        [layer] = product.layers if isinstance(product, MGRSTileProduct) else [Layer.MEASUREMENT]
        assert fm.get_product_path(product, layer).name == filename


# ---------------------------------------------------------------------------
# TestCompositeLayers
# ---------------------------------------------------------------------------


class TestCompositeLayers:
    """A composite's QA layers are files of the same product, not products of their own."""

    _COMPOSITE = RTCLRWComposite(
        mgrs_code="32TMS03",
        start_date=pd.Timestamp("20150706"),
        end_date=pd.Timestamp("20151001"),
        polarization="VH",
    )

    @pytest.fixture
    def fm(self, tmp_path: Path) -> FileManager:
        """FileManager backed by a temporary directory."""
        return FileManager(tmp_path)

    def test_each_layer_has_its_own_file(self, fm: FileManager) -> None:
        names = {layer: fm.get_product_path(self._COMPOSITE, layer).name for layer in Layer}

        assert names == {
            Layer.MEASUREMENT: "S1_RTC_LRW_32TMS03_VH_20150706_20151001.tif",
            Layer.QA_NUM: "S1_RTC_LRW_QA-NUM_32TMS03_VH_20150706_20151001.tif",
            Layer.QA_CQM: "S1_RTC_LRW_QA-CQM_32TMS03_VH_20150706_20151001.tif",
        }

    def test_a_layer_may_be_given_by_its_value(self, fm: FileManager) -> None:
        assert fm.get_product_path(self._COMPOSITE, "QA-NUM") == fm.get_product_path(
            self._COMPOSITE, Layer.QA_NUM
        )
        assert fm.get_product_path(self._COMPOSITE, "MEASUREMENT") == fm.get_product_path(
            self._COMPOSITE
        )
        assert not fm.product_exists(self._COMPOSITE, "QA-NUM")
        with pytest.raises(ValueError, match="is not a valid Layer"):
            fm.get_product_path(self._COMPOSITE, "QA-XYZ")

    @pytest.mark.parametrize("direction", ["ascending", "A"])
    def test_an_unknown_pass_direction_filter_is_refused(
        self, fm: FileManager, direction: str
    ) -> None:
        for find in (
            fm.find_rtc_pass_composites,
            fm.find_gtc_pass_composites,
            fm.find_coh_pass_composites,
        ):
            with pytest.raises(ValueError, match="must be 'Ascending' or 'Descending'"):
                # pyrefly: ignore [bad-argument-type]
                find(pass_direction=direction)

    def test_the_path_does_not_depend_on_the_layers_held(self, fm: FileManager) -> None:
        with_qa = RTCLRWComposite(
            **{k: getattr(self._COMPOSITE, k) for k in ("mgrs_code", "start_date", "end_date")},
            polarization="VH",
            layers=frozenset(Layer),
        )

        assert fm.get_product_path(with_qa) == fm.get_product_path(self._COMPOSITE)

    def test_a_pass_composite_has_no_quality_map_file(self, fm: FileManager) -> None:
        product = RTCPassComposite(
            mgrs_code="32TMS03",
            start_date=pd.Timestamp("20150706"),
            end_date=pd.Timestamp("20151001"),
            polarization="VH",
            pass_direction="Ascending",
        )

        with pytest.raises(ValueError, match="has no layer QA_CQM"):
            fm.get_product_path(product, Layer.QA_CQM)

    def test_a_per_burst_product_has_only_its_measurement(self, fm: FileManager) -> None:
        product = RTCProduct(
            30343,
            "IW1",
            _CRS_32N,
            pd.Timestamp("20150709"),
            "VH",
            pass_direction="Ascending",
            _rel_orbit_number=15,
        )

        with pytest.raises(ValueError, match="RTCProduct has no layer QA_NUM"):
            fm.product_exists(product, Layer.QA_NUM)
        assert not fm.product_exists(product)

    def test_an_unknown_product_type_is_refused(self, fm: FileManager) -> None:
        class _Unknown(MGRSTileProduct):
            @property
            def product_type(self) -> str:
                return "UNKNOWN"

        class _UnknownBurst(BaseProduct):
            @property
            def product_type(self) -> str:
                return "UNKNOWN"

        product = _Unknown("32TMS03", pd.Timestamp("20150706"), pd.Timestamp("20151001"), "VH")
        with pytest.raises(ValueError, match="Unknown product type"):
            fm.get_product_path(product)
        with pytest.raises(ValueError, match="Unknown product type"):
            fm.get_product_path(_UnknownBurst())

    def test_a_quality_map_name_does_not_parse_as_a_pass_composite(self, fm: FileManager) -> None:
        prefix_re, spec, prefix_captures = _PARSERS[RTCPassComposite]
        name = "S1_RTC_COMP_A_QA-CQM_32TMS03_VH_20150706_20151001.tif"
        assert fm._parse_filename(prefix_re, spec, name, prefix_captures) is None

    def test_find_returns_one_composite_holding_every_layer(self, fm: FileManager) -> None:
        for layer in Layer:
            fm.get_product_path(self._COMPOSITE, layer).touch()

        [found] = fm.find_rtc_lrw_products(mgrs_code="32TMS03", polarization="VH")

        assert found.layers == set(Layer)
        assert found == RTCLRWComposite(
            mgrs_code="32TMS03",
            start_date=pd.Timestamp("20150706"),
            end_date=pd.Timestamp("20151001"),
            polarization="VH",
            layers=frozenset(Layer),
        )

    def test_find_reports_the_layers_on_disk(self, fm: FileManager) -> None:
        fm.get_product_path(self._COMPOSITE).touch()
        other = RTCLRWComposite(
            mgrs_code="32TMS04",
            start_date=pd.Timestamp("20150706"),
            end_date=pd.Timestamp("20151001"),
            polarization="VH",
        )
        fm.get_product_path(other, Layer.QA_NUM).touch()

        found = fm.find_rtc_lrw_products()

        assert [(p.mgrs_code, p.layers) for p in found] == [
            ("32TMS03", {Layer.MEASUREMENT}),
            ("32TMS04", {Layer.QA_NUM}),
        ]

    def test_find_does_not_mix_composite_types(self, fm: FileManager) -> None:
        coherence = CohLRWComposite(
            mgrs_code="32TMS03",
            start_date=pd.Timestamp("20150706"),
            end_date=pd.Timestamp("20151001"),
            polarization="VH",
            baseline_days=12,
        )
        fm.get_product_path(coherence, Layer.QA_CQM).touch()

        assert fm.find_rtc_lrw_products() == []
        assert fm.find_coh_pass_composites() == []
        [found] = fm.find_coh_lrw_products(baseline_days=12)
        assert found.layers == {Layer.QA_CQM}

    def test_tightened_pol_still_parses_composites(self, fm: FileManager) -> None:
        """The tightened polarization pattern still parses VV/VH composites."""
        for pol in ("VV", "VH", "HH", "HV"):
            filename = f"S1_RTC_LRW_32TMS03_{pol}_20150706_20151001.tif"
            prefix_re, spec, prefix_captures = _PARSERS[RTCLRWComposite]
            parsed = fm._parse_filename(prefix_re, spec, filename, prefix_captures)
            assert parsed is not None, f"Failed to parse polarization {pol}"
            assert parsed["polarization"] == pol


# ---------------------------------------------------------------------------
# TestFindRTCLRWProducts
# ---------------------------------------------------------------------------


class TestFindRTCLRWProducts:
    """Tests for ``FileManager.find_rtc_lrw_products``."""

    _FILENAMES: ClassVar[list[str]] = [
        "S1_RTC_LRW_32TMS03_VH_20150706_20151001.tif",
        "S1_RTC_LRW_32TMS03_VV_20150706_20151001.tif",
        "S1_RTC_LRW_32TMS04_VH_20150706_20151001.tif",
    ]

    @pytest.fixture
    def fm(self, tmp_path: Path) -> FileManager:
        """FileManager with stub .tif files in comp_dir."""
        fm = FileManager(tmp_path)
        for name in self._FILENAMES:
            (fm.comp_dir / name).touch()
        return fm

    def test_find_all(self, fm: FileManager) -> None:
        """No filters returns all products."""
        results = fm.find_rtc_lrw_products()
        assert len(results) == 3
        assert all(isinstance(p, RTCLRWComposite) for p in results)

    def test_filter_by_mgrs_code(self, fm: FileManager) -> None:
        """mgrs_code filter restricts results to that tile."""
        results = fm.find_rtc_lrw_products(mgrs_code="32TMS03")
        assert len(results) == 2
        assert all(p.mgrs_code == "32TMS03" for p in results)

    def test_filter_by_polarization(self, fm: FileManager) -> None:
        """Polarization filter restricts results to that polarization."""
        results = fm.find_rtc_lrw_products(polarization="VH")
        assert len(results) == 2
        assert all(p.polarization == "VH" for p in results)

    def test_returns_sorted(self, fm: FileManager) -> None:
        """Results are sorted alphabetically by filename."""
        results = fm.find_rtc_lrw_products()
        names = [fm.get_product_path(p).name for p in results]
        assert names == sorted(names)

    def test_empty_directory(self, tmp_path: Path) -> None:
        """An empty directory returns an empty list."""
        assert FileManager(tmp_path).find_rtc_lrw_products() == []


# ---------------------------------------------------------------------------
# TestFindCohLRWProducts
# ---------------------------------------------------------------------------


class TestFindCohLRWProducts:
    """Tests for ``FileManager.find_coh_lrw_products``."""

    _FILENAMES: ClassVar[list[str]] = [
        "S1_COH12_LRW_32TMS03_VH_20150706_20150919.tif",
        "S1_COH06_LRW_32TMS03_VH_20150706_20150919.tif",
        "S1_COH12_LRW_32TMS03_VV_20150706_20150919.tif",
    ]

    @pytest.fixture
    def fm(self, tmp_path: Path) -> FileManager:
        """FileManager with stub .tif files in comp_dir."""
        fm = FileManager(tmp_path)
        for name in self._FILENAMES:
            (fm.comp_dir / name).touch()
        return fm

    def test_find_all(self, fm: FileManager) -> None:
        """No filters returns all products."""
        results = fm.find_coh_lrw_products()
        assert len(results) == 3
        assert all(isinstance(p, CohLRWComposite) for p in results)

    def test_filter_by_baseline_days(self, fm: FileManager) -> None:
        """baseline_days filter restricts results to that baseline."""
        results = fm.find_coh_lrw_products(baseline_days=12)
        assert len(results) == 2
        assert all(p.baseline_days == 12 for p in results)

    def test_baseline_days_is_int(self, fm: FileManager) -> None:
        """baseline_days must be parsed as int, not str."""
        results = fm.find_coh_lrw_products()
        assert all(isinstance(p.baseline_days, int) for p in results)


# ---------------------------------------------------------------------------
# TestFindRTCPassComposites
# ---------------------------------------------------------------------------


class TestFindRTCPassComposites:
    """Tests for ``FileManager.find_rtc_pass_composites``."""

    _FILENAMES: ClassVar[list[str]] = [
        "S1_RTC_COMP_A_32TMS03_VH_20150706_20151001.tif",
        "S1_RTC_COMP_D_32TMS03_VH_20150706_20151001.tif",
        "S1_RTC_COMP_A_32TMS04_VH_20150706_20151001.tif",
    ]

    @pytest.fixture
    def fm(self, tmp_path: Path) -> FileManager:
        """FileManager with stub .tif files in comp_dir."""
        fm = FileManager(tmp_path)
        for name in self._FILENAMES:
            (fm.comp_dir / name).touch()
        return fm

    def test_find_all(self, fm: FileManager) -> None:
        """No filters returns all products."""
        results = fm.find_rtc_pass_composites()
        assert len(results) == 3
        assert all(isinstance(p, RTCPassComposite) for p in results)

    def test_filter_ascending(self, fm: FileManager) -> None:
        """Ascending filter returns only ascending-pass products."""
        results = fm.find_rtc_pass_composites(pass_direction="Ascending")
        assert len(results) == 2
        assert all(p.pass_direction == "Ascending" for p in results)

    def test_filter_descending(self, fm: FileManager) -> None:
        """Descending filter returns only descending-pass products."""
        results = fm.find_rtc_pass_composites(pass_direction="Descending")
        assert len(results) == 1
        assert results[0].pass_direction == "Descending"

    def test_pass_direction_is_full_string(self, fm: FileManager) -> None:
        """pass_direction must be the full word, not the single-letter code."""
        results = fm.find_rtc_pass_composites()
        assert all(p.pass_direction in {"Ascending", "Descending"} for p in results)


# ---------------------------------------------------------------------------
# TestFindCohPassComposites
# ---------------------------------------------------------------------------


class TestFindCohPassComposites:
    """Tests for ``FileManager.find_coh_pass_composites``."""

    _FILENAMES: ClassVar[list[str]] = [
        "S1_COH12_COMP_A_32TMS03_VH_20150706_20150919.tif",
        "S1_COH12_COMP_D_32TMS03_VH_20150706_20150919.tif",
        "S1_COH06_COMP_A_32TMS03_VH_20150706_20150919.tif",
    ]

    @pytest.fixture
    def fm(self, tmp_path: Path) -> FileManager:
        """FileManager with stub .tif files in comp_dir."""
        fm = FileManager(tmp_path)
        for name in self._FILENAMES:
            (fm.comp_dir / name).touch()
        return fm

    def test_find_all(self, fm: FileManager) -> None:
        """No filters returns all products."""
        results = fm.find_coh_pass_composites()
        assert len(results) == 3
        assert all(isinstance(p, CohPassComposite) for p in results)

    def test_filter_baseline_and_direction(self, fm: FileManager) -> None:
        """Combining baseline_days and pass_direction returns exactly one match."""
        results = fm.find_coh_pass_composites(baseline_days=12, pass_direction="Ascending")
        assert len(results) == 1
        assert results[0].baseline_days == 12
        assert results[0].pass_direction == "Ascending"


def test_gtc_pass_composite_path_round_trip(tmp_path: Path) -> None:
    fm = FileManager(tmp_path)
    product = GTCPassComposite(
        mgrs_code="32TMS25",
        start_date=pd.Timestamp("2019-01-01"),
        end_date=pd.Timestamp("2019-01-31"),
        polarization="VV",
        pass_direction="Ascending",
    )
    path = fm.get_product_path(product)
    assert path == fm.comp_dir / "S1_GTC_COMP_A_32TMS25_VV_20190101_20190131.tif"
    path.touch()
    assert fm.find_gtc_pass_composites(mgrs_code="32TMS25") == [product]
    assert fm.find_gtc_pass_composites(pass_direction="Descending") == []
    assert fm.find_rtc_pass_composites() == []

    qa_path = fm.get_product_path(product, Layer.QA_NUM)
    assert qa_path.name == "S1_GTC_COMP_A_QA-NUM_32TMS25_VV_20190101_20190131.tif"
    qa_path.touch()
    [found] = fm.find_gtc_pass_composites(pass_direction="Ascending")
    assert found.layers == {Layer.MEASUREMENT, Layer.QA_NUM}


class TestStringPaths:
    """``FileManager`` accepts strings as well as path objects."""

    def test_local_string(self, tmp_path: Path) -> None:
        fm = FileManager(str(tmp_path / "data"), slc_dir=str(tmp_path / "slc"))

        assert fm.base_dir == tmp_path / "data"
        assert fm.slc_dir == tmp_path / "slc"
        assert fm.prod_dir.is_dir()

    def test_home_is_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))

        assert FileManager("~/data").base_dir == tmp_path / "data"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"base_dir": "s3://bucket/prefix"},
            {"base_dir": "data", "prod_dir": "memory://bucket/prod"},
            {"base_dir": ""},
        ],
    )
    def test_url_or_empty_string_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, str]
    ) -> None:
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match=r"URL|empty"):
            FileManager(**kwargs)
        assert not any(tmp_path.iterdir())

    def test_windows_drive_is_not_a_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        assert FileManager("C://data").base_dir == Path("C://data")


class TestFindKeepsFootprint:
    """Products rebuilt from file names keep the orbit they record."""

    def test_found_product_keeps_rel_orbit_number(self, tmp_path: Path) -> None:
        # Previously dropped, so compositing found products needed the burst map.
        fm = FileManager(tmp_path)
        product = RTCProduct(
            30343,
            "IW1",
            _CRS_32N,
            pd.Timestamp("20150709"),
            "VH",
            pass_direction="Descending",
            _rel_orbit_number=66,
        )
        fm.get_product_path(product).touch()

        with (
            patch(
                "s1bursts.burst.create_s1_orbit_number_lut",
                side_effect=AssertionError("must not resolve the burst map"),
            ),
            patch(
                "s1bursts.burst.create_s1_pass_direction_lut",
                side_effect=AssertionError("must not resolve the burst map"),
            ),
        ):
            [found] = fm.find_rtc_products()
            assert found.rel_orbit_number == 66
            assert found.pass_direction == "Descending"
        assert found == product


class TestFindPerBurstByPassDirection:
    """Per-burst products can be filtered on the pass direction their names record."""

    _FILENAMES: ClassVar[list[str]] = [
        "S1_RTC_015A_030343_IW1_VH_20150709_32N.tif",
        "S1_RTC_015A_030344_IW1_VH_20150709_32N.tif",
        "S1_RTC_066D_030343_IW1_VH_20150709_32N.tif",
        "S1_GTC_015A_030343_IW1_VH_20150709_32N.tif",
        "S1_GTC_066D_030343_IW1_VH_20150709_32N.tif",
        "S1_COH12_015A_030343_IW1_VH_20150709_20150721_32N.tif",
        "S1_COH12_066D_030343_IW1_VH_20150709_20150721_32N.tif",
        "S1_IFG12_CMPLX_015A_030343_IW1_VH_20150709_20150721_32N.tif",
        "S1_IFG12_CMPLX_066D_030343_IW1_VH_20150709_20150721_32N.tif",
    ]

    @pytest.fixture
    def fm(self, tmp_path: Path) -> FileManager:
        """FileManager with stub .tif files in prod_dir."""
        fm = FileManager(tmp_path)
        for name in self._FILENAMES:
            (fm.prod_dir / name).touch()
        return fm

    def test_rtc_by_pass_direction(self, fm: FileManager) -> None:
        ascending = fm.find_rtc_products(pass_direction="Ascending")
        descending = fm.find_rtc_products(pass_direction="Descending")

        assert [p.burst_id for p in ascending] == [30343, 30344]
        assert {p.pass_direction for p in ascending} == {"Ascending"}
        assert [p.rel_orbit_number for p in descending] == [66]

    def test_orbit_and_pass_direction_together(self, fm: FileManager) -> None:
        assert fm.find_rtc_products(rel_orbit_number=15, pass_direction="Descending") == []
        [found] = fm.find_rtc_products(rel_orbit_number=66, pass_direction="Descending")
        assert found.pass_direction == "Descending"

    def test_orbit_alone_matches_either_direction(self, fm: FileManager) -> None:
        assert len(fm.find_rtc_products(rel_orbit_number=15)) == 2

    @pytest.mark.parametrize("direction", ["ascending", "A"])
    def test_an_unknown_pass_direction_filter_is_refused(
        self, fm: FileManager, direction: str
    ) -> None:
        # Previously returned [] without a hint.
        for find in (
            fm.find_rtc_products,
            fm.find_gtc_products,
            fm.find_coherence_products,
            fm.find_interferogram_products,
        ):
            with pytest.raises(ValueError, match="must be 'Ascending' or 'Descending'"):
                # pyrefly: ignore [bad-argument-type]
                find(pass_direction=direction)

    def test_other_per_burst_products(self, fm: FileManager) -> None:
        [gtc] = fm.find_gtc_products(pass_direction="Descending")
        [coh] = fm.find_coherence_products(pass_direction="Ascending", baseline_days=12)
        [ifg] = fm.find_interferogram_products(pass_direction="Descending")

        assert (gtc.rel_orbit_number, coh.rel_orbit_number, ifg.rel_orbit_number) == (66, 15, 66)
        assert len(fm.find_rtc_products()) == 3


class TestLegacyProductNames:
    """Products named before the pass direction joined the orbit are pointed out once."""

    def test_warns_once_per_file_manager(self, tmp_path: Path) -> None:
        fm = FileManager(tmp_path)
        (fm.prod_dir / "S1_RTC_066_030346_IW1_VV_20220105_32N.tif").touch()
        (fm.prod_dir / "S1_RTC_066A_030346_IW1_VV_20220112_32N.tif").touch()

        with pytest.warns(UserWarning, match="named without their pass direction") as record:
            found = fm.find_rtc_products()
        assert len(record) == 1
        assert [p.date for p in found] == [pd.Timestamp("20220112")]

        # A second search on the same FileManager stays quiet.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fm.find_coherence_products()

    def test_current_names_do_not_warn(self, tmp_path: Path) -> None:
        fm = FileManager(tmp_path)
        (fm.prod_dir / "S1_COH12_066A_030346_IW1_VV_20220105_20220117_32N.tif").touch()
        (fm.slc_dir / "S1_SLC_066_030346_IW1_VV_20220105.zip").touch()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert len(fm.find_coherence_products()) == 1
