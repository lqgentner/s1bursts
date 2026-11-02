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
    RTCLRWQA,
    BaseProduct,
    CoherenceProduct,
    CohLRWComposite,
    CohLRWQA,
    CohPassComposite,
    CohPassQA,
    GTCPassComposite,
    GTCPassQA,
    GTCProduct,
    InterferogramProduct,
    QaMetric,
    RTCLRWComposite,
    RTCPassComposite,
    RTCPassQA,
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
        RTCLRWQA,
        "S1_RTC_LRW_QA-NUM_32TMS03_VH_20150706_20151001.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20151001"),
            "qa_metric": QaMetric.NUM,
        },
        id="rtc-lrw-qa-num",
    ),
    pytest.param(
        RTCLRWQA,
        "S1_RTC_LRW_QA-CQM_32TMS03_VH_20150706_20151001.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20151001"),
            "qa_metric": QaMetric.CQM,
        },
        id="rtc-lrw-qa-cqm",
    ),
    pytest.param(
        CohLRWQA,
        "S1_COH12_LRW_QA-NUM_32TMS03_VH_20150706_20150919.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20150919"),
            "baseline_days": 12,
            "qa_metric": QaMetric.NUM,
        },
        id="coh-lrw-qa-num",
    ),
    pytest.param(
        RTCPassQA,
        "S1_RTC_COMP_A_QA-NUM_32TMS03_VH_20150706_20151001.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20151001"),
            "pass_direction": "Ascending",
        },
        id="rtc-pass-qa-num",
    ),
    pytest.param(
        CohPassQA,
        "S1_COH12_COMP_D_QA-NUM_32TMS03_VH_20150706_20150919.tif",
        {
            "mgrs_code": "32TMS03",
            "polarization": "VH",
            "start_date": pd.Timestamp("20150706"),
            "end_date": pd.Timestamp("20150919"),
            "baseline_days": 12,
            "pass_direction": "Descending",
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
        RTCLRWQA,
        "S1_RTC_LRW_QA-NUM_32TMS03_VH_20150706_20151001.tif",
        id="rtc-lrw-qa-num",
    ),
    pytest.param(
        RTCLRWQA,
        "S1_RTC_LRW_QA-CQM_32TMS03_VH_20150706_20151001.tif",
        id="rtc-lrw-qa-cqm",
    ),
    pytest.param(
        CohLRWQA,
        "S1_COH12_LRW_QA-NUM_32TMS03_VH_20150706_20150919.tif",
        id="coh-lrw-qa-num",
    ),
    pytest.param(
        RTCPassQA,
        "S1_RTC_COMP_A_QA-NUM_32TMS03_VH_20150706_20151001.tif",
        id="rtc-pass-qa-num",
    ),
    pytest.param(
        CohPassQA,
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
        assert fm.get_product_path(product).name == filename


# ---------------------------------------------------------------------------
# TestQACollision
# ---------------------------------------------------------------------------


class TestQACollision:
    """QA filenames must not collide with composite parsers or finders."""

    @pytest.fixture
    def fm(self, tmp_path: Path) -> FileManager:
        """FileManager backed by a temporary directory."""
        return FileManager(tmp_path)

    @pytest.mark.parametrize(
        ("composite_cls", "qa_filename"),
        [
            (RTCLRWComposite, "S1_RTC_LRW_QA-NUM_32TMS03_VH_20150706_20151001.tif"),
            (RTCLRWComposite, "S1_RTC_LRW_QA-CQM_32TMS03_VH_20150706_20151001.tif"),
            (CohLRWComposite, "S1_COH12_LRW_QA-NUM_32TMS03_VH_20150706_20150919.tif"),
            (RTCPassComposite, "S1_RTC_COMP_A_QA-NUM_32TMS03_VH_20150706_20151001.tif"),
            (CohPassComposite, "S1_COH12_COMP_D_QA-NUM_32TMS03_VH_20150706_20150919.tif"),
        ],
    )
    def test_qa_does_not_parse_as_composite(
        self,
        fm: FileManager,
        composite_cls: type[BaseProduct],
        qa_filename: str,
    ) -> None:
        """A QA filename must not parse under its sibling composite spec."""
        prefix_re, spec, prefix_captures = _PARSERS[composite_cls]
        assert fm._parse_filename(prefix_re, spec, qa_filename, prefix_captures) is None

    def test_composite_does_not_parse_as_qa(self, fm: FileManager) -> None:
        """A plain composite filename must not parse under a QA spec."""
        prefix_re, spec, prefix_captures = _PARSERS[RTCLRWQA]
        result = fm._parse_filename(
            prefix_re,
            spec,
            "S1_RTC_LRW_32TMS03_VH_20150706_20151001.tif",
            prefix_captures,
        )
        assert result is None

    def test_find_composites_excludes_qa_files(self, fm: FileManager) -> None:
        """QA files in comp_dir must not be returned by the composite finder."""
        (fm.comp_dir / "S1_RTC_LRW_32TMS03_VH_20150706_20151001.tif").touch()
        (fm.comp_dir / "S1_RTC_LRW_QA-NUM_32TMS03_VH_20150706_20151001.tif").touch()
        (fm.comp_dir / "S1_RTC_LRW_QA-CQM_32TMS03_VH_20150706_20151001.tif").touch()
        results = fm.find_rtc_lrw_products()
        assert len(results) == 1
        assert all(isinstance(p, RTCLRWComposite) for p in results)
        assert results[0].mgrs_code == "32TMS03"

    def test_find_qa_excludes_composites(self, fm: FileManager) -> None:
        """The QA finder returns only QA files, not the plain composite."""
        (fm.comp_dir / "S1_RTC_LRW_32TMS03_VH_20150706_20151001.tif").touch()
        (fm.comp_dir / "S1_RTC_LRW_QA-NUM_32TMS03_VH_20150706_20151001.tif").touch()
        (fm.comp_dir / "S1_RTC_LRW_QA-CQM_32TMS03_VH_20150706_20151001.tif").touch()
        results = fm.find_rtc_lrw_qa()
        assert len(results) == 2
        assert {p.qa_metric for p in results} == {QaMetric.NUM, QaMetric.CQM}

    def test_find_qa_filter_by_metric(self, fm: FileManager) -> None:
        """The QA finder can filter by a single metric."""
        (fm.comp_dir / "S1_RTC_LRW_QA-NUM_32TMS03_VH_20150706_20151001.tif").touch()
        (fm.comp_dir / "S1_RTC_LRW_QA-CQM_32TMS03_VH_20150706_20151001.tif").touch()
        results = fm.find_rtc_lrw_qa(qa_metric=QaMetric.CQM)
        assert len(results) == 1
        assert results[0].qa_metric == QaMetric.CQM

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

    qa = GTCPassQA(
        mgrs_code="32TMS25",
        start_date=pd.Timestamp("2019-01-01"),
        end_date=pd.Timestamp("2019-01-31"),
        polarization="VV",
        pass_direction="Ascending",
    )
    qa_path = fm.get_product_path(qa)
    assert qa_path.name == "S1_GTC_COMP_A_QA-NUM_32TMS25_VV_20190101_20190131.tif"
    qa_path.touch()
    assert fm.find_gtc_pass_qa(pass_direction="Ascending") == [qa]
    assert fm.find_gtc_pass_composites() == [product]


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
