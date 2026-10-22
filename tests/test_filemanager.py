"""Tests for FileManager filename parsing, round-trips, and product discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
from pyproj import CRS
import pytest

if TYPE_CHECKING:
    from pathlib import Path
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
        "S1_RTC_015_030343_IW1_VH_20150709_32N.tif",
        {
            "burst_id": 30343,
            "sub_swath": "IW1",
            "polarization": "VH",
            "date": pd.Timestamp("20150709"),
            "crs": _CRS_32N,
        },
        id="rtc",
    ),
    pytest.param(
        GTCProduct,
        "S1_GTC_015_030343_IW1_VH_20150709_32N.tif",
        {
            "burst_id": 30343,
            "sub_swath": "IW1",
            "polarization": "VH",
            "date": pd.Timestamp("20150709"),
            "crs": _CRS_32N,
        },
        id="gtc",
    ),
    pytest.param(
        CoherenceProduct,
        "S1_COH12_015_030345_IW1_VH_20150709_20150721_32N.tif",
        {
            "burst_id": 30345,
            "sub_swath": "IW1",
            "polarization": "VH",
            "date_primary": pd.Timestamp("20150709"),
            "date_secondary": pd.Timestamp("20150721"),
            "crs": _CRS_32N,
        },
        id="coherence",
    ),
    pytest.param(
        InterferogramProduct,
        "S1_IFG12_CMPLX_015_030345_IW1_VH_20150709_20150721_32N.tif",
        {
            "burst_id": 30345,
            "sub_swath": "IW1",
            "polarization": "VH",
            "date_primary": pd.Timestamp("20150709"),
            "date_secondary": pd.Timestamp("20150721"),
            "crs": _CRS_32N,
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

    def test_wrong_type_returns_none(self, fm: FileManager) -> None:
        """An RTC filename does not parse under the CoherenceProduct spec."""
        prefix_re, spec, prefix_captures = _PARSERS[CoherenceProduct]
        result = fm._parse_filename(
            prefix_re,
            spec,
            "S1_RTC_015_030343_IW1_VH_20150709_32N.tif",
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
