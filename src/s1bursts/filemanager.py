"""File management for Sentinel-1 products with xarray integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import re
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, Self

from deepfreezer.geotiff_utils import get_utm_zone_string, utm_zone_to_crs
import pandas as pd

from s1bursts.burst import Burst
from s1bursts.product import (
    RTCLRWQA,
    BaseProduct,
    CoherenceComplexProduct,
    CoherenceProduct,
    CohLRWComposite,
    CohLRWQA,
    CohPassComposite,
    CohPassQA,
    CRSLike,
    GTCProduct,
    InterferogramProduct,
    QaMetric,
    RTCLRWComposite,
    RTCPassComposite,
    RTCPassQA,
    RTCProduct,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from upath import UPath


class _Parseable(Protocol):
    """Metadata class that can be reconstructed from a parsed-filename dict."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: ...


logger = logging.getLogger(__name__)


class FieldSpec(NamedTuple):
    """Encoding of per-field logic for the generic build/parse engine."""

    name: str  # dataclass field name (key in returned dict)
    to_str: Callable[[Any], str]  # value -> filename segment
    pattern: str  # regex for this segment (no capturing groups)
    from_str: Callable[[str], Any]  # str -> parsed value
    group: str = ""  # override regex group name (defaults to name)


# Maps orbit direction letter to full string
_ORBIT_DIRECTION = {"A": "ASCENDING", "D": "DESCENDING"}

# Shared field specs
_ORBIT = FieldSpec("rel_orbit_number", "{:03d}".format, r"\d{3}", int)
_BURST = FieldSpec("burst_id", "{:06d}".format, r"\d{6}", int)
_SWATH = FieldSpec("sub_swath", str, r"IW[1-3]|EW[1-5]", str)
_POL = FieldSpec("polarization", str, r"[A-Z]{2}", str)
_DATE = FieldSpec("date", "{:%Y%m%d}".format, r"\d{8}", pd.Timestamp)
_DATE_P = FieldSpec("date_primary", "{:%Y%m%d}".format, r"\d{8}", pd.Timestamp)
_DATE_S = FieldSpec("date_secondary", "{:%Y%m%d}".format, r"\d{8}", pd.Timestamp)
_UTM = FieldSpec("crs", get_utm_zone_string, r"\d{2}[NS]", utm_zone_to_crs, "utm_zone")
_MGRS = FieldSpec("mgrs_code", str, r"[A-Z0-9]+", str)
_SDATE = FieldSpec("start_date", "{:%Y%m%d}".format, r"\d{8}", pd.Timestamp)
_EDATE = FieldSpec("end_date", "{:%Y%m%d}".format, r"\d{8}", pd.Timestamp)

# Per-product-type field lists
_SLC_SPEC = [_ORBIT, _BURST, _SWATH, _POL, _DATE]
_BACKSCATTER_SPEC = [_ORBIT, _BURST, _SWATH, _POL, _DATE, _UTM]
_COREGISTERED_SPEC = [_ORBIT, _BURST, _SWATH, _POL, _DATE_P, _DATE_S, _UTM]
_MGRS_BASE_SPEC = [_MGRS, _POL, _SDATE, _EDATE]

# Maps each metadata class to its parse arguments: (prefix_re, spec, prefix_captures).
_PARSERS: dict[
    type[_Parseable],
    tuple[str, list[FieldSpec], list[tuple[str, str, Callable]] | None],
] = {
    Burst: ("S1_SLC", _SLC_SPEC, None),
    RTCProduct: ("S1_RTC", _BACKSCATTER_SPEC, None),
    GTCProduct: ("S1_GTC", _BACKSCATTER_SPEC, None),
    CoherenceProduct: (r"S1_COH\d{2}", _COREGISTERED_SPEC, None),
    CoherenceComplexProduct: (r"S1_COH\d{2}_CMPLX", _COREGISTERED_SPEC, None),
    InterferogramProduct: (r"S1_IFG\d{2}_CMPLX", _COREGISTERED_SPEC, None),
    RTCLRWComposite: ("S1_RTC_LRW", _MGRS_BASE_SPEC, None),
    CohLRWComposite: (
        r"S1_COH(?P<baseline_days>\d{2})_LRW",
        _MGRS_BASE_SPEC,
        [("baseline_days", "baseline_days", int)],
    ),
    RTCPassComposite: (
        r"S1_RTC_COMP_(?P<orbit_letter>[AD])",
        _MGRS_BASE_SPEC,
        [("orbit_letter", "orbit_direction", _ORBIT_DIRECTION.__getitem__)],
    ),
    CohPassComposite: (
        r"S1_COH(?P<baseline_days>\d{2})_COMP_(?P<orbit_letter>[AD])",
        _MGRS_BASE_SPEC,
        [
            ("baseline_days", "baseline_days", int),
            ("orbit_letter", "orbit_direction", _ORBIT_DIRECTION.__getitem__),
        ],
    ),
    RTCLRWQA: (
        r"S1_RTC_LRW_QA-(?P<qa_metric>NUM|CQM)",
        _MGRS_BASE_SPEC,
        [("qa_metric", "qa_metric", QaMetric)],
    ),
    CohLRWQA: (
        r"S1_COH(?P<baseline_days>\d{2})_LRW_QA-(?P<qa_metric>NUM|CQM)",
        _MGRS_BASE_SPEC,
        [
            ("baseline_days", "baseline_days", int),
            ("qa_metric", "qa_metric", QaMetric),
        ],
    ),
    RTCPassQA: (
        r"S1_RTC_COMP_(?P<orbit_letter>[AD])_QA-NUM",
        _MGRS_BASE_SPEC,
        [("orbit_letter", "orbit_direction", _ORBIT_DIRECTION.__getitem__)],
    ),
    CohPassQA: (
        r"S1_COH(?P<baseline_days>\d{2})_COMP_(?P<orbit_letter>[AD])_QA-NUM",
        _MGRS_BASE_SPEC,
        [
            ("baseline_days", "baseline_days", int),
            ("orbit_letter", "orbit_direction", _ORBIT_DIRECTION.__getitem__),
        ],
    ),
}


@dataclass
class FileManager:
    """Manages file system operations for Sentinel-1 products.

    All filename generation logic is centralized here.  Storage directories
    (``base_dir``, ``slc_dir``, …) may point to either local paths or S3
    locations via :class:`~upath.UPath`.
    """

    base_dir: Path | UPath
    """Root directory for all products (local ``Path`` or S3 ``UPath``)."""
    slc_dir: Path | UPath
    """Directory for SLC burst files."""
    prod_dir: Path | UPath
    """Directory for processed products."""
    comp_dir: Path | UPath
    """Directory for composites."""
    cog_dir: Path | UPath
    """Directory for cloud-optimized GeoTIFFs (COGs)."""

    def __init__(
        self,
        base_dir: Path | UPath,
        slc_dir: Path | UPath | None = None,
        prod_dir: Path | UPath | None = None,
        comp_dir: Path | UPath | None = None,
        cog_dir: Path | UPath | None = None,
    ) -> None:
        """Initialize FileManager with directory structure.

        Parameters
        ----------
        base_dir : Path or UPath
            Root directory for all products.  May be a local ``Path`` or a
            ``UPath`` pointing to an S3 bucket.
        slc_dir : Path or UPath, optional
            Directory for SLC burst files. Defaults to ``base_dir/s1-slc``.
        prod_dir : Path or UPath, optional
            Directory for processed products. Defaults to ``base_dir/s1-prod``.
        comp_dir : Path or UPath, optional
            Directory for composites. Defaults to ``base_dir/s1-comp``.
        cog_dir : Path or UPath, optional
            Directory for cloud-optimized GeoTIFFs (COGs). Defaults to ``base_dir/s1-cog``.
        """
        self.base_dir = base_dir

        # Set default subdirectories if not provided
        self.slc_dir = slc_dir or base_dir / "s1-slc"
        self.prod_dir = prod_dir or base_dir / "s1-prod"
        self.comp_dir = comp_dir or base_dir / "s1-comp"
        self.cog_dir = cog_dir or base_dir / "s1-cog"

        # Create all directories (UPath.mkdir on S3 is a no-op)
        for directory in [self.slc_dir, self.prod_dir, self.comp_dir, self.cog_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    # ============ SLC Operations ============

    def get_slc_path(self, burst: Burst) -> Path | UPath:
        """Generate the file path for an SLC burst.

        Parameters
        ----------
        burst : Burst
            Burst to generate the path for.

        Returns
        -------
        Path
            Expected path to the SLC zip file.
        """
        return self.slc_dir / self._build_pattern(
            "S1_SLC",
            _SLC_SPEC,
            suffix=".zip",
            rel_orbit_number=burst.rel_orbit_number,
            burst_id=burst.burst_id,
            sub_swath=burst.sub_swath,
            polarization=burst.polarization,
            date=burst.date,
        )

    def find_slc_bursts(
        self,
        rel_orbit_number: int | None = None,
        burst_id: int | None = None,
        sub_swath: str | None = None,
        polarization: str | None = None,
    ) -> list[Burst]:
        """Find SLC bursts matching criteria.

        Note: Date filtering is not supported. Use list comprehension on results
        if date filtering is needed.

        Parameters
        ----------
        rel_orbit_number : int, optional
            Filter by relative orbit number.
        burst_id : int, optional
            Filter by burst ID.
        sub_swath : str, optional
            Filter by sub-swath.
        polarization : str, optional
            Filter by polarization.

        Returns
        -------
        list[Burst]
            List of matching SLC bursts.
        """
        pattern = self._build_pattern(
            "S1_SLC",
            _SLC_SPEC,
            suffix=".zip",
            rel_orbit_number=rel_orbit_number,
            burst_id=burst_id,
            sub_swath=sub_swath,
            polarization=polarization,
        )
        return self._find_products(self.slc_dir, pattern, Burst, suffix=".zip")

    # ============ Generic Build/Parse Engine ============

    def _build_pattern(
        self,
        prefix: str,
        spec: list[FieldSpec],
        *,
        suffix: str = ".tif",
        **values: Any,
    ) -> str:
        """Build a glob pattern string from a prefix and field spec.

        Parameters
        ----------
        prefix : str
            Product type prefix (e.g. ``"S1_RTC"``).
        spec : list[FieldSpec]
            Ordered list of field specs for the filename segments after the prefix.
        suffix : str, default ".tif"
            File suffix, including the leading dot.
        **values
            Known field values; missing keys produce ``"*"`` wildcard segments.

        Returns
        -------
        str
            Glob pattern with the given suffix.
        """
        parts = [prefix]
        for f in spec:
            v = values.get(f.name)
            parts.append(f.to_str(v) if v is not None else "*")
        return "_".join(parts) + suffix

    def _parse_filename(
        self,
        prefix_re: str,
        spec: list[FieldSpec],
        filename: str,
        prefix_captures: list[tuple[str, str, Callable]] | None = None,
        *,
        suffix: str = ".tif",
    ) -> dict | None:
        """Parse a product filename using a prefix regex and field spec.

        Parameters
        ----------
        prefix_re : str
            Regex for the prefix portion of the filename (may contain named groups).
        spec : list[FieldSpec]
            Ordered list of field specs for the segments after the prefix.
        filename : str
            Filename to parse (basename only, including extension).
        prefix_captures : list of (group_name, field_name, converter), optional
            Extra named groups captured inside ``prefix_re`` that should be
            included in the result dict.  Each tuple is
            ``(regex_group_name, output_key, converter_callable)``.
        suffix : str, default ".tif"
            File suffix, including the leading dot.

        Returns
        -------
        dict | None
            Parsed field values, or ``None`` if the filename does not match.
        """
        groups = "_".join(f"(?P<{f.group or f.name}>{f.pattern})" for f in spec)
        match = re.fullmatch(rf"{prefix_re}_{groups}{re.escape(suffix)}", filename)
        if not match:
            return None
        result = {f.name: f.from_str(match.group(f.group or f.name)) for f in spec}
        if prefix_captures:
            for group_name, field_name, converter in prefix_captures:
                result[field_name] = converter(match.group(group_name))
        return result

    def _find_products[T: _Parseable](
        self,
        directory: Path | UPath,
        pattern: str,
        cls: type[T],
        *,
        suffix: str = ".tif",
    ) -> list[T]:
        """Glob a directory, parse matching filenames, and instantiate products.

        Parameters
        ----------
        directory : Path | UPath
            Directory to search.
        pattern : str
            Glob pattern (filename only, no directory).
        cls : type[_Parseable]
            Metadata class to instantiate; must have an entry in ``_PARSERS``.
        suffix : str, default ".tif"
            File suffix used to anchor the parser regex.

        Returns
        -------
        list[_Parseable]
            Sorted list of metadata instances.
        """
        prefix_re, spec, prefix_captures = _PARSERS[cls]
        files: list[Path | UPath] = sorted(directory.glob(pattern), key=lambda p: p.name)
        return [
            cls.from_dict(m)
            for f in files
            if (m := self._parse_filename(prefix_re, spec, f.name, prefix_captures, suffix=suffix))
            is not None
        ]

    # ============ RTC Product Operations ============

    def _get_rtc_path(self, product: RTCProduct) -> Path | UPath:
        """Generate the file path for an RTC backscatter product.

        Parameters
        ----------
        product : RTCProduct
            RTC product metadata.

        Returns
        -------
        Path
            Expected path to the RTC GeoTIFF file.
        """
        return self.prod_dir / self._build_pattern("S1_RTC", _BACKSCATTER_SPEC, **asdict(product))

    def find_rtc_products(
        self,
        rel_orbit_number: int | None = None,
        burst_id: int | None = None,
        sub_swath: str | None = None,
        polarization: str | None = None,
        crs: CRSLike | None = None,
    ) -> list[RTCProduct]:
        """Find RTC backscatter products matching criteria.

        Note: Date filtering is not supported. Use list comprehension on results
        if date filtering is needed.

        Parameters
        ----------
        rel_orbit_number : int, optional
            Filter by relative orbit number.
        burst_id : int, optional
            Filter by burst ID.
        sub_swath : str, optional
            Filter by sub-swath.
        polarization : str, optional
            Filter by polarization.
        crs : CRSLike, optional
            Filter by coordinate reference system (UTM zone).

        Returns
        -------
        list[RTCProduct]
            List of matching RTC products.
        """
        pattern = self._build_pattern(
            "S1_RTC",
            _BACKSCATTER_SPEC,
            rel_orbit_number=rel_orbit_number,
            burst_id=burst_id,
            sub_swath=sub_swath,
            polarization=polarization,
            crs=crs,
        )
        return self._find_products(self.prod_dir, pattern, RTCProduct)

    # ============ GTC Product Operations ============

    def _get_gtc_path(self, product: GTCProduct) -> Path | UPath:
        """Generate the file path for a GTC backscatter product.

        Parameters
        ----------
        product : GTCProduct
            GTC product metadata.

        Returns
        -------
        Path
            Expected path to the GTC GeoTIFF file.
        """
        return self.prod_dir / self._build_pattern("S1_GTC", _BACKSCATTER_SPEC, **asdict(product))

    def find_gtc_products(
        self,
        rel_orbit_number: int | None = None,
        burst_id: int | None = None,
        sub_swath: str | None = None,
        polarization: str | None = None,
        crs: CRSLike | None = None,
    ) -> list[GTCProduct]:
        """Find GTC backscatter products matching criteria.

        Note: Date filtering is not supported. Use list comprehension on results
        if date filtering is needed.

        Parameters
        ----------
        rel_orbit_number : int, optional
            Filter by relative orbit number.
        burst_id : int, optional
            Filter by burst ID.
        sub_swath : str, optional
            Filter by sub-swath.
        polarization : str, optional
            Filter by polarization.
        crs : CRSLike, optional
            Filter by coordinate reference system (UTM zone).

        Returns
        -------
        list[GTCProduct]
            List of matching GTC products.
        """
        pattern = self._build_pattern(
            "S1_GTC",
            _BACKSCATTER_SPEC,
            rel_orbit_number=rel_orbit_number,
            burst_id=burst_id,
            sub_swath=sub_swath,
            polarization=polarization,
            crs=crs,
        )
        return self._find_products(self.prod_dir, pattern, GTCProduct)

    # ============ Coherence Product Operations ============

    def _get_coherence_path(self, product: CoherenceProduct) -> Path | UPath:
        """Generate the file path for a coherence product.

        Parameters
        ----------
        product : CoherenceProduct
            Coherence product metadata.

        Returns
        -------
        Path
            Expected path to the coherence GeoTIFF file.
        """
        prefix = f"S1_COH{product.baseline_days:02d}"
        return self.prod_dir / self._build_pattern(prefix, _COREGISTERED_SPEC, **asdict(product))

    def find_coherence_products(
        self,
        rel_orbit_number: int | None = None,
        burst_id: int | None = None,
        sub_swath: str | None = None,
        polarization: str | None = None,
        baseline_days: int | None = None,
        crs: CRSLike | None = None,
    ) -> list[CoherenceProduct]:
        """Find coherence products matching criteria.

        Note: Date filtering is not supported. Use list comprehension on results
        if date filtering is needed.

        Parameters
        ----------
        rel_orbit_number : int, optional
            Filter by relative orbit number.
        burst_id : int, optional
            Filter by burst ID.
        sub_swath : str, optional
            Filter by sub-swath.
        polarization : str, optional
            Filter by polarization.
        baseline_days : int, optional
            Filter by temporal baseline.
        crs : CRSLike, optional
            Filter by coordinate reference system (UTM zone).

        Returns
        -------
        list[CoherenceProduct]
            List of matching coherence products.
        """
        bd = f"{baseline_days:02d}" if baseline_days is not None else "*"
        pattern = self._build_pattern(
            f"S1_COH{bd}",
            _COREGISTERED_SPEC,
            rel_orbit_number=rel_orbit_number,
            burst_id=burst_id,
            sub_swath=sub_swath,
            polarization=polarization,
            crs=crs,
        )
        return self._find_products(self.prod_dir, pattern, CoherenceProduct)

    # ============ Complex Coherence Product Operations ============

    def _get_coherence_complex_path(self, product: CoherenceComplexProduct) -> Path | UPath:
        """Generate the file path for a complex coherence product.

        Parameters
        ----------
        product : CoherenceComplexProduct
            Complex coherence product metadata.

        Returns
        -------
        Path
            Expected path to the coherence GeoTIFF file.
        """
        prefix = f"S1_COH{product.baseline_days:02d}_CMPLX"
        return self.prod_dir / self._build_pattern(prefix, _COREGISTERED_SPEC, **asdict(product))

    def find_coherence_complex_products(
        self,
        rel_orbit_number: int | None = None,
        burst_id: int | None = None,
        sub_swath: str | None = None,
        polarization: str | None = None,
        baseline_days: int | None = None,
        crs: CRSLike | None = None,
    ) -> list[CoherenceComplexProduct]:
        """Find complex coherence products matching criteria.

        Note: Date filtering is not supported. Use list comprehension on results
        if date filtering is needed.

        Parameters
        ----------
        rel_orbit_number : int, optional
            Filter by relative orbit number.
        burst_id : int, optional
            Filter by burst ID.
        sub_swath : str, optional
            Filter by sub-swath.
        polarization : str, optional
            Filter by polarization.
        baseline_days : int, optional
            Filter by temporal baseline.
        crs : CRSLike, optional
            Filter by coordinate reference system (UTM zone).

        Returns
        -------
        list[CoherenceComplexProduct]
            List of matching complex coherence products.
        """
        bd = f"{baseline_days:02d}" if baseline_days is not None else "*"
        pattern = self._build_pattern(
            f"S1_COH{bd}_CMPLX",
            _COREGISTERED_SPEC,
            rel_orbit_number=rel_orbit_number,
            burst_id=burst_id,
            sub_swath=sub_swath,
            polarization=polarization,
            crs=crs,
        )
        return self._find_products(self.prod_dir, pattern, CoherenceComplexProduct)

    # ============ Interferogram Product Operations ============

    def _get_interferogram_path(self, product: InterferogramProduct) -> Path | UPath:
        """Generate the file path for an interferogram product.

        Parameters
        ----------
        product : InterferogramProduct
            Interferogram product metadata.

        Returns
        -------
        Path
            Expected path to the interferogram GeoTIFF file.
        """
        prefix = f"S1_IFG{product.baseline_days:02d}_CMPLX"
        return self.prod_dir / self._build_pattern(prefix, _COREGISTERED_SPEC, **asdict(product))

    def find_interferogram_products(
        self,
        rel_orbit_number: int | None = None,
        burst_id: int | None = None,
        sub_swath: str | None = None,
        polarization: str | None = None,
        baseline_days: int | None = None,
        crs: CRSLike | None = None,
    ) -> list[InterferogramProduct]:
        """Find interferogram products matching criteria.

        Note: Date filtering is not supported. Use list comprehension on results
        if date filtering is needed.

        Parameters
        ----------
        rel_orbit_number : int, optional
            Filter by relative orbit number.
        burst_id : int, optional
            Filter by burst ID.
        sub_swath : str, optional
            Filter by sub-swath.
        polarization : str, optional
            Filter by polarization.
        baseline_days : int, optional
            Filter by temporal baseline.
        crs : CRSLike, optional
            Filter by coordinate reference system (UTM zone).

        Returns
        -------
        list[InterferogramProduct]
            List of matching interferogram products.
        """
        bd = f"{baseline_days:02d}" if baseline_days is not None else "*"
        pattern = self._build_pattern(
            f"S1_IFG{bd}_CMPLX",
            _COREGISTERED_SPEC,
            rel_orbit_number=rel_orbit_number,
            burst_id=burst_id,
            sub_swath=sub_swath,
            polarization=polarization,
            crs=crs,
        )
        return self._find_products(self.prod_dir, pattern, InterferogramProduct)

    # ============ LRW Product Operations ============

    def _get_rtc_lrw_path(self, product: RTCLRWComposite) -> Path | UPath:
        """Generate the file path for an RTC LRW intensity composite product.

        Parameters
        ----------
        product : RTCLRWComposite
            RTC LRW product metadata.

        Returns
        -------
        Path
            Expected path to the RTC LRW GeoTIFF file.
        """
        return self.comp_dir / self._build_pattern(
            "S1_RTC_LRW",
            _MGRS_BASE_SPEC,
            **asdict(product),
        )

    def find_rtc_lrw_products(
        self,
        mgrs_code: str | None = None,
        polarization: str | None = None,
    ) -> list[RTCLRWComposite]:
        """Find RTC LRW intensity composite products matching criteria.

        Parameters
        ----------
        mgrs_code : str, optional
            Filter by MGRS grid cell identifier.
        polarization : str, optional
            Filter by polarization.

        Returns
        -------
        list[RTCLRWComposite]
            List of matching RTC LRW products.
        """
        pattern = self._build_pattern(
            "S1_RTC_LRW",
            _MGRS_BASE_SPEC,
            mgrs_code=mgrs_code,
            polarization=polarization,
        )
        return self._find_products(self.comp_dir, pattern, RTCLRWComposite)

    def _get_coh_lrw_path(self, product: CohLRWComposite) -> Path | UPath:
        """Generate the file path for a coherence LRW composite product.

        Parameters
        ----------
        product : CohLRWComposite
            Coherence LRW product metadata.

        Returns
        -------
        Path
            Expected path to the coherence LRW GeoTIFF file.
        """
        prefix = f"S1_COH{product.baseline_days:02d}_LRW"
        return self.comp_dir / self._build_pattern(prefix, _MGRS_BASE_SPEC, **asdict(product))

    def find_coh_lrw_products(
        self,
        mgrs_code: str | None = None,
        polarization: str | None = None,
        baseline_days: int | None = None,
    ) -> list[CohLRWComposite]:
        """Find coherence LRW composite products matching criteria.

        Parameters
        ----------
        mgrs_code : str, optional
            Filter by MGRS grid cell identifier.
        polarization : str, optional
            Filter by polarization.
        baseline_days : int, optional
            Filter by temporal baseline in days.

        Returns
        -------
        list[CohLRWComposite]
            List of matching coherence LRW products.
        """
        bd = f"{baseline_days:02d}" if baseline_days is not None else "*"
        pattern = self._build_pattern(
            f"S1_COH{bd}_LRW",
            _MGRS_BASE_SPEC,
            mgrs_code=mgrs_code,
            polarization=polarization,
        )
        return self._find_products(self.comp_dir, pattern, CohLRWComposite)

    # ============ Pass-Composite Product Operations ============

    def _get_rtc_pass_composite_path(self, product: RTCPassComposite) -> Path | UPath:
        """Generate the file path for an RTC pass-composite product.

        Parameters
        ----------
        product : RTCPassComposite
            RTC pass-composite product metadata.

        Returns
        -------
        Path
            Expected path to the RTC pass-composite GeoTIFF file.
        """
        prefix = f"S1_RTC_COMP_{product.orbit_direction[0]}"
        return self.comp_dir / self._build_pattern(prefix, _MGRS_BASE_SPEC, **asdict(product))

    def find_rtc_pass_composites(
        self,
        mgrs_code: str | None = None,
        polarization: str | None = None,
        orbit_direction: str | None = None,
    ) -> list[RTCPassComposite]:
        """Find RTC pass-composite products matching criteria.

        Parameters
        ----------
        mgrs_code : str, optional
            Filter by MGRS grid cell identifier.
        polarization : str, optional
            Filter by polarization.
        orbit_direction : str, optional
            Filter by orbit direction (``"ASCENDING"`` or ``"DESCENDING"``).

        Returns
        -------
        list[RTCPassComposite]
            List of matching RTC pass-composite products.
        """
        od = orbit_direction[0] if orbit_direction is not None else "*"
        pattern = self._build_pattern(
            f"S1_RTC_COMP_{od}",
            _MGRS_BASE_SPEC,
            mgrs_code=mgrs_code,
            polarization=polarization,
        )
        return self._find_products(self.comp_dir, pattern, RTCPassComposite)

    def _get_coh_pass_composite_path(self, product: CohPassComposite) -> Path | UPath:
        """Generate the file path for a coherence pass-composite product.

        Parameters
        ----------
        product : CohPassComposite
            Coherence pass-composite product metadata.

        Returns
        -------
        Path
            Expected path to the coherence pass-composite GeoTIFF file.
        """
        prefix = f"S1_COH{product.baseline_days:02d}_COMP_{product.orbit_direction[0]}"
        return self.comp_dir / self._build_pattern(prefix, _MGRS_BASE_SPEC, **asdict(product))

    def find_coh_pass_composites(
        self,
        mgrs_code: str | None = None,
        polarization: str | None = None,
        orbit_direction: str | None = None,
        baseline_days: int | None = None,
    ) -> list[CohPassComposite]:
        """Find coherence pass-composite products matching criteria.

        Parameters
        ----------
        mgrs_code : str, optional
            Filter by MGRS grid cell identifier.
        polarization : str, optional
            Filter by polarization.
        orbit_direction : str, optional
            Filter by orbit direction (``"ASCENDING"`` or ``"DESCENDING"``).
        baseline_days : int, optional
            Filter by temporal baseline in days.

        Returns
        -------
        list[CohPassComposite]
            List of matching coherence pass-composite products.
        """
        bd = f"{baseline_days:02d}" if baseline_days is not None else "*"
        od = orbit_direction[0] if orbit_direction is not None else "*"
        pattern = self._build_pattern(
            f"S1_COH{bd}_COMP_{od}",
            _MGRS_BASE_SPEC,
            mgrs_code=mgrs_code,
            polarization=polarization,
        )
        return self._find_products(self.comp_dir, pattern, CohPassComposite)

    # ============ QA Auxiliary Product Operations ============

    def _get_rtc_lrw_qa_path(self, product: RTCLRWQA) -> Path | UPath:
        """Generate the file path for an RTC LRW QA auxiliary product."""
        prefix = f"S1_RTC_LRW_QA-{product.qa_metric.value}"
        return self.comp_dir / self._build_pattern(prefix, _MGRS_BASE_SPEC, **asdict(product))

    def _get_coh_lrw_qa_path(self, product: CohLRWQA) -> Path | UPath:
        """Generate the file path for a coherence LRW QA auxiliary product."""
        prefix = f"S1_COH{product.baseline_days:02d}_LRW_QA-{product.qa_metric.value}"
        return self.comp_dir / self._build_pattern(prefix, _MGRS_BASE_SPEC, **asdict(product))

    def _get_rtc_pass_qa_path(self, product: RTCPassQA) -> Path | UPath:
        """Generate the file path for an RTC pass-composite QA auxiliary product."""
        prefix = f"S1_RTC_COMP_{product.orbit_direction[0]}_QA-NUM"
        return self.comp_dir / self._build_pattern(prefix, _MGRS_BASE_SPEC, **asdict(product))

    def _get_coh_pass_qa_path(self, product: CohPassQA) -> Path | UPath:
        """Generate the file path for a coherence pass-composite QA auxiliary product."""
        prefix = f"S1_COH{product.baseline_days:02d}_COMP_{product.orbit_direction[0]}_QA-NUM"
        return self.comp_dir / self._build_pattern(prefix, _MGRS_BASE_SPEC, **asdict(product))

    def find_rtc_lrw_qa(
        self,
        mgrs_code: str | None = None,
        polarization: str | None = None,
        qa_metric: QaMetric | None = None,
    ) -> list[RTCLRWQA]:
        """Find RTC LRW QA auxiliary products matching criteria.

        Parameters
        ----------
        mgrs_code : str, optional
            Filter by MGRS grid cell identifier.
        polarization : str, optional
            Filter by polarization.
        qa_metric : QaMetric, optional
            Filter by quality metric (number of observations or CQM).

        Returns
        -------
        list[RTCLRWQA]
            List of matching RTC LRW QA products.
        """
        metric = qa_metric.value if qa_metric is not None else "*"
        pattern = self._build_pattern(
            f"S1_RTC_LRW_QA-{metric}",
            _MGRS_BASE_SPEC,
            mgrs_code=mgrs_code,
            polarization=polarization,
        )
        return self._find_products(self.comp_dir, pattern, RTCLRWQA)

    def find_coh_lrw_qa(
        self,
        mgrs_code: str | None = None,
        polarization: str | None = None,
        baseline_days: int | None = None,
        qa_metric: QaMetric | None = None,
    ) -> list[CohLRWQA]:
        """Find coherence LRW QA auxiliary products matching criteria.

        Parameters
        ----------
        mgrs_code : str, optional
            Filter by MGRS grid cell identifier.
        polarization : str, optional
            Filter by polarization.
        baseline_days : int, optional
            Filter by temporal baseline in days.
        qa_metric : QaMetric, optional
            Filter by quality metric (number of observations or CQM).

        Returns
        -------
        list[CohLRWQA]
            List of matching coherence LRW QA products.
        """
        bd = f"{baseline_days:02d}" if baseline_days is not None else "*"
        metric = qa_metric.value if qa_metric is not None else "*"
        pattern = self._build_pattern(
            f"S1_COH{bd}_LRW_QA-{metric}",
            _MGRS_BASE_SPEC,
            mgrs_code=mgrs_code,
            polarization=polarization,
        )
        return self._find_products(self.comp_dir, pattern, CohLRWQA)

    def find_rtc_pass_qa(
        self,
        mgrs_code: str | None = None,
        polarization: str | None = None,
        orbit_direction: str | None = None,
    ) -> list[RTCPassQA]:
        """Find RTC pass-composite QA (number-of-observations) products matching criteria.

        Parameters
        ----------
        mgrs_code : str, optional
            Filter by MGRS grid cell identifier.
        polarization : str, optional
            Filter by polarization.
        orbit_direction : str, optional
            Filter by orbit direction (``"ASCENDING"`` or ``"DESCENDING"``).

        Returns
        -------
        list[RTCPassQA]
            List of matching RTC pass-composite QA products.
        """
        od = orbit_direction[0] if orbit_direction is not None else "*"
        pattern = self._build_pattern(
            f"S1_RTC_COMP_{od}_QA-NUM",
            _MGRS_BASE_SPEC,
            mgrs_code=mgrs_code,
            polarization=polarization,
        )
        return self._find_products(self.comp_dir, pattern, RTCPassQA)

    def find_coh_pass_qa(
        self,
        mgrs_code: str | None = None,
        polarization: str | None = None,
        orbit_direction: str | None = None,
        baseline_days: int | None = None,
    ) -> list[CohPassQA]:
        """Find coherence pass-composite QA (number-of-observations) products matching criteria.

        Parameters
        ----------
        mgrs_code : str, optional
            Filter by MGRS grid cell identifier.
        polarization : str, optional
            Filter by polarization.
        orbit_direction : str, optional
            Filter by orbit direction (``"ASCENDING"`` or ``"DESCENDING"``).
        baseline_days : int, optional
            Filter by temporal baseline in days.

        Returns
        -------
        list[CohPassQA]
            List of matching coherence pass-composite QA products.
        """
        bd = f"{baseline_days:02d}" if baseline_days is not None else "*"
        od = orbit_direction[0] if orbit_direction is not None else "*"
        pattern = self._build_pattern(
            f"S1_COH{bd}_COMP_{od}_QA-NUM",
            _MGRS_BASE_SPEC,
            mgrs_code=mgrs_code,
            polarization=polarization,
        )
        return self._find_products(self.comp_dir, pattern, CohPassQA)

    # ============ Generic Product Path Getter ============

    def get_product_path(self, product: BaseProduct) -> Path | UPath:  # noqa: C901, PLR0912
        """
        Get path for any product type.

        Parameters
        ----------
        product : BaseProduct
            Product to get path for.

        Returns
        -------
        Path
            Path to the product file.

        Raises
        ------
        ValueError
            If product type is unknown.
        """
        match product:
            case RTCProduct():
                path = self._get_rtc_path(product)
            case GTCProduct():
                path = self._get_gtc_path(product)
            case CoherenceComplexProduct():
                path = self._get_coherence_complex_path(product)
            case CoherenceProduct():
                path = self._get_coherence_path(product)
            case InterferogramProduct():
                path = self._get_interferogram_path(product)
            case RTCLRWComposite():
                path = self._get_rtc_lrw_path(product)
            case CohLRWComposite():
                path = self._get_coh_lrw_path(product)
            case RTCPassComposite():
                path = self._get_rtc_pass_composite_path(product)
            case CohPassComposite():
                path = self._get_coh_pass_composite_path(product)
            case RTCLRWQA():
                path = self._get_rtc_lrw_qa_path(product)
            case CohLRWQA():
                path = self._get_coh_lrw_qa_path(product)
            case RTCPassQA():
                path = self._get_rtc_pass_qa_path(product)
            case CohPassQA():
                path = self._get_coh_pass_qa_path(product)
            case _:
                msg = f"Unknown product type '{type(product)}'."
                raise ValueError(msg)
        return path

    def product_exists(self, product: BaseProduct) -> bool:
        """Check if a product file exists (works for both local and S3).

        Parameters
        ----------
        product : BaseProduct
            Product to check.

        Returns
        -------
        bool
            True if the product file exists.
        """
        return self.get_product_path(product).exists()
