"""Deterministic paths and discovery for Sentinel-1 products, local or on S3."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import logging
from operator import attrgetter, itemgetter
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, Self
import warnings

from freezebase.raster import get_utm_zone_string, utm_zone_to_crs
import pandas as pd

from s1bursts.burst import Burst
from s1bursts.product import (
    BaseProduct,
    CoherenceProduct,
    CohLRWComposite,
    CohPassComposite,
    CRSInput,
    GTCPassComposite,
    GTCProduct,
    InterferogramProduct,
    Layer,
    MGRSTileProduct,
    RTCLRWComposite,
    RTCPassComposite,
    RTCProduct,
    _check_pass_direction,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from upath import UPath

    from s1bursts.query import PassDirection


class _Parseable(Protocol):
    """Metadata class that can be reconstructed from a parsed-filename dict."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: ...


logger = logging.getLogger(__name__)


# Two or more scheme characters, so a Windows drive such as "C://data" stays local.
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+://")


def _as_path(directory: str | Path | UPath) -> Path | UPath:
    """Turn a string into a local ``Path``; remote storage needs a ``UPath``."""
    if not isinstance(directory, str):
        return directory
    # An empty string would silently mean the working directory.
    if not directory:
        msg = "A directory must not be an empty string."
        raise ValueError(msg)
    # A bare URL carries no credentials or endpoint, and as a Path it would
    # silently become a local folder named after the scheme.
    if _URL_SCHEME.match(directory):
        msg = (
            f"{directory!r} is a URL. Pass a UPath instead; for S3, one from "
            "freezebase.s3.make_s3_upath() carries the credentials and endpoint."
        )
        raise ValueError(msg)
    return Path(directory).expanduser()


class FieldSpec(NamedTuple):
    """Encoding of per-field logic for the generic build/parse engine."""

    name: str  # dataclass field name (key in returned dict)
    to_str: Callable[[Any], str]  # value -> filename segment
    pattern: str  # regex for this segment (no capturing groups)
    from_str: Callable[[str], Any]  # str -> parsed value
    group: str = ""  # override regex group name (defaults to name)
    wildcard: str = "*"  # glob for this segment when its value is not given
    sep: str = "_"  # joins this segment to the one before it


# Maps pass direction letter to full string
_PASS_DIRECTION = {"A": "Ascending", "D": "Descending"}

# Shared field specs
_ORBIT = FieldSpec("rel_orbit_number", "{:03d}".format, r"\d{3}", int)
# Per-burst products append the pass direction to the orbit, "066A", keeping the
# track in one token. Fixed-width wildcards, so that filtering on one half of the
# token cannot match across the separators of a neighbouring one.
_TRACK_ORBIT = _ORBIT._replace(wildcard="???")
_PASS = FieldSpec(
    "pass_direction",
    itemgetter(0),
    "[AD]",
    _PASS_DIRECTION.__getitem__,
    wildcard="?",
    sep="",
)
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
_BACKSCATTER_SPEC = [_TRACK_ORBIT, _PASS, _BURST, _SWATH, _POL, _DATE, _UTM]
_COREGISTERED_SPEC = [_TRACK_ORBIT, _PASS, _BURST, _SWATH, _POL, _DATE_P, _DATE_S, _UTM]
_MGRS_BASE_SPEC = [_MGRS, _POL, _SDATE, _EDATE]

# A composite's QA layer adds its token after the type prefix; the measurement adds none.
_LRW_LAYER_RE = r"(?:_(?P<layer>QA-NUM|QA-CQM))?"
_UNWEIGHTED_LAYER_RE = r"(?:_(?P<layer>QA-NUM))?"
_MEASUREMENT_ONLY = frozenset({Layer.MEASUREMENT})


def _layers_from_token(token: str | None) -> frozenset[Layer]:
    """Return the layer that a composite file name's QA token, or its absence, stands for."""
    return frozenset({Layer(token)}) if token else _MEASUREMENT_ONLY


_LAYER_CAPTURE: tuple[str, str, Callable[[str], Any]] = ("layer", "layers", _layers_from_token)

# Maps each metadata class to its parse arguments: (prefix_re, spec, prefix_captures).
_PARSERS: dict[
    type[_Parseable],
    tuple[str, list[FieldSpec], list[tuple[str, str, Callable[[str], Any]]] | None],
] = {
    Burst: ("S1_SLC", _SLC_SPEC, None),
    RTCProduct: ("S1_RTC", _BACKSCATTER_SPEC, None),
    GTCProduct: ("S1_GTC", _BACKSCATTER_SPEC, None),
    CoherenceProduct: (r"S1_COH\d{2}", _COREGISTERED_SPEC, None),
    InterferogramProduct: (r"S1_IFG\d{2}_CMPLX", _COREGISTERED_SPEC, None),
    RTCLRWComposite: (rf"S1_RTC_LRW{_LRW_LAYER_RE}", _MGRS_BASE_SPEC, [_LAYER_CAPTURE]),
    CohLRWComposite: (
        rf"S1_COH(?P<baseline_days>\d{{2}})_LRW{_LRW_LAYER_RE}",
        _MGRS_BASE_SPEC,
        [("baseline_days", "baseline_days", int), _LAYER_CAPTURE],
    ),
    RTCPassComposite: (
        rf"S1_RTC_COMP_(?P<orbit_letter>[AD]){_UNWEIGHTED_LAYER_RE}",
        _MGRS_BASE_SPEC,
        [("orbit_letter", "pass_direction", _PASS_DIRECTION.__getitem__), _LAYER_CAPTURE],
    ),
    GTCPassComposite: (
        rf"S1_GTC_COMP_(?P<orbit_letter>[AD]){_UNWEIGHTED_LAYER_RE}",
        _MGRS_BASE_SPEC,
        [("orbit_letter", "pass_direction", _PASS_DIRECTION.__getitem__), _LAYER_CAPTURE],
    ),
    CohPassComposite: (
        rf"S1_COH(?P<baseline_days>\d{{2}})_COMP_(?P<orbit_letter>[AD]){_UNWEIGHTED_LAYER_RE}",
        _MGRS_BASE_SPEC,
        [
            ("baseline_days", "baseline_days", int),
            ("orbit_letter", "pass_direction", _PASS_DIRECTION.__getitem__),
            _LAYER_CAPTURE,
        ],
    ),
}


def _checked_pass_direction(value: PassDirection | None) -> PassDirection | None:
    """Return a ``pass_direction`` filter after rejecting a value no file name holds."""
    if value is not None:
        _check_pass_direction(value)
    return value


def _pass_letter(value: PassDirection | None) -> str:
    """Return the composite file name letter of a pass direction, or a wildcard for any."""
    checked = _checked_pass_direction(value)
    return checked[0] if checked is not None else "?"


# A per-burst product named before the pass direction joined the relative orbit.
_LEGACY_PRODUCT_NAME = re.compile(
    r"S1_(?:RTC|GTC|COH\d{2}|IFG\d{2}_CMPLX)_\d{3}_\d{6}_(?:IW|EW)\d_.*\.tif"
)
_CHANGELOG_URL = "https://github.com/lqgentner/s1bursts/blob/main/CHANGELOG.md"


@dataclass
class FileManager:
    """Manages file system operations for Sentinel-1 products.

    All filename generation logic is centralized here.  Storage directories
    (``base_dir``, ``slc_dir``, …) may point to either local paths or S3
    locations via ``upath.UPath``. A directory given as a string is read as a
    local path, with ``~`` expanded.

    Parameters
    ----------
    base_dir
        Root directory for all products.  May be a local ``Path`` or a
        ``UPath`` pointing to an S3 bucket.
    slc_dir
        Directory for SLC burst files. Defaults to ``base_dir/s1-slc``.
    prod_dir
        Directory for processed products. Defaults to ``base_dir/s1-prod``.
    comp_dir
        Directory for composites. Defaults to ``base_dir/s1-comp``.

    Raises
    ------
    ValueError
        If a directory is given as an empty string, or as a URL string rather
        than a ``UPath``.
    """

    base_dir: Path | UPath
    """Root directory for all products (local ``Path`` or S3 ``UPath``)."""
    slc_dir: Path | UPath
    """Directory for SLC burst files."""
    prod_dir: Path | UPath
    """Directory for processed products."""
    comp_dir: Path | UPath
    """Directory for composites."""

    def __init__(
        self,
        base_dir: str | Path | UPath,
        slc_dir: str | Path | UPath | None = None,
        prod_dir: str | Path | UPath | None = None,
        comp_dir: str | Path | UPath | None = None,
    ) -> None:
        self.base_dir = _as_path(base_dir)

        # Set default subdirectories if not provided
        self.slc_dir = _as_path(slc_dir) if slc_dir else self.base_dir / "s1-slc"
        self.prod_dir = _as_path(prod_dir) if prod_dir else self.base_dir / "s1-prod"
        self.comp_dir = _as_path(comp_dir) if comp_dir else self.base_dir / "s1-comp"

        self._legacy_names_checked = False

        # Create all directories (UPath.mkdir on S3 is a no-op)
        for directory in [self.slc_dir, self.prod_dir, self.comp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    # ============ SLC Operations ============

    def get_slc_path(self, burst: Burst) -> Path | UPath:
        """Generate the file path for an SLC burst.

        Parameters
        ----------
        burst
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
        rel_orbit_number
            Filter by relative orbit number.
        burst_id
            Filter by burst ID.
        sub_swath
            Filter by sub-swath.
        polarization
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

    def _warn_on_legacy_names(self) -> None:
        """Warn once if ``prod_dir`` holds products named without their pass direction."""
        if self._legacy_names_checked:
            return
        self._legacy_names_checked = True
        # Only names without a letter after the orbit match this glob.
        candidates = self.prod_dir.glob(
            "S1_*_[0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]_*.tif"
        )
        if any(_LEGACY_PRODUCT_NAME.fullmatch(path.name) for path in candidates):
            msg = (
                f"{self.prod_dir} holds per-burst products named without their pass "
                "direction, which the find_* methods no longer find. Rename them with "
                f"the recipe in the changelog entry on the pass direction: {_CHANGELOG_URL}"
            )
            warnings.warn(msg, stacklevel=3)

    # ============ Generic Build/Parse Engine ============

    def _build_pattern(
        self,
        prefix: str,
        spec: list[FieldSpec],
        *,
        suffix: str = ".tif",
        **values: Any,
    ) -> str:
        """Build a glob pattern from a prefix and field spec, with wildcards for missing values."""
        pattern = prefix
        for f in spec:
            v = values.get(f.name)
            pattern += f.sep + (f.to_str(v) if v is not None else f.wildcard)
        return pattern + suffix

    def _parse_filename(
        self,
        prefix_re: str,
        spec: list[FieldSpec],
        filename: str,
        prefix_captures: list[tuple[str, str, Callable[[str], Any]]] | None = None,
        *,
        suffix: str = ".tif",
    ) -> dict[str, Any] | None:
        """Parse a filename into field values, or ``None`` if it does not match the spec."""
        groups = "".join(f"{f.sep}(?P<{f.group or f.name}>{f.pattern})" for f in spec)
        match = re.fullmatch(rf"{prefix_re}{groups}{re.escape(suffix)}", filename)
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
        """Glob a directory, parse matching filenames, and instantiate products."""
        prefix_re, spec, prefix_captures = _PARSERS[cls]
        matches: list[Path | UPath] = list(directory.glob(pattern))
        files = sorted(matches, key=attrgetter("name"))
        return [
            cls.from_dict(m)
            for f in files
            if (m := self._parse_filename(prefix_re, spec, f.name, prefix_captures, suffix=suffix))
            is not None
        ]

    # ============ RTC Product Operations ============

    def _get_rtc_path(self, product: RTCProduct) -> Path | UPath:
        """Generate the file path for an RTC backscatter product."""
        return self.prod_dir / self._build_pattern("S1_RTC", _BACKSCATTER_SPEC, **asdict(product))

    def find_rtc_products(
        self,
        rel_orbit_number: int | None = None,
        burst_id: int | None = None,
        sub_swath: str | None = None,
        polarization: str | None = None,
        crs: CRSInput | None = None,
        pass_direction: PassDirection | None = None,
    ) -> list[RTCProduct]:
        """Find RTC backscatter products matching criteria.

        Note: Date filtering is not supported. Use list comprehension on results
        if date filtering is needed.

        Parameters
        ----------
        rel_orbit_number
            Filter by relative orbit number.
        burst_id
            Filter by burst ID.
        sub_swath
            Filter by sub-swath.
        polarization
            Filter by polarization.
        crs
            Filter by coordinate reference system (UTM zone).
        pass_direction
            Filter by pass direction (``"Ascending"`` or ``"Descending"``).

        Returns
        -------
        list[RTCProduct]
            List of matching RTC products.

        Raises
        ------
        ValueError
            If ``pass_direction`` is neither ``"Ascending"`` nor ``"Descending"``.

        Warns
        -----
        UserWarning
            Once per ``FileManager``, if ``prod_dir`` holds products named
            without their pass direction, which are not found.
        """
        self._warn_on_legacy_names()
        pattern = self._build_pattern(
            "S1_RTC",
            _BACKSCATTER_SPEC,
            rel_orbit_number=rel_orbit_number,
            burst_id=burst_id,
            sub_swath=sub_swath,
            polarization=polarization,
            crs=crs,
            pass_direction=_checked_pass_direction(pass_direction),
        )
        return self._find_products(self.prod_dir, pattern, RTCProduct)

    # ============ GTC Product Operations ============

    def _get_gtc_path(self, product: GTCProduct) -> Path | UPath:
        """Generate the file path for a GTC backscatter product."""
        return self.prod_dir / self._build_pattern("S1_GTC", _BACKSCATTER_SPEC, **asdict(product))

    def find_gtc_products(
        self,
        rel_orbit_number: int | None = None,
        burst_id: int | None = None,
        sub_swath: str | None = None,
        polarization: str | None = None,
        crs: CRSInput | None = None,
        pass_direction: PassDirection | None = None,
    ) -> list[GTCProduct]:
        """Find GTC backscatter products matching criteria.

        Note: Date filtering is not supported. Use list comprehension on results
        if date filtering is needed.

        Parameters
        ----------
        rel_orbit_number
            Filter by relative orbit number.
        burst_id
            Filter by burst ID.
        sub_swath
            Filter by sub-swath.
        polarization
            Filter by polarization.
        crs
            Filter by coordinate reference system (UTM zone).
        pass_direction
            Filter by pass direction (``"Ascending"`` or ``"Descending"``).

        Returns
        -------
        list[GTCProduct]
            List of matching GTC products.

        Raises
        ------
        ValueError
            If ``pass_direction`` is neither ``"Ascending"`` nor ``"Descending"``.

        Warns
        -----
        UserWarning
            Once per ``FileManager``, if ``prod_dir`` holds products named
            without their pass direction, which are not found.
        """
        self._warn_on_legacy_names()
        pattern = self._build_pattern(
            "S1_GTC",
            _BACKSCATTER_SPEC,
            rel_orbit_number=rel_orbit_number,
            burst_id=burst_id,
            sub_swath=sub_swath,
            polarization=polarization,
            crs=crs,
            pass_direction=_checked_pass_direction(pass_direction),
        )
        return self._find_products(self.prod_dir, pattern, GTCProduct)

    # ============ Coherence Product Operations ============

    def _get_coherence_path(self, product: CoherenceProduct) -> Path | UPath:
        """Generate the file path for a coherence product."""
        prefix = f"S1_COH{product.baseline_days:02d}"
        return self.prod_dir / self._build_pattern(prefix, _COREGISTERED_SPEC, **asdict(product))

    def find_coherence_products(
        self,
        rel_orbit_number: int | None = None,
        burst_id: int | None = None,
        sub_swath: str | None = None,
        polarization: str | None = None,
        baseline_days: int | None = None,
        crs: CRSInput | None = None,
        pass_direction: PassDirection | None = None,
    ) -> list[CoherenceProduct]:
        """Find coherence products matching criteria.

        Note: Date filtering is not supported. Use list comprehension on results
        if date filtering is needed.

        Parameters
        ----------
        rel_orbit_number
            Filter by relative orbit number.
        burst_id
            Filter by burst ID.
        sub_swath
            Filter by sub-swath.
        polarization
            Filter by polarization.
        baseline_days
            Filter by temporal baseline.
        crs
            Filter by coordinate reference system (UTM zone).
        pass_direction
            Filter by pass direction (``"Ascending"`` or ``"Descending"``).

        Returns
        -------
        list[CoherenceProduct]
            List of matching coherence products.

        Raises
        ------
        ValueError
            If ``pass_direction`` is neither ``"Ascending"`` nor ``"Descending"``.

        Warns
        -----
        UserWarning
            Once per ``FileManager``, if ``prod_dir`` holds products named
            without their pass direction, which are not found.
        """
        self._warn_on_legacy_names()
        bd = f"{baseline_days:02d}" if baseline_days is not None else "*"
        pattern = self._build_pattern(
            f"S1_COH{bd}",
            _COREGISTERED_SPEC,
            rel_orbit_number=rel_orbit_number,
            burst_id=burst_id,
            sub_swath=sub_swath,
            polarization=polarization,
            crs=crs,
            pass_direction=_checked_pass_direction(pass_direction),
        )
        return self._find_products(self.prod_dir, pattern, CoherenceProduct)

    # ============ Interferogram Product Operations ============

    def _get_interferogram_path(self, product: InterferogramProduct) -> Path | UPath:
        """Generate the file path for an interferogram product."""
        prefix = f"S1_IFG{product.baseline_days:02d}_CMPLX"
        return self.prod_dir / self._build_pattern(prefix, _COREGISTERED_SPEC, **asdict(product))

    def find_interferogram_products(
        self,
        rel_orbit_number: int | None = None,
        burst_id: int | None = None,
        sub_swath: str | None = None,
        polarization: str | None = None,
        baseline_days: int | None = None,
        crs: CRSInput | None = None,
        pass_direction: PassDirection | None = None,
    ) -> list[InterferogramProduct]:
        """Find interferogram products matching criteria.

        Note: Date filtering is not supported. Use list comprehension on results
        if date filtering is needed.

        Parameters
        ----------
        rel_orbit_number
            Filter by relative orbit number.
        burst_id
            Filter by burst ID.
        sub_swath
            Filter by sub-swath.
        polarization
            Filter by polarization.
        baseline_days
            Filter by temporal baseline.
        crs
            Filter by coordinate reference system (UTM zone).
        pass_direction
            Filter by pass direction (``"Ascending"`` or ``"Descending"``).

        Returns
        -------
        list[InterferogramProduct]
            List of matching interferogram products.

        Raises
        ------
        ValueError
            If ``pass_direction`` is neither ``"Ascending"`` nor ``"Descending"``.

        Warns
        -----
        UserWarning
            Once per ``FileManager``, if ``prod_dir`` holds products named
            without their pass direction, which are not found.
        """
        self._warn_on_legacy_names()
        bd = f"{baseline_days:02d}" if baseline_days is not None else "*"
        pattern = self._build_pattern(
            f"S1_IFG{bd}_CMPLX",
            _COREGISTERED_SPEC,
            rel_orbit_number=rel_orbit_number,
            burst_id=burst_id,
            sub_swath=sub_swath,
            polarization=polarization,
            crs=crs,
            pass_direction=_checked_pass_direction(pass_direction),
        )
        return self._find_products(self.prod_dir, pattern, InterferogramProduct)

    # ============ Composite Operations ============

    @staticmethod
    def _composite_prefix(product: MGRSTileProduct) -> str:
        """Return the file name prefix of a composite's measurement."""
        match product:
            case RTCLRWComposite():
                return "S1_RTC_LRW"
            case CohLRWComposite():
                return f"S1_COH{product.baseline_days:02d}_LRW"
            case RTCPassComposite():
                return f"S1_RTC_COMP_{product.pass_direction[0]}"
            case GTCPassComposite():
                return f"S1_GTC_COMP_{product.pass_direction[0]}"
            case CohPassComposite():
                return f"S1_COH{product.baseline_days:02d}_COMP_{product.pass_direction[0]}"
            case _:
                msg = f"Unknown product type '{type(product)}'."
                raise ValueError(msg)

    def _get_composite_path(self, product: MGRSTileProduct, layer: Layer) -> Path | UPath:
        """Generate the file path of one layer of a composite."""
        prefix = self._composite_prefix(product)
        if layer not in product.supported_layers:
            msg = f"{type(product).__name__} has no layer {layer.name}."
            raise ValueError(msg)
        if layer is not Layer.MEASUREMENT:
            prefix += f"_{layer.value}"
        return self.comp_dir / self._build_pattern(prefix, _MGRS_BASE_SPEC, **asdict(product))

    def _find_composites[C: MGRSTileProduct](
        self,
        prefix: str,
        cls: type[C],
        **values: Any,
    ) -> list[C]:
        """Find the files of every layer of a composite type, one product per identity."""
        # The trailing wildcard takes the QA token, so one listing finds every layer.
        pattern = self._build_pattern(f"{prefix}*", _MGRS_BASE_SPEC, **values)
        layers: defaultdict[C, set[Layer]] = defaultdict(set)
        for found in self._find_products(self.comp_dir, pattern, cls):
            layers[replace(found, layers=_MEASUREMENT_ONLY)] |= found.layers
        products = [replace(product, layers=frozenset(held)) for product, held in layers.items()]
        return sorted(products, key=lambda product: self.get_product_path(product).name)

    def find_rtc_lrw_products(
        self,
        mgrs_code: str | None = None,
        polarization: str | None = None,
    ) -> list[RTCLRWComposite]:
        """Find RTC LRW intensity composites matching criteria.

        Parameters
        ----------
        mgrs_code
            Filter by MGRS grid cell identifier.
        polarization
            Filter by polarization.

        Returns
        -------
        list[RTCLRWComposite]
            One composite per identity, holding the layers found on disk.
        """
        return self._find_composites(
            "S1_RTC_LRW", RTCLRWComposite, mgrs_code=mgrs_code, polarization=polarization
        )

    def find_coh_lrw_products(
        self,
        mgrs_code: str | None = None,
        polarization: str | None = None,
        baseline_days: int | None = None,
    ) -> list[CohLRWComposite]:
        """Find coherence LRW composites matching criteria.

        Parameters
        ----------
        mgrs_code
            Filter by MGRS grid cell identifier.
        polarization
            Filter by polarization.
        baseline_days
            Filter by temporal baseline in days.

        Returns
        -------
        list[CohLRWComposite]
            One composite per identity, holding the layers found on disk.
        """
        bd = f"{baseline_days:02d}" if baseline_days is not None else "??"
        return self._find_composites(
            f"S1_COH{bd}_LRW", CohLRWComposite, mgrs_code=mgrs_code, polarization=polarization
        )

    def find_rtc_pass_composites(
        self,
        mgrs_code: str | None = None,
        polarization: str | None = None,
        pass_direction: PassDirection | None = None,
    ) -> list[RTCPassComposite]:
        """Find RTC pass composites matching criteria.

        Parameters
        ----------
        mgrs_code
            Filter by MGRS grid cell identifier.
        polarization
            Filter by polarization.
        pass_direction
            Filter by pass direction (``"Ascending"`` or ``"Descending"``).

        Returns
        -------
        list[RTCPassComposite]
            One composite per identity, holding the layers found on disk.

        Raises
        ------
        ValueError
            If ``pass_direction`` is neither ``"Ascending"`` nor ``"Descending"``.
        """
        od = _pass_letter(pass_direction)
        return self._find_composites(
            f"S1_RTC_COMP_{od}", RTCPassComposite, mgrs_code=mgrs_code, polarization=polarization
        )

    def find_gtc_pass_composites(
        self,
        mgrs_code: str | None = None,
        polarization: str | None = None,
        pass_direction: PassDirection | None = None,
    ) -> list[GTCPassComposite]:
        """Find GTC pass composites matching criteria.

        Parameters
        ----------
        mgrs_code
            Filter by MGRS grid cell identifier.
        polarization
            Filter by polarization.
        pass_direction
            Filter by pass direction (``"Ascending"`` or ``"Descending"``).

        Returns
        -------
        list[GTCPassComposite]
            One composite per identity, holding the layers found on disk.

        Raises
        ------
        ValueError
            If ``pass_direction`` is neither ``"Ascending"`` nor ``"Descending"``.
        """
        od = _pass_letter(pass_direction)
        return self._find_composites(
            f"S1_GTC_COMP_{od}", GTCPassComposite, mgrs_code=mgrs_code, polarization=polarization
        )

    def find_coh_pass_composites(
        self,
        mgrs_code: str | None = None,
        polarization: str | None = None,
        pass_direction: PassDirection | None = None,
        baseline_days: int | None = None,
    ) -> list[CohPassComposite]:
        """Find coherence pass composites matching criteria.

        Parameters
        ----------
        mgrs_code
            Filter by MGRS grid cell identifier.
        polarization
            Filter by polarization.
        pass_direction
            Filter by pass direction (``"Ascending"`` or ``"Descending"``).
        baseline_days
            Filter by temporal baseline in days.

        Returns
        -------
        list[CohPassComposite]
            One composite per identity, holding the layers found on disk.

        Raises
        ------
        ValueError
            If ``pass_direction`` is neither ``"Ascending"`` nor ``"Descending"``.
        """
        bd = f"{baseline_days:02d}" if baseline_days is not None else "??"
        od = _pass_letter(pass_direction)
        return self._find_composites(
            f"S1_COH{bd}_COMP_{od}",
            CohPassComposite,
            mgrs_code=mgrs_code,
            polarization=polarization,
        )

    # ============ Generic Product Path Getter ============

    def get_product_path(
        self,
        product: BaseProduct,
        layer: Layer | str = Layer.MEASUREMENT,
    ) -> Path | UPath:
        """Get the path of a product's file, or of one layer of a composite.

        Parameters
        ----------
        product
            Product to get the path for.
        layer
            The layer whose file to locate, as a ``Layer`` or its value such
            as ``"QA-NUM"``; the measurement by default. The
            path of a composite's layer does not depend on whether
            ``product.layers`` holds it. A per-burst product has only its
            measurement.

        Returns
        -------
        Path
            Path to the file.

        Raises
        ------
        ValueError
            If the product type is unknown, or it has no layer ``layer``.
        """
        layer = Layer(layer)
        if isinstance(product, MGRSTileProduct):
            return self._get_composite_path(product, layer)
        if layer is not Layer.MEASUREMENT:
            msg = f"{type(product).__name__} has no layer {layer.name}."
            raise ValueError(msg)
        match product:
            case RTCProduct():
                return self._get_rtc_path(product)
            case GTCProduct():
                return self._get_gtc_path(product)
            case CoherenceProduct():
                return self._get_coherence_path(product)
            case InterferogramProduct():
                return self._get_interferogram_path(product)
            case _:
                msg = f"Unknown product type '{type(product)}'."
                raise ValueError(msg)

    def product_exists(self, product: BaseProduct, layer: Layer | str = Layer.MEASUREMENT) -> bool:
        """Check if a product's file exists (works for both local and S3).

        Parameters
        ----------
        product
            Product to check.
        layer
            The layer whose file to check, as a ``Layer`` or its value; the
            measurement by default.

        Returns
        -------
        bool
            True if the file exists.

        Raises
        ------
        ValueError
            If the product type is unknown, or it has no layer ``layer``.
        """
        return self.get_product_path(product, Layer(layer)).exists()
