"""Deterministic paths and discovery for Sentinel-1 products, local or on S3."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from functools import cache
import logging
from numbers import Integral
from operator import attrgetter, itemgetter
from pathlib import Path
import re
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
    Self,
    TypedDict,
    Unpack,
    cast,
    overload,
)
import warnings

from freezebase.mgrs import mgrs_to_crs
from freezebase.raster import get_utm_zone_string, rasterio_open, utm_zone_to_crs
import pandas as pd
from pyproj.exceptions import CRSError

from s1bursts.burst import Burst
from s1bursts.product import (
    BaseProduct,
    CoherenceProduct,
    CohLRWComposite,
    CohPassComposite,
    GTCPassComposite,
    GTCProduct,
    Layer,
    MGRSTileProduct,
    RTCLRWComposite,
    RTCPassComposite,
    RTCProduct,
    _utm_zone_crs,
)
from s1bursts.query import (
    _MAX_REL_ORBIT_NUMBER,
    _PASS_DIRECTIONS,
    _POLARIZATIONS,
    _SUB_SWATHS,
)
from s1bursts.time_range import _check_time_range

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from rasterio.io import DatasetReader
    from upath import UPath

    from s1bursts.product import CRSInput
    from s1bursts.query import PassDirection, Polarization, SubSwath
    from s1bursts.time_range import TimeRange


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
    check: Callable[[Any], object] | None = None  # rejects a filter value no name holds


def _one_of(allowed: tuple[str, ...]) -> Callable[[Any], object]:
    """Return a filter check that accepts only the values in ``allowed``."""

    def check(value: Any) -> object:
        if value not in allowed:
            msg = f"must be one of {', '.join(allowed)}."
            raise ValueError(msg)
        return value

    return check


def _int_from(minimum: int, maximum: int | None = None) -> Callable[[Any], object]:
    """Return a filter check that accepts only integers in an inclusive range."""

    def check(value: Any) -> object:
        # bool is an int, but True is no orbit or burst.
        if isinstance(value, bool) or not isinstance(value, Integral):
            msg = "must be an integer."
            raise TypeError(msg)
        number = int(value)
        if number < minimum or (maximum is not None and number > maximum):
            bound = f"{minimum}-{maximum}" if maximum is not None else f"at least {minimum}"
            msg = f"must be {bound}."
            raise ValueError(msg)
        return value

    return check


def _check_mgrs_code(value: Any) -> object:
    """Accept an upper-case MGRS reference, the form composite file names hold."""
    if not isinstance(value, str):
        msg = "must be a string."
        raise TypeError(msg)
    mgrs_to_crs(value)
    if value != value.upper():
        msg = "must be upper case, as in composite file names."
        raise ValueError(msg)
    return value


# Maps pass direction letter to full string
_PASS_DIRECTION = {"A": "Ascending", "D": "Descending"}

# Shared field specs
_ORBIT = FieldSpec(
    "rel_orbit_number", "{:03d}".format, r"\d{3}", int, check=_int_from(1, _MAX_REL_ORBIT_NUMBER)
)
# Per-burst products append the pass direction to the orbit, "066A", keeping the
# track in one token. Fixed-width wildcards, so that filtering on one half of the
# token cannot match across the separators of a neighboring one.
_TRACK_ORBIT = _ORBIT._replace(wildcard="???")
_PASS = FieldSpec(
    "pass_direction",
    itemgetter(0),
    "[AD]",
    _PASS_DIRECTION.__getitem__,
    wildcard="?",
    sep="",
    check=_one_of(_PASS_DIRECTIONS),
)
# A pass composite names its direction as a token of its own, "S1_RTC_COMP_A".
_COMPOSITE_PASS = _PASS._replace(sep="_")
_BASELINE = FieldSpec(
    "baseline_days", "{:02d}".format, r"\d{2}", int, wildcard="??", sep="", check=_int_from(1)
)
_BURST = FieldSpec("burst_id", "{:06d}".format, r"\d{6}", int, check=_int_from(1))
_SWATH = FieldSpec("sub_swath", str, r"IW[1-3]|EW[1-5]", str, check=_one_of(_SUB_SWATHS))
_POL = FieldSpec("polarization", str, r"[A-Z]{2}", str, check=_one_of(_POLARIZATIONS))
_DATE = FieldSpec("date", "{:%Y%m%d}".format, r"\d{8}", pd.Timestamp)
_DATE_P = FieldSpec("date_primary", "{:%Y%m%d}".format, r"\d{8}", pd.Timestamp)
_DATE_S = FieldSpec("date_secondary", "{:%Y%m%d}".format, r"\d{8}", pd.Timestamp)
_UTM = FieldSpec(
    "crs", get_utm_zone_string, r"\d{2}[NS]", utm_zone_to_crs, "utm_zone", check=_utm_zone_crs
)
_MGRS = FieldSpec("mgrs_code", str, r"[A-Z0-9]+", str, check=_check_mgrs_code)
_SDATE = FieldSpec("start_date", "{:%Y%m%d}".format, r"\d{8}", pd.Timestamp)
_EDATE = FieldSpec("end_date", "{:%Y%m%d}".format, r"\d{8}", pd.Timestamp)

_MEASUREMENT_ONLY = frozenset({Layer.MEASUREMENT})


def _layer_token(layer: Layer) -> str:
    """Return the token a layer adds to a composite file name; the measurement adds none."""
    return "" if layer is Layer.MEASUREMENT else f"_{layer.value}"


def _layers_from_token(token: str) -> frozenset[Layer]:
    """Return the layer that a composite file name's QA token, or its absence, stands for."""
    return frozenset({Layer(token.removeprefix("_"))}) if token else _MEASUREMENT_ONLY


# Built from one layer, parsed into the ``layers`` a found composite holds. The
# glob wildcard takes any token, so one listing finds every layer.
_LRW_LAYER = FieldSpec("layers", _layer_token, r"(?:_QA-NUM|_QA-CQM)?", _layers_from_token, sep="")
_PASS_LAYER = _LRW_LAYER._replace(pattern=r"(?:_QA-NUM)?")

# Per-product-type field lists
_SLC_FIELDS = (_TRACK_ORBIT, _PASS, _BURST, _SWATH, _POL, _DATE)
_BACKSCATTER_FIELDS = (_TRACK_ORBIT, _PASS, _BURST, _SWATH, _POL, _DATE, _UTM)
_COHERENCE_FIELDS = (_TRACK_ORBIT, _PASS, _BURST, _SWATH, _POL, _DATE_P, _DATE_S, _UTM)
_MGRS_FIELDS = (_MGRS, _POL, _SDATE, _EDATE)


# The filters of each product type. Total=False: every filter is optional, and
# None is the same as leaving it out.
class _SLCFilters(TypedDict, total=False):
    """Filters of SLC bursts."""

    rel_orbit_number: int | None
    pass_direction: PassDirection | None
    burst_id: int | None
    sub_swath: SubSwath | None
    polarization: Polarization | None


class _BurstProductFilters(_SLCFilters, total=False):
    """Filters of RTC and GTC products."""

    crs: CRSInput | None


class _CoherenceFilters(_BurstProductFilters, total=False):
    """Filters of coherence products."""

    baseline_days: int | None


class _CompositeFilters(TypedDict, total=False):
    """Filters of RTC LRW composites."""

    mgrs_code: str | None
    polarization: Polarization | None


class _CohCompositeFilters(_CompositeFilters, total=False):
    """Filters of coherence LRW composites."""

    baseline_days: int | None


class _PassCompositeFilters(_CompositeFilters, total=False):
    """Filters of RTC and GTC pass composites."""

    pass_direction: PassDirection | None


class _CohPassCompositeFilters(_PassCompositeFilters, total=False):
    """Filters of coherence pass composites."""

    baseline_days: int | None


class _NameFormat(NamedTuple):
    """How one product type is named, filed and found."""

    segments: tuple[str | FieldSpec, ...]  # literal text and fields, in name order
    filters: frozenset[str]  # the fields ``find()`` filters on
    directory: str  # the FileManager attribute of its directory
    dates: tuple[str, ...]  # the dates that must fall in a ``time_range``
    suffix: str = ".tif"

    @property
    def fields(self) -> list[FieldSpec]:
        """The fields of the name, without its literal text."""
        return [segment for segment in self.segments if isinstance(segment, FieldSpec)]


_SLC = frozenset(_SLCFilters.__optional_keys__)
_PER_BURST = frozenset(_BurstProductFilters.__optional_keys__)
_COHERENCE = frozenset(_CoherenceFilters.__optional_keys__)
_COMPOSITE = frozenset(_CompositeFilters.__optional_keys__)
_COH_COMPOSITE = frozenset(_CohCompositeFilters.__optional_keys__)
_PASS_COMPOSITE = frozenset(_PassCompositeFilters.__optional_keys__)
_COH_PASS_COMPOSITE = frozenset(_CohPassCompositeFilters.__optional_keys__)
# A coherence product is dated by its primary acquisition, as in the composites
# and ``BurstPairCollection.starting_within()``; a composite by its whole period.
_BURST_DATE = ("date",)
_PAIR_DATE = ("date_primary",)
_PERIOD = ("start_date", "end_date")

# Every type FileManager names and finds.
_NAME_FORMATS: dict[type, _NameFormat] = {
    Burst: _NameFormat(("S1_SLC", *_SLC_FIELDS), _SLC, "slc_dir", _BURST_DATE, ".zip"),
    RTCProduct: _NameFormat(("S1_RTC", *_BACKSCATTER_FIELDS), _PER_BURST, "prod_dir", _BURST_DATE),
    GTCProduct: _NameFormat(("S1_GTC", *_BACKSCATTER_FIELDS), _PER_BURST, "prod_dir", _BURST_DATE),
    CoherenceProduct: _NameFormat(
        ("S1_COH", _BASELINE, *_COHERENCE_FIELDS), _COHERENCE, "prod_dir", _PAIR_DATE
    ),
    RTCLRWComposite: _NameFormat(
        ("S1_RTC_LRW", _LRW_LAYER, *_MGRS_FIELDS), _COMPOSITE, "comp_dir", _PERIOD
    ),
    CohLRWComposite: _NameFormat(
        ("S1_COH", _BASELINE, "_LRW", _LRW_LAYER, *_MGRS_FIELDS),
        _COH_COMPOSITE,
        "comp_dir",
        _PERIOD,
    ),
    RTCPassComposite: _NameFormat(
        ("S1_RTC_COMP", _COMPOSITE_PASS, _PASS_LAYER, *_MGRS_FIELDS),
        _PASS_COMPOSITE,
        "comp_dir",
        _PERIOD,
    ),
    GTCPassComposite: _NameFormat(
        ("S1_GTC_COMP", _COMPOSITE_PASS, _PASS_LAYER, *_MGRS_FIELDS),
        _PASS_COMPOSITE,
        "comp_dir",
        _PERIOD,
    ),
    CohPassComposite: _NameFormat(
        ("S1_COH", _BASELINE, "_COMP", _COMPOSITE_PASS, _PASS_LAYER, *_MGRS_FIELDS),
        _COH_PASS_COMPOSITE,
        "comp_dir",
        _PERIOD,
    ),
}


def _name_format(cls: type) -> _NameFormat:
    """Return the name format of a type or of its nearest registered base."""
    for klass in cls.__mro__:
        if (name_format := _NAME_FORMATS.get(klass)) is not None:
            return name_format
    known = ", ".join(sorted(klass.__name__ for klass in _NAME_FORMATS))
    msg = f"Unknown product type '{cls.__name__}'; FileManager names {known}."
    raise ValueError(msg)


def _build_name(name_format: _NameFormat, values: dict[str, Any]) -> str:
    """Build a file name, or a glob pattern with wildcards for the values not given."""
    name = ""
    for segment in name_format.segments:
        if isinstance(segment, str):
            name += segment
        else:
            value = values.get(segment.name)
            name += segment.sep + (
                segment.to_str(value) if value is not None else segment.wildcard
            )
    return name + name_format.suffix


@cache
def _name_regex(name_format: _NameFormat) -> re.Pattern[str]:
    """Return the regex a file name of the format fully matches."""
    regex = "".join(
        re.escape(segment)
        if isinstance(segment, str)
        else f"{re.escape(segment.sep)}(?P<{segment.group or segment.name}>{segment.pattern})"
        for segment in name_format.segments
    )
    return re.compile(regex + re.escape(name_format.suffix))


def _parse_name(name_format: _NameFormat, filename: str) -> dict[str, Any] | None:
    """Parse a file name into field values, or ``None`` if it is not of the format."""
    match = _name_regex(name_format).fullmatch(filename)
    if match is None:
        return None
    return {f.name: f.from_str(match[f.group or f.name]) for f in name_format.fields}


def _filter_values(cls: type, name_format: _NameFormat, filters: dict[str, Any]) -> dict[str, Any]:
    """Return the given filters after rejecting a name or value no file name holds."""
    if unknown := sorted(filters.keys() - name_format.filters):
        accepted = ", ".join(sorted(name_format.filters))
        msg = (
            f"{cls.__name__} cannot be filtered on {', '.join(unknown)}; "
            f"its filters are {accepted} and time_range."
        )
        raise ValueError(msg)
    fields = {f.name: f for f in name_format.fields}
    values: dict[str, Any] = {}
    for name, value in filters.items():
        if value is None:
            continue
        spec = fields[name]
        try:
            if spec.check is not None:
                spec.check(value)
            segment = spec.to_str(value)
        except (TypeError, ValueError, CRSError) as err:
            msg = f"Invalid {name} filter {value!r}: {err}"
            raise ValueError(msg) from err
        # The check bounds the value; the name's width bounds it too.
        if not re.fullmatch(spec.pattern, segment):
            msg = f"Invalid {name} filter {value!r}: file names hold no {segment!r}."
            raise ValueError(msg)
        values[name] = value
    return values


def _merge_layers[C: MGRSTileProduct](found: list[C]) -> list[C]:
    """Merge the composites found per layer file into one per identity."""
    layers: defaultdict[C, set[Layer]] = defaultdict(set)
    for product in found:
        layers[replace(product, layers=_MEASUREMENT_ONLY)] |= product.layers
    return [replace(product, layers=frozenset(held)) for product, held in layers.items()]


# An SLC or per-burst product named before the pass direction joined the
# relative orbit, by the directory that holds them, with what becomes of them.
_LEGACY_NAMES = {
    "slc_dir": (
        re.compile(r"S1_SLC_\d{3}_\d{6}_(?:IW|EW)\d_.*\.zip"),
        "SLC bursts",
        " and BurstCatalog.download() downloads again.",
    ),
    "prod_dir": (
        re.compile(r"S1_(?:RTC|GTC|COH\d{2})_\d{3}_\d{6}_(?:IW|EW)\d_.*\.tif"),
        "per-burst products",
        ". Process them again with this version.",
    ),
}


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

        self._legacy_names_checked: set[str] = set()

        # Create all directories (UPath.mkdir on S3 is a no-op)
        for directory in [self.slc_dir, self.prod_dir, self.comp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

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

        Warns
        -----
        UserWarning
            Once per ``FileManager``, if ``slc_dir`` holds SLC bursts named
            without their pass direction, which ``BurstCatalog.download()``
            downloads again under the new name.
        """
        # Every SLC download and existence check resolves its path here.
        self._warn_on_legacy_names("slc_dir")
        return self._path(burst, Layer.MEASUREMENT)

    @overload
    def find(
        self,
        product_type: type[Burst],
        *,
        time_range: TimeRange | None = None,
        **filters: Unpack[_SLCFilters],
    ) -> list[Burst]: ...
    @overload
    def find[B: (RTCProduct, GTCProduct)](
        self,
        product_type: type[B],
        *,
        time_range: TimeRange | None = None,
        **filters: Unpack[_BurstProductFilters],
    ) -> list[B]: ...
    @overload
    def find(
        self,
        product_type: type[CoherenceProduct],
        *,
        time_range: TimeRange | None = None,
        **filters: Unpack[_CoherenceFilters],
    ) -> list[CoherenceProduct]: ...
    @overload
    def find(
        self,
        product_type: type[RTCLRWComposite],
        *,
        time_range: TimeRange | None = None,
        **filters: Unpack[_CompositeFilters],
    ) -> list[RTCLRWComposite]: ...
    @overload
    def find(
        self,
        product_type: type[CohLRWComposite],
        *,
        time_range: TimeRange | None = None,
        **filters: Unpack[_CohCompositeFilters],
    ) -> list[CohLRWComposite]: ...
    @overload
    def find(
        self,
        product_type: type[RTCPassComposite],
        *,
        time_range: TimeRange | None = None,
        **filters: Unpack[_PassCompositeFilters],
    ) -> list[RTCPassComposite]: ...
    @overload
    def find(
        self,
        product_type: type[GTCPassComposite],
        *,
        time_range: TimeRange | None = None,
        **filters: Unpack[_PassCompositeFilters],
    ) -> list[GTCPassComposite]: ...
    @overload
    def find(
        self,
        product_type: type[CohPassComposite],
        *,
        time_range: TimeRange | None = None,
        **filters: Unpack[_CohPassCompositeFilters],
    ) -> list[CohPassComposite]: ...
    def find[T: _Parseable](
        self,
        product_type: type[T],
        *,
        time_range: TimeRange | None = None,
        **filters: Any,
    ) -> list[T]:
        """Find the products of one type on disk, optionally filtered.

        Each product type is filtered on the fields its file name records;
        a filter left out, or ``None``, matches every value:

        - ``Burst`` (SLC): ``rel_orbit_number``, ``pass_direction``,
          ``burst_id``, ``sub_swath``, ``polarization``.
        - ``RTCProduct``, ``GTCProduct``: those of ``Burst``, plus ``crs``.
        - ``CoherenceProduct``: those of ``RTCProduct``, plus
          ``baseline_days``.
        - ``RTCLRWComposite``: ``mgrs_code``, ``polarization``.
        - ``CohLRWComposite``: ``mgrs_code``, ``polarization``,
          ``baseline_days``.
        - ``RTCPassComposite``, ``GTCPassComposite``: ``mgrs_code``,
          ``polarization``, ``pass_direction``.
        - ``CohPassComposite``: ``mgrs_code``, ``polarization``,
          ``pass_direction``, ``baseline_days``.

        SLC file names record the relative orbit and the pass direction, so
        the products created from a burst found here are named without the
        burst-ID map. They do not record the footprint: a burst found here
        takes its ``geometry`` from the burst-ID map, which is loaded once per
        process; see ``s1bursts.burstmap.download_burst_map``.

        Parameters
        ----------
        product_type
            The type of product to find, such as ``RTCProduct``, or ``Burst``
            for SLC bursts.
        time_range
            Keep only the products acquired on a day of this window: by
            ``date`` for SLC bursts, RTC and GTC products, by ``date_primary``
            for coherence products, as
            ``BurstPairCollection.starting_within()`` selects pairs. A
            composite is kept when its whole period, ``start_date`` through
            ``end_date``, lies in the window.
        **filters
            Field values to match, from the list above: ``rel_orbit_number``
            (1-175), ``burst_id`` and ``baseline_days`` are integers,
            ``sub_swath`` a ``SubSwath`` such as ``"IW1"``, ``polarization``
            a ``Polarization`` such as ``"VV"``, ``pass_direction``
            ``"Ascending"`` or ``"Descending"``, ``crs`` a WGS 84 / UTM zone
            as a CRS object, authority string or EPSG code, and
            ``mgrs_code`` an upper-case MGRS reference such as ``"32TMT64"``.

        Returns
        -------
        list
            The products found, sorted by file name. A composite is returned
            once per identity, holding the layers whose files were found, so
            it may lack its measurement when only QA layers are on disk.

        Raises
        ------
        ValueError
            If ``product_type`` is not a type FileManager names, or a filter
            is not one of the type's or has a value no file name holds.
        TypeError
            If ``time_range`` is not a ``TimeRange``.

        Warns
        -----
        UserWarning
            Once per ``FileManager`` and directory, when finding SLC bursts or
            per-burst products, if ``slc_dir`` or ``prod_dir`` holds files
            named without their pass direction, which are not found.
        """
        name_format = _name_format(product_type)
        time_range = _check_time_range(time_range)
        values = _filter_values(product_type, name_format, filters)
        self._warn_on_legacy_names(name_format.directory)

        directory: Path | UPath = getattr(self, name_format.directory)
        paths: list[Path | UPath] = list(directory.glob(_build_name(name_format, values)))
        found: list[T] = []
        for path in sorted(paths, key=attrgetter("name")):
            parsed = _parse_name(name_format, path.name)
            if parsed is None:
                continue
            product = product_type.from_dict(parsed)
            if time_range is None or all(
                getattr(product, date) in time_range for date in name_format.dates
            ):
                found.append(product)
        if not issubclass(product_type, MGRSTileProduct):
            return found
        # Only an MGRSTileProduct type reaches this, so T is one.
        composites = cast("list[MGRSTileProduct]", found)
        merged = sorted(_merge_layers(composites), key=lambda c: self.get_product_path(c).name)
        return cast("list[T]", merged)

    def _warn_on_legacy_names(self, directory_name: str) -> None:
        """Warn once per directory if it holds files named without their pass direction."""
        if directory_name not in _LEGACY_NAMES or directory_name in self._legacy_names_checked:
            return
        self._legacy_names_checked.add(directory_name)
        legacy_name, kind, consequence = _LEGACY_NAMES[directory_name]
        directory: Path | UPath = getattr(self, directory_name)
        # Only names without a letter after the orbit match this glob.
        candidates = directory.glob("S1_*_[0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]_*")
        if any(legacy_name.fullmatch(path.name) for path in candidates):
            msg = (
                f"{directory} holds {kind} named without their pass direction, which "
                f"FileManager.find() does not find{consequence}"
            )
            warnings.warn(msg, stacklevel=3)

    def _path(self, product: object, layer: Layer) -> Path | UPath:
        """Return the path of one layer's file of a product or SLC burst."""
        name_format = _name_format(type(product))
        values = {
            f.name: getattr(product, f.name) for f in name_format.fields if f.name != "layers"
        }
        name = _build_name(name_format, {**values, "layers": layer})
        directory: Path | UPath = getattr(self, name_format.directory)
        return directory / name

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
        supported = (
            product.supported_layers if isinstance(product, MGRSTileProduct) else _MEASUREMENT_ONLY
        )
        if layer not in supported:
            msg = f"{type(product).__name__} has no layer {layer.name}."
            raise ValueError(msg)
        return self._path(product, layer)

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

    def open(
        self,
        product: BaseProduct,
        layer: Layer | str = Layer.MEASUREMENT,
    ) -> AbstractContextManager[DatasetReader]:
        """Open a product's file, or one layer of a composite, for reading.

        Works for both local and S3 storage: an S3 ``UPath`` hands its
        endpoint and credentials to GDAL, which ``rasterio.open()`` does not.

        Parameters
        ----------
        product
            Product to open.
        layer
            The layer whose file to open, as a ``Layer`` or its value such as
            ``"QA-NUM"``; the measurement by default.

        Returns
        -------
        AbstractContextManager[DatasetReader]
            Context manager yielding the read-only rasterio dataset.

        Raises
        ------
        ValueError
            If the product type is unknown, or it has no layer ``layer``.
        """
        return rasterio_open(self.get_product_path(product, layer))
