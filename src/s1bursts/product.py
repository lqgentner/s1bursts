"""Product metadata containers for Sentinel-1 derived products."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, Self, get_args, override

from freezebase.mgrs import mgrs_to_crs
from pandas import Timestamp
from pyproj import CRS
from pyproj.exceptions import CRSError

from s1bursts.burst import _FOOTPRINT_ALIASES, BurstFootprintMixin
from s1bursts.query import PassDirection

if TYPE_CHECKING:
    from s1bursts.burst import Burst, BurstPair
    from s1bursts.query import Polarization, SubSwath


class CRSLike(Protocol):
    """CRS-like object."""

    def to_wkt(self) -> str:
        """Convert the projection to a WKT string."""
        ...


# Anything ``pyproj.CRS.from_user_input`` accepts that callers commonly pass:
# a CRS object, an authority string such as "EPSG:32632", or an EPSG code.
type CRSInput = CRSLike | str | int


# WGS 84 / UTM zones 1-60, north (EPSG:326xx) and south (EPSG:327xx).
# The ``crs`` that processes each burst, or pair, in the UTM zone of its footprint.
_AUTO_CRS = "auto"

_UTM_NORTH_EPSG = range(32601, 32661)
_UTM_SOUTH_EPSG = range(32701, 32761)


def _utm_zone_crs(crs: CRSInput) -> CRS:
    """Parse ``crs`` and require a WGS 84 / UTM zone, the one CRS product file names hold."""
    try:
        parsed = CRS.from_user_input(crs)
    except CRSError as err:
        raise ValueError(_not_a_utm_zone(crs, repr(crs))) from err
    epsg = parsed.to_epsg()
    if epsg in _UTM_NORTH_EPSG or epsg in _UTM_SOUTH_EPSG:
        return parsed
    raise ValueError(_not_a_utm_zone(crs, f"'{parsed.name}'"))


def _not_a_utm_zone(crs: CRSInput, received: str) -> str:
    """Return the message for a ``crs`` that is not a WGS 84 / UTM zone."""
    msg = (
        f"`crs` must be a WGS 84 / UTM zone, north or south (EPSG:326xx or 327xx), "
        f"received {received}. Product file names record the UTM zone. Pass the "
        'target cell\'s `crs` for products meant for a composite, or `"auto"` to '
        "process each burst in the zone of its footprint."
    )
    if isinstance(crs, str) and crs.strip().lower() == _AUTO_CRS:
        msg += ' "auto" is matched exactly, in lower case and without spaces.'
    return msg


def _check_output_crs(crs: CRSInput) -> CRS | str:
    """Return ``"auto"`` as is, or ``crs`` parsed as a WGS 84 / UTM zone."""
    if isinstance(crs, str) and crs == _AUTO_CRS:
        return _AUTO_CRS
    return _utm_zone_crs(crs)


def _output_crs(crs: CRSInput, source: Burst | BurstPair) -> CRS:
    """Return the UTM zone to geocode ``source`` in: its own zone for ``"auto"``."""
    if isinstance(crs, str) and crs == _AUTO_CRS:
        return source.utm_zone
    return _utm_zone_crs(crs)


_PASS_DIRECTIONS: tuple[str, ...] = get_args(PassDirection.__value__)


def _check_pass_direction(value: object) -> None:
    """Raise ValueError unless ``value`` is ``"Ascending"`` or ``"Descending"``."""
    if value not in _PASS_DIRECTIONS:
        msg = f"pass_direction must be 'Ascending' or 'Descending', received {value!r}."
        raise ValueError(msg)


class Layer(StrEnum):
    """One file of a composite product: its measurement or one of its quality layers.

    A quality layer adds its value to the composite's file name as a token,
    such as ``S1_RTC_LRW_QA-NUM_…``; the measurement file carries no token.
    """

    MEASUREMENT = "MEASUREMENT"
    """The composite's values: backscatter or coherence."""
    QA_NUM = "QA-NUM"
    """Number of input products contributing to each composite pixel.

    On an LRW composite this counts the products carrying a non-zero
    resolution weight, the CEOS-ARD Contributing Observations Image (SAR PFS
    v1.3.1, item 2.21). Pass composites are unweighted, so there it counts the
    products holding a finite value.
    """
    QA_CQM = "QA-CQM"
    """Composite local resolution: the reciprocal of the weighted relative area.

    Dimensionless; high values mean high (good) local resolution. The CEOS-ARD
    Composite Quality Map Image (SAR PFS v1.3.1, item 2.22), which defines it
    by reference to Small et al. (2022), eq. 6; see
    :mod:`s1bursts.resolution_weighter`. LRW composites only.
    """


@dataclass(frozen=True)
class BaseProduct(ABC):
    """Base class for all Sentinel-1 derived products.

    Products are lightweight metadata containers without file system logic.
    All filename generation and path operations are handled by FileManager.
    """

    @property
    @abstractmethod
    def product_type(self) -> str:
        """Get the product type identifier (e.g., ``"RTC"``, ``"GTC"``, ``"COH"``)."""

    def __repr__(self) -> str:
        """Return string representation of the product."""
        field_strs = []
        # Avoid resolving hidden LUT-backed fields while formatting.
        for f in fields(self):
            if not f.repr:
                continue
            value = getattr(self, f.name)
            if isinstance(value, CRS):
                field_strs.append(f"{f.name}='{value.srs}'")
            elif isinstance(value, Timestamp):
                field_strs.append(f"{f.name}='{value:%Y-%m-%d}'")
            elif isinstance(value, frozenset):
                members = ", ".join(repr(str(member)) for member in sorted(value))
                field_strs.append(f"{f.name}={{{members}}}")
            else:
                field_strs.append(f"{f.name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(field_strs)})"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Instantiate from a mapping, silently dropping unknown fields.

        Parameters
        ----------
        data
            Mapping of field names to values. A public footprint key such as
            ``rel_orbit_number`` sets its private override unless the mapping
            sets that override itself; other keys that are not init fields are
            dropped.

        Returns
        -------
        Self
            A new instance of the calling class.
        """
        init_keys = {f.name for f in fields(cls) if f.init}
        kwargs = {k: v for k, v in data.items() if k in init_keys}
        for public, private in _FOOTPRINT_ALIASES.items():
            if public in data and private in init_keys:
                kwargs.setdefault(private, data[public])
        return cls(**kwargs)


@dataclass(repr=False, frozen=True)
class GeocodedBurst(BurstFootprintMixin, BaseProduct, ABC):
    """Base class for all geocoded Sentinel-1 burst products.

    Parameters
    ----------
    burst_id
        Burst identifier.
    sub_swath
        Sub-swath identifier.
    crs
        The coordinate reference system.
    pass_direction
        Pass direction, ``"Ascending"`` or ``"Descending"``. Keyword-only.

    Raises
    ------
    ValueError
        If ``pass_direction`` is neither ``"Ascending"`` nor ``"Descending"``.
    """

    burst_id: int
    sub_swath: SubSwath
    crs: CRSLike
    # Keyword-only so the positional order of the identity fields is unchanged.
    pass_direction: PassDirection = field(kw_only=True)
    # None delegates to the mixin's burst-map lookup. The orbit is not part of
    # equality: the burst ID already fixes it, and comparing it would make ``==``
    # depend on whether a value was propagated or the burst map is installed.
    # Read-only here because the dataclass is frozen; the mixin declares it read-write.
    # pyrefly: ignore [bad-override]
    _rel_orbit_number: int | None = field(default=None, repr=False, compare=False, kw_only=True)
    # Include the resolved value in asdict()-based filenames.
    rel_orbit_number: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the pass direction and convert crs to a pyproj.CRS object."""
        # The file name keeps only its first letter, so anything else would be
        # written under a name that is never found again.
        _check_pass_direction(self.pass_direction)
        if not isinstance(self.crs, CRS):
            object.__setattr__(self, "crs", CRS.from_user_input(self.crs))


@dataclass(repr=False, frozen=True)
class BackscatterProduct(GeocodedBurst):
    """Base class for backscatter products from a single burst.

    Parameters
    ----------
    burst_id
        Burst identifier.
    sub_swath
        Sub-swath identifier.
    crs
        The coordinate reference system.
    date
        Acquisition date.
    polarization
        The polarization.
    pass_direction
        Pass direction, ``"Ascending"`` or ``"Descending"``. Keyword-only.
    """

    date: Timestamp
    polarization: Polarization

    @property
    @abstractmethod
    def is_terrain_flattened(self) -> bool:
        """Whether the product has been radiometrically terrain corrected."""

    @classmethod
    def from_burst(cls, burst: Burst, crs: CRSInput | Literal["auto"]) -> Self:
        """Create BackscatterProduct from Burst.

        Parameters
        ----------
        burst
            Burst to create product from.
        crs
            The coordinate reference system, a WGS 84 / UTM zone, north or
            south, which product file names record. Given as a CRS object, an
            authority string such as ``"EPSG:32632"`` or an EPSG code, or
            ``"auto"`` for the zone of the burst's footprint,
            ``Burst.utm_zone``.

        Returns
        -------
        BackscatterProduct
            Backscatter product metadata.

        Raises
        ------
        ValueError
            If ``crs`` is not a WGS 84 / UTM zone.
        NoUTMZoneError
            If ``crs`` is ``"auto"`` and the footprint's center lies outside
            the UTM zones.
        BurstMapUnavailableError
            If the burst carries no relative orbit number or pass direction,
            or for ``"auto"`` no footprint, and the burst map has not been
            downloaded.
        """
        return cls(
            burst_id=burst.burst_id,
            sub_swath=burst.sub_swath,
            polarization=burst.polarization,
            date=burst.date,
            crs=_output_crs(crs, burst),
            pass_direction=burst.pass_direction,
            _rel_orbit_number=burst.rel_orbit_number,
        )


@dataclass(repr=False, frozen=True)
class RTCProduct(BackscatterProduct):
    """Radiometrically Terrain Corrected (RTC) backscatter product.

    Gamma nought backscatter with terrain flattening applied.

    Parameters
    ----------
    burst_id
        Burst identifier.
    sub_swath
        Sub-swath identifier.
    crs
        The coordinate reference system.
    date
        Acquisition date.
    polarization
        The polarization.
    pass_direction
        Pass direction, ``"Ascending"`` or ``"Descending"``. Keyword-only.
    """

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return "RTC"

    @property
    @override
    def is_terrain_flattened(self) -> bool:
        """Whether the product has been radiometrically terrain corrected."""
        return True


@dataclass(repr=False, frozen=True)
class GTCProduct(BackscatterProduct):
    """Geometrically Terrain Corrected (GTC) backscatter product.

    Beta nought backscatter with geometric terrain correction only.

    Parameters
    ----------
    burst_id
        Burst identifier.
    sub_swath
        Sub-swath identifier.
    crs
        The coordinate reference system.
    date
        Acquisition date.
    polarization
        The polarization.
    pass_direction
        Pass direction, ``"Ascending"`` or ``"Descending"``. Keyword-only.
    """

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return "GTC"

    @property
    @override
    def is_terrain_flattened(self) -> bool:
        """Whether the product has been radiometrically terrain corrected."""
        return False


@dataclass(repr=False, frozen=True)
class CoherenceProduct(GeocodedBurst):
    """Coherence product from burst pair.

    Parameters
    ----------
    burst_id
        Burst identifier.
    sub_swath
        Sub-swath identifier.
    crs
        The coordinate reference system.
    date_primary
        Primary acquisition date.
    date_secondary
        Secondary acquisition date.
    polarization
        The polarization.
    pass_direction
        Pass direction, ``"Ascending"`` or ``"Descending"``. Keyword-only.
    """

    date_primary: Timestamp
    date_secondary: Timestamp
    polarization: Polarization

    @property
    def baseline_days(self) -> int:
        """Calculate temporal baseline in days.

        Returns
        -------
        int
            Temporal baseline in days.
        """
        return abs((self.date_secondary - self.date_primary).round("D").days)

    @classmethod
    def from_burst_pair(cls, pair: BurstPair, crs: CRSInput | Literal["auto"]) -> Self:
        """Create CoherenceProduct from BurstPair.

        Parameters
        ----------
        pair
            Burst pair to create product from.
        crs
            The coordinate reference system, a WGS 84 / UTM zone, north or
            south, which product file names record. Given as a CRS object, an
            authority string such as ``"EPSG:32632"`` or an EPSG code, or
            ``"auto"`` for the zone of the pair's footprint,
            ``BurstPair.utm_zone``.

        Returns
        -------
        CoherenceProduct
            Coherence product metadata.

        Raises
        ------
        ValueError
            If ``crs`` is not a WGS 84 / UTM zone.
        NoUTMZoneError
            If ``crs`` is ``"auto"`` and the footprint's center lies outside
            the UTM zones.
        BurstMapUnavailableError
            If the burst carries no relative orbit number or pass direction,
            or for ``"auto"`` no footprint, and the burst map has not been
            downloaded.
        """
        return cls(
            burst_id=pair.primary.burst_id,
            sub_swath=pair.primary.sub_swath,
            date_primary=pair.primary.date,
            date_secondary=pair.secondary.date,
            polarization=pair.primary.polarization,
            crs=_output_crs(crs, pair),
            pass_direction=pair.pass_direction,
            _rel_orbit_number=pair.rel_orbit_number,
        )

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return "COH"


_LRW_LAYERS = frozenset({Layer.MEASUREMENT, Layer.QA_NUM, Layer.QA_CQM})
# Pass composites are unweighted, so there is no resolution to map.
_PASS_LAYERS = frozenset({Layer.MEASUREMENT, Layer.QA_NUM})


def _layer_product_type(product: MGRSTileProduct, layer: Layer) -> str:
    """Product type of one layer's file: the composite's, plus the token of a QA layer."""
    if layer is Layer.MEASUREMENT:
        return product.product_type
    return f"{product.product_type}_{layer.value}"


@dataclass(repr=False, frozen=True)
class MGRSTileProduct(BaseProduct):
    """Base class for composite products using 10 km x 10 km MGRS grid squares.

    A composite is one product of several files, one per layer: the
    measurement and its quality layers. The identity fields name all of them;
    ``layers`` says which ones the product holds.

    Parameters
    ----------
    mgrs_code
        MGRS identifier.
    start_date
        Start of the compositing period (inclusive).
    end_date
        End of the compositing period (inclusive).
    polarization
        The polarization.
    layers
        The layers the product holds, a non-empty subset of
        ``supported_layers``. Keyword-only; defaults to the measurement alone.

    Raises
    ------
    ValueError
        If ``layers`` is empty or holds a layer the product type does not have,
        or a pass composite's ``pass_direction`` is neither ``"Ascending"`` nor
        ``"Descending"``.
    """

    supported_layers: ClassVar[frozenset[Layer]] = _LRW_LAYERS
    """Every layer a composite of this type can have."""

    mgrs_code: str
    start_date: Timestamp
    end_date: Timestamp
    polarization: Polarization
    layers: frozenset[Layer] = field(default=frozenset({Layer.MEASUREMENT}), kw_only=True)

    def __post_init__(self) -> None:
        """Normalize ``layers`` to a frozenset of ``Layer`` and check it."""
        layers = frozenset(Layer(layer) for layer in self.layers)
        if not layers:
            msg = "`layers` must not be empty."
            raise ValueError(msg)
        if unsupported := layers - self.supported_layers:
            names = ", ".join(sorted(layer.name for layer in unsupported))
            msg = f"{type(self).__name__} has no layer {names}."
            raise ValueError(msg)
        object.__setattr__(self, "layers", layers)

    @property
    def crs(self) -> CRS:
        """The pyproj Coordinate Reference System."""
        return mgrs_to_crs(self.mgrs_code)


@dataclass(repr=False, frozen=True)
class RTCLRWComposite(MGRSTileProduct):
    """Locally resolution weighted RTC intensity composite.

    Only RTC (radiometrically terrain corrected) products can be
    combined using LRW, as they account for terrain effects.
    The stored value is intensity (power), i.e. the squared amplitude.

    Parameters
    ----------
    mgrs_code
        MGRS grid cell identifier.
    start_date
        Start of the compositing period (inclusive).
    end_date
        End of the compositing period (inclusive).
    polarization
        The polarization.
    layers
        The layers the product holds: the measurement, ``QA-NUM`` and
        ``QA-CQM``. Keyword-only; defaults to the measurement alone.
    """

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return "RTC_LRW"


@dataclass(repr=False, frozen=True)
class CohLRWComposite(MGRSTileProduct):
    """Locally resolution weighted coherence composite.

    Parameters
    ----------
    mgrs_code
        MGRS grid cell identifier.
    start_date
        Start of the compositing period (inclusive).
    end_date
        End of the compositing period (inclusive).
    polarization
        The polarization.
    baseline_days
        Temporal baseline in days.
    layers
        The layers the product holds: the measurement, ``QA-NUM`` and
        ``QA-CQM``. Keyword-only; defaults to the measurement alone.
    """

    baseline_days: int

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return "COH_LRW"


@dataclass(repr=False, frozen=True)
class RTCPassComposite(MGRSTileProduct):
    """Unweighted temporal mean composite for a single pass direction.

    Parameters
    ----------
    mgrs_code
        MGRS grid cell identifier.
    start_date
        Start of the compositing period (inclusive).
    end_date
        End of the compositing period (inclusive).
    polarization
        The polarization.
    pass_direction
        Pass direction.
    layers
        The layers the product holds: the measurement and ``QA-NUM``.
        Keyword-only; defaults to the measurement alone.
    """

    supported_layers: ClassVar[frozenset[Layer]] = _PASS_LAYERS

    pass_direction: PassDirection

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"RTC_COMP_{self.pass_direction[0]}"

    def __post_init__(self) -> None:
        """Check the pass direction, then the layers."""
        _check_pass_direction(self.pass_direction)
        super().__post_init__()


@dataclass(repr=False, frozen=True)
class GTCPassComposite(MGRSTileProduct):
    """Unweighted temporal composite of GTC beta0 for a single pass direction.

    Parameters
    ----------
    mgrs_code
        MGRS grid cell identifier.
    start_date
        Start of the compositing period (inclusive).
    end_date
        End of the compositing period (inclusive).
    polarization
        The polarization.
    pass_direction
        Pass direction.
    layers
        The layers the product holds: the measurement and ``QA-NUM``.
        Keyword-only; defaults to the measurement alone.
    """

    supported_layers: ClassVar[frozenset[Layer]] = _PASS_LAYERS

    pass_direction: PassDirection

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"GTC_COMP_{self.pass_direction[0]}"

    def __post_init__(self) -> None:
        """Check the pass direction, then the layers."""
        _check_pass_direction(self.pass_direction)
        super().__post_init__()


@dataclass(repr=False, frozen=True)
class CohPassComposite(MGRSTileProduct):
    """Unweighted temporal mean coherence composite for a single pass direction.

    Parameters
    ----------
    mgrs_code
        MGRS grid cell identifier.
    start_date
        Start of the compositing period (inclusive).
    end_date
        End of the compositing period (inclusive).
    polarization
        The polarization.
    pass_direction
        Pass direction.
    baseline_days
        Temporal baseline in days.
    layers
        The layers the product holds: the measurement and ``QA-NUM``.
        Keyword-only; defaults to the measurement alone.
    """

    supported_layers: ClassVar[frozenset[Layer]] = _PASS_LAYERS

    pass_direction: PassDirection
    baseline_days: int

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"COH_COMP_{self.pass_direction[0]}"

    def __post_init__(self) -> None:
        """Check the pass direction, then the layers."""
        _check_pass_direction(self.pass_direction)
        super().__post_init__()
