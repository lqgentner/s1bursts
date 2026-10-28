"""Product metadata containers for Sentinel-1 derived products."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, Self, override

from freezebase.mgrs import mgrs_to_crs
from pandas import Timestamp
from pyproj import CRS

from s1bursts.burst import _FOOTPRINT_ALIASES, BurstFootprintMixin

if TYPE_CHECKING:
    from s1bursts.burst import Burst, BurstPair
    from s1bursts.query import PassDirection, Polarization, SubSwath


class CRSLike(Protocol):
    """CRS-like object."""

    def to_wkt(self) -> str:
        """Convert the projection to a WKT string."""
        ...


# Anything ``pyproj.CRS.from_user_input`` accepts that callers commonly pass:
# a CRS object, an authority string such as "EPSG:32632", or an EPSG code.
type CRSInput = CRSLike | str | int


class QaMetric(StrEnum):
    """Quality-assurance metric carried by an auxiliary composite product."""

    NUM = "NUM"
    """Number of input products contributing to each composite pixel.

    On an LRW composite this counts the products carrying a non-zero
    resolution weight, the CEOS-ARD Contributing Observations Image (SAR PFS
    v1.3.1, item 2.21). Pass-composites are unweighted, so there it counts the
    products holding a finite value.
    """
    CQM = "CQM"
    """Composite local resolution: the reciprocal of the weighted relative area.

    Dimensionless; high values mean high (good) local resolution. The CEOS-ARD
    Composite Quality Map Image (SAR PFS v1.3.1, item 2.22), which defines it
    by reference to Small et al. (2022), eq. 6; see
    :mod:`s1bursts.resolution_weighter`.
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
    """

    burst_id: int
    sub_swath: SubSwath
    crs: CRSLike
    # None delegates to the mixin's burst-map lookup. The orbit fields are not part
    # of equality: the burst ID already fixes the orbit, and comparing it would make
    # ``==`` depend on whether a value was propagated or the burst map is installed.
    # Read-only here because the dataclass is frozen; the mixin declares them read-write.
    # pyrefly: ignore [bad-override]
    _rel_orbit_number: int | None = field(default=None, repr=False, compare=False, kw_only=True)
    # pyrefly: ignore [bad-override]
    _pass_direction: PassDirection | None = field(
        default=None, repr=False, compare=False, kw_only=True
    )
    # Include the resolved value in asdict()-based filenames.
    rel_orbit_number: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Convert crs to pyproj.CRS object."""
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
    """

    date: Timestamp
    polarization: Polarization

    @property
    @abstractmethod
    def is_terrain_flattened(self) -> bool:
        """Whether the product has been radiometrically terrain corrected."""

    @classmethod
    def from_burst(cls, burst: Burst, crs: CRSInput) -> Self:
        """Create BackscatterProduct from Burst.

        Parameters
        ----------
        burst
            Burst to create product from.
        crs
            The coordinate reference system, as a CRS object, an authority string
            such as ``"EPSG:32632"`` or an EPSG code.

        Returns
        -------
        BackscatterProduct
            Backscatter product metadata.
        """
        return cls(
            burst_id=burst.burst_id,
            sub_swath=burst.sub_swath,
            polarization=burst.polarization,
            date=burst.date,
            crs=CRS.from_user_input(crs),
            _rel_orbit_number=burst.rel_orbit_number,
            # The propagated value only: the public property would fall back to the
            # burst map, which bursts found on disk need not have.
            _pass_direction=burst._pass_direction,  # noqa: SLF001
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
class CoregisteredProduct(GeocodedBurst, ABC):
    """Base class for co-registered (stacked) products from two bursts.

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
    def from_burst_pair(cls, pair: BurstPair, crs: CRSInput) -> Self:
        """Create CoregisteredProduct from BurstPair.

        Parameters
        ----------
        pair
            Burst pair to create product from.
        crs
            The coordinate reference system, as a CRS object, an authority string
            such as ``"EPSG:32632"`` or an EPSG code.

        Returns
        -------
        CoregisteredProduct
            Coregistered product metadata.
        """
        return cls(
            burst_id=pair.primary.burst_id,
            sub_swath=pair.primary.sub_swath,
            date_primary=pair.primary.date,
            date_secondary=pair.secondary.date,
            polarization=pair.primary.polarization,
            crs=CRS.from_user_input(crs),
            _rel_orbit_number=pair.rel_orbit_number,
            # See ``BackscatterProduct.from_burst``.
            _pass_direction=pair.primary._pass_direction,  # noqa: SLF001
        )


@dataclass(repr=False, frozen=True)
class CoherenceProduct(CoregisteredProduct):
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
    """

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return "COH"


@dataclass(repr=False, frozen=True)
class InterferogramProduct(CoregisteredProduct):
    """Interferogram product from burst pair.

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
    """

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return "IFG"


@dataclass(repr=False, frozen=True)
class MGRSTileProduct(BaseProduct):
    """Base class for composite products using 10 km x 10 km MGRS grid squares.

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
    """

    mgrs_code: str
    start_date: Timestamp
    end_date: Timestamp
    polarization: Polarization

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
    """

    pass_direction: PassDirection

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"RTC_COMP_{self.pass_direction[0]}"


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
    """

    pass_direction: PassDirection
    baseline_days: int

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"COH_COMP_{self.pass_direction[0]}"


@dataclass(repr=False, frozen=True)
class RTCLRWQA(MGRSTileProduct):
    """Quality-assurance auxiliary for an RTC LRW composite.

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
    qa_metric
        Quality metric stored in the file (number of observations or CQM).
    """

    qa_metric: QaMetric

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"RTC_LRW_QA-{self.qa_metric.value}"


@dataclass(repr=False, frozen=True)
class CohLRWQA(MGRSTileProduct):
    """Quality-assurance auxiliary for a coherence LRW composite.

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
    qa_metric
        Quality metric stored in the file (number of observations or CQM).
    """

    baseline_days: int
    qa_metric: QaMetric

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"COH_LRW_QA-{self.qa_metric.value}"


@dataclass(repr=False, frozen=True)
class RTCPassQA(MGRSTileProduct):
    """Contributing-observations auxiliary for an RTC pass-composite.

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
    """

    pass_direction: PassDirection

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"RTC_COMP_{self.pass_direction[0]}_QA-NUM"


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
    """

    pass_direction: PassDirection

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"GTC_COMP_{self.pass_direction[0]}"


@dataclass(repr=False, frozen=True)
class GTCPassQA(MGRSTileProduct):
    """Contributing-observations auxiliary for a GTC pass-composite.

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
    """

    pass_direction: PassDirection

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"GTC_COMP_{self.pass_direction[0]}_QA-NUM"


@dataclass(repr=False, frozen=True)
class CohPassQA(MGRSTileProduct):
    """Contributing-observations auxiliary for a coherence pass-composite.

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
    """

    pass_direction: PassDirection
    baseline_days: int

    @property
    @override
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"COH_COMP_{self.pass_direction[0]}_QA-NUM"
