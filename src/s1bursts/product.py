"""Product metadata containers for Sentinel-1 derived products."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, Self

from freezecore.mgrs import mgrs_to_crs
from pandas import Timestamp
from pyproj import CRS

from s1bursts.burst import BurstFootprintMixin

if TYPE_CHECKING:
    from s1bursts.burst import Burst, BurstPair
    from s1bursts.query import OrbitDirection, Polarization, SubSwath


class CRSLike(Protocol):
    """CRS-like object."""

    def to_wkt(self) -> str:
        """Convert the projection to a WKT string."""
        ...


class QaMetric(StrEnum):
    """Quality-assurance metric carried by an auxiliary composite product."""

    NUM = "NUM"
    """Number of valid observations contributing to each composite pixel."""
    CQM = "CQM"
    """Composite local resolution (resolution-weighted contributing area)."""


@dataclass(frozen=True)
class BaseProduct(ABC):
    """Base class for all Sentinel-1 derived products.

    Products are lightweight metadata containers without file system logic.
    All filename generation and path operations are handled by FileManager.
    """

    @property
    @abstractmethod
    def product_type(self) -> str:
        """Get the product type identifier (e.g., 'RTC', 'GTC', 'COH', 'SIM')."""

    def __repr__(self) -> str:
        """Return string representation of the product."""
        field_strs = []
        # Fields marked repr=False are skipped; this keeps __repr__ free of
        # side effects (rel_orbit_number is resolved via the burst-map LUT,
        # which may download data or raise for unknown bursts).
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
        """Instantiate from a mapping, silently dropping unknown and non-init fields.

        Parameters
        ----------
        data : dict[str, Any]
            Mapping of field names to values. Extra keys and ``init=False``
            fields are filtered before construction.

        Returns
        -------
        Self
            A new instance of the calling class.
        """
        init_keys = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in data.items() if k in init_keys})


@dataclass(repr=False, frozen=True)
class GeocodedBurst(BurstFootprintMixin, BaseProduct, ABC):
    """Base class for all geocoded Sentinel-1 burst products.

    Parameters
    ----------
    burst_id : int
        Burst identifier.
    sub_swath : {"IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"}
        Sub-swath identifier.
    crs : CRSLike
        The coordinate reference system.
    """

    burst_id: int
    sub_swath: SubSwath
    crs: CRSLike
    # Relative orbit number propagated from the source burst; None falls back to the
    # LUT via the mixin property. Keyword-only so subclasses can add required fields.
    _rel_orbit_number: int | None = field(default=None, repr=False, kw_only=True)
    # Mixin property (propagated value or LUT), declared as a field for asdict() filenames.
    rel_orbit_number: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Convert crs to pyproj.CRS object."""
        if not isinstance(self.crs, CRS):
            object.__setattr__(self, "crs", CRS(self.crs.to_wkt()))


@dataclass(repr=False, frozen=True)
class BackscatterProduct(GeocodedBurst):
    """Base class for backscatter products from a single burst.

    Parameters
    ----------
    burst_id : int
        Burst identifier.
    sub_swath : {"IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"}
        Sub-swath identifier.
    crs : CRSLike
        The coordinate reference system.
    date : Timestamp
        Acquisition date.
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    """

    date: Timestamp
    polarization: Polarization

    @property
    @abstractmethod
    def is_terrain_flattened(self) -> bool:
        """Whether the product has been radiometrically terrain corrected."""

    @classmethod
    def from_burst(cls, burst: Burst, crs: CRSLike) -> Self:
        """Create BackscatterProduct from Burst.

        Parameters
        ----------
        burst : Burst
            Burst to create product from.
        crs : CRSLike
            The coordinate reference system.

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
            crs=crs,
            _rel_orbit_number=burst.rel_orbit_number,
        )


@dataclass(repr=False, frozen=True)
class RTCProduct(BackscatterProduct):
    """Radiometrically Terrain Corrected (RTC) backscatter product.

    Gamma nought backscatter with terrain flattening applied.

    Parameters
    ----------
    burst_id : int
        Burst identifier.
    sub_swath : {"IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"}
        Sub-swath identifier.
    crs : CRSLike
        The coordinate reference system.
    date : Timestamp
        Acquisition date.
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    """

    @property
    def product_type(self) -> str:
        """Get product type identifier."""
        return "RTC"

    @property
    def is_terrain_flattened(self) -> bool:
        """Whether the product has been radiometrically terrain corrected."""
        return True


@dataclass(repr=False, frozen=True)
class GTCProduct(BackscatterProduct):
    """Geometrically Terrain Corrected (GTC) backscatter product.

    Beta nought backscatter with geometric terrain correction only.

    Parameters
    ----------
    burst_id : int
        Burst identifier.
    sub_swath : {"IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"}
        Sub-swath identifier.
    crs : CRSLike
        The coordinate reference system.
    date : Timestamp
        Acquisition date.
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    """

    @property
    def product_type(self) -> str:
        """Get product type identifier."""
        return "GTC"

    @property
    def is_terrain_flattened(self) -> bool:
        """Whether the product has been radiometrically terrain corrected."""
        return False


@dataclass(repr=False, frozen=True)
class CoregisteredProduct(GeocodedBurst, ABC):
    """Base class for co-registered (stacked) products from two bursts.

    Parameters
    ----------
    burst_id : int
        Burst identifier.
    sub_swath : {"IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"}
        Sub-swath identifier.
    crs : CRSLike
        The coordinate reference system.
    date_primary : Timestamp
        Primary acquisition date.
    date_secondary : Timestamp
        Secondary acquisition date.
    polarization : {"VV", "VH", "HH", "HV"}
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
    def from_burst_pair(cls, pair: BurstPair, crs: CRSLike) -> Self:
        """Create CoregisteredProduct from BurstPair.

        Parameters
        ----------
        pair : BurstPair
            Burst pair to create product from.
        crs : CRSLike
            The coordinate reference system.

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
            crs=crs,
            _rel_orbit_number=pair.rel_orbit_number,
        )


@dataclass(repr=False, frozen=True)
class CoherenceProduct(CoregisteredProduct):
    """Coherence product from burst pair.

    Parameters
    ----------
    burst_id : int
        Burst identifier.
    sub_swath : {"IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"}
        Sub-swath identifier.
    crs : CRSLike
        The coordinate reference system.
    date_primary : Timestamp
        Primary acquisition date.
    date_secondary : Timestamp
        Secondary acquisition date.
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    """

    @property
    def product_type(self) -> str:
        """Get product type identifier."""
        return "COH"

    @property
    def is_complex(self) -> bool:
        """Whether the product is complex."""
        return False


@dataclass(repr=False, frozen=True)
class CoherenceComplexProduct(CoregisteredProduct):
    """Complex coherence product from burst pair.

    Parameters
    ----------
    burst_id : int
        Burst identifier.
    sub_swath : {"IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"}
        Sub-swath identifier.
    crs : CRSLike
        The coordinate reference system.
    date_primary : Timestamp
        Primary acquisition date.
    date_secondary : Timestamp
        Secondary acquisition date.
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    """

    @property
    def product_type(self) -> str:
        """Get product type identifier."""
        return "COH_CMPLX"

    @property
    def is_complex(self) -> bool:
        """Whether the product is complex."""
        return True


@dataclass(repr=False, frozen=True)
class InterferogramProduct(CoregisteredProduct):
    """Interferogram product from burst pair.

    Parameters
    ----------
    burst_id : int
        Burst identifier.
    sub_swath : {"IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"}
        Sub-swath identifier.
    crs : CRSLike
        The coordinate reference system.
    date_primary : Timestamp
        Primary acquisition date.
    date_secondary : Timestamp
        Secondary acquisition date.
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    """

    @property
    def product_type(self) -> str:
        """Get product type identifier."""
        return "IFG"


@dataclass(repr=False, frozen=True)
class MGRSTileProduct(BaseProduct):
    """Base class for composite products using 10 km x 10 km MGRS grid squares.

    Parameters
    ----------
    mgrs_code : str
        MGRS identifier.
    start_date : Timestamp
        Start of the compositing period (inclusive).
    end_date : Timestamp
        End of the compositing period (inclusive).
    polarization : {"VV", "VH", "HH", "HV"}
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
    mgrs_code : str
        MGRS grid cell identifier.
    start_date : Timestamp
        Start of the compositing period (inclusive).
    end_date : Timestamp
        End of the compositing period (inclusive).
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    """

    @property
    def product_type(self) -> str:
        """Get product type identifier."""
        return "RTC_LRW"


@dataclass(repr=False, frozen=True)
class CohLRWComposite(MGRSTileProduct):
    """Locally resolution weighted coherence composite.

    Parameters
    ----------
    mgrs_code : str
        MGRS grid cell identifier.
    start_date : Timestamp
        Start of the compositing period (inclusive).
    end_date : Timestamp
        End of the compositing period (inclusive).
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    baseline_days : int
        Temporal baseline in days.
    """

    baseline_days: int

    @property
    def product_type(self) -> str:
        """Get product type identifier."""
        return "COH_LRW"


@dataclass(repr=False, frozen=True)
class RTCPassComposite(MGRSTileProduct):
    """Unweighted temporal mean composite for a single orbit-pass direction.

    Parameters
    ----------
    mgrs_code : str
        MGRS grid cell identifier.
    start_date : Timestamp
        Start of the compositing period (inclusive).
    end_date : Timestamp
        End of the compositing period (inclusive).
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    orbit_direction : {"ASCENDING", "DESCENDING"}
        Orbit pass direction.
    """

    orbit_direction: OrbitDirection

    @property
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"RTC_COMP_{self.orbit_direction[0]}"


@dataclass(repr=False, frozen=True)
class CohPassComposite(MGRSTileProduct):
    """Unweighted temporal mean coherence composite for a single orbit-pass direction.

    Parameters
    ----------
    mgrs_code : str
        MGRS grid cell identifier.
    start_date : Timestamp
        Start of the compositing period (inclusive).
    end_date : Timestamp
        End of the compositing period (inclusive).
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    orbit_direction : {"ASCENDING", "DESCENDING"}
        Orbit pass direction.
    baseline_days : int
        Temporal baseline in days.
    """

    orbit_direction: OrbitDirection
    baseline_days: int

    @property
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"COH_COMP_{self.orbit_direction[0]}"


@dataclass(repr=False, frozen=True)
class RTCLRWQA(MGRSTileProduct):
    """Quality-assurance auxiliary for an RTC LRW composite.

    Parameters
    ----------
    mgrs_code : str
        MGRS grid cell identifier.
    start_date : Timestamp
        Start of the compositing period (inclusive).
    end_date : Timestamp
        End of the compositing period (inclusive).
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    qa_metric : QaMetric
        Quality metric stored in the file (number of observations or CQM).
    """

    qa_metric: QaMetric

    @property
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"RTC_LRW_QA-{self.qa_metric.value}"


@dataclass(repr=False, frozen=True)
class CohLRWQA(MGRSTileProduct):
    """Quality-assurance auxiliary for a coherence LRW composite.

    Parameters
    ----------
    mgrs_code : str
        MGRS grid cell identifier.
    start_date : Timestamp
        Start of the compositing period (inclusive).
    end_date : Timestamp
        End of the compositing period (inclusive).
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    baseline_days : int
        Temporal baseline in days.
    qa_metric : QaMetric
        Quality metric stored in the file (number of observations or CQM).
    """

    baseline_days: int
    qa_metric: QaMetric

    @property
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"COH_LRW_QA-{self.qa_metric.value}"


@dataclass(repr=False, frozen=True)
class RTCPassQA(MGRSTileProduct):
    """Number-of-observations auxiliary for an RTC pass-composite.

    Parameters
    ----------
    mgrs_code : str
        MGRS grid cell identifier.
    start_date : Timestamp
        Start of the compositing period (inclusive).
    end_date : Timestamp
        End of the compositing period (inclusive).
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    orbit_direction : {"ASCENDING", "DESCENDING"}
        Orbit pass direction.
    """

    orbit_direction: OrbitDirection

    @property
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"RTC_COMP_{self.orbit_direction[0]}_QA-NUM"


@dataclass(repr=False, frozen=True)
class CohPassQA(MGRSTileProduct):
    """Number-of-observations auxiliary for a coherence pass-composite.

    Parameters
    ----------
    mgrs_code : str
        MGRS grid cell identifier.
    start_date : Timestamp
        Start of the compositing period (inclusive).
    end_date : Timestamp
        End of the compositing period (inclusive).
    polarization : {"VV", "VH", "HH", "HV"}
        The polarization.
    orbit_direction : {"ASCENDING", "DESCENDING"}
        Orbit pass direction.
    baseline_days : int
        Temporal baseline in days.
    """

    orbit_direction: OrbitDirection
    baseline_days: int

    @property
    def product_type(self) -> str:
        """Get product type identifier."""
        return f"COH_COMP_{self.orbit_direction[0]}_QA-NUM"
