"""Search query parameters for the Sentinel-1 burst catalog."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pyproj import CRS

from s1bursts.time_range import TimeRange, _check_time_range

type Polarization = Literal["VV", "VH", "HH", "HV"]
type Mode = Literal["IW", "EW"]
type SubSwath = Literal["IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"]
type PassDirection = Literal["Ascending", "Descending"]
type PlatformSerialID = Literal["A", "B", "C", "D"]

# The static types above are erased at runtime, so the accepted values are
# repeated here to reject a typo when the query is built rather than letting
# the CDSE API return an empty result set for it.
_POLARIZATIONS = ("VV", "VH", "HH", "HV")
_MODES = ("IW", "EW")
_SUB_SWATHS = ("IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5")
_PASS_DIRECTIONS = ("Ascending", "Descending")
_PLATFORM_SERIAL_IDS = ("A", "B", "C", "D")

_WGS84 = CRS.from_epsg(4326)

# Sentinel-1's repeat cycle covers 175 relative orbits, numbered from 1.
_MAX_REL_ORBIT_NUMBER = 175


@runtime_checkable
class HasWkt(Protocol):
    """Any Geometry that implements `wkt`."""

    @property
    def wkt(self) -> str:
        """WKT representation of the geometry."""
        ...


class SearchQuery:
    """Query parameters for searching Sentinel-1 bursts via the CDSE API.

    All parameters are optional; omitting a parameter means no filtering on that field.

    Parameters
    ----------
    time_range
        Acquisition time window. The search keeps every acquisition on any of
        its days, end included, and the same window can then name the composite
        built from the search.
    intersects
        Spatial filter; bursts must intersect this geometry, in WGS 84
        longitude and latitude. A geometry that carries a ``crs``, such as an
        ``odc.geo`` ``Geometry``, must be in EPSG:4326. The catalog rejects a
        geometry with too many vertices, such as the exact outline of an MGRS
        cell; search with its bounding box and select with
        ``BurstCollection.intersecting()``.
    burst_id
        Sentinel-1 relative burst ID, unique within each sub-swath. Must be positive.
    abs_burst_id
        Sentinel-1 absolute burst ID, globally unique across all sub-swaths and orbits.
        Must be positive.
    polarization
        Polarization channel.
    mode
        Acquisition mode.
    sub_swath
        Sub-swath identifier.
    pass_direction
        Pass direction.
    platform_serial_id
        Sentinel-1 platform serial identifier.
    rel_orbit_number
        Relative orbit number (1-175).

    Raises
    ------
    TypeError
        If ``time_range`` is not a ``TimeRange``.
    ValueError
        If any value falls outside its accepted range or set of identifiers,
        or ``intersects`` is not a geometry, is empty or is not in WGS 84.
    """

    def __init__(
        self,
        time_range: TimeRange | None = None,
        intersects: HasWkt | None = None,
        burst_id: int | None = None,
        abs_burst_id: int | None = None,
        polarization: Polarization | None = None,
        mode: Mode | None = None,
        sub_swath: SubSwath | None = None,
        pass_direction: PassDirection | None = None,
        platform_serial_id: PlatformSerialID | None = None,
        rel_orbit_number: int | None = None,
    ) -> None:
        self.time_range = _check_time_range(time_range)
        self.intersects = self._validate_geometry(intersects)
        self.burst_id = self._validate_int_range(burst_id, "burst_id", minimum=1)
        self.abs_burst_id = self._validate_int_range(abs_burst_id, "abs_burst_id", minimum=1)
        self.polarization = self._validate_choice(polarization, "polarization", _POLARIZATIONS)
        self.sub_swath = self._validate_choice(sub_swath, "sub_swath", _SUB_SWATHS)
        self.mode = self._validate_choice(mode, "mode", _MODES)
        self.pass_direction = self._validate_choice(
            pass_direction,
            "pass_direction",
            _PASS_DIRECTIONS,
        )
        self.platform_serial_id = self._validate_choice(
            platform_serial_id,
            "platform_serial_id",
            _PLATFORM_SERIAL_IDS,
        )
        self.rel_orbit_number = self._validate_int_range(
            rel_orbit_number,
            "rel_orbit_number",
            minimum=1,
            maximum=_MAX_REL_ORBIT_NUMBER,
        )

    @staticmethod
    def _validate_choice[T: str](
        value: T | None,
        field_name: str,
        allowed: tuple[str, ...],
    ) -> T | None:
        """Validate that a value is one of the accepted identifiers."""
        if value is not None and value not in allowed:
            msg = f"{field_name} must be one of {', '.join(allowed)}; received '{value}'"
            raise ValueError(msg)
        return value

    @staticmethod
    def _validate_geometry(geometry: HasWkt | None) -> HasWkt | None:
        """Validate that geometry implements the WKT protocol, is not empty and is in WGS 84."""
        if geometry is None:
            return None
        if not isinstance(geometry, HasWkt):
            msg = "`intersects` must be a Geometry that implements `wkt`."
            # Released as a ValueError; a TypeError would break callers that catch it.
            raise ValueError(msg)  # noqa: TRY004
        # An empty geometry would otherwise search without a spatial filter.
        if getattr(geometry, "is_empty", False) is True:
            msg = "`intersects` is an empty geometry; pass None to search without an area."
            raise ValueError(msg)
        # The catalog reads the WKT as longitude and latitude whatever the geometry's CRS.
        crs = getattr(geometry, "crs", None)
        if crs is not None and not CRS.from_user_input(crs).equals(_WGS84, ignore_axis_order=True):
            msg = (
                f"`intersects` is in {crs}, but the catalog takes WGS 84 longitude and "
                'latitude; pass `geometry.to_crs("EPSG:4326")`.'
            )
            raise ValueError(msg)
        return geometry

    @staticmethod
    def _validate_int_range(
        value: int | None,
        field_name: str,
        *,
        minimum: int,
        maximum: int | None = None,
    ) -> int | None:
        """Validate that an integer falls inside an inclusive range."""
        if value is None:
            return None
        if value < minimum or (maximum is not None and value > maximum):
            bound = f"{minimum}-{maximum}" if maximum is not None else f"at least {minimum}"
            msg = f"{field_name} must be {bound}; received {value}"
            raise ValueError(msg)
        return value

    def __repr__(self) -> str:
        """Return string representation showing only non-None parameters.

        Returns
        -------
        str
            String representation of the query.
        """
        attrs = [
            "time_range",
            "intersects",
            "burst_id",
            "abs_burst_id",
            "polarization",
            "mode",
            "sub_swath",
            "pass_direction",
            "platform_serial_id",
            "rel_orbit_number",
        ]
        non_none = {attr: val for attr in attrs if (val := getattr(self, attr)) is not None}
        params = ", ".join(f"{k}={v}" for k, v in non_none.items())
        return f"SearchQuery({params})"
