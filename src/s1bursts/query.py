"""Search query parameters for the Sentinel-1 burst catalog."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

import pandas as pd
from pandas import Timestamp

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
    time_range : tuple[str | Timestamp, str | Timestamp] | None, optional
        Start and end of the acquisition time window. Accepts ISO strings or Timestamps.
    intersects : HasWkt | None, optional
        Spatial filter; bursts must intersect this geometry.
    burst_id : int | None, optional
        Sentinel-1 relative burst ID, unique within each sub-swath. Must be positive.
    abs_burst_id : int | None, optional
        Sentinel-1 absolute burst ID, globally unique across all sub-swaths and orbits.
        Must be positive.
    polarization : {"VV", "VH", "HH", "HV"} | None, optional
        Polarization channel.
    mode : {"IW", "EW"} | None, optional
        Acquisition mode.
    sub_swath : {"IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"} | None, optional
        Sub-swath identifier.
    pass_direction : {"Ascending", "Descending"} | None, optional
        Pass direction.
    platform_serial_id : {"A", "B", "C", "D"} | None, optional
        Sentinel-1 platform serial identifier.
    rel_orbit_number : int | None, optional
        Relative orbit number (1-175).

    Raises
    ------
    ValueError
        If any value falls outside its accepted range or set of identifiers.
    """

    def __init__(
        self,
        time_range: tuple[str | Timestamp, str | Timestamp] | None = None,
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
        self.time_range = self._convert_to_timestamp_range(time_range)
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
        """Validate that a value is one of the accepted identifiers.

        Parameters
        ----------
        value : str | None
            Value to validate.
        field_name : str
            Name of the field, used in the error message.
        allowed : tuple of str
            Accepted values.

        Returns
        -------
        str | None
            The validated value.

        Raises
        ------
        ValueError
            If value is not one of ``allowed``.
        """
        if value is not None and value not in allowed:
            msg = f"{field_name} must be one of {', '.join(allowed)}; received '{value}'"
            raise ValueError(msg)
        return value

    @staticmethod
    def _validate_geometry(geometry: HasWkt | None) -> HasWkt | None:
        """Validate that geometry implements the WKT protocol.

        Parameters
        ----------
        geometry : HasWkt | None
            Geometry to validate.

        Returns
        -------
        HasWkt | None
            The validated geometry.

        Raises
        ------
        ValueError
            If geometry does not implement `wkt`.
        """
        if geometry is not None and not isinstance(geometry, HasWkt):
            msg = "`intersects` must be a Geometry that implements `wkt`."
            raise ValueError(msg)
        return geometry

    @staticmethod
    def _convert_to_timestamp_range(
        time_range: tuple[str | Timestamp, str | Timestamp] | None,
    ) -> tuple[Timestamp, Timestamp] | None:
        """Convert a time range tuple to Timestamps, validating ordering.

        Parameters
        ----------
        time_range : tuple[str | Timestamp, str | Timestamp] | None
            Start and end times as strings or Timestamps.

        Returns
        -------
        tuple[Timestamp, Timestamp] | None
            Converted time range, or None if input is None.

        Raises
        ------
        TypeError
            If time_range is not a sequence.
        ValueError
            If time_range does not have exactly 2 elements or start >= end.
        """
        if time_range is None:
            return None

        if not isinstance(time_range, Sequence):
            msg = f"time_range must be of type tuple, received `{type(time_range).__name__}`."
            raise TypeError(msg)
        try:
            start_raw, end_raw = time_range
        except ValueError:
            msg = "time_range tuple must have exactly 2 elements"
            raise ValueError(msg) from None

        start = pd.Timestamp(start_raw)
        end = pd.Timestamp(end_raw)

        if start >= end:
            msg = f"Start time {start} must be before end time {end}"
            raise ValueError(msg)

        return (start, end)

    @staticmethod
    def _validate_int_range(
        value: int | None,
        field_name: str,
        *,
        minimum: int,
        maximum: int | None = None,
    ) -> int | None:
        """Validate that an integer falls inside an inclusive range.

        Parameters
        ----------
        value : int | None
            Value to validate.
        field_name : str
            Name of the field, used in the error message.
        minimum : int
            Smallest accepted value.
        maximum : int, optional
            Largest accepted value; unbounded above if omitted.

        Returns
        -------
        int | None
            The validated value.

        Raises
        ------
        ValueError
            If value falls outside the range.
        """
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
