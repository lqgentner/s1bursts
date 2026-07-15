"""Search query parameters for the Sentinel-1 burst catalog."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

import pandas as pd
from pandas import Timestamp

type Polarization = Literal["VV", "VH", "HH", "HV"]
type Mode = Literal["IW", "EW"]
type SubSwath = Literal["IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"]
type OrbitDirection = Literal["ASCENDING", "DESCENDING"]
type PlatformSerialID = Literal["A", "B", "C", "D"]


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
        Sentinel-1 relative burst ID, unique within each sub-swath.
    abs_burst_id : int | None, optional
        Sentinel-1 absolute burst ID, globally unique across all sub-swaths and orbits.
    polarization : {"VV", "VH", "HH", "HV"} | None, optional
        Polarization channel.
    mode : {"IW", "EW"} | None, optional
        Acquisition mode.
    sub_swath : {"IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"} | None, optional
        Sub-swath identifier.
    orbit_direction : {"ASCENDING", "DESCENDING"} | None, optional
        Orbit direction.
    platform_serial_id : {"A", "B", "C", "D"} | None, optional
        Sentinel-1 platform serial identifier.
    rel_orbit_number : int | None, optional
        Relative orbit number (1-175).
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
        orbit_direction: OrbitDirection | None = None,
        platform_serial_id: PlatformSerialID | None = None,
        rel_orbit_number: int | None = None,
    ) -> None:
        self.time_range = self._convert_to_timestamp_range(time_range)
        self.intersects = self._validate_geometry(intersects)
        self.burst_id = self._validate_positive_int(burst_id, "burst_id")
        self.abs_burst_id = self._validate_positive_int(abs_burst_id, "abs_burst_id")
        self.polarization = polarization
        self.sub_swath = sub_swath
        self.mode = mode
        self.orbit_direction = orbit_direction
        self.platform_serial_id = platform_serial_id
        self.rel_orbit_number = self._validate_positive_int(rel_orbit_number, "rel_orbit_number")

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
    def _validate_positive_int(value: int | None, field_name: str) -> int | None:
        """Validate that an integer value is non-negative.

        Parameters
        ----------
        value : int | None
            Value to validate.
        field_name : str
            Name of the field, used in the error message.

        Returns
        -------
        int | None
            The validated value.

        Raises
        ------
        ValueError
            If value is negative.
        """
        if value is not None and value < 0:
            msg = f"{field_name} must be non-negative"
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
            "orbit_direction",
            "platform_serial_id",
            "rel_orbit_number",
        ]
        non_none = {attr: val for attr in attrs if (val := getattr(self, attr)) is not None}
        params = ", ".join(f"{k}={v}" for k, v in non_none.items())
        return f"SearchQuery({params})"
