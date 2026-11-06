"""Product stand-ins shared by the compositing tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pandas import Timestamp

# ---------------------------------------------------------------------------
# Lightweight product stubs
#
# RTCProduct and CoherenceProduct derive rel_orbit_number via a LUT lookup
# (BurstFootprintMixin._lut_lookup → create_s1_orbit_number_lut).  Accessing
# it on a real product requires a valid (burst_id, sub_swath) pair in the ESA
# burst map, which we cannot guarantee in unit tests.  Instead we use small
# NamedTuple stubs that expose only the attributes consumed by
# the mosaicking and grouping.
# ---------------------------------------------------------------------------


class RTCStub(NamedTuple):
    """Minimal RTCProduct-like stub."""

    rel_orbit_number: int
    date: Timestamp
    polarization: str
    burst_id: int = 0
    sub_swath: str = "IW1"
