"""An inclusive window of calendar days, shared by the search and the composites."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from pandas import Timedelta, Timestamp

type DateLike = date | datetime | Timestamp | str
"""A calendar day, or an instant or ISO string that names one."""


def calendar_day(value: DateLike) -> date:
    """Return the UTC calendar day ``value`` falls on.

    A zoned instant is converted to UTC first; a naive one is taken as UTC. A
    string is parsed the way :class:`pandas.Timestamp` parses it.
    """
    if isinstance(value, str):
        value = Timestamp(value)
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = Timestamp(value).tz_convert("UTC").tz_localize(None)
        return value.date()
    return value


@dataclass(frozen=True, init=False)
class TimeRange:
    """An inclusive window of calendar days, ``start`` through ``end``.

    The one object a pipeline hands to both the search and the composite:
    :class:`~s1bursts.query.SearchQuery` selects the acquisitions that fall on
    any of its days, and ``ResolutionWeighter.compose_*()`` names and tags the
    product with it. Because both take the same window, the product's dates
    describe exactly what was searched.

    Days have no time of day, so there is nothing to be exclusive about: a
    window ``2023-07-18`` through ``2023-10-15`` covers every acquisition on
    October 15. Filters that compare instants get the same span from
    :meth:`half_open`.

    Parameters
    ----------
    start, end
        First and last day of the window. An instant names the UTC calendar day
        it falls on, the resolution every date tag is written at.

    Raises
    ------
    ValueError
        If ``end`` is before ``start``.
    """

    start: date
    end: date

    def __init__(self, start: DateLike, end: DateLike) -> None:
        object.__setattr__(self, "start", calendar_day(start))
        object.__setattr__(self, "end", calendar_day(end))
        self.__post_init__()

    def __post_init__(self) -> None:
        """Check the order of the days.

        Separate from ``__init__`` because a validation library that builds a
        dataclass from its fields, pydantic for one, skips ``__init__`` but
        still calls this.
        """
        if self.end < self.start:
            msg = f"time range end {self.end} is before its start {self.start}"
            raise ValueError(msg)

    def __contains__(self, value: DateLike) -> bool:
        """Whether ``value`` falls on one of the window's days."""
        return self.start <= calendar_day(value) <= self.end

    def __str__(self) -> str:
        """Return the window as ``start..end``."""
        return f"{self.start}..{self.end}"

    def half_open(self) -> tuple[Timestamp, Timestamp]:
        """Return the window as ``[start, stop)`` naive UTC instants.

        ``stop`` is the midnight after ``end``, so a filter that keeps
        ``start <= t < stop`` keeps every instant of every day of the window.
        """
        return Timestamp(self.start), Timestamp(self.end) + Timedelta(days=1)
