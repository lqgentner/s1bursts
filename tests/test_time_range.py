"""Tests for s1bursts.time_range.TimeRange."""

from __future__ import annotations

from datetime import UTC, date, datetime

from pandas import Timestamp
import pytest

from s1bursts.time_range import TimeRange


class TestConstruction:
    def test_holds_calendar_days(self) -> None:
        window = TimeRange(date(2023, 7, 18), date(2023, 10, 15))
        assert (window.start, window.end) == (date(2023, 7, 18), date(2023, 10, 15))

    def test_accepts_iso_strings(self) -> None:
        assert TimeRange("2023-07-18", "2023-10-15") == TimeRange(
            date(2023, 7, 18), date(2023, 10, 15)
        )

    def test_an_instant_names_its_day(self) -> None:
        window = TimeRange(Timestamp("2023-07-18"), datetime(2023, 10, 15, 17, 12, 44, tzinfo=UTC))
        assert (window.start, window.end) == (date(2023, 7, 18), date(2023, 10, 15))

    def test_a_zoned_instant_names_its_utc_day(self) -> None:
        """Acquisitions are compared as UTC days, so the window is read the same way."""
        window = TimeRange(Timestamp("2023-07-18T01:00+02:00"), "2023-10-15")
        assert window.start == date(2023, 7, 17)

    def test_a_single_day_is_a_window(self) -> None:
        assert TimeRange("2023-07-18", "2023-07-18").end == date(2023, 7, 18)

    def test_end_before_start_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="before its start"):
            TimeRange("2023-10-15", "2023-07-18")

    def test_is_immutable(self) -> None:
        window = TimeRange("2023-07-18", "2023-10-15")
        with pytest.raises(AttributeError):
            window.end = date(2024, 1, 1)  # pyrefly: ignore [read-only]


class TestMembership:
    window = TimeRange("2023-07-18", "2023-10-15")

    def test_every_day_of_the_window_counts(self) -> None:
        assert date(2023, 7, 18) in self.window
        assert date(2023, 10, 15) in self.window
        assert date(2023, 7, 17) not in self.window
        assert date(2023, 10, 16) not in self.window

    def test_an_instant_on_the_last_day_counts(self) -> None:
        assert Timestamp("2023-10-15T23:59:59", tz="UTC") in self.window
        assert Timestamp("2023-10-16T00:00:00", tz="UTC") not in self.window


class TestHalfOpen:
    def test_covers_every_instant_of_every_day(self) -> None:
        start, stop = TimeRange("2023-07-18", "2023-10-15").half_open()
        assert start == Timestamp("2023-07-18")
        assert stop == Timestamp("2023-10-16")
        assert start.tzinfo is None
        assert stop.tzinfo is None
        assert start <= Timestamp("2023-10-15T17:12:44") < stop
