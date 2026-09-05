"""Tests for the SAFE comparison helpers the conformance runs report through."""

from __future__ import annotations

from lxml import etree
import numpy as np

from tests.safe_conformance import BurstSafe, _diff_gcps

LINES_PER_BURST = 1500


def _safe(gcps: list[tuple[float, float, float, float, float]]) -> BurstSafe:
    """Build a BurstSafe carrying only the GCPs and burst length under test."""
    empty = etree.fromstring("<empty/>")
    annotation = etree.fromstring(
        "<product><swathTiming>"
        f"<linesPerBurst>{LINES_PER_BURST}</linesPerBurst>"
        "</swathTiming></product>",
    )
    return BurstSafe(
        safe_dir="burst.SAFE",
        pattern="",
        members=[],
        manifest=empty,
        annotation=annotation,
        calibration=empty,
        noise=empty,
        rfi=None,
        measurement=np.empty((0, 0)),
        gcps=gcps,
    )


def _shifted(offsets: list[float]) -> tuple[BurstSafe, BurstSafe]:
    ours = [(10.0, 100.0 * i, 1.0, 2.0, 3.0) for i in range(len(offsets))]
    theirs = [
        (col, row + offset, *position)
        for (col, row, *position), offset in zip(ours, offsets, strict=True)
    ]
    # GCP tuples carry the optional height element.
    # pyrefly: ignore [bad-argument-type]
    return _safe(ours), _safe(theirs)


class TestDiffGcps:
    def test_identical_rows_are_no_difference(self) -> None:
        assert list(_diff_gcps(*_shifted([0.0, 0.0, 0.0]))) == []

    def test_a_whole_burst_shift_is_the_known_cdse_deviation(self) -> None:
        offset = float(LINES_PER_BURST)
        (message,) = _diff_gcps(*_shifted([offset, offset, offset]))

        assert message.startswith("KNOWN DEVIATION:")
        assert f"offset by {LINES_PER_BURST} lines" in message

    def test_a_uniform_partial_shift_reports_the_offset_it_saw(self) -> None:
        """The single-offset case must still name the number.

        It is what tells a real GCP disagreement from the known double shift,
        so it must survive the check for that shift.
        """
        (message,) = _diff_gcps(*_shifted([7.0, 7.0, 7.0]))

        assert message == "GCP rows differ by inconsistent offsets: [7.0]"

    def test_mixed_shifts_report_every_offset(self) -> None:
        (message,) = _diff_gcps(*_shifted([0.0, 7.0, 3.0]))

        assert message == "GCP rows differ by inconsistent offsets: [0.0, 3.0, 7.0]"

    def test_a_moved_coordinate_is_a_real_disagreement(self) -> None:
        ours = _safe([(10.0, 0.0, 1.0, 2.0, 3.0)])
        theirs = _safe([(10.0, 0.0, 1.0, 2.0, 4.0)])

        (message,) = _diff_gcps(ours, theirs)

        assert message.startswith("GCP (10.0, 0.0")

    def test_a_differing_count_is_reported_before_any_offset(self) -> None:
        (message,) = _diff_gcps(_safe([(10.0, 0.0, 1.0, 2.0, 3.0)]), _safe([]))

        assert message == "GCP count 1 != 0"
