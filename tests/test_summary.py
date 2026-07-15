"""Tests for s1bursts.summary result containers."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from s1bursts.errors import BurstDownloadError
from s1bursts.summary import ErrorResult, OkResult, OperationSummary


def make_summary() -> OperationSummary[str]:
    results = [
        OkResult(source="a", result="product-a"),
        OkResult(source="b", result="product-b", skipped=True),
        ErrorResult(source="c", error=BurstDownloadError("download failed")),
    ]
    return OperationSummary.from_results(results, start_time=0.0, end_time=120.0)


class TestOperationSummary:
    def test_from_results_counts(self) -> None:
        summary = make_summary()

        assert summary.n_total == 3
        assert summary.n_processed == 1
        assert summary.n_skipped == 1
        assert summary.n_failed == 1
        assert summary.processing_time == 120.0

    def test_products_include_skipped(self) -> None:
        summary = make_summary()

        assert summary.products == ["product-a", "product-b"]

    def test_failed_items(self) -> None:
        summary = make_summary()

        assert len(summary.failed_items) == 1
        assert isinstance(summary.failed_items[0].error, BurstDownloadError)

    def test_success_rate(self) -> None:
        assert make_summary().success_rate == pytest.approx(2 / 3)

    def test_success_rate_empty(self) -> None:
        summary = OperationSummary.from_results([], start_time=0.0, end_time=0.0)

        assert summary.success_rate == 1.0

    def test_print_summary_lists_failures(self) -> None:
        buffer = io.StringIO()
        console = Console(file=buffer, force_jupyter=False, width=120)

        make_summary().print_summary(console=console)

        output = buffer.getvalue()
        assert "Operation Summary" in output
        assert "Failed Items:" in output
        assert "BurstDownloadError" in output
        assert "download failed" in output

    def test_print_summary_without_failures(self) -> None:
        buffer = io.StringIO()
        console = Console(file=buffer, force_jupyter=False, width=120)
        summary = OperationSummary.from_results(
            [OkResult(source="a", result="product-a")],
            start_time=0.0,
            end_time=1.0,
        )

        summary.print_summary(console=console)

        assert "Failed Items:" not in buffer.getvalue()
