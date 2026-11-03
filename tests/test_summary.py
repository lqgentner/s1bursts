"""Tests for s1bursts.summary result containers."""

from __future__ import annotations

import io
import pickle

import pytest
from rich.console import Console

from s1bursts.errors import (
    BurstDownloadError,
    BurstError,
    BurstNotFoundError,
    OperationFailedError,
    S1BurstsError,
    SnapProcessingError,
)
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

    def test_raise_on_failure_returns_the_summary_without_failures(self) -> None:
        summary = OperationSummary.from_results(
            [OkResult(source="a", result="product-a")],
            start_time=0.0,
            end_time=1.0,
        )

        assert summary.raise_on_failure() is summary

    def test_raise_on_failure_lists_the_failed_items(self) -> None:
        summary = make_summary()

        with pytest.raises(OperationFailedError) as excinfo:
            summary.raise_on_failure()

        assert isinstance(excinfo.value, S1BurstsError)
        assert str(excinfo.value) == (
            "1 of 3 items failed:\n  c: BurstDownloadError: download failed"
        )
        assert excinfo.value.summary is summary
        assert excinfo.value.failed_items == summary.failed_items

    def test_the_error_keeps_the_products_that_were_made(self) -> None:
        with pytest.raises(OperationFailedError) as excinfo:
            make_summary().raise_on_failure()

        assert excinfo.value.summary.products == ["product-a", "product-b"]

    def test_the_error_survives_pickling(self) -> None:
        with pytest.raises(OperationFailedError) as excinfo:
            make_summary().raise_on_failure()

        # Round-trips an object this test made, as multiprocessing would.
        excinfo.value.add_note("while processing cell 32TMT64")
        restored = pickle.loads(pickle.dumps(excinfo.value))  # noqa: S301

        assert isinstance(restored, OperationFailedError)
        assert str(restored) == str(excinfo.value)
        assert restored.__notes__ == ["while processing cell 32TMT64"]
        assert restored.summary.products == ["product-a", "product-b"]
        (failure,) = restored.failed_items
        assert failure.source == "c"
        assert isinstance(failure.error, BurstDownloadError)

    def test_raise_on_failure_raises_when_everything_failed(self) -> None:
        summary = OperationSummary[str].from_results(
            [ErrorResult(source="a", error=BurstDownloadError("gone"))],
            start_time=0.0,
            end_time=1.0,
        )

        with pytest.raises(OperationFailedError, match="1 of 1 items failed"):
            summary.raise_on_failure()

    def test_raise_on_failure_counts_failures_beyond_the_listed_ones(self) -> None:
        summary = OperationSummary[str].from_results(
            [ErrorResult(source=i, error=BurstDownloadError("gone")) for i in range(25)],
            start_time=0.0,
            end_time=1.0,
        )

        with pytest.raises(OperationFailedError) as excinfo:
            summary.raise_on_failure()

        message = str(excinfo.value).splitlines()
        assert len(message) == 22
        assert message[-1] == "  ... and 5 more, see `failed_items`"
        assert len(excinfo.value.failed_items) == 25

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


def summary_of(*errors: Exception) -> OperationSummary[str]:
    """Return a summary of one success and one failure per error, sourced f0, f1, ..."""
    results: list[OkResult[str] | ErrorResult] = [OkResult(source="ok", result="product")]
    results += [ErrorResult(source=f"f{i}", error=error) for i, error in enumerate(errors)]
    return OperationSummary.from_results(results, start_time=0.0, end_time=1.0)


class TestRaiseOnFailureIgnore:
    def test_ignored_failures_do_not_raise_but_stay_in_the_summary(self) -> None:
        summary = summary_of(BurstNotFoundError("gone"), BurstNotFoundError("gone too"))

        assert summary.raise_on_failure(ignore=BurstNotFoundError) is summary
        assert summary.n_failed == 2
        assert [r.source for r in summary.failed_items] == ["f0", "f1"]

    def test_a_tuple_ignores_any_of_its_types(self) -> None:
        summary = summary_of(BurstNotFoundError("gone"), SnapProcessingError("gpt"))

        assert (
            summary.raise_on_failure(ignore=(BurstNotFoundError, SnapProcessingError)) is summary
        )

    def test_a_base_class_ignores_its_subclasses(self) -> None:
        summary = summary_of(BurstNotFoundError("gone"), BurstDownloadError("timeout"))

        assert summary.raise_on_failure(ignore=BurstError) is summary

    def test_other_failures_still_raise_and_count_the_ignored(self) -> None:
        summary = summary_of(BurstNotFoundError("gone"), BurstDownloadError("timeout"))

        with pytest.raises(OperationFailedError) as excinfo:
            summary.raise_on_failure(ignore=BurstNotFoundError)

        assert str(excinfo.value) == (
            "1 of 3 items failed, besides 1 ignored (BurstNotFoundError):\n"
            "  f1: BurstDownloadError: timeout"
        )
        assert [r.source for r in excinfo.value.failed_items] == ["f1"]
        assert [r.source for r in excinfo.value.summary.failed_items] == ["f0", "f1"]

    def test_the_ignored_types_survive_pickling(self) -> None:
        summary = summary_of(BurstNotFoundError("gone"), BurstDownloadError("timeout"))
        with pytest.raises(OperationFailedError) as excinfo:
            summary.raise_on_failure(ignore=BurstNotFoundError)

        # Round-trips an object this test made, as multiprocessing would.
        restored = pickle.loads(pickle.dumps(excinfo.value))  # noqa: S301

        assert restored.ignore == (BurstNotFoundError,)
        assert str(restored) == str(excinfo.value)


# The sign print_summary counts failures with.
TIMES = "\N{MULTIPLICATION SIGN}"


def printed(summary: OperationSummary[str]) -> str:
    """Return what ``print_summary`` writes, at a fixed width."""
    buffer = io.StringIO()
    summary.print_summary(console=Console(file=buffer, force_jupyter=False, width=200))
    return buffer.getvalue()


class TestPrintSummaryGroups:
    def test_identical_failures_print_as_one_counted_group(self) -> None:
        outage = [BurstDownloadError("503 Service Unavailable") for _ in range(5)]

        output = printed(summary_of(*outage, SnapProcessingError("gpt exited with 1")))

        assert output.count("503 Service Unavailable") == 1
        assert f"• 5 {TIMES} BurstDownloadError: 503 Service Unavailable" in output
        assert f"• 1 {TIMES} SnapProcessingError: gpt exited with 1" in output
        # The larger group comes first.
        assert output.index("503") < output.index("gpt exited")

    def test_a_group_shows_a_few_example_sources(self) -> None:
        output = printed(summary_of(*(BurstDownloadError("503") for _ in range(5))))

        assert "e.g. f0, f1, f2, …" in output
        assert "f3" not in output

    def test_groups_by_type_and_message(self) -> None:
        output = printed(
            summary_of(
                BurstDownloadError("503"), BurstNotFoundError("503"), BurstDownloadError("404")
            ),
        )

        assert output.count(f"• 1 {TIMES}") == 3

    def test_only_the_first_line_of_a_long_message_is_shown(self) -> None:
        first = "gpt failed: " + "x" * 200
        output = printed(summary_of(SnapProcessingError(f"{first}\nstack trace line")))

        assert "stack trace line" not in output
        assert "x" * 120 not in output
        assert "…" in output

    def test_messages_are_printed_literally(self) -> None:
        output = printed(summary_of(BurstDownloadError("bad [red]band[/red] name")))

        assert "bad [red]band[/red] name" in output

    def test_an_empty_message_shows_the_type(self) -> None:
        assert f"• 1 {TIMES} BurstDownloadError" in printed(summary_of(BurstDownloadError()))
