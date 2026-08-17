"""Shared pytest configuration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

REQUIRE_INTEGRATION_ENV = "S1BURSTS_REQUIRE_INTEGRATION"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],  # noqa: ARG001
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Turn a skipped integration test into a failure when strict mode is on.

    Integration tests skip themselves when their prerequisites are missing, so
    that the suite degrades gracefully on a machine without CDSE credentials or
    SNAP. In CI those prerequisites are all provided, so a skip there means
    something is misconfigured rather than absent — and a run that skipped
    everything is otherwise indistinguishable from one that passed.

    Setting ``S1BURSTS_REQUIRE_INTEGRATION=1`` makes each such skip a failure,
    reported against the test that skipped and carrying its original reason.
    """
    report = yield

    if not os.environ.get(REQUIRE_INTEGRATION_ENV):
        return report
    if not report.skipped or item.get_closest_marker("integration") is None:
        return report

    # For a skip, longrepr is a (path, lineno, reason) tuple.
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    report.outcome = "failed"
    report.longrepr = (
        f"{item.nodeid} was skipped while {REQUIRE_INTEGRATION_ENV} is set.\n"
        f"Reason: {reason}\n"
        "Every integration prerequisite is expected to be available here; "
        "unset the variable to allow skipping."
    )
    return report
