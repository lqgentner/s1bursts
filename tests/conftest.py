"""Shared pytest configuration."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# Each marker whose skips are prerequisite-driven, and the variable that
# declares those prerequisites present. Validation is listed too: its
# references have to be downloaded and, for HyP3, ordered ahead of time, so a
# run with nothing available skips every comparison and still reports success.
STRICT_MARKER_ENV = {
    "integration": "S1BURSTS_REQUIRE_INTEGRATION",
    "validation": "S1BURSTS_REQUIRE_VALIDATION",
}


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
    call: pytest.CallInfo[None],  # noqa: ARG001
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """Turn a skipped integration or validation test into a failure when strict mode is on.

    These tests skip themselves when their prerequisites are missing, so that
    the suite degrades gracefully on a machine without CDSE credentials or
    SNAP. Where the prerequisites are all provided, a skip means something is
    misconfigured rather than absent — and a run that skipped everything is
    otherwise indistinguishable from one that passed.

    Setting the variable named in :data:`STRICT_MARKER_ENV` for a marker makes
    each such skip a failure, reported against the test that skipped and
    carrying its original reason. Note that no validation claim skips itself
    over what it measured: which claims a scene carries is decided when the
    tests are collected, so the skips this catches are missing prerequisites
    and, for the coherence campaign, a reference that was never ordered or has
    expired.
    """
    report = yield

    if not report.skipped:
        return report
    strict = [
        variable
        for marker, variable in STRICT_MARKER_ENV.items()
        if item.get_closest_marker(marker) is not None and os.environ.get(variable)
    ]
    if not strict:
        return report
    variable = strict[0]

    # For a skip, longrepr is a (path, lineno, reason) tuple.
    reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
    report.outcome = "failed"
    report.longrepr = (
        f"{item.nodeid} was skipped while {variable} is set.\n"
        f"Reason: {reason}\n"
        "Every prerequisite is expected to be available here; "
        "unset the variable to allow skipping."
    )
    return report
