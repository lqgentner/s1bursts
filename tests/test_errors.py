"""Tests for the s1bursts exception hierarchy."""

from __future__ import annotations

import pytest

from s1bursts.errors import (
    BurstDownloadError,
    BurstError,
    BurstNotFoundError,
    ConfigurationError,
    S1BurstsError,
    SnapProcessingError,
)


@pytest.mark.parametrize(
    "exc_type",
    [BurstNotFoundError, BurstDownloadError, SnapProcessingError],
)
def test_per_item_errors_are_burst_errors(exc_type: type[Exception]) -> None:
    assert issubclass(exc_type, BurstError)
    assert issubclass(exc_type, S1BurstsError)


def test_configuration_error_is_not_a_burst_error() -> None:
    """Environment-wide failures must not be swallowed by per-item handlers."""
    assert not issubclass(ConfigurationError, BurstError)
    assert issubclass(ConfigurationError, S1BurstsError)


def test_burst_error_handler_does_not_catch_configuration_error() -> None:
    error = ConfigurationError("missing credentials")

    def batch_handler() -> None:
        try:
            raise error
        except BurstError:  # pragma: no cover - must not be reached
            pytest.fail("ConfigurationError must not be caught as BurstError")

    with pytest.raises(ConfigurationError):
        batch_handler()
