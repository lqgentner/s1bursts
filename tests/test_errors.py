"""Tests for the s1bursts exception hierarchy."""

from __future__ import annotations

import pytest

from s1bursts.errors import (
    BurstDownloadError,
    BurstError,
    BurstMapUnavailableError,
    BurstNotFoundError,
    BurstNotInMapError,
    CatalogError,
    ConfigurationError,
    S1BurstsError,
    SnapProcessingError,
    TooManyResultsError,
    TransformationGridError,
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


def test_too_many_results_error_is_not_a_burst_error() -> None:
    """An oversized search is the whole query's failure, not one burst's."""
    assert not issubclass(TooManyResultsError, BurstError)
    assert issubclass(TooManyResultsError, S1BurstsError)


@pytest.mark.parametrize(
    ("exc_type", "builtin"),
    [
        (CatalogError, RuntimeError),
        (TransformationGridError, RuntimeError),
        (BurstNotInMapError, LookupError),
    ],
)
def test_domain_errors_keep_a_builtin_base(
    exc_type: type[Exception],
    builtin: type[Exception],
) -> None:
    """Handlers written for the builtin these replaced still catch them."""
    assert issubclass(exc_type, S1BurstsError)
    assert issubclass(exc_type, builtin)
    assert not issubclass(exc_type, BurstError)


@pytest.mark.parametrize("exc_type", [TransformationGridError, BurstMapUnavailableError])
def test_missing_environment_is_a_configuration_error(exc_type: type[Exception]) -> None:
    """A missing PROJ grid or burst map affects every item, so it must fail fast."""
    assert issubclass(exc_type, ConfigurationError)


def test_burst_map_errors_are_not_their_former_builtins() -> None:
    """Neither is caught by handlers that record a missing file or key per item."""
    assert not issubclass(BurstMapUnavailableError, FileNotFoundError)
    assert not issubclass(BurstNotInMapError, KeyError)


def test_too_many_results_error_is_not_a_catalog_error() -> None:
    """An oversized query is the caller's to narrow, not an unusable response."""
    assert not issubclass(TooManyResultsError, CatalogError)


def test_burst_error_handler_does_not_catch_configuration_error() -> None:
    error = ConfigurationError("missing credentials")

    def batch_handler() -> None:
        try:
            raise error
        except BurstError:  # pragma: no cover - must not be reached
            pytest.fail("ConfigurationError must not be caught as BurstError")

    with pytest.raises(ConfigurationError):
        batch_handler()
