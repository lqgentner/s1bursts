"""Tests for the s1bursts exception hierarchy."""

from __future__ import annotations

import errno
import os
import pickle

import pytest

from s1bursts.errors import (
    BurstDownloadError,
    BurstError,
    BurstMapUnavailableError,
    BurstNotFoundError,
    BurstNotInMapError,
    CatalogError,
    CompositeError,
    CompositeInputError,
    ConfigurationError,
    EmptyCompositeError,
    InconsistentInputsError,
    OperationFailedError,
    S1BurstsError,
    SnapProcessingError,
    TooManyResultsError,
    TransformationGridError,
    _copy_error,
    _is_environment_wide,
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


def test_operation_failed_error_is_not_a_per_item_error() -> None:
    """Raised once a batch has finished, so batch handlers must not record it per item."""
    assert issubclass(OperationFailedError, S1BurstsError)
    assert not issubclass(OperationFailedError, BurstError)


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT, errno.EROFS])
def test_a_full_or_read_only_disk_is_environment_wide(code: int) -> None:
    assert _is_environment_wide(OSError(code, os.strerror(code)))


def test_an_environment_wide_cause_is_found_behind_a_wrapper() -> None:
    wrapper = RuntimeError("Failed to write COG")
    wrapper.__cause__ = OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    assert _is_environment_wide(wrapper)


@pytest.mark.parametrize(
    "message",
    [
        "TIFFWriteEncodedTile: No space left on device",
        "Error: java.io.IOException: No space left on device",
        f"write failed: {os.strerror(errno.EROFS)}",
        f"write failed: {os.strerror(errno.EDQUOT)}",
    ],
)
def test_a_message_reporting_a_full_or_read_only_disk_is_environment_wide(message: str) -> None:
    """GPT and GDAL report a full disk without an errno."""
    wrapper = RuntimeError("Failed to write COG")
    wrapper.__cause__ = RuntimeError(message)

    assert _is_environment_wide(wrapper)


def test_a_copy_keeps_type_arguments_attributes_and_cause_but_no_notes() -> None:
    cause = OSError(errno.EIO, "Input/output error")
    error = _DetailedError("boom")
    error.detail = "kept"
    error.__cause__ = cause
    error.add_note("Stopped with 1 of 2 items finished")

    duplicate = _copy_error(error)
    duplicate.add_note("another")

    assert duplicate is not error
    assert type(duplicate) is _DetailedError
    assert duplicate.args == ("boom",)
    assert duplicate.detail == "kept"
    assert duplicate.__cause__ is cause
    assert duplicate.__notes__ == ["another"]
    assert error.__notes__ == ["Stopped with 1 of 2 items finished"]


class _DetailedError(BurstError):
    """An error with an attribute of its own."""

    detail: str = ""


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(FileNotFoundError(errno.ENOENT, "missing"), id="missing-file"),
        pytest.param(OSError("no errno"), id="no-errno"),
        pytest.param(ValueError("not an OSError"), id="value-error"),
    ],
)
def test_other_errors_are_not_environment_wide(error: Exception) -> None:
    assert not _is_environment_wide(error)


def test_a_cause_cycle_ends_the_search() -> None:
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__cause__ = second
    second.__cause__ = first

    assert not _is_environment_wide(first)


def test_snap_processing_error_keeps_its_attributes_through_pickle() -> None:
    """Errors cross process boundaries, as in a process pool."""
    error = SnapProcessingError("GPT processing failed", returncode=-9, output_tail="40%")
    error.add_note("Stopped with 2 of 6 items finished")

    restored = pickle.loads(pickle.dumps(error))  # noqa: S301

    assert str(restored) == "GPT processing failed"
    assert restored.returncode == -9
    assert restored.output_tail == "40%"
    assert restored.__notes__ == ["Stopped with 2 of 6 items finished"]


def test_snap_processing_error_attributes_default_to_none() -> None:
    error = SnapProcessingError("staging failed")

    assert error.returncode is None
    assert error.output_tail is None


def test_a_full_disk_in_the_gpt_output_is_environment_wide() -> None:
    """GPT prints the IOException; the message keeps only its own error line."""
    error = SnapProcessingError(
        "GPT processing failed with exit 1: Error: Cannot write target product",
        output_tail="java.io.IOException: No space left on device",
    )

    assert _is_environment_wide(error)
    assert not _is_environment_wide(SnapProcessingError("boom", output_tail="40%"))


@pytest.mark.parametrize(
    ("exc_type", "builtin"),
    [
        (EmptyCompositeError, ValueError),
        (CompositeInputError, OSError),
    ],
)
def test_composite_errors_share_a_per_cell_base(
    exc_type: type[Exception],
    builtin: type[Exception],
) -> None:
    """One handler catches every per-cell failure, and the builtin handlers still work."""
    assert issubclass(exc_type, CompositeError)
    assert issubclass(exc_type, builtin)
    assert issubclass(CompositeError, S1BurstsError)
    assert not issubclass(CompositeError, ValueError | OSError | BurstError)


def test_inconsistent_inputs_stop_a_loop_over_cells() -> None:
    """Inputs on another grid fail every cell they are given to alike."""
    # Previously a CompositeError, so a loop over cells recorded it once per cell.
    assert not issubclass(InconsistentInputsError, CompositeError)
    assert issubclass(InconsistentInputsError, S1BurstsError)
    assert issubclass(InconsistentInputsError, ValueError)


def test_catalog_error_keeps_its_status_code_through_pickle() -> None:
    error = CatalogError("HTTP 414", status_code=414)

    restored = pickle.loads(pickle.dumps(error))  # noqa: S301

    assert str(restored) == "HTTP 414"
    assert restored.status_code == 414
    assert CatalogError("malformed").status_code is None
