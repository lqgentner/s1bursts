"""Exception hierarchy for the Sentinel-1 pipeline.

The types here separate two kinds of failure so that batch loops can catch only
the ones worth recording per item and let the rest abort the run:

- ``BurstError`` (and subclasses) are *recoverable, per-item* failures — a
  single burst could not be downloaded or processed. Batch loops catch this base
  class, record an ``ErrorResult``, and continue.
- ``ConfigurationError`` is an *environment-wide* failure (missing
  credentials, a broken PROJ database, a missing GPT executable). It affects
  every item, so it must propagate and fail fast — it is deliberately never
  caught by the per-item handlers.

Anything that is neither (a plain ``ValueError`` from a programming bug, a
third-party ``CRSError``) is also left to propagate, which keeps an
environment-wide problem from being silently logged once per item.
"""

from __future__ import annotations


class S1BurstsError(Exception):
    """Base class for all s1bursts errors."""


class ConfigurationError(S1BurstsError):
    """Invalid or missing configuration or environment.

    Raised for problems that affect every item — missing credentials, an
    unusable PROJ/GDAL install, a missing SNAP GPT executable. These propagate
    and fail the run fast; batch handlers must not catch them.
    """


class EmptyCompositeError(S1BurstsError, ValueError):
    """No input observation has a valid pixel in the target cell.

    A burst whose footprint touches a cell need not reach any of its pixels
    once geocoded. A composite of such inputs would hold no value, so none is
    written, and the cell is left without one, as if no burst had matched.
    """


class BurstError(S1BurstsError):
    """Base class for recoverable, per-item burst failures.

    Batch loops catch this type (not bare ``ValueError``/``OSError``) to record
    an ``ErrorResult`` and continue with the remaining items.
    """


class BurstNotFoundError(BurstError):
    """A requested burst or its parent product is not available at the source."""


class BurstDownloadError(BurstError):
    """Downloading or extracting a single burst failed."""


class SnapProcessingError(BurstError):
    """A SNAP GPT graph failed for a single burst or burst pair."""
