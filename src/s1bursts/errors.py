"""Exception hierarchy for the Sentinel-1 pipeline.

The types here separate two kinds of failure so that batch loops can catch only
the ones worth recording per item and let the rest abort the run:

- ``BurstError`` (and subclasses) are *recoverable, per-item* failures — a
  single burst could not be downloaded or processed. Batch loops catch this base
  class, record an ``ErrorResult``, and continue.
- ``ConfigurationError`` is an *environment-wide* failure (missing
  credentials, a missing PROJ grid or burst-ID map, a missing GPT executable).
  It affects every item, so it propagates and fails fast; per-item handlers
  do not catch it. The one exception is the credentials of a download
  endpoint: ``BurstDatabase`` turns their absence into a per-burst
  ``BurstDownloadError``, so that the other endpoint is still tried.

Every other ``S1BurstsError`` fails the whole call that raised it, as does
anything that is not one (a plain ``ValueError`` for an invalid argument, a
third-party ``CRSError``). Leaving those to propagate keeps an environment-wide
problem from being silently logged once per item.
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


class TransformationGridError(ConfigurationError, RuntimeError):
    """A PROJ grid needed for a DEM's vertical conversion is missing or could not be downloaded."""


class BurstMapUnavailableError(ConfigurationError):
    """The Sentinel-1 burst-ID map is needed but not downloaded or unreadable."""


class BurstNotInMapError(S1BurstsError, LookupError):
    """A burst's ``(burst_id, sub_swath)`` is not in the Sentinel-1 burst-ID map.

    Every burst the mission acquires is in the map, so this points to a
    malformed burst rather than one missing at the source, which is
    ``BurstNotFoundError``.
    """


class CatalogError(S1BurstsError, RuntimeError):
    """The CDSE catalog returned a response that cannot be used."""


class EmptyCompositeError(S1BurstsError, ValueError):
    """No input observation has a valid pixel in the target cell.

    A burst whose footprint touches a cell need not reach any of its pixels
    once geocoded. A composite of such inputs would hold no value, so none is
    written, and the cell is left without one, as if no burst had matched.
    """


class TooManyResultsError(S1BurstsError):
    """More bursts match a search than one catalog request returns.

    The catalog serves at most 1000 bursts per request. A query that matches
    more is refused rather than paged through, so that an oversized AOI or time
    window fails after one request. Narrow it, or split it into several
    searches and combine their results with ``+``.
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
