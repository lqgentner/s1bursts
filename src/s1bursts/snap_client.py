"""Low-level SNAP GPT client for Sentinel-1 processing."""

from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from functools import lru_cache
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any, Final, Literal, overload
import zipfile

from freezebase.errors import RasterWriteError
from freezebase.raster import (
    build_rasterio_profile,
    get_epsg_string,
    rewrite_tiff,
)
import rasterio
from rasterio.errors import RasterioError

from s1bursts import product_tags
from s1bursts.config import get_settings, warn_if_tmpdir_unusable
from s1bursts.errors import (
    BurstError,
    ConfigurationError,
    NoUTMZoneError,
    SnapProcessingError,
    _is_environment_wide,
)
from s1bursts.product import (
    CoherenceProduct,
    CRSInput,
    GTCProduct,
    InterferogramProduct,
    RTCProduct,
)
from s1bursts.settings_check import _skip_existing
from s1bursts.snap_graph import NODATA_SENTINEL, GTCConfig, RTCConfig
from s1bursts.summary import ErrorResult, OkResult, OperationResult

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from upath import UPath

    from s1bursts.burst import Burst, BurstPair
    from s1bursts.config import SNAPSettings
    from s1bursts.filemanager import FileManager
    from s1bursts.product_tags import RasterMetadata
    from s1bursts.snap_graph import CoherenceConfig, InterferogramConfig

logger = logging.getLogger(__name__)

# Non-heap memory a GPT run needs beyond `-Xmx`: JVM metaspace, thread stacks,
# GDAL's native buffers, and enough left over for the OS page cache.
_JVM_OVERHEAD_MB = 3_000

# Smallest per-worker heap that can load SNAP's operator graph.
_MIN_HEAP_MB = 512

_AUXDATA_SUBDIRS: tuple[str, ...] = ("dem", "Orbits")


def _write_graph(tmp_dir: Path, name: str, xml: str) -> Path:
    """Write a generated SNAP GPT graph XML to ``tmp_dir`` and return its path."""
    graph_file = tmp_dir / f"_{name}_graph.xml"
    graph_file.write_text(xml)
    return graph_file


def _resolve_gpt_path(bin_dir: Path | None, snap_settings: SNAPSettings) -> Path:
    """Resolve and validate the SNAP ``gpt`` executable path.

    Precedence:

    1. ``bin_dir`` (the ``bin`` argument passed to ``SNAPConfig``).
    2. ``SNAP_BIN``, a SNAP installation's ``bin`` directory.
    3. ``gpt`` discovered on ``PATH``.
    """
    resolved_bin = bin_dir or snap_settings.bin
    if resolved_bin is not None:
        candidates = (resolved_bin / name for name in ("gpt", "gpt.exe"))
        found = next((candidate for candidate in candidates if candidate.is_file()), None)
        if found is None:
            msg = f"No `gpt` executable found in SNAP bin directory: {resolved_bin}"
            raise ConfigurationError(msg)
        gpt_path = found
    else:
        which = shutil.which("gpt")
        if which is None:
            msg = (
                "Could not resolve the SNAP `gpt` executable. Set `SNAP_BIN` to your SNAP "
                "installation's `bin` directory, add `gpt` to PATH, or pass `bin` explicitly."
            )
            raise ConfigurationError(msg)
        gpt_path = Path(which)

    if not os.access(gpt_path, os.X_OK):
        msg = f"SNAP GPT executable is not executable: {gpt_path}"
        raise ConfigurationError(msg)
    return gpt_path


# Tested versions, as (major, minor). The CI matrix and README badge must agree;
# tests/test_supported_snap_versions.py enforces that.
SUPPORTED_SNAP_VERSIONS: Final[tuple[tuple[int, int], ...]] = ((13, 0), (14, 0))


_BARE_VERSION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*$")


def _as_version(text: str) -> tuple[int, ...] | None:
    """Parse a dotted version string into a comparable tuple."""
    match = _BARE_VERSION_PATTERN.match(text)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


@lru_cache(maxsize=8)
def snap_version(gpt_path: Path) -> tuple[int, ...] | None:
    """Version of the SNAP installation a ``gpt`` executable belongs to.

    Reads ``VERSION.txt`` from the installation directory. The result is cached
    per executable.

    Parameters
    ----------
    gpt_path
        The ``gpt`` executable, as resolved by ``_resolve_gpt_path``.

    Returns
    -------
    tuple of int or None
        The version as a comparable tuple, e.g. ``(14, 0, 0)``, or ``None``
        where the file is missing or unparsable.
    """
    version_file = gpt_path.parent.parent / "VERSION.txt"
    with suppress(OSError):
        return _as_version(version_file.read_text())
    return None


# First version of SNAP's `snap-dem` module that samples an external DEM where its
# georeferencing places it (SNAP-4264, snap-engine#640). It shipped in the 14.0.2
# module update; no release of the 13.x line has it. tests/test_integration.py
# measures the installed SNAP and fails if this drifts from what it does.
EXTERNAL_DEM_FIXED_IN: Final[tuple[int, ...]] = (14, 0, 2)

_SNAP_DEM_JAR: Final = Path("modules") / "org-esa-snap-snap-dem.jar"


def _jar_module_version(jar: Path) -> tuple[int, ...] | None:
    """Read the version a NetBeans module declares in its jar's manifest."""
    try:
        with zipfile.ZipFile(jar) as archive:
            manifest = archive.read("META-INF/MANIFEST.MF").decode("utf-8", "replace")
    except (OSError, KeyError, zipfile.BadZipFile):
        return None
    for line in manifest.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "OpenIDE-Module-Specification-Version":
            return _as_version(value)
    return None


def _snap_user_dir(install_dir: Path) -> Path | None:
    """Return the userdir SNAP keeps module updates in when its installation is read-only."""
    with suppress(OSError):
        for line in (install_dir / "etc" / "snap.conf").read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "default_userdir":
                expanded = os.path.expandvars(value.strip().strip('"'))
                # A variable expandvars does not know stays in the path.
                if "$" not in expanded:
                    return Path(expanded)
    with suppress(RuntimeError):
        return Path.home() / ".snap" / "system"
    return None


def snap_dem_module_version(gpt_path: Path) -> tuple[int, ...] | None:
    """Version of the ``snap-dem`` module a ``gpt`` executable runs.

    SNAP versions its modules separately from the installation: a module update
    leaves ``VERSION.txt`` unchanged. Reads the version the module's jar declares,
    in the installation and in the user's SNAP directory, where updates land when
    the installation is not writable, and returns the newer. Not cached: the module
    is updated independently of SNAP, possibly while a process is running.

    Parameters
    ----------
    gpt_path
        The ``gpt`` executable, as resolved by ``_resolve_gpt_path``.

    Returns
    -------
    tuple of int or None
        The version as a comparable tuple, e.g. ``(14, 0, 2)``, or ``None``
        where no jar is found or its manifest is unreadable.
    """
    install_dir = gpt_path.parent.parent
    clusters = [install_dir / "snap"]
    if (user_dir := _snap_user_dir(install_dir)) is not None:
        clusters.append(user_dir)
    versions = [
        version
        for cluster in clusters
        if (version := _jar_module_version(cluster / _SNAP_DEM_JAR)) is not None
    ]
    return max(versions, default=None)


def external_dem_needs_half_pixel_shift(bin_dir: Path | None = None) -> bool:
    """Whether SNAP samples an external DEM half a pixel south-east of its georeferencing.

    Where no installation is found by default, or its ``snap-dem`` version cannot
    be read, the bug is assumed present: that is the behavior of every SNAP release
    up to 14.0.1, and of the whole 13.x line.

    Parameters
    ----------
    bin_dir
        A SNAP installation's ``bin`` directory. Defaults to ``SNAP_BIN``, then
        ``gpt`` discovered on ``PATH``.

    Returns
    -------
    bool
        ``True`` if a DEM must be shifted by half a pixel to be sampled in place.

    Raises
    ------
    ConfigurationError
        If ``bin_dir`` is given but contains no ``gpt`` executable.
    """
    # Outside the try: an invalid setting is not a missing installation.
    snap_settings = get_settings().snap
    try:
        gpt_path = _resolve_gpt_path(bin_dir, snap_settings)
    except ConfigurationError:
        if bin_dir is not None:
            raise
        logger.info(
            "No SNAP installation found; assuming it samples an external DEM off by half a pixel"
        )
        return True
    version = snap_dem_module_version(gpt_path)
    if version is None:
        logger.warning(
            "Could not determine the version of SNAP's snap-dem module for '%s'; assuming it "
            "samples an external DEM half a pixel off, as releases before %s do.",
            gpt_path,
            ".".join(str(part) for part in EXTERNAL_DEM_FIXED_IN),
        )
        return True
    return version < EXTERNAL_DEM_FIXED_IN


def _extract_gpt_error(
    stdout: str | None,
    stderr: str | None,
    *,
    max_chars: int = 2000,
) -> str:
    """Pull SNAP's human-readable error out of captured ``gpt`` output.

    ``gpt`` prints an ``Error: ...`` line (and a Java stack trace) before exiting
    non-zero, but ``CalledProcessError.__str__`` carries only the exit code. This
    recovers the ``Error:`` block (up to the stack trace) so the real cause — e.g.
    ``No valid orbit file found`` — survives in the raised exception instead of
    only in a log line. Falls back to the tail of the captured output.
    """
    combined = "\n".join(part for part in (stderr, stdout) if part)
    lines = combined.splitlines()

    # Keep the first ``Error:`` block (stderr is searched before stdout), stopping
    # at the Java stack trace or the next ``Error:`` marker.
    collected: list[str] = []
    for line in lines:
        if not collected:
            if line.startswith("Error:"):
                collected.append(line)
            continue
        if line.startswith("Error:") or line.lstrip().startswith(
            ("at ", "org.esa", "Caused by:", "java."),
        ):
            break
        if line.strip():
            collected.append(line)

    detail = "\n".join(collected).strip()
    if not detail:
        # No "Error:" marker — fall back to the last few non-empty output lines.
        nonempty = [line for line in lines if line.strip()]
        detail = "\n".join(nonempty[-10:]).strip()
    return detail[:max_chars]


# A process the kernel's OOM killer ends exits with -SIGKILL.
_SIGKILL = 9

# Lines of GPT output a SnapProcessingError keeps.
_OUTPUT_TAIL_LINES = 20


def _describe_exit(returncode: int) -> str:
    """Describe a GPT exit code, naming the signal of a negative one."""
    if returncode >= 0:
        return f"exit {returncode}"
    try:
        name = signal.Signals(-returncode).name
    except ValueError:
        return f"exit {returncode}"
    hint = (
        "; usually out of memory: lower SNAPConfig.max_workers or max_ram_gb"
        if returncode == -_SIGKILL
        else ""
    )
    return f"exit {returncode} ({name}{hint})"


# What GPT's ReadOp says, in SNAP 13 and 14, of a source file no reader accepts.
# A SAFE zip cut short has lost its central directory, so a truncated SLC fails
# here, before any zip error reaches the Error: line. The path runs to the
# closing "'.", so a path holding an apostrophe is read whole.
_NO_READER = re.compile(r"No product reader found for file '(.+)'\.$", re.MULTILINE)


def _unreadable_input_hint(
    output: str,
    source_files: Mapping[str, Path],
    source_locations: Mapping[str, Path | UPath],
) -> str:
    """Return a hint naming where each input SLC that GPT found no reader for is kept."""
    unreadable = {Path(path).name for path in _NO_READER.findall(output)}
    return "".join(
        f"\nThe input SLC '{source_locations.get(name, path)}' may be truncated or corrupt: "
        "delete it and download it again."
        for name, path in source_files.items()
        if path.name in unreadable
    )


def _decode(output: str | bytes | None) -> str | None:
    """Decode captured output, which ``TimeoutExpired`` holds as bytes even in text mode."""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return output


def _output_tail(stdout: str | None, stderr: str | None) -> str | None:
    """Return the last non-empty lines GPT printed, stdout before stderr."""
    lines = [
        line for part in (stdout, stderr) if part for line in part.splitlines() if line.strip()
    ]
    return "\n".join(lines[-_OUTPUT_TAIL_LINES:]) or None


# Failures of one item: its GPT run, or staging its input or writing its product.
_ITEM_ERRORS = (BurstError, OSError, RasterioError, RasterWriteError)


def _item_failure(source: object, error: Exception) -> ErrorResult:
    """Record ``error`` as a failure of ``source`` alone, unless every item would fail alike."""
    if _is_environment_wide(error):
        raise error
    if not isinstance(error, BurstError):
        wrapped = SnapProcessingError(f"{type(error).__name__}: {error}")
        wrapped.__cause__ = error
        error = wrapped
    # Once, without a traceback: the summary carries the error itself.
    logger.error("Failed to process %s: %s", source, error)
    return ErrorResult(source=source, error=error)


def _normalize_nodata(dataset: rasterio.io.DatasetWriter, nodata: float) -> int:
    """Replace SNAP's finite sentinel and all-band zero fill with ``nodata``.

    Terrain-Correction can leave zeros in edge tiles over DEM voids. Requiring
    zero in every band preserves valid zero crossings in complex products. This
    normalization is only safe after the graph's no-data band maths has run.
    """
    replaced = 0
    for _, window in dataset.block_windows(1):
        data = dataset.read(window=window)
        marked = data == NODATA_SENTINEL
        marked[:, (data == 0).all(axis=0)] = True
        if not marked.any():
            continue
        data[marked] = nodata
        dataset.write(data, window=window)
        replaced += int(marked.any(axis=0).sum())
    if replaced:
        logger.debug("Replaced SNAP's no-data markers in %d pixels", replaced)
    return replaced


@dataclass
class SNAPConfig:
    """Runtime configuration for SNAP GPT processing.

    Parameters
    ----------
    bin
        A SNAP installation's ``bin`` directory, used to locate the ``gpt``
        executable. Defaults to ``SNAP_BIN``, then ``gpt`` discovered on ``PATH``.
    rasterio_profile
        Overrides for the rasterio profile used when rewriting GPT output
        GeoTIFFs. Merged over ``{"dtype": "float32", "nodata": "nan"}``.
    max_workers
        Number of parallel GPT subprocesses; at least 1. Defaults to
        ``SNAP_MAX_WORKERS`` (or 1).
    max_ram_gb
        Total RAM budget across all GPT processes in GB; must be positive and
        large enough to give every worker a 512 MB heap on top of 3 GB of JVM
        and OS overhead. Defaults to ``SNAP_MAX_RAM_GB`` (or 8).
    threads_per_job
        Threads per GPT process (``-q``); at least 1. Defaults to
        ``SNAP_THREADS_PER_JOB`` or ``cpu_count // max_workers``.
    cache_ratio
        Fraction of the per-process Java heap used for the GPT tile cache
        (``-c``); greater than 0 and at most 1. Defaults to
        ``SNAP_CACHE_RATIO`` (or 0.7).
    auxdata_path
        Shared SNAP auxdata cache (DEM tiles, orbit files). Defaults to
        ``SNAP_AUXDATA_PATH`` (or ``~/.snap/auxdata``).
    timeout
        Seconds one GPT run may take before it is stopped and recorded as
        failed; must be positive and finite, or ``None`` for no limit. Defaults to
        ``SNAP_TIMEOUT`` (or 3600), where ``none`` means no limit.

    Attributes
    ----------
    resolved_gpt_path
        The ``gpt`` executable this configuration runs, from ``bin``,
        ``SNAP_BIN`` or ``PATH``.
    resolved_snap_version
        Version of the SNAP installation that ``gpt`` belongs to, such as
        ``(14, 0, 0)``, or ``None`` if it cannot be determined.

    Raises
    ------
    ValueError
        If a resource argument or ``timeout`` is out of range, or if the RAM
        budget cannot cover the workers and ``max_ram_gb`` or ``max_workers``
        was passed.
    ConfigurationError
        If a ``SNAP_*`` environment variable is invalid, if the RAM budget
        cannot cover the workers and both values come from the environment,
        or if GPT or the auxdata path cannot be resolved.
    """

    bin: Path | None = None
    rasterio_profile: dict[str, Any] | None = None
    max_workers: int | None = None
    max_ram_gb: float | None = None
    threads_per_job: int | None = None
    cache_ratio: float | None = None
    auxdata_path: Path | None = None
    # None means no timeout, so the setting is read when the field is not given.
    timeout: float | None = field(default_factory=lambda: get_settings().snap.timeout)

    # Resolved resource values — set by __post_init__, not by users directly
    resolved_max_workers: int = field(init=False, repr=False, compare=False)
    resolved_xmx_mb: int = field(init=False, repr=False, compare=False)
    resolved_threads_per_job: int = field(init=False, repr=False, compare=False)
    resolved_cache_mb: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate configuration and resolve resource budgets."""
        # Update profile with default values if not specified by user
        base_profile = {"dtype": "float32", "nodata": "nan"}
        self.rasterio_profile = build_rasterio_profile(base_profile, self.rasterio_profile)
        # Each GPT run stages an uncompressed GeoTIFF through tempfile.
        warn_if_tmpdir_unusable()
        # Arguments first, so an invalid one is reported as such on a host without SNAP.
        self._validate_resources()
        snap_settings = get_settings().snap
        self._resolve_paths(snap_settings)
        self._resolve_resources(snap_settings)

    def _resolve_paths(self, snap_settings: SNAPSettings) -> None:
        """Resolve GPT and auxdata paths from explicit values and settings."""
        self.resolved_gpt_path = _resolve_gpt_path(self.bin, snap_settings)
        logger.debug("Resolved SNAP GPT executable: %s", self.resolved_gpt_path)

        # Graph band names depend on it; see `stack_band_suffixes`.
        self.resolved_snap_version = snap_version(self.resolved_gpt_path)
        if self.resolved_snap_version is None:
            logger.warning(
                "Could not determine the SNAP version of '%s'; assuming the band naming "
                "of SNAP 13 and earlier. Interferogram processing will fail if this "
                "installation is SNAP 14 or newer.",
                self.resolved_gpt_path,
            )
        else:
            logger.debug(
                "Resolved SNAP version: %s",
                ".".join(str(part) for part in self.resolved_snap_version),
            )
            if self.resolved_snap_version[:2] not in SUPPORTED_SNAP_VERSIONS:
                logger.warning(
                    "SNAP %s has not been tested with this version of s1bursts, which is "
                    "tested against %s. Processing may still work; graph band names are "
                    "chosen for the closest known naming.",
                    ".".join(str(part) for part in self.resolved_snap_version),
                    ", ".join(f"{major}.{minor}" for major, minor in SUPPORTED_SNAP_VERSIONS),
                )

        auxdata_path = self.auxdata_path or snap_settings.auxdata_path
        if auxdata_path is None:
            msg = (
                "SNAP auxdata path is None. "
                "Specify `auxdata_path` or set `SNAP_AUXDATA_PATH` in environment."
            )
            raise ConfigurationError(msg)

        self.resolved_auxdata_path = auxdata_path

    def _validate_resources(self) -> None:
        """Reject resource values that cannot describe a runnable job.

        These are caught here rather than left to GPT, which reports an
        out-of-range heap only after the JVM has already failed to start.
        """
        for name, value, minimum in (
            ("max_workers", self.max_workers, 1),
            ("threads_per_job", self.threads_per_job, 1),
        ):
            if value is not None and value < minimum:
                msg = f"`{name}` must be at least {minimum}, received {value}."
                raise ValueError(msg)

        if self.max_ram_gb is not None and self.max_ram_gb <= 0:
            msg = f"`max_ram_gb` must be positive, received {self.max_ram_gb}."
            raise ValueError(msg)

        # Written to reject NaN as well.
        if self.timeout is not None and not (math.isfinite(self.timeout) and self.timeout > 0):
            msg = (
                f"`timeout` must be a positive, finite number of seconds, received "
                f"{self.timeout}; pass None for no limit."
            )
            raise ValueError(msg)

        if self.cache_ratio is not None and not 0 < self.cache_ratio <= 1:
            msg = (
                f"`cache_ratio` must be greater than 0 and at most 1, received {self.cache_ratio}."
            )
            raise ValueError(msg)

    def _resolve_resources(self, snap_settings: SNAPSettings) -> None:
        """Resolve GPT resource parameters from explicit values, settings, and defaults."""
        n_cpus = os.cpu_count() or 4

        # max_workers: explicit > settings > 1
        self.resolved_max_workers = self.max_workers or snap_settings.max_workers

        # total RAM budget for Xmx calculation
        ram_gb = self.max_ram_gb or snap_settings.max_ram_gb
        ram_mb = int(ram_gb * 1024)

        # Reserve headroom for the JVM's own non-heap memory and the OS, then
        # split what is left between workers.
        self.resolved_xmx_mb = (ram_mb - _JVM_OVERHEAD_MB) // self.resolved_max_workers

        # A heap below this cannot load SNAP's operator graph at all, so an
        # over-subscribed budget is reported rather than quietly rounded up past
        # the RAM the caller said was available.
        if self.resolved_xmx_mb < _MIN_HEAP_MB:
            required_mb = _JVM_OVERHEAD_MB + _MIN_HEAP_MB * self.resolved_max_workers
            # Round the suggestion up, so following it actually clears the check.
            required_gb = math.ceil(required_mb / 1024 * 100) / 100
            msg = (
                f"A RAM budget of {ram_gb:g} GB across {self.resolved_max_workers} worker(s) "
                f"leaves {self.resolved_xmx_mb} MB of heap per worker, below the "
                f"{_MIN_HEAP_MB} MB SNAP needs (after {_JVM_OVERHEAD_MB} MB of JVM and OS "
                f"overhead). "
            )
            # Blame the call if it supplied either value, the environment otherwise.
            if self.max_ram_gb is not None or self.max_workers is not None:
                msg += f"Raise `max_ram_gb` to at least {required_gb:g} or lower `max_workers`."
                raise ValueError(msg)
            msg += (
                f"Raise `SNAP_MAX_RAM_GB` to at least {required_gb:g} or lower "
                "`SNAP_MAX_WORKERS`, or pass `max_ram_gb` or `max_workers`."
            )
            raise ConfigurationError(msg)

        # threads per job (-q)
        self.resolved_threads_per_job = (
            self.threads_per_job
            or snap_settings.threads_per_job
            or max(1, n_cpus // self.resolved_max_workers)
        )

        # cache per job (-c): ratio of the per-job Java heap (default: 70%).
        # Must be computed from the heap, not the raw RAM share — otherwise
        # small budgets produce a tile cache larger than -Xmx.
        cache_ratio = self.cache_ratio or snap_settings.cache_ratio
        self.resolved_cache_mb = int(self.resolved_xmx_mb * cache_ratio)


class SNAPClient:
    """Low-level client for SNAP GPT processing.

    Provides direct interface to SNAP GPT for processing Sentinel-1 bursts.
    For batch processing with progress tracking, use BurstProcessor instead.

    Parameters
    ----------
    file_manager
        File manager resolving input SLC and output product paths.
    config
        Runtime configuration for SNAP GPT; defaults to ``SNAPConfig()``.
    """

    def __init__(self, file_manager: FileManager, config: SNAPConfig | None = None) -> None:
        self.config = config or SNAPConfig()
        self.manager = file_manager

    def _run_gpt(
        self,
        graph: Path,
        source_files: dict[str, Path],
        target_file: Path,
        snap_userdir: Path,
        process_params: dict[str, str] | None = None,
        source_locations: Mapping[str, Path | UPath] | None = None,
    ) -> Path:
        """Run SNAP GPT with given parameters.

        ``source_locations`` maps a source name to where its SLC is kept, which
        a read-failure hint names instead of the staged copy GPT read.
        """
        commands = [str(self.config.resolved_gpt_path), str(graph)]

        for source_name, source_file in source_files.items():
            commands.append(f"-S{source_name}={source_file}")
        if process_params:
            for k, v in process_params.items():
                commands.append(f"-P{k}={v}")

        # Keep jblas files under the worker directory so SIGKILL cannot orphan them.
        java_tmpdir = snap_userdir / "tmp"
        java_tmpdir.mkdir(parents=True, exist_ok=True)

        commands.extend(
            [
                "-t",
                str(target_file),
                "-q",
                str(self.config.resolved_threads_per_job),
                # Clears the internal tile cache after writing a complete row
                "-x",
                "-c",
                f"{self.config.resolved_cache_mb}M",
                f"-J-Xmx{self.config.resolved_xmx_mb}M",
                f"-J-Dsnap.userdir={snap_userdir}",
                f"-J-Djava.io.tmpdir={java_tmpdir}",
            ],
        )
        if target_file.suffix == ".tif":
            # Suppresses creation of XML sidecar file
            commands.extend(["-f", "GeoTIFF"])

        logger.debug("Running command %s", " ".join(commands))

        timeout = self.config.timeout
        try:
            result = subprocess.run(  # noqa: S603
                commands,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            logger.debug("SNAP output: %s", result.stdout)
        except subprocess.CalledProcessError as e:
            target_file.unlink(missing_ok=True)
            # At debug level only: the failure is logged once, by the caller.
            logger.debug("GPT output:\nstdout: %s\nstderr: %s", e.stdout, e.stderr)
            # Preserve SNAP's cause; CalledProcessError reports only the exit code.
            detail = _extract_gpt_error(e.stdout, e.stderr) or "<no output captured>"
            msg = f"GPT processing failed with {_describe_exit(e.returncode)}: {detail}"
            output = "\n".join(part for part in (e.stdout, e.stderr) if part)
            msg += _unreadable_input_hint(output, source_files, source_locations or {})
            raise SnapProcessingError(
                msg,
                returncode=e.returncode,
                output_tail=_output_tail(e.stdout, e.stderr),
            ) from e
        except subprocess.TimeoutExpired as e:
            target_file.unlink(missing_ok=True)
            tail = _output_tail(_decode(e.stdout), _decode(e.stderr))
            msg = (
                f"GPT processing failed: timed out after {timeout:g} s (SNAPConfig.timeout). "
                f"Last output:\n{tail or '<no output captured>'}"
            )
            raise SnapProcessingError(msg, output_tail=tail) from e
        except Exception as e:
            target_file.unlink(missing_ok=True)
            msg = f"GPT processing failed: {e}"
            raise SnapProcessingError(msg) from e

        if not target_file.exists():
            msg = f"Processing completed but output not found: {target_file}"
            raise SnapProcessingError(
                msg,
                returncode=result.returncode,
                output_tail=_output_tail(result.stdout, result.stderr),
            )

        return target_file

    def _stage_slc(self, burst: Burst, tmp_dir: Path) -> Path:
        """Return a local path to ``burst``'s SLC, downloading if remote.

        ``UPath("/local/path")`` is a ``Path`` subclass, so the ``isinstance``
        check cleanly separates already-local sources from remote ones.
        """
        remote = self.manager.get_slc_path(burst)
        if isinstance(remote, Path):
            return remote
        local = tmp_dir / remote.name
        with remote.open("rb") as src, local.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return local

    def _finalize_product(
        self,
        staging_file: Path,
        product_path: Path | UPath,
        *,
        metadata: RasterMetadata,
        normalize_nodata: bool = False,
    ) -> None:
        """Rewrite the staging GeoTIFF compressed to its final location.

        SNAP's GeoTIFF writer does not embed a NODATA tag, so the profile's
        ``nodata`` is stamped onto the local staging file first;
        ``rewrite_tiff`` then carries it into the product.
        """
        profile = self.config.rasterio_profile or {}
        nodata = profile.get("nodata")
        # SNAP's GeoTIFF writer leaves a Photometric/ExtraSamples tag mismatch on
        # multi-band products. GDAL recovers on read but logs a CPLE_AppDefined warning.
        # Suppress that warning. The tags are fixed after `rewrite_tiff`.
        _rasterio_logger = logging.getLogger("rasterio")
        _prev_level = _rasterio_logger.level
        _rasterio_logger.setLevel(logging.ERROR)
        try:
            if nodata is not None:
                with rasterio.open(staging_file, "r+") as ds:
                    ds.nodata = float(nodata)
                    if normalize_nodata:
                        _normalize_nodata(ds, float(nodata))
            # The staging file is a disposable copy in a TemporaryDirectory, so move
            # it into place rather than leaving a redundant copy behind.
            rewrite_tiff(
                src_file=staging_file,
                dst_file=product_path,
                profile=profile,
                band_names=metadata.band_names,
                tags=metadata.tags,
                band_tags=metadata.band_tags or None,
                units=metadata.units,
                move=True,
            )
        finally:
            _rasterio_logger.setLevel(_prev_level)

    @overload
    def create_backscatter(
        self,
        burst: Burst,
        crs: CRSInput | Literal["auto"],
        *,
        config: RTCConfig,
        overwrite: bool = False,
    ) -> OperationResult[RTCProduct]: ...

    @overload
    def create_backscatter(
        self,
        burst: Burst,
        crs: CRSInput | Literal["auto"],
        *,
        config: GTCConfig,
        overwrite: bool = False,
    ) -> OperationResult[GTCProduct]: ...

    def create_backscatter(
        self,
        burst: Burst,
        crs: CRSInput | Literal["auto"],
        *,
        config: RTCConfig | GTCConfig,
        overwrite: bool = False,
    ) -> OperationResult[RTCProduct] | OperationResult[GTCProduct]:
        """Compute backscatter from a burst, terrain-flattened (RTC) or geocoded (GTC).

        The type of ``config`` selects the product: an ``RTCConfig`` gives an
        ``RTCProduct``, terrain-flattened gamma-0 backscatter; a ``GTCConfig``
        a ``GTCProduct``, beta-0 backscatter that is geocoded only.

        Parameters
        ----------
        burst
            Burst to process.
        crs
            The coordinate reference system of the geocoded output, a WGS 84 /
            UTM zone, north or south, which product file names record. Given as
            a CRS object, an authority string such as ``"EPSG:32632"`` or an
            EPSG code, or ``"auto"`` to process each burst in the UTM zone
            that holds its footprint (``utm_zone``). Use ``"auto"`` for
            bursts processed on their own, such as a time series, and the
            target cell's ``crs`` for products meant for a composite: an input
            in another zone than the cell raises ``InconsistentInputsError``
            when composited.
        config
            Graph-shape configuration, ``RTCConfig`` or ``GTCConfig``.
        overwrite
            Whether to overwrite an existing product. Without it, an existing
            product is returned as skipped, after checking that the settings
            it records match ``config``.

        Returns
        -------
        OperationResult
            Result of the backscatter processing. A skipped product's result
            records in ``unchecked`` and ``other_snap_version`` what its check
            could not confirm; such a skip is also logged.

        Raises
        ------
        TypeError
            If ``config`` is neither an ``RTCConfig`` nor a ``GTCConfig``.
        ValueError
            If ``crs`` is not a WGS 84 / UTM zone. Raised, not recorded,
            because a wrong ``crs`` would fail every burst. Under ``"auto"``,
            a footprint outside the UTM zones fails this item alone and is
            recorded as a ``NoUTMZoneError``.
        OSError, SnapProcessingError or RasterWriteError
            If the disk is full, over quota or read-only, which would fail
            every remaining item alike: an ``OSError`` saying so, or an error
            caused by one or whose message or GPT output says so.
        ProductSettingsMismatchError
            If the product exists, ``overwrite`` is ``False``, and the product
            records other settings than ``config``: grid spacing, DEM,
            multilook, coherence window or the interferogram's phase removal.
        OSError or RasterioError
            If the product exists, ``overwrite`` is ``False``, and it cannot
            be read to check its settings.
        """
        # Under "auto", a burst without a UTM zone fails on its own.
        if isinstance(config, RTCConfig):
            try:
                rtc = RTCProduct.from_burst(burst, crs=crs)
            except NoUTMZoneError as err:
                return _item_failure(burst, err)
            return self._create_backscatter(burst, rtc, config, overwrite=overwrite)
        if isinstance(config, GTCConfig):
            try:
                gtc = GTCProduct.from_burst(burst, crs=crs)
            except NoUTMZoneError as err:
                return _item_failure(burst, err)
            return self._create_backscatter(burst, gtc, config, overwrite=overwrite)
        msg = f"`config` must be an RTCConfig or a GTCConfig, received '{type(config).__name__}'."
        raise TypeError(msg)

    def _existing[P: RTCProduct | GTCProduct | CoherenceProduct | InterferogramProduct](
        self,
        source: Burst | BurstPair,
        product: P,
        config: RTCConfig | GTCConfig | CoherenceConfig | InterferogramConfig,
        *,
        overwrite: bool,
    ) -> OkResult[P] | None:
        """Return the skipped result of an existing product whose settings match, else None."""
        if overwrite:
            return None
        return _skip_existing(
            self.manager,
            source,
            product,
            config,
            snap_version=self.config.resolved_snap_version,
        )

    def _create_backscatter[P: RTCProduct | GTCProduct](
        self,
        burst: Burst,
        product: P,
        config: RTCConfig | GTCConfig,
        *,
        overwrite: bool,
    ) -> OperationResult[P]:
        """Run the backscatter graph of ``config`` for ``product``, unless it exists."""
        label = product.product_type
        product_path = self.manager.get_product_path(product)

        skipped = self._existing(burst, product, config, overwrite=overwrite)
        if skipped is not None:
            return skipped

        logger.info("Creating %s product: '%s'", label, product_path.name)

        with tempfile.TemporaryDirectory(prefix="s1bursts-") as tmp:
            tmp_dir = Path(tmp)
            staging_file = tmp_dir / product_path.name
            graph_file = _write_graph(
                tmp_dir,
                "backscatter",
                config.to_xml(snap_version=self.config.resolved_snap_version),
            )
            with snap_worker_userdir(self.config.resolved_auxdata_path, tmp_dir) as snap_userdir:
                try:
                    source_file = self._stage_slc(burst, tmp_dir)
                    self._run_gpt(
                        graph=graph_file,
                        source_files={"source": source_file},
                        source_locations={"source": self.manager.get_slc_path(burst)},
                        snap_userdir=snap_userdir,
                        process_params={
                            "swath": product.sub_swath,
                            "polarization": product.polarization,
                            "crs": get_epsg_string(product.crs),
                        },
                        target_file=staging_file,
                    )
                    self._finalize_product(
                        staging_file,
                        product_path,
                        metadata=product_tags.burst_metadata(
                            product,
                            config,
                            snap_version=self.config.resolved_snap_version,
                        ),
                        normalize_nodata=config.replace_zeros_with_nan,
                    )
                except _ITEM_ERRORS as e:
                    return _item_failure(burst, e)

        logger.info("Successfully created '%s'", product_path.name)
        return OkResult(source=burst, result=product)

    def create_coherence(
        self,
        pair: BurstPair,
        crs: CRSInput | Literal["auto"],
        *,
        config: CoherenceConfig,
        overwrite: bool = False,
    ) -> OperationResult[CoherenceProduct]:
        """Compute the coherence from a burst pair.

        Runs the single-shot SLC→Coherence→Terrain-Correction graph.

        Parameters
        ----------
        pair
            Burst pair to process.
        crs
            The coordinate reference system of the geocoded output, a WGS 84 /
            UTM zone, north or south, which product file names record. Given as
            a CRS object, an authority string such as ``"EPSG:32632"`` or an
            EPSG code, or ``"auto"`` to process each pair in the UTM zone
            that holds its footprint (``utm_zone``). Use ``"auto"`` for
            pairs processed on their own, such as a time series, and the
            target cell's ``crs`` for products meant for a composite: an input
            in another zone than the cell raises ``InconsistentInputsError``
            when composited.
        config
            Graph-shape configuration (window size, DEM, …).
        overwrite
            Whether to overwrite an existing product. Without it, an existing
            product is returned as skipped, after checking that the settings
            it records match ``config``.

        Returns
        -------
        OperationResult
            Result of the coherence processing. A skipped product's result
            records in ``unchecked`` and ``other_snap_version`` what its check
            could not confirm; such a skip is also logged.

        Raises
        ------
        ValueError
            If ``crs`` is not a WGS 84 / UTM zone. Raised, not recorded,
            because a wrong ``crs`` would fail every burst. Under ``"auto"``,
            a footprint outside the UTM zones fails this item alone and is
            recorded as a ``NoUTMZoneError``.
        OSError, SnapProcessingError or RasterWriteError
            If the disk is full, over quota or read-only, which would fail
            every remaining item alike: an ``OSError`` saying so, or an error
            caused by one or whose message or GPT output says so.
        ProductSettingsMismatchError
            If the product exists, ``overwrite`` is ``False``, and the product
            records other settings than ``config``: grid spacing, DEM,
            multilook, coherence window or the interferogram's phase removal.
        OSError or RasterioError
            If the product exists, ``overwrite`` is ``False``, and it cannot
            be read to check its settings.
        """
        try:
            product = CoherenceProduct.from_burst_pair(pair, crs=crs)
        except NoUTMZoneError as err:
            return _item_failure(pair, err)
        product_path = self.manager.get_product_path(product)

        skipped = self._existing(pair, product, config, overwrite=overwrite)
        if skipped is not None:
            return skipped

        logger.info("Creating coherence product: '%s'", product_path.name)

        with tempfile.TemporaryDirectory(prefix="s1bursts-") as tmp:
            tmp_dir = Path(tmp)
            staging_file = tmp_dir / product_path.name
            graph_file = _write_graph(
                tmp_dir,
                "coherence",
                config.to_xml(snap_version=self.config.resolved_snap_version),
            )
            with snap_worker_userdir(self.config.resolved_auxdata_path, tmp_dir) as snap_userdir:
                try:
                    primary_file = self._stage_slc(pair.primary, tmp_dir)
                    secondary_file = self._stage_slc(pair.secondary, tmp_dir)
                    self._run_gpt(
                        graph=graph_file,
                        source_files={
                            "primary": primary_file,
                            "secondary": secondary_file,
                        },
                        source_locations={
                            "primary": self.manager.get_slc_path(pair.primary),
                            "secondary": self.manager.get_slc_path(pair.secondary),
                        },
                        snap_userdir=snap_userdir,
                        process_params={
                            "swath": product.sub_swath,
                            "polarization": product.polarization,
                            "date1": product.date_primary.strftime("%d%b%Y"),
                            "date2": product.date_secondary.strftime("%d%b%Y"),
                            "crs": get_epsg_string(product.crs),
                        },
                        target_file=staging_file,
                    )
                    self._finalize_product(
                        staging_file,
                        product_path,
                        metadata=product_tags.burst_metadata(
                            product,
                            config,
                            snap_version=self.config.resolved_snap_version,
                        ),
                        normalize_nodata=config.replace_zeros_with_nan,
                    )
                except _ITEM_ERRORS as e:
                    return _item_failure(pair, e)

        logger.info("Successfully created '%s'", product_path.name)
        return OkResult(source=pair, result=product)

    def create_interferogram(
        self,
        pair: BurstPair,
        crs: CRSInput | Literal["auto"],
        *,
        config: InterferogramConfig,
        overwrite: bool = False,
    ) -> OperationResult[InterferogramProduct]:
        """Compute the interferogram from a burst pair.

        Parameters
        ----------
        pair
            Burst pair to process.
        crs
            The coordinate reference system of the geocoded output, a WGS 84 /
            UTM zone, north or south, which product file names record. Given as
            a CRS object, an authority string such as ``"EPSG:32632"`` or an
            EPSG code, or ``"auto"`` to process each pair in the UTM zone
            that holds its footprint (``utm_zone``). Use ``"auto"`` for
            pairs processed on their own, such as a time series, and the
            target cell's ``crs`` for products meant for a composite: an input
            in another zone than the cell raises ``InconsistentInputsError``
            when composited.
        config
            Graph-shape configuration.
        overwrite
            Whether to overwrite an existing product. Without it, an existing
            product is returned as skipped, after checking that the settings
            it records match ``config``.

        Returns
        -------
        OperationResult
            Result of the interferogram processing. A skipped product's result
            records in ``unchecked`` and ``other_snap_version`` what its check
            could not confirm; such a skip is also logged.

        Raises
        ------
        ValueError
            If ``crs`` is not a WGS 84 / UTM zone. Raised, not recorded,
            because a wrong ``crs`` would fail every burst. Under ``"auto"``,
            a footprint outside the UTM zones fails this item alone and is
            recorded as a ``NoUTMZoneError``.
        OSError, SnapProcessingError or RasterWriteError
            If the disk is full, over quota or read-only, which would fail
            every remaining item alike: an ``OSError`` saying so, or an error
            caused by one or whose message or GPT output says so.
        ProductSettingsMismatchError
            If the product exists, ``overwrite`` is ``False``, and the product
            records other settings than ``config``: grid spacing, DEM,
            multilook, coherence window or the interferogram's phase removal.
        OSError or RasterioError
            If the product exists, ``overwrite`` is ``False``, and it cannot
            be read to check its settings.
        """
        try:
            product = InterferogramProduct.from_burst_pair(pair, crs=crs)
        except NoUTMZoneError as err:
            return _item_failure(pair, err)
        product_path = self.manager.get_product_path(product)

        skipped = self._existing(pair, product, config, overwrite=overwrite)
        if skipped is not None:
            return skipped

        logger.info("Creating interferogram product: '%s'", product_path.name)

        with tempfile.TemporaryDirectory(prefix="s1bursts-") as tmp:
            tmp_dir = Path(tmp)
            staging_file = tmp_dir / product_path.name
            graph_file = _write_graph(
                tmp_dir,
                "interferogram",
                config.to_xml(snap_version=self.config.resolved_snap_version),
            )
            with snap_worker_userdir(self.config.resolved_auxdata_path, tmp_dir) as snap_userdir:
                try:
                    primary_file = self._stage_slc(pair.primary, tmp_dir)
                    secondary_file = self._stage_slc(pair.secondary, tmp_dir)
                    self._run_gpt(
                        graph=graph_file,
                        source_files={
                            "primary": primary_file,
                            "secondary": secondary_file,
                        },
                        source_locations={
                            "primary": self.manager.get_slc_path(pair.primary),
                            "secondary": self.manager.get_slc_path(pair.secondary),
                        },
                        snap_userdir=snap_userdir,
                        process_params={
                            "swath": product.sub_swath,
                            "polarization": product.polarization,
                            "date1": product.date_primary.strftime("%d%b%Y"),
                            "date2": product.date_secondary.strftime("%d%b%Y"),
                            "crs": get_epsg_string(product.crs),
                        },
                        target_file=staging_file,
                    )
                    self._finalize_product(
                        staging_file,
                        product_path,
                        metadata=product_tags.burst_metadata(
                            product,
                            config,
                            snap_version=self.config.resolved_snap_version,
                        ),
                        normalize_nodata=config.replace_zeros_with_nan,
                    )
                except _ITEM_ERRORS as e:
                    return _item_failure(pair, e)

        logger.info("Successfully created '%s'", product_path.name)
        return OkResult(source=pair, result=product)


def _populate_worker_userdir(worker_userdir: Path, shared_cache: Path) -> None:
    """Symlink tiles from the shared auxdata cache into a per-job SNAP userdir."""
    for sub in _AUXDATA_SUBDIRS:
        src_root = shared_cache / sub
        if not src_root.exists():
            continue
        dst_root = worker_userdir / "auxdata" / sub
        for src in src_root.rglob("*"):
            if not src.is_file():
                continue
            dst = dst_root / src.relative_to(src_root)
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src.resolve())


def _publish_new_files(worker_userdir: Path, shared_cache: Path) -> None:
    """Copy tiles downloaded by a job back to the shared cache.

    Symlinked files (pre-existing tiles) are skipped.  New files are written
    via an atomic replace so concurrent publishes of the same tile are safe:
    last writer wins, but all writes produce a valid file.
    """
    for sub in _AUXDATA_SUBDIRS:
        wk_root = worker_userdir / "auxdata" / sub
        if not wk_root.exists():
            continue
        sh_root = shared_cache / sub
        for path in wk_root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            dst = sh_root / path.relative_to(wk_root)
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=dst.name + ".", suffix=".tmp", dir=dst.parent)
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                shutil.copy2(path, tmp)
                tmp.replace(dst)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise


@contextmanager
def snap_worker_userdir(shared_cache: Path, worker_root: Path) -> Generator[Path]:
    """Context manager that provides an isolated SNAP userdir for one job.

    Symlinks known auxdata subdirs from *shared_cache* into a fresh per-PID
    directory under *worker_root*, yields the path to pass to
    ``-J-Dsnap.userdir``, then publishes any newly downloaded tiles back to
    *shared_cache* and removes the directory.
    """
    worker_userdir = worker_root / f"snap-{os.getpid()}"
    worker_userdir.mkdir(parents=True, exist_ok=True)
    try:
        _populate_worker_userdir(worker_userdir, shared_cache)
        yield worker_userdir
    finally:
        try:
            _publish_new_files(worker_userdir, shared_cache)
        finally:
            shutil.rmtree(worker_userdir, ignore_errors=True)
