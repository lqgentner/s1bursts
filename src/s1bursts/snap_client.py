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
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any, Final

from freezebase.raster import (
    build_rasterio_profile,
    get_epsg_string,
    rewrite_tiff,
)
import rasterio

from s1bursts import product_tags
from s1bursts.config import get_settings, warn_if_tmpdir_unusable
from s1bursts.errors import (
    BurstError,
    ConfigurationError,
    SnapProcessingError,
)
from s1bursts.product import (
    CoherenceProduct,
    CRSLike,
    GTCProduct,
    InterferogramProduct,
    RTCProduct,
)
from s1bursts.snap_graph import NODATA_SENTINEL
from s1bursts.summary import ErrorResult, OkResult, OperationResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    from upath import UPath

    from s1bursts.burst import Burst, BurstPair
    from s1bursts.config import SNAPSettings
    from s1bursts.filemanager import FileManager
    from s1bursts.product_tags import RasterMetadata
    from s1bursts.snap_graph import (
        BackscatterConfig,
        CoherenceConfig,
        InterferogramConfig,
    )

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

    Raises
    ------
    ConfigurationError
        If none of the above resolve to a candidate path, or the resolved path is
        not an executable file.
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
    gpt_path : Path
        The ``gpt`` executable, as resolved by :func:`_resolve_gpt_path`.

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


def _normalize_nodata(dataset: rasterio.io.DatasetWriter, nodata: float) -> int:
    """Replace SNAP's finite sentinel and all-band zero fill with ``nodata``.

    Terrain-Correction can leave zeros in edge tiles over DEM voids. Requiring
    zero in every band preserves valid zero crossings in complex products. This
    normalization is only safe after the graph's no-data band maths has run.

    Returns
    -------
    int
        Number of pixels replaced.
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
    bin : Path, optional
        A SNAP installation's ``bin`` directory, used to locate the ``gpt``
        executable. Defaults to ``SNAP_BIN``, then ``gpt`` discovered on ``PATH``.
    rasterio_profile : dict, optional
        Overrides for the rasterio profile used when rewriting GPT output
        GeoTIFFs. Merged over ``{"dtype": "float32", "nodata": "nan"}``.
    max_workers : int, optional
        Number of parallel GPT subprocesses; at least 1. Defaults to
        ``SNAP_MAX_WORKERS`` (or 1).
    max_ram_gb : float, optional
        Total RAM budget across all GPT processes in GB; must be positive and
        large enough to give every worker a 512 MB heap on top of 3 GB of JVM
        and OS overhead. Defaults to ``SNAP_MAX_RAM_GB`` (or 4).
    gpt_threads_per_job : int, optional
        Threads per GPT process (``-q``); at least 1. Defaults to
        ``SNAP_THREADS_PER_JOB`` or ``cpu_count // max_workers``.
    gpt_cache_ratio : float, optional
        Fraction of the per-process Java heap used for the GPT tile cache
        (``-c``); greater than 0 and at most 1. Defaults to
        ``SNAP_CACHE_RATIO`` (or 0.7).
    auxdata_path : Path, optional
        Shared SNAP auxdata cache (DEM tiles, orbit files). Defaults to
        ``SNAP_AUXDATA_PATH`` (or ``~/.snap/auxdata``).

    Raises
    ------
    ConfigurationError
        If a resource value is out of range, if the RAM budget cannot cover the
        requested workers, or if GPT or the auxdata path cannot be resolved.
    """

    bin: Path | None = None
    rasterio_profile: dict[str, Any] | None = None
    max_workers: int | None = None
    max_ram_gb: float | None = None
    gpt_threads_per_job: int | None = None
    gpt_cache_ratio: float | None = None
    auxdata_path: Path | None = None

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
            ("gpt_threads_per_job", self.gpt_threads_per_job, 1),
        ):
            if value is not None and value < minimum:
                msg = f"`{name}` must be at least {minimum}, received {value}."
                raise ConfigurationError(msg)

        if self.max_ram_gb is not None and self.max_ram_gb <= 0:
            msg = f"`max_ram_gb` must be positive, received {self.max_ram_gb}."
            raise ConfigurationError(msg)

        if self.gpt_cache_ratio is not None and not 0 < self.gpt_cache_ratio <= 1:
            msg = (
                "`gpt_cache_ratio` must be greater than 0 and at most 1, "
                f"received {self.gpt_cache_ratio}."
            )
            raise ConfigurationError(msg)

    def _resolve_resources(self, snap_settings: SNAPSettings) -> None:
        """Resolve GPT resource parameters from explicit values, settings, and defaults."""
        self._validate_resources()
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
                f"overhead). Raise `max_ram_gb` to at least {required_gb:g} "
                f"or lower `max_workers`."
            )
            raise ConfigurationError(msg)

        # threads per job (-q)
        self.resolved_threads_per_job = (
            self.gpt_threads_per_job
            or snap_settings.threads_per_job
            or max(1, n_cpus // self.resolved_max_workers)
        )

        # cache per job (-c): ratio of the per-job Java heap (default: 70%).
        # Must be computed from the heap, not the raw RAM share — otherwise
        # small budgets produce a tile cache larger than -Xmx.
        cache_ratio = self.gpt_cache_ratio or snap_settings.cache_ratio
        self.resolved_cache_mb = int(self.resolved_xmx_mb * cache_ratio)


class SNAPClient:
    """Low-level client for SNAP GPT processing.

    Provides direct interface to SNAP GPT for processing Sentinel-1 bursts.
    For batch processing with progress tracking, use BurstProcessor instead.

    Parameters
    ----------
    file_manager : FileManager
        File manager resolving input SLC and output product paths.
    config : SNAPConfig | None, default None
        Runtime configuration for SNAP GPT; defaults to ``SNAPConfig()``.

    """

    def __init__(self, file_manager: FileManager, config: SNAPConfig | None = None) -> None:
        """Initialize SNAP client.

        Parameters
        ----------
        file_manager : FileManager
            File manager resolving input SLC and output product paths.
        config : SNAPConfig | None, default None
            Runtime configuration for SNAP GPT; defaults to ``SNAPConfig()``.
        """
        self.config = config or SNAPConfig()
        self.manager = file_manager

    def _run_gpt(
        self,
        graph: Path,
        source_files: dict[str, Path],
        target_file: Path,
        snap_userdir: Path,
        process_params: dict[str, str] | None = None,
        timeout: int = 3600,
    ) -> Path:
        """Run SNAP GPT with given parameters.

        Parameters
        ----------
        graph : Path
            Path to the SNAP GPT processing graph XML file.
        source_files : dict[str, Path]
            Mapping of source parameter names to their file paths.
        target_file : Path
            Path where GPT should write the output file.
            The file type is determined based on the suffix.
        snap_userdir : Path
            Per-job SNAP user directory passed via ``-J-Dsnap.userdir``. Also
            hosts the JVM's ``java.io.tmpdir``, so that temporary files the
            process leaves behind are removed with the userdir.
        process_params : dict[str, str] | None, optional
            Additional processing parameters passed as ``-P`` flags.
        timeout : int, default 3600
            Maximum time in seconds to wait for GPT to finish.

        Returns
        -------
        Path
            Path to the output file written by GPT.

        Raises
        ------
        SnapProcessingError
            If GPT returns a non-zero exit code, times out, or the output
            file is missing.
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

        try:
            result = subprocess.run(  # noqa: S603
                commands,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            logger.debug("SNAP output: %s", result.stdout)
        except Exception as e:
            target_file.unlink(missing_ok=True)
            if (
                target_file.suffix == ".dim"
                and (data_dir := target_file.with_suffix(".data")).exists()
            ):
                shutil.rmtree(data_dir)
            if isinstance(e, subprocess.CalledProcessError):
                logger.exception(
                    "GPT failed (exit %d):\nstdout: %s\nstderr: %s",
                    e.returncode,
                    e.stdout,
                    e.stderr,
                )
                # Preserve SNAP's cause; CalledProcessError reports only the exit code.
                detail = _extract_gpt_error(e.stdout, e.stderr) or "<no output captured>"
                msg = f"GPT processing failed (exit {e.returncode}): {detail}"
            else:
                msg = f"GPT processing failed: {e}"
            raise SnapProcessingError(msg) from e

        if not target_file.exists():
            msg = f"Processing completed but output not found: {target_file}"
            raise SnapProcessingError(msg)

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

        Parameters
        ----------
        staging_file : Path
            Uncompressed GeoTIFF written by GPT.
        product_path : Path | UPath
            Final product location (local or S3).
        metadata : RasterMetadata
            Tags, band descriptions and units to write into the product.
        normalize_nodata : bool, default False
            Convert the no-data markers GPT leaves behind into the product's own
            no-data value; see :func:`_normalize_nodata`. Only pass ``True`` for
            a graph that ran the band maths, since that is what makes the markers
            unambiguous.
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

    def create_backscatter(
        self,
        burst: Burst,
        crs: CRSLike,
        *,
        config: BackscatterConfig,
        overwrite: bool = False,
    ) -> OperationResult[RTCProduct | GTCProduct]:
        """Compute the backscatter (RTC or GTC) product from a burst.

        ``config.flatten_terrain=True`` (default) produces an RTC product;
        ``False`` produces a GTC product.  All other graph-shape settings
        (DEM, multilook, resampling, …) come from ``config``.

        Parameters
        ----------
        burst : Burst
            Burst to process.
        crs : CRSLike
            The coordinate reference system of the geocoded output.
        config : BackscatterConfig
            Graph-shape configuration.
        overwrite : bool, default False
            Whether to overwrite existing product.

        Returns
        -------
        OperationResult
            Result of the backscatter processing.
        """
        product: RTCProduct | GTCProduct
        if config.flatten_terrain:
            product = RTCProduct.from_burst(burst, crs=crs)
            label = "RTC"
        else:
            product = GTCProduct.from_burst(burst, crs=crs)
            label = "GTC"
        product_path = self.manager.get_product_path(product)

        if not overwrite and self.manager.product_exists(product):
            logger.debug("%s product already exists: '%s'", label, product_path.name)
            return OkResult(source=burst, result=product, skipped=True)

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
                        snap_userdir=snap_userdir,
                        process_params={
                            "swath": product.sub_swath,
                            "polarization": product.polarization,
                            "crs": get_epsg_string(crs),
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
                except (BurstError, FileNotFoundError, RuntimeError) as e:
                    logger.exception("Failed to process burst '%s'", burst)
                    return ErrorResult(source=burst, error=e)

        logger.info("Successfully created '%s'", product_path.name)
        return OkResult(source=burst, result=product)

    def create_coherence(
        self,
        pair: BurstPair,
        crs: CRSLike,
        *,
        config: CoherenceConfig,
        overwrite: bool = False,
    ) -> OperationResult[CoherenceProduct]:
        """Compute the coherence from a burst pair.

        Runs the single-shot SLC→Coherence→Terrain-Correction graph.

        Parameters
        ----------
        pair : BurstPair
            Burst pair to process.
        crs : CRSLike
            The coordinate reference system of the geocoded output.
        config : CoherenceConfig
            Graph-shape configuration (window size, DEM, …).
        overwrite : bool, default False
            Whether to overwrite existing product.

        Returns
        -------
        OperationResult
            Result of the coherence processing.
        """
        product = CoherenceProduct.from_burst_pair(pair, crs=crs)
        product_path = self.manager.get_product_path(product)

        if not overwrite and self.manager.product_exists(product):
            logger.debug("Coherence product already exists: '%s'", product_path.name)
            return OkResult(source=pair, result=product, skipped=True)

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
                        snap_userdir=snap_userdir,
                        process_params={
                            "swath": product.sub_swath,
                            "polarization": product.polarization,
                            "date1": product.date_primary.strftime("%d%b%Y"),
                            "date2": product.date_secondary.strftime("%d%b%Y"),
                            "crs": get_epsg_string(crs),
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
                except (BurstError, FileNotFoundError, RuntimeError) as e:
                    logger.exception("Failed to process burst pair '%s'", pair)
                    return ErrorResult(source=pair, error=e)

        logger.info("Successfully created '%s'", product_path.name)
        return OkResult(source=pair, result=product)

    def create_interferogram(
        self,
        pair: BurstPair,
        crs: CRSLike,
        *,
        config: InterferogramConfig,
        overwrite: bool = False,
    ) -> OperationResult[InterferogramProduct]:
        """Compute the interferogram from a burst pair.

        Parameters
        ----------
        pair : BurstPair
            Burst pair to process.
        crs : CRSLike
            The coordinate reference system of the geocoded output.
        config : InterferogramConfig
            Graph-shape configuration.
        overwrite : bool, default False
            Whether to overwrite existing product.

        Returns
        -------
        OperationResult
            Result of the interferogram processing.
        """
        product = InterferogramProduct.from_burst_pair(pair, crs=crs)
        product_path = self.manager.get_product_path(product)

        if not overwrite and self.manager.product_exists(product):
            logger.debug("Interferogram product already exists: '%s'", product_path.name)
            return OkResult(source=pair, result=product, skipped=True)

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
                        snap_userdir=snap_userdir,
                        process_params={
                            "swath": product.sub_swath,
                            "polarization": product.polarization,
                            "date1": product.date_primary.strftime("%d%b%Y"),
                            "date2": product.date_secondary.strftime("%d%b%Y"),
                            "crs": get_epsg_string(crs),
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
                except (BurstError, FileNotFoundError, RuntimeError) as e:
                    logger.exception("Failed to process burst pair '%s'", pair)
                    return ErrorResult(source=pair, error=e)

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
def snap_worker_userdir(shared_cache: Path, worker_root: Path) -> Iterator[Path]:
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
