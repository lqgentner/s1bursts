"""Low-level SNAP GPT client for Sentinel-1 processing."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any, cast

from freezecore.raster import (
    build_rasterio_profile,
    get_epsg_string,
    rewrite_tiff,
)
import numpy as np
import rasterio

from s1bursts.config import get_settings
from s1bursts.dimap import BeamDimapDataset
from s1bursts.errors import (
    BurstError,
    ConfigurationError,
    SnapProcessingError,
)
from s1bursts.product import (
    CoherenceComplexProduct,
    CoherenceProduct,
    CRSLike,
    GTCProduct,
    InterferogramProduct,
    RTCProduct,
)
from s1bursts.snap_graph import (
    BackscatterConfig,
    CoherenceConfig,
    InterferogramConfig,
    build_terrain_correction_graph,
)
from s1bursts.summary import ErrorResult, OkResult, OperationResult
from s1bursts.utils import sliding_window_mean

if TYPE_CHECKING:
    from collections.abc import Iterator

    from upath import UPath

    from s1bursts.burst import Burst, BurstPair
    from s1bursts.config import SNAPSettings
    from s1bursts.filemanager import FileManager

logger = logging.getLogger(__name__)

_AUXDATA_SUBDIRS: tuple[str, ...] = ("dem", "Orbits")


def _write_graph(tmp_dir: Path, name: str, xml: str) -> Path:
    """Write a generated SNAP GPT graph XML to ``tmp_dir`` and return its path."""
    graph_file = tmp_dir / f"_{name}_graph.xml"
    graph_file.write_text(xml)
    return graph_file


def _apply_map_looks(geocoded_file: Path, map_looks: tuple[int, int]) -> None:
    """Coherently average a geocoded complex coherence and rewrite it as magnitude.

    Replaces the two-band ``coh_i`` / ``coh_q`` raster at *geocoded_file* with a
    single-band coherence magnitude, averaged over a ``(n_rows, n_cols)`` window
    of the map grid before the modulus is taken. Averaging the complex value
    rather than the magnitude is what suppresses the estimator's small-window
    bias, since the noise contribution carries random phase and cancels.

    Parameters
    ----------
    geocoded_file : Path
        Two-band terrain-corrected complex coherence, modified in place.
    map_looks : tuple of int
        Averaging window as ``(n_rows, n_cols)`` in geocoded pixels.
    """
    n_rows, n_cols = map_looks
    with rasterio.open(geocoded_file) as src:
        profile = src.profile
        coh = (src.read(1) + 1j * src.read(2)).astype(np.complex64)

    averaged = sliding_window_mean(coh, win_row=n_rows, win_col=n_cols)
    magnitude = np.abs(averaged).astype(np.float32)

    profile.update(count=1, dtype="float32")
    with rasterio.open(geocoded_file, "w", **profile) as dst:
        dst.write(magnitude, 1)


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
        Number of parallel GPT subprocesses. Defaults to ``SNAP_MAX_WORKERS``
        (or 1).
    max_ram_gb : float, optional
        Total RAM budget across all GPT processes in GB. Defaults to
        ``SNAP_MAX_RAM_GB`` (or 4).
    gpt_threads_per_job : int, optional
        Threads per GPT process (``-q``). Defaults to ``SNAP_THREADS_PER_JOB``
        or ``cpu_count // max_workers``.
    gpt_cache_ratio : float, optional
        Fraction of the per-process RAM budget used for the GPT tile cache
        (``-c``). Defaults to ``SNAP_CACHE_RATIO`` (or 0.7).
    auxdata_path : Path, optional
        Shared SNAP auxdata cache (DEM tiles, orbit files). Defaults to
        ``SNAP_AUXDATA_PATH`` (or ``~/.snap/auxdata``).
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
        snap_settings = get_settings().snap
        self._resolve_paths(snap_settings)
        self._resolve_resources(snap_settings)

    def _resolve_paths(self, snap_settings: SNAPSettings) -> None:
        """Resolve GPT and auxdata paths from explicit values and settings."""
        self.resolved_gpt_path = _resolve_gpt_path(self.bin, snap_settings)
        logger.debug("Resolved SNAP GPT executable: %s", self.resolved_gpt_path)

        auxdata_path = self.auxdata_path or snap_settings.auxdata_path
        if auxdata_path is None:
            msg = (
                "SNAP auxdata path is None. "
                "Specify `auxdata_path` or set `SNAP_AUXDATA_PATH` in environment."
            )
            raise ConfigurationError(msg)

        self.resolved_auxdata_path = auxdata_path

    def _resolve_resources(self, snap_settings: SNAPSettings) -> None:
        """Resolve GPT resource parameters from explicit values, settings, and defaults."""
        n_cpus = os.cpu_count() or 4

        # max_workers: explicit > settings > 1
        self.resolved_max_workers = max(1, self.max_workers or snap_settings.max_workers)

        # total RAM budget for Xmx calculation
        ram_gb = self.max_ram_gb or snap_settings.max_ram_gb
        ram_mb = int(ram_gb * 1024)

        # per-job Xmx (MB), with a floor so small RAM budgets never produce
        # a zero or negative -Xmx value
        self.resolved_xmx_mb = max(512, (ram_mb - 3_000) // self.resolved_max_workers)

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

        # The JVM ignores TMPDIR and defaults java.io.tmpdir to /tmp, where jblas
        # extracts 30 MB of native libraries and deletes them from a shutdown hook
        # that a SIGKILL skips. Keeping them under the userdir hands cleanup to
        # snap_worker_userdir, which removes it however gpt died.
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
                # Fold SNAP's own error text into the exception, since
                # CalledProcessError only reports the exit code. Without this the
                # real cause (e.g. "No valid orbit file found") never reaches the
                # ErrorResult / summary the caller sees.
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
        band_names: list[str],
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
        band_names : list[str]
            Band descriptions, one per band.
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
            # The staging file is a disposable copy in a TemporaryDirectory, so move
            # it into place rather than leaving a redundant copy behind.
            rewrite_tiff(
                src_file=staging_file,
                dst_file=product_path,
                profile=profile,
                band_names=band_names,
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
            graph_file = _write_graph(tmp_dir, "backscatter", config.to_xml())
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
                        band_names=config.output_bands,
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
    ) -> OperationResult[CoherenceProduct | CoherenceComplexProduct]:
        """Compute the coherence (magnitude or complex) from a burst pair.

        ``config.write_complex=False`` (default) runs the single-shot
        SLC→Coherence→Terrain-Correction graph and writes a magnitude
        GeoTIFF.  ``config.write_complex=True`` runs the no-TC ifg graph,
        computes complex coherence in Python, then runs a standalone
        Terrain-Correction graph; the result has ``coh_i`` / ``coh_q``
        bands.

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
        if config.write_complex:
            return self._create_coherence_complex(
                pair,
                crs,
                config=config,
                overwrite=overwrite,
            )
        magnitude = self._create_coherence_magnitude(
            pair,
            crs,
            config=config,
            overwrite=overwrite,
        )
        return cast("OperationResult[CoherenceProduct | CoherenceComplexProduct]", magnitude)

    def _create_coherence_magnitude(
        self,
        pair: BurstPair,
        crs: CRSLike,
        *,
        config: CoherenceConfig,
        overwrite: bool,
    ) -> OperationResult[CoherenceProduct]:
        product = CoherenceProduct.from_burst_pair(pair, crs=crs)
        product_path = self.manager.get_product_path(product)

        if not overwrite and self.manager.product_exists(product):
            logger.debug("Coherence product already exists: '%s'", product_path.name)
            return OkResult(source=pair, result=product, skipped=True)

        logger.info("Creating coherence product: '%s'", product_path.name)

        with tempfile.TemporaryDirectory(prefix="s1bursts-") as tmp:
            tmp_dir = Path(tmp)
            staging_file = tmp_dir / product_path.name
            graph_file = _write_graph(tmp_dir, "coherence", config.to_xml())
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
                        band_names=config.output_bands,
                    )
                except (BurstError, FileNotFoundError, RuntimeError) as e:
                    logger.exception("Failed to process burst pair '%s'", pair)
                    return ErrorResult(source=pair, error=e)

        logger.info("Successfully created '%s'", product_path.name)
        return OkResult(source=pair, result=product)

    def _create_coherence_complex(
        self,
        pair: BurstPair,
        crs: CRSLike,
        *,
        config: CoherenceConfig,
        overwrite: bool,
    ) -> OperationResult[CoherenceProduct | CoherenceComplexProduct]:
        """Run the no-TC ifg graph + Python complex coherence + standalone TC graph.

        With ``config.map_looks`` set, a coherent box average is applied on the
        geocoded grid and the magnitude written, so the product on disk is a
        plain ``COH`` and everything downstream is unchanged.
        """
        product: CoherenceProduct | CoherenceComplexProduct = (
            CoherenceProduct.from_burst_pair(pair, crs=crs)
            if config.map_looks is not None
            else CoherenceComplexProduct.from_burst_pair(pair, crs=crs)
        )
        product_path = self.manager.get_product_path(product)

        if not overwrite and self.manager.product_exists(product):
            logger.debug(
                "Complex coherence product already exists: '%s'",
                product_path.name,
            )
            return OkResult(source=pair, result=product, skipped=True)

        logger.info("Creating complex coherence product: '%s'", product_path.name)

        with tempfile.TemporaryDirectory(prefix="s1bursts-") as tmp:
            tmp_dir = Path(tmp)
            staging_file = tmp_dir / product_path.name
            ifg_file = staging_file.with_suffix(".tmp.dim")
            ifg_graph = _write_graph(tmp_dir, "coherence_ifg", config.to_xml())
            tc_graph = _write_graph(
                tmp_dir,
                "coherence_tc",
                build_terrain_correction_graph(config),
            )
            with snap_worker_userdir(self.config.resolved_auxdata_path, tmp_dir) as snap_userdir:
                try:
                    primary_file = self._stage_slc(pair.primary, tmp_dir)
                    secondary_file = self._stage_slc(pair.secondary, tmp_dir)
                    self._run_gpt(
                        graph=ifg_graph,
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
                        },
                        target_file=ifg_file,
                    )
                    window_rg, window_az = config.window
                    self._write_complex_coherence(
                        ifg_file,
                        window_rg=window_rg,
                        window_az=window_az,
                    )
                    self._run_gpt(
                        graph=tc_graph,
                        source_files={"source": ifg_file},
                        snap_userdir=snap_userdir,
                        process_params={"crs": get_epsg_string(crs)},
                        target_file=staging_file,
                    )
                    if config.map_looks is not None:
                        _apply_map_looks(staging_file, config.map_looks)
                    self._finalize_product(
                        staging_file,
                        product_path,
                        band_names=config.output_bands,
                    )
                except (BurstError, FileNotFoundError, RuntimeError) as e:
                    logger.exception("Failed to process burst pair '%s'", pair)
                    return ErrorResult(source=pair, error=e)

        logger.info("Successfully created '%s'", product_path.name)
        return OkResult(source=pair, result=product)

    def _write_complex_coherence(
        self,
        dim_file: Path,
        *,
        window_rg: int,
        window_az: int,
        delete_existing_bands: bool = True,
    ) -> None:
        r"""Read interferogram bands from a BEAM-DIMAP file and write complex coherence.

        Reads ``i_ifg``, ``q_ifg``, ``intensity1``, and
        ``intensity2`` from *dim_file*, computes complex coherence using
        a sliding-window approach that mirrors SNAP's ``CoherenceOp``.
        Replaces bands in existing dataset with new ``i_coh`` and ``q_coh``
        values

        The complex coherence formula is:

        .. math::

            \gamma = \frac{\sum(u_1 \cdot \bar{u}_2)}
                          {\sqrt{\sum|u_1|^2 \cdot \sum|u_2|^2}}

        where :math:`u_1 \cdot \bar{u}_2` is the interferogram cross-product
        already encoded in the ``i_ifg`` / ``q_ifg`` bands by SNAP, and the
        intensities :math:`|u_1|^2`, :math:`|u_2|^2` are in
        ``intensity1`` / ``intensity2``.

        Parameters
        ----------
        dim_file : Path
            Path to the BEAM-DIMAP ``.dim`` file produced by
            ``S1_SLCtoIfg-noTC.xml``.
        window_rg : int
            Coherence estimation window size in the range direction.
        window_az : int
            Coherence estimation window size in the azimuth direction.
        delete_existing_bands : bool, default: True
            Whether to delete existing bands of BEAM-DIMAP file after
            writing complex coherence bands.
        """
        logger.debug(
            "Computing complex coherence from '%s', window (rg=%d, az=%d)",
            dim_file.name,
            window_rg,
            window_az,
        )

        ds = BeamDimapDataset(dim_file)
        existing_bands = ds.band_names
        # (Az, Rg)
        i_ifg = ds.read_band("i_ifg").astype(np.float64)
        q_ifg = ds.read_band("q_ifg").astype(np.float64)
        int1 = ds.read_band("intensity1").astype(np.float64)
        int2 = ds.read_band("intensity2").astype(np.float64)

        # Complex cross-product: ifg already encodes u1 * conj(u2) from SNAP
        complex_ifg = i_ifg + 1j * q_ifg

        # NaN-safe sliding window sums (mean * window_size = sum)
        n = window_rg * window_az

        # 75 % of pixels in sliding window must be finite to produce valid output
        sliding_mean = partial(
            sliding_window_mean,
            win_row=window_az,
            win_col=window_rg,
            min_coverage=0.75,
        )
        sum_ifg = sliding_mean(complex_ifg) * n
        sum_int1 = sliding_mean(int1) * n
        sum_int2 = sliding_mean(int2) * n

        # Complex coherence = Σ(u1·ū2) / sqrt(Σ|u1|² · Σ|u2|²)
        denom = np.sqrt(sum_int1 * sum_int2)
        with np.errstate(invalid="ignore", divide="ignore"):
            coh_complex = sum_ifg / denom

        # Can't use units "real" and "imaginary": Make SNAP expect a virtual intensity
        # band immediately after; raises error in RangeDopplerGeocodingOp
        ds.add_band("i_coh", coh_complex.real.astype(np.float32), unit="coherence")
        ds.add_band("q_coh", coh_complex.imag.astype(np.float32), unit="coherence")

        if delete_existing_bands:
            for band in existing_bands:
                ds.delete_band(band)

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
            graph_file = _write_graph(tmp_dir, "interferogram", config.to_xml())
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
                        band_names=config.output_bands,
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
