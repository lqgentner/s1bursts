"""Offline unit tests for s1bursts.snap_client helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
from pandas import Timestamp
from pyproj import CRS
import pytest
import rasterio
import rasterio.transform
from upath import UPath

from s1bursts.burst import Burst, BurstPair
from s1bursts.config import SNAPSettings
from s1bursts.errors import ConfigurationError, SnapProcessingError
from s1bursts.product_tags import RasterMetadata
from s1bursts.snap_client import (
    SNAPClient,
    SNAPConfig,
    _extract_gpt_error,
    _resolve_gpt_path,
    snap_version,
    snap_worker_userdir,
)
from s1bursts.snap_graph import BackscatterConfig, CoherenceConfig
from s1bursts.summary import ErrorResult, OkResult

if TYPE_CHECKING:
    from collections.abc import Callable


def _snap_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **overrides: object,
) -> SNAPSettings:
    """Build a ``SNAPSettings`` isolated from the ambient environment/.env file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SNAP_BIN", raising=False)
    return SNAPSettings(**overrides)  # type: ignore[arg-type]


def _make_executable(path: Path) -> Path:
    path.touch()
    path.chmod(0o755)
    return path


# Representative gpt stderr for an Apply-Orbit-File failure: an ``Error:`` block
# followed by a Java stack trace.
_GPT_ORBIT_ERROR = dedent("""\
    Executing processing graph
    WARNING: eu.esa.sar.orbits.gpf.ApplyOrbitFileOp: No valid orbit file found
     done.

    Error: [NodeId: Apply-Orbit-File] No valid orbit file found for 29-JUN-2015 05:51:38.420000
    Orbit files may be downloaded from Copernicus Dataspaces
    and placed in /Users/lgentn/.snap/auxdata/Orbits/Sentinel-1/POEORB/S1A/2015/06
    org.esa.snap.core.gpf.graph.GraphException: [NodeId: Apply-Orbit-File] No valid orbit file
        at org.esa.snap.core.gpf.graph.NodeContext.initTargetProduct(NodeContext.java:73)
        at org.esa.snap.core.gpf.graph.GraphContext.initNodeContext(GraphContext.java:213)
""")


def test_extract_gpt_error_returns_error_block_without_stack_trace() -> None:
    detail = _extract_gpt_error(None, _GPT_ORBIT_ERROR)

    assert detail.startswith("Error: [NodeId: Apply-Orbit-File] No valid orbit file found")
    # The human-readable continuation lines are kept ...
    assert "Orbit files may be downloaded from Copernicus Dataspaces" in detail
    # ... but the Java stack trace is dropped.
    assert "org.esa.snap" not in detail
    assert "at org.esa.snap" not in detail


def test_extract_gpt_error_prefers_stderr_over_stdout() -> None:
    detail = _extract_gpt_error("Error: from stdout", "Error: from stderr")

    assert detail == "Error: from stderr"


def test_extract_gpt_error_falls_back_to_output_tail() -> None:
    # No "Error:" marker — fall back to the tail of the captured output.
    output = "line 1\nline 2\nsome fatal message\n"
    detail = _extract_gpt_error(None, output)

    assert detail == "line 1\nline 2\nsome fatal message"


def test_extract_gpt_error_handles_empty_output() -> None:
    assert _extract_gpt_error(None, None) == ""
    assert _extract_gpt_error("", "") == ""


def test_extract_gpt_error_truncates_to_max_chars() -> None:
    long_line = "x" * 5000
    detail = _extract_gpt_error(None, f"Error: {long_line}", max_chars=100)

    assert len(detail) == 100


class TestSnapVersion:
    """Version detection, read from the installation's VERSION.txt."""

    @staticmethod
    def _install(tmp_path: Path, version: str | None) -> Path:
        """Build a SNAP-shaped install, with or without its VERSION.txt."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        if version is not None:
            (tmp_path / "VERSION.txt").write_text(version)
        return _make_executable(bin_dir / "gpt")

    def test_reads_version_file(self, tmp_path: Path) -> None:
        gpt = self._install(tmp_path, "14.0.0\n")

        assert snap_version(gpt) == (14, 0, 0)

    def test_returns_none_without_a_version_file(self, tmp_path: Path) -> None:
        assert snap_version(self._install(tmp_path, None)) is None

    @pytest.mark.parametrize(
        ("version", "warns"),
        [("14.0.0", False), ("13.0.0", False), ("12.0.0", True), ("15.0.0", True)],
    )
    def test_config_warns_only_for_an_untested_version(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        version: str,
        warns: bool,
    ) -> None:
        """An install outside SUPPORTED_SNAP_VERSIONS should say so in the log."""
        monkeypatch.delenv("SNAP_BIN", raising=False)
        install = tmp_path / version
        bin_dir = install / "bin"
        bin_dir.mkdir(parents=True)
        _make_executable(bin_dir / "gpt")
        (install / "VERSION.txt").write_text(version)

        with caplog.at_level(logging.WARNING, logger="s1bursts.snap_client"):
            SNAPConfig(bin=bin_dir, auxdata_path=tmp_path / "auxdata")

        untested = [r for r in caplog.records if "has not been tested" in r.message]
        assert bool(untested) is warns

    def test_returns_none_for_an_unparsable_version_file(self, tmp_path: Path) -> None:
        assert snap_version(self._install(tmp_path, "not a version")) is None


class TestResolveGptPath:
    """Precedence: explicit `bin` argument > SNAP_BIN > PATH."""

    def test_explicit_bin_wins_over_snap_bin(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        explicit_bin = tmp_path / "explicit-bin"
        explicit_bin.mkdir()
        explicit_gpt = _make_executable(explicit_bin / "gpt")
        settings_bin = tmp_path / "settings-bin"
        settings_bin.mkdir()
        _make_executable(settings_bin / "gpt")
        settings = _snap_settings(tmp_path, monkeypatch, bin=settings_bin)

        assert _resolve_gpt_path(explicit_bin, settings) == explicit_gpt

    def test_snap_bin_resolves_gpt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        gpt = _make_executable(bin_dir / "gpt")
        settings = _snap_settings(tmp_path, monkeypatch, bin=bin_dir)

        assert _resolve_gpt_path(None, settings) == gpt

    def test_snap_bin_resolves_windows_style_executable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        gpt = _make_executable(bin_dir / "gpt.exe")
        settings = _snap_settings(tmp_path, monkeypatch, bin=bin_dir)

        assert _resolve_gpt_path(None, settings) == gpt

    def test_snap_bin_without_gpt_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        settings = _snap_settings(tmp_path, monkeypatch, bin=bin_dir)

        with pytest.raises(ConfigurationError, match="No `gpt` executable found"):
            _resolve_gpt_path(None, settings)

    def test_snap_bin_nonexistent_directory_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = _snap_settings(tmp_path, monkeypatch, bin=tmp_path / "missing-dir")

        with pytest.raises(ConfigurationError, match="No `gpt` executable found"):
            _resolve_gpt_path(None, settings)

    def test_falls_back_to_path_when_unconfigured(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = _snap_settings(tmp_path, monkeypatch)
        which_path = _make_executable(tmp_path / "path-gpt")
        monkeypatch.setattr("s1bursts.snap_client.shutil.which", lambda _name: str(which_path))

        assert _resolve_gpt_path(None, settings) == which_path

    def test_raises_when_nothing_resolves(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = _snap_settings(tmp_path, monkeypatch)
        monkeypatch.setattr("s1bursts.snap_client.shutil.which", lambda _name: None)

        with pytest.raises(ConfigurationError, match="Could not resolve the SNAP `gpt`"):
            _resolve_gpt_path(None, settings)

    def test_explicit_bin_gpt_must_be_executable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        explicit_bin = tmp_path / "explicit-bin"
        explicit_bin.mkdir()
        gpt = explicit_bin / "gpt"
        gpt.touch()
        gpt.chmod(0o644)
        settings = _snap_settings(tmp_path, monkeypatch)

        with pytest.raises(ConfigurationError, match="not executable"):
            _resolve_gpt_path(explicit_bin, settings)


class TestSnapWorkerUserdir:
    def test_symlinks_shared_tiles_and_publishes_new_files(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        (shared / "dem").mkdir(parents=True)
        (shared / "dem" / "tile_1.tif").write_text("existing tile")
        worker_root = tmp_path / "job"
        worker_root.mkdir()

        with snap_worker_userdir(shared, worker_root) as userdir:
            staged = userdir / "auxdata" / "dem" / "tile_1.tif"
            assert staged.is_symlink()
            assert staged.read_text() == "existing tile"
            # Simulate SNAP downloading a new tile during the job
            (userdir / "auxdata" / "dem" / "tile_2.tif").write_text("new tile")
            kept_userdir = userdir

        # New tile published back to the shared cache; symlinked one untouched
        assert (shared / "dem" / "tile_2.tif").read_text() == "new tile"
        assert (shared / "dem" / "tile_1.tif").read_text() == "existing tile"
        # Worker dir cleaned up
        assert not kept_userdir.exists()

    def test_existing_shared_tile_is_not_overwritten(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        (shared / "Orbits").mkdir(parents=True)
        (shared / "Orbits" / "orbit_a.zip").write_text("original")
        worker_root = tmp_path / "job"
        worker_root.mkdir()

        with snap_worker_userdir(shared, worker_root) as userdir:
            orbit = userdir / "auxdata" / "Orbits" / "orbit_a.zip"
            # Replace the symlink with a real (diverging) file
            orbit.unlink()
            orbit.write_text("modified")

        assert (shared / "Orbits" / "orbit_a.zip").read_text() == "original"


BURST_ID = 30346


def _make_burst(date: str) -> Burst:
    # Propagate the relative orbit number (as a burst from a CDSE search or SLC
    # filename would) so product creation needs no opt-in burst-map download.
    return Burst(
        burst_id=BURST_ID,
        sub_swath="IW1",
        polarization="VV",
        date=Timestamp(date),
        _rel_orbit_number=44,
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SNAPClient:
    """SNAPClient with a stub gpt executable and a mocked FileManager."""
    monkeypatch.delenv("SNAP_BIN", raising=False)
    monkeypatch.chdir(tmp_path)
    gpt = tmp_path / "gpt"
    gpt.touch()
    gpt.chmod(0o755)
    manager = MagicMock()
    manager.get_product_path.return_value = tmp_path / "product.tif"
    manager.product_exists.return_value = False
    manager.get_slc_path.return_value = tmp_path / "slc.zip"
    config = SNAPConfig(
        bin=tmp_path,
        auxdata_path=tmp_path / "auxdata",
    )
    return SNAPClient(manager, config)


class TestCreateBackscatter:
    CRS_UTM32 = CRS("EPSG:32632")

    def test_successful_processing(self, client: SNAPClient) -> None:
        burst = _make_burst("2022-01-01")

        with (
            patch.object(client, "_run_gpt") as mock_gpt,
            patch.object(client, "_finalize_product") as mock_finalize,
        ):
            result = client.create_backscatter(
                burst,
                crs=self.CRS_UTM32,
                config=BackscatterConfig(),
            )

        assert isinstance(result, OkResult)
        assert result.result.product_type == "RTC"
        assert not result.skipped
        params = mock_gpt.call_args.kwargs["process_params"]
        assert params["swath"] == "IW1"
        assert params["polarization"] == "VV"
        assert params["crs"] == "EPSG:32632"
        metadata = mock_finalize.call_args.kwargs["metadata"]
        assert metadata.band_names == ["Gamma0", "simulatedImage"]
        assert metadata.tags["PRODUCT_TYPE"] == "RTC"

    def test_gtc_when_terrain_flattening_disabled(self, client: SNAPClient) -> None:
        burst = _make_burst("2022-01-01")

        with (
            patch.object(client, "_run_gpt"),
            patch.object(client, "_finalize_product"),
        ):
            result = client.create_backscatter(
                burst,
                crs=self.CRS_UTM32,
                config=BackscatterConfig(flatten_terrain=False),
            )

        assert isinstance(result, OkResult)
        assert result.result.product_type == "GTC"

    def test_skips_existing_product(self, client: SNAPClient) -> None:
        client.manager.product_exists.return_value = True

        with patch.object(client, "_run_gpt") as mock_gpt:
            result = client.create_backscatter(
                _make_burst("2022-01-01"),
                crs=self.CRS_UTM32,
                config=BackscatterConfig(),
            )

        assert isinstance(result, OkResult)
        assert result.skipped
        mock_gpt.assert_not_called()

    def test_gpt_failure_returns_error_result(self, client: SNAPClient) -> None:
        burst = _make_burst("2022-01-01")

        with patch.object(
            client,
            "_run_gpt",
            side_effect=SnapProcessingError("GPT processing failed"),
        ):
            result = client.create_backscatter(
                burst,
                crs=self.CRS_UTM32,
                config=BackscatterConfig(),
            )

        assert isinstance(result, ErrorResult)
        assert isinstance(result.error, SnapProcessingError)
        assert result.source is burst


class TestFinalizeProduct:
    @staticmethod
    def _write_staging(path: Path, data: np.ndarray) -> None:
        """Write a SNAP-style staging GeoTIFF: no NODATA tag, one band per plane."""
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=data.shape[2],
            height=data.shape[1],
            count=data.shape[0],
            dtype="float32",
            crs="EPSG:32632",
            transform=rasterio.transform.from_origin(400000, 5200000, 10, 10),
        ) as ds:
            ds.write(data)

    def test_stamps_nodata_band_names_and_tags(self, client: SNAPClient, tmp_path: Path) -> None:
        """SNAP-style staging GeoTIFF (no NODATA tag) gains nodata, descriptions and tags."""
        staging = tmp_path / "staging.tif"
        product = tmp_path / "final.tif"
        data = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
        self._write_staging(staging, data[np.newaxis])

        client._finalize_product(
            staging,
            product,
            metadata=RasterMetadata(
                tags={"PRODUCT_TYPE": "RTC", "BACKSCATTER_CONVENTION": "Power"},
                band_names=["Gamma0"],
                units=["linear power"],
                band_tags=[{"LAYER_NAME": "gamma0"}],
            ),
        )

        with rasterio.open(product) as ds:
            assert ds.nodata is not None
            assert np.isnan(ds.nodata)
            assert ds.descriptions == ("Gamma0",)
            assert ds.tags()["PRODUCT_TYPE"] == "RTC"
            assert ds.tags()["BACKSCATTER_CONVENTION"] == "Power"
            assert ds.tags(1) == {"LAYER_NAME": "gamma0"}
            assert ds.units == ("linear power",)
            assert np.array_equal(ds.read(1), data, equal_nan=True)

    def test_masks_pixels_that_are_zero_in_every_band(
        self,
        client: SNAPClient,
        tmp_path: Path,
    ) -> None:
        """Terrain-Correction's zero fill becomes no-data; a single zero band does not."""
        staging = tmp_path / "staging.tif"
        product = tmp_path / "final.tif"
        # Column 0 is the zero fill; column 1 is zero in one band only.
        data = np.array(
            [[[0.0, 0.0], [0.0, 3.0]], [[0.0, 4.0], [0.0, 5.0]]],
            dtype=np.float32,
        )
        self._write_staging(staging, data)

        client._finalize_product(
            staging,
            product,
            metadata=RasterMetadata(tags={}, band_names=["i_ifg", "q_ifg"], units=["", ""]),
            mask_zero_fill=True,
        )

        with rasterio.open(product) as ds:
            assert np.isnan(ds.read(1)[:, 0]).all()
            assert np.isnan(ds.read(2)[:, 0]).all()
            assert ds.read(1)[0, 1] == 0.0
            assert ds.read(2)[0, 1] == 4.0

    def test_keeps_zeros_when_masking_is_off(self, client: SNAPClient, tmp_path: Path) -> None:
        """Without the band maths that removes them, zeros are data and are preserved."""
        staging = tmp_path / "staging.tif"
        product = tmp_path / "final.tif"
        data = np.zeros((1, 2, 2), dtype=np.float32)
        self._write_staging(staging, data)

        client._finalize_product(
            staging,
            product,
            metadata=RasterMetadata(tags={}, band_names=["Gamma0"], units=["linear power"]),
        )

        with rasterio.open(product) as ds:
            assert np.array_equal(ds.read(1), data[0])


class TestCreateCoherence:
    CRS_UTM32 = CRS("EPSG:32632")

    def _make_pair(self) -> BurstPair:
        return BurstPair(_make_burst("2022-01-01"), _make_burst("2022-01-13"))

    def test_creates_coherence_product(self, client: SNAPClient) -> None:
        with (
            patch.object(client, "_run_gpt") as mock_gpt,
            patch.object(client, "_finalize_product") as mock_finalize,
        ):
            result = client.create_coherence(
                self._make_pair(),
                crs=self.CRS_UTM32,
                config=CoherenceConfig(),
            )

        assert isinstance(result, OkResult)
        assert result.result.product_type == "COH"
        # Single-shot graph: one gpt invocation
        assert mock_gpt.call_count == 1
        params = mock_gpt.call_args.kwargs["process_params"]
        assert params["date1"] == "01Jan2022"
        assert params["date2"] == "13Jan2022"
        metadata = mock_finalize.call_args.kwargs["metadata"]
        assert metadata.band_names == ["coh"]
        assert metadata.tags["COHERENCE_WINDOW_RANGE"] == "25"

    def test_gpt_failure_returns_error_result(self, client: SNAPClient) -> None:
        pair = self._make_pair()

        with patch.object(
            client,
            "_run_gpt",
            side_effect=SnapProcessingError("boom"),
        ):
            result = client.create_coherence(pair, crs=self.CRS_UTM32, config=CoherenceConfig())

        assert isinstance(result, ErrorResult)
        assert result.source is pair


class TestResourceBudget:
    """SNAPConfig must reject a budget that cannot describe a runnable job."""

    @pytest.fixture
    def snap_bin(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.delenv("SNAP_BIN", raising=False)
        for name in ("SNAP_MAX_WORKERS", "SNAP_MAX_RAM_GB", "SNAP_CACHE_RATIO"):
            monkeypatch.delenv(name, raising=False)
        gpt = tmp_path / "gpt"
        gpt.touch()
        gpt.chmod(0o755)
        return tmp_path

    def _config(self, snap_bin: Path, **kwargs: object) -> SNAPConfig:
        return SNAPConfig(bin=snap_bin, auxdata_path=snap_bin / "auxdata", **kwargs)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("max_workers", 0, "max_workers` must be at least 1"),
            ("max_workers", -2, "max_workers` must be at least 1"),
            ("gpt_threads_per_job", 0, "gpt_threads_per_job` must be at least 1"),
            ("max_ram_gb", 0.0, "max_ram_gb` must be positive"),
            ("max_ram_gb", -4.0, "max_ram_gb` must be positive"),
            ("gpt_cache_ratio", -0.5, "gpt_cache_ratio` must be greater than 0"),
            ("gpt_cache_ratio", 1.5, "gpt_cache_ratio` must be greater than 0"),
        ],
    )
    def test_rejects_out_of_range_values(
        self,
        snap_bin: Path,
        field: str,
        value: object,
        message: str,
    ) -> None:
        with pytest.raises(ConfigurationError, match=message):
            self._config(snap_bin, **{field: value})

    def test_rejects_a_budget_too_small_for_one_worker(self, snap_bin: Path) -> None:
        """3 GB of overhead plus a 512 MB heap does not fit in 3 GB."""
        with pytest.raises(ConfigurationError, match="below the 512 MB SNAP needs"):
            self._config(snap_bin, max_ram_gb=3, max_workers=1)

    def test_rejects_a_budget_over_subscribed_by_workers(self, snap_bin: Path) -> None:
        """8 GB is ample for one worker but leaves 16 of them 324 MB of heap each."""
        with pytest.raises(ConfigurationError, match="16 worker"):
            self._config(snap_bin, max_ram_gb=8, max_workers=16)

    def test_error_names_a_budget_that_would_work(self, snap_bin: Path) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            self._config(snap_bin, max_ram_gb=4, max_workers=4)

        suggested = float(str(excinfo.value).split("at least ")[1].split()[0])
        widened = self._config(snap_bin, max_ram_gb=suggested, max_workers=4)

        assert widened.resolved_xmx_mb >= 512

    def test_never_allocates_more_heap_than_the_declared_budget(self, snap_bin: Path) -> None:
        """The per-worker floor must not push total heap past `max_ram_gb`."""
        config = self._config(snap_bin, max_ram_gb=16, max_workers=4)

        total_heap_mb = config.resolved_xmx_mb * config.resolved_max_workers

        assert total_heap_mb <= 16 * 1024

    def test_cache_never_exceeds_the_heap_it_lives_in(self, snap_bin: Path) -> None:
        config = self._config(snap_bin, max_ram_gb=16, max_workers=2, gpt_cache_ratio=1.0)

        assert config.resolved_cache_mb <= config.resolved_xmx_mb

    def test_accepts_a_boundary_budget(self, snap_bin: Path) -> None:
        """Exactly the overhead plus one minimum heap must be allowed through."""
        config = self._config(snap_bin, max_ram_gb=(3000 + 512) / 1024, max_workers=1)

        assert config.resolved_xmx_mb == 512

    def test_fractional_ram_budgets_are_accepted(self, snap_bin: Path) -> None:
        config = self._config(snap_bin, max_ram_gb=6.5, max_workers=1)

        assert config.resolved_xmx_mb == int(6.5 * 1024) - 3000


class TestRunGpt:
    """GPT invocation and failure handling, driven by a stub `gpt` executable.

    The stub is a real script, so argument assembly, the subprocess call, exit
    handling and output checks all run for real.
    """

    @pytest.fixture
    def gpt_client(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> Callable[[str], SNAPClient]:
        """Return a factory building a client around a stub `gpt` with given behaviour."""
        monkeypatch.delenv("SNAP_BIN", raising=False)

        def _client(script: str) -> SNAPClient:
            gpt = tmp_path / "gpt"
            gpt.write_text(f"#!/bin/sh\n{script}\n")
            gpt.chmod(0o755)
            manager = MagicMock()
            manager.get_slc_path.return_value = tmp_path / "slc.zip"
            config = SNAPConfig(
                bin=tmp_path,
                auxdata_path=tmp_path / "auxdata",
                max_ram_gb=8,
                gpt_threads_per_job=3,
            )
            return SNAPClient(manager, config)

        return _client

    def _invoke(self, client: SNAPClient, tmp_path: Path, target_name: str = "out.tif") -> Path:
        return client._run_gpt(
            graph=tmp_path / "graph.xml",
            source_files={"primary": tmp_path / "a.zip", "secondary": tmp_path / "b.zip"},
            target_file=tmp_path / target_name,
            snap_userdir=tmp_path / "userdir",
            process_params={"crs": "EPSG:32632"},
        )

    def test_passes_sources_parameters_and_resources_to_gpt(
        self,
        gpt_client: Callable[[str], SNAPClient],
        tmp_path: Path,
    ) -> None:
        client = gpt_client('printf "%s\\n" "$@" > "$(dirname "$0")/argv.txt"; touch "$6"')

        self._invoke(client, tmp_path)

        argv = (tmp_path / "argv.txt").read_text().splitlines()
        assert f"-Sprimary={tmp_path / 'a.zip'}" in argv
        assert f"-Ssecondary={tmp_path / 'b.zip'}" in argv
        assert "-Pcrs=EPSG:32632" in argv
        assert "-q" in argv
        assert argv[argv.index("-q") + 1] == "3"
        assert f"-J-Xmx{client.config.resolved_xmx_mb}M" in argv
        assert f"-c{client.config.resolved_cache_mb}M" not in argv  # -c takes a separate operand
        assert argv[argv.index("-c") + 1] == f"{client.config.resolved_cache_mb}M"

    def test_requests_geotiff_output_for_a_tif_target(
        self,
        gpt_client: Callable[[str], SNAPClient],
        tmp_path: Path,
    ) -> None:
        """The explicit format suppresses GPT's XML sidecar."""
        client = gpt_client('printf "%s\\n" "$@" > "$(dirname "$0")/argv.txt"; touch "$6"')

        self._invoke(client, tmp_path)

        argv = (tmp_path / "argv.txt").read_text().splitlines()
        assert argv[argv.index("-f") + 1] == "GeoTIFF"

    def test_creates_an_isolated_java_tmpdir_under_the_userdir(
        self,
        gpt_client: Callable[[str], SNAPClient],
        tmp_path: Path,
    ) -> None:
        """The jblas native libraries unpack there, so they must not land in shared /tmp."""
        client = gpt_client('printf "%s\\n" "$@" > "$(dirname "$0")/argv.txt"; touch "$6"')

        self._invoke(client, tmp_path)

        argv = (tmp_path / "argv.txt").read_text().splitlines()
        assert f"-J-Djava.io.tmpdir={tmp_path / 'userdir' / 'tmp'}" in argv
        assert (tmp_path / "userdir" / "tmp").is_dir()

    def test_returns_the_target_on_success(
        self,
        gpt_client: Callable[[str], SNAPClient],
        tmp_path: Path,
    ) -> None:
        client = gpt_client('touch "$6"')

        result = self._invoke(client, tmp_path)

        assert result == tmp_path / "out.tif"
        assert result.exists()

    def test_surfaces_snaps_own_error_text_not_just_the_exit_code(
        self,
        gpt_client: Callable[[str], SNAPClient],
        tmp_path: Path,
    ) -> None:
        client = gpt_client(
            'echo "Error: No valid orbit file found" >&2\n'
            'echo "	at org.esa.snap.Foo" >&2\n'
            "exit 1",
        )

        with pytest.raises(SnapProcessingError, match="No valid orbit file found"):
            self._invoke(client, tmp_path)

    def test_reports_the_exit_code(
        self,
        gpt_client: Callable[[str], SNAPClient],
        tmp_path: Path,
    ) -> None:
        client = gpt_client('echo "Error: boom" >&2; exit 7')

        with pytest.raises(SnapProcessingError, match=r"exit 7"):
            self._invoke(client, tmp_path)

    def test_removes_a_partial_output_after_a_failure(
        self,
        gpt_client: Callable[[str], SNAPClient],
        tmp_path: Path,
    ) -> None:
        """A half-written product must not be left where a later run would trust it."""
        client = gpt_client('touch "$6"; echo "Error: died halfway" >&2; exit 1')

        with pytest.raises(SnapProcessingError):
            self._invoke(client, tmp_path)

        assert not (tmp_path / "out.tif").exists()

    def test_removes_a_partial_beam_dimap_data_directory(
        self,
        gpt_client: Callable[[str], SNAPClient],
        tmp_path: Path,
    ) -> None:
        client = gpt_client('touch "$6"; mkdir -p "${6%.dim}.data"; exit 1')

        with pytest.raises(SnapProcessingError):
            self._invoke(client, tmp_path, target_name="out.dim")

        assert not (tmp_path / "out.dim").exists()
        assert not (tmp_path / "out.data").exists()

    def test_reports_a_timeout(
        self,
        gpt_client: Callable[[str], SNAPClient],
        tmp_path: Path,
    ) -> None:
        client = gpt_client("sleep 5")

        with pytest.raises(SnapProcessingError, match="GPT processing failed"):
            client._run_gpt(
                graph=tmp_path / "graph.xml",
                source_files={},
                target_file=tmp_path / "out.tif",
                snap_userdir=tmp_path / "userdir",
                timeout=1,
            )

    def test_reports_a_silent_run_that_produced_nothing(
        self,
        gpt_client: Callable[[str], SNAPClient],
        tmp_path: Path,
    ) -> None:
        """GPT can exit zero without writing the target; that is still a failure."""
        client = gpt_client("exit 0")

        with pytest.raises(SnapProcessingError, match="output not found"):
            self._invoke(client, tmp_path)

    def test_falls_back_to_the_output_tail_without_an_error_marker(
        self,
        gpt_client: Callable[[str], SNAPClient],
        tmp_path: Path,
    ) -> None:
        client = gpt_client('echo "something went wrong in the graph" >&2; exit 2')

        with pytest.raises(SnapProcessingError, match="something went wrong in the graph"):
            self._invoke(client, tmp_path)


class TestStageSlc:
    def test_local_slc_is_used_in_place(self, client: SNAPClient, tmp_path: Path) -> None:
        """A local path needs no copy, so the original is handed straight to GPT."""
        local = tmp_path / "slc.zip"
        local.write_bytes(b"burst")
        client.manager.get_slc_path.return_value = local

        assert client._stage_slc(_make_burst("2022-08-25"), tmp_path / "work") == local

    def test_remote_slc_is_copied_next_to_the_job(
        self,
        client: SNAPClient,
        tmp_path: Path,
    ) -> None:
        """SNAP takes filesystem paths only, so an object-store SLC must be fetched."""
        remote = UPath("memory://slcs/S1_SLC_remote.zip")
        remote.parent.mkdir(parents=True, exist_ok=True)
        remote.write_bytes(b"remote burst")
        client.manager.get_slc_path.return_value = remote
        work = tmp_path / "work"
        work.mkdir()

        staged = client._stage_slc(_make_burst("2022-08-25"), work)

        assert staged == work / "S1_SLC_remote.zip"
        assert staged.read_bytes() == b"remote burst"
