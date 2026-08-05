"""Offline unit tests for s1bursts.snap_client helpers."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock, patch

import numpy as np
from pandas import Timestamp
from pyproj import CRS
import pytest
import rasterio
import rasterio.transform

from s1bursts.burst import Burst, BurstPair
from s1bursts.config import SNAPSettings
from s1bursts.errors import ConfigurationError, SnapProcessingError
from s1bursts.snap_client import (
    SNAPClient,
    SNAPConfig,
    _extract_gpt_error,
    _resolve_gpt_path,
    snap_worker_userdir,
)
from s1bursts.snap_graph import BackscatterConfig, CoherenceConfig
from s1bursts.summary import ErrorResult, OkResult


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
        assert mock_finalize.call_args.kwargs["band_names"] == ["Gamma0", "simulatedImage"]

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
    def test_stamps_nodata_and_band_names(self, client: SNAPClient, tmp_path: Path) -> None:
        """SNAP-style staging GeoTIFF (no NODATA tag) gains nodata + descriptions."""
        staging = tmp_path / "staging.tif"
        product = tmp_path / "final.tif"
        data = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
        with rasterio.open(
            staging,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="float32",
            crs="EPSG:32632",
            transform=rasterio.transform.from_origin(400000, 5200000, 10, 10),
        ) as ds:
            ds.write(data, 1)

        client._finalize_product(staging, product, band_names=["Gamma0"])

        with rasterio.open(product) as ds:
            assert ds.nodata is not None
            assert np.isnan(ds.nodata)
            assert ds.descriptions == ("Gamma0",)
            assert np.array_equal(ds.read(1), data, equal_nan=True)


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
        assert mock_finalize.call_args.kwargs["band_names"] == ["coh"]

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
