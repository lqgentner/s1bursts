"""Offline unit tests for s1bursts.snap_client helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch
import zipfile

from freezebase.errors import RasterWriteError
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
    external_dem_needs_half_pixel_shift,
    snap_dem_module_version,
    snap_version,
    snap_worker_userdir,
)
from s1bursts.snap_graph import (
    NODATA_SENTINEL,
    BackscatterConfig,
    CoherenceConfig,
    InterferogramConfig,
)
from s1bursts.summary import ErrorResult, OkResult

if TYPE_CHECKING:
    from collections.abc import Callable


def _snap_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **overrides: Any,
) -> SNAPSettings:
    """Build a ``SNAPSettings`` isolated from the ambient environment/.env file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SNAP_BIN", raising=False)
    return SNAPSettings(**overrides)


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


def _write_module_jar(cluster: Path, manifest: str | None) -> None:
    """Write the ``snap-dem`` jar, with the given manifest text or without a manifest."""
    jar = cluster / "modules" / "org-esa-snap-snap-dem.jar"
    jar.parent.mkdir(parents=True)
    with zipfile.ZipFile(jar, "w") as archive:
        archive.writestr("org/esa/snap/dem/Placeholder.class", b"")
        if manifest is not None:
            archive.writestr("META-INF/MANIFEST.MF", manifest)


def _manifest(version: str) -> str:
    return (
        "Manifest-Version: 1.0\n"
        "OpenIDE-Module-Implementation-Version: 99.9.9\n"
        f"OpenIDE-Module-Specification-Version: {version}\n"
        "OpenIDE-Module-Build-Version: 202610051459\n"
    )


class _SnapInstall:
    """A SNAP-shaped install whose ``snap-dem`` jar is under test."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.bin_dir = root / "install" / "bin"
        self.bin_dir.mkdir(parents=True)
        self.gpt = _make_executable(self.bin_dir / "gpt")
        # Keeps the default userdir off the developer's real ~/.snap.
        self.home = root / "home"
        monkeypatch.setenv("HOME", str(self.home))
        monkeypatch.setenv("USERPROFILE", str(self.home))

    def install_module(self, version: str) -> None:
        _write_module_jar(self.root / "install" / "snap", _manifest(version))

    def install_module_with_manifest(self, manifest: str | None) -> None:
        _write_module_jar(self.root / "install" / "snap", manifest)

    def update_module_in_userdir(self, version: str, userdir: Path | None = None) -> None:
        _write_module_jar(userdir or self.home / ".snap" / "system", _manifest(version))


@pytest.fixture
def snap_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _SnapInstall:
    return _SnapInstall(tmp_path, monkeypatch)


class TestSnapDemModuleVersion:
    """The ``snap-dem`` version, read from the module jar's manifest."""

    def test_reads_the_installations_jar(self, snap_install: _SnapInstall) -> None:
        snap_install.install_module("14.0.2")

        assert snap_dem_module_version(snap_install.gpt) == (14, 0, 2)

    def test_reads_the_specification_version_not_the_implementation_version(
        self,
        snap_install: _SnapInstall,
    ) -> None:
        snap_install.install_module("14.0.1")

        assert snap_dem_module_version(snap_install.gpt) == (14, 0, 1)

    def test_userdir_update_wins_over_an_older_installation(
        self,
        snap_install: _SnapInstall,
    ) -> None:
        snap_install.install_module("14.0.0")
        snap_install.update_module_in_userdir("14.0.2")

        assert snap_dem_module_version(snap_install.gpt) == (14, 0, 2)

    def test_older_userdir_jar_does_not_hide_the_installation(
        self,
        snap_install: _SnapInstall,
    ) -> None:
        snap_install.install_module("14.0.2")
        snap_install.update_module_in_userdir("14.0.0")

        assert snap_dem_module_version(snap_install.gpt) == (14, 0, 2)

    def test_userdir_comes_from_snap_conf(
        self,
        snap_install: _SnapInstall,
        tmp_path: Path,
    ) -> None:
        custom = tmp_path / "custom-userdir"
        etc = tmp_path / "install" / "etc"
        etc.mkdir()
        (etc / "snap.conf").write_text(f'# comment\ndefault_userdir="{custom}"\n')
        snap_install.update_module_in_userdir("14.0.2", userdir=custom)

        assert snap_dem_module_version(snap_install.gpt) == (14, 0, 2)

    def test_snap_conf_userdir_expands_the_home_variable(
        self,
        snap_install: _SnapInstall,
        tmp_path: Path,
    ) -> None:
        etc = tmp_path / "install" / "etc"
        etc.mkdir()
        (etc / "snap.conf").write_text('default_userdir="${HOME}/.snap/system"\n')
        snap_install.update_module_in_userdir("14.0.2")

        assert snap_dem_module_version(snap_install.gpt) == (14, 0, 2)

    def test_unexpandable_snap_conf_userdir_falls_back_to_the_default(
        self,
        snap_install: _SnapInstall,
        tmp_path: Path,
    ) -> None:
        etc = tmp_path / "install" / "etc"
        etc.mkdir()
        (etc / "snap.conf").write_text('default_userdir="${NO_SUCH_VARIABLE_FOR_SNAP}/system"\n')
        snap_install.update_module_in_userdir("14.0.2")

        assert snap_dem_module_version(snap_install.gpt) == (14, 0, 2)

    def test_tolerates_a_missing_home_directory(
        self,
        snap_install: _SnapInstall,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        snap_install.install_module("14.0.2")

        def _no_home() -> Path:
            msg = "Could not determine home directory."
            raise RuntimeError(msg)

        monkeypatch.setattr(Path, "home", _no_home)

        assert snap_dem_module_version(snap_install.gpt) == (14, 0, 2)

    def test_does_not_cache_across_a_module_update(self, snap_install: _SnapInstall) -> None:
        snap_install.install_module("14.0.0")
        assert snap_dem_module_version(snap_install.gpt) == (14, 0, 0)

        snap_install.update_module_in_userdir("14.0.2")

        assert snap_dem_module_version(snap_install.gpt) == (14, 0, 2)

    def test_returns_none_without_a_jar(self, snap_install: _SnapInstall) -> None:
        assert snap_dem_module_version(snap_install.gpt) is None

    @pytest.mark.parametrize(
        "manifest",
        [
            None,
            "Manifest-Version: 1.0\n",
            _manifest("not a version"),
            "OpenIDE-Module-Specification-Version:\n",
        ],
        ids=["no-manifest", "no-version", "unparsable-version", "empty-version"],
    )
    def test_returns_none_for_an_unusable_manifest(
        self,
        snap_install: _SnapInstall,
        manifest: str | None,
    ) -> None:
        snap_install.install_module_with_manifest(manifest)

        assert snap_dem_module_version(snap_install.gpt) is None

    def test_returns_none_for_a_file_that_is_not_a_jar(self, snap_install: _SnapInstall) -> None:
        jar = snap_install.root / "install" / "snap" / "modules" / "org-esa-snap-snap-dem.jar"
        jar.parent.mkdir(parents=True)
        jar.write_text("not a zip file")

        assert snap_dem_module_version(snap_install.gpt) is None


class TestExternalDemNeedsHalfPixelShift:
    """Only a ``snap-dem`` older than 14.0.2, or an unknown one, needs the shift."""

    @pytest.mark.parametrize(
        ("version", "needs_shift"),
        [
            ("13.0.0", True),
            ("13.0.2", True),
            ("13.0.3", True),
            ("14.0.0", True),
            ("14.0.1", True),
            ("14.0.2", False),
            ("14.1.0", False),
            ("15.0.0", False),
        ],
    )
    def test_decides_by_the_module_version(
        self,
        snap_install: _SnapInstall,
        version: str,
        needs_shift: bool,
    ) -> None:
        snap_install.install_module(version)

        assert external_dem_needs_half_pixel_shift(snap_install.bin_dir) is needs_shift

    def test_assumes_the_bug_when_the_version_is_unknown(
        self,
        snap_install: _SnapInstall,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="s1bursts.snap_client"):
            assert external_dem_needs_half_pixel_shift(snap_install.bin_dir) is True

        assert any("snap-dem" in record.message for record in caplog.records)

    def test_assumes_the_bug_when_no_installation_is_found(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SNAP_BIN", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path))

        assert external_dem_needs_half_pixel_shift() is True

    def test_raises_for_an_explicit_directory_without_gpt(self, tmp_path: Path) -> None:
        """A typo in `bin_dir` must not silently pick the shifted behaviour."""
        empty = tmp_path / "no-gpt-here"
        empty.mkdir()

        with pytest.raises(ConfigurationError, match="No `gpt` executable"):
            external_dem_needs_half_pixel_shift(empty)

    def test_installation_defaults_to_snap_bin(
        self,
        snap_install: _SnapInstall,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        snap_install.install_module("14.0.2")
        monkeypatch.setenv("SNAP_BIN", str(snap_install.bin_dir))
        monkeypatch.chdir(snap_install.root)

        assert external_dem_needs_half_pixel_shift() is False


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
        assert metadata.band_names == ["gamma0", "scattering_area"]
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
        cast("MagicMock", client.manager).product_exists.return_value = True

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

    def test_write_failure_returns_error_result(self, client: SNAPClient) -> None:
        burst = _make_burst("2022-01-01")

        with (
            patch.object(client, "_run_gpt"),
            patch.object(
                client,
                "_finalize_product",
                side_effect=RasterWriteError("write failed"),
            ),
        ):
            result = client.create_backscatter(
                burst,
                crs=self.CRS_UTM32,
                config=BackscatterConfig(),
            )

        assert isinstance(result, ErrorResult)
        assert isinstance(result.error, RasterWriteError)

    def test_unrelated_runtime_error_propagates(self, client: SNAPClient) -> None:
        """Only a failed product write becomes a per-burst error."""
        with (
            patch.object(client, "_run_gpt"),
            patch.object(client, "_finalize_product", side_effect=RuntimeError("bug")),
            pytest.raises(RuntimeError, match="bug"),
        ):
            client.create_backscatter(
                _make_burst("2022-01-01"),
                crs=self.CRS_UTM32,
                config=BackscatterConfig(),
            )


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
                band_names=["gamma0"],
                units=["linear power"],
                band_tags=[{"LAYER_NAME": "gamma0"}],
            ),
        )

        with rasterio.open(product) as ds:
            assert ds.nodata is not None
            assert np.isnan(ds.nodata)
            assert ds.descriptions == ("gamma0",)
            assert ds.tags()["PRODUCT_TYPE"] == "RTC"
            assert ds.tags()["BACKSCATTER_CONVENTION"] == "Power"
            assert ds.tags(1) == {"LAYER_NAME": "gamma0"}
            assert ds.units == ("linear power",)
            assert np.array_equal(ds.read(1), data, equal_nan=True)

    def test_band_count_mismatch_raises_before_writing(
        self,
        client: SNAPClient,
        tmp_path: Path,
    ) -> None:
        """Metadata that doesn't fit the graph's bands is a bug, not a per-burst failure."""
        staging = tmp_path / "staging.tif"
        product = tmp_path / "final.tif"
        self._write_staging(staging, np.ones((1, 2, 2), dtype=np.float32))

        with pytest.raises(ValueError, match="band"):
            client._finalize_product(
                staging,
                product,
                metadata=RasterMetadata(
                    tags={},
                    band_names=["gamma0", "scattering_area"],
                    units=["linear power", "m2"],
                ),
            )

        assert not product.exists()

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
            metadata=RasterMetadata(tags={}, band_names=["ifg_real", "ifg_imag"], units=["", ""]),
            normalize_nodata=True,
        )

        with rasterio.open(product) as ds:
            assert np.isnan(ds.read(1)[:, 0]).all()
            assert np.isnan(ds.read(2)[:, 0]).all()
            assert ds.read(1)[0, 1] == 0.0
            assert ds.read(2)[0, 1] == 4.0

    def test_keeps_both_markers_when_normalization_is_off(
        self,
        client: SNAPClient,
        tmp_path: Path,
    ) -> None:
        """Without the band maths, neither marker is unambiguous, so neither is touched."""
        staging = tmp_path / "staging.tif"
        product = tmp_path / "final.tif"
        data = np.array([[[0.0, NODATA_SENTINEL], [1.0, 2.0]]], dtype=np.float32)
        self._write_staging(staging, data)

        client._finalize_product(
            staging,
            product,
            metadata=RasterMetadata(tags={}, band_names=["gamma0"], units=["linear power"]),
        )

        with rasterio.open(product) as ds:
            assert np.array_equal(ds.read(1), data[0])

    def test_converts_the_sentinel_to_nodata(self, client: SNAPClient, tmp_path: Path) -> None:
        """The finite marker the graph needs never reaches the product."""
        staging = tmp_path / "staging.tif"
        product = tmp_path / "final.tif"
        data = np.array([[[1.0, NODATA_SENTINEL], [2.0, 3.0]]], dtype=np.float32)
        self._write_staging(staging, data)

        client._finalize_product(
            staging,
            product,
            metadata=RasterMetadata(tags={}, band_names=["gamma0"], units=["linear power"]),
            normalize_nodata=True,
        )

        with rasterio.open(product) as ds:
            band = ds.read(1)
        assert np.isnan(band[0, 1])
        assert np.array_equal(band[[0, 1, 1], [0, 0, 1]], [1.0, 2.0, 3.0])


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
        assert metadata.band_names == ["coherence"]
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


class TestCreateInterferogram:
    CRS_UTM32 = CRS("EPSG:32632")

    def _make_pair(self) -> BurstPair:
        return BurstPair(_make_burst("2022-01-01"), _make_burst("2022-01-13"))

    def test_creates_interferogram_product(self, client: SNAPClient) -> None:
        with (
            patch.object(client, "_run_gpt") as mock_gpt,
            patch.object(client, "_finalize_product") as mock_finalize,
        ):
            result = client.create_interferogram(
                self._make_pair(),
                crs=self.CRS_UTM32,
                config=InterferogramConfig(),
            )

        assert isinstance(result, OkResult)
        assert mock_gpt.call_count == 1
        metadata = mock_finalize.call_args.kwargs["metadata"]
        assert metadata.band_names == [
            "ifg_real",
            "ifg_imag",
            "intensity_primary",
            "intensity_secondary",
        ]

    def test_write_failure_returns_error_result(self, client: SNAPClient) -> None:
        pair = self._make_pair()

        with (
            patch.object(client, "_run_gpt"),
            patch.object(
                client,
                "_finalize_product",
                side_effect=RasterWriteError("write failed"),
            ),
        ):
            result = client.create_interferogram(
                pair,
                crs=self.CRS_UTM32,
                config=InterferogramConfig(),
            )

        assert isinstance(result, ErrorResult)
        assert isinstance(result.error, RasterWriteError)
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

    def _config(self, snap_bin: Path, **kwargs: Any) -> SNAPConfig:
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
        cast("MagicMock", client.manager).get_slc_path.return_value = local

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
        cast("MagicMock", client.manager).get_slc_path.return_value = remote
        work = tmp_path / "work"
        work.mkdir()

        staged = client._stage_slc(_make_burst("2022-08-25"), work)

        assert staged == work / "S1_SLC_remote.zip"
        assert staged.read_bytes() == b"remote burst"
