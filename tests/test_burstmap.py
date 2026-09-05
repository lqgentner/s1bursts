"""Tests for the burst-ID map metadata, versioned cache paths, and opt-in loading."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch
from zipfile import ZipFile

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from s1bursts import burstmap

if TYPE_CHECKING:
    from pathlib import Path


def _fake_indexed_map() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "burst_id": [30346],
            "subswath_name": ["IW1"],
            "relative_orbit_number": [15],
            "orbit_pass": ["ASCENDING"],
            "geometry": [box(7, 46, 8, 47)],
        },
    )


class TestMetadata:
    def test_pins_sha256_and_versioned_paths(self, tmp_path: Path) -> None:
        dataset = burstmap.Sentinel1Bursts(cache_dir=tmp_path, mode="IW")
        metadata = dataset.metadata

        assert metadata.sha256 == burstmap._BURSTMAP_SHA256
        assert metadata.version == burstmap._BURSTMAP_VERSION
        assert metadata.license is None  # upstream states no explicit license
        assert metadata.license_url == burstmap._COPERNICUS_TERMS_URL
        # The release version is baked into both cache paths.
        assert burstmap._BURSTMAP_VERSION in dataset.raw_path.name
        assert burstmap._BURSTMAP_VERSION in dataset.processed_path.name

    def test_processed_path_encodes_mode(self) -> None:
        assert "s1-iw-" in burstmap.Sentinel1Bursts(mode="IW").processed_path.name
        assert "s1-ew-" in burstmap.Sentinel1Bursts(mode="EW").processed_path.name


class TestOptInLoading:
    @pytest.fixture(autouse=True)
    def _clear_lut_caches(self) -> None:
        for factory in (
            burstmap.create_s1_orbit_number_lut,
            burstmap.create_s1_pass_direction_lut,
            burstmap.create_s1_geometry_lut,
        ):
            factory.cache_clear()

    def test_missing_map_raises_actionable_error(self, tmp_path: Path) -> None:
        with (
            patch.object(
                burstmap.Sentinel1Bursts,
                "get_data",
                side_effect=FileNotFoundError("Dataset not found."),
            ),
            pytest.raises(FileNotFoundError, match="download_burst_map"),
        ):
            burstmap.create_s1_orbit_number_lut(cache_dir=tmp_path, mode="IW")

    def test_factories_do_not_auto_download(self, tmp_path: Path) -> None:
        """The LUT factories must load without triggering a download."""
        with patch.object(burstmap.Sentinel1Bursts, "get_data") as get_data:
            get_data.return_value = _fake_indexed_map()
            burstmap.create_s1_orbit_number_lut(cache_dir=tmp_path, mode="IW")

        get_data.assert_called_once_with(download=False)

    def test_pass_direction_is_title_case(self, tmp_path: Path) -> None:
        """The map spells the directions upper case; the package title case."""
        with patch.object(burstmap.Sentinel1Bursts, "get_data") as get_data:
            get_data.return_value = _fake_indexed_map()
            lut = burstmap.create_s1_pass_direction_lut(cache_dir=tmp_path, mode="IW")

        assert lut == {(30346, "IW1"): "Ascending"}


class TestDownloadBurstMap:
    def test_triggers_download_and_returns_processed_path(self, tmp_path: Path) -> None:
        with patch.object(burstmap.Sentinel1Bursts, "get_data") as get_data:
            path = burstmap.download_burst_map(cache_dir=tmp_path, mode="IW")

        get_data.assert_called_once_with(download=True)
        assert burstmap._BURSTMAP_VERSION in path.name


class TestModeValidation:
    @pytest.mark.parametrize("mode", ["SM", "iw", "", "IW1"])
    def test_rejects_an_unsupported_mode(self, tmp_path: Path, mode: str) -> None:
        with pytest.raises(ValueError, match="Invalid product"):
            # pyrefly: ignore [bad-argument-type]
            burstmap.Sentinel1Bursts(cache_dir=tmp_path, mode=mode)

    @pytest.mark.parametrize("mode", ["IW", "EW"])
    def test_accepts_both_acquisition_modes(self, tmp_path: Path, mode: str) -> None:
        # `mode` is parametrized as a plain str.
        # pyrefly: ignore [bad-argument-type]
        dataset = burstmap.Sentinel1Bursts(cache_dir=tmp_path, mode=mode)

        assert dataset.mode == mode
        assert mode.lower() in dataset.processed_path.name


class TestPrepare:
    """Extraction of the SpatiaLite database out of the SAR-MPC archive."""

    def test_reports_an_archive_without_a_database(self, tmp_path: Path) -> None:
        """A truncated or restructured download must fail loudly, not half-succeed."""
        dataset = burstmap.Sentinel1Bursts(cache_dir=tmp_path, mode="IW")
        dataset.raw_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(dataset.raw_path, "w") as archive:
            archive.writestr("S1_burstid_20220530/readme.txt", "no database here")

        with pytest.raises(FileNotFoundError, match="Couldn't locate SpatiaLite database"):
            dataset._prepare()

    def test_ignores_a_database_belonging_to_the_other_mode(self, tmp_path: Path) -> None:
        """An EW dataset must not silently fall back to the IW database."""
        dataset = burstmap.Sentinel1Bursts(cache_dir=tmp_path, mode="EW")
        dataset.raw_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(dataset.raw_path, "w") as archive:
            archive.writestr("S1_burstid/IW/sqlite/burstid.sqlite3", "not the EW map")

        with pytest.raises(FileNotFoundError, match="Couldn't locate SpatiaLite database"):
            dataset._prepare()

    def test_reads_the_database_matching_the_requested_mode(self, tmp_path: Path) -> None:
        """The selected member must be the mode's own SpatiaLite file."""
        dataset = burstmap.Sentinel1Bursts(cache_dir=tmp_path, mode="EW")
        dataset.raw_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(dataset.raw_path, "w") as archive:
            archive.writestr("S1_burstid/IW/sqlite/iw.sqlite3", "iw")
            archive.writestr("S1_burstid/EW/sqlite/ew.sqlite3", "ew")

        with (
            patch("s1bursts.burstmap.gpd.read_file") as read_file,
            patch(
                "s1bursts.burstmap.save_and_read_parquet",
            ),
        ):
            read_file.return_value = gpd.GeoDataFrame(
                {
                    "burst_id": [1],
                    "relative_orbit_number": [15],
                    "subswath_name": ["EW1"],
                    "orbit_pass": ["ASCENDING"],
                    # The real map ships 3D polygons with Z=0, which the
                    # preparation pipeline flattens.
                    "geometry": [
                        Polygon([(7, 46, 0), (8, 46, 0), (8, 47, 0), (7, 47, 0), (7, 46, 0)]),
                    ],
                },
            )
            dataset._prepare()

        assert read_file.call_args.args[0].read() == b"ew"
