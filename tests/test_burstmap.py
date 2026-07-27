"""Tests for the burst-ID map metadata, versioned cache paths, and opt-in loading."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely.geometry import box

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
            burstmap.create_s1_orbit_direction_lut,
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


class TestDownloadBurstMap:
    def test_triggers_download_and_returns_processed_path(self, tmp_path: Path) -> None:
        with patch.object(burstmap.Sentinel1Bursts, "get_data") as get_data:
            path = burstmap.download_burst_map(cache_dir=tmp_path, mode="IW")

        get_data.assert_called_once_with(download=True)
        assert burstmap._BURSTMAP_VERSION in path.name
