"""Tests for s1bursts.vrt VRT generation."""

from __future__ import annotations

from pathlib import Path

from defusedxml.ElementTree import parse
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from s1bursts.vrt import create_decibel_vrt, create_rgb_vrt

WIDTH, HEIGHT = 6, 4


def make_tiff(path: Path) -> None:
    data = np.random.default_rng(0).random((HEIGHT, WIDTH)).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=WIDTH,
        height=HEIGHT,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=from_origin(500000, 5200000, 10, 10),
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def vv_tiff(tmp_path: Path) -> Path:
    path = tmp_path / "vv.tif"
    make_tiff(path)
    return path


@pytest.fixture
def vh_tiff(tmp_path: Path) -> Path:
    path = tmp_path / "vh.tif"
    make_tiff(path)
    return path


class TestDecibelVrt:
    def test_creates_valid_vrt(self, vv_tiff: Path, tmp_path: Path) -> None:
        vrt_path = tmp_path / "vv_db.vrt"

        create_decibel_vrt(vv_tiff, vrt_path)

        root = parse(vrt_path).getroot()
        assert root.get("rasterXSize") == str(WIDTH)
        assert root.get("rasterYSize") == str(HEIGHT)
        band = root.find("VRTRasterBand")
        assert band is not None
        assert band.findtext("PixelFunctionType") == "dB"
        assert band.find("PixelFunctionArguments").get("fact") == "10"

        # GDAL must be able to open the derived VRT
        with rasterio.open(vrt_path) as src:
            assert src.width == WIDTH
            assert src.crs.to_epsg() == 32632

    def test_amplitude_uses_factor_20(self, vv_tiff: Path, tmp_path: Path) -> None:
        vrt_path = tmp_path / "vv_db.vrt"

        create_decibel_vrt(vv_tiff, vrt_path, from_intensity=False)

        root = parse(vrt_path).getroot()
        assert root.find("VRTRasterBand/PixelFunctionArguments").get("fact") == "20"

    def test_source_referenced_relative(self, vv_tiff: Path, tmp_path: Path) -> None:
        vrt_path = tmp_path / "vv_db.vrt"

        create_decibel_vrt(vv_tiff, vrt_path)

        root = parse(vrt_path).getroot()
        source = root.find("VRTRasterBand/SimpleSource/SourceFilename")
        assert source.text == "vv.tif"
        assert source.get("relativeToVRT") == "1"


class TestRgbVrt:
    def test_linear_scale_band_layout(
        self,
        vv_tiff: Path,
        vh_tiff: Path,
        tmp_path: Path,
    ) -> None:
        vrt_path = tmp_path / "rgb.vrt"

        create_rgb_vrt(vv_tiff, vh_tiff, vrt_path)

        root = parse(vrt_path).getroot()
        bands = root.findall("VRTRasterBand")
        assert [b.findtext("Description") for b in bands] == ["VV", "VH", "VV/VH"]
        assert bands[2].findtext("PixelFunctionType") == "div"

        with rasterio.open(vrt_path) as src:
            assert src.count == 3

    def test_decibel_scale_band_layout(
        self,
        vv_tiff: Path,
        vh_tiff: Path,
        tmp_path: Path,
    ) -> None:
        vrt_path = tmp_path / "rgb.vrt"

        create_rgb_vrt(vv_tiff, vh_tiff, vrt_path, decibel_scale=True)

        root = parse(vrt_path).getroot()
        bands = root.findall("VRTRasterBand")
        assert [b.findtext("Description") for b in bands] == ["VV_dB", "VH_dB", "VV_dB-VH_dB"]
        assert bands[2].findtext("PixelFunctionType") == "diff"
