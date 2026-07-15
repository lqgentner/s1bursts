"""Tests for s1bursts.dimap BEAM-DIMAP I/O."""

from __future__ import annotations

from pathlib import Path

from defusedxml.ElementTree import parse
import numpy as np
import pytest

from s1bursts.dimap import BeamDimapDataset, _write_envi_hdr

NROWS, NCOLS = 4, 5


def _band_xml(index: int, name: str, dataset_name: str) -> tuple[str, str, str]:
    """Return (data_file, spectral_band_info, crs_geoposition) XML snippets."""
    data_file = f"""
        <Data_File>
            <DATA_FILE_PATH href="{dataset_name}.data/{name}.hdr" />
            <BAND_INDEX>{index}</BAND_INDEX>
        </Data_File>"""
    band_info = f"""
        <Spectral_Band_Info>
            <BAND_INDEX>{index}</BAND_INDEX>
            <BAND_NAME>{name}</BAND_NAME>
            <BAND_RASTER_WIDTH>{NCOLS}</BAND_RASTER_WIDTH>
            <BAND_RASTER_HEIGHT>{NROWS}</BAND_RASTER_HEIGHT>
            <DATA_TYPE>float32</DATA_TYPE>
        </Spectral_Band_Info>"""
    crs_geo = f"""
    <Coordinate_Reference_System>
        <WKT>GEOGCS["WGS84"]</WKT>
    </Coordinate_Reference_System>
    <Geoposition>
        <BAND_INDEX>{index}</BAND_INDEX>
        <Geoposition_Points>
            <TIE_POINT_GRID_NAME_LAT>latitude</TIE_POINT_GRID_NAME_LAT>
            <TIE_POINT_GRID_NAME_LON>longitude</TIE_POINT_GRID_NAME_LON>
        </Geoposition_Points>
    </Geoposition>"""
    return data_file, band_info, crs_geo


def make_dimap(tmp_path: Path, bands: dict[str, np.ndarray]) -> Path:
    """Write a minimal BEAM-DIMAP dataset with the given float32 bands."""
    dataset_name = "test_product"
    dim_path = tmp_path / f"{dataset_name}.dim"
    data_dir = tmp_path / f"{dataset_name}.data"
    data_dir.mkdir()

    data_files, band_infos, crs_geos = [], [], []
    for index, (name, data) in enumerate(bands.items()):
        data.astype(">f4").tofile(data_dir / f"{name}.img")
        _write_envi_hdr(data_dir / f"{name}.hdr", name=name, nrows=NROWS, ncols=NCOLS, unit="")
        df, bi, cg = _band_xml(index, name, dataset_name)
        data_files.append(df)
        band_infos.append(bi)
        crs_geos.append(cg)

    dim_path.write_text(f"""<?xml version="1.0" encoding="ISO-8859-1"?>
<Dimap_Document name="{dataset_name}.dim">
    <Raster_Dimensions>
        <NCOLS>{NCOLS}</NCOLS>
        <NROWS>{NROWS}</NROWS>
        <NBANDS>{len(bands)}</NBANDS>
    </Raster_Dimensions>
    <Data_Access>{"".join(data_files)}
    </Data_Access>{"".join(crs_geos)}
    <Image_Interpretation>{"".join(band_infos)}
    </Image_Interpretation>
</Dimap_Document>
""")
    return dim_path


@pytest.fixture
def band_a() -> np.ndarray:
    return np.arange(NROWS * NCOLS, dtype=np.float32).reshape(NROWS, NCOLS)


@pytest.fixture
def dim_path(tmp_path: Path, band_a: np.ndarray) -> Path:
    return make_dimap(tmp_path, {"band_a": band_a, "band_b": band_a * 2})


class TestValidation:
    def test_wrong_suffix_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match=r"must end with '\.dim'"):
            BeamDimapDataset(tmp_path / "product.tif")

    def test_missing_dim_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="BEAM-DIMAP file not found"):
            BeamDimapDataset(tmp_path / "missing.dim")

    def test_missing_data_dir_raises(self, tmp_path: Path) -> None:
        dim = tmp_path / "product.dim"
        dim.write_text("<Dimap_Document/>")

        with pytest.raises(FileNotFoundError, match="data folder"):
            BeamDimapDataset(dim)


class TestReading:
    def test_band_names(self, dim_path: Path) -> None:
        assert BeamDimapDataset(dim_path).band_names == ["band_a", "band_b"]

    def test_shape(self, dim_path: Path) -> None:
        assert BeamDimapDataset(dim_path).shape == (NROWS, NCOLS)

    def test_read_band(self, dim_path: Path, band_a: np.ndarray) -> None:
        ds = BeamDimapDataset(dim_path)

        np.testing.assert_array_equal(ds.read_band("band_a"), band_a)
        np.testing.assert_array_equal(ds.read_band("band_b"), band_a * 2)

    def test_read_unknown_band_raises(self, dim_path: Path) -> None:
        with pytest.raises(KeyError, match="Band 'nope' not found"):
            BeamDimapDataset(dim_path).read_band("nope")


class TestWriting:
    def test_write_band_roundtrip(self, dim_path: Path, band_a: np.ndarray) -> None:
        ds = BeamDimapDataset(dim_path)
        new_data = band_a + 100

        ds.write_band("band_a", new_data)

        np.testing.assert_array_equal(ds.read_band("band_a"), new_data)

    def test_write_band_shape_mismatch_raises(self, dim_path: Path) -> None:
        ds = BeamDimapDataset(dim_path)

        with pytest.raises(ValueError, match="Shape mismatch"):
            ds.write_band("band_a", np.zeros((2, 2), dtype=np.float32))

    def test_write_unknown_band_raises(self, dim_path: Path, band_a: np.ndarray) -> None:
        with pytest.raises(KeyError, match="not found"):
            BeamDimapDataset(dim_path).write_band("nope", band_a)


class TestAddBand:
    def test_add_band_roundtrip(self, dim_path: Path, band_a: np.ndarray) -> None:
        ds = BeamDimapDataset(dim_path)

        ds.add_band("coh_i", band_a * 3, unit="coherence")

        assert "coh_i" in ds.band_names
        np.testing.assert_array_equal(ds.read_band("coh_i"), band_a * 3)

        # Re-open from disk: the .dim XML must register the new band
        reloaded = BeamDimapDataset(dim_path)
        assert "coh_i" in reloaded.band_names
        np.testing.assert_array_equal(reloaded.read_band("coh_i"), band_a * 3)

    def test_add_band_updates_nbands(self, dim_path: Path, band_a: np.ndarray) -> None:
        BeamDimapDataset(dim_path).add_band("new_band", band_a)

        root = parse(dim_path).getroot()
        assert root.findtext(".//Raster_Dimensions/NBANDS") == "3"

    def test_add_existing_band_raises(self, dim_path: Path, band_a: np.ndarray) -> None:
        with pytest.raises(ValueError, match="already exists"):
            BeamDimapDataset(dim_path).add_band("band_a", band_a)

    def test_add_band_shape_mismatch_raises(self, dim_path: Path) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            BeamDimapDataset(dim_path).add_band("new_band", np.zeros((2, 2)))


class TestDeleteBand:
    def test_delete_band(self, dim_path: Path) -> None:
        ds = BeamDimapDataset(dim_path)

        ds.delete_band("band_a")

        assert ds.band_names == ["band_b"]
        data_dir = dim_path.with_suffix(".data")
        assert not (data_dir / "band_a.img").exists()
        assert not (data_dir / "band_a.hdr").exists()

        reloaded = BeamDimapDataset(dim_path)
        assert reloaded.band_names == ["band_b"]

    def test_delete_band_renumbers_indices(self, dim_path: Path) -> None:
        BeamDimapDataset(dim_path).delete_band("band_a")

        root = parse(dim_path).getroot()
        assert root.findtext(".//Raster_Dimensions/NBANDS") == "1"
        remaining = root.findall(".//Image_Interpretation/Spectral_Band_Info")
        assert len(remaining) == 1
        assert remaining[0].findtext("BAND_INDEX") == "0"

    def test_delete_unknown_band_raises(self, dim_path: Path) -> None:
        with pytest.raises(KeyError, match="not found"):
            BeamDimapDataset(dim_path).delete_band("nope")

    def test_add_after_delete_keeps_dataset_consistent(
        self,
        dim_path: Path,
        band_a: np.ndarray,
    ) -> None:
        ds = BeamDimapDataset(dim_path)
        existing = ds.band_names

        ds.add_band("coh_i", band_a, unit="coherence")
        ds.add_band("coh_q", band_a, unit="coherence")
        for band in existing:
            ds.delete_band(band)

        reloaded = BeamDimapDataset(dim_path)
        assert reloaded.band_names == ["coh_i", "coh_q"]
        np.testing.assert_array_equal(reloaded.read_band("coh_q"), band_a)


class TestCleanup:
    def test_cleanup_removes_files(self, dim_path: Path) -> None:
        ds = BeamDimapDataset(dim_path)

        ds.cleanup()

        assert not dim_path.exists()
        assert not dim_path.with_suffix(".data").exists()

    def test_cleanup_is_idempotent(self, dim_path: Path) -> None:
        ds = BeamDimapDataset(dim_path)
        ds.cleanup()
        ds.cleanup()
