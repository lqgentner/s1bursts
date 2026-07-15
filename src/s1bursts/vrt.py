"""Creation of GDAL Virtual Datasets (VRTs) for decibel-scaling and false color visualization."""

from pathlib import Path

from deepfreezer.geotiff_utils import rasterio_open
from upath import UPath

type AnyPath = str | Path | UPath


def _get_raster_metadata(path: AnyPath) -> tuple[int, int, str, str]:
    """Get GDAL-compatible raster metadata.

    Parameters
    ----------
    path : str | Path | UPath
        Path to the raster file.

    Returns
    -------
    tuple of (int, int, str, str)
        A tuple of (x_size, y_size, geo_transform, projection) where
        geo_transform is a comma-separated GDAL affine transform string and
        projection is a WKT CRS string.
    """
    with rasterio_open(path) as ds:
        x_size = ds.width
        y_size = ds.height
        geo_transform = ", ".join(str(v) for v in ds.transform.to_gdal())
        projection = ds.crs.to_wkt()
    return x_size, y_size, geo_transform, projection


def create_decibel_vrt(
    src_path: AnyPath,
    output_vrt: AnyPath,
    *,
    from_intensity: bool = True,
) -> None:
    """Create a VRT file that applies a decibel conversion pixel function.

    Parameters
    ----------
    src_path : str | Path | UPath
        Path to the source raster file (linear scale).
    output_vrt : str | Path | UPath
        Path where the output VRT file will be written.
    from_intensity : bool, optional
        If True, uses a factor of 10 (intensity: 10*log10).
        If False, uses a factor of 20 (amplitude: 20*log10).
    """
    src_path = UPath(src_path)
    x_size, y_size, geo_transform, projection = _get_raster_metadata(src_path)
    fact = 10 if from_intensity else 20
    vrt_content = f"""<VRTDataset rasterXSize="{x_size}" rasterYSize="{y_size}">
  <SRS>{projection}</SRS>
  <GeoTransform>{geo_transform}</GeoTransform>

  <VRTRasterBand dataType="Float32" band="1" subClass="VRTDerivedRasterBand">
    <ColorInterp>Gray</ColorInterp>
    <Description>dB</Description>
    <NoDataValue>nan</NoDataValue>
    <PixelFunctionType>dB</PixelFunctionType>
    <PixelFunctionArguments fact="{fact}" />
    <SourceTransferType>Float32</SourceTransferType>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{src_path.name}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""

    UPath(output_vrt).write_text(vrt_content)


def create_rgb_vrt(
    vv_path: AnyPath,
    vh_path: AnyPath,
    output_vrt: AnyPath,
    *,
    decibel_scale: bool = False,
) -> None:
    """Create a 3-band RGB VRT from VV and VH rasters.

    The input scale can be linear or decibel.

    Band assignments:

    - Linear scale: Red=VV, Green=VH, Blue=VV/VH.
    - Decibel scale: Red=VV, Green=VH, Blue=VV-VH.

    Parameters
    ----------
    vv_path : str | Path | UPath
        Path to the VV polarization raster.
    vh_path : str | Path | UPath
        Path to the VH polarization raster.
    output_vrt : str | Path | UPath
        Path where the output VRT file will be written.
    decibel_scale : bool, optional
        Whether the input rasters are in decibel scale. Selects the band
        expressions and descriptions accordingly.
    """
    vv_path, vh_path = UPath(vv_path), UPath(vh_path)
    x_size, y_size, geo_transform, projection = _get_raster_metadata(vv_path)
    band_desc = ["VV_dB", "VH_dB", "VV_dB-VH_dB"] if decibel_scale else ["VV", "VH", "VV/VH"]
    operator = "diff" if decibel_scale else "div"

    vrt_content = f"""<VRTDataset rasterXSize="{x_size}" rasterYSize="{y_size}">
  <SRS>{projection}</SRS>
  <GeoTransform>{geo_transform}</GeoTransform>

  <VRTRasterBand dataType="Float32" band="1">
    <ColorInterp>Red</ColorInterp>
    <Description>{band_desc[0]}</Description>
    <NoDataValue>nan</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{vv_path.name}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>

  <VRTRasterBand dataType="Float32" band="2">
    <ColorInterp>Green</ColorInterp>
    <Description>{band_desc[1]}</Description>
    <NoDataValue>nan</NoDataValue>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{vh_path.name}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>

  <VRTRasterBand dataType="Float32" band="3" subClass="VRTDerivedRasterBand">
    <ColorInterp>Blue</ColorInterp>
    <Description>{band_desc[2]}</Description>
    <NoDataValue>nan</NoDataValue>
    <PixelFunctionType>{operator}</PixelFunctionType>
    <SourceTransferType>Float32</SourceTransferType>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{vv_path.name}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{vh_path.name}</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>"""

    UPath(output_vrt).write_text(vrt_content)
