"""Cross-check raster extraction against GDAL.

The extractor reads a window with rasterio and attaches GCPs itself. This test
has GDAL's VRT driver perform the equivalent operation over the synthetic SAFE
product and requires the two independent implementations to agree.

Offline and fast — no S3, no network, and no GDAL CLI: the reference is GDAL's
own ``vrt://`` handling, run through the GDAL bundled with rasterio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import rasterio.shutil
from upath import UPath

from s1bursts.burst_extractor import _extract_tiff, _parse_xml, _resolve_burst
from tests.fixtures.safe_product import ProductSpec, build_product

if TYPE_CHECKING:
    from pathlib import Path

    from tests.fixtures.safe_product import SyntheticProduct

# Burst to extract; interior, so a wrong window or an unshifted GCP row shows up.
BURST_INDEX = 2


def _geolocation_points(
    annotation_root: Any,
    starting_line: int,
) -> list[tuple[float, float, float, float, float]]:
    """Return grid points as (pixel, shifted line, lon, lat, height), as the port does."""
    return [
        (
            float(p.findtext("pixel")),
            float(p.findtext("line")) - starting_line,
            float(p.findtext("longitude")),
            float(p.findtext("latitude")),
            float(p.findtext("height")),
        )
        for p in annotation_root.findall(
            "geolocationGrid/geolocationGridPointList/geolocationGridPoint",
        )
    ]


@pytest.fixture(scope="module")
def extraction(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Extract one burst twice: once with the port, once with GDAL's VRT driver."""
    tmp_path = tmp_path_factory.mktemp("gdal-diff")
    product: SyntheticProduct = build_product(tmp_path, ProductSpec())
    spec = product.spec

    annotation = _parse_xml((product.root / "annotation" / product.annotation_name).read_bytes())
    manifest = _parse_xml((product.root / "manifest.safe").read_bytes())
    meta = _resolve_burst(
        annotation,
        manifest,
        relative_burst_id=spec.burst_ids[BURST_INDEX],
        sub_swath=spec.swath,
        annotation_filename=product.annotation_name,
    )
    points = _geolocation_points(annotation, meta.starting_line)
    source = product.root / "measurement" / f"{product.annotation_name[:-4]}.tiff"

    ours = tmp_path / "ours.tiff"
    _extract_tiff(UPath(source), ours, geoloc_pts=points, meta=meta)

    # One `gcp=` per grid point with the line already shifted, then a `srcwin`
    # over the burst's rows.
    gcps = "".join(
        f"gcp={pixel:g},{line:g},{lon:g},{lat:g},{height:g}&"
        for pixel, line, lon, lat, height in points
    )
    srcwin = f"srcwin=0,{meta.starting_line},{meta.number_of_samples},{meta.number_of_lines}"
    reference = tmp_path / "reference.tiff"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        rasterio.shutil.copy(f"vrt://{source}?{gcps}{srcwin}", reference, driver="GTiff")

    return {"ours": ours, "reference": reference, "meta": meta, "points": points}


def _read(path: Path) -> tuple[np.ndarray, list[tuple[float, float, float, float, float]], str]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            gcps, _crs = src.gcps
            return (
                src.read(1),
                [(g.col, g.row, g.x, g.y, g.z) for g in gcps],
                src.dtypes[0],
            )


class TestAgainstGdalVrt:
    def test_pixels_are_identical(self, extraction: dict[str, Any]) -> None:
        """The windowed rasterio read must return exactly what the VRT's srcwin does."""
        ours, _, _ = _read(extraction["ours"])
        reference, _, _ = _read(extraction["reference"])

        assert ours.shape == reference.shape
        assert np.array_equal(ours, reference)

    def test_dimensions_match_the_burst(self, extraction: dict[str, Any]) -> None:
        ours, _, _ = _read(extraction["ours"])
        meta = extraction["meta"]

        assert ours.shape == (meta.number_of_lines, meta.number_of_samples)

    def test_gcps_are_identical(self, extraction: dict[str, Any]) -> None:
        """Same points, same order, same shifted rows."""
        _, ours, _ = _read(extraction["ours"])
        _, reference, _ = _read(extraction["reference"])

        assert ours == reference

    def test_gcps_match_the_shifted_grid(self, extraction: dict[str, Any]) -> None:
        """Guards the previous test: both could agree on a wrong set of points."""
        _, ours, _ = _read(extraction["ours"])
        expected = extraction["points"]

        assert ours == expected

    def test_sample_dtype_is_preserved(self, extraction: dict[str, Any]) -> None:
        """SLC pixels are CInt16; neither path may promote them."""
        _, _, ours = _read(extraction["ours"])
        _, _, reference = _read(extraction["reference"])

        assert ours == reference == "complex_int16"

    def test_port_attaches_a_gcp_crs_that_gdal_leaves_unset(
        self,
        extraction: dict[str, Any],
    ) -> None:
        """The port declares the GCP CRS used by CDSE burst products.

        The VRT connection string has no GCP SRS option, so GDAL writes the
        points without a projection. The port declares EPSG:4326.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            with rasterio.open(extraction["ours"]) as src:
                ours_crs = src.gcps[1]
            with rasterio.open(extraction["reference"]) as src:
                reference_crs = src.gcps[1]

        assert ours_crs is not None
        assert ours_crs.to_epsg() == 4326
        assert reference_crs is None
