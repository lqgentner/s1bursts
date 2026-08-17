"""Differential test of the raster extraction against the reference tool.

Every XML edit in the port is checked against the reference script's own
expressions in ``test_esa_reference_parity.py``, but one step has no such
counterpart: the reference calls ``gdal_translate`` on a VRT connection string,
where the port reads a window with rasterio and attaches the GCPs itself. Those
are different implementations of the same operation, so this runs the
reference's exact invocation over the synthetic SAFE product and requires the
two rasters to agree.

Offline and fast — no S3, no network. Skips where the GDAL CLI is absent; the
container the integration workflow uses carries it.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING
import warnings

import numpy as np
import pytest
import rasterio
from rasterio.errors import NotGeoreferencedWarning

from s1bursts.burst_extractor import _extract_tiff, _parse_xml, _resolve_burst
from tests.fixtures.safe_product import ProductSpec, build_product

if TYPE_CHECKING:
    from pathlib import Path

    from tests.fixtures.safe_product import SyntheticProduct

pytestmark = pytest.mark.skipif(
    shutil.which("gdal_translate") is None,
    reason="gdal_translate not on PATH",
)

# Burst to extract; interior, so a wrong window or an unshifted GCP row shows up.
BURST_INDEX = 2


def _geolocation_points(
    annotation_root: object,
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
        for p in annotation_root.findall(  # type: ignore[attr-defined]
            "geolocationGrid/geolocationGridPointList/geolocationGridPoint",
        )
    ]


@pytest.fixture(scope="module")
def extraction(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Extract one burst twice: once with the port, once with gdal_translate."""
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
    _extract_tiff(source, ours, geoloc_pts=points, meta=meta)

    # The reference's invocation, verbatim: one `gcp=` per grid point with the
    # line already shifted, then a `srcwin` over the burst's rows.
    gcps = "".join(
        f"gcp={pixel:g},{line:g},{lon:g},{lat:g},{height:g}&"
        for pixel, line, lon, lat, height in points
    )
    srcwin = f"srcwin=0,{meta.starting_line},{meta.number_of_samples},{meta.number_of_lines}"
    reference = tmp_path / "reference.tiff"
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "gdal_translate",
            "-of",
            "GTiff",
            "-q",
            f"vrt://{source}?{gcps}{srcwin}",
            str(reference),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

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


class TestAgainstGdalTranslate:
    def test_pixels_are_identical(self, extraction: dict[str, object]) -> None:
        """The windowed rasterio read must return exactly what gdal_translate's srcwin does."""
        ours, _, _ = _read(extraction["ours"])  # type: ignore[arg-type]
        reference, _, _ = _read(extraction["reference"])  # type: ignore[arg-type]

        assert ours.shape == reference.shape
        assert np.array_equal(ours, reference)

    def test_dimensions_match_the_burst(self, extraction: dict[str, object]) -> None:
        ours, _, _ = _read(extraction["ours"])  # type: ignore[arg-type]
        meta = extraction["meta"]

        assert ours.shape == (meta.number_of_lines, meta.number_of_samples)  # type: ignore[attr-defined]

    def test_gcps_are_identical(self, extraction: dict[str, object]) -> None:
        """Same points, same order, same shifted rows."""
        _, ours, _ = _read(extraction["ours"])  # type: ignore[arg-type]
        _, reference, _ = _read(extraction["reference"])  # type: ignore[arg-type]

        assert ours == pytest.approx(reference)

    def test_gcps_match_the_shifted_grid(self, extraction: dict[str, object]) -> None:
        """Guards the previous test: both could agree on a wrong set of points."""
        _, ours, _ = _read(extraction["ours"])  # type: ignore[arg-type]
        expected = extraction["points"]

        assert ours == pytest.approx(expected)

    def test_sample_dtype_is_preserved(self, extraction: dict[str, object]) -> None:
        """SLC pixels are CInt16; neither path may promote them."""
        _, _, ours = _read(extraction["ours"])  # type: ignore[arg-type]
        _, _, reference = _read(extraction["reference"])  # type: ignore[arg-type]

        assert ours == reference == "complex_int16"

    def test_port_attaches_a_gcp_crs_the_reference_leaves_unset(
        self,
        extraction: dict[str, object],
    ) -> None:
        """A deliberate divergence from the script, matching what CDSE ships.

        The reference's VRT connection string has no option for a GCP SRS, so
        `gdal_translate` writes the points with no projection. CDSE's own burst
        products declare EPSG:4326, and so does the port.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            with rasterio.open(extraction["ours"]) as src:  # type: ignore[arg-type]
                ours_crs = src.gcps[1]
            with rasterio.open(extraction["reference"]) as src:  # type: ignore[arg-type]
                reference_crs = src.gcps[1]

        assert ours_crs is not None
        assert ours_crs.to_epsg() == 4326
        assert reference_crs is None
