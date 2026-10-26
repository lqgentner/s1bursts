"""Measure where SNAP samples an external DEM, and which geoid it applies.

``prepare_dem_for_snap`` exists because SNAP reads an external DEM half a pixel
south-east of where its georeferencing places it, and because the only geoid
SNAP knows is EGM96. Both are properties of SNAP, not of this package, so both
have to be measured against the installed ``gpt`` rather than asserted from a
golden file — the day either is fixed, the compensation becomes the error. The
half-pixel offset was fixed in the ``snap-dem`` 14.0.2 module update, which
``external_dem_needs_half_pixel_shift`` keys on; the probe is what says whether
that is still true of the installation at hand.

``AddElevation`` writes out what ``ElevationModel.getElevation`` returns on a
grid of our choosing, which is the call Terrain-Flattening and
Terrain-Correction make, so it isolates DEM handling from any SAR processing.
The probe is synthetic throughout: a target grid and three tiny DEMs written
here, no burst, no download, one ``gpt`` invocation.

Each DEM is a plane rising by a fixed step per pixel along one axis, so
bilinear interpolation of it is exact and the height SNAP reports states the
index it sampled at. Differencing against a run over the flat DEM removes
whatever SNAP adds per target position — the geoid it applies to an external
DEM, which ``AddElevation`` offers no way to switch off — leaving the sampled
index alone. That same difference, read the other way round, is the undulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.transform import from_origin

from s1bursts.dem import prepare_dem_for_snap

if TYPE_CHECKING:
    from collections.abc import Mapping

# One arcsecond, the posting of the Copernicus DEM this package is built around.
DEM_RES = 1.0 / 3600.0
# Bernese Alps, where EGM96 and EGM2008 sit about 1.2 m apart: far enough for a
# geoid measurement to say which of the two SNAP used.
DEM_ORIGIN_LON = 7.0
DEM_ORIGIN_LAT = 46.6
DEM_WIDTH = DEM_HEIGHT = 60

# The target grid sits well inside the DEM, so that a half-pixel shift in either
# direction still lands on interior pixels, and is fine enough to be interpolated
# rather than merely nearest-neighboured.
TARGET_MARGIN_PIXELS = 10
TARGET_RES = DEM_RES / 2
TARGET_SIZE = 40
PROBE_CENTRE = (
    DEM_ORIGIN_LON + (TARGET_MARGIN_PIXELS + TARGET_SIZE / 4) * DEM_RES,
    DEM_ORIGIN_LAT - (TARGET_MARGIN_PIXELS + TARGET_SIZE / 4) * DEM_RES,
)

# Elevations stay a long way from externalDEMNoDataValue, which SNAP would treat
# as void: the flat DEM sits at BASE and each ramp rises from it.
BASE_HEIGHT = 100.0
HEIGHT_PER_PIXEL = 10.0
NODATA = -9999.0

_GRAPH_HEADER = """<graph id="dem-probe">
  <version>1.0</version>
  <node id="read">
    <operator>Read</operator>
    <parameters><file>{target}</file></parameters>
  </node>
"""

_GRAPH_BRANCH = """  <node id="elevation_{name}">
    <operator>AddElevation</operator>
    <sources><source>read</source></sources>
    <parameters>
      <demName>External DEM</demName>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <externalDEMFile>{dem}</externalDEMFile>
      <externalDEMNoDataValue>{nodata}</externalDEMNoDataValue>
      <elevationBandName>elevation</elevationBandName>
    </parameters>
  </node>
  <node id="write_{name}">
    <operator>Write</operator>
    <sources><source>elevation_{name}</source></sources>
    <parameters><file>{output}</file><formatName>GeoTIFF</formatName></parameters>
  </node>
"""


@dataclass(frozen=True)
class SamplingOffset:
    """Displacement between where SNAP sampled a DEM and where it should have.

    Attributes
    ----------
    column : float
        Eastward displacement, in DEM pixels.
    row : float
        Southward displacement, in DEM pixels.
    scatter : float
        Largest departure from ``column`` or ``row`` across the probe grid. A
        rigid shift keeps this near zero; anything else means the offset is not
        the whole story.
    """

    column: float
    row: float
    scatter: float


@dataclass(frozen=True)
class DEMProbe:
    """What one probe run measured.

    Attributes
    ----------
    raw : SamplingOffset
        Offset for a DEM handed to SNAP as written.
    prepared : SamplingOffset
        Offset for the same DEM after ``prepare_dem_for_snap``.
    undulation : float
        Height in metres SNAP added at `PROBE_CENTRE` when it converted the
        DEM's heights to ellipsoidal, i.e. its geoid undulation there.
    """

    raw: SamplingOffset
    prepared: SamplingOffset
    undulation: float


def _write_raster(path: Path, values: np.ndarray, origin: tuple[float, float], res: float) -> Path:
    """Write a single-band WGS 84 GeoTIFF and return its path."""
    height, width = values.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(origin[0], origin[1], res, res),
    ) as dst:
        dst.write(values.astype("float32"), 1)
    return path


def _write_inputs(work_dir: Path) -> tuple[Path, dict[str, Path]]:
    """Write the target grid and the flat and ramped DEMs the probe compares."""
    rows, columns = np.mgrid[0:DEM_HEIGHT, 0:DEM_WIDTH]
    origin = (DEM_ORIGIN_LON, DEM_ORIGIN_LAT)
    dems = {
        "flat": np.full((DEM_HEIGHT, DEM_WIDTH), BASE_HEIGHT),
        "column": BASE_HEIGHT + HEIGHT_PER_PIXEL * columns,
        "row": BASE_HEIGHT + HEIGHT_PER_PIXEL * rows,
    }
    written = {
        name: _write_raster(work_dir / f"dem_{name}.tif", values, origin, DEM_RES)
        for name, values in dems.items()
    }

    target = _write_raster(
        work_dir / "target.tif",
        np.zeros((TARGET_SIZE, TARGET_SIZE)),
        (
            DEM_ORIGIN_LON + TARGET_MARGIN_PIXELS * DEM_RES,
            DEM_ORIGIN_LAT - TARGET_MARGIN_PIXELS * DEM_RES,
        ),
        TARGET_RES,
    )
    return target, written


def _run_graph(
    gpt: Path, work_dir: Path, target: Path, dems: Mapping[str, Path]
) -> dict[str, Path]:
    """Sample every DEM onto the target grid in a single `gpt` invocation."""
    outputs = {name: work_dir / f"elevation_{name}.tif" for name in dems}
    graph = _GRAPH_HEADER.format(target=target) + "".join(
        _GRAPH_BRANCH.format(
            name=name,
            dem=dem,
            nodata=NODATA,
            output=outputs[name].with_suffix(""),
        )
        for name, dem in dems.items()
    )
    graph_file = work_dir / "dem_probe.xml"
    graph_file.write_text(f"{graph}</graph>\n")

    result = subprocess.run(  # noqa: S603
        [str(gpt), str(graph_file), "-q", "2"],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if result.returncode != 0:
        msg = f"gpt failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
        raise RuntimeError(msg)
    return outputs


def _read_elevation(path: Path) -> np.ndarray:
    """Return the elevation band AddElevation appended to a probe output.

    SNAP's GeoTIFF writer drops band names, so the band is taken by position:
    AddElevation appends it to the one band the target grid carries.
    """
    with rasterio.open(path) as src:
        if src.count != 2:
            msg = (
                f"Expected the target band and one elevation band in "
                f"'{path.name}', found {src.count} band(s)"
            )
            raise RuntimeError(msg)
        return src.read(2).astype("float64")


def _true_indices() -> tuple[np.ndarray, np.ndarray]:
    """Return the DEM column and row index of every target pixel centre."""
    rows, columns = np.mgrid[0:TARGET_SIZE, 0:TARGET_SIZE]
    lon = DEM_ORIGIN_LON + TARGET_MARGIN_PIXELS * DEM_RES + (columns + 0.5) * TARGET_RES
    lat = DEM_ORIGIN_LAT - TARGET_MARGIN_PIXELS * DEM_RES - (rows + 0.5) * TARGET_RES
    # The first pixel's centre lies half a pixel inside the origin corner.
    return (
        (lon - DEM_ORIGIN_LON) / DEM_RES - 0.5,
        (DEM_ORIGIN_LAT - lat) / DEM_RES - 0.5,
    )


def _offset(elevations: Mapping[str, np.ndarray], suffix: str) -> SamplingOffset:
    """Reduce one pair of ramp samples to the offset they imply."""
    true_column, true_row = _true_indices()
    displacements = []
    for axis, truth in (("column", true_column), ("row", true_row)):
        sampled = (elevations[f"{axis}{suffix}"] - elevations["flat"]) / HEIGHT_PER_PIXEL
        displacements.append(sampled - truth)
    return SamplingOffset(
        column=float(displacements[0].mean()),
        row=float(displacements[1].mean()),
        scatter=float(max(np.ptp(displacement) for displacement in displacements)),
    )


def measure_dem_sampling(gpt: Path, work_dir: Path) -> DEMProbe:
    """Measure how the installed SNAP reads an external DEM.

    Parameters
    ----------
    gpt : Path
        The `gpt` executable to probe.
    work_dir : Path
        Directory for the synthetic inputs, the graph, and SNAP's outputs.

    Returns
    -------
    DEMProbe
        Sampling offsets for the DEM as written and as prepared, and the geoid
        undulation SNAP applied.
    """
    target, dems = _write_inputs(work_dir)
    prepared = {
        f"{name}_prepared": prepare_dem_for_snap(
            dem,
            cache_dir=work_dir / "cache",
            snap_bin=gpt.parent,
        )
        for name, dem in dems.items()
        if name != "flat"
    }
    outputs = _run_graph(gpt, work_dir, target, {**dems, **prepared})
    elevations = {name: _read_elevation(path) for name, path in outputs.items()}

    return DEMProbe(
        raw=_offset(elevations, ""),
        prepared=_offset(elevations, "_prepared"),
        undulation=float(elevations["flat"].mean()) - BASE_HEIGHT,
    )
