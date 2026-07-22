"""Dynamic SNAP GPT graph generation from typed Pydantic configs.

This module replaces the hand-maintained XML graphs that used to live under
``config/snap_graphs/``.  Each product type (backscatter, coherence,
interferogram) has a frozen Pydantic config that describes the
graph shape; ``build_*_graph(cfg)`` functions emit the corresponding SNAP
GPT XML on demand.

Per-burst values such as ``${swath}``, ``${polarization}``, ``${date1}``,
``${date2}``, ``${crs}``, ``${primary}``, ``${secondary}``, ``${source}``
are left as ``-P`` / ``-S`` placeholders so the generated XML is
config-specific but burst-agnostic; the SNAP client substitutes them at
runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, ClassVar
from xml.etree.ElementTree import Element, SubElement, indent, tostring

from lxml import etree
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

# Hardened parser: no entity resolution, no network access (XXE-safe).
_XML_PARSER = etree.XMLParser(resolve_entities=False, no_network=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PARAMS_CLASS = "com.bc.ceres.binding.dom.XppDomElement"


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------


def _to_text(value: Any) -> str | None:
    """Render a Python value the way SNAP's GraphBuilder writes it."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _add_param(parent: Element, name: str, value: Any) -> Element:
    el = SubElement(parent, name)
    el.text = _to_text(value)
    return el


def _new_graph() -> Element:
    graph = Element("graph", {"id": "Graph"})
    SubElement(graph, "version").text = "1.0"
    return graph


def _attach_sources(node: Element, sources: list[str]) -> None:
    """Attach 1+ ``sourceProduct`` entries.

    Each item is either a placeholder ``${name}`` (rendered as element text)
    or a node id (rendered as a ``refid`` attribute).
    """
    sources_el = SubElement(node, "sources")
    for i, src in enumerate(sources):
        tag = "sourceProduct" if i == 0 else f"sourceProduct.{i}"
        if src.startswith("${") and src.endswith("}"):
            SubElement(sources_el, tag).text = src
        else:
            SubElement(sources_el, tag, {"refid": src})


def _new_node(graph: Element, node_id: str, operator: str, sources: list[str]) -> Element:
    node = SubElement(graph, "node", {"id": node_id})
    SubElement(node, "operator").text = operator
    _attach_sources(node, sources)
    return node


def _new_params(node: Element, *, class_attr: bool = True) -> Element:
    attrs = {"class": _PARAMS_CLASS} if class_attr else {}
    return SubElement(node, "parameters", attrs)


def _dem_attrs(dem: str | Path) -> tuple[str, str | None]:
    """Translate a DEM spec into ``(demName, externalDEMFile)`` text values."""
    if isinstance(dem, Path):
        return "External DEM", str(dem)
    return dem, None


def _emit_xml(graph: Element, *, pretty: bool = True) -> str:
    if pretty:
        indent(graph, space="  ")
    return tostring(graph, encoding="unicode")


# ---------------------------------------------------------------------------
# Layer 1: per-operator node builders
# ---------------------------------------------------------------------------


def _apply_orbit_file(graph: Element, node_id: str, source: str) -> str:
    node = _new_node(graph, node_id, "Apply-Orbit-File", [source])
    p = _new_params(node)
    _add_param(p, "orbitType", "Sentinel Precise (Auto Download)")
    _add_param(p, "polyDegree", 3)
    _add_param(p, "continueOnFail", value=False)
    return node_id


def _thermal_noise_removal(graph: Element, node_id: str, source: str) -> str:
    node = _new_node(graph, node_id, "ThermalNoiseRemoval", [source])
    p = _new_params(node)
    _add_param(p, "selectedPolarisations", None)
    _add_param(p, "removeThermalNoise", value=True)
    _add_param(p, "outputNoise", value=False)
    _add_param(p, "reIntroduceThermalNoise", value=False)
    return node_id


def _calibration(
    graph: Element,
    node_id: str,
    source: str,
    *,
    output_band: str = "beta",
) -> str:
    """Append a ``Calibration`` node selecting ``beta``, ``sigma`` or ``gamma``."""
    node = _new_node(graph, node_id, "Calibration", [source])
    p = _new_params(node)
    _add_param(p, "sourceBands", None)
    _add_param(p, "auxFile", "Latest Auxiliary File")
    _add_param(p, "externalAuxFile", None)
    _add_param(p, "outputImageInComplex", value=False)
    _add_param(p, "outputImageScaleInDb", value=False)
    _add_param(p, "createGammaBand", value=False)
    _add_param(p, "createBetaBand", value=False)
    _add_param(p, "selectedPolarisations", None)
    _add_param(p, "outputSigmaBand", output_band == "sigma")
    _add_param(p, "outputGammaBand", output_band == "gamma")
    _add_param(p, "outputBetaBand", output_band == "beta")
    return node_id


def _topsar_deburst(graph: Element, node_id: str, source: str) -> str:
    node = _new_node(graph, node_id, "TOPSAR-Deburst", [source])
    p = _new_params(node)
    _add_param(p, "selectedPolarisations", None)
    return node_id


def _multilook(
    graph: Element,
    node_id: str,
    source: str,
    *,
    n_rg_looks: int,
    n_az_looks: int,
) -> str:
    node = _new_node(graph, node_id, "Multilook", [source])
    p = _new_params(node)
    _add_param(p, "sourceBands", None)
    _add_param(p, "nRgLooks", n_rg_looks)
    _add_param(p, "nAzLooks", n_az_looks)
    _add_param(p, "outputIntensity", value=True)
    _add_param(p, "grSquarePixel", value=True)
    return node_id


def _terrain_flattening(
    graph: Element,
    node_id: str,
    source: str,
    *,
    dem: str | Path,
    dem_resampling: str,
    nodata: float,
    apply_egm: bool,
    additional_overlap: float,
    oversampling: float,
) -> str:
    node = _new_node(graph, node_id, "Terrain-Flattening", [source])
    p = _new_params(node)
    dem_name, dem_file = _dem_attrs(dem)
    _add_param(p, "sourceBands", None)
    _add_param(p, "demName", dem_name)
    _add_param(p, "demResamplingMethod", dem_resampling)
    _add_param(p, "externalDEMFile", dem_file)
    _add_param(p, "externalDEMNoDataValue", nodata)
    _add_param(p, "externalDEMApplyEGM", apply_egm)
    _add_param(p, "outputSimulatedImage", value=True)
    _add_param(p, "outputSigma0", value=False)
    _add_param(p, "nodataValueAtSea", value=True)
    _add_param(p, "additionalOverlap", additional_overlap)
    _add_param(p, "oversamplingMultiple", oversampling)
    return node_id


def _back_geocoding(
    graph: Element,
    node_id: str,
    primary: str,
    secondary: str,
    *,
    dem: str | Path,
    dem_resampling: str,
    resampling: str,
    nodata: float,
) -> str:
    node = _new_node(graph, node_id, "Back-Geocoding", [primary, secondary])
    p = _new_params(node)
    dem_name, dem_file = _dem_attrs(dem)
    _add_param(p, "demName", dem_name)
    _add_param(p, "demResamplingMethod", dem_resampling)
    _add_param(p, "externalDEMFile", dem_file)
    _add_param(p, "externalDEMNoDataValue", nodata)
    _add_param(p, "resamplingType", resampling)
    _add_param(p, "maskOutAreaWithoutElevation", value=True)
    _add_param(p, "outputRangeAzimuthOffset", value=False)
    _add_param(p, "outputDerampDemodPhase", value=False)
    _add_param(p, "disableReramp", value=False)
    return node_id


def _coherence(
    graph: Element,
    node_id: str,
    source: str,
    *,
    win_az: int,
    win_rg: int,
    subtract_flat_earth_phase: bool,
    subtract_topographic_phase: bool,
    square_pixel: bool,
    dem: str | Path,
    nodata: float,
    apply_egm: bool,
) -> str:
    node = _new_node(graph, node_id, "Coherence", [source])
    p = _new_params(node)
    dem_name, dem_file = _dem_attrs(dem)
    _add_param(p, "cohWinAz", win_az)
    _add_param(p, "cohWinRg", win_rg)
    _add_param(p, "subtractFlatEarthPhase", subtract_flat_earth_phase)
    _add_param(p, "srpPolynomialDegree", 5)
    _add_param(p, "srpNumberPoints", 501)
    _add_param(p, "orbitDegree", 3)
    _add_param(p, "squarePixel", square_pixel)
    _add_param(p, "subtractTopographicPhase", subtract_topographic_phase)
    _add_param(p, "demName", dem_name)
    _add_param(p, "externalDEMFile", dem_file)
    _add_param(p, "externalDEMNoDataValue", nodata)
    _add_param(p, "externalDEMApplyEGM", apply_egm)
    _add_param(p, "tileExtensionPercent", 100)
    _add_param(p, "singleMaster", value=True)
    return node_id


def _interferogram(
    graph: Element,
    node_id: str,
    source: str,
    *,
    subtract_flat_earth_phase: bool,
    subtract_topographic_phase: bool,
    square_pixel: bool,
    dem: str | Path,
    nodata: float,
    apply_egm: bool,
) -> str:
    node = _new_node(graph, node_id, "Interferogram", [source])
    p = _new_params(node)
    dem_name, dem_file = _dem_attrs(dem)
    _add_param(p, "subtractFlatEarthPhase", subtract_flat_earth_phase)
    _add_param(p, "srpPolynomialDegree", 5)
    _add_param(p, "srpNumberPoints", 501)
    _add_param(p, "orbitDegree", 3)
    _add_param(p, "includeCoherence", value=False)
    _add_param(p, "cohWinAz", 3)
    _add_param(p, "cohWinRg", 15)
    _add_param(p, "squarePixel", square_pixel)
    _add_param(p, "subtractTopographicPhase", subtract_topographic_phase)
    _add_param(p, "demName", dem_name)
    _add_param(p, "externalDEMFile", dem_file)
    _add_param(p, "externalDEMNoDataValue", nodata)
    _add_param(p, "externalDEMApplyEGM", apply_egm)
    _add_param(p, "tileExtensionPercent", 100)
    _add_param(p, "outputFlatEarthPhase", value=False)
    _add_param(p, "outputTopoPhase", value=False)
    _add_param(p, "outputElevation", value=False)
    _add_param(p, "outputLatLon", value=False)
    return node_id


def _band_select(graph: Element, node_id: str, source: str, *, pattern: str) -> str:
    node = _new_node(graph, node_id, "BandSelect", [source])
    # Match the existing static graph: BandSelect's <parameters> has no class.
    p = _new_params(node, class_attr=False)
    _add_param(p, "bandNamePattern", pattern)
    return node_id


def _band_merge(graph: Element, node_id: str, sources: list[str]) -> str:
    node = _new_node(graph, node_id, "BandMerge", sources)
    p = _new_params(node)
    # SNAP convention: scientific notation matches GraphBuilder output.
    _add_param(p, "geographicError", "1.0E-5")
    return node_id


def _band_maths(
    graph: Element,
    node_id: str,
    source: str,
    *,
    target_bands: list[dict[str, Any]],
    include_variables_element: bool = True,
) -> str:
    """Append a ``BandMaths`` node.

    ``target_bands`` items: each ``dict`` has ``name``, ``type``,
    ``expression`` and optionally ``unit`` and ``noDataValue``.  The
    ``include_variables_element`` flag controls emission of the empty
    ``<variables/>`` element that some SNAP-authored graphs include.
    """
    node = _new_node(graph, node_id, "BandMaths", [source])
    p = _new_params(node)
    tbs = SubElement(p, "targetBands")
    for tb in target_bands:
        el = SubElement(tbs, "targetBand")
        _add_param(el, "name", tb["name"])
        _add_param(el, "type", tb.get("type", "float32"))
        _add_param(el, "expression", tb["expression"])
        if "unit" in tb:
            _add_param(el, "unit", tb["unit"])
        if "noDataValue" in tb:
            _add_param(el, "noDataValue", tb["noDataValue"])
    if include_variables_element:
        SubElement(p, "variables")
    return node_id


def _terrain_correction(
    graph: Element,
    node_id: str,
    source: str,
    *,
    dem: str | Path,
    nodata: float,
    apply_egm: bool,
    img_resampling: str,
    dem_resampling: str,
    grid_spacing: float,
    align_to_standard_grid: bool,
    standard_grid_origin: tuple[float, float],
    output_complex: bool = False,
) -> str:
    node = _new_node(graph, node_id, "Terrain-Correction", [source])
    p = _new_params(node)
    dem_name, dem_file = _dem_attrs(dem)
    _add_param(p, "sourceBands", None)
    _add_param(p, "demName", dem_name)
    _add_param(p, "externalDEMFile", dem_file)
    _add_param(p, "externalDEMNoDataValue", nodata)
    _add_param(p, "externalDEMApplyEGM", apply_egm)
    _add_param(p, "demResamplingMethod", dem_resampling)
    _add_param(p, "imgResamplingMethod", img_resampling)
    _add_param(p, "pixelSpacingInMeter", grid_spacing)
    _add_param(p, "mapProjection", "${crs}")
    _add_param(p, "alignToStandardGrid", align_to_standard_grid)
    _add_param(p, "standardGridOriginX", standard_grid_origin[0])
    _add_param(p, "standardGridOriginY", standard_grid_origin[1])
    _add_param(p, "nodataValueAtSea", value=True)
    _add_param(p, "saveDEM", value=False)
    _add_param(p, "saveLatLon", value=False)
    _add_param(p, "saveIncidenceAngleFromEllipsoid", value=False)
    _add_param(p, "saveLocalIncidenceAngle", value=False)
    _add_param(p, "saveProjectedLocalIncidenceAngle", value=False)
    _add_param(p, "saveSelectedSourceBand", value=True)
    _add_param(p, "saveLayoverShadowMask", value=False)
    _add_param(p, "outputComplex", output_complex)
    _add_param(p, "applyRadiometricNormalization", value=False)
    _add_param(p, "saveSigmaNought", value=False)
    _add_param(p, "saveGammaNought", value=False)
    _add_param(p, "saveBetaNought", value=False)
    _add_param(p, "incidenceAngleForSigma0", "Use projected local incidence angle from DEM")
    _add_param(p, "incidenceAngleForGamma0", "Use projected local incidence angle from DEM")
    _add_param(p, "auxFile", "Latest Auxiliary File")
    _add_param(p, "externalAuxFile", None)
    return node_id


# ---------------------------------------------------------------------------
# Mermaid converter
# ---------------------------------------------------------------------------


def xml_to_mermaid(graph_xml: str) -> str:
    """Convert a SNAP GPT XML processing graph (string) to a Mermaid script.

    Parameters
    ----------
    graph_xml : str
        SNAP GPT graph XML document.

    Returns
    -------
    str
        ``flowchart LR`` Mermaid diagram source.
    """
    root = etree.fromstring(graph_xml.encode(), parser=_XML_PARSER)

    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    input_vars: set[str] = set()

    for node in root.findall("./node[@id]"):
        node_id = node.get("id")
        if node_id is None:
            continue
        nodes[node_id] = node_id

        sources = node.find("sources")
        if sources is None:
            continue
        for source in sources:
            source_text = source.text
            if source_text and source_text.startswith("${") and source_text.endswith("}"):
                var_name = source_text[2:-1]
                input_vars.add(var_name)
                edges.append((var_name, node_id))
            else:
                ref_id = source.get("refid")
                if ref_id:
                    edges.append((ref_id, node_id))

    lines = ["flowchart LR"]
    lines.extend(f'    {var}[/"{var}"/]' for var in sorted(input_vars))
    lines.extend(f'    {node_id}["{node_id}"]' for node_id in sorted(nodes))
    lines.extend(f"    {source} --> {target}" for source, target in edges)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Layer 3: per-product graph composers
# ---------------------------------------------------------------------------


def _replace_zeros_target_bands_backscatter(*, flatten_terrain: bool) -> list[dict[str, Any]]:
    if flatten_terrain:
        return [
            {
                "name": "Gamma0",
                "type": "float32",
                "expression": (
                    "Gamma0_${swath}_${polarization} > 0 ? Gamma0_${swath}_${polarization} : NaN"
                ),
                "noDataValue": "NaN",
            },
            {
                "name": "sim",
                "type": "float32",
                "expression": "simulatedImage > 0 ? simulatedImage : NaN",
                "noDataValue": "NaN",
            },
        ]
    return [
        {
            "name": "Beta0",
            "type": "float32",
            "expression": (
                "Beta0_${swath}_${polarization} > 0 ? Beta0_${swath}_${polarization} : NaN"
            ),
            "noDataValue": "NaN",
        },
    ]


def _replace_zeros_target_bands_coherence() -> list[dict[str, Any]]:
    return [
        {
            "name": "coh",
            "type": "float32",
            "expression": (
                "coh_${swath}_${polarization}_${date1}_${date2} > 0 ? "
                "coh_${swath}_${polarization}_${date1}_${date2} : NaN"
            ),
            "noDataValue": "NaN",
        },
    ]


def _intensity_target_bands() -> list[dict[str, Any]]:
    return [
        {
            "name": "int1",
            "type": "float32",
            "expression": (
                "sq(i_${swath}_${polarization}_mst_${date1}) + "
                "sq(q_${swath}_${polarization}_mst_${date1})"
            ),
            "unit": "intensity",
            "noDataValue": 0,
        },
        {
            "name": "int2",
            "type": "float32",
            "expression": (
                "sq(i_${swath}_${polarization}_slv1_${date2}) + "
                "sq(q_${swath}_${polarization}_slv1_${date2})"
            ),
            "unit": "intensity",
            "noDataValue": 0,
        },
    ]


def _ifg_replace_zeros_target_bands() -> list[dict[str, Any]]:
    cond = "(int1 != 0) and (int2 != 0)"
    return [
        {
            "name": "i_ifg",
            "type": "float32",
            "expression": (
                f"{cond} ? i_ifg_${{swath}}_${{polarization}}_${{date1}}_${{date2}} : NaN"
            ),
            "unit": "real",
            "noDataValue": "NaN",
        },
        {
            "name": "q_ifg",
            "type": "float32",
            "expression": (
                f"{cond} ? q_ifg_${{swath}}_${{polarization}}_${{date1}}_${{date2}} : NaN"
            ),
            "unit": "imaginary",
            "noDataValue": "NaN",
        },
        {
            "name": "intensity1",
            "type": "float32",
            "expression": f"{cond} ? int1 : NaN",
            "unit": "intensity",
            "noDataValue": "NaN",
        },
        {
            "name": "intensity2",
            "type": "float32",
            "expression": f"{cond} ? int2 : NaN",
            "unit": "intensity",
            "noDataValue": "NaN",
        },
    ]


def build_backscatter_graph(cfg: BackscatterConfig) -> str:
    """Emit the SNAP GPT XML for the backscatter (RTC or GTC) pipeline."""
    g = _new_graph()
    src = _apply_orbit_file(g, "Apply-Orbit-File", "${source}")
    src = _thermal_noise_removal(g, "ThermalNoiseRemoval", src)
    src = _calibration(g, "Calibration", src, output_band="beta")
    src = _topsar_deburst(g, "TOPSAR-Deburst", src)
    if cfg.multilook is not None:
        rg, az = cfg.multilook
        src = _multilook(g, "Multilook", src, n_rg_looks=rg, n_az_looks=az)
    if cfg.flatten_terrain:
        src = _terrain_flattening(
            g,
            "Terrain-Flattening",
            src,
            dem=cfg.dem,
            dem_resampling=cfg.dem_resampling,
            nodata=cfg.external_dem_nodata,
            apply_egm=cfg.external_dem_apply_egm,
            additional_overlap=cfg.additional_overlap,
            oversampling=cfg.dem_oversampling,
        )
    if cfg.replace_zeros_with_nan:
        src = _band_maths(
            g,
            "ReplaceZerosWithNaNs",
            src,
            target_bands=_replace_zeros_target_bands_backscatter(
                flatten_terrain=cfg.flatten_terrain,
            ),
        )
    _terrain_correction(
        g,
        "Terrain-Correction",
        src,
        dem=cfg.dem,
        nodata=cfg.external_dem_nodata,
        apply_egm=cfg.external_dem_apply_egm,
        img_resampling=cfg.img_resampling,
        dem_resampling=cfg.dem_resampling,
        grid_spacing=cfg.grid_spacing,
        align_to_standard_grid=cfg.align_to_standard_grid,
        standard_grid_origin=cfg.standard_grid_origin,
    )
    return _emit_xml(g)


def _build_pair_prefix(g: Element, cfg: GeocodingConfig) -> str:
    """Emit two Apply-Orbit-File nodes plus Back-Geocoding; return its id."""
    _apply_orbit_file(g, "Apply-Orbit-File-1", "${primary}")
    _apply_orbit_file(g, "Apply-Orbit-File-2", "${secondary}")
    return _back_geocoding(
        g,
        "Back-Geocoding",
        "Apply-Orbit-File-1",
        "Apply-Orbit-File-2",
        dem=cfg.dem,
        dem_resampling=cfg.dem_resampling,
        resampling=cfg.img_resampling,
        nodata=cfg.external_dem_nodata,
    )


def build_coherence_magnitude_graph(cfg: CoherenceConfig) -> str:
    """Single-shot SLC→Coherence→Terrain-Correction graph (magnitude only)."""
    g = _new_graph()
    src = _build_pair_prefix(g, cfg)
    win_rg, win_az = cfg.window
    src = _coherence(
        g,
        "Coherence",
        src,
        win_az=win_az,
        win_rg=win_rg,
        subtract_flat_earth_phase=cfg.subtract_flat_earth_phase,
        subtract_topographic_phase=cfg.subtract_topographic_phase,
        square_pixel=cfg.square_pixel,
        dem=cfg.dem,
        nodata=cfg.external_dem_nodata,
        apply_egm=cfg.external_dem_apply_egm,
    )
    src = _topsar_deburst(g, "TOPSAR-Deburst", src)
    if cfg.replace_zeros_with_nan:
        src = _band_maths(
            g,
            "ReplaceZerosWithNaNs",
            src,
            target_bands=_replace_zeros_target_bands_coherence(),
        )
    _terrain_correction(
        g,
        "Terrain-Correction",
        src,
        dem=cfg.dem,
        nodata=cfg.external_dem_nodata,
        apply_egm=cfg.external_dem_apply_egm,
        img_resampling=cfg.img_resampling,
        dem_resampling=cfg.dem_resampling,
        grid_spacing=cfg.grid_spacing,
        align_to_standard_grid=cfg.align_to_standard_grid,
        standard_grid_origin=cfg.standard_grid_origin,
    )
    return _emit_xml(g)


def _build_ifg_branched_graph(
    g: Element,
    *,
    subtract_flat_earth_phase: bool,
    subtract_topographic_phase: bool,
    square_pixel: bool,
    dem: str | Path,
    nodata: float,
    apply_egm: bool,
) -> str:
    """Emit the Interferogram + IntensityBands branches and merge them.

    Returns the id of the deburst node sitting at the merge tail.
    """
    _interferogram(
        g,
        "Interferogram",
        "Back-Geocoding",
        subtract_flat_earth_phase=subtract_flat_earth_phase,
        subtract_topographic_phase=subtract_topographic_phase,
        square_pixel=square_pixel,
        dem=dem,
        nodata=nodata,
        apply_egm=apply_egm,
    )
    _band_select(g, "IfgSelect", "Interferogram", pattern="[iq]_ifg_.*")
    _band_maths(
        g,
        "IntensityBands",
        "Back-Geocoding",
        target_bands=_intensity_target_bands(),
        include_variables_element=False,
    )
    _band_merge(g, "BandMerge", ["IfgSelect", "IntensityBands"])
    return _topsar_deburst(g, "TOPSAR-Deburst", "BandMerge")


def build_interferogram_graph(cfg: InterferogramConfig) -> str:
    """Branched SLC→(Ifg | intensities)→merge→deburst→Terrain-Correction graph."""
    g = _new_graph()
    _build_pair_prefix(g, cfg)
    src = _build_ifg_branched_graph(
        g,
        subtract_flat_earth_phase=cfg.subtract_flat_earth_phase,
        subtract_topographic_phase=cfg.subtract_topographic_phase,
        square_pixel=cfg.square_pixel,
        dem=cfg.dem,
        nodata=cfg.external_dem_nodata,
        apply_egm=cfg.external_dem_apply_egm,
    )
    if cfg.replace_zeros_with_nan:
        src = _band_maths(
            g,
            "ReplaceZerosWithNaNs",
            src,
            target_bands=_ifg_replace_zeros_target_bands(),
            include_variables_element=False,
        )
    _terrain_correction(
        g,
        "Terrain-Correction",
        src,
        dem=cfg.dem,
        nodata=cfg.external_dem_nodata,
        apply_egm=cfg.external_dem_apply_egm,
        img_resampling=cfg.img_resampling,
        dem_resampling=cfg.dem_resampling,
        grid_spacing=cfg.grid_spacing,
        align_to_standard_grid=cfg.align_to_standard_grid,
        standard_grid_origin=cfg.standard_grid_origin,
        # i_ifg / q_ifg are tagged real / imaginary, so SNAP needs to keep
        # them paired through projection.
        output_complex=True,
    )
    return _emit_xml(g)


def build_coherence_complex_ifg_graph(cfg: CoherenceConfig) -> str:
    """Emit the *no-TC* ifg graph used by complex coherence.

    Output is a BEAM-DIMAP product with bands ``i_ifg``, ``q_ifg``,
    ``intensity1``, ``intensity2`` in radar coordinates.  Python band-math
    (driven by the snap_client) then computes the complex coherence and
    a separate Terrain-Correction graph projects the result.
    """
    g = _new_graph()
    _build_pair_prefix(g, cfg)
    _build_ifg_branched_graph(
        g,
        subtract_flat_earth_phase=cfg.subtract_flat_earth_phase,
        subtract_topographic_phase=cfg.subtract_topographic_phase,
        square_pixel=cfg.square_pixel,
        dem=cfg.dem,
        nodata=cfg.external_dem_nodata,
        apply_egm=cfg.external_dem_apply_egm,
    )
    _band_maths(
        g,
        "ReplaceZerosWithNaNs",
        "TOPSAR-Deburst",
        target_bands=_ifg_replace_zeros_target_bands(),
        include_variables_element=False,
    )
    return _emit_xml(g)


def build_terrain_correction_graph(cfg: GeocodingConfig) -> str:
    """Emit a standalone ``Terrain-Correction`` graph (replaces ``S1_TC.xml``)."""
    g = _new_graph()
    _terrain_correction(
        g,
        "Terrain-Correction",
        "${source}",
        dem=cfg.dem,
        nodata=cfg.external_dem_nodata,
        apply_egm=cfg.external_dem_apply_egm,
        img_resampling=cfg.img_resampling,
        dem_resampling=cfg.dem_resampling,
        grid_spacing=cfg.grid_spacing,
        align_to_standard_grid=cfg.align_to_standard_grid,
        standard_grid_origin=cfg.standard_grid_origin,
    )
    return _emit_xml(g)


# ---------------------------------------------------------------------------
# Layer 2: Pydantic configs
# ---------------------------------------------------------------------------

_PositiveInt = Annotated[int, Field(gt=0)]


class GeocodingConfig(BaseModel):
    """Terrain-Correction tail parameters shared by every product type.

    Used directly to emit a standalone ``Terrain-Correction`` graph (e.g. for
    the post-processing step of complex coherence) and inherited by
    each per-product config.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @field_validator("dem", mode="before")
    @classmethod
    def _coerce_dem_path(cls, v: str | Path) -> str | Path:
        """Coerce string file paths to Path so _dem_attrs routes them correctly."""
        if isinstance(v, str) and Path(v).suffix:
            return Path(v)
        return v

    grid_spacing: float = 10.0
    dem: str | Path = "Copernicus 30m Global DEM"
    img_resampling: str = "BILINEAR_INTERPOLATION"
    dem_resampling: str = "BILINEAR_INTERPOLATION"
    align_to_standard_grid: bool = True
    external_dem_nodata: float = 0.0
    external_dem_apply_egm: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def standard_grid_origin(self) -> tuple[float, float]:
        """Origin of standard grid: At pixel centers."""
        half = self.grid_spacing / 2
        return (half, half)

    def to_xml(self) -> str:
        """Build the standalone Terrain-Correction graph XML for this config."""
        return build_terrain_correction_graph(self)

    def to_mermaid(self) -> str:
        """Mermaid ``flowchart LR`` diagram source for this config's graph."""
        return xml_to_mermaid(self.to_xml())


class BackscatterConfig(GeocodingConfig):
    """Configuration for the backscatter (RTC / GTC) pipeline.

    ``flatten_terrain=True`` (default) inserts a ``Terrain-Flattening`` step
    and produces an RTC product.  ``flatten_terrain=False`` skips it and
    produces a GTC product.

    ``multilook`` is ``(n_rg_looks, n_az_looks)``; pass ``None`` to skip the
    ``Multilook`` step.
    """

    flatten_terrain: bool = True
    multilook: tuple[_PositiveInt, _PositiveInt] | None = (5, 1)
    dem_oversampling: float = 1.0
    additional_overlap: float = Field(default=0.2, ge=0.0)
    replace_zeros_with_nan: bool = True

    @property
    def output_bands(self) -> list[str]:
        """Names assigned to the output GeoTIFF bands by the SNAP client."""
        if self.flatten_terrain:
            return ["Gamma0", "simulatedImage"]
        return ["Beta0"]

    def to_xml(self) -> str:
        """Build the backscatter graph XML for this config."""
        return build_backscatter_graph(self)


class CoherenceConfig(GeocodingConfig):
    """Configuration for the coherence pipeline (magnitude or complex)."""

    window: tuple[_PositiveInt, _PositiveInt] = (25, 5)
    subtract_flat_earth_phase: bool = True
    subtract_topographic_phase: bool = True
    square_pixel: bool = False
    replace_zeros_with_nan: bool = True
    write_complex: bool = False

    @property
    def output_bands(self) -> list[str]:
        """Names assigned to the output GeoTIFF bands by the SNAP client."""
        if self.write_complex:
            return ["coh_i", "coh_q"]
        return ["coh"]

    def to_xml(self) -> str:
        """Build the coherence graph XML for this config."""
        if self.write_complex:
            return build_coherence_complex_ifg_graph(self)
        return build_coherence_magnitude_graph(self)


class InterferogramConfig(GeocodingConfig):
    """Configuration for the interferogram pipeline."""

    subtract_flat_earth_phase: bool = True
    subtract_topographic_phase: bool = True
    square_pixel: bool = False
    replace_zeros_with_nan: bool = True

    @property
    def output_bands(self) -> list[str]:
        """Names assigned to the output GeoTIFF bands by the SNAP client."""
        return ["ifg_i", "ifg_q", "intensity_1", "intensity_2"]

    def to_xml(self) -> str:
        """Build the interferogram graph XML for this config."""
        return build_interferogram_graph(self)


__all__ = [
    "BackscatterConfig",
    "CoherenceConfig",
    "GeocodingConfig",
    "InterferogramConfig",
    "build_backscatter_graph",
    "build_coherence_complex_ifg_graph",
    "build_coherence_magnitude_graph",
    "build_interferogram_graph",
    "build_terrain_correction_graph",
    "xml_to_mermaid",
]
