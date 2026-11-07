"""SNAP GPT graph generation from typed Pydantic configs.

Each product type (backscatter, coherence) has a frozen
Pydantic config that describes the graph shape; ``build_*_graph(cfg)``
functions emit the corresponding SNAP GPT XML on demand.

Per-burst values such as ``${swath}``, ``${polarization}``, ``${date1}``,
``${date2}``, ``${crs}``, ``${primary}``, ``${secondary}``, ``${source}``
are left as ``-P`` / ``-S`` placeholders so the generated XML is
config-specific but burst-agnostic; the SNAP client substitutes them at
runtime.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Annotated, Any, ClassVar, Final, Literal, Self, override
from xml.etree.ElementTree import Element, SubElement, indent, tostring

from lxml import etree
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from s1bursts.errors import ConfigurationError

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


def _new_params(node: Element) -> Element:
    return SubElement(node, "parameters", {"class": _PARAMS_CLASS})


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
    output_intensity: bool = True,
) -> str:
    node = _new_node(graph, node_id, "Multilook", [source])
    p = _new_params(node)
    _add_param(p, "sourceBands", None)
    _add_param(p, "nRgLooks", n_rg_looks)
    _add_param(p, "nAzLooks", n_az_looks)
    _add_param(p, "outputIntensity", output_intensity)
    # Pinned off, never exposed: with grSquarePixel enabled SNAP derives
    # nAzLooks from nRgLooks and the local geometry and silently discards the
    # requested azimuth looks, so the applied multilook stops matching the
    # configured one. See the note on BackscatterConfig.multilook.
    _add_param(p, "grSquarePixel", value=False)
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
    correct_coherence_bias: bool | None,
    remove_local_phase_ramp: bool | None,
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
    # Pinned off, never exposed: with squarePixel enabled SNAP 14 derives
    # cohWinAz from cohWinRg and the scene-center geometry and discards the
    # requested azimuth window, so the applied window stops matching the
    # configured one. SNAP 13 declares the parameter but never reads it.
    _add_param(p, "squarePixel", value=False)
    _add_param(p, "subtractTopographicPhase", subtract_topographic_phase)
    # ``None`` for a SNAP that does not know the option; see
    # `supports_coherence_options`.
    if correct_coherence_bias is not None:
        _add_param(p, "correctCoherenceBias", correct_coherence_bias)
    if remove_local_phase_ramp is not None:
        _add_param(p, "removeLocalPhaseRamp", remove_local_phase_ramp)
    _add_param(p, "demName", dem_name)
    _add_param(p, "externalDEMFile", dem_file)
    _add_param(p, "externalDEMNoDataValue", nodata)
    _add_param(p, "externalDEMApplyEGM", apply_egm)
    _add_param(p, "tileExtensionPercent", 100)
    _add_param(p, "singleMaster", value=True)
    return node_id


def _band_maths(
    graph: Element,
    node_id: str,
    source: str,
    *,
    target_bands: list[dict[str, Any]],
) -> str:
    """Append a ``BandMaths`` node.

    ``target_bands`` items: each ``dict`` has ``name``, ``type``,
    ``expression`` and ``noDataValue``.
    """
    node = _new_node(graph, node_id, "BandMaths", [source])
    p = _new_params(node)
    tbs = SubElement(p, "targetBands")
    for tb in target_bands:
        el = SubElement(tbs, "targetBand")
        _add_param(el, "name", tb["name"])
        _add_param(el, "type", tb["type"])
        _add_param(el, "expression", tb["expression"])
        _add_param(el, "noDataValue", tb["noDataValue"])
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
    _add_param(p, "outputComplex", value=False)
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
    graph_xml
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


# SNAP 13 detects no-data by equality, so NaN reaches interpolation instead.
# This finite marker works across supported versions, lies below every band,
# whose expressions keep only positive values, and survives SNAP's
# float32-to-double comparison exactly. `_finalize_product` converts it to the product's no-data.
NODATA_SENTINEL: Final = -1099511627776.0  # -2**40

_SENTINEL_EXPR = repr(NODATA_SENTINEL)


def _replace_zeros_target_bands_backscatter(*, flatten_terrain: bool) -> list[dict[str, Any]]:
    if flatten_terrain:
        return [
            {
                "name": "gamma0",
                "type": "float32",
                "expression": (
                    f"Gamma0_${{swath}}_${{polarization}} > 0 ? "
                    f"Gamma0_${{swath}}_${{polarization}} : {_SENTINEL_EXPR}"
                ),
                "noDataValue": _SENTINEL_EXPR,
            },
            {
                "name": "scattering_area",
                "type": "float32",
                "expression": f"simulatedImage > 0 ? simulatedImage : {_SENTINEL_EXPR}",
                "noDataValue": _SENTINEL_EXPR,
            },
        ]
    return [
        {
            "name": "beta0",
            "type": "float32",
            "expression": (
                f"Beta0_${{swath}}_${{polarization}} > 0 ? "
                f"Beta0_${{swath}}_${{polarization}} : {_SENTINEL_EXPR}"
            ),
            "noDataValue": _SENTINEL_EXPR,
        },
    ]


def _replace_zeros_target_bands_coherence() -> list[dict[str, Any]]:
    return [
        {
            "name": "coherence",
            "type": "float32",
            "expression": (
                f"coh_${{swath}}_${{polarization}}_${{date1}}_${{date2}} > 0 ? "
                f"coh_${{swath}}_${{polarization}}_${{date1}}_${{date2}} : {_SENTINEL_EXPR}"
            ),
            "noDataValue": _SENTINEL_EXPR,
        },
    ]


# SNAP 14 added the `correctCoherenceBias` and `removeLocalPhaseRamp` options
# to the Coherence operator. GPT rejects a parameter its operator does not
# know, so the graph carries them only for the versions that have them.
_SNAP_ADDED_COHERENCE_OPTIONS = (14,)


def supports_coherence_options(snap_version: tuple[int, ...] | None) -> bool:
    """Whether a SNAP version knows the Coherence operator's SNAP 14 options.

    Parameters
    ----------
    snap_version
        SNAP version as returned by ``s1bursts.snap_client.snap_version``.
        ``None`` (an installation whose version could not be read) counts as
        not supporting them.

    Returns
    -------
    bool
        ``True`` for SNAP 14 and newer.
    """
    return snap_version is not None and snap_version >= _SNAP_ADDED_COHERENCE_OPTIONS


def _coherence_option_values(
    cfg: CoherenceConfig,
    snap_version: tuple[int, ...] | None,
) -> tuple[bool | None, bool | None]:
    """Resolve the SNAP 14 Coherence options to write, ``None`` each to leave out."""
    if supports_coherence_options(snap_version):
        return (cfg.correct_coherence_bias, cfg.remove_local_phase_ramp)

    enabled = [
        name
        for name, value in (
            ("correct_coherence_bias", cfg.correct_coherence_bias),
            ("remove_local_phase_ramp", cfg.remove_local_phase_ramp),
        )
        if value
    ]
    if enabled:
        installed = (
            "the installed SNAP version could not be determined"
            if snap_version is None
            else f"the installed version is {'.'.join(str(part) for part in snap_version)}"
        )
        msg = (
            f"CoherenceConfig({', '.join(f'{name}=True' for name in enabled)}) requires "
            f"SNAP 14 or newer, but {installed}. Upgrade SNAP, or leave the option off."
        )
        raise ConfigurationError(msg)
    # Not writing them is not the same as turning them off: SNAP 14 defaults
    # `correctCoherenceBias` to true. Older SNAP has neither behavior anyway.
    return (None, None)


def build_backscatter_graph(cfg: BackscatterConfig) -> str:
    """Emit the SNAP GPT XML for the backscatter (RTC or GTC) pipeline."""
    # Only RTC has Terrain-Flattening, and only RTCConfig its settings.
    rtc = cfg if isinstance(cfg, RTCConfig) else None
    g = _new_graph()
    src = _apply_orbit_file(g, "Apply-Orbit-File", "${source}")
    src = _thermal_noise_removal(g, "ThermalNoiseRemoval", src)
    src = _calibration(g, "Calibration", src, output_band="beta")
    src = _topsar_deburst(g, "TOPSAR-Deburst", src)
    if cfg.multilook is not None:
        rg, az = cfg.multilook
        src = _multilook(
            g,
            "Multilook",
            src,
            n_rg_looks=rg,
            n_az_looks=az,
        )
    if rtc is not None:
        src = _terrain_flattening(
            g,
            "Terrain-Flattening",
            src,
            dem=rtc.dem,
            dem_resampling=rtc.dem_resampling,
            nodata=rtc.external_dem_nodata,
            apply_egm=rtc.external_dem_apply_egm,
            additional_overlap=rtc.additional_overlap,
            oversampling=rtc.dem_oversampling,
        )
    if cfg.replace_zeros_with_nan:
        src = _band_maths(
            g,
            "ReplaceZerosWithNoData",
            src,
            target_bands=_replace_zeros_target_bands_backscatter(
                flatten_terrain=rtc is not None,
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


def build_coherence_graph(
    cfg: CoherenceConfig,
    *,
    snap_version: tuple[int, ...] | None = None,
) -> str:
    """Single-shot SLC→Coherence→Terrain-Correction graph."""
    correct_coherence_bias, remove_local_phase_ramp = _coherence_option_values(cfg, snap_version)
    g = _new_graph()
    _apply_orbit_file(g, "Apply-Orbit-File-1", "${primary}")
    _apply_orbit_file(g, "Apply-Orbit-File-2", "${secondary}")
    src = _back_geocoding(
        g,
        "Back-Geocoding",
        "Apply-Orbit-File-1",
        "Apply-Orbit-File-2",
        dem=cfg.dem,
        dem_resampling=cfg.dem_resampling,
        resampling=cfg.img_resampling,
        nodata=cfg.external_dem_nodata,
    )
    win_rg, win_az = cfg.window
    src = _coherence(
        g,
        "Coherence",
        src,
        win_az=win_az,
        win_rg=win_rg,
        subtract_flat_earth_phase=cfg.subtract_flat_earth_phase,
        subtract_topographic_phase=cfg.subtract_topographic_phase,
        correct_coherence_bias=correct_coherence_bias,
        remove_local_phase_ramp=remove_local_phase_ramp,
        dem=cfg.dem,
        nodata=cfg.external_dem_nodata,
        apply_egm=cfg.external_dem_apply_egm,
    )
    src = _topsar_deburst(g, "TOPSAR-Deburst", src)
    if cfg.replace_zeros_with_nan:
        src = _band_maths(
            g,
            "ReplaceZerosWithNoData",
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


# ---------------------------------------------------------------------------
# Layer 2: Pydantic configs
# ---------------------------------------------------------------------------

_PositiveInt = Annotated[int, Field(gt=0)]
_PositiveFloat = Annotated[float, Field(gt=0)]

# Plain assignments rather than PEP 695 `type` aliases: these annotate Pydantic
# fields, and older Pydantic releases within the supported range do not resolve
# a `type` alias to its value, warning and falling back to no validation.
ResamplingMethod = Literal[
    "NEAREST_NEIGHBOUR",
    "BILINEAR_INTERPOLATION",
    "CUBIC_CONVOLUTION",
    "BISINC_5_POINT_INTERPOLATION",
    "BISINC_11_POINT_INTERPOLATION",
    "BISINC_21_POINT_INTERPOLATION",
    "BICUBIC_INTERPOLATION",
]
"""Resampling methods SNAP's Terrain-Correction operator accepts.

An unsupported name is not rejected by GPT until the graph has already run for
minutes, so it is checked when the config is built instead.
"""

DEMResamplingMethod = Literal[
    "NEAREST_NEIGHBOUR",
    "BILINEAR_INTERPOLATION",
    "CUBIC_CONVOLUTION",
    "BISINC_5_POINT_INTERPOLATION",
    "BISINC_11_POINT_INTERPOLATION",
    "BISINC_21_POINT_INTERPOLATION",
    "BICUBIC_INTERPOLATION",
    "DELAUNAY_INTERPOLATION",
]
"""Resampling methods SNAP accepts for the DEM, which adds Delaunay interpolation."""


class GeocodingConfig(BaseModel):
    """Terrain-Correction tail parameters shared by every product type.

    Abstract: inherited by each per-product config, which supplies the graph
    that precedes the shared Terrain-Correction tail via ``to_xml``.

    Parameters
    ----------
    grid_spacing
        Output pixel spacing in meters of the target CRS. Must be positive.
    dem
        DEM used for geocoding. A string selects one of SNAP's auto-download
        DEMs; a path (or a string with a file suffix) is passed as an external
        DEM.
    dem_name
        Name of an external DEM, written to product metadata and compared
        before an existing product is skipped. Required when ``dem`` is a path,
        and must not be blank; ignored for SNAP's built-in DEMs, whose ``dem``
        value is already their name. Name each DEM differently, such as by its
        source and version.
    img_resampling
        Resampling of the image during terrain correction.
    dem_resampling
        Resampling of the DEM during terrain correction.
    align_to_standard_grid
        Snap the output to a fixed global grid, so products from different
        bursts and dates share pixel boundaries and can be stacked without
        resampling.
    external_dem_nodata
        No-data value of an external DEM.
    external_dem_apply_egm
        Convert an external DEM from EGM96 geoid heights to WGS84 ellipsoidal
        heights, which is the reference terrain correction requires. SNAP does
        this automatically for its built-in ACE / Copernicus / SRTM DEMs, all of
        which are geoid-referenced; this flag is the equivalent for an external
        one and is ignored unless ``dem`` is a path.

        Set it ``True`` for an orthometric DEM (heights above mean sea level,
        the usual case for national DEMs) and leave it ``False`` only when the
        DEM already carries ellipsoidal heights. Note this default differs from
        SNAP's own, which is ``True``. Leave it ``False`` for a DEM returned by
        ``s1bursts.prepare_dem_for_snap``, which has already been converted
        to ellipsoidal heights. SNAP only applies EGM96, so preprocess DEMs in
        other vertical datums such as EGM2008 instead of enabling this option.
    replace_zeros_with_nan
        Convert SNAP's zero fill value to NaN, so no-data does not enter later
        averages as a valid low value.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @field_validator("dem", mode="before")
    @classmethod
    def _coerce_dem_path(cls, v: str | Path) -> str | Path:
        """Coerce string file paths to Path so _dem_attrs routes them correctly."""
        if isinstance(v, str) and Path(v).suffix:
            return Path(v)
        return v

    grid_spacing: _PositiveFloat = 10.0
    dem: str | Path = "Copernicus 30m Global DEM"
    dem_name: str | None = None
    img_resampling: ResamplingMethod = "BILINEAR_INTERPOLATION"
    dem_resampling: DEMResamplingMethod = "BILINEAR_INTERPOLATION"
    align_to_standard_grid: bool = True
    external_dem_nodata: float = 0.0
    external_dem_apply_egm: bool = False
    replace_zeros_with_nan: bool = True

    @model_validator(mode="after")
    def _require_dem_name(self) -> Self:
        """Reject an external DEM without a name to record it by."""
        if isinstance(self.dem, Path) and not (self.dem_name or "").strip():
            msg = (
                f"dem_name is required for an external DEM ({self.dem}) and must not be "
                "blank: products record "
                "the DEM by name, never by path. Name it by its source and version."
            )
            raise ValueError(msg)
        return self

    @computed_field
    @property
    def standard_grid_origin(self) -> tuple[float, float]:
        """Origin of standard grid: At pixel centers."""
        half = self.grid_spacing / 2
        return (half, half)

    @abstractmethod
    def to_xml(self, *, snap_version: tuple[int, ...] | None = None) -> str:
        """Build the SNAP GPT graph XML for this config.

        Parameters
        ----------
        snap_version
            SNAP version the graph will run under, from
            ``s1bursts.snap_client.snap_version``. The backscatter graph
            does not vary with it; the coherence graph writes only the
            options that version has (see ``supports_coherence_options``).
            Defaults to SNAP 13 and earlier.
        """

    def to_mermaid(self, *, snap_version: tuple[int, ...] | None = None) -> str:
        """Mermaid ``flowchart LR`` diagram source for this config's graph.

        Parameters
        ----------
        snap_version
            SNAP version the graph will run under, as for ``to_xml``. No
            diagram varies with it, but a config whose options need SNAP 14
            cannot build its graph without it.
        """
        return xml_to_mermaid(self.to_xml(snap_version=snap_version))


class BackscatterConfig(GeocodingConfig):
    """Settings shared by the backscatter pipelines, RTC and GTC.

    Abstract: pass ``RTCConfig`` for terrain-flattened gamma-0 backscatter or
    ``GTCConfig`` for beta-0 backscatter that is geocoded only. Inherits every
    parameter of ``GeocodingConfig``.

    Parameters
    ----------
    multilook
        ``(n_rg_looks, n_az_looks)`` applied after ``TOPSAR-Deburst``, before terrain
        flattening; ``None`` skips the ``Multilook`` step. Multilooking here
        cuts runtime sharply, because the reduced raster passes through both
        terrain flattening and terrain correction.

        SNAP's ``grSquarePixel`` option, which derives the azimuth looks from
        the range looks and the local geometry and discards the requested
        ``n_az_looks`` is pinned off. Therefore, the looks given are the looks applied.
    """

    multilook: tuple[_PositiveInt, _PositiveInt] | None = (5, 1)

    @property
    @abstractmethod
    def output_bands(self) -> list[str]:
        """Names assigned to the output GeoTIFF bands by the SNAP client."""

    @override
    def to_xml(self, *, snap_version: tuple[int, ...] | None = None) -> str:
        """Build the backscatter graph XML for this config.

        Takes ``snap_version`` for a uniform signature; only the coherence
        graph has options that depend on it, so it does not change this one.
        """
        return build_backscatter_graph(self)


class RTCConfig(BackscatterConfig):
    """Configuration for terrain-flattened backscatter (RTC).

    Gamma-0 backscatter, radiometrically terrain corrected: called Normalised
    Radar Backscatter (NRB) in CEOS-ARD and RTC-S1 by OPERA. The product also
    carries the local scattering area that terrain flattening divides by.
    Inherits every parameter of ``BackscatterConfig``.

    Parameters
    ----------
    dem_oversampling
        DEM oversampling factor used by ``Terrain-Flattening``. Must be positive.
    additional_overlap
        Additional overlap percentage used by ``Terrain-Flattening`` when
        searching for terrain contributions outside the tile.
    """

    dem_oversampling: _PositiveFloat = 1.0
    additional_overlap: float = Field(default=0.2, ge=0.0)

    @property
    @override
    def output_bands(self) -> list[str]:
        """Names assigned to the output GeoTIFF bands by the SNAP client."""
        return ["gamma0", "scattering_area"]


class GTCConfig(BackscatterConfig):
    """Configuration for geocoded backscatter (GTC).

    Beta-0 backscatter, geometrically terrain corrected only: geocoded, but not
    terrain-flattened. Takes exactly the parameters of ``BackscatterConfig``.
    """

    @property
    @override
    def output_bands(self) -> list[str]:
        """Names assigned to the output GeoTIFF bands by the SNAP client."""
        return ["beta0"]


class CoherenceConfig(GeocodingConfig):
    """Configuration for the coherence pipeline.

    Inherits every parameter of ``GeocodingConfig``.

    No ``Multilook`` step is applied, unlike ``BackscatterConfig``. It
    would save no runtime, since it can only sit after the coregistration and
    phase-removal stages that dominate this graph, and placing it after the
    ``Coherence`` operator averages coherence magnitudes and so cannot reduce
    the small-window estimator bias. Use ``window`` to set the looks.

    Parameters
    ----------
    window
        Coherence estimation window ``(n_rg, n_az)`` in single-look samples.
        This is the only control over the number of looks entering the
        estimator. SNAP's ``squarePixel`` option, which derives the azimuth
        window from the range window and the scene geometry, is pinned off, so
        the window given is the window applied.
    subtract_flat_earth_phase
        Remove the flat-earth phase before estimating coherence.
    subtract_topographic_phase
        Remove the topographic phase using ``dem``.
    correct_coherence_bias
        Remove the positive bias a finite estimation window leaves in the
        coherence magnitude. Requires SNAP 14; building the graph for an older
        or undetermined version raises ``ConfigurationError``. Off by default,
        unlike SNAP's own default, so that the estimator stays the one the
        window describes and stays comparable across SNAP versions.
    remove_local_phase_ramp
        Let SNAP estimate and remove a local phase ramp within each estimation
        window before averaging, which recovers coherence lost to residual
        fringes. Requires SNAP 14, as above.
    """

    window: tuple[_PositiveInt, _PositiveInt] = (25, 5)
    subtract_flat_earth_phase: bool = True
    subtract_topographic_phase: bool = True
    correct_coherence_bias: bool = False
    remove_local_phase_ramp: bool = False

    @property
    def output_bands(self) -> list[str]:
        """Names assigned to the output GeoTIFF bands by the SNAP client."""
        return ["coherence"]

    @override
    def to_xml(self, *, snap_version: tuple[int, ...] | None = None) -> str:
        """Build the coherence graph XML for this config.

        Raises
        ------
        ConfigurationError
            If ``correct_coherence_bias`` or ``remove_local_phase_ramp`` is
            enabled but ``snap_version`` is older than 14 or unknown.
        """
        return build_coherence_graph(self, snap_version=snap_version)


__all__ = [
    "BackscatterConfig",
    "CoherenceConfig",
    "DEMResamplingMethod",
    "GTCConfig",
    "GeocodingConfig",
    "RTCConfig",
    "ResamplingMethod",
    "build_backscatter_graph",
    "build_coherence_graph",
    "supports_coherence_options",
    "xml_to_mermaid",
]
