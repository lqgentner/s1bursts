"""Tests for ``s1bursts.snap_graph`` — dynamic SNAP graph generation."""

from pathlib import Path
from xml.etree.ElementTree import canonicalize

from lxml.etree import fromstring, tostring
from pydantic import ValidationError
import pytest

from s1bursts.snap_graph import (
    BackscatterConfig,
    CoherenceConfig,
    GeocodingConfig,
    InterferogramConfig,
    xml_to_mermaid,
)

_SWISSALTI3D = Path(
    "/mnt/deep-glacier-mapping/data/dem/swissalti3d/interim/swissalti3d-4326-ellipsoid.tif",
)

FIXTURES = Path(__file__).parent / "fixtures" / "snap_graphs"


# ---------------------------------------------------------------------------
# Canonicalization for golden-XML comparisons
# ---------------------------------------------------------------------------


def _normalize(xml_str: str) -> str:
    """Return a canonical XML string suitable for golden comparison.

    Strips whitespace, normalises attribute order, removes cosmetic
    differences that do not affect SNAP behaviour:

    * the ``class="com.bc.ceres.binding.dom.XppDomElement"`` attribute on
      ``<parameters>`` (sometimes present, sometimes absent in
      hand-authored SNAP XML);
    * empty ``<variables/>`` and ``<sourceBands/>`` elements (treated as
      equivalent to absent).
    """
    root = fromstring(xml_str)
    for params in root.iter("parameters"):
        params.attrib.pop("class", None)
    for parent in list(root.iter()):
        for child in list(parent):
            if (
                child.tag in {"variables", "sourceBands"}
                and len(child) == 0
                and not (child.text and child.text.strip())
            ):
                parent.remove(child)
    return canonicalize(tostring(root, encoding="unicode"), strip_text=True)


def _assert_xml_equal(expected_xml: str, actual_xml: str) -> None:
    expected = _normalize(expected_xml)
    actual = _normalize(actual_xml)
    if expected != actual:
        # Show the first divergence to make the diff readable in pytest output.
        for i, (a, b) in enumerate(zip(expected, actual, strict=False)):
            if a != b:
                ctx = 80
                start = max(0, i - ctx)
                msg = (
                    f"XML diverges at offset {i}.\n"
                    f"expected: ...{expected[start : i + ctx]}...\n"
                    f"actual:   ...{actual[start : i + ctx]}..."
                )
                raise AssertionError(msg)
        msg = "XML lengths differ but prefixes match."
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Golden-XML tests (fixtures vs. dynamic builder)
# ---------------------------------------------------------------------------


GOLDEN_CASES = [
    pytest.param(
        "S1_SLCtoRTC.xml",
        BackscatterConfig(),
        id="rtc-default",
    ),
    pytest.param(
        "S1_SLCtoRTC-noML.xml",
        BackscatterConfig(multilook=None),
        id="rtc-no-multilook",
    ),
    pytest.param(
        "S1_SLCtoGTC.xml",
        BackscatterConfig(flatten_terrain=False),
        id="gtc-default",
    ),
    pytest.param(
        "S1_SLCtoCoh-ML.xml",
        CoherenceConfig(),
        id="coherence-magnitude",
    ),
    pytest.param(
        "S1_SLCtoCoh.xml",
        CoherenceConfig(multilook=None),
        id="coherence-magnitude-no-multilook",
    ),
    pytest.param(
        "S1_SLCtoIfg.xml",
        InterferogramConfig(),
        id="interferogram",
    ),
    pytest.param(
        "S1_SLCtoIfg-noTC.xml",
        CoherenceConfig(write_complex=True, multilook=None),
        id="coherence-complex-no-tc",
    ),
    pytest.param(
        "S1_TC.xml",
        GeocodingConfig(),
        id="terrain-correction-only",
    ),
    pytest.param(
        "S1_SLCtoRTC-swissALTI3D.xml",
        BackscatterConfig(dem=_SWISSALTI3D, external_dem_apply_egm=False),
        id="rtc-swissalti3d",
    ),
    pytest.param(
        "S1_SLCtoCoh-swissALTI3D.xml",
        CoherenceConfig(
            dem=_SWISSALTI3D,
            window=(25, 5),
            multilook=None,
            external_dem_apply_egm=False,
        ),
        id="coherence-swissalti3d",
    ),
]


@pytest.mark.parametrize(("fixture", "config"), GOLDEN_CASES)
def test_dynamic_graph_matches_fixture(
    fixture: str,
    config: BackscatterConfig | CoherenceConfig | InterferogramConfig | GeocodingConfig,
) -> None:
    expected_xml = (FIXTURES / fixture).read_text()
    actual_xml = config.to_xml()
    _assert_xml_equal(expected_xml, actual_xml)


# ---------------------------------------------------------------------------
# Output bands
# ---------------------------------------------------------------------------


def test_backscatter_output_bands_rtc() -> None:
    assert BackscatterConfig().output_bands == ["Gamma0", "simulatedImage"]


def test_backscatter_output_bands_gtc() -> None:
    assert BackscatterConfig(flatten_terrain=False).output_bands == ["Beta0"]


def test_coherence_output_bands_magnitude() -> None:
    assert CoherenceConfig().output_bands == ["coh"]


def test_coherence_output_bands_complex() -> None:
    assert CoherenceConfig(write_complex=True).output_bands == ["coh_i", "coh_q"]


def test_interferogram_output_bands() -> None:
    assert InterferogramConfig().output_bands == [
        "ifg_i",
        "ifg_q",
        "intensity_1",
        "intensity_2",
    ]


# ---------------------------------------------------------------------------
# DEM specification: external (Path) flips the demName field
# ---------------------------------------------------------------------------


def test_external_dem_path_emits_external_dem_name() -> None:
    dem = Path("/mnt/glacier/dems/swissalti3d-4326-ellipsoid.tif")
    cfg = BackscatterConfig(dem=dem)
    xml = cfg.to_xml()
    root = fromstring(xml)
    dem_names = [el.text for el in root.iter("demName")]
    dem_files = [el.text for el in root.iter("externalDEMFile")]
    assert all(name == "External DEM" for name in dem_names)
    # Every place that emits demName also emits externalDEMFile pointing at
    # the user-provided path.
    assert all(file == str(dem) for file in dem_files)


def test_builtin_dem_emits_no_external_file() -> None:
    cfg = BackscatterConfig()
    root = fromstring(cfg.to_xml())
    for ext in root.iter("externalDEMFile"):
        assert ext.text is None or ext.text == ""


def test_external_dem_string_path_coerced_to_path() -> None:
    dem = "/mnt/glacier/dems/swissalti3d-4326-ellipsoid.tif"
    cfg = BackscatterConfig(dem=dem)
    root = fromstring(cfg.to_xml())
    dem_names = [el.text for el in root.iter("demName")]
    dem_files = [el.text for el in root.iter("externalDEMFile")]
    assert all(name == "External DEM" for name in dem_names)
    assert all(file == dem for file in dem_files)


# ---------------------------------------------------------------------------
# Multilook off vs on: topology check
# ---------------------------------------------------------------------------


def test_multilook_none_skips_multilook_node() -> None:
    cfg = BackscatterConfig(multilook=None)
    xml = cfg.to_xml()
    root = fromstring(xml)
    operators = {n.find("operator").text for n in root.findall("./node")}
    assert "Multilook" not in operators


def test_multilook_tuple_inserts_multilook_node() -> None:
    cfg = BackscatterConfig(multilook=(7, 2))
    xml = cfg.to_xml()
    root = fromstring(xml)
    ml = root.find('./node[@id="Multilook"]')
    assert ml is not None
    assert ml.find("./parameters/nRgLooks").text == "7"
    assert ml.find("./parameters/nAzLooks").text == "2"


# ---------------------------------------------------------------------------
# Pydantic validation
# ---------------------------------------------------------------------------


def test_coherence_window_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        CoherenceConfig(window=(0, 5))


def test_config_is_frozen() -> None:
    cfg = BackscatterConfig()
    with pytest.raises(ValueError, match="frozen"):
        cfg.flatten_terrain = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Mermaid round-trip
# ---------------------------------------------------------------------------


def test_mermaid_contains_expected_nodes_for_backscatter() -> None:
    cfg = BackscatterConfig()
    mermaid = cfg.to_mermaid()
    assert mermaid.startswith("flowchart LR")
    for op in [
        "Apply-Orbit-File",
        "ThermalNoiseRemoval",
        "Calibration",
        "TOPSAR-Deburst",
        "Multilook",
        "Terrain-Flattening",
        "ReplaceZerosWithNaNs",
        "Terrain-Correction",
    ]:
        assert op in mermaid


def test_mermaid_contains_two_orbit_files_for_coherence() -> None:
    mermaid = CoherenceConfig().to_mermaid()
    assert "Apply-Orbit-File-1" in mermaid
    assert "Apply-Orbit-File-2" in mermaid
    assert "Back-Geocoding" in mermaid


def test_mermaid_passthrough_xml_to_mermaid_function() -> None:
    cfg = BackscatterConfig()
    direct = xml_to_mermaid(cfg.to_xml())
    assert direct == cfg.to_mermaid()
