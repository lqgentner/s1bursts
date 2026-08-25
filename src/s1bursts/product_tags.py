"""GeoTIFF metadata for the products this package writes.

Pure builders: each returns the dataset tags, band descriptions, per-band tags
and band units for one raster, and performs no I/O.

The vocabulary follows the CEOS-ARD Product Family Specification for Synthetic
Aperture Radar, version 1.3.1, so that a reader can look a tag up in the
accompanying metadata specification. No product-type claim is made and no
conformance is asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import version as _distribution_version
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from s1bursts.product import (
    RTCLRWQA,
    CoherenceProduct,
    CohLRWComposite,
    CohLRWQA,
    CohPassComposite,
    CohPassQA,
    GTCProduct,
    InterferogramProduct,
    QaMetric,
    RTCLRWComposite,
    RTCPassComposite,
    RTCPassQA,
    RTCProduct,
)
from s1bursts.snap_graph import (
    BackscatterConfig,
    CoherenceConfig,
    GeocodingConfig,
    InterferogramConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.typing import NDArray
    from pandas import Timestamp

    from s1bursts.product import BackscatterProduct, GeocodedBurst, MGRSTileProduct


SPECIFICATION_URL = "https://ceos.org/ard/files/PFS/SAR/v1.3.1/CEOS-ARD_PFS_SAR_v1.3.1.pdf"
"""CEOS-ARD Product Family Specification for SAR, version 1.3.1."""

LRW_REFERENCE = "https://doi.org/10.1109/TGRS.2021.3055562"
"""Small et al. (2022), the local-resolution-weighting method of item 1.7.24."""

# The specification asks for the conversion to be declared, not applied: items
# 3.1, 2.21 and 2.22 each note that "transformation to the logarithmic decibel
# scale is not required or desired as this step can be completed by the user".
CONVERSION_EQ = "dB = 10*log10(DN)"

_ALIGNMENT_COMPOSITE = (
    "Metadata aligned with CEOS-ARD PFS SAR v1.3.1 Composite Backscatter [CB]. "
    "Not a conformance claim."
)
# The specification defines no composite coherence product, so that one names
# the two items it borrows from instead of claiming either.
_ALIGNMENT_COHERENCE = (
    "Metadata aligned with CEOS-ARD PFS SAR v1.3.1: Composite Backscatter [CB] for the "
    "compositing vocabulary, InSAR for the coherence quantity. The specification defines "
    "no composite coherence product. Not a conformance claim."
)
# A single burst is not a composite, so it must not carry the [CB] wording.
_ALIGNMENT_BURST = "Metadata aligned with CEOS-ARD PFS SAR v1.3.1. Not a conformance claim."
_ALIGNMENT_DECIBEL = (
    " Stored in decibels, which departs from the specification's stated preference "
    "for leaving the conversion to the user."
)

_PIXEL_CONVENTION = "pixel centre"

# Band unit for linear backscatter. The PFS names the two expression
# conventions "linear amplitude" and "linear power" (items 2.21, 2.22, 3.1), so
# the stored form is said in the specification's own words. The tag value stays
# the metadata spec's enum, which is [Amplitude, Power]; the two documents
# disagree here and the parameter list governs the parameter.
_LINEAR_POWER_UNIT = "linear power"
_DECIBEL_UNIT = "dB"

# This package orchestrates the run and defines the vocabulary below, so the
# product records which version wrote it. That is not cosmetic: the definition
# of QA-NUM changed between releases, and nothing else in the file says which
# of the two a reader is holding.
_PIPELINE_SOFTWARE = "s1bursts"
_PIPELINE_URL = "https://github.com/lqgentner/s1bursts"

# Processing levels follow the usual EOSDIS ladder: the SLC input is L1, a
# geocoded per-burst product is L2 (as OPERA's own RTC-S1 declares), and a
# composite gridded in space and time is L3 (as ESA's Sentinel-1 monthly
# mosaic, the closest analogue to these).
_LEVEL_SLC = "L1"
_LEVEL_BURST = "L2"
_LEVEL_COMPOSITE = "L3"

# SNAP's Coherence operator averages over an unweighted moving window.
_COHERENCE_WEIGHTING_SHAPE = "Rectangular"
_PHASE_QUALITY_METHOD = "Coherence"
_PHASE_QUALITY_REFERENCE = (
    "Magnitude of the complex correlation coefficient over the estimation window, "
    "as computed by the ESA SNAP Coherence operator."
)

UNKNOWN = "N/A"
"""Value of a provenance tag that could not be determined.

Written rather than left out, so that a reader can tell a field that was
considered and came back empty from one this package never wrote.
"""

# Provenance a composite inherits from the per-burst products it was built
# from, describing how those inputs were made. Always present on a composite,
# as ``UNKNOWN`` where the inputs do not supply a single answer.
_INPUT_KEYS = ("DEM", "PROCESSING_SOFTWARE", "PROCESSING_SOFTWARE_VERSION")
_COHERENCE_INPUT_KEYS = (
    "PHASE_QUALITY_METHOD",
    "PHASE_QUALITY_REFERENCE",
    "COHERENCE_WINDOW_RANGE",
    "COHERENCE_WINDOW_AZIMUTH",
    "COHERENCE_WEIGHTING_SHAPE",
)

type Aggregation = Literal["mean", "median"] | float


@dataclass(frozen=True)
class RasterMetadata:
    """Everything written into one output raster besides its pixels.

    Parameters
    ----------
    tags : dict[str, str]
        Dataset-level tags.
    band_names : list[str]
        Band descriptions, one per band.
    units : list[str]
        Band unit strings, one per band. Empty for a dimensionless band.
    band_tags : list[dict[str, str]]
        Per-band tags, one mapping per band.
    """

    tags: dict[str, str]
    band_names: list[str]
    units: list[str]
    band_tags: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject a band count that the three per-band lists disagree on."""
        counts = {len(self.band_names), len(self.units)}
        if self.band_tags:
            counts.add(len(self.band_tags))
        if len(counts) != 1:
            msg = (
                f"band_names, units and band_tags must describe the same number of bands; "
                f"got {len(self.band_names)}, {len(self.units)}, {len(self.band_tags)}."
            )
            raise ValueError(msg)


# ----------------------------------------------------------------------
# Shared pieces
# ----------------------------------------------------------------------


def _date(value: Timestamp) -> str:
    return f"{value:%Y-%m-%d}"


def _pipeline() -> dict[str, str]:
    """Identify the package that wrote the file, stamped fresh on every write."""
    return {
        "PIPELINE_SOFTWARE": _PIPELINE_SOFTWARE,
        "PIPELINE_SOFTWARE_VERSION": _distribution_version("s1bursts"),
        "PIPELINE_SOFTWARE_URL": _PIPELINE_URL,
    }


def _alignment(statement: str, *, decibel: bool = False) -> dict[str, str]:
    """Build the two tags that state the relationship to the specification."""
    return {
        "CEOS_ARD_ALIGNMENT": statement + (_ALIGNMENT_DECIBEL if decibel else ""),
        "CEOS_ARD_SPECIFICATION_URL": SPECIFICATION_URL,
    }


def _backscatter(*, gamma0: bool, decibel: bool) -> dict[str, str]:
    """Build the measurement and scaling tags for a backscatter raster (spec 3.1, 3.2).

    ``BACKSCATTER_CONVENTION`` is enumerated ``[Amplitude, Power]`` and
    describes the stored quantity, so decibel data gets neither it nor the
    conversion equation, which would invite a second conversion.
    """
    tags = {"BACKSCATTER_MEASUREMENT": "Gamma-0" if gamma0 else "Beta-0"}
    if decibel:
        tags["BACKSCATTER_SCALING"] = "decibel"
        tags["BACKSCATTER_CONVERSION_APPLIED"] = "10*log10(power)"
    else:
        tags["BACKSCATTER_CONVENTION"] = "Power"
        # Item 3.2 has no parameter of its own; the metadata spec puts it here.
        tags["BACKSCATTER_CONVERSION_EQ"] = CONVERSION_EQ
    return tags


def _aggregation_string(aggregation: Aggregation) -> str:
    return aggregation if isinstance(aggregation, str) else f"quantile={aggregation:g}"


def _compositing_parameters(
    aggregation: Aggregation,
    clip_max_resolution: float | None = None,
) -> str:
    """Render the arguments the composite was actually built with.

    Built from the call arguments rather than from a configuration object, so
    it cannot describe a run that did not happen.
    """
    parts = [f"aggregation={_aggregation_string(aggregation)}"]
    if clip_max_resolution is not None:
        parts.append(f"clip_max_resolution={clip_max_resolution:g}")
    else:
        parts.append("clip_max_resolution=none")
    return ";".join(parts)


def _tile(
    product: MGRSTileProduct,
    *,
    family: str,
    coherence: bool = False,
    decibel: bool = False,
) -> dict[str, str]:
    """Identity, date window and polarization, shared by every composite."""
    return {
        **_alignment(
            _ALIGNMENT_COHERENCE if coherence else _ALIGNMENT_COMPOSITE,
            decibel=decibel,
        ),
        **_pipeline(),
        "PRODUCT_TYPE": product.product_type,
        "PRODUCT_FAMILY": family,
        "PRODUCT_LEVEL": _LEVEL_COMPOSITE,
        "SOURCE_DATA_PRODUCT_LEVEL": _LEVEL_BURST,
        "MGRS_TILE": product.mgrs_code,
        "COMPOSITE_START_DATE": _date(product.start_date),
        "COMPOSITE_END_DATE": _date(product.end_date),
        "POLARIZATION": product.polarization,
        "PIXEL_COORDINATE_CONVENTION": _PIXEL_CONVENTION,
    }


def _lrw(aggregation: Aggregation, clip_max_resolution: float | None) -> dict[str, str]:
    """Compositing-method tags for a locally resolution weighted product (spec 1.7.24)."""
    return {
        "COMPOSITING_METHOD": "Local resolution weighting",
        "COMPOSITING_METHOD_REFERENCE": LRW_REFERENCE,
        "COMPOSITING_PARAMETERS": _compositing_parameters(aggregation, clip_max_resolution),
    }


def _unweighted(aggregation: Aggregation) -> dict[str, str]:
    """Compositing-method tags for a pass composite, which applies no weighting."""
    return {
        "COMPOSITING_METHOD": "Unweighted temporal aggregation",
        "COMPOSITING_PARAMETERS": f"aggregation={_aggregation_string(aggregation)}",
    }


def _inherited(
    source_tags: Mapping[str, str] | None,
    keys: Iterable[str],
    *,
    required: Iterable[str] = (),
) -> dict[str, str]:
    """Forward the named keys from the per-burst inputs' tags.

    A key in ``required`` is always emitted, as :data:`UNKNOWN` where the
    inputs did not supply it — they disagreed, or never carried it. Anything
    else is simply left out when absent, because it may not apply to this
    product at all.
    """
    source_tags = source_tags or {}
    inherited = {key: source_tags[key] for key in keys if key in source_tags}
    for key in required:
        inherited.setdefault(key, UNKNOWN)
    return inherited


def coherence_statistics(values: NDArray[np.floating[Any]]) -> dict[str, str]:
    """Mean and standard deviation of a coherence raster ([InSAR] item 3.8).

    Returns an empty mapping when no pixel is finite, so that an empty tile
    carries no statistics rather than a NaN.

    Parameters
    ----------
    values : numpy.ndarray
        Coherence values; non-finite entries are ignored.

    Returns
    -------
    dict[str, str]
        ``COHERENCE_AVERAGE`` and ``COHERENCE_STD``, or empty.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    return {
        "COHERENCE_AVERAGE": f"{finite.mean():.6g}",
        "COHERENCE_STD": f"{finite.std():.6g}",
    }


# ----------------------------------------------------------------------
# Composites
# ----------------------------------------------------------------------


def rtc_lrw_metadata(
    product: RTCLRWComposite,
    *,
    aggregation: Aggregation,
    clip_max_resolution: float | None,
    n_observations: int,
    decibel: bool,
    source_tags: Mapping[str, str] | None = None,
) -> RasterMetadata:
    """Metadata for a locally resolution weighted RTC composite.

    Parameters
    ----------
    product : RTCLRWComposite
        The product being written.
    aggregation : {"mean", "median"} or float
        Temporal aggregation actually applied.
    clip_max_resolution : float or None
        Weight clip actually applied.
    n_observations : int
        Number of orbit-date observations entering the composite (spec 2.21).
    decibel : bool
        Whether the stored values are decibels rather than linear power.
    source_tags : Mapping[str, str] or None, optional
        Tags of one of the per-burst inputs, from which the processing
        provenance is inherited.

    Returns
    -------
    RasterMetadata
        Tags, band description and unit for the composite.
    """
    tags = {
        **_tile(product, family="Composite Backscatter", decibel=decibel),
        **_backscatter(gamma0=True, decibel=decibel),
        **_lrw(aggregation, clip_max_resolution),
        "CONTRIBUTING_OBSERVATIONS": str(n_observations),
        "LAYER_NAME": "gamma0",
        "LAYER_DESCRIPTION": (
            "Locally resolution weighted gamma-0 backscatter composite, "
            "weighted by the inverse local contributing area."
        ),
        **_inherited(source_tags, _INPUT_KEYS, required=_INPUT_KEYS),
    }
    return RasterMetadata(
        tags=tags,
        band_names=[product.polarization],
        units=[_DECIBEL_UNIT if decibel else _LINEAR_POWER_UNIT],
    )


def coh_lrw_metadata(
    product: CohLRWComposite,
    *,
    aggregation: Aggregation,
    clip_max_resolution: float | None,
    n_observations: int,
    statistics: Mapping[str, str] | None = None,
    source_tags: Mapping[str, str] | None = None,
) -> RasterMetadata:
    """Metadata for a locally resolution weighted coherence composite.

    The specification defines no composite coherence product: the quantity
    vocabulary comes from [InSAR] items 3.8 and 1.7.18, the compositing
    vocabulary from [CB] item 1.7.24, and neither product type is claimed. No
    conversion equation is emitted, because coherence is a dimensionless
    magnitude on [0, 1] rather than a power quantity.

    Parameters
    ----------
    product : CohLRWComposite
        The product being written.
    aggregation : {"mean", "median"} or float
        Temporal aggregation actually applied.
    clip_max_resolution : float or None
        Weight clip actually applied.
    n_observations : int
        Number of orbit-date observations entering the composite (spec 2.21).
    statistics : Mapping[str, str] or None, optional
        Coherence statistics from :func:`coherence_statistics`.
    source_tags : Mapping[str, str] or None, optional
        Tags of one of the per-burst coherence inputs. The estimation window
        originates there and reaches the composite by no other route.

    Returns
    -------
    RasterMetadata
        Tags, band description and unit for the composite.
    """
    tags = {
        **_tile(product, family="Composite Coherence", coherence=True),
        "MEASUREMENT_TYPE": "Coherence",
        "TEMPORAL_BASELINE_DAYS": str(product.baseline_days),
        # Small et al. define LRW for backscatter power; weighting coherence by
        # local contributing area is this package's extension, so say so rather
        # than let the reference imply a method the paper does not cover.
        **(
            _lrw(aggregation, clip_max_resolution)
            | {
                "COMPOSITING_METHOD": (
                    f"Local resolution weighting (Small et al. 2022) applied to "
                    f"{product.baseline_days}-day repeat-pass coherence"
                ),
            }
        ),
        "CONTRIBUTING_OBSERVATIONS": str(n_observations),
        "LAYER_NAME": "coherence",
        "LAYER_DESCRIPTION": (
            "Locally resolution weighted repeat-pass interferometric coherence "
            "composite, weighted by the inverse local contributing area."
        ),
        **dict(statistics or {}),
        **_inherited(
            source_tags,
            (*_INPUT_KEYS, *_COHERENCE_INPUT_KEYS),
            required=_INPUT_KEYS,
        ),
    }
    return RasterMetadata(tags=tags, band_names=[product.polarization], units=[""])


def rtc_pass_metadata(
    product: RTCPassComposite,
    *,
    aggregation: Aggregation,
    n_observations: int,
    decibel: bool,
    source_tags: Mapping[str, str] | None = None,
) -> RasterMetadata:
    """Metadata for an unweighted RTC composite of a single orbit-pass direction.

    Parameters
    ----------
    product : RTCPassComposite
        The product being written.
    aggregation : {"mean", "median"} or float
        Temporal aggregation actually applied.
    n_observations : int
        Number of orbit-date observations entering the composite.
    decibel : bool
        Whether the stored values are decibels rather than linear power.
    source_tags : Mapping[str, str] or None, optional
        Tags of one of the per-burst inputs.

    Returns
    -------
    RasterMetadata
        Tags, band description and unit for the composite.
    """
    tags = {
        **_tile(product, family="Composite Backscatter", decibel=decibel),
        **_backscatter(gamma0=True, decibel=decibel),
        **_unweighted(aggregation),
        "ORBIT_DIRECTION": product.orbit_direction,
        "CONTRIBUTING_OBSERVATIONS": str(n_observations),
        "LAYER_NAME": "gamma0",
        "LAYER_DESCRIPTION": (
            "Unweighted gamma-0 backscatter composite of a single orbit-pass direction."
        ),
        **_inherited(source_tags, _INPUT_KEYS, required=_INPUT_KEYS),
    }
    return RasterMetadata(
        tags=tags,
        band_names=[product.polarization],
        units=[_DECIBEL_UNIT if decibel else _LINEAR_POWER_UNIT],
    )


def coh_pass_metadata(
    product: CohPassComposite,
    *,
    aggregation: Aggregation,
    n_observations: int,
    statistics: Mapping[str, str] | None = None,
    source_tags: Mapping[str, str] | None = None,
) -> RasterMetadata:
    """Metadata for an unweighted coherence composite of a single orbit-pass direction.

    Parameters
    ----------
    product : CohPassComposite
        The product being written.
    aggregation : {"mean", "median"} or float
        Temporal aggregation actually applied.
    n_observations : int
        Number of orbit-date observations entering the composite.
    statistics : Mapping[str, str] or None, optional
        Coherence statistics from :func:`coherence_statistics`.
    source_tags : Mapping[str, str] or None, optional
        Tags of one of the per-burst coherence inputs.

    Returns
    -------
    RasterMetadata
        Tags, band description and unit for the composite.
    """
    tags = {
        **_tile(product, family="Composite Coherence", coherence=True),
        "MEASUREMENT_TYPE": "Coherence",
        "TEMPORAL_BASELINE_DAYS": str(product.baseline_days),
        **_unweighted(aggregation),
        "ORBIT_DIRECTION": product.orbit_direction,
        "CONTRIBUTING_OBSERVATIONS": str(n_observations),
        "LAYER_NAME": "coherence",
        "LAYER_DESCRIPTION": (
            "Unweighted repeat-pass interferometric coherence composite of a single "
            "orbit-pass direction."
        ),
        **dict(statistics or {}),
        **_inherited(
            source_tags,
            (*_INPUT_KEYS, *_COHERENCE_INPUT_KEYS),
            required=_INPUT_KEYS,
        ),
    }
    return RasterMetadata(tags=tags, band_names=[product.polarization], units=[""])


def qa_metadata(
    product: RTCLRWQA | CohLRWQA | RTCPassQA | CohPassQA,
    *,
    aggregation: Aggregation,
    n_observations: int,
    clip_max_resolution: float | None = None,
    cqm_decibel: bool = False,
    source_tags: Mapping[str, str] | None = None,
) -> RasterMetadata:
    """Metadata for a QA auxiliary: contributing observations or composite quality map.

    Neither layer carries a backscatter convention or a conversion equation: a
    per-pixel observation count has no dB scaling, and the CQM declares its own.

    Parameters
    ----------
    product : RTCLRWQA or CohLRWQA or RTCPassQA or CohPassQA
        The QA product being written.
    aggregation : {"mean", "median"} or float
        Temporal aggregation actually applied to the parent composite.
    n_observations : int
        Number of orbit-date observations entering the composite (spec 2.21).
    clip_max_resolution : float or None, optional
        Weight clip actually applied; ignored for a pass composite's QA, which
        applies no weighting.
    cqm_decibel : bool, default False
        Whether a ``QA-CQM`` layer stores decibels rather than linear values.
    source_tags : Mapping[str, str] or None, optional
        Tags of one of the per-burst inputs.

    Returns
    -------
    RasterMetadata
        Tags, band description and unit for the QA layer.
    """
    # Pass composites carry only a contributing-observations layer, and are
    # unweighted, so neither the CQM branch nor the LRW tags apply to them.
    weighted = isinstance(product, RTCLRWQA | CohLRWQA)
    coherence = isinstance(product, CohLRWQA | CohPassQA)
    metric = product.qa_metric if isinstance(product, RTCLRWQA | CohLRWQA) else QaMetric.NUM

    tags = {
        **_tile(
            product,
            family="Composite Coherence" if coherence else "Composite Backscatter",
            coherence=coherence,
            decibel=cqm_decibel and metric is QaMetric.CQM,
        ),
        **(_lrw(aggregation, clip_max_resolution) if weighted else _unweighted(aggregation)),
        "CONTRIBUTING_OBSERVATIONS": str(n_observations),
        **_inherited(source_tags, _INPUT_KEYS, required=_INPUT_KEYS),
    }
    if isinstance(product, RTCPassQA | CohPassQA):
        tags["ORBIT_DIRECTION"] = product.orbit_direction

    if metric is QaMetric.NUM:
        counted = "carrying a non-zero resolution weight" if weighted else "holding a finite value"
        tags |= {
            # Deliberately not called a data mask: PFS item 2.2 asks for three
            # distinguishable states, and a zero count collapses "invalid" and
            # "no data" into one.
            "QA_LAYER": "Contributing Observations Image",
            "LAYER_NAME": "num_obs",
            "LAYER_DESCRIPTION": (
                f"Number of observations {counted} in the composite. 0 = no data."
            ),
        }
        return RasterMetadata(tags=tags, band_names=["num_obs"], units=[""])

    tags |= {
        "QA_LAYER": "Composite Quality Map Image",
        "LAYER_NAME": "cqm",
        "LAYER_DESCRIPTION": (
            "Achieved local resolution of the composite, 1/sum(W_i * A_i); "
            "dimensionless, 1.0 = flat reference facet."
        ),
    }
    if cqm_decibel:
        tags["CQM_SCALING"] = "decibel"
        tags["CQM_CONVERSION_APPLIED"] = "10*log10(cqm)"
    else:
        # <CQMdBScaling>: the declaration of a conversion, on linear data.
        tags["CQM_CONVERSION_EQ"] = CONVERSION_EQ
    return RasterMetadata(
        tags=tags, band_names=["cqm"], units=[_DECIBEL_UNIT if cqm_decibel else ""]
    )


# ----------------------------------------------------------------------
# Per-burst products
# ----------------------------------------------------------------------


def _burst(
    product: GeocodedBurst,
    *,
    config: GeocodingConfig,
    snap_version: tuple[int, ...] | None,
) -> dict[str, str]:
    """Identity and geocoding tags shared by every per-burst product."""
    tags = {
        **_alignment(_ALIGNMENT_BURST),
        **_pipeline(),
        "PRODUCT_TYPE": product.product_type,
        "PRODUCT_LEVEL": _LEVEL_BURST,
        "SOURCE_DATA_PRODUCT_LEVEL": _LEVEL_SLC,
        "BURST_ID": str(product.burst_id),
        "SUB_SWATH": product.sub_swath,
        "PIXEL_COORDINATE_CONVENTION": _PIXEL_CONVENTION,
        "PROCESSING_SOFTWARE": "ESA SNAP",
        "GRID_SPACING": f"{config.grid_spacing:g} m",
        "DEM": config.dem if isinstance(config.dem, str) else config.dem_name or UNKNOWN,
    }
    tags["PROCESSING_SOFTWARE_VERSION"] = (
        ".".join(str(part) for part in snap_version) if snap_version is not None else UNKNOWN
    )
    return tags


def backscatter_metadata(
    product: BackscatterProduct,
    config: BackscatterConfig,
    *,
    snap_version: tuple[int, ...] | None = None,
) -> RasterMetadata:
    """Metadata for a per-burst RTC or GTC product.

    Parameters
    ----------
    product : RTCProduct or GTCProduct
        The product being written.
    config : BackscatterConfig
        Configuration the graph was built from.
    snap_version : tuple of int or None, optional
        Version of the SNAP installation that ran the graph, from
        :func:`s1bursts.snap_client.snap_version`. Omitted from the tags
        when unknown.

    Returns
    -------
    RasterMetadata
        Tags, band descriptions and units for the product.

    Raises
    ------
    ValueError
        If the product and the config disagree on terrain flattening, which
        would pair a band count from one with units from the other.
    """
    gamma0 = product.is_terrain_flattened
    if gamma0 is not config.flatten_terrain:
        msg = (
            f"{type(product).__name__} and BackscatterConfig disagree on terrain "
            f"flattening: product says {gamma0}, config says {config.flatten_terrain}."
        )
        raise ValueError(msg)
    tags = {
        **_burst(product, config=config, snap_version=snap_version),
        **_backscatter(gamma0=gamma0, decibel=False),
        "POLARIZATION": product.polarization,
        "ACQUISITION_DATE": _date(product.date),
        "TERRAIN_CORRECTION": "Radiometric and geometric" if gamma0 else "Geometric",
    }
    if config.multilook is not None:
        looks_rg, looks_az = config.multilook
        tags["MULTILOOK_RANGE"] = str(looks_rg)
        tags["MULTILOOK_AZIMUTH"] = str(looks_az)

    if not gamma0:
        return RasterMetadata(
            tags=tags,
            band_names=list(config.output_bands),
            units=[_LINEAR_POWER_UNIT],
        )

    # Band 2 is the local contributing area terrain flattening divides by, the
    # layer CEOS-ARD calls scatteringArea (item 2.3) and OPERA calls
    # rtc_anf_gamma0_to_beta0.
    return RasterMetadata(
        tags=tags,
        band_names=list(config.output_bands),
        units=[_LINEAR_POWER_UNIT, ""],
        band_tags=[
            {"LAYER_NAME": "gamma0"},
            {
                "LAYER_NAME": "scatteringArea",
                "LAYER_DESCRIPTION": (
                    "Local contributing area relative to a flat reference facet; "
                    "1.0 = flat. The inverse is the local resolution weight."
                ),
            },
        ],
    )


def coherence_metadata(
    product: CoherenceProduct,
    config: CoherenceConfig,
    *,
    snap_version: tuple[int, ...] | None = None,
) -> RasterMetadata:
    """Metadata for a per-burst coherence product.

    The estimation window ([InSAR] item 1.7.18) exists only in ``config``, so
    these tags are the only route by which it reaches a composite built from
    this product.

    Parameters
    ----------
    product : CoherenceProduct
        The product being written.
    config : CoherenceConfig
        Configuration the graph was built from.
    snap_version : tuple of int or None, optional
        Version of the SNAP installation that ran the graph, from
        :func:`s1bursts.snap_client.snap_version`. Omitted from the tags
        when unknown.

    Returns
    -------
    RasterMetadata
        Tags, band description and unit for the product.
    """
    win_rg, win_az = config.window
    tags = {
        **_burst(product, config=config, snap_version=snap_version),
        "MEASUREMENT_TYPE": "Coherence",
        "POLARIZATION": product.polarization,
        "DATE_PRIMARY": _date(product.date_primary),
        "DATE_SECONDARY": _date(product.date_secondary),
        "TEMPORAL_BASELINE_DAYS": str((product.date_secondary - product.date_primary).days),
        "PHASE_QUALITY_METHOD": _PHASE_QUALITY_METHOD,
        "PHASE_QUALITY_REFERENCE": _PHASE_QUALITY_REFERENCE,
        "COHERENCE_WINDOW_RANGE": str(win_rg),
        "COHERENCE_WINDOW_AZIMUTH": str(win_az),
        "COHERENCE_WEIGHTING_SHAPE": _COHERENCE_WEIGHTING_SHAPE,
    }
    return RasterMetadata(tags=tags, band_names=list(config.output_bands), units=[""])


def interferogram_metadata(
    product: InterferogramProduct,
    config: InterferogramConfig,
    *,
    snap_version: tuple[int, ...] | None = None,
) -> RasterMetadata:
    """Metadata for a per-burst interferogram product.

    Parameters
    ----------
    product : InterferogramProduct
        The product being written.
    config : InterferogramConfig
        Configuration the graph was built from.
    snap_version : tuple of int or None, optional
        Version of the SNAP installation that ran the graph, from
        :func:`s1bursts.snap_client.snap_version`. Omitted from the tags
        when unknown.

    Returns
    -------
    RasterMetadata
        Tags, band descriptions and units for the product.
    """
    tags = {
        **_burst(product, config=config, snap_version=snap_version),
        "MEASUREMENT_TYPE": "Interferogram",
        "POLARIZATION": product.polarization,
        "DATE_PRIMARY": _date(product.date_primary),
        "DATE_SECONDARY": _date(product.date_secondary),
        "TEMPORAL_BASELINE_DAYS": str((product.date_secondary - product.date_primary).days),
        "FLAT_EARTH_PHASE_REMOVED": str(config.subtract_flat_earth_phase).lower(),
        "TOPOGRAPHIC_PHASE_REMOVED": str(config.subtract_topographic_phase).lower(),
    }
    # The two intensities are power; the complex parts are not a power quantity
    # on their own, so they get no unit and no conversion equation.
    return RasterMetadata(
        tags=tags,
        band_names=list(config.output_bands),
        units=["", "", _LINEAR_POWER_UNIT, _LINEAR_POWER_UNIT],
    )


def burst_metadata(
    product: RTCProduct | GTCProduct | CoherenceProduct | InterferogramProduct,
    config: BackscatterConfig | CoherenceConfig | InterferogramConfig,
    *,
    snap_version: tuple[int, ...] | None = None,
) -> RasterMetadata:
    """Metadata for any per-burst product, dispatched on its config.

    Parameters
    ----------
    product : RTCProduct or GTCProduct or CoherenceProduct or InterferogramProduct
        The product being written.
    config : BackscatterConfig or CoherenceConfig or InterferogramConfig
        Configuration the graph was built from.
    snap_version : tuple of int or None, optional
        Version of the SNAP installation that ran the graph, from
        :func:`s1bursts.snap_client.snap_version`. Omitted from the tags
        when unknown.

    Returns
    -------
    RasterMetadata
        Tags, band descriptions and units for the product.

    Raises
    ------
    TypeError
        If the product and config do not describe the same kind of product.
    """
    match product, config:
        case ((RTCProduct() | GTCProduct()), BackscatterConfig()):
            return backscatter_metadata(product, config, snap_version=snap_version)
        case CoherenceProduct(), CoherenceConfig():
            return coherence_metadata(product, config, snap_version=snap_version)
        case InterferogramProduct(), InterferogramConfig():
            return interferogram_metadata(product, config, snap_version=snap_version)
        case _:
            msg = (
                f"cannot build metadata for {type(product).__name__} from {type(config).__name__}."
            )
            raise TypeError(msg)


__all__ = [
    "CONVERSION_EQ",
    "LRW_REFERENCE",
    "SPECIFICATION_URL",
    "UNKNOWN",
    "RasterMetadata",
    "backscatter_metadata",
    "burst_metadata",
    "coh_lrw_metadata",
    "coh_pass_metadata",
    "coherence_metadata",
    "coherence_statistics",
    "interferogram_metadata",
    "qa_metadata",
    "rtc_lrw_metadata",
    "rtc_pass_metadata",
]
