"""GeoTIFF metadata for the products this package writes.

Pure builders: each returns the dataset tags, band descriptions, per-band tags
and band units for one raster, and performs no I/O.

Where the CEOS-ARD Product Family Specification for Synthetic Aperture Radar,
version 1.3.1, has a name for something, these tags use that name, so that a
reader can look a tag up in the accompanying metadata specification rather than
learn a private dialect. Every product records the specification it borrows
from in ``METADATA_VOCABULARY``.

Borrowing the vocabulary is not conformance, which is neither assessed nor
claimed here.

Tags named after a parameter of the CEOS-ARD SAR Metadata Specification v1.3,
upper-snake-cased, with the PFS item that asks for it::

    Tag                           Metadata parameter          PFS item
    BACKSCATTER_MEASUREMENT       BackscatterMeasurement      3.1
    BACKSCATTER_CONVENTION        BackscatterConvention       3.1
    BACKSCATTER_CONVERSION_EQ     BackscatterConversionEq     3.1 (and 3.2)
    CQM_CONVERSION_EQ             CQMdBScaling                2.22
    CONTRIBUTING_OBSERVATIONS     NbrObservations             2.21
    COMPOSITING_METHOD            CBMethod                    1.7.24
    COMPOSITING_METHOD_REFERENCE  CBReference                 1.7.24
    PIXEL_COORDINATE_CONVENTION   PixelCoordinateConvention   1.7.10
    RELATIVE_ORBIT_NUMBER         RelativeOrbitNumber         1.6.5
    PASS_DIRECTION                PassDirection               1.6.5
    MEASUREMENT_TYPE              Measurements type           [InSAR] 3.8
    COHERENCE_AVERAGE             CoherenceAverage            [InSAR] 3.8
    COHERENCE_STD                 CoherenceStd                [InSAR] 3.8
    PHASE_QUALITY_METHOD          InSARPhaseQualityMethod     [InSAR] 1.7.18
    PHASE_QUALITY_REFERENCE       InSARPhaseQualityReference  [InSAR] 1.7.18
    COHERENCE_WINDOW_RANGE        CoherenceWindowColumn       [InSAR] 1.7.18
    COHERENCE_WINDOW_AZIMUTH      CoherenceWindowLine         [InSAR] 1.7.18
    COHERENCE_WEIGHTING_SHAPE     CoherenceWeightingShape     [InSAR] 1.7.18
    TEMPORAL_BASELINE_DAYS        TemporalBaseline            [InSAR] 1.7.16

Every other tag has no counterpart in the specification and a plain
descriptive name. Among them are the processing settings: every field of the
configuration a per-burst product was made with has a tag of its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import version as _distribution_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from s1bursts.product import (
    CoherenceProduct,
    CohLRWComposite,
    CohPassComposite,
    GTCPassComposite,
    GTCProduct,
    Layer,
    RTCLRWComposite,
    RTCPassComposite,
    RTCProduct,
    _layer_product_type,
)
from s1bursts.snap_graph import (
    BackscatterConfig,
    CoherenceConfig,
    GeocodingConfig,
    GTCConfig,
    RTCConfig,
    _coherence_option_values,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from numpy.typing import NDArray
    from pandas import Timestamp

    from s1bursts.product import BackscatterProduct, GeocodedBurst, MGRSTileProduct


SPECIFICATION_URL = "https://ceos.org/ard/files/PFS/SAR/v1.3.1/CEOS-ARD_PFS_SAR_v1.3.1.pdf"
"""CEOS-ARD Product Family Specification for SAR, version 1.3.1."""

METADATA_VOCABULARY = f"CEOS-ARD PFS SAR v1.3.1; {SPECIFICATION_URL}"
"""Where the tag names and values come from, written on every product.

A pointer, not a claim: it tells a reader holding one raster on its own where
to look a term up. Conformance is neither assessed nor asserted, so the same
value is written on every product rather than being scoped per product type.
"""

LRW_REFERENCE = "https://doi.org/10.1109/TGRS.2021.3055562"
"""Small et al. (2022), the local-resolution-weighting method."""

# The specification asks products to declare this optional conversion.
CONVERSION_EQ = "dB = 10*log10(DN)"

_PIXEL_CONVENTION = "pixel centre"

# The PFS calls stored values "linear power" while the metadata enum uses "Power".
_LINEAR_POWER_UNIT = "linear power"

# The package version identifies the metadata vocabulary, including QA-NUM semantics.
_PIPELINE_SOFTWARE = "s1bursts"
_PIPELINE_URL = "https://github.com/lqgentner/s1bursts"

# Follow the EOSDIS ladder: SLC input L1, geocoded bursts L2, composites L3.
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

# The tags that record each field of RTCConfig, GTCConfig and CoherenceConfig.
# `dem_name` is recorded through DEM, and `standard_grid_origin` is derived
# from `grid_spacing`. A tag a product does not write, such as the coherence
# window of an RTC product or the no-data value of a built-in DEM, compares as
# absent on both sides.
_SETTING_TAGS: dict[str, tuple[str, ...]] = {
    "grid_spacing": ("GRID_SPACING",),
    "dem": ("DEM",),
    "img_resampling": ("IMAGE_RESAMPLING",),
    "dem_resampling": ("DEM_RESAMPLING",),
    "align_to_standard_grid": ("ALIGNED_TO_STANDARD_GRID",),
    "external_dem_nodata": ("EXTERNAL_DEM_NODATA",),
    "external_dem_apply_egm": ("EXTERNAL_DEM_EGM96_APPLIED",),
    "replace_zeros_with_nan": ("ZEROS_REPLACED_WITH_NAN",),
    "multilook": ("MULTILOOK_RANGE", "MULTILOOK_AZIMUTH"),
    "dem_oversampling": ("TERRAIN_FLATTENING_OVERSAMPLING",),
    "additional_overlap": ("TERRAIN_FLATTENING_OVERLAP",),
    "window": ("COHERENCE_WINDOW_RANGE", "COHERENCE_WINDOW_AZIMUTH"),
    "subtract_flat_earth_phase": ("FLAT_EARTH_PHASE_REMOVED",),
    "subtract_topographic_phase": ("TOPOGRAPHIC_PHASE_REMOVED",),
    "correct_coherence_bias": ("COHERENCE_BIAS_CORRECTED",),
    "remove_local_phase_ramp": ("LOCAL_PHASE_RAMP_REMOVED",),
}

# The per-burst tags that record a processing setting.
_BURST_SETTING_KEYS = tuple(key for keys in _SETTING_TAGS.values() for key in keys)

# The SNAP version, and the tags that depend on it: written as ``UNKNOWN``
# under an undetermined version, which therefore matches any other.
_SNAP_VERSION_KEY = "PROCESSING_SOFTWARE_VERSION"
_SNAP_OPTION_KEYS = ("COHERENCE_BIAS_CORRECTED", "LOCAL_PHASE_RAMP_REMOVED")

# Provenance a composite inherits from the per-burst products it was built
# from, describing how those inputs were made. Always present on a composite,
# as ``UNKNOWN`` where the inputs do not record a single answer.
_INPUT_KEYS = ("DEM", "PROCESSING_SOFTWARE", _SNAP_VERSION_KEY)
# Everything a composite inherits, where its inputs record it: the provenance,
# the processing settings the inputs share, and for coherence the estimator.
_INHERITED_KEYS = tuple(dict.fromkeys((*_INPUT_KEYS, *_BURST_SETTING_KEYS)))
_COHERENCE_INPUT_KEYS = (
    "PHASE_QUALITY_METHOD",
    "PHASE_QUALITY_REFERENCE",
    "COHERENCE_WEIGHTING_SHAPE",
)

type Aggregation = Literal["mean", "median"] | float


@dataclass(frozen=True)
class RasterMetadata:
    """Everything written into one output raster besides its pixels.

    Parameters
    ----------
    tags
        Dataset-level tags.
    band_names
        Band descriptions, one per band.
    units
        Band unit strings, one per band. Empty for a dimensionless band.
    band_tags
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


def _flag(value: bool) -> str:  # noqa: FBT001
    """Render a boolean setting as ``true`` or ``false``."""
    return "true" if value else "false"


def _number(value: float) -> str:
    """Render a float briefly, but never so briefly that two values coincide."""
    short = f"{value:g}"
    return short if float(short) == value else repr(value)


def _pipeline() -> dict[str, str]:
    """Identify the package that wrote the file, stamped fresh on every write."""
    return {
        "PIPELINE_SOFTWARE": _PIPELINE_SOFTWARE,
        "PIPELINE_SOFTWARE_VERSION": _distribution_version("s1bursts"),
        "PIPELINE_SOFTWARE_URL": _PIPELINE_URL,
    }


def _backscatter(*, gamma0: bool) -> dict[str, str]:
    """Build the measurement and scaling tags for a linear backscatter raster."""
    return {
        "BACKSCATTER_MEASUREMENT": "Gamma-0" if gamma0 else "Beta-0",
        "BACKSCATTER_CONVENTION": "Power",
        # Item 3.2 has no parameter of its own; the metadata spec puts it here.
        "BACKSCATTER_CONVERSION_EQ": CONVERSION_EQ,
    }


def _aggregation_string(aggregation: Aggregation) -> str:
    return aggregation if isinstance(aggregation, str) else f"quantile={aggregation:g}"


def _burst_mosaicking(burst_edge_trim_px: int) -> str:
    """Render the pixels dropped along each burst's data edge before mosaicking.

    Part of the compositing parameters, so a composite trimmed another way
    is not skipped as existing.
    """
    return f"burst_edge_trim_px={burst_edge_trim_px}"


def _compositing_parameters(
    aggregation: Aggregation,
    clip_max_resolution: float | None,
    burst_edge_trim_px: int,
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
    parts.append(_burst_mosaicking(burst_edge_trim_px))
    return ";".join(parts)


type Acquisitions = tuple[Timestamp, Timestamp]
"""First and last acquisition date that entered a composite."""


def _tile(
    product: MGRSTileProduct,
    *,
    family: str,
    acquisitions: Acquisitions | None = None,
    layer: Layer = Layer.MEASUREMENT,
) -> dict[str, str]:
    """Identity, date window and polarization, shared by every composite.

    ``COMPOSITE_START_DATE`` and ``COMPOSITE_END_DATE`` are the product's dates:
    the compositing window when the caller named one, the observed span
    otherwise. ``FIRST_ACQUISITION_DATE`` and ``LAST_ACQUISITION_DATE`` are
    always the observed span, so the two pairs coincide unless a window was
    given, and a reader can tell how much of the window the data covers.
    """
    first, last = acquisitions or (product.start_date, product.end_date)
    return {
        "METADATA_VOCABULARY": METADATA_VOCABULARY,
        **_pipeline(),
        "PRODUCT_TYPE": _layer_product_type(product, layer),
        "PRODUCT_FAMILY": family,
        "PRODUCT_LEVEL": _LEVEL_COMPOSITE,
        "SOURCE_DATA_PRODUCT_LEVEL": _LEVEL_BURST,
        "MGRS_TILE": product.mgrs_code,
        "COMPOSITE_START_DATE": _date(product.start_date),
        "COMPOSITE_END_DATE": _date(product.end_date),
        "FIRST_ACQUISITION_DATE": _date(first),
        "LAST_ACQUISITION_DATE": _date(last),
        "POLARIZATION": product.polarization,
        "PIXEL_COORDINATE_CONVENTION": _PIXEL_CONVENTION,
    }


def _lrw(
    aggregation: Aggregation, clip_max_resolution: float | None, burst_edge_trim_px: int
) -> dict[str, str]:
    """Compositing-method tags for a locally resolution weighted product."""
    return {
        "COMPOSITING_METHOD": "Local resolution weighting",
        "COMPOSITING_METHOD_REFERENCE": LRW_REFERENCE,
        "COMPOSITING_PARAMETERS": _compositing_parameters(
            aggregation, clip_max_resolution, burst_edge_trim_px
        ),
    }


def _unweighted(aggregation: Aggregation, burst_edge_trim_px: int) -> dict[str, str]:
    """Compositing-method tags for a pass composite, which applies no weighting."""
    return {
        "COMPOSITING_METHOD": "Unweighted temporal aggregation",
        "COMPOSITING_PARAMETERS": (
            f"aggregation={_aggregation_string(aggregation)};"
            f"{_burst_mosaicking(burst_edge_trim_px)}"
        ),
    }


def _inherited(
    source_tags: Mapping[str, str] | None,
    keys: Iterable[str],
    *,
    required: Iterable[str] = (),
) -> dict[str, str]:
    """Forward the named keys from the per-burst inputs' tags.

    A key in ``required`` is always emitted, as ``UNKNOWN`` where the
    inputs did not supply it. Anything else is simply left out when absent,
    because it may not apply to this product at all.
    """
    source_tags = source_tags or {}
    inherited = {key: source_tags[key] for key in keys if key in source_tags}
    for key in required:
        inherited.setdefault(key, UNKNOWN)
    return inherited


def band_statistics(data: NDArray[Any], nodata: float) -> dict[str, str]:
    """Exact per-band statistics, as the GDAL ``STATISTICS_*`` band-tag items.

    Computed over every pixel rather than sampled, so ``STATISTICS_APPROXIMATE``
    is never set. Returns an empty mapping when no pixel is valid, so an empty
    tile carries no statistics rather than NaNs.

    Parameters
    ----------
    data
        Band values.
    nodata
        The band's nodata value, e.g. NaN for a float composite or ``0`` for
        a uint16 count band. A pixel is valid when it is finite *and* differs
        from ``nodata``, so a stray NaN or infinity in a band with a finite
        nodata marker is excluded rather than poisoning the statistics.

    Returns
    -------
    dict[str, str]
        ``STATISTICS_MINIMUM``, ``STATISTICS_MAXIMUM``, ``STATISTICS_MEAN``,
        ``STATISTICS_STDDEV`` and ``STATISTICS_VALID_PERCENT``, or empty.
    """
    # `data != nodata` is always True when nodata is NaN (IEEE comparison),
    # so this reduces to plain finiteness in that case and adds the nodata
    # exclusion only when the marker itself is a finite value.
    valid_mask = np.isfinite(data) & (data != nodata)
    valid = data[valid_mask]
    if valid.size == 0:
        return {}
    return {
        "STATISTICS_MINIMUM": repr(float(valid.min())),
        "STATISTICS_MAXIMUM": repr(float(valid.max())),
        "STATISTICS_MEAN": repr(float(valid.mean())),
        "STATISTICS_STDDEV": repr(float(valid.std())),
        "STATISTICS_VALID_PERCENT": repr(100.0 * valid.size / data.size),
    }


def coherence_statistics(values: NDArray[np.floating[Any]]) -> dict[str, str]:
    """Mean and standard deviation of a coherence raster.

    Returns an empty mapping when no pixel is finite, so that an empty tile
    carries no statistics rather than a NaN.

    Parameters
    ----------
    values
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
    burst_edge_trim_px: int,
    clip_max_resolution: float | None,
    n_observations: int,
    source_tags: Mapping[str, str] | None = None,
    acquisitions: Acquisitions | None = None,
) -> RasterMetadata:
    """Metadata for a locally resolution weighted RTC composite.

    Parameters
    ----------
    product
        The product being written.
    aggregation
        Temporal aggregation actually applied.
    burst_edge_trim_px
        Pixels dropped along each burst's data edge before mosaicking.
    clip_max_resolution
        Weight clip actually applied.
    n_observations
        Number of orbit-date observations entering the composite.
    source_tags
        Tags of one of the per-burst inputs, from which the processing
        provenance is inherited.
    acquisitions
        First and last acquisition date that contributed; defaults to the
        product's own dates.

    Returns
    -------
    RasterMetadata
        Tags, band description and unit for the composite.
    """
    tags = {
        **_tile(product, family="Composite Backscatter", acquisitions=acquisitions),
        **_backscatter(gamma0=True),
        **_lrw(aggregation, clip_max_resolution, burst_edge_trim_px),
        "CONTRIBUTING_OBSERVATIONS": str(n_observations),
        "LAYER_NAME": "gamma0",
        "LAYER_DESCRIPTION": (
            "Gamma-0 backscatter composite, weighted by the inverse local scattering area."
        ),
        **_inherited(source_tags, _INHERITED_KEYS, required=_INPUT_KEYS),
    }
    return RasterMetadata(
        tags=tags,
        band_names=[product.polarization],
        units=[_LINEAR_POWER_UNIT],
    )


def coh_lrw_metadata(
    product: CohLRWComposite,
    *,
    aggregation: Aggregation,
    burst_edge_trim_px: int,
    clip_max_resolution: float | None,
    n_observations: int,
    statistics: Mapping[str, str] | None = None,
    source_tags: Mapping[str, str] | None = None,
    acquisitions: Acquisitions | None = None,
) -> RasterMetadata:
    """Metadata for a locally resolution weighted coherence composite.

    CEOS-ARD defines no composite coherence product: the tags borrow the
    vocabulary of its InSAR and Composite Backscatter products, and neither
    product type is claimed. No
    conversion equation is emitted, because coherence is a dimensionless
    magnitude on [0, 1] rather than a power quantity.

    Parameters
    ----------
    product
        The product being written.
    aggregation
        Temporal aggregation actually applied.
    burst_edge_trim_px
        Pixels dropped along each burst's data edge before mosaicking.
    clip_max_resolution
        Weight clip actually applied.
    n_observations
        Number of orbit-date observations entering the composite.
    statistics
        Coherence statistics from ``coherence_statistics``.
    source_tags
        Tags of one of the per-burst coherence inputs. The estimation window
        originates there and reaches the composite by no other route.
    acquisitions
        First and last acquisition date that contributed; defaults to the
        product's own dates.

    Returns
    -------
    RasterMetadata
        Tags, band description and unit for the composite.
    """
    tags = {
        **_tile(product, family="Composite Coherence", acquisitions=acquisitions),
        "MEASUREMENT_TYPE": "Coherence",
        "TEMPORAL_BASELINE_DAYS": str(product.baseline_days),
        # Small et al. define LRW for backscatter power; weighting coherence by
        # local scattering area is this package's extension, so say so rather
        # than let the reference imply a method the paper does not cover.
        **(
            _lrw(aggregation, clip_max_resolution, burst_edge_trim_px)
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
            "Repeat-pass interferometric coherence composite, "
            "weighted by the inverse local scattering area."
        ),
        **dict(statistics or {}),
        **_inherited(
            source_tags,
            (*_INHERITED_KEYS, *_COHERENCE_INPUT_KEYS),
            required=_INPUT_KEYS,
        ),
    }
    return RasterMetadata(tags=tags, band_names=[product.polarization], units=[""])


def rtc_pass_metadata(
    product: RTCPassComposite,
    *,
    aggregation: Aggregation,
    burst_edge_trim_px: int,
    n_observations: int,
    source_tags: Mapping[str, str] | None = None,
    acquisitions: Acquisitions | None = None,
) -> RasterMetadata:
    """Metadata for an unweighted RTC composite of a single pass direction.

    Parameters
    ----------
    product
        The product being written.
    aggregation
        Temporal aggregation actually applied.
    burst_edge_trim_px
        Pixels dropped along each burst's data edge before mosaicking.
    n_observations
        Number of orbit-date observations entering the composite.
    source_tags
        Tags of one of the per-burst inputs.
    acquisitions
        First and last acquisition date that contributed; defaults to the
        product's own dates.

    Returns
    -------
    RasterMetadata
        Tags, band description and unit for the composite.
    """
    tags = {
        **_tile(product, family="Composite Backscatter", acquisitions=acquisitions),
        **_backscatter(gamma0=True),
        **_unweighted(aggregation, burst_edge_trim_px),
        "PASS_DIRECTION": product.pass_direction,
        "CONTRIBUTING_OBSERVATIONS": str(n_observations),
        "LAYER_NAME": "gamma0",
        "LAYER_DESCRIPTION": (
            "Unweighted gamma-0 backscatter composite of a single pass direction."
        ),
        **_inherited(source_tags, _INHERITED_KEYS, required=_INPUT_KEYS),
    }
    return RasterMetadata(
        tags=tags,
        band_names=[product.polarization],
        units=[_LINEAR_POWER_UNIT],
    )


def gtc_pass_metadata(
    product: GTCPassComposite,
    *,
    aggregation: Aggregation,
    burst_edge_trim_px: int,
    n_observations: int,
    source_tags: Mapping[str, str] | None = None,
    acquisitions: Acquisitions | None = None,
) -> RasterMetadata:
    """Metadata for an unweighted GTC beta0 composite of a single pass direction.

    Parameters
    ----------
    product
        The product being written.
    aggregation
        Temporal aggregation actually applied.
    burst_edge_trim_px
        Pixels dropped along each burst's data edge before mosaicking.
    n_observations
        Number of orbit-date observations entering the composite.
    source_tags
        Tags of one of the per-burst inputs.
    acquisitions
        First and last acquisition date that contributed; defaults to the
        product's own dates.

    Returns
    -------
    RasterMetadata
        Tags, band description and unit for the composite.
    """
    tags = {
        **_tile(product, family="Composite Backscatter", acquisitions=acquisitions),
        **_backscatter(gamma0=False),
        **_unweighted(aggregation, burst_edge_trim_px),
        "PASS_DIRECTION": product.pass_direction,
        "CONTRIBUTING_OBSERVATIONS": str(n_observations),
        "LAYER_NAME": "beta0",
        "LAYER_DESCRIPTION": (
            "Unweighted beta-0 backscatter composite of a single pass direction, "
            "geometrically terrain corrected only."
        ),
        **_inherited(source_tags, _INHERITED_KEYS, required=_INPUT_KEYS),
    }
    return RasterMetadata(
        tags=tags,
        band_names=[product.polarization],
        units=[_LINEAR_POWER_UNIT],
    )


def coh_pass_metadata(
    product: CohPassComposite,
    *,
    aggregation: Aggregation,
    burst_edge_trim_px: int,
    n_observations: int,
    statistics: Mapping[str, str] | None = None,
    source_tags: Mapping[str, str] | None = None,
    acquisitions: Acquisitions | None = None,
) -> RasterMetadata:
    """Metadata for an unweighted coherence composite of a single pass direction.

    Parameters
    ----------
    product
        The product being written.
    aggregation
        Temporal aggregation actually applied.
    burst_edge_trim_px
        Pixels dropped along each burst's data edge before mosaicking.
    n_observations
        Number of orbit-date observations entering the composite.
    statistics
        Coherence statistics from ``coherence_statistics``.
    source_tags
        Tags of one of the per-burst coherence inputs.
    acquisitions
        First and last acquisition date that contributed; defaults to the
        product's own dates.

    Returns
    -------
    RasterMetadata
        Tags, band description and unit for the composite.
    """
    tags = {
        **_tile(product, family="Composite Coherence", acquisitions=acquisitions),
        "MEASUREMENT_TYPE": "Coherence",
        "TEMPORAL_BASELINE_DAYS": str(product.baseline_days),
        **_unweighted(aggregation, burst_edge_trim_px),
        "PASS_DIRECTION": product.pass_direction,
        "CONTRIBUTING_OBSERVATIONS": str(n_observations),
        "LAYER_NAME": "coherence",
        "LAYER_DESCRIPTION": (
            "Unweighted repeat-pass interferometric coherence composite of a single "
            "pass direction."
        ),
        **dict(statistics or {}),
        **_inherited(
            source_tags,
            (*_INHERITED_KEYS, *_COHERENCE_INPUT_KEYS),
            required=_INPUT_KEYS,
        ),
    }
    return RasterMetadata(tags=tags, band_names=[product.polarization], units=[""])


def qa_metadata(
    product: RTCLRWComposite
    | CohLRWComposite
    | RTCPassComposite
    | GTCPassComposite
    | CohPassComposite,
    layer: Layer,
    *,
    aggregation: Aggregation,
    burst_edge_trim_px: int,
    n_observations: int,
    clip_max_resolution: float | None = None,
    source_tags: Mapping[str, str] | None = None,
    acquisitions: Acquisitions | None = None,
) -> RasterMetadata:
    """Metadata for a QA layer: contributing observations or composite quality map.

    Neither layer carries a backscatter convention: a per-pixel observation
    count has no dB scaling, and the CQM declares its own conversion equation.

    Parameters
    ----------
    product
        The composite whose layer is being written.
    layer
        The QA layer being written, ``Layer.QA_NUM`` or ``Layer.QA_CQM``.
    aggregation
        Temporal aggregation actually applied to the parent composite.
    burst_edge_trim_px
        Pixels the parent composite dropped along each burst's data edge.
    n_observations
        Number of orbit-date observations entering the composite.
    clip_max_resolution
        Weight clip actually applied; ignored for a pass composite's QA, which
        applies no weighting.
    source_tags
        Tags of one of the per-burst inputs.
    acquisitions
        First and last acquisition date that contributed; defaults to the
        product's own dates.

    Returns
    -------
    RasterMetadata
        Tags, band description and unit for the QA layer.

    Raises
    ------
    ValueError
        If ``layer`` is not a QA layer of ``product``.
    """
    if layer is Layer.MEASUREMENT or layer not in product.supported_layers:
        msg = f"{type(product).__name__} has no QA layer {layer.name}."
        raise ValueError(msg)
    # Pass composites carry only a contributing-observations layer, and are
    # unweighted, so neither the CQM branch nor the LRW tags apply to them.
    weighted = isinstance(product, RTCLRWComposite | CohLRWComposite)
    coherence = isinstance(product, CohLRWComposite | CohPassComposite)

    tags = {
        **_tile(
            product,
            family="Composite Coherence" if coherence else "Composite Backscatter",
            acquisitions=acquisitions,
            layer=layer,
        ),
        **(
            _lrw(aggregation, clip_max_resolution, burst_edge_trim_px)
            if weighted
            else _unweighted(aggregation, burst_edge_trim_px)
        ),
        "CONTRIBUTING_OBSERVATIONS": str(n_observations),
        **_inherited(source_tags, _INHERITED_KEYS, required=_INPUT_KEYS),
    }
    if isinstance(product, RTCPassComposite | GTCPassComposite | CohPassComposite):
        tags["PASS_DIRECTION"] = product.pass_direction

    if layer is Layer.QA_NUM:
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
            "Composite local resolution: the reciprocal of the mean of the local scattering "
            "areas A_i, weighted with the composite's weights W_i whatever its aggregation; "
            "higher is better resolved; dimensionless."
        ),
    }
    # <CQMdBScaling>: the declaration of a conversion, on linear data.
    tags["CQM_CONVERSION_EQ"] = CONVERSION_EQ
    return RasterMetadata(tags=tags, band_names=["cqm"], units=[""])


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
        "METADATA_VOCABULARY": METADATA_VOCABULARY,
        **_pipeline(),
        "PRODUCT_TYPE": product.product_type,
        "PRODUCT_LEVEL": _LEVEL_BURST,
        "SOURCE_DATA_PRODUCT_LEVEL": _LEVEL_SLC,
        "BURST_ID": str(product.burst_id),
        "SUB_SWATH": product.sub_swath,
        "RELATIVE_ORBIT_NUMBER": str(product.rel_orbit_number),
        "PIXEL_COORDINATE_CONVENTION": _PIXEL_CONVENTION,
        "PROCESSING_SOFTWARE": "ESA SNAP",
        "PROCESSING_SOFTWARE_VERSION": _software_version(snap_version),
        "GRID_SPACING": f"{_number(config.grid_spacing)} m",
        # GeocodingConfig rejects an external DEM without a name.
        "DEM": config.dem if isinstance(config.dem, str) else cast("str", config.dem_name),
        "IMAGE_RESAMPLING": config.img_resampling,
        "DEM_RESAMPLING": config.dem_resampling,
        "ALIGNED_TO_STANDARD_GRID": _flag(config.align_to_standard_grid),
        "ZEROS_REPLACED_WITH_NAN": _flag(config.replace_zeros_with_nan),
    }
    # These settings apply to an external DEM only.
    if isinstance(config.dem, Path):
        tags["EXTERNAL_DEM_NODATA"] = _number(config.external_dem_nodata)
        tags["EXTERNAL_DEM_EGM96_APPLIED"] = _flag(config.external_dem_apply_egm)
    return tags


def _software_version(snap_version: tuple[int, ...] | None) -> str:
    """Render a SNAP version as ``PROCESSING_SOFTWARE_VERSION`` records it."""
    return ".".join(str(part) for part in snap_version) if snap_version is not None else UNKNOWN


def backscatter_metadata(
    product: BackscatterProduct,
    config: BackscatterConfig,
    *,
    snap_version: tuple[int, ...] | None = None,
) -> RasterMetadata:
    """Metadata for a per-burst RTC or GTC product.

    Parameters
    ----------
    product
        The product being written.
    config
        Configuration the graph was built from.
    snap_version
        Version of the SNAP installation that ran the graph, from
        ``s1bursts.snap_client.snap_version``. Written as ``N/A`` when unknown.

    Returns
    -------
    RasterMetadata
        Tags, band descriptions and units for the product.

    Raises
    ------
    ValueError
        If the product and the config disagree on terrain flattening, an RTC
        product with a ``GTCConfig`` or the reverse, which would pair a band
        count from one with units from the other.
    """
    gamma0 = product.is_terrain_flattened
    if gamma0 is not isinstance(config, RTCConfig):
        msg = (
            f"{type(product).__name__} and {type(config).__name__} disagree on terrain flattening."
        )
        raise ValueError(msg)
    tags = {
        **_burst(product, config=config, snap_version=snap_version),
        **_backscatter(gamma0=gamma0),
        "POLARIZATION": product.polarization,
        "ACQUISITION_DATE": _date(product.date),
        "TERRAIN_CORRECTION": "Radiometric and geometric" if gamma0 else "Geometric",
    }
    if config.multilook is not None:
        looks_rg, looks_az = config.multilook
        tags["MULTILOOK_RANGE"] = str(looks_rg)
        tags["MULTILOOK_AZIMUTH"] = str(looks_az)
    if isinstance(config, RTCConfig):
        tags["TERRAIN_FLATTENING_OVERSAMPLING"] = _number(config.dem_oversampling)
        tags["TERRAIN_FLATTENING_OVERLAP"] = _number(config.additional_overlap)

    if not gamma0:
        return RasterMetadata(
            tags=tags,
            band_names=list(config.output_bands),
            units=[_LINEAR_POWER_UNIT],
            band_tags=[{"LAYER_NAME": "beta0"}],
        )

    # Band 2 is the local scattering area terrain flattening divides by: CEOS-ARD
    # item 2.3, Small et al.'s local contributing area, OPERA's area
    # normalization factor.
    return RasterMetadata(
        tags=tags,
        band_names=list(config.output_bands),
        units=[_LINEAR_POWER_UNIT, ""],
        band_tags=[
            {"LAYER_NAME": "gamma0"},
            {
                "LAYER_NAME": "scattering_area",
                "LAYER_DESCRIPTION": (
                    "Local scattering area that terrain flattening divides beta-0 by"
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

    The estimation window exists only in ``config``, so
    these tags are the only route by which it reaches a composite built from
    this product. The options SNAP 14 added to its ``Coherence`` operator are
    recorded as ``false`` for an older SNAP, which has neither, and as
    ``N/A`` for an unknown version, under which SNAP's own defaults apply.

    Parameters
    ----------
    product
        The product being written.
    config
        Configuration the graph was built from.
    snap_version
        Version of the SNAP installation that ran the graph, from
        ``s1bursts.snap_client.snap_version``. Written as ``N/A`` when unknown.

    Returns
    -------
    RasterMetadata
        Tags, band description and unit for the product.

    Raises
    ------
    ConfigurationError
        If ``config`` enables an option of SNAP 14 but ``snap_version`` is
        older or unknown, as building its graph does.
    """
    win_rg, win_az = config.window
    # An option the graph leaves out is off under an older SNAP, which lacks it.
    no_option = UNKNOWN if snap_version is None else "false"
    correct_bias, remove_ramp = (
        no_option if value is None else _flag(value)
        for value in _coherence_option_values(config, snap_version)
    )
    tags = {
        **_burst(product, config=config, snap_version=snap_version),
        "MEASUREMENT_TYPE": "Coherence",
        "POLARIZATION": product.polarization,
        "DATE_PRIMARY": _date(product.date_primary),
        "DATE_SECONDARY": _date(product.date_secondary),
        "TEMPORAL_BASELINE_DAYS": str(product.baseline_days),
        "PHASE_QUALITY_METHOD": _PHASE_QUALITY_METHOD,
        "PHASE_QUALITY_REFERENCE": _PHASE_QUALITY_REFERENCE,
        "COHERENCE_WINDOW_RANGE": str(win_rg),
        "COHERENCE_WINDOW_AZIMUTH": str(win_az),
        "COHERENCE_WEIGHTING_SHAPE": _COHERENCE_WEIGHTING_SHAPE,
        "FLAT_EARTH_PHASE_REMOVED": _flag(config.subtract_flat_earth_phase),
        "TOPOGRAPHIC_PHASE_REMOVED": _flag(config.subtract_topographic_phase),
        "COHERENCE_BIAS_CORRECTED": correct_bias,
        "LOCAL_PHASE_RAMP_REMOVED": remove_ramp,
    }
    return RasterMetadata(
        tags=tags,
        band_names=list(config.output_bands),
        units=[""],
        band_tags=[{"LAYER_NAME": "coherence"}],
    )


def burst_metadata(
    product: RTCProduct | GTCProduct | CoherenceProduct,
    config: BackscatterConfig | CoherenceConfig,
    *,
    snap_version: tuple[int, ...] | None = None,
) -> RasterMetadata:
    """Metadata for any per-burst product, dispatched on its config.

    Parameters
    ----------
    product
        The product being written.
    config
        Configuration the graph was built from.
    snap_version
        Version of the SNAP installation that ran the graph, from
        ``s1bursts.snap_client.snap_version``. Written as ``N/A`` when unknown.

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
        case (RTCProduct(), RTCConfig()) | (GTCProduct(), GTCConfig()):
            return backscatter_metadata(product, config, snap_version=snap_version)
        case CoherenceProduct(), CoherenceConfig():
            return coherence_metadata(product, config, snap_version=snap_version)
        case _:
            msg = (
                f"cannot build metadata for {type(product).__name__} from {type(config).__name__}."
            )
            raise TypeError(msg)


def _burst_settings(
    product: RTCProduct | GTCProduct | CoherenceProduct,
    config: BackscatterConfig | CoherenceConfig,
    *,
    snap_version: tuple[int, ...] | None,
) -> dict[str, str | None]:
    """Return the setting tags a per-burst product made with ``config`` carries, None if absent.

    Taken from the tags ``burst_metadata`` writes, so they compare with a
    file's recorded tags exactly as written.
    """
    tags = burst_metadata(product, config, snap_version=snap_version).tags
    return {key: tags.get(key) for key in _BURST_SETTING_KEYS}


def _composite_settings(
    product: MGRSTileProduct,
    *,
    aggregation: Aggregation,
    burst_edge_trim_px: int,
    clip_max_resolution: float | None = None,
) -> dict[str, str]:
    """Return the setting tags a composite made with these arguments carries in every layer."""
    weighted = isinstance(product, RTCLRWComposite | CohLRWComposite)
    tags = (
        _lrw(aggregation, clip_max_resolution, burst_edge_trim_px)
        if weighted
        else _unweighted(aggregation, burst_edge_trim_px)
    )
    return {"COMPOSITING_PARAMETERS": tags["COMPOSITING_PARAMETERS"]}


__all__ = [
    "CONVERSION_EQ",
    "LRW_REFERENCE",
    "METADATA_VOCABULARY",
    "SPECIFICATION_URL",
    "UNKNOWN",
    "RasterMetadata",
    "backscatter_metadata",
    "band_statistics",
    "burst_metadata",
    "coh_lrw_metadata",
    "coh_pass_metadata",
    "coherence_metadata",
    "coherence_statistics",
    "gtc_pass_metadata",
    "qa_metadata",
    "rtc_lrw_metadata",
    "rtc_pass_metadata",
]
