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

METADATA_VOCABULARY = f"CEOS-ARD PFS SAR v1.3.1; {SPECIFICATION_URL}"
"""Where the tag names and values come from, written on every product.

A pointer, not a claim: it tells a reader holding one raster on its own where
to look a term up. Conformance is neither assessed nor asserted, so the same
value is written on every product rather than being scoped per product type.
"""

LRW_REFERENCE = "https://doi.org/10.1109/TGRS.2021.3055562"
"""Small et al. (2022), the local-resolution-weighting method of item 1.7.24."""

# The specification asks products to declare this optional conversion.
CONVERSION_EQ = "dB = 10*log10(DN)"

_PIXEL_CONVENTION = "pixel centre"

# The PFS calls stored values "linear power" while the metadata enum uses "Power".
_LINEAR_POWER_UNIT = "linear power"
_DECIBEL_UNIT = "dB"

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


def _pipeline() -> dict[str, str]:
    """Identify the package that wrote the file, stamped fresh on every write."""
    return {
        "PIPELINE_SOFTWARE": _PIPELINE_SOFTWARE,
        "PIPELINE_SOFTWARE_VERSION": _distribution_version("s1bursts"),
        "PIPELINE_SOFTWARE_URL": _PIPELINE_URL,
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


type Acquisitions = tuple[Timestamp, Timestamp]
"""First and last acquisition date that entered a composite."""


def _tile(
    product: MGRSTileProduct,
    *,
    family: str,
    acquisitions: Acquisitions | None = None,
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
        "PRODUCT_TYPE": product.product_type,
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

    A key in ``required`` is always emitted, as ``UNKNOWN`` where the
    inputs did not supply it — they disagreed, or never carried it. Anything
    else is simply left out when absent, because it may not apply to this
    product at all.
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
    """Mean and standard deviation of a coherence raster ([InSAR] item 3.8).

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
    clip_max_resolution: float | None,
    n_observations: int,
    decibel: bool,
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
    clip_max_resolution
        Weight clip actually applied.
    n_observations
        Number of orbit-date observations entering the composite (spec 2.21).
    decibel
        Whether the stored values are decibels rather than linear power.
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
        **_backscatter(gamma0=True, decibel=decibel),
        **_lrw(aggregation, clip_max_resolution),
        "CONTRIBUTING_OBSERVATIONS": str(n_observations),
        "LAYER_NAME": "gamma0",
        "LAYER_DESCRIPTION": (
            "Gamma-0 backscatter composite, weighted by the inverse local scattering area."
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
    acquisitions: Acquisitions | None = None,
) -> RasterMetadata:
    """Metadata for a locally resolution weighted coherence composite.

    The specification defines no composite coherence product: the quantity
    vocabulary comes from [InSAR] items 3.8 and 1.7.18, the compositing
    vocabulary from [CB] item 1.7.24, and neither product type is claimed. No
    conversion equation is emitted, because coherence is a dimensionless
    magnitude on [0, 1] rather than a power quantity.

    Parameters
    ----------
    product
        The product being written.
    aggregation
        Temporal aggregation actually applied.
    clip_max_resolution
        Weight clip actually applied.
    n_observations
        Number of orbit-date observations entering the composite (spec 2.21).
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
            "Repeat-pass interferometric coherence composite, "
            "weighted by the inverse local scattering area."
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
    acquisitions: Acquisitions | None = None,
) -> RasterMetadata:
    """Metadata for an unweighted RTC composite of a single pass direction.

    Parameters
    ----------
    product
        The product being written.
    aggregation
        Temporal aggregation actually applied.
    n_observations
        Number of orbit-date observations entering the composite.
    decibel
        Whether the stored values are decibels rather than linear power.
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
        **_backscatter(gamma0=True, decibel=decibel),
        **_unweighted(aggregation),
        "PASS_DIRECTION": product.pass_direction,
        "CONTRIBUTING_OBSERVATIONS": str(n_observations),
        "LAYER_NAME": "gamma0",
        "LAYER_DESCRIPTION": (
            "Unweighted gamma-0 backscatter composite of a single pass direction."
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
    acquisitions: Acquisitions | None = None,
) -> RasterMetadata:
    """Metadata for an unweighted coherence composite of a single pass direction.

    Parameters
    ----------
    product
        The product being written.
    aggregation
        Temporal aggregation actually applied.
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
        **_unweighted(aggregation),
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
    acquisitions: Acquisitions | None = None,
) -> RasterMetadata:
    """Metadata for a QA auxiliary: contributing observations or composite quality map.

    Neither layer carries a backscatter convention or a conversion equation: a
    per-pixel observation count has no dB scaling, and the CQM declares its own.

    Parameters
    ----------
    product
        The QA product being written.
    aggregation
        Temporal aggregation actually applied to the parent composite.
    n_observations
        Number of orbit-date observations entering the composite (spec 2.21).
    clip_max_resolution
        Weight clip actually applied; ignored for a pass composite's QA, which
        applies no weighting.
    cqm_decibel
        Whether a ``QA-CQM`` layer stores decibels rather than linear values.
    source_tags
        Tags of one of the per-burst inputs.
    acquisitions
        First and last acquisition date that contributed; defaults to the
        product's own dates.

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
            acquisitions=acquisitions,
        ),
        **(_lrw(aggregation, clip_max_resolution) if weighted else _unweighted(aggregation)),
        "CONTRIBUTING_OBSERVATIONS": str(n_observations),
        **_inherited(source_tags, _INPUT_KEYS, required=_INPUT_KEYS),
    }
    if isinstance(product, RTCPassQA | CohPassQA):
        tags["PASS_DIRECTION"] = product.pass_direction

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
        "LAYER_DESCRIPTION": "Composite local resolution, 1/sum(W_i * A_i); dimensionless.",
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
                "LAYER_DESCRIPTION": ("Local scattering area relative to a flat reference facet."),
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
    """
    win_rg, win_az = config.window
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
    }
    return RasterMetadata(
        tags=tags,
        band_names=list(config.output_bands),
        units=[""],
        band_tags=[{"LAYER_NAME": "coherence"}],
    )


def interferogram_metadata(
    product: InterferogramProduct,
    config: InterferogramConfig,
    *,
    snap_version: tuple[int, ...] | None = None,
) -> RasterMetadata:
    """Metadata for a per-burst interferogram product.

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
    """
    tags = {
        **_burst(product, config=config, snap_version=snap_version),
        "MEASUREMENT_TYPE": "Interferogram",
        "POLARIZATION": product.polarization,
        "DATE_PRIMARY": _date(product.date_primary),
        "DATE_SECONDARY": _date(product.date_secondary),
        "TEMPORAL_BASELINE_DAYS": str(product.baseline_days),
        "FLAT_EARTH_PHASE_REMOVED": str(config.subtract_flat_earth_phase).lower(),
        "TOPOGRAPHIC_PHASE_REMOVED": str(config.subtract_topographic_phase).lower(),
    }
    # The two intensities are power; the complex parts are not a power quantity
    # on their own, so they get no unit and no conversion equation.
    return RasterMetadata(
        tags=tags,
        band_names=list(config.output_bands),
        units=["", "", _LINEAR_POWER_UNIT, _LINEAR_POWER_UNIT],
        band_tags=[{"LAYER_NAME": name} for name in config.output_bands],
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
    "interferogram_metadata",
    "qa_metadata",
    "rtc_lrw_metadata",
    "rtc_pass_metadata",
]
