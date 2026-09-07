"""Sentinel-1 burst data structures and collections."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, overload, override
import warnings

import geopandas as gpd
import pandas as pd
from pandas import DataFrame, Series, Timedelta, Timestamp

from s1bursts import ui
from s1bursts.burstmap import (
    create_s1_geometry_lut,
    create_s1_orbit_number_lut,
    create_s1_pass_direction_lut,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator

    from shapely import MultiPolygon, Polygon

    from s1bursts.burstmap import Mode
    from s1bursts.cdse import CDSEResponse
    from s1bursts.query import PassDirection, Polarization, SubSwath


class BurstFootprintMixin:
    """Mixin that provides burst footprint attributes.

    Each attribute returns an authoritative value stored on the instance
    (propagated from the CDSE response or an SLC filename) when available, and
    otherwise falls back to the Sentinel-1 burst-ID map lookup table. The LUT is
    an opt-in ~500 MB download (see ``s1bursts.burstmap.download_burst_map``);
    it is consulted only when a footprint value was not propagated.

    Requires subclasses to declare ``burst_id: int`` and ``sub_swath: SubSwath``.
    """

    __slots__ = ()

    # Type-only and read-only: subclasses supply them as frozen dataclass fields.
    if TYPE_CHECKING:

        @property
        def burst_id(self) -> int:
            """Burst identifier."""

        @property
        def sub_swath(self) -> SubSwath:
            """Sub-swath identifier."""

    # Authoritative overrides; None here so subclasses without their own field
    # (e.g. product.GeocodedBurst) inherit the LUT-fallback behaviour.
    _rel_orbit_number: int | None = None
    _pass_direction: PassDirection | None = None
    _geometry: Polygon | MultiPolygon | None = None

    @property
    def mode(self) -> Mode:
        """Acquisition mode (``"IW"`` or ``"EW"``) derived from the sub-swath.

        Returns
        -------
        Mode
            ``"IW"`` for ``IW1``-``IW3`` sub-swaths, ``"EW"`` for ``EW1``-``EW5``.
        """
        return cast("Mode", self.sub_swath[:2])

    def _lut_lookup[T](self, lut: dict[tuple[int, SubSwath], T]) -> T:
        """Look up this burst in a LUT keyed by (burst_id, sub_swath)."""
        key = (self.burst_id, self.sub_swath)
        try:
            return lut[key]
        except KeyError:
            msg = (
                f"Burst (burst_id={self.burst_id}, sub_swath='{self.sub_swath}') "
                "not found in the Sentinel-1 burst map."
            )
            raise KeyError(msg) from None

    @property
    def rel_orbit_number(self) -> int:
        """Relative orbit number, propagated from CDSE or looked up from the burst map.

        Returns
        -------
        int
            Relative orbit number.

        Raises
        ------
        FileNotFoundError
            If the value was not propagated and the burst map has not been
            downloaded (see ``s1bursts.burstmap.download_burst_map``).
        KeyError
            If the burst is not found in the map.
        """
        if self._rel_orbit_number is not None:
            return self._rel_orbit_number
        return self._lut_lookup(create_s1_orbit_number_lut(mode=self.mode))

    @property
    def pass_direction(self) -> PassDirection:
        """Pass direction, propagated from CDSE or looked up from the burst map.

        Returns
        -------
        PassDirection
            ``"Ascending"`` or ``"Descending"``.

        Raises
        ------
        FileNotFoundError
            If the value was not propagated and the burst map has not been
            downloaded (see ``s1bursts.burstmap.download_burst_map``).
        KeyError
            If the burst is not found in the map.
        """
        if self._pass_direction is not None:
            return self._pass_direction
        return self._lut_lookup(create_s1_pass_direction_lut(mode=self.mode))

    @property
    def geometry(self) -> Polygon | MultiPolygon:
        """Burst footprint, propagated from CDSE or looked up from the burst map.

        Returns
        -------
        Polygon | MultiPolygon
            Footprint geometry. A ``MultiPolygon`` for bursts crossing the antimeridian.

        Raises
        ------
        FileNotFoundError
            If the geometry was not propagated and the burst map has not been
            downloaded (see ``s1bursts.burstmap.download_burst_map``).
        KeyError
            If the burst is not found in the map.
        """
        if self._geometry is not None:
            return self._geometry
        return self._lut_lookup(create_s1_geometry_lut(mode=self.mode))

    @property
    def full_burst_id(self) -> str:
        r"""ASF-style full burst ID.

        Consists of ``\<rel_orbit_number\>\_\<burst_id\>\_\<sub_swath\>``.

        Returns
        -------
        str
            Full burst ID string.

        Raises
        ------
        FileNotFoundError
            If ``rel_orbit_number`` was not propagated and the burst map has not
            been downloaded (see ``s1bursts.burstmap.download_burst_map``).
        KeyError
            If the burst is not found in the burst map.
        """
        return f"{self.rel_orbit_number:03}_{self.burst_id:06}_{self.sub_swath}"


def _scalar_or_none[T](value: T) -> T | None:
    """Return ``None`` for a missing scalar (``None`` or NaN), else the value unchanged."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


# Public footprint keys mapped to the private fields they populate, so
# ``Burst.from_dict`` accepts parsed-filename/response dicts.
_FOOTPRINT_ALIASES = {
    "rel_orbit_number": "_rel_orbit_number",
    "pass_direction": "_pass_direction",
    "geometry": "_geometry",
}


@dataclass(frozen=True, slots=True, eq=False)
class Burst(BurstFootprintMixin):
    """Represents a single Sentinel-1 burst.

    A burst is a segment of SAR data acquisition characterized by its spatial location,
    temporal acquisition time, and imaging parameters.

    Two bursts are considered equal when their ``burst_id``, ``sub_swath``,
    ``polarization``, and calendar day (``date`` normalized to midnight) match, so
    a burst from a CDSE search compares equal to the same acquisition reconstructed
    from an SLC filename (which carries only the day). A given footprint is
    re-acquired at most once per repeat cycle, so the day is a unique key.
    """

    burst_id: int
    sub_swath: SubSwath
    polarization: Polarization
    date: Timestamp
    abs_burst_id: int | None = field(default=None, compare=False, hash=False)
    cdse_id: str | None = field(default=None, compare=False, hash=False)
    parent_product_name: str | None = field(default=None, compare=False, hash=False)
    # Footprint values propagated from CDSE or an SLC filename; None falls back to the LUT.
    # Read-only here because the dataclass is frozen; the mixin declares them read-write.
    # pyrefly: ignore [bad-override]
    _rel_orbit_number: int | None = field(default=None, compare=False, hash=False, repr=False)
    # pyrefly: ignore [bad-override]
    _pass_direction: PassDirection | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )
    # pyrefly: ignore [bad-override]
    _geometry: Polygon | MultiPolygon | None = field(
        default=None,
        compare=False,
        hash=False,
        repr=False,
    )

    def __repr__(self) -> str:
        """Return string representation of the Burst."""
        return (
            f"Burst(burst_id='{self.burst_id:06}', sub_swath='{self.sub_swath}', "
            f"polarization='{self.polarization}', date='{self.date:%Y-%m-%d}')"
        )

    def __eq__(self, other: object) -> bool:
        """Compare identity fields with the acquisition date normalized to the day."""
        if not isinstance(other, Burst):
            return NotImplemented
        return (
            self.burst_id == other.burst_id
            and self.sub_swath == other.sub_swath
            and self.polarization == other.polarization
            and self.date.normalize() == other.date.normalize()
        )

    def __hash__(self) -> int:
        """Hash on the identity fields, with the acquisition date normalized to the day."""
        return hash(
            (self.burst_id, self.sub_swath, self.polarization, self.date.normalize()),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Instantiate from a mapping, silently dropping unknown and non-init fields.

        Public footprint keys (``rel_orbit_number``, ``pass_direction``,
        ``geometry``) are mapped onto the private override fields, so a burst
        reconstructed from a parsed SLC filename keeps its relative orbit number
        instead of re-deriving it from the LUT.

        Parameters
        ----------
        data
            Mapping of field names to values. Extra keys and ``init=False``
            fields are filtered before construction.

        Returns
        -------
        Self
            A new ``Burst`` instance.
        """
        data = {_FOOTPRINT_ALIASES.get(k, k): v for k, v in data.items()}
        init_keys = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in data.items() if k in init_keys})


@dataclass(slots=True)
class BurstPair:
    """Represents a pair of Sentinel-1 bursts for interferometric processing.

    The primary burst is typically the earlier acquisition, and the secondary
    is the later acquisition. The pair must have matching burst parameters.
    """

    primary: Burst
    secondary: Burst

    def __post_init__(self) -> None:
        """Validate burst compatibility.

        Raises
        ------
        ValueError
            If bursts have mismatched burst_id, sub_swath, or polarization,
            or if the temporal baseline is not positive.
        """
        self._validate_compatibility()

    def _validate_compatibility(self) -> None:
        """Check if bursts have matching parameters for interferometry."""
        if not (
            self.primary.burst_id == self.secondary.burst_id
            and self.primary.sub_swath == self.secondary.sub_swath
            and self.primary.polarization == self.secondary.polarization
        ):
            msg = (
                "Burst parameters don't match:\n"
                "Primary: "
                f"{self.primary.burst_id} {self.primary.sub_swath} {self.primary.polarization}\n"
                "Secondary: "
                f"{self.secondary.burst_id} {self.secondary.sub_swath} "
                f"{self.secondary.polarization}"
            )
            raise ValueError(msg)

        if self.baseline_days <= 0:
            msg = f"Expected a positive non-zero baseline, received {self.baseline_days} days."
            raise ValueError(msg)

    @property
    def full_burst_id(self) -> str:
        """ASF-style full burst ID, delegated from the primary burst.

        Returns
        -------
        str
            Full burst ID string.

        Raises
        ------
        KeyError
            If the burst is not found in the burst map.
        """
        return self.primary.full_burst_id

    @property
    def burst_id(self) -> int:
        """Burst ID of the primary burst.

        Returns
        -------
        int
            Burst ID.
        """
        return self.primary.burst_id

    @property
    def sub_swath(self) -> SubSwath:
        """Sub-swath of the primary burst.

        Returns
        -------
        SubSwath
            Sub-swath identifier.
        """
        return self.primary.sub_swath

    @property
    def polarization(self) -> Polarization:
        """Polarization of the primary burst.

        Returns
        -------
        Polarization
            Polarization channel.
        """
        return self.primary.polarization

    @property
    def rel_orbit_number(self) -> int:
        """Relative orbit number, looked up via the primary burst.

        Returns
        -------
        int
            Relative orbit number.

        Raises
        ------
        KeyError
            If the burst is not found in the burst map.
        """
        return self.primary.rel_orbit_number

    @property
    def pass_direction(self) -> PassDirection:
        """Pass direction, looked up via the primary burst.

        Returns
        -------
        PassDirection
            ``"Ascending"`` or ``"Descending"``.

        Raises
        ------
        KeyError
            If the burst is not found in the burst map.
        """
        return self.primary.pass_direction

    @property
    def geometry(self) -> Polygon | MultiPolygon:
        """Burst footprint, looked up via the primary burst.

        Returns
        -------
        Polygon | MultiPolygon
            Footprint geometry.

        Raises
        ------
        KeyError
            If the burst is not found in the burst map.
        """
        return self.primary.geometry

    @property
    def baseline_days(self) -> int:
        """Temporal baseline in days.

        Returns
        -------
        int
            Number of days between primary and secondary acquisition.
        """
        return (self.secondary.date - self.primary.date).round("D").days


class _BaseBurstCollection[TEntry]:
    """Base class for burst collections with shared functionality.

    Subclasses define their column metadata and implement ``_row_to_entry``.

    Parameters
    ----------
    df
        DataFrame with burst data. If ``None``, an empty collection is created.

    Raises
    ------
    ValueError
        If required columns are missing from the dataframe.
    """

    _UNIQUE_COLUMNS: ClassVar[list[str]]
    _ALL_COLUMNS: ClassVar[list[str]]
    # Optional columns are filled with None and excluded from validation.
    _OPTIONAL_COLUMNS: ClassVar[list[str]] = []
    _ENTRY_NAME_SINGULAR: ClassVar[str]
    _ENTRY_NAME_PLURAL: ClassVar[str]
    _FOOTPRINT_UNIQUE_COLUMNS: ClassVar[list[str]] = ["burst_id", "sub_swath"]
    _INSAR_MATCH_COLUMNS: ClassVar[list[str]] = [
        "burst_id",
        "sub_swath",
        "polarization",
    ]

    @staticmethod
    @abstractmethod
    def _row_to_entry(row: Series) -> TEntry:
        """Convert a DataFrame row to a collection entry."""
        ...

    def __init__(self, df: gpd.GeoDataFrame | None = None) -> None:
        if df is None:
            df = gpd.GeoDataFrame(
                {col: [] for col in self._ALL_COLUMNS},
                geometry="geometry",
            )
        self._validate_dataframe(df)
        df = df.copy()
        for col in self._OPTIONAL_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[self._ALL_COLUMNS]
        df = self._deduplicate_dataframe(df)
        self.df = df.sort_values(self._UNIQUE_COLUMNS)

    def _validate_dataframe(self, df: gpd.GeoDataFrame) -> None:
        """Validate the dataframe type and required columns."""
        if not isinstance(df, gpd.GeoDataFrame):
            msg = f"`df` must be a GeoDataFrame, received '{type(df).__name__}'."
            raise TypeError(msg)
        df_columns = set(df.columns)
        all_columns = set(self._ALL_COLUMNS)
        required_columns = all_columns - set(self._OPTIONAL_COLUMNS)
        missing_columns = required_columns - df_columns
        if missing_columns:
            msg = f"DataFrame is missing required columns: {missing_columns}"
            raise ValueError(msg)

        unexpected_columns = df_columns - all_columns
        if unexpected_columns:
            warnings.warn(
                f"DataFrame contains unexpected columns: {unexpected_columns}",
                stacklevel=3,
            )

    def _deduplicate_dataframe(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Remove entries with duplicate identity columns."""
        return df.drop_duplicates(subset=self._UNIQUE_COLUMNS, ignore_index=True)

    def __len__(self) -> int:
        """Return the number of entries."""
        return len(self.df)

    def __add__(self, other: Self) -> Self:
        """Concatenate two collections and remove duplicates."""
        combined_df = gpd.GeoDataFrame(
            pd.concat([self.df, other.df], ignore_index=True),
        )

        return type(self)(combined_df)

    def __iter__(self) -> Iterator[TEntry]:
        """Iterate over collection entries."""
        for _, row in self.df.iterrows():
            yield self._row_to_entry(row)

    @overload
    def __getitem__(self, index: int) -> TEntry: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> TEntry | Self:
        """Return an entry or sliced collection."""
        if isinstance(index, slice):
            return type(self)(self.df.iloc[index])
        return self._row_to_entry(self.df.iloc[index])

    def _repr_html_(self) -> str:
        """Return the rich HTML representation."""
        unique_footprints = self.df.drop_duplicates(
            subset=self._FOOTPRINT_UNIQUE_COLUMNS,
        )

        n_bursts = len(self)
        n_footprints = len(unique_footprints)

        map_svg = ui.create_footprint_svg(unique_footprints)
        world_map_svg = ui.create_world_map_with_target(unique_footprints)
        # pandas-stubs omits `_repr_html_`, so the lookup falls through to `__getattr__`.
        # pyrefly: ignore [not-callable]
        df_html = self.df[self._UNIQUE_COLUMNS]._repr_html_()
        css = ui.get_burstinfo_css()
        col_name = type(self).__name__
        entry_name = self._ENTRY_NAME_SINGULAR if n_bursts == 1 else self._ENTRY_NAME_PLURAL
        footprint_name = "Footprint" if n_footprints == 1 else "Footprints"

        return f"""
        <style>{css}</style>
        <div class="burst-info">
            <h4>{col_name}</h4>
            <div class="row">
                <div class="column">
                    <div class="info-box">
                        <div class="row">
                            <div class="column">{entry_name}</div>
                            <div class="column value">{n_bursts:,d}</div>
                        </div>
                        <div class="row">
                            <div class="column">{footprint_name}</div>
                            <div class="column value">{n_footprints:,d}</div>
                        </div>
                        <div>{world_map_svg}</div>
                    </div>
                </div>
                <div class="column svg-map">{map_svg}</div>
            </div>
            <details>
                <summary>{entry_name}</summary>
                <div class="dataframe-container">{df_html}</div>
            </details>
        </div>
        """


class BurstCollection(_BaseBurstCollection[Burst]):
    """Collection of Sentinel-1 bursts with geospatial and temporal data.

    Provides methods for working with multiple bursts, including finding
    interferometric pairs, filtering, and visualization.

    Parameters
    ----------
    df
        DataFrame with burst data. If ``None``, an empty collection is created.

    Raises
    ------
    TypeError
        If df is not a GeoDataFrame.
    ValueError
        If required columns are missing.
    """

    # Columns that make a burst unique
    _UNIQUE_COLUMNS: ClassVar[list[str]] = [
        "burst_id",
        "sub_swath",
        "polarization",
        "date",
    ]
    # All valid fields from Burst dataclass
    _ALL_COLUMNS: ClassVar[list[str]] = [
        "burst_id",
        "sub_swath",
        "polarization",
        "date",
        "abs_burst_id",
        "cdse_id",
        "parent_product_name",
        "geometry",
        "_rel_orbit_number",
        "_pass_direction",
        "_lines_per_burst",
        "_samples_per_burst",
    ]
    _OPTIONAL_COLUMNS: ClassVar[list[str]] = [
        "_rel_orbit_number",
        "_pass_direction",
        # Burst pixel dimensions, propagated from CDSE and used only to pick the
        # most complete of near-identical-timestamp duplicates.
        "_lines_per_burst",
        "_samples_per_burst",
    ]
    # Entry names for display
    _ENTRY_NAME_SINGULAR: ClassVar[str] = "Burst"
    _ENTRY_NAME_PLURAL: ClassVar[str] = "Bursts"

    def __init__(self, df: gpd.GeoDataFrame | None = None) -> None:
        super().__init__(df)
        self._remove_timestamp_duplicates()

    @staticmethod
    def _validate_pair_search(baseline: Timedelta, max_delta: Timedelta) -> None:
        """Reject baselines that cannot describe an interferometric pair."""
        if baseline <= Timedelta(0):
            msg = f"`baseline` must be a positive duration, received {baseline}."
            raise ValueError(msg)
        if max_delta < Timedelta(0):
            msg = f"`max_delta` must not be negative, received {max_delta}."
            raise ValueError(msg)

    def find_pairs(
        self,
        baseline: Timedelta | int,
        max_delta: Timedelta | int = 24,
    ) -> BurstPairCollection:
        """Find interferometric burst pairs with a specified temporal baseline.

        Parameters
        ----------
        baseline
            Temporal baseline between primary and secondary bursts.
            If integer is provided, the assumed unit is days.
        max_delta
            Maximum allowed deviation from baseline.
            If integer is provided, the assumed unit is hours.
            Default is 24 hours.

        Returns
        -------
        BurstPairCollection
            Collection of burst pairs matching the baseline criteria. Empty if
            no pair satisfies them.

        Raises
        ------
        ValueError
            If ``baseline`` is not positive or ``max_delta`` is negative.
        """
        if not isinstance(baseline, Timedelta):
            baseline = Timedelta(baseline, unit="days")
        if not isinstance(max_delta, Timedelta):
            max_delta = Timedelta(max_delta, unit="hours")

        self._validate_pair_search(baseline, max_delta)
        df_bursts = self.df.sort_values("date")
        pairs = []
        renamer_1 = {
            "date": "date_1",
            "cdse_id": "cdse_id_1",
            "abs_burst_id": "abs_burst_id_1",
            "parent_product_name": "parent_product_name_1",
        }
        renamer_2 = {
            "date": "date_2",
            "cdse_id": "cdse_id_2",
            "abs_burst_id": "abs_burst_id_2",
            "parent_product_name": "parent_product_name_2",
        }

        def match_pairs(df: DataFrame, periods: int) -> gpd.GeoDataFrame:
            timediff = -df.date.diff(-periods)
            # Find row pairs where time difference is within baseline ± max_delta
            is_match = timediff.between(baseline - max_delta, baseline + max_delta)
            if len(is_match[is_match]) == 0:
                return gpd.GeoDataFrame()
            primary_cols = df[is_match].rename(columns=renamer_1)
            secondary_cols = df.shift(-periods)[is_match][renamer_2.keys()]
            secondary_cols = secondary_cols.rename(columns=renamer_2)
            return gpd.GeoDataFrame(pd.concat([primary_cols, secondary_cols], axis=1))

        # Group by columns that must match for InSAR
        for _, group in df_bursts.groupby(self._INSAR_MATCH_COLUMNS):
            periods = 1
            pairs.append(match_pairs(group, periods))
            while (-group.date.diff(-periods) <= baseline).any():
                periods += 1
                pairs.append(match_pairs(group, periods))
        if not pairs:
            return BurstPairCollection()
        df_pairs = gpd.GeoDataFrame(pd.concat(pairs))
        if len(df_pairs) == 0:
            return BurstPairCollection()
        return BurstPairCollection(df_pairs)

    @staticmethod
    @override
    def _row_to_entry(row: Series) -> Burst:
        """Convert a DataFrame row to a Burst instance."""
        rel_orbit = _scalar_or_none(row.get("_rel_orbit_number"))
        return Burst(
            date=cast("Timestamp", row["date"]),
            burst_id=cast("int", row["burst_id"]),
            sub_swath=cast("SubSwath", row["sub_swath"]),
            polarization=cast("Polarization", row["polarization"]),
            abs_burst_id=cast("int | None", _scalar_or_none(row.get("abs_burst_id"))),
            cdse_id=cast("str | None", _scalar_or_none(row.get("cdse_id"))),
            parent_product_name=cast(
                "str | None",
                _scalar_or_none(row.get("parent_product_name")),
            ),
            _rel_orbit_number=None if rel_orbit is None else int(rel_orbit),
            _pass_direction=cast(
                "PassDirection | None",
                _scalar_or_none(row.get("_pass_direction")),
            ),
            _geometry=cast("Polygon | MultiPolygon | None", _scalar_or_none(row.get("geometry"))),
        )

    @classmethod
    def from_bursts(cls, bursts: Collection[Burst]) -> Self:
        """Create a BurstCollection from a collection of Burst objects.

        Parameters
        ----------
        bursts
            Collection of Burst instances.

        Returns
        -------
        BurstCollection
            New BurstCollection instance.
        """
        if len(bursts) == 0:
            return cls()
        df = cls._bursts_to_dataframe(bursts)
        return cls(df)

    @classmethod
    def from_cdse_response(cls, response: CDSEResponse) -> Self:
        """Create a BurstCollection directly from a CDSEResponse.

        Parameters
        ----------
        response
            CDSE API response containing burst data.

        Returns
        -------
        BurstCollection
            New BurstCollection instance.
        """
        # Get the GeoDataFrame from the response
        cdse_df = response.to_geodataframe()

        # Map CDSE column names to Burst field names
        df = gpd.GeoDataFrame(
            {
                "burst_id": cdse_df["burst_id"],
                "sub_swath": cdse_df["swath_identifier"],
                "polarization": cdse_df["polarisation_channels"],
                "date": pd.to_datetime(cdse_df["beginning_date_time"]),
                "abs_burst_id": cdse_df["absolute_burst_id"],
                "cdse_id": cdse_df["id"],
                "parent_product_name": cdse_df["parent_product_name"],
                "geometry": cdse_df["geometry"],
                # Propagated from CDSE so footprint properties need no LUT lookup.
                "_rel_orbit_number": cdse_df["relative_orbit_number"],
                "_pass_direction": cdse_df["orbit_direction"].str.capitalize(),
                # Used only to resolve near-identical-timestamp duplicates.
                "_lines_per_burst": cdse_df["lines_per_burst"],
                "_samples_per_burst": cdse_df["samples_per_burst"],
            },
            geometry="geometry",
            crs=cdse_df.crs,
        )

        return cls(df)

    @staticmethod
    def _bursts_to_dataframe(bursts: Collection[Burst]) -> gpd.GeoDataFrame:
        """Convert a collection of Burst objects to a GeoDataFrame."""
        data = [
            {
                "date": burst.date,
                "burst_id": burst.burst_id,
                "sub_swath": burst.sub_swath,
                "polarization": burst.polarization,
                "abs_burst_id": burst.abs_burst_id,
                "cdse_id": burst.cdse_id,
                "parent_product_name": burst.parent_product_name,
                # Stored overrides, not the LUT-backed properties, to avoid a burst-map load.
                "geometry": burst._geometry,  # noqa: SLF001
                "_rel_orbit_number": burst._rel_orbit_number,  # noqa: SLF001
                "_pass_direction": burst._pass_direction,  # noqa: SLF001
            }
            for burst in bursts
        ]
        return gpd.GeoDataFrame(data, geometry="geometry")

    def _remove_timestamp_duplicates(
        self,
        threshold: Timedelta | None = None,
    ) -> None:
        """Collapse near-identical-timestamp duplicates, keeping the most complete burst.

        The same physical burst can appear twice in a CDSE search: identical burst_id,
        sub_swath, and polarization with sensing times differing by microseconds.
        The two products may differ in extent, so the most complete one (largest
        ``lines_per_burst * samples_per_burst``) is kept, falling back to the most
        recent acquisition when the size is equal or unknown.
        """
        if threshold is None:
            threshold = Timedelta(hours=1)

        # Dimensions are only needed here; drop them afterwards so they don't leak
        # into pair collections or the HTML display.
        dim_cols = ["_lines_per_burst", "_samples_per_burst"]

        if len(self.df) == 0:
            self.df = self.df.drop(columns=dim_cols)
            return

        df_sorted = self.df.sort_values([*self._INSAR_MATCH_COLUMNS, "date"])

        # A new cluster starts at the first row of each footprint group or wherever
        # the gap to the previous row exceeds the threshold.
        gap = df_sorted.groupby(self._INSAR_MATCH_COLUMNS)["date"].diff()
        cluster = (gap.isna() | (gap.abs() > threshold)).cumsum()

        # Completeness ranks candidates within a cluster; missing dimensions sort
        # last (-1) so the most-recent date decides
        lines = pd.to_numeric(df_sorted["_lines_per_burst"], errors="coerce")
        samples = pd.to_numeric(df_sorted["_samples_per_burst"], errors="coerce")
        completeness = (lines * samples).fillna(-1)

        ranked = df_sorted.assign(_cluster=cluster, _completeness=completeness).sort_values(
            ["_cluster", "_completeness", "date"],
        )
        keep_idx = ranked.groupby("_cluster").tail(1).index

        cluster_sizes = cluster.groupby(cluster).transform("size")
        for idx in keep_idx:
            if cluster_sizes[idx] > 1:
                warnings.warn(
                    "Detected multiple products with nearly identical timestamp for burst "
                    f"'{self._row_to_entry(df_sorted.loc[idx])}'. "
                    "Keeping the most complete product.",
                    stacklevel=3,
                )

        self.df = (
            df_sorted.loc[keep_idx]
            .sort_values("date")
            .drop(columns=dim_cols)
            .reset_index(drop=True)
        )


class BurstPairCollection(_BaseBurstCollection[BurstPair]):
    """Collection of Sentinel-1 burst pairs for interferometric processing.

    Each entry contains a primary and secondary burst with matching spatial
    parameters but different acquisition times.
    """

    # Columns that make a burst pair unique
    _UNIQUE_COLUMNS: ClassVar[list[str]] = [
        "burst_id",
        "sub_swath",
        "polarization",
        "date_1",
        "date_2",
    ]
    # Both bursts' fields, suffixed _1 (primary) and _2 (secondary) where they differ
    _ALL_COLUMNS: ClassVar[list[str]] = [
        "burst_id",
        "sub_swath",
        "polarization",
        "date_1",
        "date_2",
        "abs_burst_id_1",
        "abs_burst_id_2",
        "cdse_id_1",
        "cdse_id_2",
        "parent_product_name_1",
        "parent_product_name_2",
        "geometry",
        "_rel_orbit_number",
        "_pass_direction",
    ]
    # Footprint overrides propagated from CDSE (shared by both bursts of a pair);
    # optional so raw-dataframe callers need not supply them.
    _OPTIONAL_COLUMNS: ClassVar[list[str]] = ["_rel_orbit_number", "_pass_direction"]
    # Entry names for display
    _ENTRY_NAME_SINGULAR: ClassVar[str] = "Burst pair"
    _ENTRY_NAME_PLURAL: ClassVar[str] = "Burst pairs"

    @staticmethod
    @override
    def _row_to_entry(row: Series) -> BurstPair:
        """Convert a DataFrame row to a BurstPair instance."""
        # Footprint overrides are shared by both bursts of the pair.
        rel_orbit = _scalar_or_none(row.get("_rel_orbit_number"))
        overrides: dict[str, Any] = {
            "_rel_orbit_number": None if rel_orbit is None else int(rel_orbit),
            "_pass_direction": cast(
                "PassDirection | None",
                _scalar_or_none(row.get("_pass_direction")),
            ),
            "_geometry": cast(
                "Polygon | MultiPolygon | None",
                _scalar_or_none(row.get("geometry")),
            ),
        }
        primary = Burst(
            date=cast("Timestamp", row["date_1"]),
            burst_id=cast("int", row["burst_id"]),
            sub_swath=cast("SubSwath", row["sub_swath"]),
            polarization=cast("Polarization", row["polarization"]),
            abs_burst_id=cast("int | None", _scalar_or_none(row.get("abs_burst_id_1"))),
            cdse_id=cast("str | None", _scalar_or_none(row.get("cdse_id_1"))),
            parent_product_name=cast(
                "str | None",
                _scalar_or_none(row.get("parent_product_name_1")),
            ),
            **overrides,
        )
        secondary = Burst(
            date=cast("Timestamp", row["date_2"]),
            burst_id=cast("int", row["burst_id"]),
            sub_swath=cast("SubSwath", row["sub_swath"]),
            polarization=cast("Polarization", row["polarization"]),
            abs_burst_id=cast("int | None", _scalar_or_none(row.get("abs_burst_id_2"))),
            cdse_id=cast("str | None", _scalar_or_none(row.get("cdse_id_2"))),
            parent_product_name=cast(
                "str | None",
                _scalar_or_none(row.get("parent_product_name_2")),
            ),
            **overrides,
        )
        return BurstPair(primary, secondary)

    @classmethod
    def from_pairs(cls, pairs: Collection[BurstPair]) -> Self:
        """Create a BurstPairCollection from a collection of BurstPair objects.

        Parameters
        ----------
        pairs
            Collection of BurstPair instances.

        Returns
        -------
        BurstPairCollection
            New BurstPairCollection instance.
        """
        if len(pairs) == 0:
            return cls()
        df = cls._pairs_to_dataframe(pairs)
        return cls(df)

    @staticmethod
    def _pairs_to_dataframe(pairs: Collection[BurstPair]) -> gpd.GeoDataFrame:
        """Convert a collection of BurstPair objects to a GeoDataFrame."""
        data = [
            {
                "burst_id": pair.burst_id,
                "sub_swath": pair.sub_swath,
                "polarization": pair.polarization,
                "date_1": pair.primary.date,
                "date_2": pair.secondary.date,
                "abs_burst_id_1": pair.primary.abs_burst_id,
                "abs_burst_id_2": pair.secondary.abs_burst_id,
                "cdse_id_1": pair.primary.cdse_id,
                "cdse_id_2": pair.secondary.cdse_id,
                "parent_product_name_1": pair.primary.parent_product_name,
                "parent_product_name_2": pair.secondary.parent_product_name,
                # Stored overrides from the primary burst (not the LUT properties).
                "geometry": pair.primary._geometry,  # noqa: SLF001
                "_rel_orbit_number": pair.primary._rel_orbit_number,  # noqa: SLF001
                "_pass_direction": pair.primary._pass_direction,  # noqa: SLF001
            }
            for pair in pairs
        ]
        return gpd.GeoDataFrame(data, geometry="geometry")

    def unpair(self) -> BurstCollection:
        """Convert burst pairs back into individual bursts.

        Separates each burst pair into its primary and secondary bursts,
        creating a BurstCollection containing all individual bursts.

        Returns
        -------
        BurstCollection
            Collection containing all primary and secondary bursts from the pairs.
        """
        df_pairs = self.df

        # Define column mappings for primary and secondary bursts
        renamer_1 = {n: n.removesuffix("_1") for n in df_pairs.columns if n.endswith("_1")}
        renamer_2 = {n: n.removesuffix("_2") for n in df_pairs.columns if n.endswith("_2")}

        # Common columns (no suffix) are shared between primary and secondary
        common_cols = set(df_pairs.columns) - set(renamer_1.keys()) - set(renamer_2.keys())
        df_primary = df_pairs[list(renamer_1.keys())].rename(columns=renamer_1)
        df_secondary = df_pairs[list(renamer_2.keys())].rename(columns=renamer_2)
        df_common = df_pairs[list(common_cols)]

        # Add common columns to both dataframes
        df_primary = pd.concat([df_primary, df_common], axis=1)
        df_secondary = pd.concat([df_secondary, df_common], axis=1)

        # Combine primary and secondary bursts
        df_all = gpd.GeoDataFrame(
            pd.concat([df_primary, df_secondary], ignore_index=True),
        )

        # BurstCollection removes duplicated bursts
        return BurstCollection(df_all)
