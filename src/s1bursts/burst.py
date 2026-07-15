"""Sentinel-1 burst data structures and collections."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast, overload
import warnings

from deepfreezer.datasets.copernicus import (
    create_s1_geometry_lut,
    create_s1_orbit_direction_lut,
    create_s1_orbit_number_lut,
)
import geopandas as gpd
import pandas as pd
from pandas import DataFrame, Series, Timedelta, Timestamp

from s1bursts import ui

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator

    from shapely import Polygon

    from s1bursts.cdse import CDSEResponse
    from s1bursts.query import OrbitDirection, Polarization, SubSwath


class BurstFootprintMixin:
    """Mixin that provides lookup-table-backed properties for burst footprint attributes.

    Requires subclasses to declare ``burst_id: int`` and ``sub_swath: SubSwath``.
    """

    __slots__ = ()
    burst_id: int
    sub_swath: SubSwath

    def _lut_lookup[T](self, lut: dict[tuple[int, SubSwath], T]) -> T:
        """Look up this burst in a LUT keyed by (burst_id, sub_swath).

        Parameters
        ----------
        lut : dict[tuple[int, str], T]
            Lookup table keyed by ``(burst_id, sub_swath)``.

        Returns
        -------
        T
            Value from the lookup table.

        Raises
        ------
        KeyError
            If the burst is not found in the map.
        """
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
        """Relative orbit number looked up from the Sentinel-1 burst map.

        Returns
        -------
        int
            Relative orbit number.

        Raises
        ------
        KeyError
            If the burst is not found in the map.
        """
        return self._lut_lookup(create_s1_orbit_number_lut())

    @property
    def orbit_direction(self) -> OrbitDirection:
        """Orbit pass direction looked up from the Sentinel-1 burst map.

        Returns
        -------
        OrbitDirection
            ``"ASCENDING"`` or ``"DESCENDING"``.

        Raises
        ------
        KeyError
            If the burst is not found in the map.
        """
        return self._lut_lookup(create_s1_orbit_direction_lut())

    @property
    def geometry(self) -> Polygon:
        """Burst footprint polygon looked up from the Sentinel-1 burst map.

        Returns
        -------
        Polygon
            Footprint geometry.

        Raises
        ------
        KeyError
            If the burst is not found in the map.
        """
        return self._lut_lookup(create_s1_geometry_lut())

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
        KeyError
            If the burst is not found in the burst map.
        """
        return f"{self.rel_orbit_number:03}_{self.burst_id:06}_{self.sub_swath}"


@dataclass(frozen=True, slots=True)
class Burst(BurstFootprintMixin):
    """
    Represents a single Sentinel-1 burst.

    A burst is a segment of SAR data acquisition characterized by its spatial location,
    temporal acquisition time, and imaging parameters.
    """

    burst_id: int
    sub_swath: SubSwath
    polarization: Polarization
    date: Timestamp
    abs_burst_id: int | None = field(default=None, compare=False, hash=False)
    cdse_id: str | None = field(default=None, compare=False, hash=False)
    parent_product_name: str | None = field(default=None, compare=False, hash=False)

    def __repr__(self) -> str:
        """Return string representation of the Burst."""
        return (
            f"Burst(burst_id='{self.burst_id:06}', sub_swath='{self.sub_swath}', "
            f"polarization='{self.polarization}', date='{self.date:%Y-%m-%d}')"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Instantiate from a mapping, silently dropping unknown and non-init fields.

        Parameters
        ----------
        data : dict[str, Any]
            Mapping of field names to values. Extra keys and ``init=False``
            fields are filtered before construction.

        Returns
        -------
        Self
            A new ``Burst`` instance.
        """
        init_keys = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in data.items() if k in init_keys})


@dataclass(slots=True)
class BurstPair:
    """
    Represents a pair of Sentinel-1 bursts for interferometric processing.

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
    def orbit_direction(self) -> OrbitDirection:
        """Orbit pass direction, looked up via the primary burst.

        Returns
        -------
        OrbitDirection
            ``"ASCENDING"`` or ``"DESCENDING"``.

        Raises
        ------
        KeyError
            If the burst is not found in the burst map.
        """
        return self.primary.orbit_direction

    @property
    def geometry(self) -> Polygon:
        """Burst footprint, looked up via the primary burst.

        Returns
        -------
        Polygon
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

    Provides iteration, indexing, concatenation, deduplication, and
    HTML display for both BurstCollection and BurstPairCollection.
    Subclasses must define the class-level column constants and implement
    ``_row_to_entry``.
    """

    # Subclasses must define these class variables
    # Columns that uniquely describe an entry
    _UNIQUE_COLUMNS: ClassVar[list[str]]
    # All columns of an entry
    _ALL_COLUMNS: ClassVar[list[str]]
    # Entry names for display
    _ENTRY_NAME_SINGULAR: ClassVar[str]
    _ENTRY_NAME_PLURAL: ClassVar[str]
    # Columns that uniquely describe a burst footprint
    _FOOTPRINT_UNIQUE_COLUMNS: ClassVar[list[str]] = ["burst_id", "sub_swath"]
    # Columns that must match to make bursts compatibile for interferometry
    _INSAR_MATCH_COLUMNS: ClassVar[list[str]] = [
        "burst_id",
        "sub_swath",
        "polarization",
    ]

    @staticmethod
    @abstractmethod
    def _row_to_entry(row: Series) -> TEntry:
        """Convert a DataFrame row to an entry (Burst or BurstPair).

        Subclasses must implement this method.
        """
        ...

    def __init__(self, df: gpd.GeoDataFrame | None = None) -> None:
        """
        Initialize collection with a GeoDataFrame.

        Parameters
        ----------
        df : gpd.GeoDataFrame | None
            DataFrame with burst data. If ``None``, an empty collection is created.

        Raises
        ------
        ValueError
            If required columns are missing from the dataframe.

        """
        if df is None:
            df = gpd.GeoDataFrame(
                {col: [] for col in self._ALL_COLUMNS},
                geometry="geometry",
            )
        self._validate_dataframe(df)
        df = df[self._ALL_COLUMNS]
        df = self._deduplicate_dataframe(df)
        self.df = df.sort_values(self._UNIQUE_COLUMNS)

    def _validate_dataframe(self, df: gpd.GeoDataFrame) -> None:
        """
        Validate that the dataframe has the required type and columns.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            DataFrame to validate.

        Raises
        ------
        TypeError
            If not a geopandas.GeoDataFrame
        ValueError
            If required columns are missing.

        """
        if not isinstance(df, gpd.GeoDataFrame):
            msg = f"`df` must be a GeoDataFrame, received '{type(df).__name__}'."
            raise TypeError(msg)
        df_columns = set(df.columns)
        all_columns = set(self._ALL_COLUMNS)
        missing_columns = all_columns - df_columns
        if missing_columns:
            msg = f"DataFrame is missing required columns: {missing_columns}"
            raise ValueError(msg)

        # Check for unexpected columns (warn but don't fail)
        unexpected_columns = df_columns - all_columns
        if unexpected_columns:
            warnings.warn(
                f"DataFrame contains unexpected columns: {unexpected_columns}",
                stacklevel=3,
            )

    def _deduplicate_dataframe(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Remove duplicate entries based on unique column combinations.

        Parameters
        ----------
        df : gpd.GeoDataFrame
            A dataframe which can contain duplicates.

        Returns
        -------
        gpd.GeoDataFrame
            The dataframe with duplicates removed.
        """
        # Remove duplicates based on the columns which uniquely identify an entry
        return df.drop_duplicates(subset=self._UNIQUE_COLUMNS, ignore_index=True)

    def __len__(self) -> int:
        """
        Return the number of entries in the collection.

        Returns
        -------
        int
            Number of entries in the collection.

        """
        return len(self.df)

    def __add__(self, other: Self) -> Self:
        """
        Concatenate two collections with duplicates removed.

        Parameters
        ----------
        other : Self
            Another collection of the same type to concatenate.

        Returns
        -------
        Self
            Concatenated collection.

        """
        # Concatenate dataframes
        combined_df = gpd.GeoDataFrame(
            pd.concat([self.df, other.df], ignore_index=True),
        )

        return type(self)(combined_df)

    def __iter__(self) -> Iterator[TEntry]:
        """
        Iterate over individual entries in the collection.

        Yields
        ------
        TEntry
            Individual entry instances from the collection (Burst or BurstPair).

        """
        for _, row in self.df.iterrows():
            yield self._row_to_entry(row)

    @overload
    def __getitem__(self, index: int) -> TEntry: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> TEntry | Self:
        """
        Get a specific entry or slice of entries from the collection.

        Parameters
        ----------
        index : int | slice
            Integer index for a single entry, or slice for multiple entries.

        Returns
        -------
        TEntry | Self
            Single entry if index is int, or new collection of same type if index is slice.

        """
        if isinstance(index, slice):
            return type(self)(self.df.iloc[index])
        return self._row_to_entry(self.df.iloc[index])

    def _repr_html_(self) -> str:
        """
        Generate HTML representation for Jupyter notebooks.

        Returns
        -------
        str
            HTML string for rich display.

        """
        # Get unique footprints
        unique_footprints = self.df.drop_duplicates(
            subset=self._FOOTPRINT_UNIQUE_COLUMNS,
        )

        # Get stats
        n_bursts = len(self)
        n_footprints = len(unique_footprints)

        # Create map SVG
        map_svg = ui.create_footprint_svg(unique_footprints)

        # Create world map with target
        world_map_svg = ui.create_world_map_with_target(unique_footprints)

        # Create dataframe HTML which only displays the unique columns
        df_html = self.df[self._UNIQUE_COLUMNS]._repr_html_()  # type: ignore[operator]

        # Create CSS and HTML structure
        css = ui.get_burstinfo_css()

        # Collection type
        col_name = type(self).__name__

        # Entry name
        entry_name = self._ENTRY_NAME_SINGULAR if n_bursts == 1 else self._ENTRY_NAME_PLURAL

        # Footprint name
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
    """
    Collection of Sentinel-1 bursts with geospatial and temporal data.

    Provides methods for working with multiple bursts, including finding
    interferometric pairs, filtering, and visualization.
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
    ]
    # Entry names for display
    _ENTRY_NAME_SINGULAR: ClassVar[str] = "Burst"
    _ENTRY_NAME_PLURAL: ClassVar[str] = "Bursts"

    def __init__(self, df: gpd.GeoDataFrame | None = None) -> None:
        """Initialize BurstCollection from a GeoDataFrame.

        Calls base class validation and deduplication, then removes bursts
        with nearly identical timestamps (CDSE API artefact).

        Parameters
        ----------
        df : gpd.GeoDataFrame | None
            DataFrame with burst data. If ``None``, an empty collection is created.

        Raises
        ------
        TypeError
            If df is not a GeoDataFrame.
        ValueError
            If required columns are missing.
        """
        super().__init__(df)
        self._remove_timestamp_duplicates()

    def find_pairs(
        self,
        baseline: Timedelta | int,
        max_delta: Timedelta | int = 24,
    ) -> BurstPairCollection:
        """
        Find interferometric burst pairs with a specified temporal baseline.

        Parameters
        ----------
        baseline : pandas.Timedelta | int
            Temporal baseline between primary and secondary bursts.
            If integer is provided, the assumed unit is days.
        max_delta : pandas.Timedelta | int
            Maximum allowed deviation from baseline.
            If integer is provided, the assumed unit is hours.
            Default is 24 hours.

        Returns
        -------
        BurstPairCollection
            Collection of burst pairs matching the baseline criteria.

        Raises
        ------
        ValueError
            If no valid burst pairs are found.
        """
        if not isinstance(baseline, Timedelta):
            baseline = Timedelta(baseline, unit="days")
        if not isinstance(max_delta, Timedelta):
            max_delta = Timedelta(max_delta, unit="hours")
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
    def _row_to_entry(row: Series) -> Burst:
        """Convert a DataFrame row to a Burst instance."""
        return Burst(
            date=cast("Timestamp", row["date"]),
            burst_id=cast("int", row["burst_id"]),
            sub_swath=cast("SubSwath", row["sub_swath"]),
            polarization=cast("Polarization", row["polarization"]),
            abs_burst_id=cast("int | None", row.get("abs_burst_id")),
            cdse_id=cast("str | None", row.get("cdse_id")),
            parent_product_name=cast("str | None", row.get("parent_product_name")),
        )

    @classmethod
    def from_bursts(cls, bursts: Collection[Burst]) -> Self:
        """
        Create a BurstCollection from a collection of Burst objects.

        Parameters
        ----------
        bursts : Collection[Burst]
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
        """
        Create a BurstCollection directly from a CDSEResponse.

        Parameters
        ----------
        response : CDSEResponse
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
            },
            geometry="geometry",
            crs=cdse_df.crs,
        )

        return cls(df)

    @staticmethod
    def _bursts_to_dataframe(bursts: Collection[Burst]) -> gpd.GeoDataFrame:
        """
        Convert a collection of Burst objects to a GeoDataFrame.

        Parameters
        ----------
        bursts : Collection[Burst]
            Collection of Burst instances.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with burst data.

        """
        data = [
            {
                "date": burst.date,
                "burst_id": burst.burst_id,
                "sub_swath": burst.sub_swath,
                "polarization": burst.polarization,
                "abs_burst_id": burst.abs_burst_id,
                "cdse_id": burst.cdse_id,
                "parent_product_name": burst.parent_product_name,
                "geometry": burst.geometry,
            }
            for burst in bursts
        ]
        return gpd.GeoDataFrame(data, geometry="geometry")

    def _remove_timestamp_duplicates(
        self,
        threshold: Timedelta | None = None,
    ) -> None:
        """Remove bursts with nearly identical timestamps, keeping the most recent.

        The CDSE API sometimes returns bursts with identical
        burst_id, sub_swath, and polarization but timestamps that differ by a
        few milliseconds. This method identifies and removes such duplicates.

        Parameters
        ----------
        threshold : Timedelta | None, optional
            Maximum time difference to consider bursts as duplicates.
            Default is 1 hour.
        """
        if threshold is None:
            threshold = Timedelta(hours=1)

        if len(self.df) == 0:
            return

        df_sorted = self.df.sort_values("date")

        # Calculate difference with next row
        time_diff = df_sorted.groupby(self._INSAR_MATCH_COLUMNS)["date"].diff(-1)

        # Find all close matches (index of older burst)
        is_close = time_diff.abs() <= threshold

        if not is_close.any():
            return

        # Warn about removed duplicates
        for _, row in df_sorted.loc[is_close].iterrows():
            warnings.warn(
                f"Detected multiple products with nearly identical timestamp for burst "
                f"'{self._row_to_entry(row)}'. "
                "Keeping burst with more recent timestamp.",
                stacklevel=3,
            )
        # Keep the more recent burst (remove older ones marked as is_close)
        self.df = df_sorted.loc[~is_close].reset_index(drop=True)


class BurstPairCollection(_BaseBurstCollection[BurstPair]):
    """
    Collection of Sentinel-1 burst pairs for interferometric processing.

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
    # All valid fields from Burst dataclass
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
    ]
    # Entry names for display
    _ENTRY_NAME_SINGULAR: ClassVar[str] = "Burst pair"
    _ENTRY_NAME_PLURAL: ClassVar[str] = "Burst pairs"

    @staticmethod
    def _row_to_entry(row: Series) -> BurstPair:
        """Convert a DataFrame row to a BurstPair instance."""
        primary = Burst(
            date=cast("Timestamp", row["date_1"]),
            burst_id=cast("int", row["burst_id"]),
            sub_swath=cast("SubSwath", row["sub_swath"]),
            polarization=cast("Polarization", row["polarization"]),
            abs_burst_id=cast("int | None", row.get("abs_burst_id_1")),
            cdse_id=cast("str | None", row.get("cdse_id_1")),
            parent_product_name=cast("str | None", row.get("parent_product_name_1")),
        )
        secondary = Burst(
            date=cast("Timestamp", row["date_2"]),
            burst_id=cast("int", row["burst_id"]),
            sub_swath=cast("SubSwath", row["sub_swath"]),
            polarization=cast("Polarization", row["polarization"]),
            abs_burst_id=cast("int | None", row.get("abs_burst_id_2")),
            cdse_id=cast("str | None", row.get("cdse_id_2")),
            parent_product_name=cast("str | None", row.get("parent_product_name_2")),
        )
        return BurstPair(primary, secondary)

    @classmethod
    def from_pairs(cls, pairs: Collection[BurstPair]) -> Self:
        """
        Create a BurstPairCollection from a collection of BurstPair objects.

        Parameters
        ----------
        pairs : Collection[BurstPair]
            Collection of Burst instances.

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
        """
        Convert a collection of BurstPair objects to a GeoDataFrame.

        Parameters
        ----------
        pairs : Collection[BurstPair]
            Collection of BurstPair instances.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with burst pair data.

        """
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
                "geometry": pair.geometry,
            }
            for pair in pairs
        ]
        return gpd.GeoDataFrame(data, geometry="geometry")

    def unpair(self) -> BurstCollection:
        """
        Convert burst pairs back into individual bursts.

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
        df_primary = pd.concat([df_primary, df_common], axis=1)  # type: ignore[assignment]
        df_secondary = pd.concat([df_secondary, df_common], axis=1)  # type: ignore[assignment]

        # Combine primary and secondary bursts
        df_all = gpd.GeoDataFrame(
            pd.concat([df_primary, df_secondary], ignore_index=True),
        )

        # BurstCollection removes duplicated bursts
        return BurstCollection(df_all)
