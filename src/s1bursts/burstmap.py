"""Sentinel-1 Burst ID map dataset and lookup tables."""

from __future__ import annotations

import functools
from io import BytesIO
import logging
from typing import TYPE_CHECKING, Literal, cast
from zipfile import ZipFile

from freezecore.utils import format_valid_options
from freezecore.vectools import drop_z_if_zero, save_and_read_parquet, simplify_multipolygons
from freezecore.vectordata import DatasetMetadata, GeoVectorData
import geopandas as gpd
from pandas.api.types import CategoricalDtype

if TYPE_CHECKING:
    from pathlib import Path

    from shapely import Polygon

logger = logging.getLogger(__name__)

type Mode = Literal["IW", "EW"]
type SubSwath = Literal["IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"]
type OrbitDirection = Literal["ASCENDING", "DESCENDING"]

S1_MODE_DESC = {
    "IW": "Interferometric Wide Swath",
    "EW": "Extra Wide Swath",
}


class Sentinel1Bursts(GeoVectorData):
    """
    Sentinel-1 Burst ID map.

    Sentinel-1 performs systematic acquisition of bursts in both IW and EW modes.
    The bursts overlap almost perfectly between different passes and are always
    located in the same place. With the deployment of the SAR processor S1-IPF 3.4,
    a new element has been added to the products annotations: the Burst ID, which
    should help the end-user to identify a burst area of interest and facilitate
    searches.

    **Important**: The Burst ID is unique in each sub-swath.
    For this reason to identify a burst univocally it is necessary to specify both
    the "*subswath_name*" (see below) and the "*burst_id*".

    Two maps are provided: one for EW and one for IW acquisition modes.
    The maps have a validity that covers the entire time span of the mission and they
    are global, i.e., they include as well information where no SAR data is acquired.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        mode: Mode = "IW",
    ) -> None:
        """
        Initialize the Sentinel-1 Burst ID map.

        Parameters
        ----------
        cache_dir : str or Path or None, default: None
            The root directory where the dataset is stored. Defaults to the 'FREEZECORE_CACHE'
            environment variable or the system cache directory (in that order).
        mode : {"IW", "EW"}, default: "IW"
            The Sentinel-1 acquisition mode.

        """
        super().__init__(cache_dir)
        # Validate mode
        try:
            S1_MODE_DESC[mode]
        except KeyError as e:
            valid_values_bulleted = format_valid_options(S1_MODE_DESC)
            msg = f"Invalid product: '{mode}'. Use one of: {valid_values_bulleted}"
            raise ValueError(msg) from e
        self.mode = mode

    @property
    def metadata(self) -> DatasetMetadata:
        """Metadata container for dataset attribution and licensing."""
        return DatasetMetadata(
            name="Sentinel-1 Burst ID Map",
            source_url="https://sar-mpc.eu/files/S1_burstid_20220530.zip",
            attribution="SAR Mission Performance Cluster (SAR-MPC)",
            version="20220530",
            additional_fields={"mode": S1_MODE_DESC[self.mode]},
        )

    @property
    def raw_path(self) -> Path:
        """Path to the raw downloaded data."""
        return self.cache_dir / "raw/sar/s1-burstid-map.zip"

    @property
    def processed_path(self) -> Path:
        """Path to the processed data."""
        return self.cache_dir / f"processed/sar/s1-{self.mode.lower()}-burstid-map.parquet"

    def _prepare(self) -> None:
        """Extract the ZIP, select most important columns and save as GeoParquet."""
        logger.info(
            "Processing data and saving '%s' to '%s'.",
            self.processed_path.name,
            self.processed_path.parent,
        )
        # Load SpatiaLite database contained in ZIP
        with ZipFile(self.raw_path) as zf:
            # Find SpatiaLite database (close to the end of the file list)
            filename: str | None = None
            for f in reversed(zf.filelist):
                if f"{self.mode}/sqlite/" in f.filename and f.filename.endswith(".sqlite3"):
                    filename = f.filename
                    break
            if filename is None:
                msg = f"Couldn't locate SpatiaLite database in '{self.raw_path.name}'."
                raise FileNotFoundError(msg)
            bursts = gpd.read_file(BytesIO(zf.read(filename)))

        # Convert column dtypes
        ordered_cat = CategoricalDtype(ordered=True)
        dtype_map: dict[str, str | CategoricalDtype] = {
            "burst_id": "uint32",
            "relative_orbit_number": "uint8",
            "subswath_name": ordered_cat,
            "orbit_pass": "category",
        }
        bursts = bursts.pipe(drop_z_if_zero).pipe(simplify_multipolygons).astype(dtype_map)
        save_and_read_parquet(bursts, self.processed_path)


@functools.cache
def create_s1_orbit_direction_lut(
    cache_dir: str | Path | None = None,
    mode: Mode = "IW",
) -> dict[tuple[int, SubSwath], OrbitDirection]:
    """Build a lookup table mapping burst footprints to their orbit pass direction.

    Loads the Sentinel-1 burst ID map and returns a dictionary keyed by
    ``(burst_id, subswath_name)`` tuples whose values are the corresponding
    orbit pass direction (``"ASCENDING"`` or ``"DESCENDING"``).

    Because the burst footprints are stationary over the lifetime of the mission,
    the orbit direction is fixed per footprint and does not depend on the
    acquisition date.

    Results are cached in memory after the first call.

    Parameters
    ----------
    cache_dir : str or Path or None, default: None
        Root directory for cached dataset files. Forwarded to
        :class:`Sentinel1Bursts`.
    mode : {"IW", "EW"}, default: "IW"
        Sentinel-1 acquisition mode. Forwarded to :class:`Sentinel1Bursts`.

    Returns
    -------
    dict[tuple[int, SubSwath], str]
        Mapping of ``(burst_id, subswath_name)`` → orbit direction string.

    """
    df = Sentinel1Bursts(cache_dir=cache_dir, mode=mode).get_data(download=True)
    df = df.set_index(["burst_id", "subswath_name"])
    return cast("dict[tuple[int, SubSwath], OrbitDirection]", df["orbit_pass"].to_dict())


@functools.cache
def create_s1_orbit_number_lut(
    cache_dir: str | Path | None = None,
    mode: Mode = "IW",
) -> dict[tuple[int, SubSwath], int]:
    """Build a lookup table mapping burst footprints to their relative orbit number.

    Loads the Sentinel-1 burst ID map and returns a dictionary keyed by
    ``(burst_id, subswath_name)`` tuples whose values are the corresponding
    relative orbit number.

    Because the burst footprints are stationary over the lifetime of the mission,
    the relative orbit number is fixed per footprint and does not depend on the
    acquisition date.

    Results are cached in memory after the first call.

    Parameters
    ----------
    cache_dir : str or Path or None, default: None
        Root directory for cached dataset files. Forwarded to
        :class:`Sentinel1Bursts`.
    mode : {"IW", "EW"}, default: "IW"
        Sentinel-1 acquisition mode. Forwarded to :class:`Sentinel1Bursts`.

    Returns
    -------
    dict[tuple[int, SubSwath], int]
        Mapping of ``(burst_id, subswath_name)`` → relative orbit number.

    """
    df = Sentinel1Bursts(cache_dir=cache_dir, mode=mode).get_data(download=True)
    df = df.set_index(["burst_id", "subswath_name"])
    return cast("dict[tuple[int, SubSwath], int]", df["relative_orbit_number"].to_dict())


@functools.cache
def create_s1_geometry_lut(
    cache_dir: str | Path | None = None,
    mode: Mode = "IW",
) -> dict[tuple[int, SubSwath], Polygon]:
    """Build a lookup table mapping burst footprints to their geometry.

    Loads the Sentinel-1 burst ID map and returns a dictionary keyed by
    ``(burst_id, subswath_name)`` tuples whose values are the corresponding
    footprint polygons.

    Because the burst footprints are stationary over the lifetime of the mission,
    the geometry is fixed per footprint and does not depend on the acquisition date.

    Results are cached in memory after the first call.

    Parameters
    ----------
    cache_dir : str or Path or None, default: None
        Root directory for cached dataset files. Forwarded to
        :class:`Sentinel1Bursts`.
    mode : {"IW", "EW"}, default: "IW"
        Sentinel-1 acquisition mode. Forwarded to :class:`Sentinel1Bursts`.

    Returns
    -------
    dict[tuple[int, SubSwath], Polygon]
        Mapping of ``(burst_id, subswath_name)`` → footprint geometry.

    """
    df = Sentinel1Bursts(cache_dir=cache_dir, mode=mode).get_data(download=True)
    df = df.set_index(["burst_id", "subswath_name"])
    return cast("dict[tuple[int, SubSwath], Polygon]", df["geometry"].to_dict())
