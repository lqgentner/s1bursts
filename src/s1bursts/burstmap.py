"""Sentinel-1 Burst ID map dataset and lookup tables."""

from __future__ import annotations

import functools
from io import BytesIO
import logging
from typing import TYPE_CHECKING, Literal, cast, override
from zipfile import ZipFile

from freezebase.utils import format_valid_options
from freezebase.vectools import drop_z_if_zero, simplify_multipolygons, write_parquet
from freezebase.vectordata import DatasetMetadata, GeoVectorData
import geopandas as gpd
from pandas.api.types import CategoricalDtype

if TYPE_CHECKING:
    from pathlib import Path

    from shapely import MultiPolygon, Polygon

logger = logging.getLogger(__name__)

type Mode = Literal["IW", "EW"]
type SubSwath = Literal["IW1", "IW2", "IW3", "EW1", "EW2", "EW3", "EW4", "EW5"]
type PassDirection = Literal["Ascending", "Descending"]

S1_MODE_DESC = {
    "IW": "Interferometric Wide Swath",
    "EW": "Extra Wide Swath",
}

# Release tag, baked into cache paths so a version bump cannot reuse a stale cache.
_BURSTMAP_VERSION = "20220530"
# SAR-MPC publishes no checksum, so this pins our own known-good copy of the ZIP.
_BURSTMAP_SHA256 = "9a37efaa7e629f9f4b807fd4a95797a8a5011b9ec3a02e686cfb3682180e4426"
_COPERNICUS_TERMS_URL = (
    "https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice"
)


class Sentinel1Bursts(GeoVectorData):
    """Sentinel-1 Burst ID map.

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

    Parameters
    ----------
    cache_dir
        The root directory where the dataset is stored. Defaults to the 'FREEZEBASE_CACHE'
        environment variable or the system cache directory (in that order).
    mode
        The Sentinel-1 acquisition mode.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        mode: Mode = "IW",
    ) -> None:
        super().__init__(cache_dir)
        # Validate mode
        try:
            S1_MODE_DESC[mode]
        except KeyError as e:
            valid_values_bulleted = format_valid_options(S1_MODE_DESC)
            msg = f"Invalid mode: '{mode}'. Use one of: {valid_values_bulleted}"
            raise ValueError(msg) from e
        self.mode = mode

    @property
    @override
    def metadata(self) -> DatasetMetadata:
        """Metadata container for dataset attribution and licensing."""
        return DatasetMetadata(
            name="Sentinel-1 Burst ID Map",
            source_url=f"https://sar-mpc.eu/files/S1_burstid_{_BURSTMAP_VERSION}.zip",
            attribution="SAR Mission Performance Cluster (SAR-MPC)",
            version=_BURSTMAP_VERSION,
            sha256=_BURSTMAP_SHA256,
            license_url=_COPERNICUS_TERMS_URL,
            description=(
                "Sentinel-1 burst-ID map distributed by SAR-MPC. No explicit license is "
                "stated upstream; use is governed by the Copernicus data terms."
            ),
            additional_fields={"mode": S1_MODE_DESC[self.mode]},
        )

    @property
    @override
    def raw_path(self) -> Path:
        """Path to the raw downloaded data."""
        return self.cache_dir / f"raw/sar/s1-burstid-map-{_BURSTMAP_VERSION}.zip"

    @property
    @override
    def processed_path(self) -> Path:
        """Path to the processed data."""
        return (
            self.cache_dir
            / f"processed/sar/s1-{self.mode.lower()}-burstid-map-{_BURSTMAP_VERSION}.parquet"
        )

    @override
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
        write_parquet(bursts, self.processed_path)


def download_burst_map(cache_dir: str | Path | None = None, *, mode: Mode = "IW") -> Path:
    """Download and prepare the Sentinel-1 burst-ID map for a given acquisition mode.

    The map is a large (~500 MB per mode) static dataset from SAR-MPC. It is only
    needed to look up the relative orbit number, pass direction, or footprint
    geometry of a burst that was **not** obtained from a CDSE search or an SLC
    filename (e.g. a manually constructed burst). It is not downloaded
    automatically; call this once to opt in. The raw download is verified against
    a pinned SHA-256.

    Parameters
    ----------
    cache_dir
        Root directory for cached dataset files. Defaults to the
        ``FREEZEBASE_CACHE`` environment variable or the system cache directory.
    mode
        Sentinel-1 acquisition mode.

    Returns
    -------
    Path
        Path to the prepared GeoParquet file.
    """
    dataset = Sentinel1Bursts(cache_dir=cache_dir, mode=mode)
    dataset.get_data(download=True)
    return dataset.processed_path


def _load_indexed_burst_map(cache_dir: str | Path | None, mode: Mode) -> gpd.GeoDataFrame:
    """Load the prepared burst map indexed by ``(burst_id, subswath_name)``.

    Does not download: the map is opt-in. Raises a clear error, pointing at
    ``download_burst_map``, when the data has not been fetched yet.
    """
    try:
        df = Sentinel1Bursts(cache_dir=cache_dir, mode=mode).get_data(download=False)
    except FileNotFoundError as e:
        msg = (
            f"The Sentinel-1 {mode} burst-ID map is required to look up this burst's "
            "orbit/geometry, but it has not been downloaded. Fetch it once (~500 MB "
            f"download) with `s1bursts.download_burst_map(mode='{mode}')`."
        )
        raise FileNotFoundError(msg) from e
    return df.set_index(["burst_id", "subswath_name"])


@functools.cache
def create_s1_pass_direction_lut(
    cache_dir: str | Path | None = None,
    mode: Mode = "IW",
) -> dict[tuple[int, SubSwath], PassDirection]:
    """Build a lookup table mapping burst footprints to their pass direction.

    Loads the Sentinel-1 burst ID map and returns a dictionary keyed by
    ``(burst_id, subswath_name)`` tuples whose values are the corresponding
    pass direction (``"Ascending"`` or ``"Descending"``).

    Because the burst footprints are stationary over the lifetime of the mission,
    the pass direction is fixed per footprint and does not depend on the
    acquisition date.

    Results are cached in memory after the first call.

    Parameters
    ----------
    cache_dir
        Root directory for cached dataset files. Forwarded to
        ``Sentinel1Bursts``.
    mode
        Sentinel-1 acquisition mode. Forwarded to ``Sentinel1Bursts``.

    Returns
    -------
    dict[tuple[int, SubSwath], str]
        Mapping of ``(burst_id, subswath_name)`` → pass direction string.
    """
    df = _load_indexed_burst_map(cache_dir, mode)
    # The burst map spells the directions upper case; the package uses the
    # CEOS-ARD <PassDirection> enumeration, which is title case.
    directions = df["orbit_pass"].str.capitalize()
    return cast("dict[tuple[int, SubSwath], PassDirection]", directions.to_dict())


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
    cache_dir
        Root directory for cached dataset files. Forwarded to
        ``Sentinel1Bursts``.
    mode
        Sentinel-1 acquisition mode. Forwarded to ``Sentinel1Bursts``.

    Returns
    -------
    dict[tuple[int, SubSwath], int]
        Mapping of ``(burst_id, subswath_name)`` → relative orbit number.
    """
    df = _load_indexed_burst_map(cache_dir, mode)
    return cast("dict[tuple[int, SubSwath], int]", df["relative_orbit_number"].to_dict())


@functools.cache
def create_s1_geometry_lut(
    cache_dir: str | Path | None = None,
    mode: Mode = "IW",
) -> dict[tuple[int, SubSwath], Polygon | MultiPolygon]:
    """Build a lookup table mapping burst footprints to their geometry.

    Loads the Sentinel-1 burst ID map and returns a dictionary keyed by
    ``(burst_id, subswath_name)`` tuples whose values are the corresponding
    footprint polygons.

    Because the burst footprints are stationary over the lifetime of the mission,
    the geometry is fixed per footprint and does not depend on the acquisition date.

    Results are cached in memory after the first call.

    Parameters
    ----------
    cache_dir
        Root directory for cached dataset files. Forwarded to
        ``Sentinel1Bursts``.
    mode
        Sentinel-1 acquisition mode. Forwarded to ``Sentinel1Bursts``.

    Returns
    -------
    dict[tuple[int, SubSwath], Polygon | MultiPolygon]
        Mapping of ``(burst_id, subswath_name)`` → footprint geometry.
    """
    df = _load_indexed_burst_map(cache_dir, mode)
    return cast("dict[tuple[int, SubSwath], Polygon | MultiPolygon]", df["geometry"].to_dict())
