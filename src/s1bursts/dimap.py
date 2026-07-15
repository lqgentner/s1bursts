"""BEAM-DIMAP dataset reader/writer for modification of ESA SNAP's native output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import warnings
import xml.etree.ElementTree as ET

from defusedxml.ElementTree import parse
import numpy as np
import rasterio
import rasterio.errors


@dataclass
class BeamDimapDataset:
    """Wrapper around a BEAM-DIMAP dataset for band-level I/O.

    SNAP writes BEAM-DIMAP products as a ``.dim`` XML metadata file alongside
    a ``.data/`` directory containing one ``<band>.img`` / ``<band>.hdr`` pair
    per band.  This class parses the ``.dim`` file to discover the band names
    and their associated binary files, then delegates to rasterio (GDAL ENVI
    driver) for reading and writes band data directly via :func:`numpy.ndarray.tofile`
    for maximum compatibility.

    Parameters
    ----------
    dim_path : Path
        Absolute or relative path to the ``.dim`` file.

    Raises
    ------
    FileNotFoundError
        If *dim_path* does not exist.
    """

    dim_path: Path
    _img_paths: dict[str, Path] | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Validate the dataset on construction.

        Raises
        ------
        ValueError
            If ``dim_path`` does not end with ``.dim``.
        FileNotFoundError
            If the ``.dim`` file or its ``.data`` directory does not exist.
        """
        self._validate_dataset()

    def _validate_dataset(self) -> None:
        """Validate the integrety of the BEAM-DIMAP dataset."""
        if self.dim_path.suffix != ".dim":
            msg = f"`dim_path` must end with '.dim', received '{self.dim_path}'."
            raise ValueError(msg)
        if not self.dim_path.exists():
            msg = f"BEAM-DIMAP file not found: {self.dim_path}"
            raise FileNotFoundError(msg)
        # Create corresponding data folder path
        self.data_path = self.dim_path.with_suffix(".data")
        if not self.data_path.is_dir():
            msg = f"BEAM-DIMAP data folder file not found: {self.data_path}"
            raise FileNotFoundError(msg)

    @property
    def img_paths(self) -> dict[str, Path]:
        """Dictionary of the band names and their ``.img`` file paths."""
        if self._img_paths is None:
            self._img_paths = self._parse_img_paths()
        return self._img_paths

    def _parse_img_paths(self) -> dict[str, Path]:
        """Parse the .dim XML file to discover band names and ``.img`` file paths."""
        self._validate_dataset()
        root = parse(self.dim_path).getroot()
        img_paths = {}
        for band_elem in root.findall("./Image_Interpretation/Spectral_Band_Info"):
            name = band_elem.findtext("BAND_NAME")
            idx = band_elem.findtext("BAND_INDEX")

            if name is None:
                msg = f"Missing BAND_NAME in '{self.dim_path}'"
                raise RuntimeError(msg)
            if idx is None:
                msg = f"Missing BAND_INDEX for band '{name}' in '{self.dim_path}'"
                raise RuntimeError(msg)

            idx_path = f"./Data_Access/Data_File[BAND_INDEX='{idx}']/DATA_FILE_PATH"
            data_file_elem = root.find(idx_path)
            if data_file_elem is None:
                msg = f"No Data_File entry for band '{name}' (index {idx}) in '{self.dim_path}'"
                raise RuntimeError(msg)

            href = data_file_elem.get("href")
            if href is None:
                msg = f"DATA_FILE_PATH missing 'href' for band '{name}' in '{self.dim_path}'"
                raise RuntimeError(msg)

            hdr_path = self.dim_path.parent / Path(href)
            if not hdr_path.exists():
                msg = f"Header file not found for band '{name}': {hdr_path}"
                raise FileNotFoundError(msg)

            img_path = hdr_path.with_suffix(".img")
            if not img_path.exists():
                msg = f"Image file not found for band '{name}': {img_path}"
                raise FileNotFoundError(msg)

            img_paths[name] = img_path
        return img_paths

    @property
    def band_names(self) -> list[str]:
        """List of band names as declared in the .dim metadata.

        Returns
        -------
        list[str]
            Band names in the order they appear in the .dim file.
        """
        return list(self.img_paths.keys())

    @property
    def shape(self) -> tuple[int, int]:
        """Raster shape ``(nrows, ncols)`` read from the first band.

        Returns
        -------
        tuple[int, int]
            ``(nrows, ncols)`` of the raster.

        Raises
        ------
        ValueError
            If no bands are available.
        """
        if not self.img_paths:
            msg = f"No bands found in {self.dim_path}"
            raise ValueError(msg)
        first_path = next(iter(self.img_paths.values()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            with rasterio.open(first_path) as ds:
                return ds.height, ds.width

    def read_band(self, name: str) -> np.ndarray:
        """Read a single band by name and return as a float32 array.

        Parameters
        ----------
        name : str
            Band name as it appears in the .dim metadata.

        Returns
        -------
        np.ndarray
            Shape ``(nrows, ncols)``, dtype ``float32``.

        Raises
        ------
        KeyError
            If *name* is not found in the dataset.
        """
        if name not in self.img_paths:
            msg = f"Band '{name}' not found. Available: {self.band_names}"
            raise KeyError(msg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            with rasterio.open(self.img_paths[name]) as ds:
                return ds.read(1).astype(np.float32)

    def write_band(self, name: str, data: np.ndarray) -> None:
        """Overwrite the binary ``.img`` file for a named band in-place.

        The ``.hdr`` ENVI header is left unchanged; only the raw pixel data
        (float32, little-endian, BSQ layout) is rewritten.

        Parameters
        ----------
        name : str
            Band name to overwrite.
        data : np.ndarray
            Array with shape matching :attr:`shape`, dtype convertible to ``float32``.

        Raises
        ------
        KeyError
            If *name* is not found in the dataset.
        ValueError
            If *data* shape does not match the dataset shape.
        """
        if name not in self.img_paths:
            msg = f"Band '{name}' not found. Available: {self.band_names}"
            raise KeyError(msg)

        expected = self.shape
        if data.shape != expected:
            msg = f"Shape mismatch for band '{name}': expected {expected}, got {data.shape}"
            raise ValueError(msg)

        stored = self.img_paths[name]
        img_path = stored.with_suffix(".img") if stored.suffix == ".hdr" else stored

        data.astype(">f4").tofile(img_path)

    def add_band(
        self,
        name: str,
        data: np.ndarray,
        unit: str = "",
    ) -> None:
        """Add a new band to the dataset, updating both the .dim XML and .hdr/.img files.

        Creates ``<name>.hdr`` and ``<name>.img`` in the ``.data/`` directory,
        then updates the ``.dim`` file to register the new band: increments
        ``NBANDS``, appends a ``Data_File`` entry in ``Data_Access``, appends a
        ``Spectral_Band_Info`` entry in ``Image_Interpretation``, and appends a
        ``Coordinate_Reference_System`` / ``Geoposition`` block.

        Parameters
        ----------
        name : str
            Name for the new band. Must not already exist in the dataset.
        data : np.ndarray
            Array with shape matching :attr:`shape`, dtype convertible to ``float32``.
        unit : str, default ""
            Physical unit string written into the ``.hdr`` description and the
            ``PHYSICAL_UNIT`` element of the ``.dim`` file (e.g. ``"real"``,
            ``"imaginary"``, ``"intensity"``).

        Raises
        ------
        ValueError
            If *name* already exists in the dataset, or if *data* shape does not
            match the dataset shape.
        """
        if name in self.img_paths:
            msg = f"Band '{name}' already exists. Use write_band() to overwrite."
            raise ValueError(msg)

        nrows, ncols = self.shape
        if data.shape != (nrows, ncols):
            msg = (
                f"Shape mismatch for new band '{name}': "
                f"expected ({nrows}, {ncols}), got {data.shape}"
            )
            raise ValueError(msg)

        data_dir = self.dim_path.with_suffix(".data")
        img_path = data_dir / f"{name}.img"
        hdr_path = data_dir / f"{name}.hdr"

        # 1. Write binary .img and ENVI .hdr (big-endian, matching SNAP's byte order = 1)
        data.astype(">f4").tofile(img_path)
        _write_envi_hdr(hdr_path, name=name, nrows=nrows, ncols=ncols, unit=unit)

        # 2. Update the .dim XML
        _update_dim_xml(
            dim_path=self.dim_path,
            name=name,
            nrows=nrows,
            ncols=ncols,
            unit=unit,
            hdr_path=hdr_path,
        )

        # 3. Register the new band in the in-memory mapping
        if self._img_paths is not None:
            self._img_paths[name] = img_path

    def delete_band(self, name: str) -> None:
        """Remove a band from the dataset, updating both the .dim XML and deleting .hdr/.img.

        Removes the ``<name>.hdr`` and ``<name>.img`` files from the ``.data/``
        directory, then updates the ``.dim`` file: decrements ``NBANDS``,
        removes the corresponding ``Data_File``, ``Spectral_Band_Info``, and
        one ``Coordinate_Reference_System`` / ``Geoposition`` pair, and
        renumbers the remaining band indices to keep them contiguous.

        Parameters
        ----------
        name : str
            Band name to remove. Must exist in the dataset.

        Raises
        ------
        KeyError
            If *name* is not found in the dataset.
        """
        if name not in self.img_paths:
            msg = f"Band '{name}' not found. Available: {self.band_names}"
            raise KeyError(msg)

        # 1. Delete binary files (.img and the accompanying .hdr)
        stored = self.img_paths[name]
        img_path = stored.with_suffix(".img") if stored.suffix == ".hdr" else stored
        img_path.unlink(missing_ok=True)
        img_path.with_suffix(".hdr").unlink(missing_ok=True)

        # 2. Update the .dim XML
        _remove_band_from_dim_xml(self.dim_path, name)

        # 3. Remove from in-memory mapping
        if self._img_paths is not None:
            del self._img_paths[name]

    def cleanup(self) -> None:
        """Delete the ``.dim`` file and its associated ``.data`` directory.

        Safe to call even if the files do not exist.
        """
        self.dim_path.unlink(missing_ok=True)
        if (data_dir := self.dim_path.with_suffix(".data")).exists():
            shutil.rmtree(data_dir)


def _write_envi_hdr(
    hdr_path: Path,
    *,
    name: str,
    nrows: int,
    ncols: int,
    unit: str,
) -> None:
    """Write an ENVI ``.hdr`` file for a single float32 BSQ band."""
    unit_str = f" - Unit: {unit}" if unit else ""
    content = (
        "ENVI\n"
        f"description = {{Sentinel-1 IW Level-1 SLC Product{unit_str}}}\n"
        f"samples = {ncols}\n"
        f"lines = {nrows}\n"
        "bands = 1\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        "data type = 4\n"
        "interleave = bsq\n"
        "byte order = 1\n"
        f"band names = {{ {name} }}\n"
        "data gain values = {1.0}\n"
        "data offset values = {0.0}\n"
    )
    hdr_path.write_text(content, encoding="ascii")


def _update_dim_xml(
    dim_path: Path,
    *,
    name: str,
    nrows: int,
    ncols: int,
    unit: str,
    hdr_path: Path,
) -> None:
    """Register a new band in the BEAM-DIMAP ``.dim`` XML file in-place."""
    tree = ET.parse(dim_path)  # noqa: S314
    root = tree.getroot()
    dataset_name = dim_path.stem

    # Increment NBANDS and get the new 0-based band index
    nbands_elem = root.find(".//Raster_Dimensions/NBANDS")
    if nbands_elem is None or nbands_elem.text is None:
        msg = "Could not find NBANDS element in .dim file"
        raise RuntimeError(msg)
    new_index = int(nbands_elem.text)
    nbands_elem.text = str(new_index + 1)

    # Append CRS block after the last Geoposition (copy structure from first existing one)
    first_crs = root.find("Coordinate_Reference_System")
    if first_crs is not None:
        _insert_after_last(root, "Geoposition", _copy_element(first_crs))

    # Append Geoposition block after the CRS we just inserted
    geo_elem = ET.Element("Geoposition")
    ET.SubElement(geo_elem, "BAND_INDEX").text = str(new_index)
    gp_points = ET.SubElement(geo_elem, "Geoposition_Points")
    ET.SubElement(gp_points, "TIE_POINT_GRID_NAME_LAT").text = "latitude"
    ET.SubElement(gp_points, "TIE_POINT_GRID_NAME_LON").text = "longitude"
    _insert_after_last(root, "Coordinate_Reference_System", geo_elem)

    # Append Data_File entry in Data_Access
    data_access = root.find("Data_Access")
    if data_access is not None:
        data_file_elem = ET.SubElement(data_access, "Data_File")
        path_elem = ET.SubElement(data_file_elem, "DATA_FILE_PATH")
        path_elem.set("href", f"{dataset_name}.data/{hdr_path.name}")
        ET.SubElement(data_file_elem, "BAND_INDEX").text = str(new_index)

    # Append Spectral_Band_Info entry in Image_Interpretation
    img_interp = root.find("Image_Interpretation")
    if img_interp is not None:
        _append_spectral_band_info(img_interp, new_index, name, nrows, ncols, unit)

    ET.indent(tree, space="    ")
    tree.write(dim_path, encoding="ISO-8859-1", xml_declaration=True)


def _append_spectral_band_info(
    img_interp: ET.Element,
    band_index: int,
    name: str,
    nrows: int,
    ncols: int,
    unit: str,
) -> None:
    """Append a ``Spectral_Band_Info`` child to *img_interp*."""
    sbi = ET.SubElement(img_interp, "Spectral_Band_Info")
    ET.SubElement(sbi, "BAND_INDEX").text = str(band_index)
    ET.SubElement(sbi, "BAND_DESCRIPTION")
    ET.SubElement(sbi, "BAND_NAME").text = name
    ET.SubElement(sbi, "BAND_RASTER_WIDTH").text = str(ncols)
    ET.SubElement(sbi, "BAND_RASTER_HEIGHT").text = str(nrows)
    ET.SubElement(sbi, "DATA_TYPE").text = "float32"
    ET.SubElement(sbi, "PHYSICAL_UNIT").text = unit
    ET.SubElement(sbi, "SOLAR_FLUX").text = "0.0"
    ET.SubElement(sbi, "BAND_WAVELEN").text = "0.0"
    ET.SubElement(sbi, "BAND_ANGULAR_VALUE").text = "-999.0"
    ET.SubElement(sbi, "BANDWIDTH").text = "0.0"
    ET.SubElement(sbi, "SCALING_FACTOR").text = "1.0"
    ET.SubElement(sbi, "SCALING_OFFSET").text = "0.0"
    ET.SubElement(sbi, "LOG10_SCALED").text = "false"
    ET.SubElement(sbi, "NO_DATA_VALUE_USED").text = "true"
    ET.SubElement(sbi, "NO_DATA_VALUE").text = "NaN"


def _remove_band_from_dim_xml(dim_path: Path, name: str) -> None:
    """Remove a band's XML entries from a BEAM-DIMAP ``.dim`` file in-place.

    Removes the ``Spectral_Band_Info``, ``Data_File``, and one
    ``Coordinate_Reference_System`` / ``Geoposition`` pair for the named band,
    decrements ``NBANDS``, and renumbers remaining band indices so they remain
    contiguous starting from 0.
    """
    tree = ET.parse(dim_path)  # noqa: S314
    root = tree.getroot()

    removed_index = _remove_spectral_band_info(root, name)
    _remove_data_file(root, removed_index)
    _remove_crs_and_geoposition(root, removed_index)

    nbands_elem = root.find(".//Raster_Dimensions/NBANDS")
    if nbands_elem is not None and nbands_elem.text is not None:
        nbands_elem.text = str(int(nbands_elem.text) - 1)

    if removed_index is not None:
        _renumber_band_indices(root, removed_index)

    ET.indent(tree, space="    ")
    tree.write(dim_path, encoding="ISO-8859-1", xml_declaration=True)


def _remove_spectral_band_info(root: ET.Element, name: str) -> int | None:
    """Remove the ``Spectral_Band_Info`` for *name* and return its band index."""
    img_interp = root.find("Image_Interpretation")
    if img_interp is None:
        return None
    for sbi in img_interp.findall("Spectral_Band_Info"):
        if sbi.findtext("BAND_NAME") == name:
            idx_text = sbi.findtext("BAND_INDEX")
            img_interp.remove(sbi)
            return int(idx_text) if idx_text is not None else None
    return None


def _remove_data_file(root: ET.Element, removed_index: int | None) -> None:
    """Remove the ``Data_File`` entry matching *removed_index* from ``Data_Access``."""
    if removed_index is None:
        return
    data_access = root.find("Data_Access")
    if data_access is None:
        return
    for df in data_access.findall("Data_File"):
        if df.findtext("BAND_INDEX") == str(removed_index):
            data_access.remove(df)
            return


def _remove_crs_and_geoposition(root: ET.Element, removed_index: int | None) -> None:
    """Remove the ``Geoposition`` for *removed_index* and its preceding CRS block."""
    if removed_index is None:
        return
    children = list(root)
    for i, child in enumerate(children):
        if child.tag == "Geoposition" and child.findtext("BAND_INDEX") == str(removed_index):
            root.remove(child)
            if i > 0 and children[i - 1].tag == "Coordinate_Reference_System":
                root.remove(children[i - 1])
            return


def _renumber_band_indices(root: ET.Element, removed_index: int) -> None:
    """Decrement all band indices greater than *removed_index* by one."""
    img_interp = root.find("Image_Interpretation")
    if img_interp is not None:
        for sbi in img_interp.findall("Spectral_Band_Info"):
            _decrement_index_if_greater(sbi.find("BAND_INDEX"), removed_index)

    data_access = root.find("Data_Access")
    if data_access is not None:
        for df in data_access.findall("Data_File"):
            _decrement_index_if_greater(df.find("BAND_INDEX"), removed_index)

    for geo in root.findall("Geoposition"):
        _decrement_index_if_greater(geo.find("BAND_INDEX"), removed_index)


def _decrement_index_if_greater(elem: ET.Element | None, threshold: int) -> None:
    """Subtract 1 from *elem*'s text if it represents an integer greater than *threshold*."""
    if elem is not None and elem.text is not None:
        idx = int(elem.text)
        if idx > threshold:
            elem.text = str(idx - 1)


def _copy_element(elem: ET.Element) -> ET.Element:
    """Return a deep copy of an :class:`xml.etree.ElementTree.Element`."""
    new = ET.Element(elem.tag, elem.attrib)
    new.text = elem.text
    new.tail = elem.tail
    for child in elem:
        new.append(_copy_element(child))
    return new


def _insert_after_last(parent: ET.Element, tag: str, new_elem: ET.Element) -> None:
    """Insert *new_elem* into *parent* immediately after the last child with *tag*.

    Falls back to appending at the end if no matching child is found.
    """
    last_idx = -1
    for i, child in enumerate(parent):
        if child.tag == tag:
            last_idx = i
    if last_idx >= 0:
        parent.insert(last_idx + 1, new_elem)
    else:
        parent.append(new_elem)
