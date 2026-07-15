"""UI utilities for creating HTML representations of Sentinel-1 data structures."""

from functools import lru_cache
from importlib.resources import files

import geopandas as gpd
from odc.geo.ui import svg_base_map
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry


def geometry_to_svg_path(geom: BaseGeometry) -> str:
    """
    Convert a Shapely geometry to an SVG path string.

    Parameters
    ----------
    geom : shapely.geometry
        Geometry to convert (Polygon or MultiPolygon).

    Returns
    -------
    str
        SVG path string.

    """
    if isinstance(geom, Polygon):
        coords = list(geom.exterior.coords)
        if len(coords) == 0:
            return ""
        path = f"M {coords[0][0]} {coords[0][1]}"
        for x, y in coords[1:]:
            path += f" L {x} {y}"
        path += " Z"
        return path
    if isinstance(geom, MultiPolygon):
        paths = [geometry_to_svg_path(poly) for poly in geom.geoms]
        return " ".join(paths)
    return ""


def create_footprint_svg(
    gdf: gpd.GeoDataFrame,
    svg_width: int = 400,
    stroke_color: str = "#66cc99",
    stroke_width: float = 1.0,
) -> str:
    """
    Create a simple SVG map of geographic footprints.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with geometry column containing footprints.
    svg_width : int, optional
        Width of SVG in pixels (default: 400).
    stroke_color : str, optional
        SVG stroke color (default: "#66cc99").
    stroke_width : float, optional
        SVG stroke width (default: 1.0).

    Returns
    -------
    str
        SVG string representation of the footprints.

    """
    if len(gdf) == 0 or "geometry" not in gdf.columns:
        return "<div class='map-placeholder'>No geometries to display</div>"

    gdf_valid = gdf[gdf.geometry.notna()].copy()
    if len(gdf_valid) == 0:
        return "<div class='map-placeholder'>No valid geometries to display</div>"
    gdf_valid = gdf_valid.drop_duplicates(subset=["geometry"])

    bounds = gdf_valid.total_bounds
    min_lon, min_lat, max_lon, max_lat = bounds
    width_deg = max_lon - min_lon
    height_deg = max_lat - min_lat

    # Calculate SVG dimensions (maintain aspect ratio)
    aspect_ratio = height_deg / width_deg if width_deg > 0 else 1
    svg_height = int(svg_width * aspect_ratio)
    svg_height = max(200, min(svg_height, 600))  # Clamp height

    # Add padding (10% on each side)
    padding = 0.1
    padded_width = width_deg * (1 + 2 * padding)
    padded_height = height_deg * (1 + 2 * padding)
    padded_min_lon = min_lon - width_deg * padding
    padded_min_lat = min_lat - height_deg * padding

    # Create SVG paths for each geometry
    paths = []
    for _, row in gdf_valid.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        # Convert geometry to SVG path
        svg_path = geometry_to_svg_path(geom)
        paths.append(svg_path)

    paths_str = " ".join(paths)

    # Scale factors: geographic coordinates to SVG pixels
    scale_x = svg_width / padded_width
    scale_y = svg_height / padded_height

    # Transform: translate to origin, scale, then flip Y-axis
    # Geographic Y (latitude) increases upward, SVG Y increases downward
    transform = (
        f"translate(0, {svg_height}) "
        f"scale({scale_x}, {-scale_y}) "
        f"translate({-padded_min_lon}, {-padded_min_lat})"
    )

    # Create SVG with transparent background and green stroke
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         width="{svg_width}" height="{svg_height}"
         viewBox="0 0 {svg_width} {svg_height}"
         style="background-color: transparent;">
        <g transform="{transform}">
            <path d="{paths_str}"
                  fill="none"
                  stroke="{stroke_color}"
                  stroke-width="{stroke_width}"
                  vector-effect="non-scaling-stroke"/>
        </g>
    </svg>
    """


def create_world_map_with_target(gdf: gpd.GeoDataFrame, size: int = 200) -> str:
    """
    Create a small world map showing the centroid of geometries.

    Calculates centroid efficiently from the bounding box of all geometries.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with geometry column.
    size : int, optional
        Size of the map in pixels (default: 200).

    Returns
    -------
    str
        SVG string of world map with target marker, or empty string if unavailable.

    """
    if len(gdf) == 0 or "geometry" not in gdf.columns:
        return ""

    gdf_valid = gdf[gdf.geometry.notna()]
    if len(gdf_valid) == 0:
        return ""
    gdf_valid = gdf_valid.drop_duplicates(subset=["geometry"])

    # Calculate centroid from bounding box
    # (Not accurate, but only used for display purposes)
    bounds = gdf_valid.total_bounds
    min_lon, min_lat, max_lon, max_lat = bounds
    centroid_lon = round((min_lon + max_lon) / 2)
    centroid_lat = round((min_lat + max_lat) / 2)
    target = (centroid_lon, centroid_lat)

    # Create world map with target marker
    return svg_base_map(sz=size, target=target)


@lru_cache
def get_burstinfo_css() -> str:
    """Load and cache CSS styles for the burst info HTML representation.

    Returns
    -------
    str
        CSS stylesheet content bundled with the package.
    """
    return files(__package__).joinpath("burstinfo.css").read_text()
