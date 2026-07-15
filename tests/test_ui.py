"""Tests for s1bursts.ui HTML/SVG helpers."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import MultiPolygon, Point, box

from s1bursts import ui


class TestGeometryToSvgPath:
    def test_polygon(self) -> None:
        path = ui.geometry_to_svg_path(box(0, 0, 1, 1))

        assert path.startswith("M ")
        assert path.endswith(" Z")
        assert path.count("L") == 4

    def test_multipolygon(self) -> None:
        multi = MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)])

        path = ui.geometry_to_svg_path(multi)

        assert path.count("M ") == 2
        assert path.count(" Z") == 2

    def test_unsupported_geometry_returns_empty(self) -> None:
        assert ui.geometry_to_svg_path(Point(0, 0)) == ""


@pytest.fixture
def footprints() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"burst_id": [1, 2]},
        geometry=[box(7, 46, 8, 47), box(8, 46, 9, 47)],
        crs="EPSG:4326",
    )


class TestCreateFootprintSvg:
    def test_valid_geometries(self, footprints: gpd.GeoDataFrame) -> None:
        svg = ui.create_footprint_svg(footprints)

        assert "<svg" in svg
        assert "path" in svg

    def test_empty_dataframe_returns_placeholder(self) -> None:
        svg = ui.create_footprint_svg(gpd.GeoDataFrame({"geometry": []}))

        assert "No geometries to display" in svg

    def test_all_nan_geometries_returns_placeholder(self) -> None:
        gdf = gpd.GeoDataFrame({"burst_id": [1]}, geometry=[None])

        svg = ui.create_footprint_svg(gdf)

        assert "No valid geometries to display" in svg


class TestCreateWorldMapWithTarget:
    def test_valid_geometries(self, footprints: gpd.GeoDataFrame) -> None:
        svg = ui.create_world_map_with_target(footprints)

        assert "<svg" in svg

    def test_empty_dataframe_returns_empty_string(self) -> None:
        assert ui.create_world_map_with_target(gpd.GeoDataFrame({"geometry": []})) == ""


def test_get_burstinfo_css_loads_package_data() -> None:
    css = ui.get_burstinfo_css()

    assert ".burst-info" in css
