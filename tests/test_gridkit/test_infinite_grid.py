from unittest import result

import numpy
import pytest
import shapely
from geopandas import GeoSeries

from gridkit import rect_grid


@pytest.mark.parametrize(
    "dx, dy, offset, point, expected_id",
    [
        (5, 5, (0, 0), (0, 0), (0, 0)),
        (5, 5, (0, 0), (-6, 6), (-2, 1)),
        (5, 5, (3, 3), (-6, 6), (-2, 0)),
        (
            10,
            5,
            (0, 0),
            numpy.array([[40.1, 8.3], [-5.1, -5.1]]),
            numpy.array([[4, 1], [-1, -2]]),
        ),
        (
            10,
            5,
            (-7, 9),
            numpy.array([[40.1, 8.3], [-5.1, -5.1]]),
            numpy.array([[3, 0], [-1, -2]]),
        ),
    ],
)
def test_cell_at_point(dx, dy, offset, point, expected_id):
    testgrid = rect_grid.RectGrid(dx=dx, dy=dy, offset=offset)
    result_id = testgrid.cell_at_point(point)
    numpy.testing.assert_allclose(result_id, expected_id)


@pytest.mark.parametrize(
    "dx, dy, offset, id, expected_center",
    [
        (10, 5, (0, 0), (0, 0), (5, 2.5)),
        (10, 5, (7, 7), (0, 0), (12, 4.5)),
        (
            10,
            5,
            (7, -7),
            numpy.array([[4, 1], [-1, -2]]),
            numpy.array([[52, 10.5], [2, -4.5]]),
        ),
    ],
)
def test_centroid(dx, dy, offset, id, expected_center):
    testgrid = rect_grid.RectGrid(dx=dx, dy=dy, offset=offset)
    center = testgrid.centroid(id)

    numpy.testing.assert_allclose(center, expected_center)


@pytest.mark.parametrize(
    "dx, dy, offset, id, expected_corners",
    [
        (
            5,
            5,
            (0, 0),
            (0, 0),
            numpy.array(
                [
                    [0, 0],
                    [5, 0],
                    [5, 5],
                    [0, 5],
                ]
            ),
        ),
        (
            10,
            5,
            (-7, 7),
            numpy.array([[2, -1], [0, 0], [1, -1]]),
            numpy.array(
                [
                    [
                        [23, -3],
                        [33, -3],
                        [33, 2],
                        [23, 2],
                    ],
                    [
                        [3, 2],
                        [13, 2],
                        [13, 7],
                        [3, 7],
                    ],
                    [
                        [13, -3],
                        [23, -3],
                        [23, 2],
                        [13, 2],
                    ],
                ]
            ),
        ),
    ],
)
def test_cell_corners(dx, dy, offset, id, expected_corners):
    testgrid = rect_grid.RectGrid(dx=dx, dy=dy, offset=offset)
    corners = testgrid.cell_corners(id)

    numpy.testing.assert_allclose(corners, expected_corners)


def test_crs():
    dx = 10
    dy = 20
    offset = (5, 10)
    crs = 3857
    grid = rect_grid.RectGrid(dx=dx, dy=dy, offset=offset, crs=crs)
    new_grid = grid.to_crs(crs=4326)

    expected_dx = 8.983152841195213e-05
    expected_dy = 2 * expected_dx

    assert new_grid.crs.to_epsg() == 4326
    numpy.testing.assert_allclose(new_grid.dx, expected_dx)
    numpy.testing.assert_allclose(new_grid.dy, expected_dy)
    numpy.testing.assert_allclose(new_grid.offset, (expected_dx / 2, expected_dy / 2))


@pytest.mark.parametrize(
    "shapely_geoms, expected_cell_ids",
    [
        (
            shapely.geometry.Point(0.5, 1.5),
            [0, 1],
        ),  # point in cell
        (
            shapely.geometry.Point(1, 1.5),
            [[0, 1], [1, 1]],
        ),  # point on edge
        (
            shapely.geometry.Point(1, 0),
            [[0, -1], [0, 0], [1, -1], [1, 0]],
        ),  # point on vertex
        (
            shapely.geometry.LineString([[1, -1], [1, 0.5]]),
            [[0, -2], [0, -1], [0, 0], [1, -2], [1, -1], [1, 0]],
        ),  # line on edge
        (
            shapely.geometry.LineString([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5]]),
            [[0, 0], [1, 0], [1, 1]],
        ),  # L shaped line covering three cells
        (
            shapely.geometry.Point(0.5, 1.5).buffer(0.1),
            [0, 1],
        ),  # Polygon in single cell
        (
            shapely.geometry.LineString([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5]]).buffer(
                0.6
            ),
            [  # L shaped polygon covering ten cells
                [-1, 0],
                [0, -1],
                [0, 0],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
                [1, 2],
                [2, 0],
                [2, 1],
            ],
        ),
        (  # test multiple geometries. The same cell should only be mentioned once
            [
                shapely.geometry.LineString([[0.5, 0.5], [1.5, 0.5]]),
                shapely.geometry.MultiPoint(
                    [
                        shapely.geometry.Point(0.5, 0.4),  # point in same cell as line
                        shapely.geometry.Point(
                            0.5, 2.5
                        ),  # point in different cell as line
                    ]
                ),
            ],
            [[0, 0], [0, 2], [1, 0]],
        ),
    ],
)
def test_intersect_geometries(
    basic_bounded_rect_grid, shapely_geoms, expected_cell_ids
):
    # test shapely geometries
    cell_ids = basic_bounded_rect_grid.intersect_geometries(shapely_geoms)
    numpy.testing.assert_allclose(cell_ids, expected_cell_ids)

    # test geopandas geoseries
    gpd_geoms = GeoSeries(shapely_geoms)
    cell_ids = basic_bounded_rect_grid.intersect_geometries(gpd_geoms)
    numpy.testing.assert_allclose(cell_ids, expected_cell_ids)
