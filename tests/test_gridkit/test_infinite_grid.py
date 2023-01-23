from unittest import result
import numpy
import pytest
from gridkit import rect_grid


@pytest.mark.parametrize("dx, dy, offset, point, expected_id",
    [
        (5, 5, (0,0), (0,0), (0,0)),
        (5, 5, (0,0), (-6,6), (-2,1)),
        (5, 5, (3,3), (-6,6), (-2,0)),
        (10, 5, (0,0), numpy.array([[40.1, 8.3], [-5.1, -5.1]]), numpy.array([[4, 1], [-1,-2]])),
        (10, 5, (-7,9), numpy.array([[40.1, 8.3], [-5.1, -5.1]]), numpy.array([[3, 0], [-1,-2]])),
    ]
)
def test_cell_at_point(dx, dy, offset, point, expected_id):

    testgrid = rect_grid.RectGrid(dx=dx, dy=dy, offset=offset)
    result_id = testgrid.cell_at_point(point)
    numpy.testing.assert_allclose(result_id, expected_id)


@pytest.mark.parametrize("dx, dy, offset, id, expected_center",
    [
        (10, 5, (0,0), (0, 0), (5, 2.5)),
        (10, 5, (7,7), (0, 0), (12, 4.5)),
        (10, 5, (7,-7),  numpy.array([[4, 1], [-1,-2]]), numpy.array([[52, 10.5],[2, -4.5]])),
    ]
)
def test_centroid(dx, dy, offset, id, expected_center):

    testgrid = rect_grid.RectGrid(dx=dx, dy=dy, offset=offset)
    center = testgrid.centroid(id)
    
    numpy.testing.assert_allclose(center, expected_center)


@pytest.mark.parametrize("dx, dy, offset, id, expected_corners",
    [
        (5, 5, (0,0), (0,0), numpy.array([
            [0, 0],
            [5, 0],
            [5, 5],
            [0, 5],
        ])),
        (10, 5, (-7,7), numpy.array([[2,-1], [0,0], [1,-1]]), numpy.array([
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
            ]
        ])),
    ]
)
def test_cell_corners(dx, dy, offset, id, expected_corners):

    testgrid = rect_grid.RectGrid(dx=dx, dy=dy, offset=offset)
    corners = testgrid.cell_corners(id)

    numpy.testing.assert_allclose(corners, expected_corners)

def test_crs():
    dx = 10
    dy = 20
    offset = (5,10)
    crs = 3857
    grid = rect_grid.RectGrid(dx=dx, dy=dy, offset=offset, crs=crs)
    new_grid = grid.to_crs(crs=4326)

    expected_dx = 8.983152841195213e-05
    expected_dy = 2 * expected_dx
    
    assert new_grid.crs.to_epsg() == 4326
    numpy.testing.assert_allclose(new_grid.dx, expected_dx)
    numpy.testing.assert_allclose(new_grid.dy, expected_dy)
    numpy.testing.assert_allclose(new_grid.offset, (expected_dx/2, expected_dy/2))





# @pytest.mark.parametrize("method", ["linear"])
# def test_interp_from_points(method):

#     testgrid = rect_grid.RectGrid(gridsize=1)
#     def func(x, y):
#         return x*(1-x)*numpy.cos(4*numpy.pi*x) * numpy.sin(4*numpy.pi*y**2)**2

#     # make points TODO: don't make em random
#     rng = numpy.random.default_rng()
#     points = 100 * rng.random((1000, 2))
#     values = func(points[:,0], points[:,1])

#     result = testgrid.interp_from_points(points, values, method=method)
