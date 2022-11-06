from gridding.rect_grid import RectGrid
import pytest
import numpy

@pytest.mark.parametrize("points, expected_ids",
    [
        [(340, -14.2), (68, -8)], # test single point as tuple
        [[[14,3],[-8,1]], [[2,1],[-2,0]]], # test multiple points as stacked list
        [numpy.array([[14,3],[-8,1]]), numpy.array([[2,1],[-2,0]])] # test multiple points as numpy ndarray
    ]
)
def test_cell_at_point(points, expected_ids):
    # TODO: take offset into account
    ids = RectGrid(dx=5,dy=2).cell_at_point(points)
    numpy.testing.assert_allclose(ids, expected_ids)


def test_centroid():
    grid = RectGrid(dx=5,dy=2)
    indices = [(-1, 1),(1, -4)]
    expected_centroids = numpy.array([
        [-2.5, 3],
        [7.5, -7]
    ])

    centroids = grid.centroid(indices)
    numpy.testing.assert_allclose(centroids, expected_centroids)


@pytest.mark.parametrize("mode, expected_bounds",[
    ["expand", (-8, 1, -3, 6.5)],
    ["contract", (-7, 1.5, -4, 6)],
    ["nearest", (-7, 1.5, -3, 6.5)],
])
def test_align_bounds(mode, expected_bounds):
    grid = RectGrid(dx=1,dy=0.5)
    bounds = (-7.2, 1.4, -3.2, 6.4)

    new_bounds = grid.align_bounds(bounds, mode=mode)
    numpy.testing.assert_allclose(new_bounds, expected_bounds)

@pytest.mark.parametrize("mode", ("expand", "contract", "nearest"))
def test_align_bounds_with_are_bounds_aligned(mode):
    grid = RectGrid(dx=3, dy=6, offset=(1, 2))
    bounds = (-7, -5, -1, 6.4)
    new_bounds = grid.align_bounds(bounds, mode=mode)
    assert grid.are_bounds_aligned(new_bounds), "Bounds are not aligned"



def test_cells_in_bounds():
    grid = RectGrid(dx=1,dy=2)
    bounds = (-5.2, 1.4, -3.2, 2.4)

    expected_ids = numpy.array([
        [-6.,  1.],
        [-5.,  1.],
        [-4.,  1.],
        [-6.,  0.],
        [-5.,  0.],
        [-4.,  0.],
    ])

    aligned_bounds = grid.align_bounds(bounds, mode="expand")
    ids, shape = grid.cells_in_bounds(aligned_bounds)
    numpy.testing.assert_allclose(ids, expected_ids)
    assert shape == (2,3)

