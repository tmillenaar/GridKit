import numpy
import pytest

from gridkit import TriGrid


@pytest.mark.parametrize(
    "shape, indices, expected_centroids",
    [
        ["pointy", (-1, -1), [-2.25, -3.46410162]],
        ["pointy", (-1, 1), [-2.25, 1.73205081]],
        [
            "pointy",
            [(0, 0), (1, -1), (1, 1)],
            [[-0.75, -0.8660254], [0.75, -3.46410162], [0.75, 1.73205081]],
        ],
    ],
)
def test_centroid(shape, indices, expected_centroids):
    # TODO: test for different shapes when implemented
    grid = TriGrid(size=3)
    centroids = grid.centroid(indices)
    numpy.testing.assert_allclose(centroids, expected_centroids)


@pytest.mark.parametrize(
    "indices, expected_centroids",
    [
        [(-1, -1), [[-2.25, -5.19615242], [-0.75, -2.59807621], [-3.75, -2.59807621]]],
        [
            [(0, 0), (1, -1), (1, 1)],
            [
                [[-0.75, -2.59807621], [0.75, 0.0], [-2.25, 0.0]],
                [[0.75, -5.19615242], [2.25, -2.59807621], [-0.75, -2.59807621]],
                [[0.75, 0.0], [2.25, 2.59807621], [-0.75, 2.59807621]],
            ],
        ],
    ],
)
def test_cell_corners(indices, expected_centroids):
    grid = TriGrid(size=3)
    centroids = grid.cell_corners(indices)
    numpy.testing.assert_allclose(centroids, expected_centroids)


@pytest.mark.parametrize(
    "points, expected_ids",
    (
        [
            (-2.2, 5.7),
            [-3, 5],
        ],
        [
            [(-0.3, -7.5), (3.6, -8.3)],
            [[0, -6], [5, -6]],
        ],
    ),
)
def test_cell_at_point(points, expected_ids):
    grid = TriGrid(size=1.4)
    ids = grid.cell_at_point(points)
    numpy.testing.assert_allclose(ids, expected_ids)


@pytest.mark.parametrize(
    "bounds, expected_ids",
    (
        [
            (-2.2, -3, 2.3, 2.1),
            [
                [-2, 2],
                [-1, 2],
                [0, 2],
                [1, 2],
                [2, 2],
                [3, 2],
                [-2, 1],
                [-1, 1],
                [0, 1],
                [1, 1],
                [2, 1],
                [3, 1],
                [-2, 0],
                [-1, 0],
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [-2, -1],
                [-1, -1],
                [0, -1],
                [1, -1],
                [2, -1],
                [3, -1],
            ],
        ],
        [
            (2.2, 3, 5.3, 6.1),
            [
                [4, 5],
                [5, 5],
                [6, 5],
                [7, 5],
                [8, 5],
                [4, 4],
                [5, 4],
                [6, 4],
                [7, 4],
                [8, 4],
                [4, 3],
                [5, 3],
                [6, 3],
                [7, 3],
                [8, 3],
            ],
        ],
        [
            (-5.3, -6.1, -2.2, -3),
            [
                [-7, -2],
                [-6, -2],
                [-5, -2],
                [-4, -2],
                [-3, -2],
                [-7, -3],
                [-6, -3],
                [-5, -3],
                [-4, -3],
                [-3, -3],
                [-7, -4],
                [-6, -4],
                [-5, -4],
                [-4, -4],
                [-3, -4],
            ],
        ],
    ),
)
def test_cells_in_bounds(bounds, expected_ids):
    grid = TriGrid(size=1.4)
    bounds = grid.align_bounds(bounds, "nearest")
    ids = grid.cells_in_bounds(bounds)
    numpy.testing.assert_allclose(ids, expected_ids)
