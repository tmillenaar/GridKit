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
@pytest.mark.parametrize("offset", ((0, 0), (-0.7, 0.3), (1, -0.2)))
def test_centroid(shape, indices, offset, expected_centroids):
    # TODO: test for different shapes when implemented
    grid = TriGrid(size=3, offset=offset)
    centroids = grid.centroid(indices)
    centroids -= offset
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
@pytest.mark.parametrize("offset", ((0, 0), (-0.7, 0.3), (1, -0.2)))
def test_cell_corners(indices, offset, expected_centroids):
    grid = TriGrid(size=3, offset=offset)
    corners = grid.cell_corners(indices)
    corners -= offset
    numpy.testing.assert_allclose(corners, expected_centroids, atol=1e-15)


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
@pytest.mark.parametrize("offset", ((0, 0), (-0.7, 0.3), (1, -0.2)))
def test_cell_at_point(points, offset, expected_ids):
    points = numpy.array(points)
    points += offset
    grid = TriGrid(size=1.4, offset=offset)
    ids = grid.cell_at_point(points)
    numpy.testing.assert_allclose(ids, expected_ids)


@pytest.mark.parametrize(
    "bounds, expected_ids, expected_shape",
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
            (6, 4),
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
            (5, 3),
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
            (5, 3),
        ],
    ),
)
@pytest.mark.parametrize("return_cell_count", [True, False])
def test_cells_in_bounds(bounds, expected_ids, expected_shape, return_cell_count):
    grid = TriGrid(size=1.4)
    bounds = grid.align_bounds(bounds, "nearest")
    result = grid.cells_in_bounds(bounds, return_cell_count=return_cell_count)
    if return_cell_count is False:
        numpy.testing.assert_allclose(result, expected_ids)
    else:
        ids, shape = result
        numpy.testing.assert_allclose(ids, expected_ids)
        numpy.testing.assert_allclose(shape, expected_shape)
        assert len(ids) == shape[0] * shape[1]
