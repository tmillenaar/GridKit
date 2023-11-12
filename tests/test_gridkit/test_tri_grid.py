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
    # TODO: test for different shapes when implemented
    grid = TriGrid(size=3)
    centroids = grid.cell_corners(indices)
    numpy.testing.assert_allclose(centroids, expected_centroids)
