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
