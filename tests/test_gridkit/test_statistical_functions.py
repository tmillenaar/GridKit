import numpy

import gridkit


def test_sum(basic_bounded_pointy_grid):
    grid = basic_bounded_pointy_grid
    result = gridkit.sum([grid, grid])
    numpy.testing.assert_allclose(2 * grid, result)


def test_mean(basic_bounded_pointy_grid):
    grid = basic_bounded_pointy_grid
    result = gridkit.mean([grid, grid])
    numpy.testing.assert_allclose(grid, result)
