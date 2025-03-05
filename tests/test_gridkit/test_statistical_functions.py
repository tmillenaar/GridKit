import numpy
import pytest

import gridkit
from gridkit import DataTile, HexGrid, RectGrid, Tile, TriGrid


@pytest.mark.parametrize("shape", ["flat", "pointy", "rect"])
def test_sum_bounded_grid(
    basic_bounded_pointy_grid, basic_bounded_flat_grid, basic_bounded_rect_grid, shape
):
    if shape == "pointy":
        grid = basic_bounded_pointy_grid
    elif shape == "flat":
        grid = basic_bounded_flat_grid
    elif shape == "rect":
        grid = basic_bounded_rect_grid
    else:
        raise ValueError(f"Unrecognized shape {shape}")

    result = gridkit.sum([grid, grid])
    numpy.testing.assert_allclose(2 * grid, result)


@pytest.mark.parametrize("shape", ["flat", "pointy", "rect"])
def test_mean(
    basic_bounded_pointy_grid, basic_bounded_flat_grid, basic_bounded_rect_grid, shape
):
    if shape == "pointy":
        grid = basic_bounded_pointy_grid
    elif shape == "flat":
        grid = basic_bounded_flat_grid
    elif shape == "rect":
        grid = basic_bounded_rect_grid
    else:
        raise ValueError(f"Unrecognized shape {shape}")

    result = gridkit.mean([grid, grid])
    numpy.testing.assert_allclose(grid, result)
