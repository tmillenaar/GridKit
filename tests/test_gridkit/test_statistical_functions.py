import numpy
import pytest

import gridkit
import gridkit.tile
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


# nocheckin, move all these to test_tile or something similar
@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1, orientation="flat"),
        TriGrid(size=1, orientation="pointy"),
        RectGrid(size=1),
        HexGrid(size=1, shape="pointy"),
        HexGrid(size=1, shape="flat"),
    ],
)
def test_combine_tiles(grid):
    grid.rotation = 12

    tile1 = Tile(grid, (-2, -3), 5, 7)
    tile2 = Tile(grid, (1, 0), 7, 5)
    dt1 = tile1.to_data_tile_with_value(-4)

    combined = gridkit.tile.combine_tiles([tile1, dt1, tile2])

    numpy.testing.assert_allclose(combined.start_id, [-2, -3])
    assert combined.nx == 10
    assert combined.ny == 8


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1, orientation="flat"),
        TriGrid(size=1, orientation="pointy"),
        RectGrid(size=1),
        HexGrid(size=1, shape="pointy"),
        HexGrid(size=1, shape="flat"),
    ],
)
def test_count_data_tile(grid):
    grid.rotation = 12

    tile1 = Tile(grid, (-2, -3), 5, 7)
    tile2 = Tile(grid, (1, 0), 7, 5).to_data_tile_with_value(3.2)
    tile3 = Tile(grid, (1, -4), 6, 10).to_data_tile_with_value(3.2)

    count = gridkit.tile.count_tiles([tile1, tile2, tile3])

    assert len(count == 0) == 14
    assert len(count == 1) == 50
    assert len(count == 2) == 28
    assert len(count == 3) == 8

    # Check nodata_value, also for tiles with different nodata values
    tile2[3, 0] = 999
    tile2[3, 3] = 999
    tile2.nodata_value = 999
    tile3[4, 0] = 997
    tile3.nodata_value = 997
    count = gridkit.tile.count_tiles([tile1, tile2, tile3])

    assert numpy.isclose(count[4, 6], 1)
    assert numpy.isclose(count[4, 3], 1)


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1, orientation="flat"),
        TriGrid(size=1, orientation="pointy"),
        RectGrid(size=1),
        HexGrid(size=1, shape="pointy"),
        HexGrid(size=1, shape="flat"),
    ],
)
def test_sum_data_tiles(grid):
    grid.rotation = 12

    tile1 = Tile(grid, (-2, -3), 5, 7).to_data_tile_with_value(-2.1)
    tile2 = Tile(grid, (1, 0), 7, 5).to_data_tile_with_value(3.2)
    tile3 = Tile(grid, (1, -4), 6, 10).to_data_tile_with_value(-1)

    summation = gridkit.tile.sum_data_tiles([tile1, tile2, tile3])

    assert len(summation == -3.1) == 6
    assert len(summation == -2.1) == 21
    assert len(summation == -1) == 24
    assert len(summation == summation.nodata_value) == 14
    assert numpy.isclose(summation[:], 0.1).sum() == 8
    assert len(summation == 2.2) == 22
    assert len(summation == 3.2) == 5

    # Check with nodata_value
    tile2[3, 3] = 999
    tile2.nodata_value = 999
    summation = gridkit.tile.sum_data_tiles([tile1, tile2, tile3])

    # Since tile2 has it's 3,3 set to a nodata value, we expect the corresponding
    # summation value to not take that number into account. This works out to be
    # at numpy id 4,6 in the combined tile and the other two tiles add to a value of -1.
    # If the nodatavalue is not properly filtered, we get a value of 998 (if the nodata value is set to 999).
    # Here we check we get the expected -1 and not the 998.
    assert numpy.isclose(summation[4, 6], -1)


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1, orientation="flat"),
        TriGrid(size=1, orientation="pointy"),
        RectGrid(size=1),
        HexGrid(size=1, shape="pointy"),
        HexGrid(size=1, shape="flat"),
    ],
)
def test_average_data_tiles(grid):
    grid.rotation = 12

    tile1 = Tile(grid, (-2, -3), 5, 7).to_data_tile_with_value(-2.1)
    tile2 = Tile(grid, (1, 0), 7, 5).to_data_tile_with_value(3.2)
    tile3 = Tile(grid, (1, -4), 6, 10).to_data_tile_with_value(-1)

    mean = gridkit.tile.average_data_tiles([tile1, tile2, tile3])

    assert len(mean == -1.55) == 6
    assert len(mean == -2.1) == 21
    assert len(mean == -1) == 24
    assert (
        ~numpy.isfinite(mean)
    ).sum() == 14  # Where count gives 0 we difide by 0 and get nan or inf
    assert numpy.isclose(mean[:], 1 / 30).sum() == 8
    assert len(mean == 1.1) == 22
    assert len(mean == 3.2) == 5

    tile2[3, 1] = 999
    tile2.nodata_value = 999

    result = gridkit.tile.average_data_tiles([tile1, tile2, tile3])

    # Since tile2 has it's 3,3 set to a nodata value, we expect the corresponding
    # summation and count values to not take that number into account. This works out to be
    # at numpy id 4,6 in the combined tile and the other two tiles add to a value of -0.5.
    assert numpy.isclose(result[4, 4], -1.55)
