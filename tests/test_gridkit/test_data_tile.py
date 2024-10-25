import numpy
import pytest

from gridkit import DataTile, Tile, TriGrid


def test_data_ravel_order():
    grid = TriGrid(size=1)
    tile = Tile(grid, (0, 0), 3, 3)
    data = numpy.arange(9).reshape((3, 3))
    data_tile = DataTile(tile, data)

    value = data_tile.value(grid.cell_at_point(grid.centroid(data_tile.indices)))
    numpy.testing.assert_equal(value, data)

    value = data_tile.value(data_tile.indices)
    numpy.testing.assert_equal(value, data)


def test_data_immutability():
    grid = TriGrid(size=1)
    tile = Tile(grid, (0, 0), 3, 3)
    data = numpy.arange(9).reshape((3, 3))
    data_tile = DataTile(tile, data)
    assert id(data) != id(data_tile.to_numpy())


def test_data_setter():
    grid = TriGrid(size=1)
    tile = Tile(grid, (0, 0), 3, 3)
    data = numpy.arange(9).reshape((3, 3))
    data_tile = DataTile(tile, data)

    data_tile[:] = data + 1
    numpy.testing.assert_equal(data + 1, data_tile.to_numpy())

    with pytest.raises(TypeError) as e:
        data_tile[:] = str

    with pytest.raises(ValueError) as e:
        data_tile[:] = [1, 2]


@pytest.mark.parametrize(
    "tile_init, expected",
    [
        [((0, 0), 2, 2), True],
        [((3, 3), 3, 3), False],
        [((3, 0), 3, 2), False],
        [((0, 3), 2, 3), False],
        [((-2, -2), 3, 3), True],
    ],
)
def test_intersects(tile_init, expected):
    grid = TriGrid(size=1)
    tile = Tile(grid, (0, 0), 3, 3)
    data = numpy.arange(9).reshape((3, 3))
    data_tile = DataTile(tile, data)

    tile2 = Tile(grid, *tile_init)
    data_tile2 = DataTile(tile2, numpy.ones((tile2.ny, tile2.nx)))

    assert data_tile.intersects(data_tile2) == expected
    assert data_tile.intersects(tile2) == expected
    assert tile.intersects(data_tile2) == expected
