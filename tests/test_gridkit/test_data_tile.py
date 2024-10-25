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
