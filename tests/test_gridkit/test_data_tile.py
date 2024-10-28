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


@pytest.mark.parametrize(
    "start_id, expected_data",
    (
        [
            [
                ((2, 0)),
                numpy.array(
                    [
                        [0, 1, 2, 1, 2],
                        [3, 4, 8, 4, 5],
                        [6, 7, 14, 7, 8],
                    ]
                ),
            ],
            [
                ((2, 1)),
                numpy.array(
                    [
                        [numpy.nan, numpy.nan, 0, 1, 2],
                        [0, 1, 5, 4, 5],
                        [3, 4, 11, 7, 8],
                        [6, 7, 8, numpy.nan, numpy.nan],
                    ]
                ),
            ],
        ]
    ),
)
def test_add_partial_overlap(start_id, expected_data):
    grid = TriGrid(size=1)
    data = numpy.arange(9).reshape((3, 3))
    tile1 = Tile(grid, (0, 0), 3, 3)
    tile2 = Tile(grid, start_id, 3, 3)
    data_tile1 = DataTile(tile1, data)

    # test with nodata_value of NaN
    data_tile2 = DataTile(tile2, data, nodata_value=numpy.nan)
    result = data_tile1 + data_tile2
    numpy.testing.assert_allclose(result.to_numpy(), expected_data)

    # test with nodata_value of 1
    data_tile3 = DataTile(tile2, data, nodata_value=-1)
    expected_data[numpy.isnan(expected_data)] = -1
    result = data_tile3 + data_tile1  # Note that the nodatavalue of left is used
    numpy.testing.assert_allclose(result.to_numpy(), expected_data)
