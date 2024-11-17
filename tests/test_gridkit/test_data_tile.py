import numpy
import pytest

from gridkit import DataTile, HexGrid, RectGrid, Tile, TriGrid


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1),
        RectGrid(size=1),
        HexGrid(size=1),
    ],
)
def test_data_ravel_order(grid):
    tile = Tile(grid, (0, 0), 3, 3)
    data = numpy.arange(9).reshape((3, 3))
    data_tile = DataTile(tile, data)

    value = data_tile.value(grid.cell_at_point(grid.centroid(data_tile.indices)))
    numpy.testing.assert_equal(value, data)

    value = data_tile.value(data_tile.indices)
    numpy.testing.assert_equal(value, data)


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1),
        RectGrid(size=1),
        HexGrid(size=1),
    ],
)
def test_data_immutability(grid):
    tile = Tile(grid, (0, 0), 3, 3)
    data = numpy.arange(9).reshape((3, 3))
    data_tile = DataTile(tile, data)
    assert id(data) != id(data_tile.to_numpy())


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1),
        RectGrid(size=1),
        HexGrid(size=1),
    ],
)
def test_data_setter(grid):
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
    "grid",
    [
        TriGrid(size=1),
        RectGrid(size=1),
        HexGrid(size=1),
    ],
)
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
def test_intersects(grid, tile_init, expected):
    tile = Tile(grid, (0, 0), 3, 3)
    data = numpy.arange(9).reshape((3, 3))
    data_tile = DataTile(tile, data)

    tile2 = Tile(grid, *tile_init)
    data_tile2 = DataTile(tile2, numpy.ones((tile2.ny, tile2.nx)))

    assert data_tile.intersects(data_tile2) == expected
    assert data_tile.intersects(data_tile2._tile) == expected
    assert data_tile.intersects(tile2) == expected
    assert data_tile.intersects(tile2._tile) == expected
    assert tile.intersects(data_tile2) == expected

    with pytest.raises(TypeError):
        # Check if appropriate error is raised when non-tile is supplied
        tile.intersects(tile.grid)


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1),
        RectGrid(size=1),
        HexGrid(size=1),
    ],
)
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
def test_add_partial_overlap(grid, start_id, expected_data):
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
    # Copy before mutating in-place to prevent it carrying over to test with next grid type
    expected_data = expected_data.copy()
    expected_data[numpy.isnan(expected_data)] = -1
    result = data_tile3 + data_tile1  # Note that the nodatavalue of left is used
    numpy.testing.assert_allclose(result.to_numpy(), expected_data)


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1),
        RectGrid(size=1),
        HexGrid(size=1),
    ],
)
def test_basic_operations_with_scalars(grid):
    tile = Tile(grid, (0, 0), 2, 2)
    data = numpy.arange(4).reshape((2, 2))
    data_tile = DataTile(tile, data)
    data_tile_nodata = data_tile.update(nodata_value=2)

    # test addition
    result = 2 + data_tile
    expected = numpy.array([[2, 3], [4, 5]])
    numpy.testing.assert_allclose(result, expected)
    result = data_tile + 2
    numpy.testing.assert_allclose(result, expected)

    result = 2 + data_tile_nodata
    expected = numpy.array([[2, 3], [2, 5]])
    numpy.testing.assert_allclose(result, expected)
    result = data_tile_nodata + 2
    numpy.testing.assert_allclose(result, expected)

    # test subtraction
    result = 2 - data_tile
    expected = numpy.array([[2, 1], [0, -1]])
    numpy.testing.assert_allclose(result, expected)
    result = data_tile - 2
    expected = numpy.array([[-2, -1], [0, 1]])
    numpy.testing.assert_allclose(result, expected)

    result = 2 - data_tile_nodata
    expected = numpy.array([[2, 1], [2, -1]])
    numpy.testing.assert_allclose(result, expected)
    result = data_tile_nodata - 2
    expected = numpy.array([[-2, -1], [2, 1]])
    numpy.testing.assert_allclose(result, expected)

    # test multiplication
    result = 2 * data_tile
    expected = numpy.array([[0, 2], [4, 6]])
    numpy.testing.assert_allclose(result, expected)
    result = data_tile * 2
    numpy.testing.assert_allclose(result, expected)

    result = 2 * data_tile_nodata
    expected = numpy.array([[0, 2], [2, 6]])
    numpy.testing.assert_allclose(result, expected)
    result = data_tile_nodata * 2
    numpy.testing.assert_allclose(result, expected)

    # test division
    result = 2 / data_tile
    expected = numpy.array([[numpy.inf, 2], [1, 2 / 3]])
    numpy.testing.assert_allclose(result, expected)
    result = data_tile / 2
    expected = numpy.array([[0, 1 / 2], [1, 3 / 2]])
    numpy.testing.assert_allclose(result, expected)

    result = 2 / data_tile_nodata
    expected = numpy.array([[numpy.inf, 2], [2, 2 / 3]])
    numpy.testing.assert_allclose(result, expected)
    result = data_tile_nodata / 2
    expected = numpy.array([[0, 1 / 2], [2, 3 / 2]])
    numpy.testing.assert_allclose(result, expected)

    # test power
    result = 2**data_tile
    expected = numpy.array([[1, 2], [4, 8]])
    numpy.testing.assert_allclose(result, expected)
    result = data_tile**2
    expected = numpy.array([[0, 1], [4, 9]])
    numpy.testing.assert_allclose(result, expected)

    result = 2**data_tile_nodata
    expected = numpy.array([[1, 2], [2, 8]])
    numpy.testing.assert_allclose(result, expected)
    result = data_tile_nodata**2
    expected = numpy.array([[0, 1], [2, 9]])
    numpy.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1),
        RectGrid(size=1),
        HexGrid(size=1),
    ],
)
@pytest.mark.parametrize("nodata", [None, 1])
def test_comparissons_with_scalars(grid, nodata):
    # initialize grid
    tile = Tile(grid, (0, 0), 2, 2)
    data = numpy.arange(4).reshape((2, 2))
    data_tile = DataTile(tile, data, nodata_value=nodata)

    # test equal
    result = data_tile == 1
    expected = numpy.array([1, 1])
    numpy.testing.assert_allclose(result.index, expected)

    # test not equal
    result = data_tile != 1
    expected = numpy.array([[0, 1], [0, 0], [1, 0]])
    numpy.testing.assert_allclose(result.index, expected)

    # test greater than
    result = data_tile > 1
    expected = numpy.array([[0, 0], [1, 0]])
    numpy.testing.assert_allclose(result.index, expected)

    # test smaller than
    result = data_tile < 1
    expected = numpy.array([0, 1])
    numpy.testing.assert_allclose(result.index, expected)

    # test greater than or equal
    result = data_tile >= 1
    expected = numpy.array([[1, 1], [0, 0], [1, 0]])
    numpy.testing.assert_allclose(result.index, expected)

    # test smaller than or equal
    result = data_tile <= 1
    expected = numpy.array([[0, 1], [1, 1]])
    numpy.testing.assert_allclose(result.index, expected)


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1),
        RectGrid(size=1),
        HexGrid(size=1),
    ],
)
def test_reduction_operators(grid):
    # initialize grid
    tile = Tile(grid, (0, 0), 2, 2)
    data = numpy.arange(4).reshape((2, 2))
    data_tile = DataTile(tile, data)
    data_tile_nodata = data_tile.update(nodata_value=3)

    # test mean
    result = data_tile.mean()
    numpy.testing.assert_allclose(result, 1.5)
    result = data_tile_nodata.mean()
    numpy.testing.assert_allclose(result, 1)

    # test median
    result = data_tile.median()
    numpy.testing.assert_allclose(result, 1.5)
    result = data_tile_nodata.median()
    numpy.testing.assert_allclose(result, 1)

    # test std
    result = data_tile.std()
    numpy.testing.assert_allclose(result, 5**0.5 / 2)
    result = data_tile_nodata.std()
    numpy.testing.assert_allclose(result, (2 / 3) ** 0.5)

    # test min
    result = data_tile.min()
    numpy.testing.assert_allclose(result, 0)
    result = data_tile_nodata.min()
    numpy.testing.assert_allclose(result, 0)

    # test max
    result = data_tile.max()
    numpy.testing.assert_allclose(result, 3)
    result = data_tile_nodata.max()
    numpy.testing.assert_allclose(result, 2)

    # test sum
    result = data_tile.sum()
    numpy.testing.assert_allclose(result, 6)
    result = data_tile_nodata.sum()
    numpy.testing.assert_allclose(result, 3)


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1),
        RectGrid(size=1),
        HexGrid(size=1),
    ],
)
def test_basic_operations_with_otehr_grids(grid):
    """The setup for this tests is as follows

    Tile 1:
        __ __ __
     __|_0|_1|_2|
    |__|_3|_4|_5|
    |__|_6|_7|_8|
    |__|__|__|

    Tile 2:
        __ __ __
     __|__|__|__|
    |_0|_1|_2|__|
    |_3|_4|_5|__|
    |_6|_7|_8|

    Overlap is:
     __ __      __ __
    |_3|_4|    |_1|_2|
    |_6|_7| vs |_4|_5|

    """
    tile1 = Tile(grid, (0, 0), 3, 3)
    tile2 = Tile(grid, (-1, -1), 3, 3)
    data = numpy.arange(9).reshape((3, 3))
    data_tile1 = DataTile(tile1, data, nodata_value=4)
    data_tile2 = DataTile(tile2, data, nodata_value=4)

    # test addition
    result = data_tile1 + data_tile2
    expected = numpy.array(
        [
            [4, 0, 1, 2],
            [0, 4, 4, 5],
            [3, 4, 12, 8],
            [6, 7, 8, 4],
        ]
    )
    numpy.testing.assert_allclose(result, expected)
    result = data_tile2 + data_tile1
    expected = numpy.array(
        [
            [4, 0, 1, 2],
            [0, 4, 4, 5],
            [3, 4, 12, 8],
            [6, 7, 8, 4],
        ]
    )
    numpy.testing.assert_allclose(result, expected)

    # test subtraction
    result = data_tile1 - data_tile2
    expected = numpy.array(
        [
            [4, 0, 1, 2],
            [0, 2, 4, 5],
            [3, 4, 2, 8],
            [6, 7, 8, 4],
        ]
    )
    numpy.testing.assert_allclose(result, expected)
    result = data_tile2 - data_tile1
    expected = numpy.array(
        [
            [4, 0, 1, 2],
            [0, -2, 4, 5],
            [3, 4, -2, 8],
            [6, 7, 8, 4],
        ]
    )
    numpy.testing.assert_allclose(result, expected)

    # test multiplication
    result = data_tile1 * data_tile2
    expected = numpy.array(
        [
            [4, 0, 1, 2],
            [0, 3, 4, 5],
            [3, 4, 35, 8],
            [6, 7, 8, 4],
        ]
    )
    numpy.testing.assert_allclose(result, expected)
    result = data_tile2 * data_tile1
    expected = numpy.array(
        [
            [4, 0, 1, 2],
            [0, 3, 4, 5],
            [3, 4, 35, 8],
            [6, 7, 8, 4],
        ]
    )
    numpy.testing.assert_allclose(result, expected)

    # test division
    result = data_tile1 / data_tile2
    expected = numpy.array(
        [
            [4, 0, 1, 2],
            [0, 3, 4, 5],
            [3, 4, 7 / 5, 8],
            [6, 7, 8, 4],
        ]
    )
    numpy.testing.assert_allclose(result, expected)
    result = data_tile2 / data_tile1
    expected = numpy.array(
        [
            [4, 0, 1, 2],
            [0, 1 / 3, 4, 5],
            [3, 4, 5 / 7, 8],
            [6, 7, 8, 4],
        ]
    )
    numpy.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1),
        RectGrid(size=1),
        HexGrid(size=1),
    ],
)
def test_crop(grid):
    tile = Tile(grid, (0, 0), 3, 3)
    data = numpy.arange(9).reshape((3, 3))
    data_tile = DataTile(tile, data)

    expected_data = numpy.array(
        [
            [4, 5],
            [7, 8],
        ]
    )  # Expect the first row of x and the last row of y to be dropped
    cropped = data_tile.crop(Tile(grid, (1, 0), 2, 2))
    numpy.testing.assert_almost_equal(cropped, expected_data)
    numpy.testing.assert_almost_equal(cropped.start_id, (1, 0))


@pytest.mark.parametrize(
    "grid, expected_centroids",
    [
        (
            TriGrid(size=1),
            [
                [[-0.5, 1.44337567], [0, 1.15470054]],
                [[-0.5, 0.28867513], [0, 0.57735027]],
            ],
        ),
        (RectGrid(size=1), [[[-1.5, 1.5], [-0.5, 1.5]], [[-1.5, 0.5], [-0.5, 0.5]]]),
        (
            HexGrid(size=1),
            [
                [[-1, 1.29903811], [0, 1.29903811]],
                [[-1.5, 0.4330127], [-0.5, 0.4330127]],
            ],
        ),
    ],
)
def test_centroid(grid, expected_centroids):
    tile = Tile(grid, (-2, 0), 2, 2)
    data = numpy.array([[0, 1], [2, 3]])
    data_tile = DataTile(tile, data)

    numpy.testing.assert_allclose(data_tile.centroid(), expected_centroids)

    # test using ids
    ids = data_tile.indices
    numpy.testing.assert_allclose(grid.centroid(ids), expected_centroids)


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1),
        RectGrid(size=1),
        HexGrid(size=1),
    ],
)
def test_overlap(grid):
    tile = Tile(grid, (-2, 0), 2, 2)
    data = numpy.array([[0, 1], [2, 3]])
    data_tile = DataTile(tile, data)

    tile2 = Tile(grid, (-1, 1), 2, 2)
    data_tile2 = DataTile(tile2, data)

    def check(overlap_tile):
        assert isinstance(overlap_tile, Tile)
        assert isinstance(overlap_tile.grid, type(grid))
        numpy.testing.assert_allclose(overlap_tile.start_id, tile2.start_id)
        assert overlap_tile.nx == 1
        assert overlap_tile.ny == 1

    # Check all combinations of data_tile and tile
    check(data_tile.overlap(data_tile2))
    check(data_tile2.overlap(data_tile))
    check(data_tile.overlap(tile2))
    check(tile.overlap(data_tile2))
    check(data_tile2.overlap(tile))
    check(tile2.overlap(data_tile))
    check(tile2.overlap(tile))
    check(tile.overlap(tile2))
