import numpy
import pytest
import shapely
import shapely.geometry

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
    data = numpy.arange(9, dtype=float).reshape((3, 3))
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
    expected = numpy.array([[numpy.inf, 2], [numpy.nan, 2 / 3]])
    numpy.testing.assert_allclose(result, expected)
    result = data_tile_nodata / 2
    expected = numpy.array([[0, 1 / 2], [numpy.nan, 3 / 2]])
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
def test_basic_operations_with_other_grids(grid):
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
    assert result.nx == 4
    assert result.ny == 4
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
    assert result.nx == 4
    assert result.ny == 4
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
    assert result.nx == 4
    assert result.ny == 4
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
    assert result.nx == 4
    assert result.ny == 4
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
    # Note: division is a special case becuase it returns a type f64, always.
    #       Dividing integers should namely return a float too, not an integer.
    #       This means also that the nodata value is hardcoded to be NaN.
    result = data_tile1 / data_tile2
    assert result.nx == 4
    assert result.ny == 4
    expected = numpy.array(
        [
            [numpy.nan, 0, 1, 2],
            [0, 3, numpy.nan, 5],
            [3, numpy.nan, 7 / 5, 8],
            [6, 7, 8, numpy.nan],
        ]
    )
    numpy.testing.assert_allclose(result, expected)
    result = data_tile2 / data_tile1
    assert result.nx == 4
    assert result.ny == 4
    expected = numpy.array(
        [
            [numpy.nan, 0, 1, 2],
            [0, 1 / 3, numpy.nan, 5],
            [3, numpy.nan, 5 / 7, 8],
            [6, 7, 8, numpy.nan],
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
def test_basic_operations_nodata_handling(grid):
    tile1 = Tile(grid, (0, 0), 3, 3)
    tile2 = Tile(grid, (-1, -1), 3, 3)
    data_tile1 = DataTile(tile1, numpy.full((3, 3), 3), nodata_value=3)
    data_tile2 = DataTile(tile2, numpy.full((3, 3), -1), nodata_value=-1)

    result = data_tile1 + data_tile2
    assert len(result == result.nodata_value) == 16
    assert len(result.nodata_cells) == 16

    result = data_tile1 - data_tile2
    assert len(result == result.nodata_value) == 16
    assert len(result.nodata_cells) == 16

    result = data_tile1 * data_tile2
    assert len(result == result.nodata_value) == 16
    assert len(result.nodata_cells) == 16

    result = data_tile1 / data_tile2
    # Cannot test using == because of NaNs
    assert numpy.all(numpy.isnan(result))
    assert len(result.nodata_cells) == 16


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


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1, rotation=13),
        RectGrid(size=1, rotation=13),
        HexGrid(size=1, rotation=13),
    ],
)
@pytest.mark.parametrize("interp_method", ["nearest", "linear", "inverse_distance"])
def test_interpolate(grid, interp_method):
    tile = Tile(grid, (-1, -1), 5, 5)

    data = numpy.arange(tile.nx * tile.ny, dtype=float).reshape((tile.ny, tile.nx))
    data_tile = DataTile(tile, data, nodata_value=-10)

    n = 100
    numpy.random.seed(0)

    corners = data_tile.corners()
    left = corners.T[0].min()
    bottom = corners.T[1].min()
    right = corners.T[0].max()
    top = corners.T[1].max()
    x = ((right - left) * numpy.random.rand(n)) + left
    numpy.random.seed(1)
    y = ((top - bottom) * numpy.random.rand(n)) + bottom
    points = numpy.stack([x, y]).T

    result = data_tile.interpolate(points, method=interp_method)

    # replace nodata_value with nan so we can check with numpy
    result[result == data_tile.nodata_value] = numpy.nan

    assert numpy.nanmax(result) <= data_tile.max()
    assert numpy.nanmin(result) >= data_tile.min()

    if interp_method == "nearest":
        ref_geoms = data_tile.to_shapely(as_multipolygon=True)
    elif interp_method == "linear" or interp_method == "inverse_distance":
        # Create a reference grid and spcify the cells in which the points should contain data
        if isinstance(grid, TriGrid):
            # Because the 6 nearby cells are used, we get a hexagonal shape for the reference grid
            ref_grid = HexGrid(size=grid.size, rotation=grid.rotation)
            ref_grid = ref_grid.update(offset=(0, ref_grid.dy / 2))
            ref_geoms = ref_grid.to_shapely(
                [[1, 2], [0, 2], [0, 1], [0, 0], [1, 0], [0, -1]], as_multipolygon=True
            )
        elif isinstance(grid, RectGrid):
            ref_grid = grid.update(
                offset=(grid.offset[0] + grid.dx / 2, grid.offset[1] - grid.dy / 2)
            )
            ref_tile = Tile(ref_grid, (-1, -1), 4, 4)
            ref_geoms = ref_tile.to_shapely(as_multipolygon=True)
        elif isinstance(grid, HexGrid):
            ref_grid = TriGrid(
                side_length=grid.dx,
                rotation=grid.rotation,
                offset=(grid.dx / 2, grid.dy / 2),
            )
            ref_tile = Tile(ref_grid, (-2, -1), 8, 4)
            ref_geoms = ref_tile.to_shapely(as_multipolygon=True)

    for i, (point, val) in enumerate(zip(points, result)):
        # Unfortunately we have to check the point for every cell in python.
        # Ideally we would leave the looping over geometries to shapely and do:
        # intersects = geoms.contains(shapely.geometry.Point(*point))
        # But that does not work because the geoms is not valid and
        # shapely.make_valid returns a collection of polygons, lines and points, which we cannot work with.
        # Since we have sufficiently few cells to check this is fine for this test and no effort is made to speed this up.
        intersects = numpy.any(
            [shapely.geometry.Point(*point).within(geom) for geom in ref_geoms.geoms]
        )
        if intersects:
            assert numpy.isfinite(val)
        else:
            assert numpy.isnan(val)


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1, rotation=13, crs=4326),
        RectGrid(size=1, rotation=13, crs=4326),
        HexGrid(size=1, rotation=13, crs=4326),
    ],
)
@pytest.mark.parametrize("interp_method", ["nearest", "linear", "inverse_distance"])
def test_resample(grid, interp_method):
    tile = Tile(grid, (-2, -2), 5, 5)
    data = numpy.arange(tile.nx * tile.ny).reshape((tile.nx, tile.ny))
    data_tile = tile.to_data_tile(data)

    new_grid = grid.update(rotation=-5, offset=(-0.2, 0.3))
    new_tile = Tile(new_grid, tile.start_id, tile.nx, tile.ny)

    result = data_tile.resample(new_tile, method=interp_method)

    # For each nodata value make sure there is at least one neighbour that is outside of the original tile
    for id, value in zip(result.indices, result.to_numpy().ravel()):
        if result.is_nodata(value):
            original_cells_near_nodata_cell = tile.grid.cells_near_point(
                result.grid.centroid(id)
            )
            # check that at least one cell is outside the original tile to warrent the nodata value
            cells_otuside_orignal_tile = original_cells_near_nodata_cell.difference(
                tile.indices
            )
            assert len(cells_otuside_orignal_tile) > 0
            # or equivalently
            assert cells_otuside_orignal_tile  # Fails when empty. Ain't that both fun and hard to read? :)

    # The new data should still be roughly in inascending order
    # Not exactly, but close enough
    data_values = [v for v in result.to_numpy().ravel() if not result.is_nodata(v)]
    assert data_values[0] < data_values[-1]

    # All interpolation methods used here smooth to some exent and never get values larger than in the original
    assert result.min() >= data_tile.min()
    assert result.max() <= data_tile.max()
    assert result.mean() > data_tile.min() and result.mean() < data_tile.max()
