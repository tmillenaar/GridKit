import numpy
import pytest

from gridkit import bounded_grid, rect_grid


def test_data_ravel_order():
    data = numpy.arange(9).reshape((3, 3))
    grid = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 3, 3))

    value = grid.value(grid.cell_at_point(grid.centroid()))
    numpy.testing.assert_equal(value, data)


def test_data_mutability():
    data = numpy.arange(9).reshape((3, 3))
    grid = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 3, 3))
    grid_mutable = rect_grid.BoundedRectGrid(
        data, bounds=(0, 0, 3, 3), prevent_copy=True
    )

    assert id(data) != id(grid.data)
    assert id(data) == id(grid_mutable.data)


def test_data_setter():
    data = numpy.arange(9).reshape((3, 3))
    grid = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 3, 3))

    grid.data = data + 1
    numpy.testing.assert_equal(data + 1, grid.data)

    with pytest.raises(TypeError) as e:
        grid.data = str
    assert (
        str(e.value)
        == "Data cannot be interpreted as a numpy.ndarray, got <class 'type'>"
    )

    with pytest.raises(ValueError) as e:
        grid.data = [1, 2]
    assert (
        str(e.value)
        == "Cannot set data that is different in size. Expected a shape of (3, 3), got (2,)."
    )


@pytest.mark.parametrize(
    "bounds, expected",
    [
        [(0, 0, 2, 2), True],
        [(3, 3, 6, 6), False],
        [(3, 0, 6, 2), False],
        [(0, 3, 2, 6), False],
        [(-2, -2, 1, 1), True],
    ],
)
def test_intersects(bounds, expected):
    data = numpy.arange(9).reshape((3, 3))
    grid1 = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 3, 3))
    grid2 = rect_grid.BoundedRectGrid(data, bounds=bounds)
    result = grid1.intersects(grid2)
    assert result == expected


@pytest.mark.parametrize(
    "bounds, expected_data",
    (
        [
            [
                (2, 0, 5, 3),
                numpy.array(
                    [
                        [0, 1, 2, 1, 2],
                        [3, 4, 8, 4, 5],
                        [6, 7, 14, 7, 8],
                    ]
                ),
            ],
            [
                (2, 1, 5, 4),
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
def test_add_partial_overlap(bounds, expected_data):
    data = numpy.arange(9).reshape((3, 3))
    grid1 = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 3, 3))

    # test with nodata_value of NaN
    grid2 = rect_grid.BoundedRectGrid(
        data.astype(float), bounds=bounds, nodata_value=numpy.nan
    )
    result = grid1 + grid2
    numpy.testing.assert_allclose(result.data, expected_data)

    # test with nodata_value of 1
    grid3 = rect_grid.BoundedRectGrid(data, bounds=bounds, nodata_value=-1)
    expected_data[numpy.isnan(expected_data)] = -1
    result = grid1 + grid3
    numpy.testing.assert_allclose(result.data, expected_data)


def test_basic_operations_with_scalars():
    # initialize grid
    data = numpy.arange(4).reshape((2, 2))
    grid = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 2, 2))
    grid_nodata = grid.copy()
    grid_nodata.nodata_value = 2

    # test addition
    result = 2 + grid
    expected = numpy.array([[2, 3], [4, 5]])
    numpy.testing.assert_allclose(result.data, expected)
    result = grid + 2
    numpy.testing.assert_allclose(result.data, expected)

    result = 2 + grid_nodata
    expected = numpy.array([[2, 3], [2, 5]])
    numpy.testing.assert_allclose(result.data, expected)
    result = grid_nodata + 2
    numpy.testing.assert_allclose(result.data, expected)

    # test subtraction
    result = 2 - grid
    expected = numpy.array([[2, 1], [0, -1]])
    numpy.testing.assert_allclose(result.data, expected)
    result = grid - 2
    expected = numpy.array([[-2, -1], [0, 1]])
    numpy.testing.assert_allclose(result.data, expected)

    result = 2 - grid_nodata
    expected = numpy.array([[2, 1], [2, -1]])
    numpy.testing.assert_allclose(result.data, expected)
    result = grid_nodata - 2
    expected = numpy.array([[-2, -1], [2, 1]])
    numpy.testing.assert_allclose(result.data, expected)

    # test multiplication
    result = 2 * grid
    expected = numpy.array([[0, 2], [4, 6]])
    numpy.testing.assert_allclose(result.data, expected)
    result = grid * 2
    numpy.testing.assert_allclose(result.data, expected)

    result = 2 * grid_nodata
    expected = numpy.array([[0, 2], [2, 6]])
    numpy.testing.assert_allclose(result.data, expected)
    result = grid_nodata * 2
    numpy.testing.assert_allclose(result.data, expected)

    # test division
    result = 2 / grid
    expected = numpy.array([[numpy.inf, 2], [1, 2 / 3]])
    numpy.testing.assert_allclose(result.data, expected)
    result = grid / 2
    expected = numpy.array([[0, 1 / 2], [1, 3 / 2]])
    numpy.testing.assert_allclose(result.data, expected)

    result = 2 / grid_nodata
    expected = numpy.array([[numpy.inf, 2], [2, 2 / 3]])
    numpy.testing.assert_allclose(result.data, expected)
    result = grid_nodata / 2
    expected = numpy.array([[0, 1 / 2], [2, 3 / 2]])
    numpy.testing.assert_allclose(result.data, expected)

    # test power
    result = 2**grid
    expected = numpy.array([[1, 2], [4, 8]])
    numpy.testing.assert_allclose(result.data, expected)
    result = grid**2
    expected = numpy.array([[0, 1], [4, 9]])
    numpy.testing.assert_allclose(result.data, expected)

    result = 2**grid_nodata
    expected = numpy.array([[1, 2], [2, 8]])
    numpy.testing.assert_allclose(result.data, expected)
    result = grid_nodata**2
    expected = numpy.array([[0, 1], [2, 9]])
    numpy.testing.assert_allclose(result.data, expected)


@pytest.mark.parametrize("nodata", [None, 1])
def test_comparissons_with_scalars(nodata):
    # initialize grid
    data = numpy.arange(4).reshape((2, 2))
    grid = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 2, 2))

    grid.nodata = nodata  # should not have an effect, set here to make sure

    # test equal
    result = grid == 1
    expected = numpy.array([1, 1])
    numpy.testing.assert_allclose(result.index, expected)

    # test not equal
    result = grid != 1
    expected = numpy.array([[0, 1], [0, 0], [1, 0]])
    numpy.testing.assert_allclose(result.index, expected)

    # # test greater than
    result = grid > 1
    expected = numpy.array([[0, 0], [1, 0]])
    numpy.testing.assert_allclose(result.index, expected)

    # # test smaller than
    result = grid < 1
    expected = numpy.array([0, 1])
    numpy.testing.assert_allclose(result.index, expected)

    # # test greater than or equal
    result = grid >= 1
    expected = numpy.array([[1, 1], [0, 0], [1, 0]])
    numpy.testing.assert_allclose(result.index, expected)

    # # test smaller than or equal
    result = grid <= 1
    expected = numpy.array([[0, 1], [1, 1]])
    numpy.testing.assert_allclose(result.index, expected)


def test_reduction_operators():
    # initialize grid
    data = numpy.arange(4).reshape((2, 2))
    grid = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 2, 2))
    grid_nodata = grid.copy()
    grid_nodata.nodata_value = 3

    # test mean
    result = grid.mean()
    numpy.testing.assert_allclose(result.data, 1.5)
    result = grid_nodata.mean()
    numpy.testing.assert_allclose(result.data, 1)

    # test median
    result = grid.median()
    numpy.testing.assert_allclose(result.data, 1.5)
    result = grid_nodata.median()
    numpy.testing.assert_allclose(result.data, 1)

    # test std
    result = grid.std()
    numpy.testing.assert_allclose(result.data, 5**0.5 / 2)
    result = grid_nodata.std()
    numpy.testing.assert_allclose(result.data, (2 / 3) ** 0.5)

    # test min
    result = grid.min()
    numpy.testing.assert_allclose(result.data, 0)
    result = grid_nodata.min()
    numpy.testing.assert_allclose(result.data, 0)

    # test max
    result = grid.max()
    numpy.testing.assert_allclose(result.data, 3)
    result = grid_nodata.max()
    numpy.testing.assert_allclose(result.data, 2)

    # test sum
    result = grid.sum()
    numpy.testing.assert_allclose(result.data, 6)
    result = grid_nodata.sum()
    numpy.testing.assert_allclose(result.data, 3)


def test_reduction_operators_arg():
    # initialize grid
    data = numpy.arange(9).reshape((3, 3))
    grid = rect_grid.BoundedRectGrid(data, bounds=(-10, -0.7, 1, 25))

    # test argmax
    idx = grid.argmax()
    value = grid.value(idx)
    assert idx.index.shape == (2,)
    numpy.testing.assert_allclose(value, data.max())

    # test argmin
    idx = grid.argmin()
    value = grid.value(idx)
    assert idx.index.shape == (2,)
    numpy.testing.assert_allclose(value, data.min())

    # test argmax with ndoata value
    grid.nodata_value = 8
    idx = grid.argmax()
    value = grid.value(idx)
    assert idx.index.shape == (2,)
    numpy.testing.assert_allclose(value, 7)

    # test argmin with nodata value
    grid.nodata_value = 0
    idx = grid.argmin()
    value = grid.value(idx)
    assert idx.index.shape == (2,)
    numpy.testing.assert_allclose(value, 1)


def test_dtype_after_division():
    data = numpy.arange(9).reshape((3, 3))
    grid = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 3, 3))
    result = grid / (grid + 1)
    assert numpy.issubdtype(
        result.dtype, float
    ), "Incorrect datatype after division operation"


def test_crop():
    data = numpy.arange(9).reshape((3, 3))
    grid = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 3, 3))
    bounds = (0.7, -1, 5, 2)

    expected_data = numpy.array(
        [
            [4, 5],
            [7, 8],
        ]
    )  # Expect the first row of x and the last row of y to be dropped
    cropped = grid.crop(bounds)

    numpy.testing.assert_almost_equal(cropped.data, expected_data)
    assert cropped.bounds == (1, 0, 3, 2)


def test_centroid():
    data = numpy.array([[0, 1], [2, 3]])
    grid = rect_grid.BoundedRectGrid(data, bounds=(-2, 0, 0, 2))

    expected_centroids = [[[-1.5, 1.5], [-0.5, 1.5]], [[-1.5, 0.5], [-0.5, 0.5]]]
    numpy.testing.assert_equal(grid.centroid(), expected_centroids)

    # test using ids
    ids = grid.indices
    numpy.testing.assert_equal(grid.centroid(ids), expected_centroids)


def test_centroid_index_comparisson():
    data = numpy.array([[0, 1], [2, 3]])
    grid = rect_grid.BoundedRectGrid(data, bounds=(-2, 0, 0.2, 2.1))
    ids = grid.indices
    numpy.testing.assert_almost_equal(grid.centroid(ids), grid.centroid())


def test_assign():
    data = numpy.array([[0, 1], [2, 3]])
    grid = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 2, 2))

    # Test assigning single value
    expected_data = numpy.array([[0, 10], [2, 10]])
    result_grid = grid.assign(10, bounds=(1, 0, 2, 2))
    numpy.testing.assert_allclose(result_grid.data, expected_data)

    # Test assigning 2D-array
    assign_data = numpy.array([[10], [20]])
    expected_data = numpy.array([[0, 10], [2, 20]])
    result_grid = grid.assign(assign_data, bounds=(1, 0, 2, 2))
    numpy.testing.assert_allclose(result_grid.data, expected_data)


@pytest.mark.parametrize(
    "bounds, expected_offset",
    (
        [(0.5, -0.5, 3.5, 1.5), (0.5, 0.5)],
        [(-0.2, 0.1, 2.8, 1.9), (0.8, 0.1)],
    ),
)
def test_offset(bounds, expected_offset):
    data = numpy.array([[0, 1, 3], [2, 3, 4]])
    grid = rect_grid.BoundedRectGrid(data, bounds=bounds)
    numpy.testing.assert_allclose(grid.offset, expected_offset)


@pytest.mark.parametrize(
    "method, expected_result",
    (
        ("nearest", [[3, 3, 4, 4, 5, 5]]),
        ("bilinear", [[numpy.nan, 2.5, 3, 3.5, 4, numpy.nan]]),
    ),
)
def test_resample(method, expected_result):
    data = numpy.array([[0, 1, 2], [3, 4, 5]], dtype=float)
    grid = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 3, 2))
    new_grid = rect_grid.RectGrid(dx=0.5, dy=1.5)

    resampled = grid.resample(new_grid, method=method)

    numpy.testing.assert_allclose(resampled.data, expected_result)
    numpy.testing.assert_allclose(resampled.bounds, (0, 0, 3, 1.5))


def test_crs():
    data = numpy.arange(9).reshape((3, 3))
    crs = 3857
    grid = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 3, 3), crs=crs)
    new_grid = grid.to_crs(crs=4326, resample_method="nearest")

    expected_dx = expected_dy = 8.983152841195214e-06

    # test crs object
    assert new_grid.crs.to_epsg() == 4326
    # test
    numpy.testing.assert_allclose(new_grid.dx, expected_dx)
    numpy.testing.assert_allclose(new_grid.dy, expected_dy)
    # test bounds
    numpy.testing.assert_allclose(
        new_grid.bounds, (0, 0, expected_dx * 3, expected_dy * 3)
    )


def test__mask_to_index():
    data = numpy.arange(9).reshape((3, 3))
    grid = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 3, 3))

    expected_ids = [
        [1, 1],
        [2, 1],
        [0, 0],
        [2, 0],
    ]

    mask = numpy.array(
        [[False, False, False], [False, True, True], [True, False, True]]
    )

    ids = grid._mask_to_index(mask)

    numpy.testing.assert_allclose(expected_ids, ids)


@pytest.mark.parametrize(
    "ids, expected_value",
    (
        [[1, 2], [6]],
        [[(3, 4), (3, 2)], [2, 8]],
        [[(0, 3), (2, 5), (4, 3), (2, 1)], 4 * [numpy.nan]],
    ),
)
def test_value_positive_bounds(ids, expected_value):
    data = numpy.arange(9).reshape((3, 3)).astype("float")
    grid = rect_grid.BoundedRectGrid(data, bounds=(1, 2, 4, 5), nodata_value=numpy.nan)

    result = grid.value(ids)

    numpy.testing.assert_allclose(expected_value, result)


@pytest.mark.parametrize(
    "ids, expected_value",
    (
        [[-4, -5], [6]],
        [[(-2, -3), (-2, -5)], [2, 8]],
        [[(-5, -4), (-3, -2), (-1, -4), (-3, -6)], 4 * [numpy.nan]],
    ),
)
def test_value_negative_bounds(ids, expected_value):
    data = numpy.arange(9).reshape((3, 3)).astype("float")
    grid = rect_grid.BoundedRectGrid(
        data, bounds=(-4, -5, -1, -2), nodata_value=numpy.nan
    )

    result = grid.value(ids)

    numpy.testing.assert_allclose(expected_value, result)


def test_value_nd_index():
    data = numpy.arange(9).reshape((3, 3)).astype("float")
    grid = rect_grid.BoundedRectGrid(data, bounds=(1, 2, 4, 5), nodata_value=numpy.nan)

    result = grid.value([[[1, 2]], [[3, 4]]])

    numpy.testing.assert_allclose(result, [6, 2])


@pytest.mark.parametrize("percentile", (2, 31, 50, 69, 98))
def test_percentile(percentile):
    data = numpy.arange(9).reshape((3, 3)).astype("float")
    grid = rect_grid.BoundedRectGrid(data, bounds=(0, 0, 3, 3))
    numpy.testing.assert_allclose(
        numpy.percentile(data, percentile), grid.percentile(percentile)
    )


@pytest.mark.parametrize(
    "method, expected_min, expected_max",
    (
        ["linear", 0.011254918521840882, 0.9697422513409796],
        ["nearest", -0.005508092250476611, 0.9740280288214683],
        ["cubic", 0.013773033571295269, 0.9919834390043766],
    ),
)
def test_interp_from_points(method, expected_min, expected_max):
    numpy.random.seed(0)
    x = 100 * numpy.random.rand(100)
    numpy.random.seed(1)
    y = 100 * numpy.random.rand(100)
    value = numpy.sin(x / (10 * numpy.pi)) * numpy.sin(y / (10 * numpy.pi))

    empty_grid = rect_grid.RectGrid(dx=5, dy=5)
    data_grid = empty_grid.interp_from_points(
        numpy.array([x, y]).T,
        value,
        method=method,
        nodata_value=float("nan"),
    )

    numpy.testing.assert_allclose(data_grid.max(), expected_max)
    numpy.testing.assert_allclose(data_grid.min(), expected_min)
    if method != "nearest":
        assert len(data_grid.nodata()) == 30


@pytest.mark.parametrize(
    "shape, expected_ids",
    [
        (
            "rect",
            (
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
            ),
        ),
        ("pointy", ([0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 0, 1, 0, 1])),
        ("flat", ([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])),
    ],
)
def test_grid_id_to_numpy_id(
    basic_bounded_rect_grid,
    basic_bounded_pointy_grid,
    basic_bounded_flat_grid,
    shape,
    expected_ids,
):
    grid_lut = dict(
        rect=basic_bounded_rect_grid,
        pointy=basic_bounded_pointy_grid,
        flat=basic_bounded_flat_grid,
    )
    grid = grid_lut[shape]
    np_ids = grid.grid_id_to_numpy_id(grid.indices.ravel())
    numpy.testing.assert_allclose(numpy.array(np_ids), expected_ids)

    # also check for error when supplying nd-index
    with pytest.raises(ValueError):
        result = grid.grid_id_to_numpy_id([[[1, 2], [1, 2]], [[10, 20], [10, 20]]])


@pytest.mark.parametrize("shape", ["rect", "pointy", "flat"])
def test_grid_id_to_numpy_id_back_and_forth(
    basic_bounded_rect_grid, basic_bounded_pointy_grid, basic_bounded_flat_grid, shape
):
    grid_lut = dict(
        rect=basic_bounded_rect_grid,
        pointy=basic_bounded_pointy_grid,
        flat=basic_bounded_flat_grid,
    )
    grid = grid_lut[shape]
    np_ids = grid.grid_id_to_numpy_id(grid.indices.ravel())
    grid_ids = grid.numpy_id_to_grid_id(np_ids)
    numpy.testing.assert_allclose(grid_ids, grid.indices.ravel())
