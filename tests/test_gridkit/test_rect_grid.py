import numpy
import pytest
import shapely

from gridkit import HexGrid
from gridkit.rect_grid import BoundedRectGrid, RectGrid


@pytest.mark.parametrize(
    "points, expected_ids",
    [
        [(340, -14.2), (68, -8)],  # test single point as tuple
        [[[14, 3], [-8, 1]], [[2, 1], [-2, 0]]],  # test multiple points as stacked list
        [
            numpy.array([[14, 3], [-8, 1]]),
            numpy.array([[2, 1], [-2, 0]]),
        ],  # test multiple points as numpy ndarray
    ],
)
def test_cell_at_point(points, expected_ids):
    # TODO: take offset into account
    ids = RectGrid(dx=5, dy=2).cell_at_point(points)
    numpy.testing.assert_allclose(ids, expected_ids)


def test_centroid():
    grid = RectGrid(dx=5, dy=2)
    indices = [(-1, 1), (1, -4)]
    expected_centroids = numpy.array([[-2.5, 3], [7.5, -7]])

    centroids = grid.centroid(indices)
    numpy.testing.assert_allclose(centroids, expected_centroids)


@pytest.mark.parametrize(
    "mode, expected_bounds",
    [
        ["expand", (-8, 1, -3, 6.5)],
        ["contract", (-7, 1.5, -4, 6)],
        ["nearest", (-7, 1.5, -3, 6.5)],
    ],
)
def test_align_bounds(mode, expected_bounds):
    grid = RectGrid(dx=1, dy=0.5)
    bounds = (-7.2, 1.4, -3.2, 6.4)

    new_bounds = grid.align_bounds(bounds, mode=mode)
    numpy.testing.assert_allclose(new_bounds, expected_bounds)


@pytest.mark.parametrize("mode", ("expand", "contract", "nearest"))
def test_align_bounds_with_are_bounds_aligned(mode):
    grid = RectGrid(dx=3, dy=6, offset=(1, 2))
    bounds = (-7, -5, -1, 6.4)
    new_bounds = grid.align_bounds(bounds, mode=mode)
    assert grid.are_bounds_aligned(new_bounds), "Bounds are not aligned"


def test_cells_in_bounds():
    grid = RectGrid(dx=1, dy=2)
    bounds = (-5.2, 1.4, -3.2, 2.4)

    expected_ids = numpy.array(
        [
            [[-6.0, 1.0], [-5.0, 1.0], [-4.0, 1.0]],
            [[-6.0, 0.0], [-5.0, 0.0], [-4.0, 0.0]],
        ]
    )

    aligned_bounds = grid.align_bounds(bounds, mode="expand")
    ids, shape = grid.cells_in_bounds(aligned_bounds, return_cell_count=True)
    numpy.testing.assert_allclose(ids, expected_ids)
    assert shape == (2, 3)
    assert ids.shape == shape


def test_crs():
    dx = 10
    dy = 20
    offset = (5, 10)
    crs = 3857
    grid = RectGrid(dx=dx, dy=dy, offset=offset, crs=crs)
    new_grid = grid.to_crs(crs=4326)

    expected_dx = 8.983152841195213e-05
    expected_dy = 2 * expected_dx

    assert new_grid.crs.to_epsg() == 4326
    numpy.testing.assert_allclose(new_grid.dx, expected_dx)
    numpy.testing.assert_allclose(new_grid.dy, expected_dy)
    numpy.testing.assert_allclose(new_grid.offset, (expected_dx / 2, expected_dy / 2))


@pytest.mark.parametrize(
    "index,expected_np_id,expected_value",
    [  # note, numpy id is in y,x
        [(0, 0), ([2, 1]), 7],
        [
            [(-1, 1), (1, -2)],  # index
            [(1, 4), (0, 2)],  # expected_np_id in [(y0, y1), (x0,x1)]
            [3, 14],  # expected_value
        ],
    ],
)
def test_grid_id_to_numpy_id(
    basic_bounded_rect_grid, index, expected_np_id, expected_value
):
    grid = basic_bounded_rect_grid

    result = grid.grid_id_to_numpy_id(index)
    numpy.testing.assert_almost_equal(result, expected_np_id)
    numpy.testing.assert_almost_equal(grid.data[result[0], result[1]], expected_value)


def test_grid_id_to_numpy_id_nd(basic_bounded_rect_grid):
    grid = basic_bounded_rect_grid
    index = [[[1, 2], [1, 2]]]
    result = grid.grid_id_to_numpy_id(index)
    numpy.testing.assert_allclose(grid.data[result], grid.value(index))


@pytest.mark.parametrize(
    "np_index,expected_grid_id",
    [  # note, numpy id is in y,x
        [(2, 1), (0, 0)],
        [
            [(1, 4), (0, 2)],  # np_index
            [(-1, 1), (1, -2)],  # expected_grid_id in [(y0, y1), (x0,x1)]
        ],
    ],
)
def test_numpy_id_to_grid_id(basic_bounded_rect_grid, np_index, expected_grid_id):
    grid = basic_bounded_rect_grid
    result = grid.numpy_id_to_grid_id(np_index)

    numpy.testing.assert_almost_equal(result, expected_grid_id)
    numpy.testing.assert_almost_equal(
        grid.data[np_index[0], np_index[1]], grid.value(result)
    )


def test_nodata_value(basic_bounded_rect_grid):
    grid = basic_bounded_rect_grid
    grid.nodata_value = None

    # test no data are nodata
    assert grid.nodata_value is None
    assert grid.nodata() is None

    # test only one value is nodata
    grid.nodata_value = 6
    numpy.testing.assert_allclose(grid.nodata(), [-1, 0])

    # test all values are nodata
    grid = grid.update(numpy.ones((grid.height, grid.width)))
    assert grid.nodata_value == 6  # make sure nodata is inhertied after update
    grid.nodata_value = 1
    numpy.testing.assert_allclose(
        grid.nodata(), grid.cells_in_bounds(grid.bounds).ravel()
    )


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic"])
def test_interp_nodata(basic_bounded_rect_grid, method):
    grid = basic_bounded_rect_grid.copy()
    grid.data[1:4, 0:4] = grid.nodata_value
    result_grid = grid.interp_nodata(method=method)

    if method == "nearest":
        numpy.testing.assert_allclose(
            result_grid, numpy.vstack([3 * [[0, 1, 2]], 2 * [[12, 13, 14]]])
        )
    else:
        numpy.testing.assert_allclose(result_grid, basic_bounded_rect_grid)


@pytest.mark.parametrize("depth", list(range(1, 7)))
@pytest.mark.parametrize("index", [[2, 1], [1, 2]])
@pytest.mark.parametrize("include_selected", [False, True])
@pytest.mark.parametrize("connect_corners", [False, True])
def test_neighbours(depth, index, include_selected, connect_corners):
    grid = RectGrid(dx=3, dy=3)
    neighbours = grid.neighbours(
        index,
        depth=depth,
        connect_corners=connect_corners,
        include_selected=include_selected,
    )

    if include_selected:
        # make sure the index is present and at the center of the neighbours array
        center_index = int(numpy.floor(len(neighbours) / 2))
        numpy.testing.assert_allclose(neighbours[center_index], index)
        # remove center index for further testing of other neighbours
        neighbours = numpy.delete(neighbours, center_index, axis=0)

    # If the neighbours are correct, there are always a multiple of 6 cells with the same distance to the center cell
    distances = numpy.linalg.norm(
        grid.centroid(neighbours) - grid.centroid(index), axis=1
    )
    for d in numpy.unique(distances):
        assert sum(distances == d) % 4 == 0

    if connect_corners:
        # No cell can be further away than 'depth' number of cells * diagonal
        assert all(
            distances <= (grid.dx**2 + grid.dy**2) ** 0.5 * depth + 1e-14
        )  # add 1e-14 in absence of atol
    else:
        # No cell can be further away than 'depth' number of cells * cell size
        assert all(distances <= grid.dx * depth)


@pytest.mark.parametrize("include_selected", [False, True])
@pytest.mark.parametrize(
    "connect_corners, expected_cell_ids",
    [
        (
            False,
            [
                [
                    [-1, 1],
                    [-2, 0],
                    [-1, 0],  # selected id
                    [0, 0],
                    [-1, -1],
                ],
                [
                    [2, 2],
                    [1, 1],
                    [2, 1],  # selected id
                    [3, 1],
                    [2, 0],
                ],
            ],
        ),
        (
            True,
            [
                [
                    [-2, 1],
                    [-1, 1],
                    [0, 1],
                    [-2, 0],
                    [-1, 0],  # selected id
                    [0, 0],
                    [-2, -1],
                    [-1, -1],
                    [0, -1],
                ],
                [
                    [1, 2],
                    [2, 2],
                    [3, 2],
                    [1, 1],
                    [2, 1],  # selected id
                    [3, 1],
                    [1, 0],
                    [2, 0],
                    [3, 0],
                ],
            ],
        ),
    ],
)
def test_neighbours_multiple_indices(
    include_selected, connect_corners, expected_cell_ids
):
    expected_cell_ids = numpy.array(expected_cell_ids)
    testgrid = RectGrid(dx=1, dy=2)

    neighbours = testgrid.neighbours(
        [[-1, 0], [2, 1]],
        connect_corners=connect_corners,
        include_selected=include_selected,
    )
    if not include_selected:
        # exclude fourth cell (center)
        center_cell = int(expected_cell_ids.shape[1] / 2)
        remaining_cells = numpy.delete(
            numpy.arange(expected_cell_ids.shape[1]), center_cell
        )
        expected_cell_ids = expected_cell_ids[:, remaining_cells]

    numpy.testing.assert_allclose(neighbours, expected_cell_ids)
    neighbours_one_point = testgrid.neighbours(
        [-1, 0], connect_corners=connect_corners, include_selected=include_selected
    )
    numpy.testing.assert_allclose(neighbours_one_point, expected_cell_ids[0])


def test_to_bounded():
    grid = RectGrid(dx=1.5, dy=1.5)
    grid.crs = 4326
    bounds = (1, 1, 5, 5)
    bounds = grid.align_bounds(bounds)
    result = grid.to_bounded(bounds)

    assert isinstance(result, BoundedRectGrid)
    assert result.data.shape == (4, 4)
    assert grid.dx == result.dx
    assert grid.dy == result.dy
    assert grid.crs.is_exact_same(result.crs)
    assert bounds == result.bounds
    assert numpy.all(numpy.isnan(result.data))


@pytest.mark.parametrize("as_mp", [True, False])
def test_to_shapely(as_mp):
    grid = RectGrid(dx=1.5, dy=1.5)
    ids = [[-3, 2], [3, -6]]
    geoms = grid.to_shapely(ids, as_multipolygon=as_mp)

    if as_mp:
        assert isinstance(geoms, shapely.geometry.MultiPolygon)
        geoms = geoms.geoms

    centroids = [[geom.centroid.x, geom.centroid.y] for geom in geoms]
    numpy.testing.assert_allclose(centroids, grid.centroid(ids))
    for geom in geoms:
        numpy.testing.assert_allclose(geom.area, grid.dx * grid.dy)


def test_cell_corners():
    grid = RectGrid(dx=1.5, dy=1.5)
    ids = [[-3, 2], [3, -6]]

    corners = grid.cell_corners(ids)

    expected_corners = numpy.array(
        [
            [[-4.5, 3.0], [-3.0, 3.0], [-3.0, 4.5], [-4.5, 4.5]],
            [[4.5, -9.0], [6.0, -9.0], [6.0, -7.5], [4.5, -7.5]],
        ]
    )

    numpy.testing.assert_allclose(corners, expected_corners)


def test_is_aligned_with():
    grid = RectGrid(dx=1.2, dy=1.2)

    is_aligned, reason = grid.is_aligned_with(grid)
    assert is_aligned
    assert reason == ""

    other_grid = RectGrid(dx=1.2, dy=1.3)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "cellsize" in reason

    other_grid = RectGrid(dx=1.3, dy=1.2)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "cellsize" in reason

    other_grid = RectGrid(dx=1.2, dy=1.2, crs=4326)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "CRS" in reason

    grid.crs = 4326
    other_grid = RectGrid(dx=1.2, dy=1.2, crs=3857)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "CRS" in reason
    grid.crs = None  # reset crs for next tests

    other_grid = RectGrid(dx=1.2, dy=1.2, offset=(0, 1))
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "offset" in reason

    other_grid = RectGrid(dx=1.2, dy=1.2, offset=(1, 0))
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "offset" in reason

    other_grid = RectGrid(dx=1.2, dy=1.1, offset=(1, 1), crs=4326)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert all(attr in reason for attr in ["CRS", "cellsize", "offset"])

    with pytest.raises(TypeError):
        other_grid = 1
        is_aligned, reason = grid.is_aligned_with(other_grid)

    other_grid = HexGrid(size=1)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "Grid type is not the same" in reason


@pytest.mark.parametrize("rot", (0, 15.5, 30, -26.2))
def test_centering_with_offset(rot):
    grid = RectGrid(dx=3, dy=3, rotation=rot)
    grid.offset = (-grid.dx / 2, -grid.dy / 2)
    numpy.testing.assert_allclose(grid.centroid([-1, -1]), [0, 0])


@pytest.mark.parametrize(
    "rot,expected_rot_mat",
    (
        (0, [[1, 0], [0, 1]]),
        (15.5, [[0.96363045, -0.26723838], [0.26723838, 0.96363045]]),
        (30, [[0.8660254, -0.5], [0.5, 0.8660254]]),
        (-26.2, [[0.89725837, 0.44150585], [-0.44150585, 0.89725837]]),
    ),
)
def test_rotation_setter(rot, expected_rot_mat):
    grid = RectGrid(dx=1.23, dy=0.987)
    grid.rotation = rot
    numpy.testing.assert_allclose(rot, grid.rotation)
    numpy.testing.assert_allclose(grid.rotation_matrix, expected_rot_mat)


def test_update():
    grid = RectGrid(dx=1, dy=2)

    new_grid = grid.update(crs=4326)
    assert grid.crs is None
    assert new_grid.crs.to_epsg() == 4326

    new_grid = grid.update(dx=0.3)
    numpy.testing.assert_allclose(grid.dx, 1)
    numpy.testing.assert_allclose(new_grid.dx, 0.3)
    numpy.testing.assert_allclose(new_grid.dy, 2)

    new_grid = grid.update(dy=0.6)
    numpy.testing.assert_allclose(grid.dy, 2)
    numpy.testing.assert_allclose(new_grid.dy, 0.6)
    numpy.testing.assert_allclose(new_grid.dx, 1)

    new_grid = grid.update(offset=(0.2, 0.3))
    numpy.testing.assert_allclose(grid.offset, (0, 0))
    numpy.testing.assert_allclose(new_grid.offset, (0.2, 0.3))

    new_grid = grid.update(rotation=2.5)
    numpy.testing.assert_allclose(grid.rotation, 0)
    numpy.testing.assert_allclose(new_grid.rotation, 2.5)


def test_dx_dy_setter():
    grid = RectGrid(dx=1.23, dy=4.56, rotation=10)
    numpy.testing.assert_allclose(grid.dx, 1.23)
    numpy.testing.assert_allclose(grid.dy, 4.56)
    grid.dx = 3.21
    grid.dy = 6.54
    numpy.testing.assert_allclose(grid.dx, 3.21)
    numpy.testing.assert_allclose(grid.dy, 6.54)
    numpy.testing.assert_allclose(grid._grid.cell_width(), 3.21)
    numpy.testing.assert_allclose(grid._grid.cell_height(), 6.54)

    with pytest.raises(ValueError):
        grid.dx = 0
    with pytest.raises(ValueError):
        grid.dx = -1
    with pytest.raises(ValueError):
        grid.dy = 0
    with pytest.raises(ValueError):
        grid.dy = -1
