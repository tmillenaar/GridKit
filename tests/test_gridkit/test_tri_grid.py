import numpy
import pytest

from gridkit import RectGrid, TriGrid
from gridkit.index import GridIndex


@pytest.mark.parametrize(
    "size, expected_radius",
    [(0.5, 0.5773502691896257), (0.6, 0.6928203230275508), (1.55, 1.78978583448784)],
)
def test_radius(size, expected_radius):
    grid = TriGrid(size=size)
    numpy.testing.assert_allclose(grid.r, expected_radius)


@pytest.mark.parametrize(
    "indices, expected_centroids",
    [
        [(-1, -1), [-2.25, -3.46410162]],
        [(-1, 1), [-2.25, 1.73205081]],
        [
            [(0, 0), (1, -1), (1, 1)],
            [[-0.75, -0.8660254], [0.75, -3.46410162], [0.75, 1.73205081]],
        ],
    ],
)
@pytest.mark.parametrize("offset", ((0, 0), (-0.7, 0.3), (1, -0.2)))
def test_centroid(indices, offset, expected_centroids):
    # TODO: test for different shapes when implemented
    grid = TriGrid(size=1.5, offset=offset)
    centroids = grid.centroid(indices)
    centroids -= offset
    numpy.testing.assert_allclose(centroids, expected_centroids)


@pytest.mark.parametrize(
    "indices, expected_corners",
    [
        [(-1, -1), [[-2.25, -5.19615242], [-0.75, -2.59807621], [-3.75, -2.59807621]]],
        [
            [(0, 0), (1, -1), (1, 1)],
            [
                [[-0.75, -2.59807621], [0.75, 0.0], [-2.25, 0.0]],
                [[0.75, -5.19615242], [2.25, -2.59807621], [-0.75, -2.59807621]],
                [[0.75, 0.0], [2.25, 2.59807621], [-0.75, 2.59807621]],
            ],
        ],
    ],
)
@pytest.mark.parametrize("offset", ((0, 0), (-1.3, 0.3), (1, -2.2)))
def test_cell_corners(indices, offset, expected_corners):
    grid = TriGrid(size=1.5, offset=offset)
    corners = grid.cell_corners(indices)
    corners -= grid.offset
    expected_corners = numpy.array(expected_corners)

    numpy.testing.assert_allclose(corners, expected_corners, atol=1e-8)


@pytest.mark.parametrize(
    "points, expected_ids",
    (
        [
            (-2.2, 5.7),
            [-3, 5],
        ],
        [
            [(-0.3, -7.5), (3.6, -8.3)],
            [[0, -6], [5, -6]],
        ],
    ),
)
@pytest.mark.parametrize("offset", ((0, 0), (-0.7, 0.3), (1, -0.2)))
def test_cell_at_point(points, offset, expected_ids):
    points = numpy.array(points)
    grid = TriGrid(size=0.7, offset=offset)
    points += grid.offset
    ids = grid.cell_at_point(points)
    numpy.testing.assert_allclose(ids, expected_ids)


@pytest.mark.parametrize(
    "bounds, expected_ids, expected_shape",
    (
        [
            (-2.2, -3, 2.3, 2.1),
            [
                [-2, 2],
                [-1, 2],
                [0, 2],
                [1, 2],
                [2, 2],
                [3, 2],
                [-2, 1],
                [-1, 1],
                [0, 1],
                [1, 1],
                [2, 1],
                [3, 1],
                [-2, 0],
                [-1, 0],
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [-2, -1],
                [-1, -1],
                [0, -1],
                [1, -1],
                [2, -1],
                [3, -1],
            ],
            (4, 6),
        ],
        [
            (2.2, 3, 5.3, 6.1),
            [
                [4, 5],
                [5, 5],
                [6, 5],
                [7, 5],
                [8, 5],
                [4, 4],
                [5, 4],
                [6, 4],
                [7, 4],
                [8, 4],
                [4, 3],
                [5, 3],
                [6, 3],
                [7, 3],
                [8, 3],
            ],
            (3, 5),
        ],
        [
            (-5.3, -6.1, -2.2, -3),
            [
                [-7, -2],
                [-6, -2],
                [-5, -2],
                [-4, -2],
                [-3, -2],
                [-7, -3],
                [-6, -3],
                [-5, -3],
                [-4, -3],
                [-3, -3],
                [-7, -4],
                [-6, -4],
                [-5, -4],
                [-4, -4],
                [-3, -4],
            ],
            (3, 5),
        ],
    ),
)
@pytest.mark.parametrize("return_cell_count", [True, False])
def test_cells_in_bounds(bounds, expected_ids, expected_shape, return_cell_count):
    grid = TriGrid(size=0.7)
    bounds = grid.align_bounds(bounds, "nearest")
    result = grid.cells_in_bounds(bounds, return_cell_count=return_cell_count)
    expected_ids = numpy.array(expected_ids).reshape((*expected_shape, 2))
    if return_cell_count is False:
        numpy.testing.assert_allclose(result, expected_ids)
    else:
        ids, shape = result
        numpy.testing.assert_allclose(ids, expected_ids)
        numpy.testing.assert_allclose(shape, expected_shape)
        assert len(ids) == shape[0] * shape[1]


@pytest.mark.parametrize(
    "index",
    [
        [0, 0],  # test single cell
        [[-6, 3], [4, -1], [5, 4], [-5, -4]],  # up
        [[-5, 3], [3, -3], [4, 4], [-6, -6]],  # down
    ],
)
@pytest.mark.parametrize("depth", range(1, 7))
@pytest.mark.parametrize("include_selected", [True, False])
@pytest.mark.parametrize("connect_corners", [True, False])
def test_neighbours(index, depth, include_selected, connect_corners):
    grid = TriGrid(size=0.7)
    all_neighbours = grid.neighbours(
        index,
        depth=depth,
        include_selected=include_selected,
        connect_corners=connect_corners,
    )
    all_relative_neighbours = grid.relative_neighbours(
        index,
        depth=depth,
        include_selected=include_selected,
        connect_corners=connect_corners,
    )

    nr_neighbours_factor = 4 if connect_corners else 1
    expected_nr_neighbours = include_selected + sum(
        [nr_neighbours_factor * 3 * (i + 1) for i in range(depth)]
    )

    def verify_neighbours(index, neighbours, relative_neigbours):
        # Compare neighbours and relative_neighbours
        numpy.testing.assert_allclose(neighbours - index, relative_neigbours)

        # Make sure include_selected targets the center cell appropriately
        if include_selected:
            assert GridIndex([0, 0]) in relative_neigbours
        else:
            assert GridIndex([0, 0]) not in relative_neigbours

        # Make sure the number of neighbours is as expected
        assert len(neighbours) == expected_nr_neighbours
        # Make sure there are no duplicate ids
        assert len(relative_neigbours.unique()) == len(relative_neigbours)
        # Make sure all cells are at least within the expected radius
        # Best would be to check the actual indices, but the arrays get too large to comfortably put in a test definition
        expected_furthest_cell_distance = (1 + 2 * depth) if connect_corners else depth
        assert numpy.all(
            abs(relative_neigbours.index.sum(axis=1)) <= expected_furthest_cell_distance
        )

    if all_neighbours.index.ndim == 2:
        verify_neighbours(index, all_neighbours, all_relative_neighbours)
    else:
        for cell_index, cell_neigbours, cell_relative_neigbours in zip(
            index, all_neighbours.index, all_relative_neighbours.index
        ):
            # FIXME: looping over GridIndex should take n-dimensionality into account,
            #        currently looping over raveled index
            verify_neighbours(
                cell_index,
                GridIndex(cell_neigbours),
                GridIndex(cell_relative_neigbours),
            )


def test_to_crs():
    grid = TriGrid(size=1.5, crs=None)
    # Expect error when `to_crs` is called on a grid where the CRS is not set
    with pytest.raises(ValueError):
        grid.to_crs(4326)

    grid = TriGrid(size=1.5, crs=3857)
    grid_degrees = grid.to_crs(4326)

    # Check cell size has changed after changing crs
    numpy.testing.assert_allclose(
        [grid_degrees.dx, grid_degrees.dy],
        [2.42919136381939e-05, 4.2074828634427165e-05],
    )
    # Make sure original grid is unaffected
    numpy.testing.assert_allclose([grid.dx, grid.dy], [1.5, 2.598076211353316])
    # Make sure nothing changed when setting to current CRS
    new_grid = grid.to_crs(3857)
    numpy.testing.assert_allclose([new_grid.dx, new_grid.dy], [1.5, 2.598076211353316])


def test_is_aligned_with():
    grid = TriGrid(size=1.2)

    is_aligned, reason = grid.is_aligned_with(grid)
    assert is_aligned
    assert reason == ""

    other_grid = TriGrid(size=1.3)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "cellsize" in reason

    other_grid = TriGrid(size=1.2, crs=4326)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "CRS" in reason

    grid.crs = 4326
    other_grid = TriGrid(size=1.2, crs=3857)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "CRS" in reason
    grid.crs = None  # reset crs for next tests

    other_grid = TriGrid(size=1.2, offset=(0, 1))
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "offset" in reason

    other_grid = TriGrid(size=1.2, offset=(1, 0))
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "offset" in reason

    # TODO: other shapes not yet implemented
    # other_grid = TriGrid(size=1.2, shape="flat")
    # is_aligned, reason = grid.is_aligned_with(other_grid)
    # assert not is_aligned
    # assert "shape" in reason

    other_grid = TriGrid(size=1.1, offset=(1, 1), crs=4326)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert all(attr in reason for attr in ["CRS", "cellsize", "offset"])

    with pytest.raises(TypeError):
        other_grid = 1
        is_aligned, reason = grid.is_aligned_with(other_grid)

    other_grid = RectGrid(dx=1, dy=1)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "Grid type is not the same" in reason


def test_is_cell_upright():
    grid = TriGrid(size=1)

    cells = numpy.arange(6) - 3
    xx, yy = numpy.meshgrid(cells, cells)
    cells = numpy.stack([xx.ravel(), yy.ravel()]).T
    expected_results = [
        False,  # 1st row
        True,
        False,
        True,
        False,
        True,
        True,  # 2nd row
        False,
        True,
        False,
        True,
        False,
        False,  # 3rd row
        True,
        False,
        True,
        False,
        True,
        True,  # 4rth row
        False,
        True,
        False,
        True,
        False,
        False,  # 5th row
        True,
        False,
        True,
        False,
        True,
        True,  # 6th row
        False,
        True,
        False,
        True,
        False,
    ]
    # test single cell as input
    result = grid.is_cell_upright(cells[0])
    numpy.testing.assert_allclose(result, expected_results[0])
    # test mulit-cell input
    result = grid.is_cell_upright(cells)
    numpy.testing.assert_allclose(result, expected_results)


@pytest.mark.parametrize(
    "points, expected_nearby_ids",
    [
        # Quadrant 1 (++)
        (
            [
                [2.6, 1.5],
                [2.3, 1.75],
                [2, 1.5],
                [2, 1],
                [2.3, 1],
                [2.6, 1.1],
            ],
            [
                [3, 2],
                [4, 2],
                [5, 2],
                [3, 1],
                [4, 1],
                [5, 1],
            ],
        ),
        # Quadrant 2 (-+)
        (
            [
                [-2, 1.5],
                [-2.5, 1.75],
                [-2.8, 1.5],
                [-2.8, 1],
                [-2.5, 1],
                [-2, 1],
            ],
            [
                [-5, 2],
                [-4, 2],
                [-3, 2],
                [-5, 1],
                [-4, 1],
                [-3, 1],
            ],
        ),
        # Quadrant 3 (--)
        (
            [
                [-2.2, -0.5],
                [-2.5, -0.3],
                [-2.8, -0.5],
                [-2.8, -1],
                [-2.5, -1],
                [-2, -1],
            ],
            [
                [-5, 0],
                [-4, 0],
                [-3, 0],
                [-5, -1],
                [-4, -1],
                [-3, -1],
            ],
        ),
        # Quadrant 4 (+-)
        (
            [
                [2.6, -0.5],
                [2.3, -0.5],
                [2, -0.5],
                [2, -1],
                [2.3, -1],
                [2.6, -1],
            ],
            [
                [3, 0],
                [4, 0],
                [5, 0],
                [3, -1],
                [4, -1],
                [5, -1],
            ],
        ),
    ],
)
@pytest.mark.parametrize("expand_axes", [True, False])
def test_cells_near_point(points, expected_nearby_ids, expand_axes):
    grid = TriGrid(size=0.6, offset=(0.2, 0.3))
    nearby_ids = grid.cells_near_point(points[0])
    # Test single point input
    numpy.testing.assert_allclose(nearby_ids, expected_nearby_ids)
    # Test for all points
    expected_nearby_ids_extended = numpy.empty(shape=(len(points), 6, 2))
    expected_nearby_ids_extended[:] = expected_nearby_ids
    if expand_axes:
        points = numpy.repeat(numpy.array(points)[None], 3, axis=0)
        expected_nearby_ids_extended = numpy.repeat(
            numpy.array(expected_nearby_ids_extended)[None], 3, axis=0
        )
    nearby_ids = grid.cells_near_point(points)
    numpy.testing.assert_allclose(nearby_ids, expected_nearby_ids_extended)


@pytest.mark.parametrize("crs", [None, 4326, 3857])
def test_bounded_crop(basic_bounded_tri_grid, crs):
    grid = basic_bounded_tri_grid
    grid.crs = 4326
    if crs == 3857:
        bounds = (-55659.9, -334111.1714019596, 222639, 557305.2572745768)
    else:
        bounds = (-0.5, -3, 2, 5)
    result = grid.crop(bounds, bounds_crs=crs)
    expected_result = numpy.array([[4, 5], [7, 8], [10, 11]])
    numpy.testing.assert_allclose(result, expected_result)

    # test with buffer_cells
    dy = grid.to_crs(crs).dy if crs else grid.dy
    bounds = (bounds[0], bounds[1] + 2 * dy, bounds[2], bounds[3])
    result = grid.crop(bounds, bounds_crs=crs, buffer_cells=2)
    expected_result = numpy.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
    numpy.testing.assert_allclose(result, expected_result)


def test_bounded_cell_corners(basic_bounded_tri_grid):
    grid = basic_bounded_tri_grid
    numpy.testing.assert_allclose(
        grid.cell_corners(), grid.cell_corners(grid.cells_in_bounds(grid.bounds))
    )


@pytest.mark.parametrize("as_multipolygon", [False, True])
def test_bounded_to_shapely(basic_bounded_tri_grid, as_multipolygon):
    grid = basic_bounded_tri_grid
    geoms1 = grid.to_shapely(as_multipolygon=as_multipolygon)
    geoms2 = grid.to_shapely(
        grid.cells_in_bounds(grid.bounds), as_multipolygon=as_multipolygon
    )

    if as_multipolygon:
        geoms1 = geoms1.geoms
        geoms2 = geoms2.geoms
    else:
        geoms1 = geoms1.ravel()
        geoms2 = geoms2.ravel()

    for geom1, geom2 in zip(geoms1, geoms2):
        assert geom1.wkb == geom2.wkb


def test_to_bounded(basic_bounded_tri_grid):
    grid = TriGrid(size=1)
    bounds = basic_bounded_tri_grid.bounds
    bounded_grid = grid.to_bounded(bounds)
    numpy.testing.assert_allclose(bounded_grid.indices, basic_bounded_tri_grid.indices)


def test_numpy_and_grid_ids(basic_bounded_tri_grid):
    grid = basic_bounded_tri_grid

    xx, yy = numpy.meshgrid(
        numpy.arange(grid.height), numpy.arange(grid.width), indexing="ij"
    )
    expected_np_ids = numpy.stack([xx.ravel(), yy.ravel()])

    np_ids = grid.grid_id_to_numpy_id(grid.indices.ravel())
    numpy.testing.assert_allclose(np_ids, expected_np_ids)

    grid_ids = grid.numpy_id_to_grid_id(np_ids)
    numpy.testing.assert_allclose(grid_ids, grid.indices.ravel())


def test_to_crs(basic_bounded_tri_grid):
    grid = basic_bounded_tri_grid
    grid.crs = 4326
    grid_3857 = grid.to_crs(3857)

    assert grid_3857.crs.to_epsg() == 3857
    numpy.testing.assert_allclose(
        grid_3857.bounds,
        (
            -111319.49079327357,
            -385622.02785329137,
            222638.98158654713,
            578433.041779937,
        ),
    )

    # make sure the original grid is not modified
    assert grid.crs.to_epsg() == 4326
    numpy.testing.assert_allclose(
        grid.bounds, (-1.0, -3.4641016151377544, 2.0, 5.196152422706632)
    )

    numpy.testing.assert_allclose(grid.data, grid_3857.data)


def test_centroid(basic_bounded_tri_grid):
    grid = basic_bounded_tri_grid
    centroids1 = grid.centroid()
    centroids2 = grid.centroid(grid.cells_in_bounds(grid.bounds))
    numpy.testing.assert_allclose(centroids1, centroids2)


@pytest.mark.parametrize(
    "method, expected_result, expected_bounds",
    (
        (
            "nearest",
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
                [12, 13, 14],
            ],
            (-0.95, -3.2908965343808667, 1.9, 4.9363448015713),
        ),
        (
            "bilinear",
            [
                [0.55, 1.25625, 2.45],
                [-371.9, -79.07916667, -286.8],
                [6.1, 7.03125, 8.01875],
                [8.95, 9.91875, 10.83125],
                [-363.4625, -70.72916667, -278.2375],
            ],
            (-0.95, -3.2908965343808667, 1.9, 4.9363448015713),
        ),
        (
            "inverse_distance",
            [
                [1.00661862e00, 1.13920527e00, 2.61187704e00],
                [-1.91589670e03, -1.61174343e02, -1.72505596e03],
                [6.79363706e00, 7.07606798e00, 8.33899655e00],
                [8.63365569e00, 9.88625995e00, 1.01918447e01],
                [-1.90876942e03, -1.53685443e02, -1.71776924e03],
            ],
            (-0.95, -3.2908965343808667, 1.9, 4.9363448015713),
        ),
    ),
)
def test_resample(
    basic_bounded_tri_grid,
    method,
    expected_result,
    expected_bounds,
):
    grid = basic_bounded_tri_grid
    new_grid = TriGrid(size=0.95)

    resampled = grid.resample(new_grid, method=method)

    numpy.testing.assert_allclose(resampled.data, expected_result)
    numpy.testing.assert_allclose(resampled.bounds, expected_bounds)


@pytest.mark.parametrize("rot", (0, 15.5, 30, -26.2))
def test_centering_with_offset(rot):
    grid = TriGrid(size=3, rotation=rot)
    grid.offset = (grid.dx / 2, grid.dy - grid.r)
    numpy.testing.assert_allclose(grid.centroid([0, 0]), [0, 0])


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
    grid = TriGrid(size=1.23)
    grid.rotation = rot
    numpy.testing.assert_allclose(rot, grid.rotation)
    numpy.testing.assert_allclose(grid.rotation_matrix, expected_rot_mat)


def test_update():
    grid = TriGrid(size=1)

    new_grid = grid.update(crs=4326)
    assert grid.crs is None
    assert new_grid.crs.to_epsg() == 4326

    new_grid = grid.update(size=0.3)
    numpy.testing.assert_allclose(grid.size, 1)
    numpy.testing.assert_allclose(new_grid.size, 0.3)

    new_grid = grid.update(offset=(0.2, 0.3))
    numpy.testing.assert_allclose(grid.offset, (0, 0))
    numpy.testing.assert_allclose(new_grid.offset, (0.2, 0.3))

    new_grid = grid.update(rotation=2.5)
    numpy.testing.assert_allclose(grid.rotation, 0)
    numpy.testing.assert_allclose(new_grid.rotation, 2.5)


def test_size_setter():
    grid = TriGrid(size=1.23, rotation=10)
    numpy.testing.assert_allclose(grid.size, 1.23)
    grid.size = 3.21
    numpy.testing.assert_allclose(grid.size, 3.21)

    with pytest.raises(ValueError):
        grid.size = 0
    with pytest.raises(ValueError):
        grid.size = -1
