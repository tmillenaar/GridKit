import numpy
import pytest

from gridkit import BoundedTriGrid, RectGrid, TriGrid
from gridkit.index import GridIndex


@pytest.mark.parametrize(
    "size, expected_radius",
    [(0.5, 0.28867513459481287), (0.6, 0.3464101615137754), (1.55, 0.89489291724392)],
)
def test_radius(size, expected_radius):
    grid = TriGrid(size=size)
    numpy.testing.assert_allclose(grid.r, expected_radius)


def test_cell_size_init_raises():

    with pytest.raises(ValueError):
        TriGrid()

    with pytest.raises(ValueError):
        TriGrid(size=1, area=1)


@pytest.mark.parametrize("area", [0.1, 123, 987.6])
def test_init_area(area):
    grid = TriGrid(area=area)
    numpy.testing.assert_allclose(grid.area, area)


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
        [(-1, -1), [[0, 0], [0.75, -1.29903811], [-0.75, -1.29903811]]],
        [
            [(0, 0), (1, -1), (1, 1)],
            [
                [[0.75, 1.29903811], [1.5, 0.0], [0.0, 0.0]],
                [[1.5, 0.0], [2.25, -1.29903811], [0.75, -1.29903811]],
                [[1.5, 2.59807621], [2.25, 1.29903811], [0.75, 1.29903811]],
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
            [-4, 4],
        ],
        [
            [(-0.3, -7.5), (3.6, -8.3)],
            [[-2, -7], [4, -7]],
        ],
    ),
)
@pytest.mark.parametrize("offset", ((0, 0), (-0.7, 0.3), (1, -0.2)))
def test_cell_at_point(points, offset, expected_ids):
    points = numpy.array(points)
    grid = TriGrid(size=1.4, offset=offset)
    points += grid.offset
    ids = grid.cell_at_point(points)
    numpy.testing.assert_allclose(ids, expected_ids)


def test_cell_at_point_v2():
    grid = TriGrid(size=0.3, offset=(0.25, 0.15))

    b = (-2, -2, 2, 2)
    px = numpy.linspace(b[0], b[2], 33)
    py = numpy.linspace(b[1], b[3], 100)
    xx, yy = numpy.meshgrid(px, py)
    points = numpy.stack([xx.ravel(), yy.ravel()]).T

    ids = grid.cell_at_point(points)
    import shapely
    import shapely.geometry

    correct = []
    geoms = grid.to_shapely(ids)
    for id, point, geom in zip(ids, points, geoms):
        assert geom.contains(
            shapely.geometry.Point(point)
        ), f"Shape does not contain point for point:{point} and id:{id}"


@pytest.mark.parametrize(
    "bounds, expected_ids, expected_shape",
    (
        [
            (-2.2, -3, 2.3, 2.1),
            [
                [-4, 1],
                [-3, 1],
                [-2, 1],
                [-1, 1],
                [0, 1],
                [1, 1],
                [-4, 0],
                [-3, 0],
                [-2, 0],
                [-1, 0],
                [0, 0],
                [1, 0],
                [-4, -1],
                [-3, -1],
                [-2, -1],
                [-1, -1],
                [0, -1],
                [1, -1],
                [-4, -2],
                [-3, -2],
                [-2, -2],
                [-1, -2],
                [0, -2],
                [1, -2],
            ],
            (4, 6),
        ],
        [
            (2.2, 3, 5.3, 6.1),
            [
                [2, 4],
                [3, 4],
                [4, 4],
                [5, 4],
                [6, 4],
                [2, 3],
                [3, 3],
                [4, 3],
                [5, 3],
                [6, 3],
                [2, 2],
                [3, 2],
                [4, 2],
                [5, 2],
                [6, 2],
            ],
            (3, 5),
        ],
        [
            (-5.3, -6.1, -2.2, -3),
            [
                [-9, -3],
                [-8, -3],
                [-7, -3],
                [-6, -3],
                [-5, -3],
                [-9, -4],
                [-8, -4],
                [-7, -4],
                [-6, -4],
                [-5, -4],
                [-9, -5],
                [-8, -5],
                [-7, -5],
                [-6, -5],
                [-5, -5],
            ],
            (3, 5),
        ],
    ),
)
@pytest.mark.parametrize("return_cell_count", [True, False])
def test_cells_in_bounds(bounds, expected_ids, expected_shape, return_cell_count):
    grid = TriGrid(size=1.4)
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


@pytest.mark.parametrize("adjust_rotation", [False, True])
def test_to_crs(adjust_rotation):
    grid = TriGrid(size=1.5, crs=None)
    # Expect error when `to_crs` is called on a grid where the CRS is not set
    with pytest.raises(ValueError):
        grid.to_crs(4326)

    grid = TriGrid(size=1.5, crs=3857)
    grid_degrees = grid.to_crs(4326, adjust_rotation=adjust_rotation)

    # Check cell size has changed after changing crs
    numpy.testing.assert_allclose(
        [grid_degrees.dx, grid_degrees.dy],
        [6.737364630896411e-06, 1.1669457849830117e-05],
    )
    # Make sure original grid is unaffected
    numpy.testing.assert_allclose([grid.dx, grid.dy], [1.5 / 2, 2.598076211353316 / 2])
    # Make sure nothing changed when setting to current CRS
    new_grid = grid.to_crs(3857, adjust_rotation=adjust_rotation)
    numpy.testing.assert_allclose(
        [new_grid.dx, new_grid.dy], [1.5 / 2, 2.598076211353316 / 2]
    )


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
        True,  # 1st row
        False,
        True,
        False,
        True,
        False,
        False,  # 2nd row
        True,
        False,
        True,
        False,
        True,
        True,  # 3rd row
        False,
        True,
        False,
        True,
        False,
        False,  # 4th row
        True,
        False,
        True,
        False,
        True,
        True,  # 5th row
        False,
        True,
        False,
        True,
        False,
        False,  # 6th row
        True,
        False,
        True,
        False,
        True,
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
                [2.1, 3.1],
                [2.3, 3.1],
                [2.5, 3.1],
                [2.1, 2.7],
                [2.3, 2.7],
                [2.5, 2.7],
            ],
            [[5, 5], [6, 5], [7, 5], [5, 4], [6, 4], [7, 4]],
        ),
        # Quadrant 2 (-+)
        (
            [
                [2.4, -2.8],
                [2.6, -2.8],
                [2.8, -2.8],
                [2.4, -3],
                [2.6, -3],
                [2.8, -3],
            ],
            [[6, -6], [7, -6], [8, -6], [6, -7], [7, -7], [8, -7]],
        ),
        # Quadrant 3 (--)
        (
            [
                [-2.4, -2.8],
                [-2.2, -2.8],
                [-2.0, -2.8],
                [-2.4, -3],
                [-2.2, -3],
                [-2.0, -3],
            ],
            [
                [-10, -6],
                [-9, -6],
                [-8, -6],
                [-10, -7],
                [-9, -7],
                [-8, -7],
            ],
        ),
        # Quadrant 4 (+-)
        (
            [
                [2.4, -0.6],
                [2.6, -0.6],
                [2.8, -0.6],
                [2.4, -0.8],
                [2.6, -0.8],
                [2.8, -0.8],
            ],
            [[6, -2], [7, -2], [8, -2], [6, -3], [7, -3], [8, -3]],
        ),
    ],
)
# @pytest.mark.parametrize("expand_axes", [True, False])
@pytest.mark.parametrize("expand_axes", [False])
@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=0.6, offset=(0.2, 0.3), orientation="flat"),
        TriGrid(size=0.6, offset=(0.3, 0.2), orientation="pointy"),
    ],
)
def test_cells_near_point(grid, points, expected_nearby_ids, expand_axes):

    points = numpy.array(points)
    expected_nearby_ids = numpy.array(expected_nearby_ids)
    if grid.orientation == "pointy":
        points = points[:, ::-1]
        expected_nearby_ids = expected_nearby_ids[:, ::-1]

    nearby_ids = grid.cells_near_point(points[0])

    # Test single point input
    # Note: we use GridIndex.intersection to figure out if the arrays match,
    if grid.orientation == "Pointy":
        numpy.testing.assert_allclose(nearby_ids, expected_nearby_ids)
    else:
        # because the order of the cells changes for the Flat orientation
        assert len(GridIndex(expected_nearby_ids).intersection(nearby_ids)) == len(
            expected_nearby_ids
        )
    # Test for all points
    expected_nearby_ids_extended = numpy.empty(shape=(len(points), 6, 2))
    expected_nearby_ids_extended[:] = expected_nearby_ids
    if expand_axes:
        points = numpy.repeat(numpy.array(points)[None], 3, axis=0)
        expected_nearby_ids_extended = numpy.repeat(
            numpy.array(expected_nearby_ids_extended)[None], 3, axis=0
        )
    nearby_ids = grid.cells_near_point(points)
    if grid.orientation == "Pointy":
        numpy.testing.assert_allclose(nearby_ids, expected_nearby_ids_extended)
    else:
        # because the order of the cells changes for the Flat orientation
        assert len(
            GridIndex(expected_nearby_ids_extended).intersection(nearby_ids)
        ) == len(expected_nearby_ids)


@pytest.mark.parametrize("crs", [None, 4326, 3857])
def test_bounded_crop(basic_bounded_tri_grid, crs):
    grid = basic_bounded_tri_grid
    grid.crs = 4326
    if crs == 3857:
        bounds = (0.0, -111325.1428663851, 222638.98158654713, 222684.20850554405)
    else:
        bounds = (0, -1, 2, 2)
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


def test_to_crs_bounded(basic_bounded_tri_grid):
    grid = basic_bounded_tri_grid
    grid.crs = 4326
    grid_3857 = grid.to_crs(3857)

    assert grid_3857.crs.to_epsg() == 3857
    numpy.testing.assert_allclose(
        grid_3857.bounds,
        (
            -55659.745397,
            -192811.013927,
            111319.490793,
            289216.52089,
        ),
    )

    # make sure the original grid is not modified
    assert grid.crs.to_epsg() == 4326
    numpy.testing.assert_allclose(
        grid.bounds, (-0.5, -1.7320508075688772, 1.0, 2.598076211353316)
    )

    numpy.testing.assert_allclose(grid.data, grid_3857.data)


def test_centroid(basic_bounded_tri_grid):
    grid = basic_bounded_tri_grid
    centroids1 = grid.centroid()
    centroids2 = grid.centroid(grid.cells_in_bounds(grid.bounds))
    numpy.testing.assert_allclose(centroids1, centroids2)


@pytest.mark.parametrize(
    "method, expected_result",
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
        ),
        (
            "bilinear",
            [
                [0.575, 1.3, 2.475],
                [-330.21666667, 4.3, -328.45],
                [6.125, 7.075, 8.025],
                [8.975, 9.925, 10.875],
                [-321.71666667, 12.7, -319.95],
            ],
        ),
        (
            "inverse_distance",
            [
                [1.98492073e00, 1.98775171e00, 2.59223915e00],
                [-3.11530678e03, -1.84398639e03, -3.11458621e03],
                [7.89146730e00, 7.90130833e00, 8.49507949e00],
                [8.50492051e00, 9.09869167e00, 9.10853270e00],
                [-3.11033141e03, -1.83865101e03, -3.10961083e03],
            ],
        ),
    ),
)
def test_resample(
    basic_bounded_tri_grid,
    method,
    expected_result,
):
    grid = basic_bounded_tri_grid
    new_grid = TriGrid(size=0.95)

    resampled = grid.resample(new_grid, method=method)

    expected_bounds = (-0.475, -1.6454482671904334, 0.95, 2.46817240078565)

    numpy.testing.assert_allclose(resampled.data, expected_result)
    numpy.testing.assert_allclose(resampled.bounds, expected_bounds)


@pytest.mark.parametrize("rot", (0, 15.5, 30, -26.2))
def test_centering_with_offset(rot):
    grid = TriGrid(size=3, rotation=rot)
    grid.offset = (grid.dx, grid.dy - grid.r)
    numpy.testing.assert_allclose(grid.centroid([-2, -1]), [0, 0])


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

    new_grid = grid.update(area=4)
    numpy.testing.assert_allclose(grid.area, 3**0.5 / 4)
    numpy.testing.assert_allclose(new_grid.area, 4)


def test_size_setter():
    grid = TriGrid(size=1.23, rotation=10)
    numpy.testing.assert_allclose(grid.size, 1.23)
    grid.size = 3.21
    numpy.testing.assert_allclose(grid.size, 3.21)

    with pytest.raises(ValueError):
        grid.size = 0
    with pytest.raises(ValueError):
        grid.size = -1


def test_area_setter():
    grid = TriGrid(size=1, rotation=10)
    numpy.testing.assert_allclose(grid.area, 3**0.5 / 4)
    grid.area = 3.21
    numpy.testing.assert_allclose(grid.area, 3.21)

    with pytest.raises(ValueError):
        grid.area = 0
    with pytest.raises(ValueError):
        grid.area = -1


@pytest.mark.parametrize("size", [0.1, 2.3, 4, 1234])
def test_area(size):
    grid = TriGrid(size=size)
    geom = grid.to_shapely((0, 0))
    numpy.testing.assert_allclose(grid.area, geom.area)


@pytest.mark.parametrize("in_place", [True, False])
@pytest.mark.parametrize(
    "target_loc",
    [
        [0, 0],
        [-2.9, -2.9],
        [-3.0, -3.0],
        [-3.1, -3.1],
    ],
)
@pytest.mark.parametrize("starting_offset", [[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.2]])
@pytest.mark.parametrize("rot", [0, 15, -69, 420])
@pytest.mark.parametrize("cell_element", ["centroid", "corner"])
@pytest.mark.parametrize("orientation", ["pointy", "flat"])
def test_anchor(target_loc, in_place, starting_offset, rot, cell_element, orientation):
    grid = TriGrid(
        size=0.3, offset=starting_offset, rotation=rot, orientation=orientation
    )

    if in_place:
        grid.anchor(target_loc, cell_element=cell_element, in_place=True)
        new_grid = grid
    else:
        new_grid = grid.anchor(target_loc, cell_element=cell_element, in_place=False)

    if cell_element == "centroid":
        numpy.testing.assert_allclose(
            new_grid.centroid(new_grid.cell_at_point(target_loc)),
            target_loc,
            atol=1e-15,
        )
    elif cell_element == "corner":
        corners = new_grid.cell_corners(new_grid.cell_at_point(target_loc))
        distances = numpy.linalg.norm(corners - target_loc, axis=1)
        assert numpy.any(numpy.isclose(distances, 0))

    if in_place:
        # verify the original grid has a new offset
        numpy.testing.assert_allclose(grid.offset, new_grid.offset)
    else:
        # verify the original grid remains unchanged
        numpy.testing.assert_allclose(grid.offset, starting_offset)


@pytest.mark.parametrize(
    "cell_element, expected_bounds, expected_data",
    [
        (
            "centroid",
            (-0.4, -0.9433756729740643, 0.6, 2.52072594216369),
            numpy.array([[0, 1], [3, 4], [6, 7], [9, 10]]),
        ),
        (
            "corner",
            (-0.4, -1.2320508075688772, 0.6, 2.232050807568877),
            numpy.array([[0, 5], [7, 4], [6, 11], [13, 10]]),
        ),
    ],
)
def test_anchor_bounded(
    basic_bounded_tri_grid, cell_element, expected_bounds, expected_data
):
    new_grid = basic_bounded_tri_grid.anchor(
        [0.1, 0.5], cell_element=cell_element, resample_method="nearest"
    )
    numpy.testing.assert_allclose(new_grid.data, expected_data)
    numpy.testing.assert_allclose(new_grid.bounds, expected_bounds)


@pytest.mark.parametrize("factor", [2, 9])
@pytest.mark.parametrize("rotation", [-23.0, 0, 456])
@pytest.mark.parametrize("offset", [(-2, 3), (0, 0), (0.1, -0.2)])
@pytest.mark.parametrize("crs", [None, 4326])
def test_subdivide(factor, rotation, offset, crs):
    grid = TriGrid(size=1, rotation=rotation, offset=offset, crs=crs)
    subgrid = grid.subdivide(factor)

    # Test for new gridsize
    numpy.testing.assert_allclose(grid.dx / factor, subgrid.dx)

    # Test for overlapping corners
    corner = grid.cell_corners([-4, 23])[-1]
    sub_corners = subgrid.cell_corners(subgrid.cell_at_point(corner))
    assert numpy.any(numpy.isclose(sub_corners, corner))

    # Test for nr of cells in parent cell
    target_cell = GridIndex([3, -2])
    target = grid.centroid(target_cell)

    # subcells_near_cell = subgrid.intersect_geometries(cell_geom)
    subcells_near_cell = subgrid.neighbours(
        subgrid.cell_at_point(target),
        connect_corners=True,
        depth=factor + 1,
        include_selected=True,
    )
    centroids = subgrid.centroid(subcells_near_cell)
    subcells_in_cell = grid.cell_at_point(centroids)
    mask = subcells_in_cell.index_1d == target_cell.index_1d
    nr_subcells_in_cell = sum(mask)
    expected_nr_subcells_in_cell = factor**2
    assert nr_subcells_in_cell == expected_nr_subcells_in_cell

    if grid.crs is None:
        assert subgrid.crs is None
    else:
        assert grid.crs.is_exact_same(subgrid.crs)


@pytest.mark.parametrize("side_length", [0.1, 123, 987.6])
def test_init_side_length(side_length):
    grid = TriGrid(side_length=side_length)
    numpy.testing.assert_allclose(grid.side_length, side_length)
    geom = grid.to_shapely([0, 0])
    numpy.testing.assert_allclose(grid.side_length, geom.exterior.length / 3)


def test_init_multiple_sizes_error():
    with pytest.raises(ValueError):
        grid = TriGrid(size=1, area=1)

    with pytest.raises(ValueError):
        grid = TriGrid(size=1, side_length=1)

    with pytest.raises(ValueError):
        grid = TriGrid(area=1, side_length=1)

    with pytest.raises(ValueError):
        grid = TriGrid(size=1, area=1, side_length=1)

    with pytest.raises(ValueError):
        grid = TriGrid()


@pytest.mark.parametrize("shape", [(3, 2), (3, 4), (5, 5)])
def test_auto_bound_init(shape):
    data = numpy.ones(shape)
    grid = BoundedTriGrid(data)

    numpy.testing.assert_allclose(grid.bounds[0], 0)
    numpy.testing.assert_allclose(grid.bounds[1], 0)
    numpy.testing.assert_allclose(grid.height, shape[0])
    numpy.testing.assert_allclose(grid.width, shape[1])
    numpy.testing.assert_allclose(grid.bounds[2] / grid.dx, shape[1])
    numpy.testing.assert_allclose(grid.bounds[3] / grid.dy, shape[0])
