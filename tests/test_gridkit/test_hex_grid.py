import numpy
import pytest
import shapely

from gridkit import RectGrid
from gridkit.hex_grid import BoundedHexGrid, HexGrid


@pytest.mark.parametrize(
    "shape, indices, expected_centroids",
    [
        ["pointy", (-1, -1), [0, -1.29903811]],
        ["pointy", (-1, 1), [0, 3.897114]],
        [
            "pointy",
            [(0, 0), (1, -1), (1, 1)],
            [
                [1.5, 1.29903811],
                [6, -1.29903811],
                [6, 3.89711432],
            ],
        ],
    ],
)
def test_centroid(shape, indices, expected_centroids):
    grid = HexGrid(size=3)
    centroids = grid.centroid(indices)
    numpy.testing.assert_allclose(centroids, expected_centroids)


@pytest.mark.parametrize(
    "shape, bounds, expected_ids, expected_shape",
    [
        ["pointy", (0, -2, 2, 2), [[0, 0], [-1, -1]], (2, 1)],
        ["pointy", (0, -2, 3, 2), [[0, 0], [-1, -1]], (2, 1)],
        ["pointy", (0, 0, 3, 5), [[-1, 1], [0, 0]], (2, 1)],
        [
            "pointy",
            (-2, -2, 2, 2),
            [
                [[-1, 0], [0, 0]],
                [[-2, -1], [-1, -1]],
            ],
            (2, 2),
        ],
        ["pointy", (-2, -2, 3, 2), [[[-1, 0], [0, 0]], [[-2, -1], [-1, -1]]], (2, 2)],
        ["pointy", (-2, 0, 2, 5), [[[-2, 1], [-1, 1]], [[-1, 0], [0, 0]]], (2, 2)],
        ["pointy", (-2, 0, 3, 5), [[[-2, 1], [-1, 1]], [[-1, 0], [0, 0]]], (2, 2)],
        ["flat", (-2, -2, 2, 2), [[[-1, -1], [-1, -2]], [[0, 0], [0, -1]]], (2, 2)],
        ["flat", (-2, -2, 2, 3), [[[-1, -1], [-1, -2]], [[0, 0], [0, -1]]], (2, 2)],
        ["flat", (0, -2, 5, 2), [[[0, 0], [0, -1]], [[1, -1], [1, -2]]], (2, 2)],
        ["flat", (0, -2, 5, 3), [[[0, 0], [0, -1]], [[1, -1], [1, -2]]], (2, 2)],
        ["flat", (-2, 0, 2, 2), [[-1, -1], [0, 0]], (2, 1)],
        ["flat", (-2, 0, 2, 3), [[-1, -1], [0, 0]], (2, 1)],
        ["flat", (-2, 2, 2, 5), [[-1, 0], [0, 1]], (2, 1)],
        ["flat", (0, 0, 5, 3), [[0, 0], [1, -1]], (2, 1)],
    ],
)
def test_cells_in_bounds(shape, bounds, expected_ids, expected_shape):
    grid = HexGrid(size=3, shape=shape)
    aligned_bounds = grid.align_bounds(bounds, mode="nearest")
    ids = grid.cells_in_bounds(aligned_bounds)
    numpy.testing.assert_allclose(ids, expected_ids)

    ids, shape = grid.cells_in_bounds(aligned_bounds, return_cell_count=True)
    numpy.testing.assert_allclose(ids, expected_ids)
    numpy.testing.assert_allclose(shape, expected_shape)


@pytest.mark.parametrize(
    "shape, point, expected_nearby_cells",
    [
        ["pointy", (2.5, 1.5), [(0, 0), (0, 1), (1, 0)]],
        ["pointy", (1.5, 2.16), [(0, 0), (-1, 1), (0, 1)]],
        ["pointy", (0.5, 1.5), [(0, 0), (-1, 0), (-1, 1)]],
        ["pointy", (0.5, 1.0), [(0, 0), (-1, -1), (-1, 0)]],
        ["pointy", (1.5, 0.43), [(0, 0), (0, -1), (-1, -1)]],
        ["pointy", (2.5, 1.0), [(0, 0), (1, 0), (0, -1)]],
        [
            "pointy",
            [(4, 4.1), (3, 4.76), (2, 4.1), (2, 3.6), (3, 3.03), (4, 3.6)],
            [
                [[0, 1], [1, 2], [1, 1]],
                [[0, 1], [0, 2], [1, 2]],
                [[0, 1], [-1, 1], [0, 2]],
                [[0, 1], [0, 0], [-1, 1]],
                [[0, 1], [1, 0], [0, 0]],
                [[0, 1], [1, 1], [1, 0]],
            ],
        ],
        ["flat", (2.5, 1.5), [(0, 0), (1, -1), (1, 0)]],
        ["flat", (2.0, 2.5), [(0, 0), (1, 0), (0, 1)]],
        ["flat", (1.0, 2.5), [(0, 0), (0, 1), (-1, 0)]],
        ["flat", (0.0, 1.5), [(0, 0), (-1, 0), (-1, -1)]],
        ["flat", (1.0, 0.5), [(0, 0), (-1, -1), (0, -1)]],
        ["flat", (2.0, 0.5), [(0, 0), (0, -1), (1, -1)]],
        [
            "flat",
            [(5, 3), (4.5, 4), (3.5, 4), (2.5, 3), (3.5, 2), (4.5, 2)],
            [
                [[1, 0], [2, 0], [2, 1]],
                [[1, 0], [2, 1], [1, 1]],
                [[1, 0], [1, 1], [0, 1]],
                [[1, 0], [0, 1], [0, 0]],
                [[1, 0], [0, 0], [1, -1]],
                [[1, 0], [1, -1], [2, 0]],
            ],
        ],
    ],
)
@pytest.mark.parametrize("expand_axes", [True, False])
def test_cells_near_point(shape, point, expected_nearby_cells, expand_axes):
    grid = HexGrid(size=3, shape=shape)
    if expand_axes:
        point = numpy.repeat(numpy.array(point)[None], 3, axis=0)
        expected_nearby_cells = numpy.repeat(
            numpy.array(expected_nearby_cells)[None], 3, axis=0
        )
    nearby_cells = grid.cells_near_point(point)
    numpy.testing.assert_allclose(nearby_cells, expected_nearby_cells)


@pytest.mark.parametrize("shape", ["pointy", "flat"])
def test_crs(shape):
    offset = (5, 10)
    crs = 3857
    grid = HexGrid(size=10, offset=offset, crs=crs, shape=shape)
    new_grid = grid.to_crs(crs=4326)

    expected_size = 8.983152841195213e-05

    assert new_grid.crs.to_epsg() == 4326
    numpy.testing.assert_allclose(new_grid.size, expected_size)


@pytest.mark.parametrize("shape", ["pointy", "flat"])
@pytest.mark.parametrize("depth", list(range(1, 7)))
@pytest.mark.parametrize("index", [[2, 1], [1, 2]])
@pytest.mark.parametrize("multi_index", [False, True])
@pytest.mark.parametrize("include_selected", [False, True])
def test_neighbours(shape, depth, index, multi_index, include_selected):
    grid = HexGrid(size=3, shape=shape)

    if multi_index:
        index = [index, index]

    neighbours = grid.neighbours(index, depth=depth, include_selected=include_selected)

    if multi_index:
        numpy.testing.assert_allclose(
            neighbours.index[0], neighbours.index[1]
        )  # make sure the neighbours are the same (since index was duplicated)
        neighbours = neighbours[0]  # continue to check single index
        index = index[0]

    if include_selected:
        # make sure the index is present and at the center of the neighbours array
        center_index = int(numpy.floor(len(neighbours) / 2))
        numpy.testing.assert_allclose(neighbours[center_index], index)
        # remove center index for further testing of other neighbours
        neighbours.index = numpy.delete(neighbours.index, center_index, axis=0)

    # If the neighbours are correct, there are always a multiple of 6 cells with the same distance to the center cell
    distances = numpy.linalg.norm(
        grid.centroid(neighbours) - grid.centroid(index), axis=1
    )
    for d in numpy.unique(distances):
        assert sum(distances == d) % 6 == 0

    # No cell can be further away than 'depth' number of cells * cell size
    assert all(distances <= grid.size * depth)


@pytest.mark.parametrize(
    "shape, method, expected_result, expected_bounds",
    (
        (
            "pointy",
            "nearest",
            [[2, 1], [2, 3], [4, 5], [6, 7]],
            (-1.0, -1.7320508075688776, 1.0, 1.7320508075688776),
        ),
        (
            "pointy",
            "bilinear",
            [
                [1.08333333, 1.75],
                [2.58333333, 3.25],
                [3.91666667, 4.58333333],
                [
                    5.41666667,
                    -827.66666667,
                ],  # FIXME: outlier value artifact of nodata value of -9999. Nodata should not be taken into account
            ],
            (-1.0, -1.7320508075688776, 1.0, 1.7320508075688776),
        ),
        (
            "flat",
            "nearest",
            [[0, 3], [0, 0], [1, 4], [1, 5]],
            (-1.0, -0.8660254037844388, 1.0, 2.5980762113533165),
        ),
        (
            "flat",
            "bilinear",
            [
                [-3.85752650e03, -6.60020632e00],
                [-2.69773379e03, 1.88397460e00],
                [1.11417424e00, 3.80847549e00],
                [-2.69684981e03, 3.03867513e00],
            ],
            (-1.0, -0.8660254037844388, 1.0, 2.5980762113533165),
        ),
    ),
)
def test_resample(
    basic_bounded_flat_grid,
    basic_bounded_pointy_grid,
    shape,
    method,
    expected_result,
    expected_bounds,
):
    new_grid = HexGrid(size=1)
    grid = basic_bounded_flat_grid if shape == "flat" else basic_bounded_pointy_grid

    resampled = grid.resample(new_grid, method=method)

    numpy.testing.assert_allclose(resampled.data, expected_result)
    numpy.testing.assert_allclose(resampled.bounds, expected_bounds)


@pytest.mark.parametrize("shape", ["pointy", "flat"])
def test_to_bounded(shape):
    grid = HexGrid(size=1.5, shape=shape)
    grid.crs = 4326
    bounds = (1, 1, 5, 5)
    bounds = grid.align_bounds(bounds)
    result = grid.to_bounded(bounds)

    assert isinstance(result, BoundedHexGrid)
    assert result.data.shape == (4, 4)
    assert grid.size == result.size
    assert grid.crs.is_exact_same(result.crs)
    assert bounds == result.bounds
    assert grid.shape == result.shape
    assert numpy.all(numpy.isnan(result.data))


@pytest.mark.parametrize("shape", ["flat", "pointy"])
def test_to_shapely_1_cell(shape):
    grid = HexGrid(size=1.5, shape=shape)
    id_ = [-3, 2]
    geom = grid.to_shapely(id_)

    centroid = [geom.centroid.x, geom.centroid.y]
    numpy.testing.assert_allclose(centroid, grid.centroid(id_))

    numpy.testing.assert_allclose(geom.area, grid.dx * grid.dy)
    # Is is not beautiful how dx*dy also gives the area of the hexagon?
    # The bits outside the rectangle perfectly compensate for the missing bits inside the rectangle


@pytest.mark.parametrize("as_mp", [True, False])
@pytest.mark.parametrize("shape", ["flat", "pointy"])
def test_to_shapely(as_mp, shape):
    grid = HexGrid(size=1.5, shape=shape)
    ids = [[-3, 2], [3, -6]]
    geoms = grid.to_shapely(ids, as_multipolygon=as_mp)

    if as_mp:
        assert isinstance(geoms, shapely.geometry.MultiPolygon)
        geoms = geoms.geoms

    centroids = [[geom.centroid.x, geom.centroid.y] for geom in geoms]
    numpy.testing.assert_allclose(centroids, grid.centroid(ids))

    for geom in geoms:
        numpy.testing.assert_allclose(geom.area, grid.dx * grid.dy)
        # Is is not beautiful how dx*dy also gives the area of the hexagon?
        # The bits outside the rectangle perfectly compensate for the missing bits inside the rectangle


@pytest.mark.parametrize(
    "shape, expected_corners",
    [
        [
            "flat",
            numpy.array(
                [
                    [-3.68060797, 5.25],
                    [-2.81458256, 5.25],
                    [-2.38156986, 4.5],
                    [-2.81458256, 3.75],
                    [-3.68060797, 3.75],
                    [-4.11362067, 4.5],
                ]
            ),
        ],
        [
            "pointy",
            numpy.array(
                [
                    [-3.0, 2.81458256],
                    [-3.0, 3.68060797],
                    [-3.75, 4.11362067],
                    [-4.5, 3.68060797],
                    [-4.5, 2.81458256],
                    [-3.75, 2.38156986],
                ]
            ),
        ],
    ],
)
def test_cell_corners(shape, expected_corners):
    grid = HexGrid(size=1.5, shape=shape)
    ids = [[-3, 2], [-3, 2]]

    corners = grid.cell_corners(ids)
    for cell_corners in corners:
        numpy.testing.assert_allclose(cell_corners, expected_corners)


def test_is_aligned_with():
    grid = HexGrid(size=1.2, shape="pointy")

    is_aligned, reason = grid.is_aligned_with(grid)
    assert is_aligned
    assert reason == ""

    other_grid = HexGrid(size=1.3)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "cellsize" in reason

    other_grid = HexGrid(size=1.2, crs=4326)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "CRS" in reason

    grid.crs = 4326
    other_grid = HexGrid(size=1.2, crs=3857)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "CRS" in reason
    grid.crs = None  # reset crs for next tests

    other_grid = HexGrid(size=1.2, offset=(0, 1))
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "offset" in reason

    other_grid = HexGrid(size=1.2, offset=(1, 0))
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "offset" in reason

    other_grid = HexGrid(size=1.2, shape="flat")
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "shape" in reason

    other_grid = HexGrid(size=1.1, shape="flat", offset=(1, 1), crs=4326)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert all(attr in reason for attr in ["CRS", "shape", "cellsize", "offset"])

    with pytest.raises(TypeError):
        other_grid = 1
        is_aligned, reason = grid.is_aligned_with(other_grid)

    other_grid = RectGrid(dx=1, dy=1)
    is_aligned, reason = grid.is_aligned_with(other_grid)
    assert not is_aligned
    assert "Grid type is not the same" in reason


@pytest.mark.parametrize("rot", (0, 15.5, 30, -26.2))
@pytest.mark.parametrize("shape", ["pointy", "flat"])
def test_centering_with_offset(shape, rot):
    grid = HexGrid(size=3, shape=shape, rotation=rot)
    if shape == "pointy":
        grid.offset = (0, grid.dy / 2)
    else:
        grid.offset = (grid.dx / 2, 0)
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
@pytest.mark.parametrize("shape", ["pointy", "flat"])
def test_rotation_setter(rot, expected_rot_mat, shape):
    grid = HexGrid(size=1.23, shape=shape)
    grid.rotation = rot
    numpy.testing.assert_allclose(rot, grid.rotation)
    if shape == "flat" and rot != 0:
        expected_rot_mat = numpy.array(expected_rot_mat).T
    numpy.testing.assert_allclose(grid.rotation_matrix, expected_rot_mat)


def test_shape_setter():
    grid = HexGrid(size=1.23, shape="pointy", rotation=10)
    expected_rot_mat = numpy.array(
        [[0.98480775, -0.17364818], [0.17364818, 0.98480775]]
    )
    assert grid.shape == "pointy"
    numpy.testing.assert_allclose(grid.rotation, 10)
    numpy.testing.assert_allclose(grid.rotation_matrix, expected_rot_mat)
    grid.shape = "flat"
    assert grid.shape == "flat"
    numpy.testing.assert_allclose(grid.rotation, 10)
    numpy.testing.assert_allclose(grid.rotation_matrix, expected_rot_mat.T)
    grid.shape = "pointy"
    assert grid.shape == "pointy"
    numpy.testing.assert_allclose(grid.rotation, 10)
    numpy.testing.assert_allclose(grid.rotation_matrix, expected_rot_mat)

    with pytest.raises(ValueError):
        grid.shape = "foo"


def test_size_setter():
    grid = HexGrid(size=1.23, shape="pointy", rotation=10)
    numpy.testing.assert_allclose(grid.size, 1.23)
    grid.size = 3.21
    numpy.testing.assert_allclose(grid.size, 3.21)

    grid = HexGrid(size=1.23, shape="flat", rotation=0, crs=4326)
    numpy.testing.assert_allclose(grid.size, 1.23)
    grid.size = 3.21
    numpy.testing.assert_allclose(grid.size, 3.21)

    with pytest.raises(ValueError):
        grid.size = 0
    with pytest.raises(ValueError):
        grid.size = -1


def test_update():
    grid = HexGrid(size=1, shape="pointy")

    new_grid = grid.update(shape="flat")
    assert grid.shape == "pointy"
    assert new_grid.shape == "flat"

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
