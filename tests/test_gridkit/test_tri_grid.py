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
    "indices, expected_centroids",
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
@pytest.mark.parametrize("offset", ((0, 0), (-0.7, 0.3), (1, -0.2)))
def test_cell_corners(indices, offset, expected_centroids):
    grid = TriGrid(size=1.5, offset=offset)
    corners = grid.cell_corners(indices)
    corners -= offset
    numpy.testing.assert_allclose(corners, expected_centroids, atol=1e-15)


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
    points += offset
    grid = TriGrid(size=0.7, offset=offset)
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
            (6, 4),
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
            (5, 3),
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
            (5, 3),
        ],
    ),
)
@pytest.mark.parametrize("return_cell_count", [True, False])
def test_cells_in_bounds(bounds, expected_ids, expected_shape, return_cell_count):
    grid = TriGrid(size=0.7)
    bounds = grid.align_bounds(bounds, "nearest")
    result = grid.cells_in_bounds(bounds, return_cell_count=return_cell_count)
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
def test_cells_near_point(points, expected_nearby_ids):
    grid = TriGrid(size=0.6, offset=(0.2, 0.3))
    nearby_ids = grid.cells_near_point(points)
    # Test single point input
    numpy.testing.assert_allclose(nearby_ids[0], expected_nearby_ids)
    # Test for all points
    expected_nearby_ids_extended = numpy.empty(shape=(len(points), 6, 2))
    expected_nearby_ids_extended[:] = expected_nearby_ids
    numpy.testing.assert_allclose(nearby_ids, expected_nearby_ids_extended)
