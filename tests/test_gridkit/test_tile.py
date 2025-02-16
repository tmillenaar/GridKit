import numpy
import pytest

from gridkit import GridIndex, HexGrid, RectGrid, Tile, TriGrid


@pytest.mark.parametrize("grid", [TriGrid(size=1), RectGrid(size=1), HexGrid(size=1)])
@pytest.mark.parametrize("start_id", [(2, 3), [-3, 2]])
@pytest.mark.parametrize("nx", [1, 7])
@pytest.mark.parametrize("ny", [1, 4])
def test_init(grid, start_id, nx, ny):
    tile = Tile(grid, start_id, nx, ny)
    assert isinstance(tile.grid, grid.__class__)
    assert grid.is_aligned_with(tile.grid)
    numpy.testing.assert_allclose(tile.start_id, start_id)
    numpy.testing.assert_allclose(tile.nx, nx)
    numpy.testing.assert_allclose(tile.ny, ny)


def test_init_errors():
    grid = TriGrid(size=1)

    # Test float nx
    with pytest.raises(ValueError):
        Tile(grid, (0, 0), 1.1, 1)

    # Test float ny
    with pytest.raises(ValueError):
        Tile(grid, (0, 0), 1, 1.1)

    # Test nx < 1
    with pytest.raises(ValueError):
        Tile(grid, (0, 0), 0, 1)

    # Test ny < 1
    with pytest.raises(ValueError):
        Tile(grid, (0, 0), 1, 0)

    # Test start_id dimentions
    with pytest.raises(ValueError):
        Tile(grid, [(0, 0), (1, 1)], 1, 1)

    # Test grid dtype
    with pytest.raises(TypeError):
        Tile(grid._grid, (0, 0), 1, 1)


@pytest.mark.parametrize(
    "grid",
    [
        TriGrid(size=1),
        RectGrid(size=1),
        HexGrid(size=1, shape="flat"),
        HexGrid(size=1, shape="pointy"),
    ],
)
@pytest.mark.parametrize("start_id", [(2, 3), [-3, 2]])
@pytest.mark.parametrize("nx", [1, 7])
@pytest.mark.parametrize("ny", [1, 4])
@pytest.mark.parametrize(
    "rotation", [0, -10, 375]
)  # should not have an impact on the outcome
def test_corner_ids(grid, start_id, nx, ny, rotation):
    grid.rotation = rotation
    tile = Tile(grid, start_id, nx, ny)
    corners = tile.corner_ids()
    assert start_id in corners
    assert (start_id[0] + nx - 1, start_id[1]) in corners
    assert (start_id[0], start_id[1] + ny - 1) in corners
    assert (start_id[0] + nx - 1, start_id[1] + ny - 1) in corners


@pytest.mark.parametrize("grid", [TriGrid(size=1), RectGrid(size=1), HexGrid(size=1)])
@pytest.mark.parametrize("start_id", [(2, 3), [-3, 2]])
@pytest.mark.parametrize("nx", [1, 7])
@pytest.mark.parametrize("ny", [1, 4])
@pytest.mark.parametrize("rotation", [0, 10, 375])
def test_corners(grid, start_id, nx, ny, rotation):
    grid.rotation = rotation
    tile = Tile(grid, start_id, nx, ny)
    corners = tile.corners()

    numpy.testing.assert_allclose(corners.shape, (4, 2))

    side_x = grid.rotation_matrix.dot((0, ny * grid.dy))
    side_y = grid.rotation_matrix.dot((nx * grid.dx, 0))

    numpy.testing.assert_allclose(corners[1] - corners[0], side_y)  # top
    numpy.testing.assert_allclose(corners[1] - corners[2], side_x)  # right
    numpy.testing.assert_allclose(corners[2] - corners[3], side_y)  # bottom
    numpy.testing.assert_allclose(corners[0] - corners[3], side_x)  # left


@pytest.mark.parametrize("grid", [TriGrid(size=1), RectGrid(size=1), HexGrid(size=1)])
@pytest.mark.parametrize("start_id", [(2, 3), [-3, 2]])
@pytest.mark.parametrize("nx", [1, 7])
@pytest.mark.parametrize("ny", [1, 4])
@pytest.mark.parametrize("rotation", [0, 10, 375])
def test_indices(grid, start_id, nx, ny, rotation):
    grid.rotation = rotation
    tile = Tile(grid, start_id, nx, ny)
    indices = tile.indices
    assert isinstance(indices, GridIndex)
    assert len(indices) == nx * ny
    assert len(indices.unique()) == nx * ny  # Make sure there are no duplicate indices
    # Test the x-id and y-id values
    assert not numpy.any(indices.x < start_id[0])
    assert not numpy.any(indices.y < start_id[1])
    # Note: exclude 'start_id[0] + nx' and 'start_id[1] + ny' because
    #       start_id itself is already included (hence '>=')
    assert not numpy.any(indices.x >= start_id[0] + nx)
    assert not numpy.any(indices.y >= start_id[1] + ny)
