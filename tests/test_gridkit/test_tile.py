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


@pytest.mark.parametrize("grid", [TriGrid(size=1), RectGrid(size=1), HexGrid(size=1)])
@pytest.mark.parametrize("start_id", [(2, 3), [-3, 2]])
@pytest.mark.parametrize("nx", [1, 7])
@pytest.mark.parametrize("ny", [1, 4])
def test_corner_ids(grid, start_id, nx, ny):
    tile = Tile(grid, start_id, nx, ny)
    corners = tile.corner_ids()
    assert GridIndex(start_id) in corners
    # FIXME: Don't convert specifically to GridIndex when this ticket is resolved: https://github.com/tmillenaar/GridKit/issues/93
    assert GridIndex((start_id[0] + nx - 1, start_id[1])) in corners
    assert GridIndex((start_id[0], start_id[1] + ny - 1)) in corners
    assert GridIndex((start_id[0] + nx - 1, start_id[1] + ny - 1)) in corners
