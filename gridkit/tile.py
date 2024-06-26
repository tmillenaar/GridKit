import numpy

from gridkit.gridkit_rs import PyHexTile, PyRectTile, PyTriTile
from gridkit.hex_grid import HexGrid
from gridkit.index import GridIndex
from gridkit.rect_grid import RectGrid
from gridkit.tri_grid import TriGrid


class Tile:

    def __init__(self, grid, start_id, nx, ny):

        if not numpy.isclose(nx % 1, 0):
            raise ValueError(f"Expected an integer for 'nx', got: {nx}")
        if nx < 1:
            raise ValueError(f"Expected 'nx' to be 1 or larger, got: {nx}")
        if not numpy.isclose(ny % 1, 0):
            raise ValueError(f"Expected an integer for 'ny', got: {ny}")
        if ny < 1:
            raise ValueError(f"Expected 'nx' to be 1 or larger, got: {ny}")
        if not len(GridIndex(start_id)) == 1:
            raise ValueError(
                "'start_id' must be a single pair of indices in the for (x,y), got: {start_id}"
            )
        sart_id = tuple(start_id)

        self.grid = grid
        if isinstance(grid, TriGrid):
            self._tile = PyTriTile(grid._grid, start_id, nx, ny)
        elif isinstance(grid, RectGrid):
            self._tile = PyRectTile(grid._grid, start_id, nx, ny)
        elif isinstance(grid, HexGrid):
            self._tile = PyHexTile(grid._grid, start_id, nx, ny)
        else:
            raise TypeError(
                f"Unexpected type for 'grid', expected a TriGrid, RectGrid or HexGrid, got a: {type(grid)}"
            )

    @property
    def start_id(self):
        return GridIndex(self._tile.start_id)

    @property
    def nx(self):
        return self._tile.nx

    @property
    def ny(self):
        return self._tile.ny

    def corner_ids(self):
        return GridIndex(self._tile.corner_ids())

    def corners(self):
        return self._tile.corners()

    def indices(self):
        return GridIndex(self._tile.indices())

    def bounds(self):
        return self._tile.bounds()
