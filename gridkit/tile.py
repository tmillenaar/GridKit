from typing import Tuple, Union

import numpy

from gridkit.base_grid import BaseGrid
from gridkit.gridkit_rs import PyO3HexTile, PyO3RectTile, PyO3TriTile
from gridkit.hex_grid import HexGrid
from gridkit.index import GridIndex
from gridkit.rect_grid import RectGrid
from gridkit.tri_grid import TriGrid


class Tile:
    """A Tile describes a set of cells defined by the ``start_id``,
    which is the cell that defines the bottom-left corner of the tile,
    and ``nx`` and ``ny``, which are the number of cells in the x and y directions, respectively.

    Each tile is associated with a particular grid and the Tile refers to a selection
    of grid indices on that grid. The associated grid can be accessed using the ``.grid`` property.

    .. Note ::

        ``nx`` and ``ny`` can be seen as 'right' and 'up', respectively, when the rotation of the grid is zero.
        If the grid is rotated, the tile rotates with it (naturally).
        This means that for a grid that is rotated 90 degrees,
        ``nx`` refers to the number of cells up, and ``ny`` refers to the number of cells to the left.

    ..

    Init parameters
    ---------------
    grid: :class:`BaseGrid`
        The :class:`.TriGrid`, :class:`.RectGrid` or :class:`.HexGrid` the tile is associated with
    start_id: Union[Tuple[int, int], GridIndex]
        The starting cell of the Tile.
        The starting cell defines the bottom-left corner of the Tile if the associated grid is not rotated.
    nx: int
        The number of cells in x direction, starting from the ``start_id``
    ny: int
        The number of cells in y direction, starting from the ``start_id``


    """

    def __init__(
        self,
        grid: BaseGrid,
        start_id: Union[Tuple[int, int], GridIndex],
        nx: int,
        ny: int,
    ):

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
                "'start_id' must be a single pair of indices in the form (x,y), got: {start_id}"
            )
        start_id = (
            tuple(start_id.index)
            if isinstance(start_id, GridIndex)
            else tuple(start_id)
        )

        self.grid = grid
        if isinstance(grid, TriGrid):
            self._tile = PyO3TriTile(grid._grid, start_id, nx, ny)
        elif isinstance(grid, RectGrid):
            self._tile = PyO3RectTile(grid._grid, start_id, nx, ny)
        elif isinstance(grid, HexGrid):
            self._tile = PyO3HexTile(grid._grid, start_id, nx, ny)
        else:
            raise TypeError(
                f"Unexpected type for 'grid', expected a TriGrid, RectGrid or HexGrid, got a: {type(grid)}"
            )

    @property
    def start_id(self):
        """The starting cell of the Tile.
        The starting cell defines the bottom-left corner of the Tile if the associated grid is not rotated.
        """
        return GridIndex(self._tile.start_id)

    @property
    def nx(self):
        """The number of cells in x direction, starting from the ``start_id``"""
        return self._tile.nx

    @property
    def ny(self):
        """The number of cells in y direction, starting from the ``start_id``"""
        return self._tile.ny

    def corner_ids(self):
        """The ids at the corners of the Tile

        Returns
        -------
        :class:`.GridIndex`
            The :class:`.GridIndex` that contains the ids of the cells at
            the corners of the Tile in order: top-left, top-right, bottom-right, bottom-left
            (assuming the assicaited grid is not rotated)
        """
        return GridIndex(self._tile.corner_ids())

    def corners(self):
        """The coordinates at the corners of the Tile

        Returns
        -------
        `numpy.ndarray`
            A two-dimensional array that contais the x and y coordinates of
            the corners in order: top-left, top-right, bottom-right, bottom-left
            (assuming the assicaited grid is not rotated)
        """
        return self._tile.corners()

    def indices(self):
        """The ids of all cells in the Tile.

        Returns
        -------
        :class:`.GridIndex`
            The :class:`.GridIndex` that contains the indices in the Tile
        """
        return GridIndex(self._tile.indices())

    def bounds(self) -> Tuple[float, float, float, float]:
        """The bounding box of the Tile in (xmin, ymin, xmax, ymax).
        If the associated grid is rotated, the this represents the bounding box
        that fully encapsulates the Tile and will contain more area than is
        covered by the rotated Tile.

        Returns
        -------
        Tuple[float, float, float, float]
            The bounding box in (xmin, ymin, xmax, ymax)
        """
        return self._tile.bounds()
