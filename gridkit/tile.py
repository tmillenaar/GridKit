from typing import Tuple, Union

import numpy

from gridkit.base_grid import BaseGrid
from gridkit.errors import AlignmentError
from gridkit.gridkit_rs import *
from gridkit.hex_grid import HexGrid
from gridkit.index import GridIndex, validate_index
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
        self.grid = grid.update()

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

    @property
    def indices(self):
        """The ids of all cells in the Tile.

        Returns
        -------
        :class:`.GridIndex`
            The :class:`.GridIndex` that contains the indices in the Tile
        """
        return GridIndex(self._tile.indices())

    @property
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

    def intersects(self, other):
        if isinstance(other, DataTile):
            other_tile = other._data_tile.get_tile()
        elif isinstance(other, Tile):
            other_tile = other._tile
        elif isinstance(other, PyO3TriDataTile):
            other_tile = other.get_tile()
        elif isinstance(other, PyO3TriTile):
            other_tile = other
        else:
            raise TypeError(
                f"Cannot determine intersection between {type(self).__name__} and {type(other).__name__}"
            )
        return self._tile.intersects(other_tile)


class DataTile:

    def __init__(self, tile: Tile, data: numpy.ndarray):
        self.grid = tile.grid
        self._data_tile = PyO3TriDataTile.from_tile(tile._tile, data.astype("float64"))

    def to_numpy(self):
        return self._data_tile.to_numpy()

    def __getitem__(self, item):
        return self.to_numpy()[item]

    def __setitem__(self, item, value):
        # Don't use update() but replace _data_tile in-place
        new_data = self.to_numpy()
        new_data[item] = value
        tile = PyO3TriTile(
            self.grid._grid, tuple(self.start_id.index), self.nx, self.ny
        )
        self._data_tile = PyO3TriDataTile.from_tile(tile, new_data)

    def update(self, grid=None, data=None, start_id=None, nx=None, ny=None):
        # TODO: Make clear that update copies the data
        if grid is None:
            grid = self.grid
        if data is None:
            data = self.to_numpy()
        if start_id is None:
            start_id = self.start_id
            if isinstance(start_id, GridIndex):
                start_id = start_id.index
        start_id = tuple(start_id)
        if nx is None:
            nx = self.nx
        if ny is None:
            ny = self.ny

        if data.shape != (ny, nx):
            raise ValueError(
                f"The shape of the supplied data ({data.shape}) does not match the shape of the tile where ny is {ny} and nx is {nx}"
            )

        return DataTile(Tile(grid, start_id, nx, ny), data)

    @property
    def start_id(self):
        """The starting cell of the Tile.
        The starting cell defines the bottom-left corner of the Tile if the associated grid is not rotated.
        """
        return GridIndex(self._data_tile.start_id())

    @property
    def nx(self):
        """The number of cells in x direction, starting from the ``start_id``"""
        return self._data_tile.nx()

    @property
    def ny(self):
        """The number of cells in y direction, starting from the ``start_id``"""
        return self._data_tile.ny()

    def get_tile(self):
        """A Tile object with the same properties as this DataTile, but without the data attached."""
        return Tile(self.grid, self.start_id, self.nx, self.ny)

    def corner_ids(self):
        """The ids at the corners of the Tile

        Returns
        -------
        :class:`.GridIndex`
            The :class:`.GridIndex` that contains the ids of the cells at
            the corners of the Tile in order: top-left, top-right, bottom-right, bottom-left
            (assuming the assicaited grid is not rotated)
        """
        return GridIndex(self._data_tile.corner_ids())

    def corners(self):
        """The coordinates at the corners of the Tile

        Returns
        -------
        `numpy.ndarray`
            A two-dimensional array that contais the x and y coordinates of
            the corners in order: top-left, top-right, bottom-right, bottom-left
            (assuming the assicaited grid is not rotated)
        """
        return self._data_tile.corners()

    @property
    def indices(self):
        """The ids of all cells in the Tile.

        Returns
        -------
        :class:`.GridIndex`
            The :class:`.GridIndex` that contains the indices in the Tile
        """
        return GridIndex(self._data_tile.indices())

    @property
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
        return self._data_tile.bounds()

    def overlap(self, other):
        is_aligned, reason = self.grid.is_aligned_with(other.grid)
        if not is_aligned:
            raise AlignmentError(
                f"Cannot find overlap of grids that are not aligned. Reason for misalignemnt: {reason}"
            )
        _data_tile = self._data_tile.overlap(other._data_tile)
        # Make a copy TODO: allow creation of Tile from PyO3...Tile and DataTile from PyO3...DataTile
        overlap = self.update()
        overlap._data_tile = _data_tile
        return overlap

    def _empty_combined_data_tile(self, other):
        _data_tile = self._data_tile._empty_combined_data_tile(other._data_tile)
        # Make a copy TODO: allow creation of Tile from PyO3...Tile and DataTile from PyO3...DataTile
        combined = self.update()
        combined._data_tile = _data_tile
        return combined

    def crop(self, crop_tile):
        _data_tile = self._data_tile.crop(crop_tile._tile, nodata_value=0)
        cropped = self.update()
        cropped._data_tile = _data_tile
        return cropped

    @validate_index
    def value(self, index=None, oob_value=numpy.nan):
        oob_value = numpy.float64(oob_value)
        original_shape = index.shape
        result = self._data_tile.value(index.ravel().index, oob_value)
        return result.reshape(original_shape)

    def intersects(self, other):
        return self.get_tile().intersects(other)

    def __add__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._add_tile(other._data_tile)
        else:
            try:
                other = float(other)
                _data_tile = self._data_tile._add_scalar(other)
            except:
                raise TypeError(f"Cannot add DataTile and `{type(other)}`")
        combined = self.update()
        combined._data_tile = _data_tile
        return combined

    def __sub__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._subtract_tile(other._data_tile)
        else:
            try:
                other = float(other)
                _data_tile = self._data_tile._subtract_scalar(other)
            except:
                raise TypeError(f"Cannot subtract DataTile and `{type(other)}`")
        combined = self.update()
        combined._data_tile = _data_tile
        return combined

    def __mul__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._multiply_tile(other._data_tile)
        else:
            try:
                other = float(other)
                _data_tile = self._data_tile._multiply_scalar(other)
            except:
                raise TypeError(f"Cannot multiply DataTile and `{type(other)}`")
        combined = self.update()
        combined._data_tile = _data_tile
        return combined

    def __truediv__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._divide_tile(other._data_tile)
        else:
            try:
                other = float(other)
                _data_tile = self._data_tile._divide_scalar(other)
            except:
                raise TypeError(f"Cannot divide DataTile and `{type(other)}`")
        combined = self.update()
        combined._data_tile = _data_tile
        return combined
