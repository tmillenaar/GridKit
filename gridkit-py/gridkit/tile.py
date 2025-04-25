import warnings
from typing import List, Literal, Tuple, Union

import numpy
from pyproj import Transformer
from shapely.geometry import MultiPoint

from gridkit.base_grid import BaseGrid
from gridkit.errors import AlignmentError
from gridkit.gridkit_rs import *
from gridkit.hex_grid import HexGrid
from gridkit.index import GridIndex, validate_index
from gridkit.rect_grid import RectGrid
from gridkit.tri_grid import TriGrid


def get_value_dtype(value):
    if isinstance(value, numpy.generic):
        return value.dtype
    return numpy.array(value).dtype


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
    grid: :class:`.BaseGrid`
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
            self._tile = PyO3Tile.from_tri_grid(grid._grid, start_id, nx, ny)
        elif isinstance(grid, RectGrid):
            self._tile = PyO3Tile.from_rect_grid(grid._grid, start_id, nx, ny)
        elif isinstance(grid, HexGrid):
            self._tile = PyO3Tile.from_hex_grid(grid._grid, start_id, nx, ny)
        else:
            raise TypeError(
                f"Unexpected type for 'grid', expected a TriGrid, RectGrid or HexGrid, got a: {type(grid)}"
            )
        self.grid = grid.update()

    @staticmethod
    def from_pyo3_tile(grid, pyo3_tile):
        return Tile(grid, pyo3_tile.start_id, pyo3_tile.nx, pyo3_tile.ny)

    def to_data_tile(self, data, nodata_value=None):
        if not data.ndim == 2:
            raise TypeError(
                f"Setting data is only allowed for 2D data. Got data with {data.ndim} dimensions."
            )
        if data.shape[0] != self.ny:
            raise ValueError(
                f"The provided data has {data.shape()[0]} elements in the first axis but the tile has {tile.ny} y-elements. Beware that a numpy array's axis order is in y,x."
            )
        if data.shape[1] != self.nx:
            raise ValueError(
                f"The provided data has {data.shape()[1]} elements in the second axis but the tile has {tile.nx} x-elements. Beware that a numpy array's axis order is in y,x."
            )
        if nodata_value is None:
            if numpy.issubdtype(data.dtype, float):
                nodata_value = numpy.nan
            else:
                nodata_value = numpy.iinfo(data.dtype).max

        return DataTile(self, data, nodata_value)

    def to_data_tile_with_value(self, fill_value, nodata_value=None):
        dtype = get_value_dtype(fill_value)
        if nodata_value is None:
            if numpy.issubdtype(dtype, float):
                nodata_value = numpy.nan
            else:
                nodata_value = numpy.iinfo(dtype).max

        # Map numpy dtypes to method suffixes
        dtype_method_map = {
            numpy.dtype("float64"): "f64",
            numpy.dtype("float32"): "f32",
            numpy.dtype("int64"): "i64",
            numpy.dtype("int32"): "i32",
            numpy.dtype("int16"): "i16",
            numpy.dtype("int8"): "i8",
            numpy.dtype("uint64"): "u64",
            numpy.dtype("uint32"): "u32",
            numpy.dtype("uint16"): "u16",
            numpy.dtype("uint8"): "u8",
            # numpy.dtype('bool'):   'bool',  # optional support
            # FIXME: add complex version
        }

        method_suffix = dtype_method_map.get(dtype)
        if method_suffix is None:
            raise TypeError(f"Unsupported dtype: {dtype}")

        method_name = f"to_data_tile_with_value_{method_suffix}"
        method = getattr(self._tile, method_name, None)

        if method is None:
            raise AttributeError(f"Method {method_name} not found on tile")

        py03_data_tile = method(fill_value, nodata_value)
        return DataTile.from_pyo3_data_tile(self.grid.update(), py03_data_tile)

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
        """The coordinates at the corners of the Tile in order: top-left, top-right, bottom-right, bottom-left
        (assuming the assicaited grid is not rotated)

        Returns
        -------
        `numpy.ndarray`
            A two-dimensional array that contais the x and y coordinates of the corners
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
    def mpl_extent(self) -> tuple:
        """Raster Bounds

        Returns
        -------
        :class:`tuple`
            The extent of the data as defined expected by matplotlib in (left, right, bottom, top) or equivalently (min-x, max-x, min-y, max-y)
        """
        corners = self.corners()
        return (
            corners[:, 0].min(),
            corners[:, 0].max(),
            corners[:, 1].min(),
            corners[:, 1].max(),
        )

    def intersects(self, other):
        """Only checks bounds, not grid type, alignment etc."""
        # TODO: Do check CRS
        if isinstance(other, DataTile):
            other_tile = other._data_tile.get_tile()
        elif isinstance(other, Tile):
            other_tile = other._tile
        elif isinstance(other, PyO3Tile):
            other_tile = other
        else:
            raise TypeError(
                f"Cannot determine intersection between {type(self).__name__} and {type(other).__name__}"
            )
        return self._tile.intersects(other_tile)

    def centroid(self, index=None):
        if index is None:
            index = self.indices
        return self.grid.centroid(index)

    def overlap(self, other):
        is_aligned, reason = self.grid.is_aligned_with(other.grid)
        if not is_aligned:
            raise AlignmentError(
                f"Cannot find overlap of grids that are not aligned. Reason for misalignemnt: {reason}"
            )
        _tile = self._tile.overlap(other._tile)
        return Tile.from_pyo3_tile(self.grid, _tile)

    @validate_index
    def to_shapely(self, index=None, as_multipolygon=True):
        if index is None:
            index = self.indices
        return self.grid.to_shapely(index, as_multipolygon=as_multipolygon)

    def tile_id_to_grid_id(self, tile_id=None, oob_value=numpy.iinfo(numpy.int64).max):
        """Convert the index referring to a cell from the tile reference frame to that of the grid.
        The tile id follows the numpy convention and is in the form [[y0, y1, y2], [x0, x1, x2]].
        The grid id is int the form [(x0, y0), (x1,y1), (x2, y2)].
        The tile id will start at the top-left at [0,0]. The grid id starts at the bottom left,
        but depending on where in the world the tile is placed it can have any starting value.

        Parameters
        ----------
        tile_id: Tuple
            The tile index in the format: ((y0, y1, y2), (x0, x1, x2)).
            If None, return a GridIndex that includes all cells in the tile.
        oob_value: int
            The value to assign to indices not in the tile. Default: maximum numpy.in64 value.

        Returns
        -------
        :class:`.GridIndex`
            The index referring to the same cell but in grid coordinates instead of tile coordinates

        Examples
        --------

        Let's first start by creating a data_tile that we can sample using either grid ids or tile ids.

        .. code-block:: python

            >>> import numpy
            >>> from gridkit import RectGrid, Tile
            >>> grid = RectGrid(size=5)
            >>> nx = ny = 4
            >>> tile = Tile(grid, start_id=(2,3), nx=nx, ny=ny)
            >>> data = numpy.arange(ny*nx).reshape(ny,nx) # Note that numpy indexes in the form (y,x)
            >>> data_tile = tile.to_data_tile(data=data)
            >>> print(data_tile.to_numpy())
            [[ 0  1  2  3]
             [ 4  5  6  7]
             [ 8  9 10 11]
             [12 13 14 15]]
            >>> tile_ids = ((1,2,3), (2,1,0))
            >>> grid_ids = data_tile.tile_id_to_grid_id(tile_ids)
            >>> print(grid_ids.index)
            [[4 5]
             [3 4]
             [2 3]]

        ..

        Note the differnce in value due to the tile's start_id of (2,3), as well as the difference in array shape.

        The method `tile_id_to_grid_id` exists on both Tile and DataTile objects.
        The tile and data_tile here are at the same location on the grid,
        the difference is just that the data_tile has data.
        Doing the id conversion should yield the same result in both cases.

        The value obtained from data_tile.value using the grid_id should be the same as obtaining
        the value from data_tile.to_numpy() using the tile_ids.
        You can shorthand this by just indexing directly on the data_tile.

        .. code-block:: python

            >>> assert grid_ids == tile.tile_id_to_grid_id(tile_ids)
            >>> print(data_tile[tile_ids])
            [ 6  9 12]
            >>> print(data_tile.value(grid_ids))
            [ 6  9 12]

        ..

        We should of course be able to get the tile_ids back using :meth:`Tile.grid_id_to_tile_id`:

        .. code-block:: python

            >>> reverted_tile_ids = tile.grid_id_to_tile_id(grid_ids)
            >>> print(reverted_tile_ids)
            (array([1, 2, 3]), array([2, 1, 0]))
            >>> numpy.testing.assert_allclose(tile_ids, reverted_tile_ids)

        ..

        # We get a tuple of numpy arrays back instead of a tuple of tuples,
        # but numerically they are the same as the tuple we started with.
        # :meth:`Tile.grid_id_to_tile_id` always returns a tuple of arrays but
        # :meth:`Tile.tile_id_to_grid_id` accepts a tuple of lists, tuple of tuples or tuple of arrays.

        See also
        --------
        :meth:`Tile.grid_id_to_tile_id`

        """
        if tile_id is None:
            return self.indices
        else:
            tile_id = numpy.array(tile_id, dtype=int)
        if not tile_id.shape[0] == 2:
            raise ValueError(
                f"""
                Expected the first dimension of tile_id to have a length of two (for x,y). Got indices in shape: {tile_id.shape}.
                This is different than the grid ids where we expect xy to be the last dimesnion.
                This is in an effort to make indexing the data as a numpy array using data[tuple(numpy_ids)] the same
                as querying values from datatile.value using grid ids botained through tile_id_to_grid_id(numpy_ids).
                Note that not only does this function need the first demension to be of size two,
                but numpy starts with the y-coordinate, so we expect: [[y0,y1,y2], [x0,x1,x2]].
            """
            )
        if tile_id.ndim == 1:
            tile_id = tile_id[numpy.newaxis]
        else:
            # Note: transpose to get the data from shape [[y0,y1,y2], [x0,x1,x2]]
            #       into shape [[x0,y0], [x1,y1], [x2,y2]] which is what the rust package works with
            tile_id = tile_id.T
        oob_value = numpy.int64(oob_value)
        return GridIndex(self._tile.tile_id_to_grid_id(tile_id, oob_value=oob_value))

    @validate_index
    def grid_id_to_tile_id(self, index=None, oob_value=numpy.iinfo(numpy.int64).max):
        """Convert the index referring to a cell from the grid reference frame to that of the tile.
        The tile id follows the numpy convention and is in the form [[y0, y1, y2], [x0, x1, x2]].
        The grid id is int the form [(x0, y0), (x1,y1), (x2, y2)].
        The tile id will start at the top-left at [0,0]. The grid id starts at the bottom left,
        but depending on where in the world the tile is placed it can have any starting value.

        Parameters
        ----------
        index: :class:`.GridIndex`
            The grid index in the form [(x0, y0), (x1,y1), (x2, y2)].
            If None, return the tile indices for each cell. Note that they are in raveled form.
        oob_value: int
            The value to assign to indices not in the tile. Default: maximum numpy.in64 value.

        Returns
        -------
        :class:`.GridIndex`
            The index referring to the same cell but in tile coordinates instead of tile coordinates

        Examples
        --------
        For an elaborate example showing going from tile_id to grid_id and back, see :meth:`Tile.tile_id_to_grid_id`

        See also
        --------
        :meth:`Tile.tile_id_to_grid_id`

        """
        if index is None:
            index = self.indices
        index = (
            index.ravel().index[None] if index.index.ndim == 1 else index.ravel().index
        )
        oob_value = numpy.int64(oob_value)
        # Note: return a tuple instead of numpy array becuase numpy indexing behaviour
        #       is different when using arrays of integers compared to tuples.
        return tuple(self._tile.grid_id_to_tile_id(index, oob_value=oob_value).T)


class DataTile(Tile):

    def __init__(self, tile: Tile, data: numpy.ndarray, nodata_value=None):
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D array, got {data.ndim} dimensions")
        if tile.ny != data.shape[0] or tile.nx != data.shape[1]:
            raise ValueError(
                f"The shape of the data {data.shape} does not match the shape of the tile: {(tile.ny, tile.nx)}"
            )
        if nodata_value is None:
            if numpy.issubdtype(data.dtype, float):
                nodata_value = numpy.nan
            else:
                nodata_value = numpy.iinfo(data.dtype.type).max
        else:
            if not get_value_dtype(nodata_value) == data.dtype:
                try:
                    nodata_value = numpy.array(nodata_value).astype(
                        data.dtype, casting="safe"
                    )
                except TypeError as e:
                    raise TypeError(
                        f"Data type of the supplied array in argument 'data' (dtype: '{data.dtype}') did not match the supplied nodata value '{nodata_value}' of type '{type(nodata_value)}'"
                    ) from e
        self.grid = tile.grid

        # Map numpy dtypes to method suffixes
        dtype_method_map = {
            numpy.dtype("float64"): "f64",
            numpy.dtype("float32"): "f32",
            numpy.dtype("int64"): "i64",
            numpy.dtype("int32"): "i32",
            numpy.dtype("int16"): "i16",
            numpy.dtype("int8"): "i8",
            numpy.dtype("uint64"): "u64",
            numpy.dtype("uint32"): "u32",
            numpy.dtype("uint16"): "u16",
            numpy.dtype("uint8"): "u8",
            # numpy.dtype('bool'):   'bool',  # optional support
            # FIXME: add complex version
        }
        method_suffix = dtype_method_map.get(data.dtype)
        if method_suffix is None:
            raise TypeError(f"Unsupported dtype: {data.dtype}")
        method_name = f"to_data_tile_{method_suffix}"
        method = getattr(tile._tile, method_name, None)
        if method is None:
            raise AttributeError(f"Method {method_name} not found on tile")
        self._data_tile = method(data, nodata_value)

        # _tile is used by the Tile parent class
        self._tile = self._data_tile.get_tile()

    @property
    def dtype(self):
        dtype_method_map = {
            "PyO3DataTileF64": "float64",
            "PyO3DataTileF32": "float32",
            "PyO3DataTileI64": "int64",
            "PyO3DataTileI32": "int32",
            "PyO3DataTileI16": "int16",
            "PyO3DataTileI8": "int8",
            "PyO3DataTileU64": "uint64",
            "PyO3DataTileU32": "uint32",
            "PyO3DataTileU16": "uint16",
            "PyO3DataTileU8": "uint8",
            # numpy.dtype('bool'):   'bool',  # optional support
            # FIXME: add complex version
        }
        return dtype_method_map[self._data_tile.__class__.__name__]

    def astype(self, dtype, nodata_value=None):
        if self.dtype == numpy.dtype(dtype):
            # Note: Do not make unnesecary copy here, but conversion does return copy.
            #       Too inconsistent? Or worth skipping the copy?
            return self

        new_data = self.to_numpy().astype(dtype)
        if nodata_value is None:
            # Note, maybe obsolete if nodata_value ever is allowed to be None?
            try:
                new_nodata_value = numpy.array(self.nodata_value, dtype=dtype)
            except ValueError:
                if numpy.issubdtype(dtype, numpy.floating):
                    new_nodata_value = numpy.nan
                elif numpy.issubdtype(dtype, numpy.integer):
                    new_nodata_value = numpy.iinfo(dtype).max
                else:
                    raise ValueError(
                        f"Unable to convert nodata_value of {self.nodata_value} to dtype {dtype}"
                    )
        result = self.update(data=new_data, nodata_value=new_nodata_value)
        # Note: since self and result have the same tile coverage it is safe to index result with ids from self
        result[self.nodata_cells] = new_nodata_value
        return result

    @property
    def nodata_value(self):
        return numpy.array(self._data_tile.nodata_value(), dtype=self.dtype)

    @nodata_value.setter
    def nodata_value(self, value):
        """Sets the nodata value of the DataTile.
        This replaces all instances of the nodata value with the new value"""
        try:
            return self._data_tile.set_nodata_value(value)
        except TypeError:
            dtype = get_value_dtype(value)
            value = numpy.array(value, dtype=dtype)
            return self._data_tile.set_nodata_value(value)

    def is_nodata(self, values):
        values = numpy.array(values)
        if values.ndim == 0:
            return self._data_tile.is_nodata(values)
        return self._data_tile.is_nodata_array(values)

    @property
    def nodata_cells(self):
        return GridIndex(self._data_tile.nodata_cells())

    @staticmethod
    def from_pyo3_data_tile(grid, pyo3_data_tile):
        tile = Tile(
            grid, pyo3_data_tile.start_id(), pyo3_data_tile.nx(), pyo3_data_tile.ny()
        )
        data = pyo3_data_tile.to_numpy()
        data_tile = DataTile(
            tile,
            data,
            nodata_value=numpy.array(pyo3_data_tile.nodata_value(), dtype=data.dtype),
        )
        return data_tile

    def to_numpy(self):
        return self._data_tile.to_numpy()

    def __array__(self, dtype=None):
        return self.to_numpy() if dtype is None else self.to_numpy().astype(dtype)

    def __getitem__(self, item):
        if isinstance(item, GridIndex):
            item = self.grid_id_to_tile_id(item)
        return self.to_numpy()[item]

    def __setitem__(self, item, value):
        # Don't use update() but replace _data_tile in-place
        if isinstance(item, GridIndex):
            item = self.grid_id_to_tile_id(item)

        new_data = self.to_numpy()
        new_data[item] = value
        if isinstance(self.grid, TriGrid):
            tile = PyO3Tile.from_tri_grid(
                self.grid._grid, tuple(self.start_id.index), self.nx, self.ny
            )
        elif isinstance(self.grid, RectGrid):
            tile = PyO3Tile.from_rect_grid(
                self.grid._grid, tuple(self.start_id.index), self.nx, self.ny
            )
        elif isinstance(self.grid, HexGrid):
            tile = PyO3Tile.from_hex_grid(
                self.grid._grid, tuple(self.start_id.index), self.nx, self.ny
            )
        else:
            raise TypeError(f"Unrecognized grid type: {self.grid}")
        new_data_tile = self.to_data_tile(new_data, self.nodata_value)
        self._data_tile = new_data_tile._data_tile

    def update(
        self, data=None, grid=None, start_id=None, nx=None, ny=None, nodata_value=None
    ):
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
        if nodata_value is None:
            nodata_value = self.nodata_value

        if data.shape != (ny, nx):
            raise ValueError(
                f"The shape of the supplied data ({data.shape}) does not match the shape of the tile where ny is {ny} and nx is {nx}"
            )
        return DataTile(Tile(grid, start_id, nx, ny), data, nodata_value=nodata_value)

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

    def crop(self, crop_tile):
        _data_tile = self._data_tile.crop(crop_tile._tile, nodata_value=0)
        cropped = DataTile(crop_tile, _data_tile.to_numpy())
        return cropped

    @validate_index
    def value(self, index=None, oob_value=None):
        if oob_value is None:
            if self.nodata_value is not None:
                oob_value = self.nodata_value
            else:
                if numpy.issubdtype(dtype, float):
                    oob_value = numpy.nan
                else:
                    oob_value = numpy.iinfo(dtype).max

        original_shape = index.shape
        index = (
            index.ravel().index[None] if index.index.ndim == 1 else index.ravel().index
        )
        result = self._data_tile.value(index, oob_value)
        return result.reshape(original_shape)

    def intersects(self, other):
        return self.get_tile().intersects(other)

    def _linear_interpolation(self, sample_points):
        if not isinstance(sample_points, numpy.ndarray):
            sample_points = numpy.array(sample_points, dtype=float)
        else:
            sample_points = sample_points.astype(float)

        original_shape = sample_points.shape
        if not original_shape[-1] == 2:
            raise ValueError(
                f"Expected the last axis of sample_points to have two elements (x,y). Got {original_shape[-1]} elements"
            )
        sample_points = sample_points.reshape(-1, 2)
        return self._data_tile.linear_interpolation(sample_points)

    def _inverse_distance_interpolation(self, sample_points, decay_constant=1):
        return self._data_tile.inverse_distance_interpolation(
            sample_points, decay_constant
        )

    def interpolate(
        self,
        sample_points,
        method: Literal["nearest", "bilinear", "inverse_distance"] = "nearest",
        **interp_kwargs,
    ):
        """Interpolate the value at the location of ``sample_points``.

        Points that are outside of the bounds of the tile are assigned `self.nodata_value`, or 'NaN' if no nodata value is set.

        Parameters
        ----------
        sample_points: :class:`numpy.ndarray`
            The coordinates of the points at which to sample the data
        method: :class:`str`, `'nearest', 'bilinear'`, optional
            The interpolation method used to determine the value at the supplied `sample_points`.
            Supported methods:
            - "nearest", for nearest neigbour interpolation, effectively sampling the value of the data cell containing the point
            - "bilinear", linear interpolation using the four cells surrounding the point
            - "inverse_distance", weighted inverse distance using the 4,3,6 nearby cells surrounding the point for Rect, Hex and Rect grid respectively.
            Default: "nearest"
        **interp_kwargs: `dict`
            The keyword argument passed to the interpolation function corresponding to the specified `method`

        Returns
        -------
        :class:`numpy.ndarray`
            The interpolated values at the supplied points

        See also
        --------
        :py:meth:`.BoundedGrid.resample`
        :py:meth:`.BaseGrid.interp_from_points`
        """
        if method == "nearest":
            new_ids = self.grid.cell_at_point(sample_points)
            return self.value(new_ids, oob_value=self.nodata_value)
        elif method == "bilinear" or method == "linear":
            return_shape = sample_points.shape[:-1]
            result = self._linear_interpolation(sample_points.reshape(-1, 2))
            return result.reshape(return_shape)
        elif method == "inverse_distance":
            decay_constant = interp_kwargs.pop("decay_constant", 1)
            return_shape = sample_points.shape[:-1]
            result = self._inverse_distance_interpolation(
                sample_points.reshape(-1, 2), decay_constant
            )
            return result.reshape(return_shape)
        raise ValueError(f"Resampling method '{method}' is not supported.")

    def resample(self, alignment_grid, method="nearest", **interp_kwargs):
        """Resample the grid onto another grid.
        This will take the locations of the grid cells of the other grid (here called ``alignment_grid``)
        and determine the value on these location based on the values of the original grid (``self``).

            The steps are as follows:
             1. Transform the bounds of the original data to the CRS of the alignment grid (if not already the same)
                No transformation is done if any of the grids has no CRS set.
             2. Find the cells of the alignment grid within these transformed bounds
             3. Find the cells of the original grid that are nearby each of the centroids of the cells found in 2.
                How many nearby cells are selected depends on the selected ``method``
             4. Interpolate the values using the supplied ``method`` at each of the centroids of the alignment grid cells selected in 2.
             5. Create a new bounded grid using the attributes of the alignment grid

        Parameters
        ----------
        alignment_grid: :class:`.BaseGrid`
            The grid with the desired attributes on which to resample.
            For the new data tile with the interpolated values, a tile extent will be chosen that closely matches
            the tile with the original data.
            If a :class:`.Tile` is provided, this step is skipped and the data is interpolated onto the cells of the
            tile as provided.

            .. Tip ::

                If the two grids don't align very well, you can play with the offset of the alignment grid to try to
                make it match the orignal grid a bit better. You could for example ``anchor`` a cell corner of the
                alignment grid to the corner of the tile like so: ``my_data_tile.resample(my_grid.anchor(my_data_tile.corners()[0], cell_element="corner"))``

            ..

        method: :class:`str`, `'nearest', 'bilinear', 'inverse_distance'`, optional
            The interpolation method used to determine the value at the supplied `sample_points`.
            Supported methods:
            - "nearest", for nearest neigbour interpolation, effectively sampling the value of the data cell containing the point
            - "bilinear", linear interpolation using the 4,3,6 nearby cells surrounding the point for Rect, Hex and Rect grid respectively.
            - "inverse_distance", weighted inverse distance using the 4,3,6 nearby cells surrounding the point for Rect, Hex and Rect grid respectively.
            Default: "nearest"
        **interp_kwargs: `dict`
            The keyword argument passed to the interpolation function corresponding to the specified `method`
        Returns
        -------
        :class:`.BoundedGrid`
            The interpolated values at the supplied points

        See also
        --------
        :py:meth:`.BoundedGrid.interpolate`
        :py:meth:`.BaseGrid.interp_from_points`

        """
        tile_is_given = isinstance(alignment_grid, Tile)
        if tile_is_given:
            tile = alignment_grid
            alignment_grid = alignment_grid.grid

        if self.grid.crs is None or alignment_grid.crs is None:
            warnings.warn(
                "`crs` not set for one or both grids. Assuming both grids have an identical CRS."
            )
            different_crs = False
        else:
            different_crs = not self.grid.crs.is_exact_same(alignment_grid.crs)

        if not tile_is_given:
            # make sure the bounds align with the grid
            if different_crs:
                # Create array that contains points around the bounds of the tile.
                # Then transform these points to the new CRS.
                # From the transformed coordinates we can determine the shape of the new bounds
                nr_cells_y = numpy.max((self.ny - 2), 0)

                top_left, top_right, bottom_right, bottom_left = self.corners()
                top_x = numpy.linspace(top_left[0], top_right[0], 2 * self.nx)
                top_y = numpy.linspace(top_left[1], top_right[1], 2 * self.nx)
                top_xy = numpy.stack([top_x, top_y])

                right_x = numpy.linspace(top_right[0], bottom_right[0], 2 * nr_cells_y)
                right_y = numpy.linspace(top_right[1], bottom_right[1], 2 * nr_cells_y)
                right_xy = numpy.stack([right_x, right_y])

                bottom_x = numpy.linspace(bottom_right[0], bottom_left[0], 2 * self.nx)
                bootom_y = numpy.linspace(bottom_right[1], bottom_left[1], 2 * self.nx)
                bottom_xy = numpy.stack([bottom_x, bootom_y])

                left_x = numpy.linspace(bottom_left[0], top_left[0], 2 * nr_cells_y)
                left_y = numpy.linspace(bottom_left[1], top_left[1], 2 * nr_cells_y)
                left_xy = numpy.stack([left_x, left_y])

                coords = numpy.hstack([top_xy, right_xy, bottom_xy, left_xy])

                transformer = Transformer.from_crs(
                    self.grid.crs, alignment_grid.crs, always_xy=True
                )
                corners_transformed = numpy.array(transformer.transform(*coords)).T

                ids = alignment_grid.cell_at_point(corners_transformed).ravel()
            else:
                corners = self.corners()
                ids = alignment_grid.cell_at_point(corners).ravel()
            min_x, min_y = numpy.min(ids, axis=0)
            max_x, max_y = numpy.max(ids, axis=0)

            # Prevent outer rows or columns with all nodata_value.
            # The centroid of the corner ids in the alignment_grid are checked against
            # the bounds of the original tile to see if the centroid of these new corner ids
            # is within the original tile. If the centroid is not within the original tile
            # the values will always be nodata_value so we don't want to include these.
            reference_corners = corners_transformed if different_crs else corners
            total_bounds = (  # in min_x, min_y, max_x, max_y
                *numpy.min(reference_corners, axis=0),
                *numpy.max(reference_corners, axis=0),
            )
            left_new, bottom_new = alignment_grid.centroid([min_x, min_y])
            right_new, top_new = alignment_grid.centroid([max_x, max_y])
            if left_new <= total_bounds[0]:
                min_x += int(
                    numpy.ceil((total_bounds[0] - left_new) / alignment_grid.dx)
                )
            if bottom_new <= total_bounds[1]:
                min_y += int(
                    numpy.ceil((total_bounds[1] - bottom_new) / alignment_grid.dy)
                )
            if right_new >= total_bounds[2]:
                max_x -= int(
                    numpy.ceil((right_new - total_bounds[2]) / alignment_grid.dx)
                )
            if top_new >= total_bounds[3]:
                max_y -= int(
                    numpy.ceil((top_new - total_bounds[3]) / alignment_grid.dy)
                )

            tile = Tile(
                alignment_grid, (min_x, min_y), max_x - min_x + 1, max_y - min_y + 1
            )
        else:  # Tile is already given
            pass  # Pass because we already have a tile to interpolate on

        new_ids = tile.indices
        shape = new_ids.shape

        new_points = alignment_grid.centroid(new_ids)

        if different_crs:
            transformer = Transformer.from_crs(
                alignment_grid.crs, self.grid.crs, always_xy=True
            )
            original_shape = new_points.shape
            raveled_new_points = new_points.reshape(-1, 2)
            transformed_points = transformer.transform(*raveled_new_points.T)
            new_points = numpy.vstack(transformed_points).T.reshape(original_shape)

        value = self.interpolate(new_points, method=method, **interp_kwargs)

        # If value id 1D, turn into 2D
        # Take into account if the 1D line of cells runs in x or y direction
        if 1 in shape:
            empty_axis = 0 if shape[0] == 1 else 1
            value = numpy.expand_dims(value, axis=empty_axis)

        nodata_value = self.nodata_value if self.nodata_value is not None else numpy.nan

        new_tile = Tile(
            alignment_grid,
            start_id=new_ids[-1, 0],
            nx=new_ids.shape[1],
            ny=new_ids.shape[0],
        )
        return DataTile(new_tile, value, nodata_value=nodata_value)

    def to_crs(self, crs, resample_method="nearest"):
        """Transforms the Coordinate Reference System (CRS) from the current CRS to the desired CRS.
        This will modify the cell size and the bounds accordingly.

        The ``crs`` attribute on the current grid must be set.

        Parameters
        ----------
        crs: Union[int, str, pyproj.CRS]
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an epsg integer (eg 4326), an authority string (eg "EPSG:4326") or a WKT string.
        resample_method: :class:`str`
            The resampling method to be used for :meth:`.DataTile.resample`.

        Returns
        -------
        :class:`~.rect_grid.BoundedRectGrid`
            A copy of the grid with modified cell spacing and bounds to match the specified CRS

        See also
        --------
        Examples:

        :ref:`Example: coordinate transformations <example coordinate transformations>`

        """
        new_inf_grid = self.grid.to_crs(crs)
        return self.resample(new_inf_grid, method=resample_method)

    def __add__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._add_tile(other._data_tile)
        else:
            try:

                dtype = get_value_dtype(other)
                # nocheckin, check other dtype with data dtype and convert in
                _data_tile = self._data_tile._add_scalar(other)
            except:
                raise TypeError(f"Cannot add DataTile and `{type(other)}`")

        combined = DataTile.from_pyo3_data_tile(self.grid, _data_tile)
        return combined

    def __radd__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._add_tile(other._data_tile)
        else:
            try:
                other = numpy.array(other, dtype=self.dtype)
                _data_tile = self._data_tile._add_scalar_reverse(other)
            except:
                raise TypeError(f"Cannot add DataTile and `{type(other)}`")

        combined = DataTile.from_pyo3_data_tile(self.grid, _data_tile)
        return combined

    def __sub__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._subtract_tile(other._data_tile)
        else:
            try:
                other = numpy.array(other, dtype=self.dtype)
                _data_tile = self._data_tile._subtract_scalar(other)
            except:
                raise TypeError(f"Cannot subtract DataTile and `{type(other)}`")
        combined = DataTile.from_pyo3_data_tile(self.grid, _data_tile)
        return combined

    def __rsub__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._subtract_tile(other._data_tile)
        else:
            try:
                other = numpy.array(other, dtype=self.dtype)
                _data_tile = self._data_tile._subtract_scalar_reverse(other)
            except:
                raise TypeError(f"Cannot subtract DataTile and `{type(other)}`")
        combined = DataTile.from_pyo3_data_tile(self.grid, _data_tile)
        return combined

    def __mul__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._multiply_tile(other._data_tile)
        else:
            try:
                other = numpy.array(other, dtype=self.dtype)
                _data_tile = self._data_tile._multiply_scalar(other)
            except:
                raise TypeError(f"Cannot multiply DataTile and `{type(other)}`")
        combined = DataTile.from_pyo3_data_tile(self.grid, _data_tile)
        return combined

    def __rmul__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._multiply_tile(other._data_tile)
        else:
            try:
                other = numpy.array(other, dtype=self.dtype)
                _data_tile = self._data_tile._multiply_scalar_reverse(other)
            except:
                raise TypeError(f"Cannot multiply DataTile and `{type(other)}`")
        combined = DataTile.from_pyo3_data_tile(self.grid, _data_tile)
        return combined

    def __truediv__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._divide_tile(other._data_tile)
        else:
            try:
                other = numpy.array(other, dtype=self.dtype)
                _data_tile = self._data_tile._divide_scalar(other)
            except:
                raise TypeError(f"Cannot divide DataTile and `{type(other)}`")
        combined = DataTile.from_pyo3_data_tile(self.grid, _data_tile)
        return combined

    def __rtruediv__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._divide_tile(other._data_tile)
        else:
            try:
                other = numpy.array(other, dtype=self.dtype)
                _data_tile = self._data_tile._divide_scalar_reverse(other)
            except:
                raise TypeError(f"Cannot divide DataTile and `{type(other)}`")
        combined = DataTile.from_pyo3_data_tile(self.grid, _data_tile)
        return combined

    def __pow__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError(
                "Elementwise raising to the power between two DataTiles is not supported."
            )
        elif isinstance(other, int):
            _data_tile = self._data_tile._powi(other)
        else:
            try:
                other = numpy.array(other, dtype=self.dtype)
                _data_tile = self._data_tile._powf(other)
            except:
                raise TypeError(f"Cannot divide DataTile and `{type(other)}`")
        combined = DataTile.from_pyo3_data_tile(self.grid, _data_tile)
        return combined

    def __rpow__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError(
                "Elementwise raising to the power between two DataTiles is not supported."
            )
        else:
            try:
                other = numpy.array(other, dtype=self.dtype)
                _data_tile = self._data_tile._powf_reverse(other)
            except:
                raise TypeError(f"Cannot divide DataTile and `{type(other)}`")
        combined = DataTile.from_pyo3_data_tile(self.grid, _data_tile)
        return combined

    def __eq__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError()
        try:
            other = numpy.array(other, dtype=self.dtype)
            return GridIndex(self._data_tile == other)
        except ValueError:
            raise TypeError(
                f"Cannot compare DataTile with object of type '{type(other)}'"
            )

    def __ne__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError()
        try:
            other = numpy.array(other, dtype=self.dtype)
            return GridIndex(self._data_tile != other)
        except ValueError:
            raise TypeError(
                f"Cannot compare DataTile with object of type '{type(other)}'"
            )

    def __ge__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError()
        try:
            other = numpy.array(other, dtype=self.dtype)
            return GridIndex(self._data_tile >= other)
        except ValueError:
            raise TypeError(
                f"Cannot compare DataTile with object of type '{type(other)}'"
            )

    def __gt__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError()
        try:
            other = numpy.array(other, dtype=self.dtype)
            return GridIndex(self._data_tile > other)
        except ValueError:
            raise TypeError(
                f"Cannot compare DataTile with object of type '{type(other)}'"
            )

    def __le__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError()
        try:
            other = numpy.array(other, dtype=self.dtype)
            return GridIndex(self._data_tile <= other)
        except ValueError:
            raise TypeError(
                f"Cannot compare DataTile with object of type '{type(other)}'"
            )

    def __lt__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError()
        try:
            other = numpy.array(other, dtype=self.dtype)
            return GridIndex(self._data_tile < other)
        except ValueError:
            raise TypeError(
                f"Cannot compare DataTile with object of type '{type(other)}'"
            )

    def max(self):
        """Get the maximum value in the data tile, disregarding the nodata_value."""
        return self._data_tile.max()

    def min(self):
        """Get the maximum value in the data tile, disregarding the nodata_value."""
        return self._data_tile.min()

    def mean(self):
        """Get the maximum value in the data tile, disregarding the nodata_value."""
        return self._data_tile.mean()

    def sum(self):
        """Get the sum of all values in the data tile, disregarding the nodata_value."""
        return self._data_tile.sum()

    def median(self):
        """Get the median value of the data tile, disregarding the nodata_value."""
        return self._data_tile.median()

    def percentile(self, percentile):
        """Get the percentile of the data tile at the specified value, disregarding the nodata_value."""
        return self._data_tile.percentile(percentile)

    def std(self):
        """Get the standard deviation of the data in the tile, disregarding the nodata_value."""
        return self._data_tile.std()


def combine_tiles(tiles):
    """Create a tile that covers all supplied tiles.

    Parameters
    ----------
    tiles: List[Tile]
        A list of tiles around which the combined tile will be created.

    Returns
    -------
    :class:`.Tile`
        A Tile that covers the supplied Tiles
    """
    pyo3_tiles = []
    for tile in tiles:
        if isinstance(tile, Tile):
            pyo3_tiles.append(tile._tile)
        elif isinstance(tile, DataTile):
            pyo3_tiles.append(tile._tile._tile)  # Man this nesting gets rediculous
        else:
            raise TypeError(
                f"Expected all Tile or DataTile objects but also got a: {type(tile)}"
            )
    pyo3_tile = tile_utils.combine_tiles(pyo3_tiles)
    return Tile.from_pyo3_tile(tiles[0].grid, pyo3_tile)


def count_tiles(tiles):
    """Count how many times a cell occurs in a tile for the list of tiles provided.
    Regions where many tiles overlap will have a high count, regions where few tiles
    overlap will have low count.

    Parameters
    ----------
    tiles: `List[Tile]` or `List[DataTile]`
        The tiles from which the overlap count will be determined.
        For each Tile that is supplied in the list, all cells in the tile contribute to the count.
        For each DataTile that is supplied in the list, the cells with a value equal to the
        nodata_value of the DataTile are ignored for the count.

    Returns
    -------
    :class:`.DataTile`
        A data tile where each value indicates by how many tiles this cell was covered.

    """
    pyo3_tiles = []
    pyo3_data_tiles = []
    for tile in tiles:
        # Note: check DataTile before Tile, because DataTile is a subclass of Tile
        if isinstance(tile, DataTile):
            pyo3_data_tiles.append(tile._data_tile)  # Man this nesting gets rediculous
        elif isinstance(tile, Tile):
            pyo3_tiles.append(tile._tile)
        else:
            raise TypeError(
                f"Expected all Tile or DataTile objects but also got a: {type(tile)}"
            )
    if pyo3_tiles:
        pyo3_data_tile = tile_utils.count_tiles(pyo3_tiles)
        result_tiles_only = DataTile.from_pyo3_data_tile(tiles[0].grid, pyo3_data_tile)
    if pyo3_data_tiles:
        pyo3_data_tile = tile_utils.count_data_tiles(pyo3_data_tiles)
        result_data_tiles_only = DataTile.from_pyo3_data_tile(
            tiles[0].grid, pyo3_data_tile
        )

    if pyo3_tiles and pyo3_data_tiles:
        result = result_tiles_only + result_data_tiles_only
        # Note: After summation, nodata values are added in the corners where no tiles are present
        #       if the DataTiles do not have the exact same coverage as the supplied Tiles.
        #       Since we want a count to say 0 where no tile is present, we replace the nans
        #       introduced by the summation with zeros.
        result[~numpy.isfinite(result)] = 0
        return result
    if pyo3_tiles:
        return result_tiles_only
    if pyo3_data_tiles:
        return result_data_tiles_only

    raise TypeError("No Tiles were found in the arguments")


def sum_data_tiles(data_tiles: List):
    """Add the DataTiles in the data_tiles list. Nodata_values will be ignored.
    Each cell will then contain the sum of the value that cell accross the supplied DataTiles.

    Parameters
    ----------
    data_tiles: `List[Tile]` or `List[DataTile]`
        A list of DataTiles to add together.

    Returns
    -------
    :class:`.DataTile`
        A data tile with the values of the provided data_tiles added together

    """
    if len(data_tiles) == 0:
        raise ValueError("No data tiles were supplied")

    grid = data_tiles[0].grid
    if not all([t.grid.is_aligned_with(grid) for t in data_tiles]):
        raise AlignmentError(
            "Not all data tiles are on the same grid. Consider resampling them all to the same grid."
        )

    result_dtype = numpy.result_type(*data_tiles)
    pyo3_tiles = []
    for tile in data_tiles:
        if isinstance(tile, DataTile):
            pyo3_tiles.append(tile.astype(result_dtype)._data_tile)
        else:
            raise TypeError(
                f"Expected all DataTile objects but also got a: {type(tile)}"
            )

    dtype_method_map = {
        numpy.dtype("float64"): tile_utils.sum_data_tile_f64,
        numpy.dtype("float32"): tile_utils.sum_data_tile_f32,
        numpy.dtype("int64"): tile_utils.sum_data_tile_i64,
        numpy.dtype("int32"): tile_utils.sum_data_tile_i32,
        numpy.dtype("int16"): tile_utils.sum_data_tile_i16,
        numpy.dtype("int8"): tile_utils.sum_data_tile_i8,
        numpy.dtype("uint64"): tile_utils.sum_data_tile_u64,
        numpy.dtype("uint32"): tile_utils.sum_data_tile_u32,
        numpy.dtype("uint16"): tile_utils.sum_data_tile_u16,
        numpy.dtype("uint8"): tile_utils.sum_data_tile_u8,
        # numpy.dtype('bool'):   'bool',  # optional support
        # FIXME: add complex version
    }

    method = dtype_method_map.get(result_dtype)
    if method is None:
        raise TypeError(f"Unsupported dtype: {result_dtype}")

    pyo3_data_tile = method(pyo3_tiles)
    return DataTile.from_pyo3_data_tile(grid, pyo3_data_tile)


def average_data_tiles(data_tiles: List):
    """Average the DataTiles in the data_tiles list. Nodata_values will be ignored.
    Each cell will then contain the mean or average value of that cell accross the supplied DataTiles.

    Parameters
    ----------
    data_tiles: `List[Tile]` or `List[DataTile]`
        A list of DataTiles to average.

    Returns
    -------
    :class:`.DataTile`
        A data tile with the averaged values of the provided data_tiles

    """
    # Simply calling `return sum(tiles) / count(tiles)` is a lot shorter, but I want to use the rust
    # logic such that the rust and python logic are the same and don't have possible inconsistencies
    # like inf instead of nan or vice versa. I want just one place for the logic.
    summed = sum_data_tiles(data_tiles)
    counted = count_tiles(data_tiles).astype(summed)
    return summed / counted
