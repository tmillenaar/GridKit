import warnings
from typing import Literal, Tuple, Union

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

    @staticmethod
    def from_bounds(grid, bounds):
        if not grid.are_bounds_aligned(bounds):
            raise ValueError(
                f"The supplied bounds are not aligned with the supplied grid. Consider calling 'grid.align_bounds' first."
            )
        bottom_left = (bounds[0] + grid.dx / 2, bounds[1] + grid.dy / 2)
        top_right = (bounds[2] + grid.dx / 2, bounds[3] - grid.dy / 2)

        # TODO: This still does not work nicely for rotated grids
        if grid.rotation != 0:
            bottom_left = grid.rotation_matrix_inv.dot(bottom_left)
            top_right = grid.rotation_matrix_inv.dot(top_right)
            tmp_grid = grid.update(rotation=0)
            bottom_left_cell = tmp_grid.cell_at_point(bottom_left)
            top_right_cell = tmp_grid.cell_at_point(top_right)
        else:
            bottom_left_cell = grid.cell_at_point(bottom_left)
            top_right_cell = grid.cell_at_point(top_right)

        # Take absolute values because rotated grids can cause flipped indices
        nx = abs(top_right_cell.x - bottom_left_cell.x)
        ny = abs(top_right_cell.y - bottom_left_cell.y)
        return Tile(grid, bottom_left_cell, nx, ny)

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

    @property
    def mpl_extent(self) -> tuple:
        """Raster Bounds

        Returns
        -------
        :class:`tuple`
            The extent of the data as defined expected by matplotlib in (left, right, bottom, top) or equivalently (min-x, max-x, min-y, max-y)
        """
        b = self.bounds
        return (b[0], b[2], b[1], b[3])

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


class DataTile(Tile):

    def __init__(self, tile: Tile, data: numpy.ndarray, nodata_value=numpy.nan):
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D array, got {data.ndim} dimensions")
        if tile.ny != data.shape[0] or tile.nx != data.shape[1]:
            raise ValueError(
                f"The shape of the data {data.shape} does not match the shape of the tile: {(tile.ny, tile.nx)}"
            )

        self.grid = tile.grid
        if nodata_value is None:
            nodata_value = numpy.nan
        self._data_tile = tile._tile.to_data_tile(data.astype("float64"), nodata_value)
        # _tile is used by the Tile parent class
        self._tile = self._data_tile.get_tile()
        self.nodata_value = nodata_value

    @staticmethod
    def from_PyO3DataTile(grid, pyo3_data_tile):
        tile = Tile(
            grid, pyo3_data_tile.start_id(), pyo3_data_tile.nx(), pyo3_data_tile.ny()
        )
        data_tile = DataTile(
            tile, pyo3_data_tile.to_numpy(), nodata_value=pyo3_data_tile.nodata_value()
        )
        return data_tile

    def to_numpy(self):
        return self._data_tile.to_numpy()

    def __array__(self, dtype=None):
        return self.to_numpy() if dtype is None else self.to_numpy().astype(dtype)

    def __getitem__(self, item):
        return self.to_numpy()[item]

    def __setitem__(self, item, value):
        # Don't use update() but replace _data_tile in-place
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
        self._data_tile = tile.to_data_tile(new_data, self.nodata_value)

    def update(
        self, grid=None, data=None, start_id=None, nx=None, ny=None, nodata_value=None
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
            oob_value = (
                self.nodata_value if not self.nodata_value is None else numpy.nan
            )
        oob_value = numpy.float64(oob_value)
        original_shape = index.shape
        result = self._data_tile.value(index.ravel().index, oob_value)
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
            return self.value(new_ids)
        elif method == "bilinear" or method == "linear":
            return_shape = sample_points.shape[:-1]
            result = self._linear_interpolation(sample_points.reshape(-1, 2))
            return result.reshape(return_shape)
        elif method == "inverse_distance":
            decay_constant = interp_kwargs.pop("decay_constant", 1)
            return self._inverse_distance_interpolation(sample_points, decay_constant)
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
        if self.grid.crs is None or alignment_grid.crs is None:
            warnings.warn(
                "`crs` not set for one or both grids. Assuming both grids have an identical CRS."
            )
            different_crs = False
        else:
            different_crs = not self.grid.crs.is_exact_same(alignment_grid.crs)

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

            # # Find the optimal new rotation
            # bounding_rect = MultiPoint(corners_transformed).minimum_rotated_rectangle
            # corners = numpy.array(bounding_rect.exterior.xy).T
            # distances = numpy.linalg.norm(corners[1:] - corners[:-1], axis=1)
            # longest_side_id = numpy.argmax(distances)
            # longest_line = numpy.array([corners[longest_side_id], corners[longest_side_id+1]])
            # horizontal_line = numpy.array([
            #     longest_line[0],
            #     [longest_line[1,0], longest_line[0,1]]
            # ])
            # # # angle_radians = numpy.arctan2(longest_line[1], horizontal_line[0])
            # # vector = longest_line[0] - longest_line[1]
            # # angle_radians = numpy.arctan2(vector[1], vector[0])
            # # angle_degrees = numpy.degrees(angle_radians)
            # #

            # new_points = numpy.array(transformer.transform([bottom_left[0], bottom_right[0]], [bottom_left[1], bottom_right[1]])).T
            # new_points = numpy.array(transformer.transform([bottom_left[0], top_left[0]], [bottom_left[1], top_left[1]])).T
            # vector = new_points[1] - new_points[0]
            # rotation = numpy.degrees(numpy.arctan2(vector[1], vector[0]))
            # alignment_grid = alignment_grid.update(rotation=rotation)
            # breakpoint()

            ids = alignment_grid.cell_at_point(corners_transformed).ravel()
        else:
            ids = alignment_grid.cell_at_point(self.corners()).ravel()
        min_x, min_y = numpy.min(ids, axis=0)
        max_x, max_y = numpy.max(ids, axis=0)

        tile = Tile(
            alignment_grid, (min_x, min_y), max_x - min_x + 1, max_y - min_y + 1
        )

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

        tile = Tile(
            alignment_grid,
            start_id=new_ids[-1, 0],
            nx=new_ids.shape[1],
            ny=new_ids.shape[0],
        )

        return DataTile(tile, value)

    def __add__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._add_tile(other._data_tile)
        else:
            try:
                other = float(other)
                _data_tile = self._data_tile._add_scalar(other)
            except:
                raise TypeError(f"Cannot add DataTile and `{type(other)}`")

        combined = DataTile.from_PyO3DataTile(self.grid, _data_tile)
        return combined

    def __radd__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._add_tile(other._data_tile)
        else:
            try:
                other = float(other)
                _data_tile = self._data_tile._add_scalar_reverse(other)
            except:
                raise TypeError(f"Cannot add DataTile and `{type(other)}`")

        combined = DataTile.from_PyO3DataTile(self.grid, _data_tile)
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

    def __rsub__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._subtract_tile(other._data_tile)
        else:
            try:
                other = float(other)
                _data_tile = self._data_tile._subtract_scalar_reverse(other)
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

    def __rmul__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._multiply_tile(other._data_tile)
        else:
            try:
                other = float(other)
                _data_tile = self._data_tile._multiply_scalar_reverse(other)
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

    def __rtruediv__(self, other):
        if isinstance(other, DataTile):
            _data_tile = self._data_tile._divide_tile(other._data_tile)
        else:
            try:
                other = float(other)
                _data_tile = self._data_tile._divide_scalar_reverse(other)
            except:
                raise TypeError(f"Cannot divide DataTile and `{type(other)}`")
        combined = self.update()
        combined._data_tile = _data_tile
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
                other = float(other)
                _data_tile = self._data_tile._powf(other)
            except:
                raise TypeError(f"Cannot divide DataTile and `{type(other)}`")
        combined = self.update()
        combined._data_tile = _data_tile
        return combined

    def __rpow__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError(
                "Elementwise raising to the power between two DataTiles is not supported."
            )
        else:
            try:
                other = float(other)
                _data_tile = self._data_tile._powf_reverse(other)
            except:
                raise TypeError(f"Cannot divide DataTile and `{type(other)}`")
        combined = self.update()
        combined._data_tile = _data_tile
        return combined

    def __eq__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError()
        try:
            other = float(other)
            return GridIndex(self._data_tile == other)
        except ValueError:
            raise TypeError(
                f"Cannot compare DataTile with object of type '{type(other)}'"
            )

    def __ne__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError()
        try:
            other = float(other)
            return GridIndex(self._data_tile != other)
        except ValueError:
            raise TypeError(
                f"Cannot compare DataTile with object of type '{type(other)}'"
            )

    def __ge__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError()
        try:
            other = float(other)
            return GridIndex(self._data_tile >= other)
        except ValueError:
            raise TypeError(
                f"Cannot compare DataTile with object of type '{type(other)}'"
            )

    def __gt__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError()
        try:
            other = float(other)
            return GridIndex(self._data_tile > other)
        except ValueError:
            raise TypeError(
                f"Cannot compare DataTile with object of type '{type(other)}'"
            )

    def __le__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError()
        try:
            other = float(other)
            return GridIndex(self._data_tile <= other)
        except ValueError:
            raise TypeError(
                f"Cannot compare DataTile with object of type '{type(other)}'"
            )

    def __lt__(self, other):
        if isinstance(other, DataTile):
            raise NotImplementedError()
        try:
            other = float(other)
            return GridIndex(self._data_tile < other)
        except ValueError:
            raise TypeError(
                f"Cannot compare DataTile with object of type '{type(other)}'"
            )

    def max(self):
        return self._data_tile.max()

    def min(self):
        return self._data_tile.min()

    def mean(self):
        return self._data_tile.mean()

    def sum(self):
        return self._data_tile.sum()

    def median(self):
        return self._data_tile.median()

    def percentile(self, percentile):
        return self._data_tile.percentile(percentile)

    def std(self):
        return self._data_tile.std()
