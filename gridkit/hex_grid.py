import warnings
from typing import Literal

import numpy
from pyproj import CRS, Transformer

from gridkit.base_grid import BaseGrid
from gridkit.bounded_grid import BoundedGrid
from gridkit.errors import AlignmentError, IntersectionError
from gridkit.gridkit_rs import PyHexGrid, interp
from gridkit.index import GridIndex, validate_index
from gridkit.rect_grid import RectGrid


class HexGrid(BaseGrid):
    """Abstraction that represents an infinite grid with hexagonal cell shape.

    Initialization parameters
    -------------------------
    size: :class:`float`
        The spacing between two cells in horizontal direction if ``shape`` is "pointy",
        or in vertical direction if ``shape`` is "flat".
    shape: `Literal["pointy", "flat"]`
        The shape of the layout of the grid.
        If ``shape`` is "pointy" the cells will be pointy side up and the regular axis will be in horizontal direction.
        If ``shape`` is "flat", the cells will be flat side up and the regular axis will be in vertical direction.
    offset: `Tuple(float, float)` (optional)
        The offset in dx and dy.
        Shifts the whole grid by the specified amount.
        The shift is always reduced to be maximum one cell size.
        If the supplied shift is larger,
        a shift will be performed such that the new center is a multiple of dx or dy away.
        Default: (0,0)
    crs: `pyproj.CRS` (optional)
        The coordinate reference system of the grid.
        The value can be anything accepted by pyproj.CRS.from_user_input(),
        such as an epsg integer (eg 4326), an authority string (eg “EPSG:4326”) or a WKT string.
        Default: None

    """

    def __init__(
        self, *args, size, shape="pointy", offset=(0, 0), rotation=0, **kwargs
    ):
        self._size = size
        self._radius = size / 3**0.5
        self._rotation = rotation if shape == "pointy" else -rotation

        if shape == "pointy":
            self._dx = size
            self._dy = 3 / 2 * self._radius
        elif shape == "flat":
            self._dy = size
            self._dx = 3 / 2 * self._radius
        else:
            raise ValueError(
                f"A HexGrid's `shape` can either be 'pointy' or 'flat', got '{shape}'"
            )

        offset_x, offset_y = offset[0], offset[1]
        offset_x = offset_x % self.dx
        offset_y = offset_y % self.dy
        offset = (offset_x, offset_y)

        self._shape = shape
        self._grid = PyHexGrid(cellsize=size, offset=offset, rotation=self._rotation)
        self.bounded_cls = BoundedHexGrid
        super(HexGrid, self).__init__(*args, **kwargs)

    @property
    def dx(self) -> float:
        """The spacing between cell centers in x-direction"""
        return self._dx

    @property
    def dy(self) -> float:
        """The spacing between cell centers in y-direction"""
        return self._dy

    @property
    def r(self) -> float:
        """The radius of the cell. The radius is defined to be the distance from the cell center to a cell corner."""
        return self._radius

    @property
    def shape(self) -> str:
        """The shape of the grid as supplied when initiating the class.
        This can be either "flat" or "pointy" referring to the top of the cells.
        """
        return self._shape

    @shape.setter
    def shape(self, value):
        """Set the shape of the grid to a new value. Possible values: 'pointy' or 'flat'"""
        if not value in ("pointy", "flat"):
            raise ValueError(
                f"Shape cannot be set to '{value}', must be either 'pointy' or 'flat'"
            )
        rot = self.rotation
        self._shape = value
        self.rotation = (
            rot  # Re-run rotation settter to update rotaiton according to new shape
        )

    @property
    def size(self) -> float:
        """The size of the cell as supplied when initiating the class.
        This is the same as dx for a flat grid and the same as dy for a pointy grid.
        """
        return self._size

    @size.setter
    def size(self, value):
        """Set the size of the grid to a new value"""
        if value <= 0:
            raise ValueError(
                f"Size of cell cannot be set to '{value}', must be larger than zero"
            )
        self._size = value
        self._grid = self._update_inner_grid(size=value)

    def to_bounded(self, bounds, fill_value=numpy.nan):
        _, shape = self.cells_in_bounds(bounds, return_cell_count=True)
        data = numpy.full(shape=shape, fill_value=fill_value)
        return self.bounded_cls(
            data=data,
            bounds=bounds,
            nodata_value=fill_value,
            shape=self.shape,
            crs=self.crs,
        )

    @validate_index
    def relative_neighbours(
        self, index, depth=1, include_selected=False, connect_corners=False
    ) -> numpy.ndarray:
        """The relative indices of the neighbouring cells.

        Parameters
        ----------
        depth: :class:`int` Default: 1
            Determines the number of neighbours that are returned.
            If `depth=1` the direct neighbours are returned.
            If `depth=2` the direct neighbours are returned, as well as the neighbours of these neighbours.
            `depth=3` returns yet another layer of neighbours, and so forth.
        index: `numpy.ndarray`
            The index of the cell of which the relative neighbours are desired.
            This is mostly relevant because in hexagonal grids the neighbouring indices differ
            when dealing with odd or even indices.
        include_selected: :class:`bool` Default: False
            Whether to include the specified cell in the return array.
            Even though the specified cell can never be a neighbour of itself,
            this can be useful when for example a weighted average of the neighbours is desired
            in which case the cell itself often should also be included.
        connect_corners: :class:`bool` Default: False
            Whether to consider cells that touch corners but not sides as neighbours.
            This is not relevant in hexagonal grids. It does nothing here.
            See :py:meth:`.RectGrid.relative_neighbours`


        Examples
        --------
        The direct neighbours of a cell can be returned by using depth=1, which is the default.
        For hexagonal grids, the relative indices of the neighbours differs depending on the index.
        There are two cases, neighbour indices for even colums and neighbour indices for odd columns,
        in the case of a grid 'pointy' shape.
        This works on rows if the grid has a 'flat' shape.

        .. code-block:: python

            >>> from gridkit.hex_grid import HexGrid
            >>> grid = HexGrid(size=3)
            >>> grid.relative_neighbours(index=[0,0]).index
            array([[-1,  1],
                   [ 0,  1],
                   [-1,  0],
                   [ 1,  0],
                   [-1, -1],
                   [ 0, -1]])
            >>> grid.relative_neighbours(index=[0,1]).index
            array([[ 0,  1],
                   [ 1,  1],
                   [-1,  0],
                   [ 1,  0],
                   [ 0, -1],
                   [ 1, -1]])

        ..

        By specifying `depth` we can include indirect neighbours from further away.
        The number of neighbours increases with depth by a factor of `depth*6`.
        So the 3rd element in the list will be `1*6 + 2*6 + 3*6 = 36`.

        .. code-block:: python

            >>> [len(grid.relative_neighbours(index=[0,0], depth=depth)) for depth in range(1,5)]
            [6, 18, 36, 60]

        ..

        The specified cell can be included if `include_selected` is set to True:

        .. code-block:: python

            >>> grid.relative_neighbours(index=[0,0], include_selected=True).index
            array([[-1,  1],
                   [ 0,  1],
                   [-1,  0],
                   [ 0,  0],
                   [ 1,  0],
                   [-1, -1],
                   [ 0, -1]])

        ..

        See also
        --------
        :py:meth:`.BaseGrid.neighbours`
        :py:meth:`.RectGrid.relative_neighbours`
        :py:meth:`.TriGrid.relative_neighbours`
        """

        if depth < 1:
            raise ValueError("'depth' cannot be lower than 1")

        original_shape = index.shape
        index = numpy.array(index.ravel())
        if index.ndim == 1:
            index = index[None]

        nr_neighbours = (
            sum(6 * numpy.arange(1, depth + 1)) + 1
        )  # Add 1 for the first cell
        nr_indices = len(index)
        neighbours = numpy.empty((nr_indices, nr_neighbours, 2), dtype=int)
        start_slice = 0
        rows = range(depth, -1, -1)

        # create top half of selection
        for i, row in enumerate(rows):  # loop from top row to bottom row
            row_length = depth + i + 1
            row_slice = slice(start_slice, start_slice + row_length)
            max_val = int(numpy.floor(row_length / 2))
            if self._shape == "pointy":
                pointy_axis = 1
                flat_axis = 0
            elif self._shape == "flat":
                pointy_axis = 0
                flat_axis = 1

            if (i % 2 == 0) == (depth % 2 == 0):
                neighbours[:, row_slice, flat_axis] = range(-max_val, max_val + 1)
            else:
                odd_mask = index[:, pointy_axis] % 2 != 0
                neighbours[odd_mask, row_slice, flat_axis] = range(
                    -max_val + 1, max_val + 1
                )
                neighbours[~odd_mask, row_slice, flat_axis] = range(-max_val, max_val)
            neighbours[:, row_slice, pointy_axis] = row
            start_slice += row_length

        # mirror top half to bottom half (leaving the center row be)
        neighbours[:, start_slice:] = neighbours[:, 0 : start_slice - row_length][::-1]
        neighbours[:, start_slice:, pointy_axis] *= -1

        if include_selected is False:
            center_cell = int(numpy.floor(neighbours.shape[1] / 2))
            neighbours = numpy.delete(neighbours, center_cell, 1)

        neighbours = neighbours.reshape(*original_shape, *neighbours.shape[-2:])
        return GridIndex(neighbours.squeeze())

    @validate_index
    def centroid(self, index=None):
        """Coordinates at the center of the cell(s) specified by `index`.

        Parameters
        ----------
        index: :class:`tuple`
            Index of the cell of which the centroid is to be calculated.
            The index consists of two integers specifying the nth cell in x- and y-direction.
            Multiple indices can be specified at once in the form of a list of indices or an Nx2 ndarray,
            where N is the number of cells.

        Returns
        -------
        `numpy.ndarray`
            The longitude and latitude of the center of each cell.
            Axes order if multiple indices are specified: (points, xy), else (xy).

        Raises
        ------
        ValueError
            No `index` parameter was supplied. `index` can only be `None` in classes that contain data.

        Examples
        --------
        Cell centers of single index are returned as an array with one dimention:

        .. code-block:: python

            >>> grid = RectGrid(dx=4, dy=1)
            >>> grid.centroid((0, 0))
            array([2. , 0.5])
            >>> grid.centroid((-1, -1))
            array([-2. , -0.5])

        ..

        Multiple cell indices can be supplied as a list of tuples or as an equivalent ndarray:

        .. code-block:: python

            >>> grid.centroid([(0, 0), (-1, -1)])
            array([[ 2. ,  0.5],
                   [-2. , -0.5]])
            >>> ids = numpy.array([[0, 0], [-1, -1]])
            >>> grid.centroid(ids)
            array([[ 2. ,  0.5],
                   [-2. , -0.5]])

        ..

        Note that the center coordinate of the cell is also dependent on the grid's offset:

        .. code-block:: python

            >>> grid = RectGrid(dx=4, dy=1, offset = (1, 0.5))
            >>> grid.centroid((0, 0))
            array([3., 1.])

        ..


        """
        if index is None:
            raise ValueError(
                "For grids that do not contain data, argument `index` is to be supplied to method `centroid`."
            )
        original_shape = (*index.shape, 2)
        index = (
            index.ravel().index[None] if index.index.ndim == 1 else index.ravel().index
        )
        if self.shape == "flat":
            index = index.T[::-1].T
        centroids = self._grid.centroid(index=index)
        if self.shape == "flat":
            centroids = centroids.T[::-1].T
        return centroids.reshape(original_shape)

    def cells_near_point(self, point):
        """Nearest 3 cells around a point.
        This includes the cell the point is contained within,
        as well as two direct neighbours of this cell and one diagonal neighbor.
        What neighbours of the containing are selected, depends on where in the cell the point is located.

        Parameters
        ----------
        point: `tuple`
            Coordinate of the point around which the cells are to be selected.
            The point consists of two floats specifying x- and y-coordinates, respectively.
            Multiple points can be specified at once in the form of a list of points or an Nx2 ndarray.

        Returns
        -------
        :class:`.GridIndex`
            The indices of the 4 nearest cells in order (top-left, top-right, bottom-left, bottom-right).
            If a single point is supplied, the four indices are returned as 1d arrays of length 2.
            If multiple points are supplied, the four indices are returned as Nx2 ndarrays.

        """
        point = numpy.array(point, dtype=float)
        original_shape = (*point.shape[:-1], 3, 2)
        point = point[None] if point.ndim == 1 else point
        point = point.reshape(-1, 2)
        if self.shape == "flat":
            point = point.T[::-1].T
        ids = self._grid.cells_near_point(point)
        if self.shape == "flat":
            ids = ids.T[::-1].T
        return GridIndex(ids.squeeze().reshape(original_shape))

    def cell_at_point(self, point):
        """Index of the cell containing the supplied point(s).

        Parameters
        ----
        point: :class:`tuple`
            Coordinate of the point for which the containing cell is to be found.
            The point consists of two floats specifying x- and y-coordinates, respectively.
            Mutliple poins can be specified at once in the form of a list of points or an Nx2 ndarray.

        Returns
        -------
        `numpy.ndarray`
            The index of the cell containing the point(s).
            If a single point is supplied, the index is returned as a 1d array of length 2.
            If multiple points are supplied, the indices are returned as Nx2 ndarrays.

        """
        point = numpy.array(point, dtype=float)
        original_shape = point.shape
        point = point[None] if point.ndim == 1 else point
        point = point.reshape(-1, 2)
        if self.shape == "flat":
            point = point.T[::-1].T
        cell_at_point = self._grid.cell_at_location(points=point)
        if self.shape == "flat":
            cell_at_point = cell_at_point.T[::-1].T
        return GridIndex(cell_at_point.squeeze().reshape(original_shape))

    @validate_index
    def cell_corners(self, index: numpy.ndarray = None) -> numpy.ndarray:
        if index is None:
            raise ValueError(
                "For grids that do not contain data, argument `index` is to be supplied to method `centroid`."
            )
        return_shape = (*index.shape, 6, 2)
        index = (
            index.ravel().index[None] if index.index.ndim == 1 else index.ravel().index
        )
        if self.shape == "flat":
            index = index.T[::-1].T
        corners = self._grid.cell_corners(index=index)
        if self.shape == "flat":
            corners = corners[:, :, ::-1]
        return corners.reshape(return_shape)

    def to_crs(self, crs):
        """Transforms the Coordinate Reference System (CRS) from the current CRS to the desired CRS.
        This will update the cell size and the origin offset.

        The ``crs`` attribute on the current grid must be set.

        Parameters
        ----------
        crs: Union[int, str, pyproj.CRS]
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an epsg integer (eg 4326), an authority string (eg "EPSG:4326") or a WKT string.

        Returns
        -------
        :class:`~.hex_grid.HexGrid`
            A copy of the grid with modified cell spacing to match the specified CRS

        See also
        --------
        Examples:

        :ref:`Example: coordinate transformations <example coordinate transformations>`

        Methods:

        :meth:`.RectGrid.to_crs`
        :meth:`.BoundedRectGrid.to_crs`
        :meth:`.BoundedHexGrid.to_crs`

        """
        if self.crs is None:
            raise ValueError(
                "Cannot transform naive grids.  "
                "Please set a crs on the object first."
            )

        crs = CRS.from_user_input(crs)

        # skip if the input CRS and output CRS are the exact same
        if self.crs.is_exact_same(crs):
            return self

        transformer = Transformer.from_crs(self.crs, crs, always_xy=True)

        new_offset = transformer.transform(*self.offset)
        point_start = transformer.transform(0, 0)

        point_end = transformer.transform(self.dx, self.dy)
        new_dx, new_dy = [end - start for (end, start) in zip(point_end, point_start)]

        if self.shape == "pointy":
            size = new_dx
        elif self.shape == "flat":
            size = new_dy

        return self.parent_grid_class(size=size, offset=new_offset, crs=crs)

    def cells_in_bounds(self, bounds, return_cell_count: bool = False):
        """Cells contained within a bounding box.

        Parameters
        ----------
        bounds: :class:`tuple`
            The bounding box in which to find the cells in (min_x, min_y, max_x, max_y)
        return_cell_count: :class:`bool`
            Return a tuple containing the nr of cells in x and y direction inside the provided bounds

        Returns
        -------
        :class:`.GridIndex`
            The indices of the cells contained in the bounds
        """
        # TODO: Simplify function. Conceptually hard to follow and not very DRY
        if self.rotation != 0:
            raise NotImplementedError(
                f"`cells_in_bounds` is not suppored for rotated grids. Roatation: {self.rotation} degrees"
            )

        if not self.are_bounds_aligned(bounds):
            raise ValueError(
                f"supplied bounds '{bounds}' are not aligned with the grid lines. Consider calling 'align_bounds' first."
            )

        # get coordinates of two diagonally opposing corner-cells
        left_top = (bounds[0] + self.dx / 4, bounds[3] - self.dy / 4)
        left_bottom = (bounds[0] + self.dx / 4, bounds[1] + self.dy / 4)
        right_top = (bounds[2] - self.dx / 4, bounds[3] - self.dy / 4)
        right_bottom = (bounds[2] - self.dx / 4, bounds[1] + self.dy / 4)

        # translate the coordinates of the corner cells into indices
        left_top_id, left_bottom_id, right_top_id, right_bottom_id = self.cell_at_point(
            [left_top, left_bottom, right_top, right_bottom]
        ).index

        if self._shape == "pointy":
            nr_cells_flat = round(((bounds[2] - bounds[0]) / self.dx))
        elif self._shape == "flat":
            nr_cells_flat = round(((bounds[3] - bounds[1]) / self.dy))

        ids_x = numpy.arange(left_bottom_id[0] - 2, right_top_id[0] + 2)
        ids_y = numpy.arange(left_bottom_id[1] - 2, right_top_id[1] + 2)
        if self._shape == "flat":
            ids_x_full, ids_y_full = numpy.meshgrid(ids_x, ids_y, indexing="ij")
            ids_y_full = numpy.fliplr(ids_y_full)
        else:
            ids_x_full, ids_y_full = numpy.meshgrid(ids_x, ids_y, indexing="xy")
            ids_y_full = numpy.flipud(ids_y_full)
        ids = numpy.vstack([ids_x_full.ravel(), ids_y_full.ravel()]).T

        # determine what cells are outside of bounding box
        centroids = self.centroid(ids).T
        error_margin = numpy.finfo(
            numpy.float32
        ).eps  # expect problems with machine precision since some cells are on the bounds by design
        oob_mask = centroids[0] < (bounds[0] - error_margin)
        oob_mask |= centroids[1] < (bounds[1] - error_margin)
        oob_mask |= centroids[0] >= (bounds[2] - error_margin)
        oob_mask |= centroids[1] >= (bounds[3] - error_margin)

        ids = ids[~oob_mask]

        shape = (
            (int(numpy.ceil(ids.shape[0] / nr_cells_flat)), nr_cells_flat)
            if nr_cells_flat != 0
            else (0, 0)
        )

        ids = GridIndex(ids.reshape((*shape, 2)))

        return (ids, shape) if return_cell_count else ids

    @property
    def parent_grid_class(self):
        return HexGrid

    def _update_inner_grid(self, size=None, offset=None, rotation=None):
        if size is None:
            size = self.size
        if offset is None:
            offset = self.offset
        if rotation is None:
            rotation = self.rotation
        return PyHexGrid(cellsize=size, offset=offset, rotation=rotation)

    def update(
        self, size=None, shape=None, offset=None, rotation=None, crs=None, **kwargs
    ):
        if size is None:
            size = self.size
        if shape is None:
            shape = self.shape
        if offset is None:
            offset = self.offset
        if rotation is None:
            rotation = self.rotation
        if crs is None:
            crs = self.crs
        return HexGrid(
            size=size, shape=shape, offset=offset, rotation=rotation, crs=crs, **kwargs
        )


class BoundedHexGrid(BoundedGrid, HexGrid):
    """
    Initialization parameters
    -------------------------
    data: `numpy.ndarray`
        A 2D ndarray containing the data
    bounds: `Tuple(float, float, float, float)`
        The extend of the data in minx, miny, maxx, maxy.
    shape: `Literal["pointy", "flat"]`
        The shape of the cells in the grid.
        If ``shape`` is "pointy", the hexagons will be standing upright on a point with the flat sides to the left and right.
        If ``shape`` is "flat", the hexagons will be flat side up and below and pointy side on the left and right.
    crs: `pyproj.CRS` (optional)
        The coordinate reference system of the grid.
        The value can be anything accepted by pyproj.CRS.from_user_input(),
        such as an epsg integer (eg 4326), an authority string (eg “EPSG:4326”) or a WKT string.
        Default: None
    """

    def __init__(self, data, *args, bounds, shape="flat", **kwargs):
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError(
                f"Incerrect bounds. Minimum value exceeds maximum value for bounds {bounds}"
            )

        if data.ndim != 2:
            raise ValueError(
                f"Expected a 2D numpy array, got data with shape {data.shape}"
            )

        if shape == "pointy":
            dx = (bounds[2] - bounds[0]) / data.shape[1]
            dy = (bounds[3] - bounds[1]) / data.shape[0]
            size = dx
            if not numpy.isclose(size / 2, dy / 3**0.5):
                raise ValueError(
                    "The supplied data cannot be covered by hexagons with sides of equal length."
                )
        elif shape == "flat":
            dx = (bounds[2] - bounds[0]) / data.shape[0]
            dy = (bounds[3] - bounds[1]) / data.shape[1]
            size = dy
            if not numpy.isclose(size / 2, dx / 3**0.5):
                raise ValueError(
                    "The supplied data cannot be covered by hexagons with sides of equal length."
                )

        offset_x = bounds[0] % dx
        offset_y = bounds[1] % dy
        offset_x = dx - offset_x if offset_x < 0 else offset_x
        offset_y = dy - offset_y if offset_y < 0 else offset_y
        offset = (
            0 if numpy.isclose(offset_x, dx) else offset_x,
            0 if numpy.isclose(offset_y, dy) else offset_y,
        )
        super(BoundedHexGrid, self).__init__(
            data, *args, size=size, bounds=bounds, offset=offset, shape=shape, **kwargs
        )

        if not self.are_bounds_aligned(bounds):
            raise AlignmentError(
                "Something went wrong, the supplied bounds are not aligned with the resulting grid."
            )

    @property
    def height(self):
        """Raster height

        Returns
        -------
        :class:`int`
            The number of grid cells in y-direction
        """
        return self.data.shape[1] if self.shape == "flat" else self.data.shape[0]

    @property
    def width(self):
        """Raster width

        Returns
        -------
        :class:`int`
            The number of grid cells in x-direction
        """
        return self.data.shape[0] if self.shape == "flat" else self.data.shape[1]

    @property
    def nr_cells(self):
        return (self.width, self.height)

    @property
    def lon(self):
        """Array of long values

        Returns
        -------
        `numpy.ndarray`
            1D-Array of size `width`, containing the longitudinal values from left to right
        """
        return numpy.linspace(
            self.bounds[0] + self.dx / 2, self.bounds[2] - self.dx / 2, self.width
        )

    @property
    def lat(self):
        """Array of lat values

        Returns
        -------
        `numpy.ndarray`
            1D-Array of size `height`, containing the latitudinal values from top to bottom
        """
        return numpy.linspace(
            self.bounds[3] - self.dy / 2, self.bounds[1] + self.dy / 2, self.height
        )

    def intersecting_cells(self, other):
        raise NotImplementedError()

    def crop(self, new_bounds, bounds_crs=None, buffer_cells=0):
        if bounds_crs is not None:
            bounds_crs = CRS.from_user_input(bounds_crs)
            transformer = Transformer.from_crs(bounds_crs, self.crs, always_xy=True)
            new_bounds = transformer.transform_bounds(*new_bounds)

        if not self.intersects(new_bounds):
            raise IntersectionError(
                f"Cannot crop grid with bounds {self.bounds} to {new_bounds} for they do not intersect."
            )
        new_bounds = self.shared_bounds(new_bounds)

        new_bounds = self.align_bounds(new_bounds, mode="contract")
        slice_y, slice_x = self._data_slice_from_bounds(new_bounds)
        if buffer_cells:
            slice_x = slice(slice_x.start - buffer_cells, slice_x.stop + buffer_cells)
            slice_y = slice(slice_y.start - buffer_cells, slice_y.stop + buffer_cells)
            new_bounds = (
                new_bounds[0] - buffer_cells * self.dx,
                new_bounds[1] - buffer_cells * self.dy,
                new_bounds[2] + buffer_cells * self.dx,
                new_bounds[3] + buffer_cells * self.dy,
            )
        # cropped_data = numpy.flipud(numpy.flipud(self._data)[slice_y, slice_x]) # TODO: fix this blasted flipping. The raster should not be stored upside down maybe
        cropped_data = self._data[slice_y, slice_x]  # Fixme: seems to be flipped?
        # cropped_data = self._data[slice_x, slice_y]
        return self.update(cropped_data, bounds=new_bounds)

    def _data_slice_from_bounds(self, bounds):
        slice_y, slice_x = super()._data_slice_from_bounds(bounds=bounds)
        return (slice_y, slice_x) if self.shape == "pointy" else (slice_x, slice_y)

    def cell_corners(self, index: numpy.ndarray = None) -> numpy.ndarray:
        if index is None:
            index = self.indices()
        return super(BoundedHexGrid, self).cell_corners(index=index)

    def to_shapely(self, index=None, as_multipolygon: bool = False):
        """Refer to :meth:`.BaseGrid.to_shapely`

        Difference with :meth:`.BaseGrid.to_shapely`:
            `index` is optional.
            If `index` is None (default) the cells containing data are used as the `index` argument.

        See also
        --------
        :meth:`.BaseGrid.to_shapely`
        :meth:`.BoundedRectGrid.to_shapely`
        """
        if index is None:
            index = self.indices
        return super().to_shapely(index, as_multipolygon)

    def _bilinear_interpolation(self, sample_points):
        """Interpolate the value at the location of `sample_points` by doing a bilinear interpolation
        using the 4 cells around the point.

        Parameters
        ----------
        sample_points: `numpy.ndarray`
            The coordinates of the points at which to sample the data

        Returns
        -------
        `numpy.ndarray`
            The interpolated values at the supplied points

        See also
        --------
        :py:meth:`.RectGrid.cell_at_point`
        """
        if not isinstance(sample_points, numpy.ndarray):
            sample_points = numpy.array(sample_points)
        original_shape = sample_points.shape
        sample_points = sample_points.reshape(-1, 2)

        all_nearby_cells = self.cells_near_point(
            sample_points
        )  # (points, nearby_cells, xy)
        nearby_centroids = self.centroid(all_nearby_cells)
        weights = interp.linear_interp_weights_triangles(
            sample_points, nearby_centroids
        )
        values = numpy.sum(weights * self.value(all_nearby_cells), axis=1)
        # TODO: remove rows and cols with nans around the edge after bilinear
        return values.reshape(*original_shape[:-1])

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
            The resampling method to be used for :meth:`.BoundedGrid.resample`.

        Returns
        -------
        :class:`~.hex_grid.BoundedHexGrid`
            A copy of the grid with modified cell spacing and bounds to match the specified CRS

        See also
        --------
        Examples:

        :ref:`Example: coordinate transformations <example coordinate transformations>`

        Methods:

        :meth:`.RectGrid.to_crs`
        :meth:`.BoundedRectGrid.to_crs`
        :meth:`.HexGrid.to_crs`

        """
        new_inf_grid = super(BoundedHexGrid, self).to_crs(crs)
        return self.resample(new_inf_grid, method=resample_method)

    def numpy_id_to_grid_id(self, np_index):
        """Turn numpy indices that select values from self.data into a GridIndex instance that represents those same cells and can be used on the Grid object.

        Parameters
        ----------
        np_index: `numpy.ndarray`
            The numpy indices to convert

        Returns
        -------
        :class:`~.index.GridIndex`
            The GridIndex representation of the cells
        """
        centroid_topleft = (self.bounds[0] + self.dx / 2, self.bounds[3] - self.dy / 2)
        index_topleft = self.cell_at_point(centroid_topleft)

        if self._shape == "pointy":
            index = numpy.array(
                [
                    index_topleft.x + np_index[1],
                    index_topleft.y
                    - np_index[
                        0
                    ],  # grid y is numpy [0] and is positive from top to bottom
                ]
            )
            offset_rows = index[1] % 2 == 1
            index[0][offset_rows] -= 1
        elif self._shape == "flat":
            index = numpy.array(
                [
                    index_topleft.x + np_index[0],
                    index_topleft.y - np_index[1],
                ]
            )
            offset_rows = index[0] % 2 == 1
            index[1, offset_rows] -= 1

        return GridIndex(index.T)

    @validate_index
    def grid_id_to_numpy_id(self, index):
        if index.index.ndim > 2:
            raise ValueError(
                "Cannot convert nd-index to numpy index. Consider flattening the index using `index.ravel()`"
            )
        if self._shape == "pointy":
            offset_rows = index.y % 2 == 1
            index.x[offset_rows] += 1
        elif self._shape == "flat":
            offset_rows = index.x % 2 == 1
            index.y[offset_rows] += 1

        centroid_topleft = (self.bounds[0] + self.dx / 2, self.bounds[3] - self.dy / 2)
        index_topleft = self.cell_at_point(centroid_topleft)
        if self._shape == "pointy":
            np_id = (index_topleft.y - index.y, index.x - index_topleft.x)
        elif self._shape == "flat":
            np_id = ((index.x - index_topleft.x), (index_topleft.y - index.y))
        return np_id

    def interp_nodata(self, *args, **kwargs):
        """Please refer to :func:`~gridkit.bounded_grid.BoundedGrid.interp_nodata`."""
        # Fixme: in the case of a rectangular grid, a performance improvement can be obtained by using scipy.interpolate.interpn
        return super(BoundedHexGrid, self).interp_nodata(*args, **kwargs)

    def centroid(self, index=None):
        if index is None:
            if not hasattr(self, "indices"):
                raise ValueError(
                    "For grids that do not contain data, argument `index` is to be supplied to method `centroid`."
                )
            index = self.indices
        return super(BoundedHexGrid, self).centroid(index)

    def update(self, new_data, bounds=None, crs=None, nodata_value=None, shape=None):
        # TODO figure out how to update size, offset
        if not bounds:
            bounds = self.bounds
        if not crs:
            crs = self.crs
        if not nodata_value:
            nodata_value = self.nodata_value
        if not shape:
            shape = self.shape
        return self.__class__(
            new_data, bounds=bounds, crs=crs, nodata_value=nodata_value, shape=shape
        )
