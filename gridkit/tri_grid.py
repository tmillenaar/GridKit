import numpy
import shapely
from pyproj import CRS, Transformer

from gridkit.base_grid import BaseGrid
from gridkit.bounded_grid import BoundedGrid
from gridkit.errors import AlignmentError, IntersectionError
from gridkit.gridkit_rs import PyTriGrid
from gridkit.index import GridIndex, validate_index


class TriGrid(BaseGrid):
    def __init__(self, *args, size, offset=(0, 0), rotation=0, **kwargs):
        self._size = size
        self._radius = size / 3**0.5
        self._rotation = rotation
        self._grid = PyTriGrid(cellsize=size, offset=offset, rotation=rotation)

        self.bounded_cls = BoundedTriGrid
        super(TriGrid, self).__init__(*args, **kwargs)

    @property
    def dx(self) -> float:
        """The spacing between cell centers in x-direction"""
        return self._grid.dx()

    @property
    def dy(self) -> float:
        """The spacing between cell centers in y-direction"""
        return self._grid.dy()

    @property
    def r(self) -> float:
        """The radius of the cell. The radius is defined to be the distance from the cell center to a cell corner."""
        return self._grid.radius()

    @property
    def size(self) -> float:
        """The size of the cell as supplied when initiating the class.
        The size is equivalent to dx, which is half a cell edge length.
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

    @validate_index
    def centroid(self, index):
        if index is None:
            raise ValueError(
                "For grids that do not contain data, argument `index` is to be supplied to method `centroid`."
            )
        original_shape = (*index.shape, 2)
        index = (
            index.ravel().index[None] if index.index.ndim == 1 else index.ravel().index
        )
        centroids = self._grid.centroid(index=index)
        return centroids.reshape(original_shape)

    @validate_index
    def cell_corners(self, index=None):
        if index is None:
            raise ValueError(
                "For grids that do not contain data, argument `index` is to be supplied to method `corners`."
            )
        return_shape = (
            *index.shape,
            3,
            2,
        )
        index = (
            index.ravel().index[None] if index.index.ndim == 1 else index.ravel().index
        )
        corners = self._grid.cell_corners(index=index)
        return corners.reshape(return_shape)

    def cell_at_point(self, point):
        point = numpy.array(point, dtype="float64")
        point = point[None] if point.ndim == 1 else point
        return_shape = point.shape
        result = self._grid.cell_at_point(point.reshape(-1, 2))
        return GridIndex(result.reshape(return_shape))

    def cells_in_bounds(self, bounds, return_cell_count=False):

        if self.rotation != 0:
            raise NotImplementedError(
                f"`cells_in_bounds` is not suppored for rotated grids. Roatation: {self.rotation} degrees"
            )

        if not self.are_bounds_aligned(bounds):
            raise ValueError(
                f"supplied bounds '{bounds}' are not aligned with the grid lines. Consider calling 'align_bounds' first."
            )
        ids, shape = self._grid.cells_in_bounds(bounds)
        ids = GridIndex(ids.reshape((*shape, 2)))
        return (ids, shape) if return_cell_count else ids

    def cells_near_point(self, point):
        point = numpy.array(point, dtype=float)
        original_shape = (*point.shape[:-1], 6, 2)
        point = point[None] if point.ndim == 1 else point
        point = point.reshape(-1, 2)
        ids = self._grid.cells_near_point(point)
        return GridIndex(ids.squeeze().reshape(original_shape))

    @validate_index
    def is_cell_upright(self, index):
        """Whether the selected cell points up or down.
        True if the cell points up, False if the cell points down.

        Parameters
        ----------
        index: `GridIndex`
            The index of the cell(s) of interest

        Returns
        -------
        `numpy.ndarray` or `bool`
            A boolean value reflecting whether the cell is upright or not.
            Or a 1d array containing the boolean values for each cell.
        """
        index = index.index[None] if index.index.ndim == 1 else index.index
        return self._grid.is_cell_upright(index=index).squeeze()

    @property
    def parent_grid_class(self):
        return TriGrid

    @validate_index
    def relative_neighbours(
        self, index=None, depth=1, connect_corners=False, include_selected=False
    ):
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
            Each cell has 3 neighbours if connect_corners is False,
            and 9 neighbours if connect_corners is True.

        See also
        --------
        :py:meth:`.BaseGrid.neighbours`
        :py:meth:`.RectGrid.relative_neighbours`
        :py:meth:`.HexGrid.relative_neighbours`
        """
        index = (
            index.ravel().index[None] if index.index.ndim == 1 else index.ravel().index
        )
        result = self._grid.relative_neighbours(
            index,
            depth=depth,
            connect_corners=connect_corners,
            include_selected=include_selected,
        )
        return GridIndex(result)

    @validate_index
    def neighbours(
        self, index=None, depth=1, connect_corners=False, include_selected=False
    ):
        index = (
            index.ravel().index[None] if index.index.ndim == 1 else index.ravel().index
        )
        result = self._grid.neighbours(
            index,
            depth=depth,
            connect_corners=connect_corners,
            include_selected=include_selected,
        )
        return GridIndex(result)

    def to_bounded(self, bounds, fill_value=numpy.nan):
        _, shape = self.cells_in_bounds(bounds, return_cell_count=True)
        data = numpy.full(shape=shape, fill_value=fill_value)
        return self.bounded_cls(
            data=data,
            bounds=bounds,
            nodata_value=fill_value,
            crs=self.crs,
        )

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
        :meth:`.BoundedTriGrid.to_crs`
        :meth:`.BoundedHexGrid.to_crs`

        """
        # FIXME: Here we determine the size, or the length of one of the sides of a cell.
        #        This changes where on the earth the difference between the CRS-es is determined.
        #        Add a location parameter with which the user can specify where this should happen.
        #        Default is at (0,0).
        # FIXME: Make it very clear the data is meant to be entered as XY which might not match how geopandas treats it based on CRS
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

        point_end = transformer.transform(
            self.dx, 0
        )  # likely different for shape='flat'
        size = numpy.linalg.norm(numpy.subtract(point_end, point_start))

        return self.parent_grid_class(size=size, offset=new_offset, crs=crs)

    def _update_inner_grid(self, size=None, offset=None, rotation=None):
        if size is None:
            size = self.size
        if offset is None:
            offset = self.offset
        if rotation is None:
            rotation = self.rotation
        return PyTriGrid(cellsize=size, offset=offset, rotation=rotation)

    def update(self, size=None, offset=None, rotation=None, crs=None, **kwargs):
        if size is None:
            size = self.size
        if offset is None:
            offset = self.offset
        if rotation is None:
            rotation = self.rotation
        if crs is None:
            crs = self.crs
        return TriGrid(size=size, offset=offset, rotation=rotation, crs=crs, **kwargs)


class BoundedTriGrid(BoundedGrid, TriGrid):
    def __init__(self, data, *args, bounds, **kwargs):
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError(
                f"Incerrect bounds. Minimum value exceeds maximum value for bounds {bounds}"
            )
        data = numpy.array(data) if not isinstance(data, numpy.ndarray) else data
        if data.ndim != 2:
            raise ValueError(
                f"Expected a 2D numpy array, got data with shape {data.shape}"
            )

        dx = (bounds[2] - bounds[0]) / data.shape[1]
        dy = (bounds[3] - bounds[1]) / data.shape[0]

        if not numpy.isclose(dy, dx * 3**0.5):
            raise ValueError(
                "The supplied data shape cannot be covered by triangles with sides of equal length with the given bounds."
            )

        offset_x = bounds[0] % dx
        offset_y = bounds[1] % dy
        offset_x = dx - offset_x if offset_x < 0 else offset_x
        offset_y = dy - offset_y if offset_y < 0 else offset_y
        offset = (
            0 if numpy.isclose(offset_x, dx) else offset_x,
            0 if numpy.isclose(offset_y, dy) else offset_y,
        )

        super(BoundedTriGrid, self).__init__(
            data, *args, size=dx, bounds=bounds, offset=offset, **kwargs
        )

        if not self.are_bounds_aligned(bounds):
            raise AlignmentError(
                "Something went wrong, the supplied bounds are not aligned with the resulting grid."
            )

    def centroid(self, index=None):
        if index is None:
            if not hasattr(self, "indices"):
                raise ValueError(
                    "For grids that do not contain data, argument `index` is to be supplied to method `centroid`."
                )
            index = self.indices
        return super(BoundedTriGrid, self).centroid(index=index)

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
            slice_x = slice(
                max(0, slice_x.start - buffer_cells),
                min(self.width, slice_x.stop + buffer_cells),
            )
            slice_y = slice(
                max(0, slice_y.start - buffer_cells),
                min(self.height, slice_y.stop + buffer_cells),
            )
            new_bounds = (
                max(new_bounds[0] - buffer_cells * self.dx, self.bounds[0]),
                max(new_bounds[1] - buffer_cells * self.dy, self.bounds[1]),
                min(new_bounds[2] + buffer_cells * self.dx, self.bounds[2]),
                min(new_bounds[3] + buffer_cells * self.dy, self.bounds[3]),
            )
        # cropped_data = numpy.flipud(numpy.flipud(self._data)[slice_y, slice_x]) # TODO: fix this blasted flipping. The raster should not be stored upside down maybe
        cropped_data = self._data[slice_y, slice_x]  # Fixme: seems to be flipped?
        # cropped_data = self._data[slice_x, slice_y]
        return self.update(cropped_data, bounds=new_bounds)

    @validate_index
    def cell_corners(self, index: numpy.ndarray = None) -> numpy.ndarray:
        if index is None:
            index = self.indices
        return super(BoundedTriGrid, self).cell_corners(index=index)

    @validate_index
    def to_shapely(self, index=None, as_multipolygon: bool = False):
        """Refer to parent method :meth:`.BaseGrid.to_shapely`

        Difference with parent method:
            `index` is optional.
            If `index` is None (default) the cells containing data are used as the `index` argument.

        See also
        --------
        :meth:`.BaseGrid.to_shapely`
        :meth:`.BoundedHexGrid.to_shapely`
        """
        if index is None:
            index = self.indices
        return super().to_shapely(index, as_multipolygon)

    def _bilinear_interpolation(self, sample_points):
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
        nearby_cells = self.cells_near_point(sample_points)
        nearby_centroids = self.centroid(nearby_cells)
        nearby_values = self.value(nearby_cells)

        if sample_points.squeeze().ndim == 1:
            sample_points = sample_points.squeeze()[None]
            nearby_centroids = nearby_centroids[None]
            nearby_values = nearby_values[None]
        values = self._grid.linear_interpolation(
            sample_points,
            nearby_centroids,
            nearby_values.astype(
                float  # FIXME: figure out generics in rust to allow for other dtypes
            ),
        )

        if len(values) == 1:
            return values[0]
        return (
            values.squeeze().reshape(*original_shape[:-1])
            if original_shape[:-1]
            else values
        )

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
        :class:`~.rect_grid.BoundedTriGrid`
            A copy of the grid with modified cell spacing and bounds to match the specified CRS

        See also
        --------
        Examples:

        :ref:`Example: coordinate transformations <example coordinate transformations>`

        Methods:

        :meth:`.RectGrid.to_crs`
        :meth:`.HexGrid.to_crs`
        :meth:`.BoundedHexGrid.to_crs`

        """
        new_inf_grid = super(BoundedTriGrid, self).to_crs(crs)
        return self.resample(new_inf_grid, method=resample_method)

    def numpy_id_to_grid_id(self, np_index):
        centroid_topleft = (self.bounds[0] + self.dx / 2, self.bounds[3] - self.dy / 2)
        index_topleft = self.cell_at_point(centroid_topleft)
        ids = numpy.array(
            [index_topleft.x + np_index[1], index_topleft.y - np_index[0]]
        )
        return GridIndex(ids.T)

    @validate_index
    def grid_id_to_numpy_id(self, index):
        if index.index.ndim > 2:
            raise ValueError(
                "Cannot convert nd-index to numpy index. Consider flattening the index using `index.ravel()`"
            )
        centroid_topleft = (self.bounds[0] + self.dx / 2, self.bounds[3] - self.dy / 2)
        index_topleft = self.cell_at_point(centroid_topleft)
        return (index_topleft.y - index.y, index.x - index_topleft.x)
