import warnings

import numpy
import scipy
from pyproj import CRS, Transformer

from gridkit.base_grid import BaseGrid
from gridkit.bounded_grid import BoundedGrid
from gridkit.errors import AlignmentError, IntersectionError
from gridkit.index import GridIndex, validate_index


class RectGrid(BaseGrid):
    def __init__(self, *args, dx, dy, **kwargs):
        self.__dx = dx
        self.__dy = dy
        self.bounded_cls = BoundedRectGrid
        super(RectGrid, self).__init__(*args, **kwargs)

    @property
    def dx(self) -> float:
        """The cellsize in x-direction"""
        return self.__dx

    @property
    def dy(self) -> float:
        """The cellsize in y-direction"""
        return self.__dy

    def relative_neighbours(
        self, depth=1, connect_corners=False, include_selected=False, index=None
    ):
        """The relative indices of the neighbouring cells.

        Parameters
        ----------
        depth: :class:`int` Default: 1
            Determines the number of neighbours that are returned.
            If `depth=1` the direct neighbours are returned.
            If `depth=2` the direct neighbours are returned, as well as the neighbours of these neighbours.
            `depth=3` returns yet another layer of neighbours, and so forth.
        include_selected: :class:`bool` Default: False
            Whether to include the specified cell in the return array.
            Even though the specified cell can never be a neighbour of itself,
            this can be useful when for example a weighted average of the neighbours is desired
            in which case the cell itself often should also be included.
        connect_corners: :class:`bool` Default: False
            Whether to consider cells that touch corners but not sides as neighbours.
            If `connect_corners` is True, the 4 cells directly touching the cell are considered neighbours.
            If `connect_corners` is True, the 8 cells surrounding the cell are considered neighbours.
            This escalates in combination with `depth` where indices in a square shape around the cell are returned
            when `connect_corners` is True, and indices in a diamond shape around the cell are returned when `connect_corners` is False.
        index: :class:`numpy.ndarray`
            The index is mostly relevant for hexagonal grids.
            For a square grid the relative neighbours are independent on the location on the grid.
            Here it is only used for the return length.
            If 10 cells are supplied to `index`, the relative neighbours are returned 10 times.
            This is to keep a consistent api between the two classes.


        Examples
        --------
        The direct neighbours of a cell can be returned by using depth=1, which is the default.
        For square grids, the number of returned neighbours depends on whether `connect_corners` is True or False:

        .. code-block:: python

            >>> from gridkit.rect_grid import RectGrid
            >>> grid = RectGrid(dx=2, dy=3)
            >>> grid.relative_neighbours().index
            array([[ 0,  1],
                   [-1,  0],
                   [ 1,  0],
                   [ 0, -1]])
            >>> grid.relative_neighbours(connect_corners=True).index
            array([[-1,  1],
                   [ 0,  1],
                   [ 1,  1],
                   [-1,  0],
                   [ 1,  0],
                   [-1, -1],
                   [ 0, -1],
                   [ 1, -1]])

        ..

        By specifying `depth` we can include indirect neighbours from further away.
        The number of neighbours increases with depth by a factor of `depth*4` or `depth*8` depending on `connect_corners` being True or False.
        So the 3rd element in the list will be `1*4 + 2*4 + 3*4 = 24` if `connect_corners` is False.
        And it will be `1*8 + 2*8 + 3*8 = 48` if `connect_corners` is True.

        .. code-block:: python

            >>> [len(grid.relative_neighbours(depth=depth)) for depth in range(1,5)]
            [4, 12, 24, 36]
            >>> [len(grid.relative_neighbours(depth=depth, connect_corners=True)) for depth in range(1,5)]
            [8, 24, 48, 80]

        ..

        The specified cell can be included if `include_selected` is set to True:

        .. code-block:: python

            >>> grid.relative_neighbours(include_selected=True).index
            array([[ 0,  1],
                   [-1,  0],
                   [ 0,  0],
                   [ 1,  0],
                   [ 0, -1]])

        ..

        See also
        --------
        :py:meth:`.BaseGrid.neighbours`
        :py:meth:`.HexGrid.relative_neighbours`
        """

        if depth < 1:
            raise ValueError("'depth' cannot be lower than 1")

        neighbours = numpy.empty(((2 * depth + 1) ** 2, 2), dtype=int)

        relative_ids_1d = numpy.arange(-depth, depth + 1)
        relative_x, relative_y = numpy.meshgrid(relative_ids_1d, relative_ids_1d[::-1])
        neighbours[:, 0], neighbours[:, 1] = numpy.ravel(relative_x), numpy.ravel(
            relative_y
        )

        if not connect_corners:
            mask = abs(numpy.multiply(*neighbours.T)) < depth
            neighbours = neighbours[mask]

        if include_selected is False:
            center_cell = int(numpy.floor(len(neighbours) / 2))
            neighbours = numpy.delete(neighbours, center_cell, 0)

        if index is not None:
            index = numpy.array(index)
            if len(index.shape) == 2:
                neighbours = numpy.repeat(neighbours[numpy.newaxis], len(index), axis=0)

        return GridIndex(neighbours)

    def centroid(self, index=None):
        """Coordinates at the center of the cell(s) specified by `index`.

        .. Warning ::
            The two values that make up an `index` are expected to be integers, and will be cast as such.

        Parameters
        ----------
        index: :class:`tuple`
            Index of the cell of which the centroid is to be calculated.
            The index consists of two integers specifying the nth cell in x- and y-direction.
            Mutliple indices can be specified at once in the form of a list of indices or an Nx2 ndarray.

        Returns
        -------
        :class:`numpy.ndarray`
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
        index = numpy.array(index, dtype="int").T
        centroids = numpy.empty_like(index, dtype=float)
        centroids[0] = index[0] * self.dx + (self.dx / 2) + self.offset[0]
        centroids[1] = index[1] * self.dy + (self.dy / 2) + self.offset[1]
        return centroids.T

    def cells_near_point(self, point):
        """Nearest 4 cells around a point.
        This includes the cell the point is contained within,
        as well as two direct neighbours of this cell and one diagonal neighbor.
        What neigbors of the containing are slected, depends on where in the cell the point is located.

        Args
        ----
        point: :class"`tuple`
            Coordinate of the point around which the cells are to be selected.
            The point consists of two floats specifying x- and y-coordinates, respectively.
            Mutliple poins can be specified at once in the form of a list of points or an Nx2 ndarray.

        Returns
        -------
        :class:`tuple`
            The indices of the 4 nearest cells in order (top-left, top-right, bottom-left, bottom-right).
            If a single point is supplied, the four indices are returned as 1d arrays of length 2.
            If multiple points are supplied, the four indices are returned as Nx2 ndarrays.

        Examples
        --------
        Nearby cell indices are returned as a tuple:

        .. code-block:: python

            >>> grid = RectGrid(dx=4, dy=1)
            >>> grid.cells_near_point((0, 0)).index
            array([[-1,  0],
                   [ 0,  0],
                   [-1, -1],
                   [ 0, -1]])
            >>> grid.cells_near_point((3, 0.75)).index
            array([[0, 1],
                   [1, 1],
                   [0, 0],
                   [1, 0]])

        ..

        If multiple points are supplied, a tuple with ndarrays is returned:

        .. code-block:: python

            >>> points = [(0, 0), (3, 0.75), (3, 0)]
            >>> nearby_cells = grid.cells_near_point(points)
            >>> nearby_cells.index
            array([[[-1,  0],
                    [ 0,  1],
                    [ 0,  0]],
            <BLANKLINE>
                   [[ 0,  0],
                    [ 1,  1],
                    [ 1,  0]],
            <BLANKLINE>
                   [[-1, -1],
                    [ 0,  0],
                    [ 0, -1]],
            <BLANKLINE>
                   [[ 0, -1],
                    [ 1,  0],
                    [ 1, -1]]])

        ..

        """

        # Split each cell into 4 quadrants in order to identify what 3 neigbors to select
        new_grid = RectGrid(dx=self.dx / 2, dy=self.dy / 2, offset=self.offset)
        ids = new_grid.cell_at_point(point)
        left_top_mask = numpy.logical_and(ids.x % 2 == 0, ids.y % 2 == 1)
        right_top_mask = numpy.logical_and(ids.x % 2 == 1, ids.y % 2 == 1)
        right_bottom_mask = numpy.logical_and(ids.x % 2 == 1, ids.y % 2 == 0)
        left_bottom_mask = numpy.logical_and(ids.x % 2 == 0, ids.y % 2 == 0)

        # obtain the bottom-left cell based on selected quadrant
        base_ids = numpy.empty_like(ids.index, dtype="int")
        # bottom-left if point in upper-right quadrant
        base_ids[right_top_mask] = numpy.floor(ids.index[right_top_mask] / 2)
        # bottom-left if point in bottom-right quadrant
        base_ids[right_bottom_mask] = numpy.floor(ids.index[right_bottom_mask] / 2)
        base_ids[right_bottom_mask, 1] -= 1
        # bottom-left if point in bottom-left quadrant
        base_ids[left_bottom_mask] = numpy.floor(ids.index[left_bottom_mask] / 2)
        base_ids[left_bottom_mask] -= 1
        # bottom-left if point in upper-left qudrant
        base_ids[left_top_mask] = numpy.floor(ids.index[left_top_mask] / 2)
        base_ids[left_top_mask, 0] -= 1

        base_ids = GridIndex(base_ids)

        # use the bottom-left cell to determine the other three
        bl_ids = base_ids.copy()
        br_ids = base_ids.copy()
        br_ids.x += 1
        tr_ids = base_ids.copy()
        tr_ids += 1
        tl_ids = base_ids.copy()
        tl_ids.y += 1

        return GridIndex([tl_ids, tr_ids, bl_ids, br_ids])

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
        :class:`numpy.ndarray`
            The index of the cell containing the point(s).
            If a single point is supplied, the index is returned as a 1d array of length 2.
            If multiple points are supplied, the indices are returned as Nx2 ndarrays.

        Examples
        --------

        .. code-block:: python

            >>> grid = RectGrid(dx=4, dy=1)
            >>> grid.cell_at_point((1, 0)).index
            array([0, 0])

        ..

        Offsets, shift the grid with respect to the coordinate system,
        and thus can influnce what cell contains the point:

        .. code-block:: python

            >>> grid = RectGrid(dx=4, dy=1, offset=(2,0))
            >>> grid.cell_at_point((1, 0)).index
            array([-1,  0])

        ..

        Multiple points can be specified as a list of points or an ndarray:

        .. code-block:: python

            >>> grid = RectGrid(dx=4, dy=1)
            >>> points = [(3, 0), (-0.5, -0.5), (5, 0)]
            >>> grid.cell_at_point(points).index
            array([[ 0,  0],
                   [-1, -1],
                   [ 1,  0]])
            >>> points = numpy.array(points)
            >>> grid.cell_at_point(points).index
            array([[ 0,  0],
                   [-1, -1],
                   [ 1,  0]])

        ..

        """
        point = numpy.array(point).T
        ids_x = numpy.floor((point[0] - self.offset[0]) / self.dx)
        ids_y = numpy.floor((point[1] - self.offset[1]) / self.dy)
        index = numpy.array([ids_x, ids_y], dtype="int").T
        return GridIndex(index)

    @validate_index
    def cell_corners(self, index: GridIndex = None) -> numpy.ndarray:
        """Return corners in (cells, corners, xy)"""
        if index is None:
            raise ValueError(
                "For grids that do not contain data, argument `index` is to be supplied to method `corners`."
            )
        centroids = self.centroid(index).T

        if len(centroids.shape) == 1:
            corners = numpy.empty((4, 2))
        else:
            corners = numpy.empty((4, 2, centroids.shape[1]))

        corners[0, 0] = centroids[0] - self.dx / 2
        corners[0, 1] = centroids[1] - self.dy / 2
        corners[1, 0] = centroids[0] + self.dx / 2
        corners[1, 1] = centroids[1] - self.dy / 2
        corners[2, 0] = centroids[0] + self.dx / 2
        corners[2, 1] = centroids[1] + self.dy / 2
        corners[3, 0] = centroids[0] - self.dx / 2
        corners[3, 1] = centroids[1] + self.dy / 2

        # swap from (corners, xy, cells) to (cells, corners, xy)
        if len(centroids.shape) > 1:
            corners = numpy.moveaxis(corners, 2, 0)

        return numpy.squeeze(corners)

    def is_aligned_with(self, other):
        if not isinstance(other, RectGrid):
            raise ValueError(f"Expected a RectGrid, got {type(other)}")
        aligned = True
        reason = ""
        reasons = []

        if self.crs is None and other.crs is None:
            pass
        elif self.crs is None:
            aligned = False
            reasons.append("CRS")
        elif not self.crs.is_exact_same(other.crs):
            aligned = False
            reasons.append("CRS")

        if not numpy.isclose(self.dx, other.dx) or not numpy.isclose(self.dy, other.dy):
            aligned = False
            reasons.append("cellsize")

        if not all(
            numpy.isclose(self.offset, other.offset, atol=1e-7)
        ):  # FIXME: atol if 1e-7 is a bandaid. It seems the offset depends slightly depending on the bounds after resampling on grid
            aligned = False
            reasons.append("offset")

        reason = (
            f"The following attributes are not the same: {reasons}"
            if reasons
            else reason
        )
        return aligned, reason

    def to_crs(self, crs):
        """Transforms the Coordinate Reference System (CRS) from the current CRS to the desired CRS.
        This will modify the cell size and the bounds accordingly.

        The ``crs`` attribute on the current GeometryArray must be set.

        Parameters
        ----------
        crs: Union[int, str, pyproj.CRS]
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an epsg integer (eg 4326), an authority string (eg "EPSG:4326") or a WKT string.

        Returns
        -------
        :class:`~.rect_grid.RectGrid`
            A copy of the grid with modified cell spacing and bounds to match the specified CRS

        See also
        --------
        :meth:`.BoundedRectGrid.to_crs`
        :meth:`.HexGrid.to_crs`
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

        return self.parent_grid_class(
            dx=abs(new_dx), dy=abs(new_dy), offset=new_offset, crs=crs
        )

    def cells_in_bounds(self, bounds):
        """
        Parameters
        ----------
        bounds: :class:`tuple`
        align_mode: :class:`str`
            Specifies when to consider a cell included in the bounds. Options:
             - contains
               select cells fully contained in the specified bounds
             - contains_center
               select cells of which the center is contained in the specified bounds
             - intersects
               select cells partially or fully contained in the specified bounds
        """

        if not self.are_bounds_aligned(bounds):
            raise ValueError(
                f"supplied bounds '{bounds}' are not aligned with the grid lines. Consider calling 'align_bounds' first."
            )

        if not self.are_bounds_aligned(bounds):
            bounds = self.align_bounds(bounds, mode="expand")

        # get coordinates of two diagonally opposing corner-cells
        left_top = (bounds[0] + self.dx / 2, bounds[3] - self.dy / 2)
        right_bottom = (bounds[2] - self.dx / 2, bounds[1] + self.dy / 2)

        # translate the coordinates of the corner cells into indices
        left_top_id, right_bottom_id = self.cell_at_point([left_top, right_bottom])

        # turn minimum and maximum indices into fully arranged array
        # TODO: think about dtype. 64 is too large. Should this be exposed? Or automated based on id range?
        ids_y = numpy.arange(
            left_top_id.y, right_bottom_id.y - 1, -1, dtype="int32"
        )  # y-axis goes from top to bottom (high to low), hence step size is -1
        ids_x = list(range(int(left_top_id.x), int(right_bottom_id.x + 1)))

        # TODO: only return ids_y and ids_x without fully filled grid
        shape = (len(ids_y), len(ids_x))
        ids = numpy.empty((2, numpy.multiply(*shape)), dtype="int32")
        ids[0] = ids_x * len(ids_y)
        # ids[1] = numpy.tile(ids_y[::-1], len(ids_x)) # invert y for ids are calculated from origin, not form the top of the bounding box
        ids[1] = numpy.repeat(ids_y, len(ids_x))

        return GridIndex(ids.T), shape

    @property
    def parent_grid_class(self):
        return RectGrid


class BoundedRectGrid(BoundedGrid, RectGrid):
    def __init__(self, data, *args, bounds, **kwargs):
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError(
                f"Incerrect bounds. Minimum value exceeds maximum value for bounds {bounds}"
            )
        dx = (bounds[2] - bounds[0]) / data.shape[1]
        dy = (bounds[3] - bounds[1]) / data.shape[0]

        offset_x = bounds[0] % dx
        offset_y = bounds[1] % dy
        offset_x = dx - offset_x if offset_x < 0 else offset_x
        offset_y = dy - offset_y if offset_y < 0 else offset_y
        offset = (
            0 if numpy.isclose(offset_x, dx) else offset_x,
            0 if numpy.isclose(offset_y, dy) else offset_y,
        )
        super(BoundedRectGrid, self).__init__(
            data, *args, dx=dx, dy=dy, bounds=bounds, offset=offset, **kwargs
        )

    @property
    def nr_cells(self):
        return (self.width, self.height)

    @property
    def lon(self):
        """Array of long values

        Returns
        -------
        :class:`numpy.ndarray`
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
        :class:`numpy.ndarray`
            1D-Array of size `height`, containing the latitudinal values from top to bottom
        """
        return numpy.linspace(
            self.bounds[3] - self.dy / 2, self.bounds[1] + self.dy / 2, self.height
        )

    def centroid(self, index=None):
        """Centroids of all cells

        Returns
        -------
        :class:`numpy.ndarray`
            Multidimensional array containing the longitude and latitude of the center of each cell respectively,
            in (width, height, lonlat)
        """
        if index is not None:
            return super(BoundedRectGrid, self).centroid(index=index)
        # get grid in shape (latlon, width, height)
        latlon = numpy.meshgrid(self.lon, self.lat, sparse=False, indexing="xy")

        # return grid in shape (width, height, lonlat)
        return numpy.array([latlon[0].ravel(), latlon[1].ravel()]).T

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
        if not self.are_bounds_aligned(bounds):
            raise ValueError(
                f"Cannot create slice from unaligned bounds {tuple(bounds)}"
            )

        difference_left = round(abs((self.bounds[0] - bounds[0]) / self.dx))
        difference_right = round(abs((self.bounds[2] - bounds[2]) / self.dx))
        slice_x = slice(
            difference_left,
            self.width
            - difference_right,  # add one for upper bound of slice is exclusive
        )

        difference_bottom = round(abs((self.bounds[1] - bounds[1]) / self.dy))
        difference_top = round(abs((self.bounds[3] - bounds[3]) / self.dy))
        slice_y = slice(
            difference_top,
            self.height
            - difference_bottom,  # add one for upper bound of slice is exclusive
        )

        return slice_y, slice_x

    def cell_corners(self, index: numpy.ndarray = None) -> numpy.ndarray:
        if index is None:
            index = self.indices()
        return super(BoundedRectGrid, self).cell_corners(index=index)

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
        """Interpolate the value at the location of `sample_points` by doing a bilinear interpolation
        using the 4 cells around the point.

        Parameters
        ----------
        sample_points: :class:`numpy.ndarray`
            The coordinates of the points at which to sample the data

        Returns
        -------
        :class:`numpy.ndarray`
            The interpolated values at the supplied points

        See also
        --------
        :py:meth:`.RectGrid.cell_at_point`
        """
        nodata_value = self.nodata_value if self.nodata_value is not None else numpy.nan
        tl_ids, tr_ids, bl_ids, br_ids = self.cells_near_point(sample_points).index

        tl_val = self.value(tl_ids, oob_value=nodata_value)
        tr_val = self.value(tr_ids, oob_value=nodata_value)
        bl_val = self.value(bl_ids, oob_value=nodata_value)
        br_val = self.value(br_ids, oob_value=nodata_value)

        # determine relative location of new point between old cell centers in x and y directions
        abs_diff = sample_points - self.centroid(bl_ids)
        x_diff = abs_diff[:, 0] / self.dx
        y_diff = abs_diff[:, 1] / self.dy

        top_val = tl_val + (tr_val - tl_val) * x_diff
        bot_val = bl_val + (br_val - bl_val) * x_diff
        values = bot_val + (top_val - bot_val) * y_diff

        # TODO: remove rows and cols with nans around the edge after bilinear
        return values

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
            The resampling method to be used for :meth:`.RectGrid.resample`.

        Returns
        -------
        :class:`~.rect_grid.BoundedRectGrid`
            A copy of the grid with modified cell spacing and bounds to match the specified CRS

        See also
        --------
        :meth:`.RectGrid.to_crs`
        :meth:`.HexGrid.to_crs`
        :meth:`.BoundedHexGrid.to_crs`

        """
        new_inf_grid = super(BoundedRectGrid, self).to_crs(
            crs, resample_method=resample_method
        )
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

    def interp_nodata(self, *args, **kwargs):
        """Please refer to :func:`~gridkit.bounded_grid.BoundedGrid.interp_nodata`."""
        # Fixme: in the case of a rectangular grid, a performance improvement can be obtained by using scipy.interpolate.interpn
        return super(BoundedRectGrid, self).interp_nodata(*args, **kwargs)
