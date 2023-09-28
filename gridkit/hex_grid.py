import warnings

import numpy
from pyproj import CRS, Transformer

from gridkit.base_grid import BaseGrid
from gridkit.bounded_grid import BoundedGrid
from gridkit.errors import AlignmentError, IntersectionError
from gridkit.index import GridIndex, validate_index
from gridkit.rect_grid import RectGrid


class HexGrid(BaseGrid):
    def __init__(self, *args, size, shape="pointy", **kwargs):
        self._size = size
        self._radius = size / 3**0.5

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

        self._shape = shape
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

    @property
    def size(self) -> float:
        """The size of the cell as supplied when initiating the class.
        This is the same as dx for a flat grid and the same as dy for a pointy grid.
        """
        return self._size

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

    def relative_neighbours(
        self, depth=1, *, index, include_selected=False, connect_corners=False
    ) -> numpy.ndarray:
        """The relative indices of the neighbouring cells.

        Parameters
        ----------
        depth: :class:`int` Default: 1
            Determines the number of neighbours that are returned.
            If `depth=1` the direct neighbours are returned.
            If `depth=2` the direct neighbours are returned, as well as the neighbours of these neighbours.
            `depth=3` returns yet another layer of neighbours, and so forth.
        index: :class:`numpy.ndarray`
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
        """

        if depth < 1:
            raise ValueError("'depth' cannot be lower than 1")

        index = numpy.array(index)
        if len(index.shape) == 1:
            index = index[numpy.newaxis]

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

        return GridIndex(neighbours if len(neighbours) > 1 else neighbours[0])

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

        if self._shape == "pointy":
            offset_rows = index[1] % 2 == 1
            centroids[0, offset_rows] += self.dx / 2
        elif self._shape == "flat":
            offset_rows = index[0] % 2 == 1
            centroids[1, offset_rows] += self.dy / 2
        return centroids.T

    def cells_near_point(self, point):
        """Nearest 3 cells around a point.
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


        """
        cell = self.cell_at_point(point)
        original_shape = cell.shape
        cell = cell.ravel()

        point = numpy.reshape(point, cell.index.shape)
        centroid = self.centroid(cell)
        distance_vector = point - centroid
        azimuth = numpy.arctan2(*distance_vector.T) * 180 / numpy.pi
        # FIXME: if there is one point, azimuth is a float in which we cannot use a mask. Make the check more elegant
        point = numpy.array(point)
        if len(point.shape) == 1:
            azimuth = numpy.array([azimuth])

        cell = cell.index
        if len(cell.shape) == 1:
            cell = numpy.expand_dims(cell, axis=0)
        az_ranges = [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300), (300, 360)]
        if self._shape == "flat":
            inconsistent_axis = (
                0  # TODO: Make consistent and inconsisten axis a property of self
            )
            consistent_axis = 1
            shift_odd_cells = [1, slice(1, 3), 2, 1, slice(1, 3), 2]
            nearby_cells_relative_idx = [
                [[1, 0], [0, 1]],
                [[1, -1], [1, 0]],
                [[0, -1], [1, -1]],
                [[-1, -1], [0, -1]],
                [[-1, 0], [-1, -1]],
                [[0, 1], [-1, 0]],
            ]
        elif self._shape == "pointy":
            inconsistent_axis = 1
            consistent_axis = 0
            azimuth += (
                30  # it is easier to work with ranges starting at 0 rather than -30
            )
            shift_odd_cells = [slice(1, 3), 1, 2, slice(1, 3), 1, 2]
            nearby_cells_relative_idx = [
                [[-1, 1], [0, 1]],
                [[0, 1], [1, 0]],
                [[1, 0], [0, -1]],
                [[0, -1], [-1, -1]],
                [[-1, -1], [-1, 0]],
                [[-1, 0], [-1, 1]],
            ]
        else:
            raise AttributeError(f"Unrecognized grid shape {self._shape}")

        nearby_cells = numpy.repeat(
            numpy.expand_dims(cell, axis=0), 3, axis=0
        )  # shape: neighbours(3), points(n), xy(2)
        odd_cells_mask = cell[:, inconsistent_axis] % 2 == 1

        # make sure azimuth is inbetween 0 and 360, and not between -180 and 180
        azimuth[azimuth <= 0] += 360
        azimuth[azimuth > 360] -= 360

        for az_range, shift_odd, cells in zip(
            az_ranges, shift_odd_cells, nearby_cells_relative_idx
        ):
            mask = numpy.logical_and(azimuth > az_range[0], azimuth <= az_range[1])
            nearby_cells[1, mask] += cells[0]
            nearby_cells[2, mask] += cells[1]
            mask_odd = numpy.logical_and(mask, odd_cells_mask)
            nearby_cells[shift_odd, mask_odd, consistent_axis] += 1

        # turn into shape: points(n), neighbours(3), xy(2)
        # points(n) will be unraveled later if multiple points were provided
        nearby_cells = numpy.swapaxes(nearby_cells, 0, 1)

        if len(nearby_cells) == 1:
            return nearby_cells[0]
        # return shape: points(...), neighbours(3), xy(2)
        # where points(...) can be any nd (unraveled) shape
        return nearby_cells.reshape((*original_shape, 3, 2))

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

        """
        point = numpy.array(point)
        point = numpy.expand_dims(point, axis=0).T if len(point.shape) == 1 else point.T

        # approach adapted after https://stackoverflow.com/a/7714148
        if self._shape == "pointy":
            flat_axis = 0
            pointy_axis = 1
            flat_stepsize = self.dx
            pointy_stepsize = self.dy
        elif self._shape == "flat":
            flat_axis = 1
            pointy_axis = 0
            flat_stepsize = self.dy
            pointy_stepsize = self.dx
        else:
            raise ValueError(
                f"A HexGrid's `shape` can either be 'pointy' or 'flat', got '{self._shape}'"
            )

        ids_pointy = numpy.floor(
            (point[pointy_axis] - self.offset[pointy_axis] - self.r / 4)
            / pointy_stepsize
        )
        even = ids_pointy % 2 == 0
        ids_flat = numpy.empty_like(ids_pointy)
        ids_flat[~even] = numpy.floor(
            (point[flat_axis][~even] - self.offset[flat_axis] - flat_stepsize / 2)
            / flat_stepsize
        )
        ids_flat[even] = numpy.floor(
            (point[flat_axis][even] - self.offset[flat_axis]) / flat_stepsize
        )

        # Finetune ambiguous points
        # Points at the top of the cell can be in this cell or in the cell to the top right or top left
        rel_loc_y = (
            (point[pointy_axis] - self.offset[pointy_axis] - self.r / 4)
            % pointy_stepsize
        ) + self.r / 4
        rel_loc_x = (point[flat_axis] - self.offset[flat_axis]) % flat_stepsize
        top_left_even = rel_loc_x / (flat_stepsize / self.r) < (
            rel_loc_y - self.r * 5 / 4
        )
        top_right_even = (self.r * 1.25 - rel_loc_y) <= (rel_loc_x - flat_stepsize) / (
            flat_stepsize / self.r
        )
        top_right_odd = (rel_loc_x - flat_stepsize / 2) / (flat_stepsize / self.r) <= (
            rel_loc_y - self.r * 5 / 4
        )
        top_right_odd &= rel_loc_x >= flat_stepsize / 2
        top_left_odd = (self.r * 1.25 - rel_loc_y) < (rel_loc_x - flat_stepsize / 2) / (
            flat_stepsize / self.r
        )
        top_left_odd &= rel_loc_x < flat_stepsize / 2

        ids_pointy[top_left_even & even] += 1
        ids_pointy[top_right_even & even] += 1
        ids_pointy[top_left_odd & ~even] += 1
        ids_pointy[top_right_odd & ~even] += 1

        ids_flat[top_left_even & even] -= 1
        ids_flat[top_left_odd & ~even] += 1

        if self._shape == "pointy":
            result = numpy.array([ids_flat, ids_pointy], dtype="int").T

        elif self._shape == "flat":
            result = numpy.array([ids_pointy, ids_flat], dtype="int").T

        return GridIndex(result)

    @validate_index
    def cell_corners(self, index: numpy.ndarray = None) -> numpy.ndarray:
        """Return corners in (cells, corners, xy)"""
        if index is None:
            raise ValueError(
                "For grids that do not contain data, argument `index` is to be supplied to method `corners`."
            )
        cell_shape = index.shape
        centroids = self.centroid(index.ravel()).T

        if len(centroids.shape) == 1:
            corners = numpy.empty((6, 2))
        else:
            corners = numpy.empty((6, 2, centroids.shape[1]))

        for i in range(6):
            angle_deg = 60 * i - 30 if self._shape == "pointy" else 60 * i
            angle_rad = numpy.pi / 180 * angle_deg
            corners[i, 0] = centroids[0] + self.r * numpy.cos(angle_rad)
            corners[i, 1] = centroids[1] + self.r * numpy.sin(angle_rad)

        # swap from (corners, xy, cells) to (cells, corners, xy)
        if len(centroids.shape) > 1:
            corners = numpy.moveaxis(corners, 2, 0)

        return corners.reshape((*cell_shape, 6, 2))

    def is_aligned_with(self, other):
        if not isinstance(other, BaseGrid):
            raise ValueError(f"Expected a (child of) BaseGrid, got {type(other)}")
        aligned = True
        reason = ""
        reasons = []
        if not other.parent_grid_class == self.parent_grid_class:
            aligned = False
            return (
                False,
                f"Grid type is not the same. This is a {self.parent_grid_class}, the other is a {other.parent_grid_class}",
            )

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
        This will update the cell size and the origin offset.

        The ``crs`` attribute on the current grid must be set.

        Parameters
        ----------
        crs Union[int, str, pyproj.CRS]
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
        if not self.are_bounds_aligned(bounds):
            raise ValueError(
                f"supplied bounds '{bounds}' are not aligned with the grid lines. Consider calling 'align_bounds' first."
            )

        if not self.are_bounds_aligned(bounds):
            bounds = self.align_bounds(bounds, mode="expand")

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


class BoundedHexGrid(BoundedGrid, HexGrid):
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

        # FIXME: speed up (numba)
        def get_weight(point, p1, p2, p3):
            def _project(point, line_points):
                """Project 'point' onto a line drawn between 'line_points[0]' and 'line_points[1]'
                Credits to https://stackoverflow.com/a/61343727/22128453
                """
                # distance between line_points[0] and line_points[1]
                line_length = numpy.sum((line_points[0] - line_points[1]) ** 2)

                # project point on line extension connecting line_points[0] and line_points[1]
                t = (
                    numpy.sum(
                        (point - line_points[0]) * (line_points[1] - line_points[0])
                    )
                    / line_length
                )

                return line_points[0] + t * (line_points[1] - line_points[0])

            side_length = numpy.linalg.norm(p1 - p2)
            dd = p2 + (p2 - p3) / 2
            if numpy.linalg.norm(dd - p1) > side_length:
                dd = p2 - (p2 - p3) / 2
            ad = dd - p1

            projected = _project(point - p1, [p3 - p1, p2 - p1])
            return numpy.linalg.norm((projected - (point - p1)) / numpy.linalg.norm(ad))

        if not isinstance(sample_points, numpy.ndarray):
            sample_points = numpy.array(sample_points)
        original_shape = sample_points.shape
        sample_points = sample_points.reshape(-1, 2)

        all_nearby_cells = self.cells_near_point(
            sample_points
        )  # (points, nearby_cells, xy)
        values = numpy.empty(len(sample_points))
        for idx, (point, nearby_cells) in enumerate(
            zip(sample_points, all_nearby_cells)
        ):
            nearby_centroids = self.centroid(nearby_cells)
            p1, p2, p3 = nearby_centroids
            weights = numpy.array(
                [
                    get_weight(point, p1, p2, p3),
                    get_weight(point, p2, p1, p3),
                    get_weight(point, p3, p2, p1),
                ]
            )
            values[idx] = numpy.sum(weights * self.value(nearby_cells))

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
        np_index: :class:`numpy.ndarray`
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
