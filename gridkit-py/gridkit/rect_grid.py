import warnings
from typing import Literal, Tuple, Union

import numpy
import scipy
from pyproj import CRS, Transformer

from gridkit.base_grid import BaseGrid
from gridkit.bounded_grid import BoundedGrid
from gridkit.errors import AlignmentError, IntersectionError
from gridkit.gridkit_rs import PyO3RectGrid
from gridkit.index import GridIndex, validate_index


class RectGrid(BaseGrid):
    """Abstraction that represents an infinite grid with square cell shape.

    Initialization parameters
    -------------------------
    dx: `float`
        The spacing between two cell centroids in horizontal direction.
        Has to be supplied together with `dx`.
        Cannot be supplied together with `area`, 'side_length' or `size`.
    dy: `float`
        The spacing between two cell centroids in vertical direction
        Has to be supplied together with `dy`.
        Cannot be supplied together with `area`, 'side_length' or `size`.
    size: `float`
        The spacing between two cell centroids in horizontal and vertical direction.
        Cannot be supplied together with `area`, 'side_length' or `dx`&`dy`.
    area: `float`
        The area of a cell. Cannot be supplied together with `size`, 'side_length' or `dx`&`dy`.
    side_length: `float`
        The lenght of the cell sides, i.e. the height and width of the cell. Cannot be supplied together with `size`, 'area' or `dx`&`dy`.
    offset: `Tuple[float, float]`, (0,0)
        The offset in dx and dy.
        Shifts the whole grid by the specified amount.
        The shift is always reduced to be maximum one cell size.
        If the supplied shift is larger,
        a shift will be performed such that the new center is a multiple of dx or dy away.
    rotation: float
        The counter-clockwise rotation of the grid around the origin in degrees.
    crs: `pyproj.CRS`, None
        The coordinate reference system of the grid.
        The value can be anything accepted by pyproj.CRS.from_user_input(),
        such as an epsg integer (eg 4326), an authority string (eg “EPSG:4326”) or a WKT string.

    See also
    --------
    :class:`.TriGrid`
    :class:`.HexGrid`
    :class:`.BoundedRectGrid`

    """

    def __init__(
        self,
        *args,
        dx=None,
        dy=None,
        size=None,
        area=None,
        side_length=None,
        offset=(0, 0),
        rotation=0,
        **kwargs,
    ):
        # Make sure only one method of defining cell size is used

        if ((dx is not None) + (dy is not None)) == 1:
            raise ValueError(
                f"Please supply both 'dx' and 'dy'. Got only '{'dx' if dx is not None else 'dy'}' and not '{'dx' if dx is None else 'dy'}'."
            )

        supplied_sizes = set()
        if dx is not None and dy is not None:
            supplied_sizes.add("dx+dy")
        if area is not None:
            supplied_sizes.add("area")
        if size is not None:
            supplied_sizes.add("size")
        if side_length is not None:
            supplied_sizes.add("side_length")
        if len(supplied_sizes) == 0:
            raise ValueError(
                "No cell size can be determined. Please supply either 'size', 'area', 'side_length' or both 'dx' and 'dy'."
            )
        if len(supplied_sizes) > 1:
            raise ValueError(
                f"Argument conflict. Please supply either 'size', 'area', 'side_length' or 'dx'&'dy' when instantiating a new RectGrid. Found: {supplied_sizes}."
            )

        # Determine cell size
        if size is not None:
            self._size = dx = dy = size
        elif area is not None:
            self._size = dx = dy = self._area_to_size(area)
        elif side_length is not None:
            self._size = dx = dy = self._side_length_to_size(side_length)
        else:
            if dx is None or dy is None:
                raise ValueError(
                    f"Found only '{'dx' if dx is not None else 'dy'}' when instantiating a new RectGrid. Please also supply '{'dx' if dx is None else 'dy'}'."
                )
            self._size = dx if numpy.isclose(dx, dy) else None

        # Instantiate attributes
        self._dx = dx
        self._dy = dy
        self._rotation = rotation
        self._grid = PyO3RectGrid(dx=dx, dy=dy, offset=tuple(offset), rotation=rotation)
        self.bounded_cls = BoundedRectGrid

        super(RectGrid, self).__init__(*args, **kwargs)

    @property
    def definition(self):
        return dict(
            dx=self.dx,
            dy=self.dy,
            offset=self.offset,
            rotation=self.rotation,
            crs=self.crs,
        )

    @property
    def side_length(self):
        """The lenght of the side of a cell.
        The length is the same as that of :meth:`.BaseGrid.cell_width`.
        In the case that cell_width and cell_height are different, only cell_widht is returned and cell_height is ignored.
        A warning is raised if that happens.
        It is advised to use :meth"`.RectGrid.cell_width` and :meth"`.RectGrid.cell_width` when dealing with RectGrids.
        This function is only implemented to keep the API aligned with the other grid types.
        """
        if not numpy.isclose(self.dx, self.dy):
            warnings.warn(
                "Not all side length are the same. Returning only 'cell_width' but note that it is different from 'cell_height'."
            )
        return self.cell_width

    def _side_length_to_size(self, side_length):
        """Find the ``size`` that corresponds to the specified length of the side of a cell.
        In the case of a RectGrid that is 1/4th the outline of the cell."""
        return side_length

    def _area_to_size(self, area):
        """Find the ``size`` that corresponds to a specific area."""
        return area**0.5

    @property
    def dx(self) -> float:
        """The cellsize in x-direction"""
        return self._dx

    @dx.setter
    def dx(self, value):
        """Set the cellsize in x-direction"""
        if value <= 0:
            raise ValueError(
                f"Size of cell cannot be set to '{value}', must be larger than zero"
            )
        self._dx = value
        self._grid = self._update_inner_grid(dx=value)

    @property
    def dy(self) -> float:
        """The cellsize in y-direction"""
        return self._dy

    @dy.setter
    def dy(self, value):
        """Set the cellsize in y-direction"""
        if value <= 0:
            raise ValueError(
                f"Size of cell cannot be set to '{value}', must be larger than zero"
            )
        self._dy = value
        self._grid = self._update_inner_grid(dy=value)

    @property
    def size(self) -> float:
        """The length of the cell sides, if all sides are of the same length.
        The returned size is 'None' if :meth:`.RectGrid.dx` and :meth:`.RectGrid.dy` are not the same length.

        See also
        --------
        :meth:`.BaseGrid.size`

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
        self._dx = value
        self._dy = value
        self._grid = self._update_inner_grid(dx=value, dy=value)

    def to_bounded(self, bounds, fill_value=numpy.nan):
        _, shape = self.cells_in_bounds(bounds, return_cell_count=True)
        data = numpy.full(shape=shape, fill_value=fill_value)
        return self.bounded_cls(
            data=data,
            bounds=bounds,
            nodata_value=fill_value,
            crs=self.crs,
        )

    def relative_neighbours(
        self, depth=1, connect_corners=False, include_selected=False, index=None
    ):
        """The relative indices of the neighbouring cells.

        Parameters
        ----------
        depth: `int`, 1
            Determines the number of neighbours that are returned.
            If `depth=1` the direct neighbours are returned.
            If `depth=2` the direct neighbours are returned, as well as the neighbours of these neighbours.
            `depth=3` returns yet another layer of neighbours, and so forth.
        include_selected: `bool`, False
            Whether to include the specified cell in the return array.
            Even though the specified cell can never be a neighbour of itself,
            this can be useful when for example a weighted average of the neighbours is desired
            in which case the cell itself often should also be included.
        connect_corners: `bool`, False
            Whether to consider cells that touch corners but not sides as neighbours.
            If `connect_corners` is True, the 4 cells directly touching the cell are considered neighbours.
            If `connect_corners` is True, the 8 cells surrounding the cell are considered neighbours.
            This escalates in combination with `depth` where indices in a square shape around the cell are returned
            when `connect_corners` is True, and indices in a diamond shape around the cell are returned when `connect_corners` is False.
        index: `numpy.ndarray`
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
        :py:meth:`.TriGrid.relative_neighbours`
        """
        depth = int(depth)
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

    @validate_index
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
        centroids = self._grid.centroid(index=index)
        return centroids.reshape(original_shape)

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
                    [ 0,  0],
                    [-1, -1],
                    [ 0, -1]],
            <BLANKLINE>
                   [[ 0,  1],
                    [ 1,  1],
                    [ 0,  0],
                    [ 1,  0]],
            <BLANKLINE>
                   [[ 0,  0],
                    [ 1,  0],
                    [ 0, -1],
                    [ 1, -1]]])

        ..

        """
        point = numpy.array(point, dtype=float)
        original_shape = (*point.shape[:-1], 4, 2)
        point = point[None] if point.ndim == 1 else point
        point = point.reshape(-1, 2)
        ids = self._grid.cells_near_point(point)
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
        point = numpy.array(point, dtype=float)
        original_shape = point.shape
        point = point.reshape(-1, 2)
        cell_ids = self._grid.cell_at_points(point)
        return GridIndex(cell_ids.reshape(original_shape))

    @validate_index
    def cell_corners(self, index: GridIndex = None) -> numpy.ndarray:
        if index is None:
            raise ValueError(
                "For grids that do not contain data, argument `index` is to be supplied to method `corners`."
            )
        original_shape = (*index.shape, 4, 2)
        index = (
            index.ravel().index[None] if index.index.ndim == 1 else index.ravel().index
        )
        corners = self._grid.cell_corners(index)
        return corners.reshape(original_shape)

    def to_crs(self, crs, location=(0, 0), adjust_rotation=False):
        """Transforms the Coordinate Reference System (CRS) from the current CRS to the desired CRS.
        This will modify the cell size and the bounds accordingly.

        The ``crs`` attribute on the current GeometryArray must be set.

        Parameters
        ----------
        crs: Union[int, str, pyproj.CRS]
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an epsg integer (eg 4326), an authority string (eg "EPSG:4326") or a WKT string.

        location: `Tuple(float, float)`, (0,0)
            The location at which to perform the conversion.
            When transforming to a new coordinate system, it matters at which location the transformation is performed.
            The chosen location will be used to determinde the cell size of the new grid.
            If you are unsure what location to use, pich the center of the area you are interested in.

            .. Warning ::

                The location is defined in the original CRS, not in the CRS supplied as the argument to this function call.

        adjust_rotation: bool
            If False, the grid in the new crs has the same rotation as the original grid.
            Since coordinate transformations often warp and rotate the grid, the original rotation is often not a good fit anymore.
            If True, set the new rotation to match the orientation of the grid at ``location`` after coordinate transformation.
            Default: False

        Returns
        -------
        :class:`~.rect_grid.RectGrid`
            A copy of the grid with modified cell spacing and bounds to match the specified CRS

        See also
        --------
        Examples:

        :ref:`Example: coordinate transformations <example coordinate transformations>`

        Methods:

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

        new_offset = transformer.transform(
            location[0] + self.offset[0], location[1] + self.offset[1]
        )

        if adjust_rotation:
            # Corner order is bottom-left, bottom-right, top-right, top-left
            trans_corners = numpy.array(
                transformer.transform(
                    *self.cell_corners(self.cell_at_point(location)).T
                )
            ).T

            new_dx = numpy.linalg.norm(trans_corners[2] - trans_corners[1])
            new_dy = numpy.linalg.norm(trans_corners[1] - trans_corners[0])

            if (new_dy > new_dx) == (self.dy > self.dx):
                vector = trans_corners[2] - trans_corners[1]
                rotation = numpy.degrees(numpy.arctan2(vector[1], vector[0]))
            else:
                tmp_dx = new_dx
                new_dx = new_dy
                new_dy = tmp_dx
                vector = trans_corners[1] - trans_corners[0]
                rotation = numpy.degrees(numpy.arctan2(vector[1], vector[0]))
        else:
            point_start = numpy.array(transformer.transform(*location))
            new_dx = numpy.linalg.norm(
                point_start
                - numpy.array(transformer.transform(location[0] + self.dx, location[1]))
            )
            new_dy = numpy.linalg.norm(
                point_start
                - numpy.array(transformer.transform(location[0], location[1] + self.dy))
            )
            rotation = self.rotation

        new_grid = self.parent_grid_class(
            dx=new_dx,
            dy=new_dy,
            offset=new_offset,
            crs=crs,
            rotation=rotation,
        )

        if adjust_rotation:
            new_grid.anchor(trans_corners[0], cell_element="corner", in_place=True)
        return new_grid

    def cells_in_bounds(self, bounds, return_cell_count: bool = False):
        """Cells contained within a bounding box.

        Parameters
        ----------
        bounds: :class:`tuple`
        return_cell_count: :class:`bool`
            Return a tuple containing the nr of cells in x and y direction inside the provided bounds

        Returns
        -------
        :class:`.GridIndex`
            The indices of the cells contained in the bounds
        """

        if self.rotation != 0:
            raise NotImplementedError(
                f"`cells_in_bounds` is not suppored for rotated grids. Roatation: {self.rotation} degrees"
            )

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

        ids = GridIndex(ids.T.reshape((*shape, 2)))
        return (ids, shape) if return_cell_count else ids

    def subdivide(self, factor: int):
        """Create a new grid that is ``factor`` times smaller than the existing grid and aligns perfectly
        with it.

        If ``factor`` is one, the side lengths of the new cells will be of the same size as
        the side lengths of the original cells, which means that the two grids will be exactly the same.
        If ``factor`` is two, the new cell sides will be half the size of the original cell sides.
        The number of cells grows quadratically with ``factor``.
        A ``factor`` of 2 results in 4 cells that fit in the original, a `factor` of 3 results in 9
        cells that fit in the original, etc..

        Parameters
        ----------
        factor: `int`
            An integer (whole number) indicating how many times smaller the new gridsize will be.
            It refers to the side of a grid cell. If ``factor`` is 1, the new grid will have cell sides
            of the same length as the cell sides of the original.
            If ``factor`` is 2, the side of the grid cell will be half the cell side length of the original.

        Returns
        -------
        :class:`.RectGrid`
            A new grid that is ``factor`` times smaller then the original grid.

        """
        if not factor % 1 == 0:
            raise ValueError(
                f"Got a 'factor' that is not a whole number. Please supply an integer. Got: {factor}"
            )

        sub_grid = self.update(
            dx=self.dx / factor, dy=self.dy / factor, rotation=self.rotation
        )
        anchor_loc = self.cell_corners([0, 0])[0]
        sub_grid.anchor(anchor_loc, cell_element="corner", in_place=True)
        return sub_grid

    @property
    def parent_grid_class(self):
        return RectGrid

    def _update_inner_grid(
        self, dx=None, dy=None, size=None, offset=None, rotation=None
    ):
        if size is not None and (dx is not None or dy is not None):
            raise ValueError(
                f"Argument conflict. Please supply either 'size' or 'dx'&'dy'. Found both."
            )
        if size is not None:
            dx = dy = size
        if dx is None:
            dx = self.dx
        if dy is None:
            dy = self.dy
        if offset is None:
            offset = self.offset
        if rotation is None:
            rotation = self.rotation
        return PyO3RectGrid(dx=dx, dy=dy, offset=offset, rotation=rotation)

    def update(
        self,
        dx=None,
        dy=None,
        size=None,
        area=None,
        offset=None,
        rotation=None,
        crs=None,
        **kwargs,
    ):
        """Modify attributes of the existing grid and return a copy.
        The original grid remains un-mutated.

        Parameters
        ----------
        dx: float
            The new horizontal spacing between two cell centroids, i.e. the new width of the cells
        dy: float
            The new vertical spacing between two cell centroids, i.e. the new height of the cells
        size: float
            The new size of the length of the cells (dx and dy)
        area: float
            The area of a cell. Cannot be supplied together with `size` or `dx`&`dy`.
        offset: Tuple[float, float]
            The new offset of the origin of the grid
        rotation: float
            The new counter-clockwise rotation of the grid in degrees.
            Can be negative for clockwise rotation.
        crs: Union[int, str, pyproj.CRS]
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an epsg integer (eg 4326), an authority string (eg "EPSG:4326") or a WKT string.

        Returns
        -------
        :class:`.RectGrid`
            A modified copy of the current grid
        """
        if not any([dx, size, area]):
            dx = self.dx
        if not any([dy, size, area]):
            dy = self.dy
        if offset is None:
            offset = self.offset
        if rotation is None:
            rotation = self.rotation
        if crs is None:
            crs = self.crs
        return RectGrid(
            dx=dx,
            dy=dy,
            size=size,
            area=area,
            offset=offset,
            rotation=rotation,
            crs=crs,
            **kwargs,
        )


class BoundedRectGrid(BoundedGrid, RectGrid):
    """
    Initialization parameters
    -------------------------
    data: `numpy.ndarray`
        A 2D ndarray containing the data
    bounds: :class:Tuple(float, float, float, float)
        The extend of the data in minx, miny, maxx, maxy.
    crs: `pyproj.CRS` (optional)
        The coordinate reference system of the grid.
        The value can be anything accepted by pyproj.CRS.from_user_input(),
        such as an epsg integer (eg 4326), an authority string (eg “EPSG:4326”) or a WKT string.
        Default: None

    See also
    --------
    :class:`.RectGrid`
    :class:`.BoundedTriGrid`
    :class:`.BoundedHexGrid`

    """

    def __init__(self, data, *args, bounds=None, **kwargs):

        data = numpy.array(data) if not isinstance(data, numpy.ndarray) else data

        if data.ndim != 2:
            raise ValueError(
                f"Expected a 2D numpy array, got data with shape {data.shape}"
            )

        if bounds is None:
            bounds = (0, 0, data.shape[1], data.shape[0])

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

    def centroid(self, index=None):
        """Centroids of all cells

        Returns
        -------
        `numpy.ndarray`
            Multidimensional array containing the longitude and latitude of the center of each cell respectively,
            in (width, height, lonlat)
        """
        if index is None:
            if not hasattr(self, "indices"):
                raise ValueError(
                    "For grids that do not contain data, argument `index` is to be supplied to method `centroid`."
                )
            index = self.indices
        return super(BoundedRectGrid, self).centroid(index=index)

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
        cropped_data = self._data[
            slice_y, slice_x
        ]  # Note: slicing numpy arrays is in order (y,x)
        return self.update(cropped_data, bounds=new_bounds)

    def cell_corners(self, index: numpy.ndarray = None) -> numpy.ndarray:
        if index is None:
            index = self.indices()
        return super(BoundedRectGrid, self).cell_corners(index=index)

    def to_shapely(self, index=None, as_multipolygon=None):
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
        return super().to_shapely(index, as_multipolygon=as_multipolygon)

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
        nodata_value = self.nodata_value if self.nodata_value is not None else numpy.nan
        tl_ids, tr_ids, bl_ids, br_ids = numpy.rollaxis(
            self.cells_near_point(sample_points).index, -2
        )

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
            The resampling method to be used for :meth:`.BoundedGrid.resample`.

        Returns
        -------
        :class:`~.rect_grid.BoundedRectGrid`
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
        new_inf_grid = super(BoundedRectGrid, self).to_crs(crs)
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

    def anchor(
        self,
        target_loc: Tuple[float, float],
        cell_element: Literal["centroid", "corner"] = "centroid",
        resample_method="nearest",
    ):
        """Position a specified part of a grid cell at a specified location.
        This shifts (the origin of) the grid such that the specified ``cell_element`` is positioned at the specified ``target_loc``.
        This is useful for example to align two grids by anchoring them to the same location.
        The data values for the new grid will need to be resampled since it has been shifted.

        Parameters
        ----------
        target_loc: Tuple[float, float]
            The coordinates of the point at which to anchor the grid in (x,y)
        cell_element: Literal["centroid", "corner"] - Default: "centroid"
            The part of the cell that is to be positioned at the specified ``target_loc``.
            Currently only "centroid" and "corner" are supported.
            When "centroid" is specified, the cell is centered around the ``target_loc``.
            When "corner" is specified, a nearby cell_corner is placed onto the ``target_loc``.
        resample_method: :class:`str`
            The resampling method to be used for :meth:`.BoundedGrid.resample`.

        Returns
        -------
        :class:`.BoundedGrid`
            The shifted and resampled grid

        See also
        --------

        :meth:`.BaseGrid.anchor`
        """
        new_inf_grid = self.parent_grid.anchor(target_loc, cell_element, in_place=False)
        return self.resample(new_inf_grid, method=resample_method)

    def interp_nodata(self, *args, **kwargs):
        """Please refer to :func:`~gridkit.bounded_grid.BoundedGrid.interp_nodata`."""
        # Fixme: in the case of a rectangular grid, a performance improvement can be obtained by using scipy.interpolate.interpn
        return super(BoundedRectGrid, self).interp_nodata(*args, **kwargs)
