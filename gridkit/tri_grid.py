import numpy
from pyproj import CRS, Transformer

from gridkit.base_grid import BaseGrid
from gridkit.gridkit_rs import PyTriGrid
from gridkit.index import GridIndex, validate_index


class TriGrid(BaseGrid):
    def __init__(self, *args, size, offset=(0, 0), **kwargs):
        self._size = size
        self._radius = size / 3**0.5

        self._grid = PyTriGrid(cellsize=size, offset=offset)

        self.bounded_cls = None
        super(TriGrid, self).__init__(*args, offset=offset, **kwargs)

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

    @property
    def offset(self) -> float:
        """The offset off the grid in dx and dy.
        The offset is never larger than the (dx, dy) of a single grid cell.
        The offset represents the shift from the origin (0,0)."""
        return self._offset

    @offset.setter
    def offset(self, value):
        """Sets the x and y value of the offset"""
        if not isinstance(value, tuple) or not len(value) == 2:
            raise TypeError(f"Expected a tuple of length 2. Got: {value}")
        self._offset = value
        # TODO: implement a generalize update method that takes the PyTriGrid into account
        self._grid = PyTriGrid(cellsize=self.size, offset=value)

    @validate_index
    def centroid(self, index):
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

        """
        if index is None:
            raise ValueError(
                "For grids that do not contain data, argument `index` is to be supplied to method `centroid`."
            )
        index = (
            index.ravel().index[None] if index.index.ndim == 1 else index.ravel().index
        )
        return self._grid.centroid(index=index).squeeze()

    @validate_index
    def cell_corners(self, index):
        index = index.index[None] if index.index.ndim == 1 else index.index
        return self._grid.cell_corners(index=index).squeeze()

    def cell_at_point(self, point):
        point = numpy.array(point, dtype="float64")
        point = point[None] if point.ndim == 1 else point
        return GridIndex(self._grid.cell_at_point(point))

    def cells_in_bounds(self, bounds, return_cell_count=False):
        if not self.are_bounds_aligned(bounds):
            raise ValueError(
                f"supplied bounds '{bounds}' are not aligned with the grid lines. Consider calling 'align_bounds' first."
            )
        ids, shape = self._grid.cells_in_bounds(bounds)
        ids = GridIndex(ids)
        return (ids, shape) if return_cell_count else ids

    def cells_near_point(self, point):
        point = numpy.array(point, dtype="float64")
        point = point[None] if point.ndim == 1 else point
        ids = self._grid.cells_near_point(point)
        return GridIndex(ids)

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

    def to_bounded(self):
        raise NotImplementedError()

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
            self.dx / 2, self.dy
        )  # likely different for shape='flat'
        size = numpy.linalg.norm(numpy.subtract(point_end, point_start))

        return self.parent_grid_class(size=size, offset=new_offset, crs=crs)
