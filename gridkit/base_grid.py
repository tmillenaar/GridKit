import abc
import functools
import warnings
from collections.abc import Iterable

import numpy
import scipy
import shapely
from pyproj import CRS, Transformer

from gridkit.gridkit_rs import shapes
from gridkit.index import GridIndex, validate_index


class BaseGrid(metaclass=abc.ABCMeta):
    """Abstraction base class that represents an infinite grid.

    Initialization parameters
    -------------------------
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
        If None, no CRS is set.
        Default: None
    """

    def __init__(self, crs=None):
        # set the CRS using crs.setter
        self._crs = None
        self.crs = crs

    @property
    def crs(self):
        """The Coordinate Reference System (CRS) represented as a ``pyproj.CRS`` object.

        Returns
        -------
        `pyproj.CRS`
            None if the CRS is not set, and to set the value it
            :getter: Returns a ``pyproj.CRS`` or None. When setting, the value
            Coordinate Reference System of the geometry objects. Can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        """
        return self._crs

    @crs.setter
    def crs(self, value):
        """Sets the value of the crs"""
        self._crs = None if not value else CRS.from_user_input(value)

    @abc.abstractmethod
    def dx(self) -> float:
        """The distance in x-direction between two adjacent cell centers."""
        pass

    @abc.abstractmethod
    def dy(self) -> float:
        """The distance in y-direction between two adjacent cell centers."""
        pass

    @property
    def cell_height(self) -> float:
        """The height of a cell"""
        return self._grid.cell_height()

    @property
    def cell_width(self) -> float:
        """The width of a cell"""
        return self._grid.cell_width()

    @property
    def offset(self) -> float:
        """The offset off the grid in dx and dy.
        The offset is never larger than the size of a single grid cell.
        The offset represents the shift from the origin (0,0)."""
        return self._grid.offset()

    @offset.setter
    def offset(self, value):
        """Sets the x and y value of the offset"""
        if not isinstance(value, tuple) or not len(value) == 2:
            raise TypeError(f"Expected a tuple of length 2. Got: {value}")
        if getattr(self, "shape", None) == "flat":  # flat hex grid
            value = value[::-1]  # swap xy to yx
        new_offset = (value[0] % self.cell_width, value[1] % self.cell_height)
        self._offset = new_offset
        self._grid = self._update_inner_grid(offset=new_offset)

    @property
    def rotation(self) -> float:
        """The counter-clockwise rotation of the grid around the origin in degrees."""
        rotation = self._rotation
        if getattr(self, "shape", None) == "flat":  # flat hex grid
            rotation = -rotation
        return rotation

    @rotation.setter
    def rotation(self, value):
        """The counter-clockwise rotation of the grid around the origin in degrees."""
        if getattr(self, "shape", None) == "flat":  # flat hex grid
            value = -value
        self._rotation = value
        self._grid = self._update_inner_grid(rotation=value)

    @property
    def rotation_matrix(self):
        """The matrix performing the counter-clockwise rotation of the grid around the origin in degrees.
        Note: makes a copy every time this is called."""
        return self._grid.rotation_matrix()

    @property
    def rotation_matrix_inv(self):
        """The matrix performing the inverse (clockwise) rotation of the grid around the origin in degrees.
        Note: makes a copy every time this is called."""
        return self._grid.rotation_matrix_inv()

    @abc.abstractmethod
    def centroid(self, index) -> float:
        """Coordinates at the center of the cell(s) specified by ``index``.

        If this method is called on a 'Bounded' class, the ``index`` argument is optional.
        In such a case the cell IDs of the cells contained in the Bounded product are returned.

        Parameters
        ----------
        index: :class:`.GridIndex`
            The index of the cell(s) of which the centroid is to be obtained.

        Returns
        -------
        `numpy.ndarray`
            Multidimensional array containing the longitude and latitude of the center of each cell respectively,
            in (width, height, lonlat)

        Raises
        ------
        ValueError
            No `index` parameter was supplied to a grid that does not contain data.
        """
        pass

    @abc.abstractmethod
    def cells_near_point(self, point) -> float:
        """The cells nearest to a point, often used in interpolation at the location of the point.
        For a TriGrid there are 6 nearby points.
        For a HexGrid there are 3 nearby points.
        For a RectGrid there are 4 nearby points.
        """
        pass

    @abc.abstractmethod
    def to_bounded(self, bounds, fill_value=numpy.nan):
        """Create a bounded version of this grid where the data in the bounds is filled with the supplied `fill_value`

        Parameters
        ----------
        bounds: :class:`tuple`
            The bounds of the area of interest in (minx, miny, maxx, maxy).
            The bounds need to be aligned to the grid.
            See :meth:`.BaseGrid.align_bounds`
        fill_value: `numpy.dtype` (optional)
            The value to assign to the newly created array that fills the supplied bounds.
            Default: numpy.nan

        Returns
        -------
        :class:`.BoundedHexGrid` or :class:`.BoundedRectGrid`
            A bounded version of the current grid where the data is filled with `fill_value`.

        See also
        --------
        :meth:`.BaseGrid.to_bounded`
        :meth:`.RectGrid.to_bounded`
        :meth:`.HexGrid.to_bounded`
        """
        pass

    @abc.abstractmethod
    def relative_neighbours(
        self, depth=1, connect_corners=False, include_selected=False
    ):
        pass

    @validate_index
    def neighbours(self, index, depth=1, connect_corners=False, include_selected=False):
        """The indices of the neighbouring cells.
        The argument 'depth' can be used to include cells from further away.

        Parameters
        ----------
        index: `numpy.ndarray`
            The index of the cell(s) of which to get the neighbours.
        depth: :class:`int` Default: 1
            Determines the number of neighbours that are returned.
            If `depth=1` the direct neighbours are returned.
            If `depth=2` the direct neighbours are returned, as well as the neighbours of these neighbours.
            `depth=3` returns yet another layer of neighbours, and so forth.
        connect_corners: :class:`bool` Default: False
            Whether to consider cells that touch corners but not sides as neighbours.
            This does not apply to :py:meth:`.HexGrid.relative_neighbours`.
            For further information on this argument, refer to :py:meth:`.RectGrid.relative_neighbours`.
        include_selected: :class:`bool` Default: False
            Whether to include the specified cell in the return array.
            Even though the specified cell can never be a neighbour of itself,
            this can be useful when for example a weighted average of the neighbours is desired
            in which case the cell itself often should also be included.

        Examples
        --------
        The direct neighbours of a cell can be returned by using depth=1, which is the default.

        .. code-block:: python

            >>> from gridkit.rect_grid import RectGrid
            >>> grid = RectGrid(dx=2, dy=3)
            >>> grid.neighbours([1,2]).index
            array([[1, 3],
                   [0, 2],
                   [2, 2],
                   [1, 1]])

        ..

        For more detailed examples:

        See also
        --------
        :py:meth:`.RectGrid.relative_neighbours`
        :py:meth:`.HexGrid.relative_neighbours`
        :py:meth:`.TriGrid.relative_neighbours`
        """
        original_shape = index.shape
        index = index.ravel()

        neighbours = self.relative_neighbours(
            depth=depth,
            connect_corners=connect_corners,
            include_selected=include_selected,
            index=index,
        )

        if len(index.index.shape) == 1:
            return neighbours + index

        # neighbours = numpy.repeat(neighbours[:, numpy.newaxis], len(index), axis=1)
        neighbours = numpy.swapaxes(neighbours, 0, 1)
        neighbours = neighbours + index
        neighbours = numpy.swapaxes(neighbours, 0, 1)

        return GridIndex(
            neighbours.reshape(*original_shape, *neighbours.shape[-2:]).squeeze()
        )

    @abc.abstractmethod
    def cell_at_point(self, point: numpy.ndarray) -> tuple:
        """Determine the ID of the cell in which `point` falls.

        Parameters
        ----------
        point: :class:`tuple`
            The coordinates of the point to which to match the cell

        Returns
        -------
        :class:`tuple`
            The ID of the cell in (x,y)
        """
        pass

    @abc.abstractmethod
    def cell_corners(self, index: numpy.ndarray) -> numpy.ndarray:
        """Coordinates of the cell corners as specified by ``index``.

        Parameters
        ----------
        index: :class:`.GridIndex`
            The indices of the cells of interest. Each id contains an `x` and `y` value.

        Returns
        -------
        `numpy.ndarray`
            An array of coordinates in (x,y) specifying each of the corners.
            The returned array will be of the same shape as the input ``index``,
            but with an extra axis containing the corners.
            The last axis is always of size 2 (x,y).
            The second to last axis is the length of the corners.
            The other axis are in the shape of the supplied index.
        """
        pass

    def is_aligned_with(self, other):
        """
        Returns True if grids are algined and False if they are not.
        Grids are considered to be aligned when:
         - they are the same type of grid
         - the CRS is the same
         - the cell_size is the same
         - the offset from origin is the same
         - the cell shape is the same

        Returns
        -------
        :class:`bool`
            Whether or not the grids are aligned
        """
        if not isinstance(other, BaseGrid):
            raise TypeError(f"Expected a (child of) BaseGrid, got {type(other)}")
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

        try:
            if not numpy.isclose(self.size, other.size):
                aligned = False
                reasons.append("cellsize")
        except AttributeError:
            if not numpy.isclose(self.dx, other.dx) or not numpy.isclose(
                self.dy, other.dy
            ):
                aligned = False
                reasons.append("cellsize")

        if not all(
            numpy.isclose(self.offset, other.offset, atol=1e-7)
        ):  # FIXME: atol if 1e-7 is a bandaid. It seems the offset depends slightly depending on the bounds after resampling on grid
            aligned = False
            reasons.append("offset")

        if getattr(self, "shape", "") != getattr(other, "shape", ""):
            aligned = False
            reasons.append("shape")

        reason = (
            f"The following attributes are not the same: {reasons}"
            if reasons
            else reason
        )
        return aligned, reason

    @abc.abstractmethod
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

        See also
        --------
        Examples:

        :ref:`Example: coordinate transformations <example coordinate transformations>`

        Methods:

        :meth:`.RectGrid.to_crs`
        :meth:`.HexGrid.to_crs`
        :meth:`.BoundedRectGrid.to_crs`
        :meth:`.BoundedHexGrid.to_crs`
        """
        pass

    @property
    def parent_grid(self):
        return self.parent_grid_class(
            dx=self.dx, dy=self.dy, offset=self.offset, crs=self.crs
        )

    @abc.abstractproperty
    def parent_grid_class(self):
        pass

    @abc.abstractmethod
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

    def are_bounds_aligned(self, bounds, separate=False):
        is_aligned = lambda val, cellsize: numpy.isclose(val, 0) or numpy.isclose(
            val, cellsize
        )
        per_axis = (
            is_aligned((bounds[0] - self.offset[0]) % self.dx, self.dx),  # left
            is_aligned((bounds[1] - self.offset[1]) % self.dy, self.dy),  # bottom
            is_aligned((bounds[2] - self.offset[0]) % self.dx, self.dx),  # right
            is_aligned((bounds[3] - self.offset[1]) % self.dy, self.dy),  # top
        )
        return per_axis if separate else numpy.all(per_axis)

    def align_bounds(self, bounds, mode="expand"):
        if self.are_bounds_aligned(bounds):
            return bounds

        if mode == "expand":
            left_rounded = numpy.floor((bounds[0] - self.offset[0]) / self.dx)
            bottom_rounded = numpy.floor((bounds[1] - self.offset[1]) / self.dy)
            right_rounded = numpy.ceil((bounds[2] - self.offset[0]) / self.dx)
            top_rounded = numpy.ceil((bounds[3] - self.offset[1]) / self.dy)
        elif mode == "contract":
            left_rounded = numpy.ceil((bounds[0] - self.offset[0]) / self.dx)
            bottom_rounded = numpy.ceil((bounds[1] - self.offset[1]) / self.dy)
            right_rounded = numpy.floor((bounds[2] - self.offset[0]) / self.dx)
            top_rounded = numpy.floor((bounds[3] - self.offset[1]) / self.dy)
        elif mode == "nearest":
            left_rounded = round((bounds[0] - self.offset[0]) / self.dx)
            bottom_rounded = round((bounds[1] - self.offset[1]) / self.dy)
            right_rounded = round((bounds[2] - self.offset[0]) / self.dx)
            top_rounded = round((bounds[3] - self.offset[1]) / self.dy)
        else:
            raise ValueError(
                f"mode = '{mode}' is not supported. Supported modes: ('expand', 'contract', 'nearest')"
            )

        return (
            left_rounded * self.dx + self.offset[0],
            bottom_rounded * self.dy + self.offset[1],
            right_rounded * self.dx + self.offset[0],
            top_rounded * self.dy + self.offset[1],
        )

    def intersect_geometries(self, geometries, suppress_point_warning=False):
        if not isinstance(geometries, Iterable):
            geometries = [geometries]
        intersecting_cells = []

        def _geom_iterator():
            """Unpack `Multi` geometries, improving performance if the geometries are far apart."""
            for geometry in geometries:
                try:
                    for geom in geometry.geoms:
                        yield geom
                except:
                    yield geometry

        for geom in _geom_iterator():
            if isinstance(geom, shapely.geometry.Point):
                if not suppress_point_warning:
                    warnings.warn(
                        "Point type geometry detected. It is more efficient to use `cell_at_point` than to use `intersect_geometries` when dealing with points"
                    )
                    suppress_point_warning = True  # Only warn once per function call
            geom_bounds = self.align_bounds(geom.bounds, mode="expand")
            geom_bounds = (  # buffer bounds to be on the safe side
                geom_bounds[0] - self.dx,
                geom_bounds[1] - self.dy,
                geom_bounds[2] + self.dx,
                geom_bounds[3] + self.dy,
            )
            cells_in_bounds = self.cells_in_bounds(geom_bounds).ravel()
            if (
                len(cells_in_bounds) == 0
            ):  # happens only if point or line lies on an edge
                geom = geom.buffer(
                    min(self.dx, self.dy) / 10
                )  # buffer may never reach further then a single cell size
                geom_bounds = self.align_bounds(geom.bounds, mode="expand")
                cells_in_bounds = self.cells_in_bounds(geom_bounds).ravel()

            cell_shapes = self.to_shapely(cells_in_bounds)
            mask = [geom.intersects(cell) for cell in cell_shapes]
            intersecting_cells.extend(cells_in_bounds[mask])
        return GridIndex(intersecting_cells).unique()

    @validate_index
    def to_shapely(self, index, as_multipolygon: bool = False):
        """Represent the cells as Shapely Polygons

        Parameters
        ----------
        index: `numpy.ndarray`
            The indices of the cells to convert to Shapely Polygons
        as_multipolygon: `numpy.ndarray`
            Returns a Shapely MultiPolygon if True, returns a list of Shapely Polygons if False

        See also
        --------
        :meth:`.BoundedRectGrid.to_shapely`
        :meth:`.BoundedHexGrid.to_shapely`
        """
        cell_arr_shape = index.shape
        vertices = self.cell_corners(index.ravel())
        if index.index.ndim == 1:
            return shapely.geometry.Polygon(vertices)
        if vertices.ndim == 2:
            vertices = vertices[numpy.newaxis]
        multipoly_wkb = shapes.multipolygon_wkb(vertices)
        multipoly = shapely.from_wkb(multipoly_wkb.hex())
        if as_multipolygon == True:
            return multipoly
        return numpy.array(multipoly.geoms).reshape(cell_arr_shape)

    def interp_from_points(
        self, points, values, method="linear", nodata_value=numpy.nan
    ):
        """Interpolate the cells containing nodata, if they are inside the convex hull of cells that do contain data.

        This function turns any set of points at arbitrary location into a regularly spaced :class:`.BoundedGrid`
        that has the properties of the current :class:`.BaseGrid` (``self``).
        :meth:`.BoundedGrid.interpolate` works in the other direction, where a values on a :class:`.BoundedGrid`
        are sampled in order to obtain values at arbitrary location.

        .. Note ::
            This function is significantly slower than :meth:`.BoundedGrid.interpolate`

        Parameters
        ----------
        point: `numpy.ndarray`
            A 2d numpy array containing the points in the form [[x1,y1], [x2,y2]]
        values: `numpy.ndarray`
            The values corresponding to the supplied `points`, used as input for interpolation
        method: :class:`str`
            The interpolation method to be used. Options are ("nearest", "linear", "cubic"). Default: "linear".

        Returns
        -------
        :class:`.BoundedGrid`
            A Bounded version of the supplied grid where the data is interpolated between the supplied points.

        See also
        --------
        :py:meth:`.BoundedGrid.resample`
        :py:meth:`.BoundedGrid.interpolate`
        """
        points = numpy.array(points)
        values = numpy.array(values)

        method_lut = dict(
            nearest=scipy.interpolate.NearestNDInterpolator,
            linear=functools.partial(
                scipy.interpolate.LinearNDInterpolator, fill_value=nodata_value
            ),
            cubic=functools.partial(
                scipy.interpolate.CloughTocher2DInterpolator, fill_value=nodata_value
            ),
        )

        if method not in method_lut:
            raise ValueError(
                f"Method '{method}' is not supported. Supported methods: {method_lut.keys()}"
            )

        coords = points.T
        bounds = (
            min(coords[0]),
            min(coords[1]),
            max(coords[0]),
            max(coords[1]),
        )
        aligned_bounds = self.align_bounds(bounds, mode="expand")
        ids, shape = self.cells_in_bounds(aligned_bounds, return_cell_count=True)
        interp_values = numpy.full(
            shape=shape, fill_value=nodata_value, dtype=values.dtype
        )

        interp_func = method_lut[method]
        nodata_mask = values == nodata_value

        interpolator = interp_func(
            points[~nodata_mask],
            values[~nodata_mask],
        )
        centroids = self.centroid(ids.ravel())

        interp_values.ravel()[:] = interpolator(centroids)
        grid_kwargs = dict(
            data=interp_values,
            bounds=aligned_bounds,
            crs=self.crs,
            nodata_value=nodata_value,
        )
        if hasattr(self, "_shape"):
            grid_kwargs["shape"] = self._shape
        return self.bounded_cls(**grid_kwargs)

    @abc.abstractmethod
    def _update_inner_grid(self, size=None, offset=None, rotation=None):
        pass

    @abc.abstractmethod
    def update(self, size=None, shape=None, offset=None, rotation=None, **kwargs):
        pass
