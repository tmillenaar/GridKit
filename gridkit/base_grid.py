import abc
import functools
import warnings
from collections.abc import Iterable

import numpy
import scipy
import shapely
from pyproj import CRS, Transformer

from gridkit.index import GridIndex, validate_index


class BaseGrid(metaclass=abc.ABCMeta):
    def __init__(self, offset=(0, 0), crs=None):
        # limit offset to be positive and max 1 cell-size
        offset_x, offset_y = offset[0], offset[1]
        offset_x = offset_x % self.dx
        offset_y = offset_y % self.dy
        self._offset = (offset_x, offset_y)

        # set the CRS using crs.setter
        self._crs = None
        self.crs = crs

    @property
    def crs(self):
        """The Coordinate Reference System (CRS) represented as a ``pyproj.CRS`` object.

        Returns
        -------
        :class:`pyproj.CRS`
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
        pass

    @abc.abstractmethod
    def dy(self) -> float:
        pass

    @property
    def offset(self) -> float:
        return self._offset

    @abc.abstractmethod
    def centroid(self) -> float:
        pass

    @abc.abstractmethod
    def cells_near_point(self) -> float:
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
        fill_value: :class:`numpy.dtype` (optional)
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

    def neighbours(self, index, depth=1, connect_corners=False, include_selected=False):
        """The indices of the neighbouring cells.
        The argument 'depth' can be used to include cells from further away.

        Parameters
        ----------
        index: :class:`numpy.ndarray`
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
        """
        if not isinstance(index, numpy.ndarray):
            index = numpy.array(index)

        neighbours = self.relative_neighbours(
            depth=depth,
            connect_corners=connect_corners,
            include_selected=include_selected,
            index=index,
        )

        if len(index.shape) == 1:
            return neighbours + index

        # neighbours = numpy.repeat(neighbours[:, numpy.newaxis], len(index), axis=1)
        neighbours = numpy.swapaxes(neighbours, 0, 1)
        neighbours = neighbours + index
        return GridIndex(numpy.swapaxes(neighbours, 0, 1))

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
    def cell_corners(
        self, index: numpy.ndarray, as_poly: bool = False
    ) -> numpy.ndarray:
        """Determine the corners of the cells as specified by `index`.

        Parameters
        ----------
        index: :class:`numpy.ndarray`
            The indices of the cells of interest. Each id contains an `x` and `y` value.
        as_poly: :class:`bool`
            Toggle that determines whether to return a :class:`numpy.ndarray` (False) or a :class:`shapely.MultiPolygon` (True).

        Returns
        -------
        :class:`tuple`
            The ID of the cell in (x,y)
        """
        pass

    @abc.abstractmethod
    def is_aligned_with(self, other):
        pass

    @abc.abstractmethod
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
        index: :class:`numpy.ndarray`
            The indices of the cells to convert to Shapely Polygons
        as_multipolygon: :class:`numpy.ndarray`
            Returns a Shapely MultiPolygon if True, returns a list of Shapely Polygons if False

        See also
        --------
        :meth:`.BoundedRectGrid.to_shapely`
        :meth:`.BoundedHexGrid.to_shapely`
        """
        cell_arr_shape = index.shape
        vertices = self.cell_corners(index.ravel())
        polygons = [shapely.geometry.Polygon(cell) for cell in vertices]
        if as_multipolygon:
            return shapely.geometry.MultiPolygon(polygons)
        return numpy.array(polygons).reshape(cell_arr_shape)

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
        point: :class:`numpy.ndarray`
            A 2d numpy array containing the points in the form [[x1,y1], [x2,y2]]
        values: :class:`numpy.ndarray`
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
        centroids = self.centroid(ids)

        interp_values.ravel()[:] = interpolator(centroids.ravel())
        grid_kwargs = dict(
            data=interp_values,
            bounds=aligned_bounds,
            crs=self.crs,
            nodata_value=nodata_value,
        )
        if hasattr(self, "_shape"):
            grid_kwargs["shape"] = self._shape
        return self.bounded_cls(**grid_kwargs)
