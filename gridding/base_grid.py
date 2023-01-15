import abc
import numpy
from pyproj import CRS, Transformer


class BaseGrid(metaclass=abc.ABCMeta):

    ## Attributes:
    # _crs
    # _origin
    # _rotation_matrix
    def __init__(self, offset=(0,0), crs=None):

        # limit offset to be positive and max 1 cell-size
        offset_x, offset_y = offset[0], offset[1]
        offset_x = offset_x % self.dx
        offset_y = offset_y % self.dy
        self._offset = (offset_x, offset_y)
        self._crs = None
        self.crs = crs

    @property
    def crs(self):
        """
        The Coordinate Reference System (CRS) represented as a ``pyproj.CRS``
        object.
        Returns None if the CRS is not set, and to set the value it
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
    def neighbours(self) -> float:
        pass

    @abc.abstractmethod
    def cell_at_point(self, point: numpy.array) -> tuple: 
        """Determine the ID of the cell in which `point` falls.

        Parameters
        ----------
        point: :class:tuple
            The coordinates of the point to which to match the cell

        Returns
        -------
        :calss:`tuple`
            The ID of the cell in (x,y)
        """
        pass

    @abc.abstractmethod
    def cell_corners(self, index: numpy.ndarray, as_poly: bool = False) -> numpy.ndarray:
        """Determine the corners of the cells as specified by `index`.

        Parameters:
        index: :class:`numpy.ndarray`
            The indices of the cells of interest. Each id contains an `x` and `y` value.
        as_poly: :class:`bool`
            Toggle that determines whether to return a :class:`numpy.ndarray` (False) or a :class:`shapely MultiPolygon` (True).

        Returns
        -------
        :calss:`tuple`
            The ID of the cell in (x,y)
        """
        pass

    @abc.abstractmethod
    def is_aligned_with(self, other):
        pass

    def interp_from_points(self) -> float:
        pass

    def to_crs(self, crs, resample_method="nearest"):
        """Transforms the 

        The ``crs`` attribute on the current GeometryArray must
        be set.  Either ``crs`` or ``epsg`` may be specified for output.
        This method will transform all points in all objects.  It has no notion
        or projecting entire geometries.  All segments joining points are
        assumed to be lines in the current projection, not geodesics.  Objects
        crossing the dateline (or other projection boundary) will have
        undesirable behavior.
        Arguments
        ----------
        crs : pyproj.CRS
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an epsg integer (eg 4326), an authority string (eg "EPSG:4326") or a WKT string.

        Returns
        -------
        
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
        new_dx, new_dy = transformer.transform(self.dx, self.dy)
        
        return self.parent_grid_class(dx=new_dx, dy=new_dy, offset=new_offset, crs=crs)

    @property
    def parent_grid(self):
        return self.parent_grid_class(dx=self.dx, dy=self.dy, offset=self.offset, crs=self.crs)

    @abc.abstractproperty
    def parent_grid_class(self):
        pass

    # def interp_from_points(self, points, values, method="linear", sparse=False):

    #     if sparse:
    #         raise NotImplementedError("Sparse grids are not yet implemented")

    #     # left, bottom = numpy.min(points, axis=0)
    #     # right, top = numpy.max(points, axis=0)
    #     bounds = [*numpy.min(points, axis=0), *numpy.max(points, axis=0)]
    #     breakpoint()
    #     ids = self.cell_at_point(points)

    #     from scipy.interpolate import LinearNDInterpolator
    #     # LinearNDInterpolator(points,values)(self.center(self.cell_at_point(points)))
    #     cell_centers = self.cells_in_bounds(bounds, as_coordinate=True)
    #     interpolator = LinearNDInterpolator(points,values)
    #     values = interpolator(cell_centers)

    def rotate(self) -> float:
        raise NotImplementedError()

    def rotation(self) -> float:
        raise NotImplementedError()

    def resample(self) -> float:
        raise NotImplementedError()

    def to_vector_file(self) -> float:
        raise NotImplementedError()

    def aggregate(self) -> float:
        pass
