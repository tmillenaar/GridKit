from multiprocessing.sharedctypes import Value
import numpy
import operator
import abc
import functools

from gridding.base_grid import BaseGrid
from gridding.errors import AlignmentError

class _BoundedGridMeta(type):
    """metaclass of the Raster class"""

    def __new__(cls, name, bases, namespace):

        # operators with a nan-base
        for op, as_idx in (
            (operator.mul, False),
            (operator.truediv, False),
            (operator.floordiv, False),
            (operator.pow, False),
            (operator.mod, False),
            (operator.eq, True),
            (operator.ne, True),
            (operator.ge, True),
            (operator.le, True),
            (operator.gt, True),
            (operator.lt, True),
        ):
            opname = "__{}__".format(op.__name__)
            opname_reversed = "__r{}__".format(op.__name__)
            normal_op, reverse_op = cls._gen_operator(op, base_value=numpy.nan, as_idx=as_idx)
            namespace[opname] = normal_op
            namespace[opname_reversed] = reverse_op
        
        # operators with a zero-base
        for op in (
            operator.add,
            operator.sub,
        ):
            opname = "__{}__".format(op.__name__)
            opname_reversed = "__r{}__".format(op.__name__)
            normal_op, reverse_op = cls._gen_operator(op, base_value=0)
            namespace[opname] = normal_op
            namespace[opname_reversed] = reverse_op

        # reduction operators
        for op, name in (
            (numpy.nansum, "sum"),
            (numpy.nanmean, "mean"),
            (numpy.nanmax, "max"),
            (numpy.nanmin, "min"),
            (numpy.nanmedian, "median"),
            (numpy.nanstd, "std"),
        ):
            namespace[name] = cls._gen_reduce_operator(op)

        # reduction operators (arg)
        for op, name in (
            (numpy.nanargmax, "argmax"),
            (numpy.nanargmin, "argmin"),
        ):
            namespace[name] = cls._gen_reduce_operator(op, as_idx=True)
        return super().__new__(cls, name, bases, namespace)

    @staticmethod
    def _gen_operator(op, base_value=numpy.nan, as_idx=False):
        """Generate operators

        Parameters
        ----------
        op: :class:`operator`
            A method from the `operator` package

        Returns
        -------
        :tuple: (`leif.core.raster._RasterMeta.<locals>.normal_op`, `leif.core.raster._RasterMeta.<locals>.reverse_op`)
            A function that takes in a left and right object to apply the supplied operation (`op`) to.\

        Raises
        ------
        NotImplementedError
            For grids of different sizes
        """

        def _grid_op(left, right, op, base_value):

            if not isinstance(right, BoundedGrid):
                data = op(left._data, right)
                return left.__class__(data, bounds=left.bounds)

            if not left.intersects(right):
                raise AlignmentError("Operation not allowed on grids that do not overlap.") # TODO rethink errors. Do we want an out of bounds error?
            
            aligned, reason = left.is_aligned_with(right)
            if not aligned:
                raise AlignmentError(f"Grids are not aligned. {reason}")

            combined_bounds = left.combined_bounds(right)
            combined_data = numpy.full((int((combined_bounds[3]-combined_bounds[1])/left.dy), int((combined_bounds[2]-combined_bounds[0])/left.dx)), numpy.nan) # data shape is y,x
            combined_grid = left.__class__(combined_data, bounds=combined_bounds)

            if base_value != numpy.nan: # nan is already the default value
                combined_grid = combined_grid.assign(base_value, bounds=left.bounds).assign(base_value, bounds=right.bounds)
            combined_left = combined_grid.assign(left.data, bounds=left.bounds, in_place=False)
            combined_right = combined_grid.assign(right.data, bounds=right.bounds, in_place=False)
            combined_data = op(combined_left._data, combined_right._data)

            return left.__class__(combined_data, bounds=combined_bounds)

        def normal_op(left, right):
            if not isinstance(right, BoundedGrid):
                data = op(left._data, right)
                grid = left.__class__(data, bounds=left.bounds) #TODO: use update method when implemented
            else:
                grid = _grid_op(left, right, op, base_value=base_value)

            return grid._mask_to_index(grid._data) if as_idx else grid #TODO: left._mask_to_index(data) works if as_idx is true

        def reverse_op(left, right):
            if not isinstance(right, BoundedGrid):
                data = op(right, left._data)
                grid = left.__class__(data, bounds=left.bounds) #TODO: use update method when implemented
            else:
                grid = _grid_op(left, right, op, base_value=base_value)
            return grid._mask_to_index(grid._data) if as_idx else grid #TODO: left._mask_to_index(data) works if as_idx is true

        return normal_op, reverse_op

    @staticmethod
    def _gen_reduce_operator(op, as_idx=False):
        def internal(self, *args, **kwargs):
            result = op(self._data, *args, **kwargs)
            if not as_idx:
                return result
            # since `as_idx`=True, assume result is the id corresponding to the raveled array
            # TODO: put lines below in function self.numpy_id_to_grid_id or similar. Think of raveled vs xy input
            np_id_x = int(result % self.width)
            np_id_y = int(numpy.floor(result / self.width))
            left_top = self.corners[0]
            left_top_id = self.cell_at_point(left_top + [self.dx / 2, - self.dy / 2])
            index = left_top_id + [np_id_x, -np_id_y]
            return index

        return internal

class BoundedGridMeta(abc.ABCMeta, _BoundedGridMeta):
    """Class that enables usage of the `_BoundedGridMeta` metaclass despite using ABCMeta as metaclass for the parent class."""
    pass

class BoundedGrid(metaclass=BoundedGridMeta):

    def __init__(self, data: numpy.ndarray, *args, bounds: tuple, **kwargs) -> None:
        self._data = data.copy()
        self._bounds = bounds
        super(BoundedGrid, self).__init__(*args, **kwargs)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        new_data = numpy.array(data)
        if new_data.dtype.name == "object":
            raise TypeError(f"Data cannot be interpreted as a numpy.ndarray, got {type(data)}")
        if new_data.shape != self.data.shape:
            raise ValueError(f"Cannot set data that is different in size. Expected a shape of {self.data.shape}, got {new_data.shape}.")
        self._data = data

    def __array__(self, dtype=None):
        return self.data

    def update(self, new_data, bounds=None):
        #TODO figure out how to update dx, dy, origin
        if not bounds:
            bounds = self.bounds
        return self.__class__(new_data, bounds=bounds)

    def copy(self):
        return self.update(self.data)

    @property
    def bounds(self) -> tuple:
        """Raster Bounds

        Returns
        -------
        :class:`tuple`
            The bounds of the data in (left, bottom, right, top) or equivalently (min-x, min-y, max-x, max-y)
        """
        return self._bounds

    @property
    def corners(self):
        b = self.bounds
        return numpy.array([
            [b[0], b[3]], # left-top
            [b[2], b[3]], # right-top
            [b[2], b[1]], # right-bottom
            [b[0], b[1]], # left-bottom
        ])

    @property
    def width(self):
        """Raster width

        Returns
        -------
        :class:`int`
            The number of grid cells in x-direction
        """
        return self._data.shape[-1]

    @property
    def height(self):
        """
        Returns
        -------
        :class:`int`
            The number of grid cells in y-direction
        """
        return self._data.shape[-2]

    @property
    def cellsize(self):
        """Get the gridsize in (dx, dy)"""
        return (self.dx, self.dy)

    @property
    def nr_cells(self):
        """Number of cells

        Returns
        -------
        :class:`int`
            The total number of cells in the grid
        """
        return self.height * self.widthF

    def intersects(self, other):
        other_bounds = other.bounds if isinstance(other, BaseGrid) else other # Allow for both grid objects and bounds
        return not (
            self.bounds[0] >= other_bounds[2]
            or self.bounds[2] <= other_bounds[0]
            or self.bounds[1] >= other_bounds[3]
            or self.bounds[3] <= other_bounds[1]
        )

    def _mask_to_index(self, mask):

        if not self._data.shape == mask.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match data shape {self._data.shape}")

        ids, shape = self.cells_in_bounds(self.bounds)
        ids = ids.reshape([*shape, 2])
        return ids[mask]


    @abc.abstractmethod
    def shared_bounds(self, other):
        pass
        
    @abc.abstractmethod
    def combined_bounds(self, other):
        pass

    @abc.abstractmethod
    def crop(self, new_bounds, bounds_crs=None):
        pass

    @abc.abstractmethod
    def intersecting_cells(self, other):
        pass

    @abc.abstractmethod
    def numpy_id_to_grid_id(self, index):
        pass

    @abc.abstractmethod
    def grid_id_to_numpy_id(self, index):
        pass

    def assign(self, data, *, anchor=None, bounds=None, in_place=True):
        if not any([anchor, bounds]):
            raise ValueError("Please supply either an 'anchor' or 'bounds' keyword to position the data in the grid.")
        new_data = self.data if in_place else self.data.copy()

        if bounds:
            slice_y, slice_x = self._data_slice_from_bounds(bounds)
            new_data[slice_y, slice_x] = data

        if anchor: # a corner or center
            raise NotImplementedError()

        return self.__class__(new_data, bounds=self.bounds, crs=self.crs)

    def value(self, index, oob_value = numpy.nan):
        """Return the value at the given cell index"""

        # Convert grid-ids into numpy-ids
        # Note: the (0,0) id for numpy refers to the top-left
        index = numpy.array(index)
        index = index[numpy.newaxis, :] if len(index.shape) == 1  else index
        left_top = self.corners[0]
        left_top_id = self.cell_at_point(left_top + [self.dx / 2, - self.dy / 2])[:,numpy.newaxis]
        index = index.T # TODO: maybe always work with xy axis first
        np_id = numpy.empty_like(index)
        np_id[0] = index[0] - left_top_id[0]
        np_id[1] = left_top_id[1] - index[1]

        # Identify any id outside the bounds
        oob_mask = numpy.where(np_id[0] >= self._data.shape[1])
        oob_mask += numpy.where(np_id[0] < 0)
        oob_mask += numpy.where(np_id[1] >= self._data.shape[0])
        oob_mask += numpy.where(np_id[1] < 0)
        oob_mask = numpy.hstack(oob_mask)

        # Return array's `dtype` needs to be float instead of integer if an id falls outside of bounds
        # For NaNs don't make sense as integer
        if numpy.any(oob_mask) and not numpy.isfinite(oob_value) and not numpy.issubdtype(self._data.dtype, numpy.float):
            print(f"Warning: dtype `{self._data.dtype}` might not support an `oob_value` of `{oob_value}`.")

        values = numpy.full(np_id.shape[1], oob_value, dtype=self._data.dtype)

        sample_mask = numpy.ones_like(values, dtype='bool')
        sample_mask[oob_mask] = False

        np_id = np_id[:, sample_mask]
        values[sample_mask] = self._data[np_id[1], np_id[0]]

        return values
        
