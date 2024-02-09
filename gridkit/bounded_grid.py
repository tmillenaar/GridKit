import abc
import functools
import operator
import warnings
from multiprocessing.sharedctypes import Value
from typing import Tuple

import numpy
import scipy
from pyproj import Transformer

import gridkit
from gridkit.base_grid import BaseGrid
from gridkit.errors import AlignmentError, IntersectionError
from gridkit.index import GridIndex, validate_index


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
            (operator.ge, True),
            (operator.le, True),
            (operator.gt, True),
            (operator.lt, True),
        ):
            opname = "__{}__".format(op.__name__)
            opname_reversed = "__r{}__".format(op.__name__)
            normal_op, reverse_op = cls._gen_operator(
                op, base_value=numpy.nan, as_idx=as_idx
            )
            namespace[opname] = normal_op
            namespace[opname_reversed] = reverse_op

        # treat equals and not-equals as special cases to acommodate NaNs
        op = functools.partial(numpy.isclose, equal_nan=True)
        normal_op, reverse_op = cls._gen_operator(op, base_value=numpy.nan, as_idx=True)
        namespace["__eq__"] = normal_op
        namespace["__req__"] = reverse_op

        op = lambda left, right: ~numpy.isclose(left, right, equal_nan=True)
        normal_op, reverse_op = cls._gen_operator(op, base_value=numpy.nan, as_idx=True)
        namespace["__ne__"] = normal_op
        namespace["__rne__"] = reverse_op

        # operators with a zero-base
        for op, name in (
            (operator.add, "add"),
            (operator.sub, "sub"),
        ):
            opname = "__{}__".format(name)
            opname_reversed = "__r{}__".format(name)
            normal_op, reverse_op = cls._gen_operator(op, base_value=0)
            namespace[opname] = normal_op
            namespace[opname_reversed] = reverse_op

        # reduction operators
        for op, name in (
            (numpy.ma.sum, "sum"),
            (numpy.ma.mean, "mean"),
            (numpy.ma.max, "max"),
            (numpy.ma.min, "min"),
            (numpy.ma.median, "median"),
            (numpy.ma.std, "std"),
        ):
            namespace[name] = cls._gen_reduce_operator(op)

        # reduction operators (arg)
        for op, name in (
            (numpy.ma.argmax, "argmax"),
            (numpy.ma.argmin, "argmin"),
        ):
            namespace[name] = cls._gen_reduce_operator(op, as_idx=True)
        return super().__new__(cls, name, bases, namespace)

    @staticmethod
    def _gen_operator(op, base_value=numpy.nan, as_idx=False):
        """Generate operators

        Parameters
        ----------
        op: any callable operator form either `numpy` or `operator`
            A function from the `operator` package. Can also be a numpy or cusotm uperator.

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
                if left.nodata_value is not None:
                    mask = left._data == left.nodata_value
                    data[mask] = left.nodata_value
                return left.__class__(data, bounds=left.bounds, crs=left.crs)

            if not left.intersects(right):
                raise AlignmentError(
                    "Operation not allowed on grids that do not overlap."
                )  # TODO rethink errors. Do we want an out of bounds error?

            aligned, reason = left.is_aligned_with(right)
            if not aligned:
                raise AlignmentError(f"Grids are not aligned. {reason}")

            # determine the dtype and nodata_value for the new grid, based on numpy casting rules
            def dominant_dtype_and_nodata_value(*grids):
                dtypes = [grid._data.dtype for grid in grids]
                dtype = numpy.result_type(*dtypes)

                nodata_values = [grid.nodata_value for grid in grids]
                if all([val == nodata_values[0] for val in nodata_values]):
                    nodata_value = nodata_values[0]
                elif all([val is not None for val in nodata_values]):
                    nodata_value = nodata_values[0]
                    warnings.warn(
                        f"Two grids have different `nodata_value`s: {nodata_values}. Using {nodata_value} for the resulting grid."
                    )
                elif all([val is None for val in nodata_values]):
                    nodata_value = None
                    warnings.warn(
                        f"No `nodata_value` was set on any of the grids. Any potential nodata gaps are filled with `{dtype.type(0)}`. Set a nodata_value by assigning a value to grid.nodata_value."
                    )
                else:  # one but not all grids have a nodata_value set
                    for val in nodata_values:
                        if val is not None:
                            nodata_value = val
                            break
                    else:
                        raise ValueError(
                            "Oops, something unexpected went wrong when determining the new nodata_value. To resolve this you could try setting the `nodata_value` attribute of your grids."
                        )
                return dtype, nodata_value

            dtype, nodata_value = dominant_dtype_and_nodata_value(left, right)

            # create combined grid, spanning both inputs
            combined_bounds = left.combined_bounds(right)
            _, shape = left.cells_in_bounds(
                combined_bounds, return_cell_count=True
            )  # TODO: split ids and shape outputs into different methods
            combined_data = numpy.full(
                shape,
                fill_value=dtype.type(0) if nodata_value is None else nodata_value,
                dtype=dtype,
            )

            combined_grid = left.update(
                combined_data,
                bounds=combined_bounds,
                nodata_value=nodata_value,
            )

            shared_bounds = left.shared_bounds(right)
            left_shared = left.crop(shared_bounds)
            right_shared = right.crop(shared_bounds)

            left_data = numpy.ma.masked_array(
                left_shared._data,
                left_shared.data == left_shared.nodata_value,
                dtype=dtype,
            )
            right_data = numpy.ma.masked_array(
                right_shared._data,
                right_shared.data == right_shared.nodata_value,
                dtype=dtype,
            )

            result = op(left_data, right_data)
            # assign data of `left` to combined_grid
            left_data = left._data.astype(dtype)
            if left.nodata_value is not None:
                left_data[left._data == left.nodata_value] = nodata_value
            combined_grid.assign(
                left_data, bounds=left.bounds, in_place=True, assign_nodata=False
            )

            # assign data of `right` to combined_grid
            right_data = right._data.astype(dtype)
            if right.nodata_value is not None:
                right_data[right._data == right.nodata_value] = nodata_value
            combined_grid.assign(
                right_data, bounds=right.bounds, in_place=True, assign_nodata=False
            )

            # overwrite shared area in combined_grid with the combined results
            count = gridkit.count([left, right])
            shared_mask = count == 2
            shared_mask_np = combined_grid.grid_id_to_numpy_id(shared_mask)
            result = op(left.value(shared_mask), right.value(shared_mask))
            combined_grid = combined_grid.astype(
                numpy.result_type(combined_grid._data.dtype, result.dtype)
            )  # when dividing the dtype changes
            combined_grid._data[shared_mask_np] = (
                result  # TODO: find more elegant way of updating data with grid ids as mask
            )

            return combined_grid

        def normal_op(left, right):
            if not isinstance(right, BoundedGrid):
                data = op(left._data, right)
                if left.nodata_value is not None:
                    nodata_np_id = numpy.where(left._data == left.nodata_value)
                    data[nodata_np_id] = left.nodata_value
                grid = left.update(data)
            else:
                grid = _grid_op(left, right, op, base_value=base_value)
            if not as_idx:
                return grid
            ids = left._mask_to_index(grid._data)
            return GridIndex(ids)

        def reverse_op(left, right):
            if not isinstance(right, BoundedGrid):
                data = op(right, left._data)
                if left.nodata_value is not None:
                    nodata_np_id = numpy.where(left._data == left.nodata_value)
                    data[nodata_np_id] = left.nodata_value
                grid = left.update(data)
            else:
                grid = _grid_op(left, right, op, base_value=base_value)
            return (
                grid._mask_to_index(grid._data) if as_idx else grid
            )  # TODO: left._mask_to_index(data) works if as_idx is true

        return normal_op, reverse_op

    @staticmethod
    def _gen_reduce_operator(op, as_idx=False):
        def internal(self, *args, **kwargs):
            data = self._data
            if not self.nodata_value is None:
                data = numpy.ma.masked_array(
                    data, numpy.isclose(data, self.nodata_value, equal_nan=True)
                )
            result = op(data, *args, **kwargs)

            if not as_idx:
                return result

            # since `as_idx`=True, assume result is the id corresponding to the raveled array
            # TODO: put lines below in function self.numpy_id_to_grid_id or similar. Think of raveled vs xy input
            np_id_x = int(result % self.width)
            np_id_y = int(numpy.floor(result / self.width))
            left_top = self.corners[0]
            left_top_id = self.cell_at_point(left_top + [self.dx / 2, -self.dy / 2])
            index = left_top_id + [np_id_x, -np_id_y]
            return index

        return internal


class _AbstractBoundedGridMeta(abc.ABCMeta, _BoundedGridMeta):
    """Class that enables usage of the :class:`~gridkit.bounded_grid._BoundedGridMeta` metaclass despite using ABCMeta as metaclass for the parent class."""

    pass


class BoundedGrid(metaclass=_AbstractBoundedGridMeta):
    def __init__(
        self,
        data: numpy.ndarray,
        *args,
        bounds: tuple,
        nodata_value=None,
        prevent_copy: bool = False,
        **kwargs,
    ) -> None:
        self._data = data if prevent_copy else data.copy()
        self._bounds = bounds
        self.nodata_value = nodata_value
        super(BoundedGrid, self).__init__(*args, **kwargs)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        new_data = numpy.array(data)
        if new_data.dtype.name == "object":
            raise TypeError(
                f"Data cannot be interpreted as a numpy.ndarray, got {type(data)}"
            )
        if new_data.shape != self.data.shape:
            raise ValueError(
                f"Cannot set data that is different in size. Expected a shape of {self.data.shape}, got {new_data.shape}."
            )
        self._data = data

    def __array__(self, dtype=None):
        if dtype:
            return self.data.astype(dtype)
        return self.data

    @property
    def dtype(self):
        return self._data.dtype

    def astype(self, dtype):
        return self.update(new_data=self._data.astype(dtype))

    def update(self, new_data, bounds=None, crs=None, nodata_value=None):
        # TODO figure out how to update dx, dy, origin
        if not bounds:
            bounds = self.bounds
        if not crs:
            crs = self.crs
        if not nodata_value:
            nodata_value = self.nodata_value
        return self.__class__(
            new_data, bounds=bounds, crs=crs, nodata_value=nodata_value
        )

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
    def mpl_extent(self) -> tuple:
        """Raster Bounds

        Returns
        -------
        :class:`tuple`
            The extent of the data as defined expected by matplotlib in (left, right, bottom, top) or equivalently (min-x, max-x, min-y, max-y)
        """
        b = self._bounds
        return (b[0], b[2], b[1], b[3])

    @property
    def corners(self):
        b = self.bounds
        return numpy.array(
            [
                [b[0], b[3]],  # left-top
                [b[2], b[3]],  # right-top
                [b[2], b[1]],  # right-bottom
                [b[0], b[1]],  # left-bottom
            ]
        )

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
        """Raster height

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
        return self.height * self.width

    def intersects(self, other):
        other_bounds = (
            other.bounds if isinstance(other, BaseGrid) else other
        )  # Allow for both grid objects and bounds
        return not (
            self.bounds[0] >= other_bounds[2]
            or self.bounds[2] <= other_bounds[0]
            or self.bounds[1] >= other_bounds[3]
            or self.bounds[3] <= other_bounds[1]
        )

    def _mask_to_index(self, mask):
        if not self._data.shape == mask.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match data shape {self._data.shape}"
            )

        ids = self.cells_in_bounds(self.bounds)
        return GridIndex(ids[mask])

    def shared_bounds(self, other):
        other_bounds = other.bounds if isinstance(other, BaseGrid) else other
        if not self.intersects(other):
            raise IntersectionError(
                f"Grid with bounds {self.bounds} does not intersect with grid with bounds {other_bounds}."
            )
        return (
            max(self.bounds[0], other_bounds[0]),
            max(self.bounds[1], other_bounds[1]),
            min(self.bounds[2], other_bounds[2]),
            min(self.bounds[3], other_bounds[3]),
        )

    def combined_bounds(self, other):
        other_bounds = other.bounds if isinstance(other, BaseGrid) else other
        return (
            min(self.bounds[0], other_bounds[0]),
            min(self.bounds[1], other_bounds[1]),
            max(self.bounds[2], other_bounds[2]),
            max(self.bounds[3], other_bounds[3]),
        )

    @abc.abstractmethod
    def crop(self, new_bounds, bounds_crs=None):
        """Cut out a slice of data contained within the supplied bounds.

        Parameters
        ----------
        new_bounds: `Tuple(minx, miny, maxx, maxy)`
            The bounds defining the area to crop, in (minx, miny, maxx, maxy).

        bounds_crs: `pyproj.CRS` (optional)
            The bounds defining the extent of the cropped data.
            The value can be anything accepted by `pyproj.CRS.from_user_input()`.

        Returns
        -------
        :class: `BoundedGrid`
            A BoundedGrid containing the data included in the cropped area contained within the bounds.
        """
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

    @property
    def indices(self):
        """Return the indices within the bounds of the data"""
        return self.cells_in_bounds(self.bounds)

    def assign(
        self, data, *, anchor=None, bounds=None, in_place=True, assign_nodata=True
    ):
        if not any([anchor, bounds]):
            raise ValueError(
                "Please supply either an 'anchor' or 'bounds' keyword to position the data in the grid."
            )
        new_data = self.data if in_place else self.data.copy()

        if bounds:
            slice_y, slice_x = self._data_slice_from_bounds(bounds)
            crop = new_data[slice_y, slice_x]
            if not assign_nodata:
                mask = data != self.nodata_value
                crop[mask] = data[mask]
            else:
                crop[:] = data
        elif anchor:  # a corner or center
            raise NotImplementedError()
        else:
            raise ValueError("Please supply one of: {anchor, bounds}")
        return self.update(new_data)

    @validate_index
    def value(self, index=None, oob_value=None):
        """Return the value at the given cell index.

        Parameters
        ----------
        index: :class:`.GridIndex` (optional)
            The index of the cell(s) of which to return the value.
            If not supplied, all values will be returned in a flattened array.
        oob_value: :class:`float` (optional)
            The value assigned to values that are 'out of bounds'.
            I.e. the value assigned to ``index`` entries not covered by the data.
            Default: numpy.nan

        Returns
        -------
        :class:`numpy.ndarray`
            The values at the supplied `index` locations
        """

        if index is None:
            index = self.indices

        # Convert grid-ids into numpy-ids
        np_id = numpy.stack(self.grid_id_to_numpy_id(index.ravel())[::-1])
        if np_id.ndim == 1:
            np_id = np_id[:, numpy.newaxis]

        # Identify any id outside the bounds
        oob_mask = numpy.where(np_id[0] >= self._data.shape[1])
        oob_mask += numpy.where(np_id[0] < 0)
        oob_mask += numpy.where(np_id[1] >= self._data.shape[0])
        oob_mask += numpy.where(np_id[1] < 0)
        oob_mask = numpy.hstack(oob_mask)

        if numpy.any(
            oob_mask
        ):  # make sure we know what nodata value to set if ids are out of bounds
            if oob_value is None and self.nodata_value is None:
                raise ValueError(
                    "Some indices do not have data. Please remove these ids, set a 'nodata_value' or supply an 'oob_value'."
                )
            oob_value = oob_value if oob_value else self.nodata_value
        else:
            oob_value = (
                oob_value if oob_value else 0
            )  # the oob_value does not matter if no ids are out of bounds

        # Return array's `dtype` needs to be float instead of integer if an id falls outside of bounds
        # For NaNs don't make sense as integer
        if (
            numpy.any(oob_mask)
            and not numpy.isfinite(oob_value)
            and not numpy.issubdtype(self._data.dtype, numpy.floating)
        ):
            print(
                f"Warning: dtype `{self._data.dtype}` might not support an `oob_value` of `{oob_value}`."
            )

        values = numpy.full(np_id.shape[1], oob_value, dtype=self._data.dtype)

        sample_mask = numpy.ones_like(values, dtype="bool")
        sample_mask[oob_mask] = False

        np_id = np_id[:, sample_mask]
        values[sample_mask] = self._data[np_id[1], np_id[0]]

        return values.reshape(index.shape)

    def nodata(self):
        if self.nodata_value is None:
            return None
        return self == self.nodata_value

    def percentile(self, value):
        return numpy.percentile(self, value)

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

    def interp_nodata(self, method="linear", in_place=False):
        """Interpolate the cells containing nodata, if they are inside the convex hull of cells that do contain data.

        Parameters
        ----------
        method: :class:`str`
            The interpolation method to be used. Options are ("nearest", "linear", "cubic"). Default: "linear".
        in_place: :class:`bool`
            Boolean flag determining whether to fill the nodata values in-place.
            In-memory values are modified if in_place is Ture.
            A copy of the grid is made where nodata values are filled if in_place is False.

        Returns
        -------
        :class:`tuple`
            A copy of the grid with interpolated values.
        """
        method_lut = dict(
            nearest=scipy.interpolate.NearestNDInterpolator,
            linear=functools.partial(
                scipy.interpolate.LinearNDInterpolator, fill_value=self.nodata_value
            ),
            cubic=scipy.interpolate.CloughTocher2DInterpolator,
        )

        if self.nodata_value is None:
            raise ValueError(
                f"Cannot interpolate nodata values if attribute 'nodata_value' is not set."
            )

        if method not in method_lut:
            raise ValueError(
                f"Method '{method}' is not supported. Supported methods: {method_lut.keys()}"
            )

        if not in_place:
            self = self.copy()

        interp_func = method_lut[method]
        values = self.data.ravel()
        nodata_mask = values == self.nodata_value
        points = self.centroid().reshape(-1, 2)
        interpolator = interp_func(
            points[~nodata_mask],
            values[~nodata_mask],
        )
        filled_values = interpolator(points[nodata_mask])
        self.data.ravel()[nodata_mask] = filled_values
        return self

    def interpolate(self, sample_points, method="nearest", **interp_kwargs):
        """Interpolate the value at the location of ``sample_points``.

        Points that are outside of the bounds of the data are assigned `self.nodata_value`, or 'NaN' if no nodata value is set.

        Parameters
        ----------
        sample_points: :class:`numpy.ndarray`
            The coordinates of the points at which to sample the data
        method: :class:`str`, `'nearest', 'bilinear'`, optional
            The interpolation method used to determine the value at the supplied `sample_points`.
            Supported methods:
            - "nearest", for nearest neigbour interpolation, effectively sampling the value of the data cell containing the point
            - "bilinear", linear interpolation using the four cells surrounding the point
            - "inverse_distance", weighted inverse distance using the 4,3,6 nearby cells surrounding the point for Rect, Hex and Rect grid respectively.
            Default: "nearest"
        **interp_kwargs: `dict`
            The keyword argument passed to the interpolation function corresponding to the specified `method`

        Returns
        -------
        :class:`numpy.ndarray`
            The interpolated values at the supplied points

        See also
        --------
        :py:meth:`.BoundedGrid.resample`
        :py:meth:`.BaseGrid.interp_from_points`
        """
        if method == "nearest":
            new_ids = self.cell_at_point(sample_points)
            return self.value(new_ids)
        elif method == "bilinear" or method == "linear":
            return_shape = sample_points.shape[:-1]
            result = self._bilinear_interpolation(sample_points.reshape(-1, 2))
            return result.reshape(return_shape)
        elif method == "inverse_distance":
            return self._inverse_distance_interpolation(sample_points, **interp_kwargs)
        raise ValueError(f"Resampling method '{method}' is not supported.")

    def _inverse_distance_interpolation(
        self, sample_points, max_nr_nans=0, decay_constant=1
    ):
        if not isinstance(sample_points, numpy.ndarray):
            sample_points = numpy.array(sample_points)
        original_shape = sample_points.shape
        if not original_shape[-1] == 2:
            raise ValueError(
                f"Expected the last axis of sample_points to have two elements (x,y). Got {original_shape[-1]} elements"
            )
        sample_points = sample_points.reshape(-1, 2)

        nearby_cells = self.cells_near_point(sample_points)

        # FIXME: needs to be conform with tri-grid
        # nearby_cells = nearby_cells.index.swapaxes(0,1) # for hexagons and rectangles

        nearby_values = self.value(nearby_cells)
        nearby_centroids = self.centroid(
            nearby_cells
        )  # shape is (points[N], cells[6], xy[2])
        # swap axes to match dimensions to sample points
        nearby_centroids = numpy.swapaxes(
            nearby_centroids, 0, 1
        )  # shape is (cells[6], points[N], xy[2])
        point_corner_vec = (
            nearby_centroids - sample_points
        )  # shape is (cells[6], points[N], xy[2])
        distances = numpy.linalg.norm(
            point_corner_vec, axis=-1
        )  # compute distance over xy vector

        # TODO: allow for different weighting equations, sch as Shepard's interpolation with different power parameters
        # weights = ((2*self.r) / distances)

        weights = numpy.exp(-((distances / decay_constant) ** 2))
        weights = weights / numpy.sum(weights, axis=0)

        # breakpoint()

        # swap axes back to match dimensions to nearby_values
        weights = numpy.swapaxes(weights, 0, 1)  # shape is (points[N], cells[6])

        # TODO: determine mask based on number of nans.
        #       The edge might have 4 nans and 2 values where the point is outside of the bounds.
        #       This might lead to edge effects.
        result = numpy.nansum(weights * nearby_values, axis=1)
        nr_nans = (~numpy.isfinite(nearby_values)).sum(axis=1)
        nan_mask = nr_nans > max_nr_nans
        result[nan_mask] = numpy.nan
        return result.reshape(original_shape[:-1])

    def resample(self, alignment_grid, method="nearest", **interp_kwargs):
        """Resample the grid onto another grid.
        This will take the locations of the grid cells of the other grid (here called ``alignment_grid``)
        and determine the value on these location based on the values of the original grid (``self``).

        The steps are as follows:
         1. Transform the bounds of the original data to the CRS of the alignment grid (if not already the same)
            No transformation is done if any of the grids has no CRS set.
         2. Find the cells of the alignment grid within these transformed bounds
         3. Find the cells of the original grid that are nearby each of the centroids of the cells found in 2.
            How many nearby cells are selected depends on the selected ``method``
         4. Interpolate the values using the supplied ``method`` at each of the centroids of the alignment grid cells selected in 2.
         5. Create a new bounded grid using the attributes of the alignment grid

        Parameters
        ----------
        alignment_grid: :class:`.BaseGrid`
            The grid with the desired attributes on which to resample.

        method: :class:`str`, `'nearest', 'bilinear', 'inverse_distance'`, optional
            The interpolation method used to determine the value at the supplied `sample_points`.
            Supported methods:
            - "nearest", for nearest neigbour interpolation, effectively sampling the value of the data cell containing the point
            - "bilinear", linear interpolation using the 4,3,6 nearby cells surrounding the point for Rect, Hex and Rect grid respectively.
            - "inverse_distance", weighted inverse distance using the 4,3,6 nearby cells surrounding the point for Rect, Hex and Rect grid respectively.
            Default: "nearest"
        **interp_kwargs: `dict`
            The keyword argument passed to the interpolation function corresponding to the specified `method`

        Returns
        -------
        :class:`.BoundedGrid`
            The interpolated values at the supplied points

        See also
        --------
        :py:meth:`.BoundedGrid.interpolate`
        :py:meth:`.BaseGrid.interp_from_points`
        """
        if self.crs is None or alignment_grid.crs is None:
            warnings.warn(
                "`crs` not set for one or both grids. Assuming both grids have an identical CRS."
            )
            different_crs = False
        else:
            different_crs = not self.crs.is_exact_same(alignment_grid.crs)

        # make sure the bounds align with the grid
        if different_crs:
            transformer = Transformer.from_crs(
                self.crs, alignment_grid.crs, always_xy=True
            )
            bounds = transformer.transform_bounds(*self.bounds)
        else:
            bounds = self.bounds

        # Align using "contract" for we cannot sample outside of the original bounds
        new_bounds = alignment_grid.align_bounds(bounds, mode="contract")

        new_ids, shape = alignment_grid.cells_in_bounds(
            bounds=new_bounds, return_cell_count=True
        )

        new_points = alignment_grid.centroid(new_ids)

        if different_crs:
            transformer = Transformer.from_crs(
                alignment_grid.crs, self.crs, always_xy=True
            )
            original_shape = new_points.shape
            raveled_new_points = new_points.reshape(-1, 2)
            transformed_points = transformer.transform(*raveled_new_points.T)
            new_points = numpy.vstack(transformed_points).T.reshape(original_shape)

        value = self.interpolate(new_points, method=method, **interp_kwargs)

        # If value id 1D, turn into 2D
        # Take into account if the 1D line of cells runs in x or y direction
        if 1 in shape:
            empty_axis = 0 if shape[0] == 1 else 1
            value = numpy.expand_dims(value, axis=empty_axis)

        nodata_value = self.nodata_value if self.nodata_value is not None else numpy.nan

        grid_kwargs = dict(
            data=value,
            bounds=new_bounds,
            crs=alignment_grid.crs,
            nodata_value=nodata_value,
        )
        if hasattr(alignment_grid, "_shape"):
            grid_kwargs["shape"] = alignment_grid._shape

        new_grid = alignment_grid.bounded_cls(**grid_kwargs)

        return new_grid
