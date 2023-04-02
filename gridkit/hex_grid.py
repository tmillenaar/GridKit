from gridkit.base_grid import BaseGrid
from gridkit.bounded_grid import BoundedGrid
from gridkit.rect_grid import RectGrid
from gridkit.errors import IntersectionError, AlignmentError

import scipy
import numpy
import warnings
from pyproj import CRS, Transformer

class HexGrid(BaseGrid):

    def __init__(self, *args, size, shape="pointy", **kwargs):
        
        self._radius = size / 3**0.5
        
        if shape == "pointy":
            self._dx = size
            self._dy = 3/2 * self._radius
        elif shape == "flat":
            self._dy = size
            self._dx = 3/2 * self._radius
        else:
            raise ValueError(f"A HexGrid's `shape` can either be 'pointy' or 'flat', got '{shape}'")
        
        self._shape = shape
        self.bounded_cls = BoundedRectGrid #TODO create a BoundedRectGrid
        super(HexGrid, self).__init__(*args, **kwargs)

    @property
    def dx(self) -> float:
        """The spacing between cell centers in x-direction
        """
        return self._dx

    @property
    def dy(self) -> float:
        """The spacing between cell centers in y-direction
        """
        return self._dy
    
    @property
    def r(self) -> float:
        """The radius of the cell. The radius is defined to be the distance from the cell center to an outer corner.
        """
        return self._radius

    def neighbours(self, index, connect_corners=False, include_selected=False, flat=False):
        """
        """
        if not isinstance(index, numpy.ndarray):
            index = numpy.array(index)

        relative_neighbours = numpy.vstack([
            numpy.array([[-1,0,1]] * 3).ravel(),
            numpy.array([[1,0,-1]] * 3).T.ravel()
        ]).T

        if self._shape == "pointy":
            even_index = index[:,1] % 2 == 0
            relative_neighbour_ids_even = {0,1,3,5,6,7}
            relative_neighbour_ids_odd = {1,2,3,5,7,8}
        elif self._shape == "flat":
            even_index = index[:,0] % 2 == 0
            relative_neighbour_ids_even = {1,3,5,6,7,8}
            relative_neighbour_ids_odd = {0,1,2,3,5,7}

        if include_selected:
            relative_neighbour_ids_even.add(4)
            relative_neighbour_ids_odd.add(4)

        # breakpoint()

        neighbours = numpy.expand_dims(index, 1)
        neighbours = numpy.repeat(neighbours, 6, axis=1)
        neighbours[even_index] += relative_neighbours[list(relative_neighbour_ids_even)]
        neighbours[~even_index] += relative_neighbours[list(relative_neighbour_ids_odd)]

        if index.ndim == 1: # make sure 'neighbours' is in the desired shape if index contains only one ID
            neighbours = neighbours.T

        return neighbours if not flat else neighbours.reshape(-1, neighbours.shape[-1])


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
            raise ValueError("For grids that do not contain data, argument `index` is to be supplied to method `centroid`.")
        index = numpy.array(index, dtype="int").T
        centroids = numpy.empty_like(index, dtype=float)
        centroids[0] = index[0] * self.dx + (self.dx / 2) + self.offset[0]
        centroids[1] = index[1] * self.dy + (self.dy / 2) + self.offset[1]

        if self._shape == "pointy":
            offset_rows = index[1] % 2 == 1
            centroids[0, offset_rows] += self.dx/2
        elif self._shape == "flat":
            offset_rows = index[0] % 2 == 1
            centroids[1, offset_rows] += self.dy/2
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
            >>> grid.cells_near_point((0, 0))
            (array([-1,  0]), array([0, 0]), array([-1, -1]), array([ 0, -1]))
            >>> grid.cells_near_point((3, 0.75))
            (array([0, 1]), array([1, 1]), array([0, 0]), array([1, 0]))

        ..

        If multiple points are supplied, a tuple with ndarrays is returned:

        .. code-block:: python

            >>> points = [(0, 0), (3, 0.75), (3, 0)]
            >>> nearby_cells = grid.cells_near_point(points)
            >>> from pprint import pprint # used for output readability purposes only
            >>> pprint(nearby_cells)
            (array([[-1,  0],
                   [ 0,  1],
                   [ 0,  0]]),
             array([[0, 0],
                   [1, 1],
                   [1, 0]]),
             array([[-1, -1],
                   [ 0,  0],
                   [ 0, -1]]),
             array([[ 0, -1],
                   [ 1,  0],
                   [ 1, -1]]))

        ..

        """

        # Split each cell into 4 quadrants in order to identify what 3 neigbors to select
        new_grid = RectGrid(
            dx = self.dx/2,
            dy = self.dy/2,
            offset = self.offset
        )
        ids = new_grid.cell_at_point(point).T
        left_top_mask = numpy.logical_and(ids[0] % 2 == 0, ids[1] % 2 == 1)
        right_top_mask = numpy.logical_and(ids[0] % 2 == 1, ids[1] % 2 == 1)
        right_bottom_mask = numpy.logical_and(ids[0] % 2 == 1, ids[1] % 2 == 0)
        left_bottom_mask = numpy.logical_and(ids[0] % 2 == 0, ids[1] % 2 == 0)

        # obtain the bottom-left cell based on selected quadrant
        base_ids = numpy.empty_like(ids, dtype="int")
        # bottom-left if point in upper-right quadrant
        base_ids[:,right_top_mask] = numpy.floor(ids[:,right_top_mask] / 2)
        # bottom-left if point in bottom-right quadrant
        base_ids[:,right_bottom_mask] = numpy.floor(ids[:,right_bottom_mask] / 2)
        base_ids[1,right_bottom_mask] -= 1
        # bottom-left if point in bottom-left quadrant
        base_ids[:,left_bottom_mask] = numpy.floor(ids[:,left_bottom_mask] / 2)
        base_ids[:,left_bottom_mask] -= 1
        # bottom-left if point in upper-left qudrant
        base_ids[:,left_top_mask] = numpy.floor(ids[:,left_top_mask] / 2)
        base_ids[0,left_top_mask] -= 1

        # use the bottom-left cell to determine the other three
        bl_ids = base_ids.copy()
        br_ids = base_ids.copy()
        br_ids[0] += 1
        tr_ids = base_ids.copy()
        tr_ids += 1
        tl_ids = base_ids.copy()
        tl_ids[1] += 1
            
        return tl_ids.T, tr_ids.T, bl_ids.T, br_ids.T


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
        point = numpy.array(point).T

        # create rectangular grid
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
            raise ValueError(f"A HexGrid's `shape` can either be 'pointy' or 'flat', got '{self._shape}'")

        ids_pointy = numpy.floor((point[pointy_axis] - self.offset[pointy_axis] - self.r / 4) / pointy_stepsize)
        even = ids_pointy % 2 == 0
        ids_flat = numpy.empty_like(ids_pointy)
        ids_flat[~even] = numpy.floor((point[flat_axis][~even] - self.offset[flat_axis] - flat_stepsize/2) / flat_stepsize)
        ids_flat[even] = numpy.floor((point[flat_axis][even] - self.offset[flat_axis]) / flat_stepsize)
        
        # Finetune ambiguous points
        # Points at the top of the cell can be in this cell or in the cell to the top right or top left
        rel_loc_y = ((point[pointy_axis] - self.offset[pointy_axis] - self.r / 4) % pointy_stepsize) + self.r / 4
        rel_loc_x = ((point[flat_axis] - self.offset[flat_axis]) % flat_stepsize)
        top_left_even = rel_loc_x / (flat_stepsize / self.r) < (rel_loc_y - self.r * 5/4)
        top_right_even = (self.r * 1.25 - rel_loc_y) < (rel_loc_x - flat_stepsize) / (flat_stepsize / self.r)
        top_right_odd = (rel_loc_x - flat_stepsize / 2) / (flat_stepsize / self.r) < (rel_loc_y - self.r * 5/4)
        top_right_odd &= rel_loc_x > flat_stepsize / 2
        top_left_odd = (self.r * 1.25 - rel_loc_y) < (rel_loc_x - flat_stepsize / 2) / (flat_stepsize / self.r)
        top_left_odd &= rel_loc_x < flat_stepsize / 2

        ids_pointy[top_left_even & even] += 1
        ids_pointy[top_right_even & even] += 1
        ids_pointy[top_left_odd & ~even] += 1
        ids_pointy[top_right_odd & ~even] += 1

        ids_flat[top_left_even & even] -= 1
        ids_flat[top_left_odd & ~even] += 1

        if self._shape == "pointy":
            return numpy.array([ids_flat, ids_pointy], dtype="int").T

        elif self._shape == "flat":
            return numpy.array([ids_pointy, ids_flat], dtype="int").T


    def cell_corners(self, index: numpy.ndarray = None) -> numpy.ndarray:
        """Return corners in (cells, corners, xy)"""
        if index is None:
            raise ValueError("For grids that do not contain data, argument `index` is to be supplied to method `corners`.")
        centroids = self.centroid(index).T
        
        if len(centroids.shape) == 1:
            corners = numpy.empty((6,2))
        else:
            corners = numpy.empty((6,2,centroids.shape[1]))

        for i in range(6):
            angle_deg = 60 * i - 30 if self._shape == "pointy" else 60 * i
            angle_rad = numpy.pi / 180 * angle_deg
            corners[i, 0] = centroids[0] + self.r * numpy.cos(angle_rad)
            corners[i, 1] = centroids[1] + self.r * numpy.sin(angle_rad)

        # swap from (corners, xy, cells) to (cells, corners, xy)
        if len(centroids.shape) > 1:
            corners = numpy.moveaxis(corners, 2, 0)

        return corners


    def is_aligned_with(self, other):
        if not isinstance(other, RectGrid):
            raise ValueError(f"Expected a RectGrid, got {type(other)}")
        aligned = True
        reason = ""
        reasons = []
        
        if (self.crs is None and other.crs is None):
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

        if not all(numpy.isclose(self.offset, other.offset, atol=1e-7)): # FIXME: atol if 1e-7 is a bandaid. It seems the offset depends slightly depending on the bounds after resampling on grid
            aligned = False
            reasons.append("offset")

        reason = f"The following attributes are not the same: {reasons}" if reasons else reason
        return aligned, reason


    def are_bounds_aligned(self, bounds, separate=False):
        if self._shape == "pointy":
            step_x = self.dx / 2
            step_y = self.dy
        elif self._shape == "flat":
            step_x = self.dx
            step_y = self.dy / 2
        is_aligned = lambda val, cellsize: numpy.isclose(val, 0) or numpy.isclose(val, cellsize)
        per_axis = (
            is_aligned((bounds[0] - self.offset[0]) % step_x, step_x), # left
            is_aligned((bounds[1] - self.offset[1]) % step_y, step_y), # bottom
            is_aligned((bounds[2] - self.offset[0]) % step_x, step_x), # right
            is_aligned((bounds[3] - self.offset[1]) % step_y, step_y)  # top
        )
        return per_axis if separate else numpy.all(per_axis)


    def align_bounds(self, bounds, mode="expand"):
        
        if self.are_bounds_aligned(bounds):
            return bounds
        
        if self._shape == "pointy":
            step_x = self.dx / 2
            step_y = self.dy
        elif self._shape == "flat":
            step_x = self.dx
            step_y = self.dy / 2

        if mode == "expand":
            return (
                numpy.floor((bounds[0] - self.offset[0]) / step_x) * step_x + self.offset[0],
                numpy.floor((bounds[1] - self.offset[1]) / step_y) * step_y + self.offset[1],
                numpy.ceil((bounds[2] - self.offset[0]) / step_x) * step_x + self.offset[0],
                numpy.ceil((bounds[3] - self.offset[1]) / step_y) * step_y + self.offset[1],
            )
        if mode == "contract":
            return (
                numpy.ceil((bounds[0] - self.offset[0]) / step_x) * step_x + self.offset[0],
                numpy.ceil((bounds[1] - self.offset[1]) / step_y) * step_y + self.offset[1],
                numpy.floor((bounds[2] - self.offset[0]) / step_x) * step_x + self.offset[0],
                numpy.floor((bounds[3] - self.offset[1]) / step_y) * step_y + self.offset[1],
            )
        if mode == "nearest":
            return (
                round((bounds[0] - self.offset[0]) / step_x) * step_x + self.offset[0],
                round((bounds[1] - self.offset[1]) / step_y) * step_y + self.offset[1],
                round((bounds[2] - self.offset[0]) / step_x) * step_x + self.offset[0],
                round((bounds[3] - self.offset[1]) / step_y) * step_y + self.offset[1],
            )
        raise ValueError(f"mode = '{mode}' is not supported. Supported modes: ('expand', 'contract', 'nearest')")

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
            raise ValueError(f"supplied bounds '{bounds}' are not aligned with the grid lines. Consider calling 'align_bounds' first.")

        if not self.are_bounds_aligned(bounds):
            bounds = self.align_bounds(bounds, mode="expand")

        # get coordinates of two diagonally opposing corner-cells
        left_top = (bounds[0] + self.dx / 2, bounds[3] - self.dy / 2)
        right_bottom = (bounds[2] - self.dx / 2, bounds[1] + self.dy / 2)

        # translate the coordinates of the corner cells into indices
        left_top_id, right_bottom_id = self.cell_at_point([left_top, right_bottom])


        # if self._shape == "pointy":
        #     even_index = left_top_id[1] % 2 == 0
        # elif self._shape == "flat":
        #     even_index = index[:,0] % 2 == 0
        #     relative_neighbour_ids_even = {1,3,5,6,7,8}
        #     relative_neighbour_ids_odd = {0,1,2,3,5,7}

        # turn minimum and maximum indices into fully arranged array
        # TODO: think about dtype. 64 is too large. Should this be exposed? Or automated based on id range?
        ids_y = numpy.arange(left_top_id[1], right_bottom_id[1]-1, -1, dtype="int32") # y-axis goes from top to bottom (high to low), hence step size is -1
        ids_x = list(range(left_top_id[0], right_bottom_id[0]+1))

        # TODO: only return ids_y and ids_x without fully filled grid
        shape = (len(ids_y), len(ids_x))
        ids = numpy.empty((2, numpy.multiply(*shape)), dtype="int32")
        ids[0] = ids_x * len(ids_y)
        ids[1] = numpy.repeat(ids_y, len(ids_x))

        return ids.T, shape

    @property
    def parent_grid_class(self):
        return HexGrid


class BoundedRectGrid(BoundedGrid, RectGrid):

    def __init__(self, data, *args, bounds, **kwargs):
        if bounds[2] <= bounds [0] or bounds[3] <= bounds[1]:
            raise ValueError(f"Incerrect bounds. Minimum value exceeds maximum value for bounds {bounds}")
        dx = (bounds[2] - bounds[0]) / data.shape[1]
        dy = (bounds[3] - bounds[1]) / data.shape[0]

        offset_x = bounds[0] % dx
        offset_y = bounds[1] % dy
        offset_x = dx - offset_x if offset_x < 0 else offset_x
        offset_y = dy - offset_y if offset_y < 0 else offset_y
        offset = (
            0 if numpy.isclose(offset_x, dx) else offset_x, 
            0 if numpy.isclose(offset_y, dy) else offset_y
        )
        super(BoundedRectGrid, self).__init__(data, *args, dx=dx, dy=dy, bounds=bounds, offset=offset, **kwargs)

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
        return numpy.linspace(self.bounds[0] + self.dx / 2, self.bounds[2] - self.dx / 2, self.width)

    @property
    def lat(self):
        """Array of lat values

        Returns
        -------
        :class:`numpy.ndarray`
            1D-Array of size `height`, containing the latitudinal values from top to bottom
        """
        return numpy.linspace(self.bounds[3] - self.dy / 2, self.bounds[1] + self.dy / 2, self.height)

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
        latlon = numpy.array(numpy.meshgrid(self.lon, self.lat, sparse=False, indexing="xy"))

        # return grid in shape (width, height, lonlat)
        return numpy.array([latlon[0].ravel(),latlon[1].ravel()]).T

    def intersecting_cells(self, other):
        raise NotImplementedError()

    def shared_bounds(self, other):
        other_bounds = other.bounds if isinstance(other, BaseGrid) else other
        if not self.intersects(other):
            raise IntersectionError(f"Grid with bounds {self.bounds} does not intersect with grid with bounds {other_bounds}.")
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

    def crop(self, new_bounds, bounds_crs=None, buffer_cells=0):

        if bounds_crs is not None:
            bounds_crs = CRS.from_user_input(bounds_crs)
            transformer = Transformer.from_crs(bounds_crs, self.crs, always_xy=True)
            new_bounds = transformer.transform_bounds(*new_bounds)

        if not self.intersects(new_bounds):
            raise IntersectionError(f"Cannot crop grid with bounds {self.bounds} to {new_bounds} for they do not intersect.")
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
        cropped_data = self._data[slice_y, slice_x] #Fixme: seems to be flipped?
        # cropped_data = self._data[slice_x, slice_y]
        return self.update(cropped_data, bounds=new_bounds)

    def _data_slice_from_bounds(self, bounds):

        if not self.are_bounds_aligned(bounds):
            raise ValueError(f"Cannot create slice from unaligned bounds {tuple(bounds)}")

        difference_left = round(abs((self.bounds[0] - bounds[0]) / self.dx))
        difference_right = round(abs((self.bounds[2] - bounds[2]) / self.dx))
        slice_x = slice(
            difference_left,
            self.width - difference_right, # add one for upper bound of slice is exclusive
        )

        difference_bottom = round(abs((self.bounds[1] - bounds[1]) / self.dy))
        difference_top = round(abs((self.bounds[3] - bounds[3]) / self.dy))
        slice_y = slice(
            difference_top,
            self.height - difference_bottom, # add one for upper bound of slice is exclusive
        )

        return slice_y, slice_x

    def cell_corners(self, index: numpy.ndarray = None) -> numpy.ndarray:
        if index is None:
            index = self.indices()
        return super(BoundedRectGrid, self).cell_corners(index=index)

    def indices(self, index: numpy.ndarray = None):
        """Return the indices"""
        # I guess this only makes sense for data grids, maybe remove the index argument?
        if index is None:
            return self.cells_in_bounds(self.bounds)
        cell_centers = self.centroid(index=index)
        return self.cell_at_point(cell_centers)

    def resample(self, alignment_grid, method="nearest"):

        if self.crs is None or alignment_grid.crs is None:
            warnings.warn("`crs` not set for one or both grids. Assuming both grids have an identical CRS.")
            different_crs = False
        else:
            different_crs = not self.crs.is_exact_same(alignment_grid.crs)

        # make sure the bounds align with the grid
        if different_crs:
            transformer = Transformer.from_crs(self.crs, alignment_grid.crs, always_xy=True)
            transformed_corners = numpy.vstack(transformer.transform(*self.corners.T)).T
            bounds = (
                min(transformed_corners[0, 0], transformed_corners[3, 0]),
                min(transformed_corners[2, 1], transformed_corners[3, 1]),
                max(transformed_corners[1, 0], transformed_corners[2, 0]),
                max(transformed_corners[0, 1], transformed_corners[1, 1])
            )
        else:
            bounds = self.bounds

        # Align using "contract" for we cannot sample outside of the original bounds
        new_bounds = alignment_grid.align_bounds(bounds, mode="contract")

        new_ids, new_shape = alignment_grid.cells_in_bounds(bounds=new_bounds)

        new_points = alignment_grid.centroid(new_ids)

        if different_crs:
            transformer = Transformer.from_crs(alignment_grid.crs, self.crs, always_xy=True)
            transformed_points = transformer.transform(*new_points.T)
            new_points = numpy.vstack(transformed_points).T

        nodata_value = self.nodata_value if self.nodata_value is not None else numpy.nan
        if method == "nearest":
            new_ids = self.cell_at_point(new_points)
            value = self.value(new_ids)
        elif method == "bilinear":
            tl_ids, tr_ids, bl_ids, br_ids = self.cells_near_point(new_points)

            tl_val = self.value(tl_ids, oob_value=nodata_value)
            tr_val = self.value(tr_ids, oob_value=nodata_value)
            bl_val = self.value(bl_ids, oob_value=nodata_value)
            br_val = self.value(br_ids, oob_value=nodata_value)

            # determine relative location of new point between old cell centers in x and y directions
            abs_diff = (new_points - self.centroid(bl_ids))
            x_diff = abs_diff[:,0] / self.dx
            y_diff = abs_diff[:,1] / self.dy

            top_val = tl_val + (tr_val - tl_val) * x_diff
            bot_val = bl_val + (br_val - bl_val) * x_diff
            value = bot_val + (top_val - bot_val) * y_diff

            # TODO: remove rows and cols with nans around the edge after bilinear
        else:
            raise ValueError(f"Resampling method '{method}' is not supported.")

        value = value.reshape(new_shape)
        new_grid = BoundedRectGrid(value, bounds=new_bounds, crs=alignment_grid.crs, nodata_value=nodata_value)

        return new_grid

    def to_crs(self, crs, resample_method="nearest"):
        new_inf_grid = super(BoundedRectGrid, self).to_crs(crs, resample_method=resample_method)
        return self.resample(new_inf_grid, method=resample_method)

    def numpy_id_to_grid_id(self, np_index):
        centroid_topleft = (self.bounds[0] + self.dx / 2, self.bounds[3] - self.dy / 2)
        index_topleft = self.cell_at_point(centroid_topleft)
        return (index_topleft[0] + np_index[1], index_topleft[1] - np_index[0])

    def grid_id_to_numpy_id(self, index):
        centroid_topleft = (self.bounds[0] + self.dx / 2, self.bounds[3] - self.dy / 2)
        index_topleft = self.cell_at_point(centroid_topleft)
        return (index_topleft[1] - index[1], index[0] - index_topleft[0])

    def interp_nodata(self, *args, **kwargs):
        """Please refer to :func:`~gridkit.bounded_grid.BoundedGrid.interp_nodata`."""
        # Fixme: in the case of a rectangular grid, a performance improvement can be obtained by using scipy.interpolate.interpn
        return super(BoundedRectGrid, self).interp_nodata(*args, **kwargs)

