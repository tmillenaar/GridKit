from typing import Literal, Tuple

import numpy
import shapely
from pyproj import CRS, Transformer

from gridkit.base_grid import BaseGrid
from gridkit.bounded_grid import BoundedGrid
from gridkit.errors import AlignmentError, IntersectionError
from gridkit.gridkit_rs import PyO3TriGrid
from gridkit.index import GridIndex, validate_index


class TriGrid(BaseGrid):
    """Abstraction that represents an infinite grid with cells in the shape of equilateral triangles.

    The size of each cell can be specified through the `size` or `area` arguments.

    Initialization parameters
    -------------------------
    size: float
        The spacing between two cell centroids in horizontal direction. Cannot be supplied together with `area` or 'side_length'.
    area: float
        The area of a cell. Cannot be supplied together with `size` or 'side_length'.
    side_length: float
        The length of the sides of a cell, which is 1/6th the cell outline. . Cannot be supplied together with `area` or 'size'.
    offset: `Tuple(float, float)` (optional)
        The offset in dx and dy.
        Shifts the whole grid by the specified amount.
        The shift is always reduced to be maximum one cell size.
        If the supplied shift is larger,
        a shift will be performed such that the new center is a multiple of dx or dy away.
        Default: (0,0)
    rotation: float
        The counter-clockwise rotation of the grid around the origin in degrees.
    crs: `pyproj.CRS` (optional)
        The coordinate reference system of the grid.
        The value can be anything accepted by pyproj.CRS.from_user_input(),
        such as an epsg integer (eg 4326), an authority string (eg “EPSG:4326”) or a WKT string.
        Default: None

    See also
    --------
    :class:`.RectGrid`
    :class:`.HexGrid`
    :class:`.BoundedTriGrid`

    """

    def __init__(
        self,
        *args,
        size=None,
        area=None,
        side_length=None,
        orientation="flat",
        offset=(0, 0),
        rotation=0,
        **kwargs,
    ):
        supplied_sizes = set()
        if area is not None:
            supplied_sizes.add("area")
        if size is not None:
            supplied_sizes.add("size")
        if side_length is not None:
            supplied_sizes.add("side_length")

        if len(supplied_sizes) == 0:
            raise ValueError(
                "No cell size can be determined. Please supply one of 'size' or 'area' or 'side_length'."
            )
        if len(supplied_sizes) > 1:
            raise ValueError(
                f"Argument conflict. Please supply either 'size' or 'area' or 'side_length'. Got: {' AND '.join(supplied_sizes)}"
            )
        if area is not None:
            size = self._area_to_size(area)
        if side_length is not None:
            size = self._side_length_to_size(side_length)

        self._size = size
        self._radius = size / 3**0.5
        self._rotation = rotation
        self._orientation = orientation
        self._grid = PyO3TriGrid(
            cellsize=size,
            offset=tuple(offset),
            rotation=rotation,
            orientation=orientation,
        )

        self.bounded_cls = BoundedTriGrid
        super(TriGrid, self).__init__(*args, **kwargs)

    @property
    def definition(self):
        return dict(
            size=self.size, offset=self.offset, rotation=self.rotation, crs=self.crs
        )

    @property
    def orientation(self) -> str:
        """The shape of the grid as supplied when initiating the class.
        This can be either "flat" or "pointy" referring to the top of the cells.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        """Set the shape of the grid to a new value. Possible values: 'pointy' or 'flat'"""
        if not value in ("pointy", "flat"):
            raise ValueError(
                f"Shape cannot be set to '{value}', must be either 'pointy' or 'flat'"
            )
        rot = self.rotation
        self._orientation = value
        self.rotation = (
            rot  # Re-run rotation settter to update rotaiton according to new shape
        )

    def _area_to_size(self, area):
        """Find the ``size`` that corresponds to a specific area."""
        return 2 * (area / 3**0.5) ** 0.5

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
    def side_length(self):
        """The length of the side of a cell.
        For a TriGrid, the side length is the same as `HexGrid.size`"""
        return self.size

    def _side_length_to_size(self, side_length):
        """Find the ``size`` that corresponds to the specified length of the side of a cell.
        In the case of a TriGrid that is 1/3rd the outline of the cell."""
        return side_length

    @validate_index
    def centroid(self, index):
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

    @validate_index
    def cell_corners(self, index=None):
        if index is None:
            raise ValueError(
                "For grids that do not contain data, argument `index` is to be supplied to method `corners`."
            )
        return_shape = (
            *index.shape,
            3,
            2,
        )
        index = (
            index.ravel().index[None] if index.index.ndim == 1 else index.ravel().index
        )
        corners = self._grid.cell_corners(index=index)
        return corners.reshape(return_shape)

    def cell_at_point(self, point):
        point = numpy.array(point, dtype="float64")
        point = point[None] if point.ndim == 1 else point
        return_shape = point.shape
        result = self._grid.cell_at_points(point.reshape(-1, 2))
        return GridIndex(result.reshape(return_shape))

    def cells_in_bounds(self, bounds, return_cell_count=False):

        if self.rotation != 0:
            raise NotImplementedError(
                f"`cells_in_bounds` is not suppored for rotated grids. Roatation: {self.rotation} degrees"
            )

        if not self.are_bounds_aligned(bounds):
            raise ValueError(
                f"supplied bounds '{bounds}' are not aligned with the grid lines. Consider calling 'align_bounds' first."
            )
        ids, shape = self._grid.cells_in_bounds(bounds)
        ids = GridIndex(ids.reshape((*shape, 2)))
        return (ids, shape) if return_cell_count else ids

    def cells_near_point(self, point):
        point = numpy.array(point, dtype=float)
        original_shape = (*point.shape[:-1], 6, 2)
        point = point[None] if point.ndim == 1 else point
        point = point.reshape(-1, 2)
        ids = self._grid.cells_near_point(point)
        return GridIndex(ids.squeeze().reshape(original_shape))

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
        :class:`.TriGrid`
            A new grid that is ``factor`` times smaller then the original grid.

        """
        if not factor % 1 == 0:
            raise ValueError(
                f"Got a 'factor' that is not a whole number. Please supply an integer. Got: {factor}"
            )

        sub_grid = self.update(size=self.size / factor, rotation=self.rotation)
        anchor_loc = self.cell_corners([0, 0])[0]
        sub_grid.anchor(anchor_loc, cell_element="corner", in_place=True)
        return sub_grid

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
            depth=int(depth),
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

    def to_bounded(self, bounds, fill_value=numpy.nan):
        _, shape = self.cells_in_bounds(bounds, return_cell_count=True)
        data = numpy.full(shape=shape, fill_value=fill_value)
        return self.bounded_cls(
            data=data,
            bounds=bounds,
            nodata_value=fill_value,
            crs=self.crs,
        )

    def to_crs(self, crs, location=(0, 0), adjust_rotation=False):
        """Transforms the Coordinate Reference System (CRS) from the current CRS to the desired CRS.
        This will update the cell size and the origin offset.

        The ``crs`` attribute on the current grid must be set.

        Parameters
        ----------
        crs: Union[int, str, pyproj.CRS]
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an epsg integer (eg 4326), an authority string (eg "EPSG:4326") or a WKT string.


        location: (float, float) (default: (0,0))
            The location at which to perform the conversion.
            When transforming to a new coordinate system, it matters at which location the transformation is performed.
            The chosen location will be used to determinde the cell size of the new grid.
            If you are unsure what location to use, pich the center of the area you are interested in.

            .. Warning ::

                The location is defined in the original CRS, not in the CRS supplied as the argument to this function call.

        adjust_rotation: bool (default: False)
            If False, the grid in the new crs has the same rotation as the original grid.
            Since coordinate transformations often warp and rotate the grid, the original rotation is often not a good fit anymore.
            If True, set the new rotation to match the orientation of the grid at ``location`` after coordinate transformation.

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
        :meth:`.BoundedTriGrid.to_crs`
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

        new_offset = transformer.transform(
            location[0] + self.offset[0], location[1] + self.offset[1]
        )

        point_start = numpy.array(transformer.transform(*location))
        point_end = transformer.transform(location[0] + self.size, location[1])

        if adjust_rotation:
            cell_id = self.cell_at_point(location)
            # move over two cells for the next cell is flipped and thus it's centroid not at the same y position
            centroids = self.centroid([cell_id, (cell_id.x + 2, cell_id.y)])
            trans_centroids = numpy.array(transformer.transform(*centroids.T)).T
            vector = trans_centroids[1] - trans_centroids[0]
            rotation = numpy.degrees(numpy.arctan2(vector[1], vector[0]))
            trans_corners = numpy.array(
                transformer.transform(
                    *self.cell_corners(self.cell_at_point(location)).T
                )
            ).T
            area = shapely.Polygon(trans_corners).area
        else:
            rotation = self.rotation
            new_dx = numpy.linalg.norm(
                point_start
                - numpy.array(transformer.transform(location[0] + self.dx, location[1]))
            )
            new_dy = numpy.linalg.norm(
                point_start
                - numpy.array(transformer.transform(location[0], location[1] + self.dy))
            )
            area = new_dx * new_dy
        size = numpy.linalg.norm(numpy.subtract(point_end, point_start))

        # new_grid = self.parent_grid_class(area=area, offset=new_offset, crs=crs, rotation=rotation)
        new_grid = self.parent_grid_class(
            size=size, offset=new_offset, crs=crs, rotation=rotation
        )

        if adjust_rotation:
            new_grid.anchor(trans_corners[0], cell_element="corner", in_place=True)
        return new_grid

    def _update_inner_grid(
        self, size=None, offset=None, rotation=None, orientation=None
    ):
        if size is None:
            size = self.size
        if offset is None:
            offset = self.offset
        if rotation is None:
            rotation = self.rotation
        if orientation is None:
            orientation = self.orientation
        return PyO3TriGrid(
            cellsize=size,
            offset=offset,
            rotation=rotation,
            orientation=orientation,
        )

    def update(
        self,
        size=None,
        area=None,
        offset=None,
        rotation=None,
        orientation=None,
        crs=None,
        **kwargs,
    ):
        """Modify attributes of the existing grid and return a copy.
        The original grid remains un-mutated.

        Parameters
        ----------
        size: `float`
            The new spacing between cell centers in x-direction. Cannot be supplied together with ``area``.
        area: float
            The area of a cell. Cannot be supplied together with ``size``.
        offset: `Tuple[float, float]`
            The new offset of the origin of the grid
        rotation: `float`
            The new counter-clockwise rotation of the grid in degrees.
            Can be negative for clockwise rotation.
        orientation: Literal["flat", "pointy"]
            The orientation of the cells. Options are:

             - "flat": The cells point up and down. They are positioned like you would expect a triangle to lie on the ground.
             - "pointy": The cells point left and right. They are positioned standing on a corner if you will.
        crs: Union[int, str, pyproj.CRS]
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an epsg integer (eg 4326), an authority string (eg "EPSG:4326") or a WKT string.

        Returns
        -------
        :class:`.RectGrid`
            A modified copy of the current grid
        """
        if size is None and area is None:
            size = self.size
        if offset is None:
            offset = self.offset
        if rotation is None:
            rotation = self.rotation
        if orientation is None:
            orientation = self.orientation
        if crs is None:
            crs = self.crs
        return TriGrid(
            size=size,
            area=area,
            offset=offset,
            rotation=rotation,
            orientation=orientation,
            crs=crs,
            **kwargs,
        )


class BoundedTriGrid(BoundedGrid, TriGrid):
    """A HexGrid with data encapsulated within a bounding box.

    Initialization parameters
    -------------------------
    data: `numpy.ndarray`
        A 2D ndarray containing the data
    bounds: `Tuple(float, float, float, float)`
        The extend of the data in minx, miny, maxx, maxy.
    crs: `pyproj.CRS` (optional)
        The coordinate reference system of the grid.
        The value can be anything accepted by pyproj.CRS.from_user_input(),
        such as an epsg integer (eg 4326), an authority string (eg “EPSG:4326”) or a WKT string.
        Default: None

    See also
    --------
    :class:`.TriGrid`
    :class:`.BoundedRectGrid`
    :class:`.BoundedHexGrid`

    """

    def __init__(self, data, *args, bounds=None, orientation="flat", **kwargs):

        data = numpy.array(data) if not isinstance(data, numpy.ndarray) else data

        if data.ndim != 2:
            raise ValueError(
                f"Expected a 2D numpy array, got data with shape {data.shape}"
            )

        if bounds is None:
            bounds = (0, 0, data.shape[1], data.shape[0] * 3**0.5)

        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError(
                f"Incerrect bounds. Minimum value exceeds maximum value for bounds {bounds}"
            )

        dx = (bounds[2] - bounds[0]) / data.shape[1]
        dy = (bounds[3] - bounds[1]) / data.shape[0]

        if orientation == "flat":
            if not numpy.isclose(dy, (dx * 3**0.5)):
                raise ValueError(
                    "The supplied data shape cannot be covered by triangles with sides of equal length with the given bounds."
                )
        elif orientation == "pointy":
            if not numpy.isclose(dx, (dy * 3**0.5)):
                raise ValueError(
                    "The supplied data shape cannot be covered by triangles with sides of equal length with the given bounds."
                )
        else:
            raise ValueError(
                f"Unrecognized orientation: {orientation}. Expecte 'flat' or 'pointy'"
            )

        offset_x = bounds[0] % dx
        offset_y = bounds[1] % dy
        offset_x = dx - offset_x if offset_x < 0 else offset_x
        offset_y = dy - offset_y if offset_y < 0 else offset_y
        offset = (
            0 if numpy.isclose(offset_x, dx) else offset_x,
            0 if numpy.isclose(offset_y, dy) else offset_y,
        )
        size = 2 * dx if orientation == "flat" else 2 * dy

        super(BoundedTriGrid, self).__init__(
            data,
            *args,
            size=size,
            bounds=bounds,
            offset=offset,
            orientation=orientation,
            **kwargs,
        )

        if not self.are_bounds_aligned(bounds):
            raise AlignmentError(
                "Something went wrong, the supplied bounds are not aligned with the resulting grid."
            )

    def centroid(self, index=None):
        if index is None:
            if not hasattr(self, "indices"):
                raise ValueError(
                    "For grids that do not contain data, argument `index` is to be supplied to method `centroid`."
                )
            index = self.indices
        return super(BoundedTriGrid, self).centroid(index=index)

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
            slice_x = slice(
                max(0, slice_x.start - buffer_cells),
                min(self.width, slice_x.stop + buffer_cells),
            )
            slice_y = slice(
                max(0, slice_y.start - buffer_cells),
                min(self.height, slice_y.stop + buffer_cells),
            )
            new_bounds = (
                max(new_bounds[0] - buffer_cells * self.dx, self.bounds[0]),
                max(new_bounds[1] - buffer_cells * self.dy, self.bounds[1]),
                min(new_bounds[2] + buffer_cells * self.dx, self.bounds[2]),
                min(new_bounds[3] + buffer_cells * self.dy, self.bounds[3]),
            )
        # cropped_data = numpy.flipud(numpy.flipud(self._data)[slice_y, slice_x]) # TODO: fix this blasted flipping. The raster should not be stored upside down maybe
        cropped_data = self._data[slice_y, slice_x]  # Fixme: seems to be flipped?
        # cropped_data = self._data[slice_x, slice_y]
        return self.update(cropped_data, bounds=new_bounds)

    @validate_index
    def cell_corners(self, index: numpy.ndarray = None) -> numpy.ndarray:
        if index is None:
            index = self.indices
        return super(BoundedTriGrid, self).cell_corners(index=index)

    @validate_index
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
        if not isinstance(sample_points, numpy.ndarray):
            sample_points = numpy.array(sample_points, dtype=float)
        else:
            sample_points = sample_points.astype(float)

        original_shape = sample_points.shape
        if not original_shape[-1] == 2:
            raise ValueError(
                f"Expected the last axis of sample_points to have two elements (x,y). Got {original_shape[-1]} elements"
            )
        sample_points = sample_points.reshape(-1, 2)
        nearby_cells = self.cells_near_point(sample_points)
        nearby_centroids = self.centroid(nearby_cells)
        nearby_values = self.value(nearby_cells)

        if sample_points.squeeze().ndim == 1:
            sample_points = sample_points.squeeze()[None]
            nearby_centroids = nearby_centroids[None]
            nearby_values = nearby_values[None]
        values = self._grid.linear_interpolation(
            sample_points,
            nearby_centroids,
            nearby_values.astype(
                float  # FIXME: figure out generics in rust to allow for other dtypes
            ),
        )

        if len(values) == 1:
            return values[0]
        return (
            values.squeeze().reshape(*original_shape[:-1])
            if original_shape[:-1]
            else values
        )

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
        :class:`.BoundedTriGrid`
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
        new_inf_grid = super(BoundedTriGrid, self).to_crs(crs)
        return self.resample(new_inf_grid, method=resample_method)

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

    def numpy_id_to_grid_id(self, np_index):
        centroid_topleft = (self.bounds[0] + self.dx / 2, self.bounds[3] - self.dy / 2)
        index_topleft = self.cell_at_point(centroid_topleft)
        ids = numpy.array(
            [index_topleft.x + np_index[1] - 1, index_topleft.y - np_index[0]]
        )
        return GridIndex(ids.T)

    @validate_index
    def grid_id_to_numpy_id(self, index):
        if index.index.ndim > 2:
            raise ValueError(
                "Cannot convert nd-index to numpy index. Consider flattening the index using `index.ravel()`"
            )
        centroid_topleft = (
            self.bounds[0] + self.cell_width / 2,
            self.bounds[3] - self.cell_height / 2,
        )
        index_topleft = self.cell_at_point(centroid_topleft)
        return (index_topleft.y - index.y, index.x - index_topleft.x + 1)
