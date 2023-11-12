from gridkit.base_grid import BaseGrid
from gridkit.index import GridIndex, validate_index
from gridkit_rs import PyTriGrid


class TriGrid(BaseGrid):
    def __init__(self, *args, size, shape="pointy", **kwargs):
        self._size = size
        self._radius = size / 3**0.5

        self._grid = PyTriGrid(cellsize=size)

        self._shape = shape
        self.bounded_cls = None
        super(TriGrid, self).__init__(*args, **kwargs)

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

    @validate_index
    def centroid(self, index):
        index = index.index[None] if index.index.ndim == 1 else index.index
        return self._grid.centroid(index=index).squeeze()

    @validate_index
    def cell_corners(self, index):
        index = index.index[None] if index.index.ndim == 1 else index.index
        return self._grid.cell_corners(index=index).squeeze()

    def cell_at_point(self, point):
        return GridIndex(self._grid.cell_at_point(point))

    def cells_in_bounds(self, bounds):
        return GridIndex(self._grid.cells_in_bounds(bounds))

    def cells_in_bounds_py(self, bounds, return_cell_count: bool = False):
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

        import numpy

        # translate the coordinates of the corner cells into indices
        left_top_id, left_bottom_id, right_top_id, right_bottom_id = self.cell_at_point(
            numpy.stack([left_top, left_bottom, right_top, right_bottom])
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

    def cells_near_point(self):
        raise NotImplementedError()

    def is_aligned_with(self):
        raise NotImplementedError()

    def parent_grid_class(self):
        raise NotImplementedError()

    def relative_neighbours(self):
        raise NotImplementedError()

    def to_bounded(self):
        raise NotImplementedError()

    def to_crs(self):
        raise NotImplementedError()
