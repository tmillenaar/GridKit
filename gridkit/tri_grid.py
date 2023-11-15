import numpy

from gridkit.base_grid import BaseGrid
from gridkit.index import GridIndex, validate_index
from gridkit_rs import PyTriGrid


class TriGrid(BaseGrid):
    def __init__(self, *args, size, shape="pointy", offset=(0,0), **kwargs):
        self._size = size
        self._radius = size / 3**0.5

        self._grid = PyTriGrid(cellsize=size, offset=offset)

        self._shape = shape
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
        point = numpy.array(point)
        point = point[None] if point.ndim == 1 else point
        return GridIndex(self._grid.cell_at_point(point))

    def cells_in_bounds(self, bounds, return_cell_count=False):
        ids, shape = self._grid.cells_in_bounds(bounds)
        ids = GridIndex(ids)
        return (ids, shape) if return_cell_count else ids

    def cells_near_point(self):
        raise NotImplementedError()

    def is_aligned_with(self):
        raise NotImplementedError()

    def parent_grid_class(self):
        raise TriGrid

    def relative_neighbours(self):
        raise NotImplementedError()

    def to_bounded(self):
        raise NotImplementedError()

    def to_crs(self):
        raise NotImplementedError()
