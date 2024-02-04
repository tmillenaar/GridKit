import numpy
import pytest

from gridkit import hex_grid, rect_grid, tri_grid
from gridkit.io import read_geotiff


@pytest.fixture(scope="function")
def basic_bounded_rect_grid():
    data = numpy.arange(3 * 5).reshape(5, 3)
    return rect_grid.BoundedRectGrid(data, bounds=(-1, -2, 2, 3), nodata_value=-9999)


@pytest.fixture(scope="function")
def basic_bounded_pointy_grid():
    data = numpy.arange(4 * 2).reshape(4, 2)
    bounds = (-1.5, -2.598076211353316, 1.5, 2.598076211353316)
    return hex_grid.BoundedHexGrid(
        data, bounds=bounds, nodata_value=-9999, shape="pointy"
    )


@pytest.fixture(scope="function")
def basic_bounded_flat_grid():
    data = numpy.arange(2 * 3).reshape(2, 3)
    bounds = (-1.299038105676658, -1.5, 1.299038105676658, 3.0)
    return hex_grid.BoundedHexGrid(
        data, bounds=bounds, nodata_value=-9999, shape="flat"
    )


@pytest.fixture(scope="function")
def basic_bounded_tri_grid():
    data = numpy.arange(3 * 5).reshape(5, 3)
    # breakpoint()
    # bounds = (-1.0, -3.4641016151377544, 2.0, 3.4641016151377544)
    # grid = tri_grid.TriGrid(size=1)
    # grid.align_bounds((-1, -3, 2, 4))
    bounds = (-1.0, -3.4641016151377544, 2.0, 5.196152422706632)
    return tri_grid.BoundedTriGrid(data, bounds=bounds, nodata_value=-9999)
