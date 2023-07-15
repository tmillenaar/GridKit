import numpy
import pytest

from gridkit import hex_grid, rect_grid
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
