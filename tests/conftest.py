import pytest
import numpy
from gridkit import rect_grid
from gridkit.io import read_geotiff

@pytest.fixture(scope="function")
def basic_bounded_rect_grid():
    data = numpy.arange(3*5).reshape(5,3)
    return rect_grid.BoundedRectGrid(data, bounds=(-1,-2,2,3))

@pytest.fixture(scope="function")
def wgs84_grid():
    return read_geotiff("tests/data/do_not_commit/GeogToWGS84GeoKey5.tif")
