import numpy
import pytest
from gridkit import raster_to_data_tile, read_raster, write_raster


def test_read_raster():
    path = "tests/data/alps_landuse.tiff"
    grid = read_raster(path)

    expected_bounds = (4116200.0, 2575600.0, 4218700.0, 2624900.0)
    numpy.testing.assert_allclose(grid.bounds, expected_bounds)
    assert grid.dx == 100.0
    assert grid.dy == 100.0
    assert grid.crs.to_epsg() == 3035


def test_write_raster(tmp_path, basic_bounded_rect_grid):
    filepath = write_raster(basic_bounded_rect_grid, tmp_path / "out.tiff")
    assert filepath.exists()
    assert filepath.is_file()
    assert filepath.suffix == ".tiff"


def test_raster_to_data_tile():
    path = "tests/data/alps_landuse.tiff"
    grid_tile = raster_to_data_tile(path)

    expected_corners = [
        [4116200.0, 2624900.0],
        [4218700.0, 2624900.0],
        [4218700.0, 2575600.0],
        [4116200.0, 2575600.0],
    ]
    numpy.testing.assert_allclose(grid_tile.corners(), expected_corners)
    assert grid_tile.grid.dx == 100.0
    assert grid_tile.grid.dy == 100.0
    assert grid_tile.grid.crs.to_epsg() == 3035
