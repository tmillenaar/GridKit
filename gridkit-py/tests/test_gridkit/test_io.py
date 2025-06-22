import numpy
import pytest

from gridkit import (DataTile, RectGrid, Tile, raster_to_data_tile,
                     read_raster, write_raster)


def test_read_raster():
    path = "tests/data/alps_landuse.tiff"
    grid = read_raster(path)

    expected_bounds = (4116200.0, 2575600.0, 4218700.0, 2624900.0)
    numpy.testing.assert_allclose(grid.bounds, expected_bounds)
    assert grid.dx == 100.0
    assert grid.dy == 100.0
    assert grid.crs.to_epsg() == 3035


def test_write_raster(tmp_path, basic_bounded_rect_grid):
    filepath = write_raster(basic_bounded_rect_grid, "./out_bounded.tiff")
    filepath = write_raster(basic_bounded_rect_grid, tmp_path / "out.tiff")
    assert filepath.exists()
    assert filepath.is_file()
    assert filepath.suffix == ".tiff"

    result = raster_to_data_tile(filepath)
    assert numpy.testing.assert_allclose(result, basic_bounded_rect_grid.data)
    assert numpy.testing.assert_allclose(result.offset, basic_bounded_rect_grid.offset)
    assert result.rotation == 0
    assert numpy.testing.assert_allclose(result.dx, basic_bounded_rect_grid.dx)
    assert numpy.testing.assert_allclose(result.dy, basic_bounded_rect_grid.dy)
    assert numpy.testing.assert_allclose(
        result.start_id, basic_bounded_rect_grid.start_id
    )


def test_write_raster_data_tile(tmp_path):
    grid = RectGrid(dx=0.2, dy=0.3)
    tile = Tile(grid, (-1, -2), 4, 5)
    data = numpy.arange(tile.nx * tile.ny).reshape(tile.ny, tile.nx)
    data_tile = tile.to_data_tile(data)
    filepath = write_raster(data_tile, "./out.tiff")
    filepath = write_raster(data_tile, tmp_path / "out.tiff")
    assert filepath.exists()
    assert filepath.is_file()
    assert filepath.suffix == ".tiff"

    result = raster_to_data_tile(filepath)
    assert numpy.testing.assert_allclose(result, data)
    assert numpy.testing.assert_allclose(result.offset, data.offset)
    assert result.rotation == 0
    assert numpy.testing.assert_allclose(result.dx, grid.dx)
    assert numpy.testing.assert_allclose(result.dy, grid.dy)
    assert numpy.testing.assert_allclose(result.start_id, tile.start_id)


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
