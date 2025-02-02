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


@pytest.mark.parametrize("bounds", [None, [4200000.0, 2600000.0, 4210000.0, 2610000.0]])
@pytest.mark.parametrize("border_buffer", [0, 400, -300])
def test_raster_to_data_tile(bounds, border_buffer):
    path = "tests/data/alps_landuse.tiff"
    grid_tile = raster_to_data_tile(path, bounds=bounds, border_buffer=border_buffer)

    if bounds:
        expected_bounds = numpy.array(bounds)
        expected_bounds[:2] -= border_buffer
        expected_bounds[2:] += border_buffer
        expected_nx = expected_ny = 100.0 + (2 * border_buffer / 100)
    else:
        expected_bounds = numpy.array((4116200.0, 2575600.0, 4218700.0, 2624900.0))
        expected_nx, expected_ny = (1025, 493)
        if border_buffer <= 0:
            expected_bounds[:2] -= border_buffer
            expected_bounds[2:] += border_buffer
            expected_nx += 2 * border_buffer / 100
            expected_ny += 2 * border_buffer / 100
    numpy.testing.assert_allclose(grid_tile.bounds, expected_bounds)
    assert grid_tile.grid.dx == 100.0
    assert grid_tile.grid.dy == 100.0
    assert grid_tile.nx == expected_nx
    assert grid_tile.ny == expected_ny
    assert grid_tile.grid.crs.to_epsg() == 3035
