from gridkit import read_raster, write_raster


def test_read_raster():
    path = "tests/data/alps_landuse.tiff"
    grid = read_raster(path)

    expected_bounds = (4116200.0, 2575600.0, 4218700.0, 2624900.0)
    assert grid.bounds == expected_bounds
    assert grid.dx == 100.0
    assert grid.dy == 100.0
    assert grid.crs.to_epsg() == 3035


def test_write_raster(tmp_path, basic_bounded_rect_grid):
    filepath = write_raster(basic_bounded_rect_grid, tmp_path / "out.tiff")
    assert filepath.exists()
    assert filepath.is_file()
    assert filepath.suffix == ".tiff"
