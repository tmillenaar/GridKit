from gridding.rect_grid import BoundedRectGrid
import rasterio
import numpy


def read_geotiff(path, bands=1):
    """Read data from a GeoTIFF

    Returns
    -------
    Raster
        The contents of the GeoTIFF read into a Raster

    """
    if not isinstance(bands, int):
        raise NotImplementedError("Reading multiple bounds is not yet supported.")

    with rasterio.open(path) as raster_file:
        crs = str(raster_file.crs)
        import numpy
        # data = numpy.flipud(raster_file.read(bands))
        data = raster_file.read(bands)
        # breakpoint()
        b = raster_file.bounds
        bounds = (b.left, b.bottom, b.right, b.top)
        nodata = raster_file.nodata
    # breakpoint()
    # print("")

    grid = BoundedRectGrid(data, bounds=bounds, crs=crs, nodata_value=nodata)
    return grid

# path = "tests/data/do_not_commit/wildfires.tiff"
# read_geotiff(path, bands=1)

def write_raster(grid, path):
    transform = rasterio.transform.from_bounds(*grid.bounds, grid.width, grid.height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=1, # nr bands
        dtype=grid._data.dtype,
        crs=grid.crs,
        nodata=0, # Fixme: make flexible
        transform=transform,
    ) as dst:
        dst.write(numpy.expand_dims(grid._data, 0))
    return path



if __name__ == "__main__":
    # path_t115 = "/home/timo/mounts/glusterfs/tests/2020/timo/JPL/insar/t115.tiff"
    # path_t42 = "/home/timo/mounts/glusterfs/tests/2020/timo/JPL/insar/t42.tiff"

    # grid_115 = read_geotiff(path_t115, bands=1)
    # grid_42 = read_geotiff(path_t42, bands=1)

    grid = read_geotiff("/Users/mara/Documents/Timo/Projects/gridding/test/data/do_not_commit/wildfires.tiff")

    from gridding import plotting
    from gridding import rect_grid
    gridding.plot_raster(resampled, "original.png")
    new_grid = rect_grid.RectGrid(
        dx = grid.dx*2,
        dy = grid.dy*2,
        offset = grid.offset
    )
    resampled = grid.resample(new_grid)
    gridding.plot_raster(resampled, "resampled.png")
    breakpoint()