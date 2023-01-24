from gridkit.rect_grid import BoundedRectGrid
import rasterio
import numpy
from pyproj import CRS, Transformer


def read_geotiff(path, bands=1, bounds=None, bounds_crs=None):
    """Read data from a GeoTIFF

    Returns
    -------
    :class:`~gridkit.rect_grid.BoundedRectGrid`
        The contents of the GeoTIFF read into a Raster

    """
    if not isinstance(bands, int):
        raise NotImplementedError("Reading multiple bounds is not yet supported.")

    with rasterio.open(path) as raster_file:
        crs = str(raster_file.crs)

        if bounds is not None:
            if bounds_crs is not None:
                bounds_crs = CRS.from_user_input(bounds_crs)
                transformer = Transformer.from_crs(bounds_crs, crs, always_xy=True)
                bounds = transformer.transform_bounds(*bounds)
            ul = raster_file.index(bounds[0], bounds[3])
            lr = raster_file.index(bounds[2], bounds[1])
            # Create a window from the indices
            window = rasterio.windows.Window.from_slices((ul[0], lr[0]), (ul[1], lr[1]))
            bounds = rasterio.windows.bounds(window, raster_file.transform) # update the bounds to those of the window
        else:
            window = None
            b = raster_file.bounds if bounds is None else bounds
            bounds = (b.left, b.bottom, b.right, b.top)

        data = raster_file.read(bands, window=window)        
        nodata = raster_file.nodata

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

    grid = read_geotiff("/Users/mara/Documents/Timo/Projects/gridkit/test/data/do_not_commit/wildfires.tiff")

    from gridkit import plotting
    from gridkit import rect_grid
    gridkit.plot_raster(resampled, "original.png")
    new_grid = rect_grid.RectGrid(
        dx = grid.dx*2,
        dy = grid.dy*2,
        offset = grid.offset
    )
    resampled = grid.resample(new_grid)
    gridkit.plot_raster(resampled, "resampled.png")
    breakpoint()