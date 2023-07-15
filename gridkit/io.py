import numpy
import rasterio
from pyproj import CRS, Transformer

from gridkit.rect_grid import BoundedRectGrid


def read_geotiff(path, bands=1, bounds=None, bounds_crs=None, border_buffer=0):
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
            top, left = raster_file.index(bounds[0], bounds[3])
            bottom, right = raster_file.index(bounds[2], bounds[1])

            if border_buffer:  # note rasterio slices from top to bottom
                left -= border_buffer
                right += border_buffer
                top -= border_buffer
                bottom += border_buffer

            # Create a window from the indices
            window = rasterio.windows.Window.from_slices((top, bottom), (left, right))
            bounds = rasterio.windows.bounds(
                window, raster_file.transform
            )  # update the bounds to those of the window
        else:
            window = None
            b = raster_file.bounds if bounds is None else bounds
            bounds = (b.left, b.bottom, b.right, b.top)

        data = raster_file.read(bands, window=window)
        nodata = raster_file.nodata

    grid = BoundedRectGrid(data, bounds=bounds, crs=crs, nodata_value=nodata)
    return grid


def write_raster(grid, path):
    transform = rasterio.transform.from_bounds(*grid.bounds, grid.width, grid.height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=1,  # nr bands
        dtype=grid._data.dtype,
        crs=grid.crs,
        nodata=grid.nodata_value,
        transform=transform,
    ) as dst:
        dst.write(numpy.expand_dims(grid._data.copy(), 0))
    return path
