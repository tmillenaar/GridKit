import warnings
from typing import Tuple

import numpy
import rasterio
from pyproj import CRS, Transformer

from gridkit.rect_grid import BoundedRectGrid


def read_raster(
    path: str,
    bounds: Tuple[float, float, float, float] = None,
    bounds_crs: CRS = None,
    border_buffer: float = 0,
):
    """Read data from a GeoTIFF

    Parameters
    ----------
    path: :class:`str`
        The path to the file. This needs to be a file that is supported by rasterio.
    bounds: Tuple(float, float, float, float)
        The bounds of the are of interest. Only the data within the supplied bounds is read from the input file.
    bounds_crs: `pyproj.CRS`
        The Coordinte Reference System (CRS) of the supploed bounds.
        If the CRS of the bounds does not match that of the input files,
        the bounds are converted to that of the input file before reading.
    border_buffer: :class:`int`
        A buffer to apply to the supplied `bounds` to read in a larger slice of the area.

    Returns
    -------
    :class:`~gridkit.rect_grid.BoundedRectGrid`
        The contents of the GeoTIFF in the form of a BoundedRectGrid

    See also
    --------
    :func:`.write_raster`
    """
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

        data = raster_file.read(1, window=window)
        nodata = raster_file.nodata

    grid = BoundedRectGrid(data, bounds=bounds, crs=crs, nodata_value=nodata)
    return grid


def read_geotiff(*args, bands=1, **kwargs):
    """Deprecated, please refer to :func:`read_raster`"""
    warnings.warn(
        "gridkit's 'read_geotiff' has been replaced with 'read_raster' and will be removed in a future update."
    )
    return read_raster(*args, **kwargs)


def write_raster(grid, path):
    """Write a BoundedRectGrid to a raster file (eg .tiff).

    Parameters
    ----------
    grid: :class:`.BoundedRectGrid`
        The grid to write to a raster file.
        This can only be a BoundedRectGrid.
    path: :class:`str`
        The locatin of the file to write to (eg ./my_raster.tiff).

    Returns
    -------
    :class:`str`
        The path pointing to the written file

    See also
    --------
    :func:`read_raster`
    """
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
