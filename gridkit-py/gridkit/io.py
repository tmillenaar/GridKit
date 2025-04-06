import warnings
from typing import Tuple

import numpy
import rasterio
from gridkit.rect_grid import BoundedRectGrid, RectGrid
from gridkit.tile import DataTile, Tile
from pyproj import CRS, Transformer


def read_raster(
    path: str,
    bounds: Tuple[float, float, float, float] = None,
    bounds_crs: CRS = None,
    border_buffer: float = 0,
):
    """Read data from a GeoTIFF

    Parameters
    ----------
    path: `str`
        The path to the file. This needs to be a file that is supported by rasterio.
    bounds: `Tuple(float, float, float, float)`, None
        The bounds of the are of interest. Only the data within the supplied bounds is read from the input file.
    bounds_crs: `pyproj.CRS`, None
        The Coordinte Reference System (CRS) of the supploed bounds.
        If the CRS of the bounds does not match that of the input files,
        the bounds are converted to that of the input file before reading.
    border_buffer: `int`, 0
        A buffer to apply to the supplied `bounds` to read in a larger slice of the area.

    Returns
    -------
    :class:`.BoundedRectGrid`
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
    path: `str`
        The locatin of the file to write to (eg ./my_raster.tiff).

    Returns
    -------
    `str`
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


# TODO: Add a write data_tile to raster function
def raster_to_data_tile(
    path,
    bounds=None,
    bounds_crs: CRS = None,
    border_buffer: float = 0,
    band=1,
):
    """Read a raster as a DataTile. Only a single band can be read.

    Parameters
    ----------
    path: `str`, `file object`, `PathLike object`, `FilePath`, or `MemoryFile`
        A filename or URL, a file object opened in binary ('rb') mode, a
        Path object, or one of the rasterio classes that provides the
        dataset-opening interface.
    bounds: `Tuple(float, float, float, float)`, None
        The bounding box of the data to read in (min-x, min-y, max-x, max-y). Can be used to read a smaller
        section of the dataset. If the bounds are larger than the raster bounds,
        the full raster will be read.
    bounds_crs: `pyproj.CRS`, None
        The Coordinate Reference System (CRS) that matches the supplied `bounds` argument.
        The `bounds` are then converted to the CRS that matches the dataset before reading
        the tile. This allows for reading a section of the dataset defined in any CRS,
        unrelated to the CRS of the dataset.

        .. Note ::

            The dataset will be read in the CRS corresponding to the data file,
            not the bounds_crs.

    border_buffer: `float`, 0
        Enlarge the supplied `bounds` with the specified border_radius.
        A negative border_radius can be supplied to decrease the bounds.
        The unit of the border_radius is that of bounds_crs, or that of the
        dataset CRS if bounds_crs is not supplied.
    band: `int`, 0
        The index of the band to read. Default: 1.
        Note that raster bands start at 1, so if you
        want to read the first band, supply band=1, not band=0.

    Returns
    -------
    class:`.DataTile`
        A data tile with spatial properties based on the input raster
    """
    with rasterio.open(path) as raster_file:
        crs = str(raster_file.crs)

        if bounds is None and border_buffer < 0:
            # Allow for shrinking of full dataset using border_buffer if no bounds are supplied
            # If border_buffer is zero or larger then there is no point in cropping, just return the full dataset
            b = raster_file.bounds if bounds is None else bounds
            bounds = (b.left, b.bottom, b.right, b.top)

        if bounds is not None:

            bounds = numpy.array(bounds)

            if border_buffer:  # note rasterio slices from top to bottom
                bounds[:2] -= border_buffer
                bounds[2:] += border_buffer

            if bounds_crs is not None:
                bounds_crs = CRS.from_user_input(bounds_crs)
                transformer = Transformer.from_crs(bounds_crs, crs, always_xy=True)
                bounds = transformer.transform_bounds(*bounds)
            top, left = raster_file.index(bounds[0], bounds[3])
            bottom, right = raster_file.index(bounds[2], bounds[1])

            # Create a window from the indices
            window = rasterio.windows.Window.from_slices((top, bottom), (left, right))
            bounds = rasterio.windows.bounds(
                window, raster_file.transform
            )  # update the bounds to those of the window
        else:
            window = None
            b = raster_file.bounds if bounds is None else bounds
            bounds = (b.left, b.bottom, b.right, b.top)

        data = raster_file.read(band, window=window)
        nodata = raster_file.nodata

    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError(
            f"Incerrect bounds. Minimum value exceeds maximum value for bounds {bounds}"
        )

    dx = (bounds[2] - bounds[0]) / data.shape[1]
    dy = (bounds[3] - bounds[1]) / data.shape[0]

    offset_x = bounds[0] % dx
    offset_y = bounds[1] % dy
    offset_x = dx - offset_x if offset_x < 0 else offset_x
    offset_y = dy - offset_y if offset_y < 0 else offset_y
    offset = (
        0 if numpy.isclose(offset_x, dx) else offset_x,
        0 if numpy.isclose(offset_y, dy) else offset_y,
    )

    grid = RectGrid(dx=dx, dy=dy, offset=offset, crs=crs)
    start_id = grid.cell_at_point(bounds[:2])
    tile = Tile(grid, start_id, nx=data.shape[1], ny=data.shape[0])
    data_tile = DataTile(tile, data, nodata_value=raster_file.nodata)
    return data_tile
