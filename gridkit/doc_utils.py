import warnings
from typing import List, Union

import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy
import shapely


def generate_2d_scatter_doughnut(num_points: float, radius: float) -> numpy.ndarray:
    """Generate a 2d doughnut shape of points distributed using gaussian noise.

    Parameters
    ----------
    num_points: :class:`float`
        The total number of points that make up the doughnut shape.
    radius: :class:`float`
        The radius of the circle around which the points are scattered

    Returns
    -------
    :class:`numpy.ndarray`
        The points that make up the doughnut shape
    """
    # Generate angles uniformly spaced around a circle
    angles = numpy.linspace(0, 2 * numpy.pi, num_points)

    # Generate coordinates of the points
    x_coords = radius * numpy.cos(angles)
    y_coords = radius * numpy.sin(angles)

    # Apply Gaussian noise to the coordinates
    numpy.random.seed(0)
    x_coords += numpy.random.normal(0, 1, num_points)
    y_coords += numpy.random.normal(0, 1, num_points)
    return numpy.array([x_coords, y_coords]).T


def plot_polygons(
    geoms: List[shapely.Polygon],
    colors: Union[numpy.ndarray, str] = None,
    cmap: str = "viridis",
    filled: bool = True,
    ax=plt,
    **kwargs,
):
    """Plot polygons on a map and color them based on the supplied ``values``

    Parameters
    ----------
    geoms: List[shapely.Polygon]
        A list of shapely polygons to draw
    colors: `Union[numpy.ndarray, str]`
        If a string is supplied, the string is assumed to be the name of a matplotlib color, e.g. 'green'.
        If colors is a numpy array or a list,
        these values will be used to color the supplied ``geoms`` according the the supplied ``cmap``.
        If None, the all polygons will be black.
    cmap: :class:`str`
        The matplitlib complient colormap name to use
        Will be ignored if the 'values' argument not is supplied.
    filled: :class:`bool`
        Whether only the outline of the polygon should be drawn (False) or the polygon should be filled (True)
    ax: `matplotlib.axes.Axes` (optional)
        The matplotlib axis object to plot on.
        If an axis object is supplied, the plot will be edited in-place.
        Default: `matplotlib.pyplot`

    Returns
    -------
    None

    """
    # ravel geoms if necessary
    if isinstance(geoms, numpy.ndarray):
        geoms = geoms.ravel()

    if isinstance(colors, str) or colors is None:
        colors = [colors if colors else "black"] * len(geoms)
    if all(isinstance(c, str) for c in colors):
        pass  # already got passed a list of color names
    else:
        # create colormap that matches our values
        cmap = getattr(pl.cm, cmap)
        values = colors
        vmin = numpy.nanmin(values)
        values_normalized = values - vmin
        vmax = numpy.nanmax(values_normalized)
        values_normalized = values_normalized / vmax
        colors = cmap(
            values_normalized
        ).squeeze()  # squeeze to remove empty axes (when values is pandas series)
        colors[numpy.all(colors == 0, axis=1)] += 1  # turn black (nodata) to white

    # plot each cell as a polygon with color
    for geom, color in zip(geoms, colors):
        if filled:
            ax.fill(*geom.exterior.xy, color=color, **kwargs)
        else:
            ax.plot(*geom.exterior.xy, color=color, **kwargs)
