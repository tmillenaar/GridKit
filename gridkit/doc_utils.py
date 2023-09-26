from typing import List

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
    geoms: List[shapely.Polygon], values: numpy.ndarray, cmap: str, ax=plt
):
    """Plot polygons on a map and color them based on the supplied ``values``

    Parameters
    ----------
    geoms: List[shapely.Polygon]
        A list of shapely polygons to draw
    values: :class:`numpy.ndarray`
        The values corresponding the the respective ``geoms``.
        These values will be used to color the supplied ``geoms`` according the the supplied ``cmap``.
    cmal: :class:`str`
        The matplitlib complient colormap name to use
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

    # create colormap that matches our values
    cmap = getattr(pl.cm, cmap)
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
        ax.fill(*geom.exterior.xy, alpha=1.0, color=color)
