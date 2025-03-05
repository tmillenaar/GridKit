import warnings
from typing import List, Union

import matplotlib.patches as mpatches
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy
import shapely
import shapely.geometry
from matplotlib.collections import PatchCollection


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
    fill: bool = True,
    filled: bool = None,
    ax=None,
    add_colorbar=False,
    **kwargs,
):
    """Plot polygons on a map and color them based on the supplied ``values``.

    .. warning ::

        This function is a workaround to have matplotlib plot arbitrary shapes.
        It is not performant and thus not suitable for large amounts of data.
        Moreover, sometimes the shapes don't seem to align well in the corners.
        This is a matplotlib visualization issue, likely related to the resolution of the figure.
        The jagged corners tend to disappear when zooming in closely.

    ..

    Parameters
    ----------
    geoms: List[`shapely.Polygon`]
        A list of shapely polygons to draw
    colors: `Union[numpy.ndarray, str]`
        If a string is supplied, the string is assumed to be the name of a matplotlib color, e.g. 'green'.
        If colors is a numpy array or a list,
        these values will be used to color the supplied ``geoms`` according the the supplied ``cmap``.
        If None, the all polygons will be black.
    cmap: :class:`str`
        The matplitlib complient colormap name to use
        Will be ignored if the 'values' argument not is supplied.
    fill: :class:`bool`
        Whether only the outline of the polygon should be drawn (False) or the polygon should be filled (True).
        If True, the 'edgecolor' (outline of each cell) will also be colored.
        If both a fill color and a small cell outline are desired, supply the keyword argument `edgecolors=None`
        alongside `colors`. If both a fill color and a different cell outline color are desired,
        supply a custom array for `edgecolor` alongside `colors`.
    ax: `matplotlib.axes.Axes` (optional)
        The matplotlib axis object to plot on.
        If an axis object is supplied, the plot will be edited in-place.
        Default: `matplotlib.pyplot`
    **kwargs:
        Keyword arguments passed to matplotlib's PatchCollection, see:
        https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.Collection

    Returns
    -------
    None

    """
    if filled is not None:
        warnings.warn(
            """The argument 'filled' for doc_utils has been deprecated in favor of 'fill' and will be removed in a future version."""
        )
        fill = filled

    if ax is None:
        ax = plt.gca()

    # Sanetize input
    if isinstance(geoms, numpy.ndarray):
        geoms = geoms.ravel()
        geoms = shapely.MultiPolygon(iter(geoms))
    elif isinstance(geoms, list):
        geoms = shapely.MultiPolygon(geoms)
    elif isinstance(geoms, shapely.geometry.Polygon):
        geoms = shapely.geometry.MultiPolygon([geoms])
    elif isinstance(geoms, shapely.geometry.base.GeometrySequence):
        geoms = geoms._parent
    bounds = geoms.bounds

    if colors is None:
        colors = "black"
    if isinstance(colors, str):
        colors = numpy.full(shape=len(geoms.geoms), fill_value=colors)
    elif all(isinstance(c, str) for c in colors):
        pass  # already got passed a list of color names
    elif (
        isinstance(colors, numpy.ndarray)
        and colors.ndim > 1
        and (colors.shape[-1] == 3 or colors.shape == 4)
    ):
        pass  # Assume rgb(a) values were supplied. Do nothing.
    else:
        # create colormap that matches our values
        cmap = getattr(pl.cm, cmap)
        values = colors
        vmin = kwargs.pop("vmin", numpy.nanmin(values))
        vmax = kwargs.pop("vmax", numpy.nanmax(values))
        values_normalized = values - vmin
        values_normalized = values_normalized / numpy.nanmax(values_normalized)
        colors = cmap(
            values_normalized
        ).squeeze()  # squeeze to remove empty axes (when values is pandas series)
        colors[numpy.all(colors == 0, axis=1)] += 1  # turn black (nodata) to white
        kwargs.setdefault("clim", (vmin, vmax))

    polygons = []
    for geom in geoms.geoms:
        polygon = mpatches.Polygon(numpy.array(geom.exterior.coords))
        polygons.append(polygon)

    if fill:
        kwargs["facecolors"] = colors
        kwargs.setdefault("edgecolors", colors)
    else:
        kwargs["facecolors"] = "None"
        kwargs["edgecolors"] = colors

    im = ax.add_artist(
        PatchCollection(
            patches=polygons,
            cmap=cmap,
            **kwargs,
        )
    )

    if add_colorbar:
        plt.colorbar(im)

    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])

    return im
