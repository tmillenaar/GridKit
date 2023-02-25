"""
Interpolate from points
=======================

Fill a BoundedGrid based on point data

Introduction
============

In this example a BoundedGrid is created from a set of points by interpolating between the points.
For this operation we need:

#. locations of the points
#. values of the points
#. a grid to interpolate on

In this example, we simulate the point data using a 2d sinusoidal equation.
We sample the equation at pseudo-random locations.

"""

# sphinx_gallery_thumbnail_number = -1

import numpy

numpy.random.seed(0)
x = 100 * numpy.random.rand(100)
numpy.random.seed(1)
y = 100 * numpy.random.rand(100)
values = numpy.sin(x/(10*numpy.pi)) * numpy.sin(y/(10*numpy.pi))
points = numpy.array([x,y]).T

# %%
# 
# This input data looks as follows:

import matplotlib.pyplot as plt
plt.scatter(x,y,c=values)
plt.colorbar()
plt.title("Simulated input values")
plt.show()

# %%
# 
# Now we can create a grid and interpolate onto that grid.

from gridkit import rect_grid

empty_grid = rect_grid.RectGrid(dx=5, dy=5)
data_grid = empty_grid.interp_from_points(points, values)

# %%
# 
# Currently the interpolation methods "nearest", "linear" and "cubic" are supported,
# based on scpiy's ``NearestNDInterpolator``, ``LinearNDInterpolator`` and ``CloughTocher2DInterpolator``, respectively.
# Here we try each method and plot them next to each other to compare.
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 5))

for ax, method in zip(axes, ("nearest", "linear", "cubic")):
    import matplotlib.pyplot as plt
    data_grid = empty_grid.interp_from_points(
        points,
        values,
        method=method,
    )
    ax.imshow(data_grid, extent=data_grid.mpl_extent)
    ax.scatter(x, y, c=values, edgecolors='black')
    ax.set_title(method, fontdict={'fontsize': 30})

fig.tight_layout()
plt.show()

# %%
#
# In this example, the `cubic` interpolation represents the original function the best, but this will differ per usecase.
# Note how the BoundedGrid has nodata values at the border in the "linear" and "cubic" cases.
# The bounds of the grid are automatically chosen to align with the supplied grid and accommodate all points.
# Cells of which the center is not within the convex hull of the data,
# but are within the selected bounds, get a `nodata_value` as specified in ``interp_from_points`` (default numpy.nan).
