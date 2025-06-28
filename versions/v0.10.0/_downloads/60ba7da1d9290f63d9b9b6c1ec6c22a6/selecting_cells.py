"""
.. _example selecting cells:

Cell selection using other grids
================================

Identify specific cells using other grids

Introduction
------------

Grids of different shapes and sizes often interact with each other in predictable ways.
This is demonstrated also in :ref:`example hexagon grids`.
This can be used to identify cells from one grid using an other grid.
This may sound abstract, but in this example images will be provided to show what is meant here.

In this example a grid is created where the cells are colored based on how they relate to a second grid.

Defining matching grids
-----------------------

For two grids to overlap in predictable ways, we need to be mindful of our grid definitions.

Let's create two grids, both centered around zero,
where one grid's cells are exactly three times larger than the other grid's cells.

"""

# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy

from gridkit import HexGrid
from gridkit.doc_utils import plot_polygons

# create a grids
fine_grid = HexGrid(size=1, shape="pointy")
fine_grid.offset = (0, fine_grid.dy / 2)
coarse_grid = HexGrid(
    size=3 * fine_grid.dx, offset=(0, 3 * fine_grid.dy / 2), shape="pointy"
)

# %%
#
# Now we have a fine grid, which is the one we want to color,
# and we have a course grid which we will use to do this.
# First, let's plot the outlines of the two grids so we can verify our grids indeed overlap as expected.
# To plot the outlines we can select an area of interest and represent the cells in this area as polygons.
# These polygons can then be plotted. Since the grids were defined to be centered around zero,
# let's center our are of interest around zero as well.

# define an area of interest, slightly larger than what we want to plot
bounds = (-7, -7, 7, 7)

fine_bounds = fine_grid.align_bounds(bounds)
fine_cell_ids = fine_grid.cells_in_bounds(fine_bounds)
fine_shapes = fine_grid.to_shapely(fine_cell_ids)

coarse_bounds = coarse_grid.align_bounds(bounds)
coarse_cell_ids = coarse_grid.cells_in_bounds(coarse_bounds)
coarse_shapes = coarse_grid.to_shapely(coarse_cell_ids)

# %%
#
# Let's plot our grids in the same image so we can compare them.
plot_polygons(fine_shapes, fill=False, linewidth=1, colors="purple")
plot_polygons(coarse_shapes, fill=False, linewidth=2, colors="orange")

plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.show()


# %%
#
# The grids seem to align nicely. Also, there are exactly three purple cells
# between two orange cell edges, as intended.
# Arguably, there are three categories of purple cells when compared to the orange grid.
# The first category contains the purple cells at the center of each of the orange cells.
# Secondly we have the neighbours of these center cells and lastly we have the purple cells at the
# vertices of the orange cells.
# Let's start by coloring in the center cells.

# coarse_centroids = numpy.array([geom.centroid.xy for geom in coarse_shapes.geoms])
coarse_centroids = coarse_grid.centroid(coarse_cell_ids)
center_cells = fine_grid.cell_at_point(coarse_centroids)
center_shapes = fine_grid.to_shapely(center_cells)

plot_polygons(fine_shapes, fill=False, linewidth=1, colors="purple")
plot_polygons(center_shapes, fill=True, colors="limegreen", alpha=0.6)
plot_polygons(coarse_shapes, fill=False, linewidth=2, colors="orange")

plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.show()

# %%
#
# Next, let's use these center cells to find their neighbours and color them too.

# return axes ('inital_cells', 'neighbours', 'xy') with shape (42, 6, 2)
center_neighbour_cells = fine_grid.neighbours(center_cells)
# flatten cell_ids to axes ('all_neigbours', 'xy') with shape (252, 2)
center_neighbour_cells = center_neighbour_cells.ravel()
center_neighbour_shapes = fine_grid.to_shapely(center_neighbour_cells)

plot_polygons(center_shapes, fill=True, colors="limegreen", alpha=0.6)
plot_polygons(center_neighbour_shapes, fill=True, colors="sandybrown", alpha=0.6)
plot_polygons(fine_shapes, fill=False, linewidth=1, colors="purple")
plot_polygons(coarse_shapes, fill=False, linewidth=2, colors="orange")

plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.show()

# %%
#
# Lastly, let's find coordinates of the vertices of the orange cells.
# We then find what purple cells are a these coordinates and color them as well.

vertices = coarse_grid.cell_corners(coarse_cell_ids).reshape((-1, 2))
vertices_cells = fine_grid.cell_at_point(vertices)
vertices_cells = vertices_cells.unique()  # drop duplicate ids
vertices_shapes = fine_grid.to_shapely(vertices_cells)

plot_polygons(center_shapes, fill=True, colors="limegreen", alpha=0.6)
plot_polygons(center_neighbour_shapes, fill=True, colors="sandybrown", alpha=0.6)
plot_polygons(vertices_shapes, fill=True, colors="darkcyan", alpha=0.6)
plot_polygons(fine_shapes, fill=False, linewidth=1, colors="purple")
plot_polygons(coarse_shapes, fill=False, linewidth=2, colors="orange")

plt.xlim(-4.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.show()

# %%
#
# Now the purple grid is colored, based on it's relation to the orange grid.
