"""
.. _example resampling:

Resampling
==========

Resample data from one grid onto another.

Introduction
------------

Resampling is a general term for creating a new dataset based on an existing one.
This new dataset can have a different resolution, cell shape or Coordinate Reference System (CRS).

In GridKit, resampling is generally done by defining a new grid and resampling onto that new grid.
In this example a DEM is read from a geotiff file.
The read function returns a BoundedRectGrid object that will serve as the initial state.
Several grids with different cell shapes are then defined on which the DEM data can be resampled.
"""

# sphinx_gallery_thumbnail_number = -1

from gridkit import HexGrid, RectGrid, read_raster

dem = read_raster(
    "../../tests/data/alps_dem.tiff", bounds=(28300, 167300, 28700, 167700)
)
print("Original resolution in (dx, dy):", dem.cellsize)

# %%
#
# .. Warning ::
#
#    Be mindful of the projection of your dataset.
#    This dataset is in ETRS89, which is defined in degrees, which can introduce a bias.
#    Grids defined in degrees generally do not have consistent spacing in x and y direction,
#    meaning that one degree north is not the same distance as one degree south.
#    It is generally recommended to work with grids defined in a local CRS that is not
#    defined in degrees but either meters or feet.
#
#
# Resampling onto the a grid
# --------------------------
#
# Now we have read the DEM, we can define a new grid and resample onto it
# Let's define a coarser grid, effectively downsampling the data.
# The reason for this is that we can better compare grids if we can actually distinguish the cells.
# Let's define a rectangular grid with a cell size of 10x10 degrees.
# I am calling it rectdem for later on I will define hexagonal ones as well.
# To make sure it worked we can print out the cellsize after resampling
rectdem = dem.resample(RectGrid(dx=10, dy=10, crs=dem.crs), method="bilinear")
print("Downsampled resolution in (dx, dy):", rectdem.cellsize)

# %%
#
# Let's plot the data to see what it looks like after downsampling

import matplotlib.pyplot as plt

plt.imshow(rectdem, extent=rectdem.mpl_extent, cmap="terrain")
plt.show()


# %%
#
# Now let's do the same, but on hexagonal grids.
# There are two flavours, "pointy" and "flat" hexagonal grids.
# Let's show both so we can compare them both to each other and to the downsampled rectangular grid.
# Hexagonal cells are smaller than square cells when given the same width,
# so to make a more fair visual comparisson let's use a slightly larger cell width.
# This way we will have a roughly equal number of cells covering the area.

hexdem_flat = dem.resample(
    HexGrid(size=11, shape="flat", crs=dem.crs), method="bilinear"
)
hexdem_pointy = dem.resample(
    HexGrid(size=11, shape="pointy", crs=dem.crs), method="bilinear"
)


# %%
#
# Now let's create two new figures and populate these with the colored shapes of the two downsampled hexagon grids.
# Since there is no 'imshow' equivalent for hexagons in matplotlib, we use our own :func:`gridkit.doc_utils.plot_polygons`
# function, which is less performant but works on generalized shapes.
#
from gridkit.doc_utils import plot_polygons

# define two new figures
fig_flat, ax_flat = plt.subplots()
fig_pointy, ax_pointy = plt.subplots()

plot_polygons(
    hexdem_flat.to_shapely(),
    colors=hexdem_flat.data.ravel(),
    cmap="terrain",
    ax=ax_flat,
)
plot_polygons(
    hexdem_pointy.to_shapely(),
    colors=hexdem_pointy.data.ravel(),
    cmap="terrain",
    ax=ax_pointy,
)

# Format the plot
for hexdem, ax in zip((hexdem_flat, hexdem_pointy), (ax_flat, ax_pointy)):
    ax.set_xlim(hexdem.bounds[0], hexdem.bounds[2])
    ax.set_ylim(hexdem.bounds[1], hexdem.bounds[3])
    ax.set_aspect("equal")
plt.show()

# %%
#
# This example can of course also be used to upsample your data.
#
# .. Note ::
#
#    The three images in this example look different, but they are all equally 'correct'.
#    The visual difference results from the difference in positioning of the cells.
#    Generally hexagon grids better represent rounded features,
#    whereas rectangular grids are generally easier to work with and are more widespread.
