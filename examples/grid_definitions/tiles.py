"""
.. _example tile:

Grid Tiles
==========

Convenient way to refer to a collection of cells in a grid.

Introduction
------------

The Tile object is designed to be an easy way to refer to a collection of cells on a grid.
In order to create one, we supply it the grid is is associated with,
the starting cell id in the bottom left and the number of cells in x and in y direction.

Let's plot the cells contained in a tile and show the tile outline.

"""

# sphinx_gallery_thumbnail_number = -2

from matplotlib import pyplot as plt
from shapely.geometry import Polygon

from gridkit import HexGrid, Tile
from gridkit.doc_utils import plot_polygons

grid = HexGrid(size=1, offset=(0.3, 0.5), rotation=18)

tile = Tile(
    grid=grid,
    start_id=[5, 5],
    nx=8,
    ny=5,
)

geoms = grid.to_shapely(tile.indices, as_multipolygon=True)

bbox = Polygon(tile.corners())
plot_polygons(geoms, fill=True, linewidth=2, alpha=0.3)
plot_polygons(bbox, fill=False, linewidth=2, colors="red")
plt.show()

# %%
#
# Note that the outline of the tile cuts through the cells on the left and right,
# and note how the left side is missing coverage of some half-cells and on the right
# some cells extend beyond the border. This is deliberate and is designed in a way where the
# two neighboring tiles don't share any cells.
# To demonstrate this, let's plot several tiles next to each other:
#

tiles = [
    Tile(grid, [-3, 5], nx=8, ny=5),  # 0
    Tile(grid, [5, 5], nx=8, ny=5),  # 1
    Tile(grid, [5 + 8, 5], nx=8, ny=5),  # 2
    Tile(grid, [-1, 0], nx=10, ny=5),  # 3
    Tile(grid, [9, 0], nx=10, ny=5),  # 4
]

bboxes = [Polygon(t.corners()) for t in tiles]
for i, (tile, color) in enumerate(
    zip(tiles, ["peru", "deepskyblue", "orange", "teal", "purple"])
):
    geoms = tile.grid.to_shapely(
        tile.indices, as_multipolygon=True
    )  # Note how I call tile.grid.to_shapely, read on for explenation
    plot_polygons(geoms, fill=True, linewidth=2, alpha=0.3, colors=color)
    center = tile.corners().mean(axis=0)
    plt.text(*center, i, size=30)
plot_polygons(bboxes, fill=False, linewidth=2, colors="red")
plt.show()

# %%
#
# This is set up such that multiple tiles can cover a larger plane together which
# is usefull for distributed computing, though GridKit does not handle any of this
# distribution itself. It is up to the user to use this tileability together with a tool
# such as Dask or Multiporcessing.
# This of course only works well if all tiles are based off the same grid.
#
# Note that every Tile has it's own copy of the grid. If we mutate the original grid,
# the .grid associated with each tile remains unaffected.
# Be mindful of what grid you refer to. It is easy to make the mistake where
# the original grid is modified and the indices of the tile are used with the
# ``grid`` variable instead of tile.grid.
# Here I will show what it might look like if you use the tile indices on a different grid:
#

grid.rotation -= 5

bboxes = [Polygon(t.corners()) for t in tiles]
for i, (tile, color) in enumerate(
    zip(tiles, ["peru", "deepskyblue", "orange", "teal", "purple"])
):
    geoms = grid.to_shapely(
        tile.indices, as_multipolygon=True
    )  # Note how I call grid.to_shapely and not tile.grid.to_shapely!
    plot_polygons(geoms, fill=True, linewidth=2, alpha=0.3, colors=color)
    center = tile.corners().mean(axis=0)
    plt.text(*center, i, size=30)
plot_polygons(bboxes, fill=False, linewidth=2, colors="red")
plt.show()
