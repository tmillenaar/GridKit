"""
.. _example rotated animation:

Rotation animation
==================

Animated visualization of grid rotation

Introduction
------------

Grids can be rotated around the origin.

Some more info on rotations is in example :ref:`rotated_grids.py <example rotated grids>`.

In this example, an animation is created where the grid rotates around the origin.
The grid is centered around zero and rotated a little more every frame.
Since hexagonal grids have a six-fold symmetry, we only have to animate from
0-60 and loop it. It will look like a continuous rotation.
Every frame, a new grid is created that is centered around 0,0 and rotated 
two degrees more than the previous frame.

"""

# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy
from matplotlib.animation import FuncAnimation

from gridkit import HexGrid
from gridkit.doc_utils import plot_polygons

# Initialize
grid = HexGrid(size=1, rotation=0)
center_id = [-1, -1]
ids = grid.neighbours(center_id, depth=4)
distance = numpy.linalg.norm(grid.centroid(ids) - grid.centroid(center_id), axis=1)


def update_frame(rotation):
    ax.clear()
    ax.scatter(0, 0)
    rotated_grid = HexGrid(size=1, rotation=rotation, offset=(0, grid.dy / 2))
    geoms = rotated_grid.to_shapely(ids, as_multipolygon=True)
    im = plot_polygons(geoms.geoms, colors=distance, fill=True, ax=ax)
    ax.set_title(f"Rotation: {rotation} degrees")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)


# Create animation
fig, ax = plt.subplots()
ax.set_aspect("equal")
anim = FuncAnimation(
    fig, update_frame, frames=range(0, 60, 2), repeat=True, interval=50
)
