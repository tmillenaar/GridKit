"""
.. _example hexagonal automata:

Cellular automata: Hexagonal automata
=====================================


This script is based on the example shown in :ref:`game_of_life.py <example game of life>`, but here a Hexagonal grid is used with slightly differnt rules.
Here the 6 neighbours around a cell are used. The rules are:

    1. If a cell is alive, it stays alive if two and only two of it's neighbours are alive.
    2. If a cell is dead, it becomes alive if one and only one of it's neighbours is alive.

This makes remarkably beautiful patterns on larger scales.
A cell in a line has two neighbours, so it remains alive.
In this configuration, straight lines are stable and corners are not.
Because of this, the pattern can only grow form corners and not from a line. This results in remarkable patterns when played out over longer sessions.

Also pretty cool, notice that on frame 1 we have a ring that forms a hexagon and again on frames 3, 7 and 15.
Do you see the pattern?

I encourage you to play with this, though matplotlib as a backend is not very performant when animating many shapes.

"""

# sphinx_gallery_thumbnail_number = -1

import matplotlib.pyplot as plt
import numpy
from matplotlib.animation import FuncAnimation

from gridkit import BoundedHexGrid
from gridkit.doc_utils import plot_polygons

# Initialize
shape = (33, 33)
data = numpy.zeros(shape)
grid = BoundedHexGrid(data=data)
geoms = grid.to_shapely()  # Used for plotting

neighbours = grid.neighbours(grid.indices)
neighbours = numpy.stack(grid.grid_id_to_numpy_id(neighbours.ravel())).T.reshape(
    (*shape, 6, 2)
)

# Animate
fig, ax = plt.subplots()
ax.set_aspect("equal")


def init():
    data.ravel()[:] = 0
    data[16, 16] = 1


def update_frame(frame):
    ax.clear()
    plot_polygons(geoms, colors=data.ravel(), cmap="gray_r", fill=True, ax=ax)
    ax.set_title(frame)
    ax.set_axis_off()
    nb = data[
        neighbours[1 : shape[0] - 1, 1 : shape[1] - 1, :, 0],
        neighbours[1 : shape[0] - 1, 1 : shape[1] - 1, :, 1],
    ]
    nr_nb_live = nb.sum(axis=-1)
    is_live = data[1 : shape[0] - 1, 1 : shape[1] - 1] == 1
    data[1 : shape[0] - 1, 1 : shape[1] - 1] = (
        0 + is_live * (nr_nb_live == 2) + ~is_live * (nr_nb_live == 1)
    )


anim = FuncAnimation(
    fig, update_frame, init_func=init, frames=range(0, 16), repeat=True, interval=300
)
plt.show()
