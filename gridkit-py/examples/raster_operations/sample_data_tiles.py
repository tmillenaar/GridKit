import numpy
from shapely import Polygon

from gridkit import DataTile, RectGrid, Tile
from gridkit.doc_utils import plot_polygons

# Create a new grid
# Use float as data type because NaNs nicely show up as transparent in matplotlib
data = numpy.arange(25 * 25, dtype=float).reshape(25, 25)
grid = RectGrid(size=1, rotation=7.2)
tile = Tile(grid, (-20, -15), data.shape[1], data.shape[0])
data_tile = DataTile(
    tile,
    data,
    nodata_value=numpy.nan,
)


upscaled_grid = RectGrid(size=0.1, rotation=-16.3)
upscaled_tile = Tile(upscaled_grid, (-100, 35), 60, 60)
upscaled_data_tile = upscaled_tile.sample_nearest_from(data_tile)

downscaled_grid = RectGrid(size=1.3, rotation=32)
downscaled_tile = Tile(downscaled_grid, (-1, 0), 12, 12)
downscaled_data_tile = downscaled_tile.sample_nearest_from(data_tile)

# Plot the grid in it's native CRS
import matplotlib.pyplot as plt

plot_polygons(
    data_tile.to_shapely(),
    colors=data_tile.to_numpy().ravel(),
    cmap="viridis",
    add_colorbar=True,
    set_zoom_to_bounds=False,
)

plot_polygons(
    downscaled_data_tile.to_shapely(),
    colors=downscaled_data_tile.to_numpy().ravel(),
    cmap="viridis",
    add_colorbar=True,
    set_zoom_to_bounds=False,
)
plot_polygons(  # Plot 2, show tile bounds
    Polygon(downscaled_data_tile.corners()),
    colors="red",
    fill=False,
)

plot_polygons(
    upscaled_data_tile.to_shapely(),
    colors=upscaled_data_tile.to_numpy().ravel(),
    cmap="viridis",
    add_colorbar=True,
    set_zoom_to_bounds=False,
)
plot_polygons(  # Plot 2, show tile bounds
    Polygon(upscaled_data_tile.corners()),
    colors="red",
    fill=False,
)

plt.xlim(-25, 10)
plt.ylim(-20, 15)
plt.show()
