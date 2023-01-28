"""
NDVI 
====

Operation using two raster bands

In this example we read two Sentinel 2 raster bands and combine them to calculate the Normalized Difference Vegetation Index (NDVI).

After reading, the dtype is increased from int16 to int32 because the addition will result in values that are too large for int16.

We don't have to align these bands, for they are from the same Sentinel 2 acquisition.
In many cases it will be nesecary to resample one band on the other.

Also, a numpy warning is raised because we devide by zero for some cells.
These turn into NaN-values.
This is remedied in the script by replacing each NaN-value with 0 afterwards.

"""
from gridkit.io import read_geotiff
import matplotlib.pyplot as plt
import numpy

# Read in the bands required to determine the NDVI
band_4 = read_geotiff("../../tests/data/2022-07-08-00:00_2022-07-08-23:59_Sentinel-2_L2A_B04_(Raw).tiff").astype("int32")
band_8 = read_geotiff("../../tests/data/2022-07-08-00:00_2022-07-08-23:59_Sentinel-2_L2A_B08_(Raw).tiff").astype("int32")

# Determine the NDVI
ndvi = (band_8 - band_4)/(band_8 + band_4)
ndvi.data[~numpy.isfinite(ndvi)] = 0 # fill NaN values resulting from a division by 0

# Plot the result
mpl_extent = (band_4.bounds[0], band_4.bounds[2], band_4.bounds[1], band_4.bounds[3])
fig, ax = plt.subplots(1)
im_ndvi = ax.imshow(ndvi, cmap="RdYlGn", extent=mpl_extent, vmin=-1, vmax=1)
fig.colorbar(im_ndvi, ax=ax, fraction=0.022, pad=0.01)
ax.set_xlabel("lon")
ax.set_ylabel("lat")
ax.set_title(f"NDVI of scene in the alps \n EPSG:{ndvi.crs.to_epsg()}")

plt.show()
