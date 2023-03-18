.. _getting_started:

Getting Started
================

What is GridKit?
----------------

GridKit is an open source Python library that provides a high-level API to work with regularly tiled grids.

**Core features:**

 * tessellation of an infinite surface [#]_
 * working with geospatial data

    * geocoded grids
    * resampling
    * coordinate transformations (akin to geopanda's `to_crs`)

 * bridging the gap between raster and vector data

    * gridcell-vector intersections
    * sampling
    * interpolation
    * aggregation statistics (planned)

 * grids of various shapes

    * rectangular
    * hexahonal (planned)
    * triangular (planned)


.. rubric:: Footnotes

.. [#] In practice the "*infinite surface*" is bounded by machine precision. No grid cell can be referenced of which the index does not fit in a 64 byte integer.


What GridKit is not
-------------------

Many common operations are done on grids or rasters.
Restrictions therefore keep clarity around what features we want to add, and what we want to leave to other packages.

Most notably, there are many software solutions out there that focuss on Rasters, or bounded rectangular grids (some of these are mentioned below).
Often, these software packages focuss around operations on raster bands.
Instead, GridKit is build around infinite grids and works with indices reprecenting individual grid cells.
Here, the grid cell is the fundamental unit instead of the Raster band.

Some features will overlap, such as cropping and resampling.
Other features that are prevelent in these types of software might be missing in GridKit, such as stacking raster bands or slope and aspect calculations.



**GridKit is not:**

 * GDAL
    GDAL is a versitile toolbox of everything geospatial, for both vector and raster operations.
    It's raster operations are specifically tuned to work with bounded geospatial rectangular grids, often with multiple data bands.
    The great advantage of GDAl is that the amount and veriety raster operations that is supported out of the box, is unmatched.
    In GridKit, no attempt will be made to match GDAL in this regard.
    However, while python bindings for GDAL are available, the use is not very Pythonic.
    One example of this is that for every raster operation, GDAl expects a path to a raster file as input and writes a new raster file as output.
    While the rasters can be written to memory as a workaround to save the IO, this is not a style of programming most python developers are comfortable with.
    In GridKit, the grids are python classes which allows for a more pythonic style of programming and easier chaining of operations.
 * Rasterio
    Rasterio, as the name suggest, is a package that focusses on the reading and writing of raster files.
    Besides reading and writing, Rasterio provides functionality for warping, cropping and interpolating data.
    Some of this functionality overlaps with that of GridKit.
    Most notably rasterio uses affine transforms to represent the position of a raster, whereas gridkit uses the bounding box.
    We have no intention of re-inventing the wheel, and happily use Rasterio for the reading and writing of geotiffs.
 * RioXarray
 * Scikit-image (Skimage)



Basic examples
--------------

Raster sampling
^^^^^^^^^^^^^^^

Aggregation
^^^^^^^^^^^

Interpolation
^^^^^^^^^^^^^



Grid Indices
------------


Offsets
-------

