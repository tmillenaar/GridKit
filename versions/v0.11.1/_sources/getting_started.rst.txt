.. _getting_started:

Getting Started
================

Quick example
-------------
As a quick demonstration a grid and two points are created.
Then the grid cells that cover those points will be obtained
and finally the polygon description of these cells will be obtained.

.. code-block:: python

   >>> from gridkit import HexGrid
   >>> grid = HexGrid(size=5)
   >>> points = [[2, 7], [-6, 2]]
   >>> cell_ids = grid.cell_at_point(points)
   >>> geoms = grid.to_shapely(cell_ids)
   >>> geoms
   array([<POLYGON ((2.5 5.052, 2.5 7.939, 0 9.382, -2.5 7.939, -2.5 5.052, 0 3.608, 2...>,
         <POLYGON ((-5 0.722, -5 3.608, -7.5 5.052, -10 3.608, -10 0.722, -7.5 -0.722...>],
         dtype=object)

..

More elaborate examples can be found in the :ref:`Example gallery <example_gallery>`

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
Restrictions therefore keep clarity around which features make sense for GridKit, and which are best left to to other packages.

Most notably, there are many software solutions out there that focuss on rasters, or bounded rectangular grids (some of these are mentioned below).
Often, these software packages focuss around operations on raster bands.
GridKit, in contrast, is build around infinite grids and works with indices reprecenting individual grid cells.
Here, the grid cell is the fundamental unit instead of the raster band.

Some features will overlap, such as cropping and resampling.
Other features you may have come to expect from raster oriented software might be missing in GridKit.


**GridKit is not:**

 * GDAL
    GDAL is a versitile toolbox of everything geospatial, for both vector and raster operations.
    It's raster operations are specifically tuned to work with bounded geospatial rectangular grids, often with multiple data bands.
    The great advantage of GDAl is that the amount and veriety raster operations that is supported out of the box, is unmatched.
    In GridKit, no attempt will be made to match GDAL in this regard.
    However, while python bindings for GDAL are available, the use is not very Pythonic.
    One example of this is that for every raster operation, GDAL expects a path to a raster file as input and writes a new raster file as output.
    While the rasters can be written to memory as a workaround to save the IO, this is not a style of programming most python developers are comfortable with.
    In GridKit, the grids are python classes which allows for a more pythonic style of programming and easier chaining of operations.
 * Rasterio
    Rasterio, as the name suggest, is a package that focusses on the reading and writing of raster files.
    Besides reading and writing, Rasterio provides functionality for warping, cropping and interpolating data.
    Some of this functionality overlaps with that of GridKit.
    Most notably rasterio uses affine transforms to represent the position of a raster, whereas gridkit uses the bounding box.
    GridKit utilizes Rasterio for the reading and writing of geotiffs.
 * RioXarray
    RioXarray is a package that acts, not surprisingly, as an interface between Rasterio (rio) and Xarray.
    This is a powerful combination that converts raster files to Xarray objects.
    This is a more pythonic way of working with data than the methods mentioned above,
    but is still fundamentally oriented around the bounded rectangular datatype, with the option for multiple bands.
    If your aim is primarily to work with raster bands, give RioXarray a look.

**Conclusion, when to use GridKit**

Gridkit is not nesecarily build to work with multiband rasters.
If this is the kind of data you are most likely to use, you may consider one of the packages mentined in the previous section.

GridKit's cell-based approach can be beneficial if you want to:
 
 * relate rasters of different spatial extends
 * relate vector data to grid cells
 * exercise fine grained control over particular cells and their properties, such as obtaining a cell's corner locations or finding it's neighbouring cells


Infinite grids, the fundamentals
--------------------------------

A grid is little more than a surface that is subdivided into cells.
We chan choose how we subdivide this surface by choosing the type shape and size of the cells, as well as an origin (a starting point).
Once we have chose these parameters, we can say something about each cell, like it's position relative to the origin or relative other cells.
Since this plane has no defined end, the cells can keep on tiling to infinity. Hence the term '*infinite grid*'.

A grid that is defined by it's cell shape, size and the grid's origin does not need to store a lot of data in memory. 
All other information regarding grid cells of interest (such as their area, center location, corner locations or neighbours) can be calculated on demand.

Bounded vs infinite grids, an example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Out of practicality, most data that can be thought of as a grid is bounded in the real world.
A computer screen can be thought of as a rectangular grid of say 1920x1080 pixels.
This is clearly a bounded grid, where pixel (2000, 500) has little meaning, for it is out of the bounds of the screen and hence does not reflect a real-world pixel.
However, when two screens are placed next to each other and attached to the same device, 
the screens can be combined and their shared pixel space can thought of as one of 3840x1080 pixels.
In this scenario pixel (2000, 500) refers to a pixel on the second screen.
Conceptually, these two screens then sare the same grid.
This is an illustration of bounded data living on an infinte grid.

Cell indices, what are they?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In GridKit, one cell in a grid is set to be cell (0,0), the starting cell.
From this cell you can move one or more cells to the left, right, down or up.
The first number indicates how many cells you moved in the x-direction (left or right).
The second number indicates how many cells you moved in the y-direction (up or down).
The directions up and right are defined to be the positive directins.
Thus, a cell with index (-2,3) is two cells left of cell (0,0), and three cells above it.
Similarly, cell (-1,3) is one cell right of cell (-2,3), but on the same row.

Based on the cell size, all kinds of information can be calculated.
For example, a rectangular grid where the cell size in x-direciton is 1, and in y-direction is 2, the center of the first cell (0,0) is at location *x=0.5*, *y=1*).


Offsets
^^^^^^^
Two grids with the same cell shape and the same cell dimentions, are not nesecarily the same.
If their origin differs, the grids are shifted with respect to each other.
This shift can be less then the size of a cell.
If this is the case, the grid lines of the two grids do not nicely overlap.
The distance between a grid line of one grid and the grid line of another grid is considered to be the *offset* between the two grids.
If this is the case, the grids are not *aligned*.
When one grid is offset by exactly one cell (or a multiple), the gridlines of the two grids still overlap.
Such grids are considered to be *aligned*.
In fact, in GridKit there is no distinction between these two grids, they are considered to be one and the same.
To refer back to the screen example, the two screens that are attached to the same device conceptually occupy the same pixel-space.
Hence they live on the same grid, with the same origin.
In this conceptualization, the second screen's first pixel would start at index (1921,0).

