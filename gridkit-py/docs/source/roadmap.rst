Roadmap
=======

This page provides an overview of the major themes in GridKit's development.
Each of these topics fits in the vision of what the package is meant to become.
These items are estimated to requires a significant amount of effort to implement.

This page is meant to give a rough overview.
For these items, there is no time estimation and there are no guarantees a topic on this page will make it in the package.
Unexpected issues may prevent the adoption of this feature.

A topic not being on this page, naturally, does not exclude it from possible adoption.
Smaller topics are not on this page and recommendations for new features can be made in the `issue tracker <https://github.com/tmillenaar/GridKit/issues>`_.

On the roadmap
-----------------------

Hexagonal and triangular cells
""""""""""""""""""""""""""""""
Rectangular grids are far from the only types of grids that can tile the plane.
However, there are only three types of primitive shapes that can tile the plane without the need of a second shape to make it fit.
These shapes are triangles, rectangles and hexagons.
In GridKit, these three shapes should be supported.
Combinations of shapes that tile the plane come in an infinite veriety.
While these can yield interesting patterns, for practical purposes GridKit will only support single shapes that tile the plane.


Sparse data grids
"""""""""""""""""
The most common way to store rasters is as a 2D-array.
This is also the way the data of the grid is stored in GridKit, through the :class:`~gridkit.bounded_grid.BoundedGrid` mixin.
This approach makes sense for continuous datasets, possibly with a nodata value here and there.
However, this approach does not lend itself well for sparse data sets.
To store sparse data, it is a lot more memory efficient to store a list of cell IDs along with a list of values.
Any cell not in this list will be considered to contain no data.
A new `SparseGrid` can be created alongside the :class:`~gridkit.bounded_grid.BoundedGrid` to store sparse data.
Some operations will only work on bounded data, like convolutions for example.
Other operations are more suitable for sparse data, like aggregation and possibly A* pathfinding.
Some operations (such as aggregation) might get the keyword argument '`sparse`', specifying whether a sparse or bounded grid is to be returned. 


Aggregation
"""""""""""
There are several ways of converting a dataset of points to a grid.
Currently, only interpolation is supported.
Another method is aggregation, where all points contained within the same cell are `aggregated`, or combined.
There are of course multiple ways by which these points can be combined.
The value of the operation is then stored on that grid cell.

Common operations when aggregating are:

 * count (number of points in cell)
 * mean
 * standard deviation
 * `n` th percentile


FFT and convolutions
""""""""""""""""""""
The most common image processing operations use Fast Fourier Transforms (FFT) and/or convolutions.
In fact, a convolution `can be done through the use of an FFT <https://en.wikipedia.org/wiki/Convolution_theorem>`_.
While GridKit is not trying to be an image processing library, these operations can be very powerful.
For rectangular grids, `numpy.fft.fft2 <https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html>`_ and 
`scipy.fft.fft2 <https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft2.html>`_ can be used to determine the FFT.
Hence there is no real need to implement this directly into gridkit.
However, when hexagonal grids are implemented, these approaches cannot be used.
Algorithms for the `Hexagonal Fast Fourier Transform (HFFT) <https://en.wikipedia.org/wiki/Hexagonal_fast_Fourier_transform>`_ have been devised.
This can be implemented in gridkit to allow for basic image processing on hexagonal grids.
Because of the hexagonal grids better represend curves, and each neighbour is equidistant, image processing on hexagonal data can be beneficial.


Pathfinding
"""""""""""
Pathfinding is a common usecase for grids and is not currently part of GridKit.
Adding a weighted A* pathfinding algorithm only makes sense.
Other pathfinding algorithm could be added too, but A* would be a good start.
It is a general algorithm that could be performed on grids of different shapes.
The obstacles could be defined either as bounded or sparse data, see the section on Sparse Data Grids.



Not on the Roadmap
------------------

Some operations make sense for grids, but are not on the roadmap for GridKit.
Sometimes this is simply because it has not been considered yet,
but sometimes this is because it does not fit with the project's vision, dispite it being a common operation to perform on grids.
Some features that fall in the latter category are mentioned here, in an attempt to explain why this is not being considered at the moment.

If you think one of these features should, in fact, be considered for the roadmap, you can try to make a compelling case in the `issue tracker <https://github.com/tmillenaar/GridKit/issues>`_

Storing data on edges and vertices
""""""""""""""""""""""""""""""""""
There are many types of operations that could benefit from storing the values not in the cell center, but on the vertices or edges instead.
Some solutions utilize some or all of these in combination.
An example would be storing temperature values in a cell and flux at the edges.

While there is clear value in this feature, it is a large effort to develop this and, more importantly, a lot harder to maintain.
Every new feature would have to work on cells, edges and vertices, or specifically exclude a data type and yield the appropriate error.


Irregualr grids
"""""""""""""""
Not all grids are regular grids. Irregular grids often are grids where the cell shapes vary in shape or size.
This approach, does not match with the architecture of GridKit's implementation.
Implementations involving irregualr grids need to store the information of each cell, such as the location of the vertices.
In GridKit, the location of a cell is inferred, not stored.
This is possible because the cell distribution of a regular grid is predicatable.

