.. _release_notes:

Release notes
=============

This is the changelog listing the changes of GridKit over time.

Version 1.0.0 (nocheckin, add date)
-----------------------------------
Features
 - Introduce DataTile class to replace BoundedGrid because of their ability to represent rotated Tiles
 - Replace usages to bounded grids in examples with data tiles
 - :func:`.count`, :func:`.sum` and :func:`.mean` now also apply to data tiles
 - TriGrid orign moved and size definition halved to better match the RectGrid and HexGrid, see https://github.com/tmillenaar/GridKit/issues/94
 - Provide a flat and pointy version of TriGrid
 - Fix offset having to be specified in (y,x) if for 'flat' HexGrids, now (x,y)
 - Add set_zoom_to_bounds option to :func:`.doc_utils.plot_polygons` to allow to turn it off

Fixes
 - adjust example :ref:`flower_of_life.py <example flower of life>` to work with shapely 2.1.0
 - properly raise error if :meth:`.BaseGrid.align_bounds` or :meth:`.BaseGrid.are_bounds_aligned` is called on a rotated grid.
   Before this just gave unreliable results.

Deprecations
 - Remove as_multipolygon argument from :meth:`.BaseGrid.to_shapely()` and always return a Shapely object.
   The user can call `.geoms` on the Shapely object to get an iterable.
   Reason: the to_shapely() function always created a Shapely object and then called .geoms on
   it if as_multipolygon was set to False. This was implicit and one might assume that
   this saves the creation of a Shapely object hence being more performant. This was removed
   to make this more explicit since now the user has to call .geoms for themselves if they want
   an iterable.
 - Remove 'filled' keyword argument from :func:`gridkit.doc_utils.plot_polygons` in favor of 'fill'

Misc
 - Drop support for python 3.9
 - Moved :meth:`.BaseGrid.anchor` to Rust
 - Remove DeprecationWarning for flat HexGrids, turns out they are useful in data tiles and bounded grid contexts

Version 0.14.1 (December 25, 2024)
----------------------------------
Features
 - Add arguments ``location`` and ``adjust_rotation`` to :meth:`.TriGrid.to_crs`, :meth:`.RectGrid.to_crs` and :meth:`.HexGrid.to_crs` for more accurate conversions.

Misc
 - Build using python3.9 because for 3.8 the geopandas/fiona documentation dependency installs incompatible versions

Version 0.14.0 (August 11, 2024)
--------------------------------
Featrures
 - A new :class:`.Tile` class that references a set of cells and has some convenience methods
   that describe the tile, such as :attr:`.Tile.indices` and :attr:`.Tile.corners`.
   This class is takes a similar role to the :meth:`.BaseGrid.cells_in_bounds` method,
   but is able to work with rotated grids. The intent is that in the long run a DataTile
   will replace the BoundedGrid for this reason.

Documentation
 - Add example :ref:`tiles.py <example tiles>` which explains the usage of the new :class:`.Tile` class.
 - Use more neighbours in example :ref:`flower_of_life.py <example flower of life>` since the final flower
   was missing some circles in the bottom left.

Misc
 - Rename the PyO3 classes PyTriGrid, PyRectGrid and PyHexGrid to PyO3TriGrid, PyO3RectGrid and PyO3HexGrid, respectively.
   This is done to avoid confusion. From the Rust perspective these represent Python classes but from the Python perspective
   these represent Rust classes. PyO3 seems to be less ambiguous for it makes sense from both perspectives.

Version 0.13.0 (July 10, 2024)
------------------------------
Features
 - Option to initialize grid using ``side_length`` instead of ``size`` or ``area``

Fixes
 - Comparisson operators for :class:`.GridIndex` now also work when comparing to non-grid index classes.
   For example ``GridIndex([1,2]) == (1,2)`` will result in ``True``.
   By extension, ``(1,2) in GridIndex([[1,2], [0,0]])`` now also works as expected.

Documentation
 - Add example :ref:`flower_of_life.py <example flower of life>`

Version 0.12.1 (Jun 17, 2024)
-----------------------------

Misc
 - Make compatible with numpy v2.0.0 while remaining compatible with earlier versions

Version 0.12.0 (May 10, 2024)
-----------------------------

Featrures
 - Create a new grid with a smaller gridsize that fits perfectily inside the orignal grid using :meth:`.TriGrid.subdivide`, :meth:`.RectGrid.subdivide` or :meth:`.HexGrid.subdivide`

Documentation
 - Add tip to :ref:`triangles_in_hexes.py <example triangles in hexes>` hinting to the use of the new ``subdivide`` and ``anchor`` methods.

Version 0.11.1 (June 01, 2024)
------------------------------

Features
 - Add method ``anchor`` to Bounded Grids (grids with data) that resamples the data after shifting, see :meth:`.BoundedTriGrid.anchor`, :meth:`.BoundedRectGrid.anchor` and :meth:`.BoundedHexGrid.anchor`
 - Shift nearby corner to specified location using ``cell_element="corner"`` in :meth:`.BaseGrid.anchor` and their bounded equavalents mentioned above
 - Add an easy method to access all important paramers defining the grid: :meth:`.BaseGrid.definition`

Version 0.11.0 (May 29, 2024)
------------------------------
.. _release notes v0_11_0:


Features
 - Easier shifting of grids using :meth:`.BaseGrid.anchor`

Fixes
 - Fix issue in HexGrid where offsets were incorrectly applied when the supplied offsets were not between 0:cell size
 - Fix 'flat' HexGrids rotating in the other direction
 - Fix error regarding the datatype when supplying the offset for a TriGrid as a non-tuple iterable such as a list or numpy array
 - Fix :meth:`.HexGrid.cell_at_point` not properly taking x-offset into account
 - Fix issue where the offset would flip for 'flat' HexGrids when using the offset setter but not when calling `grid.update(offset=new_offset)`

Documentation
 - Simplify centering of grids in examples :ref:`selecting_cells.py <example selecting cells>`, :ref:`rotation_animation.py <example rotated animation>` and :ref:`rotation_animation.py <example hexagon grids>`

Deprecations
 - 'flat' ``shape`` for HexGrid will be deprecated in favor of ``rotation`` in v1.0.0. A warning will be raised on class initiation.

Known Issues
 - The implementation of 'flat' HexGrids is done by swapping the x and y axes compared to a 'pointy' grid.
   With this release, several issues related to the offset were fixed, but this implementation aspect now leaks into the offset.
   This means the user might specify an offset of (0,1) and expect a shift of 1 in the y-axis but the shift occurs in the x-axis.
   Since 'flat' HexGrids will be deprecated in release v1.0.0, fixing this is not worth the effort, meaning this leaky abstraction will be deliberately ignored.

Version 0.10.0 (April 21, 2024)
-------------------------------
Features
 - Add :meth:`.GridIndex.sort`
 - Allow plotting of RGB(A) values in :func:`.doc_utils.plot_polygons`
 - Improve performance of :func:`.doc_utils.plot_polygons`
 - Improve performance of initializing a new :class:`.GridIndex` if the supplied indices are already in an appropriate numpy integer ndarray.

Fixes
 - Fix incorrect :meth:`.HexGrid.relative_neighbours` and by extension :meth:`.BaseGrid.neighbours` for :class:`.HexGrid` when supplying multiple grid indices at a time

Documentation
 - Add example :ref:`2d_diff_hex_anim.py <example diffusion>`

Version 0.9.2 (April 03, 2024)
------------------------------
Features
 - Add new initialization argument ``area`` to :class:`.TriGrid`, :class:`.RectGrid` and :class:`.HexGrid` for specifying the cell area of the grid
 - Add new property :meth:`.BaseGrid.area`
 - Add ``shape`` to :class:`.RectGrid` to further unify the class API between the three grid types

Fixes
 - Prevent passing rotation argument to Bounded grids, which were not designed with rotation in mind because that breaks the tiling.

Documentation
 - Update example :ref:`resampling.py <example resampling>`
 - Improve docstrings about initialization of :class:`.TriGrid`, :class:`.RectGrid` and :class:`.HexGrid`

Version 0.9.1 (March 17, 2024)
------------------------------
Features
 - Change the following attributes using a setter: ``rotation``, ``offset`` and (``size`` for :class:`.TriGrid` and :class:`.HexGrid`) or (``dx`` and ``dy`` for :class:`.RectGrid`)
 - Add :meth:`.BaseGrid.cell_height` and :meth:`.BaseGrid.cell_width`
 - Add :meth:`.BaseGrid.update` method for easily making small changes to grid specs

Fixes
 - ``offset`` is now properly taken into account for :meth:`.HexGrid.cell_at_point`
 - Less restrictive offsets by limiting offset for all grids with :meth:`.BaseGrid.cell_height` and :meth:`.BaseGrid.cell_width` instead of dx and dy.
 - Rotation of "flat" :class:`.HexGrid` is no longer in the opposite direction (was clockwise)

Documentation
 - Add example :ref:`rotation_animation.py <example rotated animation>`.

Version 0.9.0 (March 10, 2024)
------------------------------
Features
 - Rotation for :class:`.TriGrid`, :class:`.RectGrid` and :class:`.HexGrid`
     - Note: not for the bounded versions

Fixes:
 - Fixed :meth:`.RectGrid.cells_near_point` returning incorrect cells for negative points

Documentation
 - Add example :ref:`rotated_grids.py <example rotated grids>`.

Version 0.8.0 (March 03, 2024)
------------------------------
Fixes
 - Return :class:`.GridIndex` from :meth:`.HexGrid.cells_near_point`
 - Align return shape of index :meth:`.RectGrid.cells_near_point` with those of :meth:`.TriGrid.cells_near_point` and :meth:`.HexGrid.cells_near_point` (!API change)
 - Allow multi-dimensional input and returns form method `cells_near_point` on the three grid types

Misc
 - Move the following methods to Rust:

     - :meth:`.RectGrid.cells_near_point`
     - :meth:`.HexGrid.cells_near_point`


Version 0.7.3 (February 25, 2024)
---------------------------------
Fixes
 - Properly handle negative offsets in Rust grid classes

Misc
 - Move the following methods to Rust:

     - :meth:`.RectGrid.centroid`
     - :meth:`.RectGrid.cell_at_point`
     - :meth:`.RectGrid.cell_corners`
     - :meth:`.HexGrid.centroid`
     - :meth:`.HexGrid.cell_at_point`
     - :meth:`.HexGrid.cell_corners`

    This is done in preparation of rotation of un-bounded grids and provides a minor speedup.

Version 0.7.2 (February 18, 2024)
---------------------------------
Features
 - Replace ``GridIndex._1d_view`` with :meth:`.GridIndex.index_1d`, which is an int64 instead of a custom data type.
 - Replace ``index._nd_view`` with :meth:`.GridIndex.from_index_1d`

Fixes
 - Remove redundant array allocation in :meth:`.TriGrid.cells_in_bounds`

Documentation
 - Remove ``dask_geopandas`` dependency in example :ref:`aggregate_dask.py <example aggregate_dask>`. Use :meth:`.GridIndex.index_1d` instead.
 - Use numpy array :meth:`.GridIndex.index_1d` in example :ref:`aggregate.py <example aggregate>` instead of a python list of :class:`.GridIndex` objects.

Version 0.7.1 (February 11, 2024)
---------------------------------
Fixes
 - Remove allocation of unused array

Documentation
 - Add building of Rust binary to the :ref:`contributing guide <contributing>`

Misc
 - Improve performance of :meth:`.BaseGrid.to_shapely`

Version 0.7.0 (February 04, 2024)
---------------------------------
Features
 - Add :class:`.BoundedTriGrid`
 - Improved performance of linear resampling for :class:`.BoundedHexGrid`
 - "inverse_distance" interpolation method for :meth:`.BoundedGrid.resample` and :meth:`.BoundedGrid.interpolate`

Fixes
 - Fixed incorrect cell returned for points in :meth:`.TriGrid.cell_at_point` near the cell edge
 - Allow for nd input in :meth:`.TriGrid.cell_at_point`


Version 0.6.0 (January 07, 2024)
--------------------------------
Features
 - Add :class:`.TriGrid` (Only base variant, BoundedTriGrid is yet to come)

Fixes
 - :meth:`.BaseGrid.to_shapely` now properly handles ND input
 - :meth:`.HexGrid.relative_neighbours` now properly handles ND input

Documentation
 - Add example :ref:`triangles_in_hexes.py <example triangles in hexes>`
 - :func:`.doc_utils.plot_polygons` used in examples now plots both lines and filled polygons

Misc
 - Add Rust bindings using the maturin package
 - Renamed the test rasters used in example :ref:`ndvi.py <example ndvi>` because Windows failed on special characters in the name
 - Put index as first argument instead of second in :meth:`.HexGrid.relative_neighbours`

CICD
 - Retire setup.py in favour of pyproject.toml
 - Build package using maturin
 - Test deploy for linux, macos and windows before uploading the sdist to PyPi

Version 0.5.1 (October 08, 2023)
--------------------------------
Fixes
 - :meth:`.BaseGrid.to_shapely()` now returns single Polygon if a single GridIndex was supplied

Documentation
 - Add example :ref:`aggregate_dask.py <example aggregate_dask>`

Version 0.5.0 (October 01, 2023)
--------------------------------
Features
 - Make return argument `shape` optional in :meth:`.BaseGrid.cells_in_bounds` by adding the `return_cell_shape` argument (default False)
 - Structure the :class:`.GridIndex` returned by :meth:`.BaseGrid.cells_in_bounds` in the shape of the grid (2D)
 - Now the return shape of :meth:`.BaseGrid.to_shapely` is the same as the input shape of the `index` argument (if `as_multipolygon` is `False`)
 - Allow :meth:`.BoundedRectGrid.centroid` to be called without specifying the `index` argument, use the cells in it's bounds by default
 - Better error when `index` is not supplied to `centroid` method on grids that are not bounded

Misc
 - Remove placeholder methods that no longer fit the curent API
 - Add tests for :meth:`.BaseGrid.to_shapely`
 - Add tests for :meth:`.BaseGrid.cell_corners`

Version 0.4.8 (September 18, 2023)
----------------------------------
Features
 - Add methods :meth:`.RectGrid.to_bounded` and :meth:`.HexGrid.to_bounded` to turn an infinite grid into a bounded grid.

Version 0.4.7 (September 10, 2023)
----------------------------------
Features
 - :meth:`~gridkit.index.concat` for combining :class:`.GridIndex` objects

Documentation
 - Fixed problems related to slicing 'flat' :class:`.BoundedHexGrid` objects
 - Swap formerly incorrect :meth:`.BoundedHexGrid.height` and :meth:`.BoundedHexGrid.width` for 'flat' :class:`.BoundedHexGrid` objects
 - Fixed nesting issue in menu navigation
 - Add colorbars to example :ref:`partial_overlap.py <example partial overlap>`
 - Simplify example :ref:`elevation_distribution_per_landcover.py <example elevation distribution landcover>`

Misc
 - Add basic tests for statistical functions :func:`~gridkit._statistical_functions.sum`, :func:`~gridkit._statistical_functions.mean`

CICD
 - Allow for manual triggering of documentation pipeline

Version 0.4.6 (September 4, 2023)
---------------------------------
Features
 - Make 'index' argument optional in :meth:`.BoundedGrid.value`

Documentation
 - Add example :ref:`partial_overlap.py <example partial overlap>`
 - Update the way docs are build in the :ref:`contributing guide <contributing>`
 - Improve docstring of :meth:`.BoundedGrid.value`

Version 0.4.5 (August 27, 2023)
-------------------------------
Fixes
 - Replace all mentions of ``read_geotiff`` in example gallery to ``write_geotiff``
 - build docs without referencing setup.py

Misc
 - Add test to verify if the documentation builds succesfully
 - Add docs_require to tests_require in setup.py
 - remove restriction on sphinx version

Version 0.4.4 (August 27, 2023)
-------------------------------
Fixes
 - Add missing matplotlib to docs_require

Version 0.4.3 (August 27, 2023)
-------------------------------
Fixes
 - Pin sphinx version to prevent docs build step from erroring

Version 0.4.2 (August 27, 2023)
-------------------------------
Fixes
 - Fix ``to_crs`` on :class:`.HexGrid` and :class:`.RectGrid` (only worked on bounded equivalents)

Documentation
 - Improved docstrings for ``to_crs`` on :class:`.BaseGrid`,  :class:`.HexGrid`,  :class:`.RectGrid`,  :class:`.BoundedHexGrid` and  :class:`.BoundedRectGrid`
 - Add docstrings to :func:`.read_raster` and :func:`.write_raster`

Misc
 - Import :class:`.GridIndex`, :func:`.validate_index`, :class:`.BaseGrid`, :class:`.RectGrid`, :class:`.HexGrid`, :class:`.BoundedRectGrid` and :class:`.BoundedHexGrid` as part of gridkit to make for more convenient importing (eg `from gridkit import HexGrid`)
 - Move pytest and matplotlib requirements from requirements.txt to tests_require in setup.py
 - Rename :func:`.read_geotiff` to :func:`.read_raster`. The former will be deprecated in a future release.

Version 0.4.1 (August 20, 2023)
-------------------------------
Features
 - make :class:`~gridkit.index.GridIndex` hashable so it works as pandas index
 - remove any empty axis on :class:`~gridkit.index.GridIndex` initialization

Documentation
 - create example script :ref:`aggregate.py <example aggregate>`
 - rename ``Shape interactions`` section to ``Vector data interactions``
 - create ``doc_utils.py`` to contain helper functions for plotting and input generation used in examples

Version 0.4.0 (August 13, 2023)
-------------------------------
Features
 - :class:`~gridkit.index.GridIndex` class to unify index representation
 - :func:`~gridkit.index.validate_index` decorator to turn any index represetntation into a GridIndex on function call
 - Operations that return grid indices now return GridIndex instances instead of numpy arrays

Version 0.3.1 (July 23, 2023)
-----------------------------
Features
 - add :meth:`~gridkit.hex_grid.BoundedHexGrid.numpy_id_to_grid_id()` to :class:`~gridkit.hex_grid.BoundedHexGrid`
 - add :meth:`~gridkit.hex_grid.BoundedHexGrid.grid_id_to_numpy_id()` to :class:`~gridkit.hex_grid.BoundedHexGrid`

Fixes
 - resolve shift in data when using comparisson and mathematical operators on BoudedHexGrid

Documentation
 - Add examle on coordinate transformations

Version 0.3.0 (July 16, 2023)
-----------------------------

Features
 - Resample method for BoundedHexGrid
 - Bilinear interpolation method for BoundedHexGrid
 - Split ``Interpolate`` method from ``resample`` method
 - Codecov integration

CICD
 - black and isort checks in test pipeline

Documentation
 - Add missing docstrings to resample method

Misc
 - reformat python files using black and isort
 - move ``Resample`` method one step up in the inheritance hierarchy, to BoundedGrid

Version 0.2.0 (July 10, 2023)
-----------------------------

Features
 - Add hex_grid.HexGrid class
 - Add hex_grid.BoundedHexGrid class
 - `to_shapely()` on bounded grids returns the shapes in the bounds when no index is supplied
 - add action for pytest and doctest on push
 - turn bounded_grid.indices into a property

Fixes
 - set proper version when documentation is build

Documentation
 - build documentation when tagged instead of merged in main
 - add example "Hexagon grids"
 - add example "Cell selection using other grids"
 - add example "Resampling"
 - use hexagons instead of squares in example "Interpolate from points"


Version 0.1.1 (March 17, 2023)
------------------------------

Fixes
 - Fix `__version__`` missing an ending quotation mark


Version 0.1.0 (March 17, 2023)
------------------------------
 - release first version to PyPi
