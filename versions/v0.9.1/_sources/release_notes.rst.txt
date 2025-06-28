.. _release_notes:

Release notes
================

Version 0.9.1 (March 17, 2024)
------------------------------
Features
 - Change the following attributes using a setter: ``rotation``, ``offset`` and (``size`` for :class:`.TriGrid` and :class:`.HexGrid`) or (``dx`` and ``dy`` for :class:`.RectGrid`)
 - Add :meth:`.BaseGrid.cell_height` and :meth:`.BaseGrid.cell_width`
 - Add :meth:`.BaseGrid.update` method for easily making small changes to grid specs

Fixes
 - ``offset`` is now properly taken into account for :meth:`.HexGrid.cell_at_point`
 - Less restrictive offsets by limiting offset for all grids with :meth:`.BaseGrid.cell_height` and :meth:`.BaseGrid.cell_width` instead of dx and dy.
 - Rotation of "flat" :class:`.HexGrids` is no longer in the opposite direction (was clockwise)

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
