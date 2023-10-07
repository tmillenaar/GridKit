.. _release_notes:

Release notes
================

Version 0.5.1 (October 08, 2023)
----------------------------------
Fixes
 - :meth:`.BaseGrid.to_shapely()` now returns single Polygon if a single GridIndex was supplied

Documentation
 - Add example :ref:`aggregate_dask.py <example aggregate_dask>`

Version 0.5.0 (October 01, 2023)
----------------------------------
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
