.. _release_notes:

Release notes
================

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
