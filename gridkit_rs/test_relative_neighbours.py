import numpy

import gridkit_rs
from gridkit_rs import PointyHexGrid

grid = PointyHexGrid(cellsize=1.7)

ids = numpy.array([[1, 2], [-4, 5]])

print(ids)
result = grid.relative_neighbours(ids, 1, False, False)
print(result)
