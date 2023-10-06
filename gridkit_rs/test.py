from time import time
import numpy
from gridkit_rs import PointyHexGrid

grid = PointyHexGrid(cellsize=1.7)

# ids = numpy.array([[1,2],[3,4]], dtype=numpy.float64)
bla = numpy.arange(int(3e8), dtype=numpy.float64)
ids = numpy.vstack([
    bla,
    bla
])
bla = None
print(ids.shape)

start_rs = time()
grid.cell_at_locations( ids )
end_rs = time()

# print(result)


import gridkit
grid2 = gridkit.HexGrid(shape="pointy", size=1.7)

start_py = time()
grid2.cell_at_point(ids)
end_py = time()
# print(result2.index)

print(f"Duration Rust: {end_rs-start_rs} seconds")
print(f"Duration Python: {end_py-start_py} seconds")
