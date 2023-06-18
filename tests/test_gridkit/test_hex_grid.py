import numpy
import pytest

from gridkit.hex_grid import HexGrid


@pytest.mark.parametrize("shape, indices, expected_centroids", [
        ["pointy", (-1,-1), [0, -1.29903811]],
        ["pointy", (-1,1), [0, 3.897114]],
        ["pointy", [
            (0,0),(1,-1),(1,1)
        ], [
            [1.5, 1.29903811],
            [6, -1.29903811],
            [6, 3.89711432],
        ]]
])
def test_centroid(shape, indices, expected_centroids):
    grid = HexGrid(size=3)
    centroids = grid.centroid(indices)
    numpy.testing.assert_allclose(centroids, expected_centroids)


@pytest.mark.parametrize("shape, bounds, expected_ids", [
    ["pointy", (0,-2,2,2), [[0,0],[-1,-1]]],
    ["pointy", (0,-2,3,2), [[0,0],[-1,-1],[0,-1]]],
    ["pointy", (0,0,1,5), [[-1,1],[0,0]]],
    ["pointy", (0,0,3,5), [[-1,1],[0,1],[0,0]]],
    ["pointy", (-2,-2,2,2), [[-1,0],[0,0],[-1,-1]]],
    ["pointy", (-2,-2,3,2), [[-1,0],[0,0],[-1,-1],[0,-1]]],
    ["pointy", (-2,0,2,5), [[-1,1],[-1,0],[0,0]]],
    ["pointy", (-2,0,3,5), [[-1,1],[0,1],[-1,0],[0,0]]],
    ["flat", (-2,-2,2,2), [[-1,-1],[0,-1],[0,0]]],
    ["flat", (-2,-2,2,3), [[-1,-1],[-1,0],[0,-1],[0,0]]],
    ["flat", (0,-2,5,2), [[0,-1],[0,0],[1,-1]]],
    ["flat", (0,-2,5,3), [[0,-1],[0,0],[1,-1],[1,0]]],
    ["flat", (-2,0,2,2), [[-1,-1],[0,0]]],
    ["flat", (-2,0,2,3), [[-1,-1],[-1,0],[0,0]]],
    ["flat", (-2,2,2,3), [[-1,0],[0,0]]],
    ["flat", (-2,2,2,5), [[-1,0],[0,0],[0,1]]],
    ["flat", (0,0,5,1), [[0,0],[1,-1]]],
    ["flat", (0,0,5,3), [[0,0],[1,-1],[1,0]]],
])
def test_cells_in_bounds(shape, bounds, expected_ids):
    grid = HexGrid(size=3, shape=shape)
    aligned_bounds = grid.align_bounds(bounds, mode="nearest")
    ids, _ = grid.cells_in_bounds(aligned_bounds) #TODO: remove returned None shape

    import matplotlib.pyplot as plt
    def bounds_to_poly(bounds): 
        return (
            [[bounds[0], bounds[0], bounds[2], bounds[2], bounds[0]],
            [bounds[1], bounds[3], bounds[3], bounds[1], bounds[1]]]
        )
    fig, ax = plt.subplots()
    ax.fill(
            *bounds_to_poly(aligned_bounds),
            alpha=0.5,
            color="green"
        )
    for poly, id in zip(grid.to_shapely(ids), ids):
        x, y = poly.exterior.xy
        ax.fill(
            x,y,
            alpha=0.5,
            color="red"
        )
        ax.text(*grid.centroid(id), id)
    fig.savefig(f"test_{bounds}.png")
    numpy.testing.assert_allclose(ids, expected_ids)


@pytest.mark.parametrize("shape, point, expected_nearby_cells", [
    ["pointy", (2.5, 1.5 ), [(0,0), (0, 1), (1, 0)]],
    ["pointy", (1.5, 2.16), [(0,0), (-1, 1), (0, 1)]],
    ["pointy", (0.5, 1.5 ), [(0,0), (-1, 0), (-1, 1)]],
    ["pointy", (0.5, 1.0 ), [(0,0), (-1, -1), (-1, 0)]],
    ["pointy", (1.5, 0.43), [(0,0), (0, -1), (-1, -1)]],
    ["pointy", (2.5, 1.0 ), [(0,0), (1, 0), (0, -1)]],
    ["pointy", [
        (4, 4.1 ),
        (3, 4.76),
        (2, 4.1 ),
        (2, 3.6 ),
        (3, 3.03),
        (4, 3.6 )
    ], [
        [[0,1], [1, 2], [1, 1]],
        [[0,1], [0, 2], [1, 2]],
        [[0,1], [-1, 1], [0, 2]],
        [[0,1], [0, 0], [-1, 1]],
        [[0,1], [1, 0], [0, 0]],
        [[0,1], [1, 1], [1, 0]]
    ]],
    ["flat", (2.5, 1.5 ), [(0,0), (1, -1), (1, 0)]],
    ["flat", (2.0, 2.5), [(0,0), (1, 0), (0, 1)]],
    ["flat", (1.0, 2.5 ), [(0,0), (0, 1), (-1, 0)]],
    ["flat", (0.0, 1.5 ), [(0,0), (-1, 0), (-1, -1)]],
    ["flat", (1.0, 0.5), [(0,0), (-1, -1), (0, -1)]],
    ["flat", (2.0, 0.5 ), [(0,0), (0, -1), (1, -1)]],
    ["flat", [
        (5, 3 ),
        (4.5, 4),
        (3.5, 4 ),
        (2.5, 3 ),
        (3.5, 2),
        (4.5, 2 )
    ], [
        [[1,0], [2, 0], [2, 1]],
        [[1,0], [2, 1], [1, 1]],
        [[1,0], [1, 1], [0, 1]],
        [[1,0], [0, 1], [0, 0]],
        [[1,0], [0, 0], [1, -1]],
        [[1,0], [1, -1], [2, 0]]
    ]],
])
def test_cells_near_point(shape, point, expected_nearby_cells):
    grid = HexGrid(size=3, shape=shape)
    nearby_cells = grid.cells_near_point(point)
    numpy.testing.assert_allclose(nearby_cells, expected_nearby_cells)


@pytest.mark.parametrize("shape", ["pointy", "flat"])
@pytest.mark.parametrize("depth", list(range(1,7)))
@pytest.mark.parametrize("index", [[2,1], [1,2]])
@pytest.mark.parametrize("include_selected", [False, True])
def test_neighbours(shape, depth, index, include_selected):
    grid = HexGrid(size=3, shape=shape)
    neighbours = grid.neighbours(index, depth=depth, include_selected=include_selected)

    if include_selected:
        # make sure the index is present and at the center of the neighbours array
        center_index = int(numpy.floor(len(neighbours) / 2))
        numpy.testing.assert_allclose(neighbours[center_index], index)
        # remove center index for further testing of other neighbours
        neighbours = numpy.delete(neighbours, center_index, axis=0)

    # If the neighbours are correct, there are always a multiple of 6 cells with the same distance to the center cell
    distances = numpy.linalg.norm(grid.centroid(neighbours) - grid.centroid(index), axis=1)
    for d in numpy.unique(distances):
        assert sum(distances == d) % 6 == 0

    # No cell can be further away than 'depth' number of cells * cell size
    assert all(distances <= grid.size * depth)
