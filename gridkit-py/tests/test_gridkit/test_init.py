import subprocess
import sys

import gridkit


def test_initialized_objects():
    expected_attributes = [
        "GridIndex",
        "validate_index",
        "BaseGrid",
        "RectGrid",
        "HexGrid",
        "BoundedRectGrid",
        "BoundedHexGrid",
        "mean",
        "sum",
        "count",
        "read_raster",
        "write_raster",
        "set_num_threads",
        "get_num_threads",
    ]
    for attr in expected_attributes:
        assert hasattr(gridkit, attr), f"Missing attribute '{attr}' from gridkit module"


def test_num_threads_api():
    # Run in a fresh interpreter so this test owns the process-local pool.
    code = """
import gridkit
import numpy as np

assert gridkit.get_num_threads() >= 1

try:
    gridkit.set_num_threads(2)
except RuntimeError:
    # Built without the `parallel` feature: control is unavailable.
    assert gridkit.get_num_threads() == 1
else:
    assert gridkit.get_num_threads() == 2
    grid = gridkit.HexGrid(size=1.0)
    ids = np.arange(8192, dtype=np.int64).reshape((-1, 2))
    grid.cell_corners(ids)
    gridkit.set_num_threads(4)
    assert gridkit.get_num_threads() == 4
    gridkit.set_num_threads(None)
    assert gridkit.get_num_threads() >= 1
    try:
        gridkit.set_num_threads(0)
    except ValueError:
        pass
    else:
        raise AssertionError("set_num_threads(0) should raise ValueError")
"""
    subprocess.run([sys.executable, "-c", code], check=True)
