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
        "num_threads",
    ]
    for attr in expected_attributes:
        assert hasattr(gridkit, attr), f"Missing attribute '{attr}' from gridkit module"


def test_num_threads_api():
    # Run in a fresh interpreter so the lazy thread pool has not been built yet,
    # which is what makes set_num_threads() usable.
    code = """
import gridkit

assert gridkit.num_threads() >= 1

try:
    gridkit.set_num_threads(2)
except RuntimeError:
    # Built without the `parallel` feature: control is unavailable.
    assert gridkit.num_threads() == 1
else:
    assert gridkit.num_threads() == 2
    gridkit.set_num_threads(None)
    assert gridkit.num_threads() >= 1
    try:
        gridkit.set_num_threads(0)
    except ValueError:
        pass
    else:
        raise AssertionError("set_num_threads(0) should raise ValueError")
"""
    subprocess.run([sys.executable, "-c", code], check=True)
