use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;

use pyo3::{pymodule, types::PyModule, PyResult, Python};

mod tri_grid;

#[pyclass]
struct PyTriGrid {
    cellsize: f64,
    _grid: tri_grid::TriGrid,
}

#[pymethods]
impl PyTriGrid {
    #[new]
    fn new(cellsize: f64, offset: (f64, f64)) -> Self {
        let _grid = tri_grid::TriGrid { cellsize, offset };
        PyTriGrid { cellsize, _grid }
    }

    fn cell_height(&self) -> f64 {
        self._grid.cell_height()
    }

    fn radius(&self) -> f64 {
        self._grid.radius()
    }

    fn cell_width(&self) -> f64 {
        self._grid.cell_width()
    }

    fn dx(&self) -> f64 {
        self._grid.dx()
    }

    fn dy(&self) -> f64 {
        self._grid.dy()
    }

    fn centroid<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray2<'py, i64>,
    ) -> &'py PyArray2<f64> {
        let index = index.as_array();
        self._grid.centroid(&index).into_pyarray(py)
    }

    fn cell_corners<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray2<'py, i64>,
    ) -> &'py PyArray3<f64> {
        let index = index.as_array();
        self._grid.cell_corners(&index).into_pyarray(py)
    }

    fn cell_at_point<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<i64> {
        let points = points.as_array();
        self._grid.cell_at_point(&points).into_pyarray(py)
    }

    fn cells_in_bounds<'py>(
        &self,
        py: Python<'py>,
        bounds: (f64, f64, f64, f64),
    ) -> (&'py PyArray2<i64>, (usize, usize)) {
        let (bounds, shape) = self._grid.cells_in_bounds(&bounds);
        (bounds.into_pyarray(py), shape)
    }

    fn relative_neighbours<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray2<'py, i64>,
        depth: i64,
        connect_corners: bool,
        include_selected: bool,
    ) -> &'py PyArray3<i64> {
        let index = index.as_array();
        if connect_corners {
            self._grid
                .all_neighbours(&index, depth, include_selected, false)
                .into_pyarray(py)
        } else {
            self._grid
                .direct_neighbours(&index, depth, include_selected, false)
                .into_pyarray(py)
        }
    }

    fn neighbours<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray2<'py, i64>,
        depth: i64,
        connect_corners: bool,
        include_selected: bool,
    ) -> &'py PyArray3<i64> {
        let index = index.as_array();
        if connect_corners {
            self._grid
                .all_neighbours(&index, depth, include_selected, true)
                .into_pyarray(py)
        } else {
            self._grid
                .direct_neighbours(&index, depth, include_selected, true)
                .into_pyarray(py)
        }
    }

    fn cells_near_point<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray3<i64> {
        self._grid
            .cells_near_point(&point.as_array())
            .into_pyarray(py)
    }

    fn is_cell_upright<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray2<'py, i64>,
    ) -> &'py PyArray1<bool> {
        self._grid
            .is_cell_upright(&index.as_array())
            .into_pyarray(py)
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn gridkit_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTriGrid>()?;
    Ok(())
}