use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::{pymodule, pyfunction, wrap_pyfunction, wrap_pymodule, types::PyModule, PyResult, Python};

mod utils;
mod tri_grid;
mod rect_grid;
mod hex_grid;
mod vector_shapes;
mod interpolate;

#[pyclass]
struct PyTriGrid {
    cellsize: f64,
    rotation: f64,
    _grid: tri_grid::TriGrid,
}

#[pymethods]
impl PyTriGrid {
    #[new]
    fn new(cellsize: f64, offset: (f64, f64), rotation: f64) -> Self {
        let _grid = tri_grid::TriGrid::new( cellsize, offset, rotation);
        PyTriGrid { cellsize, rotation, _grid }
    }

    fn offset(&self) -> (f64, f64) {
        self._grid.offset
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
    
    fn rotation_matrix<'py>(
        &self,
        py: Python<'py>,
    ) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix.clone().into_pyarray(py)
    }

    fn rotation_matrix_inv<'py>(
        &self,
        py: Python<'py>,
    ) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix_inv.clone().into_pyarray(py)
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

    fn linear_interpolation<'py>(
        &self,
        py: Python<'py>,
        sample_point: PyReadonlyArray2<'py, f64>,
        nearby_value_locations: PyReadonlyArray3<'py, f64>,
        nearby_values: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray1<f64>  {
        self._grid.linear_interpolation(
            &sample_point.as_array(),
            &nearby_value_locations.as_array(),
            &nearby_values.as_array(),
        ).into_pyarray(py)
    }
}


#[pyclass]
struct PyRectGrid {
    dx: f64,
    dy: f64,
    rotation: f64,
    _grid: rect_grid::RectGrid,
}

#[pymethods]
impl PyRectGrid {
    #[new]
    fn new(dx: f64, dy: f64, offset: (f64, f64), rotation: f64) -> Self {
        let _grid = rect_grid::RectGrid::new(dx, dy, offset, rotation);
        PyRectGrid { dx, dy, rotation, _grid }
    }

    fn cell_height(&self) -> f64 {
        self._grid.cell_height()
    }

    fn cell_width(&self) -> f64 {
        self._grid.cell_width()
    }

    fn offset(&self) -> (f64, f64) {
        self._grid.offset
    }

    fn rotation_matrix<'py>(
        &self,
        py: Python<'py>,
    ) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix.clone().into_pyarray(py)
    }

    fn rotation_matrix_inv<'py>(
        &self,
        py: Python<'py>,
    ) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix_inv.clone().into_pyarray(py)
    }

    fn centroid<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray2<'py, i64>,
    ) -> &'py PyArray2<f64> {
        let index = index.as_array();
        self._grid.centroid(&index).into_pyarray(py)
    }

    fn cell_at_point<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<i64> {
        let points = points.as_array();
        self._grid.cell_at_point(&points).into_pyarray(py)
    }

    fn cell_corners<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray2<'py, i64>,
    ) -> &'py PyArray3<f64> {
        let index = index.as_array();
        self._grid.cell_corners(&index).into_pyarray(py)
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
}

#[pyclass]
struct PyHexGrid {
    cellsize: f64,
    rotation: f64,
    _grid: hex_grid::HexGrid,
}

#[pymethods]
impl PyHexGrid {
    #[new]
    fn new(cellsize: f64, offset: (f64, f64), rotation: f64) -> Self {
        let _grid = hex_grid::HexGrid::new(cellsize, offset, rotation);
        PyHexGrid { cellsize, rotation, _grid}
    }

    fn cell_height(&self) -> f64 {
        self._grid.cell_height()
    }

    fn cell_width(&self) -> f64 {
        self._grid.cell_width()
    }

    fn offset(&self) -> (f64, f64) {
        self._grid.offset
    }

    fn radius(&self) -> f64 {
        self._grid.radius()
    }

    fn dx(&self) -> f64 {
        self._grid.dx()
    }

    fn dy(&self) -> f64 {
        self._grid.dy()
    }
    
    fn rotation_matrix<'py>(
        &self,
        py: Python<'py>,
    ) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix.clone().into_pyarray(py)
    }

    fn rotation_matrix_inv<'py>(
        &self,
        py: Python<'py>,
    ) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix_inv.clone().into_pyarray(py)
    }

    fn centroid<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray2<'py, i64>,
    ) -> &'py PyArray2<f64> {
        let index = index.as_array();
        self._grid.centroid(&index).into_pyarray(py)
    }

    fn cell_at_location<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<i64> {
        let points = points.as_array();
        self._grid.cell_at_location(&points).into_pyarray(py)
    }

    fn cell_corners<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray2<'py, i64>,
    ) -> &'py PyArray3<f64> {
        let index = index.as_array();
        self._grid.cell_corners(&index).into_pyarray(py)
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
}

#[pyfunction]
fn linear_interp_weights_triangles<'py>(
    py: Python<'py>,
    sample_point: PyReadonlyArray2<'py, f64>,
    nearby_value_locations: PyReadonlyArray3<'py, f64>,
) -> &'py PyArray2<f64> {
    let weights = interpolate::linear_interp_weights_triangles(
        &sample_point.as_array(),
        &nearby_value_locations.as_array(),
    );
    return weights.into_pyarray(py)
}

#[pymodule]
fn interp(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(linear_interp_weights_triangles, module)?)?;
    Ok(())
}

#[pyfunction]
fn multipolygon_wkb<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray3<'py, f64>,
) -> &'py PyByteArray {
    let geom_wkb = vector_shapes::coords_to_multipolygon_wkb(&coords.as_array());
    PyByteArray::new(py, &geom_wkb)
}

#[pymodule]
fn shapes(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(multipolygon_wkb, module)?)?;
    Ok(())
}

#[pymodule]
fn gridkit_rs(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyTriGrid>()?;
    module.add_class::<PyRectGrid>()?;
    module.add_class::<PyHexGrid>()?;
    module.add_wrapped(wrap_pymodule!(interp))?;
    module.add_wrapped(wrap_pymodule!(shapes))?;
    Ok(())
}
