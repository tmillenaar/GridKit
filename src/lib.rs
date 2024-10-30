use std::ops::Add;

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::*;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::{
    pyfunction, pymodule, types::PyModule, wrap_pyfunction, wrap_pymodule, PyResult, Python,
};

mod data_tile;
mod grid;
mod hex_grid;
mod interpolate;
mod rect_grid;
mod tile;
mod tri_grid;
mod utils;
mod vector_shapes;

use crate::data_tile::DataTile;
use crate::grid::GridTraits;
use crate::tile::TileTraits;

#[pyclass]
#[derive(Clone)]
struct PyO3TriDataTile {
    _data_tile: data_tile::DataTile,
    grid: PyO3TriGrid,
}

#[pymethods]
impl PyO3TriDataTile {
    #[new]
    fn new<'py>(
        grid: PyO3TriGrid,
        start_id: (i64, i64),
        nx: u64,
        ny: u64,
        data: PyReadonlyArray2<'py, f64>,
        nodata_value: f64,
    ) -> Self {
        let _grid = grid::Grid::TriGrid(grid._grid.clone());
        let _data_tile = data_tile::DataTile::new(
            _grid,
            start_id,
            nx,
            ny,
            data.as_array().to_owned(),
            nodata_value,
        );
        PyO3TriDataTile { _data_tile, grid }
    }
    #[staticmethod]
    fn from_tile<'py>(
        tile: PyO3TriTile,
        data: PyReadonlyArray2<'py, f64>,
        nodata_value: f64,
    ) -> Self {
        let _data_tile = data_tile::DataTile {
            tile: tile._tile.clone(),
            data: data.as_array().to_owned(),
            nodata_value: nodata_value,
        };
        PyO3TriDataTile {
            _data_tile: _data_tile,
            grid: tile.grid,
        }
    }

    fn start_id(&self) -> (i64, i64) {
        self._data_tile.tile.start_id
    }

    fn nx(&self) -> u64 {
        self._data_tile.tile.nx
    }

    fn ny(&self) -> u64 {
        self._data_tile.tile.ny
    }

    fn get_tile<'py>(&self, py: Python<'py>) -> PyO3TriTile {
        PyO3TriTile::new(self.grid.clone(), self.start_id(), self.nx(), self.ny())
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self._data_tile.data.clone().into_pyarray(py)
    }

    fn corner_ids<'py>(&self, py: Python<'py>) -> &'py PyArray2<i64> {
        self._data_tile.corner_ids().into_pyarray(py)
    }

    fn corners<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self._data_tile.corners().into_pyarray(py)
    }

    fn indices<'py>(&self, py: Python<'py>) -> &'py PyArray3<i64> {
        self._data_tile.indices().into_pyarray(py)
    }

    fn bounds<'py>(&self, py: Python<'py>) -> (f64, f64, f64, f64) {
        self._data_tile.bounds()
    }

    fn intersects<'py>(&self, py: Python<'py>, other: &PyO3TriTile) -> bool {
        self._data_tile.intersects(&other._tile)
    }

    fn overlap<'py>(&self, py: Python<'py>, other: &PyO3TriDataTile) -> PyResult<PyO3TriTile> {
        match self._data_tile.overlap(&other._data_tile.get_tile()) {
            Ok(new_tile) => Ok(PyO3TriTile {
                grid: self.grid.clone(),
                start_id: new_tile.start_id,
                nx: new_tile.nx,
                ny: new_tile.ny,
                _tile: new_tile.clone(),
            }),
            Err(e) => Err(PyException::new_err(e)), // TODO: return custom exception for nicer try-catch on python end
        }
    }

    fn _add_scalar<'py>(&self, py: Python<'py>, value: f64) -> PyO3TriDataTile {
        let _data_tile = self._data_tile.clone() + value;
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _add_scalar_reverse<'py>(&self, py: Python<'py>, value: f64) -> PyO3TriDataTile {
        let _data_tile = value + self._data_tile.clone();
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _add_tile<'py>(&self, py: Python<'py>, other: PyO3TriDataTile) -> PyO3TriDataTile {
        let _data_tile = self._data_tile.clone() + other._data_tile;
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _subtract_scalar<'py>(&self, py: Python<'py>, value: f64) -> PyO3TriDataTile {
        let _data_tile = self._data_tile.clone() - value;
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _subtract_scalar_reverse<'py>(&self, py: Python<'py>, value: f64) -> PyO3TriDataTile {
        let _data_tile = value - self._data_tile.clone();
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _subtract_tile<'py>(&self, py: Python<'py>, other: PyO3TriDataTile) -> PyO3TriDataTile {
        let _data_tile = self._data_tile.clone() - other._data_tile;
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _multiply_scalar<'py>(&self, py: Python<'py>, value: f64) -> PyO3TriDataTile {
        let _data_tile = self._data_tile.clone() * value;
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _multiply_scalar_reverse<'py>(&self, py: Python<'py>, value: f64) -> PyO3TriDataTile {
        let _data_tile = value * self._data_tile.clone();
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _multiply_tile<'py>(&self, py: Python<'py>, other: PyO3TriDataTile) -> PyO3TriDataTile {
        let _data_tile = self._data_tile.clone() * other._data_tile;
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _divide_scalar<'py>(&self, py: Python<'py>, value: f64) -> PyO3TriDataTile {
        let _data_tile = self._data_tile.clone() / value;
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _divide_scalar_reverse<'py>(&self, py: Python<'py>, value: f64) -> PyO3TriDataTile {
        let _data_tile = value / self._data_tile.clone();
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _divide_tile<'py>(&self, py: Python<'py>, other: PyO3TriDataTile) -> PyO3TriDataTile {
        let _data_tile = self._data_tile.clone() / other._data_tile;
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _powf<'py>(&self, py: Python<'py>, value: f64) -> PyO3TriDataTile {
        let _data_tile = self._data_tile.powf(value);
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _powf_reverse<'py>(&self, py: Python<'py>, value: f64) -> PyO3TriDataTile {
        let _data_tile = self._data_tile.powf_reverse(value);
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn _powi<'py>(&self, py: Python<'py>, value: i32) -> PyO3TriDataTile {
        let _data_tile = self._data_tile.powi(value);
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn __eq__<'py>(&self, py: Python<'py>, value: f64) -> &'py PyArray2<i64> {
        self._data_tile.equals_value(value).into_pyarray(py)
    }

    fn __ne__<'py>(&self, py: Python<'py>, value: f64) -> &'py PyArray2<i64> {
        self._data_tile.not_equals_value(value).into_pyarray(py)
    }

    fn __gt__<'py>(&self, py: Python<'py>, value: f64) -> &'py PyArray2<i64> {
        self._data_tile.greater_than_value(value).into_pyarray(py)
    }

    fn __ge__<'py>(&self, py: Python<'py>, value: f64) -> &'py PyArray2<i64> {
        self._data_tile.greater_equals_value(value).into_pyarray(py)
    }

    fn __lt__<'py>(&self, py: Python<'py>, value: f64) -> &'py PyArray2<i64> {
        self._data_tile.lower_than_value(value).into_pyarray(py)
    }

    fn __le__<'py>(&self, py: Python<'py>, value: f64) -> &'py PyArray2<i64> {
        self._data_tile.lower_equals_value(value).into_pyarray(py)
    }

    fn max<'py>(&self, py: Python<'py>) -> f64 {
        self._data_tile.max()
    }

    fn min<'py>(&self, py: Python<'py>) -> f64 {
        self._data_tile.min()
    }

    fn sum<'py>(&self, py: Python<'py>) -> f64 {
        self._data_tile.sum()
    }

    fn mean<'py>(&self, py: Python<'py>) -> f64 {
        self._data_tile.mean()
    }

    fn median<'py>(&self, py: Python<'py>) -> f64 {
        self._data_tile.median()
    }

    fn percentile<'py>(&self, py: Python<'py>, percentile: f64) -> PyResult<f64> {
        match self._data_tile.percentile(percentile) {
            Ok(value) => Ok(value),
            Err(e) => Err(PyValueError::new_err(e)),
        }
    }

    fn std<'py>(&self, py: Python<'py>) -> f64 {
        self._data_tile.std()
    }

    fn _empty_combined_data_tile<'py>(
        &self,
        py: Python<'py>,
        other: PyO3TriDataTile,
    ) -> PyO3TriDataTile {
        let _data_tile = self
            ._data_tile
            ._empty_combined_tile(&other._data_tile, self._data_tile.nodata_value);
        PyO3TriDataTile {
            _data_tile,
            grid: self.grid.clone(),
        }
    }

    fn value<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray2<'py, i64>,
        nodata_value: f64,
    ) -> &'py PyArray1<f64> {
        self._data_tile
            .value(&index.as_array(), nodata_value)
            .into_pyarray(py)
    }

    fn crop<'py>(
        &self,
        py: Python<'py>,
        crop_tile: PyO3TriTile,
        nodata_value: f64,
    ) -> PyResult<PyO3TriDataTile> {
        match self._data_tile.crop(&crop_tile._tile, nodata_value) {
            Ok(new_tile) => Ok(PyO3TriDataTile {
                _data_tile: new_tile,
                grid: self.grid.clone(),
            }),
            Err(e) => Err(PyException::new_err(e)), // TODO: return custom exception for nicer try-catch on python end
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct PyO3TriTile {
    #[pyo3(get, set)]
    grid: PyO3TriGrid,
    #[pyo3(get, set)]
    start_id: (i64, i64),
    #[pyo3(get, set)]
    nx: u64,
    #[pyo3(get, set)]
    ny: u64,
    _tile: tile::Tile,
}

#[pymethods]
impl PyO3TriTile {
    #[new]
    fn new(grid: PyO3TriGrid, start_id: (i64, i64), nx: u64, ny: u64) -> Self {
        let tmp = grid.clone(); // FIXME: both PyTile and Tile need 'grid' but are in fact different structs (Py)...Grid
        let grid = grid::Grid::TriGrid(grid._grid);
        let _tile = tile::Tile {
            grid,
            start_id,
            nx,
            ny,
        };
        let grid = tmp;
        PyO3TriTile {
            grid,
            start_id,
            nx,
            ny,
            _tile,
        }
    }

    fn corner_ids<'py>(&self, py: Python<'py>) -> &'py PyArray2<i64> {
        self._tile.corner_ids().into_pyarray(py)
    }

    fn corners<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self._tile.corners().into_pyarray(py)
    }

    fn indices<'py>(&self, py: Python<'py>) -> &'py PyArray3<i64> {
        self._tile.indices().into_pyarray(py)
    }

    fn bounds<'py>(&self, py: Python<'py>) -> (f64, f64, f64, f64) {
        self._tile.bounds()
    }

    fn intersects<'py>(&self, py: Python<'py>, other: &PyO3TriTile) -> bool {
        self._tile.intersects(&other._tile)
    }

    fn overlap<'py>(&self, py: Python<'py>, other: &PyO3TriTile) -> PyResult<PyO3TriTile> {
        match self._tile.overlap(&other._tile) {
            Ok(new_tile) => Ok(PyO3TriTile {
                grid: self.grid.clone(),
                start_id: new_tile.start_id,
                nx: new_tile.nx,
                ny: new_tile.ny,
                _tile: new_tile,
            }),
            Err(e) => Err(PyException::new_err(e)), // TODO: return custom exception for nicer try-catch on python end
        }
    }
}

#[pyclass]
struct PyO3RectTile {
    #[pyo3(get, set)]
    grid: PyO3RectGrid,
    #[pyo3(get, set)]
    start_id: (i64, i64),
    #[pyo3(get, set)]
    nx: u64,
    #[pyo3(get, set)]
    ny: u64,
    _tile: tile::Tile,
}

#[pymethods]
impl PyO3RectTile {
    #[new]
    fn new(grid: PyO3RectGrid, start_id: (i64, i64), nx: u64, ny: u64) -> Self {
        let tmp = grid.clone(); // FIXME: both PyTile and Tile need 'grid' but are in fact different structs (Py)...Grid
        let grid = grid::Grid::RectGrid(grid._grid);
        let _tile = tile::Tile {
            grid,
            start_id,
            nx,
            ny,
        };
        let grid = tmp;
        PyO3RectTile {
            grid,
            start_id,
            nx,
            ny,
            _tile,
        }
    }

    fn corner_ids<'py>(&self, py: Python<'py>) -> &'py PyArray2<i64> {
        self._tile.corner_ids().into_pyarray(py)
    }

    fn corners<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self._tile.corners().into_pyarray(py)
    }

    fn indices<'py>(&self, py: Python<'py>) -> &'py PyArray3<i64> {
        self._tile.indices().into_pyarray(py)
    }

    fn bounds<'py>(&self, py: Python<'py>) -> (f64, f64, f64, f64) {
        self._tile.bounds()
    }
}

#[pyclass]
struct PyO3HexTile {
    #[pyo3(get, set)]
    grid: PyO3HexGrid,
    #[pyo3(get, set)]
    start_id: (i64, i64),
    #[pyo3(get, set)]
    nx: u64,
    #[pyo3(get, set)]
    ny: u64,
    _tile: tile::Tile,
}

#[pymethods]
impl PyO3HexTile {
    #[new]
    fn new(grid: PyO3HexGrid, start_id: (i64, i64), nx: u64, ny: u64) -> Self {
        let tmp = grid.clone(); // FIXME: both PyTile and Tile need 'grid' but are in fact different structs (Py)...Grid
        let grid = grid::Grid::HexGrid(grid._grid);
        let _tile = tile::Tile {
            grid,
            start_id,
            nx,
            ny,
        };
        let grid = tmp;
        PyO3HexTile {
            grid,
            start_id,
            nx,
            ny,
            _tile,
        }
    }

    fn corner_ids<'py>(&self, py: Python<'py>) -> &'py PyArray2<i64> {
        self._tile.corner_ids().into_pyarray(py)
    }

    fn corners<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self._tile.corners().into_pyarray(py)
    }

    fn indices<'py>(&self, py: Python<'py>) -> &'py PyArray3<i64> {
        self._tile.indices().into_pyarray(py)
    }

    fn bounds<'py>(&self, py: Python<'py>) -> (f64, f64, f64, f64) {
        self._tile.bounds()
    }
}

#[derive(Clone)]
#[pyclass]
struct PyO3TriGrid {
    cellsize: f64,
    rotation: f64,
    _grid: tri_grid::TriGrid,
}

#[pymethods]
impl PyO3TriGrid {
    #[new]
    fn new(cellsize: f64, offset: (f64, f64), rotation: f64) -> Self {
        let _grid = tri_grid::TriGrid::new(cellsize, offset, rotation);
        PyO3TriGrid {
            cellsize,
            rotation,
            _grid,
        }
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

    fn rotation_matrix<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix().into_pyarray(py)
    }

    fn rotation_matrix_inv<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix_inv().into_pyarray(py)
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
    ) -> &'py PyArray1<f64> {
        self._grid
            .linear_interpolation(
                &sample_point.as_array(),
                &nearby_value_locations.as_array(),
                &nearby_values.as_array(),
            )
            .into_pyarray(py)
    }
}

#[derive(Clone)]
#[pyclass]
struct PyO3RectGrid {
    dx: f64,
    dy: f64,
    rotation: f64,
    _grid: rect_grid::RectGrid,
}

#[pymethods]
impl PyO3RectGrid {
    #[new]
    fn new(dx: f64, dy: f64, offset: (f64, f64), rotation: f64) -> Self {
        let _grid = rect_grid::RectGrid::new(dx, dy, offset, rotation);
        PyO3RectGrid {
            dx,
            dy,
            rotation,
            _grid,
        }
    }

    fn cell_height(&self) -> f64 {
        self._grid.cell_height()
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

    fn offset(&self) -> (f64, f64) {
        self._grid.offset
    }

    fn rotation_matrix<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix().into_pyarray(py)
    }

    fn rotation_matrix_inv<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix_inv().into_pyarray(py)
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

#[derive(Clone)]
#[pyclass]
struct PyO3HexGrid {
    cellsize: f64,
    rotation: f64,
    _grid: hex_grid::HexGrid,
}

#[pymethods]
impl PyO3HexGrid {
    #[new]
    fn new(cellsize: f64, offset: (f64, f64), rotation: f64) -> Self {
        let _grid = hex_grid::HexGrid::new(cellsize, offset, rotation);
        PyO3HexGrid {
            cellsize,
            rotation,
            _grid,
        }
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

    fn rotation_matrix<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix().into_pyarray(py)
    }

    fn rotation_matrix_inv<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix_inv().into_pyarray(py)
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
    return weights.into_pyarray(py);
}

#[pymodule]
fn interp(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(linear_interp_weights_triangles, module)?)?;
    Ok(())
}

#[pyfunction]
fn multipolygon_wkb<'py>(py: Python<'py>, coords: PyReadonlyArray3<'py, f64>) -> &'py PyByteArray {
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
    module.add_class::<PyO3TriGrid>()?;
    module.add_class::<PyO3RectGrid>()?;
    module.add_class::<PyO3HexGrid>()?;
    module.add_class::<PyO3TriTile>()?;
    module.add_class::<PyO3TriDataTile>()?;
    module.add_class::<PyO3RectTile>()?;
    module.add_class::<PyO3HexTile>()?;
    module.add_wrapped(wrap_pymodule!(interp))?;
    module.add_wrapped(wrap_pymodule!(shapes))?;
    Ok(())
}
