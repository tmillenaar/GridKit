use std::ops::Add;

use grid::Orientation;
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

#[derive(Clone)]
// #[enum_delegate::implement(GridTraits)]
pub enum PyO3Grid {
    PyO3TriGrid(PyO3TriGrid),
    PyO3RectGrid(PyO3RectGrid),
    PyO3HexGrid(PyO3HexGrid),
}

#[pyclass]
#[derive(Clone)]
struct PyO3DataTile {
    _data_tile: data_tile::DataTile<f64>,
    _tile: PyO3Tile,
}

#[pyclass]
#[derive(Clone)]
struct PyO3Tile {
    // #[pyo3(get, set)]
    _grid: PyO3Grid,
    #[pyo3(get, set)]
    start_id: (i64, i64),
    #[pyo3(get, set)]
    nx: u64,
    #[pyo3(get, set)]
    ny: u64,
    _tile: tile::Tile,
}

#[pymethods]
impl PyO3Tile {
    #[staticmethod]
    fn from_tri_grid(grid: PyO3TriGrid, start_id: (i64, i64), nx: u64, ny: u64) -> Self {
        let tmp = grid.clone(); // FIXME: both PyTile and Tile need 'grid' but are in fact different structs (Py)...Grid
        let grid = grid::Grid::TriGrid(grid._grid);
        let _tile = tile::Tile {
            grid,
            start_id,
            nx,
            ny,
        };
        let grid = tmp;
        PyO3Tile {
            _grid: PyO3Grid::PyO3TriGrid(grid),
            start_id,
            nx,
            ny,
            _tile,
        }
    }

    #[staticmethod]
    fn from_rect_grid(grid: PyO3RectGrid, start_id: (i64, i64), nx: u64, ny: u64) -> Self {
        let tmp = grid.clone(); // FIXME: both PyTile and Tile need 'grid' but are in fact different structs (Py)...Grid
        let grid = grid::Grid::RectGrid(grid._grid);
        let _tile = tile::Tile {
            grid,
            start_id,
            nx,
            ny,
        };
        let grid = tmp;
        PyO3Tile {
            _grid: PyO3Grid::PyO3RectGrid(grid),
            start_id,
            nx,
            ny,
            _tile,
        }
    }

    #[staticmethod]
    fn from_hex_grid(grid: PyO3HexGrid, start_id: (i64, i64), nx: u64, ny: u64) -> Self {
        let tmp = grid.clone(); // FIXME: both PyTile and Tile need 'grid' but are in fact different structs (Py)...Grid
        let grid = grid::Grid::HexGrid(grid._grid);
        let _tile = tile::Tile {
            grid,
            start_id,
            nx,
            ny,
        };
        let grid = tmp;
        PyO3Tile {
            _grid: PyO3Grid::PyO3HexGrid(grid),
            start_id,
            nx,
            ny,
            _tile,
        }
    }

    fn to_data_tile<'py>(
        &self,
        data: PyReadonlyArray2<'py, f64>,
        nodata_value: f64,
    ) -> PyO3DataTile {
        let grid = match &self._grid {
            PyO3Grid::PyO3TriGrid(grid) => grid::Grid::TriGrid(grid._grid.clone()),
            PyO3Grid::PyO3RectGrid(grid) => grid::Grid::RectGrid(grid._grid.clone()),
            PyO3Grid::PyO3HexGrid(grid) => grid::Grid::HexGrid(grid._grid.clone()),
        };
        let _tile = tile::Tile {
            grid: grid,
            start_id: self.start_id,
            nx: self.nx,
            ny: self.ny,
        };
        let _data_tile = DataTile {
            tile: _tile,
            data: data.as_array().to_owned(),
            nodata_value: nodata_value,
        };
        PyO3DataTile {
            _data_tile: _data_tile,
            _tile: self.clone(),
        }
    }

    fn to_data_tile_with_value<'py>(&self, fill_value: f64, nodata_value: f64) -> PyO3DataTile {
        let _data_tile = self._tile.to_data_tile_with_value(fill_value, nodata_value);
        PyO3DataTile {
            _data_tile: _data_tile,
            _tile: self.clone(),
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

    fn intersects<'py>(&self, py: Python<'py>, other: &PyO3Tile) -> bool {
        self._tile.intersects(&other._tile)
    }

    fn overlap<'py>(&self, py: Python<'py>, other: &PyO3Tile) -> PyResult<PyO3Tile> {
        match self._tile.overlap(&other._tile) {
            Ok(new_tile) => {
                let grid = match &new_tile.grid {
                    grid::Grid::TriGrid(grid) => PyO3Grid::PyO3TriGrid(PyO3TriGrid::new(
                        grid.cellsize,
                        &grid.cell_orientation.to_string(),
                        grid.offset.into(),
                        grid.rotation(),
                    )?),
                    grid::Grid::RectGrid(grid) => PyO3Grid::PyO3RectGrid(PyO3RectGrid::new(
                        grid.dx(),
                        grid.dy(),
                        grid.offset.into(),
                        grid.rotation(),
                    )),
                    grid::Grid::HexGrid(grid) => PyO3Grid::PyO3HexGrid(PyO3HexGrid::new(
                        grid.cellsize,
                        &grid.cell_orientation.to_string(),
                        grid.offset.into(),
                        grid.rotation(),
                    )?),
                };
                Ok(PyO3Tile {
                    _grid: grid,
                    start_id: new_tile.start_id,
                    nx: new_tile.nx,
                    ny: new_tile.ny,
                    _tile: new_tile,
                })
            }
            Err(e) => Err(PyException::new_err(e)), // TODO: return custom exception for nicer try-catch on python end
        }
    }

    fn tile_id_to_grid_id<'py>(
        &self,
        py: Python<'py>,
        tile_ids: PyReadonlyArray2<'py, i64>,
        oob_value: i64,
    ) -> &'py PyArray2<i64> {
        let index = tile_ids.as_array();
        self._tile
            .tile_id_to_grid_id(&index, oob_value)
            .into_pyarray(py)
    }

    fn grid_id_to_tile_id<'py>(
        &self,
        py: Python<'py>,
        grid_ids: PyReadonlyArray2<'py, i64>,
        oob_value: i64,
    ) -> &'py PyArray2<i64> {
        let index = grid_ids.as_array();
        self._tile
            .grid_id_to_tile_id(&index, oob_value)
            .into_pyarray(py)
    }
}

#[pymethods]
impl PyO3DataTile {
    fn start_id(&self) -> (i64, i64) {
        self._data_tile.tile.start_id
    }

    fn nx(&self) -> u64 {
        self._data_tile.tile.nx
    }

    fn ny(&self) -> u64 {
        self._data_tile.tile.ny
    }

    fn nodata_value(&self) -> f64 {
        self._data_tile.nodata_value
    }

    fn set_nodata_value(&mut self, nodata_value: f64) {
        self._data_tile.set_nodata_value(nodata_value);
    }

    fn is_nodata(&self, value: f64) -> bool {
        self._data_tile.is_nodata(value)
    }

    fn get_tile<'py>(&self, py: Python<'py>) -> PyO3Tile {
        self._tile.clone()
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

    fn intersects<'py>(&self, py: Python<'py>, other: &PyO3Tile) -> bool {
        self._data_tile.intersects(&other._tile)
    }

    fn overlap<'py>(&self, py: Python<'py>, other: &PyO3DataTile) -> PyResult<PyO3Tile> {
        match self._data_tile.overlap(&other._data_tile.get_tile()) {
            Ok(new_tile) => {
                let tile = match &self._tile._grid {
                    PyO3Grid::PyO3TriGrid(grid) => PyO3Tile::from_tri_grid(
                        grid.clone(),
                        new_tile.start_id,
                        new_tile.nx,
                        new_tile.ny,
                    ),
                    PyO3Grid::PyO3RectGrid(grid) => PyO3Tile::from_rect_grid(
                        grid.clone(),
                        new_tile.start_id,
                        new_tile.nx,
                        new_tile.ny,
                    ),
                    PyO3Grid::PyO3HexGrid(grid) => PyO3Tile::from_hex_grid(
                        grid.clone(),
                        new_tile.start_id,
                        new_tile.nx,
                        new_tile.ny,
                    ),
                };
                Ok(tile)
            }
            Err(e) => Err(PyException::new_err(e)), // TODO: return custom exception for nicer try-catch on python end
        }
    }

    fn linear_interpolation<'py>(
        &self,
        py: Python<'py>,
        sample_point: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray1<f64> {
        self._data_tile
            .linear_interpolation(&sample_point.as_array())
            .into_pyarray(py)
    }

    fn inverse_distance_interpolation<'py>(
        &self,
        py: Python<'py>,
        sample_point: PyReadonlyArray2<'py, f64>,
        decay_constant: f64,
    ) -> &'py PyArray1<f64> {
        self._data_tile
            .inverse_distance_interpolation(&sample_point.as_array(), decay_constant)
            .into_pyarray(py)
    }

    fn _add_scalar<'py>(&self, py: Python<'py>, value: f64) -> PyO3DataTile {
        let _data_tile = self._data_tile.clone() + value;
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _add_scalar_reverse<'py>(&self, py: Python<'py>, value: f64) -> PyO3DataTile {
        let _data_tile = data_tile::Scalar(value) + self._data_tile.clone();
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _add_tile<'py>(&self, py: Python<'py>, other: PyO3DataTile) -> PyO3DataTile {
        let _data_tile = self._data_tile.clone() + other._data_tile;
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _subtract_scalar<'py>(&self, py: Python<'py>, value: f64) -> PyO3DataTile {
        let _data_tile = self._data_tile.clone() - value;
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _subtract_scalar_reverse<'py>(&self, py: Python<'py>, value: f64) -> PyO3DataTile {
        let _data_tile = data_tile::Scalar(value) - self._data_tile.clone();
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _subtract_tile<'py>(&self, py: Python<'py>, other: PyO3DataTile) -> PyO3DataTile {
        let _data_tile = self._data_tile.clone() - other._data_tile;
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _multiply_scalar<'py>(&self, py: Python<'py>, value: f64) -> PyO3DataTile {
        let _data_tile = self._data_tile.clone() * value;
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _multiply_scalar_reverse<'py>(&self, py: Python<'py>, value: f64) -> PyO3DataTile {
        let _data_tile = data_tile::Scalar(value) * self._data_tile.clone();
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _multiply_tile<'py>(&self, py: Python<'py>, other: PyO3DataTile) -> PyO3DataTile {
        let _data_tile = self._data_tile.clone() * other._data_tile;
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _divide_scalar<'py>(&self, py: Python<'py>, value: f64) -> PyO3DataTile {
        let _data_tile = self._data_tile.clone() / value;
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _divide_scalar_reverse<'py>(&self, py: Python<'py>, value: f64) -> PyO3DataTile {
        let _data_tile = data_tile::Scalar(value) / self._data_tile.clone();
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _divide_tile<'py>(&self, py: Python<'py>, other: PyO3DataTile) -> PyO3DataTile {
        let _data_tile = self._data_tile.clone() / other._data_tile;
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _powf<'py>(&self, py: Python<'py>, value: f64) -> PyO3DataTile {
        let _data_tile = self._data_tile.powf(value);
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _powf_reverse<'py>(&self, py: Python<'py>, value: f64) -> PyO3DataTile {
        let _data_tile = self._data_tile.powf_reverse(value);
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
        }
    }

    fn _powi<'py>(&self, py: Python<'py>, value: i32) -> PyO3DataTile {
        let _data_tile = self._data_tile.powi(value);
        PyO3DataTile {
            _data_tile,
            _tile: self._tile.clone(),
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

    fn _empty_combined_data_tile<'py>(&self, py: Python<'py>, other: PyO3DataTile) -> PyO3DataTile {
        let _data_tile = self
            ._data_tile
            ._empty_combined_tile(&other._data_tile, self._data_tile.nodata_value);
        let tile = match &self._tile._grid {
            PyO3Grid::PyO3TriGrid(grid) => PyO3Tile::from_tri_grid(
                grid.clone(),
                _data_tile.tile.start_id,
                _data_tile.tile.nx,
                _data_tile.tile.ny,
            ),
            PyO3Grid::PyO3RectGrid(grid) => PyO3Tile::from_rect_grid(
                grid.clone(),
                _data_tile.tile.start_id,
                _data_tile.tile.nx,
                _data_tile.tile.ny,
            ),
            PyO3Grid::PyO3HexGrid(grid) => PyO3Tile::from_hex_grid(
                grid.clone(),
                _data_tile.tile.start_id,
                _data_tile.tile.nx,
                _data_tile.tile.ny,
            ),
        };

        PyO3DataTile {
            _data_tile: _data_tile,
            _tile: tile,
        }
    }

    fn value<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray2<'py, i64>,
        nodata_value: f64,
    ) -> &'py PyArray1<f64> {
        self._data_tile
            .values(&index.as_array(), nodata_value)
            .into_pyarray(py)
    }

    fn crop<'py>(
        &self,
        py: Python<'py>,
        crop_tile: PyO3Tile,
        nodata_value: f64,
    ) -> PyResult<PyO3DataTile> {
        match self._data_tile.crop(&crop_tile._tile, nodata_value) {
            Ok(new_tile) => Ok(PyO3DataTile {
                _data_tile: new_tile,
                _tile: self._tile.clone(),
            }),
            Err(e) => Err(PyException::new_err(e)), // TODO: return custom exception for nicer try-catch on python end
        }
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
    fn new(
        cellsize: f64,
        cell_orientation: &str,
        offset: (f64, f64),
        rotation: f64,
    ) -> PyResult<Self> {
        match Orientation::from_string(cell_orientation) {
            Some(cell_orientation) => {
                let mut _grid = tri_grid::TriGrid::new(cellsize, cell_orientation);
                _grid.set_offset(offset.into());
                _grid.set_rotation(rotation);
                Ok(PyO3TriGrid {
                    cellsize,
                    rotation,
                    _grid,
                })
            }
            None => Err(PyException::new_err(format!(
                "Unrecognized cell_orientation. Use 'Flat' or 'Pointy'. Got {}",
                cell_orientation
            ))),
        }
    }

    fn offset(&self) -> (f64, f64) {
        self._grid.offset.into()
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
        &self._grid.rotation_matrix().clone().into_pyarray(py)
    }

    fn rotation_matrix_inv<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix_inv().clone().into_pyarray(py)
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
        let mut _grid = rect_grid::RectGrid::new(dx, dy);
        _grid.set_offset(offset.into());
        _grid.set_rotation(rotation);
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
        self._grid.offset.into()
    }

    fn rotation_matrix<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix().clone().into_pyarray(py)
    }

    fn rotation_matrix_inv<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix_inv().clone().into_pyarray(py)
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
    fn new(
        cellsize: f64,
        cell_orientation: &str,
        offset: (f64, f64),
        rotation: f64,
    ) -> PyResult<Self> {
        match Orientation::from_string(cell_orientation) {
            Some(cell_orientation) => {
                let mut _grid = hex_grid::HexGrid::new(cellsize, cell_orientation);
                _grid.set_offset(offset.into());
                _grid.set_rotation(rotation);
                Ok(PyO3HexGrid {
                    cellsize,
                    rotation,
                    _grid,
                })
            }
            None => Err(PyException::new_err(format!(
                "Unrecognized cell_orientation. Use 'Flat' or 'Pointy'. Got {}",
                cell_orientation
            ))),
        }
    }

    fn cell_orientation<'py>(&self, py: Python<'py>) -> &'py PyString {
        PyString::new(py, &self._grid.cell_orientation.to_string())
    }

    fn set_cell_orientation(&mut self, cell_orientation: &str) -> PyResult<()> {
        match Orientation::from_string(cell_orientation) {
            Some(cell_orientation) => Ok(self._grid.set_cell_orientation(cell_orientation)),
            None => Err(PyException::new_err(format!(
                "Unrecognized cell_orientation. Use 'Flat' or 'Pointy'. Got {}",
                cell_orientation
            ))),
        }
    }

    fn cell_height(&self) -> f64 {
        self._grid.cell_height()
    }

    fn cell_width(&self) -> f64 {
        self._grid.cell_width()
    }

    fn offset(&self) -> (f64, f64) {
        self._grid.offset.into()
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
        &self._grid.rotation_matrix().clone().into_pyarray(py)
    }

    fn rotation_matrix_inv<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        &self._grid.rotation_matrix_inv().clone().into_pyarray(py)
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

#[pyfunction]
fn combine_tiles<'py>(py: Python<'py>, tiles: &PyList) -> PyResult<PyO3Tile> {
    let tiles_vec: Vec<tile::Tile> = tiles
        .iter()
        .map(|item| {
            let py_o3_tile: PyO3Tile = item.extract().unwrap(); // Extract PyO3Tile
            py_o3_tile._tile.clone() // Get the Tile reference and clone it
        })
        .collect();
    let _tile = tile::combine_tiles(&tiles_vec);
    let _grid = tiles[0].extract::<PyO3Tile>().unwrap()._grid;
    let tile = PyO3Tile {
        _grid: _grid,
        start_id: _tile.start_id,
        nx: _tile.nx,
        ny: _tile.ny,
        _tile,
    };
    Ok(tile)
}

#[pyfunction]
fn count_tiles<'py>(py: Python<'py>, tiles: &PyList) -> PyResult<PyO3DataTile> {
    let tiles_vec: Vec<tile::Tile> = tiles
        .iter()
        .map(|item| {
            let py_o3_tile: PyO3Tile = item.extract().unwrap();
            py_o3_tile._tile.clone()
        })
        .collect();
    let _data_tile = tile::count_tiles(&tiles_vec);
    let _tile = tiles[0].extract::<PyO3Tile>().unwrap();
    let tile = PyO3DataTile { _data_tile, _tile };
    Ok(tile)
}

#[pyfunction]
fn count_data_tiles<'py>(py: Python<'py>, data_tiles: &PyList) -> PyResult<PyO3DataTile> {
    // Like count_tiles but does not count cells with a nodata_value
    let tiles_vec: Vec<DataTile<f64>> = data_tiles
        .iter()
        .map(|item| {
            let py_o3_tile: PyO3DataTile = item.extract().unwrap();
            py_o3_tile._data_tile
        })
        .collect();
    let _data_tile = tile::count_data_tiles(&tiles_vec);
    let _tile = data_tiles[0]
        .extract::<PyO3DataTile>()
        .unwrap()
        .get_tile(py);
    let tile = PyO3DataTile { _data_tile, _tile };
    Ok(tile)
}

#[pyfunction]
fn sum_data_tile<'py>(py: Python<'py>, data_tiles: &PyList) -> PyResult<PyO3DataTile> {
    let tiles_vec: Vec<DataTile<f64>> = data_tiles
        .iter()
        .map(|item| {
            let py_o3_tile: PyO3DataTile = item.extract().unwrap();
            py_o3_tile._data_tile
        })
        .collect();
    let _data_tile = tile::sum_data_tiles(&tiles_vec);
    let _tile = data_tiles[0]
        .extract::<PyO3DataTile>()
        .unwrap()
        .get_tile(py);
    let tile = PyO3DataTile { _data_tile, _tile };
    Ok(tile)
}

#[pyfunction]
fn average_data_tile<'py>(py: Python<'py>, data_tiles: &PyList) -> PyResult<PyO3DataTile> {
    let tiles_vec: Vec<DataTile<f64>> = data_tiles
        .iter()
        .map(|item| {
            let py_o3_tile: PyO3DataTile = item.extract().unwrap();
            py_o3_tile._data_tile
        })
        .collect();
    let _data_tile = tile::average_data_tiles(&tiles_vec);
    let _tile = data_tiles[0]
        .extract::<PyO3DataTile>()
        .unwrap()
        .get_tile(py);
    let tile = PyO3DataTile { _data_tile, _tile };
    Ok(tile)
}

#[pymodule]
fn tile_utils(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(combine_tiles, module)?)?;
    module.add_function(wrap_pyfunction!(count_tiles, module)?)?;
    module.add_function(wrap_pyfunction!(count_data_tiles, module)?)?;
    module.add_function(wrap_pyfunction!(sum_data_tile, module)?)?;
    module.add_function(wrap_pyfunction!(average_data_tile, module)?)?;
    Ok(())
}

#[pymodule]
fn gridkit_rs(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyO3TriGrid>()?;
    module.add_class::<PyO3RectGrid>()?;
    module.add_class::<PyO3HexGrid>()?;
    module.add_class::<PyO3Tile>()?;
    module.add_class::<PyO3DataTile>()?;
    module.add_wrapped(wrap_pymodule!(interp))?;
    module.add_wrapped(wrap_pymodule!(shapes))?;
    module.add_wrapped(wrap_pymodule!(tile_utils))?;
    Ok(())
}
