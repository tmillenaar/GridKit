use std::ops::Add;
// use num_traits::{Bounded, FromPrimitive, Num, ToPrimitive};

use grid::Orientation;
use gridkit::CellElement;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyReadonlyArrayDyn,
};
use pyo3::exceptions::*;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::{
    pyfunction, pymodule, types::PyModule, wrap_pyfunction, wrap_pymodule, PyResult, Python,
};

use gridkit::data_tile;
use gridkit::grid;
use gridkit::hex_grid;
use gridkit::interpolate;
use gridkit::rect_grid;
use gridkit::tile;
use gridkit::tri_grid;
use gridkit::utils;
use gridkit::vector_shapes;

use crate::data_tile::DataTile;
use crate::grid::GridTraits;
use crate::tile::TileTraits;

macro_rules! impl_pydata_tile {
    ($name:ident, $type:ty, $f64type:ident) => {
        #[pyclass]
        #[derive(Clone)]
        pub struct $name {
            _data_tile: data_tile::DataTile<$type>,
            _tile: PyO3Tile,
        }
        #[pymethods]
        impl $name {
            fn start_id(&self) -> (i64, i64) {
                self._data_tile.tile.start_id
            }

            fn nx(&self) -> u64 {
                self._data_tile.tile.nx
            }

            fn ny(&self) -> u64 {
                self._data_tile.tile.ny
            }

            fn nodata_value(&self) -> $type {
                self._data_tile.nodata_value
            }

            fn set_nodata_value(&mut self, nodata_value: $type) {
                self._data_tile.set_nodata_value(nodata_value);
            }

            fn is_nodata(&self, value: $type) -> bool {
                self._data_tile.is_nodata(&value)
            }

            fn is_nodata_array<'py>(
                &self,
                values: PyReadonlyArrayDyn<'py, $type>,
                py: Python<'py>,
            ) -> &'py PyArrayDyn<bool> {
                self._data_tile
                    .is_nodata_array(&values.as_array())
                    .into_pyarray(py)
            }

            fn nodata_cells<'py>(&self, py: Python<'py>) -> &'py PyArray2<i64> {
                self._data_tile.nodata_cells().into_pyarray(py)
            }

            fn get_tile<'py>(&self, _py: Python<'py>) -> PyO3Tile {
                self._tile.clone()
            }

            fn to_numpy<'py>(&self, py: Python<'py>) -> &'py PyArray2<$type> {
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

            fn intersects<'py>(&self, _py: Python<'py>, other: &PyO3Tile) -> bool {
                self._data_tile.intersects(&other._tile)
            }

            fn overlap<'py>(&self, _py: Python<'py>, other: &$name) -> PyResult<PyO3Tile> {
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

            fn _add_scalar<'py>(&self, _py: Python<'py>, value: $type) -> Self {
                let _data_tile = &self._data_tile + value;
                $name {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _add_scalar_reverse<'py>(&self, _py: Python<'py>, value: $type) -> Self {
                let _data_tile: data_tile::DataTile<$type> =
                    data_tile::Scalar::<$type>(value) + &self._data_tile;
                $name {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _add_tile<'py>(&self, _py: Python<'py>, other: $name) -> Self {
                let _data_tile = &self._data_tile + &other._data_tile;
                $name {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _subtract_scalar<'py>(&self, _py: Python<'py>, value: $type) -> Self {
                let _data_tile = &self._data_tile - value;
                $name {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _subtract_scalar_reverse<'py>(&self, _py: Python<'py>, value: $type) -> Self {
                let _data_tile = data_tile::Scalar(value) - &self._data_tile;
                $name {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _subtract_tile<'py>(&self, _py: Python<'py>, other: $name) -> Self {
                let _data_tile = &self._data_tile - &other._data_tile;
                $name {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _multiply_scalar<'py>(&self, _py: Python<'py>, value: $type) -> Self {
                let _data_tile = &self._data_tile * value;
                $name {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _multiply_scalar_reverse<'py>(&self, _py: Python<'py>, value: $type) -> Self {
                let _data_tile = data_tile::Scalar(value) * &self._data_tile;
                $name {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _multiply_tile<'py>(&self, _py: Python<'py>, other: $name) -> Self {
                let _data_tile = &self._data_tile * &other._data_tile;
                $name {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _divide_scalar<'py>(&self, _py: Python<'py>, value: $type) -> $f64type {
                let _data_tile = &self._data_tile / value;
                $f64type {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _divide_scalar_reverse<'py>(&self, _py: Python<'py>, value: $type) -> $f64type {
                let _data_tile = data_tile::Scalar(value) / &self._data_tile;
                $f64type {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _divide_tile<'py>(&self, _py: Python<'py>, other: $name) -> $f64type {
                let _data_tile = &self._data_tile / &other._data_tile;
                $f64type {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _powf<'py>(&self, _py: Python<'py>, value: f64) -> $f64type {
                let _data_tile = self._data_tile.powf(value);
                $f64type {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _powf_reverse<'py>(&self, _py: Python<'py>, value: f64) -> $f64type {
                // FIXME: Only for float??
                let _data_tile = self._data_tile.powf_reverse(value);
                $f64type {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn _powi<'py>(&self, _py: Python<'py>, value: i32) -> $f64type {
                let _data_tile = self._data_tile.powi(value);
                $f64type {
                    _data_tile,
                    _tile: self._tile.clone(),
                }
            }

            fn __eq__<'py>(&self, py: Python<'py>, value: $type) -> &'py PyArray2<i64> {
                self._data_tile.equals_value(value).into_pyarray(py)
            }

            fn __ne__<'py>(&self, py: Python<'py>, value: $type) -> &'py PyArray2<i64> {
                self._data_tile.not_equals_value(value).into_pyarray(py)
            }

            fn __gt__<'py>(&self, py: Python<'py>, value: $type) -> &'py PyArray2<i64> {
                self._data_tile.greater_than_value(value).into_pyarray(py)
            }

            fn __ge__<'py>(&self, py: Python<'py>, value: $type) -> &'py PyArray2<i64> {
                self._data_tile.greater_equals_value(value).into_pyarray(py)
            }

            fn __lt__<'py>(&self, py: Python<'py>, value: $type) -> &'py PyArray2<i64> {
                self._data_tile.lower_than_value(value).into_pyarray(py)
            }

            fn __le__<'py>(&self, py: Python<'py>, value: $type) -> &'py PyArray2<i64> {
                self._data_tile.lower_equals_value(value).into_pyarray(py)
            }

            fn max<'py>(&self, _py: Python<'py>) -> $type {
                self._data_tile.max()
            }

            fn min<'py>(&self, _py: Python<'py>) -> $type {
                self._data_tile.min()
            }

            fn sum<'py>(&self, _py: Python<'py>) -> $type {
                self._data_tile.sum()
            }

            fn mean<'py>(&self, _py: Python<'py>) -> f64 {
                self._data_tile.mean()
            }

            fn median<'py>(&self, _py: Python<'py>) -> f64 {
                self._data_tile.median()
            }

            fn percentile<'py>(&self, _py: Python<'py>, percentile: f64) -> PyResult<f64> {
                match self._data_tile.percentile(percentile) {
                    Ok(value) => Ok(value),
                    Err(e) => Err(PyValueError::new_err(e)),
                }
            }

            fn std<'py>(&self, _py: Python<'py>) -> f64 {
                self._data_tile.std()
            }

            fn _empty_combined_data_tile<'py>(&self, _py: Python<'py>, other: $name) -> Self {
                // Fixme: other should maybe be tile and not datatile?
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

                $name {
                    _data_tile: _data_tile,
                    _tile: tile,
                }
            }

            fn value<'py>(
                &self,
                py: Python<'py>,
                index: PyReadonlyArray2<'py, i64>,
                nodata_value: $type,
            ) -> &'py PyArray1<$type> {
                self._data_tile
                    .values(&index.as_array(), nodata_value)
                    .into_pyarray(py)
            }

            fn crop<'py>(
                &self,
                _py: Python<'py>,
                crop_tile: PyO3Tile,
                nodata_value: $type,
            ) -> PyResult<$name> {
                match self._data_tile.crop(&crop_tile._tile, nodata_value) {
                    Ok(new_tile) => Ok($name {
                        _data_tile: new_tile,
                        _tile: self._tile.clone(),
                    }),
                    Err(e) => Err(PyException::new_err(e)), // TODO: return custom exception for nicer try-catch on python end
                }
            }
        }
    };
}

impl_pydata_tile!(PyO3DataTileF64, f64, PyO3DataTileF64);
impl_pydata_tile!(PyO3DataTileF32, f32, PyO3DataTileF64);
impl_pydata_tile!(PyO3DataTileI64, i64, PyO3DataTileF64);
impl_pydata_tile!(PyO3DataTileI32, i32, PyO3DataTileF64);
impl_pydata_tile!(PyO3DataTileI16, i16, PyO3DataTileF64);
impl_pydata_tile!(PyO3DataTileI8, i8, PyO3DataTileF64);
impl_pydata_tile!(PyO3DataTileU64, u64, PyO3DataTileF64);
impl_pydata_tile!(PyO3DataTileU32, u32, PyO3DataTileF64);
impl_pydata_tile!(PyO3DataTileU16, u16, PyO3DataTileF64);
impl_pydata_tile!(PyO3DataTileU8, u8, PyO3DataTileF64);

#[derive(Clone)]
pub enum PyO3DataTile {
    F64(PyO3DataTileF64),
    F32(PyO3DataTileF32),
    I64(PyO3DataTileI64),
    I32(PyO3DataTileI32),
    I16(PyO3DataTileI16),
    I8(PyO3DataTileI8),
    U64(PyO3DataTileU64),
    U32(PyO3DataTileU32),
    U16(PyO3DataTileU16),
    U8(PyO3DataTileU8),
    // Bool(PyO3DataTileBool),
    // Complex(PyO3DataTileComplex),
}

impl<'py> FromPyObject<'py> for PyO3DataTile {
    fn extract(obj: &'py PyAny) -> PyResult<Self> {
        if let Ok(v) = obj.extract::<PyRef<'py, PyO3DataTileF64>>() {
            return Ok(Self::F64(v.clone()));
        }
        if let Ok(v) = obj.extract::<PyRef<'py, PyO3DataTileI64>>() {
            return Ok(Self::I64(v.clone()));
        }
        if let Ok(v) = obj.extract::<PyRef<'py, PyO3DataTileU64>>() {
            return Ok(Self::U64(v.clone()));
        }
        if let Ok(v) = obj.extract::<PyRef<'py, PyO3DataTileI32>>() {
            return Ok(Self::I32(v.clone()));
        }
        if let Ok(v) = obj.extract::<PyRef<'py, PyO3DataTileU32>>() {
            return Ok(Self::U32(v.clone()));
        }
        if let Ok(v) = obj.extract::<PyRef<'py, PyO3DataTileI16>>() {
            return Ok(Self::I16(v.clone()));
        }
        if let Ok(v) = obj.extract::<PyRef<'py, PyO3DataTileU16>>() {
            return Ok(Self::U16(v.clone()));
        }
        if let Ok(v) = obj.extract::<PyRef<'py, PyO3DataTileI8>>() {
            return Ok(Self::I8(v.clone()));
        }
        if let Ok(v) = obj.extract::<PyRef<'py, PyO3DataTileU8>>() {
            return Ok(Self::U8(v.clone()));
        }
        // if let Ok(v) = obj.extract::<PyRef<'py, PyO3DataTileBool>>() {
        //     return Ok(Self::Bool(v.clone()));
        // }
        // if let Ok(v) = obj.extract::<PyRef<'py, PyO3DataTileComplex>>() {
        //     return Ok(Self::Complex(v.clone()));
        // }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Unsupported PyO3DataTile type",
        ))
    }
}

impl PyO3DataTile {
    pub fn get_tile(&self) -> &PyO3Tile {
        match self {
            PyO3DataTile::F64(grid) => &grid._tile,
            PyO3DataTile::F32(grid) => &grid._tile,
            PyO3DataTile::I64(grid) => &grid._tile,
            PyO3DataTile::I32(grid) => &grid._tile,
            PyO3DataTile::I16(grid) => &grid._tile,
            PyO3DataTile::I8(grid) => &grid._tile,
            PyO3DataTile::U64(grid) => &grid._tile,
            PyO3DataTile::U32(grid) => &grid._tile,
            PyO3DataTile::U16(grid) => &grid._tile,
            PyO3DataTile::U8(grid) => &grid._tile,
        }
    }
}

#[derive(Clone)]
// #[enum_delegate::implement(GridTraits)]
pub enum PyO3Grid {
    PyO3TriGrid(PyO3TriGrid),
    PyO3RectGrid(PyO3RectGrid),
    PyO3HexGrid(PyO3HexGrid),
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

macro_rules! impl_to_data_tile {
    // This macro will create a function that converts to a `DataTile` of the given type
    ($name:ident, $type:ty, $data_tile_type:ident) => {
        fn $name<'py>(
            &self,
            data: PyReadonlyArray2<'py, $type>,
            nodata_value: $type,
        ) -> $data_tile_type {
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
            $data_tile_type {
                _data_tile: _data_tile,
                _tile: self.clone(),
            }
        }
    };
}

impl PyO3Tile {
    impl_to_data_tile!(_to_data_tile_f64, f64, PyO3DataTileF64);
    impl_to_data_tile!(_to_data_tile_f32, f32, PyO3DataTileF32);
    impl_to_data_tile!(_to_data_tile_i64, i64, PyO3DataTileI64);
    impl_to_data_tile!(_to_data_tile_i32, i32, PyO3DataTileI32);
    impl_to_data_tile!(_to_data_tile_u64, u64, PyO3DataTileU64);
    impl_to_data_tile!(_to_data_tile_u32, u32, PyO3DataTileU32);
    impl_to_data_tile!(_to_data_tile_i16, i16, PyO3DataTileI16);
    impl_to_data_tile!(_to_data_tile_u16, u16, PyO3DataTileU16);
    impl_to_data_tile!(_to_data_tile_i8, i8, PyO3DataTileI8);
    impl_to_data_tile!(_to_data_tile_u8, u8, PyO3DataTileU8);
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

    pub fn to_data_tile_f64<'py>(
        &self,
        data: PyReadonlyArray2<'py, f64>,
        nodata_value: f64,
    ) -> PyO3DataTileF64 {
        self._to_data_tile_f64(data, nodata_value)
    }

    pub fn to_data_tile_f32<'py>(
        &self,
        data: PyReadonlyArray2<'py, f32>,
        nodata_value: f32,
    ) -> PyO3DataTileF32 {
        self._to_data_tile_f32(data, nodata_value)
    }

    pub fn to_data_tile_i64<'py>(
        &self,
        data: PyReadonlyArray2<'py, i64>,
        nodata_value: i64,
    ) -> PyO3DataTileI64 {
        self._to_data_tile_i64(data, nodata_value)
    }

    pub fn to_data_tile_i32<'py>(
        &self,
        data: PyReadonlyArray2<'py, i32>,
        nodata_value: i32,
    ) -> PyO3DataTileI32 {
        self._to_data_tile_i32(data, nodata_value)
    }

    pub fn to_data_tile_u64<'py>(
        &self,
        data: PyReadonlyArray2<'py, u64>,
        nodata_value: u64,
    ) -> PyO3DataTileU64 {
        self._to_data_tile_u64(data, nodata_value)
    }

    pub fn to_data_tile_u32<'py>(
        &self,
        data: PyReadonlyArray2<'py, u32>,
        nodata_value: u32,
    ) -> PyO3DataTileU32 {
        self._to_data_tile_u32(data, nodata_value)
    }

    pub fn to_data_tile_i16<'py>(
        &self,
        data: PyReadonlyArray2<'py, i16>,
        nodata_value: i16,
    ) -> PyO3DataTileI16 {
        self._to_data_tile_i16(data, nodata_value)
    }

    pub fn to_data_tile_u16<'py>(
        &self,
        data: PyReadonlyArray2<'py, u16>,
        nodata_value: u16,
    ) -> PyO3DataTileU16 {
        self._to_data_tile_u16(data, nodata_value)
    }

    pub fn to_data_tile_i8<'py>(
        &self,
        data: PyReadonlyArray2<'py, i8>,
        nodata_value: i8,
    ) -> PyO3DataTileI8 {
        self._to_data_tile_i8(data, nodata_value)
    }

    pub fn to_data_tile_u8<'py>(
        &self,
        data: PyReadonlyArray2<'py, u8>,
        nodata_value: u8,
    ) -> PyO3DataTileU8 {
        self._to_data_tile_u8(data, nodata_value)
    }

    pub fn to_data_tile_with_value_f64<'py>(
        &self,
        fill_value: f64,
        nodata_value: f64,
    ) -> PyO3DataTileF64 {
        let _data_tile = self._tile.to_data_tile_with_value(fill_value, nodata_value);
        PyO3DataTileF64 {
            _data_tile,
            _tile: self.clone(),
        }
    }

    pub fn to_data_tile_with_value_f32<'py>(
        &self,
        fill_value: f32,
        nodata_value: f32,
    ) -> PyO3DataTileF32 {
        let _data_tile = self._tile.to_data_tile_with_value(fill_value, nodata_value);
        PyO3DataTileF32 {
            _data_tile,
            _tile: self.clone(),
        }
    }

    pub fn to_data_tile_with_value_i64<'py>(
        &self,
        fill_value: i64,
        nodata_value: i64,
    ) -> PyO3DataTileI64 {
        let _data_tile = self._tile.to_data_tile_with_value(fill_value, nodata_value);
        PyO3DataTileI64 {
            _data_tile,
            _tile: self.clone(),
        }
    }

    pub fn to_data_tile_with_value_i32<'py>(
        &self,
        fill_value: i32,
        nodata_value: i32,
    ) -> PyO3DataTileI32 {
        let _data_tile = self._tile.to_data_tile_with_value(fill_value, nodata_value);
        PyO3DataTileI32 {
            _data_tile,
            _tile: self.clone(),
        }
    }

    pub fn to_data_tile_with_value_u64<'py>(
        &self,
        fill_value: u64,
        nodata_value: u64,
    ) -> PyO3DataTileU64 {
        let _data_tile = self._tile.to_data_tile_with_value(fill_value, nodata_value);
        PyO3DataTileU64 {
            _data_tile,
            _tile: self.clone(),
        }
    }

    pub fn to_data_tile_with_value_u32<'py>(
        &self,
        fill_value: u32,
        nodata_value: u32,
    ) -> PyO3DataTileU32 {
        let _data_tile = self._tile.to_data_tile_with_value(fill_value, nodata_value);
        PyO3DataTileU32 {
            _data_tile,
            _tile: self.clone(),
        }
    }

    pub fn to_data_tile_with_value_i16<'py>(
        &self,
        fill_value: i16,
        nodata_value: i16,
    ) -> PyO3DataTileI16 {
        let _data_tile = self._tile.to_data_tile_with_value(fill_value, nodata_value);
        PyO3DataTileI16 {
            _data_tile,
            _tile: self.clone(),
        }
    }

    pub fn to_data_tile_with_value_u16<'py>(
        &self,
        fill_value: u16,
        nodata_value: u16,
    ) -> PyO3DataTileU16 {
        let _data_tile = self._tile.to_data_tile_with_value(fill_value, nodata_value);
        PyO3DataTileU16 {
            _data_tile,
            _tile: self.clone(),
        }
    }

    pub fn to_data_tile_with_value_i8<'py>(
        &self,
        fill_value: i8,
        nodata_value: i8,
    ) -> PyO3DataTileI8 {
        let _data_tile = self._tile.to_data_tile_with_value(fill_value, nodata_value);
        PyO3DataTileI8 {
            _data_tile,
            _tile: self.clone(),
        }
    }

    pub fn to_data_tile_with_value_u8<'py>(
        &self,
        fill_value: u8,
        nodata_value: u8,
    ) -> PyO3DataTileU8 {
        let _data_tile = self._tile.to_data_tile_with_value(fill_value, nodata_value);
        PyO3DataTileU8 {
            _data_tile,
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

    fn intersects<'py>(&self, _py: Python<'py>, other: &PyO3Tile) -> bool {
        self._tile.intersects(&other._tile)
    }

    fn overlap<'py>(&self, _py: Python<'py>, other: &PyO3Tile) -> PyResult<PyO3Tile> {
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

    fn anchor<'py>(
        &self,
        _py: Python<'py>,
        target_loc: (f64, f64),
        cell_element: String,
    ) -> PyO3TriGrid {
        let target_loc: [f64; 2] = target_loc.into();
        let cell_element =
            CellElement::from_string(&cell_element).expect("Unsupported cell_element {}");
        let mut new_grid = self.clone();
        new_grid._grid.anchor_inplace(&target_loc, cell_element);
        new_grid
    }

    fn anchor_inplace<'py>(
        &mut self,
        _py: Python<'py>,
        target_loc: (f64, f64),
        cell_element: String,
    ) {
        let target_loc: [f64; 2] = target_loc.into();
        let cell_element =
            CellElement::from_string(&cell_element).expect("Unsupported cell_element {}");
        self._grid.anchor_inplace(&target_loc, cell_element);
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

    fn cell_at_points<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<i64> {
        let points = points.as_array();
        self._grid.cell_at_points(&points).into_pyarray(py)
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

    fn anchor<'py>(
        &self,
        _py: Python<'py>,
        target_loc: (f64, f64),
        cell_element: String,
    ) -> PyO3RectGrid {
        let target_loc: [f64; 2] = target_loc.into();
        let cell_element =
            CellElement::from_string(&cell_element).expect("Unsupported cell_element {}");
        let mut new_grid = self.clone();
        new_grid._grid.anchor_inplace(&target_loc, cell_element);
        new_grid
    }

    fn anchor_inplace<'py>(
        &mut self,
        _py: Python<'py>,
        target_loc: (f64, f64),
        cell_element: String,
    ) {
        let target_loc: [f64; 2] = target_loc.into();
        let cell_element =
            CellElement::from_string(&cell_element).expect("Unsupported cell_element {}");
        self._grid.anchor_inplace(&target_loc, cell_element);
    }

    fn centroid<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray2<'py, i64>,
    ) -> &'py PyArray2<f64> {
        let index = index.as_array();
        self._grid.centroid(&index).into_pyarray(py)
    }

    fn cell_at_points<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<i64> {
        let points = points.as_array();
        self._grid.cell_at_points(&points).into_pyarray(py)
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

    fn anchor<'py>(
        &self,
        _py: Python<'py>,
        target_loc: (f64, f64),
        cell_element: String,
    ) -> PyO3HexGrid {
        let target_loc: [f64; 2] = target_loc.into();
        let cell_element = CellElement::from_string(&cell_element)
            .expect(&format!("Unsupported cell_element {}", &cell_element));
        let mut new_grid = self.clone();
        new_grid._grid.anchor_inplace(&target_loc, cell_element);
        new_grid
    }

    fn anchor_inplace<'py>(
        &mut self,
        _py: Python<'py>,
        target_loc: (f64, f64),
        cell_element: String,
    ) {
        let target_loc: [f64; 2] = target_loc.into();
        let cell_element = CellElement::from_string(&cell_element)
            .expect(&format!("Unsupported cell_element {}", &cell_element));
        self._grid.anchor_inplace(&target_loc, cell_element);
    }
    fn centroid<'py>(
        &self,
        py: Python<'py>,
        index: PyReadonlyArray2<'py, i64>,
    ) -> &'py PyArray2<f64> {
        let index = index.as_array();
        self._grid.centroid(&index).into_pyarray(py)
    }

    fn cell_at_points<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<i64> {
        let points = points.as_array();
        self._grid.cell_at_points(&points).into_pyarray(py)
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
fn combine_tiles<'py>(_py: Python<'py>, tiles: &PyList) -> PyResult<PyO3Tile> {
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
fn count_tiles<'py>(_py: Python<'py>, tiles: &PyList) -> PyResult<PyO3DataTileU64> {
    let tiles_vec: Vec<tile::Tile> = tiles
        .iter()
        .map(|item| {
            let py_o3_tile: PyO3Tile = item.extract().unwrap();
            py_o3_tile._tile.clone()
        })
        .collect();
    let _data_tile = tile::count_tiles(&tiles_vec);
    let _tile = tiles[0].extract::<PyO3Tile>().unwrap();
    let tile = PyO3DataTileU64 { _data_tile, _tile };
    Ok(tile)
}

#[pyfunction]
fn count_data_tiles<'py>(py: Python<'py>, data_tiles: &PyList) -> PyResult<PyO3DataTileU64> {
    // TODO: add check if tiles are aligned. Currently done in python but would be robust to add check here too.
    let tiles_as_coverage: Vec<_> = data_tiles
        .iter()
        .map(|item| {
            let pyo3_tile: PyO3DataTile = item.extract().unwrap();
            match pyo3_tile {
                PyO3DataTile::F64(data_tile) => data_tile._data_tile.data_cells_uint(),
                PyO3DataTile::F32(data_tile) => data_tile._data_tile.data_cells_uint(),
                PyO3DataTile::I64(data_tile) => data_tile._data_tile.data_cells_uint(),
                PyO3DataTile::I32(data_tile) => data_tile._data_tile.data_cells_uint(),
                PyO3DataTile::I16(data_tile) => data_tile._data_tile.data_cells_uint(),
                PyO3DataTile::I8(data_tile) => data_tile._data_tile.data_cells_uint(),
                PyO3DataTile::U64(data_tile) => data_tile._data_tile.data_cells_uint(),
                PyO3DataTile::U32(data_tile) => data_tile._data_tile.data_cells_uint(),
                PyO3DataTile::U16(data_tile) => data_tile._data_tile.data_cells_uint(),
                PyO3DataTile::U8(data_tile) => data_tile._data_tile.data_cells_uint(),
            }
        })
        .collect();

    let pyo3_tile: PyO3DataTile = data_tiles[0].extract().unwrap();
    let pyo3_grid = match pyo3_tile {
        PyO3DataTile::F64(pyo3_tile) => pyo3_tile._tile._grid.to_owned(),
        PyO3DataTile::F32(pyo3_tile) => pyo3_tile._tile._grid.to_owned(),
        PyO3DataTile::I64(pyo3_tile) => pyo3_tile._tile._grid.to_owned(),
        PyO3DataTile::I32(pyo3_tile) => pyo3_tile._tile._grid.to_owned(),
        PyO3DataTile::I16(pyo3_tile) => pyo3_tile._tile._grid.to_owned(),
        PyO3DataTile::I8(pyo3_tile) => pyo3_tile._tile._grid.to_owned(),
        PyO3DataTile::U64(pyo3_tile) => pyo3_tile._tile._grid.to_owned(),
        PyO3DataTile::U32(pyo3_tile) => pyo3_tile._tile._grid.to_owned(),
        PyO3DataTile::U16(pyo3_tile) => pyo3_tile._tile._grid.to_owned(),
        PyO3DataTile::U8(pyo3_tile) => pyo3_tile._tile._grid.to_owned(),
    };

    let combined_tile = tile::sum_data_tiles(&tiles_as_coverage);

    let _tile = PyO3Tile {
        _grid: pyo3_grid.clone(),
        start_id: combined_tile.get_tile().start_id,
        nx: combined_tile.get_tile().nx,
        ny: combined_tile.get_tile().ny,
        _tile: combined_tile.get_tile().to_owned(),
    };
    Ok(PyO3DataTileU64 {
        _data_tile: combined_tile,
        _tile,
    })
}

macro_rules! impl_sum_data_tile {
    ($func_name:ident, $PyO3DataTile:ident, $type:ty) => {
        #[pyfunction]
        fn $func_name<'py>(_py: Python<'py>, data_tiles: &PyList) -> PyResult<$PyO3DataTile> {
            // Note: I am returning PyObject because PyO3DataTile is an enum.
            //       I could return a more specific version like PyO3DataTileF64,
            //       but that would require a version of this function for each type.
            //       Using PyObject is easier but we lose static typing in Python.
            let tiles_vec: Vec<DataTile<$type>> = data_tiles
                .iter()
                .map(|item| {
                    let py_o3_tile: $PyO3DataTile = item.extract().unwrap();
                    py_o3_tile._data_tile
                })
                .collect();
            let _data_tile = tile::sum_data_tiles(&tiles_vec);
            let _tile = data_tiles[0]
                .extract::<$PyO3DataTile>()
                .unwrap()
                // .get_tile()
                ._tile;
            let tile = $PyO3DataTile { _data_tile, _tile };
            Ok(tile)
        }
    };
}

impl_sum_data_tile!(sum_data_tile_f64, PyO3DataTileF64, f64);
impl_sum_data_tile!(sum_data_tile_f32, PyO3DataTileF32, f32);
impl_sum_data_tile!(sum_data_tile_i64, PyO3DataTileI64, i64);
impl_sum_data_tile!(sum_data_tile_i32, PyO3DataTileI32, i32);
impl_sum_data_tile!(sum_data_tile_i16, PyO3DataTileI16, i16);
impl_sum_data_tile!(sum_data_tile_i8, PyO3DataTileI8, i8);
impl_sum_data_tile!(sum_data_tile_u64, PyO3DataTileU64, u64);
impl_sum_data_tile!(sum_data_tile_u32, PyO3DataTileU32, u32);
impl_sum_data_tile!(sum_data_tile_u16, PyO3DataTileU16, u16);
impl_sum_data_tile!(sum_data_tile_u8, PyO3DataTileU8, u8);

#[pymodule]
fn tile_utils(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(combine_tiles, module)?)?;
    module.add_function(wrap_pyfunction!(count_tiles, module)?)?;
    module.add_function(wrap_pyfunction!(count_data_tiles, module)?)?;
    module.add_function(wrap_pyfunction!(sum_data_tile_f64, module)?)?;
    module.add_function(wrap_pyfunction!(sum_data_tile_f32, module)?)?;
    module.add_function(wrap_pyfunction!(sum_data_tile_i64, module)?)?;
    module.add_function(wrap_pyfunction!(sum_data_tile_i32, module)?)?;
    module.add_function(wrap_pyfunction!(sum_data_tile_i16, module)?)?;
    module.add_function(wrap_pyfunction!(sum_data_tile_i8, module)?)?;
    module.add_function(wrap_pyfunction!(sum_data_tile_u64, module)?)?;
    module.add_function(wrap_pyfunction!(sum_data_tile_u32, module)?)?;
    module.add_function(wrap_pyfunction!(sum_data_tile_u16, module)?)?;
    module.add_function(wrap_pyfunction!(sum_data_tile_u8, module)?)?;
    Ok(())
}

#[pymodule]
fn gridkit_rs(_py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<PyO3TriGrid>()?;
    module.add_class::<PyO3RectGrid>()?;
    module.add_class::<PyO3HexGrid>()?;
    module.add_class::<PyO3Tile>()?;
    module.add_class::<PyO3DataTileF64>()?;
    module.add_class::<PyO3DataTileF32>()?;
    module.add_class::<PyO3DataTileI64>()?;
    module.add_class::<PyO3DataTileI32>()?;
    module.add_class::<PyO3DataTileI16>()?;
    module.add_class::<PyO3DataTileI8>()?;
    module.add_class::<PyO3DataTileU64>()?;
    module.add_class::<PyO3DataTileU32>()?;
    module.add_class::<PyO3DataTileU16>()?;
    module.add_class::<PyO3DataTileU8>()?;
    // TODO: add complex128 (in rust probably num_complex::Complex::Complex<f64>)
    // TODO: add support for boolean arrays
    module.add_wrapped(wrap_pymodule!(interp))?;
    module.add_wrapped(wrap_pymodule!(shapes))?;
    module.add_wrapped(wrap_pymodule!(tile_utils))?;
    Ok(())
}
