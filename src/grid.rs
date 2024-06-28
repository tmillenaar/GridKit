use numpy::ndarray::*;
use crate::tri_grid::*;
use crate::rect_grid::*;
use crate::hex_grid::*;
use enum_delegate;

#[enum_delegate::register]
pub trait GridTraits {
    fn dx(&self) -> f64;
    fn dy(&self) -> f64;
    fn offset(&self) -> (f64, f64);
    fn radius(&self) -> f64;
    fn rotation(&self) -> f64;
    fn rotation_matrix(&self) -> Array2<f64>;
    fn rotation_matrix_inv(&self) -> Array2<f64>;
    fn centroid_xy_no_rot(&self, x: i64, y: i64) -> (f64, f64);
    fn centroid(&self, index: &ArrayView2<i64>) -> Array2<f64>;
}

#[derive(Clone)]
#[enum_delegate::implement(GridTraits)]
pub enum Grid {
    TriGrid(TriGrid),
    RectGrid(RectGrid),
    HexGrid(HexGrid),
}
