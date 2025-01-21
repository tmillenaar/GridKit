use crate::hex_grid::*;
use crate::rect_grid::*;
use crate::tri_grid::*;
use enum_delegate;
use numpy::ndarray::*;

#[enum_delegate::register]
pub trait GridTraits {
    fn dx(&self) -> f64;
    fn dy(&self) -> f64;
    fn set_cellsize(&mut self, cellsize: f64);
    fn offset(&self) -> (f64, f64);
    fn set_offset(&mut self, offset: (f64, f64));
    fn rotation(&self) -> f64;
    fn set_rotation(&mut self, rotation: f64);
    fn rotation_matrix(&self) -> &Array2<f64>;
    fn rotation_matrix_inv(&self) -> &Array2<f64>;
    fn radius(&self) -> f64;
    fn cell_height(&self) -> f64;
    fn cell_width(&self) -> f64;
    fn centroid_xy_no_rot(&self, x: i64, y: i64) -> (f64, f64);
    fn centroid(&self, index: &ArrayView2<i64>) -> Array2<f64>;
    fn cell_at_point(&self, points: &ArrayView2<f64>) -> Array2<i64>;
    fn cell_corners(&self, index: &ArrayView2<i64>) -> Array3<f64>;
    fn cells_near_point(&self, points: &ArrayView2<f64>) -> Array3<i64>;
    // fn linear_interpolation(
    //     &self,
    //     sample_points: &ArrayView2<f64>,
    //     nearby_value_locations: &ArrayView3<f64>,
    //     nearby_values: &ArrayView2<f64>,
    //     nodata_value: f64,
    // ) -> Array1<f64>;
}

#[derive(Clone)]
#[enum_delegate::implement(GridTraits)]
pub enum Grid {
    TriGrid(TriGrid),
    RectGrid(RectGrid),
    HexGrid(HexGrid),
}

#[derive(Clone)]
pub enum Orientation {
    Flat,
    Pointy,
}

impl ToString for Orientation {
    fn to_string(&self) -> String {
        match self {
            Orientation::Flat => "falt".to_string(),
            Orientation::Pointy => "pointy".to_string(),
        }
    }
}

impl Orientation {
    pub fn from_string(cell_orientation: &str) -> Option<Self> {
        match cell_orientation.to_lowercase().as_str() {
            "flat" => Some(Orientation::Flat),
            "pointy" => Some(Orientation::Pointy),
            _ => None,
        }
    }
}
