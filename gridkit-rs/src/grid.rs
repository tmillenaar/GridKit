use crate::hex_grid::*;
use crate::rect_grid::*;
use crate::tri_grid::*;
use enum_delegate;
use ndarray::*;

#[enum_delegate::register]
pub trait GridTraits {
    fn get_grid(&self) -> Grid;
    fn dx(&self) -> f64;
    fn dy(&self) -> f64;
    fn set_cellsize(&mut self, cellsize: f64);
    fn offset(&self) -> [f64; 2];
    fn set_offset(&mut self, offset: [f64; 2]);
    fn anchor(&self, target_loc: &[f64; 2], cell_element: CellElement) -> Grid {
        let mut grid = self.get_grid();
        grid.anchor_inplace(target_loc, cell_element);
        grid
    }
    fn anchor_inplace(&mut self, target_loc: &[f64; 2], cell_element: CellElement) {
        let target_loc = ArrayView1::<f64>::from(target_loc);
        let target_loc_2d = &target_loc.into_shape((1, 2)).unwrap();
        let current_cell = self.cell_at_points(&target_loc_2d);

        // Force rotation to zero before determining new offset
        let orig_rot = self.rotation();
        // let orig_target_loc = target_loc.clone();
        let target_loc = self.rotation_matrix_inv().dot(&target_loc);
        let target_loc_2d = target_loc.clone().into_shape((1, 2)).unwrap();
        self.set_rotation(0.);

        let diff = match cell_element {
            CellElement::Centroid => {
                // Get (dx, dy) to centroid of cell at target_loc
                let diff = &target_loc - &self.centroid(&current_cell.view());
                diff.slice(s![0, ..]).to_owned()
            }
            CellElement::Corner => {
                let corners = self.cell_corners(&current_cell.view());
                let corners = corners.slice(s![0, .., ..]); // Select first cell since we only have one (3D -> 2D)
                let diffs = &target_loc - &corners;
                let distances = diffs.mapv(|x| x.powi(2)).sum_axis(Axis(1)).mapv(f64::sqrt);

                // Get (dx, dy) to closest corner
                let mut min_dist = f64::MAX;
                let mut min_dist_id = 0;
                for i in 0..distances.len() {
                    if distances[Ix1(i)] < min_dist {
                        min_dist = distances[Ix1(i)];
                        min_dist_id = i;
                    }
                }
                diffs.slice(s![min_dist_id, ..]).to_owned()
            }
        };
        let mut new_offset = [
            self.offset()[0] + diff[Ix1(0)],
            self.offset()[1] + diff[Ix1(1)],
        ];
        self.set_offset(new_offset);

        match self.get_grid() {
            // A clone happens in `get_grid` which kinda defeats the point of the `inplace` bit of this function
            Grid::TriGrid(grid) => match cell_element {
                CellElement::Centroid => {
                    // Make sure if the target_loc was in an upright cell, the aligned centroid is also from an upright cell.
                    // Same for downward cells
                    let cell_after_diff_shift = grid.cell_at_points(&target_loc_2d.view());
                    if !(grid.is_cell_upright(&current_cell.view())
                        == grid.is_cell_upright(&cell_after_diff_shift.view()))
                    {
                        new_offset[grid.consistent_axis()] += grid.stepsize_consistent_axis();
                        self.set_offset(new_offset);
                    }
                }
                CellElement::Corner => {
                    // Check the target_loc actually intersects one of the new corners.
                    // If the offset got wrapped, we might need to shift by another dx.
                    let cell_after_diff_shift = grid.cell_at_points(&target_loc_2d.view());
                    let corners = grid.cell_corners(&cell_after_diff_shift.view());
                    let corners = corners.slice(s![0, .., ..]); // Select first cell since we only have one (3D -> 2D)
                    let diffs = &target_loc - &corners;
                    let distances = diffs
                        .mapv(|diff| diff.powi(2))
                        .sum_axis(Axis(1))
                        .mapv(f64::sqrt);
                    let mut is_target_on_corner = false;
                    for dist in distances.iter() {
                        if *dist < 10. * f64::EPSILON {
                            is_target_on_corner = true;
                        }
                    }
                    if !is_target_on_corner {
                        new_offset[grid.consistent_axis()] += grid.stepsize_consistent_axis();
                        self.set_offset(new_offset);
                    }
                }
            },
            Grid::RectGrid(grid) => {
                // nothing to do here
            }
            Grid::HexGrid(grid) => {
                // nothing to do here
            }
        }
        // Rotate original grid back
        self.set_rotation(orig_rot);
    }
    fn rotation(&self) -> f64;
    fn set_rotation(&mut self, rotation: f64);
    fn rotation_matrix(&self) -> &Array2<f64>;
    fn rotation_matrix_inv(&self) -> &Array2<f64>;
    fn radius(&self) -> f64;
    fn cell_height(&self) -> f64;
    fn cell_width(&self) -> f64;
    fn centroid_xy_no_rot(&self, x: i64, y: i64) -> [f64; 2];
    fn centroid(&self, index: &ArrayView2<i64>) -> Array2<f64>;
    fn cell_at_point(&self, point: &[f64; 2]) -> [i64; 2] {
        // Note: convenience function when dealing with a single point. This function
        // moves the point back and forth between the stack and heap so if you have multiple
        // points to process or already have an ndarray, use cell_at_points
        let point = ArrayView1::<f64>::from(point).into_shape((1, 2)).unwrap();
        let result = self.cell_at_points(&point);
        [result[Ix2(0, 0)], result[Ix2(0, 1)]]
    }
    fn cell_at_points(&self, points: &ArrayView2<f64>) -> Array2<i64>;
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
pub enum CellElement {
    Centroid,
    Corner,
}

impl ToString for CellElement {
    fn to_string(&self) -> String {
        match self {
            CellElement::Centroid => "centroid".to_string(),
            CellElement::Corner => "corner".to_string(),
        }
    }
}

impl CellElement {
    pub fn from_string(cell_element: &str) -> Option<Self> {
        match cell_element.to_lowercase().as_str() {
            "centroid" => Some(CellElement::Centroid),
            "corner" => Some(CellElement::Corner),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub enum Orientation {
    Flat,
    Pointy,
}

impl ToString for Orientation {
    fn to_string(&self) -> String {
        match self {
            Orientation::Flat => "flat".to_string(),
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
