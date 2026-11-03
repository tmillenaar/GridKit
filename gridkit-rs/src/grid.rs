use crate::hex_grid::*;
use crate::rect_grid::*;
use crate::tri_grid::*;
use crate::utils::{is_close, NUMERIC_ATOL, NUMERIC_RTOL};
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
    fn anchor(&self, target_loc: &[f64; 2], cell_element: CellElement) -> Grid
    where
        Self: Sync,
    {
        let mut grid = self.get_grid();
        grid.anchor_inplace(target_loc, cell_element);
        grid
    }
    fn anchor_inplace(&mut self, target_loc: &[f64; 2], cell_element: CellElement)
    where
        Self: Sync,
    {
        let target_loc = ArrayView1::<f64>::from(target_loc);
        // Keep the internal single-point representation 2D. The public array
        // methods accept arbitrary shapes, but the anchoring logic below uses
        // an explicit leading cell axis.
        let target_loc_2d = target_loc.into_shape((1, 2)).unwrap();
        let current_cell = self.cell_at_points(target_loc_2d);

        // Force rotation to zero before determining new offset
        let orig_rot = self.rotation();
        // let orig_target_loc = target_loc.clone();
        let target_loc = self.rotation_matrix_inv().dot(&target_loc);
        let target_loc_2d = target_loc.clone().into_shape((1, 2)).unwrap();
        self.set_rotation(0.);

        let diff = match cell_element {
            CellElement::Centroid => {
                // Get (dx, dy) to centroid of cell at target_loc
                let diff = &target_loc - &self.centroid(current_cell.view());
                diff.slice(s![0, ..]).to_owned()
            }
            CellElement::Corner => {
                let corners = self.cell_corners(current_cell.view());
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
                    let cell_after_diff_shift = grid.cell_at_points(target_loc_2d.view());
                    if !(grid.is_cell_upright(current_cell.view())
                        == grid.is_cell_upright(cell_after_diff_shift.view()))
                    {
                        new_offset[grid.consistent_axis()] += grid.stepsize_consistent_axis();
                        self.set_offset(new_offset);
                    }
                }
                CellElement::Corner => {
                    // Check the target_loc actually intersects one of the new corners.
                    // If the offset got wrapped, we might need to shift by another dx.
                    let cell_after_diff_shift = grid.cell_at_points(target_loc_2d.view());
                    let corners = grid.cell_corners(cell_after_diff_shift.view());
                    let corners = corners.slice(s![0, .., ..]); // Select first cell since we only have one (3D -> 2D)
                    let diffs = &target_loc - &corners;
                    let distances = diffs
                        .mapv(|diff| diff.powi(2))
                        .sum_axis(Axis(diffs.ndim() - 1))
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
            Grid::RectGrid(_grid) => {
                // nothing to do here
            }
            Grid::HexGrid(_grid) => {
                // nothing to do here
            }
        }
        // Rotate original grid back
        self.set_rotation(orig_rot);
    }
    fn rotation(&self) -> f64;
    fn set_rotation(&mut self, rotation: f64);
    fn rotation_matrix(&self) -> ArrayView2<f64>;
    fn rotation_matrix_inv(&self) -> ArrayView2<f64>;
    fn radius(&self) -> f64;
    fn cell_height(&self) -> f64;
    fn cell_width(&self) -> f64;
    fn centroid_xy_no_rot(&self, x: i64, y: i64) -> [f64; 2];

    /// Coordinates at the center of the cell(s) specified by `index`.
    ///
    /// `index` may have any shape as long as its last axis is of length 2 and
    /// holds the `(x, y)` cell ids. The result has the same shape as `index`.
    /// This is the Rust equivalent of the numpy `ravel` + `reshape` dance in
    /// the Python implementation.
    fn centroid<D>(&self, index: ArrayView<i64, D>) -> Array<f64, D>
    where
        D: Dimension,
        Self: Sync,
    {
        crate::utils::map_point_pairs_batched(index, |index| {
            let mut centroids = Array2::<f64>::zeros((index.shape()[0], 2));
            for cell_id in 0..centroids.shape()[0] {
                let point = self.centroid_xy_no_rot(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
                centroids[Ix2(cell_id, 0)] = point[0];
                centroids[Ix2(cell_id, 1)] = point[1];
            }
            if self.rotation() != 0. {
                let rotation_matrix = self.rotation_matrix();
                // Applying the rotation by hand avoids the temporary allocation
                // that a `rotation_matrix.dot(&centroid)` call would make per cell.
                let cos = rotation_matrix[[0, 0]];
                let sin = rotation_matrix[[1, 0]];
                for cell_id in 0..centroids.shape()[0] {
                    let x = centroids[Ix2(cell_id, 0)];
                    let y = centroids[Ix2(cell_id, 1)];
                    centroids[Ix2(cell_id, 0)] = cos * x - sin * y;
                    centroids[Ix2(cell_id, 1)] = sin * x + cos * y;
                }
            }
            centroids
        })
    }

    fn cell_at_point(&self, point: &[f64; 2]) -> [i64; 2] {
        // Note: convenience function when dealing with a single point. This function
        // moves the point back and forth between the stack and heap so if you have multiple
        // points to process or already have an ndarray, use cell_at_points
        let result = self.cell_at_points(point.into());
        [result[Ix1(0)], result[Ix1(1)]]
    }

    /// The id of the cell each point falls in.
    ///
    /// `points` may have any shape as long as its last axis is of length 2 and
    /// holds the `(x, y)` coordinates. The result has the same shape as `points`.
    fn cell_at_points<D>(&self, points: ArrayView<f64, D>) -> Array<i64, D>
    where
        D: Dimension;

    /// Coordinates of the corners of the cell(s) specified by `index`.
    ///
    /// `index` may have any shape as long as its last axis is of length 2 and
    /// holds the `(x, y)` cell ids. The result has one dimension more than
    /// `index`: its leading axes match `index`, followed by `(corner, xy)`.
    fn cell_corners<D>(&self, index: ArrayView<i64, D>) -> Array<f64, D::Larger>
    where
        D: Dimension;

    /// The cells nearest to the point(s), often used for interpolation.
    ///
    /// `points` may have any shape as long as its last axis is of length 2.
    /// The result has one dimension more than `points`: its leading axes match,
    /// followed by `(nearby_cell, xy)`.
    fn cells_near_point<D>(&self, points: ArrayView<f64, D>) -> Array<i64, D::Larger>
    where
        D: Dimension;

    /// All cells within `depth` steps of `index`, as the full window of cells
    /// around it that stepping along both axes can reach.
    ///
    /// Set `include_selected` to keep the cells of `index` themselves in the
    /// result; they always sit in the middle of the window. Set `add_cell_id` to
    /// offset the result by the ids in `index` (as `neighbours` does) rather than
    /// returning the relative offsets (as `relative_neighbours` does).
    ///
    /// Mirrors `BaseGrid.neighbours(connect_corners=True)` and
    /// `BaseGrid.relative_neighbours(connect_corners=True)` in Python. The shape
    /// of the window is grid specific: a square for `RectGrid`, a hexagon for
    /// `TriGrid`.
    fn all_neighbours<D>(
        &self,
        index: ArrayView<i64, D>,
        depth: u64,
        include_selected: bool,
        add_cell_id: bool,
    ) -> Array<i64, D::Larger>
    where
        D: Dimension;

    /// The cells within `depth` steps of `index` that are also within `depth` steps
    /// measured in the sense of the two axes of this grid type, so the diamond (for
    /// `RectGrid`) or rhombus (for `TriGrid`) shaped window.
    ///
    /// The arguments are the same as for [`GridTraits::all_neighbours`]. Mirrors
    /// `BaseGrid.neighbours(connect_corners=False)` and
    /// `BaseGrid.relative_neighbours(connect_corners=False)` in Python.
    fn direct_neighbours<D>(
        &self,
        index: ArrayView<i64, D>,
        depth: u64,
        include_selected: bool,
        add_cell_id: bool,
    ) -> Array<i64, D::Larger>
    where
        D: Dimension;

    /// Check whether this grid is aligned with `other`.
    ///
    /// Grids are considered aligned when they are the same type of grid and
    /// their cellsize/dx-dy, offset and rotation are the same.
    fn is_aligned_with(&self, other: &Grid) -> bool;

    /// Create a new grid that is `factor` times finer than this grid and that
    /// aligns perfectly with it.
    ///
    /// The number of cells grows quadratically with `factor`: a `factor` of 2
    /// results in 4 cells that fit in the original, a `factor` of 3 in 9.
    ///
    /// A `factor` of 0 would divide the cellsize by zero, so it yields an unchanged
    /// copy of this grid rather than a grid with infinitely small cells.
    fn subdivide(&self, factor: u64) -> Grid;
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[enum_delegate::implement(GridTraits)]
pub enum Grid {
    TriGrid(TriGrid),
    RectGrid(RectGrid),
    HexGrid(HexGrid),
}

/// The name of the concrete grid type, as used in the Python class names.
///
/// Used to report grid type mismatches, mirroring the `parent_grid_class`
/// comparison in `BaseGrid.is_aligned_with`.
pub fn grid_type_name(grid: &Grid) -> &'static str {
    match grid {
        Grid::TriGrid(_) => "TriGrid",
        Grid::RectGrid(_) => "RectGrid",
        Grid::HexGrid(_) => "HexGrid",
    }
}

#[derive(Clone)]
pub enum CellElement {
    Centroid,
    Corner,
}

#[cfg(all(test, feature = "parallel"))]
mod parallel_batch_tests {
    use super::*;
    use std::time::Instant;

    const POINTS: usize = 2_097_152;

    fn assert_parallel_speed<T, F>(name: &str, call: F)
    where
        T: std::fmt::Debug + PartialEq,
        F: Fn() -> T,
    {
        crate::utils::set_num_threads(Some(1)).unwrap();
        let start = Instant::now();
        let serial = call();
        let serial_time = start.elapsed();

        crate::utils::set_num_threads(Some(2)).unwrap();
        // Exclude one-time Rayon worker startup from the measured query.
        let _ = call();
        let start = Instant::now();
        let parallel = call();
        let parallel_time = start.elapsed();

        eprintln!(
            "{name}: 1 thread = {:.3}s, 2 threads = {:.3}s, speedup = {:.2}x",
            serial_time.as_secs_f64(),
            parallel_time.as_secs_f64(),
            serial_time.as_secs_f64() / parallel_time.as_secs_f64()
        );
        assert_eq!(serial, parallel, "{name} changed its result or ordering");
        assert!(
            serial_time >= parallel_time.mul_f32(1.5),
            "{name} did not achieve the required 1.5x speedup"
        );
    }

    fn test_grid_methods(
        grid_name: &str,
        mut grid: Grid,
        indices: &Array2<i64>,
        points: &Array2<f64>,
    ) {
        grid.set_rotation(31.);

        assert_parallel_speed(&format!("{grid_name}/centroid"), || {
            grid.centroid(indices.view()).into_dyn()
        });
        assert_parallel_speed(&format!("{grid_name}/cell_at_points"), || {
            grid.cell_at_points(points.view()).into_dyn()
        });
        assert_parallel_speed(&format!("{grid_name}/cell_corners"), || {
            grid.cell_corners(indices.view()).into_dyn()
        });
        assert_parallel_speed(&format!("{grid_name}/cells_near_point"), || {
            grid.cells_near_point(points.view()).into_dyn()
        });
        assert_parallel_speed(&format!("{grid_name}/all_neighbours"), || {
            grid.all_neighbours(indices.view(), 1, false, true).into_dyn()
        });
        assert_parallel_speed(&format!("{grid_name}/direct_neighbours"), || {
            grid.direct_neighbours(indices.view(), 1, false, true).into_dyn()
        });
    }

    #[test]
    #[ignore = "performance test; run with --ignored --nocapture"]
    fn batched_grid_methods_parallelize_without_reordering() {
        let indices = Array2::from_shape_fn((POINTS, 2), |(row, axis)| {
            if axis == 0 {
                (row % 2048) as i64 - 1024
            } else {
                (row / 2048) as i64 - 128
            }
        });
        let points = indices.mapv(|value| value as f64 + 0.37);

        test_grid_methods(
            "rect",
            Grid::RectGrid(RectGrid::new(12.3, 8.7)),
            &indices,
            &points,
        );
        test_grid_methods(
            "hex",
            Grid::HexGrid(HexGrid::new(12.3, Orientation::Flat)),
            &indices,
            &points,
        );
        test_grid_methods(
            "tri",
            Grid::TriGrid(TriGrid::new(12.3, Orientation::Flat)),
            &indices,
            &points,
        );
    }
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

#[derive(Clone, Debug, Hash, PartialEq)]
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
    pub fn from_string(orientation: &str) -> Option<Self> {
        match orientation.to_lowercase().as_str() {
            "flat" => Some(Orientation::Flat),
            "pointy" => Some(Orientation::Pointy),
            _ => None,
        }
    }
}
