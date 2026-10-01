use crate::hex_grid::*;
use crate::rect_grid::*;
use crate::tri_grid::*;
use crate::utils::{isclose, NUMERIC_ATOL, NUMERIC_RTOL};
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
    fn all_neighbours(
        &self,
        index: &ArrayView2<i64>,
        depth: i64,
        include_selected: bool,
        add_cell_id: bool,
    ) -> Array3<i64>;

    /// The cells within `depth` steps of `index` that are also within `depth` steps
    /// measured in the sense of the two axes of this grid type, so the diamond (for
    /// `RectGrid`) or rhombus (for `TriGrid`) shaped window.
    ///
    /// The arguments are the same as for [`GridTraits::all_neighbours`]. Mirrors
    /// `BaseGrid.neighbours(connect_corners=False)` and
    /// `BaseGrid.relative_neighbours(connect_corners=False)` in Python.
    fn direct_neighbours(
        &self,
        index: &ArrayView2<i64>,
        depth: i64,
        include_selected: bool,
        add_cell_id: bool,
    ) -> Array3<i64>;

    /// Check whether this grid is aligned with `other` and, if it is not, why.
    ///
    /// Grids are considered aligned when they are the same type of grid and
    /// their CRS, cellsize, offset and rotation are the same.
    ///
    /// Returns a tuple containing a boolean and a string. The boolean indicates
    /// whether or not the grids are aligned, the string contains the reason for
    /// the misalignment and is empty when the grids are aligned.
    ///
    /// Mirrors `BaseGrid.is_aligned_with` in `gridkit/base_grid.py`. Note that
    /// the CRS is not part of the Rust grid, so the CRS is not compared here.
    fn is_aligned_with(&self, other: &Grid) -> (bool, String);

    /// Create a new grid that is `factor` times finer than this grid and that
    /// aligns perfectly with it.
    ///
    /// The number of cells grows quadratically with `factor`: a `factor` of 2
    /// results in 4 cells that fit in the original, a `factor` of 3 in 9.
    ///
    /// The return type is the `Grid` enum rather than `Self` because
    /// `HexGrid.subdivide` returns a *triangular* grid: a hexagon cannot be
    /// tiled by smaller hexagons, so the only way to keep every corner of the
    /// parent cell on a corner of the sub grid is to fall back to triangles,
    /// which divide a hexagon into 6 `factor`² cells.
    ///
    /// `factor` is unsigned, so there are no negative factors. A `factor` of 0
    /// would divide the cellsize by zero, so it yields an unchanged copy of this
    /// grid rather than a grid with infinitely small cells. Note that Python's
    /// `RectGrid.subdivide` raises a `ValueError` for a `factor` below 1, so this
    /// is a deliberate divergence.
    ///
    /// Mirrors `RectGrid.subdivide` in `gridkit/rect_grid.py`,
    /// `TriGrid.subdivide` in `gridkit/tri_grid.py` and `HexGrid.subdivide` in
    /// `gridkit/hex_grid.py`.
    fn subdivide(&self, factor: u64) -> Grid;
    // fn linear_interpolation(
    //     &self,
    //     sample_points: &ArrayView2<f64>,
    //     nearby_value_locations: &ArrayView3<f64>,
    //     nearby_values: &ArrayView2<f64>,
    //     nodata_value: f64,
    // ) -> Array1<f64>;
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

/// The `cellsize` of a grid, which is the counterpart of Python's `size`.
///
/// `RectGrid` is included for exhaustiveness, even though it compares `dx`/`dy`
/// in its own `is_aligned_with`.
fn cellsize_of(grid: &Grid) -> f64 {
    match grid {
        Grid::TriGrid(grid) => grid.cellsize,
        Grid::RectGrid(grid) => grid.dx(),
        Grid::HexGrid(grid) => grid.cellsize,
    }
}

/// The `orientation` of a grid, or `None` for a `RectGrid`.
///
/// Mirrors Python's `getattr(grid, "orientation", "")`, where `RectGrid` has no
/// `orientation` and falls back to the empty string.
fn orientation_of(grid: &Grid) -> Option<&Orientation> {
    match grid {
        Grid::TriGrid(grid) => Some(&grid.orientation),
        Grid::RectGrid(_) => None,
        Grid::HexGrid(grid) => Some(&grid.orientation),
    }
}

/// Shared implementation of [`GridTraits::is_aligned_with`] for the grid types
/// that have a `cellsize` and an `orientation`.
///
/// `RectGrid` compares `dx`/`dy` and has no orientation, so it implements its own
/// version. Note that the CRS is not part of the Rust grid, so the CRS check of
/// `BaseGrid.is_aligned_with` has no counterpart here.
///
/// Mirrors `BaseGrid.is_aligned_with` in `gridkit/base_grid.py`.
pub(crate) fn is_aligned_with(
    grid: &Grid,
    other: &Grid,
    cellsize: f64,
    offset: [f64; 2],
    orientation: &Orientation,
) -> (bool, String) {
    // Python returns early on a type mismatch rather than accumulating reasons,
    // so only the grid type is reported in that case.
    if std::mem::discriminant(grid) != std::mem::discriminant(other) {
        return (
            false,
            format!(
                "Grid type is not the same. This is a {}, the other is a {}",
                grid_type_name(grid),
                grid_type_name(other)
            ),
        );
    }

    let mut reasons: Vec<&str> = Vec::new();

    // Hex and triangular grids always have a `size`, so Python takes the
    // `numpy.isclose(self.size, other.size)` branch rather than the `dx`/`dy`
    // fallback that `RectGrid` uses. `cellsize` is the Rust counterpart of that
    // `size`.
    if !isclose(cellsize, cellsize_of(other), NUMERIC_RTOL, NUMERIC_ATOL) {
        reasons.push("cellsize");
    }

    // FIXME: the 1e-7 tolerance is a bandaid, taken from Python. The offset
    //        seems to depend slightly on the bounds after resampling.
    if !(isclose(offset[0], other.offset()[0], NUMERIC_RTOL, 1e-7)
        && isclose(offset[1], other.offset()[1], NUMERIC_RTOL, 1e-7))
    {
        reasons.push("offset");
    }

    // Unlike `RectGrid`, both grid types here expose an `orientation`, which
    // Python compares via `getattr(self, "orientation", "")`. A `RectGrid` on
    // either side has already been rejected by the type check above, so the
    // orientations are always both present here.
    if orientation_of(other).is_some_and(|other| orientation != other) {
        reasons.push("orientation");
    }

    if grid.rotation() != other.rotation() {
        reasons.push("rotation");
    }

    if reasons.is_empty() {
        return (true, String::new());
    }
    (
        false,
        format!("The following attributes are not the same: {reasons:?}"),
    )
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
