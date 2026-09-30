use std::hash::{Hash, Hasher};

use crate::grid::GridTraits;
use crate::utils::*;
use ndarray::*;

#[derive(Clone, Debug)]
pub struct RectGrid {
    pub _dx: f64,
    pub _dy: f64,
    pub offset: [f64; 2],
    pub _rotation: f64,
    pub _rotation_matrix: Array2<f64>,
    pub _rotation_matrix_inv: Array2<f64>,
}

impl PartialEq for RectGrid {
    // Needs manual implementation, derive PartialEq does not work on floats because of NaN etc.
    fn eq(&self, other: &Self) -> bool {
        self._dx.to_bits() == other._dx.to_bits() &&
        self._dy.to_bits() == other._dy.to_bits() &&
        self.offset[0].to_bits() == other.offset[0].to_bits() &&
        self.offset[1].to_bits() == other.offset[1].to_bits() &&
        self._rotation.to_bits() == other._rotation.to_bits()
    }
}

impl Eq for RectGrid {}

impl Hash for RectGrid {
    // Needs manual implementation, derive Hash does not work on floats because of NaN etc.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self._dx.to_bits().hash(state);
        self._dy.to_bits().hash(state);
        self.offset[0].to_bits().hash(state);
        self.offset[1].to_bits().hash(state);
        self._rotation.to_bits().hash(state);
    }
}

impl GridTraits for RectGrid {
    fn get_grid(&self) -> crate::Grid {
        crate::Grid::RectGrid(self.to_owned())
    }

    fn dx(&self) -> f64 {
        self._dx
    }
    fn dy(&self) -> f64 {
        self._dy
    }

    fn set_cellsize(&mut self, cellsize: f64) {
        self._dx = cellsize;
        self._dy = cellsize;
    }

    fn offset(&self) -> [f64; 2] {
        self.offset
    }

    fn set_offset(&mut self, offset: [f64; 2]) {
        self.offset = normalize_offset(offset, self.cell_width(), self.cell_height())
    }

    fn rotation(&self) -> f64 {
        self._rotation
    }
    fn set_rotation(&mut self, rotation: f64) {
        self._rotation = rotation;
        self._rotation_matrix = rotation_matrix_from_angle(rotation);
        self._rotation_matrix_inv = rotation_matrix_from_angle(-rotation);
    }
    fn rotation_matrix(&self) -> &Array2<f64> {
        &self._rotation_matrix
    }
    fn rotation_matrix_inv(&self) -> &Array2<f64> {
        &self._rotation_matrix_inv
    }

    fn radius(&self) -> f64 {
        ((self._dx / 2.).powi(2) + (self._dy / 2.).powi(2)).powf(0.5)
    }

    fn centroid_xy_no_rot(&self, x: i64, y: i64) -> [f64; 2] {
        let centroid_x = x as f64 * self.dx() + (self.dx() / 2.) + self.offset[0];
        let centroid_y = y as f64 * self.dy() + (self.dy() / 2.) + self.offset[1];
        [centroid_x, centroid_y]
    }
    fn centroid(&self, index: &ArrayView2<i64>) -> Array2<f64> {
        let mut centroids = Array2::<f64>::zeros((index.shape()[0], 2));

        for cell_id in 0..centroids.shape()[0] {
            let point = self.centroid_xy_no_rot(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
            centroids[Ix2(cell_id, 0)] = point[0];
            centroids[Ix2(cell_id, 1)] = point[1];
        }

        if self.rotation() != 0. {
            for cell_id in 0..centroids.shape()[0] {
                let mut centroid = centroids.slice_mut(s![cell_id, ..]);
                let cent_rot = self._rotation_matrix.dot(&centroid);
                centroid.assign(&cent_rot);
            }
        }
        centroids
    }

    fn cell_height(&self) -> f64 {
        self._dy
    }

    fn cell_width(&self) -> f64 {
        self._dx
    }

    fn cell_at_points(&self, points: &ArrayView2<f64>) -> Array2<i64> {
        let shape = points.shape();
        let mut index = Array2::<i64>::zeros((shape[0], shape[1]));
        for cell_id in 0..points.shape()[0] {
            let point = points.slice(s![cell_id, ..]);
            let point = self._rotation_matrix_inv.dot(&point);
            let id_x = ((point[Ix1(0)] - self.offset[0]) / self.dx()).floor() as i64;
            let id_y = ((point[Ix1(1)] - self.offset[1]) / self.dy()).floor() as i64;
            index[Ix2(cell_id, 0)] = id_x;
            index[Ix2(cell_id, 1)] = id_y;
        }
        index
    }

    fn cell_corners(&self, index: &ArrayView2<i64>) -> Array3<f64> {
        let mut corners = Array3::<f64>::zeros((index.shape()[0], 4, 2));
        for cell_id in 0..index.shape()[0] {
            let id_x = index[Ix2(cell_id, 0)];
            let id_y = index[Ix2(cell_id, 1)];
            let [centroid_x, centroid_y] = self.centroid_xy_no_rot(id_x, id_y);
            corners[Ix3(cell_id, 0, 0)] = centroid_x - self.dx() / 2.;
            corners[Ix3(cell_id, 0, 1)] = centroid_y - self.dy() / 2.;
            corners[Ix3(cell_id, 1, 0)] = centroid_x + self.dx() / 2.;
            corners[Ix3(cell_id, 1, 1)] = centroid_y - self.dy() / 2.;
            corners[Ix3(cell_id, 2, 0)] = centroid_x + self.dx() / 2.;
            corners[Ix3(cell_id, 2, 1)] = centroid_y + self.dy() / 2.;
            corners[Ix3(cell_id, 3, 0)] = centroid_x - self.dx() / 2.;
            corners[Ix3(cell_id, 3, 1)] = centroid_y + self.dy() / 2.;
        }

        if self.rotation() != 0. {
            for cell_id in 0..corners.shape()[0] {
                for corner_id in 0..corners.shape()[1] {
                    let mut corner_xy = corners.slice_mut(s![cell_id, corner_id, ..]);
                    let rotated_corner_xy = self._rotation_matrix.dot(&corner_xy);
                    corner_xy.assign(&rotated_corner_xy);
                }
            }
        }
        corners
    }

    fn cells_near_point(&self, points: &ArrayView2<f64>) -> Array3<i64> {
        let mut nearby_cells = Array3::<i64>::zeros((points.shape()[0], 4, 2));
        let index = self.cell_at_points(points);

        // FIXME: Find a way to not clone points in the case of no rotation
        //        If points is made mutable within the conditional, it is dropped from scope and nothing changed
        let mut points = points.to_owned();
        if self.rotation() != 0. {
            for cell_id in 0..points.shape()[0] {
                let mut point = points.slice_mut(s![cell_id, ..]);
                let point_rot = self._rotation_matrix_inv.dot(&point);
                point.assign(&point_rot);
            }
        }

        for cell_id in 0..points.shape()[0] {
            let rel_loc_x: f64 = modulus(points[Ix2(cell_id, 0)] - self.offset[0], self.dx());
            let rel_loc_y: f64 = modulus(points[Ix2(cell_id, 1)] - self.offset[1], self.dy());
            let id_x = index[Ix2(cell_id, 0)];
            let id_y = index[Ix2(cell_id, 1)];
            match (rel_loc_x, rel_loc_y) {
                // Top-left quadrant
                (x, y) if x <= self.dx() / 2. && y >= self.dy() / 2. => {
                    nearby_cells[Ix3(cell_id, 0, 0)] = -1 + id_x;
                    nearby_cells[Ix3(cell_id, 0, 1)] = 1 + id_y;
                    nearby_cells[Ix3(cell_id, 1, 0)] = 0 + id_x;
                    nearby_cells[Ix3(cell_id, 1, 1)] = 1 + id_y;
                    nearby_cells[Ix3(cell_id, 2, 0)] = -1 + id_x;
                    nearby_cells[Ix3(cell_id, 2, 1)] = 0 + id_y;
                    nearby_cells[Ix3(cell_id, 3, 0)] = 0 + id_x;
                    nearby_cells[Ix3(cell_id, 3, 1)] = 0 + id_y;
                }
                // Top-right quadrant
                (x, y) if x >= self.dx() / 2. && y >= self.dy() / 2. => {
                    nearby_cells[Ix3(cell_id, 0, 0)] = 0 + id_x;
                    nearby_cells[Ix3(cell_id, 0, 1)] = 1 + id_y;
                    nearby_cells[Ix3(cell_id, 1, 0)] = 1 + id_x;
                    nearby_cells[Ix3(cell_id, 1, 1)] = 1 + id_y;
                    nearby_cells[Ix3(cell_id, 2, 0)] = 0 + id_x;
                    nearby_cells[Ix3(cell_id, 2, 1)] = 0 + id_y;
                    nearby_cells[Ix3(cell_id, 3, 0)] = 1 + id_x;
                    nearby_cells[Ix3(cell_id, 3, 1)] = 0 + id_y;
                }
                // Bottom-left quadrant
                (x, y) if x <= self.dx() / 2. && y <= self.dy() / 2. => {
                    nearby_cells[Ix3(cell_id, 0, 0)] = -1 + id_x;
                    nearby_cells[Ix3(cell_id, 0, 1)] = 0 + id_y;
                    nearby_cells[Ix3(cell_id, 1, 0)] = 0 + id_x;
                    nearby_cells[Ix3(cell_id, 1, 1)] = 0 + id_y;
                    nearby_cells[Ix3(cell_id, 2, 0)] = -1 + id_x;
                    nearby_cells[Ix3(cell_id, 2, 1)] = -1 + id_y;
                    nearby_cells[Ix3(cell_id, 3, 0)] = 0 + id_x;
                    nearby_cells[Ix3(cell_id, 3, 1)] = -1 + id_y;
                }
                // Bottom-right quadrant
                _ => {
                    nearby_cells[Ix3(cell_id, 0, 0)] = 0 + id_x;
                    nearby_cells[Ix3(cell_id, 0, 1)] = 0 + id_y;
                    nearby_cells[Ix3(cell_id, 1, 0)] = 1 + id_x;
                    nearby_cells[Ix3(cell_id, 1, 1)] = 0 + id_y;
                    nearby_cells[Ix3(cell_id, 2, 0)] = 0 + id_x;
                    nearby_cells[Ix3(cell_id, 2, 1)] = -1 + id_y;
                    nearby_cells[Ix3(cell_id, 3, 0)] = 1 + id_x;
                    nearby_cells[Ix3(cell_id, 3, 1)] = -1 + id_y;
                }
            }
        }
        nearby_cells
    }
}

impl RectGrid {
    pub fn new(dx: f64, dy: f64) -> Self {
        let _rotation_matrix = rotation_matrix_from_angle(0.);
        let _rotation_matrix_inv = rotation_matrix_from_angle(-0.);
        RectGrid {
            _dx: dx,
            _dy: dy,
            offset: [0., 0.],
            _rotation: 0.,
            _rotation_matrix,
            _rotation_matrix_inv,
        }
    }

    /// Set the cellsize in the x-direction.
    ///
    /// The cellsize must be larger than zero, mirroring the validation in
    /// `gridkit/rect_grid.py`'s `dx` setter.
    pub fn set_cellsize_x(&mut self, cellsize_x: f64) -> Result<(), String> {
        if cellsize_x <= 0. {
            return Err(format!(
                "Size of cell cannot be set to '{cellsize_x}', must be larger than zero"
            ));
        }
        self._dx = cellsize_x;
        Ok(())
    }

    /// Set the cellsize in the y-direction.
    ///
    /// The cellsize must be larger than zero, mirroring the validation in
    /// `gridkit/rect_grid.py`'s `dy` setter.
    pub fn set_cellsize_y(&mut self, cellsize_y: f64) -> Result<(), String> {
        if cellsize_y <= 0. {
            return Err(format!(
                "Size of cell cannot be set to '{cellsize_y}', must be larger than zero"
            ));
        }
        self._dy = cellsize_y;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::CellElement;

    const TOL: f64 = 1e-9;

    // ---------------------------------------------------------------------
    // Assertion helpers
    // ---------------------------------------------------------------------

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {}, got {} (tolerance {})",
            expected,
            actual,
            tol
        );
    }

    /// Compare an (n, 2) f64 array against a flat list of expected values.
    fn assert_xy_close(actual: &Array2<f64>, expected: &[f64], tol: f64) {
        assert_eq!(actual.shape(), &[expected.len() / 2, 2], "unexpected shape");
        for (i, exp) in expected.chunks(2).enumerate() {
            assert_close(actual[[i, 0]], exp[0], tol);
            assert_close(actual[[i, 1]], exp[1], tol);
        }
    }

    /// Compare an (n, 2) i64 array against a flat list of expected values.
    fn assert_ids(actual: &Array2<i64>, expected: &[i64]) {
        assert_eq!(actual.shape(), &[expected.len() / 2, 2], "unexpected shape");
        for (i, exp) in expected.chunks(2).enumerate() {
            assert_eq!(actual[[i, 0]], exp[0]);
            assert_eq!(actual[[i, 1]], exp[1]);
        }
    }

    /// Compare an (n, m, 2) f64 array against a flat list of expected values.
    fn assert_xyz_close(actual: &Array3<f64>, expected: &[f64], tol: f64) {
        let (n, m) = (actual.shape()[0], actual.shape()[1]);
        assert_eq!(n * m * 2, expected.len(), "unexpected shape");
        for i in 0..n {
            for j in 0..m {
                let base = (i * m + j) * 2;
                assert_close(actual[[i, j, 0]], expected[base], tol);
                assert_close(actual[[i, j, 1]], expected[base + 1], tol);
            }
        }
    }

    // ---------------------------------------------------------------------
    // radius  (regression tests for the dx/dy half-diagonal bug)
    // ---------------------------------------------------------------------

    #[test]
    fn radius_is_half_the_diagonal() {
        // The radius of a cell is the distance from its center to one of its
        // corners, so for a non-square cell it must depend on both dx and dy.
        let grid = RectGrid::new(2., 3.);
        assert_close(grid.radius(), 1.802775637731995, 1e-12);

        let grid = RectGrid::new(4., 2.);
        assert_close(grid.radius(), 2.23606797749979, 1e-12);
    }

    #[test]
    fn radius_of_square_cell_is_unchanged() {
        // Guard against regressing the square-cell case, which the old
        // dy-typo implementation happened to get right.
        let grid = RectGrid::new(1., 1.);
        assert_close(grid.radius(), 0.7071067811865476, 1e-12);
    }

    // ---------------------------------------------------------------------
    // test_cell_at_point
    // ---------------------------------------------------------------------

    #[test]
    fn cell_at_point_single() {
        // A single point, mirroring the "test single point as tuple" case.
        let grid = RectGrid::new(5., 2.);
        let id = grid.cell_at_point(&[340., -14.2]);
        assert_eq!(id, [68, -8]);
    }

    #[test]
    fn cell_at_points_multiple() {
        // Multiple points, mirroring the "stacked list" / "ndarray" cases.
        let grid = RectGrid::new(5., 2.);
        let points = array![[14., 3.], [-8., 1.]];
        let ids = grid.cell_at_points(&points.view());
        assert_ids(&ids, &[2, 1, -2, 0]);
    }

    #[test]
    fn cell_at_point_and_points_agree() {
        let grid = RectGrid::new(5., 2.);
        let points = array![[14., 3.], [-8., 1.], [340., -14.2]];
        let ids = grid.cell_at_points(&points.view());
        for i in 0..points.shape()[0] {
            let single = grid.cell_at_point(&[points[[i, 0]], points[[i, 1]]]);
            assert_eq!([ids[[i, 0]], ids[[i, 1]]], single);
        }
    }

    // ---------------------------------------------------------------------
    // test_centroid
    // ---------------------------------------------------------------------

    #[test]
    fn centroid() {
        let grid = RectGrid::new(5., 2.);
        let index = array![[-1, 1], [1, -4]];
        let centroids = grid.centroid(&index.view());
        assert_xy_close(&centroids, &[-2.5, 3., 7.5, -7.], TOL);
    }

    // ---------------------------------------------------------------------
    // test_cell_corners
    // ---------------------------------------------------------------------

    #[test]
    fn cell_corners() {
        let grid = RectGrid::new(1.5, 1.5);
        let index = array![[-3, 2], [3, -6]];
        let corners = grid.cell_corners(&index.view());

        // Corner order is (top-left, top-right, bottom-right, bottom-left).
        let expected = [
            -4.5, 3.0, -3.0, 3.0, -3.0, 4.5, -4.5, 4.5, //
            4.5, -9.0, 6.0, -9.0, 6.0, -7.5, 4.5, -7.5,
        ];
        assert_xyz_close(&corners, &expected, TOL);
    }

    // ---------------------------------------------------------------------
    // test_centering_with_offset
    // ---------------------------------------------------------------------

    #[test]
    fn centering_with_offset() {
        for rot in [0., 15.5, 30., -26.2] {
            let mut grid = RectGrid::new(3., 3.);
            grid.set_rotation(rot);
            grid.set_offset([-grid.dx() / 2., -grid.dy() / 2.]);

            let index = array![[-1, -1]];
            let centroids = grid.centroid(&index.view());
            assert_close(centroids[[0, 0]], 0., TOL);
            assert_close(centroids[[0, 1]], 0., TOL);
        }
    }

    // ---------------------------------------------------------------------
    // test_rotation_setter
    // ---------------------------------------------------------------------

    #[test]
    fn rotation_setter() {
        let cases: [(f64, [[f64; 2]; 2]); 4] = [
            (0., [[1., 0.], [0., 1.]]),
            (15.5, [[0.96363045, -0.26723838], [0.26723838, 0.96363045]]),
            (30., [[0.8660254, -0.5], [0.5, 0.8660254]]),
            (-26.2, [[0.89725837, 0.44150585], [-0.44150585, 0.89725837]]),
        ];

        for (rot, expected) in cases {
            let mut grid = RectGrid::new(1.23, 0.987);
            grid.set_rotation(rot);
            assert_close(grid.rotation(), rot, TOL);

            let matrix = grid.rotation_matrix();
            for i in 0..2 {
                for j in 0..2 {
                    assert_close(matrix[[i, j]], expected[i][j], 1e-8);
                }
            }
        }
    }

    #[test]
    fn rotation_matrix_inv_is_the_inverse() {
        for rot in [0., 15.5, 30., -26.2, 420.] {
            let mut grid = RectGrid::new(1.23, 0.987);
            grid.set_rotation(rot);
            let m = grid.rotation_matrix();
            let m_inv = grid.rotation_matrix_inv();
            let product = m.dot(m_inv);
            assert_close(product[[0, 0]], 1., TOL);
            assert_close(product[[0, 1]], 0., TOL);
            assert_close(product[[1, 0]], 0., TOL);
            assert_close(product[[1, 1]], 1., TOL);
        }
    }

    // ---------------------------------------------------------------------
    // test_dx_dy_setter
    // ---------------------------------------------------------------------

    #[test]
    fn dx_dy_setter() {
        let mut grid = RectGrid::new(1.23, 4.56);
        grid.set_rotation(10.);
        assert_close(grid.dx(), 1.23, TOL);
        assert_close(grid.dy(), 4.56, TOL);

        grid.set_cellsize_x(3.21).unwrap();
        grid.set_cellsize_y(6.54).unwrap();
        assert_close(grid.dx(), 3.21, TOL);
        assert_close(grid.dy(), 6.54, TOL);
        assert_close(grid.cell_width(), 3.21, TOL);
        assert_close(grid.cell_height(), 6.54, TOL);
    }

    #[test]
    fn dx_dy_setter_rejects_non_positive_values() {
        // Python raises ValueError; the Rust setters return Err.
        let mut grid = RectGrid::new(1.23, 4.56);

        for invalid in [0., -1.] {
            assert!(
                grid.set_cellsize_x(invalid).is_err(),
                "setting dx to {invalid} should have failed"
            );
            assert!(
                grid.set_cellsize_y(invalid).is_err(),
                "setting dy to {invalid} should have failed"
            );
        }

        // A rejected value must not have been applied.
        assert_close(grid.dx(), 1.23, TOL);
        assert_close(grid.dy(), 4.56, TOL);
    }

    // ---------------------------------------------------------------------
    // test_anchor
    // ---------------------------------------------------------------------

    #[test]
    fn anchor_to_centroid() {
        let target_locs = [[0., 0.], [-2.9, -2.9], [-3.0, -3.0], [-3.1, -3.1]];
        let starting_offsets = [[0., 0.], [0.1, 0.], [0., 0.1], [0.1, 0.2]];
        let rotations = [0., 15., -69., 420.];

        for target_loc in target_locs {
            for starting_offset in starting_offsets {
                for rot in rotations {
                    let mut grid = RectGrid::new(0.3, 0.4);
                    grid.set_offset(starting_offset);
                    grid.set_rotation(rot);
                    grid.anchor_inplace(&target_loc, CellElement::Centroid);

                    let id = grid.cell_at_point(&target_loc);
                    let index = array![[id[0], id[1]]];
                    let centroid = grid.centroid(&index.view());
                    assert_close(centroid[[0, 0]], target_loc[0], 1e-9);
                    assert_close(centroid[[0, 1]], target_loc[1], 1e-9);
                }
            }
        }
    }

    #[test]
    fn anchor_to_corner() {
        let target_locs = [[0., 0.], [-2.9, -2.9], [-3.0, -3.0], [-3.1, -3.1]];
        let starting_offsets = [[0., 0.], [0.1, 0.], [0., 0.1], [0.1, 0.2]];
        let rotations = [0., 15., -69., 420.];

        for target_loc in target_locs {
            for starting_offset in starting_offsets {
                for rot in rotations {
                    let mut grid = RectGrid::new(0.3, 0.4);
                    grid.set_offset(starting_offset);
                    grid.set_rotation(rot);
                    grid.anchor_inplace(&target_loc, CellElement::Corner);

                    // The target location must coincide with one of the corners
                    // of the cell that contains it.
                    let id = grid.cell_at_point(&target_loc);
                    let index = array![[id[0], id[1]]];
                    let corners = grid.cell_corners(&index.view());
                    let on_corner = (0..4).any(|i| {
                        let dx = corners[[0, i, 0]] - target_loc[0];
                        let dy = corners[[0, i, 1]] - target_loc[1];
                        (dx * dx + dy * dy).sqrt() <= 1e-9
                    });
                    assert!(
                        on_corner,
                        "target {:?} is not a corner of the anchored cell \
                         (offset {:?}, rotation {})",
                        target_loc,
                        grid.offset(),
                        rot
                    );
                }
            }
        }
    }

    #[test]
    fn anchor_in_place_leaves_original_intact() {
        // The non-in-place variant of the Python test: anchoring a copy must
        // not modify the source grid's offset.
        let starting_offset = [0.1, 0.2];
        let mut grid = RectGrid::new(0.3, 0.4);
        grid.set_offset(starting_offset);

        let original_offset = grid.offset();

        let mut copy = grid.clone();
        copy.anchor_inplace(&[-2.9, -2.9], CellElement::Centroid);

        assert_close(grid.offset()[0], original_offset[0], 1e-15);
        assert_close(grid.offset()[1], original_offset[1], 1e-15);
        assert_ne!(
            copy.offset(),
            original_offset,
            "anchoring the copy should have changed its offset"
        );
    }

    #[test]
    fn anchor_returns_a_new_grid_and_keeps_the_source() {
        // `GridTraits::anchor` is the non-mutating form of `anchor_inplace`.
        let mut grid = RectGrid::new(0.3, 0.4);
        grid.set_offset([0.1, 0.2]);
        let original_offset = grid.offset();

        let anchored = grid.anchor(&[-2.9, -2.9], CellElement::Centroid);

        assert_eq!(grid.offset(), original_offset);
        let id = anchored.cell_at_point(&[-2.9, -2.9]);
        let index = array![[id[0], id[1]]];
        let centroid = anchored.centroid(&index.view());
        assert_close(centroid[[0, 0]], -2.9, 1e-9);
        assert_close(centroid[[0, 1]], -2.9, 1e-9);
    }

    // ---------------------------------------------------------------------
    // cells_near_point  (underlying primitive of bounded-grid interpolation)
    // ---------------------------------------------------------------------

    #[test]
    fn cells_near_point_contains_containing_cell_and_its_corner_neighbours() {
        // `cells_near_point` returns the containing cell plus the three cells
        // that share a corner with it. The *position* of the containing cell in
        // the returned 4 depends on which quadrant of it the point lies in, so
        // the assertions below are set-based.
        let mut grid = RectGrid::new(1., 2.);
        grid.set_offset([0.25, 0.5]);

        let (id_x, id_y) = (2, 3);
        let centroid = grid.centroid(&array![[id_x, id_y]].view());
        let (cx, cy) = (centroid[[0, 0]], centroid[[0, 1]]);

        // Four sample points, one in each quadrant of the containing cell.
        let quadrants = [
            (cx - 0.25, cy + 0.5), // top-left
            (cx + 0.25, cy + 0.5), // top-right
            (cx - 0.25, cy - 0.5), // bottom-left
            (cx + 0.25, cy - 0.5), // bottom-right
            (cx, cy),              // exactly on the centroid
        ];

        for (px, py) in quadrants {
            let points = array![[px, py]];
            let nearby = grid.cells_near_point(&points.view());

            assert_eq!(nearby.shape(), &[1, 4, 2]);

            let containing = grid.cell_at_point(&[px, py]);
            assert_eq!([containing[0], containing[1]], [id_x, id_y]);

            let mut cells: Vec<(i64, i64)> = (0..4)
                .map(|i| (nearby[[0, i, 0]], nearby[[0, i, 1]]))
                .collect();
            cells.sort_unstable();
            cells.dedup();
            assert_eq!(cells.len(), 4, "cells near ({px}, {py}) are not distinct");

            // The containing cell must be present ...
            assert!(
                cells.contains(&(id_x, id_y)),
                "containing cell ({id_x}, {id_y}) missing for point ({px}, {py})"
            );
            // ... and every returned cell is within one step on both axes.
            for (x, y) in &cells {
                assert!((x - id_x).abs() <= 1 && (y - id_y).abs() <= 1);
            }
        }
    }

    #[test]
    fn cells_near_point_covers_exactly_one_quadrant() {
        // Pin down the exact 4 cells returned for a point in the top-left
        // quadrant of cell (2, 3) on a dx=1, dy=2 grid with offset (0.25, 0.5).
        let mut grid = RectGrid::new(1., 2.);
        grid.set_offset([0.25, 0.5]);

        let centroid = grid.centroid(&array![[2, 3]].view());
        let points = array![[centroid[[0, 0]] - 0.25, centroid[[0, 1]] + 0.5]];
        let nearby = grid.cells_near_point(&points.view());

        let mut cells: Vec<(i64, i64)> = (0..4)
            .map(|i| (nearby[[0, i, 0]], nearby[[0, i, 1]]))
            .collect();
        cells.sort_unstable();
        cells.dedup();
        assert_eq!(cells, vec![(1, 3), (1, 4), (2, 3), (2, 4)]);
    }
}
