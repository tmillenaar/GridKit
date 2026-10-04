use std::hash::{Hash, Hasher};

use crate::grid::{grid_type_name, CellElement, Grid, GridTraits};
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
        self._dx.to_bits() == other._dx.to_bits()
            && self._dy.to_bits() == other._dy.to_bits()
            && self.offset[0].to_bits() == other.offset[0].to_bits()
            && self.offset[1].to_bits() == other.offset[1].to_bits()
            && self._rotation.to_bits() == other._rotation.to_bits()
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
            //let point = self._rotation_matrix_inv.dot(&point); // nocheckin, this line causes
            //slowdown. Consider a separate version of the function that does not do rotation
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

    fn all_neighbours(
        &self,
        index: &ArrayView2<i64>,
        depth: i64,
        include_selected: bool,
        add_cell_id: bool,
    ) -> Array3<i64> {
        self._neighbours(index, depth, include_selected, add_cell_id, false)
    }

    fn direct_neighbours(
        &self,
        index: &ArrayView2<i64>,
        depth: i64,
        include_selected: bool,
        add_cell_id: bool,
    ) -> Array3<i64> {
        self._neighbours(index, depth, include_selected, add_cell_id, true)
    }

    fn is_aligned_with(&self, other: &Grid) -> bool {
        if let Grid::RectGrid(other) = other {
            if !isclose(self.dx(), other.dx(), NUMERIC_RTOL, NUMERIC_ATOL)
                || !isclose(self.dy(), other.dy(), NUMERIC_RTOL, NUMERIC_ATOL)
            {
                return false;
            }
            if !(isclose(self.offset()[0], other.offset()[0], NUMERIC_RTOL, 1e-7)
                && isclose(self.offset()[1], other.offset()[1], NUMERIC_RTOL, 1e-7))
            {
                return false;
            }
            if self.rotation() != other.rotation() {
                return false;
            }
            true
        } else {
            false
        }
    }

    fn subdivide(&self, factor: u64) -> Grid {
        if factor == 0 {
            return Grid::RectGrid(self.clone());
        }
        let factor = factor as f64;

        let mut sub_grid = self.clone();
        sub_grid
            .set_cellsize_x(self.dx() / factor)
            .expect("subdividing by a factor of at least one cannot zero the cellsize");
        sub_grid
            .set_cellsize_y(self.dy() / factor)
            .expect("subdividing by a factor of at least one cannot zero the cellsize");

        // Anchor the sub grid to the top left corner of the parent's cell (0, 0).
        // The corner is taken from the *parent*, so it is unaffected by the cellsize
        // change above.
        let corners = self.cell_corners(&array![[0i64, 0i64]].view());
        let anchor_loc = [corners[[0, 0, 0]], corners[[0, 0, 1]]];
        sub_grid.anchor_inplace(&anchor_loc, CellElement::Corner);

        Grid::RectGrid(sub_grid)
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

    /// Shared implementation of the `GridTraits` neighbour methods.
    ///
    /// `direct_only` selects the diamond shaped window (`connect_corners=false`)
    /// over the full square one.
    fn _neighbours(
        &self,
        index: &ArrayView2<i64>,
        depth: i64,
        include_selected: bool,
        add_cell_id: bool,
        direct_only: bool,
    ) -> Array3<i64> {
        // Python raises `ValueError("'depth' cannot be lower than 1")`.
        assert!(depth >= 1, "'depth' cannot be lower than 1");
        let add_cell_id = add_cell_id as i64;

        // Python builds the full `(2 * depth + 1)^2` window by raveling a meshgrid in
        // C order, which means the rows run from y = +depth down to y = -depth and
        // within a row x runs from -depth to +depth. `direct_only` then keeps the
        // cells for which `|x * y| < depth`, the cells that are at most `depth` steps
        // away when stepping along the axes.
        let mut relative: Vec<[i64; 2]> = Vec::new();
        for row in (0..=(2 * depth)).rev() {
            let y = row - depth;
            for column in 0..=(2 * depth) {
                let x = column - depth;
                if direct_only && (x * y).abs() >= depth {
                    continue;
                }
                // The selected cell always sits exactly in the middle of the window,
                // so skipping it here is the same as deleting the middle element.
                if !include_selected && x == 0 && y == 0 {
                    continue;
                }
                relative.push([x, y]);
            }
        }

        let mut neighbours = Array3::<i64>::zeros((index.shape()[0], relative.len(), 2));
        for cell_id in 0..neighbours.shape()[0] {
            for (neighbour_id, [rel_x, rel_y]) in relative.iter().enumerate() {
                neighbours[Ix3(cell_id, neighbour_id, 0)] =
                    rel_x + add_cell_id * index[Ix2(cell_id, 0)];
                neighbours[Ix3(cell_id, neighbour_id, 1)] =
                    rel_y + add_cell_id * index[Ix2(cell_id, 1)];
            }
        }
        neighbours
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex_grid::HexGrid;
    use crate::tri_grid::TriGrid;
    use crate::Orientation;

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

    /// Compare an (n, m, 2) i64 array against a flat list of expected values.
    fn assert_ids_3d(actual: &Array3<i64>, expected: &[i64]) {
        let (n, m) = (actual.shape()[0], actual.shape()[1]);
        assert_eq!(n * m * 2, expected.len(), "unexpected shape");
        for i in 0..n {
            for j in 0..m {
                let base = (i * m + j) * 2;
                assert_eq!(
                    [[actual[[i, j, 0]], actual[[i, j, 1]]]],
                    [[expected[base], expected[base + 1]]]
                );
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

    // ---------------------------------------------------------------------
    // all_neighbours / direct_neighbours
    // ---------------------------------------------------------------------

    #[test]
    fn direct_neighbours_depth_one() {
        // Mirrors the doctest of `RectGrid.relative_neighbours`: a diamond of
        // 4 cells around (0, 0), y descending within each row.
        let grid = RectGrid::new(1., 2.);
        let neighbours = grid.direct_neighbours(&array![[0i64, 0i64]].view(), 1, false, false);
        assert_eq!(neighbours.shape(), &[1, 4, 2]);
        assert_ids_3d(&neighbours, &[0, 1, -1, 0, 1, 0, 0, -1]);
    }

    #[test]
    fn all_neighbours_depth_one() {
        // Mirrors `relative_neighbours(connect_corners=True)`: the full 3x3
        // square without the selected cell.
        let grid = RectGrid::new(1., 2.);
        let neighbours = grid.all_neighbours(&array![[0i64, 0i64]].view(), 1, false, false);
        assert_eq!(neighbours.shape(), &[1, 8, 2]);
        assert_ids_3d(
            &neighbours,
            &[-1, 1, 0, 1, 1, 1, -1, 0, 1, 0, -1, -1, 0, -1, 1, -1],
        );
    }

    #[test]
    fn include_selected_places_the_cell_at_the_centre() {
        // Python inserts the selected cell at index `floor(len / 2)`. The
        // diamond and the square both have an odd number of cells when the
        // selected cell is included, so that index is well defined.
        let grid = RectGrid::new(1., 2.);

        for depth in 1..=6 {
            for direct_only in [true, false] {
                for add_cell_id in [false, true] {
                    let (with, without) = if direct_only {
                        (
                            grid.direct_neighbours(
                                &array![[7i64, -3i64]].view(),
                                depth,
                                true,
                                add_cell_id,
                            ),
                            grid.direct_neighbours(
                                &array![[7i64, -3i64]].view(),
                                depth,
                                false,
                                add_cell_id,
                            ),
                        )
                    } else {
                        (
                            grid.all_neighbours(
                                &array![[7i64, -3i64]].view(),
                                depth,
                                true,
                                add_cell_id,
                            ),
                            grid.all_neighbours(
                                &array![[7i64, -3i64]].view(),
                                depth,
                                false,
                                add_cell_id,
                            ),
                        )
                    };

                    assert_eq!(with.shape()[1], without.shape()[1] + 1);
                    let centre = with.shape()[1] / 2;
                    let selected = if add_cell_id { [7, -3] } else { [0, 0] };
                    assert_eq!([[with[[0, centre, 0]], with[[0, centre, 1]]]], [selected]);

                    // Dropping the selected cell must give exactly the same
                    // cells, in the same order, as the shorter array.
                    for i in 0..without.shape()[1] {
                        let j = if i < centre { i } else { i + 1 };
                        assert_eq!(
                            [[without[[0, i, 0]], without[[0, i, 1]]]],
                            [[with[[0, j, 0]], with[[0, j, 1]]]]
                        );
                    }
                }
            }
        }
    }

    /// Independently count the cells a neighbour query should return, by walking
    /// the window rather than by raveling and filtering it.
    fn expected_neighbour_count(
        depth: i64,
        connect_corners: bool,
        include_selected: bool,
    ) -> usize {
        let mut count = 0;
        for y in -depth..=depth {
            for x in -depth..=depth {
                if !connect_corners && (x * y).abs() >= depth {
                    continue;
                }
                if !include_selected && x == 0 && y == 0 {
                    continue;
                }
                count += 1;
            }
        }
        count
    }

    #[test]
    fn neighbour_count_grows_with_depth() {
        // The doctest of `RectGrid.relative_neighbours` pins the diamond counts
        // at [4, 12, 24, 36] and the square counts at [8, 24, 48, 80] for depth
        // 1 to 4. Note that neither shape grows by a constant factor per depth:
        // the diamond holds `2 * depth * (depth + 1)` cells only for depth 1 to 3.
        const DOCUMENTED_DIAMOND: [usize; 4] = [4, 12, 24, 36];
        const DOCUMENTED_SQUARE: [usize; 4] = [8, 24, 48, 80];

        let grid = RectGrid::new(1., 2.);
        let index = array![[0i64, 0i64]];

        for depth in 1..=6 {
            for include_selected in [false, true] {
                let diamond = grid
                    .direct_neighbours(&index.view(), depth, include_selected, false)
                    .shape()[1];
                let square = grid
                    .all_neighbours(&index.view(), depth, include_selected, false)
                    .shape()[1];

                assert_eq!(
                    diamond,
                    expected_neighbour_count(depth, false, include_selected),
                    "diamond, depth {depth}, include_selected {include_selected}"
                );
                assert_eq!(
                    square,
                    expected_neighbour_count(depth, true, include_selected),
                    "square, depth {depth}, include_selected {include_selected}"
                );

                // Including the selected cell adds exactly one cell.
                if include_selected {
                    let without = grid
                        .direct_neighbours(&index.view(), depth, false, false)
                        .shape()[1];
                    assert_eq!(diamond, without + 1);
                }

                if depth <= 4 {
                    if include_selected {
                        continue;
                    }
                    assert_eq!(diamond, DOCUMENTED_DIAMOND[depth as usize - 1]);
                    assert_eq!(square, DOCUMENTED_SQUARE[depth as usize - 1]);
                }
            }
        }
    }

    #[test]
    fn neighbours_multiple_indices() {
        // The literal expected ids of `test_neighbours_multiple_indices`, on a
        // dx=1, dy=2 grid.
        let grid = RectGrid::new(1., 2.);
        let index = array![[-1i64, 0i64], [2i64, 1i64]];

        // connect_corners=False, include_selected=False
        assert_ids_3d(
            &grid.direct_neighbours(&index.view(), 1, false, true),
            &[
                -1, 1, -2, 0, 0, 0, -1, -1, //
                2, 2, 1, 1, 3, 1, 2, 0,
            ],
        );

        // connect_corners=True, include_selected=True
        assert_ids_3d(
            &grid.all_neighbours(&index.view(), 1, true, true),
            &[
                -2, 1, -1, 1, 0, 1, -2, 0, -1, 0, 0, 0, -2, -1, -1, -1, 0, -1, //
                1, 2, 2, 2, 3, 2, 1, 1, 2, 1, 3, 1, 1, 0, 2, 0, 3, 0,
            ],
        );
    }

    #[test]
    fn neighbours_single_and_multiple_indices_agree() {
        // `relative_neighbours` only uses `index` for its repeat count on a
        // rectangular lattice, so a single index must give the same ids as the
        // matching row of the multi index call.
        let grid = RectGrid::new(1., 2.);

        for depth in 1..=4 {
            for connect_corners in [false, true] {
                for include_selected in [false, true] {
                    let index = array![[-1i64, 0i64], [2i64, 1i64]];

                    let many = if connect_corners {
                        grid.all_neighbours(&index.view(), depth, include_selected, true)
                    } else {
                        grid.direct_neighbours(&index.view(), depth, include_selected, true)
                    };

                    for (row, expected) in [(-1i64, 0i64), (2i64, 1i64)].iter().enumerate() {
                        let one = array![[expected.0, expected.1]];
                        let single = if connect_corners {
                            grid.all_neighbours(&one.view(), depth, include_selected, true)
                        } else {
                            grid.direct_neighbours(&one.view(), depth, include_selected, true)
                        };

                        assert_eq!(single.shape(), &[1, many.shape()[1], 2]);
                        for i in 0..single.shape()[1] {
                            assert_eq!(
                                [[single[[0, i, 0]], single[[0, i, 1]]]],
                                [[many[[row, i, 0]], many[[row, i, 1]]]]
                            );
                        }

                        // The relative ids plus the cell id are the absolute ids.
                        let relative = if connect_corners {
                            grid.all_neighbours(&one.view(), depth, include_selected, false)
                        } else {
                            grid.direct_neighbours(&one.view(), depth, include_selected, false)
                        };
                        for i in 0..single.shape()[1] {
                            assert_eq!(
                                [[single[[0, i, 0]], single[[0, i, 1]]]],
                                [[
                                    relative[[0, i, 0]] + expected.0,
                                    relative[[0, i, 1]] + expected.1
                                ]]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn neighbours_respect_depth_and_connect_corners() {
        // The invariants `test_neighbours` asserts, on a square dx=dy=3 grid.
        let grid = RectGrid::new(3., 3.);

        for depth in 1..=6 {
            for connect_corners in [false, true] {
                for include_selected in [false, true] {
                    let index = array![[2i64, 1i64]];
                    let neighbours = if connect_corners {
                        grid.all_neighbours(&index.view(), depth, include_selected, true)
                    } else {
                        grid.direct_neighbours(&index.view(), depth, include_selected, true)
                    };

                    let centroids = grid.centroid(&neighbours.slice(s![0, .., ..]));
                    let center = grid.centroid(&index.view());
                    let offsets = centroids - &center;

                    // `test_neighbours` deletes the selected cell from the array
                    // before measuring, because it sits at distance zero on its
                    // own and would form a group of one.
                    let distances: Vec<f64> = offsets
                        .axis_iter(Axis(0))
                        .enumerate()
                        .filter(|(i, _)| !include_selected || *i != neighbours.shape()[1] / 2)
                        .map(|(_, xy)| (xy[Ix1(0)].powi(2) + xy[Ix1(1)].powi(2)).sqrt())
                        .collect();

                    // Cells at the same distance from the center always come in
                    // multiples of four.
                    let mut sorted = distances.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mut counts = vec![1usize];
                    for window in sorted.windows(2) {
                        if (window[0] - window[1]).abs() < 1e-9 {
                            *counts.last_mut().unwrap() += 1;
                        } else {
                            counts.push(1);
                        }
                    }
                    for count in &counts {
                        assert_eq!(
                            count % 4,
                            0,
                            "depth {depth}, connect_corners {connect_corners}, \
                             include_selected {include_selected}: {counts:?}"
                        );
                    }

                    if connect_corners {
                        // No cell can be further away than `depth` diagonals.
                        let limit =
                            (grid.dx().powi(2) + grid.dy().powi(2)).sqrt() * depth as f64 + 1e-14;
                        assert!(distances.iter().all(|d| *d <= limit));
                    } else {
                        // No cell can be further away than `depth` cell sizes.
                        let limit = grid.dx() * depth as f64;
                        assert!(distances.iter().all(|d| *d <= limit));
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "'depth' cannot be lower than 1")]
    fn depth_below_one_is_rejected() {
        let grid = RectGrid::new(1., 2.);
        grid.direct_neighbours(&array![[0i64, 0i64]].view(), 0, false, false);
    }

    // ---------------------------------------------------------------------
    // subdivide
    // ---------------------------------------------------------------------

    #[test]
    fn subdivide_scales_the_cells() {
        let mut grid = RectGrid::new(1., 0.7);
        grid.set_rotation(-23.);

        for factor in [1u64, 2, 3, 9] {
            let sub_grid = grid.subdivide(factor);
            let sub_grid = match sub_grid {
                Grid::RectGrid(g) => g,
                _ => panic!("subdivide on RectGrid should return RectGrid"),
            };
            assert_close(sub_grid.dx(), 1. / factor as f64, TOL);
            assert_close(sub_grid.dy(), 0.7 / factor as f64, TOL);
        }
    }

    #[test]
    fn subdivide_keeps_parent_corners_on_a_corner() {
        for rotation in [-23., 0., 456.] {
            for offset in [[-2., 3.], [0., 0.], [0.1, -0.2]] {
                let mut grid = RectGrid::new(1., 0.7);
                grid.set_rotation(rotation);
                grid.set_offset(offset);

                for factor in [2u64, 9] {
                    let sub_grid = grid.subdivide(factor);
                    let sub_grid_ref = match &sub_grid {
                        Grid::RectGrid(g) => g.clone(),
                        _ => panic!("subdivide on RectGrid should return RectGrid"),
                    };

                    // Take the last corner of a cell and check that it coincides
                    // with one of the corners of the sub cell containing it.
                    let corners = grid.cell_corners(&array![[-4i64, 23i64]].view());
                    let corner = [corners[[0, 3, 0]], corners[[0, 3, 1]]];

                    let id = sub_grid_ref.cell_at_point(&corner);
                    let sub_corners = sub_grid_ref.cell_corners(&array![[id[0], id[1]]].view());
                    let on_corner = (0..4).any(|i| {
                        let dx = sub_corners[[0, i, 0]] - corner[0];
                        let dy = sub_corners[[0, i, 1]] - corner[1];
                        (dx * dx + dy * dy).sqrt() <= 1e-9
                    });
                    assert!(
                        on_corner,
                        "corner {corner:?} is not a corner of the sub cell containing it \
                         (rotation {rotation}, offset {offset:?}, factor {factor})"
                    );
                }
            }
        }
    }

    #[test]
    fn subdivide_yields_factor_squared_subcells_per_parent_cell() {
        for factor in [2u64, 9] {
            for rotation in [-23., 0., 456.] {
                for offset in [[-2., 3.], [0., 0.], [0.1, -0.2]] {
                    let mut grid = RectGrid::new(1., 0.7);
                    grid.set_rotation(rotation);
                    grid.set_offset(offset);

                    let sub_grid = grid.subdivide(factor);
                    let sub_grid_ref = match &sub_grid {
                        Grid::RectGrid(g) => g.clone(),
                        _ => panic!("subdivide on RectGrid should return RectGrid"),
                    };

                    // The sub cells that could possibly be inside the parent cell
                    // (3, -2) are the (2 * factor + 1)^2 sub cells around the one
                    // holding the parent centroid.
                    let target = grid.centroid(&array![[3i64, -2i64]].view());
                    let start = sub_grid_ref.cell_at_point(&[target[[0, 0]], target[[0, 1]]]);
                    let candidates = sub_grid_ref.all_neighbours(
                        &array![[start[0], start[1]]].view(),
                        factor as i64,
                        true,
                        true,
                    );

                    let sub_centroids = sub_grid_ref.centroid(&candidates.slice(s![0, .., ..]));
                    let in_cell = grid.cell_at_points(&sub_centroids.view());
                    let nr_in_cell = (0..in_cell.shape()[0])
                        .filter(|i| in_cell[[*i, 0]] == 3 && in_cell[[*i, 1]] == -2)
                        .count();

                    assert_eq!(
                        nr_in_cell,
                        (factor * factor) as usize,
                        "rotation {rotation}, offset {offset:?}, factor {factor}"
                    );
                }
            }
        }
    }

    #[test]
    fn subdivide_by_one_is_the_same_grid() {
        let mut grid = RectGrid::new(1.23, 4.56);
        grid.set_rotation(15.5);
        grid.set_offset([0.1, 0.2]);

        assert_eq!(grid.subdivide(1), Grid::RectGrid(grid.clone()));
    }

    #[test]
    fn subdivide_by_zero_is_the_same_grid() {
        // `factor` is unsigned, so a `factor` of 0 is the only invalid input left.
        // Dividing the cellsize by zero is not meaningful, so this yields an
        // unchanged copy instead. Note this diverges from Python, which raises a
        // `ValueError` for any `factor` below 1.
        let mut grid = RectGrid::new(1.23, 4.56);
        grid.set_rotation(15.5);
        grid.set_offset([0.1, 0.2]);

        assert_eq!(grid.subdivide(0), Grid::RectGrid(grid));
    }

    // ---------------------------------------------------------------------
    // is_aligned_with
    //
    // The CRS is deliberately not part of the Rust grid, so the two CRS cases of
    // `test_is_aligned_with` are not covered here.
    // ---------------------------------------------------------------------

    #[test]
    fn is_aligned_with_an_identical_grid() {
        let grid = RectGrid::new(1.2, 1.2);
        let other = RectGrid::new(1.2, 1.2);

        assert!(grid.is_aligned_with(&Grid::RectGrid(other)));
    }

    #[test]
    fn is_aligned_with_reports_a_differing_cellsize() {
        let grid = RectGrid::new(1.2, 1.2);

        for other in [RectGrid::new(1.2, 1.3), RectGrid::new(1.3, 1.2)] {
            assert!(!grid.is_aligned_with(&Grid::RectGrid(other)));
        }
    }

    #[test]
    fn is_aligned_with_reports_a_differing_offset() {
        let grid = RectGrid::new(1.2, 1.2);

        for offset in [[0., 1.], [1., 0.]] {
            let mut other = RectGrid::new(1.2, 1.2);
            other.set_offset(offset);
            assert!(!grid.is_aligned_with(&Grid::RectGrid(other)));
        }
    }

    #[test]
    fn is_aligned_with_reports_a_differing_rotation() {
        let grid = RectGrid::new(1.2, 1.2);
        let mut other = RectGrid::new(1.2, 1.2);
        other.set_rotation(15.5);
        assert!(!grid.is_aligned_with(&Grid::RectGrid(other)));
    }

    #[test]
    fn is_aligned_with_reports_a_differing_grid_type() {
        let grid = RectGrid::new(1.2, 1.2);

        let others = vec![
            Grid::HexGrid(HexGrid::new(1., Orientation::Flat)),
            Grid::TriGrid(TriGrid::new(1., Orientation::Flat)),
        ];
        for other in others {
            assert!(!grid.is_aligned_with(&other));
        }
    }

    #[test]
    fn is_aligned_with_reports_multiple_reasons() {
        let grid = RectGrid::new(1.2, 1.2);
        let mut other = RectGrid::new(1.2, 1.1);
        other.set_offset([1., 1.]);
        assert!(!grid.is_aligned_with(&Grid::RectGrid(other)));
    }

    #[test]
    fn is_aligned_with_leaves_both_grids_intact() {
        let mut grid = RectGrid::new(1.2, 1.2);
        grid.set_offset([0.3, 0.4]);
        let mut other = RectGrid::new(1.2, 1.1);
        other.set_offset([1., 1.]);

        let before_grid = (grid.dx(), grid.dy(), grid.offset(), grid.rotation());
        let before_other = (other.dx(), other.dy(), other.offset(), other.rotation());

        assert!(!grid.is_aligned_with(&Grid::RectGrid(other.clone())));
        assert!(!other.is_aligned_with(&Grid::RectGrid(grid.clone())));

        assert_eq!(
            before_grid,
            (grid.dx(), grid.dy(), grid.offset(), grid.rotation())
        );
        assert_eq!(
            before_other,
            (other.dx(), other.dy(), other.offset(), other.rotation())
        );
    }

    #[test]
    fn is_aligned_with_is_reachable_through_the_grid_enum() {
        // The `GridTraits` implementation should behave the same whether it is
        // reached through the concrete type or through the enum.
        let grid = Grid::RectGrid(RectGrid::new(1.2, 1.2));
        let other = Grid::RectGrid(RectGrid::new(1.2, 1.3));

        assert_eq!(
            grid.is_aligned_with(&other),
            RectGrid::new(1.2, 1.2).is_aligned_with(&other)
        );
        assert!(!grid.is_aligned_with(&other));
    }

    #[test]
    fn neighbours_are_reachable_through_the_grid_enum() {
        // `all_neighbours`/`direct_neighbours` are `GridTraits` methods, so
        // `enum_delegate` has to forward them for `Grid` to be usable with them.
        let grid = Grid::RectGrid(RectGrid::new(1.2, 1.2));
        let concrete = RectGrid::new(1.2, 1.2);
        let index = array![[3i64, -2i64]];

        assert_eq!(
            grid.all_neighbours(&index.view(), 2, true, true),
            concrete.all_neighbours(&index.view(), 2, true, true)
        );
        assert_eq!(
            grid.direct_neighbours(&index.view(), 2, false, false),
            concrete.direct_neighbours(&index.view(), 2, false, false)
        );
    }
}
