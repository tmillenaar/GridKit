use crate::grid::GridTraits;
use crate::tile::*;
use crate::utils::*;
use ndarray::*;

#[derive(Clone)]
pub struct RectGrid {
    pub _dx: f64,
    pub _dy: f64,
    pub offset: [f64; 2],
    pub _rotation: f64,
    pub _rotation_matrix: Array2<f64>,
    pub _rotation_matrix_inv: Array2<f64>,
}

impl GridTraits for RectGrid {
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
        ((self._dy / 2.).powi(2) + (self._dy / 2.).powi(2)).powf(0.5)
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

    fn cell_at_point(&self, points: &ArrayView2<f64>) -> Array2<i64> {
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
        let index = self.cell_at_point(points);

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
            let rel_loc_x: f64 = modulus((points[Ix2(cell_id, 0)] - self.offset[0]), self.dx());
            let rel_loc_y: f64 = modulus((points[Ix2(cell_id, 1)] - self.offset[1]), self.dy());
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

    fn set_cellsize_x(&mut self, cellsize_x: f64) {
        self._dx = cellsize_x;
    }

    fn set_cellsize_y(&mut self, cellsize_y: f64) {
        self._dy = cellsize_y;
    }
}
