use numpy::ndarray::*;
use crate::utils::*;

pub struct RectGrid {
    pub _dx: f64,
    pub _dy: f64,
    pub offset: (f64, f64),
}

impl RectGrid {
    pub fn new(dx: f64, dy: f64, offset: (f64, f64)) -> Self {
        let offset = normalize_offset(offset, dx, dy);
        let (_dx, _dy) = (dx, dy); // rename in order to pass to struct
        RectGrid { _dx, _dy, offset }
    }

    pub fn cell_height(&self) -> f64 {
        self._dy
    }

    pub fn cell_width(&self) -> f64 {
        self._dx
    }

    pub fn dx(&self) -> f64 {
        self._dx
    }

    pub fn dy(&self) -> f64 {
        self._dy
    }

    pub fn centroid(&self, index: &ArrayView2<i64>) -> Array2<f64> {
        let mut centroids = Array2::<f64>::zeros((index.shape()[0], 2));

        for cell_id in 0..centroids.shape()[0] {
            let point = self.centroid_single_point(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
            centroids[Ix2(cell_id, 0)] = point.0;
            centroids[Ix2(cell_id, 1)] = point.1;
        }
        centroids
    }

    fn centroid_single_point(&self, x: i64, y: i64) -> (f64, f64) {
        let centroid_x = x as f64 * self.dx() + (self.dx() / 2.) + self.offset.0;
        let centroid_y = y as f64 * self.dy() + (self.dy() / 2.) + self.offset.1;
        (centroid_x, centroid_y)
    }

    pub fn cell_at_point(&self, points: &ArrayView2<f64>) -> Array2<i64> {
        let shape = points.shape();
        let mut index = Array2::<i64>::zeros((shape[0], shape[1]));
        for cell_id in 0..points.shape()[0] {
            let id_x = ((points[Ix2(cell_id, 0)] - self.offset.0) / self.dx()).floor() as i64;
            let id_y = ((points[Ix2(cell_id, 1)] - self.offset.1) / self.dy()).floor() as i64;
            index[Ix2(cell_id, 0)] = id_x;
            index[Ix2(cell_id, 1)] = id_y;
        }
        index
    }

    pub fn cell_corners(&self, index: &ArrayView2<i64>) -> Array3<f64> {
        let mut corners = Array3::<f64>::zeros((index.shape()[0], 4, 2));
        for cell_id in 0..index.shape()[0] {
            let id_x = index[Ix2(cell_id, 0)];
            let id_y = index[Ix2(cell_id, 1)];
            let (centroid_x, centroid_y) = self.centroid_single_point(id_x, id_y);
            corners[Ix3(cell_id, 0, 0)] = centroid_x - self.dx() / 2.;
            corners[Ix3(cell_id, 0, 1)] = centroid_y - self.dy() / 2.;
            corners[Ix3(cell_id, 1, 0)] = centroid_x + self.dx() / 2.;
            corners[Ix3(cell_id, 1, 1)] = centroid_y - self.dy() / 2.;
            corners[Ix3(cell_id, 2, 0)] = centroid_x + self.dx() / 2.;
            corners[Ix3(cell_id, 2, 1)] = centroid_y + self.dy() / 2.;
            corners[Ix3(cell_id, 3, 0)] = centroid_x - self.dx() / 2.;
            corners[Ix3(cell_id, 3, 1)] = centroid_y + self.dy() / 2.;
        }
        corners
    }
}
