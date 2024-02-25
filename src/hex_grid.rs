use numpy::ndarray::*;
use crate::utils::*;

pub struct HexGrid {
    pub cellsize: f64,
    pub offset: (f64, f64),
    pub rotation: f64,
    pub rotation_matrix: Array2<f64>,
    pub rotation_matrix_inv: Array2<f64>,
}

impl HexGrid {
    pub fn new(cellsize: f64, offset: (f64, f64), rotation: f64) -> Self {
        let rotation_matrix = _rotation_matrix(rotation);
        let rotation_matrix_inv = _rotation_matrix(-rotation);
        // TODO: Find a way to normalize_offset without having to instantiate tmp object
        let self_tmp = HexGrid { cellsize, offset, rotation, rotation_matrix, rotation_matrix_inv };
        let offset = normalize_offset(offset, self_tmp.dx(), self_tmp.dy());
        let rotation_matrix = _rotation_matrix(rotation);
        let rotation_matrix_inv = _rotation_matrix(-rotation);
        HexGrid { cellsize, offset, rotation, rotation_matrix, rotation_matrix_inv }
    }

    pub fn radius(&self) -> f64 {
        self.cellsize / 3_f64.powf(0.5)
    }

    pub fn dx(&self) -> f64 {
        self.cellsize
    }

    pub fn dy(&self) -> f64 {
        3. / 2. * self.radius()
    }

    pub fn centroid(&self, index: &ArrayView2<i64>) -> Array2<f64> {
        let mut centroids = Array2::<f64>::zeros((index.shape()[0], 2));

        for cell_id in 0..centroids.shape()[0] {
            let point = self.centroid_xy_no_rot(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
            centroids[Ix2(cell_id, 0)] = point.0;
            centroids[Ix2(cell_id, 1)] = point.1;
        }

        if self.rotation != 0. {
            for cell_id in 0..centroids.shape()[0] {
                let mut centroid = centroids.slice_mut(s![cell_id, ..]);
                let cent_rot = self.rotation_matrix.dot(&centroid);
                centroid.assign(&cent_rot);
            }
        }
        centroids
    }

    fn centroid_xy_no_rot(&self, x: i64, y: i64) -> (f64, f64) {
        let mut centroid_x = x as f64 * self.dx() + (self.dx() / 2.) + self.offset.0;
        let centroid_y = y as f64 * self.dy() + (self.dy() / 2.) + self.offset.1;

        if !iseven(y) {
            centroid_x  = centroid_x + self.dx() / 2.;
        }
        (centroid_x, centroid_y)
    }

    pub fn cell_at_location(&self, points: &ArrayView2<f64>) -> Array2<i64> {
        let mut index = Array2::<i64>::zeros((points.shape()[0], 2));
        for cell_id in 0..points.shape()[0] {
            let point = points.slice(s![cell_id, ..]);
            let point = self.rotation_matrix_inv.dot(&point);
            
            // let x = points[Ix2(cell_id, 0)];
            // let y = points[Ix2(cell_id, 1)];
            let x = point[Ix1(0)];
            let y = point[Ix1(1)];

            // determine initial id_y
            let mut id_y = ((y - self.radius() / 4.) / self.dy()).floor();
            let is_offset = modulus(id_y, 2.) != 0.;
            let mut id_x: f64;

            // determine initial id_x
            if is_offset == true {
                id_x = (x - self.dx() / 2.) / self.dx();
            } else {
                id_x = x / self.dx();
            }
            id_x = id_x.floor();

            // refine id_x and id_y
            // Example: points at the top of the cell can be in this cell or in the cell to the top right or top left
            let rel_loc_y = modulus(y - self.radius() / 4., self.dy()) + self.radius() / 4.;
            let rel_loc_x = modulus(x, self.dx());

            let mut in_top_left: bool;
            let mut in_top_right: bool;
            if is_offset == true {
                in_top_left = (self.radius() * 1.25 - rel_loc_y)
                    < ((rel_loc_x - 0.5 * self.dx()) / (self.dx() / self.radius()));
                in_top_left = in_top_left && (rel_loc_x < (0.5 * self.dx()));
                in_top_right = (rel_loc_x - 0.5 * self.dx()) / (self.dx() / self.radius())
                    <= (rel_loc_y - self.radius() * 1.25);
                in_top_right = in_top_right && rel_loc_x >= (0.5 * self.dx());
                if in_top_left == true {
                    id_y = id_y + 1.;
                    id_x = id_x + 1.;
                }
                else if in_top_right == true {
                    id_y = id_y + 1.;
                }
            } else {
                in_top_left =
                    rel_loc_x / (self.dx() / self.radius()) < (rel_loc_y - self.radius() * 5. / 4.);
                in_top_right = (self.radius() * 1.25 - rel_loc_y)
                    <= (rel_loc_x - self.dx()) / (self.dx() / self.radius());
                if in_top_left == true {
                    id_y = id_y + 1.;
                    id_x = id_x - 1.;
                }
                else if in_top_right == true {
                    id_y = id_y + 1.;
                }
            }
        index[Ix2(cell_id, 0)] = id_x as i64;
        index[Ix2(cell_id, 1)] = id_y as i64;
        }
    index
    }

    pub fn cell_corners(&self, index: &ArrayView2<i64>) -> Array3<f64> {
        let mut corners = Array3::<f64>::zeros((index.shape()[0], 6, 2));

        for cell_id in 0..index.shape()[0] {
            for corner_id in 0..6 {
                let angle_deg = 60. * corner_id as f64 - 30.;
                let angle_rad = angle_deg * std::f64::consts::PI / 180.;
                let centroid = self.centroid_xy_no_rot(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
                corners[Ix3(cell_id, corner_id, 0)] = centroid.0 + self.radius() * angle_rad.cos();
                corners[Ix3(cell_id, corner_id, 1)] = centroid.1 + self.radius() * angle_rad.sin();
            }
        }

        if self.rotation != 0. {
            for cell_id in 0..corners.shape()[0] {
                for corner_id in 0..6 {
                    let mut corner_xy = corners.slice_mut(s![cell_id, corner_id, ..]);
                    let rotated_corner_xy = self.rotation_matrix.dot(&corner_xy);
                    corner_xy.assign(&rotated_corner_xy);
                }
            }
        }
        corners
    }
}