use crate::grid::{GridTraits, Orientation};
use crate::utils::*;
use numpy::ndarray::Slice;
use numpy::ndarray::*;

#[derive(Clone)]
pub struct HexGrid {
    pub cellsize: f64,
    pub offset: [f64; 2],
    pub cell_orientation: Orientation,
    pub _rotation: f64,
    pub _rotation_matrix: Array2<f64>,
    pub _rotation_matrix_inv: Array2<f64>,
}

impl GridTraits for HexGrid {
    fn dx(&self) -> f64 {
        match self.cell_orientation {
            Orientation::Pointy => self.cellsize,
            Orientation::Flat => 3. / 2. * self.radius(),
        }
    }
    fn dy(&self) -> f64 {
        match self.cell_orientation {
            Orientation::Pointy => 3. / 2. * self.radius(),
            Orientation::Flat => self.cellsize,
        }
    }

    fn set_cellsize(&mut self, cellsize: f64) {
        self.cellsize = cellsize;
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
        self.cellsize / 3_f64.powf(0.5)
    }

    fn cell_height(&self) -> f64 {
        match self.cell_orientation {
            Orientation::Pointy => self.radius() * 2.,
            Orientation::Flat => self.dy(),
        }
    }

    fn cell_width(&self) -> f64 {
        match self.cell_orientation {
            Orientation::Pointy => self.dx(),
            Orientation::Flat => self.radius() * 2.,
        }
    }

    fn centroid_xy_no_rot(&self, x: i64, y: i64) -> [f64; 2] {
        let mut centroid_x = x as f64 * self.dx() + (self.dx() / 2.) + self.offset[0];
        let mut centroid_y = y as f64 * self.dy() + (self.dy() / 2.) + self.offset[1];

        match self.cell_orientation() {
            Orientation::Pointy => {
                if !iseven(y) {
                    centroid_x = centroid_x + self.dx() / 2.;
                }
            }
            Orientation::Flat => {
                if !iseven(x) {
                    centroid_y = centroid_y + self.dy() / 2.;
                }
            }
        }

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

    fn cell_at_point(&self, points: &ArrayView2<f64>) -> Array2<i64> {
        // For the sake of simplicity here I will define it all in terms of a pointy grid.
        // dx, dy etc will be used but they will refer to the consistent/inconsistent axes
        // so it also works for flat grids. While in the context of flat grids the meanings
        // are flipped, it is easier to read in terms of x and y axes rather than (in)consistent axes.
        let mut index = Array2::<i64>::zeros((points.shape()[0], 2));

        let dx = self.stepsize_consistent_axis();
        let dy = self.stepsize_inconsistent_axis();
        let id_x_axis = self.consistent_axis();
        let id_y_axis = self.inconsistent_axis();
        let offset_x = self.offset[id_x_axis];
        let offset_y = self.offset[id_y_axis];

        for cell_id in 0..points.shape()[0] {
            let point = points.slice(s![cell_id, ..]);
            let point = self._rotation_matrix_inv.dot(&point);

            let x = point[Ix1(id_x_axis)];
            let y = point[Ix1(id_y_axis)];

            // determine initial id_y
            let mut id_y = ((y - offset_y - self.radius() / 4.) / dy).floor();
            let is_offset = modulus(id_y, 2.) != 0.;
            let mut id_x: f64;

            // determine initial id_x
            if is_offset == true {
                id_x = (x - offset_x - dx / 2.) / dx;
            } else {
                id_x = (x - offset_x) / dx;
            }
            id_x = id_x.floor();

            // refine id_x and id_y
            // Example: points at the top of the cell's bounding box can be in this cell or in the cell to the top right or top left
            let rel_loc_y = modulus(y - offset_y - self.radius() / 4., dy) + self.radius() / 4.;
            let rel_loc_x = modulus(x - offset_x, dx);

            let mut in_top_left: bool;
            let mut in_top_right: bool;
            if is_offset == true {
                in_top_left = (self.radius() * 1.25 - rel_loc_y)
                    < ((rel_loc_x - 0.5 * dx) / (dx / self.radius()));
                in_top_left = in_top_left && (rel_loc_x < (0.5 * dx));
                in_top_right = (rel_loc_x - 0.5 * dx) / (dx / self.radius())
                    <= (rel_loc_y - self.radius() * 1.25);
                in_top_right = in_top_right && rel_loc_x >= (0.5 * dx);
                if in_top_left == true {
                    id_y = id_y + 1.;
                    id_x = id_x + 1.;
                } else if in_top_right == true {
                    id_y = id_y + 1.;
                }
            } else {
                in_top_left =
                    rel_loc_x / (dx / self.radius()) < (rel_loc_y - self.radius() * 5. / 4.);
                in_top_right =
                    (self.radius() * 1.25 - rel_loc_y) <= (rel_loc_x - dx) / (dx / self.radius());
                if in_top_left == true {
                    id_y = id_y + 1.;
                    id_x = id_x - 1.;
                } else if in_top_right == true {
                    id_y = id_y + 1.;
                }
            }

            index[Ix2(cell_id, id_x_axis)] = id_x as i64;
            index[Ix2(cell_id, id_y_axis)] = id_y as i64;
        }
        index
    }

    fn cell_corners(&self, index: &ArrayView2<i64>) -> Array3<f64> {
        let mut corners = Array3::<f64>::zeros((index.shape()[0], 6, 2));

        for cell_id in 0..index.shape()[0] {
            for corner_id in 0..6 {
                let angle_deg = match self.cell_orientation() {
                    Orientation::Pointy => 60. * corner_id as f64 - 30.,
                    Orientation::Flat => 60. * corner_id as f64,
                };
                let angle_rad = angle_deg * std::f64::consts::PI / 180.;
                let centroid =
                    self.centroid_xy_no_rot(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
                corners[Ix3(cell_id, corner_id, 0)] = centroid[0] + self.radius() * angle_rad.cos();
                corners[Ix3(cell_id, corner_id, 1)] = centroid[1] + self.radius() * angle_rad.sin();
            }
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
        // There are 6 options for the nearby cells,
        // based on where in a cell the point is located.
        // Hence there are 6 sections within the cell,
        // each 60 degrees wide (360/6 = 60).
        // We divide the quadrants based on the angle between
        // the point and pure up, as measured from the centroid of the cell.
        let mut nearby_cells = Array3::<i64>::zeros((points.shape()[0], 3, 2));

        // Since points are rotated in cell_at_point, perform this call before
        // rotating the points in this function.
        // Ideally we have a version cell_at_point that does not rotate, same
        // as we do with centroid_xy_no_rot.
        let cell_ids = self.cell_at_point(&points);

        // Rotate points only if nesecary
        // We only want to clone/copy the data in the points variable
        // if it needs rotating. This is hard for the Rust compiler
        // because points cannot be either owned or a view depending on a condition.
        // It needs to either always be owned or always be a view.
        // Since we don't always want to own the data we make sure that
        // the variable points is always a view.
        // However, the rotated data does need to be stored in a different
        // variable that lives at least as long as the view.
        // To satisfy the compiler we declare points_ before the if-block and
        // then return the view from the if-block which we use to shadow the original
        // points variable.
        let mut points_: Array2<f64>;
        let points: ArrayView2<f64> = if self.rotation() != 0. {
            // Create an owned copy of `points` and apply rotation.
            points_ = points.to_owned();
            for cell_id in 0..points.shape()[0] {
                let mut point = points_.slice_mut(s![cell_id, ..]);
                let point_rot = self._rotation_matrix_inv.dot(&point);
                point.assign(&point_rot);
            }
            points_.view()
        } else {
            points.view()
        };

        for cell_id in 0..points.shape()[0] {
            // Determine the azimuth based on the direction vector from the cell centroid to the point
            let centroid =
                self.centroid_xy_no_rot(cell_ids[Ix2(cell_id, 0)], cell_ids[Ix2(cell_id, 1)]);
            let direction_x = points[Ix2(cell_id, 0)] - centroid[0];
            let direction_y = points[Ix2(cell_id, 1)] - centroid[1];
            let mut azimuth = direction_x.atan2(direction_y) * 180. / std::f64::consts::PI;
            match self.cell_orientation() {
                // it is easier to work with ranges starting at 0 rather than -30;
                Orientation::Pointy => {
                    azimuth += 30.;
                }
                Orientation::Flat => {}
            }
            // make sure azimuth is inbetween 0 and 360, and not between -180 and 180
            azimuth += 360. * (azimuth <= 0.) as i64 as f64;
            azimuth -= 360. * (azimuth > 360.) as i64 as f64;
            // Determine which of the 6 quadratns is in based on the azimuth
            let quadrant_id = (azimuth / 60.).floor() as usize;
            // The first nearby cell is the cell the point is located in.
            // So this point can be excluded from the slice.
            // It is already 0,0 which is the correct relative ID
            let mut nearby_cells_slice = nearby_cells.slice_mut(s![cell_id, 1.., ..]);

            // Offset the nearby cells by one depending whether the cell points up or down
            let is_offset = !iseven(cell_ids[Ix2(cell_id, self.inconsistent_axis())]) as i64;
            match self.cell_orientation() {
                Orientation::Pointy => {
                    match quadrant_id {
                        0 => {
                            // azimuth 0-60
                            nearby_cells_slice.assign(&array![[-1, 1], [0, 1]]);
                            nearby_cells[Ix3(cell_id, 1, 0)] += 1 * is_offset;
                            nearby_cells[Ix3(cell_id, 2, 0)] += 1 * is_offset;
                        }
                        1 => {
                            // azimuth 60-120
                            nearby_cells_slice.assign(&array![[0, 1], [1, 0]]);
                            nearby_cells[Ix3(cell_id, 1, 0)] += 1 * is_offset;
                        }
                        2 => {
                            // azimuth 120-180
                            nearby_cells_slice.assign(&array![[1, 0], [0, -1]]);
                            nearby_cells[Ix3(cell_id, 2, 0)] += 1 * is_offset;
                        }
                        3 => {
                            // azimuth 180-240
                            nearby_cells_slice.assign(&array![[0, -1], [-1, -1]]);
                            nearby_cells[Ix3(cell_id, 1, 0)] += 1 * is_offset;
                            nearby_cells[Ix3(cell_id, 2, 0)] += 1 * is_offset;
                        }
                        4 => {
                            // azimuth 240- 300
                            nearby_cells_slice.assign(&array![[-1, -1], [-1, 0]]);
                            nearby_cells[Ix3(cell_id, 1, 0)] += 1 * is_offset;
                        }
                        _ => {
                            // azimuth 300- 360
                            nearby_cells_slice.assign(&array![[-1, 0], [-1, 1]]);
                            nearby_cells[Ix3(cell_id, 2, 0)] += 1 * is_offset;
                        }
                    }
                }
                Orientation::Flat => {
                    match quadrant_id {
                        0 => {
                            // azimuth 0-60
                            nearby_cells_slice.assign(&array![[1, 0], [0, 1]]);
                            nearby_cells[Ix3(cell_id, 1, 1)] += 1 * is_offset;
                        }
                        1 => {
                            // azimuth 60-120
                            nearby_cells_slice.assign(&array![[1, -1], [1, 0]]);
                            nearby_cells[Ix3(cell_id, 1, 1)] += is_offset;
                            nearby_cells[Ix3(cell_id, 2, 1)] += is_offset;
                        }
                        2 => {
                            // azimuth 120-180
                            nearby_cells_slice.assign(&array![[0, -1], [1, -1]]);
                            nearby_cells[Ix3(cell_id, 2, 1)] += 1 * is_offset;
                        }
                        3 => {
                            // azimuth 180-240
                            nearby_cells_slice.assign(&array![[-1, -1], [0, -1]]);
                            nearby_cells[Ix3(cell_id, 1, 1)] += 1 * is_offset;
                        }
                        4 => {
                            // azimuth 240- 300
                            nearby_cells_slice.assign(&array![[-1, 0], [-1, -1]]);
                            nearby_cells[Ix3(cell_id, 1, 1)] += 1 * is_offset;
                            nearby_cells[Ix3(cell_id, 2, 1)] += 1 * is_offset;
                        }
                        _ => {
                            // azimuth 300- 360
                            nearby_cells_slice.assign(&array![[0, 1], [-1, 0]]);
                            nearby_cells[Ix3(cell_id, 2, 1)] += 1 * is_offset;
                        }
                    }
                }
            }

            // Add cell ID to relative ID
            for nearby_cell_id in 0..3 {
                nearby_cells[Ix3(cell_id, nearby_cell_id, 0)] += cell_ids[Ix2(cell_id, 0)];
                nearby_cells[Ix3(cell_id, nearby_cell_id, 1)] += cell_ids[Ix2(cell_id, 1)];
            }
        }
        nearby_cells
    }
}

impl HexGrid {
    pub fn new(cellsize: f64, cell_orientation: Orientation) -> Self {
        let _rotation_matrix = rotation_matrix_from_angle(0.);
        let _rotation_matrix_inv = rotation_matrix_from_angle(-0.);
        HexGrid {
            cellsize,
            offset: [0., 0.],
            cell_orientation,
            _rotation: 0.,
            _rotation_matrix,
            _rotation_matrix_inv,
        }
    }

    pub fn cell_orientation(&self) -> &Orientation {
        &self.cell_orientation
    }

    pub fn set_cell_orientation(&mut self, cell_orientation: Orientation) {
        self.cell_orientation = cell_orientation;
    }

    fn consistent_axis(&self) -> usize {
        match self.cell_orientation {
            Orientation::Pointy => 0,
            Orientation::Flat => 1,
        }
    }

    fn inconsistent_axis(&self) -> usize {
        match self.cell_orientation {
            Orientation::Pointy => 1,
            Orientation::Flat => 0,
        }
    }

    fn stepsize_consistent_axis(&self) -> f64 {
        match self.cell_orientation {
            Orientation::Pointy => self.dx(),
            Orientation::Flat => self.dy(),
        }
    }

    fn stepsize_inconsistent_axis(&self) -> f64 {
        match self.cell_orientation {
            Orientation::Pointy => self.dy(),
            Orientation::Flat => self.dx(),
        }
    }
}
