use geo::intersects;
use numpy::ndarray::*;
use crate::utils::*;
use crate::interpolate::*;

use geo_types::{LineString, Point, Coord, Geometry};
use geo::Intersects;

pub struct RectGrid {
    pub _dx: f64,
    pub _dy: f64,
    pub offset: (f64, f64),
    pub rotation: f64,
    pub rotation_matrix: Array2<f64>,
    pub rotation_matrix_inv: Array2<f64>,
}

impl RectGrid {
    pub fn new(dx: f64, dy: f64, offset: (f64, f64), rotation: f64) -> Self {
        let rotation_matrix = _rotation_matrix(rotation);
        let rotation_matrix_inv = _rotation_matrix(-rotation);
        let offset = normalize_offset(offset, dx, dy);
        let (_dx, _dy) = (dx, dy); // rename in order to pass to struct
        RectGrid { _dx, _dy, offset, rotation, rotation_matrix, rotation_matrix_inv }
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
        let centroid_x = x as f64 * self.dx() + (self.dx() / 2.) + self.offset.0;
        let centroid_y = y as f64 * self.dy() + (self.dy() / 2.) + self.offset.1;
        (centroid_x, centroid_y)
    }

    pub fn cell_at_point(&self, points: &ArrayView2<f64>) -> Array2<i64> {
        let shape = points.shape();
        let mut index = Array2::<i64>::zeros((shape[0], shape[1]));
        for cell_id in 0..points.shape()[0] {
            let point = points.slice(s![cell_id, ..]);
            let point = self.rotation_matrix_inv.dot(&point);
            let id_x = ((point[Ix1(0)] - self.offset.0) / self.dx()).floor() as i64;
            let id_y = ((point[Ix1(1)] - self.offset.1) / self.dy()).floor() as i64;
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
            let (centroid_x, centroid_y) = self.centroid_xy_no_rot(id_x, id_y);
            corners[Ix3(cell_id, 0, 0)] = centroid_x - self.dx() / 2.;
            corners[Ix3(cell_id, 0, 1)] = centroid_y - self.dy() / 2.;
            corners[Ix3(cell_id, 1, 0)] = centroid_x + self.dx() / 2.;
            corners[Ix3(cell_id, 1, 1)] = centroid_y - self.dy() / 2.;
            corners[Ix3(cell_id, 2, 0)] = centroid_x + self.dx() / 2.;
            corners[Ix3(cell_id, 2, 1)] = centroid_y + self.dy() / 2.;
            corners[Ix3(cell_id, 3, 0)] = centroid_x - self.dx() / 2.;
            corners[Ix3(cell_id, 3, 1)] = centroid_y + self.dy() / 2.;
        }
        
        if self.rotation != 0. {
            for cell_id in 0..corners.shape()[0] {
                for corner_id in 0..corners.shape()[1] {
                    let mut corner_xy = corners.slice_mut(s![cell_id, corner_id, ..]);
                    let rotated_corner_xy = self.rotation_matrix.dot(&corner_xy);
                    corner_xy.assign(&rotated_corner_xy);
                }
            }
        }
        corners
    }

    pub fn cells_near_point(&self, points: &ArrayView2<f64>) -> Array3<i64> {

        let mut nearby_cells = Array3::<i64>::zeros((points.shape()[0], 4, 2));
        let index = self.cell_at_point(points);

        // FIXME: Find a way to not clone points in the case of no rotation
        //        If points is made mutable within the conditional, it is dropped from scope and nothing changed
        let mut points = points.to_owned();
        if self.rotation != 0. {
            for cell_id in 0..points.shape()[0] {
                let mut point = points.slice_mut(s![cell_id, ..]);
                let point_rot = self.rotation_matrix_inv.dot(&point);
                point.assign(&point_rot);
            }
        }

        for cell_id in 0..points.shape()[0] {
            let rel_loc_x: f64 = modulus((points[Ix2(cell_id, 0)] - self.offset.0), self.dx());
            let rel_loc_y: f64 = modulus((points[Ix2(cell_id, 1)] - self.offset.1), self.dy());
            let id_x = index[Ix2(cell_id, 0)];
            let id_y = index[Ix2(cell_id, 1)];
            match (rel_loc_x, rel_loc_y) {
                // Top-left quadrant
                (x, y) if x <= self.dx() / 2. && y >= self.dy() / 2. => {
                    nearby_cells[Ix3(cell_id, 0, 0)] = -1 + id_x;
                    nearby_cells[Ix3(cell_id, 0, 1)] =  1 + id_y;
                    nearby_cells[Ix3(cell_id, 1, 0)] =  0 + id_x;
                    nearby_cells[Ix3(cell_id, 1, 1)] =  1 + id_y;
                    nearby_cells[Ix3(cell_id, 2, 0)] = -1 + id_x;
                    nearby_cells[Ix3(cell_id, 2, 1)] =  0 + id_y;
                    nearby_cells[Ix3(cell_id, 3, 0)] =  0 + id_x;
                    nearby_cells[Ix3(cell_id, 3, 1)] =  0 + id_y;
                }
                // Top-right quadrant
                (x, y) if x >= self.dx() / 2. && y >= self.dy() / 2. => {
                    nearby_cells[Ix3(cell_id, 0, 0)] =  0 + id_x;
                    nearby_cells[Ix3(cell_id, 0, 1)] =  1 + id_y;
                    nearby_cells[Ix3(cell_id, 1, 0)] =  1 + id_x;
                    nearby_cells[Ix3(cell_id, 1, 1)] =  1 + id_y;
                    nearby_cells[Ix3(cell_id, 2, 0)] =  0 + id_x;
                    nearby_cells[Ix3(cell_id, 2, 1)] =  0 + id_y;
                    nearby_cells[Ix3(cell_id, 3, 0)] =  1 + id_x;
                    nearby_cells[Ix3(cell_id, 3, 1)] =  0 + id_y;
                }
                // Bottom-left quadrant
                (x, y) if x <= self.dx() / 2. && y <= self.dy() / 2. => {
                    nearby_cells[Ix3(cell_id, 0, 0)] = -1 + id_x;
                    nearby_cells[Ix3(cell_id, 0, 1)] =  0 + id_y;
                    nearby_cells[Ix3(cell_id, 1, 0)] =  0 + id_x;
                    nearby_cells[Ix3(cell_id, 1, 1)] =  0 + id_y;
                    nearby_cells[Ix3(cell_id, 2, 0)] = -1 + id_x;
                    nearby_cells[Ix3(cell_id, 2, 1)] = -1 + id_y;
                    nearby_cells[Ix3(cell_id, 3, 0)] =  0 + id_x;
                    nearby_cells[Ix3(cell_id, 3, 1)] = -1 + id_y;
                }
                // Bottom-right quadrant
                _ => {
                    nearby_cells[Ix3(cell_id, 0, 0)] =  0 + id_x;
                    nearby_cells[Ix3(cell_id, 0, 1)] =  0 + id_y;
                    nearby_cells[Ix3(cell_id, 1, 0)] =  1 + id_x;
                    nearby_cells[Ix3(cell_id, 1, 1)] =  0 + id_y;
                    nearby_cells[Ix3(cell_id, 2, 0)] =  0 + id_x;
                    nearby_cells[Ix3(cell_id, 2, 1)] = -1 + id_y;
                    nearby_cells[Ix3(cell_id, 3, 0)] =  1 + id_x;
                    nearby_cells[Ix3(cell_id, 3, 1)] = -1 + id_y;
                }
            }
        }
        nearby_cells
    }

    pub fn cells_intersecting_line(&self, p1: &ArrayView1<f64>, p2: &ArrayView1<f64>) -> Array2<i64> {
        // This function returns the ids of the cells that intersect the line defined by the outer points p1 and p2.
        // This is done through an infinite loop where we start at the cell that contains p1, 
        // check if any of the line intersects any of the corners or sides of the cell and depending
        // on which side or corner is intersected, we find the next cell and repeat the process.
        // Since the line also has an intersection where it 'entered' the new cell,
        // we ignore this intersection and look for anohter.
        // The loop is terminated when the line does not intersect any new corners or lines,
        // or when corner or cell that intersects the line also intersects point2, the end of the line.
        // Since the corners are also part of the lines as far as the line.intersects function is concerned,
        // we check the corners first and skip the lines check if a corner is intersected. 
        // We then also have to ignore the lines that contain the corner when checking intersections for the next cell.
        //
        // The layout of the corners and lines with their indices are counter-clockwise starting at the bottom-left:
        //
        // C3 -- L2 -- C2
        //  |          |
        // L3          L1
        //  |          |
        // C0 -- L0 -- C1
        //
        // Where C0 represents the first corner (corner index 0), C1 represents corner index 1 and so forth.
        // The sampe applies to the lines, where L0 is the first line.
        //
        let mut ids = Array2::<i64>::zeros((0, 2));
        let mut cell_id = self.cell_at_point(&p1.into_shape((1, p1.len())).unwrap());

        // Create a LineString from the supplied endpoints
        let point1 = Coord::<f64> {x:p1[Ix1(0)], y:p1[Ix1(1)]};
        let point2 = Coord::<f64> {x:p2[Ix1(0)], y:p2[Ix1(1)]};
        let line = LineString::new(vec![point1, point2]);

        let mut side_intersection_id: i64 = -1;
        let mut corner_intersection_id: i64 = -1;
        let mut skip_corners = array![false, false, false, false];
        let mut skip_sides = array![false, false, false, false];

        // Check if the line starts on a cell corner. If so, 
        // find out which of the connecting cells is the true starting cell and
        // add the starting corner to 'skip_corners' when determining the next cell.
        let mut p1_on_corner: bool = false;
        let corners = self.cell_corners(&cell_id.view());
        for i in 0..4 { // loop over corners
            p1_on_corner = point1.intersects(&Coord::<f64> {x:corners[Ix3(0,i,0)], y:corners[Ix3(0,i,1)]});
            if p1_on_corner {
                break
            }
        }
        if p1_on_corner {
            let direction_x = p2[Ix1(0)] - p1[Ix1(0)];
            let direction_y = p2[Ix1(1)] - p1[Ix1(1)];
            let mut azimuth = direction_x.atan2(direction_y) * 180. / std::f64::consts::PI;
            azimuth += 360. * (azimuth <= 0.) as i64 as f64;
            azimuth -= 360. * (azimuth > 360.) as i64 as f64;
            match azimuth { // Line continues in top-right direction
                az if (az >= 0.0 && az <= 90.0) => {
                    // Already in correct starting cell
                    skip_corners = array![true, false, false, false];
                    skip_sides = array![true, false, false, true];
                } // Line continues in bottom-right direction
                az if (az > 90.0 && az <= 180.0) => {
                    cell_id[Ix2(0,1)] -= 1;
                    skip_corners = array![false, false, false, true];
                    skip_sides = array![false, false, true, true];
                } // Line continues in bottom-left direction
                az if (az > 180.0 && az <= 270.0) => {
                    cell_id[Ix2(0,0)] -= 1;
                    cell_id[Ix2(0,1)] -= 1;
                    skip_corners = array![false, false, true, false];
                    skip_sides = array![false, true, true, false];
                }
                _ => { // Line continues in top-left direction
                    cell_id[Ix2(0,0)] -= 1;
                    skip_corners = array![false, true, false, false];
                    skip_sides = array![true, true, false, false];
                }
            }
        }

        // Also check if the line starts on a cell side. If so, 
        // find out which of the connecting cells is the true starting cell and
        // add the starting side to 'skip_sides' when determining the next cell.
        let mut p1_on_side: i8 = -1;
        let sides = vec![
                LineString::new(vec![Coord::<f64> {x:corners[Ix3(0,0,0)], y:corners[Ix3(0,0,1)]},Coord::<f64> {x:corners[Ix3(0,1,0)], y:corners[Ix3(0,1,1)]}]),
                LineString::new(vec![Coord::<f64> {x:corners[Ix3(0,1,0)], y:corners[Ix3(0,1,1)]},Coord::<f64> {x:corners[Ix3(0,2,0)], y:corners[Ix3(0,2,1)]}]),
                LineString::new(vec![Coord::<f64> {x:corners[Ix3(0,2,0)], y:corners[Ix3(0,2,1)]},Coord::<f64> {x:corners[Ix3(0,3,0)], y:corners[Ix3(0,3,1)]}]),
                LineString::new(vec![Coord::<f64> {x:corners[Ix3(0,3,0)], y:corners[Ix3(0,3,1)]},Coord::<f64> {x:corners[Ix3(0,0,0)], y:corners[Ix3(0,0,1)]}]),
            ];
        for i in 0..4 { // loop over sides
            let intersects = sides[i].intersects(&point1);
            if intersects {
                p1_on_side = i as i8;
                break
            }
        }
        match p1_on_side {
            // Because of the way cell_at_point handles the boundary conditions,
            // we only need to check for the point being on the left or bottom sides.
            // Points on the right or top sides are considered to be in the next cell over.
            0 => { // Bottom-left corner
                if p2[Ix1(1)] > p1[Ix1(1)] { // Line continues towards the top
                    // cell_id already correct
                    skip_sides = array![true, false, false, false];
                } else { // Line continues towards the bottom
                    cell_id[Ix2(0,1)] -= 1;
                    skip_sides = array![false, false, true, false];
                }
            }
            3 => { // Top-left corner
                if p2[Ix1(0)] > p1[Ix1(0)] { // Line continues towards the right
                    // cell_id already correct
                    skip_sides = array![false, false, false, true];
                } else { // Line continues towards the left
                    cell_id[Ix2(0,0)] -= 1;
                    skip_sides = array![false, true, false, false];
                }
            }
            _ => {} // No intersection, check sides next
        }

        // Add starting cell to return ids
        let _ = ids.push(Axis(0), array![cell_id[Ix2(0,0)], cell_id[Ix2(0,1)]].view());

        // Loop and continuousely find the next cell until the end of the line is reached.
        loop {
            let corners = self.cell_corners(&cell_id.view());
            let mut reached_point2 = false;
            corner_intersection_id = -1;
            for i in 0..4 { // Loop over corners
                if skip_corners[i] { // Discount the intersection towards previous cell
                    continue;
                }
                let corner_coord = Coord::<f64> {x:corners[Ix3(0,i,0)], y:corners[Ix3(0,i,1)]};
                let intersects = line.intersects(&corner_coord);
                if intersects {
                    if corner_coord.intersects(&point2) {
                        reached_point2 = true;
                    }
                    corner_intersection_id = i as i64;
                    break;
                }
            }

            if reached_point2 {
                // Line ends in this cell, break infinite loop and return
                break
            }

            match corner_intersection_id {
                // Adjust the cell-id to reflect the next cell and mark the oppisite corner and connecting sides to be skipped.
                // To demonstrate what I mean with skipping the opposite corner and connecting sides:
                //   If the line now intersects the top-right corner, from the perspective of the next cell
                //   the line intersects the bottom-left corner, which is the one we want to ignore in the next iteration.
                //   This corner also belongs to the bottom and left sides, which we also want to ignore in the next iteration.
                0 => { // Bottom-left corner
                    cell_id[Ix2(0,0)] -= 1;
                    cell_id[Ix2(0,1)] -= 1;
                    skip_corners = array![false, false, true, false];
                    skip_sides = array![false, true, true, false];
                }
                1 => { // Bottom-right corner
                    cell_id[Ix2(0,0)] += 1;
                    cell_id[Ix2(0,1)] -= 1;
                    skip_corners = array![false, false, false, true];
                    skip_sides = array![false, false, true, true];
                }
                2 => { // Top-right corner
                    cell_id[Ix2(0,0)] += 1;
                    cell_id[Ix2(0,1)] += 1;
                    skip_corners = array![true, false, false, false];
                    skip_sides = array![true, false, false, true];
                }
                3 => { // Top-left corner
                    cell_id[Ix2(0,0)] -= 1;
                    cell_id[Ix2(0,1)] += 1;
                    skip_corners = array![false, true, false, false];
                    skip_sides = array![true, true, false, false];
                }
                _ => {} // No intersection, check sides next
            }

            // Add previous cell to vec and don't bother checking side intersections if we have a corner intersection
            if corner_intersection_id != -1 {
                let _ = ids.push(Axis(0), cell_id.slice(s![0, ..]).view());
                continue
            }

            // Since there is no corner intersection, reset skip_corners
            skip_corners = array![false, false, false, false];

            // Check insersection on sides
            let sides = vec![
                LineString::new(vec![Coord::<f64> {x:corners[Ix3(0,0,0)], y:corners[Ix3(0,0,1)]},Coord::<f64> {x:corners[Ix3(0,1,0)], y:corners[Ix3(0,1,1)]}]),
                LineString::new(vec![Coord::<f64> {x:corners[Ix3(0,1,0)], y:corners[Ix3(0,1,1)]},Coord::<f64> {x:corners[Ix3(0,2,0)], y:corners[Ix3(0,2,1)]}]),
                LineString::new(vec![Coord::<f64> {x:corners[Ix3(0,2,0)], y:corners[Ix3(0,2,1)]},Coord::<f64> {x:corners[Ix3(0,3,0)], y:corners[Ix3(0,3,1)]}]),
                LineString::new(vec![Coord::<f64> {x:corners[Ix3(0,3,0)], y:corners[Ix3(0,3,1)]},Coord::<f64> {x:corners[Ix3(0,0,0)], y:corners[Ix3(0,0,1)]}]),
            ];

            side_intersection_id = -1;
            for i in 0..sides.len() {
                if skip_sides[i] { // Discount the intersection towards previous cell
                    continue;
                }
                let intersects = sides[i].intersects(&line);
                if intersects {
                    if sides[i].intersects(&point2) {
                        reached_point2 = true;
                    }
                    side_intersection_id = i as i64;
                    break;
                }
            }

            if reached_point2 {
                // Line ends in this cell, break infinite loop and return
                break
            }

            match side_intersection_id {
                // Adjust the cell-id to reflect the next cell and mark the oppisite side to be skipped.
                // To demonstrate what I mean with skipping the opposite side:
                //   If the line now intersects the top side, from the perspective of the next cell
                //   the line intersects the bottom side, which is the one we want to ignore in the next iteration.
                0 => { // Bottom side
                    cell_id[Ix2(0,1)] -= 1;
                    skip_sides = array![false, false, true, false];
                }
                1 => { // Right side
                    cell_id[Ix2(0,0)] += 1;
                    skip_sides = array![false, false, false, true];
                }
                2 => { // Top side
                    cell_id[Ix2(0,1)] += 1;
                    skip_sides = array![true, false, false, false];
                }
                3 => { // Left side
                    cell_id[Ix2(0,0)] -= 1;
                    skip_sides = array![false, true, false, false];
                }
                _ => { // No intersection
                    // Reached the end point of the line, break infinite loop and return from function
                    break;
                }
            }
            let _ = ids.push(Axis(0), cell_id.slice(s![0, ..]).view());
        }
        return ids
    }
        
}
