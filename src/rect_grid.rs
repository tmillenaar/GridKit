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
        let mut ids = Array2::<i64>::zeros((0, 2));
        let mut cell_id = self.cell_at_point(&p1.into_shape((1, p1.len())).unwrap());
        let _ = ids.push(Axis(0), array![cell_id[Ix2(0,0)], cell_id[Ix2(0,1)]].view());
        // let sides = array![
        //     [[corners[Ix3(0,0,0)], corners[Ix3(0,0,1)]],[corners[Ix3(0,1,0)],corners[Ix3(0,1,1)]]],
        //     [[corners[Ix3(0,1,0)], corners[Ix3(0,1,1)]],[corners[Ix3(0,2,0)],corners[Ix3(0,2,1)]]],
        //     [[corners[Ix3(0,2,0)], corners[Ix3(0,2,1)]],[corners[Ix3(0,3,0)],corners[Ix3(0,3,1)]]],
        //     [[corners[Ix3(0,3,0)], corners[Ix3(0,3,1)]],[corners[Ix3(0,0,0)],corners[Ix3(0,0,1)]]],
        // ];
        // for i in 0..corners.shape()[0] {

        let point1 = Coord::<f64> {x:p1[Ix1(0)], y:p1[Ix1(1)]};
        let point2 = Coord::<f64> {x:p2[Ix1(0)], y:p2[Ix1(1)]};
        let line = LineString::new(vec![point1, point2]);

        let mut intersection_id: i64 = -1;
        let mut counter = 0;
        loop {
            let corners = self.cell_corners(&cell_id.view());
            let sides = vec![
                LineString::new(vec![Coord::<f64> {x:corners[Ix3(0,0,0)], y:corners[Ix3(0,0,1)]},Coord::<f64> {x:corners[Ix3(0,1,0)], y:corners[Ix3(0,1,1)]}]),
                LineString::new(vec![Coord::<f64> {x:corners[Ix3(0,1,0)], y:corners[Ix3(0,1,1)]},Coord::<f64> {x:corners[Ix3(0,2,0)], y:corners[Ix3(0,2,1)]}]),
                LineString::new(vec![Coord::<f64> {x:corners[Ix3(0,2,0)], y:corners[Ix3(0,2,1)]},Coord::<f64> {x:corners[Ix3(0,3,0)], y:corners[Ix3(0,3,1)]}]),
                LineString::new(vec![Coord::<f64> {x:corners[Ix3(0,3,0)], y:corners[Ix3(0,3,1)]},Coord::<f64> {x:corners[Ix3(0,0,0)], y:corners[Ix3(0,0,1)]}]),
            ];
            
            let mut entered_through_side_id: i64;
            match intersection_id { // convert intersection_id of previous cell to id of same side for new cell_id
                0 => {entered_through_side_id = 2;}
                1 => {entered_through_side_id = 3;}
                2 => {entered_through_side_id = 0;}
                3 => {entered_through_side_id = 1;}
                _ => {entered_through_side_id = -1;}
            }

            intersection_id = -1;

            for i in 0..sides.len() {
                // TODO: Handle case where line intersects cell corner
                println!("{},{}", entered_through_side_id, intersection_id);
                if i as i64 == entered_through_side_id { // Discount the intersection towards previous cell
                    continue;
                }
                let intersects = sides[i].intersects(&line);
                if intersects {
                    intersection_id = i as i64;
                    break;
                }
                
            }
            println!("{:?}, {}", cell_id, intersection_id);
            match intersection_id {
                0 => { // Bottom side
                    cell_id[Ix2(0,1)] -= 1;
                    // cell_id = array![[cell_id[Ix2(0,0)], cell_id[Ix2(0,1)] - 1]];
                    // let _ = ids.push(Axis(0), array![cell_id[Ix2(0,0)], cell_id[Ix2(0,1)] - 1].view());
                }
                1 => { // Right side
                    cell_id[Ix2(0,0)] += 1;
                    // cell_id = array![[cell_id[Ix2(0,0)] + 1, cell_id[Ix2(0,1)]]];
                    // let _ = ids.push(Axis(0), array![cell_id[Ix2(0,0)] + 1, cell_id[Ix2(0,1)]].view());
                }
                2 => { // Top side
                    cell_id[Ix2(0,1)] += 1;
                    // cell_id = array![[cell_id[Ix2(0,0)], cell_id[Ix2(0,1)] + 1]];
                    // let _ = ids.push(Axis(0), array![cell_id[Ix2(0,0)], cell_id[Ix2(0,1)] + 1].view());
                }
                3 => { // Left side
                    cell_id[Ix2(0,0)] -= 1;
                    // cell_id = array![[cell_id[Ix2(0,0)] - 1, cell_id[Ix2(0,1)]]];
                    // let _ = ids.push(Axis(0), array![cell_id[Ix2(0,0)] - 1, cell_id[Ix2(0,1)]].view());
                }
                _ => { // No intersection
                    // Reached the end of the line, break infinite loop and return from function
                    break;
                }
            }
            let _ = ids.push(Axis(0), cell_id.slice(s![0, ..]).view());
            counter += 1;
            if counter == 50 {
                break
            }
        }

            // TODO: Remove previous intersection from sides before checking new intersection
            // TODO: Loop until no intersection

            

            // let projected_vec_1 = projected_1 - &sides.slice(s![i, 0, ..]);
            // let projected_vec_2 = projected_2 - &sides.slice(s![i, 1, ..]);
            // // Check if the direction of the projected arrow is the same
            // let same_x = (projected_vec_1[Ix1(0)] > 0.) == (projected_vec_2[Ix1(0)] > 0.);
            // let same_y = (projected_vec_1[Ix1(1)] > 0.) == (projected_vec_2[Ix1(1)] > 0.);
            // if (!same_x || !same_y) {
            //     // line crossing cell_side
            //     println!("{:?}, {:?}", projected_vec_1, projected_vec_2);
            // }
        // for points in sides.axis_iter(Axis(0)){}
        // ids.into()
        ids
        // let mut ids_arr = Array2::<f64>::zeros((ids.shape()[0], 2));
    }    
}
