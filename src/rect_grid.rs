use numpy::ndarray::*;
use crate::utils::*;
use crate::tile::*;
use crate::grid::*;

#[derive(Clone)]
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

    pub fn tiles_from_bounds(&self, bounds: (f64,f64,f64,f64), nr_tiles_x: i64, nr_tiles_y: i64) -> Vec<Tile> {
        let mut tiles = Vec::with_capacity((nr_tiles_x * nr_tiles_y) as usize);
        let origin = self.cell_at_point(&(array![[bounds.0 + self.dx() / 2., bounds.1 + self.dy() / 2.]]).view());
        let x_cells = origin[Ix2(0,0)]..(origin[Ix2(0,0)] + ((bounds.2 - bounds.0) / self.dx()) as i64);
        let y_cells = origin[Ix2(0,1)]..(origin[Ix2(0,1)] + ((bounds.3 - bounds.1) / self.dy()) as i64);

        let normal_nx = (x_cells.end - x_cells.start) / nr_tiles_x;
        let remainder_x = (x_cells.end - x_cells.start) - (nr_tiles_x - 1) * normal_nx;
        let normal_ny = (y_cells.end - y_cells.start) / nr_tiles_y;
        let remainder_y = (y_cells.end - y_cells.start) - (nr_tiles_y - 1) * normal_ny;
        
        let mut nx = normal_nx as u64;
        let mut ny = normal_ny as u64;

        // Add every row but the last one
        for i_y in 0..(nr_tiles_y-1) {
            nx = normal_nx as u64;
            // Add all but the last tiles in row
            for i_x in 0..(nr_tiles_x-1) {
                let start_id = (
                    origin[Ix2(0,0)] + i_x * normal_nx,
                    origin[Ix2(0,1)] + i_y * normal_ny
                );
                let grid = Grid::RectGrid(self.clone());
                // let grid = *self.clone();
                // let grid = Grid::RectGrid(grid);
                tiles.push(
                    Tile{grid, start_id, nx, ny}
                );
            }
            // Add last x-tile in row
            nx = remainder_x as u64;
            let start_id = (
                origin[Ix2(0,0)] + (nr_tiles_x-1) * normal_nx,
                origin[Ix2(0,1)] + i_y * normal_ny
            );
            let grid = Grid::RectGrid(self.clone());
            tiles.push(
                Tile{grid, start_id, nx, ny}
            );
        }
        // Add the final row, except the very last tile
        ny = remainder_y as u64;
        nx = normal_nx as u64;
        for i_x in 0..(nr_tiles_x-1) {
            let start_id = (
                origin[Ix2(0,0)] + i_x * normal_nx,
                origin[Ix2(0,1)] + (nr_tiles_y-1) * normal_ny
            );
            let grid = Grid::RectGrid(self.clone());
            tiles.push(
                Tile{grid, start_id, nx, ny}
            );
        }
        // Add the final tile
        nx = remainder_x as u64;
        let start_id = (
            origin[Ix2(0,0)] + (nr_tiles_x-1) * normal_nx,
            origin[Ix2(0,1)] + (nr_tiles_y-1) * normal_ny
        );
        let grid = Grid::RectGrid(self.clone());
        tiles.push(
            Tile{grid, start_id, nx, ny}
        );
        return tiles
    }
}
