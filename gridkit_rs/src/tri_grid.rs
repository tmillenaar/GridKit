use numpy::ndarray::*;

fn iseven(val: i64) -> bool {
    val % 2 == 0
}
pub struct TriGrid {
    pub cellsize: f64,
}

impl TriGrid {

    pub fn new(cellsize: f64) -> Self {
        TriGrid{
            cellsize,
        }
    }

    pub fn cell_height(&self) -> f64 {
        self.cellsize * (3_f64).sqrt() / 2.
    }

    pub fn radius(&self) -> f64 {
        2. / 3. * self.cell_height()
    }

    pub fn cell_width(&self) -> f64 {
        self.cellsize
    }

    pub fn dx(&self) -> f64 {
        self.cellsize / 2.
    }
    
    pub fn dy(&self) -> f64 {
        self.cell_height()
    }

    pub fn centroid(
        &self,
        index: &ArrayView2<i64>,
    ) -> Array2<f64> {
        let mut centroids = Array2::<f64>::zeros((index.shape()[0], 2));
        
        for cell_id in 0..centroids.shape()[0] {
            centroids[Ix2(cell_id, 0)] = index[Ix2(cell_id, 0)] as f64 * self.dx() - (self.dx() / 2.);
            centroids[Ix2(cell_id, 1)] = index[Ix2(cell_id, 1)] as f64 * self.dy() - (self.dy() / 2.);
    
            let vertical_offset = self.radius() - 0.5 * self.dy();
            if iseven(index[Ix2(cell_id, 0)]) == iseven(index[Ix2(cell_id, 1)]) {
                centroids[Ix2(cell_id, 1)]  = centroids[Ix2(cell_id, 1)]  + vertical_offset;
            } else {
                centroids[Ix2(cell_id, 1)]  = centroids[Ix2(cell_id, 1)]  - vertical_offset;
            }
        }
        centroids
    }

    pub fn cell_corners(
        &self,
        index: &ArrayView2<i64>,
    ) -> Array3<f64> {
        let centroids = self.centroid(index);
        let mut corners = Array3::<f64>::zeros((index.shape()[0], 3, 2));
    
        for cell_id in 0..corners.shape()[0] {
            if iseven(index[Ix2(cell_id, 0)]) == iseven(index[Ix2(cell_id, 1)]) {
                corners[Ix3(cell_id, 0, 0)] = centroids[Ix2(cell_id, 0)];
                corners[Ix3(cell_id, 0, 1)] = centroids[Ix2(cell_id, 1)] - self.radius();
                corners[Ix3(cell_id, 1, 0)] = centroids[Ix2(cell_id, 0)] + self.dx();
                corners[Ix3(cell_id, 1, 1)] = centroids[Ix2(cell_id, 1)] + (self.cell_height() - self.radius());
                corners[Ix3(cell_id, 2, 0)] = centroids[Ix2(cell_id, 0)] - self.dx();
                corners[Ix3(cell_id, 2, 1)] = centroids[Ix2(cell_id, 1)] + (self.cell_height() - self.radius());
            } else {
                corners[Ix3(cell_id, 0, 0)] = centroids[Ix2(cell_id, 0)];
                corners[Ix3(cell_id, 0, 1)] = centroids[Ix2(cell_id, 1)] + self.radius();
                corners[Ix3(cell_id, 1, 0)] = centroids[Ix2(cell_id, 0)] + self.dx();
                corners[Ix3(cell_id, 1, 1)] = centroids[Ix2(cell_id, 1)] - (self.cell_height() - self.radius());
                corners[Ix3(cell_id, 2, 0)] = centroids[Ix2(cell_id, 0)] - self.dx();
                corners[Ix3(cell_id, 2, 1)] = centroids[Ix2(cell_id, 1)] - (self.cell_height() - self.radius());
            }
        }
        corners
    }

    pub fn cell_at_point(
        &self,
        points: &ArrayView2<f64>,
    ) -> Array2<i64> {
        let mut index = Array2::<i64>::zeros((points.shape()[0], 2));
        for cell_id in 0..points.shape()[0] {
            index[Ix2(cell_id, 0)] = (1. + (points[Ix2(cell_id, 0)]) / self.dx()).floor() as i64;
            index[Ix2(cell_id, 1)] = (1. + points[Ix2(cell_id, 1)] / self.dy()).floor() as i64;

            let rel_loc_x: f64 = (points[Ix2(cell_id, 0)].abs()) % self.dx();
            let rel_loc_y: f64 = points[Ix2(cell_id, 1)].abs() % self.dy();

            let mut downward_cell = iseven(index[Ix2(cell_id, 0)]) != iseven(index[Ix2(cell_id, 1)]);
            if index[Ix2(cell_id, 1)] > 0 {
                downward_cell = !downward_cell;
            }
            if downward_cell {
                // Bottom left of square that belongs to triangle to the left
                let is_inside = (rel_loc_x / self.dx()) + self.cell_width() / 2. > (self.cell_height() - rel_loc_y) / self.dy();
                if !is_inside {
                    let factor = if index[Ix2(cell_id, 0)] > 0 {-1} else {1};
                    index[Ix2(cell_id, 0)] = index[Ix2(cell_id, 0)] + factor;
                }
                // Bottom right of square that belongs to triangle to the right
                let is_inside = (rel_loc_x / self.dx()) - self.cell_width() / 2. < (rel_loc_y) / self.dy();
                if !is_inside {
                    let factor = if index[Ix2(cell_id, 0)] > 0 {1} else {-1};
                    index[Ix2(cell_id, 0)] = index[Ix2(cell_id, 0)] + factor;
                }
            } else {
                // Top left of square that belongs to triangle to the left
                let is_inside = (rel_loc_x / self.dx()) + self.cell_width() / 2. > rel_loc_y / self.dy();
                if !is_inside {
                    let factor = if index[Ix2(cell_id, 0)] > 0 {-1} else {1};
                    index[Ix2(cell_id, 0)] = index[Ix2(cell_id, 0)] + factor;
                }
                // Top right of square that belongs to triangle to the right
                let is_inside = (rel_loc_x / self.dx()) - self.cell_width() / 2. < (self.cell_height() - rel_loc_y) / self.dy();
                if !is_inside {
                    let factor = if index[Ix2(cell_id, 0)] > 0 {1} else {-1};
                    index[Ix2(cell_id, 0)] = index[Ix2(cell_id, 0)] + factor;
                }
            }
        }
        index
    }

    pub fn cells_in_bounds(
        &self,
        bounds: &(f64, f64, f64, f64),
    ) -> (Array2<i64>, (usize, usize)) {
        
        // Get ids of the cells at diagonally opposing corners of the bounds
        // TODO: allow calling of cell_at_point with single point (tuple or 1d array)
        let left_bottom: Array2<f64> = array![[bounds.0 + self.dx() / 4., bounds.1 + self.dy() / 4.]];
        let right_top: Array2<f64> = array![[bounds.2 - self.dx() / 4., bounds.3 - self.dy() / 4.]];
        
        // translate the coordinates of the corner cells into indices
        let left_bottom_id = self.cell_at_point(&left_bottom.view());
        let right_top_id = self.cell_at_point(&right_top.view());

        // use the cells at the corners to determine 
        // the ids in x an y direction as separate 1d arrays
        let minx = left_bottom_id[Ix2(0, 0)] - 2;
        let maxx = right_top_id[Ix2(0, 0)] + 2;

        let miny = left_bottom_id[Ix2(0, 1)] - 2;
        let maxy = right_top_id[Ix2(0, 1)] + 2;

        // fill raveled meshgrid if the centroid is in the bounds
        let nr_cells_x: usize = ((bounds.2 - bounds.0) / self.dx()).round() as usize;
        let nr_cells_y: usize = ((bounds.3 - bounds.1) / self.dy()).round() as usize;
        let mut index = Array2::<i64>::zeros((nr_cells_x * nr_cells_y, 2));
        let mut cell_id: usize = 0;
        let mut centroid: Array2<f64>;
        for y in (miny..=maxy).rev() {
            for x in minx..=maxx {
                centroid = self.centroid(&array!([x, y]).view()); // TODO: allow for calls using tuple or 1d array
                if (centroid[Ix2(0, 0)] > bounds.0) & // x > minx
                   (centroid[Ix2(0, 0)] < bounds.2) & // x < maxx
                   (centroid[Ix2(0, 1)] > bounds.1) & // y > miny
                   (centroid[Ix2(0, 1)] < bounds.3) { // y < maxy
                        index[Ix2(cell_id, 0)] = x;
                        index[Ix2(cell_id, 1)] = y;
                        cell_id += 1;
                }
            }
        }



        (index, (nr_cells_x, nr_cells_y))
    }

}