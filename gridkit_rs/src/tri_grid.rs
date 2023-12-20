use numpy::ndarray::*;

fn iseven(val: i64) -> bool {
    val % 2 == 0
}
pub struct TriGrid {
    pub cellsize: f64,
    pub offset: (f64, f64),
}

impl TriGrid {
    pub fn new(cellsize: f64, offset: (f64, f64)) -> Self {
        TriGrid { cellsize, offset }
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
        let centroid_x = x as f64 * self.dx() - (self.dx() / 2.) + self.offset.0;
        let mut centroid_y = y as f64 * self.dy() - (self.dy() / 2.) + self.offset.1;

        let vertical_offset = self.radius() - 0.5 * self.dy();
        if iseven(x) == iseven(y) {
            centroid_y = centroid_y + vertical_offset;
        } else {
            centroid_y = centroid_y - vertical_offset;
        }

        (centroid_x, centroid_y)
    }

    pub fn cell_corners(&self, index: &ArrayView2<i64>) -> Array3<f64> {
        let centroids = self.centroid(index);
        let mut corners = Array3::<f64>::zeros((index.shape()[0], 3, 2));

        for cell_id in 0..corners.shape()[0] {
            if iseven(index[Ix2(cell_id, 0)]) == iseven(index[Ix2(cell_id, 1)]) {
                corners[Ix3(cell_id, 0, 0)] = centroids[Ix2(cell_id, 0)];
                corners[Ix3(cell_id, 0, 1)] = centroids[Ix2(cell_id, 1)] - self.radius();
                corners[Ix3(cell_id, 1, 0)] = centroids[Ix2(cell_id, 0)] + self.dx();
                corners[Ix3(cell_id, 1, 1)] =
                    centroids[Ix2(cell_id, 1)] + (self.cell_height() - self.radius());
                corners[Ix3(cell_id, 2, 0)] = centroids[Ix2(cell_id, 0)] - self.dx();
                corners[Ix3(cell_id, 2, 1)] =
                    centroids[Ix2(cell_id, 1)] + (self.cell_height() - self.radius());
            } else {
                corners[Ix3(cell_id, 0, 0)] = centroids[Ix2(cell_id, 0)];
                corners[Ix3(cell_id, 0, 1)] = centroids[Ix2(cell_id, 1)] + self.radius();
                corners[Ix3(cell_id, 1, 0)] = centroids[Ix2(cell_id, 0)] + self.dx();
                corners[Ix3(cell_id, 1, 1)] =
                    centroids[Ix2(cell_id, 1)] - (self.cell_height() - self.radius());
                corners[Ix3(cell_id, 2, 0)] = centroids[Ix2(cell_id, 0)] - self.dx();
                corners[Ix3(cell_id, 2, 1)] =
                    centroids[Ix2(cell_id, 1)] - (self.cell_height() - self.radius());
            }
        }
        corners
    }

    pub fn cell_at_point(&self, points: &ArrayView2<f64>) -> Array2<i64> {
        let mut index = Array2::<i64>::zeros((points.shape()[0], 2));
        for cell_id in 0..points.shape()[0] {
            index[Ix2(cell_id, 0)] =
                (1. + (points[Ix2(cell_id, 0)] - self.offset.0) / self.dx()).floor() as i64;
            index[Ix2(cell_id, 1)] =
                (1. + (points[Ix2(cell_id, 1)] - self.offset.1) / self.dy()).floor() as i64;

            let rel_loc_x: f64 = ((points[Ix2(cell_id, 0)] - self.offset.0).abs()) % self.dx();
            let rel_loc_y: f64 = ((points[Ix2(cell_id, 1)] - self.offset.1).abs()) % self.dy();

            let mut downward_cell =
                iseven(index[Ix2(cell_id, 0)]) != iseven(index[Ix2(cell_id, 1)]);
            if index[Ix2(cell_id, 1)] > 0 {
                downward_cell = !downward_cell;
            }
            if downward_cell {
                // Bottom left of square that belongs to triangle to the left
                let is_inside = (rel_loc_x / self.dx()) + self.cell_width() / 2.
                    > (self.cell_height() - rel_loc_y) / self.dy();
                if !is_inside {
                    let factor = if index[Ix2(cell_id, 0)] > 0 { -1 } else { 1 };
                    index[Ix2(cell_id, 0)] = index[Ix2(cell_id, 0)] + factor;
                }
                // Bottom right of square that belongs to triangle to the right
                let is_inside =
                    (rel_loc_x / self.dx()) - self.cell_width() / 2. < (rel_loc_y) / self.dy();
                if !is_inside {
                    let factor = if index[Ix2(cell_id, 0)] > 0 { 1 } else { -1 };
                    index[Ix2(cell_id, 0)] = index[Ix2(cell_id, 0)] + factor;
                }
            } else {
                // Top left of square that belongs to triangle to the left
                let is_inside =
                    (rel_loc_x / self.dx()) + self.cell_width() / 2. > rel_loc_y / self.dy();
                if !is_inside {
                    let factor = if index[Ix2(cell_id, 0)] > 0 { -1 } else { 1 };
                    index[Ix2(cell_id, 0)] = index[Ix2(cell_id, 0)] + factor;
                }
                // Top right of square that belongs to triangle to the right
                let is_inside = (rel_loc_x / self.dx()) - self.cell_width() / 2.
                    < (self.cell_height() - rel_loc_y) / self.dy();
                if !is_inside {
                    let factor = if index[Ix2(cell_id, 0)] > 0 { 1 } else { -1 };
                    index[Ix2(cell_id, 0)] = index[Ix2(cell_id, 0)] + factor;
                }
            }
        }
        index
    }

    pub fn cells_in_bounds(&self, bounds: &(f64, f64, f64, f64)) -> (Array2<i64>, (usize, usize)) {
        // Get ids of the cells at diagonally opposing corners of the bounds
        // TODO: allow calling of cell_at_point with single point (tuple or 1d array)
        let left_bottom: Array2<f64> =
            array![[bounds.0 + self.dx() / 4., bounds.1 + self.dy() / 4.]];
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
                   (centroid[Ix2(0, 1)] < bounds.3)
                {
                    // y < maxy
                    index[Ix2(cell_id, 0)] = x;
                    index[Ix2(cell_id, 1)] = y;
                    cell_id += 1;
                }
            }
        }

        (index, (nr_cells_x, nr_cells_y))
    }

    pub fn all_neighbours(
        &self,
        index: &ArrayView2<i64>,
        depth: i64,
        include_selected: bool,
        add_cell_id: bool,
    ) -> Array3<i64> {
        let add_cell_id = add_cell_id as i64;
        let mut total_nr_neighbours: usize = include_selected as usize;
        let mut nr_neighbours_factor: usize;
        let max_nr_cols: i64;
        let nr_rows: i64;

        nr_neighbours_factor = 4;
        max_nr_cols = 1 + 4 * depth;
        nr_rows = 1 + 2 * depth;

        for i in 0..depth {
            total_nr_neighbours = total_nr_neighbours + nr_neighbours_factor * 3 * (i + 1) as usize;
        }
        let mut relative_neighbours =
            Array3::<i64>::zeros((index.shape()[0], total_nr_neighbours, 2));

        let mut nr_cells_per_colum_upward = Array1::<i64>::zeros((nr_rows as usize,));
        for row_id in 0..(depth) {
            nr_cells_per_colum_upward[Ix1(row_id.try_into().unwrap())] =
                max_nr_cols - 2 * (depth - 1 - row_id);
        }
        for row_id in depth..(nr_rows) {
            nr_cells_per_colum_upward[Ix1(row_id.try_into().unwrap())] =
                max_nr_cols - 2 * (row_id - depth);
        }

        let mut counter: usize = 0;

        let mut nr_cells_per_colum_downward = Array1::<i64>::zeros((nr_rows as usize,));
        for i in 0..nr_rows {
            let i = i as usize;
            nr_cells_per_colum_downward[Ix1(i)] =
                nr_cells_per_colum_upward[Ix1(nr_rows as usize - 1 - i)];
        }

        let mut counter: usize = 0;
        let mut nr_cells_per_colum: &Array1<i64>;
        for cell_id in 0..relative_neighbours.shape()[0] {
            counter = 0;

            let downward_cell = iseven(index[Ix2(cell_id, 0)]) != iseven(index[Ix2(cell_id, 1)]);
            if downward_cell {
                nr_cells_per_colum = &nr_cells_per_colum_downward;
            } else {
                nr_cells_per_colum = &nr_cells_per_colum_upward;
            }

            for rel_row_id in (0..nr_rows).rev() {
                let nr_cells_in_colum = nr_cells_per_colum[Ix1(rel_row_id as usize)];
                for rel_col_id in 0..nr_cells_in_colum {
                    relative_neighbours[Ix3(cell_id, counter, 0)] = rel_col_id
                        - ((nr_cells_in_colum as f64 / 2.).floor() as i64)
                        + (add_cell_id * index[Ix2(cell_id, 0)]);
                    relative_neighbours[Ix3(cell_id, counter, 1)] =
                        depth - rel_row_id + (add_cell_id * index[Ix2(cell_id, 1)]);
                    counter = counter + 1;
                    // Skip selected center cell if include_selected is false
                    counter = counter
                        - (!include_selected
                            && (rel_row_id == depth)
                            && (rel_col_id == (2 * depth))) as usize;
                }
            }
        }

        relative_neighbours
    }

    pub fn direct_neighbours(
        &self,
        index: &ArrayView2<i64>,
        depth: i64,
        include_selected: bool,
        add_cell_id: bool,
    ) -> Array3<i64> {
        let add_cell_id = add_cell_id as i64;
        let mut total_nr_neighbours: usize = include_selected as usize;

        let max_nr_cols = 1 + 2 * depth;
        let nr_rows = 1 + depth;

        for i in 0..depth {
            total_nr_neighbours = total_nr_neighbours + 3 * (i + 1) as usize;
        }
        let mut relative_neighbours =
            Array3::<i64>::zeros((index.shape()[0], total_nr_neighbours, 2));

        let mut nr_cells_per_colum_upward = Array1::<i64>::zeros((nr_rows as usize,));
        for row_id in 0..(depth / 2) {
            nr_cells_per_colum_upward[Ix1(row_id.try_into().unwrap())] =
                max_nr_cols - 2 * (depth / 2 - row_id);
        }
        for row_id in (depth / 2)..(nr_rows) {
            nr_cells_per_colum_upward[Ix1(row_id.try_into().unwrap())] =
                max_nr_cols - 2 * (row_id - depth / 2);
        }
        let mut nr_cells_per_colum_downward = Array1::<i64>::zeros((nr_rows as usize,));
        for i in 0..nr_rows {
            let i = i as usize;
            nr_cells_per_colum_downward[Ix1(i)] =
                nr_cells_per_colum_upward[Ix1(nr_rows as usize - 1 - i)];
        }

        let mut counter: usize = 0;
        let mut y_offset: i64;
        let mut skip_cell: bool;
        let mut nr_cells_per_colum: &Array1<i64>;
        for cell_id in 0..relative_neighbours.shape()[0] {
            counter = 0;

            let downward_cell = iseven(index[Ix2(cell_id, 0)]) == iseven(index[Ix2(cell_id, 1)]);
            if downward_cell {
                nr_cells_per_colum = &nr_cells_per_colum_downward;
            } else {
                nr_cells_per_colum = &nr_cells_per_colum_upward;
            }

            for rel_row_id in (0..nr_rows).rev() {
                let nr_cells_in_colum = nr_cells_per_colum[Ix1(rel_row_id as usize)];
                for rel_col_id in 0..nr_cells_in_colum {
                    let partial_row: i64;
                    if downward_cell {
                        if iseven(depth) {
                            partial_row = nr_rows - 1;
                        } else {
                            partial_row = 0;
                        }
                        skip_cell = rel_row_id == partial_row
                            && if (((index[Ix2(cell_id, 0)]) > 0) == (index[Ix2(cell_id, 1)] > 0)) {
                                iseven(rel_col_id) != iseven(index[Ix2(cell_id, 0)])
                            } else {
                                iseven(rel_col_id) == iseven(index[Ix2(cell_id, 0)])
                            };
                        y_offset = ((depth as f64 / 2.).floor() as i64);
                    } else {
                        if iseven(depth) {
                            partial_row = 0;
                        } else {
                            partial_row = nr_rows - 1;
                        }
                        skip_cell = rel_row_id == partial_row
                            && if (((index[Ix2(cell_id, 0)]) > 0) == (index[Ix2(cell_id, 1)] > 0)) {
                                iseven(rel_col_id) == iseven(index[Ix2(cell_id, 0)])
                            } else {
                                iseven(rel_col_id) != iseven(index[Ix2(cell_id, 0)])
                            };
                        y_offset = (depth as f64 / 2.).ceil() as i64;
                    }
                    if counter < relative_neighbours.shape()[1] {
                        if !skip_cell {
                            relative_neighbours[Ix3(cell_id, counter, 0)] = rel_col_id
                                - ((nr_cells_in_colum as f64 / 2.).floor() as i64)
                                + (add_cell_id * index[Ix2(cell_id, 0)]);
                            relative_neighbours[Ix3(cell_id, counter, 1)] =
                                depth - rel_row_id - y_offset
                                    + (add_cell_id * index[Ix2(cell_id, 1)]);
                            counter = counter + 1;
                        }
                    }

                    // Skip selected center cell if include_selected is false
                    counter = counter
                        - (!include_selected
                            && (nr_cells_in_colum == max_nr_cols)
                            && (rel_col_id == depth)) as usize;
                }
            }
        }

        relative_neighbours
    }
}
