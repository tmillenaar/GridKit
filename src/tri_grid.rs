use numpy::ndarray::*;
use crate::interpolate;
use crate::utils::*;

pub struct TriGrid {
    pub cellsize: f64,
    pub offset: (f64, f64),
    pub rotation: f64,
    pub rotation_matrix: Array2<f64>,
    pub rotation_matrix_inv: Array2<f64>,
}

impl TriGrid {
    pub fn new(cellsize: f64, offset: (f64, f64), rotation: f64) -> Self {
        let rotation_matrix = _rotation_matrix(rotation);
        let rotation_matrix_inv = _rotation_matrix(-rotation);
        // TODO: Find a way to normalize_offset without having to instantiate tmp object
        let self_tmp = TriGrid { cellsize, offset, rotation, rotation_matrix, rotation_matrix_inv };
        let offset = normalize_offset(offset, self_tmp.cell_width(), self_tmp.cell_height());
        let rotation_matrix = _rotation_matrix(rotation);
        let rotation_matrix_inv = _rotation_matrix(-rotation);
        TriGrid { cellsize, offset, rotation, rotation_matrix, rotation_matrix_inv }
    }

    pub fn cell_height(&self) -> f64 {
        self.cellsize * (3_f64).sqrt()
    }

    pub fn cell_width(&self) -> f64 {
        self.cellsize * 2.
    }

    pub fn radius(&self) -> f64 {
        2. / 3. * self.cell_height()
    }

    pub fn dx(&self) -> f64 {
        self.cellsize
    }

    pub fn dy(&self) -> f64 {
        self.cell_height()
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
        let mut corners = Array3::<f64>::zeros((index.shape()[0], 3, 2));

        for cell_id in 0..corners.shape()[0] {
            let (centroid_x, centroid_y) = self.centroid_xy_no_rot(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
            if iseven(index[Ix2(cell_id, 0)]) == iseven(index[Ix2(cell_id, 1)]) {
                corners[Ix3(cell_id, 0, 0)] = centroid_x;
                corners[Ix3(cell_id, 0, 1)] = centroid_y - self.radius();
                corners[Ix3(cell_id, 1, 0)] = centroid_x + self.dx();
                corners[Ix3(cell_id, 1, 1)] =
                    centroid_y + (self.cell_height() - self.radius());
                corners[Ix3(cell_id, 2, 0)] = centroid_x - self.dx();
                corners[Ix3(cell_id, 2, 1)] =
                    centroid_y + (self.cell_height() - self.radius());
            } else {
                corners[Ix3(cell_id, 0, 0)] = centroid_x;
                corners[Ix3(cell_id, 0, 1)] = centroid_y + self.radius();
                corners[Ix3(cell_id, 1, 0)] = centroid_x + self.dx();
                corners[Ix3(cell_id, 1, 1)] =
                    centroid_y - (self.cell_height() - self.radius());
                corners[Ix3(cell_id, 2, 0)] = centroid_x - self.dx();
                corners[Ix3(cell_id, 2, 1)] =
                    centroid_y - (self.cell_height() - self.radius());
            }
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

    pub fn cell_at_point(&self, points: &ArrayView2<f64>) -> Array2<i64> {
        let mut index = Array2::<i64>::zeros((points.shape()[0], 2));
        for cell_id in 0..points.shape()[0] {
            let point = points.slice(s![cell_id, ..]);
            let point = self.rotation_matrix_inv.dot(&point);
            index[Ix2(cell_id, 0)] =
                (1. + (point[Ix1(0)] - self.offset.0) / self.dx()).floor() as i64;
            index[Ix2(cell_id, 1)] =
                (1. + (point[Ix1(1)] - self.offset.1) / self.dy()).floor() as i64;

            let rel_loc_x: f64 = ((point[Ix1(0)] - self.offset.0).abs()) % self.dx();
            let rel_loc_y: f64 = ((point[Ix1(1)] - self.offset.1).abs()) % self.dy();

            let mut downward_cell =
                iseven(index[Ix2(cell_id, 0)]) != iseven(index[Ix2(cell_id, 1)]);
            if index[Ix2(cell_id, 1)] > 0 {
                downward_cell = !downward_cell;
            }
            if downward_cell {
                // Bottom left of square that belongs to triangle to the left
                let is_inside = (rel_loc_x / self.dx()) + 0.5
                    > (self.cell_height() - rel_loc_y) / self.dy();
                if !is_inside {
                    let factor = if index[Ix2(cell_id, 0)] > 0 { -1 } else { 1 };
                    index[Ix2(cell_id, 0)] = index[Ix2(cell_id, 0)] + factor;
                }
                // Bottom right of square that belongs to triangle to the right
                let is_inside =
                    (rel_loc_x / self.dx()) - 0.5 < (rel_loc_y) / self.dy();
                if !is_inside {
                    let factor = if index[Ix2(cell_id, 0)] > 0 { 1 } else { -1 };
                    index[Ix2(cell_id, 0)] = index[Ix2(cell_id, 0)] + factor;
                }
            } else {
                // Top left of square that belongs to triangle to the left
                let is_inside =
                    (rel_loc_x / self.dx()) + 0.5 > rel_loc_y / self.dy();
                if !is_inside {
                    let factor = if index[Ix2(cell_id, 0)] > 0 { -1 } else { 1 };
                    index[Ix2(cell_id, 0)] = index[Ix2(cell_id, 0)] + factor;
                }
                // Top right of square that belongs to triangle to the right
                let is_inside = (rel_loc_x / self.dx()) - 0.5
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
        for y in (miny..=maxy).rev() {
            for x in minx..=maxx {
                let (centroid_x, centroid_y) = self.centroid_xy_no_rot(x, y);
                if (centroid_x > bounds.0) & // x > minx
                   (centroid_x < bounds.2) & // x < maxx
                   (centroid_y > bounds.1) & // y > miny
                   (centroid_y < bounds.3)   // y < maxy
                {
                    index[Ix2(cell_id, 0)] = x;
                    index[Ix2(cell_id, 1)] = y;
                    cell_id += 1;
                }
            }
        }

        (index, (nr_cells_y, nr_cells_x))
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
        for row_id in 0..depth {
            nr_cells_per_colum_upward[Ix1(row_id as usize)] =
                max_nr_cols - 2 * (depth - 1 - row_id);
        }
        for row_id in depth..(nr_rows) {
            nr_cells_per_colum_upward[Ix1(row_id as usize)] =
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
            nr_cells_per_colum_upward[Ix1(row_id as usize)] =
                max_nr_cols - 2 * (depth / 2 - row_id);
        }
        for row_id in (depth / 2)..(nr_rows) {
            nr_cells_per_colum_upward[Ix1(row_id as usize)] =
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

    pub fn cells_near_point(&self, points: &ArrayView2<f64>) -> Array3<i64> {
        let mut nearby_cells = Array3::<i64>::zeros((points.shape()[0], 6, 2));
        // TODO:
        // Condense this into a single loop
        let cell_ids = self.cell_at_point(points);
        let corners = self.cell_corners(&cell_ids.view());


        if self.rotation != 0. {
            let mut points = points.to_owned();
            for cell_id in 0..points.shape()[0] {
                let mut point = points.slice_mut(s![cell_id, ..]);
                let point_rot = self.rotation_matrix_inv.dot(&point);
                point.assign(&point_rot);
            }
        }

        // Define arguments to be used when determining the minimum distance
        let mut min_dist: f64 = 0.;
        let mut nearest_corner_id: usize = 0;
        for cell_id in 0..corners.shape()[0] {
            // - compute id of min distance
            for corner_id in 0..corners.shape()[1] {
                let x = corners[Ix3(cell_id, corner_id, 0)];
                let y = corners[Ix3(cell_id, corner_id, 1)];
                let dx = points[Ix2(cell_id, 0)] - x;
                let dy = points[Ix2(cell_id, 1)] - y;
                let distance = (dx.powi(2) + dy.powi(2)).powf(0.5);
                if corner_id == 0 {
                    nearest_corner_id = corner_id;
                    min_dist = distance;
                } else if distance <= min_dist {
                    nearest_corner_id = corner_id;
                    min_dist = distance;
                }
            }

            // Define the relative ids of the nearby points with respect to the cell that contains the point
            // The nearby cells will depend on which corner of the cell the point is located at, and
            // whether the cell is pointing up or down.
            let rel_nearby_cells: Array2<i64>;
            if self._is_cell_upright(cell_ids[Ix2(cell_id, 0)], cell_ids[Ix2(cell_id, 1)]) {
                match nearest_corner_id {
                    0 => {
                        rel_nearby_cells =
                            array![[-1, 1], [0, 1], [1, 1], [-1, 0], [0, 0], [1, 0],];
                    }
                    1 => {
                        rel_nearby_cells =
                            array![[0, 0], [1, 0], [2, 0], [0, -1], [1, -1], [2, -1],];
                    }
                    2 => {
                        rel_nearby_cells =
                            array![[-2, 0], [-1, 0], [0, 0], [-2, -1], [-1, -1], [0, -1],];
                    }
                    _ => {
                        panic!("Invalid nearest corner id: {}. Expected the corner triangle ID to be any of (0,1,2)", nearest_corner_id);
                    }
                }
            } else {
                // Triangle points upright
                match nearest_corner_id {
                    0 => {
                        rel_nearby_cells =
                            array![[-1, 0], [0, 0], [1, 0], [-1, -1], [0, -1], [1, -1],];
                    }
                    1 => {
                        rel_nearby_cells = array![[0, 1], [1, 1], [2, 1], [0, 0], [1, 0], [2, 0],];
                    }
                    2 => {
                        rel_nearby_cells =
                            array![[-2, 1], [-1, 1], [0, 1], [-2, 0], [-1, 0], [0, 0],];
                    }
                    _ => {
                        panic!("Invalid nearest corner id: {}. Expected the corner triangle ID to be any of (0,1,2)", nearest_corner_id);
                    }
                }
            }
            // Insert ids into return array for current cell_id
            nearby_cells
                .slice_mut(s![cell_id, .., ..])
                .assign(&(rel_nearby_cells + cell_ids.slice(s![cell_id, ..]))); // Try inserting slice?
        }

        nearby_cells
    }

    fn _is_cell_upright(&self, id_x: i64, id_y: i64) -> bool {
        iseven(id_x) != iseven(id_y)
    }

    pub fn is_cell_upright(&self, index: &ArrayView2<i64>) -> Array1<bool> {
        let mut cells = Array1::<bool>::from_elem(index.shape()[0], false);
        for cell_id in 0..cells.shape()[0] {
            cells[Ix1(cell_id)] =
                self._is_cell_upright(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
        }
        cells
    }

    pub fn linear_interpolation (
        &self,
        sample_points: &ArrayView2<f64>,
        nearby_value_locations: &ArrayView3<f64>,
        nearby_values: &ArrayView2<f64>,
    ) -> Array1<f64> {
        let mut values = Array1::<f64>::zeros(sample_points.shape()[0]);
        Zip::from(&mut values)
            .and(sample_points.axis_iter(Axis(0)))
            .and(nearby_value_locations.axis_iter(Axis(0)))
            .and(nearby_values.axis_iter(Axis(0)))
            .for_each(|
                new_val,
                point,
                val_locs,
                near_vals
            | {
                // get 2 nearest point ids
                let point_to_centroid_vecs = &val_locs - &point;
                let mut near_pnt_1: usize = 0;
                let mut near_pnt_2: usize = 0;
                let mut near_dist_1: f64 = f64::MAX;
                let mut near_dist_2: f64 = f64::MAX;
                for (i, vec) in point_to_centroid_vecs.axis_iter(Axis(0)).enumerate() {
                    let dist = crate::interpolate::vec_norm_1d(&vec);
                    if (dist <= near_dist_1) {
                        near_dist_2 = near_dist_1;
                        near_dist_1 = dist;
                        near_pnt_2 = near_pnt_1;
                        near_pnt_1 = i;
                    } else if (dist <= near_dist_2) {
                        near_dist_2 = dist;
                        near_pnt_2 = i;
                    }
                }
                // mean of 6 (val and centroid)
                let mean_centroid = val_locs.mean_axis(Axis(0)).unwrap();
                let mean_val = near_vals.mean().unwrap();
                let near_pnt_locs =  array![
                    [val_locs[Ix2(near_pnt_1, 0)], val_locs[Ix2(near_pnt_1, 1)]],
                    [val_locs[Ix2(near_pnt_2, 0)], val_locs[Ix2(near_pnt_2, 1)]],
                    [mean_centroid[Ix1(0)], mean_centroid[Ix1(1)]],
                ];
                let near_pnt_vals =  array![
                    near_vals[Ix1(near_pnt_1)],
                    near_vals[Ix1(near_pnt_2)],
                    mean_val,
                ];
                let weights = crate::interpolate::linear_interp_weights_single_triangle(
                    &point,
                    &near_pnt_locs.view()
                );
                *new_val = (&weights * &near_pnt_vals).sum();
            });
        values
    }
}
