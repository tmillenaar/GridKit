#![allow(unused_parens)]

use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use crate::grid::{CellElement, Grid, GridTraits, Orientation};
use crate::utils::*;
use ndarray::*;

#[derive(Clone, Debug)]
pub struct TriGrid {
    pub cellsize: f64,
    pub offset: [f64; 2],
    pub orientation: Orientation,
    pub _rotation: f64,
    pub _rotation_matrix: Array2<f64>,
    pub _rotation_matrix_inv: Array2<f64>,
}

impl PartialEq for TriGrid {
    // Needs manual implementation, derive PartialEq does not work on floats because of NaN etc.
    fn eq(&self, other: &Self) -> bool {
        is_close(self.cellsize, other.cellsize)
            && is_close(self.offset[0], other.offset[0])
            && is_close(self.offset[1], other.offset[1])
            && is_close(self._rotation, other._rotation)
            && self.orientation == other.orientation
    }
}

impl Eq for TriGrid {}

impl Hash for TriGrid {
    // Needs manual implementation, derive Hash does not work on floats because of NaN etc.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cellsize.to_bits().hash(state);
        self.offset[0].to_bits().hash(state);
        self.offset[1].to_bits().hash(state);
        self.orientation.hash(state);
        self._rotation.to_bits().hash(state);
    }
}

impl GridTraits for TriGrid {
    fn get_grid(&self) -> crate::Grid {
        crate::Grid::TriGrid(self.to_owned())
    }

    fn dx(&self) -> f64 {
        match self.orientation() {
            Orientation::Flat => self.cellsize / 2.,
            Orientation::Pointy => self.cell_width(),
        }
    }
    fn dy(&self) -> f64 {
        match self.orientation() {
            Orientation::Flat => self.cell_height(),
            Orientation::Pointy => self.cellsize / 2.,
        }
    }

    fn set_cellsize(&mut self, cellsize: f64) {
        self.cellsize = cellsize;
    }

    fn offset(&self) -> [f64; 2] {
        self.offset
    }
    fn set_offset(&mut self, offset: [f64; 2]) {
        self.offset = normalize_offset(offset, self.cell_width(), self.cell_height());
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
        match self.orientation() {
            Orientation::Flat => 2. / 3. * self.cell_height(),
            Orientation::Pointy => 2. / 3. * self.cell_width(),
        }
    }

    fn cell_height(&self) -> f64 {
        match self.orientation() {
            Orientation::Flat => self.dx() * (3_f64).sqrt(),
            Orientation::Pointy => self.cellsize,
        }
    }

    fn cell_width(&self) -> f64 {
        match self.orientation() {
            Orientation::Flat => self.cellsize,
            Orientation::Pointy => self.dy() * (3_f64).sqrt(),
        }
    }

    fn centroid_xy_no_rot(&self, x: i64, y: i64) -> [f64; 2] {
        let mut centroid_x: f64;
        let mut centroid_y: f64;

        match self.orientation() {
            Orientation::Flat => {
                centroid_x = x as f64 * self.dx() + self.dx() + self.offset[0];
                centroid_y = y as f64 * self.dy() + (self.dy() / 2.) + self.offset[1];
                let vertical_offset = self.radius() - 0.5 * self.dy();
                if iseven(x) == iseven(y) {
                    centroid_y = centroid_y - vertical_offset;
                } else {
                    centroid_y = centroid_y + vertical_offset;
                }
            }
            Orientation::Pointy => {
                centroid_x = x as f64 * self.dx() + (self.dx() / 2.) + self.offset[0];
                centroid_y = y as f64 * self.dy() + self.dy() + self.offset[1];
                let horizontal_offset = self.radius() - 0.5 * self.dx();
                if iseven(x) == iseven(y) {
                    centroid_x = centroid_x - horizontal_offset;
                } else {
                    centroid_x = centroid_x + horizontal_offset;
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

    fn cell_at_points(&self, points: &ArrayView2<f64>) -> Array2<i64> {
        let mut index = Array2::<i64>::zeros((points.shape()[0], 2));

        let dx = self.stepsize_consistent_axis();
        let dy = self.stepsize_inconsistent_axis();
        let id_x_axis = self.consistent_axis();
        let id_y_axis = self.inconsistent_axis();
        let offset_x = self.offset[id_x_axis];
        let offset_y = self.offset[id_y_axis];
        let cell_width = match self.orientation() {
            Orientation::Flat => self.cell_width(),
            Orientation::Pointy => self.cell_height(),
        };
        let cell_height = match self.orientation() {
            Orientation::Flat => self.cell_height(),
            Orientation::Pointy => self.cell_width(),
        };

        for cell_id in 0..points.shape()[0] {
            let point = points.slice(s![cell_id, ..]);
            let point = self._rotation_matrix_inv.dot(&point);
            index[Ix2(cell_id, id_x_axis)] =
                (1. + (point[Ix1(id_x_axis)] - offset_x) / dx).floor() as i64;
            index[Ix2(cell_id, id_y_axis)] =
                ((point[Ix1(id_y_axis)] - offset_y) / cell_height).floor() as i64;
            index[Ix2(cell_id, id_x_axis)] = 2
                * ((point[Ix1(id_x_axis)] - offset_x
                    + dx * !iseven(index[Ix2(cell_id, id_y_axis)]) as i64 as f64)
                    / cell_width)
                    .floor() as i64
                - 1 * (!iseven(index[Ix2(cell_id, id_y_axis)]) as i64);

            // TODO: Fix this 3rd dimension of cell_id=0. I.e. fix cell_corners needing to take multiple ids at once
            let cell_origin = self.cell_corners(&index.slice(s![cell_id..cell_id + 1, ..]));
            let cell_origin = cell_origin.slice(s![0, 2, ..]);

            let cell_origin: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> =
                self._rotation_matrix_inv.dot(&cell_origin);
            let cell_origin_x = cell_origin[Ix1(id_x_axis)];
            let cell_origin_y = cell_origin[Ix1(id_y_axis)];
            let rel_loc_x: f64 = point[Ix1(id_x_axis)] - cell_origin_x;
            let rel_loc_y: f64 = point[Ix1(id_y_axis)] - cell_origin_y;

            let y_threshold_left: f64;
            let y_threshold_right: f64;
            let slope = dy / dx;
            // Descrtibes the equation that forms the left border of the triangle
            let left_eq = |x: f64| -> f64 { slope * x };
            // Descrtibes the equation that forms the right border of the triangle
            let right_eq = |x: f64| -> f64 { -slope * x + 2. * cell_height };
            y_threshold_left = left_eq(rel_loc_x);
            y_threshold_right = right_eq(rel_loc_x);
            let id_shift = if rel_loc_y > y_threshold_left {
                -1
            } else if rel_loc_y > y_threshold_right {
                1
            } else {
                0
            };

            index[Ix2(cell_id, id_x_axis)] = index[Ix2(cell_id, id_x_axis)] + id_shift;
        }
        index
    }

    fn cell_corners(&self, index: &ArrayView2<i64>) -> Array3<f64> {
        let mut corners = Array3::<f64>::zeros((index.shape()[0], 3, 2));

        for cell_id in 0..corners.shape()[0] {
            let [centroid_x, centroid_y] =
                self.centroid_xy_no_rot(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
            match self.orientation() {
                Orientation::Flat => {
                    if iseven(index[Ix2(cell_id, 0)]) == iseven(index[Ix2(cell_id, 1)]) {
                        // Cell with flat base at bottom and pointing up
                        corners[Ix3(cell_id, 0, 0)] = centroid_x; // top-x
                        corners[Ix3(cell_id, 0, 1)] = centroid_y + self.radius(); // top-y
                        corners[Ix3(cell_id, 1, 0)] = centroid_x + self.dx(); // bottom-right-x
                        corners[Ix3(cell_id, 1, 1)] =
                            centroid_y - (self.cell_height() - self.radius()); // bottom-right-y
                        corners[Ix3(cell_id, 2, 0)] = centroid_x - self.dx(); // bottom-left-x
                        corners[Ix3(cell_id, 2, 1)] =
                            centroid_y - (self.cell_height() - self.radius()); //bottom-left-y
                    } else {
                        // Cell with flat base at top and pointing down
                        corners[Ix3(cell_id, 0, 0)] = centroid_x; // bottom-x
                        corners[Ix3(cell_id, 0, 1)] = centroid_y - self.radius(); // bottom-y
                        corners[Ix3(cell_id, 1, 0)] = centroid_x + self.dx(); // top-right-x
                        corners[Ix3(cell_id, 1, 1)] =
                            centroid_y + (self.cell_height() - self.radius()); // top-right-y
                        corners[Ix3(cell_id, 2, 0)] = centroid_x - self.dx(); // top-left-x
                        corners[Ix3(cell_id, 2, 1)] =
                            centroid_y + (self.cell_height() - self.radius()); // top-left-y
                    }
                }
                Orientation::Pointy => {
                    if iseven(index[Ix2(cell_id, 0)]) == iseven(index[Ix2(cell_id, 1)]) {
                        // Cell with flat base on left side, pointing right
                        corners[Ix3(cell_id, 0, 0)] = centroid_x + self.radius(); // right-x
                        corners[Ix3(cell_id, 0, 1)] = centroid_y; // right-y
                        corners[Ix3(cell_id, 1, 0)] =
                            centroid_x - (self.cell_width() - self.radius()); // top-left-x
                        corners[Ix3(cell_id, 1, 1)] = centroid_y + self.dy(); // top-left-y
                        corners[Ix3(cell_id, 2, 0)] =
                            centroid_x - (self.cell_width() - self.radius()); // bottom-left-x
                        corners[Ix3(cell_id, 2, 1)] = centroid_y - self.dy(); // bottom-left-y
                    } else {
                        // Cell with flat base on right side, pointing left
                        corners[Ix3(cell_id, 0, 0)] = centroid_x - self.radius(); // left-x
                        corners[Ix3(cell_id, 0, 1)] = centroid_y; // left-y
                        corners[Ix3(cell_id, 1, 0)] =
                            centroid_x + (self.cell_width() - self.radius()); // top-right-x
                        corners[Ix3(cell_id, 1, 1)] = centroid_y + self.dy(); // top-right-y
                        corners[Ix3(cell_id, 2, 0)] =
                            centroid_x + (self.cell_width() - self.radius()); // bottom-right-x
                        corners[Ix3(cell_id, 2, 1)] = centroid_y - self.dy(); // bottom-right-y
                    }
                }
            };
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
        let mut nearby_cells = Array3::<i64>::zeros((points.shape()[0], 6, 2));
        // TODO:
        // Condense this into a single loop
        let cell_ids = self.cell_at_points(points);
        let corners = self.cell_corners(&cell_ids.view());

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
            match self.orientation() {
                Orientation::Flat => {
                    if !self._is_cell_upright(cell_ids[Ix2(cell_id, 0)], cell_ids[Ix2(cell_id, 1)])
                    {
                        // Triangle points upright
                        match nearest_corner_id {
                            0 => {
                                rel_nearby_cells =
                                    array![[-1, 0], [0, 0], [1, 0], [-1, -1], [0, -1], [1, -1],];
                            }
                            1 => {
                                rel_nearby_cells =
                                    array![[0, 1], [1, 1], [2, 1], [0, 0], [1, 0], [2, 0],];
                            }
                            2 => {
                                rel_nearby_cells =
                                    array![[-2, 1], [-1, 1], [0, 1], [-2, 0], [-1, 0], [0, 0],];
                            }
                            _ => {
                                panic!("Invalid nearest corner id: {}. Expected the corner triangle ID to be any of (0,1,2)", nearest_corner_id);
                            }
                        }
                    } else {
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
                    }
                }
                Orientation::Pointy => {
                    if !self._is_cell_upright(cell_ids[Ix2(cell_id, 0)], cell_ids[Ix2(cell_id, 1)])
                    {
                        // Triangle points left
                        match nearest_corner_id {
                            0 => {
                                rel_nearby_cells =
                                    array![[0, -1], [0, 0], [0, 1], [-1, -1], [-1, 0], [-1, 1],];
                            }
                            1 => {
                                rel_nearby_cells =
                                    array![[1, 0], [1, 1], [1, 2], [0, 0], [0, 1], [0, 2],];
                            }
                            2 => {
                                rel_nearby_cells =
                                    array![[1, -2], [1, -1], [1, 0], [0, -2], [0, -1], [0, 0],];
                            }
                            _ => {
                                panic!("Invalid nearest corner id: {}. Expected the corner triangle ID to be any of (0,1,2)", nearest_corner_id);
                            }
                        }
                    } else {
                        match nearest_corner_id {
                            // Note: these indices are just like those of the Pointy version, but swapped xy.
                            0 => {
                                rel_nearby_cells =
                                    array![[1, -1], [1, 0], [1, 1], [0, -1], [0, 0], [0, 1],];
                            }
                            1 => {
                                rel_nearby_cells =
                                    array![[0, 1], [0, 2], [-1, 2], [-1, 1], [-1, 0], [0, 0]];
                                // array![[1, 0], [2, 0], [2, -1], [1, -1], [0, -1], [0, 0],];
                            }
                            2 => {
                                rel_nearby_cells =
                                    array![[0, -2], [0, -1], [0, 0], [-1, -2], [-1, -1], [-1, 0],];
                            }
                            _ => {
                                panic!("Invalid nearest corner id: {}. Expected the corner triangle ID to be any of (0,1,2)", nearest_corner_id);
                            }
                        }
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

    fn all_neighbours(
        &self,
        index: &ArrayView2<i64>,
        depth: u64,
        include_selected: bool,
        add_cell_id: bool,
    ) -> Array3<i64> {
        let depth = depth as i64;

        let add_cell_id = add_cell_id as i64;
        let mut total_nr_neighbours = include_selected as usize;
        let nr_neighbours_factor: usize;
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
            nr_cells_per_colum_upward[Ix1(row_id as usize)] = max_nr_cols - 2 * (row_id - depth);
        }

        let mut nr_cells_per_colum_downward = Array1::<i64>::zeros((nr_rows as usize,));
        for i in 0..nr_rows {
            let i = i as usize;
            nr_cells_per_colum_downward[Ix1(i)] =
                nr_cells_per_colum_upward[Ix1(nr_rows as usize - 1 - i)];
        }

        let mut counter: usize;
        let mut nr_cells_per_colum: &Array1<i64>;
        for cell_id in 0..relative_neighbours.shape()[0] {
            counter = 0;

            let downward_cell =
                !self._is_cell_upright(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
            if downward_cell {
                nr_cells_per_colum = &nr_cells_per_colum_downward;
            } else {
                nr_cells_per_colum = &nr_cells_per_colum_upward;
            }

            let id_x_axis = self.consistent_axis();
            let id_y_axis = self.inconsistent_axis();

            for rel_row_id in (0..nr_rows).rev() {
                let nr_cells_in_colum = nr_cells_per_colum[Ix1(rel_row_id as usize)];
                for rel_col_id in 0..nr_cells_in_colum {
                    relative_neighbours[Ix3(cell_id, counter, id_x_axis)] = rel_col_id
                        - ((nr_cells_in_colum as f64 / 2.).floor() as i64)
                        + (add_cell_id * index[Ix2(cell_id, id_x_axis)]);
                    relative_neighbours[Ix3(cell_id, counter, id_y_axis)] =
                        depth - rel_row_id + (add_cell_id * index[Ix2(cell_id, id_y_axis)]);
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

    fn direct_neighbours(
        &self,
        index: &ArrayView2<i64>,
        depth: u64,
        include_selected: bool,
        add_cell_id: bool,
    ) -> Array3<i64> {
        let depth = depth as i64;
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

        let id_x_axis = self.consistent_axis();
        let id_y_axis = self.inconsistent_axis();

        let mut counter: usize;
        let mut y_offset: i64;
        let mut skip_cell: bool;
        for cell_id in 0..relative_neighbours.shape()[0] {
            counter = 0;
            let upright_cell =
                self._is_cell_upright(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
            let flip_vertically: i64;
            if upright_cell {
                flip_vertically = 1;
            } else {
                flip_vertically = -1;
            }
            for rel_row_id in (0..nr_rows) {
                let partial_row: i64;
                let nr_cells_in_colum: i64;
                if iseven(depth) {
                    partial_row = 0;
                    nr_cells_in_colum = nr_cells_per_colum_upward[Ix1(rel_row_id as usize)];
                } else {
                    partial_row = nr_rows - 1;
                    nr_cells_in_colum = nr_cells_per_colum_upward[Ix1(rel_row_id as usize)];
                }

                for rel_col_id in 0..nr_cells_in_colum {
                    if iseven(depth) {
                        skip_cell = rel_row_id == partial_row && !iseven(rel_col_id);
                    } else {
                        skip_cell = rel_row_id == partial_row && !iseven(rel_col_id);
                    }
                    y_offset = ((depth as f64 / 2.).floor() as i64);
                    if counter < relative_neighbours.shape()[1] {
                        if !skip_cell {
                            relative_neighbours[Ix3(cell_id, counter, id_x_axis)] = flip_vertically
                                * (rel_col_id - (nr_cells_in_colum as f64 / 2.).floor() as i64)
                                + (add_cell_id * index[Ix2(cell_id, id_x_axis)]);
                            relative_neighbours[Ix3(cell_id, counter, id_y_axis)] = flip_vertically
                                * (depth - rel_row_id - y_offset - !iseven(depth) as i64)
                                + (add_cell_id * index[Ix2(cell_id, id_y_axis)]);
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

    fn is_aligned_with(&self, other: &Grid) -> bool {
        if let Grid::TriGrid(other) = other {
            if !is_close(self.cellsize, other.cellsize) {
                return false;
            }
            if !(is_close(self.offset()[0], other.offset()[0])
                && is_close(self.offset()[1], other.offset()[1]))
            {
                return false;
            }
            if self.orientation != other.orientation {
                return false;
            }
            if self.rotation() != other.rotation() {
                return false;
            }
            true
        } else {
            false
        }
    }

    fn subdivide(&self, factor: u64) -> Grid {
        if factor == 0 {
            return Grid::TriGrid(self.clone());
        }

        // `factor` of at least one cannot drive the cellsize to zero, so no
        // validation is needed here.
        let mut sub_grid = self.clone();
        sub_grid.set_cellsize(self.cellsize / factor as f64);

        // Anchor the sub grid to the top corner of the parent's cell (0, 0), as
        // Python's `TriGrid.subdivide` does. The corner is taken from the *parent*,
        // so it is unaffected by the cellsize change above.
        let corners = self.cell_corners(&array![[0i64, 0i64]].view());
        let anchor_loc = [corners[[0, 0, 0]], corners[[0, 0, 1]]];
        sub_grid.anchor_inplace(&anchor_loc, CellElement::Corner);

        Grid::TriGrid(sub_grid)
    }
}

impl TriGrid {
    pub fn new(cellsize: f64, orientation: Orientation) -> Self {
        let _rotation_matrix = rotation_matrix_from_angle(0.);
        let _rotation_matrix_inv = rotation_matrix_from_angle(-0.);
        TriGrid {
            cellsize,
            offset: [0., 0.],
            orientation: orientation,
            _rotation: 0.,
            _rotation_matrix,
            _rotation_matrix_inv,
        }
    }

    pub fn orientation(&self) -> &Orientation {
        &self.orientation
    }

    pub fn set_orientation(&mut self, orientation: Orientation) {
        self.orientation = orientation;
    }

    pub fn consistent_axis(&self) -> usize {
        match self.orientation {
            Orientation::Pointy => 1,
            Orientation::Flat => 0,
        }
    }

    pub fn inconsistent_axis(&self) -> usize {
        match self.orientation {
            Orientation::Pointy => 0,
            Orientation::Flat => 1,
        }
    }

    pub fn stepsize_consistent_axis(&self) -> f64 {
        match self.orientation {
            Orientation::Pointy => self.dy(),
            Orientation::Flat => self.dx(),
        }
    }

    pub fn stepsize_inconsistent_axis(&self) -> f64 {
        match self.orientation {
            Orientation::Pointy => self.dx(),
            Orientation::Flat => self.dy(),
        }
    }

    // TODO: remove, I don't think we want this functionality at all, but we are using it from python at the moment
    pub fn cells_in_bounds(&self, bounds: &(f64, f64, f64, f64)) -> (Array2<i64>, (usize, usize)) {
        // Get ids of the cells at diagonally opposing corners of the bounds
        // TODO: allow calling of cell_at_points with single point (tuple or 1d array)
        let left_bottom: Array2<f64> =
            array![[bounds.0 + self.dx() / 4., bounds.1 + self.dy() / 4.]];
        let right_top: Array2<f64> =
            array![[bounds.2 - self.cell_width() / 4., bounds.3 - self.dy() / 4.]];

        // translate the coordinates of the corner cells into indices
        let left_bottom_id = self.cell_at_points(&left_bottom.view());
        let right_top_id = self.cell_at_points(&right_top.view());

        // use the cells at the corners to determine
        // the ids in x an y direction as separate 1d arrays
        let minx = left_bottom_id[Ix2(0, 0)] - 2;
        let maxx = right_top_id[Ix2(0, 0)] + 2;

        let miny = left_bottom_id[Ix2(0, 1)] - 2;
        let maxy = right_top_id[Ix2(0, 1)] + 2;

        // fill raveled meshgrid if the centroid is in the bounds, left bound is inclusive
        let nr_cells_x: usize = ((bounds.2 - bounds.0) / self.dx()).round() as usize;
        let nr_cells_y: usize = ((bounds.3 - bounds.1) / self.dy()).round() as usize;
        let mut index = Array2::<i64>::zeros((nr_cells_x * nr_cells_y, 2));
        let mut cell_id: usize = 0;
        for y in (miny..=maxy).rev() {
            for x in minx..=maxx {
                let [centroid_x, centroid_y] = self.centroid_xy_no_rot(x, y);
                if (centroid_x >= bounds.0) & // x > minx
                   (centroid_x < bounds.2) & // x < maxx
                   (centroid_y > bounds.1) & // y > miny
                   (centroid_y < bounds.3)
                // y < maxy
                {
                    index[Ix2(cell_id, 0)] = x;
                    index[Ix2(cell_id, 1)] = y;
                    cell_id += 1;
                }
            }
        }

        (index, (nr_cells_y, nr_cells_x))
    }

    fn _is_cell_upright(&self, id_x: i64, id_y: i64) -> bool {
        // FIXME: Name makes no sense if grid is flat maybe is_cell_flipped?

        // if (id_x >= 0) == (id_y >= 0) {
        iseven(id_x) == iseven(id_y)
        // } else {
        //     iseven(id_x) == iseven(id_y)
        // }
    }

    pub fn is_cell_upright(&self, index: &ArrayView2<i64>) -> Array1<bool> {
        let mut cells = Array1::<bool>::from_elem(index.shape()[0], false);
        for cell_id in 0..cells.shape()[0] {
            cells[Ix1(cell_id)] =
                self._is_cell_upright(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
        }
        cells
    }

    pub fn linear_interpolation(
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
            .for_each(|new_val, point, val_locs, near_vals| {
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
                let near_pnt_locs = array![
                    [val_locs[Ix2(near_pnt_1, 0)], val_locs[Ix2(near_pnt_1, 1)]],
                    [val_locs[Ix2(near_pnt_2, 0)], val_locs[Ix2(near_pnt_2, 1)]],
                    [mean_centroid[Ix1(0)], mean_centroid[Ix1(1)]],
                ];
                let near_pnt_vals = array![
                    near_vals[Ix1(near_pnt_1)],
                    near_vals[Ix1(near_pnt_2)],
                    mean_val,
                ];
                let weights = crate::interpolate::linear_interp_weights_single_triangle(
                    &point,
                    &near_pnt_locs.view(),
                );
                *new_val = (&weights * &near_pnt_vals).sum();
            });
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "expected {b} to be within {tol} of {a} (difference {})",
            (a - b).abs()
        );
    }

    fn assert_ids_3d(result: &Array3<i64>, expected: &[i64]) {
        assert_eq!(result.shape()[2], 2);
        assert_eq!(result.len(), expected.len());
        let mut flat = Vec::with_capacity(result.len());
        for cell in 0..result.shape()[0] {
            for neighbour in 0..result.shape()[1] {
                flat.push(result[[cell, neighbour, 0]]);
                flat.push(result[[cell, neighbour, 1]]);
            }
        }
        assert_eq!(flat, expected);
    }

    // ---------------------------------------------------------------------
    // all_neighbours / direct_neighbours
    //
    // These were already implemented; the tests below pin the behaviour that
    // `test_neighbours` in `test_tri_grid.py` asserts.
    // ---------------------------------------------------------------------

    #[test]
    fn direct_neighbours_depth_one() {
        // A cell has 3 side-sharing neighbours when corners do not count.
        let grid = TriGrid::new(0.7, Orientation::Flat);
        let neighbours = grid.direct_neighbours(&array![[0i64, 0i64]].view(), 1, false, false);
        assert_eq!(neighbours.shape(), &[1, 3, 2]);
        assert_ids_3d(&neighbours, &[-1, 0, 1, 0, 0, -1]);
    }

    #[test]
    fn all_neighbours_depth_one() {
        // With corners connected, the window holds 12 cells: the 6 surrounding
        // each of the 3 neighbours, plus the neighbours themselves.
        let grid = TriGrid::new(0.7, Orientation::Flat);
        let neighbours = grid.all_neighbours(&array![[0i64, 0i64]].view(), 1, false, false);
        assert_eq!(neighbours.shape(), &[1, 12, 2]);
        assert_ids_3d(
            &neighbours,
            &[
                -1, -1, 0, -1, 1, -1, //
                -2, 0, -1, 0, 1, 0, 2, 0, //
                -2, 1, -1, 1, 0, 1, 1, 1, 2, 1,
            ],
        );
    }

    #[test]
    fn neighbour_count_matches_the_documented_formula() {
        // `test_neighbours` in Python expects
        // `include_selected + sum(factor * 3 * (i + 1) for i in range(depth))`,
        // where `factor` is 4 with corners and 1 without.
        let grid = TriGrid::new(0.7, Orientation::Flat);
        let index = array![[0i64, 0i64]];

        for depth in 1..=6u64 {
            for connect_corners in [false, true] {
                let factor = if connect_corners { 4i64 } else { 1i64 };
                let expected = ((0..depth)
                    .map(|i| factor as i64 * 3 * (i as i64 + 1))
                    .sum::<i64>()
                    + 1) as usize;

                let with = if connect_corners {
                    grid.all_neighbours(&index.view(), depth, true, false)
                } else {
                    grid.direct_neighbours(&index.view(), depth, true, false)
                };
                let without = if connect_corners {
                    grid.all_neighbours(&index.view(), depth, false, false)
                } else {
                    grid.direct_neighbours(&index.view(), depth, false, false)
                };

                assert_eq!(with.shape()[1], expected);
                assert_eq!(without.shape()[1], expected - 1);
            }
        }
    }

    #[test]
    fn neighbours_are_unique_and_respect_the_expected_radius() {
        // The two remaining assertions of the Python `test_neighbours`: no
        // duplicate ids, and every cell within `1 + 2 * depth` (with corners) or
        // `depth` (without) steps along the axes.
        let grid = TriGrid::new(0.7, Orientation::Flat);

        // The Python test uses upright and downward cells alike, since the
        // window is mirrored for downward cells.
        for index in [
            array![[0i64, 0i64]],
            array![[-6i64, 3i64], [4i64, -1i64], [5i64, 4i64], [-5i64, -4i64]],
            array![[-5i64, 3i64], [3i64, -3i64], [4i64, 4i64], [-6i64, -6i64]],
        ] {
            for depth in 1..=6u64 {
                for connect_corners in [false, true] {
                    let max_radius = if connect_corners {
                        1 + 2 * depth
                    } else {
                        depth
                    };

                    let neighbours = if connect_corners {
                        grid.all_neighbours(&index.view(), depth, false, false)
                    } else {
                        grid.direct_neighbours(&index.view(), depth, false, false)
                    };

                    for cell_id in 0..neighbours.shape()[0] {
                        let mut ids: Vec<[i64; 2]> = (0..neighbours.shape()[1])
                            .map(|i| [neighbours[[cell_id, i, 0]], neighbours[[cell_id, i, 1]]])
                            .collect();
                        let total = ids.len();
                        ids.sort_unstable();
                        ids.dedup();
                        assert_eq!(ids.len(), total, "duplicate cell in the neighbours");

                        for id in &ids {
                            assert!(
                                (id[0] + id[1]).abs() <= max_radius as i64,
                                "cell {id:?} is further than {max_radius} steps away"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn include_selected_is_the_only_difference_including_the_centre() {
        let grid = TriGrid::new(0.7, Orientation::Flat);
        let index = array![[-6i64, 3i64], [4i64, -1i64]];

        for depth in 1..=6u64 {
            for connect_corners in [false, true] {
                let with = if connect_corners {
                    grid.all_neighbours(&index.view(), depth, true, false)
                } else {
                    grid.direct_neighbours(&index.view(), depth, true, false)
                };
                let without = if connect_corners {
                    grid.all_neighbours(&index.view(), depth, false, false)
                } else {
                    grid.direct_neighbours(&index.view(), depth, false, false)
                };

                assert_eq!(with.shape()[1], without.shape()[1] + 1);

                for cell_id in 0..with.shape()[0] {
                    // Exactly one [0, 0] entry with the selected cell included.
                    let zeroes = (0..with.shape()[1])
                        .filter(|i| with[[cell_id, *i, 0]] == 0 && with[[cell_id, *i, 1]] == 0)
                        .count();
                    assert_eq!(zeroes, 1);
                }
            }
        }
    }

    #[test]
    fn neighbours_single_and_multiple_indices_agree() {
        let grid = TriGrid::new(0.7, Orientation::Flat);
        let index = array![[-6i64, 3i64], [4i64, -1i64], [5i64, 4i64]];

        for depth in 1..=6u64 {
            for include_selected in [false, true] {
                let many = grid.all_neighbours(&index.view(), depth, include_selected, true);

                for (row, expected) in [(-6i64, 3i64), (4, -1), (5, 4)].iter().enumerate() {
                    let one = array![[expected.0, expected.1]];
                    let single = grid.all_neighbours(&one.view(), depth, include_selected, true);

                    assert_eq!(single.shape(), &[1, many.shape()[1], 2]);
                    for i in 0..single.shape()[1] {
                        assert_eq!(
                            [[single[[0, i, 0]], single[[0, i, 1]]]],
                            [[many[[row, i, 0]], many[[row, i, 1]]]]
                        );
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // subdivide
    // ---------------------------------------------------------------------

    /// Unwrap the sub grid that `TriGrid::subdivide` returns.
    fn as_tri_grid(sub_grid: Grid) -> TriGrid {
        match sub_grid {
            Grid::TriGrid(grid) => grid,
            _ => panic!("subdivide on TriGrid should return TriGrid"),
        }
    }

    #[test]
    fn subdivide_scales_the_cells() {
        let mut grid = TriGrid::new(1., Orientation::Flat);
        grid.set_rotation(-23.);

        for factor in [1u64, 2, 3, 9] {
            let sub_grid = as_tri_grid(grid.subdivide(factor));
            assert_close(sub_grid.cellsize, 1. / factor as f64, TOL);
        }
    }

    #[test]
    fn subdivide_keeps_the_orientation_and_rotation() {
        // Python's `TriGrid.subdivide` passes `rotation=self.rotation` and leaves
        // the orientation alone.
        let mut grid = TriGrid::new(1.4, Orientation::Pointy);
        grid.set_rotation(-23.);

        let sub_grid = as_tri_grid(grid.subdivide(3));
        assert_eq!(sub_grid.orientation, Orientation::Pointy);
        assert_close(sub_grid.rotation(), -23., TOL);
    }

    #[test]
    fn subdivide_keeps_parent_corners_on_a_corner() {
        for rotation in [-23., 0., 456.] {
            for offset in [[-2., 3.], [0., 0.], [0.1, -0.2]] {
                let mut grid = TriGrid::new(1., Orientation::Flat);
                grid.set_rotation(rotation);
                grid.set_offset(offset);

                for factor in [2u64, 9] {
                    let sub_grid = as_tri_grid(grid.subdivide(factor));

                    // Take a corner of a parent cell and check that it coincides
                    // with one of the corners of the sub cell containing it.
                    let corners = grid.cell_corners(&array![[-4i64, 23i64]].view());
                    let corner = [corners[[0, 2, 0]], corners[[0, 2, 1]]];

                    let id = sub_grid.cell_at_point(&corner);
                    let sub_corners = sub_grid.cell_corners(&array![[id[0], id[1]]].view());
                    let on_corner = (0..3).any(|i| {
                        let dx = sub_corners[[0, i, 0]] - corner[0];
                        let dy = sub_corners[[0, i, 1]] - corner[1];
                        (dx * dx + dy * dy).sqrt() <= 1e-9
                    });
                    assert!(
                        on_corner,
                        "corner {corner:?} is not a corner of the sub cell containing it \
                         (rotation {rotation}, offset {offset:?}, factor {factor})"
                    );
                }
            }
        }
    }

    #[test]
    fn subdivide_yields_factor_squared_subcells_per_parent_cell() {
        for factor in [2u64, 9] {
            for rotation in [-23., 0., 456.] {
                for offset in [[-2., 3.], [0., 0.], [0.1, -0.2]] {
                    let mut grid = TriGrid::new(1., Orientation::Flat);
                    grid.set_rotation(rotation);
                    grid.set_offset(offset);

                    let sub_grid = as_tri_grid(grid.subdivide(factor));

                    // The sub cells that could possibly be inside the parent cell
                    // (3, -2) are the ones around the sub cell holding the parent
                    // centroid. Python uses `depth=factor + 1`.
                    let target = grid.centroid(&array![[3i64, -2i64]].view());
                    let start = sub_grid.cell_at_point(&[target[[0, 0]], target[[0, 1]]]);
                    let candidates = sub_grid.all_neighbours(
                        &array![[start[0], start[1]]].view(),
                        factor + 1,
                        true,
                        true,
                    );

                    let sub_centroids = sub_grid.centroid(&candidates.slice(s![0, .., ..]));
                    let in_cell = grid.cell_at_points(&sub_centroids.view());
                    let nr_in_cell = (0..in_cell.shape()[0])
                        .filter(|i| in_cell[[*i, 0]] == 3 && in_cell[[*i, 1]] == -2)
                        .count();

                    assert_eq!(
                        nr_in_cell,
                        (factor * factor) as usize,
                        "rotation {rotation}, offset {offset:?}, factor {factor}"
                    );
                }
            }
        }
    }

    #[test]
    fn subdivide_by_one_is_the_same_grid() {
        let mut grid = TriGrid::new(1.23, Orientation::Flat);
        grid.set_rotation(15.5);
        grid.set_offset([0.1, 0.2]);

        let sub_grid = as_tri_grid(grid.subdivide(1));

        // Compared with a tolerance rather than bit for bit: re-anchoring on the
        // parent corner recomputes the offset, which can land one ULP away from
        // the original. Python's `subdivide(1)` returns the same grid for the same
        // reason, through the same re-anchoring.
        assert_close(sub_grid.cellsize, grid.cellsize, TOL);
        assert_close(sub_grid.rotation(), grid.rotation(), TOL);
        assert_eq!(sub_grid.orientation, grid.orientation);
        assert_close(sub_grid.offset()[0], grid.offset()[0], TOL);
        assert_close(sub_grid.offset()[1], grid.offset()[1], TOL);
    }

    #[test]
    fn subdivide_by_zero_is_the_same_grid() {
        // `factor` is unsigned, so a `factor` of 0 is the only invalid input left.
        // Dividing the cellsize by zero is not meaningful, so this yields an
        // unchanged copy instead. Note this diverges from Python, which raises a
        // `ValueError` for any `factor` below 1.
        let mut grid = TriGrid::new(1.23, Orientation::Flat);
        grid.set_rotation(15.5);
        grid.set_offset([0.1, 0.2]);

        assert_eq!(grid.subdivide(0), Grid::TriGrid(grid));
    }
}

#[cfg(test)]
mod is_aligned_with_tests {
    use super::*;
    use crate::hex_grid::HexGrid;
    use crate::rect_grid::RectGrid;

    // Mirrors `test_is_aligned_with` in `test_tri_grid.py`, minus the CRS cases
    // (the Rust grid has no CRS) and the TypeError case (the Rust signature
    // takes a `&Grid`, so a non-grid cannot be passed). The Python test's
    // orientation case is commented out as "other shapes not yet implemented",
    // but `TriGrid` does expose an orientation and the Rust implementation
    // compares it, so it is covered here.
    fn flat() -> TriGrid {
        TriGrid::new(1.2, Orientation::Flat)
    }

    #[test]
    fn an_identical_grid_is_aligned() {
        assert!(flat().is_aligned_with(&Grid::TriGrid(flat())));
    }

    #[test]
    fn a_differing_cellsize_is_reported() {
        assert!(!flat().is_aligned_with(&Grid::TriGrid(TriGrid::new(1.3, Orientation::Flat))));
    }

    #[test]
    fn a_differing_offset_is_reported() {
        for offset in [[0., 1.], [1., 0.]] {
            let mut other = flat();
            other.set_offset(offset);
            assert!(!flat().is_aligned_with(&Grid::TriGrid(other)));
        }
    }

    #[test]
    fn a_differing_orientation_is_reported() {
        let other = TriGrid::new(1.2, Orientation::Pointy);
        assert!(!flat().is_aligned_with(&Grid::TriGrid(other)));
    }

    #[test]
    fn a_differing_rotation_is_reported() {
        let mut other = flat();
        other.set_rotation(15.5);
        assert!(!flat().is_aligned_with(&Grid::TriGrid(other)));
    }

    #[test]
    fn a_differing_grid_type_is_reported() {
        let grid = flat();

        let others = vec![
            Grid::RectGrid(RectGrid::new(1.2, 1.2)),
            Grid::HexGrid(HexGrid::new(1.2, Orientation::Pointy)),
        ];
        for other in others {
            assert!(!grid.is_aligned_with(&other));
            // The method is symmetric.
            assert!(!other.is_aligned_with(&Grid::TriGrid(flat())));
        }
    }

    #[test]
    fn multiple_reasons_are_reported_together() {
        let mut other = TriGrid::new(1.1, Orientation::Flat);
        other.set_offset([1., 1.]);
        assert!(!flat().is_aligned_with(&Grid::TriGrid(other)));
    }

    #[test]
    fn both_grids_are_left_intact() {
        let mut grid = flat();
        grid.set_offset([0.3, 0.4]);
        let mut other = TriGrid::new(1.1, Orientation::Flat);
        other.set_offset([1., 1.]);
        other.set_rotation(15.5);

        let snapshot = |g: &TriGrid| (g.cellsize, g.offset(), g.rotation(), g.orientation.clone());
        let before_grid = snapshot(&grid);
        let before_other = snapshot(&other);

        assert!(!grid.is_aligned_with(&Grid::TriGrid(other.clone())));
        assert!(!other.is_aligned_with(&Grid::TriGrid(grid.clone())));

        assert_eq!(before_grid, snapshot(&grid));
        assert_eq!(before_other, snapshot(&other));
    }

    #[test]
    fn it_is_reachable_through_the_grid_enum() {
        // `is_aligned_with` is a `GridTraits` method, so `enum_delegate` has to
        // forward it for `Grid` to be usable with it.
        let grid = Grid::TriGrid(flat());
        let other = Grid::TriGrid(flat());

        assert!(grid.is_aligned_with(&other));

        let mismatched = Grid::TriGrid(TriGrid::new(1.3, Orientation::Flat));
        assert_eq!(
            grid.is_aligned_with(&mismatched),
            flat().is_aligned_with(&mismatched)
        );
    }
}
