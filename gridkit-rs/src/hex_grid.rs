use std::hash::{Hash, Hasher};

use crate::grid::{CellElement, Grid, GridTraits, Orientation};
use crate::tri_grid::TriGrid;
use crate::utils::*;
use ndarray::*;

#[derive(Clone, Debug)]
pub struct HexGrid {
    pub cellsize: f64,
    pub offset: [f64; 2],
    pub orientation: Orientation,
    pub _rotation: f64,
    pub _rotation_matrix: Array2<f64>,
    pub _rotation_matrix_inv: Array2<f64>,
}

impl PartialEq for HexGrid {
    // Needs manual implementation, derive PartialEq does not work on floats because of NaN etc.
    fn eq(&self, other: &Self) -> bool {
        is_close(self.cellsize, other.cellsize)
            && is_close(self.offset[0], other.offset[0])
            && is_close(self.offset[1], other.offset[1])
            && is_close(self._rotation, other._rotation)
            && self.orientation == other.orientation
    }
}

impl Eq for HexGrid {}

impl Hash for HexGrid {
    // Needs manual implementation, derive Hash does not work on floats because of NaN etc.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cellsize.to_bits().hash(state);
        self.offset[0].to_bits().hash(state);
        self.offset[1].to_bits().hash(state);
        self.orientation.hash(state);
        self._rotation.to_bits().hash(state);
    }
}

impl GridTraits for HexGrid {
    fn get_grid(&self) -> crate::Grid {
        crate::Grid::HexGrid(self.to_owned())
    }

    fn dx(&self) -> f64 {
        match self.orientation {
            Orientation::Pointy => self.cellsize,
            Orientation::Flat => 3. / 2. * self.radius(),
        }
    }
    fn dy(&self) -> f64 {
        match self.orientation {
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

    fn set_offset(&mut self, mut offset: [f64; 2]) {
        if modulus(
            (offset[self.inconsistent_axis()] / self.stepsize_inconsistent_axis()).floor(),
            2.,
        ) != 0.
        {
            // Row is odd
            offset[self.consistent_axis()] -= self.stepsize_consistent_axis() / 2.
        }
        self.offset = normalize_offset(offset, self.dx(), self.dy());
    }

    fn rotation(&self) -> f64 {
        self._rotation
    }

    fn set_rotation(&mut self, rotation: f64) {
        self._rotation = rotation;
        self._rotation_matrix = rotation_matrix_from_angle(rotation);
        self._rotation_matrix_inv = rotation_matrix_from_angle(-rotation);
    }

    fn rotation_matrix(&self) -> ArrayView2<f64> {
        self._rotation_matrix.view()
    }
    fn rotation_matrix_inv(&self) -> ArrayView2<f64> {
        self._rotation_matrix_inv.view()
    }

    fn radius(&self) -> f64 {
        self.cellsize / 3_f64.sqrt()
    }

    fn cell_height(&self) -> f64 {
        match self.orientation {
            Orientation::Pointy => self.radius() * 2.,
            Orientation::Flat => self.dy(),
        }
    }

    fn cell_width(&self) -> f64 {
        match self.orientation {
            Orientation::Pointy => self.dx(),
            Orientation::Flat => self.radius() * 2.,
        }
    }

    fn centroid_xy_no_rot(&self, x: i64, y: i64) -> [f64; 2] {
        let mut centroid_x = x as f64 * self.dx() + (self.dx() / 2.) + self.offset[0];
        let mut centroid_y = y as f64 * self.dy() + (self.dy() / 2.) + self.offset[1];

        match self.orientation() {
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
    fn cell_at_points<D>(&self, points: ArrayView<f64, D>) -> Array<i64, D>
    where
        D: Dimension,
    {
        crate::utils::map_point_pairs(points, |points| {
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

            // The radius and these derived constants do not depend on the point,
            // so hoist them out of the loop instead of recomputing them (and the
            // sqrt inside `radius()`) for every point.
            let radius = self.radius();
            let radius_quarter = radius / 4.;
            let radius_1_25 = radius * 1.25;
            let radius_5_4 = radius * 5. / 4.;
            let dx_over_radius = dx / radius;

            // `rotate` is loop-invariant. Splitting this into one loop with rotation
            // and one without, and replacing the ndarray `dot` below with hand-written
            // scalar math, were both benchmarked and made no measurable difference.
            // The branch is kept inline and the clearer `dot` is used.
            let rotate = self.rotation() != 0.;

            for cell_id in 0..points.shape()[0] {
                let point_no_rot = points.slice(s![cell_id, ..]);
                let rotated;
                let point = if rotate {
                    rotated = self._rotation_matrix_inv.dot(&point_no_rot);
                    rotated.view()
                } else {
                    point_no_rot
                };

                let x = point[Ix1(id_x_axis)];
                let y = point[Ix1(id_y_axis)];

                // determine initial id_y
                let mut id_y = ((y - offset_y - radius_quarter) / dy).floor();
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
                let rel_loc_y = modulus(y - offset_y - radius_quarter, dy) + radius_quarter;
                let rel_loc_x = modulus(x - offset_x, dx);

                let mut in_top_left: bool;
                let mut in_top_right: bool;
                if is_offset == true {
                    in_top_left =
                        (radius_1_25 - rel_loc_y) < ((rel_loc_x - 0.5 * dx) / dx_over_radius);
                    in_top_left = in_top_left && (rel_loc_x < (0.5 * dx));
                    in_top_right =
                        (rel_loc_x - 0.5 * dx) / dx_over_radius <= (rel_loc_y - radius_1_25);
                    in_top_right = in_top_right && rel_loc_x >= (0.5 * dx);
                    if in_top_left == true {
                        id_y = id_y + 1.;
                        id_x = id_x + 1.;
                    } else if in_top_right == true {
                        id_y = id_y + 1.;
                    }
                } else {
                    in_top_left = rel_loc_x / dx_over_radius < (rel_loc_y - radius_5_4);
                    in_top_right = (radius_1_25 - rel_loc_y) <= (rel_loc_x - dx) / dx_over_radius;
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
        })
    }

    fn cell_corners<D>(&self, index: ArrayView<i64, D>) -> Array<f64, D::Larger>
    where
        D: Dimension,
    {
        crate::utils::map_point_pairs_fanout(index, |index| {
            let mut corners = Array3::<f64>::zeros((index.shape()[0], 6, 2));

            for cell_id in 0..index.shape()[0] {
                for corner_id in 0..6 {
                    let angle_deg = match self.orientation() {
                        Orientation::Pointy => 60. * corner_id as f64 - 30.,
                        Orientation::Flat => 60. * corner_id as f64,
                    };
                    let angle_rad = angle_deg * std::f64::consts::PI / 180.;
                    let centroid =
                        self.centroid_xy_no_rot(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
                    corners[Ix3(cell_id, corner_id, 0)] =
                        centroid[0] + self.radius() * angle_rad.cos();
                    corners[Ix3(cell_id, corner_id, 1)] =
                        centroid[1] + self.radius() * angle_rad.sin();
                }
            }

            if self.rotation() != 0. {
                // Applying the rotation by hand avoids the temporary allocation
                // that a `_rotation_matrix.dot(&corner)` call would make per corner.
                let cos = self._rotation_matrix[[0, 0]];
                let sin = self._rotation_matrix[[1, 0]];
                for cell_id in 0..corners.shape()[0] {
                    for corner_id in 0..corners.shape()[1] {
                        let x = corners[Ix3(cell_id, corner_id, 0)];
                        let y = corners[Ix3(cell_id, corner_id, 1)];
                        corners[Ix3(cell_id, corner_id, 0)] = cos * x - sin * y;
                        corners[Ix3(cell_id, corner_id, 1)] = sin * x + cos * y;
                    }
                }
            }
            corners
        })
    }

    fn cells_near_point<D>(&self, points: ArrayView<f64, D>) -> Array<i64, D::Larger>
    where
        D: Dimension,
    {
        crate::utils::map_point_pairs_fanout(points, |points| {
            // There are 6 options for the nearby cells,
            // based on where in a cell the point is located.
            // Hence there are 6 sections within the cell,
            // each 60 degrees wide (360/6 = 60).
            // We divide the quadrants based on the angle between
            // the point and pure up, as measured from the centroid of the cell.
            let mut nearby_cells = Array3::<i64>::zeros((points.shape()[0], 3, 2));

            // Since points are rotated in cell_at_points, perform this call before
            // rotating the points in this function.
            // Ideally we have a version cell_at_points that does not rotate, same
            // as we do with centroid_xy_no_rot.
            let cell_ids = self.cell_at_points(points);

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
                // Applying the rotation by hand avoids the temporary allocation
                // that a `_rotation_matrix_inv.dot(&point)` call would make per point.
                let cos = self._rotation_matrix[[0, 0]];
                let sin = self._rotation_matrix[[1, 0]];
                for cell_id in 0..points_.shape()[0] {
                    let x = points_[Ix2(cell_id, 0)];
                    let y = points_[Ix2(cell_id, 1)];
                    points_[Ix2(cell_id, 0)] = cos * x + sin * y;
                    points_[Ix2(cell_id, 1)] = -sin * x + cos * y;
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
                match self.orientation() {
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
                match self.orientation() {
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
        })
    }

    fn all_neighbours<D>(
        &self,
        index: ArrayView<i64, D>,
        depth: u64,
        include_selected: bool,
        add_cell_id: bool,
    ) -> Array<i64, D::Larger>
    where
        D: Dimension,
    {
        // Python's `HexGrid.relative_neighbours` documents `connect_corners` as
        // "not relevant in hexagonal grids" and it indeed returns the same result
        // for both values, so both trait methods share one implementation.
        crate::utils::map_point_pairs_fanout(index, |index| {
            self._neighbours(index, depth, include_selected, add_cell_id)
        })
    }

    fn direct_neighbours<D>(
        &self,
        index: ArrayView<i64, D>,
        depth: u64,
        include_selected: bool,
        add_cell_id: bool,
    ) -> Array<i64, D::Larger>
    where
        D: Dimension,
    {
        crate::utils::map_point_pairs_fanout(index, |index| {
            self._neighbours(index, depth, include_selected, add_cell_id)
        })
    }

    fn subdivide(&self, factor: u64) -> Grid {
        // A hexagon cannot be tiled by smaller hexagons, so a *TriGrid* is
        // returned from `HexGrid.subdivide` to keep the alignment exact. The
        // triangle edge length is the hexagon radius, so the sub grid's `size`
        // (which for a TriGrid is the side length) is `self.radius() / factor`.
        if factor == 0 {
            return Grid::HexGrid(self.clone());
        }

        // Add 30 degrees for 'pointy' orientation and keep the rotation
        // for 'flat', so the triangular cells line up with the hexagon corners.
        let extra_rotation = match self.orientation {
            Orientation::Pointy => 30.,
            Orientation::Flat => 0.,
        };

        let mut sub_grid = TriGrid::new(self.radius() / factor as f64, Orientation::Flat);
        sub_grid.set_rotation(self.rotation() + extra_rotation);
        sub_grid.set_offset(self.offset());

        // Anchor the sub grid to the top corner of the parent's cell (0, 0). The corner
        // is taken from the *parent*, so it is unaffected by the cellsize change above.
        let corners = self.cell_corners(array![0i64, 0i64].view());
        let anchor_loc = [corners[[0, 0]], corners[[0, 1]]];
        sub_grid.anchor_inplace(&anchor_loc, CellElement::Corner);

        Grid::TriGrid(sub_grid)
    }

    fn is_aligned_with(&self, other: &Grid) -> bool {
        if let Grid::HexGrid(other) = other {
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
}

impl HexGrid {
    /// Shared implementation of the `GridTraits` neighbour methods.
    fn _neighbours(
        &self,
        index: ArrayView2<i64>,
        depth: u64,
        include_selected: bool,
        add_cell_id: bool,
    ) -> Array3<i64> {
        let add_cell_id = add_cell_id as i64;

        // Python sizes the array as `sum(6 * arange(1, depth + 1)) + 1`, i.e.
        // including the selected cell, and deletes it afterwards when needed.
        // `6 * (1 + 2 + ... + depth)` is `3 * depth * (depth + 1)`.
        let nr_neighbours = (3 * depth * (depth + 1) + 1) as usize;

        // Python works in terms of a `pointy_axis` and a `flat_axis`, which are
        // the inconsistent and consistent axes of this grid respectively.
        let pointy_axis = self.inconsistent_axis();
        let flat_axis = self.consistent_axis();

        // Python always builds the selected cell into the middle of the array and
        // deletes it afterwards when `include_selected` is False. Building the
        // relative ids as a flat list first lets the same code serve both cases
        // without a second pass over the data.
        let nr_cells = index.shape()[0];
        let mut relative = vec![[0i64; 2]; nr_cells * nr_neighbours];

        for cell_id in 0..nr_cells {
            let is_odd_row = !iseven(index[Ix2(cell_id, pointy_axis)]);
            let row_start = cell_id * nr_neighbours;
            let cell_slice = &mut relative[row_start..row_start + nr_neighbours];

            // Python loops over `rows = range(depth, -1, -1)`, tracking the row
            // number `i` separately from the row *value*, which is `depth - i`.
            let mut start_slice = 0usize;
            for i in 0..=depth {
                let row = depth - i;
                let row_length = (depth + i + 1) as usize;
                let max_val = row_length as i64 / 2;

                // Rows alternate between a full range of flat ids and a pair of
                // ranges that depend on the parity of the cell's row. Both
                // branches produce exactly `row_length` ids.
                let (lo, hi) = if (i % 2 == 0) == (depth % 2 == 0) {
                    (-max_val, max_val + 1)
                } else if is_odd_row {
                    (-max_val + 1, max_val + 1)
                } else {
                    (-max_val, max_val)
                };

                for flat_id in lo..hi {
                    cell_slice[start_slice][flat_axis] = flat_id;
                    cell_slice[start_slice][pointy_axis] = row as i64;
                    start_slice += 1;
                }
            }

            // Mirror the top half onto the bottom half, leaving the centre row in
            // place. The centre row is the widest one, at `2 * depth + 1` cells,
            // and is the last row written by the loop above.
            let centre_row_start = start_slice - (2 * depth + 1) as usize;
            let mut mirrored = start_slice;
            for source in (0..centre_row_start).rev() {
                cell_slice[mirrored] = cell_slice[source];
                cell_slice[mirrored][pointy_axis] = -cell_slice[source][pointy_axis];
                mirrored += 1;
            }
            debug_assert_eq!(mirrored, nr_neighbours);

            // Python deletes `floor(nr_neighbours / 2)`, which is the selected cell
            // sitting exactly in the middle. Shift the tail down in place so the
            // remaining cells keep their order; the final slot is then unused.
            if !include_selected {
                let centre = nr_neighbours / 2;
                debug_assert_eq!((cell_slice[centre][0], cell_slice[centre][1]), (0, 0));
                for i in centre..nr_neighbours - 1 {
                    cell_slice[i] = cell_slice[i + 1];
                }
            }
        }

        let kept = if include_selected {
            nr_neighbours
        } else {
            nr_neighbours - 1
        };
        let mut neighbours = Array3::<i64>::zeros((nr_cells, kept, 2));
        for cell_id in 0..nr_cells {
            let row_start = cell_id * nr_neighbours;
            for cell in 0..kept {
                neighbours[Ix3(cell_id, cell, 0)] =
                    relative[row_start + cell][0] + add_cell_id * index[Ix2(cell_id, 0)];
                neighbours[Ix3(cell_id, cell, 1)] =
                    relative[row_start + cell][1] + add_cell_id * index[Ix2(cell_id, 1)];
            }
        }
        neighbours
    }
    pub fn new(cellsize: f64, orientation: Orientation) -> Self {
        let _rotation_matrix = rotation_matrix_from_angle(0.);
        let _rotation_matrix_inv = rotation_matrix_from_angle(-0.);
        HexGrid {
            cellsize,
            offset: [0., 0.],
            orientation,
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
            Orientation::Pointy => 0,
            Orientation::Flat => 1,
        }
    }

    pub fn inconsistent_axis(&self) -> usize {
        match self.orientation {
            Orientation::Pointy => 1,
            Orientation::Flat => 0,
        }
    }

    pub fn stepsize_consistent_axis(&self) -> f64 {
        match self.orientation {
            Orientation::Pointy => self.dx(),
            Orientation::Flat => self.dy(),
        }
    }

    pub fn stepsize_inconsistent_axis(&self) -> f64 {
        match self.orientation {
            Orientation::Pointy => self.dy(),
            Orientation::Flat => self.dx(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tri_grid::TriGrid;

    const TOL: f64 = 1e-9;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol,
            "expected {b} to be within {tol} of {a} (difference {})",
            (a - b).abs()
        );
    }

    fn assert_ids_3d(result: ArrayView3<i64>, expected: &[i64]) {
        assert_eq!(result.shape()[2], 2); // xy coordinates in last axis
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
    // ---------------------------------------------------------------------

    #[test]
    fn direct_neighbours_depth_zero() {
        // No neighbours in case depth is set to 0
        let grid = HexGrid::new(3., Orientation::Pointy);
        let neighbours = grid.direct_neighbours(array![[0i64, 0i64]].view(), 0, false, false);
        assert_eq!(neighbours.shape(), &[1, 0, 2]);
        assert_ids_3d(neighbours.view(), &[]);
    }

    #[test]
    fn direct_neighbours_depth_zero_with_self() {
        // No neighbours in case depth is set to 0
        let grid = HexGrid::new(3., Orientation::Pointy);
        let neighbours = grid.direct_neighbours(array![[0i64, -1i64]].view(), 0, true, true);
        assert_eq!(neighbours.shape(), &[1, 1, 2]);
        assert_ids_3d(neighbours.view(), &[0, -1]);
    }

    #[test]
    fn direct_neighbours_depth_zero_with_self_relative() {
        // No neighbours in case depth is set to 0
        let grid = HexGrid::new(3., Orientation::Pointy);
        let neighbours = grid.direct_neighbours(array![[0i64, -1i64]].view(), 0, true, false);
        assert_eq!(neighbours.shape(), &[1, 1, 2]);
        assert_ids_3d(neighbours.view(), &[0, 0]);
    }

    #[test]
    fn direct_neighbours_depth_one() {
        let grid = HexGrid::new(3., Orientation::Pointy);
        let neighbours = grid.direct_neighbours(array![[0i64, 0i64]].view(), 1, false, false);
        assert_eq!(neighbours.shape(), &[1, 6, 2]);
        assert_ids_3d(
            neighbours.view(),
            &[-1, 1, 0, 1, -1, 0, 1, 0, 0, -1, -1, -1],
        );
    }

    #[test]
    fn direct_neighbours_depend_on_the_parity_of_the_row() {
        // The second doctest of `HexGrid.relative_neighbours`: on an odd row the
        // flat ids shift by one, because odd rows are offset by half a cell.
        let grid = HexGrid::new(3., Orientation::Pointy);
        let neighbours = grid.direct_neighbours(array![[0i64, 1i64]].view(), 1, false, false);
        assert_ids_3d(neighbours.view(), &[0, 1, 1, 1, -1, 0, 1, 0, 1, -1, 0, -1]);
    }

    #[test]
    fn neighbours_of_a_flat_grid() {
        // Python returns the same ids for index [0, 0] and [0, 1] on a flat
        // grid: the parity that matters is that of the *x* index, which is 0 for
        // both.
        let grid = HexGrid::new(1., Orientation::Flat);
        let expected = &[1, -1, 1, 0, 0, -1, 0, 1, -1, 0, -1, -1];

        assert_ids_3d(
            grid.direct_neighbours(array![[0i64, 0i64]].view(), 1, false, false)
                .view(),
            expected,
        );
        assert_ids_3d(
            grid.direct_neighbours(array![[0i64, 1i64]].view(), 1, false, false)
                .view(),
            expected,
        );
    }

    #[test]
    fn all_neighbours_ignore_connect_corners() {
        // Python's `HexGrid.relative_neighbours` documents `connect_corners` as
        // irrelevant for hexagonal grids and returns the same ids either way, so
        // `all_neighbours` and `direct_neighbours` must agree.
        for orientation in [Orientation::Pointy, Orientation::Flat] {
            let grid = HexGrid::new(3., orientation.clone());

            for depth in 1..=6 {
                for include_selected in [false, true] {
                    let index = array![[2i64, 1i64], [-3i64, 4i64]];
                    assert_eq!(
                        grid.all_neighbours(index.view(), depth, include_selected, true),
                        grid.direct_neighbours(index.view(), depth, include_selected, true)
                    );
                }
            }
        }
    }

    #[test]
    fn neighbour_count_grows_with_depth() {
        // The doctest of `HexGrid.relative_neighbours` pins the counts at
        // [6, 18, 36, 60] for depth 1 to 4 without the selected cell, and one
        // more with it.
        const DOCUMENTED: [usize; 4] = [6, 18, 36, 60];

        let grid = HexGrid::new(3., Orientation::Pointy);
        let index = array![[0i64, 0i64]];

        for depth in 1..=6 {
            let without = grid.direct_neighbours(index.view(), depth, false, false);
            let with = grid.direct_neighbours(index.view(), depth, true, false);

            assert_eq!(with.shape()[1], without.shape()[1] + 1);
            if (depth as usize) <= DOCUMENTED.len() {
                assert_eq!(without.shape()[1], DOCUMENTED[depth as usize - 1]);
            }
            // The closed form Python uses, `3 * depth * (depth + 1)`.
            assert_eq!(with.shape()[1], (3 * depth * (depth + 1) + 1) as usize);
        }
    }

    #[test]
    fn include_selected_places_the_cell_at_the_centre() {
        // Python deletes the selected cell at `floor(len / 2)`, so with
        // `include_selected` it has to sit back at exactly that index.
        let grid = HexGrid::new(3., Orientation::Pointy);

        for depth in 1..=6 {
            let index = array![[7i64, -3i64]];
            let with = grid.direct_neighbours(index.view(), depth, true, true);
            let without = grid.direct_neighbours(index.view(), depth, false, true);

            let centre = with.shape()[1] / 2;
            assert_eq!([[with[[0, centre, 0]], with[[0, centre, 1]]]], [[7, -3]]);

            // Dropping the selected cell must give exactly the same cells, in
            // the same order, as the shorter array.
            for i in 0..without.shape()[1] {
                let j = if i < centre { i } else { i + 1 };
                assert_eq!(
                    [[without[[0, i, 0]], without[[0, i, 1]]]],
                    [[with[[0, j, 0]], with[[0, j, 1]]]]
                );
            }
        }
    }

    #[test]
    fn neighbours_multiple_indices() {
        // The literal expected ids of `test_relative_neighbours`, which passes a
        // two-row index of [4, 4] and [3, 5].
        let grid = HexGrid::new(1., Orientation::Pointy);
        let index = array![[4i64, 4i64], [3i64, 5i64]];

        assert_ids_3d(
            grid.direct_neighbours(index.view(), 1, false, true).view(),
            &[
                3, 5, 4, 5, 3, 4, 5, 4, 4, 3, 3, 3, //
                3, 6, 4, 6, 2, 5, 4, 5, 4, 4, 3, 4,
            ],
        );
    }

    #[test]
    fn neighbours_single_and_multiple_indices_agree() {
        // `relative_neighbours` only uses `index` for its repeat count and for the
        // parity of the row, so a single index must give the same ids as the
        // matching row of a multi index call.
        let grid = HexGrid::new(1., Orientation::Pointy);
        let index = array![[-1i64, 0i64], [2i64, 1i64], [-3i64, 2i64]];

        for depth in 1..=6 {
            for include_selected in [false, true] {
                let many = grid.direct_neighbours(index.view(), depth, include_selected, true);

                for (row, expected) in [(-1i64, 0i64), (2, 1), (-3, 2)].iter().enumerate() {
                    let one = array![[expected.0, expected.1]];
                    let single = grid.direct_neighbours(one.view(), depth, include_selected, true);

                    assert_eq!(single.shape(), &[1, many.shape()[1], 2]);
                    for i in 0..single.shape()[1] {
                        assert_eq!(
                            [[single[[0, i, 0]], single[[0, i, 1]]]],
                            [[many[[row, i, 0]], many[[row, i, 1]]]]
                        );
                    }

                    // The relative ids plus the cell id are the absolute ids.
                    let relative =
                        grid.direct_neighbours(one.view(), depth, include_selected, false);
                    for i in 0..single.shape()[1] {
                        assert_eq!(
                            [[single[[0, i, 0]], single[[0, i, 1]]]],
                            [[
                                relative[[0, i, 0]] + expected.0,
                                relative[[0, i, 1]] + expected.1
                            ]]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn neighbours_respect_depth_and_are_unique() {
        // The invariants `test_neighbours` asserts: cells at the same distance from
        // the centre always come in multiples of six, nothing is further away than
        // `depth` cell sizes, and no cell is listed twice.
        for orientation in [Orientation::Pointy, Orientation::Flat] {
            let grid = HexGrid::new(3., orientation.clone());

            for depth in 1..=6 {
                for index in [array![[2i64, 1i64]], array![[1i64, 2i64]]] {
                    let neighbours = grid.direct_neighbours(index.view(), depth, false, true);

                    let centroids = grid.centroid(neighbours.slice(s![0, .., ..]));
                    let center = grid.centroid(index.view());
                    let offsets = centroids - &center;

                    // No cell is listed twice, and the selected cell is not
                    // among them.
                    let mut ids: Vec<[i64; 2]> = (0..neighbours.shape()[1])
                        .map(|i| [neighbours[[0, i, 0]], neighbours[[0, i, 1]]])
                        .collect();
                    let total = ids.len();
                    ids.sort_unstable();
                    ids.dedup();
                    assert_eq!(ids.len(), total, "duplicate cell in the neighbours");

                    let distances: Vec<f64> = offsets
                        .axis_iter(Axis(0))
                        .map(|xy| (xy[Ix1(0)].powi(2) + xy[Ix1(1)].powi(2)).sqrt())
                        .collect();

                    // Cells at the same distance always come in multiples of six.
                    let mut distances = distances;
                    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mut run = 1;
                    for i in 1..distances.len() {
                        if (distances[i] - distances[i - 1]).abs() < 1e-6 {
                            run += 1;
                        } else {
                            assert_eq!(run % 6, 0, "run of {run} cells at one distance");
                            run = 1;
                        }
                    }
                    assert_eq!(run % 6, 0, "run of {run} cells at one distance");

                    // Nothing can be further away than `depth` cell sizes.
                    for distance in &distances {
                        assert!(
                            *distance <= grid.cellsize * depth as f64 + 1e-9,
                            "cell at {distance} is further than {depth} cellsizes away"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn neighbours_are_reachable_through_the_grid_enum() {
        // `all_neighbours`/`direct_neighbours` are `GridTraits` methods, so
        // `enum_delegate` has to forward them for `Grid` to be usable with them.
        let concrete = HexGrid::new(3., Orientation::Pointy);
        let grid = Grid::HexGrid(concrete.clone());
        let index = array![[3i64, -2i64]];

        assert_eq!(
            grid.all_neighbours(index.view(), 2, true, true),
            concrete.all_neighbours(index.view(), 2, true, true)
        );
        assert_eq!(
            grid.direct_neighbours(index.view(), 2, false, false),
            concrete.direct_neighbours(index.view(), 2, false, false)
        );
    }

    // ---------------------------------------------------------------------
    // subdivide
    // ---------------------------------------------------------------------

    /// Unwrap the triangular sub grid that `HexGrid::subdivide` returns.
    fn as_tri_grid(sub_grid: Grid) -> TriGrid {
        match sub_grid {
            Grid::TriGrid(grid) => grid,
            _ => panic!("subdivide on HexGrid should return TriGrid"),
        }
    }

    #[test]
    fn subdivide_returns_a_tri_grid() {
        // A hexagon cannot be tiled by smaller hexagons, so Python returns a
        // TriGrid. The triangle side length is the hexagon radius, so the
        // sub grid's `cellsize` (which for a TriGrid is the side length) is
        // `r / factor`.
        for orientation in [Orientation::Pointy, Orientation::Flat] {
            let grid = HexGrid::new(3., orientation.clone());

            for factor in [1u64, 2, 3, 9] {
                let sub_grid = as_tri_grid(grid.subdivide(factor));
                assert_close(sub_grid.cellsize, grid.radius() / factor as f64, TOL);
            }
        }
    }

    #[test]
    fn subdivide_rotates_the_tri_grid_for_pointy_cells() {
        // Python adds 30 degrees for 'pointy' orientation and keeps the rotation
        // for 'flat', so the triangles line up with the hexagon corners.
        let mut pointy = HexGrid::new(3., Orientation::Pointy);
        pointy.set_rotation(11.);
        assert_close(as_tri_grid(pointy.subdivide(2)).rotation(), 41., TOL);

        let mut flat = HexGrid::new(3., Orientation::Flat);
        flat.set_rotation(11.);
        assert_close(as_tri_grid(flat.subdivide(2)).rotation(), 11., TOL);
    }

    #[test]
    fn subdivide_keeps_parent_corners_on_a_corner() {
        for orientation in [Orientation::Pointy, Orientation::Flat] {
            for rotation in [-23., 0., 456.] {
                for offset in [[-2., 3.], [0., 0.], [0.1, -0.2]] {
                    let mut grid = HexGrid::new(1., orientation.clone());
                    grid.set_rotation(rotation);
                    grid.set_offset(offset);

                    for factor in [2u64, 9] {
                        let sub_grid = as_tri_grid(grid.subdivide(factor));

                        // Take a corner of a parent cell and check that it coincides
                        // with one of the corners of the sub cell containing it.
                        let corners = grid.cell_corners(array![[-4i64, 23i64]].view());
                        let corner = [corners[[0, 5, 0]], corners[[0, 5, 1]]];

                        let id = sub_grid.cell_at_point(&corner);
                        let sub_corners = sub_grid.cell_corners(array![[id[0], id[1]]].view());
                        let on_corner = (0..3).any(|i| {
                            let dx = sub_corners[[0, i, 0]] - corner[0];
                            let dy = sub_corners[[0, i, 1]] - corner[1];
                            (dx * dx + dy * dy).sqrt() <= 1e-9
                        });
                        assert!(
                            on_corner,
                            "corner {corner:?} is not a corner of the sub cell containing it \
                             ({orientation:?}, rotation {rotation}, offset {offset:?}, \
                             factor {factor})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn subdivide_yields_six_factor_squared_subcells_per_parent_cell() {
        // Six triangles per hexagon side, so the parent cell is divided into
        // `6 * factor^2` sub cells.
        for orientation in [Orientation::Pointy, Orientation::Flat] {
            for factor in [2u64, 9] {
                for rotation in [-23., 0., 456.] {
                    for offset in [[-2., 3.], [0., 0.], [0.1, -0.2]] {
                        let mut grid = HexGrid::new(1., orientation.clone());
                        grid.set_rotation(rotation);
                        grid.set_offset(offset);

                        let sub_grid = as_tri_grid(grid.subdivide(factor));

                        // The sub cells that could possibly be inside the parent cell
                        // (3, -2) are the ones around the sub cell holding the parent
                        // centroid. Triangular windows reach further per step than the
                        // square ones, hence the generous depth.
                        let target = grid.centroid(array![[3i64, -2i64]].view());
                        let start = sub_grid.cell_at_point(&[target[[0, 0]], target[[0, 1]]]);
                        let candidates = sub_grid.all_neighbours(
                            array![[start[0], start[1]]].view(),
                            factor + 1,
                            true,
                            true,
                        );

                        let sub_centroids = sub_grid.centroid(candidates.slice(s![0, .., ..]));
                        let in_cell = grid.cell_at_points(sub_centroids.view());
                        let nr_in_cell = (0..in_cell.shape()[0])
                            .filter(|i| in_cell[[*i, 0]] == 3 && in_cell[[*i, 1]] == -2)
                            .count();

                        assert_eq!(
                            nr_in_cell,
                            (6 * factor * factor) as usize,
                            "{orientation:?}, rotation {rotation}, offset {offset:?}, \
                             factor {factor}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn subdivide_by_one_matches_the_documented_workaround() {
        // Python's docstring shows how to build a smaller *HexGrid* by hand. A
        // factor of 1 must give the same cellsize and, after anchoring, the same
        // geometry as the equivalent hand-built grid.
        let mut grid = HexGrid::new(3., Orientation::Pointy);
        grid.set_rotation(-23.);
        grid.set_offset([0.1, -0.2]);

        let sub_grid = as_tri_grid(grid.subdivide(1));
        assert_close(sub_grid.cellsize, grid.radius(), TOL);
    }

    #[test]
    fn subdivide_by_zero_is_the_same_grid() {
        // `factor` is unsigned, so a `factor` of 0 is the only invalid input left.
        // Dividing the cellsize by zero is not meaningful, so this yields an
        // unchanged copy instead. Note this diverges from Python, which raises a
        // `ValueError` for any `factor` below 1.
        let mut grid = HexGrid::new(1.23, Orientation::Pointy);
        grid.set_rotation(15.5);
        grid.set_offset([0.1, 0.2]);

        assert_eq!(grid.subdivide(0), Grid::HexGrid(grid));
    }
}

#[cfg(test)]
mod is_aligned_with_tests {
    use super::*;
    use crate::rect_grid::RectGrid;
    use crate::tri_grid::TriGrid;

    // Mirrors `test_is_aligned_with` in `test_hex_grid.py`, minus the CRS cases
    // (the Rust grid has no CRS) and the TypeError case (the Rust signature
    // takes a `&Grid`, so a non-grid cannot be passed).
    fn pointy() -> HexGrid {
        HexGrid::new(1.2, Orientation::Pointy)
    }

    #[test]
    fn an_identical_grid_is_aligned() {
        assert!(pointy().is_aligned_with(&Grid::HexGrid(pointy())));
    }

    #[test]
    fn a_differing_cellsize_is_reported() {
        assert!(!pointy().is_aligned_with(&Grid::HexGrid(HexGrid::new(1.3, Orientation::Pointy))));
    }

    #[test]
    fn a_differing_offset_is_reported() {
        for offset in [[0., 1.], [1., 0.]] {
            let mut other = HexGrid::new(1.2, Orientation::Pointy);
            other.set_offset(offset);
            assert!(!pointy().is_aligned_with(&Grid::HexGrid(other)));
        }
    }

    #[test]
    fn a_differing_orientation_is_reported() {
        assert!(!pointy().is_aligned_with(&Grid::HexGrid(HexGrid::new(1.2, Orientation::Flat))));
    }

    #[test]
    fn a_differing_rotation_is_reported() {
        let mut other = pointy();
        other.set_rotation(15.5);
        assert!(!pointy().is_aligned_with(&Grid::HexGrid(other)));
    }

    #[test]
    fn a_differing_grid_type_is_reported() {
        let grid = pointy();

        let others = vec![
            Grid::RectGrid(RectGrid::new(1.2, 1.2)),
            Grid::TriGrid(TriGrid::new(1.2, Orientation::Pointy)),
        ];
        for other in others {
            assert!(!grid.is_aligned_with(&other));
            assert!(!other.is_aligned_with(&Grid::HexGrid(pointy())));
        }
    }

    #[test]
    fn multiple_reasons_are_reported_together() {
        let mut other = HexGrid::new(1.1, Orientation::Flat);
        other.set_offset([1., 1.]);
        assert!(!pointy().is_aligned_with(&Grid::HexGrid(other)));
    }

    #[test]
    fn both_grids_are_left_intact() {
        let mut grid = pointy();
        grid.set_offset([0.3, 0.4]);
        let mut other = HexGrid::new(1.1, Orientation::Flat);
        other.set_offset([1., 1.]);
        other.set_rotation(15.5);

        let snapshot = |g: &HexGrid| (g.cellsize, g.offset(), g.rotation(), g.orientation.clone());
        let before_grid = snapshot(&grid);
        let before_other = snapshot(&other);

        assert!(!grid.is_aligned_with(&Grid::HexGrid(other.clone())));
        assert!(!other.is_aligned_with(&Grid::HexGrid(grid.clone())));

        assert_eq!(before_grid, snapshot(&grid));
        assert_eq!(before_other, snapshot(&other));
    }

    #[test]
    fn it_is_reachable_through_the_grid_enum() {
        // `is_aligned_with` is a `GridTraits` method, so `enum_delegate` has to
        // forward it for `Grid` to be usable with it.
        let grid = Grid::HexGrid(pointy());
        let other = Grid::HexGrid(pointy());

        assert!(grid.is_aligned_with(&other));

        let mismatched = Grid::HexGrid(HexGrid::new(1.3, Orientation::Pointy));
        assert_eq!(
            grid.is_aligned_with(&mismatched),
            HexGrid::new(1.2, Orientation::Pointy).is_aligned_with(&mismatched)
        );
    }
}
