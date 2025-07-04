use crate::grid::*;
use crate::tile::*;
use crate::RectGrid;
use core::f64;
use ndarray::*;
use num_traits::{AsPrimitive, Bounded, FromPrimitive, Num, NumCast, ToPrimitive};
use std::any::Any;
use std::f64::consts::E;
use std::f64::NAN;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

#[derive(Clone)]
pub struct DataTile<T> {
    pub tile: Tile,
    pub data: Array2<T>,
    pub nodata_value: T,
}

fn to_f64<
    T: Num + Clone + Copy + PartialEq + Bounded + ToPrimitive + FromPrimitive + PartialOrd + NumCast,
>(
    val: T,
) -> f64 {
    val.to_f64().unwrap_or(f64::NAN)
}

fn is_nodata_value_f64(val: f64, nodata_value: f64) -> bool {
    if nodata_value.is_nan() {
        return val.is_nan();
    }
    // Normal check for when self.nodata_value is not a nan
    val == nodata_value
}

fn is_nodata_value<
    T: Num + Clone + Copy + PartialEq + Bounded + ToPrimitive + FromPrimitive + PartialOrd + NumCast,
>(
    val: T,
    nodata_value: T,
) -> bool {
    // The following if-block is a dirty way of checking for nans which
    // are a float only concept. T.is_nan() does not exist and Rust does
    // not allow fow matching of T. Ideally we would have been able to do
    // something like match T {f64 => {handle_nan..}}.
    // Since that is not possible. I also tried something like
    // `impl IsNoData<T> for DataTile<T>` and
    // `impl IsNoData<f64> for DataTile<f64>`
    // But again that is not supported because I get an error saying:
    // conflicting implementation for f64. That leads us to the following hack.
    // This might be solved by `specialization` suggested here:
    // https://github.com/rust-lang/rfcs/pull/1210
    // But at the time of writing this is not avaialable in Rust stable (only nightly).
    if let Some(nodata_float) = nodata_value.to_f64() {
        if let Some(val_float) = val.to_f64() {
            if nodata_float.is_nan() {
                return val_float.is_nan();
            }
        }
    }
    // Normal check for when self.nodata_value is not a nan
    val == nodata_value
}

impl<
        T: Num
            + Clone
            + Copy
            + PartialEq
            + Bounded
            + ToPrimitive
            + FromPrimitive
            + PartialOrd
            + NumCast,
    > TileTraits for DataTile<T>
{
    fn get_tile(&self) -> &Tile {
        &self.tile
    }

    fn get_grid(&self) -> &Grid {
        &self.tile.grid
    }
}

impl<
        T: Num
            + Clone
            + Copy
            + PartialEq
            + Bounded
            + ToPrimitive
            + FromPrimitive
            + PartialOrd
            + NumCast,
    > DataTile<T>
{
    pub fn new(
        grid: Grid,
        start_id: (i64, i64),
        nx: u64,
        ny: u64,
        data: Array2<T>,
        nodata_value: T,
    ) -> Self {
        let tile = Tile {
            grid,
            start_id,
            nx,
            ny,
        };
        DataTile {
            tile: tile,
            data,
            nodata_value,
        }
    }

    pub fn from_bounds_as_rect(
        bounds: (f64, f64, f64, f64),
        data: Array2<T>,
        nodata_value: T,
    ) -> Self {
        // Note: not used in python, that has it's own version. This one is untested.
        let nx = data.shape()[1] as u64;
        let ny = data.shape()[0] as u64;
        let dx = (bounds.2 - bounds.0) / nx as f64;
        let dy = (bounds.3 - bounds.1) / ny as f64;
        let mut grid = RectGrid::new(dx, dy);
        let start_cell_centroid = [bounds.0 + dx/2., bounds.1+dy/2.];
        grid.anchor_inplace(&start_cell_centroid, CellElement::Centroid);
        let start_id = grid.cell_at_point(&start_cell_centroid);
        let tile = Tile {
            grid: Grid::RectGrid(grid),
            start_id: start_id.into(), // TODO: align tpye of start_id, use [] intead of ()
            nx,
            ny,
        };
        DataTile {
            tile: tile,
            data,
            nodata_value,
        }
    }

    pub fn into_dtype<U>(&self) -> DataTile<U>
    where
        T: AsPrimitive<U>,
        U: 'static + Clone,
        T: 'static, // Needed to use AsPrimitive safely
        T: Copy,
        U: Copy,
    {
        let data = self.data.map(|x| (*x).as_());
        let nodata_value = self.nodata_value.as_();
        DataTile {
            tile: self.tile.to_owned(),
            data,
            nodata_value,
        }
    }

    pub fn set_nodata_value(&mut self, nodata_value: T) {
        let old_nodata_value = self.nodata_value;
        for val in self.data.iter_mut() {
            if is_nodata_value(*val, old_nodata_value) {
                *val = nodata_value;
            }
        }
        self.nodata_value = nodata_value;
    }

    pub fn is_nodata(&self, &value: &T) -> bool {
        is_nodata_value(value, self.nodata_value)
    }

    pub fn is_nodata_array(&self, value: &ArrayViewD<T>) -> ArrayD<bool> {
        let mut result = Array::default(value.shape());
        for (idx, val) in value.indexed_iter() {
            result[idx] = self.is_nodata(&val);
        }
        result
    }

    pub fn nodata_cells(&self) -> Array2<i64> {
        // Returns ids of cells with nodata value

        let nodata_mask = self.is_nodata_array(&self.data.view().into_dyn());
        // Note: nodata_mask.sum() returns a bool, which is not what we are after
        //       when summing a boolean array. So I'll do the sum myself.
        let mut nr_nodata: usize = 0;
        for val in nodata_mask.iter() {
            if *val {
                nr_nodata += 1;
            }
        }

        // Now for each true in nodata_mask, get the index
        let mut result = Array2::<i64>::zeros((nr_nodata, 2));
        let ids = self.indices();
        let mut result_id = 0;
        for id_y in 0..ids.shape()[0] {
            for id_x in 0..ids.shape()[1] {
                if nodata_mask[Ix2(id_y, id_x)] {
                    result[Ix2(result_id, 0)] = ids[Ix3(id_y, id_x, 0)];
                    result[Ix2(result_id, 1)] = ids[Ix3(id_y, id_x, 1)];
                    result_id += 1;
                }
            }
        }
        result
    }

    pub fn data_cells(&self) -> DataTile<bool> {
        // Returns a DataTile with boolean values where each true indicates the cell has valid data
        let mut result = Array2::<bool>::default((self.tile.ny as usize, self.tile.nx as usize));
        for (idx, val) in self.data.indexed_iter() {
            result[idx] = !self.is_nodata(&val);
        }
        DataTile {
            tile: self.get_tile().to_owned(),
            data: result,
            nodata_value: false, // Maybe make nodata_value optional??
        }
    }

    pub fn data_cells_uint(&self) -> DataTile<u64> {
        // Returns a DataTile with u64 values where each 1 indicates the cell has valid data
        let mut result = Array2::<u64>::default((self.tile.ny as usize, self.tile.nx as usize));
        for (idx, val) in self.data.indexed_iter() {
            if self.is_nodata(&val) {
                result[idx] = 0;
            } else {
                result[idx] = 1;
            }
        }
        DataTile {
            tile: self.get_tile().to_owned(),
            data: result,
            nodata_value: u64::MAX,
        }
    }

    pub fn _empty_combined_tile(&self, other: &DataTile<T>, nodata_value: T) -> DataTile<T> {
        let tile = self.tile.combined_tile(&other.tile);
        let data = Array2::from_elem((tile.ny as usize, tile.nx as usize), nodata_value);
        DataTile {
            tile,
            data,
            nodata_value,
        }
    }

    pub fn crop(&self, crop_tile: &Tile, nodata_value: T) -> Result<DataTile<T>, String> {
        // Handle the case where the DataTile(self) and crop_tile only partially overlap by getting the overlap of the two
        // and using that as the crop_tile.
        let crop_tile = self.overlap(crop_tile)?;

        let mut new_data =
            Array2::from_elem((crop_tile.ny as usize, crop_tile.nx as usize), nodata_value);

        let (end_slice_col, start_slice_row) = self
            .grid_id_to_tile_id_xy(crop_tile.start_id.0, crop_tile.start_id.1)
            .unwrap();
        // Note: We first subtract one from nx and ny to get the id of the top left corner
        //       For this tile we do not get out of bounds of the tile.
        //       Because this is then used as the upper end of a slice we add the 1 back because
        //       slice ends are exclusive.
        let end_id = (
            crop_tile.start_id.0 + crop_tile.nx as i64 - 1,
            crop_tile.start_id.1 + crop_tile.ny as i64 - 1,
        );
        let (start_slice_col, end_slice_row) =
            self.grid_id_to_tile_id_xy(end_id.0, end_id.1).unwrap();
        let data_slice = &self.data.slice(s![
            start_slice_col as usize..(end_slice_col + 1) as usize,
            start_slice_row as usize..(end_slice_row + 1) as usize
        ]);
        let mut new_data_slice =
            new_data.slice_mut(s![0..crop_tile.ny as usize, 0..crop_tile.nx as usize,]);
        new_data_slice.assign(&data_slice);
        Ok(DataTile {
            tile: crop_tile,
            data: new_data,
            nodata_value: self.nodata_value,
        })
    }

    pub fn value(&self, i_x: i64, i_y: i64, nodata_value: T) -> T {
        let id_result = self.grid_id_to_tile_id_xy(i_x, i_y);
        match id_result {
            Ok((id_col, id_row)) => {
                // Note: indexing into array is in order y,x
                return self.data[Ix2(id_col as usize, id_row as usize)];
            }
            Err(e) => {
                // If id is out of bounds, set value to nodata_value
                return nodata_value;
            }
        }
    }

    pub fn values(&self, index: &ArrayView2<i64>, nodata_value: T) -> Array1<T> {
        // let tile_index = self.grid_id_to_tile_id(index, i64::MAX);
        let mut values = Array1::<T>::zeros(index.shape()[0]);
        for cell_id in 0..values.shape()[0] {
            values[Ix1(cell_id)] =
                self.value(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)], nodata_value);
        }
        values
    }

    pub fn linear_interpolation(&self, sample_points: &ArrayView2<f64>) -> Array1<f64> {
        let grid = self.get_grid();
        let nearby_cells = grid.cells_near_point(sample_points);

        let original_shape = nearby_cells.raw_dim();
        let raveled_shape = (original_shape[0] * original_shape[1], original_shape[2]);

        // Get values at nearby cells
        let nearby_cells = nearby_cells.into_shape(raveled_shape).unwrap();
        let nearby_values = self.values(&nearby_cells.view(), self.nodata_value);
        let nearby_values = nearby_values
            .into_shape((original_shape[0], original_shape[1]))
            .unwrap();

        // Get coordinates of nearby cells
        let nearby_centroids = grid.centroid(&nearby_cells.view());
        let nearby_centroids: Array3<f64> = nearby_centroids.into_shape(original_shape).unwrap();

        let mut values = Array1::<f64>::zeros(sample_points.shape()[0]);
        let nodata_value = self.nodata_value.to_f64().unwrap_or(f64::NAN);
        match grid {
            // FIXME: I think the interpolation logic should not live on the grid, but it is used by BoundedGrid implementation
            Grid::TriGrid(grid) => {
                Zip::from(&mut values)
                    .and(sample_points.axis_iter(Axis(0)))
                    .and(nearby_centroids.axis_iter(Axis(0)))
                    .and(nearby_values.axis_iter(Axis(0)))
                    .for_each(|new_val, point, val_locs, near_vals| {
                        let mut nodata_present: bool = false;
                        for val in near_vals {
                            if self.is_nodata(&val) {
                                *new_val = nodata_value;
                                nodata_present = true;
                                break;
                            }
                        }
                        // I wanted to break but it seems the foreach does not count as a loop
                        if nodata_present == false {
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
                            // Convert type T to f64. Linear interpolation does not make sense on integers and the like.
                            let near_vals = near_vals.mapv(|x| x.to_f64().unwrap_or(f64::NAN));
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
                        }
                    });
            }
            Grid::RectGrid(grid) => {
                let mut nearby_cells = grid.cells_near_point(sample_points);
                let mut sliced_ids = nearby_cells.slice_mut(s![.., 0, ..]);

                let tl_val = self.values(&sliced_ids.view(), self.nodata_value);
                sliced_ids = nearby_cells.slice_mut(s![.., 1, ..]);
                let tr_val = self.values(&sliced_ids.view(), self.nodata_value);
                sliced_ids = nearby_cells.slice_mut(s![.., 3, ..]);
                let br_val = self.values(&sliced_ids.view(), self.nodata_value);
                // Note: End with slice 2, which are the ids at bottom left.
                //       These are used again so we take these as the last slice.
                //       That way they don't get overwritten again.
                sliced_ids = nearby_cells.slice_mut(s![.., 2, ..]);
                let bl_val = self.values(&sliced_ids.view(), self.nodata_value);

                // Sliced_ids here should contain the bottom-left ids
                let mut abs_diff = sample_points - grid.centroid(&sliced_ids.view());
                if grid.rotation() != 0. {
                    for i in 0..abs_diff.shape()[0] {
                        let mut diff = abs_diff.slice_mut(s![i, ..]);
                        let diff_local = grid._rotation_matrix_inv.dot(&diff);
                        diff.assign(&diff_local);
                    }
                }
                let x_diff = &abs_diff.slice(s![.., 0]) / grid.dx();
                let y_diff = &abs_diff.slice(s![.., 1]) / grid.dy();

                for i in 0..sample_points.shape()[0] {
                    let is_nodata = self.is_nodata(&tl_val[Ix1(i)])
                        || self.is_nodata(&tr_val[Ix1(i)])
                        || self.is_nodata(&bl_val[Ix1(i)])
                        || self.is_nodata(&br_val[Ix1(i)]);
                    if is_nodata {
                        values[Ix1(i)] = to_f64(self.nodata_value);
                    } else {
                        // Should look like so:
                        // let top_val = &tl_val + (&tr_val - &tl_val) * &x_diff;
                        // let bot_val = &bl_val + (&br_val - &bl_val) * &x_diff;
                        // values = &bot_val + (&top_val - &bot_val) * &y_diff;
                        let top_val: f64 = to_f64(tl_val[Ix1(i)])
                            + (to_f64(tr_val[Ix1(i)]) - to_f64(tl_val[Ix1(i)])) * &x_diff[Ix1(i)];
                        let bot_val: f64 = to_f64(bl_val[Ix1(i)])
                            + (to_f64(br_val[Ix1(i)]) - to_f64(bl_val[Ix1(i)])) * &x_diff[Ix1(i)];
                        values[Ix1(i)] = &bot_val + (&top_val - &bot_val) * &y_diff[Ix1(i)];
                    }
                }
            }
            Grid::HexGrid(grid) => {
                let all_nearby_cells = grid.cells_near_point(sample_points); // (points, nearby_cells, xy)

                let original_shape = all_nearby_cells.raw_dim();
                let raveled_shape = (original_shape[0] * original_shape[1], original_shape[2]);

                // Get values at nearby cells
                let all_nearby_cells = all_nearby_cells.into_shape(raveled_shape).unwrap();
                let nearby_centroids = grid
                    .centroid(&all_nearby_cells.view())
                    .into_shape(original_shape)
                    .unwrap();
                let weights = crate::interpolate::linear_interp_weights_triangles(
                    &sample_points,
                    &nearby_centroids.view(),
                );

                let all_nearby_values = self
                    .values(&all_nearby_cells.view(), self.nodata_value)
                    .into_shape((original_shape[0], original_shape[1]))
                    .unwrap();
                for i in 0..values.shape()[0] {
                    let nearby_values_slice = all_nearby_values.slice(s![i, ..]);
                    let mut is_nodata = false;
                    for val in nearby_values_slice.iter() {
                        if *val == self.nodata_value {
                            is_nodata = true;
                            break;
                        }
                    }
                    if is_nodata {
                        values[Ix1(i)] = to_f64(self.nodata_value);
                    } else {
                        let mut weighted_sum: f64 = 0.;
                        for w in 0..(weights.shape()[1] as usize) {
                            weighted_sum +=
                                weights[Ix2(i, w)] * to_f64(nearby_values_slice[Ix1(w)]);
                        }
                        values[Ix1(i)] = weighted_sum;
                    }
                }
            }
        }
        values
    }

    pub fn inverse_distance_interpolation(
        &self,
        sample_points: &ArrayView2<f64>,
        decay_constant: f64,
    ) -> Array1<f64> {
        let grid = self.get_grid();
        let mut result = Array1::<f64>::zeros(sample_points.shape()[0]);
        // Nearby_cells shape: (nr_sample_points, nearby_cells, xy)
        let all_nearby_cells = grid.cells_near_point(sample_points);
        for i in 0..sample_points.shape()[0] {
            let nearby_cells = all_nearby_cells.slice(s![i, .., ..]);
            let nearby_values = self.values(&nearby_cells, self.nodata_value);
            let nearby_centroids = grid.centroid(&nearby_cells);
            let point_corner_vec = (nearby_centroids - sample_points.slice(s![i, ..]));
            let distances = point_corner_vec.map_axis(Axis(1), |row| {
                row.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
            });

            // TODO: allow for different weighting equations, sch as Shepard's interpolation with different power parameters
            let mut weights =
                (-(distances / decay_constant).map(|x| x.powi(2))).map(|x| E.powf(*x));
            weights = &weights / &weights.sum_axis(Axis(0));

            let mut result_val = 0.;
            for j in 0..weights.shape()[0] {
                let val = nearby_values[Ix1(j)];
                if val == self.nodata_value {
                    result_val = to_f64(self.nodata_value);
                    break;
                }
                result_val += weights[Ix1(j)] * to_f64(val);
            }
            result[Ix1(i)] = result_val;
        }
        result
    }

    pub fn _slice_tile_mut(&mut self, tile: &Tile) -> ArrayBase<ViewRepr<&mut T>, Dim<[usize; 2]>> {
        // ArrayView2<&mut f64> is translated to &&mut so I specify the ArrayBase syntax as return type instead.

        // Note: We first subtract one from nx and ny to get the id of the top left corner.
        //       For this cell we do not get out of bounds of the tile.
        //       Because this is then used as the upper end of a slice we add the 1 back because
        //       slice ends are exclusive.
        let (end_slice_col, start_slice_row) = self
            .grid_id_to_tile_id_xy(tile.start_id.0, tile.start_id.1)
            .unwrap();
        let (start_slice_col, end_slice_row) = self
            .grid_id_to_tile_id_xy(
                tile.start_id.0 + tile.nx as i64 - 1,
                tile.start_id.1 + tile.ny as i64 - 1,
            )
            .unwrap();
        let mut data_slice = self.data.slice_mut(s![
            start_slice_col as usize..(end_slice_col + 1) as usize,
            start_slice_row as usize..(end_slice_row + 1) as usize
        ]);
        data_slice
    }

    pub fn _assign_data_in_place(&mut self, data_tile: &DataTile<T>) -> () {
        let self_nodata_value = self.nodata_value; // Get nodata value while it is still allowed, before mutable borrowing self
        let mut data_slice = self._slice_tile_mut(data_tile.get_tile());
        for (self_val, new_val) in data_slice.iter_mut().zip(data_tile.data.iter()) {
            if data_tile.is_nodata(new_val) {
                // If new value is nodata value in it's original data tile, also make it nodata value of self here.
                *self_val = self_nodata_value;
            } else {
                *self_val = *new_val
            }
        }
    }

    pub fn powf(&self, exponent: f64) -> DataTile<f64> {
        let mut data = Array2::<f64>::zeros((self.tile.ny as usize, self.tile.nx as usize));
        let nodata_value = to_f64(self.nodata_value);
        for id_x in 0..self.tile.ny {
            for id_y in 0..self.tile.nx {
                let val = self.data[Ix2(id_y as usize, id_x as usize)];
                if self.is_nodata(&val) {
                    data[Ix2(id_y as usize, id_x as usize)] = nodata_value;
                } else {
                    data[Ix2(id_y as usize, id_x as usize)] = to_f64(val).powf(exponent);
                }
            }
        }
        DataTile {
            tile: self.tile.to_owned(),
            data: data,
            nodata_value: nodata_value,
        }
    }

    pub fn powf_reverse(&self, base: f64) -> DataTile<f64> {
        let mut data = Array2::<f64>::zeros((self.tile.ny as usize, self.tile.nx as usize));
        let nodata_value = to_f64(self.nodata_value);
        for id_x in 0..self.tile.ny {
            for id_y in 0..self.tile.nx {
                let val = self.data[Ix2(id_y as usize, id_x as usize)];
                if self.is_nodata(&val) {
                    data[Ix2(id_y as usize, id_x as usize)] = nodata_value;
                } else {
                    data[Ix2(id_y as usize, id_x as usize)] = base.powf(to_f64(val));
                }
            }
        }
        DataTile {
            tile: self.tile.to_owned(),
            data: data,
            nodata_value: nodata_value,
        }
    }

    pub fn powi(&self, exponent: i32) -> DataTile<f64> {
        let mut data = Array2::<f64>::zeros((self.tile.ny as usize, self.tile.nx as usize));
        let nodata_value = to_f64(self.nodata_value);
        for id_x in 0..self.tile.ny {
            for id_y in 0..self.tile.nx {
                let val = self.data[Ix2(id_y as usize, id_x as usize)];
                if self.is_nodata(&val) {
                    data[Ix2(id_y as usize, id_x as usize)] = nodata_value;
                } else {
                    data[Ix2(id_y as usize, id_x as usize)] = to_f64(val).powi(exponent);
                }
            }
        }
        DataTile {
            tile: self.tile.to_owned(),
            data: data,
            nodata_value: nodata_value,
        }
    }

    pub fn equals_value(&self, value: T) -> Array2<i64> {
        let mut index_vec = Vec::new();

        for id_y in 0..self.tile.ny {
            for id_x in 0..self.tile.nx {
                let tile_val = self.data[Ix2(id_y as usize, id_x as usize)];
                if tile_val == value {
                    let (grid_id_x, grid_id_y) = self
                        .tile
                        .tile_id_to_grid_id_xy(id_y as i64, id_x as i64)
                        .unwrap();
                    index_vec.push([grid_id_x, grid_id_y]);
                }
            }
        }

        // Convert the vector of results into an Array2
        let nr_matches = index_vec.len();
        let mut index = Array2::<i64>::zeros((nr_matches, 2));
        for (i, [x, y]) in index_vec.iter().enumerate() {
            index[(i, 0)] = *x;
            index[(i, 1)] = *y;
        }

        index
    }

    pub fn not_equals_value(&self, value: T) -> Array2<i64> {
        let mut index_vec = Vec::new();

        for id_y in 0..self.tile.ny {
            for id_x in 0..self.tile.nx {
                let tile_val = self.data[Ix2(id_y as usize, id_x as usize)];
                if tile_val != value && !self.is_nodata(&tile_val) {
                    let (grid_id_x, grid_id_y) = self
                        .tile
                        .tile_id_to_grid_id_xy(id_y as i64, id_x as i64)
                        .unwrap();
                    index_vec.push([grid_id_x, grid_id_y]);
                }
            }
        }

        // Convert the vector of results into an Array2
        let nr_matches = index_vec.len();
        let mut index = Array2::<i64>::zeros((nr_matches, 2));
        for (i, [x, y]) in index_vec.iter().enumerate() {
            index[(i, 0)] = *x;
            index[(i, 1)] = *y;
        }

        index
    }

    pub fn greater_than_value(&self, value: T) -> Array2<i64> {
        let mut index_vec = Vec::new();

        for id_y in 0..self.tile.ny {
            for id_x in 0..self.tile.nx {
                if self.data[Ix2(id_y as usize, id_x as usize)] > value {
                    let (grid_id_x, grid_id_y) = self
                        .tile
                        .tile_id_to_grid_id_xy(id_y as i64, id_x as i64)
                        .unwrap();
                    index_vec.push([grid_id_x, grid_id_y]);
                }
            }
        }

        // Convert the vector of results into an Array2
        let nr_matches = index_vec.len();
        let mut index = Array2::<i64>::zeros((nr_matches, 2));
        for (i, [x, y]) in index_vec.iter().enumerate() {
            index[(i, 0)] = *x;
            index[(i, 1)] = *y;
        }

        index
    }

    pub fn greater_equals_value(&self, value: T) -> Array2<i64> {
        let mut index_vec = Vec::new();

        for id_y in 0..self.tile.ny {
            for id_x in 0..self.tile.nx {
                if self.data[Ix2(id_y as usize, id_x as usize)] >= value {
                    let (grid_id_x, grid_id_y) = self
                        .tile
                        .tile_id_to_grid_id_xy(id_y as i64, id_x as i64)
                        .unwrap();
                    index_vec.push([grid_id_x, grid_id_y]);
                }
            }
        }

        // Convert the vector of results into an Array2
        let nr_matches = index_vec.len();
        let mut index = Array2::<i64>::zeros((nr_matches, 2));
        for (i, [x, y]) in index_vec.iter().enumerate() {
            index[(i, 0)] = *x;
            index[(i, 1)] = *y;
        }

        index
    }

    pub fn lower_than_value(&self, value: T) -> Array2<i64> {
        let mut index_vec = Vec::new();

        for id_y in 0..self.tile.ny {
            for id_x in 0..self.tile.nx {
                if self.data[Ix2(id_y as usize, id_x as usize)] < value {
                    let (grid_id_x, grid_id_y) = self
                        .tile
                        .tile_id_to_grid_id_xy(id_y as i64, id_x as i64)
                        .unwrap();
                    index_vec.push([grid_id_x, grid_id_y]);
                }
            }
        }

        // Convert the vector of results into an Array2
        let nr_matches = index_vec.len();
        let mut index = Array2::<i64>::zeros((nr_matches, 2));
        for (i, [x, y]) in index_vec.iter().enumerate() {
            index[(i, 0)] = *x;
            index[(i, 1)] = *y;
        }

        index
    }

    pub fn lower_equals_value(&self, value: T) -> Array2<i64> {
        let mut index_vec = Vec::new();

        for id_y in 0..self.tile.ny {
            for id_x in 0..self.tile.nx {
                if self.data[Ix2(id_y as usize, id_x as usize)] <= value {
                    let (grid_id_x, grid_id_y) = self
                        .tile
                        .tile_id_to_grid_id_xy(id_y as i64, id_x as i64)
                        .unwrap();
                    index_vec.push([grid_id_x, grid_id_y]);
                }
            }
        }

        // Convert the vector of results into an Array2
        let nr_matches = index_vec.len();
        let mut index = Array2::<i64>::zeros((nr_matches, 2));
        for (i, [x, y]) in index_vec.iter().enumerate() {
            index[(i, 0)] = *x;
            index[(i, 1)] = *y;
        }

        index
    }

    pub fn max(&self) -> T {
        let mut max_val = T::min_value();
        for val in self.data.iter() {
            if !self.is_nodata(&val) && *val > max_val {
                max_val = *val;
            }
        }
        if max_val == T::min_value() {
            max_val = self.nodata_value
        }
        max_val
    }

    pub fn min(&self) -> T {
        let mut min_val = T::max_value();
        for val in self.data.iter() {
            if !self.is_nodata(&val) && *val < min_val {
                min_val = *val;
            }
        }
        if min_val == T::max_value() {
            min_val = self.nodata_value
        }
        min_val
    }

    pub fn sum(&self) -> T {
        let mut summed = T::zero();
        for val in self.data.iter() {
            if !self.is_nodata(&val) {
                summed = summed + *val;
            }
        }
        summed
    }

    pub fn mean(&self) -> f64 {
        let mut summed = T::zero();
        let mut nr_values_with_data = 0;
        for val in self.data.iter() {
            if !self.is_nodata(&val) {
                summed = summed + *val;
                nr_values_with_data += 1;
            }
        }
        to_f64(summed) / to_f64(nr_values_with_data)
    }

    pub fn median(&self) -> f64 {
        self.percentile(50.).unwrap()
    }

    pub fn percentile(&self, percentile: f64) -> Result<f64, String> {
        // Validate percentile input
        if percentile < 0.0 || percentile > 100.0 {
            let error_message = format!(
                "Percentile value needs to be between 0 and 100, got {}",
                percentile
            )
            .to_string();
            return Err(error_message);
        }

        let mut sorted: Vec<T> = self.data.iter().cloned().collect();
        sorted.retain(|&x| !self.is_nodata(&x));
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        if n == 0 {
            return Err("Cannot find percentile of an empty array".to_string());
        }

        // Calculate the fractional index for the desired percentile
        let rank = (percentile / 100.0) * (n as f64 - 1.0);
        let lower_index = rank.floor() as usize;
        let upper_index = rank.ceil() as usize;

        // Interpolate if needed
        if lower_index == upper_index {
            return Ok(to_f64(sorted[lower_index])); // Exact match at an integer index
        }

        let weight_upper = rank - lower_index as f64;
        let weight_lower = 1.0 - weight_upper;
        Ok(to_f64(sorted[lower_index]) * weight_lower + to_f64(sorted[upper_index]) * weight_upper)
    }

    pub fn std(&self) -> f64 {
        let mean = self.mean();

        // Compute the squared differences from the mean
        let filtered = self.data.iter().filter(|&&x| !self.is_nodata(&x));
        let variance_sum: f64 = filtered
            .clone()
            .map(|&x| (x.to_f64().unwrap_or(f64::NAN) - mean).powi(2))
            .sum();
        let variance = variance_sum / filtered.count() as f64;

        variance.sqrt() // Return the square root of the variance
    }
}

/// Newtype wrapper for `T`, allowing `T + DataTile<T>` operations.
#[derive(Clone, Copy)]
pub struct Scalar<T>(pub T);

impl<T> Add<T> for &DataTile<T>
where
    T: Num
        + Clone
        + Copy
        + PartialEq
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + PartialOrd
        + NumCast
        + Add<Output = T>,
{
    type Output = DataTile<T>;

    fn add(self, scalar: T) -> DataTile<T> {
        let mut data = self.data.to_owned();
        for val in data.iter_mut() {
            if !self.is_nodata(&val) {
                *val = *val + scalar;
            }
        }
        DataTile {
            tile: self.tile.clone(),
            data,
            nodata_value: self.nodata_value,
        }
    }
}

impl<T> Add<&DataTile<T>> for Scalar<T>
where
    T: Num
        + Clone
        + Copy
        + PartialEq
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + PartialOrd
        + NumCast
        + Add<Output = T>,
{
    type Output = DataTile<T>;

    fn add(self, data_tile: &DataTile<T>) -> DataTile<T> {
        let mut data = data_tile.data.to_owned();
        for val in data.iter_mut() {
            if *val == data_tile.nodata_value {
                *val = data_tile.nodata_value;
            } else {
                *val = self.0 + *val;
            }
        }
        DataTile {
            tile: data_tile.tile.clone(),
            data,
            nodata_value: data_tile.nodata_value,
        }
    }
}

impl<
        T: Num
            + Clone
            + Copy
            + PartialEq
            + Bounded
            + ToPrimitive
            + FromPrimitive
            + PartialOrd
            + NumCast,
    > Add<&DataTile<T>> for &DataTile<T>
{
    type Output = DataTile<T>;

    fn add(self, other: &DataTile<T>) -> DataTile<T> {
        // Create full span DataTile
        let mut out_data_tile = self._empty_combined_tile(other, self.nodata_value);

        let _ = out_data_tile._assign_data_in_place(self);
        let _ = out_data_tile._assign_data_in_place(other);

        // Insert added overlap between self and other
        if self.intersects(&other.tile) {
            // Note that it should unwrap the crop() calls here. Since we already checked if the tiles overlap,
            // crop() should not return an error. If it does, something more fundamental is wrong.
            let overlap_self = self.crop(&other.tile, self.nodata_value).unwrap();
            let overlap_other = other.crop(&self.tile, self.nodata_value).unwrap();

            // Calculate overlapping values taking nodata values into account
            // The nodata value will be taken from self and not other.
            // If the the value of the element in question is nodata for either self or other,
            // the result is set to be nodata value of self.
            let mut overlap_data = Array::zeros(overlap_self.data.raw_dim());
            Zip::from(&mut overlap_data)
                .and(&overlap_self.data)
                .and(&overlap_other.data)
                .for_each(|result_val, &self_val, &other_val| {
                    if self.is_nodata(&self_val) || other.is_nodata(&other_val) {
                        *result_val = self.nodata_value;
                    } else {
                        *result_val = self_val + other_val;
                    }
                });

            let overlap_tile = overlap_self.tile; // Might as well have taken overlap_other.tile, they should be the same
            let overlap_data_tile = DataTile {
                tile: overlap_tile,
                data: overlap_data,
                nodata_value: self.nodata_value,
            };
            let _ = out_data_tile._assign_data_in_place(&overlap_data_tile);
        }
        out_data_tile
    }
}

impl<T> Sub<T> for &DataTile<T>
where
    T: Num
        + Clone
        + Copy
        + PartialEq
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + PartialOrd
        + NumCast
        + Sub<Output = T>,
{
    type Output = DataTile<T>;

    fn sub(self, scalar: T) -> DataTile<T> {
        let mut data = self.data.to_owned();
        for val in data.iter_mut() {
            if !self.is_nodata(&val) {
                *val = *val - scalar;
            }
        }
        DataTile {
            tile: self.tile.clone(),
            data,
            nodata_value: self.nodata_value,
        }
    }
}

impl<T> Sub<&DataTile<T>> for Scalar<T>
where
    T: Num
        + Clone
        + Copy
        + PartialEq
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + PartialOrd
        + NumCast
        + Sub<Output = T>,
{
    type Output = DataTile<T>;

    fn sub(self, data_tile: &DataTile<T>) -> DataTile<T> {
        let mut data = data_tile.data.to_owned();
        for val in data.iter_mut() {
            if *val == data_tile.nodata_value {
                *val = data_tile.nodata_value;
            } else {
                *val = self.0 - *val;
            }
        }
        DataTile {
            tile: data_tile.tile.clone(),
            data,
            nodata_value: data_tile.nodata_value,
        }
    }
}

impl<
        T: Num
            + Clone
            + Copy
            + PartialEq
            + Bounded
            + ToPrimitive
            + FromPrimitive
            + PartialOrd
            + NumCast,
    > Sub<&DataTile<T>> for &DataTile<T>
{
    type Output = DataTile<T>;

    fn sub(self, other: &DataTile<T>) -> DataTile<T> {
        // Create full span DataTile
        let mut out_data_tile = self._empty_combined_tile(other, self.nodata_value);

        let _ = out_data_tile._assign_data_in_place(self);
        let _ = out_data_tile._assign_data_in_place(other);

        // Insert added overlap between self and other
        if self.intersects(&other.tile) {
            // Note that it should unwrap the crop() calls here. Since we already checked if the tiles overlap,
            // crop() should not return an error. If it does, something more fundamental is wrong.
            let overlap_self = self.crop(&other.tile, self.nodata_value).unwrap();
            let overlap_other = other.crop(&self.tile, self.nodata_value).unwrap();

            // Calculate overlapping values taking nodata values into account
            // The nodata value will be taken from self and not other.
            // If the the value of the element in question is nodata for either self or other,
            // the result is set to be nodata value of self.
            let mut overlap_data = Array::zeros(overlap_self.data.raw_dim());
            Zip::from(&mut overlap_data)
                .and(&overlap_self.data)
                .and(&overlap_other.data)
                .for_each(|result_val, &self_val, &other_val| {
                    if self.is_nodata(&self_val) || other.is_nodata(&other_val) {
                        *result_val = self.nodata_value;
                    } else {
                        *result_val = self_val - other_val;
                    }
                });

            let overlap_tile = overlap_self.tile; // Might as well have taken overlap_other.tile, they should be the same
            let overlap_data_tile = DataTile {
                tile: overlap_tile,
                data: overlap_data,
                nodata_value: self.nodata_value,
            };
            let _ = out_data_tile._assign_data_in_place(&overlap_data_tile);
        }
        out_data_tile
    }
}

impl<T> Mul<&DataTile<T>> for Scalar<T>
where
    T: Num
        + Clone
        + Copy
        + PartialEq
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + PartialOrd
        + NumCast
        + Mul<Output = T>,
{
    type Output = DataTile<T>;

    fn mul(self, data_tile: &DataTile<T>) -> DataTile<T> {
        let mut data = data_tile.data.to_owned();
        for val in data.iter_mut() {
            if *val == data_tile.nodata_value {
                *val = data_tile.nodata_value;
            } else {
                *val = self.0 * *val;
            }
        }
        DataTile {
            tile: data_tile.tile.clone(),
            data,
            nodata_value: data_tile.nodata_value,
        }
    }
}

impl<T> Mul<T> for &DataTile<T>
where
    T: Num
        + Clone
        + Copy
        + PartialEq
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + PartialOrd
        + NumCast
        + Add<Output = T>,
{
    type Output = DataTile<T>;

    fn mul(self, scalar: T) -> DataTile<T> {
        let mut data = self.data.to_owned();
        for val in data.iter_mut() {
            if self.is_nodata(&val) {
                *val = self.nodata_value;
            } else {
                *val = *val * scalar
            }
        }
        DataTile {
            tile: self.tile.clone(),
            data: data,
            nodata_value: self.nodata_value,
        }
    }
}

impl<
        T: Num
            + Clone
            + Copy
            + PartialEq
            + Bounded
            + ToPrimitive
            + FromPrimitive
            + PartialOrd
            + NumCast,
    > Mul<&DataTile<T>> for &DataTile<T>
{
    type Output = DataTile<T>;

    fn mul(self, other: &DataTile<T>) -> DataTile<T> {
        // Create full span DataTile
        let mut out_data_tile = self._empty_combined_tile(&other, self.nodata_value);

        let _ = out_data_tile._assign_data_in_place(self);
        let _ = out_data_tile._assign_data_in_place(other);

        // Insert added overlap between self and other
        if self.intersects(&other.tile) {
            // Note that it should unwrap the crop() calls here. Since we already checked if the tiles overlap,
            // crop() should not return an error. If it does, something more fundamental is wrong.
            let overlap_self = self.crop(&other.tile, self.nodata_value).unwrap();
            let overlap_other = other.crop(&self.tile, self.nodata_value).unwrap();

            // Calculate overlapping values taking nodata values into account
            // The nodata value will be taken from self and not other.
            // If the the value of the element in question is nodata for either self or other,
            // the result is set to be nodata value of self.
            let mut overlap_data = Array::zeros(overlap_self.data.raw_dim());
            Zip::from(&mut overlap_data)
                .and(&overlap_self.data)
                .and(&overlap_other.data)
                .for_each(|result_val, &self_val, &other_val| {
                    if self.is_nodata(&self_val) || other.is_nodata(&other_val) {
                        *result_val = self.nodata_value;
                    } else {
                        *result_val = self_val * other_val;
                    }
                });

            let overlap_tile = overlap_self.tile; // Might as well have taken overlap_other.tile, they should be the same
            let overlap_data_tile = DataTile {
                tile: overlap_tile,
                data: overlap_data,
                nodata_value: self.nodata_value,
            };
            let _ = out_data_tile._assign_data_in_place(&overlap_data_tile);
        }
        out_data_tile
    }
}

impl<T> Div<&DataTile<T>> for Scalar<T>
where
    T: Num
        + Clone
        + Copy
        + PartialEq
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + PartialOrd
        + NumCast
        + Div<Output = T>,
{
    type Output = DataTile<f64>;

    fn div(self, data_tile: &DataTile<T>) -> DataTile<f64> {
        let mut data =
            Array2::<f64>::default((data_tile.data.shape()[0], data_tile.data.shape()[1]));

        for (new_val, orig_val) in data.iter_mut().zip(data_tile.data.iter()) {
            if data_tile.is_nodata(orig_val) {
                *new_val = f64::NAN;
            } else {
                *new_val = to_f64(self.0) / to_f64(*orig_val);
            }
        }
        DataTile::<f64> {
            tile: data_tile.tile.clone(),
            data,
            nodata_value: f64::NAN,
        }
    }
}

impl<T> Div<T> for &DataTile<T>
where
    T: Num
        + Clone
        + Copy
        + PartialEq
        + Bounded
        + ToPrimitive
        + FromPrimitive
        + PartialOrd
        + NumCast
        + Div<Output = T>,
{
    type Output = DataTile<f64>;

    fn div(self, scalar: T) -> DataTile<f64> {
        let mut data = Array2::<f64>::default((self.data.shape()[0], self.data.shape()[1]));
        for (new_val, orig_val) in data.iter_mut().zip(&self.data) {
            if self.is_nodata(orig_val) {
                *new_val = f64::NAN;
            } else {
                *new_val = to_f64(*orig_val) / to_f64(scalar)
            }
        }
        DataTile {
            tile: self.tile.clone(),
            data: data,
            nodata_value: f64::NAN,
        }
    }
}

impl<
        T: Num
            + Clone
            + Copy
            + PartialEq
            + Bounded
            + ToPrimitive
            + FromPrimitive
            + AsPrimitive<f64>
            + PartialOrd
            + NumCast,
    > Div<&DataTile<T>> for &DataTile<T>
{
    type Output = DataTile<f64>;

    fn div(self, other: &DataTile<T>) -> DataTile<f64> {
        // Create full span DataTile
        let tile = self.tile.combined_tile(&other.tile);
        let nodata_value = f64::NAN;
        let data = Array2::<f64>::from_elem((tile.ny as usize, tile.nx as usize), nodata_value);
        let mut out_data_tile = DataTile {
            tile,
            data,
            nodata_value,
        };
        let mut other: DataTile<f64> = other.into_dtype::<f64>();
        other.set_nodata_value(f64::NAN);
        let self_f64: DataTile<f64> = self.into_dtype::<f64>();
        let _ = out_data_tile._assign_data_in_place(&other);
        let _ = out_data_tile._assign_data_in_place(&self_f64);

        // Insert added overlap between self and other
        if self.intersects(&other.tile) {
            // Note that it should unwrap the crop() calls here. Since we already checked if the tiles overlap,
            // crop() should not return an error. If it does, something more fundamental is wrong.
            let overlap_self = self.crop(&other.tile, self.nodata_value).unwrap();
            let overlap_other = other.crop(&self.tile, other.nodata_value).unwrap();

            // Calculate overlapping values taking nodata values into account
            // The nodata value will be taken from self and not other.
            // If the the value of the element in question is nodata for either self or other,
            // the result is set to be nodata value of self.
            let mut overlap_data = Array::zeros(overlap_self.data.raw_dim());
            Zip::from(&mut overlap_data)
                .and(&overlap_self.data)
                .and(&overlap_other.data)
                .for_each(|result_val, &self_val, &other_val| {
                    if self.is_nodata(&self_val) || other.is_nodata(&other_val) {
                        *result_val = f64::NAN;
                    } else {
                        *result_val = to_f64(self_val) / to_f64(other_val);
                    }
                });

            let overlap_tile = overlap_self.tile; // Might as well have taken overlap_other.tile, they should be the same
            let overlap_data_tile = DataTile {
                tile: overlap_tile,
                data: overlap_data,
                nodata_value: f64::NAN,
            };
            let _ = out_data_tile._assign_data_in_place(&overlap_data_tile);
        }
        out_data_tile
    }
}

impl<
        T: Num
            + Clone
            + Copy
            + PartialEq
            + Bounded
            + ToPrimitive
            + FromPrimitive
            + PartialOrd
            + NumCast,
    > Index<(usize, usize)> for DataTile<T>
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[(index.0, index.1)]
    }
}

impl<
        T: Num
            + Clone
            + Copy
            + PartialEq
            + Bounded
            + ToPrimitive
            + FromPrimitive
            + PartialOrd
            + NumCast,
    > IndexMut<(usize, usize)> for DataTile<T>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[(index.0, index.1)]
    }
}
