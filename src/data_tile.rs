use crate::grid::*;
use crate::tile::*;
use core::f64;
use numpy::ndarray::*;
use std::any::Any;
use std::f64::consts::E;
use std::f64::NAN;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

#[derive(Clone)]
pub struct DataTile {
    pub tile: Tile,
    pub data: Array2<f64>,
    pub nodata_value: f64,
}

impl TileTraits for DataTile {
    fn get_tile(&self) -> &Tile {
        &self.tile
    }

    fn get_grid(&self) -> &Grid {
        &self.tile.grid
    }
}

impl DataTile {
    pub fn new(
        grid: Grid,
        start_id: (i64, i64),
        nx: u64,
        ny: u64,
        data: Array2<f64>,
        nodata_value: f64,
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

    pub fn set_nodata_value(&mut self, nodata_value: f64) {
        for val in self.data.iter_mut() {
            if *val == self.nodata_value {
                *val = nodata_value;
            }
        }
        self.nodata_value = nodata_value;
    }

    pub fn is_nodata(&self, value: f64) -> bool {
        if f64::is_nan(self.nodata_value) {
            return f64::is_nan(value);
        }
        return value == self.nodata_value;
    }

    pub fn _empty_combined_tile(&self, other: &DataTile, nodata_value: f64) -> DataTile {
        let tile = self.tile.combined_tile(&other.tile);
        let data = Array2::from_elem((tile.ny as usize, tile.nx as usize), nodata_value);
        DataTile {
            tile,
            data,
            nodata_value,
        }
    }

    pub fn crop(&self, crop_tile: &Tile, nodata_value: f64) -> Result<DataTile, String> {
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

    pub fn value(&self, i_x: i64, i_y: i64, nodata_value: f64) -> f64 {
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

    pub fn values(&self, index: &ArrayView2<i64>, nodata_value: f64) -> Array1<f64> {
        // let tile_index = self.grid_id_to_tile_id(index, i64::MAX);
        let mut values = Array1::<f64>::zeros(index.shape()[0]);
        for cell_id in 0..values.shape()[0] {
            values[Ix1(cell_id)] =
                self.value(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)], nodata_value);
            // println!("Indexing in {},{} gives value {}", tile_index[Ix2(cell_id, 0)], tile_index[Ix2(cell_id, 1)], values[Ix1(cell_id)])
        }
        println!("{:?}", self.data);
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
        let nearby_values: Array2<f64> = nearby_values
            .into_shape((original_shape[0], original_shape[1]))
            .unwrap();

        // Get coordinates of nearby cells
        let nearby_centroids = grid.centroid(&nearby_cells.view());
        let nearby_centroids: Array3<f64> = nearby_centroids.into_shape(original_shape).unwrap();

        let mut values = Array1::<f64>::zeros(sample_points.shape()[0]);
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
                            if *val == self.nodata_value {
                                *new_val = self.nodata_value;
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

                let top_val = &tl_val + (&tr_val - &tl_val) * &x_diff;
                let bot_val = &bl_val + (&br_val - &bl_val) * &x_diff;
                values = &bot_val + (&top_val - &bot_val) * &y_diff;

                for i in 0..sample_points.shape()[0] {
                    let is_nodata = tl_val[Ix1(i)] == self.nodata_value
                        || tr_val[Ix1(i)] == self.nodata_value
                        || bl_val[Ix1(i)] == self.nodata_value
                        || br_val[Ix1(i)] == self.nodata_value;
                    if is_nodata {
                        values[Ix1(i)] = self.nodata_value;
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
                        values[Ix1(i)] = self.nodata_value;
                    } else {
                        values[Ix1(i)] = (&weights.slice(s![i, ..]) * &nearby_values_slice).sum();
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
                    result_val = self.nodata_value;
                    break;
                }
                result_val += weights[Ix1(j)] * val;
            }
            result[Ix1(i)] = result_val;
        }
        result
    }

    pub fn _slice_tile_mut(
        &mut self,
        tile: &Tile,
    ) -> ArrayBase<ViewRepr<&mut f64>, Dim<[usize; 2]>> {
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

    pub fn _assign_data_in_place(&mut self, data_tile: &DataTile) -> () {
        let mut data_slice = self._slice_tile_mut(data_tile.get_tile());
        data_slice.assign(&data_tile.data);
    }

    pub fn powf(&self, exponent: f64) -> DataTile {
        let mut new_tile = self.to_owned();
        new_tile.data.mapv_inplace(|val| {
            if val == self.nodata_value {
                val
            } else {
                val.powf(exponent)
            }
        });
        new_tile
    }

    pub fn powf_reverse(&self, base: f64) -> DataTile {
        let mut new_tile = self.to_owned();
        new_tile.data.mapv_inplace(|val| {
            if val == self.nodata_value {
                val
            } else {
                base.powf(val)
            }
        });
        new_tile
    }

    pub fn powi(&self, exponent: i32) -> DataTile {
        let mut new_tile = self.to_owned();
        new_tile.data.mapv_inplace(|val| {
            if val == self.nodata_value {
                val
            } else {
                val.powi(exponent)
            }
        });
        new_tile
    }

    pub fn equals_value(&self, value: f64) -> Array2<i64> {
        let mut index_vec = Vec::new();

        for id_y in 0..self.tile.ny {
            for id_x in 0..self.tile.nx {
                let tile_val = self.data[Ix2(id_y as usize, id_x as usize)];
                if tile_val == value || (f64::is_nan(tile_val) && f64::is_nan(value)) {
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

    pub fn not_equals_value(&self, value: f64) -> Array2<i64> {
        let mut index_vec = Vec::new();

        for id_y in 0..self.tile.ny {
            for id_x in 0..self.tile.nx {
                let tile_val = self.data[Ix2(id_y as usize, id_x as usize)];
                if tile_val != value && !(f64::is_nan(tile_val) && f64::is_nan(value)) {
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

    pub fn greater_than_value(&self, value: f64) -> Array2<i64> {
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

    pub fn greater_equals_value(&self, value: f64) -> Array2<i64> {
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

    pub fn lower_than_value(&self, value: f64) -> Array2<i64> {
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

    pub fn lower_equals_value(&self, value: f64) -> Array2<i64> {
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

    pub fn max(&self) -> f64 {
        let mut max_val = f64::NEG_INFINITY;
        for val in self.data.iter() {
            if *val != self.nodata_value && *val > max_val {
                max_val = *val;
            }
        }
        if max_val == f64::NEG_INFINITY {
            max_val = self.nodata_value
        }
        max_val
    }

    pub fn min(&self) -> f64 {
        let mut min_val = f64::INFINITY;
        for val in self.data.iter() {
            if *val != self.nodata_value && *val < min_val {
                min_val = *val;
            }
        }
        if min_val == f64::INFINITY {
            min_val = self.nodata_value
        }
        min_val
    }

    pub fn sum(&self) -> f64 {
        let mut summed = 0.;
        for val in self.data.iter() {
            if *val != self.nodata_value {
                summed += *val;
            }
        }
        summed
    }

    pub fn mean(&self) -> f64 {
        let mut summed = 0.;
        let mut nr_values_with_data = 0;
        for val in self.data.iter() {
            if *val != self.nodata_value {
                summed += *val;
                nr_values_with_data += 1;
            }
        }
        summed / nr_values_with_data as f64
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

        let mut sorted: Vec<f64> = self.data.iter().cloned().collect();
        sorted.retain(|&x| x != self.nodata_value);
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
            return Ok(sorted[lower_index]); // Exact match at an integer index
        }

        let weight_upper = rank - lower_index as f64;
        let weight_lower = 1.0 - weight_upper;
        Ok(sorted[lower_index] * weight_lower + sorted[upper_index] * weight_upper)
    }

    pub fn std(&self) -> f64 {
        let mean = self.mean();

        // Compute the squared differences from the mean
        let filtered = self.data.iter().filter(|&&x| x != self.nodata_value);
        let variance_sum: f64 = filtered.clone().map(|&x| (x - mean).powi(2)).sum();
        let variance = variance_sum / filtered.count() as f64;

        variance.sqrt() // Return the square root of the variance
    }
}

impl Add<DataTile> for f64 {
    type Output = DataTile;

    fn add(self, data_tile: DataTile) -> DataTile {
        let mut data = data_tile.data.to_owned();
        for val in data.iter_mut() {
            if *val == data_tile.nodata_value {
                *val = data_tile.nodata_value;
            } else {
                *val = self + *val;
            }
        }
        DataTile {
            tile: data_tile.tile,
            data: data,
            nodata_value: data_tile.nodata_value,
        }
    }
}

impl Add<f64> for DataTile {
    type Output = DataTile;

    fn add(self, scalar: f64) -> DataTile {
        let mut data = self.data.to_owned();
        for val in data.iter_mut() {
            if *val == self.nodata_value {
                *val = self.nodata_value;
            } else {
                *val = *val + scalar
            }
        }
        DataTile {
            tile: self.tile,
            data: data,
            nodata_value: self.nodata_value,
        }
    }
}

impl Add<DataTile> for DataTile {
    type Output = DataTile;

    fn add(self, other: DataTile) -> DataTile {
        // Create full span DataTile
        let mut out_data_tile = self._empty_combined_tile(&other, self.nodata_value);

        let _ = out_data_tile._assign_data_in_place(&self);
        let _ = out_data_tile._assign_data_in_place(&other);

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
                    if self_val == self.nodata_value || other_val == other.nodata_value {
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

impl Sub<DataTile> for f64 {
    type Output = DataTile;

    fn sub(self, data_tile: DataTile) -> DataTile {
        let mut data = data_tile.data.to_owned();
        for val in data.iter_mut() {
            if *val == data_tile.nodata_value {
                *val = data_tile.nodata_value;
            } else {
                *val = self - *val;
            }
        }
        DataTile {
            tile: data_tile.tile,
            data: data,
            nodata_value: data_tile.nodata_value,
        }
    }
}

impl Sub<f64> for DataTile {
    type Output = DataTile;

    fn sub(self, scalar: f64) -> DataTile {
        let mut data = self.data.to_owned();
        for val in data.iter_mut() {
            if *val == self.nodata_value {
                *val = self.nodata_value;
            } else {
                *val = *val - scalar
            }
        }
        DataTile {
            tile: self.tile,
            data: data,
            nodata_value: self.nodata_value,
        }
    }
}

impl Sub<DataTile> for DataTile {
    type Output = DataTile;

    fn sub(self, other: DataTile) -> DataTile {
        // Create full span DataTile
        let mut out_data_tile = self._empty_combined_tile(&other, self.nodata_value);

        let _ = out_data_tile._assign_data_in_place(&self);
        let _ = out_data_tile._assign_data_in_place(&other);

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
                    if self_val == self.nodata_value || other_val == other.nodata_value {
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

impl Mul<DataTile> for f64 {
    type Output = DataTile;

    fn mul(self, data_tile: DataTile) -> DataTile {
        let mut data = data_tile.data.to_owned();
        for val in data.iter_mut() {
            if *val == data_tile.nodata_value {
                *val = data_tile.nodata_value;
            } else {
                *val = self * *val;
            }
        }
        DataTile {
            tile: data_tile.tile,
            data: data,
            nodata_value: data_tile.nodata_value,
        }
    }
}

impl Mul<f64> for DataTile {
    type Output = DataTile;

    fn mul(self, scalar: f64) -> DataTile {
        let mut data = self.data.to_owned();
        for val in data.iter_mut() {
            if *val == self.nodata_value {
                *val = self.nodata_value;
            } else {
                *val = *val * scalar
            }
        }
        DataTile {
            tile: self.tile,
            data: data,
            nodata_value: self.nodata_value,
        }
    }
}

impl Mul<DataTile> for DataTile {
    type Output = DataTile;

    fn mul(self, other: DataTile) -> DataTile {
        // Create full span DataTile
        let mut out_data_tile = self._empty_combined_tile(&other, self.nodata_value);

        let _ = out_data_tile._assign_data_in_place(&self);
        let _ = out_data_tile._assign_data_in_place(&other);

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
                    if self_val == self.nodata_value || other_val == other.nodata_value {
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

impl Div<DataTile> for f64 {
    type Output = DataTile;

    fn div(self, data_tile: DataTile) -> DataTile {
        let mut data = data_tile.data.to_owned();
        for val in data.iter_mut() {
            if *val == data_tile.nodata_value {
                *val = data_tile.nodata_value;
            } else {
                *val = self / *val;
            }
        }
        DataTile {
            tile: data_tile.tile,
            data: data,
            nodata_value: data_tile.nodata_value,
        }
    }
}

impl Div<f64> for DataTile {
    type Output = DataTile;

    fn div(self, scalar: f64) -> DataTile {
        let mut data = self.data.to_owned();
        for val in data.iter_mut() {
            if *val == self.nodata_value {
                *val = self.nodata_value;
            } else {
                *val = *val / scalar
            }
        }
        DataTile {
            tile: self.tile,
            data: data,
            nodata_value: self.nodata_value,
        }
    }
}

impl Div<DataTile> for DataTile {
    type Output = DataTile;

    fn div(self, other: DataTile) -> DataTile {
        // Create full span DataTile
        let mut out_data_tile = self._empty_combined_tile(&other, self.nodata_value);
        let _ = out_data_tile._assign_data_in_place(&other);
        let _ = out_data_tile._assign_data_in_place(&self);

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
                    if self_val == self.nodata_value || other_val == other.nodata_value {
                        *result_val = self.nodata_value;
                    } else {
                        *result_val = self_val / other_val;
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

impl Index<(usize, usize)> for DataTile {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[(index.0, index.1)]
    }
}

impl IndexMut<(usize, usize)> for DataTile {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[(index.0, index.1)]
    }
}
