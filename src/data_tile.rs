use crate::grid::*;
use crate::tile::*;
use core::f64;
use numpy::ndarray::*;
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

        let (start_slice_x, end_slice_y) = self
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
        let (end_slice_x, start_slice_y) = self.grid_id_to_tile_id_xy(end_id.0, end_id.1).unwrap();
        let data_slice = &self.data.slice(s![
            start_slice_y as usize..(end_slice_y + 1) as usize,
            start_slice_x as usize..(end_slice_x + 1) as usize
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

    pub fn value(&self, index: &ArrayView2<i64>, nodata_value: f64) -> Array1<f64> {
        let mut values = Array1::<f64>::zeros(index.shape()[0]);
        for cell_id in 0..values.shape()[0] {
            let id_result =
                self.grid_id_to_tile_id_xy(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
            match id_result {
                Ok((id_x, id_y)) => {
                    // Note: indexing into array is in order y,x
                    values[Ix1(cell_id)] = self.data[Ix2(id_y as usize, id_x as usize)];
                }
                Err(e) => {
                    // If id is out of bounds, set value to nodata_value
                    values[Ix1(cell_id)] = nodata_value;
                }
            }
        }
        values
    }

    fn _assign_data_in_place(&mut self, data_tile: &DataTile) -> Result<(), String> {
        // Note: We first subtract one from nx and ny to get the id of the top left corner
        //       For this tile we do not get out of bounds of the tile.
        //       Because this is then used as the upper end of a slice we add the 1 back because
        //       slice ends are exclusive.
        let (start_slice_x, end_slice_y) = self
            .grid_id_to_tile_id_xy(data_tile.tile.start_id.0, data_tile.tile.start_id.1)
            .unwrap();
        let (end_slice_x, start_slice_y) = self
            .grid_id_to_tile_id_xy(
                data_tile.tile.start_id.0 + data_tile.tile.nx as i64 - 1,
                data_tile.tile.start_id.1 + data_tile.tile.ny as i64 - 1,
            )
            .unwrap();
        let mut data_slice = self.data.slice_mut(s![
            start_slice_y as usize..(end_slice_y + 1) as usize,
            start_slice_x as usize..(end_slice_x + 1) as usize
        ]);
        data_slice.assign(&data_tile.data);
        Ok(())
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
                if self.data[Ix2(id_y as usize, id_x as usize)] == value {
                    let (grid_id_x, grid_id_y) = self
                        .tile
                        .tile_id_to_grid_id_xy(id_x as i64, id_y as i64)
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
                if self.data[Ix2(id_y as usize, id_x as usize)] != value {
                    let (grid_id_x, grid_id_y) = self
                        .tile
                        .tile_id_to_grid_id_xy(id_x as i64, id_y as i64)
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
                        .tile_id_to_grid_id_xy(id_x as i64, id_y as i64)
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
                        .tile_id_to_grid_id_xy(id_x as i64, id_y as i64)
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
                        .tile_id_to_grid_id_xy(id_x as i64, id_y as i64)
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
                        .tile_id_to_grid_id_xy(id_x as i64, id_y as i64)
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

        // // Compute the squared differences from the mean
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
