use crate::grid::*;
use crate::tile::*;
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
}

impl Add<f64> for DataTile {
    type Output = DataTile;

    fn add(self, scalar: f64) -> DataTile {
        DataTile {
            tile: self.tile,
            data: &self.data + scalar,
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
            let overlap_data = overlap_self.data + overlap_other.data;
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

impl Sub<f64> for DataTile {
    type Output = DataTile;

    fn sub(self, scalar: f64) -> DataTile {
        DataTile {
            tile: self.tile,
            data: &self.data - scalar,
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
            let overlap_data = overlap_self.data - overlap_other.data;
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

impl Mul<f64> for DataTile {
    type Output = DataTile;

    fn mul(self, scalar: f64) -> DataTile {
        DataTile {
            tile: self.tile,
            data: &self.data * scalar,
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
            let overlap_data = overlap_self.data * overlap_other.data;
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

impl Div<f64> for DataTile {
    type Output = DataTile;

    fn div(self, scalar: f64) -> DataTile {
        DataTile {
            tile: self.tile,
            data: &self.data / scalar,
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
            let overlap_data = overlap_self.data / overlap_other.data;
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
