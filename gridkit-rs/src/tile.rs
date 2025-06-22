use num_traits::{Bounded, FromPrimitive, Num, ToPrimitive, NumCast};
use std::{f32::MAX, f64, i64, u64};

use crate::{data_tile::DataTile, grid::*, hex_grid::HexGrid};
use ndarray::*;

#[derive(Clone, Debug)]
pub struct Tile {
    pub grid: Grid,
    pub start_id: (i64, i64),
    pub nx: u64,
    pub ny: u64,
}

#[enum_delegate::register]
pub trait TileTraits {
    fn get_tile(&self) -> &Tile;

    fn get_grid(&self) -> &Grid;

    fn to_data_tile_with_value<
        T: Num + Clone + Copy + PartialEq + Bounded + ToPrimitive + FromPrimitive + PartialOrd + NumCast,
    >(
        &self,
        fill_value: T,
        nodata_value: T,
    ) -> DataTile<T> {
        let data = Array2::from_elem(
            [self.get_tile().ny as usize, self.get_tile().nx as usize],
            fill_value,
        );
        DataTile {
            tile: self.get_tile().clone(),
            data,
            nodata_value,
        }
    }

    fn corner_ids<'py>(&self) -> Array2<i64> {
        self.get_tile().corner_ids()
    }

    fn corners<'py>(&self) -> Array2<f64> {
        self.get_tile().corners()
    }

    fn indices<'py>(&self) -> Array3<i64> {
        self.get_tile().indices()
    }

    fn intersects(&self, other: &Tile) -> bool {
        self.get_tile().intersects(other)
    }

    fn overlap(&self, other: &Tile) -> Result<Tile, String> {
        self.get_tile().overlap(other)
    }

    fn combined_tile(&self, other: &Tile) -> Tile {
        self.get_tile().combined_tile(other)
    }

    fn grid_id_to_tile_id(&self, grid_ids: &ArrayView2<i64>, oob_value: i64) -> Array2<i64> {
        let mut tile_ids = Array2::<i64>::zeros((grid_ids.shape()[0], 2));
        for cell_id in 0..grid_ids.shape()[0] {
            match self.grid_id_to_tile_id_xy(grid_ids[Ix2(cell_id, 0)], grid_ids[Ix2(cell_id, 1)]) {
                Ok((col, row)) => {
                    tile_ids[Ix2(cell_id, 0)] = col;
                    tile_ids[Ix2(cell_id, 1)] = row;
                }
                Err(e) => {
                    tile_ids[Ix2(cell_id, 0)] = oob_value;
                    tile_ids[Ix2(cell_id, 1)] = oob_value;
                }
            }
        }
        tile_ids
    }

    fn grid_id_to_tile_id_xy(&self, id_x: i64, id_y: i64) -> Result<(i64, i64), String> {
        let tile = self.get_tile();
        let grid = self.get_grid();
        let tile_id_x = id_x - tile.start_id.0;
        // Flip y, for grid start_id is bottom left and array origin is top left as per numpy convention
        let tile_id_y = (tile.start_id.1 + tile.ny as i64 - 1) - id_y;

        // Check out of bounds
        if tile_id_x < 0
            || tile_id_y < 0
            || tile_id_x >= self.get_tile().nx as i64
            || tile_id_y >= self.get_tile().ny as i64
        {
            let error_message = format!(
                "Grid ID ({}, {}) not in Tile ID with start_id: ({}, {}), nx: {}, ny: {}. Array id would have been ({},{})",
                id_x, id_y, tile.start_id.0, tile.start_id.1, tile.nx, tile.ny, tile_id_x, tile_id_y
            )
            .to_string();
            return Err(error_message);
        }
        // Note: return y,x for array indexing as per numpy convention
        return Ok((tile_id_y, tile_id_x));
    }

    fn tile_id_to_grid_id(&self, tile_ids: &ArrayView2<i64>, oob_value: i64) -> Array2<i64> {
        let mut grid_ids = Array2::<i64>::zeros((tile_ids.shape()[0], 2));
        for cell_id in 0..grid_ids.shape()[0] {
            match self.tile_id_to_grid_id_xy(tile_ids[Ix2(cell_id, 0)], tile_ids[Ix2(cell_id, 1)]) {
                Ok((x, y)) => {
                    grid_ids[Ix2(cell_id, 0)] = x;
                    grid_ids[Ix2(cell_id, 1)] = y;
                }
                Err(e) => {
                    grid_ids[Ix2(cell_id, 0)] = oob_value;
                    grid_ids[Ix2(cell_id, 1)] = oob_value;
                }
            }
        }
        grid_ids
    }

    fn tile_id_to_grid_id_xy(&self, id_col: i64, id_row: i64) -> Result<(i64, i64), String> {
        let tile = self.get_tile();
        // Check out of bounds
        if id_row < 0
            || id_col < 0
            || id_row >= self.get_tile().nx as i64
            || id_col >= self.get_tile().ny as i64
        {
            let error_message = format!(
                "Tile ID ({},{}) not in Tile with start_id: ({}, {}), nx: {}, ny: {}.",
                id_col, id_row, tile.start_id.0, tile.start_id.1, tile.nx, tile.ny
            )
            .to_string();
            return Err(error_message);
        }

        // Do conversion
        let grid_id_x = id_row + tile.start_id.0;
        // Flip y, for grid start_id is bottom left and array origin is top left as per numpy convention
        // Subtract 1 to account for base-0
        let grid_id_y = (tile.ny as i64 - id_col) + tile.start_id.1 - 1;
        return Ok((grid_id_x, grid_id_y));
    }
}

impl TileTraits for Tile {
    fn get_tile(&self) -> &Tile {
        &self
    }

    fn get_grid(&self) -> &Grid {
        &self.grid
    }

    fn corner_ids(&self) -> Array2<i64> {
        let (x0, y0) = self.start_id;
        let ids = array![
            [x0, y0 + self.ny as i64 - 1],                      // top-left
            [x0 + self.nx as i64 - 1, y0 + self.ny as i64 - 1], // top-right
            [x0 + self.nx as i64 - 1, y0],                      // bottom-right
            [x0, y0],                                           // bottom-left
        ];
        ids
    }

    fn corners(&self) -> Array2<f64> {
        let start_corner_x = self.start_id.0 as f64 * self.grid.dx() + self.grid.offset()[0];
        let start_corner_y = self.start_id.1 as f64 * self.grid.dy() + self.grid.offset()[1];
        let side_length_x = self.nx as f64 * self.grid.dx();
        let side_length_y = self.ny as f64 * self.grid.dy();

        let mut corners = array![
            [start_corner_x, start_corner_y + side_length_y],
            [
                start_corner_x + side_length_x,
                start_corner_y + side_length_y
            ],
            [start_corner_x + side_length_x, start_corner_y],
            [start_corner_x, start_corner_y],
        ];

        // Rotate if necessary
        if self.grid.rotation() != 0. {
            let rotation_matrix = self.grid.rotation_matrix();
            for cell_id in 0..corners.shape()[0] {
                let mut corner = corners.slice_mut(s![cell_id, ..]);
                let corner_rot = rotation_matrix.dot(&corner);
                corner.assign(&corner_rot);
            }
        }
        corners
    }

    fn indices(&self) -> Array3<i64> {
        let mut indices = Array3::<i64>::zeros((self.ny as usize, self.nx as usize, 2));
        for iy in 0..self.ny {
            for ix in 0..self.nx {
                indices[Ix3(iy as usize, ix as usize, 0)] = self.start_id.0 + ix as i64;
                // Note: index y from top to bottom to compy with numpy standard, hence self.ny - iy
                //       Also subtract -1. Best way to think about this is to reason from a tile with ny=1.
                //       In that case there is only one tile, start_id so we need counter adding self.ny=1.
                indices[Ix3(iy as usize, ix as usize, 1)] =
                    self.start_id.1 + (self.ny - 1 - iy) as i64;
            }
        }
        indices
    }

    fn intersects(&self, other: &Tile) -> bool {
        return !(self.start_id.0 >= (other.start_id.0 + other.nx as i64)
            || (self.start_id.0 + self.nx as i64) <= other.start_id.0
            || self.start_id.1 >= (other.start_id.1 + other.ny as i64)
            || (self.start_id.1 + other.ny as i64) <= other.start_id.1);
    }

    fn overlap(&self, other: &Tile) -> Result<Tile, String> {
        if !self.intersects(&other) {
            return Err("Tiles do not overlap".to_string());
        }

        let min_x = std::cmp::max(self.start_id.0, other.start_id.0);
        let min_y = std::cmp::max(self.start_id.1, other.start_id.1);
        let max_x = std::cmp::min(
            self.start_id.0 + self.nx as i64,
            other.start_id.0 + other.nx as i64,
        );
        let max_y = std::cmp::min(
            self.start_id.1 + self.ny as i64,
            other.start_id.1 + other.ny as i64,
        );

        Ok(Tile {
            grid: self.grid.clone(),
            start_id: (min_x, min_y),
            nx: (max_x - min_x) as u64,
            ny: (max_y - min_y) as u64,
        })
    }

    fn combined_tile(&self, other: &Tile) -> Tile {
        // Determine start tile (bottom left) (get min of bot-left)
        // Determine dx and dy of combined tile (top right) (get max of top-right)
        // Fill with nodata values
        let min_x_id = i64::min(self.start_id.0, other.start_id.0);
        let min_y_id = i64::min(self.start_id.1, other.start_id.1);
        let max_x_id = i64::max(
            self.start_id.0 + self.nx as i64,
            other.start_id.0 + other.nx as i64,
        );
        let max_y_id = i64::max(
            self.start_id.1 + self.ny as i64,
            other.start_id.1 + other.ny as i64,
        );

        let nx = max_x_id - min_x_id;
        let ny = max_y_id - min_y_id;

        Tile {
            grid: self.grid.clone(),
            start_id: (min_x_id, min_y_id),
            nx: nx as u64,
            ny: ny as u64,
        }
    }
}

pub fn combine_tiles<T: TileTraits>(tiles: &Vec<T>) -> Tile {
    let mut min_x: i64 = i64::MAX;
    let mut min_y: i64 = i64::MAX;
    let mut max_x: i64 = i64::MIN;
    let mut max_y: i64 = i64::MIN;
    let _ = for tile in tiles {
        let t = tile.get_tile();
        // Min x
        if t.start_id.0 < min_x {
            min_x = t.start_id.0;
        };
        // Min y
        if t.start_id.1 < min_y {
            min_y = t.start_id.1;
        };
        // Max x
        let tile_max_x = t.start_id.0 + t.nx as i64;
        if tile_max_x > max_x {
            max_x = tile_max_x;
        };
        // Max y
        let tile_max_y = t.start_id.1 + t.ny as i64;
        if tile_max_y > max_y {
            max_y = tile_max_y;
        };
    };
    return Tile {
        grid: tiles[0].get_grid().clone(),
        start_id: (min_x, min_y),
        nx: (max_x - min_x) as u64,
        ny: (max_y - min_y) as u64,
    };
}

pub fn count_tiles<T: TileTraits>(tiles: &Vec<T>) -> DataTile<u64> {
    let mut combined_tile = combine_tiles(tiles).to_data_tile_with_value(0, u64::MAX);
    for tile in tiles {
        let mut data_slice = combined_tile._slice_tile_mut(&tile.get_tile());
        data_slice.map_inplace(|val| *val += 1);
    }
    combined_tile
}

pub fn count_data_tiles<
    T: Num + Clone + Copy + PartialEq + Bounded + ToPrimitive + FromPrimitive + PartialOrd + NumCast,
>(
    tiles: &Vec<DataTile<T>>,
) -> DataTile<u64> {
    let mut combined_tile = combine_tiles(tiles).to_data_tile_with_value(0, 0);
    for tile in tiles {
        let mut data_slice = combined_tile._slice_tile_mut(&tile.get_tile());
        for (count_val, tile_val) in data_slice.iter_mut().zip(tile.data.iter()) {
            if *tile_val != tile.nodata_value {
                *count_val += 1;
            }
        }
    }
    combined_tile
}

pub fn sum_data_tiles<
    T: Num + Clone + Copy + PartialEq + Bounded + ToPrimitive + FromPrimitive + PartialOrd + NumCast,
>(
    tiles: &Vec<DataTile<T>>,
) -> DataTile<T> {
    let nodata_value = tiles[0].nodata_value;
    let mut combined_tile =
        combine_tiles(tiles).to_data_tile_with_value(nodata_value, nodata_value);
    for tile in tiles {
        let mut data_slice = combined_tile._slice_tile_mut(&tile.get_tile());
        for (current_val, val_to_add) in data_slice.iter_mut().zip(tile.data.iter()) {
            if tile.is_nodata(val_to_add) {
                // Don't add any cell in the tile that is falgged as nodata
                continue;
            }
            // It would have been nice to use combined_tile.is_nodata here,
            // but since combined_tile._slice_tile_mut mutably borrows self,
            // we cannot call other methods on combined_tile as long as the data_slice
            // is being held on to. Since we base the nodata_value on the first tile,
            if tiles[0].is_nodata(current_val) {
                *current_val = *val_to_add; // Overwriting default, only on first modification
            } else {
                *current_val = *current_val + *val_to_add; // If already has a value, start summing
            }
        }
    }
    combined_tile
}

pub fn average_data_tiles<
    T: Num + Clone + Copy + PartialEq + Bounded + ToPrimitive + FromPrimitive + PartialOrd + NumCast,
>(
    tiles: &Vec<DataTile<T>>,
) -> DataTile<f64> {
    let summed = sum_data_tiles(tiles);
    let counted = count_data_tiles(tiles);
    let summed_data = summed.data.mapv(|x| x.to_f64().unwrap_or(f64::NAN));
    let counted_data = counted.data.mapv(|x| x.to_f64().unwrap_or(f64::NAN));
    let result = summed_data / counted_data;
    DataTile {
        tile: summed.tile.clone(),
        data: result,
        nodata_value: f64::NAN,
    }
}
