use crate::grid::*;
use crate::tile::*;
use numpy::ndarray::*;
use std::ops::Add;

#[derive(Clone)]
pub struct DataTile {
    pub tile: Tile,
    pub data: Array2<f64>
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
        data: Array2<f64>
    ) -> Self {
        let tile = Tile{ grid, start_id, nx, ny};
        DataTile {
            tile: tile,
            data,
        }
    }

    pub fn _empty_combined_tile(&self, other: &DataTile, nodata_value: f64) -> DataTile {
        // Determine start tile (bottom left) (get min of bot-left)
        // Determine dx and dy of combined tile (top right) (get max of top-right)
        // Fill with nodata values
        let min_x_id = i64::min(self.tile.start_id.0, other.tile.start_id.0);
        let min_y_id = i64::min(self.tile.start_id.1, other.tile.start_id.1);
        let max_x_id = i64::max(self.tile.start_id.0 + self.tile.nx as i64, other.tile.start_id.0 + other.tile.nx as i64);
        let max_y_id = i64::max(self.tile.start_id.1 + self.tile.ny as i64, other.tile.start_id.1 + other.tile.ny as i64);

        let nx = max_x_id - min_x_id;
        let ny = max_y_id - min_y_id;

        let tile = Tile{
            grid: self.tile.grid.clone(),
            start_id: (min_x_id, min_y_id),
            nx: nx as u64,
            ny: ny as u64
        };
        let data = Array2::from_elem((nx as usize, ny as usize), nodata_value);
        DataTile { tile, data }
    }
}

impl Add<f64> for DataTile {
    type Output = DataTile;

    fn add(self, scalar: f64) -> DataTile {
        DataTile {
            tile: self.tile,
            data: &self.data + scalar,
        }
    }
}

impl Add<DataTile> for DataTile {
    type Output = DataTile;

    fn add(self, other: DataTile) -> DataTile {
        // TODO: Take tile overlap into account: IMPORTATNT!
        DataTile {
            tile: self.tile,
            data: &self.data + &other.data, // Add element-wise using ndarray
        }
    }
}
