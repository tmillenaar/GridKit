#![allow(unused_parens)]
use num_traits::{Bounded, FromPrimitive, Num, NumCast, ToPrimitive};
use std::{f32::MAX, f64, i64, u64};

use crate::{data_tile::DataTile, grid::*};
use ndarray::*;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
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
        T: Num
            + Clone
            + Copy
            + PartialEq
            + Bounded
            + ToPrimitive
            + FromPrimitive
            + PartialOrd
            + NumCast,
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

    fn intersects_bounds(&self, bounds: &[f64; 4]) -> bool {
        self.get_tile().intersects_bounds(bounds)
    }

    fn intersection_bounds(&self, other: &Tile) -> Option<(f64, f64, f64, f64)> {
        self.get_tile().intersection_bounds(other)
    }

    fn intersection_corners(&self, other: &Tile) -> Option<Array2<f64>> {
        self.get_tile().intersection_corners(other)
    }

    fn overlap(&self, other: &Tile) -> Result<Tile, String> {
        self.get_tile().overlap(other)
    }

    fn combined_tile(&self, other: &Tile) -> Tile {
        self.get_tile().combined_tile(other)
    }

    /// Convert grid ids to tile ids.
    ///
    /// `grid_ids` may have any shape as long as its last axis is of length 2 and
    /// holds the `(x, y)` grid ids. The result has the same shape; ids outside
    /// the tile are set to `oob_value`.
    fn grid_id_to_tile_id<D>(&self, grid_ids: ArrayView<i64, D>, oob_value: i64) -> Array<i64, D>
    where
        D: Dimension,
    {
        crate::utils::map_point_pairs(grid_ids, |grid_ids| {
            let mut tile_ids = Array2::<i64>::zeros((grid_ids.shape()[0], 2));
            for cell_id in 0..grid_ids.shape()[0] {
                match self
                    .grid_id_to_tile_id_xy(grid_ids[Ix2(cell_id, 0)], grid_ids[Ix2(cell_id, 1)])
                {
                    Ok((col, row)) => {
                        tile_ids[Ix2(cell_id, 0)] = col;
                        tile_ids[Ix2(cell_id, 1)] = row;
                    }
                    Err(_e) => {
                        tile_ids[Ix2(cell_id, 0)] = oob_value;
                        tile_ids[Ix2(cell_id, 1)] = oob_value;
                    }
                }
            }
            tile_ids
        })
    }

    fn grid_id_to_tile_id_xy(&self, id_x: i64, id_y: i64) -> Result<(i64, i64), String> {
        let tile = self.get_tile();
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

    /// Convert tile ids to grid ids.
    ///
    /// `tile_ids` may have any shape as long as its last axis is of length 2 and
    /// holds the `(row, col)` tile ids. The result has the same shape; ids
    /// outside the tile are set to `oob_value`.
    fn tile_id_to_grid_id<D>(&self, tile_ids: ArrayView<i64, D>, oob_value: i64) -> Array<i64, D>
    where
        D: Dimension,
    {
        crate::utils::map_point_pairs(tile_ids, |tile_ids| {
            let mut grid_ids = Array2::<i64>::zeros((tile_ids.shape()[0], 2));
            for cell_id in 0..grid_ids.shape()[0] {
                match self
                    .tile_id_to_grid_id_xy(tile_ids[Ix2(cell_id, 0)], tile_ids[Ix2(cell_id, 1)])
                {
                    Ok((x, y)) => {
                        grid_ids[Ix2(cell_id, 0)] = x;
                        grid_ids[Ix2(cell_id, 1)] = y;
                    }
                    Err(_e) => {
                        grid_ids[Ix2(cell_id, 0)] = oob_value;
                        grid_ids[Ix2(cell_id, 1)] = oob_value;
                    }
                }
            }
            grid_ids
        })
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
        /// Returns corners in order:
        /// top-left, top-right, bottom-right, bottom-left
        /// 0 > 1
        ///     v
        /// 3 < 2
        ///
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
            // Applying the rotation by hand avoids the temporary allocation
            // that a `rotation_matrix.dot(&corner)` call would make per corner.
            let cos = rotation_matrix[[0, 0]];
            let sin = rotation_matrix[[1, 0]];
            for cell_id in 0..corners.shape()[0] {
                let x = corners[Ix2(cell_id, 0)];
                let y = corners[Ix2(cell_id, 1)];
                corners[Ix2(cell_id, 0)] = cos * x - sin * y;
                corners[Ix2(cell_id, 1)] = sin * x + cos * y;
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
        // Note: It is easier to match tiles on their ids. This seems elegant
        //       because we don't need to care about rotation. This was how it
        //       was first implemented but it required self and other to be on
        //       the same grid. This was not enforced so you would get weird
        //       results if this was not upheld. Rather than forcing the grids
        //       to be aligned we use the tile corner coordinates. This way
        //       any tile can be checked for intersection with any other tile
        //       purely based on location, regardless of the host grid. This
        //       does have the downside that we need to account for rotated
        //       grids where the corner that was originally the on the left
        //       side might now be the rightmost corner.
        let corners_self = self.corners();
        let corners_other = other.corners();

        // Get min and max values.
        // Note: This is a thorn in the eye as .min() does not work
        //       on ndarrays of dtype float because of nan and inf values.
        let left_self = corners_self
            .slice(s![.., 0])
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap();
        let bottom_self = corners_self
            .slice(s![.., 1])
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap();
        let right_self = corners_self
            .slice(s![.., 0])
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap();
        let top_self = corners_self
            .slice(s![.., 1])
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap();

        let left_other = corners_other
            .slice(s![.., 0])
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap();
        let bottom_other = corners_other
            .slice(s![.., 1])
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap();
        let right_other = corners_other
            .slice(s![.., 0])
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap();
        let top_other = corners_other
            .slice(s![.., 1])
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap();

        return !(left_self >= right_other
            || right_self <= left_other
            || bottom_self >= top_other
            || top_self <= bottom_other);
    }

    fn intersects_bounds(&self, bounds: &[f64; 4]) -> bool {
        let corners = self.corners();

        // Get tile bounds.
        let left = corners
            .slice(s![.., 0])
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap();
        let bottom = corners
            .slice(s![.., 1])
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap();
        let right = corners
            .slice(s![.., 0])
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap();
        let top = corners
            .slice(s![.., 1])
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap();

        let [left_other, bottom_other, right_other, top_other] = bounds;

        !(left >= *right_other
            || right <= *left_other
            || bottom >= *top_other
            || top <= *bottom_other)
    }

    fn intersection_bounds(&self, other: &Tile) -> Option<(f64, f64, f64, f64)> {
        let corners_self = self.corners();
        let corners_other = other.corners();

        let left_self = corners_self
            .slice(s![.., 0])
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap();
        let bottom_self = corners_self
            .slice(s![.., 1])
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap();
        let right_self = corners_self
            .slice(s![.., 0])
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap();
        let top_self = corners_self
            .slice(s![.., 1])
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap();

        let left_other = corners_other
            .slice(s![.., 0])
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap();
        let bottom_other = corners_other
            .slice(s![.., 1])
            .iter()
            .copied()
            .reduce(f64::min)
            .unwrap();
        let right_other = corners_other
            .slice(s![.., 0])
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap();
        let top_other = corners_other
            .slice(s![.., 1])
            .iter()
            .copied()
            .reduce(f64::max)
            .unwrap();

        // Compute intersection bounds
        let minx = left_self.max(left_other);
        let miny = bottom_self.max(bottom_other);
        let maxx = right_self.min(right_other);
        let maxy = top_self.min(top_other);

        // Check if there is an intersection
        if minx < maxx && miny < maxy {
            Some((minx, miny, maxx, maxy))
        } else {
            None
        }
    }

    fn intersection_corners(&self, other: &Tile) -> Option<Array2<f64>> {
        if let Some((minx, miny, maxx, maxy)) = self.intersection_bounds(other) {
            Some(arr2(&[
                [minx, maxy],
                [maxx, maxy],
                [maxx, miny],
                [minx, miny],
            ]))
        } else {
            None
        }
    }

    fn overlap(&self, other: &Tile) -> Result<Tile, String> {
        /// Returns a tile that covers the intersection between the tiles `self` and `other`.
        ///
        /// ## Keywords
        /// intersection, crop, overlap, clip
        ///
        if self.get_grid() != other.get_grid() {
            return Err("Tiles do not live on the same grid.".to_string());
        }

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

impl Tile {
    /// Return a copy of this tile with `nx` x `ny` cells whose grid has been
    /// scaled by `factor`, keeping the start corner and rotation unchanged.
    ///
    /// This is the shared workhorse for [`Tile::upscale`] and
    /// [`Tile::downscale`]. It assumes the caller has already checked that the
    /// new dimensions fit the tile exactly.
    fn _rescale(&self, nx: u64, ny: u64, factor: f64) -> Tile {
        let mut grid = self.get_grid().clone();
        // Scale cellsize
        match &mut grid {
            Grid::TriGrid(tri_grid) => {
                tri_grid.cellsize *= factor;
            }
            Grid::RectGrid(rect_grid) => {
                rect_grid._dx *= factor;
                rect_grid._dy *= factor;
            }
            Grid::HexGrid(hex_grid) => {
                hex_grid.cellsize *= factor;
            }
        };

        let tile_corners = self.corners();
        let start_corner_x = tile_corners[Ix2(3, 0)];
        let start_corner_y = tile_corners[Ix2(3, 1)];
        dbg!(start_corner_x);
        dbg!(start_corner_y);
        grid.anchor_inplace(&[start_corner_x, start_corner_y], CellElement::Corner);

        // let start_id = grid.cell_at_point(&[
        //     start_corner_x + grid.dx() / 4.,
        //     start_corner_y + grid.dy() / 4.,
        // ]);
        let start_id = grid.cell_at_point(&[start_corner_x, start_corner_y]);

        Tile {
            grid,
            start_id: start_id.into(),
            nx,
            ny,
        }
    }

    /// Return a tile with `factor` times as many cells along both axes,
    /// subdividing each cell into `factor` x `factor` smaller cells.
    ///
    /// The tile keeps exactly the same bounding box and rotation. A `factor` of
    /// 0 leaves the tile unchanged (mirroring [`GridTraits::subdivide`]).
    pub fn upscale(&self, factor: u64) -> Tile {
        if factor == 0 {
            return self.clone();
        }
        let nx = self
            .nx
            .checked_mul(factor)
            .expect("upscaling the tile would overflow the number of cells");
        let ny = self
            .ny
            .checked_mul(factor)
            .expect("upscaling the tile would overflow the number of cells");
        self._rescale(nx, ny, 1. / factor as f64)
    }

    /// Return a tile with `factor` times fewer cells along both axes.
    ///
    /// The coarser cells are `factor` times larger, so the tile keeps exactly
    /// the same bounding box and rotation. This is the inverse of
    /// [`Tile::upscale`] (which subdivides the tile).
    ///
    /// `factor` must be nonzero and must divide both `nx` and `ny`, otherwise
    /// the coarser cells would not fit the tile exactly and an error is
    /// returned.
    pub fn downscale(&self, factor: u64) -> Result<Tile, String> {
        if factor == 0 {
            return Err("Cannot downscale a tile by a factor of 0.".to_string());
        }
        if self.nx % factor != 0 || self.ny % factor != 0 {
            return Err(format!(
                "Cannot downscale a tile of {} x {} cells by a factor of {}: \
                 the factor must divide both nx and ny so the coarser cells still \
                 fit the tile's bounding box exactly.",
                self.nx, self.ny, factor
            ));
        }
        Ok(self._rescale(self.nx / factor, self.ny / factor, factor as f64))
    }

    /// Sample `tile_to_sample` at this tile's cell centroids using nearest-neighbour sampling.
    pub fn sample_nearest_from<T>(&self, tile_to_sample: &DataTile<T>) -> DataTile<T>
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
            + Default
            + Send
            + Sync,
    {
        let centroids = self.grid.centroid(self.indices().view());
        let source_indices = tile_to_sample.tile.grid.cell_at_points(centroids.view());
        let data = tile_to_sample.values(source_indices.view(), tile_to_sample.nodata_value);
        DataTile::new(
            self.grid.clone(),
            self.start_id,
            self.nx,
            self.ny,
            data,
            tile_to_sample.nodata_value,
        )
    }

    /// Sample `tile_to_sample` at this tile's cell centroids using bilinear interpolation.
    pub fn sample_bilinear_from<T>(&self, tile_to_sample: &DataTile<T>) -> DataTile<f64>
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
            + Default
            + Send
            + Sync,
    {
        let centroids = self.grid.centroid(self.indices().view());
        let data = tile_to_sample.linear_interpolation(centroids.view());
        DataTile::new(
            self.grid.clone(),
            self.start_id,
            self.nx,
            self.ny,
            data,
            tile_to_sample.nodata_value.to_f64().unwrap_or(f64::NAN),
        )
    }

    /// Sample `tile_to_sample` at this tile's cell centroids using inverse-distance interpolation.
    pub fn sample_inverse_distance_from<T>(
        &self,
        tile_to_sample: &DataTile<T>,
        decay_constant: f64,
    ) -> DataTile<f64>
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
            + Default
            + Send
            + Sync,
    {
        let centroids = self.grid.centroid(self.indices().view());
        let data = tile_to_sample.inverse_distance_interpolation(centroids.view(), decay_constant);
        DataTile::new(
            self.grid.clone(),
            self.start_id,
            self.nx,
            self.ny,
            data,
            tile_to_sample.nodata_value.to_f64().unwrap_or(f64::NAN),
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex_grid::HexGrid;
    use crate::rect_grid::RectGrid;
    use crate::tri_grid::TriGrid;
    use crate::Orientation;
    use ndarray::array;

    #[test]
    fn grid_queries_accept_the_3d_indices_directly() {
        let grid = Grid::RectGrid(RectGrid::new(5., 2.));
        let tile = Tile {
            grid,
            start_id: (0, 0),
            nx: 3,
            ny: 2,
        };

        // `indices()` is (ny, nx, 2); the grid queries accept it as-is and
        // return arrays with the same leading axes.
        let indices = tile.indices();
        assert_eq!(indices.shape(), &[2, 3, 2]);

        let centroids = tile.get_grid().centroid(indices.view());
        assert_eq!(centroids.shape(), &[2, 3, 2]);

        let corners = tile.get_grid().cell_corners(indices.view());
        assert_eq!(corners.shape(), &[2, 3, 4, 2]);

        let near = tile.get_grid().cells_near_point(centroids.view());
        assert_eq!(near.shape(), &[2, 3, 4, 2]);

        let tile_ids = tile.grid_id_to_tile_id(indices.view(), i64::MAX);
        assert_eq!(tile_ids.shape(), &[2, 3, 2]);
    }

    #[test]
    fn data_tile_values_accept_the_3d_indices_directly() {
        let grid = Grid::RectGrid(RectGrid::new(5., 2.));
        let data = array![[1, 2, 3], [4, 5, 6]];
        let data_tile = DataTile::new(grid, (0, 0), 3, 2, data, -1);

        let indices = data_tile.indices();
        let values = data_tile.values(indices.view(), -1);
        assert_eq!(values.shape(), &[2, 3]);

        // Flattened access agrees with the 3D access.
        let flat = indices.clone().into_shape((6, 2)).unwrap();
        let expected = data_tile
            .values(flat.view(), -1)
            .into_shape((2, 3))
            .unwrap();
        assert_eq!(values, expected);
    }

    #[test]
    fn samples_nearest_at_target_centroids_and_fills_outside_with_nodata() {
        let source_grid = Grid::RectGrid(RectGrid::new(1., 1.));
        let source = DataTile::new(
            source_grid.clone(),
            (0, 0),
            2,
            2,
            array![[1, 2], [3, 4]],
            -1,
        );
        let target = Tile {
            grid: source_grid,
            start_id: (-1, 0),
            nx: 3,
            ny: 2,
        };

        let sampled = target.sample_nearest_from(&source);
        assert_eq!(sampled.data, array![[-1, 1, 2], [-1, 3, 4]]);
        assert_eq!(sampled.nodata_value, -1);
    }

    #[test]
    fn samples_bilinear_and_inverse_distance_at_target_centroids() {
        let source_grid = Grid::RectGrid(RectGrid::new(1., 1.));
        let source = DataTile::new(
            source_grid.clone(),
            (0, 0),
            2,
            2,
            array![[1., 2.], [3., 4.]],
            f64::NAN,
        );
        let mut target_grid = source_grid;
        target_grid.set_offset([0.25, 0.25]);
        let target = Tile {
            grid: target_grid,
            start_id: (0, 0),
            nx: 1,
            ny: 1,
        };

        let bilinear = target.sample_bilinear_from(&source);
        let inverse_distance = target.sample_inverse_distance_from(&source, 1.);
        assert_eq!(bilinear.data.shape(), &[1, 1]);
        assert_eq!(inverse_distance.data.shape(), &[1, 1]);
        assert!(bilinear.data.iter().all(|value| !value.is_nan()));
        assert!(inverse_distance.data.iter().all(|value| !value.is_nan()));
        assert!((bilinear.data[[0, 0]] - 2.75).abs() < 1e-12);
        assert!((inverse_distance.data[[0, 0]] - 2.6224593312018545).abs() < 1e-12);
    }

    #[test]
    fn interpolated_samples_use_nodata_outside_source_tile() {
        let source = DataTile::new(
            Grid::RectGrid(RectGrid::new(1., 1.)),
            (0, 0),
            1,
            1,
            array![[7.]],
            -99.,
        );
        let target = Tile {
            grid: source.tile.grid.clone(),
            start_id: (1, 0),
            nx: 1,
            ny: 1,
        };

        assert_eq!(target.sample_bilinear_from(&source).data[[0, 0]], -99.);
        assert_eq!(
            target.sample_inverse_distance_from(&source, 1.).data[[0, 0]],
            -99.
        );
    }

    // ---------------------------------------------------------------------
    // upscale / downscale
    // ---------------------------------------------------------------------

    /// One grid of each type, non-trivial offset and rotation so the resizing
    /// code cannot accidentally lean on the origin or an axis-aligned grid.
    fn sample_grids() -> Vec<Grid> {
        let mut rect = RectGrid::new(2., 3.);
        rect.set_offset([0.4, -1.2]);
        rect.set_rotation(31.);

        let mut hex_flat = HexGrid::new(2., Orientation::Flat);
        hex_flat.set_offset([0.4, -1.2]);
        hex_flat.set_rotation(31.);

        let mut hex_pointy = HexGrid::new(2., Orientation::Pointy);
        hex_pointy.set_offset([-0.7, 0.3]);
        hex_pointy.set_rotation(-12.);

        let mut tri_flat = TriGrid::new(2., Orientation::Flat);
        tri_flat.set_offset([0.4, -1.2]);
        tri_flat.set_rotation(31.);

        let mut tri_pointy = TriGrid::new(2., Orientation::Pointy);
        tri_pointy.set_offset([-0.7, 0.3]);
        tri_pointy.set_rotation(-12.);

        vec![
            Grid::RectGrid(rect),
            Grid::HexGrid(hex_flat),
            Grid::HexGrid(hex_pointy),
            Grid::TriGrid(tri_flat),
            Grid::TriGrid(tri_pointy),
        ]
    }

    fn assert_corners_close(a: &Array2<f64>, b: &Array2<f64>) {
        assert_eq!(a.shape(), b.shape(), "corner arrays have different shapes");
        for i in 0..a.shape()[0] {
            for j in 0..a.shape()[1] {
                assert!(
                    (a[[i, j]] - b[[i, j]]).abs() < 1e-9,
                    "corner [{i}, {j}] differs: {} vs {}",
                    a[[i, j]],
                    b[[i, j]]
                );
            }
        }
    }

    #[test]
    fn upscale_keeps_the_bounding_box_and_rotation() {
        for grid in sample_grids() {
            let grid_name = grid_type_name(&grid).to_string();
            let tile = Tile {
                grid,
                start_id: (2, -3),
                nx: 4,
                ny: 2,
            };

            let up = tile.upscale(3);

            assert_eq!(up.nx, 12, "{grid_name}");
            assert_eq!(up.ny, 6, "{grid_name}");
            assert_eq!(
                grid_type_name(&up.grid),
                grid_type_name(&tile.grid),
                "{grid_name}"
            );
            assert!((up.grid.rotation() - tile.grid.rotation()).abs() < 1e-12);
            assert!((up.grid.dx() - tile.grid.dx() / 3.).abs() < 1e-9);
            assert!((up.grid.dy() - tile.grid.dy() / 3.).abs() < 1e-9);
            assert_corners_close(&up.corners(), &tile.corners());
        }
    }

    #[test]
    fn upscale_by_zero_leaves_the_tile_unchanged() {
        for grid in sample_grids() {
            let tile = Tile {
                grid,
                start_id: (2, -3),
                nx: 4,
                ny: 2,
            };
            let up = tile.upscale(0);
            assert_eq!(up.nx, tile.nx);
            assert_eq!(up.ny, tile.ny);
            assert_corners_close(&up.corners(), &tile.corners());
        }
    }

    #[test]
    fn downscale_keeps_the_bounding_box_and_rejects_bad_factors() {
        for grid in sample_grids() {
            println!("==========+++++==========");
            let grid_name = grid_type_name(&grid).to_string();
            let tile = Tile {
                grid,
                start_id: (-5, 7),
                nx: 6,
                ny: 4,
            };

            let down = tile.downscale(2).unwrap();
            assert_eq!(down.nx, 3, "{grid_name}");
            assert_eq!(down.ny, 2, "{grid_name}");
            dbg!(&tile);
            dbg!(&down);
            dbg!(&tile.corners());
            dbg!(&down.corners());
            assert_corners_close(&down.corners(), &tile.corners());

            // Only downscale by a factor that divides both dimensions.
            assert!(tile.downscale(0).is_err(), "{grid_name}");
            assert!(tile.downscale(4).is_err(), "{grid_name}");
            assert!(tile.downscale(5).is_err(), "{grid_name}");
        }
    }

    #[test]
    fn downscale_then_upscale_round_trips_the_corners() {
        for grid in sample_grids() {
            let tile = Tile {
                grid,
                start_id: (-5, 7),
                nx: 6,
                ny: 4,
            };
            let round_tripped = tile.downscale(2).unwrap().upscale(2);
            assert_corners_close(&round_tripped.corners(), &tile.corners());
        }
    }

    #[test]
    fn rescaling_re_derives_the_start_id() {
        // With a zero offset and an aligned tile, upscaling multiplies the ids
        // and the offset stays zero.
        let grid = Grid::RectGrid(RectGrid::new(2., 3.));
        let tile = Tile {
            grid: grid.clone(),
            start_id: (4, -6),
            nx: 8,
            ny: 6,
        };
        assert_eq!(tile.upscale(2).start_id, (8, -12));
        assert_eq!(tile.grid.offset(), [0., 0.]);

        let down = tile.downscale(2).unwrap();
        assert_eq!(down.start_id, (2, -3));
        assert_eq!(down.grid.offset(), [0., 0.]);
        assert_corners_close(&down.corners(), &tile.corners());

        // When the start id is not aligned to the coarser grid, the offset is
        // shifted so the bounds are still preserved.
        let unaligned = Tile {
            grid,
            start_id: (3, -5),
            nx: 8,
            ny: 6,
        };
        let down = unaligned.downscale(2).unwrap();
        assert!((down.grid.offset()[0] - 2.).abs() < 1e-12);
        assert!((down.grid.offset()[1] - 3.).abs() < 1e-12);
        assert_corners_close(&down.corners(), &unaligned.corners());
    }

    #[test]
    fn resized_tile_can_sample_the_original_data_tile() {
        // A one-cell source tile whose value is sampled on the finer grid.
        let source = DataTile::new(
            Grid::RectGrid(RectGrid::new(4., 4.)),
            (0, 0),
            1,
            1,
            array![[7.]],
            -1.,
        );

        let target = source.get_tile().upscale(4);
        assert_eq!((target.nx, target.ny), (4, 4));
        let sampled = target.sample_nearest_from(&source);
        assert!(sampled.data.iter().all(|value| *value == 7.));
    }

    #[test]
    fn data_tile_resizing_methods_error_with_a_resampling_hint() {
        let data_tile = DataTile::new(
            Grid::RectGrid(RectGrid::new(1., 1.)),
            (0, 0),
            2,
            2,
            array![[1, 2], [3, 4]],
            -1,
        );

        for err in [
            data_tile.upscale(2).err().unwrap(),
            data_tile.downscale(2).err().unwrap(),
        ] {
            assert!(err.contains("get_tile()"), "missing hint in: {err}");
            assert!(
                err.contains("sample_nearest_from"),
                "missing resampling suggestion in: {err}"
            );
        }
    }
}
