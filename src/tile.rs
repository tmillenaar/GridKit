use std::f32::MAX;

use crate::grid::*;
use numpy::{ndarray::*, IntoPyArray};

#[derive(Clone)]
pub struct Tile {
    pub grid: Grid,
    pub start_id: (i64, i64),
    pub nx: u64,
    pub ny: u64,
}

#[enum_delegate::register]
pub trait TileTraits {
    fn get_tile(&self) -> &Tile;
    
    fn corner_ids<'py>(&self) -> Array2<i64> {
        self.get_tile().corner_ids()
    }

    fn corners<'py>(&self) -> Array2<f64> {
        self.get_tile().corners()
    }

    fn indices<'py>(&self) -> Array3<i64> {
        self.get_tile().indices()
    }

    fn bounds<'py>(&self) -> (f64, f64, f64, f64) {
        self.get_tile().bounds()
    }

    fn intersects(&self, other: &Tile) -> bool {
        self.get_tile().intersects(other)
    }

    fn overlap(&self, other: &Tile) -> Tile {
        self.get_tile().overlap(other)
    }
}

impl TileTraits for Tile {

    fn get_tile(&self) -> &Tile {
        &self
    }

    fn corner_ids(&self) -> Array2<i64> {
        let (x0, y0) = self.start_id;
        let ids = array![
            [x0, y0 + self.ny as i64 - 1], // top-left
            [x0 + self.nx as i64 -1, y0 + self.ny as i64 - 1], // top-right
            [x0 + self.nx as i64 -1, y0], // bottom-right
            [x0, y0], // bottom-left
        ];
        ids
    }

    fn corners(&self) -> Array2<f64> {
        let start_corner_x = self.start_id.0 as f64 * self.grid.dx() + self.grid.offset().0;
        let start_corner_y = self.start_id.1 as f64 * self.grid.dy() + self.grid.offset().1;
        let side_length_x = self.nx as f64 * self.grid.dx();
        let side_length_y = self.ny as f64 * self.grid.dy();

        let mut corners = array![
            [start_corner_x, start_corner_y + side_length_y],
            [start_corner_x + side_length_x, start_corner_y + side_length_y],
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
                indices[Ix3(iy as usize, ix as usize, 1)] = self.start_id.1 + iy as i64;
            }
        }
        indices
    }

    fn bounds(&self) -> (f64, f64, f64, f64) {
        let corners: ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>> = self.corners();

        // Note: something like the following might have been neat:
        //        (
        //            corners.slice(s![..,0]).min(),
        //            corners.slice(s![..,1]).min(),
        //            corners.slice(s![..,0]).max(),
        //            corners.slice(s![..,1]).max(),
        //        ) 
        //       But that likely is less performant and it does not work with n-dimensional arrays.
        //       See: https://stackoverflow.com/questions/62544170/how-can-i-get-or-create-the-equivalent-of-numpys-amax-function-with-rusts-ndar/62547757
        let mut xmin = f64::MAX;
        let mut xmax = f64::MIN;
        let mut ymin = f64::MAX;
        let mut ymax = f64::MIN;
        for corner in corners.axis_iter(Axis(0)) {
            let x = corner[Ix1(0)];
            let y = corner[Ix1(1)];
            if x < xmin {
                xmin = x;
            }
            if x > xmax {
                xmax = x;
            }
            if y < ymin {
                ymin = y;
            }
            if y > ymax {
                ymax = y;
            }
        };
        (xmin, ymin, xmax, ymax)
    }

    fn intersects(&self, other: &Tile) -> bool {
        let self_bounds = self.bounds();
        let other_bounds = other.bounds();
        return !(
            self_bounds.0 >= other_bounds.2
            || self_bounds.2 <= other_bounds.0
            || self_bounds.1 >= other_bounds.3
            || self_bounds.3 <= other_bounds.1
        )
    }
    
    fn overlap(&self, other: &Tile) -> Tile {
        // return (
        //     max(self.bounds[0], other_bounds[0]),
        //     max(self.bounds[1], other_bounds[1]),
        //     min(self.bounds[2], other_bounds[2]),
        //     min(self.bounds[3], other_bounds[3]),
        // )
        let min_x = std::cmp::max(self.start_id.0, other.start_id.0);
        let min_y = std::cmp::max(self.start_id.1, other.start_id.1);
        let max_x = std::cmp::min(self.start_id.0 + self.nx as i64, other.start_id.0 + other.nx as i64);
        let max_y = std::cmp::min(self.start_id.1 + self.ny as i64, other.start_id.1 + other.ny as i64);

        Tile{
            grid: self.grid.clone(),
            start_id: (min_x, min_y),
            nx: (max_x - min_x) as u64,
            ny: (max_y - min_y) as u64
        }
    }

    
}
