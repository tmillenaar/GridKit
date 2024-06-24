use crate::grid::*;
use numpy::{ndarray::*, IntoPyArray};

#[derive(Clone)]
pub struct Tile {
    pub grid: Grid,
    pub start_id: (i64, i64),
    pub nx: u64,
    pub ny: u64,
}

impl Tile {
    pub fn corner_ids(&self) -> Array2<i64> {
        let (x0, y0) = self.start_id;
        let ids = array![
            [x0, y0], // bottom-left
            [x0, y0 + self.ny as i64 - 1], // top-left
            [x0 + self.nx as i64 -1, y0 + self.ny as i64 - 1], // top-right
            [x0 + self.nx as i64 -1, y0], // bottom-right
        ];
        ids
    }

    pub fn bounding_corners(&self) -> Array2<f64> {
        let corner_ids = self.corner_ids();

        // Determine coordinates of the centroids of the corner-cells in local (non-roated) space
        let mut centroids = Array2::<f64>::zeros((corner_ids.shape()[0], 2));
        for i in 0..corner_ids.shape()[0] {
            let centroid = self.grid.centroid_xy_no_rot(corner_ids[Ix2(i,0)], corner_ids[Ix2(i, 1)]);
            centroids[Ix2(i, 0)] = centroid.0;
            centroids[Ix2(i, 1)] = centroid.1;
        }

        // Convert centroids at corner-cells to coordinates of the corners in local (non-roated) space
        let dx: f64 = self.grid.dx();
        let dy: f64 = self.grid.dy();
        let mut corners = array![
            [centroids[Ix2(0,0)] - dx / 2., centroids[Ix2(0,1)] - dy / 2.],
            [centroids[Ix2(1,0)] - dx / 2., centroids[Ix2(1,1)] + dy / 2.],
            [centroids[Ix2(2,0)] + dx / 2., centroids[Ix2(2,1)] + dy / 2.],
            [centroids[Ix2(3,0)] + dx / 2., centroids[Ix2(3,1)] - dy / 2.],
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

    pub fn indices(&self) -> Array3<i64> {
        let mut indices = Array3::<i64>::zeros((self.ny as usize, self.nx as usize, 2));
        for iy in 0..(self.ny) {
            for ix in 0..(self.nx) {
                indices[Ix3(iy as usize, ix as usize, 0)] = self.start_id.0 + ix as i64;
                indices[Ix3(iy as usize, ix as usize, 1)] = self.start_id.1 + iy as i64;
            }
        }
        indices
    }

    pub fn bbox(&self) -> (f64, f64, f64, f64) {
        let corners = self.bounding_corners();
        // FIXME: weird order of slices to get xmin, ymin, xmax, ymax
        (corners[Ix2(1,0)], corners[Ix2(0,1)], corners[Ix2(3,0)], corners[Ix2(2,1)])
    }
}
