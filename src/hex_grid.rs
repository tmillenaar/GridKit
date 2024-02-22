use numpy::ndarray::*;

fn iseven(val: i64) -> bool {
    val % 2 == 0
}
pub struct HexGrid {
    pub cellsize: f64,
    pub offset: (f64, f64),
}

impl HexGrid {
    pub fn new(cellsize: f64, offset: (f64, f64)) -> Self {
        // let (_dx, _dy) = (dx,dy);
        // // Note: In Rust, % is the remainder.
        // //       In Python, % is the modulus.
        // //       Here we want the modulus, see https://stackoverflow.com/q/31210357
        // let offset_x = ((offset.0 % dx) + dx ) % dx;
        // let offset_y = ((offset.1 % dy) + dy ) % dy;
        let offset = offset;
        HexGrid { cellsize, offset }
    }

    pub fn radius(&self) -> f64 {
        self.cellsize / 3_f64.powf(0.5)
    }

    // pub fn cell_height(&self) -> f64 {
    // }

    // pub fn cell_width(&self) -> f64 {
    // }

    pub fn dx(&self) -> f64 {
        self.cellsize
    }

    pub fn dy(&self) -> f64 {
        3. / 2. * self.radius()
    }

    pub fn centroid(&self, index: &ArrayView2<i64>) -> Array2<f64> {
        let mut centroids = Array2::<f64>::zeros((index.shape()[0], 2));

        for cell_id in 0..centroids.shape()[0] {
            let point = self.centroid_single_point(index[Ix2(cell_id, 0)], index[Ix2(cell_id, 1)]);
            centroids[Ix2(cell_id, 0)] = point.0;
            centroids[Ix2(cell_id, 1)] = point.1;
        }
        centroids
    }

    fn centroid_single_point(&self, x: i64, y: i64) -> (f64, f64) {
        let mut centroid_x = x as f64 * self.dx() + (self.dx() / 2.) + self.offset.0;
        let centroid_y = y as f64 * self.dy() + (self.dy() / 2.) + self.offset.1;

        if !iseven(y) {
            centroid_x  = centroid_x + self.dx() / 2.;
        }
        (centroid_x, centroid_y)
    }

}