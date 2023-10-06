use pyo3::prelude::*;
use numpy::PyArray2;
use numpy::PyReadonlyArray2;
use numpy::ndarray::{Array2};
use numpy::{ToPyArray};

#[pyclass]
struct PointyHexGrid {
    cellsize: f64,
}

#[pymethods]
impl PointyHexGrid {

    #[new]
    fn new(cellsize: f64) -> Self {
        PointyHexGrid{
            cellsize,
        }
    }
    fn radius(&self) -> f64 {
        self.cellsize / (3_f64).sqrt()
    }
    fn dx(&self) -> f64 {
        self.cellsize
    }
    fn dy(&self) -> f64 {
        (3. / 2.) * self.radius()
    }

    fn cell_at_locations<'py>(&self, points: PyReadonlyArray2<'py, f64>) -> Py<PyArray2<i64>> {
        let shape = points.dims();
        // let mut cells = unsafe { Array2::<i64>::uninitialized(shape) };
        let mut cells = Array2::<i64>::zeros(shape);

        for i in 0..shape[0] {
            let (x, y) = self.cell_at_location(*points.get((i, 0)).unwrap(), *points.get((i, 1)).unwrap());
            cells[(i, 0)] = x;
            cells[(i, 1)] = y;
        }

        let py = points.py();
        cells.to_pyarray(py).to_owned()
    }

    fn cell_at_location(&self, x: f64, y: f64) -> (i64, i64) {
        // determine initial id_y
        let mut id_y = ((y - self.radius() / 4.) / self.dy()).floor();
        let is_offset = id_y % 2. != 0.;
        let mut id_x: f64;

        // determine initial id_x
        if is_offset == true {
            id_x = (x - self.dx() / 2.) / self.dx();
        } else {
            id_x = x / self.dx();
        }
        id_x = id_x.floor();

        // refine id_x and id_y
        // Example: points at the top of the cell can be in this cell or in the cell to the top right or top left
        let rel_loc_y = (y - self.radius() / 4.) % self.dy() + self.radius() / 4.;
        let rel_loc_x = x % self.dx();

        let mut in_top_left: bool;
        let mut in_top_right: bool;
        if is_offset == true {
            in_top_left = (self.radius() * 1.25 - rel_loc_y)
                < ((rel_loc_x - 0.5 * self.dx()) / (self.dx() / self.radius()));
            in_top_left = in_top_left && (rel_loc_x < (0.5 * self.dx()));
            in_top_right = (rel_loc_x - 0.5 * self.dx()) / (self.dx() / self.radius())
                <= (rel_loc_y - self.radius() * 1.25);
            in_top_right = in_top_right && rel_loc_x >= (0.5 * self.dx());
            if in_top_left == true {
                id_y = id_y + 1.;
                id_x = id_x + 1.;
            }
            else if in_top_right == true {
                id_y = id_y + 1.;
            }
        } else {
            in_top_left =
                rel_loc_x / (self.dx() / self.radius()) < (rel_loc_y - self.radius() * 5. / 4.);
            in_top_right = (self.radius() * 1.25 - rel_loc_y)
                <= (rel_loc_x - self.dx()) / (self.dx() / self.radius());
            if in_top_left == true {
                id_y = id_y + 1.;
                id_x = id_x - 1.;
            }
            else if in_top_right == true {
                id_y = id_y + 1.;
            }
        }

        (id_x as i64, id_y as i64)
    }
}


/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn gridkit_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<PointyHexGrid>()?;
    Ok(())
}
