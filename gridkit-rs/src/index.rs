use ndarray::*;
use std::collections::hash_set::HashSet;

#[derive(Clone)]
pub struct GridIndex {
    pub index: ArrayD<i64>,
}

impl GridIndex {
    pub fn new(index: ArrayD<i64>) -> Self {
        assert!(index.shape().last().map_or(false, |&dim| dim == 2), "Last dimension of index must be of length 2, one for x and one for y.");
        GridIndex { index }
    }

    pub fn x(&self) -> ArrayViewD<i64> {
        let ndim = self.index.ndim();
        self.index.index_axis(Axis(ndim-1), 0)
    }

    pub fn y(&self) -> ArrayViewD<i64> {
        let ndim = self.index.ndim();
        self.index.index_axis(Axis(ndim-1), 1)
    }

    pub fn shape(&self) -> &[usize] {
        // We remove the last index containing xy.
        // Since rust cannot dynamically slice tuples, we need to first create a vec
        // This is not a big performance hit since the number of dimensions will never be large.
        let shape = self.index.shape();
        let shape = &shape[..shape.len()-1];
        shape
    }

    pub fn set_index(&mut self, index: ArrayViewD<i64>) {
        assert!(index.shape().last().map_or(false, |&dim| dim == 2), "Last dimension of index must be of length 2, one for x and one for y.");
        self.index = index.to_owned();
    }

    pub fn intersection(&mut self, index: GridIndex) -> GridIndex {
        let x_self = self.index.index_axis(Axis(self.index.ndim() - 1), 0);
        let y_self = self.index.index_axis(Axis(self.index.ndim() - 1), 1);

        let mut pair_set_self: HashSet<(i64, i64)> = HashSet::new();
        for (x, y) in x_self.iter().zip(y_self.iter()) {
            pair_set_self.insert((*x, *y));
        }

        let x_other = index.index.index_axis(Axis(self.index.ndim() - 1), 0);
        let y_other = index.index.index_axis(Axis(self.index.ndim() - 1), 1);

        let mut vec_of_pairs: Vec<[i64; 2]> = Vec::new();
        for (x, y) in x_other.iter().zip(y_other.iter()) {
            if pair_set_self.contains(&(*x,*y)) {
                vec_of_pairs.push([*x,*y]);
            }
        }

        let shape = (vec_of_pairs.len(), 2);
        let intersection = Array2::from_shape_vec(shape, vec_of_pairs.into_iter().flatten().collect()).unwrap();
        GridIndex{ index:intersection.into_dyn() }
    }
}
