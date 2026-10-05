use ndarray::*;

pub fn iseven(val: i64) -> bool {
    val % 2 == 0
}

pub fn modulus(val: f64, modulus: f64) -> f64 {
    ((val % modulus) + modulus) % modulus
}

pub fn normalize_offset(offset: [f64; 2], dx: f64, dy: f64) -> [f64; 2] {
    // Note: In Rust, % is the remainder.
    //       In Python, % is the modulus.
    //       Here we want the modulus, see https://stackoverflow.com/q/31210357
    let offset_x = modulus(offset[0], dx);
    let offset_y = modulus(offset[1], dy);
    [offset_x, offset_y]
}

/// Relative tolerance used as `numpy.is_close` default.
pub const NUMERIC_RTOL: f64 = 1e-5;
/// Absolute tolerance used as `numpy.is_close` default.
pub const NUMERIC_ATOL: f64 = 1e-8;

/// Compare two floats the way `numpy.is_close` does: `|a - b| <= atol + rtol * |b|`.
pub fn is_close_with_atol_and_rtol(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

pub fn is_close_with_rtol(a: f64, b: f64, rtol: f64) -> bool {
    is_close_with_atol_and_rtol(a, b, rtol, NUMERIC_ATOL)
}

pub fn is_close_with_atol(a: f64, b: f64, atol: f64) -> bool {
    is_close_with_atol_and_rtol(a, b, NUMERIC_RTOL, atol)
}

pub fn is_close(a: f64, b: f64) -> bool {
    is_close_with_atol_and_rtol(a, b, NUMERIC_RTOL, NUMERIC_ATOL)
}

pub fn rotation_matrix_from_angle(angle_deg: f64) -> Array2<f64> {
    let angle_rad = angle_deg.to_radians();
    let cos_angle = angle_rad.cos();
    let sin_angle = angle_rad.sin();
    array![[cos_angle, -sin_angle], [sin_angle, cos_angle]]
}

/// Reshape `index` (any shape ending in an axis of length 2) to `(N, 2)`, run
/// the `(N, 2) -> (N, 2)` function `f` on it, and reshape the result back to
/// the original shape.
///
/// This is the Rust equivalent of numpy's `ravel` + `reshape`. It lets a
/// function written for a stack of `(x, y)` pairs be applied to an array of any
/// rank whose last axis is the `(x, y)` pair, e.g. the `(ny, nx, 2)` array
/// returned by [`crate::Tile::indices`].
///
/// Non-contiguous inputs are handled: they are copied into a contiguous layout
/// before being reshaped.
pub fn map_point_pairs<D, A, B, F>(index: &ArrayView<A, D>, f: F) -> Array<B, D>
where
    D: Dimension,
    A: Clone,
    F: FnOnce(&ArrayView2<A>) -> Array2<B>,
{
    let dim = index.raw_dim();
    assert_eq!(
        *dim.slice()
            .last()
            .expect("expected a last axis of length 2"),
        2
    );
    let pattern = dim.clone().into_pattern();
    let raveled = index.to_shape((dim.size() / 2, 2)).unwrap();
    let result = f(&raveled.view());
    result
        .into_shape(pattern)
        .expect("output of `f` is not compatible with the input shape")
}

/// Like [`map_point_pairs`], but for functions that add an axis before the
/// trailing `(x, y)` axis, e.g. `(N, 2) -> (N, K, 2)`.
///
/// The output shape is the leading axes of the input followed by `(K, 2)`,
/// which is one dimension more than the input. `K` is read from the output of
/// `f` so the same helper works for e.g. cell corners and near-point lookups.
pub fn map_point_pairs_fanout<D, A, B, F>(index: &ArrayView<A, D>, f: F) -> Array<B, D::Larger>
where
    D: Dimension,
    A: Clone,
    F: FnOnce(&ArrayView2<A>) -> Array3<B>,
{
    let dim = index.raw_dim();
    assert_eq!(
        *dim.slice()
            .last()
            .expect("expected a last axis of length 2"),
        2
    );
    let raveled = index.to_shape((dim.size() / 2, 2)).unwrap();
    let result = f(&raveled.view());
    let mut new_dims = dim.slice()[..dim.ndim() - 1].to_vec();
    new_dims.push(result.shape()[1]);
    new_dims.push(2);
    result
        .into_shape(IxDyn(&new_dims))
        .expect("intermediate reshape failed")
        .into_dimensionality::<D::Larger>()
        .expect("output rank does not match the input rank plus one")
}

/// Like [`map_point_pairs`], but for functions that reduce the trailing
/// `(x, y)` pair to a single value, e.g. `(N, 2) -> (N,)`.
///
/// The output shape is the input shape with the last axis removed.
pub fn map_point_pairs_reduce<D, A, B, F>(index: &ArrayView<A, D>, f: F) -> Array<B, D::Smaller>
where
    D: Dimension + RemoveAxis,
    A: Clone,
    F: FnOnce(&ArrayView2<A>) -> Array1<B>,
{
    let dim = index.raw_dim();
    assert_eq!(
        *dim.slice()
            .last()
            .expect("expected a last axis of length 2"),
        2
    );
    let smaller = dim.clone().remove_axis(Axis(dim.ndim() - 1));
    let raveled = index.to_shape((dim.size() / 2, 2)).unwrap();
    let result = f(&raveled.view());
    result
        .into_shape(smaller.into_pattern())
        .expect("output of `f` is not compatible with the reduced input shape")
}
