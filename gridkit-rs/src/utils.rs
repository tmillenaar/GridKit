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

/// Relative tolerance used as `numpy.isclose` default.
pub const NUMERIC_RTOL: f64 = 1e-5;
/// Absolute tolerance used as `numpy.isclose` default.
pub const NUMERIC_ATOL: f64 = 1e-8;

/// Compare two floats the way `numpy.isclose` does: `|a - b| <= atol + rtol * |b|`.
pub fn isclose_with_atol_and_rtol(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

pub fn isclose_with_rtol(a: f64, b: f64, rtol: f64) -> bool {
    isclose_with_atol_and_rtol(a, b, rtol, NUMERIC_ATOL)
}

pub fn isclose_with_atol(a: f64, b: f64, atol: f64) -> bool {
    isclose_with_atol_and_rtol(a, b, NUMERIC_RTOL, atol)
}

pub fn isclose(a: f64, b: f64) -> bool {
    isclose_with_atol_and_rtol(a, b, NUMERIC_RTOL, NUMERIC_ATOL)
}

pub fn rotation_matrix_from_angle(angle_deg: f64) -> Array2<f64> {
    let angle_rad = angle_deg.to_radians();
    let cos_angle = angle_rad.cos();
    let sin_angle = angle_rad.sin();
    array![[cos_angle, -sin_angle], [sin_angle, cos_angle]]
}
