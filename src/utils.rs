use numpy::ndarray::*;

pub fn iseven(val: i64) -> bool {
    val % 2 == 0
}

pub fn modulus(val: f64, modulus: f64) -> f64 {
    ((val % modulus) + modulus ) % modulus
}

pub fn normalize_offset(offset: (f64, f64), dx: f64, dy: f64) -> (f64, f64) {
    // Note: In Rust, % is the remainder.
    //       In Python, % is the modulus.
    //       Here we want the modulus, see https://stackoverflow.com/q/31210357
    let offset_x = modulus(offset.0, dx);
    let offset_y = modulus(offset.1, dy);
    (offset_x, offset_y)
}

pub fn _rotation_matrix(angle_deg: f64) -> Array2<f64> {
    let angle_rad = angle_deg.to_radians();
    let cos_angle = angle_rad.cos();
    let sin_angle = angle_rad.sin();
    array![
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ]
}
