pub fn iseven(val: i64) -> bool {
    val % 2 == 0
}

pub fn normalize_offset(offset: (f64, f64), dx: f64, dy: f64) -> (f64, f64) {
    // Note: In Rust, % is the remainder.
    //       In Python, % is the modulus.
    //       Here we want the modulus, see https://stackoverflow.com/q/31210357
    let offset_x = ((offset.0 % dx) + dx ) % dx;
    let offset_y = ((offset.1 % dy) + dy ) % dy;
    (offset_x, offset_y)
}