use numpy::ndarray::*;

pub fn vec_norm_1d(
    vec: &ArrayView1<f64>,
) -> f64 {
    let mut squared_sum = 0.;
    for item in vec.iter() {
        squared_sum += item.powi(2);
    }
    return squared_sum.powf(0.5)
}

fn _project(
    point: &Array1<f64>,
    line_point_1: &Array1<f64>,
    line_point_2: &Array1<f64>
) -> Array1<f64> {
    let vec_line = line_point_1 - line_point_2;
    let vec_point = point - line_point_1;
    let line_length = vec_norm_1d(&vec_line.view());
    let scaled_vec_line = &vec_line / line_length;
    let scaled_vec_point = &vec_point / line_length;
    let projected_factor = scaled_vec_line.dot(&scaled_vec_point);

    let projected = (projected_factor * &vec_line) + line_point_1;
    return projected
}

pub fn linear_interp_weights_single_triangle(
    sample_point: &ArrayView1<f64>,
    nearby_val_locs: &ArrayView2<f64>
) -> Array1<f64> {
    let mut p1 = nearby_val_locs.slice(s![0, ..]);
    let mut p2 = nearby_val_locs.slice(s![1, ..]);
    let mut p3 = nearby_val_locs.slice(s![2, ..]);
    let mut weights = Array1::<f64>::zeros(3);

    for nearby_pnt_id in 0..3 {
        let side_length = vec_norm_1d(&(&p1 - &p2).view());
        let mut midpoint_opposite_side = &p2 + (&p2 - &p3) / 2.;
        let mut median = &midpoint_opposite_side - &p1;
        if vec_norm_1d(&median.view()) > side_length {
            midpoint_opposite_side = &p2 - (&p2 - &p3) / 2.;
            median = &midpoint_opposite_side - &p1;
        }

        let projected: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = _project(
            &(sample_point - &p1),
            &(&p2 - &p1),
            &(&p3 - &p1),
        );
        let vec_p1_to_point = sample_point - &p1;
        let point_to_projection = &projected - &vec_p1_to_point;
        let medain_length = vec_norm_1d(&median.view());
        let weight = vec_norm_1d(&(point_to_projection / medain_length).view() );
        weights[Ix1(nearby_pnt_id)] = weight;
        
        // Shift all nearby point references down
        let tmp = p1;
        p1 = p2;
        p2 = p3;
        p3 = tmp;
    }
    return weights
}

pub fn linear_interp_weights_triangles(
    sample_points: &ArrayView2<f64>,
    nearby_val_locs: &ArrayView3<f64>
) -> Array2<f64> {
    let nr_points = sample_points.shape()[0];
    let mut weights = Array2::<f64>::zeros((nr_points, 3));
    for sample_pnt_id in 0..nr_points {
        let w = linear_interp_weights_single_triangle(
            &sample_points.slice(s![sample_pnt_id, ..]),
            &nearby_val_locs.slice(s![sample_pnt_id, .., ..])
        );
        weights.slice_mut(s![sample_pnt_id, ..]).assign(&w);
    }
    return weights
}