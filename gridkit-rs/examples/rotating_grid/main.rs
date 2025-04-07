use bresenham::Bresenham;
use gridkit::*;
use ndarray::*;

fn draw_hexagon(canvas: &mut Vec<Vec<char>>, points: &Array3<f64>) {
    for cell_id in 0..points.shape()[0] {
        for vertex_id in 0..points.shape()[1] {
            // Add canvas width so we can center around 0,0, or add acnchor to rust version
            let x0 = points[Ix3(cell_id, vertex_id, 0)];
            let y0 = points[Ix3(cell_id, vertex_id, 1)];
            let mut next_vertex_id = vertex_id + 1;
            if next_vertex_id == points.shape()[1] {
                next_vertex_id = 0;
            }
            let x1 = points[Ix3(cell_id, next_vertex_id, 0)];
            let y1 = points[Ix3(cell_id, next_vertex_id, 1)];
            for (x, y) in Bresenham::new((x0 as isize, y0 as isize), (x1 as isize, y1 as isize)) {
                if y >= 0 && y < canvas.len() as isize && x >= 0 && x < canvas[0].len() as isize {
                    canvas[y as usize][x as usize] = 'o';
                }
            }
        }
    }
}

fn print_canvas(canvas: &[Vec<char>]) {
    for row in canvas.iter().rev() {
        println!("{}", row.iter().collect::<String>());
    }
}
fn main() {
    let grid = gridkit::TriGrid::new(10., Orientation::Flat);

    let point: Array2<f64> = Array2::from_shape_vec((1, 2), vec![42., 25.]).unwrap();
    let center_cell = grid.cell_at_point(&point.view()); // Fixme do a cell_at_point_xy
    let neighbours = grid.direct_neighbours(&center_cell.view(), 3, false, true);
    let neighbour_shape = neighbours.shape();
    let raveled_neighbours = neighbours
        .clone() // FIXME
        .into_shape((neighbour_shape[0] * neighbour_shape[1], 2))
        .expect("Unable to ravel neighbours"); // Ravel cells
    let corners = grid.cell_corners(&raveled_neighbours.view());

    let mut canvas = vec![vec![' '; 100]; 50];
    draw_hexagon(&mut canvas, &corners);
    print_canvas(&canvas);
}
