use numpy::ndarray::*;
use geo::{MultiPolygon, Polygon, LineString, Point, Coord};
use wkb;
use geo_types;


pub fn coords_to_polygon(
    coords: &ArrayView2<f64>,
) -> Vec<u8> {
    let mut points: Vec<Coord<f64>> = Vec::new();

    // Iterate over points
    for point_index in 0..coords.shape()[0] {
        // Replace this with your actual coordinates from the ndarray
        let x = coords[Ix2(point_index, 0)];
        let y = coords[Ix2(point_index, 1)];
        points.push(Point::new(x, y).into());
    }

    // Create a Polygon from the points and add it to the list
    let polygon = geo_types::Polygon::new(LineString(points), vec![]);
    let wkb_geometry: geo_types::Geometry<f64> = polygon.into();
    return wkb::geom_to_wkb(&wkb_geometry).unwrap();
}