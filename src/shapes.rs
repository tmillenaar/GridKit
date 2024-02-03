use numpy::ndarray::*;
use geo::{MultiPolygon, Polygon, LineString, Point, Coord};
use wkb::geom_to_wkb;
use geo_types;


pub fn coords_to_multipolygon_wkb(
    coords: &ArrayView3<f64>,
) -> Vec<u8> {
    let mut polygons: Vec<geo_types::Polygon<f64>> = Vec::new();
    // Iterate over points
    for poly_index in 0..coords.shape()[0] {
        let mut points: Vec<Coord<f64>> = Vec::new();
        for point_index in 0..coords.shape()[1] {
            // Replace this with your actual coordinates from the ndarray
            let x = coords[Ix3(poly_index, point_index, 0)];
            let y = coords[Ix3(poly_index, point_index, 1)];
            points.push(Point::new(x, y).into());
        }
        let polygon = geo_types::Polygon::new(LineString(points), vec![]);
        polygons.push(polygon);
    }

    // Create a Polygon from the points and add it to the list
    let multi_poly = geo_types::MultiPolygon(polygons);
    let wkb_geometry: geo_types::Geometry<f64> = multi_poly.into();
    return wkb::geom_to_wkb(&wkb_geometry).unwrap();
}