use numpy::ndarray::*;
use wkb::geom_to_wkb;
use geo_types::{MultiPolygon, Polygon, LineString, Point, Coord, Geometry};

pub fn coords_to_multipolygon_wkb(
    coords: &ArrayView3<f64>,
) -> Vec<u8> {
    let polygons: Vec<Polygon<f64>> = (0..coords.shape()[0])
        .map(|poly_index| {
            let points: Vec<Coord<f64>> = (0..coords.shape()[1])
                .map(|point_index| {
                    let x = coords[Ix3(poly_index, point_index, 0)];
                    let y = coords[Ix3(poly_index, point_index, 1)];
                    Point::new(x, y).into()
                })
                .collect();

            Polygon::new(LineString(points), vec![])
        })
        .collect();

    let multi_polygon = MultiPolygon(polygons);
    let wkb_geometry: Geometry<f64> = multi_polygon.into();
    wkb::geom_to_wkb(&wkb_geometry).unwrap()
}
