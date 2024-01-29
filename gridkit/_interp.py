import numpy


# FIXME: move to Rust
def get_linear_weight_triangle(point, p1, p2, p3):
    """Get the weight for p1 when linearly interpolating
    at point p inside a triangle.

    Parameters
    ----------
    point: `numpy.ndarray`
        A numpy array containing the x and y value of the desired location
    p1: `numpy.ndarray`
        The corner on the triangle of which the weight is desired
    p2: `numpy.ndarray`
        The second corner on the triangle
    p3: `numpy.ndarray`
        The third corner on the triangle

    Returns
    -------
    `float`
        The weight of p1 when linearly interpolating at location point
    """

    def _project(point, line_points):
        """Project 'point' onto a line drawn between 'line_points[0]' and 'line_points[1]'
        This solution is based on a dot product between the line and the line between
        the point and one of the line_points.
        Alternative solution: https://stackoverflow.com/a/61343727/22128453
        """
        vec_line = line_points[1] - line_points[0]
        vec_point = point - line_points[0]
        line_length = numpy.linalg.norm(vec_line)
        projected_factor = numpy.dot(vec_line / line_length, vec_point / line_length)
        projected = projected_factor * vec_line + line_points[0]

        return projected

    side_length = numpy.linalg.norm(p1 - p2)
    # Determine the median line that divides a triangle in two
    midpoint_opposite_side = p2 + (p2 - p3) / 2
    median = midpoint_opposite_side - p1
    # The median line should be shorter than the side length
    # If it is longer we need to be on the other side of p2
    if numpy.linalg.norm(median) > side_length:
        midpoint_opposite_side = p2 - (p2 - p3) / 2
        median = midpoint_opposite_side - p1

    projected = _project(point - p1, [p3 - p1, p2 - p1])

    return numpy.linalg.norm((projected - (point - p1)) / numpy.linalg.norm(median))
