import matplotlib.pyplot as plt
import numpy
import pytest
import shapely

from gridkit import doc_utils


def test_generate_2d_scatter_doughnut():
    radius = 2.5
    points = doc_utils.generate_2d_scatter_doughnut(1000, radius)
    median_norm = numpy.median([numpy.linalg.norm(p) for p in points])
    numpy.testing.assert_allclose(median_norm, radius, atol=0.3)


@pytest.mark.parametrize("filled", [True, False])
@pytest.mark.parametrize(
    "colors",
    [
        lambda nr: nr * ["brown"],
        lambda nr: "brown",
        lambda nr: numpy.arange(nr),
    ],
)
def test_plot_polygons(filled, colors):
    nr_shapes = 5

    ax = plt.subplot()
    point = shapely.Point(1, 2)
    poly = point.buffer(0.5)

    colors = colors(nr_shapes)
    doc_utils.plot_polygons(
        nr_shapes * [poly], colors=colors, cmap="viridis", ax=ax, filled=filled
    )

    # check if shapes have been added
    if filled:
        artists = ax.get_children()
        polygons = [a for a in artists if isinstance(a, plt.Polygon)]
        assert len(polygons) == nr_shapes
    else:
        assert len(ax.lines) == nr_shapes
    # clear the plot in preparation for the next test
    plt.clf()
