import operator

import numpy
import pytest

from gridkit.index import GridIndex, validate_index


def test_unique():
    index_raw = [(1, 2), (2, 1), (1, 1), (1, 2)]
    expected_unique = [(1, 1), (1, 2), (2, 1)]
    index = GridIndex(index_raw)

    index_unique = index.unique()
    numpy.testing.assert_allclose(index_unique, expected_unique)

    index_unique, index_ = index.unique(return_index=True)
    numpy.testing.assert_allclose(index_unique, expected_unique)
    numpy.testing.assert_allclose(index_, [2, 0, 1])

    index_unique, inverse = index.unique(return_inverse=True)
    numpy.testing.assert_allclose(index_unique, expected_unique)
    numpy.testing.assert_allclose(inverse, [1, 2, 0, 1])

    index_unique, counts = index.unique(return_counts=True)
    numpy.testing.assert_allclose(index_unique, expected_unique)
    numpy.testing.assert_allclose(counts, [1, 2, 1])


def test_intersection():
    index1 = GridIndex([(1, 2), (2, 1), (1, 1), (1, 2)])
    index2 = GridIndex([(1, 1), (1, 2), (0, 1)])
    expected_intersection = [(1, 1), (1, 2)]
    intersection = index1.intersection(index2)
    numpy.testing.assert_allclose(intersection, expected_intersection)


def test_difference():
    index1 = GridIndex([(1, 2), (2, 1), (1, 1), (1, 2)])
    index2 = GridIndex([(1, 1), (1, 2), (0, 1)])

    # test index1 diff index2
    difference = index1.difference(index2)
    numpy.testing.assert_allclose(difference, [(2, 1)])

    # test index2 diff index1
    difference = index2.difference(index1)
    numpy.testing.assert_allclose(difference, [(0, 1)])


def test_isdisjoint():
    index1 = GridIndex([(1, 2), (2, 1), (1, 1), (1, 2)])
    index2 = GridIndex([(1, 1), (1, 2), (0, 1)])
    index3 = GridIndex([(0, 0)])

    isdisjoint = index1.isdisjoint(index2)
    assert isdisjoint == False

    isdisjoint = index1.isdisjoint(index3)
    assert isdisjoint == True


def test_issubset():
    index1 = GridIndex([(0, 1), (1, 1)])
    index2 = GridIndex([(1, 1), (1, 2), (0, 1)])

    issubset = index1.issubset(index2)
    assert issubset == True

    issubset = index2.issubset(index1)
    assert issubset == False


def test_issuperset():
    index1 = GridIndex([(0, 1), (1, 1)])
    index2 = GridIndex([(1, 1), (1, 2), (0, 1)])

    issubset = index1.issuperset(index2)
    assert issubset == False

    issubset = index2.issuperset(index1)
    assert issubset == True


@pytest.mark.parametrize(
    "op, expected_result",
    [
        (numpy.add, numpy.array([[2, 3], [1, 3]])),
        (numpy.subtract, numpy.array([[-2, -1], [-3, -1]])),
        (numpy.multiply, numpy.array([[0, 2], [-2, 2]])),
        (numpy.true_divide, numpy.array([[0, 0.5], [-0.5, 0.5]])),
        (numpy.floor_divide, numpy.array([[0, 0], [-1, 0]])),
        (numpy.power, numpy.array([[0, 1], [1, 1]])),
        (numpy.mod, numpy.array([[0, 1], [1, 1]])),
        (numpy.greater_equal, numpy.array([[False, False], [False, False]])),
        (numpy.less_equal, numpy.array([[True, True], [True, True]])),
        (numpy.greater, numpy.array([[False, False], [False, False]])),
        (numpy.less, numpy.array([[True, True], [True, True]])),
    ],
)
def test_operator(op, expected_result):
    index = GridIndex([(0, 1), (-1, 1)])

    result = op(index, 2)
    numpy.testing.assert_allclose(result, expected_result)


def test_copy():
    index = GridIndex([(0, 1), (-1, 1)])
    assert id(index.index) != id(index.copy().index)


def test_xy():
    index = GridIndex([(0, 1), (-1, 1)])
    numpy.testing.assert_allclose(index.x, [0, -1])
    numpy.testing.assert_allclose(index.y, [1, 1])


@pytest.mark.parametrize(
    "index",
    (
        [(0, 1)],
        [0, 1],
        ((0, 0), (-1, -2)),
        [[1, 1], [-1, -2]],
        numpy.array([-2, 0]),
        numpy.arange(6).reshape(3, 2),
        GridIndex([0, 1]),
        GridIndex([(0, -1), (-1, -2)]),
    ),
)
def test_validate_index(index):
    @validate_index
    def assert_index(index, *args, **kwargs):
        assert isinstance(index, GridIndex)

    assert_index(index)


def test_rase_shape_error():
    ids = numpy.arange(4 * 3 * 2).reshape(4, 3, 2)
    with pytest.raises(ValueError):
        GridIndex(ids)
