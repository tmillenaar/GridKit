import numpy
from index import GridIndex

def test_unique():
    index_raw = [(1,2), (2,1), (1,1), (1,2)]
    expected_unique = [(1,1), (1,2), (2,1)]
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
    index1 = GridIndex([(1,2), (2,1), (1,1), (1,2)])
    index2 = GridIndex([(1,1), (1,2), (0,1)])
    expected_intersection = [(1,1), (1,2)]
    intersection = index1.intersection(index2)
    numpy.testing.assert_allclose(intersection, expected_intersection)

def test_difference():
    index1 = GridIndex([(1,2), (2,1), (1,1), (1,2)])
    index2 = GridIndex([(1,1), (1,2), (0,1)])

    # test index1 diff index2
    difference = index1.difference(index2)
    numpy.testing.assert_allclose(difference, [(2,1)])

    # test index2 diff index1
    difference = index2.difference(index1)
    numpy.testing.assert_allclose(difference, [(0,1)])


def test_isdisjoint():
    index1 = GridIndex([(1,2), (2,1), (1,1), (1,2)])
    index2 = GridIndex([(1,1), (1,2), (0,1)])
    index3 = GridIndex([(0,0)])

    isdisjoint = index1.isdisjoint(index2)
    assert isdisjoint == False

    isdisjoint = index1.isdisjoint(index3)
    assert isdisjoint == True

def test_issubset():
    index1 = GridIndex([(0,1), (1,1)])
    index2 = GridIndex([(1,1), (1,2), (0,1)])

    issubset = index1.issubset(index2)
    assert issubset == True

    issubset = index2.issubset(index1)
    assert issubset == False

def test_issuperset():
    index1 = GridIndex([(0,1), (1,1)])
    index2 = GridIndex([(1,1), (1,2), (0,1)])

    issubset = index1.issuperset(index2)
    assert issubset == False

    issubset = index2.issuperset(index1)
    assert issubset == True
