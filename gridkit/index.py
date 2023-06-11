import numpy

class GridIndex():

    def __init__(self, index):

        self.index = numpy.asarray(index)

    def unique(self, **kwargs):
        return numpy.unique(self.index, axis=0, **kwargs)

    def intersection(self, other):
        if not isinstance(other, GridIndex):
            other = GridIndex(other)
        intersection = numpy.intersect1d(self, other)
        return _nd_view(intersection)

    def difference(self, other):
        if not isinstance(other, GridIndex):
            other = GridIndex(other)
        difference = numpy.setdiff1d(self, other)
        return _nd_view(difference)

    def isdisjoint(self, other):
        return ~numpy.isin(self, other).any()

    def issubset(self, other):
        return numpy.isin(self, other).all()

    def issuperset(self, other):
        return numpy.isin(other, self).all()

    def __array__(self, dtype=None):
        return self._1d_view

    @property
    def _1d_view(self):
        """Create a structured array where each (x,y) pair is seen as a single entitiy"""
        index = self.index
        formats = numpy.full(len(index), index.dtype) if index.shape[0] > 1 else 2*[index.dtype]
        dtype={'names':['f0', 'f1'], 'formats':formats}
        return index.view(dtype)

    # TODO: standard operators https://docs.python.org/3/library/operator.html

def _nd_view(index):
    """Turn 1d-view into ndarray"""
    if index.shape[0] == 0: # return index if empty
        return index
    return index.view(int).reshape(-1, 2)
