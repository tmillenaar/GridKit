import numpy


class _IndexMeta(type):
    def __new__(cls, name, bases, namespace):
        # numpys with a nan-base
        for op, as_idx in (
            (numpy.add, False),
            (numpy.subtract, False),
            (numpy.multiply, False),
            (numpy.true_divide, False),
            (numpy.floor_divide, False),
            (numpy.power, False),
            (numpy.mod, False),
            (numpy.greater_equal, True),
            (numpy.less_equal, True),
            (numpy.greater, True),
            (numpy.less, True),
        ):
            opname = "__{}__".format(op.__name__)
            opname_reversed = "__r{}__".format(op.__name__)
            namespace[opname] = op
            namespace[opname_reversed] = op
        return super().__new__(cls, name, bases, namespace)


class GridIndex(metaclass=_IndexMeta):
    def __init__(self, index):
        self.index = numpy.asarray(index)

    def unique(self, **kwargs):
        return numpy.unique(self.index, axis=0, **kwargs)

    def intersection(self, other):
        if not isinstance(other, GridIndex):
            other = GridIndex(other)
        intersection = numpy.intersect1d(self._1d_view, other._1d_view)
        return _nd_view(intersection)

    def difference(self, other):
        if not isinstance(other, GridIndex):
            other = GridIndex(other)
        difference = numpy.setdiff1d(self._1d_view, other._1d_view)
        return _nd_view(difference)

    def isdisjoint(self, other):
        return ~numpy.isin(self._1d_view, other._1d_view).any()

    def issubset(self, other):
        return numpy.isin(self._1d_view, other._1d_view).all()

    def issuperset(self, other):
        return numpy.isin(other._1d_view, self._1d_view).all()

    def __array__(self, dtype=None):
        return self.index

    @property
    def _1d_view(self):
        """Create a structured array where each (x,y) pair is seen as a single entitiy"""
        index = self.index
        formats = (
            numpy.full(len(index), index.dtype)
            if index.shape[0] > 1
            else 2 * [index.dtype]
        )
        dtype = {"names": ["f0", "f1"], "formats": formats}
        return index.view(dtype)

    # TODO: standard numpys https://docs.python.org/3/library/operator.html


def _nd_view(index):
    """Turn 1d-view into ndarray"""
    if index.shape[0] == 0:  # return index if empty
        return index
    return index.view(int).reshape(-1, 2)


def validate_index(func):
    def wrapper(index, *args, **kwargs):
        if not isinstance(index, GridIndex):
            index = GridIndex(index)
        return func(index, *args, **kwargs)

    return wrapper
