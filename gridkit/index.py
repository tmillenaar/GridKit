import inspect
from typing import Union

import numpy


class _IndexMeta(type):
    """Metaclass for GridIndex which implements the basic operators"""

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
    """Index to be used with grid classes.
    These GridIndex class instances contain the references to the grid cells by ID (idx, idy).
    Several methods are implemented to treat the indices as sets, where one (idx, idy) pair is considered one unit, one cell-ID.
    The GridIndex class also allows for generic operators such as: (+, -, *, /, //, **, %, >=, <=, >, <)

    Parameters
    ----------
    index: Union[numpy.ndarray, list, tuple, GridIndex]
        The index containing the cell-id.
        This is assumed to either be a single (idx, idy) pair or a list, tuple or ndarray containing multiple of such pairs.

    Raises
    ------
    ValueError
        When the shape of the index does not match the expected shape of (2,) or (N,2) where N is the number of cells, a ValuError is raised.
    """

    def __init__(self, index):
        self.index = numpy.asarray(index)

        if self.index.shape == (2,):
            self.index = self.index[numpy.newaxis]

        if self.index.ndim > 2 or (self.index.ndim > 1 and self.index.shape[-1] != 2):
            raise ValueError(
                f"Unexpected index shape. Expected a shape of (2,) or (N, 2), got {self.index.shape}."
            )

    def __len__(self):
        """The number of indices"""
        return len(self._1d_view)

    @property
    def x(self):
        """The X-component of the cell-IDs"""
        return self.index[:, 0]

    @property
    def y(self):
        """The Y-component of the cell-IDs"""
        return self.index[:, 1]

    def unique(self, **kwargs):
        """The unique IDs contained in the index. Remove duplicate IDs.

        Parameters
        ----------
        **kwargs:
            The keyword arguments to pass to numpy.unique
        """
        return numpy.unique(self.index, axis=0, **kwargs)

    def intersection(self, other):
        """The intersection of two GridIndex instances. Keep the IDs contained in both.

        Parameters
        ----------
        **other:
            The GridIndex instance to compare with
        """
        if not isinstance(other, GridIndex):
            other = GridIndex(other)
        intersection = numpy.intersect1d(self._1d_view, other._1d_view)
        return _nd_view(intersection)

    def difference(self, other):
        """The differenceof two GridIndex instances. Keep the IDs contained in ``self`` that are not in ``other``.

        Parameters
        ----------
        **other:
            The GridIndex instance to compare with
        """
        if not isinstance(other, GridIndex):
            other = GridIndex(other)
        difference = numpy.setdiff1d(self._1d_view, other._1d_view)
        return _nd_view(difference)

    def isdisjoint(self, other):
        """True if none of the IDs in ``self`` are in ``other``. False if any ID in ``self`` is also in ``other``.

        Parameters
        ----------
        **other:
            The GridIndex instance to compare with
        """
        return ~numpy.isin(self._1d_view, other._1d_view).any()

    def issubset(self, other):
        """True if all of the IDs in ``self`` are in ``other``. False if not all IDs in ``self`` is also in ``other``.

        Parameters
        ----------
        **other:
            The GridIndex instance to compare with
        """
        return numpy.isin(self._1d_view, other._1d_view).all()

    def issuperset(self, other):
        """True if all of the IDs in ``other`` are in ``self``. False if not all IDs in ``other`` is also in ``self``.

        Parameters
        ----------
        **other:
            The GridIndex instance to compare with
        """
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

    def copy(self):
        """Return an immutable copy of self."""
        return GridIndex(self.index.copy())


def _nd_view(index):
    """Turn 1d-view into ndarray"""
    if index.shape[0] == 0:  # return index if empty
        return index
    return index.view(int).reshape(-1, 2)


def validate_index(func):
    """Decorator to convert the index argument of a function to a GridIndex object."""

    def wrapper(*args, **kwargs):
        """Inner function to convert the index argument to a GridIndex object.

        Parameters
        ----------
        index: Union[numpy.ndarray, list, tuple, GridIndex]
            The index referring to the grid IDs
        *args:
            The arguments to be passed to the wrapped function
        *kwargs:
            The keyword arguments to be passed to the wrapped function
        """
        arg_names = inspect.signature(func).parameters
        new_args = []
        for key, value in zip(arg_names, args):
            if key == "index" and value is not None:
                value = GridIndex(value)
            new_args.append(value)
        new_kwargs = {}
        for key, value in kwargs.items():
            if key == "index" and value is not None:
                value = GridIndex(value)
            new_kwargs[key] = value

        return func(*new_args, **new_kwargs)

    return wrapper
