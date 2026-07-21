"""Unit tests for ``DTMCMC.likelihood._freeze``.

``_freeze`` turns a baked ``inputs`` constant into a hashable, value-equal
surrogate used to key the compiled-handle memo. These tests exercise each
branch (ndarray, tuple recursion, hashable leaf, unhashable rejection) and
confirm that every component of the ndarray key can independently distinguish
two otherwise-identical arrays.
"""

from typing import NamedTuple

import numpy as np
import pytest
from numba import njit

from DTMCMC.likelihood import _freeze


class Unhashable:
    """Module-level unhashable type, so its ``__qualname__`` is stable."""

    __hash__ = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# ndarray branch                                                              #
# --------------------------------------------------------------------------- #


def test_ndarray_branch_returns_expected_surrogate() -> None:
    """An ndarray freezes to (shape, dtype.str, byteorder, C-flag, F-flag, bytes)."""
    arr = np.arange(6.0).reshape(2, 3)
    expected = (
        arr.shape,
        arr.dtype.str,
        arr.dtype.byteorder,
        arr.flags['C_CONTIGUOUS'],
        arr.flags['F_CONTIGUOUS'],
        arr.tobytes(),
    )
    assert _freeze(arr) == expected


def test_ndarray_surrogate_is_hashable() -> None:
    """The whole point: the surrogate can key a dict where the array cannot."""
    frozen = _freeze(np.arange(4))
    assert {frozen: 1}[frozen] == 1


def test_equal_arrays_freeze_equal_distinct_arrays_do_not() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    c = np.array([1.0, 2.0, 4.0])
    assert _freeze(a) == _freeze(b)
    assert _freeze(a) != _freeze(c)


# --------------------------------------------------------------------------- #
# each ndarray key component can independently "bite"                         #
# --------------------------------------------------------------------------- #


def test_shape_component_bites() -> None:
    """Same bytes and dtype, different shape -> distinct keys."""
    a = np.arange(6).reshape(2, 3)
    b = np.arange(6).reshape(3, 2)
    assert a.tobytes() == b.tobytes()
    assert a.dtype == b.dtype
    assert _freeze(a) != _freeze(b)


def test_dtype_component_bites() -> None:
    """Same bytes and shape, different dtype -> distinct keys."""
    a = np.array([1, 2, 3], dtype=np.int32)
    b = a.view(np.float32)
    assert a.tobytes() == b.tobytes()
    assert a.shape == b.shape
    assert _freeze(a) != _freeze(b)


def test_c_contiguity_component_bites() -> None:
    """Same shape/dtype/bytes, differ only in C_CONTIGUOUS -> distinct keys."""
    strided = np.arange(12).reshape(4, 3)[::2]  # C=False, F=False
    c_contig = np.array([[0, 1, 2], [6, 7, 8]])  # C=True,  F=False
    assert strided.tobytes() == c_contig.tobytes()
    assert strided.shape == c_contig.shape
    assert strided.dtype == c_contig.dtype
    assert strided.flags['F_CONTIGUOUS'] == c_contig.flags['F_CONTIGUOUS']
    assert strided.flags['C_CONTIGUOUS'] != c_contig.flags['C_CONTIGUOUS']
    assert _freeze(strided) != _freeze(c_contig)


def test_f_contiguity_component_bites() -> None:
    """Same shape/dtype/bytes and C-flag, differ only in F_CONTIGUOUS -> distinct keys."""
    strided = np.arange(12).reshape(4, 3)[::2]  # C=False, F=False
    f_contig = np.asfortranarray([[0, 1, 2], [6, 7, 8]])  # C=False, F=True
    assert strided.tobytes() == f_contig.tobytes()
    assert strided.shape == f_contig.shape
    assert strided.dtype == f_contig.dtype
    assert strided.flags['C_CONTIGUOUS'] == f_contig.flags['C_CONTIGUOUS']
    assert strided.flags['F_CONTIGUOUS'] != f_contig.flags['F_CONTIGUOUS']
    assert _freeze(strided) != _freeze(f_contig)


def test_contents_component_bites() -> None:
    """Same shape/dtype/layout, different bytes -> distinct keys."""
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 4])
    assert _freeze(a) != _freeze(b)


def test_byteorder_differing_arrays_get_distinct_keys() -> None:
    """Little- vs big-endian arrays never collide.

    NOTE: the standalone ``dtype.byteorder`` component is redundant with
    ``dtype.str`` -- numpy resolves native order into ``dtype.str`` and only
    exposes a non-native byte order, so whenever ``byteorder`` differs
    ``dtype.str`` (and the raw bytes) already differ. This test therefore
    asserts the outcome that matters (distinct keys), not that ``byteorder``
    bites in isolation, which is not achievable.
    """
    little = np.array([1, 2, 3], dtype='<i4')
    big = np.array([1, 2, 3], dtype='>i4')
    assert _freeze(little) != _freeze(big)


# --------------------------------------------------------------------------- #
# tuple branch / recursion                                                    #
# --------------------------------------------------------------------------- #


def test_tuple_branch_recurses_into_elements() -> None:
    arr = np.arange(3)
    frozen = _freeze((1, arr, 'x'))
    assert frozen == (1, _freeze(arr), 'x')
    assert {frozen: 0}[frozen] == 0  # hashable end-to-end


def test_nested_structure_recursion() -> None:
    """Recursion reaches arrays nested arbitrarily deep inside tuples."""
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0])
    nested_a = (0, (a, (2, a)))
    nested_b = (0, (b, (2, b)))
    nested_c = (0, (b, (2, np.array([9.0, 9.0]))))
    assert _freeze(nested_a) == _freeze(nested_b)
    assert _freeze(nested_a) != _freeze(nested_c)


def test_namedtuple_inputs_recursion() -> None:
    """A NamedTuple (the real ``inputs`` shape) freezes hashably."""

    class Inputs(NamedTuple):
        n_par: int
        low_lims: np.ndarray
        high_lims: np.ndarray

    a = Inputs(2, np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
    b = Inputs(2, np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
    c = Inputs(2, np.array([-5.0, -5.0]), np.array([9.0, 5.0]))
    assert _freeze(a) == _freeze(b)
    assert _freeze(a) != _freeze(c)
    assert {_freeze(a): 1}[_freeze(b)] == 1


# --------------------------------------------------------------------------- #
# hashable-leaf branch (incl. jit-compiled handle)                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('value', [0, 1.5, 'text', None, True, (), (1, 2)])
def test_hashable_leaf_returned_unchanged(value: object) -> None:
    assert _freeze(value) == value


def test_jit_compiled_handle_is_treated_as_hashable_leaf() -> None:
    """A numba CPUDispatcher is hashable, so it passes through untouched."""

    @njit
    def handle(x: float) -> float:
        return x + 1.0

    assert _freeze(handle) is handle


# --------------------------------------------------------------------------- #
# unhashable rejection branch                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ('value', 'name'),
    [
        ([1, 2, 3], 'list'),
        ({'a': 1}, 'dict'),
        ({1, 2}, 'set'),
        (Unhashable(), 'Unhashable'),
    ],
)
def test_unhashable_leaf_raises_typeerror_naming_the_type(value: object, name: str) -> None:
    with pytest.raises(TypeError, match=name):
        _freeze(value)


def test_unhashable_nested_inside_tuple_is_reported() -> None:
    """Recursion surfaces an unhashable buried inside a tuple, naming its type."""
    with pytest.raises(TypeError, match='list'):
        _freeze((1, [2, 3]))
