"""Regression guards for the owner half of the compiled-handle memo key.

``AbstractNativeLikelihood.__init__`` memoizes compiled handles on
``(owner, role, _freeze(inputs))``. ``owner`` must be the class object itself:
keying it on ``type(self).__qualname__`` let two distinct likelihood classes
that share a name across modules — with value-equal ``inputs`` — silently reuse
the first one's handles, so the second sampler targeted the wrong distribution
with no warning. ``tests/test_freeze.py`` covers the ``inputs`` half of the same
key; these tests pin both halves of the owner contract: distinct classes never
alias, and instances of one class still share a single compiled handle so
structurally identical samplers keep sharing one kernel program.
"""

from typing import override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import NativeLoglikeCall, RectangularInputs, RectangularLikelihood, _freeze

_N_PAR = 2
_LOW_LIMS = np.full(_N_PAR, -1.0)
_HIGH_LIMS = np.full(_N_PAR, 1.0)


@njit(inline='always')
def _alpha_loglike_native(_params_in: NDArray[np.floating], _inputs: RectangularInputs) -> float:
    return 1.0


@njit(inline='always')
def _beta_loglike_native(_params_in: NDArray[np.floating], _inputs: RectangularInputs) -> float:
    return 99.0


class _MemoOwnerAlpha(RectangularLikelihood[RectangularInputs]):
    """Flat likelihood pinned at 1.0."""

    def __init__(self) -> None:
        RectangularLikelihood.__init__(self, _N_PAR, _LOW_LIMS, _HIGH_LIMS)

    @property
    @override
    def loglike_fn(self) -> NativeLoglikeCall[RectangularInputs]:
        return _alpha_loglike_native


class _MemoOwnerBeta(RectangularLikelihood[RectangularInputs]):
    """Flat likelihood pinned at 99.0, otherwise identical to the alpha class."""

    def __init__(self) -> None:
        RectangularLikelihood.__init__(self, _N_PAR, _LOW_LIMS, _HIGH_LIMS)

    @property
    @override
    def loglike_fn(self) -> NativeLoglikeCall[RectangularInputs]:
        return _beta_loglike_native


# Two likelihood classes defined in separate modules can share a __qualname__.
# Forcing the collision here reproduces that without building throwaway modules;
# the bounds are value-equal too, so the qualname was the whole of the old key.
_MemoOwnerBeta.__qualname__ = _MemoOwnerAlpha.__qualname__


def test_classes_sharing_a_qualname_do_not_share_compiled_handles() -> None:
    """The collision that made a sampler silently target the wrong distribution."""
    alpha = _MemoOwnerAlpha()
    beta = _MemoOwnerBeta()

    # the two halves of the old key really do collide, so the memo is forced to
    # separate them on the class object alone
    assert type(alpha).__qualname__ == type(beta).__qualname__
    assert _freeze(alpha.inputs) == _freeze(beta.inputs)

    params = np.zeros(_N_PAR)
    assert alpha.loglike_fn_baked(params) == 1.0
    assert beta.loglike_fn_baked(params) == 99.0
    assert alpha.loglike_fn_baked is not beta.loglike_fn_baked


def test_instances_of_one_class_share_every_compiled_handle() -> None:
    """The sharing the memo exists for: one compiled handle per (class, role, inputs)."""
    first = _MemoOwnerAlpha()
    second = _MemoOwnerAlpha()

    # prior_draw is keyed in _HANDLE_MEMO_NO_PARAMS, the rest in _HANDLE_MEMO_PARAMS
    assert first.prior_draw_fn_baked is second.prior_draw_fn_baked
    assert first.loglike_fn_baked is second.loglike_fn_baked
    assert first.prior_factor_fn_baked is second.prior_factor_fn_baked
    assert first.check_bounds_fn_baked is second.check_bounds_fn_baked
    assert first.correct_bounds_fn_baked is second.correct_bounds_fn_baked
    assert first.validate_bounds_fn_baked is second.validate_bounds_fn_baked
    assert first.prior_proposal_fn_baked is second.prior_proposal_fn_baked
