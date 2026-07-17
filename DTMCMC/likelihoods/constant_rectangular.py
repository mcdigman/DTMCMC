"""A constant likelihood over a rectangular uniform prior.

The posterior equals the box prior exactly, which makes this the reference
target for prior-recovery review runs: a run with any other likelihood and
``zero_loglike=True`` over the same bounds has the same target. Proposal
internals may still use that run's original likelihood by design.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import RectangularInputs, RectangularLikelihood
from DTMCMC.numba_backend import NativeLoglikeCall

# default rectangular bounds, borrowed from the banana likelihood's
# n >= 2 dimensions; override per instance via the constructor
low_lim: float = -100.0
high_lim: float = 100.0


@njit(inline='always')
def _loglike_native(_params_in: NDArray[np.floating], _state: RectangularInputs) -> float:
    """Return the constant log likelihood in the native kernel."""
    return 0.0


class ConstantRectangularLikelihood(RectangularLikelihood):
    """Identically-zero log likelihood over standard rectangular bounds."""

    def __init__(
        self,
        n_par: int = 4,
        low_lims: NDArray[np.floating] | None = None,
        high_lims: NDArray[np.floating] | None = None,
    ) -> None:
        """Create the class, defaulting to the module-level bounds in every dimension."""
        low_arr = np.full(n_par, low_lim) if low_lims is None else low_lims.copy()
        high_arr = np.full(n_par, high_lim) if high_lims is None else high_lims.copy()

        RectangularLikelihood.__init__(self, n_par, low_arr, high_arr)

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Return a constant log likelihood over the whole parameter space."""
        del params_in
        return 0.0

    def bind_native_loglike(self) -> NativeLoglikeCall[RectangularInputs]:
        """Return the per-class native zero log likelihood."""
        return _loglike_native
