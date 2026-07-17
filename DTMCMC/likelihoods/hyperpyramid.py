"""the 2D hyper-pyramid lkelihood, adapted from https://github.com/joshspeagle/dynesty/blob/master/demos/"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import RectangularInputs, RectangularLikelihood
from DTMCMC.numba_backend import NativeLoglikeCall

# constants
low_lim: float = -15.0
high_lim: float = 15.0
s: float = 0.5
sigma: float = 1.0
center: float = 0.0


@njit()
def get_loglike(x: NDArray[np.floating]) -> float:
    """Get the likelihood"""
    res: float = -(max(np.abs((x - center) / sigma)) ** (1.0 / s))
    return res


@njit(inline='always')
def _loglike_native(params_in: NDArray[np.floating], _state: RectangularInputs) -> float:
    """Per-class native log likelihood; the pyramid needs no instance state."""
    return get_loglike(params_in)


class HyperpyramidLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""

    def __init__(self, n_par: int = 2) -> None:
        """Create the class and store any object specific variables"""
        if n_par != 2:
            msg = 'HyperpyramidLikelihood is 2D; n_par must be 2'
            raise ValueError(msg)
        low_lims = np.full(n_par, low_lim)
        high_lims = np.full(n_par, high_lim)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Get the log likelihood given a set of parameters v"""
        return get_loglike(params_in)

    def bind_native_loglike(self) -> NativeLoglikeCall[RectangularInputs]:
        """Return the per-class native log likelihood."""
        return _loglike_native
