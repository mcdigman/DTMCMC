"""the 2D hyper-pyramid lkelihood, adapted from https://github.com/joshspeagle/dynesty/blob/master/demos/"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import RectangularLikelihood

# constants
low_lim = -15.0
high_lim = 15.0
s, sigma = 0.5, 1.0
center = 0.0


@njit()
def get_loglike(x: NDArray[np.floating]) -> float:
    """Get the likelihood"""
    return -(max(np.abs((x - center) / sigma)) ** (1.0 / s))


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
