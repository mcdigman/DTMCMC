"""the 2D hyper-pyramid lkelihood, adapted from https://github.com/joshspeagle/dynesty/blob/master/demos/"""

from typing import override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import LoglikeFn, RectangularLikelihood

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

    @override
    def _make_loglike(self) -> LoglikeFn:
        return get_loglike
