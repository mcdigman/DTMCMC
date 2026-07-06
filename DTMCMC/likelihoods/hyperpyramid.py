"""the 2D hyper-pyramid lkelihood, adapted from https://github.com/joshspeagle/dynesty/blob/master/demos/"""
import numpy as np
from numba import njit

from DTMCMC.likelihood import RectangularLikelihood

#constants
low_lim = -15.
high_lim = 15.
s, sigma = 0.5, 1.
center = 0.


@njit()
def get_loglike(x):
    """Get the likelihood"""
    return -max(np.abs((x - center) / sigma))**(1. / s)


class HyperpyramidLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self) -> None:
        """Create the class and store any object specific variables"""
        n_par = 2
        low_lims = np.full(n_par, low_lim)
        high_lims = np.full(n_par, high_lim)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    def get_loglike(self,params_in):
        """Get the log likelihood given a set of parameters v"""
        return get_loglike(params_in)
