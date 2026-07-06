"""a gaussian mixture likelihood in n dimensions with 2 unequal modes at +/-5"""
import numpy as np
from numba import njit

from DTMCMC.likelihood import RectangularLikelihood, check_bounds_rectangular

#constants
low_lim = -10.
high_lim = 10.


@njit()
def get_loglike(v,n_par):
    """Get likelihood for gaussian mixture"""
    res1 = np.log(1. /(3*(2. * np.pi)**(n_par/2)))
    res2 = np.log(2. /(3*(2. * np.pi)**(n_par/2)))
    for itrp in range(n_par):
        res1 += -(v[itrp]-5)**2/2
        res2 += -(v[itrp]+5)**2/2

    res = np.logaddexp(res1,res2)

    return res


class GaussianMixtureLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self,n_par=50) -> None:
        """Create the class and store any object specific variables"""
        low_lims = np.full(n_par, low_lim)
        high_lims = np.full(n_par, high_lim)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    def get_loglike(self,params_in):
        """Get the log likelihood given a set of parameters v"""
        return get_loglike(params_in,self.n_par)


@njit()
def gen_draws(n_draws,n_par,attempt_lim=10000):
    """Get posterior draws"""
    draws = np.zeros((n_draws,n_par))
    for itrk in range(n_draws):
        itra = 0
        mode_choose = np.random.uniform(0.,1.)
        if mode_choose<1./3.:
            draw_loc = np.random.normal(5,1,n_par)
        else:
            draw_loc = np.random.normal(-5,1,n_par)

        while not check_bounds_rectangular(draw_loc, np.full(n_par, low_lim), np.full(n_par, high_lim)):
            if itra==attempt_lim:
                msg = 'Failed to find valid posterior point.'
                raise RuntimeError(msg)

            #redraw if it doesn't fit
            mode_choose = np.random.uniform(0.,1.)
            if mode_choose<1./3.:
                draw_loc = np.random.normal(5,1,n_par)
            else:
                draw_loc = np.random.normal(-5,1,n_par)
            itra += 1
        draws[itrk] = draw_loc
    return draws
