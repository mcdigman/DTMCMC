"""The ar(1) likelihood in n dimensions"""
import numpy as np
from numba import njit

from DTMCMC.likelihood import RectangularLikelihood, check_bounds_rectangular

#constants
low_lim = -10.
high_lim = 10.
alpha = 0.9
beta = np.sqrt(1-alpha**2)
const = np.log(1. / np.sqrt(2. * np.pi*beta**2))  # normalization constant


@njit()
def get_loglike(v,n_par):
    """Get the likelihood for ar1"""
    res = 0.
    x_next = 0.
    res += const-1/2*(v[0]-x_next)**2
    x_next = alpha*v[0]

    for itrp in range(1,n_par):
        res += const-1/2/beta**2*(v[itrp]-x_next)**2
        x_next = alpha*v[itrp]
    return res


class Ar1Likelihood(RectangularLikelihood):
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
        draw_loc = np.zeros(n_par)
        draw_loc[0] = np.random.normal(0.,1.)
        for itrp in range(1,n_par):
            n1 = np.random.normal(alpha*draw_loc[itrp-1],beta)
            draw_loc[itrp] = n1

        while not check_bounds_rectangular(draw_loc, np.full(n_par, low_lim), np.full(n_par, high_lim)):
            if itra==attempt_lim:
                msg = 'Failed to find valid posterior point.'
                raise RuntimeError(msg)

            draw_loc = np.zeros(n_par)
            draw_loc[0] = np.random.normal(0.,1.)
            for itrp in range(1,n_par):
                n1 = np.random.normal(alpha*draw_loc[itrp-1],beta)
                draw_loc[itrp] = n1
            itra += 1
        draws[itrk] = draw_loc
    return draws
