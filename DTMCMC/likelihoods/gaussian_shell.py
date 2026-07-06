"""two shell likelihood, adapted from https://github.com/joshspeagle/dynesty/blob/master/demos/Examples%20--%20Gaussian%20Shells.ipynb"""
import numpy as np
from numba import njit

from DTMCMC.likelihood import RectangularLikelihood, check_bounds_rectangular

#constants
low_lim = -40.
high_lim = 40.

r = 2.  # radius
w = 0.1  # width
r1 = 3.5
c1 = np.array([-r1, 0.])  # center of shell 1
c2 = np.array([r1, 0.])  # center of shell 2
const = np.log(1. / np.sqrt(2. * np.pi * w**2))  # normalization constant

n_par = 2


@njit()
def logcirc(theta, c):
    """Helper function for log likelihood of a single shell"""
    d = np.sqrt(np.sum((theta - c)**2, axis=-1))
    return const - (d - r)**2 / (2. * w**2)


@njit()
def get_loglike(theta):
    """Get the likelihood of two gaussian shells"""
    res = np.logaddexp(logcirc(theta, c1), logcirc(theta, c2))
    return res


class GaussianShellLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self) -> None:
        """Create the class and store any object specific variables"""
        low_lims = np.full(n_par, low_lim)
        high_lims = np.full(n_par, high_lim)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    def get_loglike(self,params_in):
        """Get the log likelihood given a set of parameters v"""
        return get_loglike(params_in)


@njit()
def gen_draws(n_draws,n_par,attempt_lim=10000):
    """Get posterior draws"""
    draws = np.zeros((n_draws,n_par))
    for itrk in range(n_draws):
        itra = 0
        mode_select = np.random.randint(0,2)
        draw_phase = np.random.uniform(0.,2*np.pi)
        draw_dist = np.random.normal(r1,w)
        draw_coord = np.array([np.cos(draw_phase)*draw_dist,np.sin(draw_phase)*draw_dist])
        if mode_select==0:
            draw_loc = draw_coord+c1
        else:
            draw_loc = draw_coord+c2

        while not check_bounds_rectangular(draw_loc, np.full(n_par, low_lim), np.full(n_par, high_lim)):
            if itra==attempt_lim:
                msg = 'Failed to find valid posterior point.'
                raise RuntimeError(msg)

            mode_select = np.random.randint(0,2)
            draw_phase = np.random.uniform(0.,2*np.pi)
            draw_dist = np.random.normal(r1,w)
            draw_coord = np.array([np.cos(draw_phase)*draw_dist,np.sin(draw_phase)*draw_dist])
            if mode_select==0:
                draw_loc = draw_coord+c1
            else:
                draw_loc = draw_coord+c2

        draws[itrk] = draw_loc
    return draws
