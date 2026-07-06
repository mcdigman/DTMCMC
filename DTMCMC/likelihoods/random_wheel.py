"""the random wheel likelihood"""
import numpy as np
from numba import njit

from DTMCMC.likelihood import RectangularLikelihood, check_bounds_rectangular

#constants
low_lim = -40.
high_lim = 40.

# defining constants
w = 1.  # width
r1 = 30.

c1 = np.array([-3.7602844 , 13.64930874])  # location of mode 1
c2 = np.array([  2.89532714, -10.62313047])  # location of mode 2
c3 = np.array([ 7.14762605, 29.96712803])  # location of mode 3
c4 = np.array([27.82821259, -5.52804758])  # location of mode 4
c5 = np.array([-10.55158202,  -1.57069936])  # location of mode 5
c6 = np.array([-20.67569031, -28.2846108 ])  # location of mode 6
c7 = np.array([10.48702906, -0.46073795])  # location of mode 7
c8 = np.array([ -1.76706143, -18.33279469])  # location of mode 8
c9 = np.array([17.37321326, -3.39999183])  # location of mode 9

cs = np.array([c1,c2,c3,c4,c5,c6,c7,c8,c9])
const = np.log(1. / np.sqrt(2. * np.pi * w**2))  # normalization constant

n_par = 2


@njit()
def gaussian(v, c):
    """Helper for log likelihood of a gaussian"""
    res = 0.
    for itrp in range(n_par):
        res += const-1/(2*w**2)*(v[itrp]-c[itrp])**2
    return res


@njit()
def get_loglike(v):
    """Get the likelihood for our wheel potential"""
    res = np.logaddexp(gaussian(v, c1), gaussian(v, c2))
    res = np.logaddexp(res, gaussian(v, c3))
    res = np.logaddexp(res, gaussian(v, c4))
    res = np.logaddexp(res, gaussian(v, c5))
    res = np.logaddexp(res, gaussian(v, c6))
    res = np.logaddexp(res, gaussian(v, c7))
    res = np.logaddexp(res, gaussian(v, c8))
    res = np.logaddexp(res, gaussian(v, c9))
    return res


class RandomWheelLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self, n_par=2) -> None:
        """Create the class and store any object specific variables"""
        assert n_par == 2
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
        mode_choose = np.random.randint(0,9)
        draw_loc = cs[mode_choose]+np.random.normal(0.,w,2)
        while not check_bounds_rectangular(draw_loc, np.full(n_par, low_lim), np.full(n_par, high_lim)):
            if itra==attempt_lim:
                msg = 'Failed to find valid posterior point.'
                raise RuntimeError(msg)

            mode_choose = np.random.randint(0,9)
            draw_loc = cs[mode_choose]+np.random.normal(0.,w,2)
            itra += 1

        draws[itrk] = draw_loc
    return draws
