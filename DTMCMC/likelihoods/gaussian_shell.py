"""two shell likelihood, adapted from https://github.com/joshspeagle/dynesty/blob/master/demos/Examples%20--%20Gaussian%20Shells.ipynb"""
import numba as nb
import numpy as np
from correction_helpers import reflect_into_range
from numba import njit
from numba.experimental import jitclass

from DTMCMC.likelihood import check_bounds_rectangular

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
def loglike(theta):
    """Get the likelihood of two gaussian shells"""
    res = np.logaddexp(logcirc(theta, c1), logcirc(theta, c2))
    return res

@njit()
def prior_factor(v):
    """Get the prior factor if one is needed"""
    return 0.

@njit()
def prior_draw():
    """Get a prior draw"""
    return np.random.uniform(low_lim,high_lim,2)

@njit()
def correct_bounds(v):
    """Wrap the parameters into range"""
    v[0] = reflect_into_range(v[0],low_lim,high_lim)
    v[1] = reflect_into_range(v[1],low_lim,high_lim)
    return v

@njit()
def check_bounds(v):
    """Check if a sample is within the prior range"""
    for itrp in range(v.size):
        if not low_lim<v[itrp]<high_lim:
            return False
    return True

@jitclass([('n_par',nb.int64)])
class Likelihood():
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self):
        """Create the class and store any object specific variables"""
        self.n_par = n_par
        return

    def loglike(self,v):
        """Get the log likelihood given a set of parameters v"""
        return loglike(v)

    def prior_draw(self):
        """Get a draw from the prior"""
        return prior_draw()

    def prior_proposal(self,v_in):
        """Get a proposal from the prior"""
        v_out = prior_draw()
        return v_out,prior_factor(v_in)-prior_factor(v_out)

    def prior_factor(self,v):
        """Get the density factor for prior draws, if the prior draws are not uniform"""
        return prior_factor(v)

    def correct_bounds(self,v):
        """Correct the bounds of a draw to be in range, if allowed for this likelihood"""
        return correct_bounds(v)

    def check_bounds(self,v):
        """Check if the bounds of a draw are in the prior range but do not change them"""
        return check_bounds(v)

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

        draw_loc = np.random.normal(0.,1,n_par)
        while not check_bounds_rectangular(draw_loc, np.full(n_par, low_lim), np.full(n_par, high_lim)):
            if itra==attempt_lim:
                print('failed to find valid posterior point')
                assert False

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
