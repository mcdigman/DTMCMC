"""an n dimensional normal distribution"""
import numpy as np
from numba import njit

from DTMCMC.likelihood import RectangularLikelihood, check_bounds_rectangular


# @njit()
def get_loglike(v):
    """Get an n dimensional gaussian likelihood"""
    const = np.log(1. / np.sqrt(2. * np.pi))  # normalization constant
    res = v.shape[0] * const
    for itrp in range(v.shape[0]):
        res += -1 / 2 * v[itrp]**2
    return res


# n dimensional unit normal motivated by the 100d considerations in
# https://statmodeling.stat.columbia.edu/2017/03/15/ensemble-methods-doomed-fail-high-dimensions/

# @jitclass([('n_par',nb.int64),('epsilons',nb.float64[:])])
class GaussianLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self, n_par=100, cutoff=5) -> None:
        """Create the class and store any object specific variables"""
        low_lims = np.full(n_par, -cutoff)
        high_lims = np.full(n_par, cutoff)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    def get_loglike(self, v):
        """Get the log likelihood given a set of parameters v"""
        return get_loglike(v)


@njit()
def gen_draws(n_draws, n_par, cutoff, attempt_lim=10000):
    """Get posterior draws"""
    draws = np.zeros((n_draws, n_par))
    for itrk in range(n_draws):
        itra = 0
        draw_loc = np.random.normal(0., 1, n_par)
        while not check_bounds_rectangular(draw_loc, cutoff):
            if itra == attempt_lim:
                msg = 'failed to find valid posterior point'
                raise RuntimeError(msg)

            draw_loc = np.random.normal(0., 1, n_par)
        draws[itrk] = draw_loc
    return draws


@njit()
def drawposterior(n, Ts, n_par, cutoff):
    """For truncated normal we can draw from the posterior for testing purposes"""
    samples = np.zeros((n, Ts.size, n_par))
    for itrt in range(Ts.size):
        for itrn in range(n):
            if np.isfinite(Ts[itrt]):
                sample_loc = np.random.normal(0., np.sqrt(Ts[itrt]), n_par)
                itrlim = 0
                while not check_bounds_rectangular(sample_loc, cutoff):
                    if itrlim == 100000:
                        print(itrt, itrn, itrlim)
                        msg = 'failed to find valid posterior point'
                        raise RuntimeError(msg)
                    sample_loc = np.random.normal(0., np.sqrt(Ts[itrt]), n_par)
                    itrlim += 1
                samples[itrn, itrt] = sample_loc
            else:
                samples[itrn, itrt] = np.random.uniform(-cutoff, cutoff, n_par)
    return samples
