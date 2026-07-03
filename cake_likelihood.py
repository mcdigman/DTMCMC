"""an n dimensional normal distribution"""
import numpy as np
from scipy.special import gamma

from DTMCMC.likelihood import RectangularLikelihood


def get_cake_tier_logL(v, amp, width, exponent):
    n_par = v.shape[0]

    dim_part = gamma(1 + n_par / 2) / (np.pi**(n_par / 2))
    res = np.log(amp * dim_part / (2**(n_par / exponent) * width**n_par * gamma((exponent + n_par) / exponent)))

    # get the squared distance from the center
    r2_got = 0.
    for itrp in range(v.shape[0]):
        r2_got += v[itrp]**2

    res += -1 / (2 * width**exponent) * r2_got**(exponent / 2)
    # for itrp in range(0,v.shape[0]):
    #    res += -1/(2*width**exponent)*v[itrp]**exponent

    return res


# @njit()
def get_loglike(params_in):
    """Get a 'cake' likelihood"""
    amp1 = 0.5
    amp2 = 0.5
    width1 = 4.
    width2 = 0.1
    exp1 = 8
    exp2 = 2

    res1 = get_cake_tier_logL(params_in, amp1, width1, exp1)
    res2 = get_cake_tier_logL(params_in, amp2, width2, exp2)

    res = np.logaddexp(res1, res2)
    return res


# n dimensional unit normal motivated by the 100d considerations in
# https://statmodeling.stat.columbia.edu/2017/03/15/ensemble-methods-doomed-fail-high-dimensions/

# @jitclass([('n_par',nb.int64),('epsilons',nb.float64[:])])
class CakeLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self, n_par=2, cutoff=10) -> None:
        """Create the class and store any object specific variables"""
        low_lims = np.full(n_par, -cutoff)
        high_lims = np.full(n_par, cutoff)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    def get_loglike(self, params_in):
        """Get the log likelihood given a set of parameters v"""
        return get_loglike(params_in)
