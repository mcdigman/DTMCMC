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


# default two-tier cake: a wide flat-topped tier and a narrow Gaussian
# spike, equal mixture weights (each tier integrates to exactly its amp)
CAKE_DEFAULT_AMPS = (0.5, 0.5)
CAKE_DEFAULT_WIDTHS = (4., 0.1)
CAKE_DEFAULT_EXPONENTS = (8, 2)


# @njit()
def get_loglike(params_in, amps=CAKE_DEFAULT_AMPS, widths=CAKE_DEFAULT_WIDTHS, exponents=CAKE_DEFAULT_EXPONENTS):
    """Get a 'cake' likelihood: logaddexp over the mixture tiers"""
    res = get_cake_tier_logL(params_in, amps[0], widths[0], exponents[0])
    for itrm in range(1, len(amps)):
        res = np.logaddexp(res, get_cake_tier_logL(params_in, amps[itrm], widths[itrm], exponents[itrm]))
    return res


# n dimensional unit normal motivated by the 100d considerations in
# https://statmodeling.stat.columbia.edu/2017/03/15/ensemble-methods-doomed-fail-high-dimensions/

# @jitclass([('n_par',nb.int64),('epsilons',nb.float64[:])])
class CakeLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self, n_par=2, cutoff=10, amps=CAKE_DEFAULT_AMPS, widths=CAKE_DEFAULT_WIDTHS, exponents=CAKE_DEFAULT_EXPONENTS) -> None:
        """Create the class and store any object specific variables

        The tier parameters default to the historical hardcoded cake
        (identical values, guarded by the golden-run test); passing
        custom amps/widths/exponents gives the tunable cake family.
        """
        assert len(amps) == len(widths) == len(exponents)
        assert len(amps) >= 1
        self.amps = tuple(amps)
        self.widths = tuple(widths)
        self.exponents = tuple(exponents)

        low_lims = np.full(n_par, -cutoff)
        high_lims = np.full(n_par, cutoff)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    def get_loglike(self, params_in):
        """Get the log likelihood given a set of parameters v"""
        return get_loglike(params_in, self.amps, self.widths, self.exponents)
