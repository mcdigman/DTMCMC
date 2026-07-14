"""an n dimensional normal distribution"""

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gamma

from DTMCMC.likelihood import RectangularLikelihood

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_cake_tier_logL(v: NDArray[np.floating], amp: float, width: float, exponent: int | float) -> float:
    n_par = v.shape[0]

    dim_part = gamma(1 + n_par / 2) / (np.pi ** (n_par / 2))
    res = np.log(amp * dim_part / (2 ** (n_par / exponent) * width**n_par * gamma((exponent + n_par) / exponent)))

    # get the squared distance from the center
    r2_got = 0.0
    for itrp in range(v.shape[0]):
        r2_got += v[itrp] ** 2

    res += -1 / (2 * width**exponent) * r2_got ** (exponent / 2)
    # for itrp in range(0,v.shape[0]):
    #    res += -1/(2*width**exponent)*v[itrp]**exponent

    return res


# default two-tier cake: a wide flat-topped tier and a narrow Gaussian
# spike, equal mixture weights (each tier integrates to exactly its amp)
CAKE_DEFAULT_AMPS = (0.5, 0.5)
CAKE_DEFAULT_WIDTHS = (4.0, 0.1)
CAKE_DEFAULT_EXPONENTS = (8, 2)


# @njit()
def get_loglike(
    params_in: NDArray[np.floating],
    amps: tuple[float, ...] = CAKE_DEFAULT_AMPS,
    widths: tuple[float, ...] = CAKE_DEFAULT_WIDTHS,
    exponents: tuple[int, ...] | tuple[float, ...] = CAKE_DEFAULT_EXPONENTS,
) -> float:
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

    def __init__(
        self,
        n_par: int = 2,
        cutoff: int = 10,
        amps: tuple[float, ...] = CAKE_DEFAULT_AMPS,
        widths: tuple[float, ...] = CAKE_DEFAULT_WIDTHS,
        exponents: tuple[int, ...] | tuple[float, ...] = CAKE_DEFAULT_EXPONENTS,
    ) -> None:
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

        # the log-normalizations cost two gamma calls per tier, so hoist
        # them out of the per-evaluation path; the expressions are
        # identical to get_cake_tier_logL's, so values are bit-identical
        # (guarded by the golden-run test)
        dim_part = gamma(1 + n_par / 2) / (np.pi ** (n_par / 2))
        self._tier_lognorms = tuple(
            np.log(amp * dim_part / (2 ** (n_par / exponent) * width**n_par * gamma((exponent + n_par) / exponent)))
            for amp, width, exponent in zip(self.amps, self.widths, self.exponents, strict=True)
        )
        self._tier_coefs = tuple(
            -1 / (2 * width**exponent) for width, exponent in zip(self.widths, self.exponents, strict=True)
        )
        self._tier_powers = tuple(exponent / 2 for exponent in self.exponents)

        low_lims = np.full(n_par, -cutoff)
        high_lims = np.full(n_par, cutoff)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Get the log likelihood given a set of parameters v"""
        r2_got = 0.0
        for itrp in range(params_in.shape[0]):
            r2_got += params_in[itrp] ** 2

        res = self._tier_lognorms[0] + self._tier_coefs[0] * r2_got ** self._tier_powers[0]
        for itrm in range(1, len(self.amps)):
            res = np.logaddexp(
                res, self._tier_lognorms[itrm] + self._tier_coefs[itrm] * r2_got ** self._tier_powers[itrm]
            )
        return res
