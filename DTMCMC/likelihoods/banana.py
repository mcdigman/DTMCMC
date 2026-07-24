"""the banana likelihood in n dimensions"""

from typing import override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import NativeLoglikeCall, RectangularInputs, RectangularLikelihood

# constants
# limits for first two parameters
low_lim01: float = -10000.0
high_lim01: float = 10000.0
# limits for n>=2 parameters
low_limn: float = -100.0
high_limn: float = 100.0
# see https://link.springer.com/content/pdf/10.1007/s001800050022.pdf
# and https://www.tandfonline.com/doi/pdf/10.1198/jcgs.2009.06134?needAccess=true
# 20d banana is default
B: float = 0.1  # bananacity parameter


@njit()
def get_loglike(v: NDArray[np.floating], n_par: int) -> float:
    """Get the log likelihood for the 'banana' test case"""
    res: float = -(v[0] ** 2) / 200 - 1 / 2 * (v[1] + B * v[0] ** 2 - 100 * B) ** 2
    for itrp in range(2, n_par):
        res += -1 / 2 * v[itrp] ** 2
    return res


@njit(inline='always')
def _loglike_native(params_in: NDArray[np.floating], inputs: RectangularInputs) -> float:
    """Per-class native log likelihood."""
    return get_loglike(params_in, inputs.n_par)


class BananaLikelihood(RectangularLikelihood[RectangularInputs]):
    """class to manage the likelihood-specific essential functions for the sampler"""

    def __init__(self, n_par: int = 20) -> None:
        """Create the class and store any object specific variables"""
        if n_par < 2:
            msg = 'BananaLikelihood requires n_par >= 2'
            raise ValueError(msg)
        low_lims = np.full(n_par, low_limn)
        high_lims = np.full(n_par, high_limn)
        low_lims[:2] = low_lim01
        high_lims[:2] = high_lim01

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    @property
    @override
    def loglike_fn(self) -> NativeLoglikeCall[RectangularInputs]:
        return _loglike_native
