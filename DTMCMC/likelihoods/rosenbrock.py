"""generate an n dimensional rosenbrock likelihood, motivated by 1509.02230.pdf"""

from typing import override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import LoglikeFn, RectangularLikelihood, check_bounds_arrays, memoized_handle

# constants

low_lim = -10.0
high_lim = 10.0


@njit()
def get_loglike(v: NDArray[np.floating], n_par: int) -> float:
    """Get the rosenbrock likelihood"""
    res = 0.0
    for itrp2 in range(n_par // 2):
        res += -100 * (v[itrp2 * 2] ** 2 - v[itrp2 * 2 + 1]) ** 2 - (v[itrp2 * 2] - 1) ** 2
    return res


class RosenbrockLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""

    def __init__(self, n_par: int = 20) -> None:
        """Create the class and store any object specific variables"""
        if n_par % 2 != 0:
            msg = 'RosenbrockLikelihood requires an even n_par'
            raise ValueError(msg)
        low_lims = np.full(n_par, low_lim)
        high_lims = np.full(n_par, high_lim)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    @override
    def _make_loglike(self) -> LoglikeFn:
        n_par = self.n_par

        def build() -> LoglikeFn:
            def loglike(params_in: NDArray[np.floating]) -> float:
                return get_loglike(params_in, n_par)

            return loglike

        return memoized_handle(('rosenbrock_loglike', n_par), build)


@njit()
def gen_draws(n_draws: int, n_par: int, attempt_lim: int = 10000) -> NDArray[np.floating]:
    """Get posterior draws"""
    low_lims = np.full(n_par, low_lim)
    high_lims = np.full(n_par, high_lim)
    draws = np.zeros((n_draws, n_par))
    for itrk in range(n_draws):
        itra = 0
        while True:
            draw_loc = np.zeros(n_par)
            for itrp in range(n_par // 2):
                n1 = np.random.normal(1.0, np.sqrt(1.0 / 2.0))
                n2 = np.random.normal(n1**2, np.sqrt(1 / 200))
                draw_loc[2 * itrp] = n1
                draw_loc[2 * itrp + 1] = n2
            if check_bounds_arrays(draw_loc, low_lims, high_lims):
                break
            itra += 1
            if itra == attempt_lim:
                msg = 'Failed to find valid posterior point.'
                raise RuntimeError(msg)
        draws[itrk] = draw_loc
    return draws
