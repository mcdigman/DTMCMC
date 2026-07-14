"""The ar(1) likelihood in n dimensions"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import RectangularLikelihood, check_bounds_rectangular

# constants
low_lim: float = -10.0
high_lim: float = 10.0
alpha: float = 0.9
beta: float = np.sqrt(1 - alpha**2)
const: float = np.log(1.0 / np.sqrt(2.0 * np.pi * beta**2))  # conditional-term normalization constant
const0: float = np.log(1.0 / np.sqrt(2.0 * np.pi))  # first-term (unit-variance stationary) normalization constant


@njit()
def get_loglike(v: NDArray[np.floating], n_par: int) -> float:
    """Get the likelihood for ar1"""
    res: float = 0.0
    x_next: float = 0.0
    res += const0 - 1 / 2 * (v[0] - x_next) ** 2
    x_next = alpha * v[0]

    for itrp in range(1, n_par):
        res += const - 1 / 2 / beta**2 * (v[itrp] - x_next) ** 2
        x_next = alpha * v[itrp]
    return res


class Ar1Likelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""

    def __init__(self, n_par: int = 50) -> None:
        """Create the class and store any object specific variables"""
        low_lims = np.full(n_par, low_lim)
        high_lims = np.full(n_par, high_lim)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Get the log likelihood given a set of parameters v"""
        return get_loglike(params_in, self.n_par)


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
            draw_loc[0] = np.random.normal(0.0, 1.0)
            for itrp in range(1, n_par):
                n1 = np.random.normal(alpha * draw_loc[itrp - 1], beta)
                draw_loc[itrp] = n1
            if check_bounds_rectangular(draw_loc, low_lims, high_lims):
                break
            itra += 1
            if itra == attempt_lim:
                msg = 'Failed to find valid posterior point.'
                raise RuntimeError(msg)
        draws[itrk] = draw_loc
    return draws
