"""a gaussian mixture likelihood in n dimensions with 2 unequal modes at +/-5"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import RectangularInputs, RectangularLikelihood, check_bounds_rectangular
from DTMCMC.numba_backend import NativeLoglikeCall

# constants
low_lim: float = -10.0
high_lim: float = 10.0


@njit()
def get_loglike(v: NDArray[np.floating], n_par: int) -> float:
    """Get likelihood for gaussian mixture"""
    res1: float = np.log(1.0 / (3 * (2.0 * np.pi) ** (n_par / 2)))
    res2: float = np.log(2.0 / (3 * (2.0 * np.pi) ** (n_par / 2)))
    for itrp in range(n_par):
        res1 += -((v[itrp] - 5) ** 2) / 2
        res2 += -((v[itrp] + 5) ** 2) / 2

    res: float = np.logaddexp(res1, res2)

    return res


@njit(inline='always')
def _loglike_native(params_in: NDArray[np.floating], state: RectangularInputs) -> float:
    """Per-class native log likelihood reading n_par from the state bundle."""
    return get_loglike(params_in, state.n_par)


class GaussianMixtureLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""

    def __init__(self, n_par: int = 50) -> None:
        """Create the class and store any object specific variables"""
        low_lims = np.full(n_par, low_lim)
        high_lims = np.full(n_par, high_lim)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Get the log likelihood given a set of parameters v"""
        return get_loglike(params_in, self.n_par)

    def bind_native_loglike(self) -> NativeLoglikeCall[RectangularInputs]:
        """Return the per-class native log likelihood."""
        return _loglike_native


@njit()
def gen_draws(n_draws: int, n_par: int, attempt_lim: int = 10000) -> NDArray[np.floating]:
    """Get posterior draws"""
    low_lims = np.full(n_par, low_lim)
    high_lims = np.full(n_par, high_lim)
    inputs = RectangularInputs(n_par, low_lims, high_lims)
    draws = np.zeros((n_draws, n_par))
    for itrk in range(n_draws):
        itra = 0
        while True:
            mode_choose = np.random.uniform(0.0, 1.0)
            if mode_choose < 1.0 / 3.0:
                draw_loc = np.random.normal(5, 1, n_par)
            else:
                draw_loc = np.random.normal(-5, 1, n_par)
            if check_bounds_rectangular(draw_loc, inputs):
                break
            itra += 1
            if itra == attempt_lim:
                msg = 'Failed to find valid posterior point.'
                raise RuntimeError(msg)
        draws[itrk] = draw_loc
    return draws
