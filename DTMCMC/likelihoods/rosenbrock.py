"""generate an n dimensional rosenbrock likelihood, motivated by 1509.02230.pdf"""

from typing import override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import NativeLoglikeCall, RectangularInputs, RectangularLikelihood, check_bounds_rectangular

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


@njit(inline='always')
def _loglike_native(params_in: NDArray[np.floating], inputs: RectangularInputs) -> float:
    """Per-class native log likelihood."""
    return get_loglike(params_in, inputs.n_par)


class RosenbrockLikelihood(RectangularLikelihood[RectangularInputs]):
    """class to manage the likelihood-specific essential functions for the sampler"""

    def __init__(self, n_par: int = 20) -> None:
        """Create the class and store any object specific variables"""
        if n_par % 2 != 0:
            msg = 'RosenbrockLikelihood requires an even n_par'
            raise ValueError(msg)
        low_lims = np.full(n_par, low_lim)
        high_lims = np.full(n_par, high_lim)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    @property
    @override
    def loglike_fn(self) -> NativeLoglikeCall[RectangularInputs]:
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
            draw_loc = np.zeros(n_par)
            for itrp in range(n_par // 2):
                n1 = np.random.normal(1.0, np.sqrt(1.0 / 2.0))
                n2 = np.random.normal(n1**2, np.sqrt(1 / 200))
                draw_loc[2 * itrp] = n1
                draw_loc[2 * itrp + 1] = n2
            if check_bounds_rectangular(draw_loc, inputs):
                break
            itra += 1
            if itra == attempt_lim:
                msg = 'Failed to find valid posterior point.'
                raise RuntimeError(msg)
        draws[itrk] = draw_loc
    return draws
