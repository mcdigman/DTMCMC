"""an n dimensional normal distribution"""

from typing import override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import NativeLoglikeCall, RectangularInputs, RectangularLikelihood, check_bounds_rectangular


@njit()
def get_loglike(v: NDArray[np.floating]) -> float:
    """Get an n dimensional gaussian likelihood"""
    const: float = np.log(1.0 / np.sqrt(2.0 * np.pi))  # normalization constant
    res: float = v.shape[0] * const
    for itrp in range(v.shape[0]):
        res += -1 / 2 * v[itrp] ** 2
    return res


@njit(inline='always')
def _loglike_native(params_in: NDArray[np.floating], _inputs: RectangularInputs) -> float:
    """Per-class native log likelihood."""
    return get_loglike(params_in)


# n dimensional unit normal motivated by the 100d considerations in
# https://statmodeling.stat.columbia.edu/2017/03/15/ensemble-methods-doomed-fail-high-dimensions/


class GaussianLikelihood(RectangularLikelihood[RectangularInputs]):
    """class to manage the likelihood-specific essential functions for the sampler"""

    def __init__(self, n_par: int = 100, cutoff: int = 5) -> None:
        """Create the class and store any object specific variables"""
        low_lims = np.full(n_par, -cutoff)
        high_lims = np.full(n_par, cutoff)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    @property
    @override
    def loglike_fn(self) -> NativeLoglikeCall[RectangularInputs]:
        return _loglike_native


@njit()
def gen_draws(
    n_draws: int, n_par: int, low_lims: NDArray[np.floating], high_lims: NDArray[np.floating], attempt_lim: int = 10000
) -> NDArray[np.floating]:
    """Get posterior draws"""
    draws = np.zeros((n_draws, n_par))
    inputs = RectangularInputs(n_par, low_lims, high_lims)
    for itrk in range(n_draws):
        itra = 0
        draw_loc = np.random.normal(0.0, 1, n_par)
        while not check_bounds_rectangular(draw_loc, inputs):
            itra += 1
            if itra >= attempt_lim:
                msg = 'failed to find valid posterior point'
                raise RuntimeError(msg)

            draw_loc = np.random.normal(0.0, 1, n_par)
        draws[itrk] = draw_loc
    return draws


@njit()
def drawposterior(
    n: int, Ts: NDArray[np.floating], n_par: int, low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]
) -> NDArray[np.floating]:
    """For truncated normal we can draw from the posterior for testing purposes"""
    inputs = RectangularInputs(n_par, low_lims, high_lims)
    samples = np.zeros((n, Ts.size, n_par))
    for itrt in range(Ts.size):
        for itrn in range(n):
            if np.isfinite(Ts[itrt]):
                sample_loc = np.random.normal(0.0, np.sqrt(Ts[itrt]), n_par)
                itrlim = 0
                while not check_bounds_rectangular(sample_loc, inputs):
                    if itrlim == 100000:
                        print(itrt, itrn, itrlim)
                        msg = 'failed to find valid posterior point'
                        raise RuntimeError(msg)
                    sample_loc = np.random.normal(0.0, np.sqrt(Ts[itrt]), n_par)
                    itrlim += 1
                samples[itrn, itrt] = sample_loc
            else:
                for itrp in range(n_par):
                    samples[itrn, itrt, itrp] = np.random.uniform(low_lims[itrp], high_lims[itrp])
    return samples
