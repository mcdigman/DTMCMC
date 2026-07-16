"""the random wheel likelihood"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import RectangularLikelihood, RectangularNativeState, check_bounds_rectangular
from DTMCMC.numba_backend import NativeLoglikeCall

# constants
low_lim = -40.0
high_lim = 40.0

w = 1.0  # width

c1 = np.array([-3.7602844, 13.64930874])  # location of mode 1
c2 = np.array([2.89532714, -10.62313047])  # location of mode 2
c3 = np.array([7.14762605, 29.96712803])  # location of mode 3
c4 = np.array([27.82821259, -5.52804758])  # location of mode 4
c5 = np.array([-10.55158202, -1.57069936])  # location of mode 5
c6 = np.array([-20.67569031, -28.2846108])  # location of mode 6
c7 = np.array([10.48702906, -0.46073795])  # location of mode 7
c8 = np.array([-1.76706143, -18.33279469])  # location of mode 8
c9 = np.array([17.37321326, -3.39999183])  # location of mode 9

cs = np.array([c1, c2, c3, c4, c5, c6, c7, c8, c9])
const = np.log(1.0 / np.sqrt(2.0 * np.pi * w**2))  # normalization constant


@njit()
def gaussian(v: NDArray[np.floating], c: NDArray[np.floating]) -> float:
    """Helper for log likelihood of a gaussian"""
    res = 0.0
    for itrp in range(c.shape[0]):
        res += const - 1 / (2 * w**2) * (v[itrp] - c[itrp]) ** 2
    return res


@njit()
def get_loglike(v: NDArray[np.floating]) -> float:
    """Get the likelihood for our wheel potential"""
    res = gaussian(v, cs[0])
    for itrm in range(1, cs.shape[0]):
        res = np.logaddexp(res, gaussian(v, cs[itrm]))
    return res


@njit(inline='always')
def _loglike_native(params_in: NDArray[np.floating], _state: RectangularNativeState) -> float:
    """Per-class native log likelihood; the wheel needs no instance state."""
    return get_loglike(params_in)


class RandomWheelLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""

    def __init__(self, n_par: int = 2) -> None:
        """Create the class and store any object specific variables"""
        if n_par != 2:
            msg = 'RandomWheelLikelihood is 2D; n_par must be 2'
            raise ValueError(msg)
        low_lims = np.full(n_par, low_lim)
        high_lims = np.full(n_par, high_lim)

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Get the log likelihood given a set of parameters v"""
        return get_loglike(params_in)

    def bind_native_loglike(self) -> NativeLoglikeCall[RectangularNativeState]:
        """Return the per-class native log likelihood."""
        return _loglike_native


@njit()
def gen_draws(n_draws: int, n_par: int, attempt_lim: int = 10000) -> NDArray[np.floating]:
    """Get posterior draws"""
    low_lims = np.full(n_par, low_lim)
    high_lims = np.full(n_par, high_lim)
    draws = np.zeros((n_draws, n_par))
    for itrk in range(n_draws):
        itra = 0
        while True:
            mode_choose = np.random.randint(0, cs.shape[0])
            draw_loc = cs[mode_choose] + np.random.normal(0.0, w, cs.shape[1])
            if check_bounds_rectangular(draw_loc, low_lims, high_lims):
                break
            itra += 1
            if itra == attempt_lim:
                msg = 'Failed to find valid posterior point.'
                raise RuntimeError(msg)
        draws[itrk] = draw_loc
    return draws
