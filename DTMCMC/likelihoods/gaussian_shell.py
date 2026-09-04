"""two shell likelihood, adapted from https://github.com/joshspeagle/dynesty/blob/master/demos/Examples%20--%20Gaussian%20Shells.ipynb"""

from typing import override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import NativeLoglikeCall, RectangularInputs, RectangularLikelihood, check_bounds_rectangular

# constants
low_lim: float = -40.0
high_lim: float = 40.0

r: float = 2.0  # radius
w: float = 0.1  # width
r1 = 3.5
c1: NDArray[np.floating] = np.array([-r1, 0.0])  # center of shell 1
c2: NDArray[np.floating] = np.array([r1, 0.0])  # center of shell 2
const: float = np.log(1.0 / np.sqrt(2.0 * np.pi * w**2))  # normalization constant


@njit()
def logcirc(theta: NDArray[np.floating], c: NDArray[np.floating]) -> float:
    """Helper function for log likelihood of a single shell"""
    d: float = np.sqrt(np.sum((theta - c) ** 2))
    res: float = const - (d - r) ** 2 / (2.0 * w**2)
    return res


@njit()
def get_loglike(theta: NDArray[np.floating]) -> float:
    """Get the likelihood of two gaussian shells"""
    res: float = np.logaddexp(logcirc(theta, c1), logcirc(theta, c2))
    return res


@njit(inline='always')
def _loglike_native(params_in: NDArray[np.floating], _inputs: RectangularInputs) -> float:
    """Per-class native log likelihood."""
    return get_loglike(params_in)


@njit()
def draw_shell_radius() -> float:
    """Draw a radius from the shell's radial marginal density ~ d*exp(-(d-r)^2/(2w^2)).

    Rejection sampling with an N(r, w) proposal and acceptance probability d/d_cap
    supplies the polar Jacobian factor d that a bare N(r, w) draw omits.
    """
    d_cap = r + 8.0 * w
    while True:
        d = np.random.normal(r, w)
        if 0.0 < d <= d_cap and np.random.uniform(0.0, 1.0) < d / d_cap:
            return d


class GaussianShellLikelihood(RectangularLikelihood[RectangularInputs]):
    """class to manage the likelihood-specific essential functions for the sampler"""

    def __init__(self, n_par: int = 2) -> None:
        """Create the class and store any object specific variables"""
        if n_par != 2:
            msg = 'GaussianShellLikelihood is 2D; n_par must be 2'
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
            mode_select = np.random.randint(0, 2)
            draw_phase = np.random.uniform(0.0, 2 * np.pi)
            draw_dist = draw_shell_radius()
            draw_coord = np.array([np.cos(draw_phase) * draw_dist, np.sin(draw_phase) * draw_dist])
            if mode_select == 0:
                draw_loc = draw_coord + c1
            else:
                draw_loc = draw_coord + c2
            if check_bounds_rectangular(draw_loc, inputs):
                break
            itra += 1
            if itra >= attempt_lim:
                msg = 'Failed to find valid posterior point.'
                raise RuntimeError(msg)
        draws[itrk] = draw_loc
    return draws
