"""Constant likelihood with an independent Gaussian prior in every dimension."""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import RectangularLikelihood
from DTMCMC.numba_backend import jittable_likelihood


@njit(inline='always')
def get_loglike(
    _params: NDArray[np.floating],
    _n_par: int,
    _prior_mean: NDArray[np.floating],
    _prior_std: NDArray[np.floating],
) -> float:
    """Return a constant log likelihood over the whole parameter space."""
    return 0.0


@njit(inline='always')
def gaussian_prior_draw(
    n_par: int,
    _low_lims: NDArray[np.floating],
    _high_lims: NDArray[np.floating],
    state: tuple[int, NDArray[np.floating], NDArray[np.floating]],
) -> NDArray[np.floating]:
    """Draw from the independent Gaussian prior."""
    _n_par, prior_mean, prior_std = state
    return prior_mean + prior_std * np.random.normal(0.0, 1.0, n_par)


@njit(inline='always')
def gaussian_prior_factor(
    params: NDArray[np.floating],
    state: tuple[int, NDArray[np.floating], NDArray[np.floating]],
) -> float:
    """Return the Gaussian log density, up to an irrelevant constant."""
    _n_par, prior_mean, prior_std = state
    standardized = (params - prior_mean) / prior_std
    return -0.5 * float(np.sum(standardized * standardized)) - float(np.sum(np.log(prior_std)))


@jittable_likelihood(
    get_loglike,
    state_attrs=('n_par', 'prior_mean', 'prior_std'),
    prior_draw=gaussian_prior_draw,
    prior_factor=gaussian_prior_factor,
)
class UniformGaussianPriorLikelihood(RectangularLikelihood):
    """Known Gaussian target produced by a constant likelihood and Gaussian prior."""

    def __init__(self, n_par: int = 4, prior_mean: float = 0.0, prior_std: float = 1.0) -> None:
        if prior_std <= 0.0:
            msg = 'prior_std must be positive'
            raise ValueError(msg)
        super().__init__(n_par, np.full(n_par, -np.inf), np.full(n_par, np.inf))
        self.prior_mean = np.full(n_par, prior_mean)
        self.prior_std = np.full(n_par, prior_std)

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Evaluate the constant likelihood."""
        self.n_evals += 1
        return get_loglike(params_in, self.n_par, self.prior_mean, self.prior_std)

    def prior_draw(self) -> NDArray[np.floating]:
        """Draw from the Gaussian prior."""
        return gaussian_prior_draw(
            self.n_par,
            self.low_lims,
            self.high_lims,
            (self.n_par, self.prior_mean, self.prior_std),
        )

    def prior_factor(self, params_in: NDArray[np.floating]) -> float:
        """Return the Gaussian log prior density up to a constant."""
        return gaussian_prior_factor(params_in, (self.n_par, self.prior_mean, self.prior_std))
