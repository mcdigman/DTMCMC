"""Constant likelihood with an independent Gaussian prior in every dimension."""

from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import RectangularLikelihood

if TYPE_CHECKING:
    from collections.abc import Callable


@njit(inline='always')
def get_loglike(_params: NDArray[np.floating]) -> float:
    """Return a constant log likelihood over the whole parameter space."""
    return 0.0


@njit(inline='always')
def gaussian_prior_draw(
    n_par: int,
    prior_mean: NDArray[np.floating],
    prior_std: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Draw from the independent Gaussian prior."""
    return prior_mean + prior_std * np.random.normal(0.0, 1.0, n_par)


@njit(inline='always')
def gaussian_prior_factor(
    params: NDArray[np.floating],
    prior_mean: NDArray[np.floating],
    prior_std: NDArray[np.floating],
) -> float:
    """Return the Gaussian log density, up to an irrelevant constant."""
    standardized = (params - prior_mean) / prior_std
    return -0.5 * float(np.sum(standardized * standardized)) - float(np.sum(np.log(prior_std)))


class UniformGaussianPriorLikelihood(RectangularLikelihood):
    """Known Gaussian target produced by a constant likelihood and Gaussian prior."""

    def __init__(self, n_par: int = 4, prior_mean: float = 0.0, prior_std: float = 1.0) -> None:
        if prior_std <= 0.0:
            msg = 'prior_std must be positive'
            raise ValueError(msg)
        super().__init__(n_par, np.full(n_par, -np.inf), np.full(n_par, np.inf))
        # frozen: the native bindings bake these arrays in by reference
        self.prior_mean = np.full(n_par, prior_mean)
        self.prior_std = np.full(n_par, prior_std)
        self.prior_mean.setflags(write=False)
        self.prior_std.setflags(write=False)

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Evaluate the constant likelihood."""
        return get_loglike(params_in)

    def prior_draw(self) -> NDArray[np.floating]:
        """Draw from the Gaussian prior."""
        return gaussian_prior_draw(self.n_par, self.prior_mean, self.prior_std)

    def prior_factor(self, params_in: NDArray[np.floating]) -> float:
        """Return the Gaussian log prior density up to a constant."""
        return gaussian_prior_factor(params_in, self.prior_mean, self.prior_std)

    def bind_native_loglike(self) -> Callable[[NDArray[np.floating]], float]:
        """The constant loglike is already stateless and jitted."""
        return get_loglike

    def bind_native_prior_draw(self) -> Callable[[], NDArray[np.floating]]:
        """Return the Gaussian draw with the prior arrays baked in."""
        n_par = self.n_par
        prior_mean = self.prior_mean
        prior_std = self.prior_std

        @njit(inline='always')
        def prior_draw_native() -> NDArray[np.floating]:
            return gaussian_prior_draw(n_par, prior_mean, prior_std)

        return prior_draw_native

    def bind_native_prior_factor(self) -> Callable[[NDArray[np.floating]], float]:
        """Return the Gaussian log density with the prior arrays baked in."""
        prior_mean = self.prior_mean
        prior_std = self.prior_std

        @njit(inline='always')
        def prior_factor_native(params: NDArray[np.floating]) -> float:
            return gaussian_prior_factor(params, prior_mean, prior_std)

        return prior_factor_native
