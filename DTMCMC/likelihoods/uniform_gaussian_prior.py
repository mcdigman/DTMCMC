"""Constant likelihood with an independent Gaussian prior in every dimension."""

from typing import override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import (
    LoglikeFn,
    PriorDrawFn,
    PriorFactorFn,
    RectangularLikelihood,
    memoized_handle,
)


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
        prior_mean_arr = np.full(n_par, prior_mean)
        prior_std_arr = np.full(n_par, prior_std)
        prior_mean_arr.setflags(write=False)
        prior_std_arr.setflags(write=False)
        self._prior_mean: NDArray[np.floating] = prior_mean_arr
        self._prior_std: NDArray[np.floating] = prior_std_arr
        super().__init__(n_par, np.full(n_par, -np.inf), np.full(n_par, np.inf))

    def _prior_key(self) -> tuple[object, ...]:
        return (self.n_par, self._prior_mean.tobytes(), self._prior_std.tobytes())

    @override
    def _make_loglike(self) -> LoglikeFn:
        return get_loglike

    @override
    def _make_prior_draw(self) -> PriorDrawFn:
        n_par, prior_mean, prior_std = self._n_par, self._prior_mean, self._prior_std

        def build() -> PriorDrawFn:
            def prior_draw() -> NDArray[np.floating]:
                return gaussian_prior_draw(n_par, prior_mean, prior_std)

            return prior_draw

        return memoized_handle(('uniform_gaussian_prior_draw', *self._prior_key()), build)

    @override
    def _make_prior_factor(self) -> PriorFactorFn:
        prior_mean, prior_std = self._prior_mean, self._prior_std

        def build() -> PriorFactorFn:
            def prior_factor(params_in: NDArray[np.floating]) -> float:
                return gaussian_prior_factor(params_in, prior_mean, prior_std)

            return prior_factor

        return memoized_handle(('uniform_gaussian_prior_factor', *self._prior_key()), build)
