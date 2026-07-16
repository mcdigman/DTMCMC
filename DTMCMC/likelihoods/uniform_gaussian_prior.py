"""Constant likelihood with an independent Gaussian prior in every dimension."""

from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import RectangularLikelihood
from DTMCMC.numba_backend import NativeLoglikeCall, NativePriorDrawCall, NativePriorFactorCall


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


class UniformGaussianNativeState(NamedTuple):
    """Runtime state bundle: the rectangular fields plus the Gaussian prior arrays."""

    n_par: int
    low_lims: NDArray[np.float64]
    high_lims: NDArray[np.float64]
    prior_mean: NDArray[np.float64]
    prior_std: NDArray[np.float64]


@njit(inline='always')
def _loglike_native(params_in: NDArray[np.floating], _state: UniformGaussianNativeState) -> float:
    """Per-class native log likelihood; the constant needs no instance state."""
    return get_loglike(params_in)


@njit(inline='always')
def _prior_draw_native(state: UniformGaussianNativeState) -> NDArray[np.floating]:
    """Per-class native Gaussian prior draw reading the state bundle."""
    return gaussian_prior_draw(state.n_par, state.prior_mean, state.prior_std)


@njit(inline='always')
def _prior_factor_native(params_in: NDArray[np.floating], state: UniformGaussianNativeState) -> float:
    """Per-class native Gaussian log prior density reading the state bundle."""
    return gaussian_prior_factor(params_in, state.prior_mean, state.prior_std)


class UniformGaussianPriorLikelihood(RectangularLikelihood):
    """Known Gaussian target produced by a constant likelihood and Gaussian prior."""

    def __init__(self, n_par: int = 4, prior_mean: float = 0.0, prior_std: float = 1.0) -> None:
        if prior_std <= 0.0:
            msg = 'prior_std must be positive'
            raise ValueError(msg)
        super().__init__(n_par, np.full(n_par, -np.inf), np.full(n_par, np.inf))
        # frozen so the Python path and the native state bundles always agree
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

    def native_state(self) -> UniformGaussianNativeState:
        """Return the rectangular fields plus the Gaussian prior arrays."""
        return UniformGaussianNativeState(self.n_par, self.low_lims, self.high_lims, self.prior_mean, self.prior_std)

    def bind_native_loglike(self) -> NativeLoglikeCall[UniformGaussianNativeState]:
        """Return the per-class native log likelihood."""
        return _loglike_native

    def bind_native_prior_draw(self) -> NativePriorDrawCall[UniformGaussianNativeState]:
        """Return the per-class native Gaussian prior draw."""
        return _prior_draw_native

    def bind_native_prior_factor(self) -> NativePriorFactorCall[UniformGaussianNativeState]:
        """Return the per-class native Gaussian log prior density."""
        return _prior_factor_native
