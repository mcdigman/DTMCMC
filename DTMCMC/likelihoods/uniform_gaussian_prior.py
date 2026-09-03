"""Constant likelihood with an independent Gaussian prior in every dimension."""

from typing import NamedTuple, override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import NativeLoglikeCall, NativePriorDrawCall, NativePriorFactorCall, RectangularLikelihood


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


class UniformRectangularGaussianInputs(NamedTuple):
    """Compile-time fixed inputs."""

    n_par: int
    low_lims: NDArray[np.floating]
    high_lims: NDArray[np.floating]
    prior_mean: NDArray[np.floating]
    prior_std: NDArray[np.floating]


@njit(inline='always')
def _loglike_native(_params_in: NDArray[np.floating], _inputs: UniformRectangularGaussianInputs) -> float:
    """Per-class native log likelihood."""
    return 0.0


@njit(inline='always')
def _prior_draw_native(inputs: UniformRectangularGaussianInputs) -> NDArray[np.floating]:
    """Per-class native Gaussian prior draw."""
    return gaussian_prior_draw(inputs.n_par, inputs.prior_mean, inputs.prior_std)


@njit(inline='always')
def _prior_factor_native(params_in: NDArray[np.floating], inputs: UniformRectangularGaussianInputs) -> float:
    """Per-class native Gaussian log prior density."""
    return gaussian_prior_factor(params_in, inputs.prior_mean, inputs.prior_std)


class UniformGaussianPriorLikelihood(RectangularLikelihood[UniformRectangularGaussianInputs]):
    """Known Gaussian target produced by a constant likelihood and Gaussian prior."""

    def __init__(self, n_par: int = 4, prior_mean: float = 0.0, prior_std: float = 1.0) -> None:
        if prior_std <= 0.0:
            msg = 'prior_std must be positive'
            raise ValueError(msg)

        prior_mean_arr = np.full(n_par, prior_mean)
        prior_std_arr = np.full(n_par, prior_std)
        prior_mean_arr.setflags(write=False)
        prior_std_arr.setflags(write=False)

        low_lims = np.full(n_par, -np.inf)
        low_lims.setflags(write=False)

        high_lims = np.full(n_par, np.inf)
        high_lims.setflags(write=False)

        self._inputs_gauss: UniformRectangularGaussianInputs = UniformRectangularGaussianInputs(
            n_par, low_lims, high_lims, prior_mean_arr, prior_std_arr
        )
        super().__init__(n_par, low_lims, high_lims)

    @property
    @override
    def loglike_fn(self) -> NativeLoglikeCall[UniformRectangularGaussianInputs]:
        return _loglike_native

    @property
    @override
    def prior_draw_fn(self) -> NativePriorDrawCall[UniformRectangularGaussianInputs]:
        return _prior_draw_native

    @property
    @override
    def prior_factor_fn(self) -> NativePriorFactorCall[UniformRectangularGaussianInputs]:
        return _prior_factor_native

    @property
    @override
    def inputs(self) -> UniformRectangularGaussianInputs:
        """Return the rectangular fields plus the Gaussian prior arrays."""
        return self._inputs_gauss
