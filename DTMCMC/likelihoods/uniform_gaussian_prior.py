"""Constant likelihood with an independent Gaussian prior in every dimension."""

from typing import NamedTuple, override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import RectangularLikelihood


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


class UniformRectangularGaussianInputs(NamedTuple):
    """Compile-time fixed inputs."""

    n_par: int
    low_lims: NDArray[np.floating]
    high_lims: NDArray[np.floating]
    prior_mean: NDArray[np.floating]
    prior_std: NDArray[np.floating]


@njit(inline='always')
def _loglike_native(params_in: NDArray[np.floating], _inputs: UniformRectangularGaussianInputs) -> float:
    """Per-class native log likelihood."""
    return get_loglike(params_in)


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

    loglike_fn = staticmethod(_loglike_native)
    prior_draw_fn = staticmethod(_prior_draw_native)
    prior_factor_fn = staticmethod(_prior_factor_native)

    def __init__(self, n_par: int = 4, prior_mean: float = 0.0, prior_std: float = 1.0) -> None:
        if prior_std <= 0.0:
            msg = 'prior_std must be positive'
            raise ValueError(msg)
        super().__init__(n_par, np.full(n_par, -np.inf), np.full(n_par, np.inf))

        prior_mean_arr = np.full(n_par, prior_mean)
        prior_std_arr = np.full(n_par, prior_std)
        prior_mean_arr.setflags(write=False)
        prior_std_arr.setflags(write=False)
        self._inputs_gauss: UniformRectangularGaussianInputs = UniformRectangularGaussianInputs(
            self.n_par, self.low_lims, self.high_lims, prior_mean_arr, prior_std_arr
        )

    # def get_loglike(self, params_in: NDArray[np.floating]) -> float:
    #    """Evaluate the constant likelihood."""
    #    return get_loglike(params_in)

    # def prior_draw(self) -> NDArray[np.floating]:
    #    """Draw from the Gaussian prior."""
    #    return gaussian_prior_draw(self.n_par, self.prior_mean, self.prior_std)

    # def prior_factor(self, params_in: NDArray[np.floating]) -> float:
    #    """Return the Gaussian log prior density up to a constant."""
    #    return gaussian_prior_factor(params_in, self.prior_mean, self.prior_std)

    @property
    @override
    def inputs(self) -> UniformRectangularGaussianInputs:
        """Return the rectangular fields plus the Gaussian prior arrays."""
        return self._inputs_gauss

    # def bind_native_loglike(self) -> NativeLoglikeCall[UniformRectangularGaussianInputs]:
    #    """Return the per-class native log likelihood."""
    #    return _loglike_native

    # def bind_native_prior_draw(self) -> NativePriorDrawCall[UniformRectangularGaussianInputs]:
    #    """Return the per-class native Gaussian prior draw."""
    #    return _prior_draw_native

    # def bind_native_prior_factor(self) -> NativePriorFactorCall[UniformRectangularGaussianInputs]:
    #    """Return the per-class native Gaussian log prior density."""
    #    return _prior_factor_native
