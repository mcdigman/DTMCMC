"""C 2023 Matthew C. Digman
abstract class to hold a likelihood object
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.correction_helpers import reflect_into_range
from DTMCMC.numba_backend import NativeBackendUnsupportedError, NativeLikelihoodFunctions

if TYPE_CHECKING:
    from collections.abc import Callable


@runtime_checkable
class AbstractLikelihood(Protocol):
    """Structural interface the engine requires of a likelihood object.

    Likelihood objects are stateless: every attribute is fixed at
    construction and no method mutates the object. Evaluation counting is
    handled by the sampler's LikelihoodEvalTracker, not the likelihood.
    """

    n_par: int

    def __init__(self, n_par: int) -> None:
        """Initialize the likelihood.
        input: n_par integer, how many dimensions in the parameter space
        """
        ...

    def get_loglike(self, params_in: NDArray[np.floating], /) -> float:
        """Get the log likelihood at the specified parameters.
        input:
            params_in: a 1D float array of parameters
        output:
            logL: a scalar float likelihood
        """
        ...

    def prior_draw(self) -> NDArray[np.floating]:
        """Get a draw from the priors for this likelihood.
        output:
            params: a 1D float array of parameters
        """
        ...

    def prior_factor(self, params_in: NDArray[np.floating], /) -> float:
        """Get the untempered log prior density for the input parameters.
        input:
            params_in: the parameters to consider
        output:
            prior_factor: log prior density up to an additive constant
        """
        ...

    def correct_bounds(self, params_in: NDArray[np.floating], /) -> NDArray[np.floating]:
        """Correct the bounds for the input parameters to be within the prior range.
        input:
            params_in: the point with possibly incorrect parameters
        output:
            params_out: the point with corrected parameters
        """
        ...

    def check_bounds(self, params_in: NDArray[np.floating], /) -> bool:
        """Check if the specified point is within the prior volume
        input:
            params_in: the point to be checkout
        output:
            valid: a scalar boolean which is True is the point is valid in the prior volume and false otherwise
        """
        ...

    def validate_bounds(self, params_in: NDArray[np.floating], /) -> tuple[NDArray[np.floating], bool]:
        """Check if the specified point is within the prior volume, try to correct if not."""
        ...

    def get_epsilons(self, /) -> NDArray[np.floating]:
        """Get epsilons by dimension for fisher matrix computation."""
        ...

    def get_labels(self) -> list[str]:
        """Get formatted axis labels for corner plots"""
        ...

    def format_samples_output(
        self, samples_store: NDArray[np.floating], params_fid: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Purely a convenience function for making corner plots:
        if we desire to do any adjustments to input samples to make corner plots
        look nice, for example converting some dimension the raw parameter
        to Delta that parameter, or changing the units, we can do that here
        """
        ...


@njit()
def correct_bounds_rectangular(
    v: NDArray[np.floating], low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Wrap parameters into range"""
    for itrp in range(v.size):
        v[itrp] = reflect_into_range(v[itrp], low_lims[itrp], high_lims[itrp])
    return v


@njit()
def prior_draw_rectangular(
    n_par: int, low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Get a uniform prior draw with rectangular walls"""
    draw = np.zeros(n_par)
    for itrp in range(n_par):
        draw[itrp] = np.random.uniform(low_lims[itrp], high_lims[itrp])

    return draw


@njit()
def check_bounds_rectangular(
    v: NDArray[np.floating], low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]
) -> bool:
    """Check if a sample is within the prior range"""
    for itrp in range(v.size):
        if not low_lims[itrp] <= v[itrp] <= high_lims[itrp]:
            return False
    return True


@njit()
def validate_bounds_rectangular(
    params_in: NDArray[np.floating], low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]
) -> tuple[NDArray[np.floating], bool]:
    success: bool = check_bounds_rectangular(params_in, low_lims, high_lims)
    if not success:
        # try to make the point in bounds and fail if unsuccesful
        new_point = correct_bounds_rectangular(params_in, low_lims, high_lims)
        success = check_bounds_rectangular(params_in, low_lims, high_lims)
    else:
        new_point = params_in
    return new_point, success


@njit(inline='always')
def prior_factor_uniform(_params_in: NDArray[np.floating]) -> float:
    """Log density of a uniform prior, up to an additive constant."""
    return 0.0


class RectangularLikelihood(AbstractLikelihood):
    """Handle a likelihood with rectangular bounds
    by default assume a uniform prior

    The bounds arrays are copied and frozen read-only at construction:
    native bindings bake them into compiled closures by reference, so they
    must never be rebound or mutated afterwards. Subclasses opt into native
    execution by overriding bind_native_loglike (and the other bind_native_*
    hooks when they override the corresponding Python methods).
    """

    def __init__(self, n_par: int, low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]) -> None:
        low_arr = np.array(low_lims, dtype=np.float64)
        high_arr = np.array(high_lims, dtype=np.float64)
        low_arr.setflags(write=False)
        high_arr.setflags(write=False)
        self.low_lims: NDArray[np.floating] = low_arr
        self.high_lims: NDArray[np.floating] = high_arr

        assert self.low_lims.size == n_par
        assert self.high_lims.size == n_par

        self.n_par = n_par

    def correct_bounds(self, params_in: NDArray[np.floating]) -> NDArray[np.floating]:
        """Correct bounds for rectangular walls"""
        return correct_bounds_rectangular(params_in, self.low_lims, self.high_lims)

    def check_bounds(self, params_in: NDArray[np.floating]) -> bool:
        """Check bounds for rectangular walls"""
        return check_bounds_rectangular(params_in, self.low_lims, self.high_lims)

    def prior_factor(self, params_in: NDArray[np.floating]) -> float:
        """Get the density factor for prior draws assuming a uniform prior"""
        del params_in
        return 0.0

    def prior_draw(self) -> NDArray[np.floating]:
        """Get a draw from the prior"""
        return prior_draw_rectangular(self.n_par, self.low_lims, self.high_lims)

    def validate_bounds(self, params_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], bool]:
        """Check the parameters and correct if required."""
        return validate_bounds_rectangular(params_in, self.low_lims, self.high_lims)

    def get_epsilons(self) -> NDArray[np.floating]:
        """Special helper for FisherJumpManager
        if this likelihood has special epsilons specified for fisher matrix jumps, get them here,
        otherwise just return zeros
        """
        return np.zeros(self.n_par)

    def get_labels(self) -> list[str]:
        """Get formatted axis labels for corner plots"""
        return [r'$v_' + str(itrp) + '$' for itrp in range(self.n_par)]

    def format_samples_output(
        self, samples_store: NDArray[np.floating], params_fid: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Purely a convenience function for making corner plots:
        if we desire to do any adjustments to input samples to make corner plots
        look nice, for example converting some dimension the raw parameter
        to Delta that parameter, or changing the units, we can do that here
        """
        return samples_store.copy(), params_fid.copy()

    def bind_native_loglike(self) -> Callable[[NDArray[np.floating]], float]:
        """Return a jitted ``(params) -> float`` log-likelihood closure.

        Subclasses opt into native execution by returning an ``@njit``
        closure with their instance constants baked in; the default
        declines, which keeps the sampler on the Python path.
        """
        msg = f'{type(self).__qualname__} does not provide a native log-likelihood binding'
        raise NativeBackendUnsupportedError(msg)

    def bind_native_prior_draw(self) -> Callable[[], NDArray[np.floating]]:
        """Return a jitted ``() -> params`` draw with the rectangular bounds baked in."""
        n_par = self.n_par
        low_lims = self.low_lims
        high_lims = self.high_lims

        @njit(inline='always')
        def prior_draw_native() -> NDArray[np.floating]:
            return prior_draw_rectangular(n_par, low_lims, high_lims)

        return prior_draw_native

    def bind_native_prior_factor(self) -> Callable[[NDArray[np.floating]], float]:
        """Return a jitted ``(params) -> float`` log prior density (uniform default)."""
        return prior_factor_uniform

    def bind_native_validate_bounds(self) -> Callable[[NDArray[np.floating]], tuple[NDArray[np.floating], bool]]:
        """Return a jitted ``(params) -> (params, ok)`` closure with the bounds baked in."""
        low_lims = self.low_lims
        high_lims = self.high_lims

        @njit(inline='always')
        def validate_bounds_native(params_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], bool]:
            return validate_bounds_rectangular(params_in, low_lims, high_lims)

        return validate_bounds_native

    def bind_native(self) -> NativeLikelihoodFunctions:
        """Assemble the baked native likelihood functions for the block kernel."""
        return NativeLikelihoodFunctions(
            loglike=self.bind_native_loglike(),
            prior_draw=self.bind_native_prior_draw(),
            prior_factor=self.bind_native_prior_factor(),
            validate_bounds=self.bind_native_validate_bounds(),
        )
