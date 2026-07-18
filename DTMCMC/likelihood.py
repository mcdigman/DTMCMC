"""C 2023 Matthew C. Digman
abstract class to hold a likelihood object
"""

from abc import ABC
from typing import Any, NamedTuple, Protocol, cast, override, runtime_checkable

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.correction_helpers import reflect_into_range
from DTMCMC.numba_backend import (
    NativeBackendUnsupportedError,
    NativeLikelihoodFunctions,
    NativeLoglikeCall,
    NativePriorDrawCall,
    NativePriorFactorCall,
    NativeValidateBoundsCall,
)


@runtime_checkable
class AbstractLikelihood[InputType](Protocol):
    """Structural interface the engine requires of a likelihood object.

    Likelihood objects are stateless: every attribute is fixed at
    construction and no method mutates the object. Evaluation counting is
    handled by the sampler's LikelihoodEvalTracker, not the likelihood.
    """

    @property
    def inputs(self) -> InputType:
        """Get the read-only NamedTuple storing all likelihood input attributes."""
        ...

    @property
    def n_par(self) -> int:
        """Get the read-only number of parameters."""
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


class RectangularBoundsProtocol(Protocol):
    @property
    def n_par(self) -> int: ...

    @property
    def low_lims(self) -> NDArray[np.floating]: ...

    @property
    def high_lims(self) -> NDArray[np.floating]: ...


class RectangularInputs(NamedTuple):
    """Compile-time inputs for the stateless rectangular likelihood parent class.

    Subclasses whose functions need extra instance values define
    their own NamedTuple that also exposes ``n_par``/``low_lims``/
    ``high_lims`` fields.
    """

    n_par: int
    low_lims: NDArray[np.floating]
    high_lims: NDArray[np.floating]


@njit()
def correct_bounds_rectangular(v: NDArray[np.floating], inputs: RectangularBoundsProtocol) -> NDArray[np.floating]:
    """Wrap parameters into range"""
    for itrp in range(v.size):
        v[itrp] = reflect_into_range(v[itrp], inputs.low_lims[itrp], inputs.high_lims[itrp])
    return v


@njit()
def prior_draw_rectangular(inputs: RectangularBoundsProtocol) -> NDArray[np.floating]:
    """Get a uniform prior draw with rectangular walls"""
    draw = np.zeros(inputs.n_par)
    for itrp in range(inputs.n_par):
        draw[itrp] = np.random.uniform(inputs.low_lims[itrp], inputs.high_lims[itrp])

    return draw


@njit()
def check_bounds_rectangular(v: NDArray[np.floating], inputs: RectangularBoundsProtocol) -> bool:
    """Check if a sample is within the prior range"""
    for itrp in range(v.size):
        if not inputs.low_lims[itrp] <= v[itrp] <= inputs.high_lims[itrp]:
            return False
    return True


@njit()
def validate_bounds_rectangular(
    params_in: NDArray[np.floating], inputs: RectangularBoundsProtocol
) -> tuple[NDArray[np.floating], bool]:
    success: bool = check_bounds_rectangular(params_in, inputs)
    if not success:
        # try to make the point in bounds and fail if unsuccesful
        new_point = correct_bounds_rectangular(params_in, inputs)
        success = check_bounds_rectangular(params_in, inputs)
    else:
        new_point = params_in
    return new_point, success


@njit(inline='always')
def prior_factor_rectangular(_params_in: NDArray[np.floating], _inputs: RectangularBoundsProtocol) -> float:
    """Log density of a uniform prior, up to an additive constant."""
    return 0.0


@njit(inline='always')
def _unavailable_loglike_fn(_params_in: NDArray[np.floating], _inputs: RectangularBoundsProtocol) -> float:
    """Return the per-class jitted ``(params, state) -> float`` log likelihood."""
    msg = 'No wired native log-likelihood binding for this path.'
    raise NativeBackendUnsupportedError(msg)


class AbstractNativeLikelihood[InputType](AbstractLikelihood[InputType], ABC):
    loglike_fn: NativeLoglikeCall[InputType]
    prior_draw_fn: NativePriorDrawCall[InputType]
    prior_factor_fn: NativePriorFactorCall[InputType]
    validate_bounds_fn: NativeValidateBoundsCall[InputType]

    @override
    def prior_factor(self, params_in: NDArray[np.floating]) -> float:
        """Get the density factor for prior draws assuming a uniform prior"""
        return self.prior_factor_fn(params_in, self.inputs)

    @override
    def prior_draw(self) -> NDArray[np.floating]:
        """Get a draw from the prior"""
        return self.prior_draw_fn(self.inputs)

    @override
    def validate_bounds(self, params_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], bool]:
        """Check the parameters and correct if required."""
        return self.validate_bounds_fn(params_in, self.inputs)

    @override
    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Get the log likelihood at the specified parameters.
        input:
            params_in: a 1D float array of parameters
        output:
            logL: a scalar float likelihood
        """
        return self.loglike_fn(params_in, self.inputs)


class RectangularLikelihood[InputType: RectangularBoundsProtocol](AbstractNativeLikelihood[InputType]):
    """Handle a likelihood with rectangular bounds
    by default assume a uniform prior

    The bounds arrays are copied and frozen read-only at construction:
    bindings bake them into compiled closures by reference, so they
    must never be rebound or mutated afterwards. Subclasses opt into native
    execution by overriding bind_native_loglike (and the other bind_native_*
    hooks when they override the corresponding Python methods).

    """

    prior_draw_fn: NativePriorDrawCall[InputType] = staticmethod(prior_draw_rectangular)
    prior_factor_fn: NativePriorFactorCall[InputType] = staticmethod(prior_factor_rectangular)
    validate_bounds_fn: NativeValidateBoundsCall[InputType] = staticmethod(validate_bounds_rectangular)
    loglike_fn: NativeLoglikeCall[InputType] = staticmethod(_unavailable_loglike_fn)

    def __init__(self, n_par: int, low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]) -> None:
        if low_lims.size != n_par or high_lims.size != n_par:
            msg = 'Input arrays must have size=n_par'
            raise ValueError(msg)

        low_arr = low_lims.copy()
        low_arr.setflags(write=False)

        high_arr = high_lims.copy()
        high_arr.setflags(write=False)

        self._inputs_rect: RectangularInputs = RectangularInputs(n_par, low_arr, high_arr)

    @property
    @override
    def inputs(self) -> InputType:
        """Read-only return of the inputs fixed at construction."""
        return cast('InputType', self._inputs_rect)

    @property
    def low_lims(self) -> NDArray[np.floating]:
        """Read-only lower rectangular bounds (fixed at construction)."""
        return self._inputs_rect.low_lims

    @property
    def high_lims(self) -> NDArray[np.floating]:
        """Read-only upper rectangular bounds (fixed at construction)."""
        return self._inputs_rect.high_lims

    @property
    @override
    def n_par(self) -> int:
        """Read-only return of the number of parameters."""
        return self._inputs_rect.n_par

    @override
    def correct_bounds(self, params_in: NDArray[np.floating]) -> NDArray[np.floating]:
        """Correct bounds for rectangular walls"""
        return correct_bounds_rectangular(params_in, self._inputs_rect)

    @override
    def check_bounds(self, params_in: NDArray[np.floating]) -> bool:
        """Check bounds for rectangular walls"""
        return check_bounds_rectangular(params_in, self._inputs_rect)

    @override
    def get_epsilons(self) -> NDArray[np.floating]:
        """Special helper for FisherJumpManager
        if this likelihood has special epsilons specified for fisher matrix jumps, get them here,
        otherwise just return zeros
        """
        return np.zeros(self.n_par)

    @override
    def get_labels(self) -> list[str]:
        """Get formatted axis labels for corner plots"""
        return [r'$v_' + str(itrp) + '$' for itrp in range(self.n_par)]

    @override
    def format_samples_output(
        self, samples_store: NDArray[np.floating], params_fid: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Purely a convenience function for making corner plots:
        if we desire to do any adjustments to input samples to make corner plots
        look nice, for example converting some dimension the raw parameter
        to Delta that parameter, or changing the units, we can do that here
        """
        return samples_store.copy(), params_fid.copy()

    # def bind_native_prior_draw(self) -> NativePriorDrawCall[Any]:
    #    """Return the per-class jitted ``(state) -> params`` rectangular uniform draw."""
    #    return prior_draw_rectangular

    # def bind_native_prior_factor(self) -> NativePriorFactorCall[Any]:
    #    """Return the per-class jitted ``(params, state) -> float`` uniform log density."""
    #    return prior_factor_rectangular

    # def bind_native_validate_bounds(self) -> NativeValidateBoundsCall[Any]:
    #     """Return the per-class jitted ``(params, state) -> (params, ok)`` validator."""
    #     return validate_bounds_rectangular

    def bind_native(self) -> NativeLikelihoodFunctions[Any]:
        """Assemble the per-class native likelihood functions for the block kernel."""
        return NativeLikelihoodFunctions(
            loglike=self.loglike_fn,
            prior_draw=self.prior_draw_fn,
            prior_factor=self.prior_factor_fn,
            validate_bounds=self.validate_bounds_fn,
        )
