"""C 2023 Matthew C. Digman
abstract class to hold a likelihood object
"""

from typing import Any, NamedTuple, Protocol, runtime_checkable

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
class CoreLikelihood(Protocol):
    """Minimal structural interface the sampler core requires of a likelihood.

    Likelihood objects are stateless: every attribute is fixed at
    construction and no method mutates the object. Evaluation counting is
    handled by the sampler's EvalAccounting, not the likelihood.
    """

    n_par: int

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

    def validate_bounds(self, params_in: NDArray[np.floating], /) -> tuple[NDArray[np.floating], bool]:
        """Check if the specified point is within the prior volume, try to correct if not."""
        ...


@runtime_checkable
class FisherSupportLikelihood(Protocol):
    """Additional structural interface the Fisher jump manager requires."""

    def correct_bounds(self, params_in: NDArray[np.floating], /) -> NDArray[np.floating]:
        """Correct the bounds for the input parameters to be within the prior range.
        input:
            params_in: the point with possibly incorrect parameters
        output:
            params_out: the point with corrected parameters
        """
        ...

    def get_epsilons(self, /) -> NDArray[np.floating]:
        """Get epsilons by dimension for fisher matrix computation."""
        ...


@runtime_checkable
class PresentableLikelihood(Protocol):
    """Presentation/output interface used by plotting and reporting helpers."""

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


@runtime_checkable
class AbstractLikelihood(CoreLikelihood, FisherSupportLikelihood, PresentableLikelihood, Protocol):
    """Full structural likelihood interface: core sampling, Fisher support, and presentation.

    The sampler itself only requires CoreLikelihood; FisherSupportLikelihood
    is validated by the Fisher jump manager, and PresentableLikelihood by
    the plotting helpers that consume it.
    """

    def check_bounds(self, params_in: NDArray[np.floating], /) -> bool:
        """Check if the specified point is within the prior volume
        input:
            params_in: the point to be checkout
        output:
            valid: a scalar boolean which is True is the point is valid in the prior volume and false otherwise
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


class RectangularNativeState(NamedTuple):
    """Runtime state bundle for the rectangular native likelihood defaults.

    Subclasses whose native functions need extra instance values define
    their own NamedTuple that also exposes ``n_par``/``low_lims``/
    ``high_lims`` fields (the rectangular default functions access them
    structurally) and override ``native_state``.
    """

    n_par: int
    low_lims: NDArray[np.float64]
    high_lims: NDArray[np.float64]


@njit(inline='always')
def _prior_draw_rectangular_native(state: RectangularNativeState) -> NDArray[np.floating]:
    """Uniform prior draw over the rectangular bounds in the state bundle."""
    return prior_draw_rectangular(state.n_par, state.low_lims, state.high_lims)


@njit(inline='always')
def _prior_factor_uniform_native(_params_in: NDArray[np.floating], _state: RectangularNativeState) -> float:
    """Log density of a uniform prior, up to an additive constant."""
    return 0.0


@njit(inline='always')
def _validate_bounds_rectangular_native(
    params_in: NDArray[np.floating], state: RectangularNativeState
) -> tuple[NDArray[np.floating], bool]:
    """Validate against the rectangular bounds in the state bundle."""
    return validate_bounds_rectangular(params_in, state.low_lims, state.high_lims)


class RectangularLikelihood(AbstractLikelihood):
    """Handle a likelihood with rectangular bounds
    by default assume a uniform prior

    The bounds are copied at construction into private read-only arrays
    exposed through the ``low_lims``/``high_lims`` properties, so they can
    be neither rebound nor mutated afterwards: the Python path and the
    native state bundles always agree. Subclasses opt into native execution
    by overriding bind_native_loglike (plus the other bind_native_* hooks
    and native_state when they override the corresponding Python methods or
    need extra state fields).
    """

    def __init__(self, n_par: int, low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]) -> None:
        low_arr = np.array(low_lims, dtype=np.float64)
        high_arr = np.array(high_lims, dtype=np.float64)
        low_arr.setflags(write=False)
        high_arr.setflags(write=False)
        self._low_lims: NDArray[np.float64] = low_arr
        self._high_lims: NDArray[np.float64] = high_arr

        assert self._low_lims.size == n_par
        assert self._high_lims.size == n_par

        self.n_par = n_par

    @property
    def low_lims(self) -> NDArray[np.float64]:
        """Read-only lower rectangular bounds (fixed at construction)."""
        return self._low_lims

    @property
    def high_lims(self) -> NDArray[np.float64]:
        """Read-only upper rectangular bounds (fixed at construction)."""
        return self._high_lims

    def correct_bounds(self, params_in: NDArray[np.floating]) -> NDArray[np.floating]:
        """Correct bounds for rectangular walls"""
        return correct_bounds_rectangular(params_in, self._low_lims, self._high_lims)

    def check_bounds(self, params_in: NDArray[np.floating]) -> bool:
        """Check bounds for rectangular walls"""
        return check_bounds_rectangular(params_in, self._low_lims, self._high_lims)

    def prior_factor(self, params_in: NDArray[np.floating]) -> float:
        """Get the density factor for prior draws assuming a uniform prior"""
        del params_in
        return 0.0

    def prior_draw(self) -> NDArray[np.floating]:
        """Get a draw from the prior"""
        return prior_draw_rectangular(self.n_par, self._low_lims, self._high_lims)

    def validate_bounds(self, params_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], bool]:
        """Check the parameters and correct if required."""
        return validate_bounds_rectangular(params_in, self._low_lims, self._high_lims)

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

    # the hooks are typed over an Any state so a subclass can narrow every
    # return to its own state bundle type (a concrete NamedTuple is not a
    # subtype of RectangularNativeState, so a nominal base type would make
    # precise subclass annotations override-incompatible); the rectangular
    # defaults themselves require the bundle to expose the
    # RectangularNativeState fields structurally

    def native_state(self) -> Any:
        """Return the runtime state bundle consumed by the native functions.

        Re-read at every block entry; subclasses with extra native state
        return their own NamedTuple exposing at least the rectangular
        fields.
        """
        return RectangularNativeState(self.n_par, self._low_lims, self._high_lims)

    def bind_native_loglike(self) -> NativeLoglikeCall[Any]:
        """Return the per-class jitted ``(params, state) -> float`` log likelihood.

        Subclasses opt into native execution by returning a module-level
        ``@njit`` function reading any instance values from the state
        bundle; the default declines, which keeps the sampler on the Python
        path.
        """
        msg = f'{type(self).__qualname__} does not provide a native log-likelihood binding'
        raise NativeBackendUnsupportedError(msg)

    def bind_native_prior_draw(self) -> NativePriorDrawCall[Any]:
        """Return the per-class jitted ``(state) -> params`` rectangular uniform draw."""
        return _prior_draw_rectangular_native

    def bind_native_prior_factor(self) -> NativePriorFactorCall[Any]:
        """Return the per-class jitted ``(params, state) -> float`` uniform log density."""
        return _prior_factor_uniform_native

    def bind_native_validate_bounds(self) -> NativeValidateBoundsCall[Any]:
        """Return the per-class jitted ``(params, state) -> (params, ok)`` validator."""
        return _validate_bounds_rectangular_native

    def bind_native(self) -> NativeLikelihoodFunctions[Any]:
        """Assemble the per-class native likelihood functions for the block kernel."""
        return NativeLikelihoodFunctions(
            loglike=self.bind_native_loglike(),
            prior_draw=self.bind_native_prior_draw(),
            prior_factor=self.bind_native_prior_factor(),
            validate_bounds=self.bind_native_validate_bounds(),
        )
