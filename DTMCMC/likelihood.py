"""C 2023 Matthew C. Digman
abstract class to hold a likelihood object
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol, cast, override, runtime_checkable
from warnings import warn

import numpy as np
from numba import njit
from numba.core import types as nb_types
from numba.core.errors import NumbaError
from numba.extending import is_jitted, register_jitable
from numpy.typing import NDArray

from DTMCMC.correction_helpers import reflect_into_range
from DTMCMC.numba_backend import (
    NativeBackendUnsupportedError,
    NativeCheckBoundsCall,
    NativeCorrectBoundsCall,
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


type LoglikeFn = Callable[[NDArray[np.floating]], float]
type PriorDrawFn = Callable[[], NDArray[np.floating]]
type PriorFactorFn = Callable[[NDArray[np.floating]], float]
type ValidateBoundsFn = Callable[[NDArray[np.floating]], tuple[NDArray[np.floating], bool]]
type CheckBoundsFn = Callable[[NDArray[np.floating]], bool]
type CorrectBoundsFn = Callable[[NDArray[np.floating]], NDArray[np.floating]]


# compiled-dispatcher store keyed by the underlying function object: a
# memoized implementation function maps to one dispatcher, so equal-config
# objects share one compiled handle (the key reference also keeps the
# implementation alive, keeping its id stable)
_COMPILED_MEMO: dict[Callable[..., object], Callable[..., object]] = {}

# argument types used to force ahead-of-first-call compilation of a handle;
# they mirror the hot-path call (a 1D C-contiguous float64 parameter
# vector) but do not restrict the handle: the compiled dispatcher still
# lazily specializes for any other argument types it is later called with
_PARAMS_PROBE_ARGS: tuple[nb_types.Type, ...] = (nb_types.Array(nb_types.float64, 1, 'C'),)  # type: ignore[no-untyped-call]
_NO_PROBE_ARGS: tuple[nb_types.Type, ...] = ()


# TODO narrow types
def compile_handle[F: PriorDrawFn | LoglikeFn | ValidateBoundsFn | CheckBoundsFn | CorrectBoundsFn](
    fn: F, probe_args: tuple[nb_types.Type, ...], owner: str, role: str
) -> F:
    """Eagerly nopython-compile a behavior handle, falling back to plain Python.

    Compilation is forced without executing ``fn``, so no RNG stream is
    consumed. On a numba compilation failure the plain function is
    returned unchanged with a warning naming the owning class and role;
    the owning object then routes through the Python kernel path.
    """
    got = _COMPILED_MEMO.get(fn)
    if got is not None:
        return cast('F', got)
    # numba's Dispatcher is untyped generically, so the boundary is Any-typed
    handle = cast('Callable[..., object]', fn)
    dispatcher: Any = handle if is_jitted(handle) else njit(inline='always')(handle)
    try:
        dispatcher.compile(probe_args)
    except NumbaError as exc:
        warn(
            f'{owner} {role} failed nopython compilation and will run as plain Python: {exc}',
            RuntimeWarning,
            stacklevel=2,
        )
        return fn
    _COMPILED_MEMO[fn] = dispatcher
    return cast('F', dispatcher)


LIKELIHOOD_HANDLE_ROLES: tuple[tuple[str, str], ...] = (
    ('loglike_fn', 'get_loglike'),
    ('prior_draw_fn', 'prior_draw'),
    ('prior_factor_fn', 'prior_factor'),
    ('validate_bounds_fn', 'validate_bounds'),
    ('check_bounds_fn', 'check_bounds'),
    ('correct_bounds_fn', 'correct_bounds'),
)

# value-keyed store of shared handles: two objects constructed with equal
# baked constants reuse one compiled handle (and can therefore share one
# compiled kernel program downstream). Entries live for the process, which
# also keeps the baked arrays alive.
_HANDLE_MEMO_NO_PARAMS: dict[tuple[object, ...], PriorDrawFn] = {}
_HANDLE_MEMO_PARAMS: dict[tuple[object, ...], LoglikeFn | ValidateBoundsFn | CheckBoundsFn | CorrectBoundsFn] = {}


def _freeze(value: object) -> object:
    """Return a hashable, value-equal stand-in for a baked constant.

    NamedTuple ``inputs`` may hold numpy arrays, which are unhashable, so key
    the memo on a frozen surrogate that compares equal iff the array contents
    (and dtype/shape) match.
    """
    if isinstance(value, np.ndarray):
        return (
            value.shape,
            value.dtype.str,
            value.dtype.byteorder,
            value.flags['C_CONTIGUOUS'],
            value.flags['F_CONTIGUOUS'],
            value.tobytes(),
        )
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    try:
        hash(value)
    except TypeError as exc:
        msg = f'Everything in the inputs NamedTuple must be hashable or a rule to hash it provided. Failed for {type(value).__qualname__}'
        raise TypeError(msg) from exc

    return value


def build_from_handle_no_params[S](
    fn: Callable[[S], NDArray[np.floating]], inputs: S, owner: str, role: str
) -> PriorDrawFn:
    key = (owner, role, _freeze(inputs))
    got = _HANDLE_MEMO_NO_PARAMS.get(key)
    if got is not None:
        return got
    if role == 'prior_draw':

        def build_prior_draw() -> PriorDrawFn:
            def baked() -> NDArray[np.floating]:
                return fn(inputs)

            return baked

        got = compile_handle(build_prior_draw(), _NO_PROBE_ARGS, owner, role)
        _HANDLE_MEMO_NO_PARAMS[key] = got
        return got
    msg = f'Unrecognized role {role}'
    raise NotImplementedError(msg)


def build_from_handle_params[S, T: bool | float | NDArray[np.floating] | tuple[NDArray[np.floating], bool]](
    fn: Callable[[NDArray[np.floating], S], T], inputs: S, owner: str, role: str
) -> LoglikeFn | ValidateBoundsFn | CheckBoundsFn | CorrectBoundsFn:
    key = (owner, role, _freeze(inputs))
    got = _HANDLE_MEMO_PARAMS.get(key)
    if got is not None:
        return got

    if role == 'get_loglike':

        def build_loglike() -> LoglikeFn:
            def baked(params_in: NDArray[np.floating]) -> T:
                return fn(params_in, inputs)

            return cast('LoglikeFn', baked)

        got = compile_handle(build_loglike(), _PARAMS_PROBE_ARGS, owner, role)
        _HANDLE_MEMO_PARAMS[key] = got
        return got
    if role == 'prior_factor':

        def build_prior_factor() -> PriorFactorFn:
            def baked(params_in: NDArray[np.floating]) -> T:
                return fn(params_in, inputs)

            return cast('PriorFactorFn', baked)

        got = compile_handle(build_prior_factor(), _PARAMS_PROBE_ARGS, owner, role)
        _HANDLE_MEMO_PARAMS[key] = got
        return got
    if role == 'validate_bounds':

        def build_validate_bounds() -> ValidateBoundsFn:
            def baked(params_in: NDArray[np.floating]) -> T:
                return fn(params_in, inputs)

            return cast('ValidateBoundsFn', baked)

        got = compile_handle(build_validate_bounds(), _PARAMS_PROBE_ARGS, owner, role)
        _HANDLE_MEMO_PARAMS[key] = got
        return got
    if role == 'check_bounds':

        def build_check_bounds() -> CheckBoundsFn:
            def baked(params_in: NDArray[np.floating]) -> T:
                return fn(params_in, inputs)

            return cast('CheckBoundsFn', baked)

        got = compile_handle(build_check_bounds(), _PARAMS_PROBE_ARGS, owner, role)
        _HANDLE_MEMO_PARAMS[key] = got
        return got
    if role == 'correct_bounds':

        def build_correct_bounds() -> CorrectBoundsFn:
            def baked(params_in: NDArray[np.floating]) -> T:
                return fn(params_in, inputs)

            return cast('CorrectBoundsFn', baked)

        got = compile_handle(build_correct_bounds(), _PARAMS_PROBE_ARGS, owner, role)
        _HANDLE_MEMO_PARAMS[key] = got
        return got
    msg = f'Unrecognized role {role}'
    raise NotImplementedError(msg)


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


# @njit()
@register_jitable
def correct_bounds_rectangular(v: NDArray[np.floating], inputs: RectangularBoundsProtocol) -> NDArray[np.floating]:
    """Wrap parameters into range"""
    for itrp in range(v.size):
        v[itrp] = reflect_into_range(v[itrp], inputs.low_lims[itrp], inputs.high_lims[itrp])
    return v


# @njit()
@register_jitable
def prior_draw_rectangular(inputs: RectangularBoundsProtocol) -> NDArray[np.floating]:
    """Get a uniform prior draw with rectangular walls"""
    draw = np.zeros(inputs.n_par)
    for itrp in range(inputs.n_par):
        draw[itrp] = np.random.uniform(inputs.low_lims[itrp], inputs.high_lims[itrp])

    return draw


# @njit()
@register_jitable
def check_bounds_rectangular(v: NDArray[np.floating], inputs: RectangularBoundsProtocol) -> bool:
    """Check if a sample is within the prior range"""
    for itrp in range(v.size):
        if not inputs.low_lims[itrp] <= v[itrp] <= inputs.high_lims[itrp]:
            return False
    return True


# @njit()
@register_jitable
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


# @njit(inline='always')
@register_jitable
def prior_factor_rectangular(_params_in: NDArray[np.floating], _inputs: RectangularBoundsProtocol) -> float:
    """Log density of a uniform prior, up to an additive constant."""
    return 0.0


# @njit(inline='always')
@register_jitable
def _unavailable_loglike_fn(_params_in: NDArray[np.floating], _inputs: RectangularBoundsProtocol) -> float:
    """Return the per-class jitted ``(params, state) -> float`` log likelihood."""
    msg = 'No wired native log-likelihood binding for this path.'
    raise NativeBackendUnsupportedError(msg)


class AbstractNativeLikelihood[InputType](ABC):
    loglike_fn: NativeLoglikeCall[InputType]
    prior_draw_fn: NativePriorDrawCall[InputType]
    prior_factor_fn: NativePriorFactorCall[InputType]
    validate_bounds_fn: NativeValidateBoundsCall[InputType]
    check_bounds_fn: NativeCheckBoundsCall[InputType]
    correct_bounds_fn: NativeCorrectBoundsCall[InputType]

    def __init__(self) -> None:
        owner = type(self).__qualname__
        self._loglike_fn_baked: LoglikeFn = staticmethod(
            cast('LoglikeFn', build_from_handle_params(self.loglike_fn, self.inputs, owner, 'get_loglike'))
        )
        self._prior_draw_fn_baked: PriorDrawFn = staticmethod(
            build_from_handle_no_params(self.prior_draw_fn, self.inputs, owner, 'prior_draw')
        )
        self._prior_factor_fn_baked: PriorFactorFn = staticmethod(
            cast('PriorFactorFn', build_from_handle_params(self.prior_factor_fn, self.inputs, owner, 'prior_factor'))
        )
        self._validate_bounds_fn_baked: ValidateBoundsFn = staticmethod(
            cast(
                'ValidateBoundsFn',
                build_from_handle_params(self.validate_bounds_fn, self.inputs, owner, 'validate_bounds'),
            )
        )
        self._check_bounds_fn_baked: CheckBoundsFn = staticmethod(
            cast('CheckBoundsFn', build_from_handle_params(self.check_bounds_fn, self.inputs, owner, 'check_bounds'))
        )
        self._correct_bounds_fn_baked: CorrectBoundsFn = staticmethod(
            cast(
                'CorrectBoundsFn',
                build_from_handle_params(self.correct_bounds_fn, self.inputs, owner, 'correct_bounds'),
            )
        )
        return

    def prior_factor(self, params_in: NDArray[np.floating]) -> float:
        """Get the density factor for prior draws assuming a uniform prior"""
        return self._prior_factor_fn_baked(params_in)

    def prior_draw(self) -> NDArray[np.floating]:
        """Get a draw from the prior"""
        return self._prior_draw_fn_baked()

    def validate_bounds(self, params_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], bool]:
        """Check the parameters and correct if required."""
        return self._validate_bounds_fn_baked(params_in)

    def check_bounds(self, params_in: NDArray[np.floating]) -> bool:
        """Check the parameters."""
        return self._check_bounds_fn_baked(params_in)

    def correct_bounds(self, params_in: NDArray[np.floating]) -> NDArray[np.floating]:
        """Correct bounds without checking if required first."""
        return self._correct_bounds_fn_baked(params_in)

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Get the log likelihood at the specified parameters.
        input:
            params_in: a 1D float array of parameters
        output:
            logL: a scalar float likelihood
        """
        return self._loglike_fn_baked(params_in)

    @property
    @abstractmethod
    def inputs(self) -> InputType:
        """Get the read-only NamedTuple storing all likelihood input attributes."""
        ...


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
    correct_bounds_fn = staticmethod(correct_bounds_rectangular)
    check_bounds_fn = staticmethod(check_bounds_rectangular)

    def __init__(self, n_par: int, low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]) -> None:
        if low_lims.size != n_par or high_lims.size != n_par:
            msg = 'Input arrays must have size=n_par'
            raise ValueError(msg)

        low_arr = low_lims.copy()
        low_arr.setflags(write=False)

        high_arr = high_lims.copy()
        high_arr.setflags(write=False)

        self._inputs_rect: RectangularInputs = RectangularInputs(n_par, low_arr, high_arr)

        super().__init__()

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
    def n_par(self) -> int:
        """Read-only return of the number of parameters."""
        return self._inputs_rect.n_par

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
