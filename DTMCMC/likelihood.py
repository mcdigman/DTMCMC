"""C 2023 Matthew C. Digman
abstract class to hold a likelihood object
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol, cast, final, override, runtime_checkable
from warnings import warn

import numpy as np
from numba import njit
from numba.core import types as nb_types
from numba.core.errors import NumbaError
from numba.extending import is_jitted, register_jitable
from numpy.typing import NDArray

from DTMCMC.correction_helpers import reflect_into_range

type LoglikeFn = Callable[[NDArray[np.floating]], float]
type PriorDrawFn = Callable[[], NDArray[np.floating]]
type PriorFactorFn = Callable[[NDArray[np.floating]], float]
type PriorProposalFn = Callable[[NDArray[np.floating]], tuple[NDArray[np.floating], float, bool]]
type ValidateBoundsFn = Callable[[NDArray[np.floating]], tuple[NDArray[np.floating], bool]]
type CheckBoundsFn = Callable[[NDArray[np.floating]], bool]
type CorrectBoundsFn = Callable[[NDArray[np.floating]], NDArray[np.floating]]


@dataclass(frozen=True)
class NativeLikelihoodFunctions[InputType]:
    """Per-class function handles for the key functionality that needs them."""

    loglike: LoglikeFn
    prior_draw: PriorDrawFn
    prior_factor: PriorFactorFn
    prior_proposal: PriorProposalFn
    validate_bounds: ValidateBoundsFn
    check_bounds: CheckBoundsFn
    correct_bounds: CorrectBoundsFn


type NativeLoglikeCall[InputType] = Callable[[NDArray[np.floating], InputType], float]
type NativePriorDrawCall[InputType] = Callable[[InputType], NDArray[np.floating]]
type NativePriorFactorCall[InputType] = Callable[[NDArray[np.floating], InputType], float]
type NativePriorProposalCall[InputType] = Callable[
    [NDArray[np.floating], InputType], tuple[NDArray[np.floating], float, bool]
]
type NativeValidateBoundsCall[InputType] = Callable[
    [NDArray[np.floating], InputType], tuple[NDArray[np.floating], bool]
]
type NativeCorrectBoundsCall[InputType] = Callable[[NDArray[np.floating], InputType], NDArray[np.floating]]

type NativeCheckBoundsCall[InputType] = Callable[[NDArray[np.floating], InputType], bool]


@runtime_checkable
class AbstractLikelihood[InputType](Protocol):
    """Structural interface the engine requires of a likelihood object.

    Likelihood objects are stateless: every attribute is fixed at
    construction and no method mutates the object.
    """

    @property
    def loglike_fn(self) -> NativeLoglikeCall[InputType]: ...
    @property
    def prior_draw_fn(self) -> NativePriorDrawCall[InputType]: ...
    @property
    def prior_factor_fn(self) -> NativePriorFactorCall[InputType]: ...
    @property
    def prior_proposal_fn(self) -> NativePriorProposalCall[InputType]: ...
    @property
    def validate_bounds_fn(self) -> NativeValidateBoundsCall[InputType]: ...
    @property
    def check_bounds_fn(self) -> NativeCheckBoundsCall[InputType]: ...
    @property
    def correct_bounds_fn(self) -> NativeCorrectBoundsCall[InputType]: ...
    @property
    def loglike_fn_baked(self) -> LoglikeFn: ...
    @property
    def prior_draw_fn_baked(self) -> PriorDrawFn: ...
    @property
    def prior_factor_fn_baked(self) -> PriorFactorFn: ...
    @property
    def validate_bounds_fn_baked(self) -> ValidateBoundsFn: ...
    @property
    def check_bounds_fn_baked(self) -> CheckBoundsFn: ...
    @property
    def correct_bounds_fn_baked(self) -> CorrectBoundsFn: ...
    @property
    def prior_proposal_fn_baked(self) -> PriorProposalFn: ...

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

    def prior_proposal(self, params_in: NDArray[np.floating], /) -> tuple[NDArray[np.floating], float, bool]:
        """Get a prior proposal."""
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

    @property
    def bind_native(self, /) -> NativeLikelihoodFunctions[InputType]:
        """Get the bundle of functions as baked callables."""
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


class CompilationFallbackWarning(RuntimeWarning):
    """A native behavior handle failed nopython compilation and fell back to plain Python."""


class NativeBackendUnsupportedError(RuntimeError):
    """The requested object graph has no complete native binding."""


class NativeBackendCompilationError(RuntimeError):
    """The requested object graph has no complete native binding."""


# argument types used to force ahead-of-first-call compilation of a handle;
# they mirror the hot-path call (a 1D C-contiguous float64 parameter
# vector) but do not restrict the handle: the compiled dispatcher still
# lazily specializes for any other argument types it is later called with
_PARAMS_PROBE_ARGS: tuple[nb_types.Type, ...] = (nb_types.Array(nb_types.float64, 1, 'C'),)  # type: ignore[no-untyped-call]
_NO_PROBE_ARGS: tuple[nb_types.Type, ...] = ()


def compile_handle[
    F: Callable[
        ..., object
    ]  # PriorDrawFn | LoglikeFn | ValidateBoundsFn | CheckBoundsFn | CorrectBoundsFn | PriorFactorFn | PriorProposalFn
](fn: F, probe_args: tuple[nb_types.Type, ...]) -> tuple[F, str | None]:
    """Eagerly nopython-compile a behavior handle, falling back to plain Python.

    Compilation is forced without executing ``fn``, so no RNG stream is
    consumed. On a numba compilation failure the plain function is
    returned unchanged with the compilation error; the owning likelihood
    aggregates all of its failed behavior handles into one warning.
    """
    # numba's Dispatcher is untyped generically, so the boundary is Any-typed
    handle = cast('Callable[..., object]', fn)
    dispatcher: Any = handle if is_jitted(handle) else njit(inline='always')(handle)
    try:
        dispatcher.compile(probe_args)
    except NumbaError as exc:
        return fn, str(exc)
    return cast('F', dispatcher), None


LIKELIHOOD_HANDLE_ROLES: tuple[tuple[str, str], ...] = (
    ('loglike_fn', 'get_loglike'),
    ('prior_draw_fn', 'prior_draw'),
    ('prior_factor_fn', 'prior_factor'),
    ('prior_proposal_fn', 'prior_proposal'),
    ('validate_bounds_fn', 'validate_bounds'),
    ('check_bounds_fn', 'check_bounds'),
    ('correct_bounds_fn', 'correct_bounds'),
)

# value-keyed store of shared handles: two objects constructed with equal
# baked constants reuse one compiled handle (and can therefore share one
# compiled kernel program downstream). Entries live for the process, which
# also keeps the baked arrays alive.
_HANDLE_MEMO_NO_PARAMS: dict[tuple[object, ...], tuple[PriorDrawFn, str | None]] = {}
_HANDLE_MEMO_PARAMS: dict[
    tuple[object, ...],
    tuple[LoglikeFn | ValidateBoundsFn | CheckBoundsFn | CorrectBoundsFn | PriorFactorFn | PriorProposalFn, str | None],
] = {}


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
) -> tuple[PriorDrawFn, str | None]:
    key = (owner, role, _freeze(inputs))
    got = _HANDLE_MEMO_NO_PARAMS.get(key)
    if got is not None:
        return got
    if role == 'prior_draw':

        def build_prior_draw() -> PriorDrawFn:
            def baked() -> NDArray[np.floating]:
                return fn(inputs)

            return baked

        got = compile_handle(build_prior_draw(), _NO_PROBE_ARGS)
        _HANDLE_MEMO_NO_PARAMS[key] = got
        return got
    msg = f'Unrecognized role {role}'
    raise NotImplementedError(msg)


def build_from_handle_params[
    S,
    T: bool
    | float
    | NDArray[np.floating]
    | tuple[NDArray[np.floating], bool]
    | tuple[NDArray[np.floating], float, bool],
](
    fn: Callable[[NDArray[np.floating], S], T], inputs: S, owner: str, role: str
) -> tuple[
    LoglikeFn | ValidateBoundsFn | CheckBoundsFn | CorrectBoundsFn | PriorProposalFn | PriorFactorFn, str | None
]:
    key = (owner, role, _freeze(inputs))
    got = _HANDLE_MEMO_PARAMS.get(key)
    if got is not None:
        return got

    if role == 'get_loglike':

        def build_loglike() -> LoglikeFn:
            def baked(params_in: NDArray[np.floating]) -> T:
                return fn(params_in, inputs)

            return cast('LoglikeFn', baked)

        got = compile_handle(build_loglike(), _PARAMS_PROBE_ARGS)
        _HANDLE_MEMO_PARAMS[key] = got
        return got
    if role == 'prior_factor':

        def build_prior_factor() -> PriorFactorFn:
            def baked(params_in: NDArray[np.floating]) -> T:
                return fn(params_in, inputs)

            return cast('PriorFactorFn', baked)

        got = compile_handle(build_prior_factor(), _PARAMS_PROBE_ARGS)
        _HANDLE_MEMO_PARAMS[key] = got
        return got
    if role == 'prior_proposal':

        def build_prior_proposal() -> PriorProposalFn:
            def baked(params_in: NDArray[np.floating]) -> T:
                return fn(params_in, inputs)

            return cast('PriorProposalFn', baked)

        got = compile_handle(build_prior_proposal(), _PARAMS_PROBE_ARGS)
        _HANDLE_MEMO_PARAMS[key] = got
        return got
    if role == 'validate_bounds':

        def build_validate_bounds() -> ValidateBoundsFn:
            def baked(params_in: NDArray[np.floating]) -> T:
                return fn(params_in, inputs)

            return cast('ValidateBoundsFn', baked)

        got = compile_handle(build_validate_bounds(), _PARAMS_PROBE_ARGS)
        _HANDLE_MEMO_PARAMS[key] = got
        return got
    if role == 'check_bounds':

        def build_check_bounds() -> CheckBoundsFn:
            def baked(params_in: NDArray[np.floating]) -> T:
                return fn(params_in, inputs)

            return cast('CheckBoundsFn', baked)

        got = compile_handle(build_check_bounds(), _PARAMS_PROBE_ARGS)
        _HANDLE_MEMO_PARAMS[key] = got
        return got
    if role == 'correct_bounds':

        def build_correct_bounds() -> CorrectBoundsFn:
            def baked(params_in: NDArray[np.floating]) -> T:
                return fn(params_in, inputs)

            return cast('CorrectBoundsFn', baked)

        got = compile_handle(build_correct_bounds(), _PARAMS_PROBE_ARGS)
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


# @njit(inline='always')
@register_jitable
def prior_factor_rectangular(_params_in: NDArray[np.floating], _inputs: RectangularBoundsProtocol) -> float:
    """Log density of a uniform prior, up to an additive constant."""
    return 0.0


# @njit(inline='always')
@register_jitable
def _unavailable_loglike_fn(_params_in: NDArray[np.floating], _inputs: RectangularBoundsProtocol) -> float:
    """Return the per-class jitted ``(params, inputs) -> float`` log likelihood."""
    msg = 'No user-specified likelihood function provided.'
    raise NativeBackendUnsupportedError(msg)


def _validate_bounds_function[InputType](
    check_bounds_fn: NativeCheckBoundsCall[InputType], correct_bounds_fn: NativeCorrectBoundsCall[InputType]
) -> NativeValidateBoundsCall[InputType]:
    @register_jitable
    def validate_bounds_fn(params_in: NDArray[np.floating], inputs: InputType) -> tuple[NDArray[np.floating], bool]:
        success: bool = check_bounds_fn(params_in, inputs)
        if not success:
            # try to make the point in bounds and fail if unsuccesful
            new_point = correct_bounds_fn(params_in, inputs)
            success = check_bounds_fn(new_point, inputs)
        else:
            new_point = params_in
        return new_point, success

    return cast('NativeValidateBoundsCall[InputType]', validate_bounds_fn)


def _prior_proposal_function[InputType](
    prior_draw_fn: NativePriorDrawCall[InputType], prior_factor_fn: NativePriorFactorCall[InputType]
) -> NativePriorProposalCall[InputType]:
    @register_jitable
    def proposal_function(
        params_in: NDArray[np.floating], inputs: InputType
    ) -> tuple[NDArray[np.floating], float, bool]:
        params_new = prior_draw_fn(inputs)
        prior_factor = prior_factor_fn(params_in, inputs) - prior_factor_fn(params_new, inputs)
        return params_new, prior_factor, True

    return cast('NativePriorProposalCall[InputType]', proposal_function)


class AbstractNativeLikelihood[InputType](ABC):
    def __init__(self) -> None:
        owner = type(self).__qualname__
        failures: list[tuple[str, str]] = []

        loglike_handle, failure = build_from_handle_params(self.loglike_fn, self.inputs, owner, 'get_loglike')
        self._loglike_fn_baked: LoglikeFn = cast('LoglikeFn', loglike_handle)
        if failure is not None:
            failures.append(('get_loglike', failure))

        self._prior_draw_fn_baked, failure = build_from_handle_no_params(
            self.prior_draw_fn, self.inputs, owner, 'prior_draw'
        )
        if failure is not None:
            failures.append(('prior_draw', failure))

        prior_factor_handle, failure = build_from_handle_params(
            self.prior_factor_fn, self.inputs, owner, 'prior_factor'
        )
        self._prior_factor_fn_baked: PriorFactorFn = cast('PriorFactorFn', prior_factor_handle)
        if failure is not None:
            failures.append(('prior_factor', failure))

        check_bounds_handle, failure = build_from_handle_params(
            self.check_bounds_fn, self.inputs, owner, 'check_bounds'
        )
        self._check_bounds_fn_baked: CheckBoundsFn = cast('CheckBoundsFn', check_bounds_handle)
        if failure is not None:
            failures.append(('check_bounds', failure))

        correct_bounds_handle, failure = build_from_handle_params(
            self.correct_bounds_fn, self.inputs, owner, 'correct_bounds'
        )
        self._correct_bounds_fn_baked: CorrectBoundsFn = cast(
            'CorrectBoundsFn',
            correct_bounds_handle,
        )
        if failure is not None:
            failures.append(('correct_bounds', failure))

        self._validate_bounds_fn = _validate_bounds_function(self.check_bounds_fn, self.correct_bounds_fn)

        validate_bounds_handle, failure = build_from_handle_params(
            self.validate_bounds_fn, self.inputs, owner, 'validate_bounds'
        )
        self._validate_bounds_fn_baked: ValidateBoundsFn = cast(
            'ValidateBoundsFn',
            validate_bounds_handle,
        )
        if failure is not None:
            failures.append(('validate_bounds', failure))

        self._prior_proposal_fn = _prior_proposal_function(self.prior_draw_fn, self.prior_factor_fn)
        prior_proposal_handle, failure = build_from_handle_params(
            self.prior_proposal_fn, self.inputs, owner, 'prior_proposal'
        )
        self._prior_proposal_fn_baked = cast(
            'PriorProposalFn',
            prior_proposal_handle,
        )
        if failure is not None:
            failures.append(('prior_proposal', failure))

        if failures:
            roles = ', '.join(role for role, _failure in failures)
            details = '\n'.join(f'{role}: {failure}' for role, failure in failures)
            warn(
                f'{owner} {roles} failed nopython compilation and will run as plain Python:\n{details}',
                CompilationFallbackWarning,
                stacklevel=2,
            )

        return

    @property
    @abstractmethod
    def loglike_fn(self) -> NativeLoglikeCall[InputType]: ...
    @property
    @abstractmethod
    def prior_draw_fn(self) -> NativePriorDrawCall[InputType]: ...
    @property
    @abstractmethod
    def prior_factor_fn(self) -> NativePriorFactorCall[InputType]: ...
    @property
    @abstractmethod
    def validate_bounds_fn(self) -> NativeValidateBoundsCall[InputType]: ...
    @property
    @abstractmethod
    def check_bounds_fn(self) -> NativeCheckBoundsCall[InputType]: ...
    @property
    @abstractmethod
    def correct_bounds_fn(self) -> NativeCorrectBoundsCall[InputType]: ...
    @property
    @final
    def prior_proposal_fn(self) -> NativePriorProposalCall[InputType]:
        return self._prior_proposal_fn

    @property
    @final
    def loglike_fn_baked(self) -> LoglikeFn:
        return self._loglike_fn_baked

    @property
    @final
    def prior_draw_fn_baked(self) -> PriorDrawFn:
        return self._prior_draw_fn_baked

    @property
    @final
    def prior_factor_fn_baked(self) -> PriorFactorFn:
        return self._prior_factor_fn_baked

    @property
    @final
    def validate_bounds_fn_baked(self) -> ValidateBoundsFn:
        return self._validate_bounds_fn_baked

    @property
    @final
    def check_bounds_fn_baked(self) -> CheckBoundsFn:
        return self._check_bounds_fn_baked

    @property
    @final
    def correct_bounds_fn_baked(self) -> CorrectBoundsFn:
        return self._correct_bounds_fn_baked

    @property
    @final
    def prior_proposal_fn_baked(self) -> PriorProposalFn:
        return self._prior_proposal_fn_baked

    @final
    def prior_factor(self, params_in: NDArray[np.floating]) -> float:
        """Get the density factor for prior draws assuming a uniform prior"""
        return self.prior_factor_fn_baked(params_in)

    @final
    def prior_draw(self) -> NDArray[np.floating]:
        """Get a draw from the prior"""
        return self.prior_draw_fn_baked()

    @final
    def prior_proposal(self, params_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], float, bool]:
        """Get a prior proposal."""
        return self.prior_proposal_fn_baked(params_in)

    @final
    def validate_bounds(self, params_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], bool]:
        """Check the parameters and correct if required."""
        return self.validate_bounds_fn_baked(params_in)

    @final
    def check_bounds(self, params_in: NDArray[np.floating]) -> bool:
        """Check the parameters."""
        return self.check_bounds_fn_baked(params_in)

    @final
    def correct_bounds(self, params_in: NDArray[np.floating]) -> NDArray[np.floating]:
        """Correct bounds without checking if required first."""
        return self.correct_bounds_fn_baked(params_in)

    @final
    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Get the log likelihood at the specified parameters.
        input:
            params_in: a 1D float array of parameters
        output:
            logL: a scalar float likelihood
        """
        return self.loglike_fn_baked(params_in)

    @property
    @final
    def bind_native(self) -> NativeLikelihoodFunctions[Any]:
        """Assemble the per-class native likelihood functions for the block kernel."""
        return NativeLikelihoodFunctions(
            loglike=self.loglike_fn_baked,
            prior_draw=self.prior_draw_fn_baked,
            prior_factor=self.prior_factor_fn_baked,
            validate_bounds=self.validate_bounds_fn_baked,
            prior_proposal=self.prior_proposal_fn_baked,
            check_bounds=self.check_bounds_fn_baked,
            correct_bounds=self.correct_bounds_fn_baked,
        )

    @property
    @abstractmethod
    def inputs(self) -> InputType:
        """Get the read-only NamedTuple storing all likelihood input attributes."""
        ...

    @property
    @abstractmethod
    def n_par(self) -> int:
        """Number of parameters."""
        ...

    @abstractmethod
    def get_epsilons(self) -> NDArray[np.floating]:
        """Special helper for FisherJumpManager
        if this likelihood has special epsilons specified for fisher matrix jumps, get them here,
        otherwise just return zeros
        """
        ...

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


class RectangularLikelihood[InputType: RectangularBoundsProtocol](AbstractNativeLikelihood[InputType]):
    """Handle a likelihood with rectangular bounds
    by default assume a uniform prior

    The bounds arrays are copied and frozen read-only at construction:
    bindings bake them into compiled closures by reference, so they
    must never be rebound or mutated afterwards. Subclasses opt into native
    execution by overriding bind_native_loglike (and the other bind_native_*
    hooks when they override the corresponding Python methods).

    """

    _prior_draw_fn: NativePriorDrawCall[InputType] = staticmethod(prior_draw_rectangular)
    _prior_factor_fn: NativePriorFactorCall[InputType] = staticmethod(prior_factor_rectangular)
    _loglike_fn: NativeLoglikeCall[InputType] = staticmethod(_unavailable_loglike_fn)
    _correct_bounds_fn: NativeCorrectBoundsCall[InputType] = staticmethod(correct_bounds_rectangular)
    _check_bounds_fn: NativeCheckBoundsCall[InputType] = staticmethod(check_bounds_rectangular)

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
    def loglike_fn(self) -> NativeLoglikeCall[InputType]:
        return self._loglike_fn

    @property
    @override
    def prior_draw_fn(self) -> NativePriorDrawCall[InputType]:
        return self._prior_draw_fn

    @property
    @override
    def prior_factor_fn(self) -> NativePriorFactorCall[InputType]:
        return self._prior_factor_fn

    @property
    @override
    def validate_bounds_fn(self) -> NativeValidateBoundsCall[InputType]:
        return self._validate_bounds_fn

    @property
    @override
    def check_bounds_fn(self) -> NativeCheckBoundsCall[InputType]:
        return self._check_bounds_fn

    @property
    @override
    def correct_bounds_fn(self) -> NativeCorrectBoundsCall[InputType]:
        return self._correct_bounds_fn

    @property
    @override
    def inputs(self) -> InputType:
        """Read-only return of the inputs fixed at construction."""
        return cast('InputType', self._inputs_rect)

    @property
    @final
    def low_lims(self) -> NDArray[np.floating]:
        """Read-only lower rectangular bounds (fixed at construction)."""
        return self._inputs_rect.low_lims

    @property
    @final
    def high_lims(self) -> NDArray[np.floating]:
        """Read-only upper rectangular bounds (fixed at construction)."""
        return self._inputs_rect.high_lims

    @property
    @override
    def n_par(self) -> int:
        """Read-only return of the number of parameters."""
        return self._inputs_rect.n_par

    @override
    def get_epsilons(self) -> NDArray[np.floating]:
        """Special helper for FisherJumpManager
        if this likelihood has special epsilons specified for fisher matrix jumps, get them here,
        otherwise just return zeros
        """
        return np.zeros(self.n_par)
