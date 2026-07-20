"""C 2023 Matthew C. Digman
abstract class to hold a likelihood object
"""

from abc import ABC
from collections.abc import Callable
from typing import Any, Protocol, cast, override, runtime_checkable
from warnings import warn

import numpy as np
from numba import njit
from numba.core import types as nb_types
from numba.core.errors import NumbaError
from numba.extending import is_jitted
from numpy.typing import NDArray

from DTMCMC.correction_helpers import reflect_into_range

type LoglikeFn = Callable[[NDArray[np.floating]], float]
type PriorDrawFn = Callable[[], NDArray[np.floating]]
type PriorFactorFn = Callable[[NDArray[np.floating]], float]
type ValidateBoundsFn = Callable[[NDArray[np.floating]], tuple[NDArray[np.floating], bool]]
type CheckBoundsFn = Callable[[NDArray[np.floating]], bool]
type CorrectBoundsFn = Callable[[NDArray[np.floating]], NDArray[np.floating]]

# argument types used to force ahead-of-first-call compilation of a handle;
# they mirror the hot-path call (a 1D C-contiguous float64 parameter
# vector) but do not restrict the handle: the compiled dispatcher still
# lazily specializes for any other argument types it is later called with
_PARAMS_PROBE_ARGS: tuple[nb_types.Type, ...] = (nb_types.Array(nb_types.float64, 1, 'C'),)
_NO_PROBE_ARGS: tuple[nb_types.Type, ...] = ()


@runtime_checkable
class AbstractLikelihood(Protocol):
    """Structural interface the engine requires of a likelihood object.

    Likelihood objects are stateless: every attribute is fixed at
    construction and no method mutates the object. Evaluation counting is
    handled by the sampler's eval accounting, not the likelihood.
    """

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


# compiled-dispatcher store keyed by the underlying function object: a
# memoized implementation function maps to one dispatcher, so equal-config
# objects share one compiled handle (the key reference also keeps the
# implementation alive, keeping its id stable)
_COMPILED_MEMO: dict[Callable[..., object], Callable[..., object]] = {}


def compile_handle[F: Callable[..., object]](fn: F, probe_args: tuple[nb_types.Type, ...], owner: str, role: str) -> F:
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
    dispatcher: Any = fn if is_jitted(fn) else njit(inline='always')(fn)
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


# value-keyed store of shared handles: two objects constructed with equal
# baked constants reuse one compiled handle (and can therefore share one
# compiled kernel program downstream). Entries live for the process, which
# also keeps the baked arrays alive.
_HANDLE_MEMO: dict[tuple[object, ...], Callable[..., object]] = {}


def memoized_handle[F: Callable[..., object]](key: tuple[object, ...], build: Callable[[], F]) -> F:
    """Return the handle stored for ``key``, building and storing it if absent.

    ``key`` must start with a token unique to the calling factory and
    contain only the values that determine the built handle's behavior
    (the baked constants); arrays enter the key as ``tobytes()`` bytes.
    """
    got = _HANDLE_MEMO.get(key)
    if got is None:
        got = build()
        _HANDLE_MEMO[key] = got
    return cast('F', got)


LIKELIHOOD_HANDLE_ROLES: tuple[tuple[str, str], ...] = (
    ('loglike_fn', 'get_loglike'),
    ('prior_draw_fn', 'prior_draw'),
    ('prior_factor_fn', 'prior_factor'),
    ('validate_bounds_fn', 'validate_bounds'),
)


def role_handle_by_names(like_obj: AbstractLikelihood, fn_name: str, method_name: str) -> Callable[..., object]:
    """Return the single behavior implementation for one likelihood role.

    Handle-building likelihoods resolve the implementation into the
    ``*_fn`` attribute at construction; a likelihood without the attribute
    (a plain structural implementation of the protocol) contributes its
    bound method, which routes the graph through the Python kernel path.
    """
    fn: Callable[..., object] | None = getattr(like_obj, fn_name, None)
    if fn is None:
        return cast('Callable[..., object]', getattr(like_obj, method_name))
    return fn


# the casts below restate what the AbstractLikelihood protocol and the
# AbstractNativeLikelihood attribute declarations already guarantee about
# each role's call signature
def loglike_handle(like_obj: AbstractLikelihood) -> LoglikeFn:
    """Return the log-likelihood implementation for ``like_obj``."""
    return cast('LoglikeFn', role_handle_by_names(like_obj, 'loglike_fn', 'get_loglike'))


def prior_draw_handle(like_obj: AbstractLikelihood) -> PriorDrawFn:
    """Return the prior-draw implementation for ``like_obj``."""
    return cast('PriorDrawFn', role_handle_by_names(like_obj, 'prior_draw_fn', 'prior_draw'))


def prior_factor_handle(like_obj: AbstractLikelihood) -> PriorFactorFn:
    """Return the prior log-density implementation for ``like_obj``."""
    return cast('PriorFactorFn', role_handle_by_names(like_obj, 'prior_factor_fn', 'prior_factor'))


def validate_bounds_handle(like_obj: AbstractLikelihood) -> ValidateBoundsFn:
    """Return the bounds-validation implementation for ``like_obj``."""
    return cast('ValidateBoundsFn', role_handle_by_names(like_obj, 'validate_bounds_fn', 'validate_bounds'))


def _defining_class(cls: type, name: str) -> type:
    """Return the most-derived class in the MRO that defines ``name``."""
    for klass in cls.__mro__:
        if name in vars(klass):
            return klass
    msg = f'{cls.__qualname__} has no definition of {name}'
    raise AttributeError(msg)


class AbstractNativeLikelihood(ABC):
    """Base class building one compiled-or-plain handle per likelihood behavior.

    Each behavior (log likelihood, prior draw, prior density, bounds
    handling) has exactly one implementation, resolved once at
    construction:

    * A subclass returns a plain function from the matching ``_make_*``
      factory, closing over its constants as local variables (never over
      ``self``); construction compiles it to nopython eagerly, warning
      and keeping the plain function if compilation fails.
    * Alternatively a subclass overrides the public method itself; the
      bound method then becomes the handle and the object routes through
      the Python kernel path. An overriding method must not call its own
      handle attribute.

    The public methods are thin wrappers over the handles. A subclass
    defining both the method and the factory in the same class is
    ambiguous and fails at construction.
    """

    loglike_fn: LoglikeFn
    prior_draw_fn: PriorDrawFn
    prior_factor_fn: PriorFactorFn
    validate_bounds_fn: ValidateBoundsFn
    check_bounds_fn: CheckBoundsFn
    correct_bounds_fn: CorrectBoundsFn

    def _make_loglike(self) -> LoglikeFn | None:
        """Return the log-likelihood implementation to compile, if any."""
        return None

    def _make_prior_draw(self) -> PriorDrawFn | None:
        """Return the prior-draw implementation to compile, if any."""
        return None

    def _make_prior_factor(self) -> PriorFactorFn | None:
        """Return the prior log-density implementation to compile, if any."""
        return None

    def _make_validate_bounds(self) -> ValidateBoundsFn | None:
        """Return the bounds-validation implementation to compile, if any."""
        return None

    def _make_check_bounds(self) -> CheckBoundsFn | None:
        """Return the bounds-check implementation to compile, if any."""
        return None

    def _make_correct_bounds(self) -> CorrectBoundsFn | None:
        """Return the bounds-correction implementation to compile, if any."""
        return None

    def _resolve_handle[F: Callable[..., object]](
        self, method_name: str, factory_name: str, made: F | None, probe_args: tuple[nb_types.Type, ...]
    ) -> F:
        """Pick the single implementation for one behavior.

        The deepest definition wins: a public-method override defined
        strictly below the class defining the factory takes precedence
        and routes through Python; otherwise the factory's function is
        compiled. Defining both in the same class raises.
        """
        cls = type(self)
        method_cls = _defining_class(cls, method_name)
        factory_cls = _defining_class(cls, factory_name)
        if method_cls is not AbstractNativeLikelihood and issubclass(method_cls, factory_cls):
            if method_cls is factory_cls:
                msg = f'{cls.__qualname__} defines both {method_name} and {factory_name}; provide exactly one'
                raise TypeError(msg)
            return cast('F', getattr(self, method_name))
        if made is None:
            msg = f'{cls.__qualname__} provides neither a {factory_name} implementation nor a {method_name} override'
            raise TypeError(msg)
        return compile_handle(made, probe_args, cls.__qualname__, factory_name)

    def _init_handles(self) -> None:
        """Resolve and (where applicable) compile every behavior handle.

        Must be called exactly once, at the end of the subclass
        constructor: the ``_make_*`` factories read instance constants,
        so every attribute they consume must already be assigned.
        """
        self.loglike_fn = self._resolve_handle('get_loglike', '_make_loglike', self._make_loglike(), _PARAMS_PROBE_ARGS)
        self.prior_draw_fn = self._resolve_handle(
            'prior_draw', '_make_prior_draw', self._make_prior_draw(), _NO_PROBE_ARGS
        )
        self.prior_factor_fn = self._resolve_handle(
            'prior_factor', '_make_prior_factor', self._make_prior_factor(), _PARAMS_PROBE_ARGS
        )
        self.validate_bounds_fn = self._resolve_handle(
            'validate_bounds', '_make_validate_bounds', self._make_validate_bounds(), _PARAMS_PROBE_ARGS
        )
        self.check_bounds_fn = self._resolve_handle(
            'check_bounds', '_make_check_bounds', self._make_check_bounds(), _PARAMS_PROBE_ARGS
        )
        self.correct_bounds_fn = self._resolve_handle(
            'correct_bounds', '_make_correct_bounds', self._make_correct_bounds(), _PARAMS_PROBE_ARGS
        )

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Get the log likelihood at the specified parameters.
        input:
            params_in: a 1D float array of parameters
        output:
            logL: a scalar float likelihood
        """
        return self.loglike_fn(params_in)

    def prior_draw(self) -> NDArray[np.floating]:
        """Get a draw from the prior"""
        return self.prior_draw_fn()

    def prior_factor(self, params_in: NDArray[np.floating]) -> float:
        """Get the untempered log prior density for the input parameters"""
        return self.prior_factor_fn(params_in)

    def validate_bounds(self, params_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], bool]:
        """Check the parameters and correct if required."""
        return self.validate_bounds_fn(params_in)

    def check_bounds(self, params_in: NDArray[np.floating]) -> bool:
        """Check if the specified point is within the prior volume"""
        return self.check_bounds_fn(params_in)

    def correct_bounds(self, params_in: NDArray[np.floating]) -> NDArray[np.floating]:
        """Correct the input parameters to be within the prior range"""
        return self.correct_bounds_fn(params_in)


@njit()
def check_bounds_arrays(
    v: NDArray[np.floating], low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]
) -> bool:
    """Check if a sample is within the rectangular prior range"""
    for itrp in range(v.size):
        if not low_lims[itrp] <= v[itrp] <= high_lims[itrp]:
            return False
    return True


@njit()
def correct_bounds_arrays(
    v: NDArray[np.floating], low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Wrap parameters into the rectangular range"""
    for itrp in range(v.size):
        v[itrp] = reflect_into_range(v[itrp], low_lims[itrp], high_lims[itrp])
    return v


@njit()
def prior_draw_arrays(
    n_par: int, low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Get a uniform prior draw with rectangular walls"""
    draw = np.zeros(n_par)
    for itrp in range(n_par):
        draw[itrp] = np.random.uniform(low_lims[itrp], high_lims[itrp])

    return draw


@njit()
def validate_bounds_arrays(
    params_in: NDArray[np.floating], low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]
) -> tuple[NDArray[np.floating], bool]:
    """Check a point against the rectangular range, trying to correct it if outside"""
    success: bool = check_bounds_arrays(params_in, low_lims, high_lims)
    if not success:
        # try to make the point in bounds and fail if unsuccesful
        new_point = correct_bounds_arrays(params_in, low_lims, high_lims)
        success = check_bounds_arrays(params_in, low_lims, high_lims)
    else:
        new_point = params_in
    return new_point, success


@njit(inline='always')
def prior_factor_flat(_params_in: NDArray[np.floating]) -> float:
    """Log density of a uniform prior, up to an additive constant."""
    return 0.0


class RectangularLikelihood(AbstractNativeLikelihood):
    """Handle a likelihood with rectangular bounds
    by default assume a uniform prior

    The bounds arrays are copied and frozen read-only at construction:
    the default handles bake them into compiled closures by reference, so
    they must never be rebound or mutated afterwards. Subclasses implement
    ``_make_loglike`` (and override the other ``_make_*`` factories when
    the uniform-prior rectangular defaults do not apply), assigning every
    instance constant the factories read before calling this constructor.
    """

    def __init__(self, n_par: int, low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]) -> None:
        if low_lims.size != n_par or high_lims.size != n_par:
            msg = 'Input arrays must have size=n_par'
            raise ValueError(msg)

        low_arr = low_lims.copy()
        low_arr.setflags(write=False)

        high_arr = high_lims.copy()
        high_arr.setflags(write=False)

        self._n_par: int = int(n_par)
        self._low_lims: NDArray[np.floating] = low_arr
        self._high_lims: NDArray[np.floating] = high_arr
        self._bounds_key: tuple[object, ...] = (self._n_par, low_arr.tobytes(), high_arr.tobytes())
        self._init_handles()

    @property
    def low_lims(self) -> NDArray[np.floating]:
        """Read-only lower rectangular bounds (fixed at construction)."""
        return self._low_lims

    @property
    def high_lims(self) -> NDArray[np.floating]:
        """Read-only upper rectangular bounds (fixed at construction)."""
        return self._high_lims

    @property
    def n_par(self) -> int:
        """Read-only return of the number of parameters."""
        return self._n_par

    @override
    def _make_prior_draw(self) -> PriorDrawFn:
        n_par, low_lims, high_lims = self._n_par, self._low_lims, self._high_lims

        def build() -> PriorDrawFn:
            def prior_draw() -> NDArray[np.floating]:
                return prior_draw_arrays(n_par, low_lims, high_lims)

            return prior_draw

        return memoized_handle(('rectangular_prior_draw', *self._bounds_key), build)

    @override
    def _make_prior_factor(self) -> PriorFactorFn:
        return prior_factor_flat

    @override
    def _make_validate_bounds(self) -> ValidateBoundsFn:
        low_lims, high_lims = self._low_lims, self._high_lims

        def build() -> ValidateBoundsFn:
            def validate_bounds(params_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], bool]:
                return validate_bounds_arrays(params_in, low_lims, high_lims)

            return validate_bounds

        return memoized_handle(('rectangular_validate_bounds', *self._bounds_key), build)

    @override
    def _make_check_bounds(self) -> CheckBoundsFn:
        low_lims, high_lims = self._low_lims, self._high_lims

        def build() -> CheckBoundsFn:
            def check_bounds(params_in: NDArray[np.floating]) -> bool:
                return check_bounds_arrays(params_in, low_lims, high_lims)

            return check_bounds

        return memoized_handle(('rectangular_check_bounds', *self._bounds_key), build)

    @override
    def _make_correct_bounds(self) -> CorrectBoundsFn:
        low_lims, high_lims = self._low_lims, self._high_lims

        def build() -> CorrectBoundsFn:
            def correct_bounds(params_in: NDArray[np.floating]) -> NDArray[np.floating]:
                return correct_bounds_arrays(params_in, low_lims, high_lims)

            return correct_bounds

        return memoized_handle(('rectangular_correct_bounds', *self._bounds_key), build)

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
