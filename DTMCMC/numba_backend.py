"""Serial block programs assembled from per-class native bindings.

One control flow serves every kernel backend: the block-advance program is
assembled once from plain recursive closure factories (the Numba equivalent
of ``functools.partial``) over the components' bound functions, in two
flavors that differ only in the wrapper applied to each assembled link —
``'numba'`` nopython-compiles the chain, ``'python'`` runs the identical
call chain uncompiled while still invoking the same bound handles. Both
flavors therefore call the same functions in the same order, so their
fixed-seed behavior is bit-identical whenever every handle is compiled.
Compiled programs are cached by the identity of the bound functions, so
structurally identical samplers share one compiled kernel and Numba's own
dispatcher handles any residual signature specialization. No strings, no
``exec``: every line is visible to linters and type checkers.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, cast
from warnings import warn

import numpy as np
from numba import njit, typeof
from numba.core.errors import NumbaError
from numba.extending import is_jitted

# runtime import, not TC002: numba evaluates the lazy annotations of the
# assembled links when the flavor wrapper compiles them, which ruff cannot
# see because njit arrives through the wrapper rather than as a decorator
from numpy.typing import NDArray  # noqa: TC002

from DTMCMC.jump_manager import (
    NativeJumpCall,
    NativePostStepCall,
    choose_prob_helper,
)
from DTMCMC.likelihood import (
    NativeBackendCompilationError,
    NativeBackendUnsupportedError,
    NativeLikelihoodFunctions,
)
from DTMCMC.mcmc_kernel_helpers import mcmc_decision_helper

if TYPE_CHECKING:
    from DTMCMC.eval_accounting import EvalAccounting
    from DTMCMC.exchange_manager import NativeExchangeFunctions
    from DTMCMC.likelihood import (
        AbstractLikelihood,
    )
    from DTMCMC.proposal_manager import AbstractProposalManager
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder
    from DTMCMC.tracker_manager import TrackerManager

type KernelBackendMode = Literal['auto', 'numba', 'python']
type KernelFlavor = Literal['numba', 'python']

_VALID_MODES: tuple[KernelBackendMode, ...] = ('auto', 'numba', 'python')

# Heterogeneous per-manager rows: Callable parameters are contravariant, so
# no common state parameter type can express "each row takes exactly its own
# manager's state"; Any is the deliberate element type here.
type ManagerJumpCalls = tuple[tuple[NativeJumpCall[Any], ...], ...]
type ManagerPostSteps = tuple[NativePostStepCall[Any] | None, ...]


def _defining_class(cls: type, name: str) -> type | None:
    """Return the most-derived class in the MRO that defines ``name``."""
    for klass in cls.__mro__:
        if name in vars(klass):
            return klass
    return None


def _stale_native_override(cls: type, method_name: str, hook_names: tuple[str, ...]) -> str | None:
    """Detect a Python-behavior override that outruns its native binding.

    Every program flavor executes the bound functions, so a subclass that
    overrides a paired Python method without also overriding a hook would
    have its override silently ignored — the inherited binding would run
    the ancestor's behavior. Absent members are protocol-conformance
    problems, and hooks held as instance attributes are invisible to class
    introspection and assumed consistent; only the provable case — the
    method defined strictly below every hook — is reported.
    """
    py_cls = _defining_class(cls, method_name)
    if py_cls is None:
        return None
    native_cls: type | None = None
    for hook_name in hook_names:
        hook_cls = _defining_class(cls, hook_name)
        if hook_cls is not None and (native_cls is None or issubclass(hook_cls, native_cls)):
            native_cls = hook_cls
    if native_cls is not None and py_cls is not native_cls and issubclass(py_cls, native_cls):
        return f'{cls.__qualname__} overrides {method_name} without overriding {" or ".join(hook_names)}'
    return None


_LIKELIHOOD_DIVERGENCE_PAIRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('get_loglike', ('loglike_fn', 'bind_native')),
    ('prior_draw', ('prior_draw_fn', 'bind_native')),
    ('prior_factor', ('prior_factor_fn', 'bind_native')),
    ('prior_proposal', ('prior_proposal_fn', 'bind_native')),
    ('validate_bounds', ('validate_bounds_fn', 'bind_native')),
    ('check_bounds', ('check_bounds_fn', 'bind_native')),
    ('correct_bounds', ('correct_bounds_fn', 'bind_native')),
)

_EXCHANGE_DIVERGENCE_PAIRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('is_exchange_step', ('is_exchange_step_native', 'bind_native')),
    ('do_ptmcmc_exchange', ('exchange_native', 'bind_native')),
)


def _manager_divergence(manager: object) -> str | None:
    """Require per-step behavior to come from the native post-step binding.

    Both program flavors run the assembled post-step chain, never the
    ``post_step_update`` method, so per-step behavior expressed only as a
    method override would be silently skipped.
    """
    cls = type(manager)
    hook_cls = _defining_class(cls, 'bind_native_post_step')
    py_cls = _defining_class(cls, 'post_step_update')
    if hook_cls is None:
        if py_cls is not None:
            return f'{cls.__qualname__} defines post_step_update without defining bind_native_post_step'
        return None
    if py_cls is not None and py_cls is not hook_cls and issubclass(py_cls, hook_cls):
        return f'{cls.__qualname__} overrides post_step_update without overriding bind_native_post_step'
    return None


def find_native_divergences[LikelihoodType: AbstractLikelihood[NamedTuple]](
    proposal_manager: AbstractProposalManager[LikelihoodType], like_obj: LikelihoodType
) -> list[str]:
    """List Python-contract overrides that would diverge from the bound behavior.

    Checked at sampler construction in every kernel backend mode: falling
    back to the Python program no longer protects such a component, because
    the Python program executes the same bound functions as the compiled
    program.
    """
    problems: list[str] = []
    for method_name, hook_names in _LIKELIHOOD_DIVERGENCE_PAIRS:
        problem = _stale_native_override(type(like_obj), method_name, hook_names)
        if problem is not None:
            problems.append(f'likelihood {problem}')
    for method_name, hook_names in _EXCHANGE_DIVERGENCE_PAIRS:
        problem = _stale_native_override(type(proposal_manager.exchange_manager), method_name, hook_names)
        if problem is not None:
            problems.append(f'exchange manager {problem}')
    managers: tuple[tuple[str, object], ...] = (
        ('aggregate proposal manager', proposal_manager),
        *((f'proposal manager {idx}', manager) for idx, manager in enumerate(proposal_manager.managers)),
    )
    for label, manager in managers:
        problem = _manager_divergence(manager)
        if problem is not None:
            problems.append(f'{label} {problem}')
        problem = _stale_native_override(type(manager), 'dispatch_jump', ('jumps',))
        if problem is not None:
            problems.append(f'{label} {problem}')
    for idx, jump in enumerate(proposal_manager.jumps):
        problem = _stale_native_override(type(jump), '__call__', ('handle',))
        if problem is not None:
            problems.append(f'jump {idx} {problem}')
    return problems


def _capability_gaps[LikelihoodType: AbstractLikelihood[NamedTuple]](
    proposal_manager: AbstractProposalManager[LikelihoodType], like_obj: LikelihoodType
) -> tuple[list[str], int]:
    """Describe what blocks the compiled program, and count compiled handles.

    Eligibility is detected by capability, not API shape: every function the
    kernel closes over must already be nopython-compiled (eager compilation
    at component construction leaves a plain-Python handle behind exactly
    when compilation failed) and every runtime bundle must be numba-typable.
    The compiled-handle count distinguishes a partially compiled graph
    (warn in auto mode) from a graph with no native participation (silent).
    """
    gaps: list[str] = []
    compiled = 0

    def check_handle(description: str, handle: Callable[..., object]) -> None:
        nonlocal compiled
        if is_jitted(handle):
            compiled += 1
        else:
            gaps.append(f'{description} failed Numba compilation or is not nopython-compiled')

    def check_typable(description: str, value: object) -> None:
        try:
            typeof(value)  # type: ignore[no-untyped-call]
        except ValueError, NumbaError:
            gaps.append(f'{description} is not numba-typable')

    like_name = type(like_obj).__qualname__
    likelihood_natives = like_obj.bind_native
    check_handle(f'likelihood {like_name} get_loglike handle', likelihood_natives.loglike)
    check_handle(f'likelihood {like_name} prior_factor handle', likelihood_natives.prior_factor)
    check_handle(f'likelihood {like_name} validate_bounds handle', likelihood_natives.validate_bounds)

    exchange_manager = proposal_manager.exchange_manager
    exchange_name = type(exchange_manager).__qualname__
    exchange_natives = exchange_manager.bind_native
    check_handle(f'exchange manager {exchange_name} is_exchange_step handle', exchange_natives.is_exchange_step)
    check_handle(f'exchange manager {exchange_name} exchange handle', exchange_natives.exchange)
    check_typable(f'exchange manager {exchange_name} inputs', exchange_manager.inputs)

    for idx, manager in enumerate(proposal_manager.managers):
        manager_name = type(manager).__qualname__
        for jump in manager.jumps:
            check_handle(f'proposal manager {idx} {manager_name} jump {type(jump).__qualname__} handle', jump.handle)
        if _defining_class(type(manager), 'bind_native_post_step') is not None:
            check_handle(f'proposal manager {idx} {manager_name} post-step binding', manager.bind_native_post_step)
        if _defining_class(type(manager), 'native_state') is not None:
            check_typable(f'proposal manager {idx} {manager_name} native state', manager.native_state)

    return gaps, compiled


def _check_flattening[LikelihoodType: AbstractLikelihood[NamedTuple]](
    proposal_manager: AbstractProposalManager[LikelihoodType],
) -> None:
    """Require the aggregate jump list to be the ordered flattening of the managers'.

    Both program flavors dispatch per manager against the aggregate's
    flattened ``jump_probs`` columns, so this is a precondition of every
    kernel backend mode.
    """
    flattened = [jump for manager in proposal_manager.managers for jump in manager.jumps]
    if len(flattened) != len(proposal_manager.jumps) or any(
        actual is not expected for actual, expected in zip(proposal_manager.jumps, flattened, strict=True)
    ):
        msg = 'proposal_manager.jumps must be the ordered identity-preserving flattening of managers[*].jumps'
        raise NativeBackendUnsupportedError(msg)


class _LocalDispatchCall(Protocol):
    """Internal chain link: one manager's jumps behind a local index."""

    def __call__(
        self, idx_jump: int, sample_point: NDArray[np.floating], itrt: int, state: object, /
    ) -> tuple[NDArray[np.floating], float, bool]:
        """Dispatch the manager-local jump index."""
        ...


class _ManagerDispatchCall(Protocol):
    """Internal chain link: all managers' jumps behind a flattened index."""

    def __call__(
        self,
        idx_jump: int,
        sample_point: NDArray[np.floating],
        itrt: int,
        states: tuple[object, ...],
        /,
    ) -> tuple[NDArray[np.floating], float, bool]:
        """Dispatch the flattened jump index."""
        ...


class _WrapCall(Protocol):
    """Flavor wrapper applied to every assembled link of a program."""

    def __call__[F: Callable[..., object]](self, fn: F, /) -> F:
        """Return the link as the flavor executes it."""
        ...


def _wrap_python[F: Callable[..., object]](fn: F) -> F:
    """Identity link wrapper: the Python program runs the chain uncompiled."""
    return fn


def _wrap_njit_inline[F: Callable[..., object]](fn: F) -> F:
    """Inline nopython link wrapper for the compiled program's chain."""
    return cast('F', njit(inline='always')(fn))


def _wrap_njit[F: Callable[..., object]](fn: F) -> F:
    """Nopython wrapper for the compiled program's outermost kernel."""
    return cast('F', njit()(fn))


def _build_local_dispatch(wrap: _WrapCall, jump_calls: tuple[NativeJumpCall[Any], ...]) -> _LocalDispatchCall:
    """Dispatch a manager-local jump index over one manager's bound jumps.

    The recursion happens in Python at assembly time; under the compiled
    wrapper (``inline='always'``) Numba flattens the chain into the same
    branch sequence a hand-written ``if/elif`` ladder would produce, and
    the identity wrapper leaves the equivalent plain-Python chain. The
    caller guarantees the index is within range, so the final leaf executes
    unconditionally.
    """
    head = jump_calls[0]
    if len(jump_calls) == 1:

        def dispatch_leaf(
            _idx_jump: int,
            sample_point: NDArray[np.floating],
            itrt: int,
            state: object,
        ) -> tuple[NDArray[np.floating], float, bool]:
            return head(sample_point, itrt, state)

        return wrap(dispatch_leaf)
    rest = _build_local_dispatch(wrap, jump_calls[1:])

    def dispatch(
        idx_jump: int,
        sample_point: NDArray[np.floating],
        itrt: int,
        state: object,
    ) -> tuple[NDArray[np.floating], float, bool]:
        if idx_jump == 0:
            return head(sample_point, itrt, state)
        return rest(idx_jump - 1, sample_point, itrt, state)

    return wrap(dispatch)


def _build_manager_dispatch(wrap: _WrapCall, per_manager_calls: ManagerJumpCalls) -> _ManagerDispatchCall:
    """Dispatch a flattened jump index over the managers' bound jumps.

    Peels the heterogeneous ``manager_states`` runtime tuple with static
    indexing (``states[0]`` / ``states[1:]``) so each manager's jumps see
    exactly their own manager's runtime native state.
    """
    head = _build_local_dispatch(wrap, per_manager_calls[0])
    n_head = len(per_manager_calls[0])
    if len(per_manager_calls) == 1:

        def dispatch_last(
            idx_jump: int,
            sample_point: NDArray[np.floating],
            itrt: int,
            states: tuple[object, ...],
        ) -> tuple[NDArray[np.floating], float, bool]:
            return head(idx_jump, sample_point, itrt, states[0])

        return wrap(dispatch_last)
    rest = _build_manager_dispatch(wrap, per_manager_calls[1:])

    def dispatch(
        idx_jump: int,
        sample_point: NDArray[np.floating],
        itrt: int,
        states: tuple[object, ...],
    ) -> tuple[NDArray[np.floating], float, bool]:
        if idx_jump < n_head:
            return head(idx_jump, sample_point, itrt, states[0])
        return rest(idx_jump - n_head, sample_point, itrt, states[1:])

    return wrap(dispatch)


def _post_step_noop(_states: tuple[object, ...], _samples_row: NDArray[np.floating]) -> None:
    """Per-step update for managers with no native per-step work."""
    return


def _build_post_chain(wrap: _WrapCall, post_steps: ManagerPostSteps) -> NativePostStepCall[tuple[object, ...]]:
    """Chain the managers' bound per-step updates in manager order.

    ``post_steps`` has one entry per manager (None when idle) so the chain
    peels the ``manager_states`` tuple in step with the dispatch chain.
    """
    if not post_steps:
        return wrap(_post_step_noop)
    head = post_steps[0]
    rest = _build_post_chain(wrap, post_steps[1:]) if len(post_steps) > 1 else None
    if head is None:
        if rest is None:
            return wrap(_post_step_noop)
        rest_after_skip = rest

        def post_skip(states: tuple[object, ...], samples_row: NDArray[np.floating]) -> None:
            rest_after_skip(states[1:], samples_row)

        return wrap(post_skip)
    if rest is None:

        def post_leaf(states: tuple[object, ...], samples_row: NDArray[np.floating]) -> None:
            head(states[0], samples_row)

        return wrap(post_leaf)
    rest_fn = rest

    def post_all(states: tuple[object, ...], samples_row: NDArray[np.floating]) -> None:
        head(states[0], samples_row)
        rest_fn(states[1:], samples_row)

    return wrap(post_all)


class SerialBlockKernel[ExchangeInputType](Protocol):
    """Signature shared by both flavors of the assembled block program."""

    def __call__(
        self,
        samples: NDArray[np.floating],
        logLs: NDArray[np.floating],
        chain_track: NDArray[np.int64],
        betas: NDArray[np.floating],
        jump_probs: NDArray[np.floating],
        exchange_tracker: NDArray[np.int64],
        esd_exchange: NDArray[np.floating],
        accept_record: NDArray[np.int64],
        esd_record: NDArray[np.floating],
        manager_states: tuple[object, ...],
        exchange_inputs: ExchangeInputType,
        jump_internal_evals: NDArray[np.int64],
        zero_loglike: bool,
        /,
    ) -> tuple[int, int]:
        """Advance one block; return the target and declared-internal eval counts."""
        ...


def make_serial_kernel[ExchangeInputType](
    flavor: KernelFlavor,
    per_manager_calls: ManagerJumpCalls,
    post_steps: ManagerPostSteps,
    likelihood_natives: NativeLikelihoodFunctions[NamedTuple],
    exchange_natives: NativeExchangeFunctions[ExchangeInputType],
) -> SerialBlockKernel[ExchangeInputType]:
    """Assemble one flavor of the serial block program for one bound graph.

    Only per-class functions are closed over; every instance-specific value
    arrives through the runtime state bundles. The two flavors are the same
    assembly: ``'numba'`` nopython-compiles every link and the kernel,
    ``'python'`` applies the identity wrapper so the identical call chain
    runs uncompiled while still invoking the same (typically compiled)
    bound handles, preserving fixed-seed bit-exactness between flavors.
    The kernel returns the counts of target likelihood evaluations
    performed and of declared jump-internal evaluations incurred.
    """
    wrap_link: _WrapCall = _wrap_njit_inline if flavor == 'numba' else _wrap_python
    wrap_kernel: _WrapCall = _wrap_njit if flavor == 'numba' else _wrap_python
    dispatch = _build_manager_dispatch(wrap_link, per_manager_calls)
    post_all = _build_post_chain(wrap_link, post_steps)
    loglike = likelihood_natives.loglike
    prior_factor = likelihood_natives.prior_factor
    validate_bounds = likelihood_natives.validate_bounds
    is_exchange_step = exchange_natives.is_exchange_step
    do_exchange = exchange_natives.exchange

    def advance_block_serial(
        samples: NDArray[np.floating],
        logLs: NDArray[np.floating],
        chain_track: NDArray[np.int64],
        betas: NDArray[np.floating],
        jump_probs: NDArray[np.floating],
        exchange_tracker: NDArray[np.int64],
        esd_exchange: NDArray[np.floating],
        accept_record: NDArray[np.int64],
        esd_record: NDArray[np.floating],
        manager_states: tuple[object, ...],
        exchange_inputs: ExchangeInputType,
        jump_internal_evals: NDArray[np.int64],
        zero_loglike: bool,
    ) -> tuple[int, int]:
        block_size = samples.shape[0] - 1
        n_chain = samples.shape[1]
        n_target_evals = 0
        n_internal_evals = 0
        for itrb in range(1, block_size + 1):
            if is_exchange_step(itrb, exchange_inputs):
                do_exchange(
                    itrb, samples, logLs, n_chain, betas, exchange_tracker, esd_exchange, chain_track, exchange_inputs
                )
            else:
                for itrt in range(n_chain):
                    idx_jump = choose_prob_helper(jump_probs[itrt])
                    sample_point = samples[itrb - 1, itrt]
                    new_point, density_fac, success = dispatch(idx_jump, sample_point, itrt, manager_states)
                    n_internal_evals += jump_internal_evals[idx_jump]
                    if success:
                        new_point, success = validate_bounds(new_point)
                    if success:
                        density_fac += prior_factor(new_point) - prior_factor(sample_point)
                    if success:
                        if zero_loglike:
                            logL_new = 0.0
                        else:
                            logL_new = loglike(new_point)
                        n_target_evals += 1
                    else:
                        logL_new = -np.inf
                    mcmc_decision_helper(
                        itrb,
                        samples,
                        logLs,
                        betas,
                        accept_record,
                        esd_record,
                        itrt,
                        new_point,
                        logL_new,
                        density_fac,
                        idx_jump,
                    )
                chain_track[itrb, :] = chain_track[itrb - 1, :]
            post_all(manager_states, samples[itrb])
        return n_target_evals, n_internal_evals

    return wrap_kernel(advance_block_serial)


@dataclass(frozen=True)
class NativeSerialProgram:
    """One assembled program flavor plus strong references to its bound functions.

    Holding the bound functions prevents garbage collection (and therefore
    ``id`` reuse) from aliasing a stale program onto freshly created
    functions while the program is cached; a hook that violates the
    per-class stability contract only costs a duplicate cache entry.
    """

    flavor: KernelFlavor
    # the exchange-input parameter is erased at this boundary:
    # AbstractProposalManager exposes AbstractExchangeManager[Any]
    kernel: SerialBlockKernel[Any]
    likelihood_natives: NativeLikelihoodFunctions[NamedTuple]
    per_manager_calls: ManagerJumpCalls
    post_steps: ManagerPostSteps
    exchange_natives: NativeExchangeFunctions[Any]


_PROGRAM_CACHE: dict[tuple[object, ...], NativeSerialProgram] = {}


def _graph_identity[LikelihoodType: AbstractLikelihood[NamedTuple]](
    proposal_manager: AbstractProposalManager[LikelihoodType], like_obj: LikelihoodType
) -> tuple[object, ...]:
    """Cheap per-block identity of the component object graph.

    Used only to detect that the graph changed and a re-resolve is needed;
    program reuse is keyed structurally by ``_structural_key``.
    """
    manager_graph = tuple(
        (type(manager), id(manager), tuple((type(jump), id(jump)) for jump in manager.jumps))
        for manager in proposal_manager.managers
    )
    exchange_manager = proposal_manager.exchange_manager
    return (
        (type(like_obj), id(like_obj)),
        manager_graph,
        (type(exchange_manager), id(exchange_manager)),
    )


def _structural_key(
    likelihood_natives: NativeLikelihoodFunctions[NamedTuple],
    per_manager_calls: ManagerJumpCalls,
    post_steps: ManagerPostSteps,
    exchange_natives: NativeExchangeFunctions[Any],
) -> tuple[object, ...]:
    """Program cache key: the identities of the bound per-class functions.

    Two samplers whose components bind the same function objects share one
    program; the cached program holds strong references to those functions,
    so the ids in the key cannot be recycled while the entry lives.
    """
    return (
        (
            id(likelihood_natives.loglike),
            id(likelihood_natives.prior_draw),
            id(likelihood_natives.prior_factor),
            id(likelihood_natives.prior_proposal),
            id(likelihood_natives.validate_bounds),
            id(likelihood_natives.check_bounds),
            id(likelihood_natives.correct_bounds),
        ),
        tuple(
            (None if post is None else id(post), tuple(id(call) for call in calls))
            for post, calls in zip(post_steps, per_manager_calls, strict=True)
        ),
        (id(exchange_natives.is_exchange_step), id(exchange_natives.exchange)),
    )


def _compile_for_args(kernel: SerialBlockKernel[Any], args: tuple[object, ...]) -> None:
    """Eagerly compile the compiled-flavor kernel for the exact runtime signature.

    Compilation is forced without executing the kernel, so a NumbaError
    here is guaranteed to leave the block arrays, tracker state, and RNG
    streams untouched — the only condition under which auto mode may fall
    back and run the same block through the Python program. A NumbaError
    raised while the kernel executes (user-compiled code may raise any
    exception class) must never be treated as a compilation failure.
    """
    dispatcher: Any = kernel  # numba Dispatcher: typeof_pyval/compile are untyped
    arg_types = tuple(dispatcher.typeof_pyval(arg) for arg in args)
    dispatcher.compile(arg_types)


class NativeSerialBackend[LikelihoodType: AbstractLikelihood[NamedTuple]]:
    """Bind, cache, and run the serial block programs for one sampler."""

    def __init__(self, mode: str) -> None:
        if mode not in _VALID_MODES:
            msg = f'kernel_backend must be one of {_VALID_MODES}; got {mode!r}'
            raise ValueError(msg)
        # membership in the Literal-typed tuple narrows mode for the checkers
        self.mode: KernelBackendMode = mode
        self.graph_identity: tuple[object, ...] | None = None
        self.program: NativeSerialProgram | None = None
        self.warned_identities: set[tuple[object, ...]] = set()
        self._manager_has_state: tuple[bool, ...] = ()
        self._jump_internal_evals: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        self._jump_internal_known: bool = True
        self._kernel_ready: bool = False

    def _bind_program(
        self, proposal_manager: AbstractProposalManager[LikelihoodType], like_obj: LikelihoodType, flavor: KernelFlavor
    ) -> NativeSerialProgram:
        """Bind every component and fetch or assemble one program flavor."""
        likelihood_natives = like_obj.bind_native
        per_manager_calls: list[tuple[NativeJumpCall[Any], ...]] = []
        post_steps: list[NativePostStepCall[Any] | None] = []
        manager_has_state: list[bool] = []
        for manager in proposal_manager.managers:
            per_manager_calls.append(tuple(jump.handle for jump in manager.jumps))
            manager_has_state.append(_defining_class(type(manager), 'native_state') is not None)
            has_post = _defining_class(type(manager), 'bind_native_post_step') is not None
            post_steps.append(manager.bind_native_post_step if has_post else None)
        assert sum(len(calls) for calls in per_manager_calls) == proposal_manager.jump_probs.shape[1]
        exchange_natives = proposal_manager.exchange_manager.bind_native

        declared: list[int | None] = [getattr(jump, 'declared_internal_evals', None) for jump in proposal_manager.jumps]
        self._jump_internal_known = all(value is not None for value in declared)
        self._jump_internal_evals = np.array([0 if value is None else value for value in declared], dtype=np.int64)
        self._manager_has_state = tuple(manager_has_state)

        if flavor == 'python':
            # nothing is compiled, so there is nothing to share through the cache
            kernel = make_serial_kernel(
                'python', tuple(per_manager_calls), tuple(post_steps), likelihood_natives, exchange_natives
            )
            return NativeSerialProgram(
                flavor='python',
                kernel=kernel,
                likelihood_natives=likelihood_natives,
                per_manager_calls=tuple(per_manager_calls),
                post_steps=tuple(post_steps),
                exchange_natives=exchange_natives,
            )
        key = _structural_key(likelihood_natives, tuple(per_manager_calls), tuple(post_steps), exchange_natives)
        cached = _PROGRAM_CACHE.get(key)
        if cached is not None:
            return cached
        kernel = make_serial_kernel(
            'numba', tuple(per_manager_calls), tuple(post_steps), likelihood_natives, exchange_natives
        )
        program = NativeSerialProgram(
            flavor='numba',
            kernel=kernel,
            likelihood_natives=likelihood_natives,
            per_manager_calls=tuple(per_manager_calls),
            post_steps=tuple(post_steps),
            exchange_natives=exchange_natives,
        )
        _PROGRAM_CACHE[key] = program
        return program

    def _resolve(self, proposal_manager: AbstractProposalManager[LikelihoodType], like_obj: LikelihoodType) -> None:
        """Select and bind the program for the current graph, once per identity.

        ``'python'`` mode always binds the Python program. ``'numba'`` mode
        requires full capability and raises otherwise (not memoized: strict
        mode re-raises on every attempted block). ``'auto'`` binds the
        compiled program when every handle is compiled, warning once per
        graph before using the Python program otherwise.
        """
        identity = _graph_identity(proposal_manager, like_obj)
        if identity == self.graph_identity:
            return
        _check_flattening(proposal_manager)
        self._kernel_ready = False
        if self.mode == 'python':
            self.program = self._bind_program(proposal_manager, like_obj, 'python')
            self.graph_identity = identity
            return
        gaps, compiled = _capability_gaps(proposal_manager, like_obj)
        program: NativeSerialProgram | None = None
        if not gaps:
            try:
                program = self._bind_program(proposal_manager, like_obj, 'numba')
            except NativeBackendUnsupportedError as exc:
                # a hook declined at bind time
                gaps = [str(exc)]
        if program is None:
            detail = '; '.join(gaps)
            if self.mode == 'numba':
                msg = (
                    f"kernel_backend='numba' requires a fully compiled graph; "
                    f'these bindings failed Numba compilation or are unavailable: {detail}'
                )
                raise NativeBackendCompilationError(msg)
            if compiled and identity not in self.warned_identities:
                warn(
                    f"kernel_backend='auto' found a partially compiled graph and "
                    f'will run the Python program; missing: {detail}',
                    RuntimeWarning,
                    stacklevel=3,
                )
                self.warned_identities.add(identity)
            program = self._bind_program(proposal_manager, like_obj, 'python')
        self.graph_identity = identity
        self.program = program

    def advance_block(
        self,
        T_ladder: TemperatureLadder,
        logLs: NDArray[np.floating],
        samples: NDArray[np.floating],
        chain_track: NDArray[np.int64],
        proposal_manager: AbstractProposalManager[LikelihoodType],
        like_obj: LikelihoodType,
        tracker_manager: TrackerManager[LikelihoodType],
        eval_accounting: EvalAccounting,
        zero_loglike: bool,
    ) -> KernelFlavor:
        """Advance one block through the resolved program; return the flavor that ran."""
        self._resolve(proposal_manager, like_obj)
        program = self.program
        assert program is not None
        # runtime state bundles are re-read at every block entry, so
        # configuration changes between blocks behave identically in both flavors
        manager_states = tuple(
            manager.native_state if has_state else None
            for manager, has_state in zip(proposal_manager.managers, self._manager_has_state, strict=True)
        )
        exchange_inputs = proposal_manager.exchange_manager.inputs
        args = (
            samples,
            logLs,
            chain_track,
            T_ladder.betas,
            proposal_manager.jump_probs,
            tracker_manager.exchange_tracker,
            tracker_manager.esd_exchange,
            tracker_manager.accept_record,
            tracker_manager.esd_record,
            manager_states,
            exchange_inputs,
            self._jump_internal_evals,
            zero_loglike,
        )
        if program.flavor == 'numba' and not self._kernel_ready:
            # separate compilation from execution: only a failure here, with
            # the block arrays and RNG streams still untouched, may fall back.
            # A later block with a changed state signature lazily recompiles
            # inside the call and a failure there propagates — never a replay.
            try:
                _compile_for_args(program.kernel, args)
            except NumbaError as exc:
                if self.mode == 'numba':
                    cls = type(like_obj)
                    msg = f'fully compiled-bindable graph failed Numba compilation for {cls.__qualname__} '
                    raise NativeBackendCompilationError(msg) from exc
                warn(
                    f"kernel_backend='auto' compiled kernel failed to compile and will run the Python program: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                program = self._bind_program(proposal_manager, like_obj, 'python')
                self.program = program
            else:
                self._kernel_ready = True
        # runtime exceptions (numba or otherwise) propagate: execution may
        # already have consumed RNG draws and mutated trackers, so rerunning
        # the block would silently double-count its side effects
        n_target_evals, n_internal_evals = program.kernel(*args)
        eval_accounting.proposal_targets += n_target_evals
        eval_accounting.proposal_internal += n_internal_evals
        if not self._jump_internal_known:
            eval_accounting.complete = False
        return program.flavor
