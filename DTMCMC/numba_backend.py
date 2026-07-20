"""Native (nopython) serial backend assembled from per-instance behavior handles.

The likelihood's constants are baked into its compiled handles at
construction (see DTMCMC.likelihood); the kernel closes over those handles
directly. Everything genuinely mutable — the component managers' buffers
and counters — travels in a small per-manager runtime state bundle
returned by ``native_state()`` and re-read at every block entry, so
between-block updates behave exactly like the Python path and nothing
mutable is ever baked into compiled code (Numba treats closure-captured
arrays as read-only non-aliasing constants, so baking mutable or
rebindable state risks silently stale reads).

Programs are cached by the identity of the bound functions: samplers whose
components resolve to the same handle objects (equal baked constants reuse
memoized handles, see ``DTMCMC.likelihood.memoized_handle``) share one
compiled kernel, and Numba's own dispatcher handles any residual signature
specialization. The assembly uses plain recursive closure factories (the
Numba equivalent of ``functools.partial``): no strings, no ``exec``, and
every line is visible to linters and type checkers.

A bound native jump has the ``AbstractJump.__call__`` signature plus the
owning manager's runtime state; likelihood handles have the
``AbstractLikelihood`` method signatures.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast
from warnings import warn

import numpy as np
from numba import njit
from numba.core.errors import NumbaError
from numba.extending import is_jitted
from numpy.typing import NDArray

from DTMCMC.jump_manager import AbstractJump, AbstractJumpManager, JumpManager, choose_prob_helper
from DTMCMC.likelihood import (
    LIKELIHOOD_HANDLE_ROLES,
    AbstractLikelihood,
    LoglikeFn,
    PriorFactorFn,
    ValidateBoundsFn,
    loglike_handle,
    prior_factor_handle,
    role_handle_by_names,
    validate_bounds_handle,
)
from DTMCMC.mcmc_kernel_helpers import mcmc_decision_helper

if TYPE_CHECKING:
    from collections.abc import Callable

    from DTMCMC.eval_accounting import EvalAccounting
    from DTMCMC.exchange_manager import AbstractExchangeManager
    from DTMCMC.proposal_manager import AbstractProposalManager
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder
    from DTMCMC.tracker_manager import TrackerManager

_VALID_MODES = ('auto', 'numba', 'python')

ManagerStateT_contra = TypeVar('ManagerStateT_contra', contravariant=True)


class NativeBackendUnsupportedError(RuntimeError):
    """The requested object graph has no complete native binding."""


class NativeBackendCompilationError(RuntimeError):
    """A fully bindable native graph failed to compile or execute in Numba."""


class NativeJumpCall(Protocol[ManagerStateT_contra]):
    """Jitted jump: ``AbstractJump.__call__`` plus the manager runtime state."""

    def __call__(
        self,
        sample_point: NDArray[np.floating],
        itrt: int,
        manager_state: ManagerStateT_contra,
        /,
    ) -> tuple[NDArray[np.floating], float, bool]:
        """Return the proposed point, its density factor, and success."""
        ...


class NativePostStepCall(Protocol[ManagerStateT_contra]):
    """Jitted per-step manager update over the manager's runtime state."""

    def __call__(self, state: ManagerStateT_contra, samples_row: NDArray[np.floating], /) -> None:
        """Update the manager state in place after one step of all chains."""
        ...


class NativeExchangeStepCall(Protocol):
    """Jitted exchange cadence with the schedule constants baked in."""

    def __call__(self, itrb: int, /) -> bool:
        """Return whether step ``itrb`` is an exchange step."""
        ...


class NativeExchangeCall(Protocol):
    """Jitted exchange executor with the strategy constants baked in."""

    def __call__(
        self,
        itrb: int,
        samples: NDArray[np.floating],
        logLs: NDArray[np.floating],
        n_chain: int,
        betas: NDArray[np.floating],
        exchange_tracker: NDArray[np.int64],
        esd_exchange: NDArray[np.floating],
        chain_track: NDArray[np.int64],
        /,
    ) -> None:
        """Execute one exchange step."""
        ...


@dataclass(frozen=True)
class NativeExchangeFunctions:
    """Jitted exchange schedule and executor with baked constants."""

    is_exchange_step: NativeExchangeStepCall
    exchange: NativeExchangeCall


def _defining_class(cls: type, name: str) -> type | None:
    """Return the most-derived class in the MRO that defines ``name``."""
    for klass in cls.__mro__:
        if name in vars(klass):
            return klass
    return None


def _stale_native_override(cls: type, method_name: str, hook_names: tuple[str, ...]) -> str | None:
    """Detect a Python-behavior override that outruns its native binding.

    A native hook only describes the Python methods defined at or above the
    class that defines the hook. If a subclass overrides a paired Python
    method without also overriding a hook, the inherited binding would run
    the ancestor's behavior — silently wrong results — so the component is
    reported unsupported instead.
    """
    py_cls = _defining_class(cls, method_name)
    if py_cls is None:
        return None
    native_cls: type | None = None
    for hook_name in hook_names:
        hook_cls = _defining_class(cls, hook_name)
        if hook_cls is not None and (native_cls is None or issubclass(hook_cls, native_cls)):
            native_cls = hook_cls
    if native_cls is None:
        return f'{cls.__qualname__} defines {method_name} without defining {" or ".join(hook_names)}'
    if py_cls is not native_cls and issubclass(py_cls, native_cls):
        return f'{cls.__qualname__} overrides {method_name} without overriding {" or ".join(hook_names)}'
    return None


def _likelihood_problem(like_obj: AbstractLikelihood) -> str | None:
    python_roles = [
        method_name
        for fn_name, method_name in LIKELIHOOD_HANDLE_ROLES
        if not is_jitted(role_handle_by_names(like_obj, fn_name, method_name))
    ]
    if python_roles:
        return f'{type(like_obj).__qualname__} has no nopython-compiled implementation of {", ".join(python_roles)}'
    return None


def _jump_problem(jump: object) -> str | None:
    cls = type(jump)
    if _defining_class(cls, 'bind_native') is None:
        return f'{cls.__qualname__} does not define bind_native'
    return _stale_native_override(cls, '__call__', ('bind_native',))


def _exchange_problem(exchange_manager: object) -> str | None:
    cls = type(exchange_manager)
    if _defining_class(cls, 'bind_native') is None:
        return f'{cls.__qualname__} does not define bind_native'
    for method_name in ('is_exchange_step', 'do_ptmcmc_exchange'):
        problem = _stale_native_override(cls, method_name, ('bind_native',))
        if problem is not None:
            return problem
    return None


def _manager_binding(manager: object) -> tuple[str | None, bool]:
    """Classify a component manager's native binding.

    Returns ``(problem, explicit)``. A manager that inherits the base
    ``JumpManager.post_step_update`` no-op is bindable with no native work
    at all, but that passive eligibility does not count as explicit native
    intent for the mixed-graph warning.
    """
    cls = type(manager)
    hook_cls = _defining_class(cls, 'bind_native_post_step')
    py_cls = _defining_class(cls, 'post_step_update')
    if hook_cls is not None:
        if py_cls is not None and py_cls is not hook_cls and issubclass(py_cls, hook_cls):
            return (
                f'{cls.__qualname__} overrides post_step_update without overriding bind_native_post_step',
                False,
            )
        return None, True
    if py_cls is not None and py_cls is not JumpManager:
        return f'{cls.__qualname__} overrides post_step_update without defining bind_native_post_step', False
    return None, _defining_class(cls, 'native_state') is not None


def _binding_inventory[LikelihoodType: AbstractLikelihood](
    proposal_manager: AbstractProposalManager[LikelihoodType], like_obj: LikelihoodType
) -> tuple[list[str], int]:
    """Return unsupported-component descriptions and the explicit-binding count."""
    problems: list[str] = []
    explicit = 0

    problem = _likelihood_problem(like_obj)
    if problem is None:
        explicit += 1
    else:
        problems.append(f'likelihood {problem}')

    problem = _exchange_problem(proposal_manager.exchange_manager)
    if problem is None:
        explicit += 1
    else:
        problems.append(f'exchange manager {problem}')

    for idx, manager in enumerate(proposal_manager.managers):
        problem, is_explicit = _manager_binding(manager)
        if problem is not None:
            problems.append(f'proposal manager {idx} {problem}')
        elif is_explicit:
            explicit += 1
        for jump_idx, jump in enumerate(manager.jumps):
            problem = _jump_problem(jump)
            if problem is None:
                explicit += 1
            else:
                problems.append(f'proposal manager {idx} jump {jump_idx} {problem}')

    return problems, explicit


def _declared_jump_internal_evals[LikelihoodType: AbstractLikelihood](
    proposal_manager: AbstractProposalManager[LikelihoodType],
) -> tuple[NDArray[np.int64], bool]:
    """Collect the per-jump declared internal evaluation costs in flattened order.

    Returns the cost array and whether every jump declared one; a missing
    declaration contributes 0 to the array but marks the accounting
    incomplete — an unknown cost is never silently treated as zero.
    """
    declared = [getattr(jump, 'declared_internal_evals', None) for jump in proposal_manager.jumps]
    known = all(value is not None for value in declared)
    return np.array([0 if value is None else value for value in declared], dtype=np.int64), known


def _python_jump_call(jump: AbstractJump[Any]) -> NativeJumpCall[Any]:
    """Adapt a jump's Python ``__call__`` to the bound-jump calling convention."""

    def jump_call(
        sample_point: NDArray[np.floating], itrt: int, _state: object
    ) -> tuple[NDArray[np.floating], float, bool]:
        return jump(sample_point, itrt)

    return jump_call


def _python_post_step(manager: AbstractJumpManager[Any]) -> NativePostStepCall[Any]:
    """Adapt a manager's ``post_step_update`` to the bound-post-step convention."""

    def post_step(_state: object, samples_row: NDArray[np.floating]) -> None:
        manager.post_step_update(samples_row)

    return post_step


def _python_exchange(exchange_manager: AbstractExchangeManager, T_ladder: TemperatureLadder) -> NativeExchangeFunctions:
    """Adapt an exchange manager's Python methods to the bound convention."""

    def exchange(
        itrb: int,
        samples: NDArray[np.floating],
        logLs: NDArray[np.floating],
        _n_chain: int,
        _betas: NDArray[np.floating],
        exchange_tracker: NDArray[np.int64],
        esd_exchange: NDArray[np.floating],
        chain_track: NDArray[np.int64],
    ) -> None:
        exchange_manager.do_ptmcmc_exchange(itrb, samples, logLs, T_ladder, exchange_tracker, esd_exchange, chain_track)

    return NativeExchangeFunctions(is_exchange_step=exchange_manager.is_exchange_step, exchange=exchange)


def _check_flattening[LikelihoodType: AbstractLikelihood](
    proposal_manager: AbstractProposalManager[LikelihoodType],
) -> None:
    """Require the aggregate jump list to be the ordered flattening of the managers'."""
    flattened = [jump for manager in proposal_manager.managers for jump in manager.jumps]
    if len(flattened) != len(proposal_manager.jumps) or any(
        actual is not expected for actual, expected in zip(proposal_manager.jumps, flattened, strict=True)
    ):
        msg = 'proposal_manager.jumps must be the ordered identity-preserving flattening of managers[*].jumps'
        raise NativeBackendUnsupportedError(msg)


class _LocalDispatchCall(Protocol):
    """Internal chain link: one manager's jumps behind a local index."""

    def __call__(
        self, idx_jump: int, sample_point: NDArray[np.floating], itrt: int, state: Any, /
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
        states: tuple[Any, ...],
        /,
    ) -> tuple[NDArray[np.floating], float, bool]:
        """Dispatch the flattened jump index."""
        ...


class _WrapCall(Protocol):
    """Decorator applied to every assembled chain link (njit or identity)."""

    def __call__[F: Callable[..., object]](self, fn: F, /) -> F:
        """Return the (possibly compiled) link."""
        ...


def _no_wrap[F: Callable[..., object]](fn: F) -> F:
    """Identity link decorator for the interpreted assembly."""
    return fn


def _build_local_dispatch(jump_calls: tuple[NativeJumpCall[Any], ...], wrap: _WrapCall) -> _LocalDispatchCall:
    """Dispatch a manager-local jump index over one manager's bound jumps.

    The recursion happens in Python at assembly time; with
    ``inline='always'`` Numba flattens the chain into the same branch
    sequence a hand-written ``if/elif`` ladder would produce. The caller
    guarantees the index is within range, so the final leaf executes
    unconditionally.
    """
    head = jump_calls[0]
    if len(jump_calls) == 1:

        @wrap
        def dispatch_leaf(
            _idx_jump: int, sample_point: NDArray[np.floating], itrt: int, state: object
        ) -> tuple[NDArray[np.floating], float, bool]:
            return head(sample_point, itrt, state)

        return dispatch_leaf
    rest = _build_local_dispatch(jump_calls[1:], wrap)

    @wrap
    def dispatch(
        idx_jump: int, sample_point: NDArray[np.floating], itrt: int, state: object
    ) -> tuple[NDArray[np.floating], float, bool]:
        if idx_jump == 0:
            return head(sample_point, itrt, state)
        return rest(idx_jump - 1, sample_point, itrt, state)

    return dispatch


def _build_manager_dispatch(
    per_manager_calls: tuple[tuple[NativeJumpCall[Any], ...], ...],
    wrap: _WrapCall,
) -> _ManagerDispatchCall:
    """Dispatch a flattened jump index over the managers' bound jumps.

    Peels the heterogeneous ``manager_states`` runtime tuple with static
    indexing (``states[0]`` / ``states[1:]``) so each manager's jumps see
    exactly their own manager's runtime native state.
    """
    head = _build_local_dispatch(per_manager_calls[0], wrap)
    n_head = len(per_manager_calls[0])
    if len(per_manager_calls) == 1:

        @wrap
        def dispatch_last(
            idx_jump: int,
            sample_point: NDArray[np.floating],
            itrt: int,
            states: tuple[object, ...],
        ) -> tuple[NDArray[np.floating], float, bool]:
            return head(idx_jump, sample_point, itrt, states[0])

        return dispatch_last
    rest = _build_manager_dispatch(per_manager_calls[1:], wrap)

    @wrap
    def dispatch(
        idx_jump: int,
        sample_point: NDArray[np.floating],
        itrt: int,
        states: tuple[object, ...],
    ) -> tuple[NDArray[np.floating], float, bool]:
        if idx_jump < n_head:
            return head(idx_jump, sample_point, itrt, states[0])
        return rest(idx_jump - n_head, sample_point, itrt, states[1:])

    return dispatch


@njit(inline='always')
def _post_step_noop(_states: tuple[object, ...], _samples: NDArray[np.floating]) -> None:
    """Per-step update for managers with no native per-step work."""
    return


def _build_post_chain(
    post_steps: tuple[NativePostStepCall[Any] | None, ...], wrap: _WrapCall
) -> NativePostStepCall[Any]:
    """Chain the managers' bound per-step updates in manager order.

    ``post_steps`` has one entry per manager (None when idle) so the chain
    peels the ``manager_states`` tuple in step with the dispatch chain.
    """
    if not post_steps:
        return _post_step_noop
    head = post_steps[0]
    rest = _build_post_chain(post_steps[1:], wrap) if len(post_steps) > 1 else None
    if head is None:
        if rest is None:
            return _post_step_noop
        rest_after_skip = rest

        @wrap
        def post_skip(states: tuple[object, ...], samples_row: NDArray[np.floating]) -> None:
            rest_after_skip(states[1:], samples_row)

        return post_skip
    if rest is None:

        @wrap
        def post_leaf(states: tuple[object, ...], samples_row: NDArray[np.floating]) -> None:
            head(states[0], samples_row)

        return post_leaf
    rest_fn = rest

    @wrap
    def post_all(states: tuple[object, ...], samples_row: NDArray[np.floating]) -> None:
        head(states[0], samples_row)
        rest_fn(states[1:], samples_row)

    return post_all


def make_serial_kernel(
    per_manager_calls: tuple[tuple[NativeJumpCall[Any], ...], ...],
    post_steps: tuple[NativePostStepCall[Any] | None, ...],
    loglike: LoglikeFn,
    prior_factor: PriorFactorFn,
    validate_bounds: ValidateBoundsFn,
    exchange_natives: NativeExchangeFunctions,
    *,
    jit: bool,
) -> Callable[..., tuple[int, int]]:
    """Assemble the serial block kernel for one bound graph structure.

    The single control flow serves both backends: with ``jit`` the chain
    links and the block loop compile to nopython; without it the same
    closures run interpreted over the same bound functions (so the two
    paths call the same jitted primitives in the same order and stay
    bit-exact). The closed-over handles carry only construction-frozen
    constants; every mutable value arrives through the runtime state
    bundles. Returns the counts of target likelihood evaluations
    performed and of declared jump-internal evaluations incurred in the
    block.
    """
    wrap: _WrapCall = cast('_WrapCall', njit(inline='always')) if jit else _no_wrap
    dispatch = _build_manager_dispatch(per_manager_calls, wrap)
    post_all = _build_post_chain(post_steps, wrap)
    is_exchange_step = exchange_natives.is_exchange_step
    do_exchange = exchange_natives.exchange
    kernel_wrap: _WrapCall = cast('_WrapCall', njit()) if jit else _no_wrap

    @kernel_wrap
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
        jump_internal_evals: NDArray[np.int64],
        zero_loglike: bool,
    ) -> tuple[int, int]:
        block_size = samples.shape[0] - 1
        n_chain = samples.shape[1]
        n_target_evals = 0
        n_internal_evals = 0
        for itrb in range(1, block_size + 1):
            if is_exchange_step(itrb):
                do_exchange(itrb, samples, logLs, n_chain, betas, exchange_tracker, esd_exchange, chain_track)
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

    return advance_block_serial


@dataclass(frozen=True)
class NativeSerialProgram:
    """Shared kernel plus strong references to the bound functions.

    Holding the bound functions prevents garbage collection (and therefore
    ``id`` reuse) from aliasing a stale program onto freshly created
    functions while the program is cached; a handle that violates the
    stability contract only costs a duplicate cache entry.
    """

    kernel: Callable[..., tuple[int, int]]
    likelihood_handles: tuple[LoglikeFn, PriorFactorFn, ValidateBoundsFn]
    per_manager_calls: tuple[tuple[NativeJumpCall[Any], ...], ...]
    post_steps: tuple[NativePostStepCall[Any] | None, ...]
    exchange_natives: NativeExchangeFunctions


_PROGRAM_CACHE: dict[tuple[object, ...], NativeSerialProgram] = {}


def _graph_identity[LikelihoodType: AbstractLikelihood](
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
    likelihood_handles: tuple[LoglikeFn, PriorFactorFn, ValidateBoundsFn],
    per_manager_calls: tuple[tuple[NativeJumpCall[Any], ...], ...],
    post_steps: tuple[NativePostStepCall[Any] | None, ...],
    exchange_natives: NativeExchangeFunctions,
) -> tuple[object, ...]:
    """Program cache key: the identities of the bound functions.

    Two samplers whose components resolve to the same handle objects share
    one program (equal baked constants reuse memoized handles); the cached
    program holds strong references to those functions, so the ids in the
    key cannot be recycled while the entry lives.
    """
    return (
        tuple(id(handle) for handle in likelihood_handles),
        tuple(
            (None if post is None else id(post), tuple(id(call) for call in calls))
            for post, calls in zip(post_steps, per_manager_calls, strict=True)
        ),
        (id(exchange_natives.is_exchange_step), id(exchange_natives.exchange)),
    )


class NativeSerialBackend[LikelihoodType: AbstractLikelihood]:
    """Bind, cache, and execute the native serial block kernel for one sampler."""

    def __init__(self, mode: str) -> None:
        if mode not in _VALID_MODES:
            msg = f'kernel_backend must be one of {_VALID_MODES}; got {mode!r}'
            raise ValueError(msg)
        self.mode = mode
        self.graph_identity: tuple[object, ...] | None = None
        self.program: NativeSerialProgram | None = None
        self.warned_identities: set[tuple[object, ...]] = set()
        self._manager_has_state: tuple[bool, ...] = ()
        self._jump_internal_evals: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        self._jump_internal_known: bool = True

    def _bind_program(
        self, proposal_manager: AbstractProposalManager[LikelihoodType], like_obj: LikelihoodType
    ) -> NativeSerialProgram:
        """Bind every component and fetch or assemble the structural program."""
        _check_flattening(proposal_manager)
        likelihood_handles = (
            loglike_handle(like_obj),
            prior_factor_handle(like_obj),
            validate_bounds_handle(like_obj),
        )
        per_manager_calls: list[tuple[NativeJumpCall[Any], ...]] = []
        post_steps: list[NativePostStepCall[Any] | None] = []
        manager_has_state: list[bool] = []
        for manager in proposal_manager.managers:
            per_manager_calls.append(tuple(jump.bind_native() for jump in manager.jumps))
            manager_has_state.append(_defining_class(type(manager), 'native_state') is not None)
            has_post = _defining_class(type(manager), 'bind_native_post_step') is not None
            post_steps.append(manager.bind_native_post_step() if has_post else None)
        assert sum(len(calls) for calls in per_manager_calls) == proposal_manager.jump_probs.shape[1]
        exchange_natives = proposal_manager.exchange_manager.bind_native()

        declared: list[int | None] = [getattr(jump, 'declared_internal_evals', None) for jump in proposal_manager.jumps]
        self._jump_internal_known = all(value is not None for value in declared)
        self._jump_internal_evals = np.array([0 if value is None else value for value in declared], dtype=np.int64)
        self._manager_has_state = tuple(manager_has_state)

        key = _structural_key(likelihood_handles, tuple(per_manager_calls), tuple(post_steps), exchange_natives)
        cached = _PROGRAM_CACHE.get(key)
        if cached is not None:
            return cached
        kernel = make_serial_kernel(
            tuple(per_manager_calls), tuple(post_steps), *likelihood_handles, exchange_natives, jit=True
        )
        program = NativeSerialProgram(
            kernel=kernel,
            likelihood_handles=likelihood_handles,
            per_manager_calls=tuple(per_manager_calls),
            post_steps=tuple(post_steps),
            exchange_natives=exchange_natives,
        )
        _PROGRAM_CACHE[key] = program
        return program

    def _resolve(self, proposal_manager: AbstractProposalManager[LikelihoodType], like_obj: LikelihoodType) -> None:
        identity = _graph_identity(proposal_manager, like_obj)
        if identity == self.graph_identity:
            return

        problems, explicit = _binding_inventory(proposal_manager, like_obj)
        if not problems:
            try:
                program = self._bind_program(proposal_manager, like_obj)
            except NativeBackendUnsupportedError as exc:
                # binding declined (e.g. the flattening contract is violated)
                problems = [str(exc)]
            else:
                self.graph_identity = identity
                self.program = program
                return

        detail = '; '.join(problems)
        if self.mode == 'numba':
            # not memoized: strict mode re-raises on every attempted block
            msg = f"kernel_backend='numba' requires a fully native-bindable graph; missing: {detail}"
            raise NativeBackendUnsupportedError(msg)
        self.graph_identity = identity
        self.program = None
        if explicit and identity not in self.warned_identities:
            warn(
                f"kernel_backend='auto' found a mixed native/Python graph and will use Python; missing: {detail}",
                RuntimeWarning,
                stacklevel=3,
            )
            self.warned_identities.add(identity)

    def _python_block_kernel(
        self,
        proposal_manager: AbstractProposalManager[LikelihoodType],
        like_obj: LikelihoodType,
        T_ladder: TemperatureLadder,
    ) -> Callable[..., tuple[int, int]]:
        """Assemble the interpreted block kernel over the Python-facing bindings.

        Rebuilt at every block: the closures are cheap and re-capture the
        current ladder and graph objects, so between-block reconfiguration
        needs no invalidation. The interpreted assembly binds the
        Python-facing surface — jump ``__call__`` in one aggregate group,
        manager ``post_step_update``, the exchange methods, and the
        likelihood methods — so wrappers, spies, and Python-only overrides
        all behave exactly as when called directly; the methods delegate to
        the same compiled primitives the native kernel binds.
        """
        jump_calls = tuple(_python_jump_call(jump) for jump in proposal_manager.jumps)
        assert len(jump_calls) == proposal_manager.jump_probs.shape[1]
        post_steps = tuple(_python_post_step(manager) for manager in proposal_manager.managers)
        exchange_natives = _python_exchange(proposal_manager.exchange_manager, T_ladder)
        return make_serial_kernel(
            (jump_calls,),
            post_steps,
            like_obj.get_loglike,
            like_obj.prior_factor,
            like_obj.validate_bounds,
            exchange_natives,
            jit=False,
        )

    @staticmethod
    def _account(
        eval_accounting: EvalAccounting[LikelihoodType],
        n_target_evals: int,
        n_internal_evals: int,
        internal_known: bool,
    ) -> None:
        """Fold one block's evaluation counts into the sampler accounting."""
        eval_accounting.proposal_targets += n_target_evals
        eval_accounting.proposal_internal += n_internal_evals
        if not internal_known:
            eval_accounting.complete = False

    def advance_block(
        self,
        T_ladder: TemperatureLadder,
        logLs: NDArray[np.floating],
        samples: NDArray[np.floating],
        chain_track: NDArray[np.int64],
        proposal_manager: AbstractProposalManager[LikelihoodType],
        like_obj: LikelihoodType,
        tracker_manager: TrackerManager[LikelihoodType],
        eval_accounting: EvalAccounting[LikelihoodType],
        zero_loglike: bool,
    ) -> str:
        """Advance one block, compiled when the graph binds, interpreted otherwise.

        Returns the name of the path used ('numba' or 'python').
        """
        if self.mode != 'python':
            self._resolve(proposal_manager, like_obj)
        if self.program is not None and self.mode != 'python':
            program = self.program
            # runtime state bundles are re-read at every block entry, so
            # configuration changes between blocks reach the kernel
            manager_states = tuple(
                manager.native_state() if has_state else None
                for manager, has_state in zip(proposal_manager.managers, self._manager_has_state, strict=True)
            )
            try:
                n_target_evals, n_internal_evals = program.kernel(
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
                    self._jump_internal_evals,
                    zero_loglike,
                )
            except NumbaError as exc:
                # NumbaError here means the kernel failed to compile, which
                # happens before any execution, so the block arrays are
                # untouched and the interpreted path below can run the block
                # from samples[0]
                if self.mode == 'numba':
                    msg = 'fully native-bindable graph failed Numba compilation or execution'
                    raise NativeBackendCompilationError(msg) from exc
                self.program = None
                warn(
                    f"kernel_backend='auto' native kernel failed to compile and will use Python: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                self._account(eval_accounting, n_target_evals, n_internal_evals, self._jump_internal_known)
                return 'numba'

        jump_internal_evals, internal_known = _declared_jump_internal_evals(proposal_manager)
        kernel = self._python_block_kernel(proposal_manager, like_obj, T_ladder)
        n_target_evals, n_internal_evals = kernel(
            samples,
            logLs,
            chain_track,
            T_ladder.betas,
            proposal_manager.jump_probs,
            tracker_manager.exchange_tracker,
            tracker_manager.esd_exchange,
            tracker_manager.accept_record,
            tracker_manager.esd_record,
            tuple(None for _ in proposal_manager.managers),
            jump_internal_evals,
            zero_loglike,
        )
        self._account(eval_accounting, n_target_evals, n_internal_evals, internal_known)
        return 'python'
