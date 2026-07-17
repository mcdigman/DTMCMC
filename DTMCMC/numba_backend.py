"""Native (nopython) serial backend assembled from per-class jitted functions.

Every component of a proposal graph — the likelihood, each jump, each
component manager, and the exchange manager — may opt into native execution
by exposing ``bind_native*`` hooks. A hook returns per-class jitted functions
(the same function objects on every call); everything instance-specific
travels in a small per-component runtime state bundle returned by
``native_state()`` and re-read at every block entry, so configuration
changes between blocks behave exactly like the Python path and nothing
mutable is ever baked into compiled code (Numba treats closure-captured
arrays as read-only non-aliasing constants, so baking mutable or rebindable
state risks silently stale reads).

Because the bound functions are per-class constants, the assembled block
program depends only on the graph structure, not on which instances built
it: programs are cached by the identity of the bound functions, so
structurally identical samplers share one compiled kernel and Numba's own
dispatcher handles any residual signature specialization. The assembly uses
plain recursive closure factories (the Numba equivalent of
``functools.partial``): no strings, no ``exec``, and every line is visible
to linters and type checkers.

A bound native jump has the ``AbstractJump.__call__`` signature plus the
owning manager's runtime state and the likelihood's runtime state.
likelihood functions have the ``AbstractLikelihood`` method signatures.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, Protocol, TypeVar
from warnings import warn

import numpy as np
from numba import njit
from numba.core.errors import NumbaError
from numpy.typing import NDArray

from DTMCMC.jump_manager import JumpManager, choose_prob_helper
from DTMCMC.mcmc_kernel_helpers import mcmc_decision_helper

if TYPE_CHECKING:
    from collections.abc import Callable

    from DTMCMC.eval_accounting import EvalAccounting
    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.proposal_manager import AbstractProposalManager
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder
    from DTMCMC.tracker_manager import TrackerManager

_VALID_MODES = ('auto', 'numba', 'python')

LikeStateT = TypeVar('LikeStateT')
LikeStateT_contra = TypeVar('LikeStateT_contra', contravariant=True)
ManagerStateT_contra = TypeVar('ManagerStateT_contra', contravariant=True)
ExchangeStateT = TypeVar('ExchangeStateT')
ExchangeStateT_contra = TypeVar('ExchangeStateT_contra', contravariant=True)


class NativeBackendUnsupportedError(RuntimeError):
    """The requested object graph has no complete native binding."""


class NativeBackendCompilationError(RuntimeError):
    """A fully bindable native graph failed to compile or execute in Numba."""


class NativeLoglikeCall(Protocol[LikeStateT_contra]):
    """Jitted log-likelihood: ``AbstractLikelihood.get_loglike`` plus the state bundle."""

    def __call__(self, params_in: NDArray[np.floating], state: LikeStateT_contra, /) -> float:
        """Return the log likelihood at ``params_in``."""
        ...


class NativePriorDrawCall(Protocol[LikeStateT_contra]):
    """Jitted prior draw: ``AbstractLikelihood.prior_draw`` plus the state bundle."""

    def __call__(self, state: LikeStateT_contra, /) -> NDArray[np.floating]:
        """Return one draw from the prior."""
        ...


class NativePriorFactorCall(Protocol[LikeStateT_contra]):
    """Jitted prior log density: ``AbstractLikelihood.prior_factor`` plus the state bundle."""

    def __call__(self, params_in: NDArray[np.floating], state: LikeStateT_contra, /) -> float:
        """Return the untempered log prior density up to an additive constant."""
        ...


class NativeValidateBoundsCall(Protocol[LikeStateT_contra]):
    """Jitted bounds validation: ``AbstractLikelihood.validate_bounds`` plus the state bundle."""

    def __call__(
        self, params_in: NDArray[np.floating], state: LikeStateT_contra, /
    ) -> tuple[NDArray[np.floating], bool]:
        """Return the (possibly corrected) point and whether it is in bounds."""
        ...


class NativeJumpCall(Protocol[ManagerStateT_contra, LikeStateT_contra]):
    """Jitted jump: ``AbstractJump.__call__`` plus the manager and likelihood states."""

    def __call__(
        self,
        sample_point: NDArray[np.floating],
        itrt: int,
        manager_state: ManagerStateT_contra,
        like_state: LikeStateT_contra,
        /,
    ) -> tuple[NDArray[np.floating], float, bool]:
        """Return the proposed point, its density factor, and success."""
        ...


class NativePostStepCall(Protocol[ManagerStateT_contra]):
    """Jitted per-step manager update over the manager's runtime state."""

    def __call__(self, state: ManagerStateT_contra, samples_row: NDArray[np.floating], /) -> None:
        """Update the manager state in place after one step of all chains."""
        ...


class NativeExchangeStepCall(Protocol[ExchangeStateT_contra]):
    """Jitted exchange cadence over the exchange manager's runtime state."""

    def __call__(self, itrb: int, state: ExchangeStateT_contra, /) -> bool:
        """Return whether step ``itrb`` is an exchange step."""
        ...


class NativeExchangeCall(Protocol[ExchangeStateT_contra]):
    """Jitted exchange executor over the exchange manager's runtime state."""

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
        state: ExchangeStateT_contra,
        /,
    ) -> None:
        """Execute one exchange step."""
        ...


@dataclass(frozen=True)
class NativeLikelihoodFunctions(Generic[LikeStateT]):
    """Per-class jitted likelihood functions consuming a runtime state bundle.

    Signatures match the ``AbstractLikelihood`` methods minus ``self`` plus
    the state; the kernel treats the state as opaque, so it never sees
    bounds, priors, or any other likelihood internals. Hooks must return the
    same function objects on every call so structurally identical samplers
    share one compiled program.
    """

    loglike: NativeLoglikeCall[Any]
    prior_draw: NativePriorDrawCall[Any]
    prior_factor: NativePriorFactorCall[Any]
    validate_bounds: NativeValidateBoundsCall[Any]


@dataclass(frozen=True)
class NativeExchangeFunctions(Generic[ExchangeStateT]):
    """Per-class jitted exchange schedule and executor over a runtime state bundle."""

    is_exchange_step: NativeExchangeStepCall[Any]
    exchange: NativeExchangeCall[Any]


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


_LIKELIHOOD_NATIVE_PAIRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ('get_loglike', ('bind_native_loglike', 'bind_native')),
    ('prior_draw', ('bind_native_prior_draw', 'bind_native')),
    ('prior_factor', ('bind_native_prior_factor', 'bind_native')),
    ('validate_bounds', ('bind_native_validate_bounds', 'bind_native')),
)


def _likelihood_problem(like_obj: AbstractLikelihood[NamedTuple]) -> str | None:
    cls = type(like_obj)
    if _defining_class(cls, 'bind_native') is None:
        return f'{cls.__qualname__} does not define bind_native'
    if _defining_class(cls, 'inputs') is None:
        return f'{cls.__qualname__} does not define inputs'
    for method_name, hook_names in _LIKELIHOOD_NATIVE_PAIRS:
        problem = _stale_native_override(cls, method_name, hook_names)
        if problem is not None:
            return problem
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
    if _defining_class(cls, 'native_state') is None:
        return f'{cls.__qualname__} does not define native_state'
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


def _binding_inventory[LikelihoodType: AbstractLikelihood[NamedTuple]](
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


def _check_flattening[LikelihoodType: AbstractLikelihood[NamedTuple]](
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
        self, idx_jump: int, sample_point: NDArray[np.floating], itrt: int, state: Any, like_state: Any, /
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
        like_state: Any,
        /,
    ) -> tuple[NDArray[np.floating], float, bool]:
        """Dispatch the flattened jump index."""
        ...


def _build_local_dispatch(jump_calls: tuple[NativeJumpCall[Any, Any], ...]) -> _LocalDispatchCall:
    """Dispatch a manager-local jump index over one manager's bound jumps.

    The recursion happens in Python at assembly time; with
    ``inline='always'`` Numba flattens the chain into the same branch
    sequence a hand-written ``if/elif`` ladder would produce. The caller
    guarantees the index is within range, so the final leaf executes
    unconditionally.
    """
    head = jump_calls[0]
    if len(jump_calls) == 1:

        @njit(inline='always')
        def dispatch_leaf(
            _idx_jump: int, sample_point: NDArray[np.floating], itrt: int, state: object, like_state: object
        ) -> tuple[NDArray[np.floating], float, bool]:
            return head(sample_point, itrt, state, like_state)

        return dispatch_leaf
    rest = _build_local_dispatch(jump_calls[1:])

    @njit(inline='always')
    def dispatch(
        idx_jump: int, sample_point: NDArray[np.floating], itrt: int, state: object, like_state: object
    ) -> tuple[NDArray[np.floating], float, bool]:
        if idx_jump == 0:
            return head(sample_point, itrt, state, like_state)
        return rest(idx_jump - 1, sample_point, itrt, state, like_state)

    return dispatch


def _build_manager_dispatch(
    per_manager_calls: tuple[tuple[NativeJumpCall[Any, Any], ...], ...],
) -> _ManagerDispatchCall:
    """Dispatch a flattened jump index over the managers' bound jumps.

    Peels the heterogeneous ``manager_states`` runtime tuple with static
    indexing (``states[0]`` / ``states[1:]``) so each manager's jumps see
    exactly their own manager's runtime native state.
    """
    head = _build_local_dispatch(per_manager_calls[0])
    n_head = len(per_manager_calls[0])
    if len(per_manager_calls) == 1:

        @njit(inline='always')
        def dispatch_last(
            idx_jump: int,
            sample_point: NDArray[np.floating],
            itrt: int,
            states: tuple[object, ...],
            like_state: object,
        ) -> tuple[NDArray[np.floating], float, bool]:
            return head(idx_jump, sample_point, itrt, states[0], like_state)

        return dispatch_last
    rest = _build_manager_dispatch(per_manager_calls[1:])

    @njit(inline='always')
    def dispatch(
        idx_jump: int,
        sample_point: NDArray[np.floating],
        itrt: int,
        states: tuple[object, ...],
        like_state: object,
    ) -> tuple[NDArray[np.floating], float, bool]:
        if idx_jump < n_head:
            return head(idx_jump, sample_point, itrt, states[0], like_state)
        return rest(idx_jump - n_head, sample_point, itrt, states[1:], like_state)

    return dispatch


@njit(inline='always')
def _post_step_noop(_states: tuple[object, ...], _samples: NDArray[np.floating]) -> None:
    """Per-step update for managers with no native per-step work."""
    return


def _build_post_chain(post_steps: tuple[NativePostStepCall[Any] | None, ...]) -> NativePostStepCall[Any]:
    """Chain the managers' bound per-step updates in manager order.

    ``post_steps`` has one entry per manager (None when idle) so the chain
    peels the ``manager_states`` tuple in step with the dispatch chain.
    """
    if not post_steps:
        return _post_step_noop
    head = post_steps[0]
    rest = _build_post_chain(post_steps[1:]) if len(post_steps) > 1 else None
    if head is None:
        if rest is None:
            return _post_step_noop
        rest_after_skip = rest

        @njit(inline='always')
        def post_skip(states: tuple[object, ...], samples_row: NDArray[np.floating]) -> None:
            rest_after_skip(states[1:], samples_row)

        return post_skip
    if rest is None:

        @njit(inline='always')
        def post_leaf(states: tuple[object, ...], samples_row: NDArray[np.floating]) -> None:
            head(states[0], samples_row)

        return post_leaf
    rest_fn = rest

    @njit(inline='always')
    def post_all(states: tuple[object, ...], samples_row: NDArray[np.floating]) -> None:
        head(states[0], samples_row)
        rest_fn(states[1:], samples_row)

    return post_all


def make_serial_kernel(
    per_manager_calls: tuple[tuple[NativeJumpCall[Any, Any], ...], ...],
    post_steps: tuple[NativePostStepCall[Any] | None, ...],
    likelihood_natives: NativeLikelihoodFunctions[Any],
    exchange_natives: NativeExchangeFunctions[Any],
) -> Callable[..., tuple[int, int]]:
    """Assemble the jitted serial block kernel for one bound graph structure.

    Only per-class functions are closed over; every instance-specific value
    arrives through the runtime state bundles, so the kernel is reusable
    across all samplers whose components bind the same functions. Returns
    the counts of target likelihood evaluations performed and of declared
    jump-internal evaluations incurred in the block.
    """
    dispatch = _build_manager_dispatch(per_manager_calls)
    post_all = _build_post_chain(post_steps)
    loglike = likelihood_natives.loglike
    prior_factor = likelihood_natives.prior_factor
    validate_bounds = likelihood_natives.validate_bounds
    is_exchange_step = exchange_natives.is_exchange_step
    do_exchange = exchange_natives.exchange

    @njit()
    def advance_block_numba_serial(
        samples: NDArray[np.floating],
        logLs: NDArray[np.floating],
        chain_track: NDArray[np.int64],
        betas: NDArray[np.floating],
        jump_probs: NDArray[np.floating],
        exchange_tracker: NDArray[np.int64],
        esd_exchange: NDArray[np.floating],
        accept_record: NDArray[np.int64],
        esd_record: NDArray[np.floating],
        like_state: object,
        manager_states: tuple[object, ...],
        exchange_state: object,
        jump_internal_evals: NDArray[np.int64],
        zero_loglike: bool,
    ) -> tuple[int, int]:
        block_size = samples.shape[0] - 1
        n_chain = samples.shape[1]
        n_target_evals = 0
        n_internal_evals = 0
        for itrb in range(1, block_size + 1):
            if is_exchange_step(itrb, exchange_state):
                do_exchange(
                    itrb, samples, logLs, n_chain, betas, exchange_tracker, esd_exchange, chain_track, exchange_state
                )
            else:
                for itrt in range(n_chain):
                    idx_jump = choose_prob_helper(jump_probs[itrt])
                    sample_point = samples[itrb - 1, itrt]
                    new_point, density_fac, success = dispatch(idx_jump, sample_point, itrt, manager_states, like_state)
                    n_internal_evals += jump_internal_evals[idx_jump]
                    if success:
                        new_point, success = validate_bounds(new_point, like_state)
                    if success:
                        density_fac += prior_factor(new_point, like_state) - prior_factor(sample_point, like_state)
                    if success:
                        if zero_loglike:
                            logL_new = 0.0
                        else:
                            logL_new = loglike(new_point, like_state)
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

    return advance_block_numba_serial


@dataclass(frozen=True)
class NativeSerialProgram:
    """Shared kernel plus strong references to the per-class bound functions.

    Holding the bound functions prevents garbage collection (and therefore
    ``id`` reuse) from aliasing a stale program onto freshly created
    functions while the program is cached; a hook that violates the
    per-class stability contract only costs a duplicate cache entry.
    """

    kernel: Callable[..., tuple[int, int]]
    likelihood_natives: NativeLikelihoodFunctions[Any]
    per_manager_calls: tuple[tuple[NativeJumpCall[Any, Any], ...], ...]
    post_steps: tuple[NativePostStepCall[Any] | None, ...]
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
    likelihood_natives: NativeLikelihoodFunctions[Any],
    per_manager_calls: tuple[tuple[NativeJumpCall[Any, Any], ...], ...],
    post_steps: tuple[NativePostStepCall[Any] | None, ...],
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
            id(likelihood_natives.validate_bounds),
        ),
        tuple(
            (None if post is None else id(post), tuple(id(call) for call in calls))
            for post, calls in zip(post_steps, per_manager_calls, strict=True)
        ),
        (id(exchange_natives.is_exchange_step), id(exchange_natives.exchange)),
    )


class NativeSerialBackend[LikelihoodType: AbstractLikelihood[NamedTuple]]:
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
        likelihood_natives = like_obj.bind_native()
        per_manager_calls: list[tuple[NativeJumpCall[Any, Any], ...]] = []
        post_steps: list[NativePostStepCall[Any] | None] = []
        manager_has_state: list[bool] = []
        for manager in proposal_manager.managers:
            per_manager_calls.append(tuple(jump.bind_native(likelihood_natives) for jump in manager.jumps))
            manager_has_state.append(_defining_class(type(manager), 'native_state') is not None)
            has_post = _defining_class(type(manager), 'bind_native_post_step') is not None
            post_steps.append(manager.bind_native_post_step() if has_post else None)
        assert sum(len(calls) for calls in per_manager_calls) == proposal_manager.jump_probs.shape[1]
        exchange_natives = proposal_manager.exchange_manager.bind_native()

        declared: list[int | None] = [getattr(jump, 'declared_internal_evals', None) for jump in proposal_manager.jumps]
        self._jump_internal_known = all(value is not None for value in declared)
        self._jump_internal_evals = np.array([0 if value is None else value for value in declared], dtype=np.int64)
        self._manager_has_state = tuple(manager_has_state)

        key = _structural_key(likelihood_natives, tuple(per_manager_calls), tuple(post_steps), exchange_natives)
        cached = _PROGRAM_CACHE.get(key)
        if cached is not None:
            return cached
        kernel = make_serial_kernel(tuple(per_manager_calls), tuple(post_steps), likelihood_natives, exchange_natives)
        program = NativeSerialProgram(
            kernel=kernel,
            likelihood_natives=likelihood_natives,
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
                # a hook declined at bind time (e.g. no native log-likelihood)
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

    def try_advance_block(
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
    ) -> bool:
        """Run a native block when the graph is bindable, otherwise return False."""
        if self.mode == 'python':
            return False
        self._resolve(proposal_manager, like_obj)
        if self.program is None:
            return False

        program = self.program
        # runtime state bundles are re-read at every block entry, so
        # configuration changes between blocks behave like the Python path
        like_state = like_obj.inputs
        manager_states = tuple(
            manager.native_state() if has_state else None
            for manager, has_state in zip(proposal_manager.managers, self._manager_has_state, strict=True)
        )
        exchange_state = proposal_manager.exchange_manager.native_state()
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
                like_state,
                manager_states,
                exchange_state,
                self._jump_internal_evals,
                zero_loglike,
            )
        except NumbaError as exc:
            # NumbaError here means the kernel failed to compile, which
            # happens before any execution, so the block arrays are untouched
            # and the Python path can rerun the block from samples[0]
            if self.mode == 'numba':
                msg = 'fully native-bindable graph failed Numba compilation or execution'
                raise NativeBackendCompilationError(msg) from exc
            self.program = None
            warn(
                f"kernel_backend='auto' native kernel failed to compile and will use Python: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        eval_accounting.proposal_targets += n_target_evals
        eval_accounting.proposal_internal += n_internal_evals
        if not self._jump_internal_known:
            eval_accounting.complete = False
        return True
