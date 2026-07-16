"""Native (nopython) serial backend assembled from baked jitted closures.

Every component of a proposal graph — the likelihood, each jump, each
component manager, and the exchange manager — may opt into native execution
by exposing ``bind_native*`` hooks that return Numba-jitted closures with the
component's immutable configuration absorbed at bind time as compile-time
constants (scalars, tuples, and frozen arrays such as rectangular bounds).

Mutable state cannot be baked: Numba types closure-captured arrays as
read-only and assumes they alias nothing, so a write through any other name
is not reliably visible to a baked read within one kernel invocation.
Arrays a manager mutates (the DE ring buffer and counters, the Fisher scale
arrays) therefore travel as an opaque per-manager runtime state, must be
identity-stable — updated in place, never rebound — for the lifetime of the
sampler, and are routed to that manager's jumps and per-step update by the
dispatch chain.

The backend assembles one monomorphic block kernel per concrete object graph
out of those closures using plain recursive closure factories (the Numba
equivalent of ``functools.partial``): no strings, no ``exec``, and every line
is visible to linters and type checkers. A bound native jump has the
``AbstractJump.__call__`` signature plus the owning manager's runtime state,
and the bound likelihood functions have the ``AbstractLikelihood`` method
signatures minus ``self``, so the native contract mirrors the Python
protocols directly.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from warnings import warn

import numpy as np
from numba import njit
from numba.core.errors import NumbaError
from numpy.typing import NDArray

from DTMCMC.jump_manager import JumpManager, choose_prob_helper
from DTMCMC.mcmc_kernel_helpers import mcmc_decision_helper

if TYPE_CHECKING:
    from collections.abc import Callable

    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.proposal_manager import AbstractProposalManager
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder
    from DTMCMC.tracker_manager import LikelihoodEvalTracker, TrackerManager

    NativeJumpCall = Callable[..., tuple[NDArray[np.floating], float, bool]]
    NativePostStepCall = Callable[..., None]

_VALID_MODES = ('auto', 'numba', 'python')


class NativeBackendUnsupportedError(RuntimeError):
    """The requested object graph has no complete native binding."""


class NativeBackendCompilationError(RuntimeError):
    """A fully bindable native graph failed to compile or execute in Numba."""


@dataclass(frozen=True)
class NativeLikelihoodFunctions:
    """Jitted likelihood closures with all instance state baked in.

    Signatures match the ``AbstractLikelihood`` methods minus ``self``;
    the kernel never sees bounds, priors, or any other likelihood state.
    """

    loglike: Callable[[NDArray[np.floating]], float]
    prior_draw: Callable[[], NDArray[np.floating]]
    prior_factor: Callable[[NDArray[np.floating]], float]
    validate_bounds: Callable[[NDArray[np.floating]], tuple[NDArray[np.floating], bool]]


@dataclass(frozen=True)
class NativeExchangeFunctions:
    """Jitted exchange schedule and executor with strategy constants baked in.

    ``exchange`` receives ``(itrb, samples, logLs, n_chain, betas,
    exchange_tracker, esd_exchange, chain_track)``.
    """

    is_exchange_step: Callable[[int], bool]
    exchange: Callable[..., None]


@runtime_checkable
class NativeBindableLikelihood(Protocol):
    """Likelihood that can bake its state into jitted closures."""

    def bind_native(self) -> NativeLikelihoodFunctions:
        """Return the baked native likelihood functions."""
        ...


@runtime_checkable
class NativeBindableJump(Protocol):
    """Jump that can bind a jitted equivalent of its ``__call__`` method.

    The bound closure has signature ``(sample_point, itrt, manager_state) ->
    (new_point, density_fac, success)``: ``__call__`` plus the owning
    manager's runtime native state (None for a stateless manager), which
    carries any mutable arrays that Numba's read-only closure capture cannot.
    """

    def bind_native(self, likelihood_natives: NativeLikelihoodFunctions) -> NativeJumpCall:
        """Return the jitted jump closure."""
        ...


@runtime_checkable
class NativeBindableJumpManager(Protocol):
    """Component manager that can bind its mutable state and per-step update.

    ``bind_native_state`` returns the manager's identity-stable mutable
    runtime state (e.g. a tuple of arrays updated in place), shared by its
    jumps and its per-step update; leave it undefined for a stateless
    manager. ``bind_native_post_step`` returns a jitted ``(state,
    samples_row) -> None`` closure, or None when there is no per-step work.
    """

    def bind_native_state(self) -> object:
        """Return the manager's mutable runtime native state."""
        ...

    def bind_native_post_step(self) -> NativePostStepCall | None:
        """Return the jitted per-step update closure, or None if idle."""
        ...


@runtime_checkable
class NativeBindableExchangeManager(Protocol):
    """Exchange manager that can bake its schedule and executor."""

    def bind_native(self) -> NativeExchangeFunctions:
        """Return the baked native exchange functions."""
        ...


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
    method without also overriding a hook, the inherited binding would bake
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


def _likelihood_problem(like_obj: AbstractLikelihood) -> str | None:
    cls = type(like_obj)
    if _defining_class(cls, 'bind_native') is None:
        return f'{cls.__qualname__} does not define bind_native'
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
    return None, _defining_class(cls, 'bind_native_state') is not None


def _binding_inventory(
    proposal_manager: AbstractProposalManager, like_obj: AbstractLikelihood
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


def _check_flattening(proposal_manager: AbstractProposalManager) -> None:
    """Require the aggregate jump list to be the ordered flattening of the managers'."""
    flattened = [jump for manager in proposal_manager.managers for jump in manager.jumps]
    if len(flattened) != len(proposal_manager.jumps) or any(
        actual is not expected for actual, expected in zip(proposal_manager.jumps, flattened, strict=True)
    ):
        msg = 'proposal_manager.jumps must be the ordered identity-preserving flattening of managers[*].jumps'
        raise NativeBackendUnsupportedError(msg)


def _build_local_dispatch(jump_calls: tuple[NativeJumpCall, ...]) -> NativeJumpCall:
    """Dispatch a manager-local jump index over one manager's bound jumps.

    The recursion happens in Python at bind time; with ``inline='always'``
    Numba flattens the chain into the same branch sequence a hand-written
    ``if/elif`` ladder would produce. The caller guarantees the index is
    within range, so the final leaf executes unconditionally.
    """
    head = jump_calls[0]
    if len(jump_calls) == 1:

        @njit(inline='always')
        def dispatch_leaf(
            _idx_jump: int, sample_point: NDArray[np.floating], itrt: int, state: object
        ) -> tuple[NDArray[np.floating], float, bool]:
            return head(sample_point, itrt, state)

        return dispatch_leaf
    rest = _build_local_dispatch(jump_calls[1:])

    @njit(inline='always')
    def dispatch(
        idx_jump: int, sample_point: NDArray[np.floating], itrt: int, state: object
    ) -> tuple[NDArray[np.floating], float, bool]:
        if idx_jump == 0:
            return head(sample_point, itrt, state)
        return rest(idx_jump - 1, sample_point, itrt, state)

    return dispatch


def _build_manager_dispatch(per_manager_calls: tuple[tuple[NativeJumpCall, ...], ...]) -> NativeJumpCall:
    """Dispatch a flattened jump index over the managers' bound jumps.

    Peels the heterogeneous ``manager_states`` runtime tuple with static
    indexing (``states[0]`` / ``states[1:]``) so each manager's jumps see
    exactly their own manager's mutable native state.
    """
    head = _build_local_dispatch(per_manager_calls[0])
    n_head = len(per_manager_calls[0])
    if len(per_manager_calls) == 1:

        @njit(inline='always')
        def dispatch_last(
            idx_jump: int, sample_point: NDArray[np.floating], itrt: int, states: tuple[object, ...]
        ) -> tuple[NDArray[np.floating], float, bool]:
            return head(idx_jump, sample_point, itrt, states[0])

        return dispatch_last
    rest = _build_manager_dispatch(per_manager_calls[1:])

    @njit(inline='always')
    def dispatch(
        idx_jump: int, sample_point: NDArray[np.floating], itrt: int, states: tuple[object, ...]
    ) -> tuple[NDArray[np.floating], float, bool]:
        if idx_jump < n_head:
            return head(idx_jump, sample_point, itrt, states[0])
        return rest(idx_jump - n_head, sample_point, itrt, states[1:])

    return dispatch


@njit(inline='always')
def _post_step_noop(_states: tuple[object, ...], _samples: NDArray[np.floating]) -> None:
    """Per-step update for managers with no native per-step work."""
    return


def _build_post_chain(post_steps: tuple[NativePostStepCall | None, ...]) -> NativePostStepCall:
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
    per_manager_calls: tuple[tuple[NativeJumpCall, ...], ...],
    post_steps: tuple[NativePostStepCall | None, ...],
    likelihood_natives: NativeLikelihoodFunctions,
    exchange_natives: NativeExchangeFunctions,
) -> Callable[..., int]:
    """Assemble the jitted serial block kernel for one bound object graph.

    All graph structure is closed over as compile-time constants; the
    runtime arguments are the sampler-owned mutable arrays plus the
    ``manager_states`` tuple of per-manager mutable state (Numba treats
    captured arrays as read-only non-aliasing constants, so mutated state
    cannot be baked). Returns the number of likelihood evaluations performed
    in the block.
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
        manager_states: tuple[object, ...],
    ) -> int:
        block_size = samples.shape[0] - 1
        n_chain = samples.shape[1]
        n_evals = 0
        for itrb in range(1, block_size + 1):
            if is_exchange_step(itrb):
                do_exchange(itrb, samples, logLs, n_chain, betas, exchange_tracker, esd_exchange, chain_track)
            else:
                for itrt in range(n_chain):
                    idx_jump = choose_prob_helper(jump_probs[itrt])
                    sample_point = samples[itrb - 1, itrt]
                    new_point, density_fac, success = dispatch(idx_jump, sample_point, itrt, manager_states)
                    if success:
                        new_point, success = validate_bounds(new_point)
                    if success:
                        density_fac += prior_factor(new_point) - prior_factor(sample_point)
                    if success:
                        logL_new = loglike(new_point)
                        n_evals += 1
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
        return n_evals

    return advance_block_numba_serial


@dataclass(frozen=True)
class NativeSerialProgram:
    """Bound kernel plus strong references to the objects baked into it.

    ``manager_states`` holds the per-manager mutable state tuple passed to
    the kernel on every block. Holding the bound components prevents garbage
    collection (and therefore ``id`` reuse) from aliasing a stale kernel onto
    a new object graph while the program is cached.
    """

    kernel: Callable[..., int]
    manager_states: tuple[object, ...]
    bound_refs: tuple[object, ...]


def _graph_signature(proposal_manager: AbstractProposalManager, like_obj: AbstractLikelihood) -> tuple[object, ...]:
    """Identity signature of the bound graph: baked closures are per-instance."""
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


def _bind_program(proposal_manager: AbstractProposalManager, like_obj: AbstractLikelihood) -> NativeSerialProgram:
    """Bind every component and assemble the kernel for one concrete graph."""
    _check_flattening(proposal_manager)
    likelihood_natives = like_obj.bind_native()  # type: ignore[attr-defined]
    per_manager_calls = []
    post_steps: list[NativePostStepCall | None] = []
    manager_states = []
    for manager in proposal_manager.managers:
        per_manager_calls.append(
            tuple(
                jump.bind_native(likelihood_natives)  # type: ignore[attr-defined]
                for jump in manager.jumps
            )
        )
        has_state = _defining_class(type(manager), 'bind_native_state') is not None
        manager_states.append(manager.bind_native_state() if has_state else None)  # type: ignore[attr-defined]
        has_post = _defining_class(type(manager), 'bind_native_post_step') is not None
        post_steps.append(manager.bind_native_post_step() if has_post else None)  # type: ignore[attr-defined]
    assert sum(len(calls) for calls in per_manager_calls) == proposal_manager.jump_probs.shape[1]
    exchange_natives = proposal_manager.exchange_manager.bind_native()  # type: ignore[attr-defined]
    kernel = make_serial_kernel(tuple(per_manager_calls), tuple(post_steps), likelihood_natives, exchange_natives)
    return NativeSerialProgram(
        kernel=kernel, manager_states=tuple(manager_states), bound_refs=(like_obj, proposal_manager)
    )


class NativeSerialBackend:
    """Bind, cache, and execute the native serial block kernel for one sampler."""

    def __init__(self, mode: str) -> None:
        if mode not in _VALID_MODES:
            msg = f'kernel_backend must be one of {_VALID_MODES}; got {mode!r}'
            raise NotImplementedError(msg)
        self.mode = mode
        self.signature: tuple[object, ...] | None = None
        self.program: NativeSerialProgram | None = None
        self.warned_signatures: set[tuple[object, ...]] = set()

    def _resolve(self, proposal_manager: AbstractProposalManager, like_obj: AbstractLikelihood) -> None:
        signature = _graph_signature(proposal_manager, like_obj)
        if signature == self.signature:
            return

        problems, explicit = _binding_inventory(proposal_manager, like_obj)
        if not problems:
            try:
                program = _bind_program(proposal_manager, like_obj)
            except NativeBackendUnsupportedError as exc:
                # a hook declined at bind time (e.g. no native log-likelihood)
                problems = [str(exc)]
            else:
                self.signature = signature
                self.program = program
                return

        detail = '; '.join(problems)
        if self.mode == 'numba':
            # not memoized: strict mode re-raises on every attempted block
            msg = f"kernel_backend='numba' requires a fully native-bindable graph; missing: {detail}"
            raise NativeBackendUnsupportedError(msg)
        self.signature = signature
        self.program = None
        if explicit and signature not in self.warned_signatures:
            warn(
                f"kernel_backend='auto' found a mixed native/Python graph and will use Python; missing: {detail}",
                RuntimeWarning,
                stacklevel=3,
            )
            self.warned_signatures.add(signature)

    def try_advance_block(
        self,
        T_ladder: TemperatureLadder,
        logLs: NDArray[np.floating],
        samples: NDArray[np.floating],
        chain_track: NDArray[np.int64],
        proposal_manager: AbstractProposalManager,
        like_obj: AbstractLikelihood,
        tracker_manager: TrackerManager,
        eval_tracker: LikelihoodEvalTracker,
    ) -> bool:
        """Run a native block when the graph is bindable, otherwise return False."""
        if self.mode == 'python':
            return False
        self._resolve(proposal_manager, like_obj)
        if self.program is None:
            return False

        program = self.program
        try:
            n_evals = program.kernel(
                samples,
                logLs,
                chain_track,
                T_ladder.betas,
                proposal_manager.jump_probs,
                tracker_manager.exchange_tracker,
                tracker_manager.esd_exchange,
                tracker_manager.accept_record,
                tracker_manager.esd_record,
                program.manager_states,
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
        eval_tracker.count(n_evals)
        return True
