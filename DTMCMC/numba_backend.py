"""Registry-driven nopython backend for the serial PTMCMC block transition.

Likelihoods, jumps, component managers, and exchange managers opt in without
becoming jitclasses.  The public objects remain ordinary Python objects; this
module snapshots their Numba-compatible state at block entry and generates a
monomorphic block function for the concrete ordered proposal graph.
"""

from dataclasses import dataclass
from types import FunctionType
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeVar, cast
from warnings import warn

import numpy as np
from numba import njit
from numba.core.errors import NumbaError
from numpy.typing import NDArray

from DTMCMC.likelihood import prior_draw_rectangular, validate_bounds_rectangular
from DTMCMC.mcmc_kernel_helpers import mcmc_decision_helper

if TYPE_CHECKING:
    from collections.abc import Callable

    from DTMCMC.jump_manager import AbstractJump
    from DTMCMC.proposal_manager import AbstractProposalManager
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder
    from DTMCMC.tracker_manager import TrackerManager


ClassT = TypeVar('ClassT', bound=type[Any])


class RectangularBoundsLikelihood(Protocol):
    """Likelihood surface required by the default rectangular native adapter."""

    n_par: int
    n_evals: int
    low_lims: NDArray[np.floating]
    high_lims: NDArray[np.floating]


class NativeSamplerState(NamedTuple):
    """Mutable sampler arrays consumed by a generated block program."""

    betas: NDArray[np.floating]
    logLs: NDArray[np.floating]
    samples: NDArray[np.floating]
    chain_track: NDArray[np.int64]


class NativeTrackerState(NamedTuple):
    """Mutable diagnostic arrays consumed by a generated block program."""

    exchange_tracker: NDArray[np.int64]
    esd_exchange: NDArray[np.floating]
    accept_record: NDArray[np.int64]
    esd_record: NDArray[np.floating]


class NativeLikelihoodState(NamedTuple):
    """Runtime-constant likelihood values and extension-defined state."""

    n_par: int
    low_lims: NDArray[np.floating]
    high_lims: NDArray[np.floating]
    custom: tuple[Any, ...]


class NativeProposalState(NamedTuple):
    """Proposal probabilities and one native state per component manager."""

    jump_probs: NDArray[np.floating]
    manager_states: tuple[Any, ...]


class NativeExchangeState(NamedTuple):
    """One extension-defined native state for the exchange manager."""

    custom: Any


@njit(inline='always')
def _uniform_prior_draw(
    n_par: int,
    low_lims: NDArray[np.floating],
    high_lims: NDArray[np.floating],
    _state: tuple[Any, ...],
) -> NDArray[np.floating]:
    return prior_draw_rectangular(n_par, low_lims, high_lims)


@njit(inline='always')
def _uniform_prior_factor(_params: NDArray[np.floating], _state: tuple[Any, ...]) -> float:
    return 0.0


@njit(inline='always')
def _rectangular_validate_bounds(
    params: NDArray[np.floating],
    low_lims: NDArray[np.floating],
    high_lims: NDArray[np.floating],
    _state: tuple[Any, ...],
) -> tuple[NDArray[np.floating], bool]:
    return validate_bounds_rectangular(params, low_lims, high_lims)


def _ensure_njit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return a Numba dispatcher for ``func`` without double-wrapping one."""
    if hasattr(func, 'py_func'):
        return func
    return cast('Callable[..., Any]', njit(func))


def _adapt_loglike(loglike: Callable[..., float], n_state: int) -> Callable[..., float]:
    """Generate an adapter from ``(params, *state)`` to ``(params, state)``."""
    native_loglike = _ensure_njit(loglike)
    state_args = ''.join(f', state[{idx}]' for idx in range(n_state))
    source = f'def adapted(params, state):\n    return native_loglike(params{state_args})\n'
    namespace: dict[str, Any] = {'native_loglike': native_loglike}
    exec(compile(source, '<dtmcmc-likelihood-adapter>', 'exec'), namespace)  # noqa: S102
    return njit(inline='always')(namespace['adapted'])


def _default_bounds_getter(
    like_obj: RectangularBoundsLikelihood,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Get rectangular bounds without copying them."""
    return np.asarray(like_obj.low_lims), np.asarray(like_obj.high_lims)


@dataclass(frozen=True)
class NativeLikelihoodSpec:
    """Native functions and Python-side state extraction for a likelihood."""

    loglike: Callable[..., float]
    state_getter: Callable[[Any], tuple[Any, ...]]
    bounds_getter: Callable[[Any], tuple[NDArray[np.floating], NDArray[np.floating]]]
    prior_draw: Callable[..., NDArray[np.floating]]
    prior_factor: Callable[..., float]
    validate_bounds: Callable[..., tuple[NDArray[np.floating], bool]]


@dataclass(frozen=True)
class NativeJumpSpec:
    """Factory for a jump kernel specialized to one likelihood contract."""

    factory: Callable[[NativeLikelihoodSpec], Callable[..., tuple[NDArray[np.floating], float, bool]]]


@dataclass(frozen=True)
class NativeJumpManagerSpec:
    """State translation and per-step transition for a component manager."""

    state_getter: Callable[[Any], Any]
    state_setter: Callable[[Any, Any], None]
    post_step: Callable[..., Any]


@dataclass(frozen=True)
class NativeExchangeManagerSpec:
    """State translation, scheduling, and execution for an exchange manager."""

    state_getter: Callable[[Any], Any]
    is_exchange_step: Callable[..., bool]
    exchange: Callable[..., None]


_LIKELIHOOD_REGISTRY: dict[type[Any], NativeLikelihoodSpec] = {}
_JUMP_REGISTRY: dict[type[Any], NativeJumpSpec] = {}
_JUMP_MANAGER_REGISTRY: dict[type[Any], NativeJumpManagerSpec] = {}
_EXCHANGE_MANAGER_REGISTRY: dict[type[Any], NativeExchangeManagerSpec] = {}


def jittable_likelihood(
    loglike: Callable[..., float],
    *,
    state_attrs: tuple[str, ...] = (),
    prior_draw: Callable[..., NDArray[np.floating]] | None = None,
    prior_factor: Callable[..., float] | None = None,
    validate_bounds: Callable[..., tuple[NDArray[np.floating], bool]] | None = None,
    bounds_getter: Callable[[Any], tuple[NDArray[np.floating], NDArray[np.floating]]] | None = None,
) -> Callable[[ClassT], ClassT]:
    """Register an ordinary likelihood class for generated native execution.

    ``loglike`` has signature ``(params, *state)``. Values named by
    ``state_attrs`` are read at each block entry. Rectangular bounds and a
    uniform prior are supplied by default; non-uniform priors can provide
    native draw and log-density functions using the standardized signatures
    documented below::

        prior_draw(n_par, low_lims, high_lims, state) -> params
        prior_factor(params, state) -> log_density
        validate_bounds(params, low_lims, high_lims, state) -> (params, ok)
    """
    adapted_loglike = _adapt_loglike(loglike, len(state_attrs))
    native_prior_draw = _uniform_prior_draw if prior_draw is None else _ensure_njit(prior_draw)
    native_prior_factor = _uniform_prior_factor if prior_factor is None else _ensure_njit(prior_factor)
    native_validate = _rectangular_validate_bounds if validate_bounds is None else _ensure_njit(validate_bounds)
    get_bounds = _default_bounds_getter if bounds_getter is None else bounds_getter

    def decorate(cls: ClassT) -> ClassT:
        def get_state(like_obj: Any) -> tuple[Any, ...]:
            return tuple(getattr(like_obj, name) for name in state_attrs)

        _LIKELIHOOD_REGISTRY[cls] = NativeLikelihoodSpec(
            loglike=adapted_loglike,
            state_getter=get_state,
            bounds_getter=get_bounds,
            prior_draw=native_prior_draw,
            prior_factor=native_prior_factor,
            validate_bounds=native_validate,
        )
        return cls

    return decorate


def jittable_jump(
    native_function: Callable[..., tuple[NDArray[np.floating], float, bool]] | None = None,
    *,
    factory: Callable[[NativeLikelihoodSpec], Callable[..., tuple[NDArray[np.floating], float, bool]]] | None = None,
) -> Callable[[ClassT], ClassT]:
    """Register a jump class with a native function or likelihood-aware factory.

    A direct native function receives ``(sample_point, itrt, manager_state,
    likelihood_state)``. A factory receives the concrete native likelihood
    specification and returns a function with that same runtime signature.
    """
    if (native_function is None) == (factory is None):
        msg = 'jittable_jump requires exactly one of native_function or factory'
        raise TypeError(msg)

    jump_factory: Callable[[NativeLikelihoodSpec], Callable[..., tuple[NDArray[np.floating], float, bool]]]
    if factory is None:
        assert native_function is not None
        ensured = _ensure_njit(native_function)

        def direct_factory(_spec: NativeLikelihoodSpec) -> Callable[..., tuple[NDArray[np.floating], float, bool]]:
            return ensured

        jump_factory = direct_factory
    else:
        jump_factory = factory

    def decorate(cls: ClassT) -> ClassT:
        _JUMP_REGISTRY[cls] = NativeJumpSpec(factory=jump_factory)
        return cls

    return decorate


def jittable_jump_manager(
    *,
    state_getter: Callable[[Any], Any],
    state_setter: Callable[[Any, Any], None],
    post_step: Callable[..., Any],
) -> Callable[[ClassT], ClassT]:
    """Register native state translation for a component proposal manager."""
    native_post_step = _ensure_njit(post_step)

    def decorate(cls: ClassT) -> ClassT:
        _JUMP_MANAGER_REGISTRY[cls] = NativeJumpManagerSpec(
            state_getter=state_getter,
            state_setter=state_setter,
            post_step=native_post_step,
        )
        return cls

    return decorate


def jittable_exchange_manager(
    *,
    state_getter: Callable[[Any], Any],
    is_exchange_step: Callable[..., bool],
    exchange: Callable[..., None],
) -> Callable[[ClassT], ClassT]:
    """Register native state, cadence, and execution for an exchange manager."""
    native_schedule = _ensure_njit(is_exchange_step)
    native_exchange = _ensure_njit(exchange)

    def decorate(cls: ClassT) -> ClassT:
        _EXCHANGE_MANAGER_REGISTRY[cls] = NativeExchangeManagerSpec(
            state_getter=state_getter,
            is_exchange_step=native_schedule,
            exchange=native_exchange,
        )
        return cls

    return decorate


class NativeBackendUnsupportedError(RuntimeError):
    """The requested object graph has no complete native registration."""


class NativeBackendCompilationError(RuntimeError):
    """A fully registered native graph failed to compile or execute in Numba."""


@dataclass(frozen=True)
class NativeSerialProgram:
    """Compiled generated kernel plus Python-side state translators."""

    kernel: Callable[..., tuple[tuple[Any, ...], int]]
    manager_specs: tuple[NativeJumpManagerSpec, ...]
    likelihood_spec: NativeLikelihoodSpec
    exchange_spec: NativeExchangeManagerSpec
    source: str


_PROGRAM_CACHE: dict[tuple[Any, ...], NativeSerialProgram] = {}


def _graph_signature(proposal_manager: AbstractProposalManager, like_obj: Any) -> tuple[Any, ...]:
    manager_graph = tuple(
        (type(manager), tuple(type(jump) for jump in manager.jumps)) for manager in proposal_manager.managers
    )
    return type(like_obj), manager_graph, type(proposal_manager.exchange_manager)


def _registration_inventory(proposal_manager: AbstractProposalManager, like_obj: Any) -> tuple[list[str], int, int]:
    """Return missing component descriptions and registered/total counts."""
    components: list[tuple[str, type[Any], dict[type[Any], Any]]] = [
        ('likelihood', type(like_obj), _LIKELIHOOD_REGISTRY),
        ('exchange manager', type(proposal_manager.exchange_manager), _EXCHANGE_MANAGER_REGISTRY),
    ]
    for idx, manager in enumerate(proposal_manager.managers):
        components.append((f'proposal manager {idx}', type(manager), _JUMP_MANAGER_REGISTRY))
        for jump_idx, jump in enumerate(manager.jumps):
            components.append((f'proposal manager {idx} jump {jump_idx}', type(jump), _JUMP_REGISTRY))

    missing = [
        f'{role} {component_type.__qualname__}'
        for role, component_type, registry in components
        if component_type not in registry
    ]
    registered = len(components) - len(missing)
    return missing, registered, len(components)


def _jump_manager_indices(proposal_manager: AbstractProposalManager) -> tuple[int, ...]:
    """Map the aggregate flattened jump order back to component-manager states."""
    flattened: list[AbstractJump] = []
    manager_indices: list[int] = []
    for manager_idx, manager in enumerate(proposal_manager.managers):
        flattened.extend(manager.jumps)
        manager_indices.extend([manager_idx] * len(manager.jumps))
    if len(flattened) != len(proposal_manager.jumps) or any(
        actual is not expected for actual, expected in zip(proposal_manager.jumps, flattened, strict=True)
    ):
        msg = 'proposal_manager.jumps must be the ordered identity-preserving flattening of managers[*].jumps'
        raise NativeBackendUnsupportedError(msg)
    return tuple(manager_indices)


def _make_serial_program(
    proposal_manager: AbstractProposalManager,
    likelihood_spec: NativeLikelihoodSpec,
    exchange_spec: NativeExchangeManagerSpec,
) -> NativeSerialProgram:
    """Generate and JIT a monomorphic block program for one concrete graph."""
    manager_specs = tuple(_JUMP_MANAGER_REGISTRY[type(manager)] for manager in proposal_manager.managers)
    manager_indices = _jump_manager_indices(proposal_manager)
    jump_specs = tuple(_JUMP_REGISTRY[type(jump)] for jump in proposal_manager.jumps)
    jump_functions = tuple(spec.factory(likelihood_spec) for spec in jump_specs)

    namespace: dict[str, Any] = {
        'choose_prob_helper': __import__('DTMCMC.jump_manager', fromlist=['choose_prob_helper']).choose_prob_helper,
        'mcmc_decision_helper': mcmc_decision_helper,
        'loglike': likelihood_spec.loglike,
        'prior_factor': likelihood_spec.prior_factor,
        'validate_bounds': likelihood_spec.validate_bounds,
        'exchange_is_step': exchange_spec.is_exchange_step,
        'exchange_execute': exchange_spec.exchange,
    }
    for idx, jump_function in enumerate(jump_functions):
        namespace[f'jump_{idx}'] = jump_function
    for idx, manager_spec in enumerate(manager_specs):
        namespace[f'manager_post_{idx}'] = manager_spec.post_step

    lines = [
        'def advance_block_numba_serial(sampler, tracker, likelihood, proposal, exchange):',
        '    samples = sampler.samples',
        '    logLs = sampler.logLs',
        '    chain_track = sampler.chain_track',
        '    betas = sampler.betas',
        '    block_size = samples.shape[0] - 1',
        '    n_chain = samples.shape[1]',
        '    n_evals = 0',
    ]
    lines.extend(f'    manager_state_{idx} = proposal.manager_states[{idx}]' for idx in range(len(manager_specs)))
    lines.extend(
        [
            '    for itrb in range(1, block_size + 1):',
            '        if exchange_is_step(itrb, exchange.custom):',
            '            exchange_execute(',
            '                itrb, samples, logLs, n_chain, betas,',
            '                tracker.exchange_tracker, tracker.esd_exchange,',
            '                chain_track, exchange.custom,',
            '            )',
            '        else:',
            '            for itrt in range(n_chain):',
            '                idx_jump = choose_prob_helper(proposal.jump_probs[itrt])',
            '                sample_point = samples[itrb - 1, itrt]',
        ]
    )
    for jump_idx, manager_idx in enumerate(manager_indices):
        keyword = 'if' if jump_idx == 0 else 'elif'
        lines.extend(
            [
                f'                {keyword} idx_jump == {jump_idx}:',
                f'                    new_point, density_fac, success = jump_{jump_idx}(',
                f'                        sample_point, itrt, manager_state_{manager_idx}, likelihood,',
                '                    )',
            ]
        )
    lines.extend(
        [
            '                else:',
            "                    raise RuntimeError('generated proposal index out of range')",
            '                if success:',
            '                    new_point, success = validate_bounds(',
            '                        new_point, likelihood.low_lims, likelihood.high_lims, likelihood.custom,',
            '                    )',
            '                if success:',
            '                    density_fac += prior_factor(new_point, likelihood.custom) - prior_factor(',
            '                        sample_point, likelihood.custom,',
            '                    )',
            '                if success:',
            '                    logL_new = loglike(new_point, likelihood.custom)',
            '                    n_evals += 1',
            '                else:',
            '                    logL_new = -np.inf',
            '                mcmc_decision_helper(',
            '                    itrb, samples, logLs, betas, tracker.accept_record, tracker.esd_record,',
            '                    itrt, new_point, logL_new, density_fac, idx_jump,',
            '                )',
            '            chain_track[itrb, :] = chain_track[itrb - 1, :]',
        ]
    )
    namespace['np'] = np
    lines.extend(
        f'        manager_state_{idx} = manager_post_{idx}(manager_state_{idx}, samples[itrb])'
        for idx in range(len(manager_specs))
    )
    states_expr = ', '.join(f'manager_state_{idx}' for idx in range(len(manager_specs)))
    if len(manager_specs) == 1:
        states_expr += ','
    lines.append(f'    return ({states_expr}), n_evals')
    source = '\n'.join(lines) + '\n'

    exec(compile(source, '<dtmcmc-generated-serial-kernel>', 'exec'), namespace)  # noqa: S102
    function = namespace['advance_block_numba_serial']
    if not isinstance(function, FunctionType):
        msg = 'generated serial kernel was not a Python function'
        raise TypeError(msg)
    kernel = njit()(function)
    return NativeSerialProgram(kernel, manager_specs, likelihood_spec, exchange_spec, source)


def _resolve_program(
    signature: tuple[Any, ...], proposal_manager: AbstractProposalManager, like_obj: Any
) -> NativeSerialProgram:
    cached = _PROGRAM_CACHE.get(signature)
    if cached is not None:
        return cached
    likelihood_spec = _LIKELIHOOD_REGISTRY[type(like_obj)]
    exchange_spec = _EXCHANGE_MANAGER_REGISTRY[type(proposal_manager.exchange_manager)]
    program = _make_serial_program(proposal_manager, likelihood_spec, exchange_spec)
    _PROGRAM_CACHE[signature] = program
    return program


class NativeSerialBackend:
    """Resolve, cache, and execute the generated backend for one sampler."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.signature: tuple[Any, ...] | None = None
        self.program: NativeSerialProgram | None = None
        self.warned_signatures: set[tuple[Any, ...]] = set()

    def _resolve(self, proposal_manager: AbstractProposalManager, like_obj: Any) -> None:
        signature = _graph_signature(proposal_manager, like_obj)
        if signature == self.signature:
            return

        self.signature = signature
        self.program = None
        missing, registered, total = _registration_inventory(proposal_manager, like_obj)
        if missing:
            detail = '; '.join(missing)
            if self.mode == 'numba':
                msg = f"kernel_backend='numba' requires a fully registered graph; missing: {detail}"
                raise NativeBackendUnsupportedError(msg)
            if registered and signature not in self.warned_signatures:
                warn(
                    f"kernel_backend='auto' found a mixed native/Python graph and will use Python; missing: {detail}",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self.warned_signatures.add(signature)
            assert 0 <= registered < total
            return

        self.program = _resolve_program(signature, proposal_manager, like_obj)

    def try_advance_block(
        self,
        T_ladder: TemperatureLadder,
        logLs: NDArray[np.floating],
        samples: NDArray[np.floating],
        chain_track: NDArray[np.int64],
        proposal_manager: AbstractProposalManager,
        like_obj: Any,
        tracker_manager: TrackerManager,
    ) -> bool:
        """Run a native block when the graph is eligible, otherwise return false."""
        if self.mode == 'python':
            return False
        self._resolve(proposal_manager, like_obj)
        if self.program is None:
            return False

        program = self.program
        low_lims, high_lims = program.likelihood_spec.bounds_getter(like_obj)
        likelihood_state = NativeLikelihoodState(
            like_obj.n_par,
            low_lims,
            high_lims,
            program.likelihood_spec.state_getter(like_obj),
        )
        manager_states = tuple(
            spec.state_getter(manager)
            for spec, manager in zip(program.manager_specs, proposal_manager.managers, strict=True)
        )
        exchange_state = NativeExchangeState(program.exchange_spec.state_getter(proposal_manager.exchange_manager))
        sampler_state = NativeSamplerState(T_ladder.betas, logLs, samples, chain_track)
        tracker_state = NativeTrackerState(
            tracker_manager.exchange_tracker,
            tracker_manager.esd_exchange,
            tracker_manager.accept_record,
            tracker_manager.esd_record,
        )
        proposal_state = NativeProposalState(proposal_manager.jump_probs, manager_states)

        try:
            updated_states, n_evals = program.kernel(
                sampler_state,
                tracker_state,
                likelihood_state,
                proposal_state,
                exchange_state,
            )
        except NumbaError as exc:
            msg = 'fully registered serial-native graph failed Numba compilation or execution'
            raise NativeBackendCompilationError(msg) from exc

        for spec, manager, state in zip(program.manager_specs, proposal_manager.managers, updated_states, strict=True):
            spec.state_setter(manager, state)
        like_obj.n_evals += n_evals
        return True
