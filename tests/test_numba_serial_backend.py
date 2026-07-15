"""Regression tests for the registry-driven serial nopython block kernel."""

import configparser
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pytest
from numba import njit
from numpy.typing import NDArray

from DTMCMC.de_manager import DEJumpManager
from DTMCMC.dtmcmc_sampler import DTMCMCSampler
from DTMCMC.exchange_manager import (
    NULL_TARGETS,
    ExchangeManager,
    do_ptmcmc_exchange,
)
from DTMCMC.jump_manager import AbstractJump, JumpManager
from DTMCMC.likelihood import RectangularLikelihood
from DTMCMC.likelihoods.normal_nd import GaussianLikelihood
from DTMCMC.likelihoods.uniform_gaussian_prior import UniformGaussianPriorLikelihood
from DTMCMC.numba_backend import (
    NativeBackendCompilationError,
    NativeBackendUnsupportedError,
    NativeLikelihoodState,
    jittable_exchange_manager,
    jittable_jump,
    jittable_jump_manager,
    jittable_likelihood,
)
from DTMCMC.proposal_manager import ProposalManager
from DTMCMC.rng_helpers import reset_seed_guard_for_tests, seed_run
from DTMCMC.temperature_ladder_helpers import GeometricTemperatureLadder
from experiments.harness.runner import build_sampler
from experiments.harness.spec import RunSpec

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_spec(
    likelihood_name: str = 'gaussian',
    *,
    n_par: int = 4,
    n_steps: int = 8,
    block_size: int = 4,
    n_chain: int = 6,
    n_cold: int = 1,
    store_thin: int = 1,
    exchange_strategy: str = 'sequential',
    track_full_exchanges: bool = False,
    proposal_overrides: dict[str, dict[str, Any]] | None = None,
) -> RunSpec:
    proposals: dict[str, dict[str, Any]] = {
        'FisherJumpManager': {'verbose_fisher': False},
        'DEJumpManager': {'de_size': 16},
    }
    if proposal_overrides is not None:
        for section, values in proposal_overrides.items():
            proposals.setdefault(section, {}).update(values)
    return RunSpec.from_dict(
        {
            'name': f'numba_serial_{likelihood_name}',
            'seed': 20260714,
            'likelihood': {'name': likelihood_name}
            if likelihood_name == 'hawaii'
            else {'name': likelihood_name, 'n_par': n_par},
            'ladder': {
                'kind': 'geometric',
                'n_chain': n_chain,
                'n_cold': n_cold,
                'T_cold': 1.0,
                'T_min': 1.0,
                'T_max': 100.0,
                'n_inf_final': 1,
            },
            'run': {
                'n_steps': n_steps,
                'block_size': block_size,
                'store_thin': store_thin,
                'checkpoint_every_blocks': 1,
            },
            'exchange': {'strategy': exchange_strategy, 'track_full_exchanges': track_full_exchanges},
            'proposals': proposals,
        }
    )


def _snapshot(sampler, like_obj) -> dict[str, object]:
    de_manager = next(manager for manager in sampler.proposal_manager.managers if isinstance(manager, DEJumpManager))
    tracker = sampler.tracker_manager
    return {
        'samples': sampler.samples.copy(),
        'logLs': sampler.logLs.copy(),
        'chain_track': sampler.chain_track.copy(),
        'samples_store': sampler.samples_store.copy(),
        'logLs_store': sampler.logLs_store.copy(),
        'accept_record': tracker.accept_record.copy(),
        'esd_record': tracker.esd_record.copy(),
        'exchange_tracker': tracker.exchange_tracker.copy(),
        'esd_exchange': tracker.esd_exchange.copy(),
        'cycle_tracker': tracker.cycle_tracker.copy(),
        'flow_up': np.asarray(tracker.flow_up_archive),
        'flow_labeled': np.asarray(tracker.flow_labeled_archive),
        'de_buffer': de_manager.de_buffer.copy(),
        'de_write': de_manager.itrde_write,
        'de_count': de_manager.itrde_count,
        'n_evals': like_obj.n_evals,
        'store_idx': sampler.store_idx,
        'store_counter': sampler.store_counter,
        'itrn': sampler.itrn,
    }


def _run(
    spec: RunSpec,
    backend: str,
    *,
    jump_label: str | None = None,
    like_factory: Callable[[], object] | None = None,
) -> tuple[dict[str, object], str]:
    reset_seed_guard_for_tests()
    try:
        seed_run(spec.seed)
        like_obj = None if like_factory is None else like_factory()
        sampler, like_obj = build_sampler(spec, like_obj=like_obj, kernel_backend=backend)  # type: ignore[arg-type]
        if jump_label is not None:
            jump_idx = sampler.proposal_manager.jump_labels_array.index(jump_label)
            sampler.proposal_manager.jump_probs.fill(0.0)
            sampler.proposal_manager.jump_probs[:, jump_idx] = 1.0
        for _ in range(spec.n_blocks):
            sampler.advance_block()
        return _snapshot(sampler, like_obj), sampler.last_kernel_backend
    finally:
        reset_seed_guard_for_tests()


def _assert_snapshots_equal(python_state: dict[str, object], numba_state: dict[str, object]) -> None:
    assert python_state.keys() == numba_state.keys()
    for key, python_value in python_state.items():
        numba_value = numba_state[key]
        if isinstance(python_value, np.ndarray):
            np.testing.assert_array_equal(numba_value, python_value, err_msg=key)
        else:
            assert numba_value == python_value, key


def test_default_serial_kernel_is_bit_exact_to_python() -> None:
    """The native block path preserves the complete fixed-seed serial state."""
    spec = _make_spec()
    python_state, python_backend = _run(spec, 'python')
    numba_state, numba_backend = _run(spec, 'numba')

    assert python_backend == 'python'
    assert numba_backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


@pytest.mark.parametrize(
    'exchange_strategy',
    ['random', 'sequential', 'adjacent', 'null', 'reverse_sequential', 'alternate_sequential'],
)
def test_every_exchange_strategy_is_bit_exact(exchange_strategy: str) -> None:
    """Generated dispatch delegates every built-in exchange strategy without drift."""
    spec = _make_spec(n_steps=8, exchange_strategy=exchange_strategy)
    python_state, _ = _run(spec, 'python')
    numba_state, backend = _run(spec, 'numba')
    assert backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


@pytest.mark.parametrize('track_full_exchanges', [False, True])
def test_random_exchange_tracking_modes_are_bit_exact(track_full_exchanges: bool) -> None:
    """Random pairing preserves both compact and full exchange tracking."""
    spec = _make_spec(
        n_steps=8,
        exchange_strategy='random',
        track_full_exchanges=track_full_exchanges,
    )
    python_state, _ = _run(spec, 'python')
    numba_state, backend = _run(spec, 'numba')
    assert backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


@pytest.mark.parametrize(
    ('block_size', 'n_steps', 'store_thin', 'de_thin', 'n_cold'),
    [
        (5, 15, 3, 2, 1),
        (6, 18, 4, 3, 2),
        (7, 21, 5, 4, 2),
    ],
)
def test_thinning_counters_and_multiple_cold_chains_are_bit_exact(
    block_size: int,
    n_steps: int,
    store_thin: int,
    de_thin: int,
    n_cold: int,
) -> None:
    """Odd/even non-divisor thinning values preserve cross-block counter state."""
    spec = _make_spec(
        n_steps=n_steps,
        block_size=block_size,
        n_cold=n_cold,
        store_thin=store_thin,
        proposal_overrides={'DEJumpManager': {'de_thin': de_thin}},
    )
    python_state, _ = _run(spec, 'python')
    numba_state, backend = _run(spec, 'numba')
    assert backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


@pytest.mark.parametrize(
    'jump_label',
    [
        'Std All-D',
        'Std Random-D',
        'DE Std All-D',
        'DE Std Random-D',
        'DE Big All-D',
        'DE Big Random-D',
        'Prior All-D',
        'Blank Jump',
    ],
)
def test_supported_proposals_are_bit_exact_to_python(jump_label: str) -> None:
    """Every non-Cholesky built-in proposal has no fixed-seed drift."""
    spec = _make_spec(n_steps=4)
    python_state, _ = _run(spec, 'python', jump_label=jump_label)
    numba_state, numba_backend = _run(spec, 'numba', jump_label=jump_label)

    assert numba_backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


def test_eggbox_custom_native_contract_is_bit_exact() -> None:
    """A likelihood can override generated prior and bounds operations."""
    spec = _make_spec('eggbox', n_par=3, n_steps=4)
    python_state, _ = _run(spec, 'python', jump_label='Prior All-D')
    numba_state, numba_backend = _run(spec, 'numba', jump_label='Prior All-D')

    assert numba_backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


@njit()
def _extension_loglike(params: NDArray[np.floating], center: float, _prior_rate: float) -> float:
    return -float(np.sum((params - center) ** 2))


@njit()
def _extension_prior_draw(
    n_par: int,
    low_lims: NDArray[np.floating],
    high_lims: NDArray[np.floating],
    state: tuple[float, float],
) -> NDArray[np.floating]:
    """Draw a separable truncated exponential by inverse CDF."""
    rate = state[1]
    draw = np.zeros(n_par)
    for itrp in range(n_par):
        mass = 1.0 - np.exp(-rate * (high_lims[itrp] - low_lims[itrp]))
        draw[itrp] = low_lims[itrp] - np.log(1.0 - np.random.uniform(0.0, 1.0) * mass) / rate
    return draw


@njit()
def _extension_prior_factor(params: NDArray[np.floating], state: tuple[float, float]) -> float:
    return -state[1] * float(np.sum(params))


@jittable_likelihood(
    _extension_loglike,
    state_attrs=('center', 'prior_rate'),
    prior_draw=_extension_prior_draw,
    prior_factor=_extension_prior_factor,
)
class _ExtensionLikelihood(RectangularLikelihood):
    """Small external-style likelihood with a non-uniform prior."""

    def __init__(self, n_par: int = 4, center: float = 0.75, prior_rate: float = 0.5) -> None:
        super().__init__(n_par, np.zeros(n_par), np.full(n_par, 3.0))
        self.center = center
        self.prior_rate = prior_rate

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        self.n_evals += 1
        return _extension_loglike(params_in, self.center, self.prior_rate)

    def prior_draw(self) -> NDArray[np.floating]:
        return _extension_prior_draw(self.n_par, self.low_lims, self.high_lims, (self.center, self.prior_rate))

    def prior_factor(self, params_in: NDArray[np.floating]) -> float:
        return _extension_prior_factor(params_in, (self.center, self.prior_rate))


class _CustomManagerState(NamedTuple):
    scale: float


@njit(inline='always')
def _custom_jump_helper(sample_point: NDArray[np.floating], scale: float) -> tuple[NDArray[np.floating], float, bool]:
    return sample_point + scale * np.random.normal(0.0, 1.0, sample_point.size), 0.0, True


@njit(inline='always')
def _custom_jump_native(
    sample_point: NDArray[np.floating],
    _itrt: int,
    state: _CustomManagerState,
    _likelihood: NativeLikelihoodState,
) -> tuple[NDArray[np.floating], float, bool]:
    return _custom_jump_helper(sample_point, state.scale)


@jittable_jump(_custom_jump_native)
class _CustomJump(AbstractJump):
    """External-style jump whose type is unknown to numba_backend.py."""

    def __init__(self, manager: _CustomManager) -> None:
        self.manager = manager
        self.print_name = 'Custom Gaussian'

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        del itrt
        return _custom_jump_helper(sample_point, self.manager.scale)


def _get_custom_manager_state(manager: Any) -> _CustomManagerState:
    return _CustomManagerState(manager.scale)


def _set_custom_manager_state(_manager: Any, _state: _CustomManagerState) -> None:
    """The custom scale is constant during each block."""


@njit(inline='always')
def _post_custom_manager_state(state: _CustomManagerState, _samples: NDArray[np.floating]) -> _CustomManagerState:
    return state


@jittable_jump_manager(
    state_getter=_get_custom_manager_state,
    state_setter=_set_custom_manager_state,
    post_step=_post_custom_manager_state,
)
class _CustomManager(JumpManager):
    """External-style manager registered without backend source changes."""

    def __init__(self, T_ladder, like_obj, scale: float = 0.2) -> None:
        self.scale = scale
        super().__init__(T_ladder, like_obj, [_CustomJump(self)])

    def set_jump_weights(self) -> None:
        self.jump_weights = np.ones((self.n_chain, self.n_jump_types))

    def record_config(self, config_in: configparser.ConfigParser) -> None:
        del config_in


class _PythonOnlyJump(AbstractJump):
    print_name = 'Python only'

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        del itrt
        return sample_point.copy(), 0.0, True


class _PythonOnlyManager(JumpManager):
    def __init__(self, T_ladder, like_obj) -> None:
        super().__init__(T_ladder, like_obj, [_PythonOnlyJump()])

    def set_jump_weights(self) -> None:
        self.jump_weights = np.ones((self.n_chain, self.n_jump_types))

    def record_config(self, config_in: configparser.ConfigParser) -> None:
        del config_in


class _PythonOnlyExchange(ExchangeManager):
    def is_exchange_step(self, itrb: int) -> bool:
        del itrb
        return False

    def do_ptmcmc_exchange(self, *args: Any, **kwargs: Any) -> None:
        del args
        del kwargs
        msg = 'unscheduled exchange should not execute'
        raise AssertionError(msg)


class _EveryThirdExchangeState(NamedTuple):
    track_full_exchanges: bool


@njit(inline='always')
def _every_third_exchange_schedule(itrb: int, _state: _EveryThirdExchangeState) -> bool:
    return itrb % 3 == 0


@njit(inline='always')
def _every_third_exchange_native(
    itrb: int,
    samples: NDArray[np.floating],
    logLs: NDArray[np.floating],
    n_chain: int,
    betas: NDArray[np.floating],
    exchange_tracker: NDArray[np.int64],
    esd_exchange: NDArray[np.floating],
    chain_track: NDArray[np.int64],
    state: _EveryThirdExchangeState,
) -> None:
    do_ptmcmc_exchange(
        itrb - 1,
        samples,
        logLs,
        n_chain,
        betas,
        exchange_tracker,
        esd_exchange,
        chain_track,
        NULL_TARGETS,
        state.track_full_exchanges,
    )


def _get_every_third_exchange_state(manager: Any) -> _EveryThirdExchangeState:
    return _EveryThirdExchangeState(manager.track_full_exchanges)


@jittable_exchange_manager(
    state_getter=_get_every_third_exchange_state,
    is_exchange_step=_every_third_exchange_schedule,
    exchange=_every_third_exchange_native,
)
class _EveryThirdExchange(ExchangeManager):
    def is_exchange_step(self, itrb: int) -> bool:
        return _every_third_exchange_schedule(itrb, _get_every_third_exchange_state(self))

    def do_ptmcmc_exchange(
        self,
        itrb: int,
        samples: NDArray[np.floating],
        logLs: NDArray[np.floating],
        T_ladder,
        exchange_tracker: NDArray[np.int64],
        esd_exchange: NDArray[np.floating],
        chain_track: NDArray[np.int64],
    ) -> None:
        _every_third_exchange_native(
            itrb,
            samples,
            logLs,
            T_ladder.n_chain,
            T_ladder.betas,
            exchange_tracker,
            esd_exchange,
            chain_track,
            _get_every_third_exchange_state(self),
        )


def _standalone_snapshot(sampler: DTMCMCSampler, like_obj: RectangularLikelihood) -> dict[str, object]:
    tracker = sampler.tracker_manager
    return {
        'samples': sampler.samples.copy(),
        'logLs': sampler.logLs.copy(),
        'chain_track': sampler.chain_track.copy(),
        'accept_record': tracker.accept_record.copy(),
        'esd_record': tracker.esd_record.copy(),
        'n_evals': like_obj.n_evals,
    }


class _UndecoratedGaussian(GaussianLikelihood):
    """An extension that must not inherit its parent's native contract."""

    def get_loglike(self, params_in: np.ndarray) -> float:
        self.n_evals += 1
        return float(-np.sum(np.abs(params_in)))


def _run_standalone_graph(
    backend: str, *, registered: bool, custom_exchange: bool = False
) -> tuple[dict[str, object], str]:
    reset_seed_guard_for_tests()
    try:
        seed_run(20260715)
        like_obj = GaussianLikelihood(2) if registered else _UndecoratedGaussian(2)
        ladder = GeometricTemperatureLadder(4, n_cold=1, T_max=20.0, n_inf_final=1)
        config = configparser.ConfigParser()
        config.read('default_config.ini')
        config['ProposalManager']['only_prior_hot'] = 'False'
        manager = _CustomManager(ladder, like_obj) if registered else _PythonOnlyManager(ladder, like_obj)
        exchange: ExchangeManager
        if custom_exchange:
            exchange = _EveryThirdExchange(NULL_TARGETS, False)
        elif registered:
            exchange = ExchangeManager(NULL_TARGETS, False)
        else:
            exchange = _PythonOnlyExchange(NULL_TARGETS, False)
        proposal = ProposalManager(ladder, like_obj, (manager,), exchange, config)
        sampler = DTMCMCSampler(
            ladder,
            like_obj,
            block_size=5,
            store_size=5,
            proposal_manager=proposal,
            starting_samples=np.zeros((ladder.n_chain, like_obj.n_par)),
            kernel_backend=backend,
        )
        sampler.advance_block()
        return _standalone_snapshot(sampler, like_obj), sampler.last_kernel_backend
    finally:
        reset_seed_guard_for_tests()


def _bad_native_loglike(params: NDArray[np.floating]) -> float:
    return params.this_attribute_does_not_exist()  # type: ignore[attr-defined,no-any-return]


@jittable_likelihood(_bad_native_loglike)
class _BadNativeLikelihood(RectangularLikelihood):
    """Valid Python likelihood with an intentionally invalid native function."""

    def __init__(self) -> None:
        super().__init__(4, np.full(4, -5.0), np.full(4, 5.0))

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        self.n_evals += 1
        return -float(np.sum(params_in * params_in))


@njit(inline='always')
def _many_state_loglike(params: NDArray[np.floating], a: float, b: float, c: float, d: float, e: float) -> float:
    return -float(np.sum(params * params)) + a + b + c + d + e


@jittable_likelihood(_many_state_loglike, state_attrs=('a', 'b', 'c', 'd', 'e'))
class _ManyStateLikelihood(RectangularLikelihood):
    """Regression likelihood with more than the former four-state limit."""

    def __init__(self) -> None:
        super().__init__(4, np.full(4, -5.0), np.full(4, 5.0))
        self.a, self.b, self.c, self.d, self.e = 1.0, 2.0, 3.0, 4.0, 5.0

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        self.n_evals += 1
        return _many_state_loglike(params_in, self.a, self.b, self.c, self.d, self.e)


def test_external_nonuniform_prior_contract_is_bit_exact() -> None:
    """The decorator contract supports extension likelihood prior weights."""
    spec = _make_spec(n_steps=4)
    factory = _ExtensionLikelihood
    python_state, _ = _run(spec, 'python', jump_label='Prior All-D', like_factory=factory)
    numba_state, numba_backend = _run(spec, 'numba', jump_label='Prior All-D', like_factory=factory)

    assert numba_backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


def test_external_jump_and_manager_registry_is_bit_exact() -> None:
    """A new manager and jump need no hard-coded backend dispatch changes."""
    python_state, python_backend = _run_standalone_graph('python', registered=True)
    numba_state, numba_backend = _run_standalone_graph('numba', registered=True)
    assert python_backend == 'python'
    assert numba_backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


def test_external_exchange_cadence_registry_is_bit_exact() -> None:
    """Generated control flow uses a registered scheduler rather than even/odd hard-coding."""
    python_state, _ = _run_standalone_graph('python', registered=True, custom_exchange=True)
    numba_state, backend = _run_standalone_graph('numba', registered=True, custom_exchange=True)
    assert backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


def test_fully_undecorated_auto_graph_falls_back_without_warning() -> None:
    """Auto remains quiet when no graph component opts into native execution."""
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter('always')
        _state, backend = _run_standalone_graph('auto', registered=False)
    assert backend == 'python'
    assert warning_records == []


@pytest.mark.parametrize('backend', ['auto', 'numba'])
def test_registered_native_compile_failure_is_loud(backend: str) -> None:
    """A broken decorated function never degrades silently to Python."""
    with pytest.raises(NativeBackendCompilationError, match='failed Numba compilation'):
        _run(_make_spec(n_steps=4), backend, like_factory=_BadNativeLikelihood)


def test_likelihood_adapter_has_no_fixed_state_arity_limit() -> None:
    """Generated likelihood adapters support extension state beyond four fields."""
    python_state, _ = _run(_make_spec(n_steps=4), 'python', like_factory=_ManyStateLikelihood)
    numba_state, backend = _run(_make_spec(n_steps=4), 'numba', like_factory=_ManyStateLikelihood)
    assert backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


def test_uniform_likelihood_gaussian_prior_contract_is_bit_exact() -> None:
    """The harness-visible non-uniform prior example preserves all serial state."""
    spec = _make_spec('uniform_gaussian_prior', n_par=3, n_steps=8)

    def factory() -> UniformGaussianPriorLikelihood:
        return UniformGaussianPriorLikelihood(n_par=3, prior_mean=1.5, prior_std=0.75)

    python_state, _ = _run(spec, 'python', jump_label='Prior All-D', like_factory=factory)
    numba_state, numba_backend = _run(spec, 'numba', jump_label='Prior All-D', like_factory=factory)

    assert numba_backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


@pytest.mark.parametrize(
    'likelihood_name',
    [
        'ar1',
        'banana',
        'cake',
        'gaussian',
        'gaussian_mixture',
        'gaussian_shell',
        'hyperpyramid',
        'random_wheel',
        'rosenbrock',
        'spoke_wheel',
        'uniform_gaussian_prior',
    ],
)
def test_builtin_likelihood_compiles_native(likelihood_name: str) -> None:
    """All supported built-ins can compile and execute the native block."""
    spec = _make_spec(likelihood_name, n_par=2, n_steps=4)
    _state, backend = _run(spec, 'numba')
    assert backend == 'numba'


def test_mixed_auto_graph_warns_once_and_falls_back() -> None:
    """Auto warns once when built-in native proposals meet an unregistered likelihood."""
    spec = _make_spec(n_steps=4)
    with pytest.warns(RuntimeWarning, match='mixed native/Python graph') as warning_records:
        _state, selected = _run(spec, 'auto', like_factory=lambda: _UndecoratedGaussian(n_par=4))
    assert selected == 'python'
    assert len(warning_records) == 1


def test_numba_backend_rejects_undecorated_likelihood() -> None:
    """Explicit Numba selection is strict about missing registrations."""
    spec = _make_spec(n_steps=4)
    with pytest.raises(NativeBackendUnsupportedError, match='likelihood _UndecoratedGaussian'):
        _run(spec, 'numba', like_factory=lambda: _UndecoratedGaussian(n_par=4))


def test_hawaii_likelihood_warns_in_auto_and_is_rejected_by_numba() -> None:
    """The scipy-interpolated Hawaii likelihood remains an intentional fallback."""
    with pytest.warns(RuntimeWarning, match='HawaiiLikelihood'):
        _state, selected = _run(_make_spec('hawaii', n_steps=4), 'auto')
    assert selected == 'python'
    with pytest.raises(NativeBackendUnsupportedError, match='HawaiiLikelihood'):
        _run(_make_spec('hawaii', n_steps=4), 'numba')


def test_active_full_fisher_jump_is_bit_exact() -> None:
    """The Cholesky Fisher proposal is part of the fully registered graph."""
    spec = _make_spec(
        n_par=2,
        n_steps=4,
        proposal_overrides={
            'FisherJumpManager': {
                'use_chol_fishers': True,
                'cold_fisher_weight': 1.0,
                'hot_fisher_weight': 1.0,
            },
        },
    )
    python_state, _ = _run(spec, 'python', jump_label='Fisher All-D')
    numba_state, backend = _run(spec, 'numba', jump_label='Fisher All-D')
    assert backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


def test_python_backend_disables_only_generated_kernel() -> None:
    """The debug switch leaves the ordinary sampler behavior available."""
    state, backend = _run(_make_spec(n_steps=4), 'python')
    assert backend == 'python'
    assert state['itrn'] == 4


def test_invalid_kernel_backend_is_rejected() -> None:
    spec = _make_spec(n_steps=4)
    reset_seed_guard_for_tests()
    try:
        seed_run(spec.seed)
        with pytest.raises(ValueError, match='kernel_backend'):
            build_sampler(spec, kernel_backend='cuda')
    finally:
        reset_seed_guard_for_tests()
