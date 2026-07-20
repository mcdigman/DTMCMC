"""Regression tests for the binding-driven serial nopython block kernel."""

import configparser
import typing
import warnings
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, override

import numpy as np
import pytest
from numba import njit
from numpy.typing import NDArray

from DTMCMC.auxilliary_manager import BlankJump
from DTMCMC.de_manager import DEJumpManager, DEStandardFullJump
from DTMCMC.dtmcmc_sampler import DTMCMCSampler
from DTMCMC.exchange_manager import (
    NULL_TARGETS,
    ExchangeManager,
    ExchangeNativeInputs,
    do_ptmcmc_exchange,
)
from DTMCMC.fisher_manager import FisherJumpManager, SigmaFullJump
from DTMCMC.jump_manager import AbstractJump, JumpManager
from DTMCMC.likelihood import (
    AbstractLikelihood,
    LoglikeFn,
    PriorDrawFn,
    PriorFactorFn,
    RectangularLikelihood,
)
from DTMCMC.likelihoods.normal_nd import GaussianLikelihood
from DTMCMC.likelihoods.uniform_gaussian_prior import UniformGaussianPriorLikelihood
from DTMCMC.numba_backend import (
    NativeBackendCompilationError,
    NativeBackendUnsupportedError,
    NativeExchangeFunctions,
    NativeJumpCall,
)
from DTMCMC.prior_manager import PriorFullJump
from DTMCMC.proposal_manager import ProposalManager
from DTMCMC.rng_helpers import reset_seed_guard_for_tests, seed_run
from DTMCMC.temperature_ladder_helpers import GeometricTemperatureLadder, TemperatureLadder
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
) -> RunSpec[Any]:
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


def _snapshot(sampler: DTMCMCSampler[Any]) -> dict[str, object]:
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
        'eval_accounting': asdict(sampler.eval_accounting),
        'store_idx': sampler.store_idx,
        'store_counter': sampler.store_counter,
        'itrn': sampler.itrn,
    }


def _run(
    spec: RunSpec[Any],
    backend: str,
    *,
    jump_label: str | None = None,
    like_factory: Callable[[], object] | None = None,
) -> tuple[dict[str, object], str]:
    reset_seed_guard_for_tests()
    try:
        seed_run(spec.seed)
        like_obj: Any = None if like_factory is None else like_factory()
        sampler, _like_obj = build_sampler(spec, like_obj=like_obj, kernel_backend=backend)
        if jump_label is not None:
            jump_idx = sampler.proposal_manager.jump_labels.index(jump_label)
            sampler.proposal_manager.jump_probs.fill(0.0)
            sampler.proposal_manager.jump_probs[:, jump_idx] = 1.0
        for _ in range(spec.n_blocks):
            sampler.advance_block()
        return _snapshot(sampler), sampler.last_kernel_backend
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
    """Bound dispatch delegates every built-in exchange strategy without drift."""
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


def test_eggbox_native_binding_is_bit_exact() -> None:
    """A likelihood with baked constants preserves prior-jump serial state."""
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
    rate: float,
) -> NDArray[np.floating]:
    """Draw a separable truncated exponential by inverse CDF."""
    draw = np.zeros(n_par)
    for itrp in range(n_par):
        mass = 1.0 - np.exp(-rate * (high_lims[itrp] - low_lims[itrp]))
        draw[itrp] = low_lims[itrp] - np.log(1.0 - np.random.uniform(0.0, 1.0) * mass) / rate
    return draw


@njit()
def _extension_prior_factor(params: NDArray[np.floating], rate: float) -> float:
    return -rate * float(np.sum(params))


class _ExtensionLikelihood(RectangularLikelihood):
    """Small external-style likelihood with a non-uniform prior.

    Bakes its per-instance constants through unmemoized closures, the
    simplest external extension pattern.
    """

    def __init__(self, n_par: int = 4, center: float = 0.75, prior_rate: float = 0.5) -> None:
        self.center = center
        self.prior_rate = prior_rate
        super().__init__(n_par, np.zeros(n_par), np.full(n_par, 3.0))

    @override
    def _make_loglike(self) -> LoglikeFn:
        center, prior_rate = self.center, self.prior_rate

        def loglike(params: NDArray[np.floating]) -> float:
            return _extension_loglike(params, center, prior_rate)

        return loglike

    @override
    def _make_prior_draw(self) -> PriorDrawFn:
        n_par, low_lims, high_lims, rate = self.n_par, self.low_lims, self.high_lims, self.prior_rate

        def prior_draw() -> NDArray[np.floating]:
            return _extension_prior_draw(n_par, low_lims, high_lims, rate)

        return prior_draw

    @override
    def _make_prior_factor(self) -> PriorFactorFn:
        rate = self.prior_rate

        def prior_factor(params: NDArray[np.floating]) -> float:
            return _extension_prior_factor(params, rate)

        return prior_factor


@njit(inline='always')
def _custom_jump_helper(sample_point: NDArray[np.floating], scale: float) -> tuple[NDArray[np.floating], float, bool]:
    return sample_point + scale * np.random.normal(0.0, 1.0, sample_point.size), 0.0, True


class _CustomJump[LikelihoodType: AbstractLikelihood](AbstractJump[LikelihoodType]):
    """External-style jump whose type is unknown to numba_backend.py.

    Binds a per-instance closure rather than a per-class function: allowed
    (the program cache just cannot share it across samplers).
    """

    declared_internal_evals = 0

    def __init__(self, manager: _CustomManager[LikelihoodType]) -> None:
        self.manager = manager
        self.print_name = 'Custom Gaussian'

    def bind_native(self) -> NativeJumpCall[None]:
        scale = self.manager.scale

        @njit(inline='always')
        def native_call(
            sample_point: NDArray[np.floating], _itrt: int, _inputs: None
        ) -> tuple[NDArray[np.floating], float, bool]:
            return _custom_jump_helper(sample_point, scale)

        return native_call

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        del itrt
        return _custom_jump_helper(sample_point, self.manager.scale)


class _CustomManager[LikelihoodType: AbstractLikelihood](JumpManager[LikelihoodType]):
    """External-style manager bound without backend source changes.

    Its post_step_update is the inherited base no-op, so no explicit native
    post-step binding is required.
    """

    def __init__(self, T_ladder: TemperatureLadder, like_obj: LikelihoodType, scale: float = 0.2) -> None:
        self.scale = scale
        super().__init__(T_ladder, like_obj, [_CustomJump(self)])

    @override
    def set_jump_weights(self) -> None:
        self.jump_weights = np.ones((self.n_chain, self.n_jump_types))

    @override
    def record_config(self, config_in: configparser.ConfigParser) -> None:
        del config_in


class _PythonOnlyJump[LikelihoodType: AbstractLikelihood](AbstractJump[LikelihoodType]):
    print_name = 'Python only'

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        del itrt
        return sample_point.copy(), 0.0, True


class _PythonOnlyManager[LikelihoodType: AbstractLikelihood](JumpManager[LikelihoodType]):
    def __init__(self, T_ladder: TemperatureLadder, like_obj: LikelihoodType) -> None:
        super().__init__(T_ladder, like_obj, [_PythonOnlyJump()])

    @override
    def set_jump_weights(self) -> None:
        self.jump_weights = np.ones((self.n_chain, self.n_jump_types))

    @override
    def record_config(self, config_in: configparser.ConfigParser) -> None:
        del config_in


class _PythonOnlyExchange(ExchangeManager):
    """Overrides the Python schedule without rebinding: must not run natively."""

    @override
    def is_exchange_step(self, itrb: int) -> bool:
        del itrb
        return False

    @override
    def do_ptmcmc_exchange(self, *args: Any, **kwargs: Any) -> None:
        del args
        del kwargs
        msg = 'unscheduled exchange should not execute'
        raise AssertionError(msg)


@njit(inline='always')
def _every_third_exchange_schedule(itrb: int) -> bool:
    return itrb % 3 == 0


@njit(inline='always')
def _every_third_schedule_native(itrb: int, _state: ExchangeNativeInputs) -> bool:
    return _every_third_exchange_schedule(itrb)


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
    state: ExchangeNativeInputs,
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


class _EveryThirdExchange(ExchangeManager):
    @override
    def is_exchange_step(self, itrb: int) -> bool:
        return _every_third_exchange_schedule(itrb)

    @override
    def do_ptmcmc_exchange(
        self,
        itrb: int,
        samples: NDArray[np.floating],
        logLs: NDArray[np.floating],
        T_ladder: TemperatureLadder,
        exchange_tracker: NDArray[np.int64],
        esd_exchange: NDArray[np.floating],
        chain_track: NDArray[np.int64],
    ) -> None:
        do_ptmcmc_exchange(
            itrb - 1,
            samples,
            logLs,
            T_ladder.n_chain,
            T_ladder.betas,
            exchange_tracker,
            esd_exchange,
            chain_track,
            NULL_TARGETS,
            self.inputs.track_full_exchanges,
        )

    @override
    def bind_native(self) -> NativeExchangeFunctions[ExchangeNativeInputs]:
        return NativeExchangeFunctions(
            is_exchange_step=_every_third_schedule_native, exchange=_every_third_exchange_native
        )


def _standalone_snapshot[LikelihoodType: AbstractLikelihood](
    sampler: DTMCMCSampler[LikelihoodType],
) -> dict[str, object]:
    tracker = sampler.tracker_manager
    return {
        'samples': sampler.samples.copy(),
        'logLs': sampler.logLs.copy(),
        'chain_track': sampler.chain_track.copy(),
        'accept_record': tracker.accept_record.copy(),
        'esd_record': tracker.esd_record.copy(),
        'eval_accounting': asdict(sampler.eval_accounting),
    }


class _UndecoratedGaussian(GaussianLikelihood):
    """An extension that must not inherit its parent's native binding."""

    @override
    def get_loglike(self, params_in: np.ndarray) -> float:
        return float(-np.sum(np.abs(params_in)))


@njit()
def _broken_native_jump_impl(sample_point: NDArray[np.floating]) -> float:
    return sample_point.this_attribute_does_not_exist()  # type: ignore[attr-defined,no-any-return]


class _BrokenNativeJump[LikelihoodType: AbstractLikelihood](AbstractJump[LikelihoodType]):
    """Jitted native binding that only fails once the kernel compiles it."""

    print_name = 'Broken native'
    declared_internal_evals = 0

    def bind_native(self) -> NativeJumpCall[None]:
        @njit(inline='always')
        def native_call(
            sample_point: NDArray[np.floating], _itrt: int, _state: None
        ) -> tuple[NDArray[np.floating], float, bool]:
            return sample_point + _broken_native_jump_impl(sample_point), 0.0, True

        return native_call

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        del itrt
        return sample_point.copy(), 0.0, True


class _BrokenNativeJumpManager[LikelihoodType: AbstractLikelihood](JumpManager[LikelihoodType]):
    def __init__(self, T_ladder: TemperatureLadder, like_obj: LikelihoodType) -> None:
        super().__init__(T_ladder, like_obj, [_BrokenNativeJump()])

    @override
    def set_jump_weights(self) -> None:
        self.jump_weights = np.ones((self.n_chain, self.n_jump_types))

    @override
    def record_config(self, config_in: configparser.ConfigParser) -> None:
        del config_in


def _run_standalone_graph(
    backend: str, *, bindable: bool, custom_exchange: bool = False, broken_jump: bool = False
) -> tuple[dict[str, object], str]:
    reset_seed_guard_for_tests()
    try:
        seed_run(20260715)
        like_obj = GaussianLikelihood(2) if bindable else _UndecoratedGaussian(2)
        ladder = GeometricTemperatureLadder(4, n_cold=1, T_max=20.0, n_inf_final=1)
        config = configparser.ConfigParser()
        config.read('default_config.ini')
        config['ProposalManager']['only_prior_hot'] = 'False'
        manager: JumpManager[Any]
        if broken_jump:
            manager = _BrokenNativeJumpManager(ladder, like_obj)
        elif bindable:
            manager = _CustomManager(ladder, like_obj)
        else:
            manager = _PythonOnlyManager(ladder, like_obj)
        exchange: ExchangeManager
        if custom_exchange:
            exchange = _EveryThirdExchange(NULL_TARGETS, False)
        elif bindable:
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
        return _standalone_snapshot(sampler), sampler.last_kernel_backend
    finally:
        reset_seed_guard_for_tests()


def _plain_python_norm(params: NDArray[np.floating]) -> float:
    return float(np.sum(np.abs(params)))


class _UncompilableLikelihood(RectangularLikelihood):
    """Provides a loglike factory whose function cannot compile to nopython.

    The function calls an untyped plain-Python global, so the eager
    construction-time compile fails; the plain closure still runs
    correctly on the Python path.
    """

    def __init__(self) -> None:
        super().__init__(4, np.full(4, -5.0), np.full(4, 5.0))

    @override
    def _make_loglike(self) -> LoglikeFn:
        def loglike(params: NDArray[np.floating]) -> float:
            return -_plain_python_norm(params)

        return loglike


@njit(inline='always')
def _many_input_loglike(params: NDArray[np.floating], a: float, b: float, c: float, d: float, e: float) -> float:
    return -float(np.sum(params * params)) + a + b + c + d + e


class _ManyInputLikelihood(RectangularLikelihood):
    """Regression likelihood whose handles bake many extension constants."""

    def __init__(self) -> None:
        self.a, self.b, self.c, self.d, self.e = 1.0, 2.0, 3.0, 4.0, 5.0
        super().__init__(4, np.full(4, -5.0), np.full(4, 5.0))

    @override
    def _make_loglike(self) -> LoglikeFn:
        a, b, c, d, e = self.a, self.b, self.c, self.d, self.e

        def loglike(params: NDArray[np.floating]) -> float:
            return _many_input_loglike(params, a, b, c, d, e)

        return loglike


def test_external_nonuniform_prior_contract_is_bit_exact() -> None:
    """The binding hooks support extension likelihood prior weights."""
    spec = _make_spec(n_steps=4)
    factory = _ExtensionLikelihood
    python_state, _ = _run(spec, 'python', jump_label='Prior All-D', like_factory=factory)
    numba_state, numba_backend = _run(spec, 'numba', jump_label='Prior All-D', like_factory=factory)

    assert numba_backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


def test_external_jump_and_manager_binding_is_bit_exact() -> None:
    """A new manager and jump need no backend dispatch changes."""
    python_state, python_backend = _run_standalone_graph('python', bindable=True)
    numba_state, numba_backend = _run_standalone_graph('numba', bindable=True)
    assert python_backend == 'python'
    assert numba_backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


def test_external_exchange_cadence_binding_is_bit_exact() -> None:
    """Bound control flow uses the bound scheduler rather than even/odd hard-coding."""
    python_state, _ = _run_standalone_graph('python', bindable=True, custom_exchange=True)
    numba_state, backend = _run_standalone_graph('numba', bindable=True, custom_exchange=True)
    assert backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


def test_fully_unbindable_auto_graph_falls_back_without_warning() -> None:
    """Auto remains quiet when no graph component opts into native execution."""
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter('always')
        _state, backend = _run_standalone_graph('auto', bindable=False)
    assert backend == 'python'
    assert warning_records == []


def test_native_compile_failure_raises_in_numba_mode() -> None:
    """A jitted binding that fails kernel compilation is loud when required."""
    with pytest.raises(NativeBackendCompilationError, match='failed Numba compilation'):
        _run_standalone_graph('numba', bindable=True, broken_jump=True)


def test_native_compile_failure_warns_once_and_falls_back_in_auto_mode() -> None:
    """Auto degrades to Python with a single warning when compilation fails."""
    with pytest.warns(RuntimeWarning, match='failed to compile') as warning_records:
        _state, selected = _run_standalone_graph('auto', bindable=True, broken_jump=True)
    assert selected == 'python'
    compile_warnings = [record for record in warning_records if 'failed to compile' in str(record.message)]
    assert len(compile_warnings) == 1


def test_uncompilable_likelihood_warns_at_construction_and_runs_python() -> None:
    """An intended-native function that cannot compile falls back at construction.

    The construction-time eager compile warns and keeps the plain
    function; the auto sampler then runs the graph on the Python path
    (with the usual one-time mixed-graph warning), and the strict numba
    backend rejects it naming the role.
    """
    with pytest.warns(RuntimeWarning, match='failed nopython compilation'):
        like = _UncompilableLikelihood()
    assert like.get_loglike(np.ones(4)) == -4.0

    with pytest.warns(RuntimeWarning) as warning_records:
        state, selected = _run(_make_spec(n_steps=4), 'auto', like_factory=_UncompilableLikelihood)
    assert selected == 'python'
    mixed_warnings = [record for record in warning_records if 'mixed native/Python' in str(record.message)]
    assert len(mixed_warnings) == 1
    assert state['itrn'] == 4

    with (
        pytest.warns(RuntimeWarning, match='failed nopython compilation'),
        pytest.raises(NativeBackendUnsupportedError, match='no nopython-compiled implementation of get_loglike'),
    ):
        _run(_make_spec(n_steps=4), 'numba', like_factory=_UncompilableLikelihood)


def test_native_binding_supports_many_input_fields() -> None:
    """State bundles support extension state well beyond the rectangular fields."""
    python_state, _ = _run(_make_spec(n_steps=4), 'python', like_factory=_ManyInputLikelihood)
    numba_state, backend = _run(_make_spec(n_steps=4), 'numba', like_factory=_ManyInputLikelihood)
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
        'constant_rectangular',
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
    """Auto warns once when built-in native proposals meet an unbindable likelihood."""
    spec = _make_spec(n_steps=4)
    with pytest.warns(RuntimeWarning, match='mixed native/Python graph') as warning_records:
        _state, selected = _run(spec, 'auto', like_factory=lambda: _UndecoratedGaussian(n_par=4))
    assert selected == 'python'
    assert len(warning_records) == 1


def test_numba_backend_rejects_stale_likelihood_override() -> None:
    """A subclass overriding get_loglike must not inherit its parent's binding."""
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
    """The Cholesky Fisher proposal is part of the fully bound graph."""
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


def test_rectangular_bounds_are_immutable() -> None:
    """PR #40 review F001: a bound graph cannot be diverged by rebinding bounds.

    The bounds are private read-only arrays behind properties, so both the
    rebind and the in-place write that produced silent native/Python
    divergence now fail immediately.
    """
    like = GaussianLikelihood(n_par=2)
    with pytest.raises(AttributeError):
        like.low_lims = np.full(2, -1.0)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        like.high_lims = np.full(2, 1.0)  # type: ignore[misc]
    with pytest.raises(ValueError, match='read-only'):
        like.low_lims[0] = 0.0


def test_structurally_identical_samplers_share_one_program() -> None:
    """PR #40 review F002: same-structure samplers reuse one compiled kernel.

    Equal-config likelihoods resolve to value-memoized handles, so the
    program cache key matches, a second sampler resolves to the same
    program, and Numba compiles no additional kernel signature.
    """
    spec = _make_spec(n_steps=4)
    reset_seed_guard_for_tests()
    try:
        seed_run(spec.seed)
        first, _ = build_sampler(spec, kernel_backend='numba')
        for _ in range(spec.n_blocks):
            first.advance_block()
        first_program = first._native_serial_backend.program
        assert first_program is not None
        kernel: Any = first_program.kernel  # numba Dispatcher: signatures is untyped
        n_signatures = len(kernel.signatures)

        second, _ = build_sampler(spec, kernel_backend='numba')
        second.advance_block()
        assert second.last_kernel_backend == 'numba'
        assert second._native_serial_backend.program is first_program
        assert len(kernel.signatures) == n_signatures
    finally:
        reset_seed_guard_for_tests()


class _IncompleteLikelihood:
    """Deliberately missing prior_draw/prior_factor/validate_bounds.

    Does not inherit any base: structural conformance is what the sampler
    checks, and nominal Protocol inheritance would satisfy isinstance
    vacuously while providing stub methods that return None.
    """

    @property
    def n_par(self) -> int:
        return 2

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        del params_in
        return 0.0


def test_incomplete_likelihood_fails_fast_with_conformance_error() -> None:
    """PR #40 review F003: core validation fires before the likelihood is used.

    The failure is the structural conformance TypeError naming the missing
    protocol, not a raw AttributeError from initialization touching a
    missing method.
    """
    ladder = GeometricTemperatureLadder(4, n_cold=1, T_max=20.0, n_inf_final=1)
    nonconforming = typing.cast('AbstractLikelihood', _IncompleteLikelihood())
    with pytest.raises(TypeError, match='n_par, get_loglike, prior_draw, prior_factor, validate_bounds'):
        DTMCMCSampler(ladder, nonconforming, 8, 8, kernel_backend='python')


class _FisherDeficientLikelihood:
    """Implements a Likelihood Without Required Fisher support methods."""

    @property
    def n_par(self) -> int:
        return 2

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        return -float(np.sum(params_in * params_in))

    def prior_draw(self) -> NDArray[np.floating]:
        return np.zeros(self.n_par)

    def prior_factor(self, params_in: NDArray[np.floating]) -> float:
        del params_in
        return 0.0

    def validate_bounds(self, params_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], bool]:
        return params_in, True


def test_fisher_manager_requires_fisher_support_likelihood() -> None:
    """PR #40 review F003: the Fisher manager validates its extra requirements."""
    ladder = GeometricTemperatureLadder(4, n_cold=1, T_max=20.0, n_inf_final=1)
    config = configparser.ConfigParser()
    config.read('default_config.ini')
    deficient = typing.cast('AbstractLikelihood', _FisherDeficientLikelihood())
    with pytest.raises(TypeError, match='correct_bounds and get_epsilons'):
        FisherJumpManager(ladder, deficient, np.zeros((4, 2)), config)


@pytest.mark.parametrize(
    'hook',
    [
        RectangularLikelihood._make_prior_draw,
        RectangularLikelihood._make_validate_bounds,
        GaussianLikelihood._make_loglike,
        SigmaFullJump.bind_native,
        DEStandardFullJump.bind_native,
        PriorFullJump.bind_native,
        BlankJump.bind_native,
        DEJumpManager.bind_native_post_step,
        DEJumpManager.native_state,
        ExchangeManager.bind_native,
        ExchangeManager.inputs.fget,
    ],
)
def test_native_binding_type_hints_resolve_at_runtime(hook: Any) -> None:
    """PR #40 review F005: extension hook annotations are runtime-introspectable."""
    hints = typing.get_type_hints(hook)
    assert 'return' in hints
