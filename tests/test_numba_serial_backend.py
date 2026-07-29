"""Regression tests for the binding-driven serial block programs (both flavors)."""

import configparser
import typing
import warnings
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any, NamedTuple, override

import numpy as np
import pytest
import scipy.interpolate
from numba import njit
from numba.core.errors import TypingError
from numpy.typing import NDArray

from DTMCMC.auxilliary_manager import BlankJump
from DTMCMC.de_manager import DEJumpManager, DEStandardFullJump
from DTMCMC.dtmcmc_sampler import DTMCMCSampler
from DTMCMC.exchange_manager import (
    NULL_TARGETS,
    ExchangeManager,
    ExchangeNativeInputs,
    NativeExchangeCall,
    NativeExchangeStepCall,
    do_ptmcmc_exchange,
)
from DTMCMC.fisher_manager import FisherJumpManager, SigmaFullJump
from DTMCMC.jump_manager import AbstractNativeJump, JumpManager
from DTMCMC.likelihood import (
    AbstractLikelihood,
    CompilationFallbackWarning,
    NativeBackendCompilationError,
    NativeBackendUnsupportedError,
    RectangularBoundsProtocol,
    RectangularLikelihood,
)
from DTMCMC.likelihoods.normal_nd import GaussianLikelihood
from DTMCMC.likelihoods.uniform_gaussian_prior import UniformGaussianPriorLikelihood
from DTMCMC.prior_manager import PriorFullJump
from DTMCMC.proposal_manager import ProposalManager
from DTMCMC.rng_helpers import reset_seed_guard_for_tests, seed_run
from DTMCMC.temperature_ladder_helpers import GeometricTemperatureLadder, TemperatureLadder
from experiments.harness.artifact import collect_provenance, read_attrs, validate, write_artifact
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


class _ExtensionNativeInput(NamedTuple):
    """External-style inputs."""

    n_par: int
    low_lims: NDArray[np.floating]
    high_lims: NDArray[np.floating]
    center: float
    prior_rate: float


@njit(inline='always')
def _extension_loglike_native(params: NDArray[np.floating], inputs: _ExtensionNativeInput) -> float:
    return _extension_loglike(params, inputs.center, inputs.prior_rate)


@njit(inline='always')
def _extension_prior_draw_native(inputs: _ExtensionNativeInput) -> NDArray[np.floating]:
    return _extension_prior_draw(inputs.n_par, inputs.low_lims, inputs.high_lims, inputs.prior_rate)


@njit(inline='always')
def _extension_prior_factor_native(params: NDArray[np.floating], inputs: _ExtensionNativeInput) -> float:
    return _extension_prior_factor(params, inputs.prior_rate)


class _ExtensionLikelihood(RectangularLikelihood[_ExtensionNativeInput]):
    """Small external-style likelihood with a non-uniform prior."""

    def __init__(self, n_par: int = 4, center: float = 0.75, prior_rate: float = 0.5) -> None:
        self.center = center
        self.prior_rate = prior_rate
        super().__init__(n_par, np.zeros(n_par), np.full(n_par, 3.0))

    @property
    @override
    def inputs(self) -> _ExtensionNativeInput:
        return _ExtensionNativeInput(self.n_par, self.low_lims, self.high_lims, self.center, self.prior_rate)

    @property
    @override
    def prior_draw_fn(self):
        return _extension_prior_draw_native

    @property
    @override
    def prior_factor_fn(self):
        return _extension_prior_factor_native

    @property
    @override
    def loglike_fn(self):
        return _extension_loglike_native


class _CustomNativeState(NamedTuple):
    scale: float


@njit(inline='always')
def _custom_jump_helper(sample_point: NDArray[np.floating], scale: float) -> tuple[NDArray[np.floating], float, bool]:
    return sample_point + scale * np.random.normal(0.0, 1.0, sample_point.size), 0.0, True


@njit(inline='always')
def _custom_jump_native(
    sample_point: NDArray[np.floating],
    _itrt: int,
    inputs: _CustomNativeState,
) -> tuple[NDArray[np.floating], float, bool]:
    return _custom_jump_helper(sample_point, inputs.scale)


class _CustomJump[LikelihoodType: AbstractLikelihood[_CustomNativeState]](
    AbstractNativeJump[LikelihoodType, _CustomNativeState]
):
    """External-style jump whose type is unknown to numba_backend.py.

    Binds a per-instance closure rather than a per-class function: allowed
    (the program cache just cannot share it across samplers).
    """

    def __init__(self, manager: _CustomManager[LikelihoodType]) -> None:
        print_name = 'Custom Gaussian'
        super().__init__(_custom_jump_native, manager, print_name)

    @property
    @override
    def declared_internal_evals(self) -> int:
        return 0


class _CustomManager[LikelihoodType: AbstractLikelihood[_CustomNativeState]](
    JumpManager[LikelihoodType, _CustomNativeState]
):
    """External-style manager bound without backend source changes.

    Its post_step_update is the inherited base no-op, so no explicit native
    post-step binding is required.
    """

    def __init__(self, T_ladder: TemperatureLadder, like_obj: LikelihoodType, scale: float = 0.2) -> None:
        self._state = _CustomNativeState(scale=scale)
        super().__init__(T_ladder, like_obj, [_CustomJump(self)])

    @override
    def set_jump_weights(self) -> None:
        self.jump_weights = np.ones((self.n_chain, self.n_jump_types))

    @override
    def record_config(self, config_in: configparser.ConfigParser) -> None:
        del config_in

    @property
    @override
    def native_state(self) -> _CustomNativeState:
        return self._state


@njit(inline='always')
def _every_third_exchange_schedule_native(itrb: int, _input: ExchangeNativeInputs) -> bool:
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
    inputs: ExchangeNativeInputs,
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
        inputs.track_full_exchanges,
    )


class _EveryThirdExchange(ExchangeManager):
    @property
    @override
    def is_exchange_step_native(self) -> NativeExchangeStepCall[ExchangeNativeInputs]:
        return _every_third_exchange_schedule_native

    @property
    @override
    def exchange_native(self) -> NativeExchangeCall[ExchangeNativeInputs]:
        return _every_third_exchange_native


def _standalone_snapshot[LikelihoodType: AbstractLikelihood[Any]](
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


def _run_standalone_graph(backend: str, *, custom_exchange: bool = False) -> tuple[dict[str, object], str]:
    reset_seed_guard_for_tests()
    try:
        seed_run(20260715)
        like_obj: Any = GaussianLikelihood(2)
        ladder = GeometricTemperatureLadder(4, n_cold=1, T_max=20.0, n_inf_final=1)
        config = configparser.ConfigParser()
        config.read('default_config.ini')
        config['ProposalManager']['only_prior_hot'] = 'False'
        manager = _CustomManager(ladder, like_obj)
        exchange: ExchangeManager
        if custom_exchange:
            exchange = _EveryThirdExchange(NULL_TARGETS, False)
        else:
            exchange = ExchangeManager(NULL_TARGETS, False)
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


def _bad_native_loglike(params: NDArray[np.floating], _inputs: Any) -> float:
    return float(
        np.linalg.norm(
            scipy.interpolate.InterpolatedUnivariateSpline(np.arange(-5, 5), np.arange(-5, 5) ** 2, k=3)(params)
        )
    )
    # return params.this_attribute_does_not_exist()  # type: ignore[attr-defined,no-any-return]


class _BadNativeLikelihood[InputType: RectangularBoundsProtocol](RectangularLikelihood[RectangularBoundsProtocol]):
    """Valid Python likelihood with an intentionally invalid native function."""

    def __init__(self) -> None:
        super().__init__(4, np.full(4, -5.0), np.full(4, 5.0))

    @property
    @override
    def loglike_fn(self):
        return _bad_native_loglike


@njit(inline='always')
def _many_input_loglike(params: NDArray[np.floating], a: float, b: float, c: float, d: float, e: float) -> float:
    return -float(np.sum(params * params)) + a + b + c + d + e


class _ManyInputNativeInput(NamedTuple):
    n_par: int
    low_lims: NDArray[np.floating]
    high_lims: NDArray[np.floating]
    a: float
    b: float
    c: float
    d: float
    e: float


@njit(inline='always')
def _many_input_loglike_native(params: NDArray[np.floating], inputs: _ManyInputNativeInput) -> float:
    return _many_input_loglike(params, inputs.a, inputs.b, inputs.c, inputs.d, inputs.e)


class _ManyInputLikelihood(RectangularLikelihood[_ManyInputNativeInput]):
    """Regression likelihood whose input carries many extension fields."""

    def __init__(self) -> None:
        self.a, self.b, self.c, self.d, self.e = 1.0, 2.0, 3.0, 4.0, 5.0
        n_par = 4
        low_lims = np.full(n_par, -5.0)
        low_lims.setflags(write=False)
        high_lims = np.full(n_par, 5.0)
        high_lims.setflags(write=False)
        self._inputs = _ManyInputNativeInput(n_par, low_lims, high_lims, self.a, self.b, self.c, self.d, self.e)
        super().__init__(n_par, low_lims, high_lims)

    @property
    @override
    def loglike_fn(self):
        return _many_input_loglike_native

    @property
    @override
    def inputs(self) -> _ManyInputNativeInput:
        return self._inputs


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
    python_state, python_backend = _run_standalone_graph('python')
    numba_state, numba_backend = _run_standalone_graph('numba')
    assert python_backend == 'python'
    assert numba_backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


def test_external_exchange_cadence_binding_is_bit_exact() -> None:
    """Bound control flow uses the bound scheduler rather than even/odd hard-coding."""
    python_state, _ = _run_standalone_graph('python', custom_exchange=True)
    numba_state, backend = _run_standalone_graph('numba', custom_exchange=True)
    assert backend == 'numba'
    _assert_snapshots_equal(python_state, numba_state)


def test_native_compile_failure_warns_during_eager_likelihood_compilation() -> None:
    """A bad handle is diagnosed once while the shared public handle is built."""
    with pytest.warns(CompilationFallbackWarning, match='_BadNativeLikelihood get_loglike failed') as warning_records:
        like_obj: Any = _BadNativeLikelihood()
    assert len(warning_records) == 1

    # The fallback is a normal public likelihood handle, so the forced Python
    # orchestrator can run without discovering a second compilation failure.
    with warnings.catch_warnings():
        warnings.simplefilter('error', CompilationFallbackWarning)
        state, selected = _run(_make_spec(n_steps=4), 'python', like_factory=lambda: like_obj)
    assert selected == 'python'
    assert state['itrn'] == 4


def test_native_compile_failure_raises_in_numba_mode() -> None:
    """A broken native binding is loud when the native backend is required."""
    with pytest.raises(NativeBackendCompilationError, match='failed Numba compilation'):
        _run(_make_spec(n_steps=4), 'numba', like_factory=_BadNativeLikelihood)


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


def test_hawaii_likelihood_warns_in_auto_and_is_rejected_by_numba() -> None:
    """The scipy-interpolated Hawaii likelihood remains an intentional fallback."""
    with pytest.warns(RuntimeWarning, match='HawaiiLikelihood'):
        _state, selected = _run(_make_spec('hawaii', n_steps=4), 'auto')
    assert selected == 'python'
    with pytest.raises(NativeBackendCompilationError, match='HawaiiLikelihood'):
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
    inputs = like.inputs
    assert inputs.low_lims is like.low_lims
    assert inputs.high_lims is like.high_lims


def test_structurally_identical_samplers_share_one_program() -> None:
    """PR #40 review F002: same-structure samplers reuse one compiled kernel.

    The program cache keys on the bound per-class functions rather than
    object identities, so a second sampler resolves to the same program and
    Numba compiles no additional kernel signature.
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
    """Deliberately missing prior_draw/prior_factor/validate_bounds."""

    def __init__(self) -> None:
        pass

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
    with pytest.raises(TypeError, match='n_par, get_loglike, prior_draw, prior_factor, validate_bounds'):
        DTMCMCSampler(ladder, _IncompleteLikelihood(), 8, 8, kernel_backend='python')  # type: ignore[type-var] # pyrefly: ignore[bad-specialization]


class _FisherDeficientLikelihood:
    """Implements a Likelihood Without Required Fisher support methods."""

    def __init__(self) -> None:
        pass

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
    with pytest.raises(TypeError, match='correct_bounds and get_epsilons'):
        FisherJumpManager(ladder, _FisherDeficientLikelihood(), np.zeros((4, 2)), config)  # type: ignore[type-var] # pyrefly: ignore[bad-specialization]


def _build_standalone_sampler(
    like_obj: Any,
    *,
    manager_cls: Callable[[TemperatureLadder, Any], JumpManager[Any, Any]] = _CustomManager,
    exchange: ExchangeManager | None = None,
    kernel_backend: str = 'python',
) -> DTMCMCSampler[Any]:
    """Assemble a one-manager sampler around externally supplied components."""
    ladder = GeometricTemperatureLadder(4, n_cold=1, T_max=20.0, n_inf_final=1)
    config = configparser.ConfigParser()
    config.read('default_config.ini')
    config['ProposalManager']['only_prior_hot'] = 'False'
    manager = manager_cls(ladder, like_obj)
    if exchange is None:
        exchange = ExchangeManager(NULL_TARGETS, False)
    proposal = ProposalManager(ladder, like_obj, (manager,), exchange, config)
    return DTMCMCSampler(
        ladder,
        like_obj,
        block_size=5,
        store_size=5,
        proposal_manager=proposal,
        starting_samples=np.zeros((ladder.n_chain, like_obj.n_par)),
        kernel_backend=kernel_backend,
    )


class _StaleLoglikeLikelihood(GaussianLikelihood):
    """Overrides the Python contract below the loglike_fn hook definition."""

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:  # type: ignore[misc]
        return -float(np.sum(params_in))


class _StalePostStepManager(_CustomManager[Any]):
    """Adds per-step behavior as a method only, with no post-step binding."""

    def post_step_update(self, samples: NDArray[np.floating]) -> None:  # type: ignore[misc]
        del samples


class _StaleCadenceExchange(ExchangeManager):
    """Overrides the exchange cadence below the native schedule hook."""

    def is_exchange_step(self, itrb: int) -> bool:  # type: ignore[misc]
        return itrb % 3 == 0


@pytest.mark.parametrize('backend', ['python', 'auto', 'numba'])
def test_stale_loglike_override_fails_fast_in_every_mode(backend: str) -> None:
    """A method override that outruns its binding is a construction error.

    Both program flavors execute the bound functions, so routing such a
    graph to the Python program would no longer honor the override; the
    divergence is rejected at construction in every kernel backend mode.
    """
    reset_seed_guard_for_tests()
    try:
        seed_run(20260717)
        with pytest.raises(TypeError, match='overrides get_loglike without overriding'):
            _build_standalone_sampler(_StaleLoglikeLikelihood(2), kernel_backend=backend)
    finally:
        reset_seed_guard_for_tests()


def test_stale_post_step_override_fails_fast() -> None:
    """Per-step behavior must arrive as a native binding, not a method override."""
    reset_seed_guard_for_tests()
    try:
        seed_run(20260717)
        with pytest.raises(TypeError, match='overrides post_step_update without overriding bind_native_post_step'):
            _build_standalone_sampler(GaussianLikelihood(2), manager_cls=_StalePostStepManager)
    finally:
        reset_seed_guard_for_tests()


def test_stale_exchange_cadence_override_fails_fast() -> None:
    """The exchange schedule must arrive as a native binding, not a method override."""
    reset_seed_guard_for_tests()
    try:
        seed_run(20260717)
        with pytest.raises(TypeError, match='overrides is_exchange_step without overriding'):
            _build_standalone_sampler(GaussianLikelihood(2), exchange=_StaleCadenceExchange(NULL_TARGETS, False))
    finally:
        reset_seed_guard_for_tests()


def test_flattening_violation_rejected_in_python_mode() -> None:
    """The unified assembly needs the flattening invariant in every mode.

    Both flavors dispatch per manager against the aggregate's flattened
    jump_probs columns, so a reordered aggregate jump list is rejected
    rather than silently misaligning probabilities with proposals.
    """
    reset_seed_guard_for_tests()
    try:
        seed_run(20260718)
        spec = _make_spec(n_steps=4)
        sampler, _like_obj = build_sampler(spec, kernel_backend='python')
        jumps = typing.cast('list[Any]', sampler.proposal_manager.jumps)
        jumps.append(jumps.pop(0))
        with pytest.raises(NativeBackendUnsupportedError, match='ordered identity-preserving flattening'):
            sampler.advance_block()
    finally:
        reset_seed_guard_for_tests()


_UNTYPABLE_GLOBAL = object()


@njit()
def _uncompilable_kernel(
    _samples: NDArray[np.floating],
    _logLs: NDArray[np.floating],
    _chain_track: NDArray[np.int64],
    _betas: NDArray[np.floating],
    _jump_probs: NDArray[np.floating],
    _exchange_tracker: NDArray[np.int64],
    _esd_exchange: NDArray[np.floating],
    _accept_record: NDArray[np.int64],
    _esd_record: NDArray[np.floating],
    _manager_states: tuple[object, ...],
    _exchange_inputs: object,
    _jump_internal_evals: NDArray[np.int64],
    _zero_loglike: bool,
) -> tuple[int, int]:
    """Kernel stand-in whose nopython compilation always fails."""
    # numba cannot type a bare object() global: TypingError at compile
    return _UNTYPABLE_GLOBAL  # type: ignore[return-value]


def test_auto_kernel_failure_backstop_is_bit_exact_with_python() -> None:
    """A NumbaError while compiling the kernel falls back losslessly in auto.

    The kernel is compiled for the exact runtime signature before any
    execution, so the Python program reruns the block from untouched
    arrays: the whole run stays bit-exact with a pure python-mode run of
    the same seed.
    """
    spec = _make_spec(n_steps=12, block_size=4)
    python_state, _ = _run(spec, 'python')
    reset_seed_guard_for_tests()
    try:
        seed_run(spec.seed)
        sampler, _like_obj = build_sampler(spec, kernel_backend='auto')
        sampler.advance_block()
        flavor_used: str = sampler.last_kernel_backend
        assert flavor_used == 'numba'
        backend = sampler._native_serial_backend
        program = backend.program
        assert program is not None
        assert program.flavor == 'numba'

        backend.program = replace(program, kernel=_uncompilable_kernel)
        backend._kernel_ready = False
        with pytest.warns(RuntimeWarning, match='failed to compile'):
            sampler.advance_block()
        flavor_used = sampler.last_kernel_backend
        assert flavor_used == 'python'
        sampler.advance_block()
        flavor_used = sampler.last_kernel_backend
        assert flavor_used == 'python'
        auto_state = _snapshot(sampler)
    finally:
        reset_seed_guard_for_tests()
    _assert_snapshots_equal(python_state, auto_state)


class _ReplayProbeState(NamedTuple):
    scale: float
    counter: NDArray[np.int64]


@njit()
def _replay_probe_post_step(state: _ReplayProbeState, _samples_row: NDArray[np.floating]) -> None:
    """Mutate runtime state, then raise a NumbaError subclass mid-block."""
    state.counter[0] += 1
    if state.counter[0] == 4:
        msg = 'runtime failure after partial block execution'
        raise TypingError(msg)


class _ReplayProbeManager(_CustomManager[Any]):
    """Manager whose compiled post-step fails at runtime after tracker writes."""

    def __init__(self, T_ladder: TemperatureLadder, like_obj: Any, scale: float = 0.2) -> None:
        self._replay_state = _ReplayProbeState(scale=scale, counter=np.zeros(1, dtype=np.int64))
        super().__init__(T_ladder, like_obj, scale)

    @property
    @override
    def native_state(self) -> _ReplayProbeState:  # type: ignore[override]
        return self._replay_state

    @property
    @override
    def bind_native_post_step(self) -> Any:
        return _replay_probe_post_step


def test_runtime_numba_error_is_never_replayed() -> None:
    """P1 review fix: a runtime NumbaError must propagate, not trigger replay.

    Only a failure of the pre-execution compilation step may engage the
    auto fallback; an error raised while the compiled kernel executes
    (after RNG draws and tracker writes) propagates instead of silently
    rerunning the block and double-counting its side effects.
    """
    reset_seed_guard_for_tests()
    try:
        seed_run(20260719)
        sampler = _build_standalone_sampler(
            GaussianLikelihood(2), manager_cls=_ReplayProbeManager, kernel_backend='auto'
        )
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            with pytest.raises(TypingError, match='after partial block execution'):
                sampler.advance_block()
        # the jump steps completed before the failing post-step (itrb 1 and 3
        # of the 5-step block; even steps are exchanges) stay single-counted
        assert int(sampler.tracker_manager.accept_record[:2].sum()) == 8
    finally:
        reset_seed_guard_for_tests()


def _uncompiled_jump_call(
    sample_point: NDArray[np.floating], _itrt: int, state: _CustomNativeState
) -> tuple[NDArray[np.floating], float, bool]:
    """Jump handle deliberately left as plain Python."""
    return sample_point + state.scale, 0.0, True


class _UncompiledJump[LikelihoodType: AbstractLikelihood[Any]](AbstractNativeJump[LikelihoodType, _CustomNativeState]):
    """Jump whose handle is uncompiled: a jump-level capability gap."""

    def __init__(self, manager: _UncompiledJumpManager[LikelihoodType]) -> None:
        super().__init__(_uncompiled_jump_call, manager, 'Uncompiled Jump')

    @property
    @override
    def declared_internal_evals(self) -> int:
        return 0


class _UncompiledJumpManager[LikelihoodType: AbstractLikelihood[Any]](JumpManager[LikelihoodType, _CustomNativeState]):
    """Manager whose only jump handle is uncompiled; everything else compiles."""

    def __init__(self, T_ladder: TemperatureLadder, like_obj: LikelihoodType, scale: float = 0.2) -> None:
        self._state = _CustomNativeState(scale=scale)
        super().__init__(T_ladder, like_obj, [_UncompiledJump(self)])

    @override
    def set_jump_weights(self) -> None:
        self.jump_weights = np.ones((self.n_chain, self.n_jump_types))

    @override
    def record_config(self, config_in: configparser.ConfigParser) -> None:
        del config_in

    @property
    @override
    def native_state(self) -> _CustomNativeState:
        return self._state


class _UntypableState(NamedTuple):
    scale: float
    blob: object


def _untypable_state_jump_call(
    sample_point: NDArray[np.floating], _itrt: int, state: _UntypableState
) -> tuple[NDArray[np.floating], float, bool]:
    return sample_point + state.scale, 0.0, True


def _untypable_state_post_step(_state: _UntypableState, _samples_row: NDArray[np.floating]) -> None:
    return


class _UntypableStateJump[LikelihoodType: AbstractLikelihood[Any]](AbstractNativeJump[LikelihoodType, _UntypableState]):
    def __init__(self, manager: _UntypableStateManager[LikelihoodType]) -> None:
        super().__init__(_untypable_state_jump_call, manager, 'Untypable State Jump')

    @property
    @override
    def declared_internal_evals(self) -> int:
        return 0


class _UntypableStateManager[LikelihoodType: AbstractLikelihood[Any]](JumpManager[LikelihoodType, _UntypableState]):
    """Manager whose runtime state bundle numba cannot type.

    Its handles are plain Python so the fallback program can still run the
    graph; the state itself is the targeted capability gap.
    """

    def __init__(self, T_ladder: TemperatureLadder, like_obj: LikelihoodType, scale: float = 0.2) -> None:
        self._state = _UntypableState(scale=scale, blob=object())
        super().__init__(T_ladder, like_obj, [_UntypableStateJump(self)])

    @override
    def set_jump_weights(self) -> None:
        self.jump_weights = np.ones((self.n_chain, self.n_jump_types))

    @override
    def record_config(self, config_in: configparser.ConfigParser) -> None:
        del config_in

    @property
    @override
    def native_state(self) -> _UntypableState:
        return self._state

    @property
    @override
    def bind_native_post_step(self) -> Any:
        return _untypable_state_post_step


def _uncompiled_exchange_step(itrb: int, _inputs: ExchangeNativeInputs) -> bool:
    return itrb % 2 == 0


def _uncompiled_exchange(
    itrb: int,
    samples: NDArray[np.floating],
    logLs: NDArray[np.floating],
    n_chain: int,
    betas: NDArray[np.floating],
    exchange_tracker: NDArray[np.int64],
    esd_exchange: NDArray[np.floating],
    chain_track: NDArray[np.int64],
    inputs: ExchangeNativeInputs,
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
        inputs.track_full_exchanges,
    )


class _UncompiledExchange(ExchangeManager):
    """Exchange manager whose native bindings are plain Python."""

    @property
    @override
    def is_exchange_step_native(self) -> NativeExchangeStepCall[ExchangeNativeInputs]:
        return _uncompiled_exchange_step

    @property
    @override
    def exchange_native(self) -> NativeExchangeCall[ExchangeNativeInputs]:
        return _uncompiled_exchange


@pytest.mark.parametrize(
    ('manager_cls', 'use_uncompiled_exchange', 'gap_match'),
    [
        (_UncompiledJumpManager, False, '_UncompiledJump handle'),
        (_UntypableStateManager, False, 'native state is not numba-typable'),
        (_CustomManager, True, 'is_exchange_step handle'),
    ],
    ids=['jump-handle', 'manager-state', 'exchange-binding'],
)
def test_component_capability_gaps_fall_back_in_auto_and_raise_in_numba(
    manager_cls: Any, use_uncompiled_exchange: bool, gap_match: str
) -> None:
    """Jump-, manager-state-, and exchange-level gaps drive the backend policy.

    Capability detection is component-scoped, not likelihood-scoped: each
    gap makes auto mode warn and run the Python program on the partially
    compiled graph, and makes strict numba mode raise.
    """
    reset_seed_guard_for_tests()
    try:
        seed_run(20260728)
        exchange = _UncompiledExchange(NULL_TARGETS, False) if use_uncompiled_exchange else None
        sampler = _build_standalone_sampler(
            GaussianLikelihood(2), manager_cls=manager_cls, exchange=exchange, kernel_backend='auto'
        )
        with pytest.warns(RuntimeWarning, match=gap_match):
            sampler.advance_block()
        assert sampler.last_kernel_backend == 'python'
        assert sampler.itrn == 5

        reset_seed_guard_for_tests()
        seed_run(20260728)
        exchange = _UncompiledExchange(NULL_TARGETS, False) if use_uncompiled_exchange else None
        strict_sampler = _build_standalone_sampler(
            GaussianLikelihood(2), manager_cls=manager_cls, exchange=exchange, kernel_backend='numba'
        )
        with pytest.raises(NativeBackendCompilationError, match=gap_match):
            strict_sampler.advance_block()
    finally:
        reset_seed_guard_for_tests()


def test_auto_partial_graph_warning_is_memoized_per_graph() -> None:
    """The auto fallback warns exactly once per graph identity.

    Later blocks on the unchanged graph resolve silently through the
    identity short-circuit, and a forced re-resolve of the same identity
    is silenced by the warned-identities memo.
    """
    reset_seed_guard_for_tests()
    try:
        seed_run(20260729)
        sampler = _build_standalone_sampler(
            GaussianLikelihood(2), manager_cls=_UncompiledJumpManager, kernel_backend='auto'
        )
        with pytest.warns(RuntimeWarning, match='partially compiled graph') as warning_records:
            sampler.advance_block()
        assert sum('partially compiled graph' in str(record.message) for record in warning_records) == 1

        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            sampler.advance_block()
            sampler._native_serial_backend.graph_identity = None
            sampler.advance_block()
        assert sampler.last_kernel_backend == 'python'
    finally:
        reset_seed_guard_for_tests()


def test_auto_fallback_artifact_distinguishes_requested_and_executed_backend(tmp_path: Any) -> None:
    """Artifact provenance distinguishes an auto request from Python fallback."""
    reset_seed_guard_for_tests()
    try:
        spec = _make_spec(n_par=2, n_steps=5, block_size=5, n_chain=4).with_kernel_backend('auto')
        child_seed_python, child_seed_numba = seed_run(spec.seed)
        sampler = _build_standalone_sampler(
            GaussianLikelihood(2), manager_cls=_UncompiledJumpManager, kernel_backend=spec.kernel_backend
        )
        with pytest.warns(RuntimeWarning, match='partially compiled graph'):
            sampler.advance_block()
        assert sampler.last_kernel_backend == 'python'

        provenance = collect_provenance(
            spec.seed,
            child_seed_python,
            child_seed_numba,
            spec_toml=spec.to_toml_text(),
            proposal_config_ini=spec.resolved_config_text(),
        )
        artifact_path = tmp_path / 'auto_fallback.h5'
        write_artifact(
            artifact_path,
            spec,
            sampler,
            sampler.eval_accounting,
            provenance,
            finalized=True,
            wall_seconds=0.0,
        )

        assert validate(artifact_path, mode='complete') == []
        attrs = read_attrs(artifact_path)
        assert str(attrs['spec_toml']) == spec.to_toml_text()
        assert spec.kernel_backend == 'auto'
        assert str(attrs['kernel_backend_executed']) == 'python'
    finally:
        reset_seed_guard_for_tests()


def test_explicit_starting_samples_initialize_state() -> None:
    """Supplied starting_samples seed samples[0] and the starting logLs exactly.

    starting_samples=None draws fresh points from the prior instead, so an
    explicit input must be adopted verbatim: the stored starting array, the
    block's row 0, and the per-chain starting logLs all reflect the
    supplied points.
    """
    reset_seed_guard_for_tests()
    try:
        seed_run(20260730)
        like_obj: Any = GaussianLikelihood(2)
        ladder = GeometricTemperatureLadder(4, n_cold=1, T_max=20.0, n_inf_final=1)
        config = configparser.ConfigParser()
        config.read('default_config.ini')
        config['ProposalManager']['only_prior_hot'] = 'False'
        manager = _CustomManager(ladder, like_obj)
        proposal = ProposalManager(ladder, like_obj, (manager,), ExchangeManager(NULL_TARGETS, False), config)
        starting = 0.125 * np.arange(ladder.n_chain * like_obj.n_par, dtype=np.float64).reshape(
            ladder.n_chain, like_obj.n_par
        )
        sampler = DTMCMCSampler(
            ladder,
            like_obj,
            block_size=5,
            store_size=5,
            proposal_manager=proposal,
            starting_samples=starting.copy(),
            kernel_backend='python',
        )
        np.testing.assert_array_equal(sampler.starting_samples, starting)
        np.testing.assert_array_equal(sampler.samples[0], starting)
        for itrt in range(ladder.n_chain):
            assert sampler.logLs[0, itrt] == like_obj.get_loglike(starting[itrt])
    finally:
        reset_seed_guard_for_tests()


@pytest.mark.parametrize(
    'hook',
    [
        RectangularLikelihood.bind_native.fget,  # type: ignore[attr-defined]
        RectangularLikelihood.loglike_fn.fget,  # type: ignore[attr-defined]
        RectangularLikelihood.inputs.fget,  # type: ignore[attr-defined]
        GaussianLikelihood.loglike_fn.fget,  # type: ignore[attr-defined]
        SigmaFullJump.__call__,
        DEStandardFullJump.__call__,
        PriorFullJump.__call__,
        BlankJump.__call__,
        DEJumpManager.bind_native_post_step.fget,  # type: ignore[attr-defined]
        DEJumpManager.native_state.fget,  # type: ignore[attr-defined]
        ExchangeManager.bind_native.fget,  # type: ignore[attr-defined]
        ExchangeManager.inputs.fget,  # type: ignore[attr-defined]
    ],
)
def test_native_binding_type_hints_resolve_at_runtime(hook: Any) -> None:
    """PR #40 review F005: extension hook annotations are runtime-introspectable."""
    hints = typing.get_type_hints(hook)
    assert 'return' in hints
