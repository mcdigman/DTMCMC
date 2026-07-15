"""Regression tests for the generated serial nopython block kernel."""

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from numba import njit
from numpy.typing import NDArray

from DTMCMC.de_manager import DEJumpManager
from DTMCMC.likelihood import RectangularLikelihood
from DTMCMC.likelihoods.normal_nd import GaussianLikelihood
from DTMCMC.numba_backend import jittable_likelihood
from DTMCMC.rng_helpers import reset_seed_guard_for_tests, seed_run
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
                'n_chain': 6,
                'n_cold': 1,
                'T_cold': 1.0,
                'T_min': 1.0,
                'T_max': 100.0,
                'n_inf_final': 1,
            },
            'run': {
                'n_steps': n_steps,
                'block_size': block_size,
                'store_thin': 1,
                'checkpoint_every_blocks': 1,
            },
            'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
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
    """Every proposal implemented by the first native milestone has no drift."""
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


def test_external_nonuniform_prior_contract_is_bit_exact() -> None:
    """The decorator contract supports extension likelihood prior weights."""
    spec = _make_spec(n_steps=4)
    factory = _ExtensionLikelihood
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
    ],
)
def test_builtin_likelihood_compiles_native(likelihood_name: str) -> None:
    """All supported built-ins can compile and execute the native block."""
    spec = _make_spec(likelihood_name, n_par=2, n_steps=4)
    _state, backend = _run(spec, 'numba')
    assert backend == 'numba'


class _UndecoratedGaussian(GaussianLikelihood):
    """An extension that must not inherit its parent's native contract."""

    def get_loglike(self, params_in: np.ndarray) -> float:
        self.n_evals += 1
        return float(-np.sum(np.abs(params_in)))


@pytest.mark.parametrize('backend', ['auto', 'numba'])
def test_undecorated_likelihood_subclass_falls_back_silently(backend: str) -> None:
    """Unsupported extension objects remain usable in either selecting mode."""
    spec = _make_spec(n_steps=4)
    _state, selected = _run(spec, backend, like_factory=lambda: _UndecoratedGaussian(n_par=4))
    assert selected == 'python'


def test_hawaii_likelihood_falls_back_silently() -> None:
    """The scipy-interpolated Hawaii likelihood remains on the Python path."""
    _state, selected = _run(_make_spec('hawaii', n_steps=4), 'numba')
    assert selected == 'python'


def test_active_full_fisher_jump_falls_back_silently() -> None:
    """An active unsupported jump selects the whole Python block path."""
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
    _state, backend = _run(spec, 'numba')
    assert backend == 'python'


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
