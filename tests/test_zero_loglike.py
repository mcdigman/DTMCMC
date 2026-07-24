"""Prior-recovery review mode and its constant-likelihood reference target.

Scientific sampler reviews rerun with the likelihood forced to a constant so
the sampler must reproduce the prior. These tests pin the contract: target
evaluations are zero in both kernel backends while the original likelihood
continues to drive proposal-internal calculations such as Fisher stencils.
"""

import tomllib
from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pytest

from DTMCMC.de_manager import DEJumpManager
from DTMCMC.eval_accounting import LoglikeCallSpy
from DTMCMC.fisher_manager import FisherJumpManager
from DTMCMC.likelihoods.banana import BananaLikelihood
from DTMCMC.likelihoods.constant_rectangular import ConstantRectangularLikelihood, high_lim, low_lim
from DTMCMC.rng_helpers import reset_seed_guard_for_tests, seed_run
from experiments.harness.artifact import read_attrs, validate
from experiments.harness.runner import build_sampler, run_from_spec
from experiments.harness.spec import RunSpec

N_PAR = 4
SEED = 20260720

# banana's rectangular bounds at N_PAR = 4, reused verbatim for the
# constant-likelihood reference run
_BANANA_BOUNDS = BananaLikelihood(N_PAR)

# Proposals whose construction and dispatch do not depend on likelihood
# values. Fisher-manager proposals are covered separately because zero mode
# deliberately retains the original likelihood for their stencils.
TARGET_EQUIVALENT_JUMP_LABELS = [
    'DE Std All-D',
    'DE Std Random-D',
    'DE Big All-D',
    'DE Big Random-D',
    'Prior All-D',
    'Blank Jump',
]

FISHER_MANAGER_JUMP_LABELS = ['Fisher All-D', 'Std All-D', 'Std Random-D']


def _make_spec(likelihood_table: dict[str, Any], *, zero_loglike: bool = False) -> RunSpec[Any]:
    return RunSpec.from_dict(
        {
            'name': 'zero_loglike_equivalence',
            'seed': SEED,
            'likelihood': likelihood_table,
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
                'n_steps': 8,
                'block_size': 4,
                'store_thin': 1,
                'checkpoint_every_blocks': 1,
                'zero_loglike': zero_loglike,
            },
            'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
            'proposals': {
                'FisherJumpManager': {'verbose_fisher': False, 'use_chol_fishers': True},
                'DEJumpManager': {'de_size': 16},
            },
        }
    )


def _snapshot(sampler: Any) -> dict[str, object]:
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
        'de_buffer': de_manager.de_buffer.copy(),
        'de_write': de_manager.itrde_write,
        'de_count': de_manager.itrde_count,
        'eval_accounting': asdict(sampler.eval_accounting),
        'itrn': sampler.itrn,
    }


def _run(
    likelihood_table: dict[str, Any],
    backend: str,
    jump_label: str | None,
    *,
    zero_loglike: bool,
    like_obj: ConstantRectangularLikelihood | None = None,
) -> tuple[dict[str, object], Any]:
    reset_seed_guard_for_tests()
    try:
        spec = _make_spec(likelihood_table, zero_loglike=zero_loglike)
        seed_run(spec.seed)
        sampler, _like_obj = build_sampler(spec, like_obj=like_obj, kernel_backend=backend)
        if jump_label is not None:
            jump_idx = sampler.proposal_manager.jump_labels.index(jump_label)
            sampler.proposal_manager.jump_probs.fill(0.0)
            sampler.proposal_manager.jump_probs[:, jump_idx] = 1.0
        for _ in range(spec.n_blocks):
            sampler.advance_block()
        assert sampler.last_kernel_backend == backend
        return _snapshot(sampler), sampler
    finally:
        reset_seed_guard_for_tests()


@pytest.mark.parametrize('backend', ['python', 'numba'])
@pytest.mark.parametrize('jump_label', TARGET_EQUIVALENT_JUMP_LABELS)
def test_zero_mode_matches_constant_target(backend: str, jump_label: str) -> None:
    """Zero mode equals the constant target when proposal internals are likelihood-independent.

    Same seed, bounds, ladder, and proposal settings: the banana run with
    its log likelihood zeroed must reproduce the constant-rectangular run
    exactly for these proposals — every sample, tracker, and DE-buffer entry.
    """
    constant_like = ConstantRectangularLikelihood(
        n_par=N_PAR,
        low_lims=_BANANA_BOUNDS.low_lims,
        high_lims=_BANANA_BOUNDS.high_lims,
    )
    constant_state, constant_sampler = _run(
        {'name': 'constant_rectangular', 'n_par': N_PAR},
        backend,
        jump_label,
        zero_loglike=False,
        like_obj=constant_like,
    )
    zeroed_state, zeroed_sampler = _run({'name': 'banana', 'n_par': N_PAR}, backend, jump_label, zero_loglike=True)

    assert isinstance(constant_sampler.like_obj, ConstantRectangularLikelihood)
    assert isinstance(zeroed_sampler.like_obj, BananaLikelihood)
    assert all(
        cast('Any', manager).like_obj is zeroed_sampler.like_obj for manager in zeroed_sampler.proposal_manager.managers
    )

    assert constant_state.keys() == zeroed_state.keys()
    for key, constant_value in constant_state.items():
        zeroed_value = zeroed_state[key]
        if isinstance(constant_value, np.ndarray):
            np.testing.assert_array_equal(zeroed_value, constant_value, err_msg=key)
        else:
            assert zeroed_value == constant_value, key

    assert zeroed_sampler.eval_accounting.proposal_targets > 0


@pytest.mark.parametrize('jump_label', FISHER_MANAGER_JUMP_LABELS)
def test_zero_mode_fisher_manager_jumps_match_python_and_numba(jump_label: str) -> None:
    """Forced Fisher and sigma jumps are bit-exact across zero-mode backends."""
    python_state, python_sampler = _run({'name': 'banana', 'n_par': N_PAR}, 'python', jump_label, zero_loglike=True)
    numba_state, numba_sampler = _run({'name': 'banana', 'n_par': N_PAR}, 'numba', jump_label, zero_loglike=True)

    assert python_state.keys() == numba_state.keys()
    for key, python_value in python_state.items():
        numba_value = numba_state[key]
        if isinstance(python_value, np.ndarray):
            np.testing.assert_array_equal(numba_value, python_value, err_msg=key)
        else:
            assert numba_value == python_value, key

    python_fisher = next(
        manager for manager in python_sampler.proposal_manager.managers if isinstance(manager, FisherJumpManager)
    )
    numba_fisher = next(
        manager for manager in numba_sampler.proposal_manager.managers if isinstance(manager, FisherJumpManager)
    )
    for field in ('sigma_diags', 'fishers', 'chol_fishers', 'sigma_scales', 'gamma_mults'):
        np.testing.assert_array_equal(getattr(numba_fisher, field), getattr(python_fisher, field), err_msg=field)

    assert python_sampler.zero_loglike
    assert numba_sampler.zero_loglike
    assert np.all(python_sampler.logLs == 0.0)
    assert np.all(numba_sampler.logLs == 0.0)


def test_constant_rectangular_defaults() -> None:
    """The constant likelihood defaults to the borrowed banana box and zero loglike."""
    like = ConstantRectangularLikelihood(n_par=3)
    np.testing.assert_array_equal(like.low_lims, np.full(3, low_lim))
    np.testing.assert_array_equal(like.high_lims, np.full(3, high_lim))
    assert like.get_loglike(np.zeros(3)) == 0.0
    assert like.prior_factor(np.zeros(3)) == 0.0
    assert like.loglike_fn(np.zeros(3), like.inputs) == 0.0


def test_zero_mode_preserves_original_fisher_likelihood() -> None:
    """Fisher construction is identical with and without zero mode."""
    reset_seed_guard_for_tests()
    try:
        zeroed_spec = _make_spec({'name': 'banana', 'n_par': N_PAR}, zero_loglike=True)
        seed_run(zeroed_spec.seed)
        zeroed_sampler, _ = build_sampler(zeroed_spec, kernel_backend='python')

        reset_seed_guard_for_tests()
        regular_spec = _make_spec({'name': 'banana', 'n_par': N_PAR})
        seed_run(regular_spec.seed)
        regular_sampler, _ = build_sampler(regular_spec, kernel_backend='python')
    finally:
        reset_seed_guard_for_tests()

    assert isinstance(zeroed_sampler.like_obj, BananaLikelihood)
    assert all(
        cast('Any', manager).like_obj is zeroed_sampler.like_obj for manager in zeroed_sampler.proposal_manager.managers
    )
    zeroed_fisher = next(
        manager for manager in zeroed_sampler.proposal_manager.managers if isinstance(manager, FisherJumpManager)
    )
    regular_fisher = next(
        manager for manager in regular_sampler.proposal_manager.managers if isinstance(manager, FisherJumpManager)
    )
    assert zeroed_fisher.like_obj is zeroed_sampler.like_obj
    np.testing.assert_array_equal(zeroed_sampler.starting_samples, regular_sampler.starting_samples)
    np.testing.assert_array_equal(zeroed_fisher.sigma_diags, regular_fisher.sigma_diags)
    np.testing.assert_array_equal(zeroed_fisher.fishers, regular_fisher.fishers)
    np.testing.assert_array_equal(zeroed_fisher.chol_fishers, regular_fisher.chol_fishers)
    assert np.all(zeroed_sampler.starting_logLs == 0.0)
    assert np.any(regular_sampler.starting_logLs != 0.0)


@pytest.mark.parametrize('backend', ['python', 'numba'])
def test_zero_mode_skips_only_sampler_target_calls(backend: str) -> None:
    """Target calls are skipped while Fisher's real-likelihood calls remain accounted."""
    reset_seed_guard_for_tests()
    try:
        spec = _make_spec({'name': 'banana', 'n_par': N_PAR}, zero_loglike=True)
        seed_run(spec.seed)
        like_obj = BananaLikelihood(N_PAR)
        with LoglikeCallSpy(like_obj) as spy:
            sampler, _ = build_sampler(spec, like_obj=like_obj, kernel_backend=backend)
            fisher = next(
                manager for manager in sampler.proposal_manager.managers if isinstance(manager, FisherJumpManager)
            )
            assert spy.n_calls == fisher.declared_construction_evals
            assert sampler.eval_accounting.initialization == spy.n_calls + sampler.n_chain
            jump_idx = sampler.proposal_manager.jump_labels.index('Blank Jump')
            sampler.proposal_manager.jump_probs.fill(0.0)
            sampler.proposal_manager.jump_probs[:, jump_idx] = 1.0
            if backend == 'python':
                sampler.advance_block()

        # The spy's Python counting closure is intentionally not a native
        # binding. Resolve and compile the strict native backend only after
        # the original baked handle has been restored.
        if backend == 'numba':
            sampler.advance_block()

        assert sampler.last_kernel_backend == backend
        if backend == 'python':
            assert spy.n_calls == 2 * fisher.declared_construction_evals
        else:
            assert spy.n_calls == fisher.declared_construction_evals
            assert sampler.eval_accounting.post_block == fisher.declared_construction_evals
        assert sampler.eval_accounting.proposal_targets > 0
        assert np.all(sampler.logLs == 0.0)
    finally:
        reset_seed_guard_for_tests()


def test_run_from_spec_persists_zero_mode_in_artifact(tmp_path: str) -> None:
    """The effective zero-mode setting survives run_from_spec as artifact provenance."""
    reset_seed_guard_for_tests()
    try:
        spec = _make_spec({'name': 'banana', 'n_par': N_PAR}, zero_loglike=True)
        artifact_path = run_from_spec(spec, tmp_path)
    finally:
        reset_seed_guard_for_tests()

    assert validate(artifact_path, mode='complete') == []
    attrs = read_attrs(artifact_path)
    embedded: RunSpec[Any] = RunSpec.from_dict(tomllib.loads(str(attrs['spec_toml'])))
    assert embedded.zero_loglike
