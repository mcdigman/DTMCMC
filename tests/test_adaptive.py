"""Phase 5 acceptance tests: ladder-update hook and adaptive controller.

Criterion 1: post-freeze equivalence is bit-exact — the frozen sampler's
full state copied into a fresh fixed-ladder sampler, both streams
reseeded identically, one block advanced, identical output. Criterion 2:
the adaptive entropy ladder converges to the gold ladder's dS profile on
cake 5D without human input. Criterion 3: the golden test stays green
(separate file, unchanged).
"""

from typing import Any

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import experiments.adaptive as adaptive_module
from DTMCMC.de_manager import DEJumpManager
from DTMCMC.fisher_manager import FisherJumpManager
from DTMCMC.rng_helpers import reset_seed_guard_for_tests, seed_run
from DTMCMC.temperature_ladder_helpers import (
    GeometricTemperatureLadder,
    Ts_to_betas,
    get_spacing_integrated,
    remap_ladder_indices,
    standardize_input_vars,
)
from experiments.adaptive import AdaptiveLadderController
from experiments.harness.paths import resolve
from experiments.harness.runner import CountingLikelihood, build_likelihood, build_sampler, run_from_spec
from experiments.harness.spec import ADAPTIVE_MODES, RunSpec
from tests.test_harness import TINY_GAUSSIAN_SPEC, make_tiny_spec


def test_remap_ladder_indices_hand_computed() -> None:
    """Both D6 remap rules on a hand-checkable ladder pair."""
    Ts_old = np.array([1., 4., 16., np.inf])
    Ts_new = np.array([1., 2., 8., 64.])

    # nearest in log T: 2 is sqrt(1*4) -> tie resolved to first (index 0);
    # 8 is sqrt(4*16) -> index 1; 64 is closer to 16 than to inf(=1e300)
    assert_array_equal(remap_ladder_indices(Ts_old, Ts_new, 'nearest'), [0, 0, 1, 2])
    # at-or-hotter: coolest old rung at or above each new temperature
    assert_array_equal(remap_ladder_indices(Ts_old, Ts_new, 'at_or_hotter'), [0, 1, 2, 3])
    # a new rung hotter than every finite old rung falls back to the hottest
    assert_array_equal(remap_ladder_indices(np.array([1., 4.]), np.array([16.]), 'at_or_hotter'), [1])

    with pytest.raises(ValueError, match='unknown remap rule'):
        remap_ladder_indices(Ts_old, Ts_new, 'nonsense')


@pytest.mark.usefixtures('fresh_seed_guard')
def test_apply_ladder_update_rebinds_and_remaps() -> None:
    """The hook rebinds every alias and remaps state per D6."""
    spec = make_tiny_spec()
    seed_run(spec.seed)
    sampler, _like_obj = build_sampler(spec)
    sampler.advance_block()

    old_Ts = np.asarray(sampler.Ts).copy()
    old_states = sampler.samples[0].copy()
    old_logLs = sampler.logLs[0].copy()
    de_manager = next(m for m in sampler.proposal_manager.managers if isinstance(m, DEJumpManager))
    fisher_manager = next(m for m in sampler.proposal_manager.managers if isinstance(m, FisherJumpManager))
    old_gamma = fisher_manager.gamma_mults.copy()
    # tag each buffer column so the remap is checkable
    for itrt in range(sampler.n_chain):
        de_manager.de_buffer[:, itrt, :] = float(itrt)
    n_archives_before = len(sampler.tracker_manager.cycle_archive)

    new_ladder = GeometricTemperatureLadder(spec.n_chain, n_cold=1, T_cold=1., T_min=1., T_max=10., n_inf_final=1)
    sampler.apply_ladder_update(new_ladder, 'at_or_hotter')

    # aliases rebound to the new ladder's arrays (the stale-alias footgun)
    assert sampler.T_ladder is new_ladder
    assert sampler.betas is new_ladder.betas
    assert sampler.Ts is new_ladder.Ts
    assert sampler.proposal_manager.T_ladder is new_ladder
    for manager in sampler.proposal_manager.managers:
        assert manager.T_ladder is new_ladder

    # chain states remapped to the nearest new temperature, logLs carried
    state_sources = remap_ladder_indices(old_Ts, np.asarray(new_ladder.Ts), 'nearest')
    assert_array_equal(sampler.samples[0], old_states[state_sources])
    assert_array_equal(sampler.logLs[0], old_logLs[state_sources])
    assert_array_equal(sampler.chain_track[0], np.arange(spec.n_chain))

    # DE columns remapped per the requested rule
    buffer_sources = remap_ladder_indices(old_Ts, np.asarray(new_ladder.Ts), 'at_or_hotter')
    for itrt in range(sampler.n_chain):
        assert np.all(de_manager.de_buffer[:, itrt, :] == float(buffer_sources[itrt]))

    # Fisher temperature scaling refreshed for the new betas
    assert not np.array_equal(fisher_manager.gamma_mults, old_gamma)

    # trackers segmented: archives grew, cycle tracker reinitialized
    assert len(sampler.tracker_manager.cycle_archive) == n_archives_before + 1
    assert_array_equal(sampler.tracker_manager.cycle_tracker[0][1:], np.full(spec.n_chain - 1, -1))
    assert_array_equal(sampler.tracker_manager.cycle_tracker[2], np.zeros(spec.n_chain, dtype=np.int64))


def _copy_full_state(source, target) -> None:
    """Copy the complete dynamical state of one sampler into another.

    Covers everything the next block's evolution depends on: chain state,
    iteration counter (Fisher refresh schedule), storage cursors, DE
    buffer + write cursors, and the Fisher matrices/scales (which may be
    stale by up to fisher_downsample blocks — recomputing them from the
    copied samples would not match, plan Phase 5 criterion 1).
    """
    target.samples[:] = source.samples
    target.logLs[:] = source.logLs
    target.chain_track[:] = source.chain_track
    target.samples_store[:] = source.samples_store
    target.logLs_store[:] = source.logLs_store
    target.store_idx = source.store_idx
    target.store_counter = source.store_counter
    target.itrn = source.itrn
    target.logL_means = [arr.copy() for arr in source.logL_means]
    target.logL2_means = [arr.copy() for arr in source.logL2_means]
    target.logL3_means = [arr.copy() for arr in source.logL3_means]
    target.logL4_means = [arr.copy() for arr in source.logL4_means]
    target.logL5_means = [arr.copy() for arr in source.logL5_means]
    target.logL6_means = [arr.copy() for arr in source.logL6_means]
    target.logL_prod11_means = [arr.copy() for arr in source.logL_prod11_means]
    target.logL_prod21_means = [arr.copy() for arr in source.logL_prod21_means]
    target.logL_prod12_means = [arr.copy() for arr in source.logL_prod12_means]
    target.logL_vars = [arr.copy() for arr in source.logL_vars]

    src_de = next(m for m in source.proposal_manager.managers if isinstance(m, DEJumpManager))
    tgt_de = next(m for m in target.proposal_manager.managers if isinstance(m, DEJumpManager))
    tgt_de.de_buffer[:] = src_de.de_buffer
    tgt_de.itrde_write = src_de.itrde_write
    tgt_de.itrde_count = src_de.itrde_count

    src_fisher = next(m for m in source.proposal_manager.managers if isinstance(m, FisherJumpManager))
    tgt_fisher = next(m for m in target.proposal_manager.managers if isinstance(m, FisherJumpManager))
    tgt_fisher.sigma_diags = src_fisher.sigma_diags.copy()
    tgt_fisher.fishers = src_fisher.fishers.copy()
    tgt_fisher.chol_fishers = src_fisher.chol_fishers.copy()
    tgt_fisher.sigma_scales = src_fisher.sigma_scales.copy()
    tgt_fisher.gamma_mults = src_fisher.gamma_mults.copy()

    target.proposal_manager.jump_weights = source.proposal_manager.jump_weights.copy()
    target.proposal_manager.jump_probs = source.proposal_manager.jump_probs.copy()

    target.tracker_manager.cycle_tracker[:] = source.tracker_manager.cycle_tracker
    target.tracker_manager.accept_record[:] = source.tracker_manager.accept_record
    target.tracker_manager.exchange_tracker[:] = source.tracker_manager.exchange_tracker
    target.tracker_manager.esd_record[:] = source.tracker_manager.esd_record
    target.tracker_manager.esd_exchange[:] = source.tracker_manager.esd_exchange
    target.tracker_manager.itrb = source.tracker_manager.itrb


def test_post_freeze_bit_exact_equivalence() -> None:
    """Acceptance 1: post-freeze adaptive == fixed-ladder sampler, bit-exact."""
    reset_seed_guard_for_tests()
    spec = make_tiny_spec(n_steps=64 * 40, block_size=64)
    seed_run(spec.seed)

    controller = AdaptiveLadderController(mode='entropy', update_every_blocks=4, freeze_criterion=(0.08, 2))
    like_a = CountingLikelihood(build_likelihood(spec))
    initial_ladder = controller.initial_ladder(like_a, spec.n_chain, spec.n_cold)
    sampler_a, _ = build_sampler(spec, like_obj=like_a, T_ladder=initial_ladder)

    for _ in range(spec.n_blocks):
        sampler_a.advance_block()
        controller.post_block(sampler_a)
        if controller.frozen:
            break
    assert controller.frozen, 'controller must reach hard freeze without human input'
    # a couple more frozen-path blocks so the state is post-freeze generic
    sampler_a.advance_block()
    sampler_a.advance_block()

    # fresh fixed-ladder sampler on the frozen ladder; construction consumes
    # draws, then the full state is overwritten with copies
    like_b = CountingLikelihood(build_likelihood(spec))
    sampler_b, _ = build_sampler(spec, like_obj=like_b, T_ladder=sampler_a.T_ladder)
    _copy_full_state(sampler_a, sampler_b)

    continuation_seed = 987654
    reset_seed_guard_for_tests()
    seed_run(continuation_seed)
    sampler_a.advance_block()

    reset_seed_guard_for_tests()
    seed_run(continuation_seed)
    sampler_b.advance_block()
    reset_seed_guard_for_tests()

    assert_array_equal(sampler_a.samples, sampler_b.samples)
    assert_array_equal(sampler_a.logLs, sampler_b.logLs)
    assert_array_equal(sampler_a.chain_track, sampler_b.chain_track)
    assert_array_equal(sampler_a.logLs_store, sampler_b.logLs_store)


def test_adaptive_entropy_converges_to_gold(tmp_path) -> None:
    """Acceptance 2: unattended convergence to the gold dS structure on cake 5D.

    Convergence is asserted structurally against the gold entropy
    profile: the run must hard-freeze on its own, resolve the phase
    transition (gold packs 7 of 11 finite rungs at T <= 2; a starved
    log-uniform ladder has 1 there), and space every non-coldest link at
    the gold per-link scale. The extreme spike tip below T ~ 1.15 keeps
    refining across updates (the coldest link still carries several gold
    nats when the freeze fires) — the refinement-rate measurement is
    E3's job; this test pins the discovered structure.
    """
    reset_seed_guard_for_tests()
    data: dict[str, Any] = {key: dict(value) if isinstance(value, dict) else value for key, value in TINY_GAUSSIAN_SPEC.items()}
    data['name'] = 'adaptive_cake'
    data['seed'] = 555
    data['likelihood'] = {'name': 'cake', 'n_par': 5, 'cutoff': 10}
    data['ladder'] = {'kind': 'geometric', 'n_chain': 12, 'n_cold': 1}
    data['run'] = {'n_steps': 512 * 320, 'block_size': 512, 'store_thin': 16, 'n_record': -1, 'checkpoint_every_blocks': 160}
    data['adaptive'] = {'mode': 'entropy', 'update_every_blocks': 8, 'forgetting': 0.15, 'freeze_dlog': 0.05, 'freeze_consecutive': 3}
    spec = RunSpec.from_dict(data)

    artifact_path = run_from_spec(spec, tmp_path)
    reset_seed_guard_for_tests()

    with h5py.File(str(artifact_path), 'r') as hf:
        assert bool(hf['ladder/history'].attrs['frozen']), 'run must end frozen without human input'
        final_Ts = np.asarray(hf['ladder/Ts'])
        history_Ts = np.asarray(hf['ladder/history/Ts'])
    assert history_Ts.shape[0] >= 8

    # gold dS profile: evaluate the gold cumulative entropy at the adaptive
    # rung positions (interpolated dS profile comparison)
    Ts_gold = np.load(resolve('data/Ts_cake_gold.npy'))
    vars_gold = np.load(resolve('data/vars_cake_gold.npy'))
    keep = Ts_gold >= 1.
    betas_use, vars_use = standardize_input_vars(Ts_to_betas(Ts_gold[keep]), vars_gold[keep])
    s_integral = get_spacing_integrated(vars_use, betas_use, False)

    finite_Ts = np.sort(final_Ts[np.isfinite(final_Ts)])
    betas_rungs = 1. / finite_Ts
    # interpolate S(beta) (integral is over descending beta; flip ascending)
    s_at_rungs = np.asarray(np.interp(betas_rungs[::-1], betas_use[::-1], s_integral[::-1]))[::-1]
    increments = np.abs(np.diff(s_at_rungs))
    gold_per_link = s_integral[-1] / (finite_Ts.size - 1)

    # the transition cluster is resolved (log-uniform starvation puts 1 rung here)
    assert int(np.sum(finite_Ts <= 2.0)) >= 4
    # every link above the spike tip sits at the gold per-link scale
    assert float(increments[1:].max()) <= 2.5 * gold_per_link
    # the still-refining spike tip is bounded
    assert float(increments[0]) <= 8.0


def test_adaptive_modes_sync_and_validation() -> None:
    """Spec-layer mode set matches the controller's; bad modes raise."""
    assert ADAPTIVE_MODES == adaptive_module.ADAPTIVE_MODES
    with pytest.raises(ValueError, match='unknown adaptive mode'):
        AdaptiveLadderController(mode='nonsense')


# reuse the shared guard fixture from the harness tests
from tests.test_harness import fresh_seed_guard  # noqa: E402, F401
