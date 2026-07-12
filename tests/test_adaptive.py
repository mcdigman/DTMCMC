"""Phase 5 acceptance tests: ladder-update hook and adaptive controller.

Criterion 1: post-freeze equivalence is bit-exact — the frozen sampler's
full state copied into a fresh fixed-ladder sampler, both streams
reseeded identically, one block advanced, identical output. Criterion 2:
the adaptive entropy ladder converges to the gold ladder's dS profile on
cake 5D without human input. Criterion 3: the golden test stays green
(separate file, unchanged).
"""

from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal

if TYPE_CHECKING:
    from DTMCMC.dtmcmc_sampler import DTMCMCSampler
    from experiments.harness.runner import LikelihoodLike

import experiments.adaptive as adaptive_module
from DTMCMC.de_manager import DEJumpManager
from DTMCMC.fisher_manager import FisherJumpManager
from DTMCMC.rng_helpers import reset_seed_guard_for_tests, seed_run
from DTMCMC.temperature_ladder_helpers import (
    GeometricTemperatureLadder,
    TemperatureLadder,
    Ts_to_betas,
    get_spacing_integrated,
    remap_ladder_indices,
    standardize_input_vars,
)
from DTMCMC.tracker_manager import TrackerManager
from experiments.adaptive import AdaptiveLadderController
from experiments.harness.paths import resolve
from experiments.harness.runner import CountingLikelihood, build_likelihood, build_sampler, run_from_spec
from experiments.harness.spec import ADAPTIVE_MODES, RunSpec, SpecError
from tests.test_harness import TINY_GAUSSIAN_SPEC, make_tiny_spec


def test_remap_ladder_indices_hand_computed() -> None:
    """Both D6 buffer remap rules on a hand-checkable ladder pair."""
    Ts_old = np.array([1., 4., 16., np.inf])
    Ts_new = np.array([1., 2., 10., 64.])

    # nearest in log T: 2 is sqrt(1*4), a log-equidistant tie containing
    # the receiving slot, which therefore keeps its slot (D6); 10 is
    # nearer 16 than 4; 64 is closer to 16 than to inf(=1e300)
    assert_array_equal(remap_ladder_indices(Ts_old, Ts_new, 'nearest'), [0, 1, 2, 2])
    # at-or-hotter: coolest old rung at or above each new temperature
    assert_array_equal(remap_ladder_indices(Ts_old, Ts_new, 'at_or_hotter'), [0, 1, 2, 3])
    # a new rung hotter than every finite old rung falls back to the hottest
    assert_array_equal(remap_ladder_indices(np.array([1., 4.]), np.array([16.]), 'at_or_hotter'), [1])
    # an exact-T tie NOT containing the receiving slot goes to the lowest
    # tied slot (D6): new slot 3 at T=2 ties old duplicate rungs 1 and 2
    assert_array_equal(remap_ladder_indices(np.array([1., 2., 2., 8.]), np.array([0.5, 0.6, 0.7, 2.]), 'at_or_hotter'), [0, 0, 0, 1])

    with pytest.raises(ValueError, match='unknown remap rule'):
        remap_ladder_indices(Ts_old, Ts_new, 'nonsense')


def test_remap_ladder_indices_identical_is_identity() -> None:
    """An identical ladder — including duplicate temperatures — maps every slot to itself (D6)."""
    Ts = np.array([1., 1., 2., 4., np.inf])
    for rule in ('nearest', 'at_or_hotter'):
        assert_array_equal(remap_ladder_indices(Ts, Ts, rule), np.arange(Ts.size))


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

    # chain states remap by temperature rank — the identity for
    # equal-size sorted ladders, a bijection: no walker cloned or lost
    assert_array_equal(sampler.samples[0], old_states)
    assert_array_equal(sampler.logLs[0], old_logLs)
    assert_array_equal(sampler.chain_track[0], np.arange(spec.n_chain))

    # DE columns remapped per the requested rule, then each resourced
    # rung's current state written at the newest buffer row (D6
    # self-inclusion restoration)
    buffer_sources = remap_ladder_indices(old_Ts, np.asarray(new_ladder.Ts), 'at_or_hotter')
    row_newest = (de_manager.itrde_write - 1) % de_manager.de_size
    for itrt in range(sampler.n_chain):
        if buffer_sources[itrt] == itrt:
            assert np.all(de_manager.de_buffer[:, itrt, :] == float(itrt))
        else:
            other_rows = np.delete(np.arange(de_manager.de_size), row_newest)
            assert np.all(de_manager.de_buffer[other_rows, itrt, :] == float(buffer_sources[itrt]))
            assert_array_equal(de_manager.de_buffer[row_newest, itrt, :], sampler.samples[0][itrt])

    # Fisher temperature scaling refreshed for the new betas
    assert not np.array_equal(fisher_manager.gamma_mults, old_gamma)

    # trackers segmented: archives grew, cycle tracker reinitialized,
    # round-trip event log boundary recorded
    assert len(sampler.tracker_manager.cycle_archive) == n_archives_before + 1
    assert_array_equal(sampler.tracker_manager.cycle_tracker[0][1:], np.full(spec.n_chain - 1, -1))
    assert_array_equal(sampler.tracker_manager.cycle_tracker[2], np.zeros(spec.n_chain, dtype=np.int64))
    assert sampler.tracker_manager.rt_segment_itrns == [sampler.itrn]


@pytest.mark.usefixtures('fresh_seed_guard')
def test_apply_ladder_update_identical_ladder_strict_noop() -> None:
    """Acceptance 4: an identical-ladder update (duplicate temperatures, n_cold > 1) is a strict no-op for states, logLs, and DE buffers."""
    data: dict[str, Any] = {key: dict(value) if isinstance(value, dict) else value for key, value in TINY_GAUSSIAN_SPEC.items()}
    data['ladder'] = {'kind': 'geometric', 'n_chain': 6, 'n_cold': 2, 'T_cold': 1.0, 'T_min': 1.0, 'T_max': 100.0, 'n_inf_final': 1}
    spec = RunSpec.from_dict(data)
    seed_run(spec.seed)
    sampler, _like_obj = build_sampler(spec)
    sampler.advance_block()
    assert float(np.asarray(sampler.Ts)[0]) == float(np.asarray(sampler.Ts)[1]), 'test premise: duplicate cold temperatures'

    old_states = sampler.samples[0].copy()
    old_logLs = sampler.logLs[0].copy()
    de_manager = next(m for m in sampler.proposal_manager.managers if isinstance(m, DEJumpManager))
    old_buffer = de_manager.de_buffer.copy()

    same_ladder = TemperatureLadder(spec.n_cold, np.asarray(sampler.Ts).copy())
    sampler.apply_ladder_update(same_ladder, 'at_or_hotter')

    assert_array_equal(sampler.samples[0], old_states)
    assert_array_equal(sampler.logLs[0], old_logLs)
    assert_array_equal(de_manager.de_buffer, old_buffer)


@pytest.mark.usefixtures('fresh_seed_guard')
def test_apply_ladder_update_extension_bijection_and_self_inclusion() -> None:
    """Acceptance 4 under cold support extension: state remap stays a bijection and every resourced rung's buffer column contains its current state."""
    spec = make_tiny_spec()
    seed_run(spec.seed)
    hot_ladder = GeometricTemperatureLadder(spec.n_chain, n_cold=1, T_cold=4., T_min=4., T_max=100., n_inf_final=1)
    sampler, _like_obj = build_sampler(spec, T_ladder=hot_ladder)
    sampler.advance_block()

    old_states = sampler.samples[0].copy()
    de_manager = next(m for m in sampler.proposal_manager.managers if isinstance(m, DEJumpManager))

    extended = GeometricTemperatureLadder(spec.n_chain, n_cold=1, T_cold=1., T_min=1., T_max=100., n_inf_final=1)
    sampler.apply_ladder_update(extended, 'at_or_hotter')

    # bijection: every walker keeps its slot, none cloned or discarded
    assert_array_equal(sampler.samples[0], old_states)

    # the extension resources sub-support rungs many-to-one (D6); each
    # resourced rung's column must contain its own current state
    buffer_sources = remap_ladder_indices(np.asarray(hot_ladder.Ts), np.asarray(extended.Ts), 'at_or_hotter')
    resourced = np.flatnonzero(buffer_sources != np.arange(spec.n_chain))
    assert resourced.size > 0, 'test premise: the extension must resource at least one rung'
    for itrt in resourced:
        rows_matching = np.all(de_manager.de_buffer[:, itrt, :] == sampler.samples[0][itrt], axis=1)
        assert bool(rows_matching.any())


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

    controller = AdaptiveLadderController(mode='entropy', update_every_blocks=4, freeze_criterion=(0.08, 2), budget_blocks=10**6)
    like_a = CountingLikelihood(build_likelihood(spec))
    initial_ladder = controller.initial_ladder(like_a, spec.n_chain, spec.n_cold)
    sampler_a, _ = build_sampler(spec, like_obj=like_a, T_ladder=initial_ladder)

    for _ in range(spec.n_blocks):
        sampler_a.advance_block()
        controller.post_block(sampler_a)
        if controller.frozen:
            break
    assert controller.frozen, 'controller must reach hard freeze without human input'
    assert controller.frozen_by == 'criterion'
    # sub-threshold rebuilds are held, never applied: the engine segments
    # exactly once per applied update, and reaching the freeze requires
    # held evaluations (plan D6/Phase 5)
    n_applied = sum(record.applied for record in controller.history)
    assert len(sampler_a.tracker_manager.rt_segment_itrns) == n_applied
    assert any(not record.applied for record in controller.history)
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


# acceptance 2 battery seeds, pre-registered (amended per PR #16 review):
# the set deliberately includes 556/559/560, which froze starved under the
# pre-witness controller (stability-only freeze on the cold-cap fixed point)
CONVERGENCE_BATTERY_SEEDS = (555, 556, 557, 558, 559, 560)


@pytest.mark.parametrize('seed', CONVERGENCE_BATTERY_SEEDS)
def test_adaptive_entropy_converges_to_gold(tmp_path, seed: int) -> None:
    """Acceptance 2: unattended convergence to the gold dS structure on cake 5D.

    Run as an N-seed battery (N >= 6, seeds pre-registered): every run
    must hard-freeze via the coupling witness — not the budget — within
    the test budget, resolve the phase transition (gold packs 7 of 11
    finite rungs at T <= 2; a starved log-uniform ladder has 1 there),
    and space every non-coldest link at the gold per-link scale. The
    extreme spike tip below T ~ 1.15 keeps refining across updates (the
    coldest link still carries several gold nats when the freeze fires)
    — the refinement-rate measurement is E3's job; this battery pins the
    discovered structure.
    """
    reset_seed_guard_for_tests()
    data: dict[str, Any] = {key: dict(value) if isinstance(value, dict) else value for key, value in TINY_GAUSSIAN_SPEC.items()}
    data['name'] = 'adaptive_cake'
    data['seed'] = seed
    data['likelihood'] = {'name': 'cake', 'n_par': 5, 'cutoff': 10}
    data['ladder'] = {'kind': 'geometric', 'n_chain': 12, 'n_cold': 1}
    data['run'] = {'n_steps': 512 * 320, 'block_size': 512, 'store_thin': 16, 'checkpoint_every_blocks': 160}
    data['adaptive'] = {'mode': 'entropy', 'update_every_blocks': 8, 'forgetting': 0.15, 'freeze_dlog': 0.05, 'freeze_consecutive': 3, 'budget_blocks': 288}
    spec = RunSpec.from_dict(data)

    artifact_path = run_from_spec(spec, tmp_path)
    reset_seed_guard_for_tests()

    with h5py.File(str(artifact_path), 'r') as hf:
        assert bool(hf['ladder/history'].attrs['frozen']), 'run must end frozen without human input'
        assert str(hf['ladder/history'].attrs['frozen_by']) == 'criterion', 'freeze must fire via the coupling witness, not the budget'
        final_Ts = np.asarray(hf['ladder/Ts'])
        history_applied = np.asarray(hf['ladder/history/applied'])
    assert int(history_applied.sum()) >= 8

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


@pytest.mark.usefixtures('fresh_seed_guard')
def test_freeze_requires_coupling_witness(monkeypatch) -> None:
    """Acceptance-adjacent: rebuild stability alone must not freeze (plan Phase 5).

    Identical configuration and streams to
    test_post_freeze_bit_exact_equivalence, which freezes via the
    criterion — but with the coupling witness suppressed (a pure
    observer, so RNG streams are unchanged) the controller must keep
    holding instead of certifying stability it cannot corroborate.
    """
    spec = make_tiny_spec(n_steps=64 * 40, block_size=64)
    seed_run(spec.seed)
    monkeypatch.setattr(TrackerManager, 'get_n_cycles', lambda self: np.zeros(self.n_chain, dtype=np.int64))

    controller = AdaptiveLadderController(mode='entropy', update_every_blocks=4, freeze_criterion=(0.08, 2), budget_blocks=10**6)
    like_obj = CountingLikelihood(build_likelihood(spec))
    sampler, _ = build_sampler(spec, like_obj=like_obj, T_ladder=controller.initial_ladder(like_obj, spec.n_chain, spec.n_cold))
    for _ in range(spec.n_blocks):
        sampler.advance_block()
        controller.post_block(sampler)

    assert not controller.frozen
    assert controller.frozen_by == ''
    # the schedule reached the holding pattern (sub-threshold rebuilds at
    # the target window); only the witness kept the freeze from firing
    n_consecutive = controller.freeze_criterion[1]
    assert all(not record.applied for record in controller.history[-n_consecutive:])


@pytest.mark.usefixtures('fresh_seed_guard')
def test_freeze_requires_trips_in_every_streak_window(monkeypatch) -> None:
    """A single lucky round trip must not certify a starved ladder.

    get_n_cycles is patched to a nonzero CONSTANT: the open-segment
    witness (the plan's floor) is green throughout, but no NEW trips
    ever arrive between evaluations. The per-window witness must
    therefore never assemble a freeze streak — this is exactly the
    hold-lengthened-segment loophole the strengthening closes.
    """
    spec = make_tiny_spec(n_steps=64 * 40, block_size=64)
    seed_run(spec.seed)
    monkeypatch.setattr(TrackerManager, 'get_n_cycles', lambda self: np.ones(self.n_chain, dtype=np.int64))

    controller = AdaptiveLadderController(mode='entropy', update_every_blocks=4, freeze_criterion=(0.08, 2), budget_blocks=10**6)
    like_obj = CountingLikelihood(build_likelihood(spec))
    sampler, _ = build_sampler(spec, like_obj=like_obj, T_ladder=controller.initial_ladder(like_obj, spec.n_chain, spec.n_cold))
    for _ in range(spec.n_blocks):
        sampler.advance_block()
        controller.post_block(sampler)

    assert not controller.frozen
    assert controller.frozen_by == ''


@pytest.mark.usefixtures('fresh_seed_guard')
def test_budget_freeze_records_reason() -> None:
    """A run exhausting budget_blocks unfrozen hard-freezes with frozen_by='budget' (plan Phase 5)."""
    spec = make_tiny_spec(n_steps=64 * 8, block_size=64)
    seed_run(spec.seed)
    controller = AdaptiveLadderController(mode='entropy', update_every_blocks=8, budget_blocks=4)
    like_obj = CountingLikelihood(build_likelihood(spec))
    sampler, _ = build_sampler(spec, like_obj=like_obj, T_ladder=controller.initial_ladder(like_obj, spec.n_chain, spec.n_cold))
    for _ in range(spec.n_blocks):
        sampler.advance_block()
        controller.post_block(sampler)

    assert controller.frozen
    assert controller.frozen_by == 'budget'
    # adaptation was capped before the first cadence: the hook never ran
    # and the remaining blocks took the post-freeze fixed-ladder path
    assert len(sampler.tracker_manager.rt_segment_itrns) == 0
    assert controller.history == []


@pytest.mark.usefixtures('fresh_seed_guard')
def test_eggbox_mode_retention_gate() -> None:
    """Acceptance 5 (pre-E3): occupied-mode counts among cold-slot DE-buffer columns are preserved across a support-extension update.

    'Cold-slot' is operationalized as the coldest half of the ladder —
    where mode identity is physically meaningful and where the
    extension's many-to-one crowd-out concentrates. Failure reopens the
    extension-case buffer rule as the D6 pilot A/B before E3 runs.
    """
    n_par = 2
    data: dict[str, Any] = {key: dict(value) if isinstance(value, dict) else value for key, value in TINY_GAUSSIAN_SPEC.items()}
    data['name'] = 'eggbox_gate'
    data['likelihood'] = {'name': 'eggbox', 'n_par': n_par}
    data['run'] = {'n_steps': 64 * 8, 'block_size': 64, 'store_thin': 1, 'checkpoint_every_blocks': 4}
    spec = RunSpec.from_dict(data)
    seed_run(spec.seed)

    hot_ladder = GeometricTemperatureLadder(spec.n_chain, n_cold=1, T_cold=4., T_min=4., T_max=100., n_inf_final=1)
    sampler, _like_obj = build_sampler(spec, T_ladder=hot_ladder)
    for _ in range(spec.n_blocks):
        sampler.advance_block()

    def occupied_modes(buffer_cold_slice: np.ndarray) -> set[tuple[int, ...]]:
        # eggbox maxima sit at x_i = 2 pi k: nearest-mode assignment per row
        rows = buffer_cold_slice.reshape(-1, n_par)
        return {tuple(mode) for mode in np.round(rows / (2. * np.pi)).astype(int)}

    de_manager = next(m for m in sampler.proposal_manager.managers if isinstance(m, DEJumpManager))
    n_cold_half = spec.n_chain // 2
    modes_before = occupied_modes(de_manager.de_buffer[:, :n_cold_half, :])

    extended = GeometricTemperatureLadder(spec.n_chain, n_cold=1, T_cold=1., T_min=1., T_max=100., n_inf_final=1)
    sampler.apply_ladder_update(extended, 'at_or_hotter')
    modes_after = occupied_modes(de_manager.de_buffer[:, :n_cold_half, :])

    assert len(modes_before) > 1, 'test premise: the cold columns must start multimodal'
    assert len(modes_after) >= len(modes_before)


def test_adaptive_spec_validation() -> None:
    """[adaptive] rejects unknown keys, missing budget_blocks, and sub-unit T_min_factor."""
    base: dict[str, Any] = {key: dict(value) if isinstance(value, dict) else value for key, value in TINY_GAUSSIAN_SPEC.items()}
    base['adaptive'] = {'mode': 'entropy', 'budget_blocks': 8}
    RunSpec.from_dict(base)

    bad_tables = (
        {'mode': 'entropy', 'budget_blocks': 8, 'forgeting': 0.15},    # typo'd key must fail loudly
        {'mode': 'entropy'},                                           # budget_blocks is required
        {'mode': 'entropy', 'budget_blocks': 8, 'T_min_factor': 0.9},  # sub-unit rungs locked out pending amendment
    )
    for table in bad_tables:
        data = dict(base)
        data['adaptive'] = table
        with pytest.raises(SpecError):
            RunSpec.from_dict(data)


def test_adaptive_modes_sync_and_validation() -> None:
    """Spec-layer mode set matches the controller's; bad modes raise."""
    assert ADAPTIVE_MODES == adaptive_module.ADAPTIVE_MODES
    with pytest.raises(ValueError, match='unknown adaptive mode'):
        AdaptiveLadderController(mode='nonsense')
    with pytest.raises(ValueError, match='budget_blocks'):
        AdaptiveLadderController(mode='entropy')
    with pytest.raises(ValueError, match='unknown var_estimator'):
        AdaptiveLadderController(mode='entropy', budget_blocks=8, var_estimator=7)


class _StubLikelihood:
    """Deterministic LikelihoodLike stand-in (no RNG draws)."""

    def __init__(self) -> None:
        self._count = 0

    def prior_draw(self) -> np.ndarray:
        return np.zeros(1)

    def get_loglike(self, _x: np.ndarray) -> float:
        self._count += 1
        return float(self._count % 2)


class _StubTracker:
    """Tracker stand-in whose witness is always green."""

    def __init__(self, n_chain: int) -> None:
        self.n_chain = n_chain

    def get_n_cycles(self) -> np.ndarray:
        return np.ones(self.n_chain, dtype=np.int64)


class _StubSampler:
    """Minimal sampler surface driving post_block with scripted moments."""

    def __init__(self, Ts: np.ndarray, block_size: int = 64) -> None:
        self.Ts = np.asarray(Ts)
        self.n_chain = self.Ts.size
        self.n_cold = 1
        self.block_size = block_size
        self.itrn = 0
        self.logL_means: list[np.ndarray] = []
        self.logL2_means: list[np.ndarray] = []
        self.tracker_manager = _StubTracker(self.n_chain)

    def feed_block(self, vars_profile: list[float]) -> None:
        self.logL_means.append(np.zeros(self.n_chain))
        self.logL2_means.append(np.asarray(vars_profile))
        self.itrn += self.block_size

    def apply_ladder_update(self, new_ladder: TemperatureLadder, _remap_rule: str) -> None:
        self.Ts = np.asarray(new_ladder.Ts)


def _coldest_ratio(Ts: np.ndarray) -> float:
    finite = np.sort(Ts[np.isfinite(Ts)])
    return float(finite[1] / finite[0])


def test_pessimistic_var_estimator_ratchets_and_ages_out() -> None:
    """var_estimator=1 believes the largest recent variance; =0 averages it away.

    Scripted moments at fixed recurring temperatures: one high-variance
    segment at the coldest rung amid low-variance segments. The
    pessimistic estimator must keep the cold end packed while the high
    estimate is inside its rolling window (where the weighted mean has
    already diluted it), and must relax once the window ages it out.
    """
    Ts_fixed = np.array([1., 2., 4., 8., 16., np.inf])
    low = [0.5, 0.5, 0.5, 0.5, 0.5, 0.25]
    high_cold = [50., 0.5, 0.5, 0.5, 0.5, 0.25]

    controllers = {}
    for estimator in (0, 1):
        controller = AdaptiveLadderController(
            mode='entropy', update_every_blocks=1, forgetting=0.,
            freeze_criterion=(1.e9, 10**6), budget_blocks=10**6, var_estimator=estimator,
        )
        # huge freeze_dlog holds every rebuild, so the stub's temperatures
        # recur every segment — the regime where the estimators differ
        controller.initial_ladder(cast('LikelihoodLike', _StubLikelihood()), Ts_fixed.size, 1)
        stub = _StubSampler(Ts_fixed)
        sampler = cast('DTMCMCSampler', stub)
        feeds = [low, high_cold] + [low] * (adaptive_module.VAR_HISTORY_LENGTH + 3)
        for vars_profile in feeds:
            stub.feed_block(vars_profile)
            controller.post_block(sampler)
        controllers[estimator] = controller

    # three low segments after the spike: still inside the rolling window,
    # the pessimistic candidate stays packed at the cold end while the
    # weighted mean has diluted the spike
    idx_probe = 4
    ratio_pessimistic = _coldest_ratio(controllers[1].history[idx_probe].Ts)
    ratio_mean = _coldest_ratio(controllers[0].history[idx_probe].Ts)
    assert ratio_pessimistic < ratio_mean

    # once VAR_HISTORY_LENGTH low segments have passed, the unreconfirmed
    # spike ages out and the pessimistic candidate relaxes again
    ratio_pessimistic_late = _coldest_ratio(controllers[1].history[-1].Ts)
    assert ratio_pessimistic_late > ratio_pessimistic


# reuse the shared guard fixture from the harness tests
from tests.test_harness import fresh_seed_guard  # noqa: E402, F401
