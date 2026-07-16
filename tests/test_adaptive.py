"""Phase 5 acceptance tests: ladder-update hook and adaptive controller.

Criterion 1: post-freeze equivalence is bit-exact — the frozen sampler's
full state copied into a fresh fixed-ladder sampler, both streams
reseeded identically, one block advanced, identical output. Criterion 2:
the reference-anchored cake battery — unattended adaptation on a cake
phase transition ends frozen with the T=1 readout passing sample-space
gates. Criterion 3: the golden test stays green (separate file,
unchanged). Heavy batteries carry the slow marker.
"""

from typing import TYPE_CHECKING, Any, cast

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal

if TYPE_CHECKING:
    from DTMCMC.dtmcmc_sampler import DTMCMCSampler
    from DTMCMC.likelihood import AbstractLikelihood

import experiments.adaptive as adaptive_module
from DTMCMC.de_manager import DEJumpManager
from DTMCMC.fisher_manager import FisherJumpManager
from DTMCMC.likelihoods.cake_likelihood import CakeLikelihood
from DTMCMC.rng_helpers import get_rng, reset_seed_guard_for_tests, seed_run
from DTMCMC.temperature_ladder_helpers import (
    AcceptanceTemperatureLadder,
    GeometricTemperatureLadder,
    TemperatureLadder,
    get_spacing_integrated,
    remap_ladder_indices,
    standardize_input_vars,
)
from DTMCMC.tracker_manager import TrackerManager
from experiments.adaptive import AdaptiveLadderController
from experiments.gates import GateReport, dedup_rows, ladder_entropy_gates, moment_gates, nn_gate, radial_mixture_gates
from experiments.harness.runner import (
    build_adaptive_controller,
    build_likelihood,
    build_sampler,
    run_from_spec,
)
from experiments.harness.spec import ADAPTIVE_MODES, RunSpec, SpecError
from experiments.reference_samplers import cake_logL_radial, cake_moment_r2, cake_tempered_cumulants, draw_cake
from tests.battery_common import adaptive_spec_data, assert_readout_structure, load_post_freeze
from tests.test_harness import TINY_GAUSSIAN_SPEC, make_tiny_spec


def test_remap_ladder_indices_hand_computed() -> None:
    """Both D6 buffer remap rules on a hand-checkable ladder pair."""
    Ts_old = np.array([1.0, 4.0, 16.0, np.inf])
    Ts_new = np.array([1.0, 2.0, 10.0, 64.0])

    # nearest in log T: 2 is sqrt(1*4), a log-equidistant tie containing
    # the receiving slot, which therefore keeps its slot (D6); 10 is
    # nearer 16 than 4; 64 is closer to 16 than to inf(=1e300)
    assert_array_equal(remap_ladder_indices(Ts_old, Ts_new, 'nearest'), [0, 1, 2, 2])
    # no_remap preserves DE columns by slot and is bijective for this
    # equal-size ladder update.
    assert_array_equal(remap_ladder_indices(Ts_old, Ts_new, 'no_remap'), [0, 1, 2, 3])
    # at-or-hotter: coolest old rung at or above each new temperature
    assert_array_equal(remap_ladder_indices(Ts_old, Ts_new, 'at_or_hotter'), [0, 1, 2, 3])
    # a new rung hotter than every finite old rung falls back to the hottest
    assert_array_equal(remap_ladder_indices(np.array([1.0, 4.0]), np.array([16.0]), 'at_or_hotter'), [1])
    # an exact-T tie NOT containing the receiving slot goes to the lowest
    # tied slot (D6): new slot 3 at T=2 ties old duplicate rungs 1 and 2
    assert_array_equal(
        remap_ladder_indices(np.array([1.0, 2.0, 2.0, 8.0]), np.array([0.5, 0.6, 0.7, 2.0]), 'at_or_hotter'),
        [0, 0, 0, 1],
    )

    with pytest.raises(ValueError, match='unknown remap rule'):
        remap_ladder_indices(Ts_old, Ts_new, 'nonsense')


def test_remap_ladder_indices_identical_is_identity() -> None:
    """An identical ladder — including duplicate temperatures — maps every slot to itself (D6)."""
    Ts = np.array([1.0, 1.0, 2.0, 4.0, np.inf])
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

    new_ladder = GeometricTemperatureLadder(spec.n_chain, n_cold=1, T_cold=1.0, T_min=1.0, T_max=10.0, n_inf_final=1)
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
    data: dict[str, Any] = {
        key: dict(value) if isinstance(value, dict) else value for key, value in TINY_GAUSSIAN_SPEC.items()
    }
    data['ladder'] = {
        'kind': 'geometric',
        'n_chain': 6,
        'n_cold': 2,
        'T_cold': 1.0,
        'T_min': 1.0,
        'T_max': 100.0,
        'n_inf_final': 1,
    }
    spec = RunSpec.from_dict(data)
    seed_run(spec.seed)
    sampler, _like_obj = build_sampler(spec)
    sampler.advance_block()
    assert float(np.asarray(sampler.Ts)[0]) == float(np.asarray(sampler.Ts)[1]), (
        'test premise: duplicate cold temperatures'
    )

    old_states = sampler.samples[0].copy()
    old_logLs = sampler.logLs[0].copy()
    de_manager = next(m for m in sampler.proposal_manager.managers if isinstance(m, DEJumpManager))
    old_buffer = de_manager.de_buffer.copy()

    same_ladder = TemperatureLadder(np.asarray(sampler.Ts).copy(), n_cold=spec.n_cold)
    sampler.apply_ladder_update(same_ladder, 'at_or_hotter')

    assert_array_equal(sampler.samples[0], old_states)
    assert_array_equal(sampler.logLs[0], old_logLs)
    assert_array_equal(de_manager.de_buffer, old_buffer)


@pytest.mark.usefixtures('fresh_seed_guard')
def test_apply_ladder_update_extension_bijection_and_self_inclusion() -> None:
    """Acceptance 4 under cold support extension: state remap stays a bijection and every resourced rung's buffer column contains its current state."""
    spec = make_tiny_spec()
    seed_run(spec.seed)
    hot_ladder = GeometricTemperatureLadder(spec.n_chain, n_cold=1, T_cold=4.0, T_min=4.0, T_max=100.0, n_inf_final=1)
    sampler, _like_obj = build_sampler(spec, T_ladder=hot_ladder)
    sampler.advance_block()

    old_states = sampler.samples[0].copy()
    de_manager = next(m for m in sampler.proposal_manager.managers if isinstance(m, DEJumpManager))

    extended = GeometricTemperatureLadder(spec.n_chain, n_cold=1, T_cold=1.0, T_min=1.0, T_max=100.0, n_inf_final=1)
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

    controller = AdaptiveLadderController(
        mode='entropy', update_every_blocks=4, freeze_criterion=(0.08, 2), budget_blocks=10**6
    )
    like_a = build_likelihood(spec)
    initial_ladder = controller.initial_ladder(like_a, spec.n_chain, spec.n_cold)
    sampler_a, _ = build_sampler(spec, like_obj=like_a, T_ladder=initial_ladder)

    for _ in range(spec.n_blocks):
        sampler_a.advance_block()
        controller.post_block(sampler_a)
        if controller.frozen:
            break
    assert controller.frozen, 'controller must reach hard freeze'
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
    like_b = build_likelihood(spec)
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


# acceptance 2 battery seeds
CONVERGENCE_BATTERY_SEEDS = (555, 556, 557, 558, 559, 560)

# The battery cake uses the same 5D two-tier structure as the default
# cake, with a wider narrow tier so the slow test can gate sample-space
# recovery directly.
BATTERY_CAKE_WIDTHS = (4.0, 0.15)
BATTERY_NARROW_R2 = 2.25  # tier-assignment radius^2 (narrow spike sigma 0.4)
BATTERY_N_BLOCKS = 320
BATTERY_BUDGET_BLOCKS = 240
BATTERY_STORE_THIN = 8


def _battery_reference(rng_seed: int = 20260712) -> np.ndarray:
    """Exact reference draws for the battery cake."""
    return draw_cake(20000, 5, get_rng(rng_seed), widths=BATTERY_CAKE_WIDTHS)


def _battery_entropy_profile() -> tuple[np.ndarray, np.ndarray]:
    """Analytic entropy profile S(beta) for the battery cake over T in [0.9, 4].

    Computed by radial quadrature (cake_tempered_cumulants). The span
    stops at T = 4 because hotter temperatures pick up prior-box corner
    variance outside the spherical quadrature approximation.
    """
    betas = 1.0 / np.geomspace(0.9, 4.0, 64)
    _, vars_quad = cake_tempered_cumulants(betas, 5, widths=BATTERY_CAKE_WIDTHS)
    betas_use, vars_use = standardize_input_vars(betas, vars_quad)
    s_profile = get_spacing_integrated(vars_use, betas_use, False)
    return betas_use, s_profile


def _cake_battery_gates(run: dict[str, Any], reference: np.ndarray, seed: int) -> GateReport:
    """The full reference-anchored gate stack for one battery run."""
    report = GateReport()
    cold = run['cold']
    cold_unique = dedup_rows(cold)

    # posterior recovery in sample space
    ref_r2_mean = cake_moment_r2(5, widths=BATTERY_CAKE_WIDTHS)
    ref_narrow = float((((reference**2).sum(axis=1)) < BATTERY_NARROW_R2).mean())
    report.merge(
        radial_mixture_gates(
            cold,
            r2_threshold=BATTERY_NARROW_R2,
            narrow_frac_ref=ref_narrow,
            narrow_frac_tol=0.15,
            r2_mean_ref=ref_r2_mean,
            r2_mean_rtol=0.25,
            min_tier_flips=200,
        )
    )
    per_coord_var = np.full(5, ref_r2_mean / 5.0)
    report.merge(moment_gates(cold, np.zeros(5), per_coord_var, mean_tol_sigmas=0.25, var_ratio_bounds=(0.7, 1.3)))
    report.merge(nn_gate(reference, cold_unique, threshold=2.0, n_use=2000, rng=get_rng(seed)))

    ref_logL_mean = float(cake_logL_radial(np.linalg.norm(reference, axis=1), 5, widths=BATTERY_CAKE_WIDTHS).mean())
    logL_mean = float(run['cold_logLs'].mean())
    report.stats['logL_mean'] = logL_mean
    if abs(logL_mean - ref_logL_mean) > 1.5:
        report.violations.append(f'cold logL mean {logL_mean:.2f} vs reference {ref_logL_mean:.2f} (band 1.5 nats)')

    # ladder structure against the analytic entropy profile
    betas_profile, s_profile = _battery_entropy_profile()
    finite_Ts = run['final_Ts'][np.isfinite(run['final_Ts'])]
    report.merge(ladder_entropy_gates(finite_Ts, betas_profile, s_profile, tip_max_nats=2.5, link_max_nats=2.5))
    return report


@pytest.mark.slow
@pytest.mark.parametrize('seed', CONVERGENCE_BATTERY_SEEDS)
def test_adaptive_cake_battery_recovers_posterior(tmp_path, seed: int) -> None:
    """Acceptance 2: the cake battery freezes and passes reference gates."""
    reset_seed_guard_for_tests()
    data = adaptive_spec_data(
        'adaptive_cake_battery',
        seed,
        {'name': 'cake', 'n_par': 5, 'cutoff': 10, 'widths': list(BATTERY_CAKE_WIDTHS)},
        n_chain=48,
        block_size=1024,
        n_blocks=BATTERY_N_BLOCKS,
        budget_blocks=BATTERY_BUDGET_BLOCKS,
        store_thin=BATTERY_STORE_THIN,
    )
    spec = RunSpec.from_dict(data)
    artifact_path = run_from_spec(spec, tmp_path)
    reset_seed_guard_for_tests()

    run = load_post_freeze(artifact_path)
    assert_readout_structure(run)
    assert run['n_applied'] >= 6

    report = _cake_battery_gates(run, _battery_reference(), seed)
    assert report.passed, (seed, report.violations, report.stats)


@pytest.mark.slow
def test_adaptive_cake_battery_whole_run_control(tmp_path) -> None:
    """Old-behavior control: a whole-run DE buffer warns and passes the same gates.

    The ring-buffer battery above certifies the production regime (the
    buffer turns over — and forgets adaptation burn-in — well before the
    post-freeze readout window); this control documents that the
    never-forgetting whole-run configuration draws the harness warning
    and, on this target, does not change the verdict.
    """
    reset_seed_guard_for_tests()
    data = adaptive_spec_data(
        'adaptive_cake_whole_run',
        555,
        {'name': 'cake', 'n_par': 5, 'cutoff': 10, 'widths': list(BATTERY_CAKE_WIDTHS)},
        n_chain=48,
        block_size=1024,
        n_blocks=BATTERY_N_BLOCKS,
        budget_blocks=BATTERY_BUDGET_BLOCKS,
        store_thin=BATTERY_STORE_THIN,
        de_window_blocks=None,
    )
    spec = RunSpec.from_dict(data)
    with pytest.warns(UserWarning, match='never forgets burn-in'):
        artifact_path = run_from_spec(spec, tmp_path)
    reset_seed_guard_for_tests()

    run = load_post_freeze(artifact_path)
    assert_readout_structure(run)
    report = _cake_battery_gates(run, _battery_reference(), 555)
    assert report.passed, (report.violations, report.stats)


@pytest.mark.slow
@pytest.mark.usefixtures('fresh_seed_guard')
def test_adaptive_acceptance_mode_realizes_equal_exchange_rates() -> None:
    """The adaptive acceptance-mode ladder realizes ~equal NN exchange rates.

    Closes the predicted->realized loop for the ADAPTIVE acceptance path
    (the fixed-ladder form lives in test_metrics). cold_cap_links=0
    isolates the spacing rule — the cold-edge cap deliberately overrides
    it and would tighten the coldest links past the equal-acceptance
    target — and T_min_factor=1 keeps the readout pin at the cold edge so
    no interior link is plug-distorted. The realized rates are measured
    over the final tracker segment, which runs entirely on the frozen
    geometry. Calibration: observed interior spread 0.031 at this seed;
    the 0.12 band matches the fixed-ladder realized-equality test (its
    equal ladder measured 0.059 vs 0.487 for the lopsided control).
    """
    data = adaptive_spec_data(
        'acceptance_flatness',
        778,
        {'name': 'gaussian', 'n_par': 4, 'cutoff': 5},
        n_chain=8,
        block_size=256,
        n_blocks=160,
        budget_blocks=120,
        t_min_factor=1.0,
        mode='acceptance',
    )
    data['adaptive']['cold_cap_links'] = 0
    spec = RunSpec.from_dict(data)
    assert spec.adaptive is not None
    seed_run(spec.seed)
    controller = build_adaptive_controller(spec.adaptive)
    like_obj = build_likelihood(spec)
    sampler, _ = build_sampler(
        spec, like_obj=like_obj, T_ladder=controller.initial_ladder(like_obj, spec.n_chain, spec.n_cold)
    )
    for _ in range(spec.n_blocks):
        sampler.advance_block()
        controller.post_block(sampler)

    assert controller.frozen, 'acceptance-mode adaptation must freeze within budget'
    ladder = sampler.T_ladder
    assert isinstance(ladder, AcceptanceTemperatureLadder), (
        'cap disabled: the frozen ladder is the raw acceptance ladder'
    )

    _full, rates, _total = sampler.tracker_manager.get_exchange_rate_summary(0)
    # chains 1..5: both NN links lie among the finite spaced rungs
    # (n_chain=8 with one inf rung; chain 6's upper link reaches the inf edge)
    interior = np.asarray(rates[1:6], dtype=np.float64)
    assert np.all(np.isfinite(interior))
    assert interior.max() - interior.min() < 0.12, interior
    assert np.all(np.abs(interior - ladder.achieved_acceptance) < 0.12), (interior, ladder.achieved_acceptance)


def _warm_started_gold_sampler(seed: int, n_blocks: int, de_size: int):
    """File-based cake ladder with every chain warm-started at posterior draws."""
    data: dict[str, Any] = {
        'name': 'gold_warm',
        'seed': seed,
        'likelihood': {'name': 'cake', 'n_par': 5, 'cutoff': 10},
        'ladder': {
            'kind': 'entropy_file',
            'n_chain': 12,
            'n_cold': 1,
            'Ts_file': 'data/Ts_cake_gold.npy',
            'vars_file': 'data/vars_cake_gold.npy',
        },
        'run': {'n_steps': 512 * n_blocks, 'block_size': 512, 'store_thin': 16, 'checkpoint_every_blocks': n_blocks},
        'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
        'proposals': {'FisherJumpManager': {'verbose_fisher': False}, 'DEJumpManager': {'de_size': de_size}},
    }
    spec = RunSpec.from_dict(data)
    seed_run(spec.seed)
    sampler, _like_obj = build_sampler(spec)
    like_obj = CakeLikelihood(n_par=5, cutoff=10)
    warm = draw_cake(sampler.n_chain, 5, get_rng(1000 + seed))
    for itrt in range(sampler.n_chain):
        sampler.samples[0, itrt, :] = warm[itrt]
        sampler.logLs[0, itrt] = like_obj.get_loglike(np.ascontiguousarray(warm[itrt]))
    return spec, sampler


@pytest.mark.slow
@pytest.mark.parametrize('seed', [555, 556])
def test_gold_ladder_warm_start_preserves_target(seed: int) -> None:
    """A warm-started file-based cake ladder should preserve broad tier statistics."""
    reset_seed_guard_for_tests()
    n_blocks = 640
    spec, sampler = _warm_started_gold_sampler(seed, n_blocks, de_size=512 * n_blocks)
    for _ in range(spec.n_blocks):
        sampler.advance_block()
    reset_seed_guard_for_tests()

    second_half = sampler.samples_store[sampler.store_size // 2 :, 0, :]
    r2 = (second_half**2).sum(axis=1)
    narrow_mask = r2 < 1.0
    n_flips = int(np.count_nonzero(np.diff(narrow_mask.astype(np.int8))))

    assert 1.0 <= float(r2.mean()) <= 13.5, f'E[r^2] {r2.mean():.2f} drifted out of the equilibrium band'
    assert 0.02 <= float(narrow_mask.mean()) <= 0.96, f'narrow fraction {narrow_mask.mean():.3f} collapsed to one tier'
    assert n_flips >= 50, f'only {n_flips} tier flips: tiers stopped interconverting'


@pytest.mark.slow
def test_small_de_buffer_fails_posterior_gate() -> None:
    """Negative control: a very short DE buffer fails this posterior gate."""
    reset_seed_guard_for_tests()
    n_blocks = 320
    with pytest.warns(UserWarning, match='DE buffer memory'):
        spec, sampler = _warm_started_gold_sampler(555, n_blocks, de_size=256)
    for _ in range(spec.n_blocks):
        sampler.advance_block()
    reset_seed_guard_for_tests()

    second_half = sampler.samples_store[sampler.store_size // 2 :, 0, :]
    r2 = (second_half**2).sum(axis=1)
    narrow_frac = float((r2 < 1.0).mean())

    # the preservation bands from test_gold_ladder_warm_start_preserves_target
    assert float(r2.mean()) < 1.0, 'the short-buffer negative control no longer fails as expected'
    assert narrow_frac > 0.96


@pytest.mark.slow
def test_adaptive_default_cake_structural(tmp_path) -> None:
    """Default cake: adaptation completes and the readout stays nondegenerate."""
    reset_seed_guard_for_tests()
    data = adaptive_spec_data(
        'adaptive_cake_structural',
        558,
        {'name': 'cake', 'n_par': 5, 'cutoff': 10},
        n_chain=12,
        block_size=512,
        n_blocks=BATTERY_N_BLOCKS,
        budget_blocks=BATTERY_BUDGET_BLOCKS,
        store_thin=BATTERY_STORE_THIN,
    )
    spec = RunSpec.from_dict(data)
    artifact_path = run_from_spec(spec, tmp_path)
    reset_seed_guard_for_tests()

    run = load_post_freeze(artifact_path)
    assert_readout_structure(run)
    assert run['n_applied'] >= 5

    r2 = (run['cold'] ** 2).sum(axis=1)
    narrow_frac = float((r2 < 1.0).mean())
    assert narrow_frac <= 0.98, 'spike-tier collapse: the readout no longer leaves the spike'
    assert 0.05 <= float(r2.mean()) <= 15.5


@pytest.mark.usefixtures('fresh_seed_guard')
def test_freeze_requires_coupling_witness(monkeypatch) -> None:
    """Acceptance-adjacent: rebuild stability alone must not freeze.

    Identical configuration and streams to
    test_post_freeze_bit_exact_equivalence, which freezes via the
    criterion — but with the coupling witness suppressed (a pure
    observer, so RNG streams are unchanged) the controller must keep
    holding.
    """
    spec = make_tiny_spec(n_steps=64 * 40, block_size=64)
    seed_run(spec.seed)
    monkeypatch.setattr(TrackerManager, 'n_cycles', property(lambda self: np.zeros(self.n_chain, dtype=np.int64)))

    controller = AdaptiveLadderController(
        mode='entropy', update_every_blocks=4, freeze_criterion=(0.08, 2), budget_blocks=10**6
    )
    like_obj = build_likelihood(spec)
    sampler, _ = build_sampler(
        spec, like_obj=like_obj, T_ladder=controller.initial_ladder(like_obj, spec.n_chain, spec.n_cold)
    )
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
    """A stale round-trip count must not satisfy every freeze window.

    n_cycles is patched to a nonzero CONSTANT: the open-segment
    witness (the plan's floor) is green throughout, but no NEW trips
    ever arrive between evaluations. The per-window witness therefore
    never assembles a freeze streak.
    """
    spec = make_tiny_spec(n_steps=64 * 40, block_size=64)
    seed_run(spec.seed)
    monkeypatch.setattr(TrackerManager, 'n_cycles', property(lambda self: np.ones(self.n_chain, dtype=np.int64)))

    controller = AdaptiveLadderController(
        mode='entropy', update_every_blocks=4, freeze_criterion=(0.08, 2), budget_blocks=10**6
    )
    like_obj = build_likelihood(spec)
    sampler, _ = build_sampler(
        spec, like_obj=like_obj, T_ladder=controller.initial_ladder(like_obj, spec.n_chain, spec.n_cold)
    )
    for _ in range(spec.n_blocks):
        sampler.advance_block()
        controller.post_block(sampler)

    assert not controller.frozen
    assert controller.frozen_by == ''


@pytest.mark.usefixtures('fresh_seed_guard')
def test_budget_freeze_records_reason() -> None:
    """A run exhausting budget_blocks unfrozen hard-freezes with frozen_by='budget'."""
    spec = make_tiny_spec(n_steps=64 * 8, block_size=64)
    seed_run(spec.seed)
    controller = AdaptiveLadderController(mode='entropy', update_every_blocks=8, budget_blocks=4)
    like_obj = build_likelihood(spec)
    sampler, _ = build_sampler(
        spec, like_obj=like_obj, T_ladder=controller.initial_ladder(like_obj, spec.n_chain, spec.n_cold)
    )
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
    """Occupied-mode counts among cold-slot DE-buffer columns survive an update."""
    n_par = 2
    data: dict[str, Any] = {
        key: dict(value) if isinstance(value, dict) else value for key, value in TINY_GAUSSIAN_SPEC.items()
    }
    data['name'] = 'eggbox_gate'
    data['likelihood'] = {'name': 'eggbox', 'n_par': n_par}
    data['run'] = {'n_steps': 64 * 8, 'block_size': 64, 'store_thin': 1, 'checkpoint_every_blocks': 4}
    spec = RunSpec.from_dict(data)
    seed_run(spec.seed)

    hot_ladder = GeometricTemperatureLadder(spec.n_chain, n_cold=1, T_cold=4.0, T_min=4.0, T_max=100.0, n_inf_final=1)
    sampler, _like_obj = build_sampler(spec, T_ladder=hot_ladder)
    for _ in range(spec.n_blocks):
        sampler.advance_block()

    def occupied_modes(buffer_cold_slice: np.ndarray) -> set[tuple[int, ...]]:
        # eggbox maxima sit at x_i = 2 pi k: nearest-mode assignment per row
        rows = buffer_cold_slice.reshape(-1, n_par)
        return {tuple(mode) for mode in np.round(rows / (2.0 * np.pi)).astype(int)}

    de_manager = next(m for m in sampler.proposal_manager.managers if isinstance(m, DEJumpManager))
    n_cold_half = spec.n_chain // 2
    modes_before = occupied_modes(de_manager.de_buffer[:, :n_cold_half, :])

    extended = GeometricTemperatureLadder(spec.n_chain, n_cold=1, T_cold=1.0, T_min=1.0, T_max=100.0, n_inf_final=1)
    sampler.apply_ladder_update(extended, 'at_or_hotter')
    modes_after = occupied_modes(de_manager.de_buffer[:, :n_cold_half, :])

    assert len(modes_before) > 1, 'test premise: the cold columns must start multimodal'
    assert len(modes_after) >= len(modes_before)


def test_adaptive_spec_validation() -> None:
    """[adaptive] rejects unknown keys, missing budget_blocks, and out-of-range T_min_factor."""
    base: dict[str, Any] = {
        key: dict(value) if isinstance(value, dict) else value for key, value in TINY_GAUSSIAN_SPEC.items()
    }
    base['adaptive'] = {'mode': 'entropy', 'budget_blocks': 8}
    RunSpec.from_dict(base)

    # sub-unit T_min_factor is supported now that storage is index-based
    good = dict(base)
    good['adaptive'] = {
        'mode': 'entropy',
        'budget_blocks': 8,
        'T_min_factor': 0.9,
        'remap_rule': 'no_remap',
        'window_extension_factor': 3.0,
        'ds_link_cap': 2.0,
        'cold_cap_links': 5,
        'cap_ratio_min': 1.02,
        'cap_ratio_max': 1.5,
        'var_history_length': 6,
        'pool_dlog_tol': 0.015,
        'discard_blocks_after_update': 2,
        'min_updates_at_target': 4,
    }
    RunSpec.from_dict(good)

    bad_tables = (
        {'mode': 'entropy', 'budget_blocks': 8, 'forgeting': 0.15},  # typo'd key must fail loudly
        {'mode': 'entropy'},  # budget_blocks is required
        {
            'mode': 'entropy',
            'budget_blocks': 8,
            'T_min_factor': 1.5,
        },  # target is a multiple of T=1, capped at the readout
        {'mode': 'entropy', 'budget_blocks': 8, 'T_min_factor': 0.0},  # and must be positive
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
    with pytest.raises(ValueError, match='unknown remap_rule'):
        AdaptiveLadderController(mode='entropy', budget_blocks=8, remap_rule='nonsense')


class _StubLikelihood:
    """Deterministic AbstractLikelihood stand-in (no RNG draws)."""

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

    @property
    def n_cycles(self) -> np.ndarray:
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
    Ts_fixed = np.array([1.0, 2.0, 4.0, 8.0, 16.0, np.inf])
    low = [0.5, 0.5, 0.5, 0.5, 0.5, 0.25]
    high_cold = [50.0, 0.5, 0.5, 0.5, 0.5, 0.25]

    controllers = {}
    for estimator in (0, 1):
        controller = AdaptiveLadderController(
            mode='entropy',
            update_every_blocks=1,
            forgetting=0.0,
            freeze_criterion=(1.0e9, 10**6),
            budget_blocks=10**6,
            var_estimator=estimator,
            discard_blocks_after_update=0,
        )
        # huge freeze_dlog holds every rebuild, so the stub's temperatures
        # recur every segment — the regime where the estimators differ
        controller.initial_ladder(cast('AbstractLikelihood', _StubLikelihood()), Ts_fixed.size, 1)
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


def test_pool_tolerance_preserves_variance_history() -> None:
    """Rungs drifting within pool_dlog_tol keep feeding one pool row.

    Exact float keying spawned a fresh length-1 variance history at
    every applied rebuild, so the pessimistic max degenerated to the
    latest estimate precisely during active adaptation. With tolerance
    matching, a rung drifting 0.5% between segments accumulates history;
    with the tolerance disabled it splits into separate rows.
    """
    for tol, expect_merged in ((0.02, True), (0.0, False)):
        controller = AdaptiveLadderController(
            mode='entropy',
            update_every_blocks=1,
            freeze_criterion=(1.0e9, 10**6),
            budget_blocks=10**6,
            pool_dlog_tol=tol,
            discard_blocks_after_update=0,
        )
        controller.initial_ladder(cast('AbstractLikelihood', _StubLikelihood()), 6, 1)
        stub = _StubSampler(np.array([1.0, 2.0, 4.0, 8.0, 16.0, np.inf]))
        sampler = cast('DTMCMCSampler', stub)
        stub.feed_block([0.5, 0.5, 0.5, 0.5, 0.5, 0.25])
        controller.post_block(sampler)
        # the coldest rung drifts 0.5% (well under converged link spacing)
        stub.Ts = np.array([1.005, 2.0, 4.0, 8.0, 16.0, np.inf])
        stub.feed_block([9.0, 0.5, 0.5, 0.5, 0.5, 0.25])
        controller.post_block(sampler)

        cold_rows = [idx for idx, T_loc in enumerate(controller._pool_Ts) if np.isfinite(T_loc) and T_loc < 1.5]
        histories = [controller._pool_var_history[idx] for idx in cold_rows]
        if expect_merged:
            assert len(cold_rows) == 1
            assert len(histories[0]) == 2
            assert max(histories[0]) == 9.0
        else:
            assert len(cold_rows) == 2
            assert all(len(history) == 1 for history in histories)


def test_discard_blocks_after_update_drops_transients() -> None:
    """The head of each segment is excluded from the pooled statistics.

    The remap leaves chains equilibrated to their previous temperatures,
    so the first post-update block measures a transient. A cadence-2
    controller with discard 1 must pool only the second block of each
    segment, including the first segment.
    """
    controller = AdaptiveLadderController(
        mode='entropy',
        update_every_blocks=2,
        freeze_criterion=(1.0e9, 10**6),
        budget_blocks=10**6,
        discard_blocks_after_update=1,
        pool_dlog_tol=0.0,
    )
    controller.initial_ladder(cast('AbstractLikelihood', _StubLikelihood()), 6, 1)
    stub = _StubSampler(np.array([1.0, 2.0, 4.0, 8.0, 16.0, np.inf]))
    sampler = cast('DTMCMCSampler', stub)
    stub.feed_block([999.0, 999.0, 999.0, 999.0, 999.0, 999.0])  # transient: must not pool
    controller.post_block(sampler)
    stub.feed_block([0.5, 0.5, 0.5, 0.5, 0.5, 0.25])
    controller.post_block(sampler)

    coldest_idx = int(np.argmin([T_loc if np.isfinite(T_loc) else np.inf for T_loc in controller._pool_Ts]))
    assert controller._pool_vars[coldest_idx] == 0.5
    assert max(controller._pool_var_history[coldest_idx]) == 0.5


def test_resolve_cap_links_scales_with_chain_count() -> None:
    """The auto cold-cap link count scales with the ladder instead of a fixed 3."""
    controller = AdaptiveLadderController(mode='entropy', budget_blocks=8)
    assert controller._resolve_cap_links(12, 1) == 3  # historical floor at battery scale
    assert controller._resolve_cap_links(64, 1) == 15
    explicit = AdaptiveLadderController(mode='entropy', budget_blocks=8, cold_cap_links=7)
    assert explicit._resolve_cap_links(64, 1) == 7


def test_cap_cold_links_skips_readout_pin_and_can_disable() -> None:
    """The cap never moves a rung pinned at T_cold, and cold_cap_links=0 disables it."""
    controller = AdaptiveLadderController(
        mode='entropy',
        budget_blocks=8,
        cap_ratio_bounds=(1.01, 1.02),
    )
    # pool rows with tiny variance make the measured cap ratio clip to its
    # minimum, so every non-pinned capped link must move
    for T_loc, var_loc in ((0.9, 1.0e-6), (1.0, 1.0e-6), (4.0, 1.0e-6), (64.0, 1.0e-6)):
        controller._pool_Ts.append(T_loc)
        controller._pool_means.append(0.0)
        controller._pool_vars.append(var_loc)
        controller._pool_weights.append(1.0)
        controller._pool_var_history.append([var_loc])

    ladder = TemperatureLadder(np.array([0.9, 1.0, 2.0, 4.0, 16.0, np.inf]), T_cold=1.0, n_cold=1)
    capped = controller._cap_cold_links(ladder, 1)
    capped_Ts = np.asarray(capped.Ts)
    # the readout pin is untouched even though its link violates the cap
    assert 1.0 in capped_Ts
    assert capped.get_arg_cold().size == 1
    # the link above the pin was pulled down to the clipped ratio
    assert capped_Ts[2] <= 1.0 * 1.02 + 1.0e-12
    # the pinned link consumes no cap slot: all three resolved cap links
    # land on real links, so the third one (ending at the 16 rung) is capped
    assert capped_Ts[4] <= capped_Ts[3] * 1.02 + 1.0e-12
    # disabling the cap returns the ladder unchanged
    disabled = AdaptiveLadderController(mode='entropy', budget_blocks=8, cold_cap_links=0)
    assert disabled._cap_cold_links(ladder, 1) is ladder


@pytest.mark.usefixtures('fresh_seed_guard')
def test_adaptive_t_min_factor_pins_readout_with_subcold_rungs(tmp_path) -> None:
    """T_min_factor < 1 extends the ladder below T=1 while recording stays at T=1.

    End-to-end on the tiny Gaussian: after the window reaches its
    sub-unit target the readout chains are pinned at exactly T=1 with
    rungs below them, and the artifact's record map points at the T=1
    chains rather than the coldest ones.
    """
    data: dict[str, Any] = {
        key: dict(value) if isinstance(value, dict) else value for key, value in TINY_GAUSSIAN_SPEC.items()
    }
    data['name'] = 'tmin_gaussian'
    data['ladder'] = {'kind': 'geometric', 'n_chain': 8, 'n_cold': 1}
    data['run'] = {'n_steps': 64 * 24, 'block_size': 64, 'store_thin': 1, 'checkpoint_every_blocks': 8}
    data['adaptive'] = {
        'mode': 'entropy',
        'update_every_blocks': 2,
        'freeze_dlog': 0.02,
        'freeze_consecutive': 3,
        'budget_blocks': 20,
        'T_min_factor': 0.9,
    }
    spec = RunSpec.from_dict(data)

    artifact_path = run_from_spec(spec, tmp_path)

    with h5py.File(str(artifact_path), 'r') as hf:
        final_Ts = np.asarray(hf['ladder/Ts'])
        record_indices = np.asarray(hf['store/record_indices'])
        history_indices = np.asarray(hf['store/record_history_indices'])
        t_cold_window = np.asarray(hf['ladder/history/t_cold_window'])

    assert float(t_cold_window[-1]) <= 0.9 + 1.0e-12, 'window must reach its sub-unit target within budget'
    finite_Ts = final_Ts[np.isfinite(final_Ts)]
    assert np.min(finite_Ts) < 1.0, 'sub-readout rungs must be present'
    assert record_indices.size == 1
    assert final_Ts[record_indices[0]] == 1.0, 'the recorded chain is the T=1 readout, not the coldest rung'
    assert record_indices[0] > 0, 'the readout chain sits interior to the sorted ladder'
    # the record map tracked the move off index 0 when sub-cold rungs appeared
    assert history_indices[0][0] == 0


# reuse the shared guard fixture from the harness tests
from tests.test_harness import fresh_seed_guard  # noqa: E402, F401
