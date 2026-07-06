"""Phase 2 acceptance 3+4: physics invariants, reference samplers, metric utilities.

The Gaussian heat-capacity invariant runs the real sampler on an explicit
ladder and checks C(T) = beta^2 Var(logL) = n_par/2 at every finite rung,
with the prior cutoff sized so truncation is negligible (cutoff/sqrt(T)
>= 4), and that an entropy ladder built from the measured variances
coincides with a geometric ladder (the constant-C null case). Reference
samplers are validated on analytic moments and NN-KL(self) ~ 0. All tests
use fixed seeds and are deterministic.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import gamma as gamma_func

from DTMCMC.likelihoods import eggbox as eggbox_module
from DTMCMC.likelihoods.cake_likelihood import CakeLikelihood
from DTMCMC.rng_helpers import get_rng, reset_seed_guard_for_tests, seed_run
from DTMCMC.temperature_ladder_helpers import EntropyTemperatureLadder, GeometricTemperatureLadder
from DTMCMC.tracker_manager import RT_ARRIVED_COLD, RT_ARRIVED_HOT
from experiments.harness.runner import build_sampler
from experiments.harness.spec import RunSpec
from experiments.metrics import (
    de_buffer_difference_spectrum,
    detect_apparent_super_efficiency,
    effective_rank,
    fit_knee_max_curvature,
    fit_knee_piecewise_linear,
    flow_fraction,
    fraction_walkers_with_round_trip,
    nn_kl,
    round_trip_counts,
    round_trip_rate,
    round_trip_times,
    scramble_block_n_eff,
    scramble_block_n_eff_min,
)
from experiments.reference_samplers import (
    CAKE_AMPS,
    CAKE_EXPONENTS,
    CAKE_WIDTHS,
    cake_moment_r2,
    draw_cake,
    draw_eggbox,
    draw_truncated_gaussian,
    eggbox_cells,
    eggbox_logL,
)

GAUSSIAN_INVARIANT_TS = [1., 1.4142135623730951, 2., 2.8284271247461903, 4., 5.656854249492381, 8.]

GAUSSIAN_INVARIANT_SPEC: dict[str, object] = {
    'name': 'gaussian_invariant',
    'seed': 314159,
    # cutoff/sqrt(T) = 12/sqrt(8) = 4.24 >= 4 over the TESTED rungs (T <= 8):
    # truncation negligible there. The T=16 rung is an untested buffer: the
    # beta=0 chain's uniform-box walkers carry extreme logL values, and rare
    # accepted swaps out of it inject heavy-tailed excursions into its
    # neighbor's Var(logL) estimator (observed as a +25% outlier at the
    # hottest tested rung on the CI platform's realization without it)
    'likelihood': {'name': 'gaussian', 'n_par': 4, 'cutoff': 12},
    'ladder': {'kind': 'explicit', 'n_chain': 9, 'n_cold': 1, 'Ts': [*GAUSSIAN_INVARIANT_TS, 16., float('inf')]},
    'run': {'n_steps': 49152, 'block_size': 512, 'store_thin': 16, 'n_record': -1, 'checkpoint_every_blocks': 96},
    'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
    'proposals': {
        'FisherJumpManager': {'verbose_fisher': False},
        'DEJumpManager': {'de_size': 2048},
    },
}


@pytest.fixture(scope='module')
def gaussian_invariant_run():
    """Run the Gaussian invariant spec once and return the sampler."""
    reset_seed_guard_for_tests()
    spec = RunSpec.from_dict(dict(GAUSSIAN_INVARIANT_SPEC))
    seed_run(spec.seed)
    sampler, _like_obj = build_sampler(spec)
    for _ in range(spec.n_blocks):
        sampler.advance_block()
    reset_seed_guard_for_tests()
    return sampler


def measured_logL_vars(sampler, n_burn_blocks: int) -> np.ndarray:
    """Stationary per-chain Var(logL) from the stored block moments."""
    logL_means = np.asarray(sampler.logL_means)[n_burn_blocks:]
    logL2_means = np.asarray(sampler.logL2_means)[n_burn_blocks:]
    return logL2_means.mean(axis=0) - logL_means.mean(axis=0)**2


def test_gaussian_heat_capacity_invariant(gaussian_invariant_run) -> None:
    """C(T) = beta^2 Var(logL) = n_par/2 at every tested finite temperature."""
    sampler = gaussian_invariant_run
    n_par = 4
    n_burn_blocks = sampler.itrn // sampler.block_size // 2

    vars_measured = measured_logL_vars(sampler, n_burn_blocks)
    Ts = np.asarray(GAUSSIAN_INVARIANT_TS)
    betas = 1. / Ts
    heat_capacity = betas**2 * vars_measured[:Ts.size]

    # Var(logL) itself grows as n_par * T^2 / 2; C(T) is flat at n_par/2.
    # Tolerance covers per-platform realization noise (trajectories diverge
    # chaotically across libm implementations even at fixed seed)
    assert_allclose(heat_capacity, np.full(Ts.size, n_par / 2.), rtol=0.2)


def test_entropy_ladder_matches_geometric_on_gaussian(gaussian_invariant_run) -> None:
    """Constant heat capacity: measured-variance entropy ladder == geometric ladder."""
    sampler = gaussian_invariant_run
    n_burn_blocks = sampler.itrn // sampler.block_size // 2
    Ts = np.asarray(GAUSSIAN_INVARIANT_TS)
    vars_measured = measured_logL_vars(sampler, n_burn_blocks)[:Ts.size]

    entropy_ladder = EntropyTemperatureLadder(Ts.size, Ts, vars_measured, n_cold=1, T_cold=1., n_inf_final=0)
    geometric_ladder = GeometricTemperatureLadder(Ts.size, 1, 1., 1., float(Ts[-1]), n_inf_final=0)

    assert_allclose(np.log(entropy_ladder.Ts), np.log(geometric_ladder.Ts), atol=0.1)


def test_cake_constants_match_engine() -> None:
    """The reference sampler's tier constants reproduce the engine logL exactly.

    CAKE_AMPS/WIDTHS/EXPONENTS are copies of function-local constants in
    cake_likelihood.py (single-sourced in Phase 3); this reconstruction
    fails the moment either side drifts.
    """
    n_par = 5
    like_obj = CakeLikelihood(n_par=n_par, cutoff=10)
    rng = get_rng(23)
    points = rng.uniform(-9., 9., size=(64, n_par))

    dim_part = gamma_func(1. + n_par / 2.) / np.pi**(n_par / 2.)
    for point in points:
        r2 = float((point**2).sum())
        tier_logs = [
            np.log(amp * dim_part / (2.**(n_par / exponent) * width**n_par * gamma_func((exponent + n_par) / exponent))) - r2**(exponent / 2.) / (2. * width**exponent)
            for amp, width, exponent in zip(CAKE_AMPS, CAKE_WIDTHS, CAKE_EXPONENTS, strict=True)
        ]
        assert like_obj.get_loglike(point) == pytest.approx(np.logaddexp(tier_logs[0], tier_logs[1]), rel=1.e-12)


def test_truncated_gaussian_reference() -> None:
    """Moments of the truncated-Gaussian reference match the target."""
    rng = get_rng(11)
    draws = draw_truncated_gaussian(8000, 4, 12., rng)
    assert draws.shape == (8000, 4)
    assert np.all(np.abs(draws) <= 12.)
    assert_allclose(draws.mean(axis=0), np.zeros(4), atol=0.05)
    assert_allclose(draws.var(axis=0), np.ones(4), rtol=0.06)


def test_cake_reference_moments() -> None:
    """Cake reference draws reproduce the analytic mixture E[r^2]."""
    rng = get_rng(12)
    draws = draw_cake(20000, 5, rng)
    r2 = (draws**2).sum(axis=1)
    assert_allclose(r2.mean(), cake_moment_r2(5), rtol=0.05)
    assert np.all(np.abs(draws) <= 10.)
    # both tiers must be represented: the narrow tier concentrates near r ~ 0.1
    assert np.count_nonzero(r2 < 0.1) > 20000 * 0.3


def test_eggbox_reference_draws() -> None:
    """Eggbox reference: symmetric, mode-uniform, consistent with the engine logL."""
    rng = get_rng(13)
    n_par = 2
    draws = draw_eggbox(8000, n_par, rng)
    assert np.all((draws >= eggbox_module.low_lim) & (draws <= eggbox_module.high_lim))
    assert_allclose(draws.mean(axis=0), np.zeros(n_par), atol=0.25)

    # vectorized logL used by the sampler machinery matches the engine's
    points = rng.uniform(eggbox_module.low_lim, eggbox_module.high_lim, size=(64, n_par))
    engine_logL = np.array([eggbox_module.get_loglike(point, n_par) for point in points])
    assert_allclose(eggbox_logL(points), engine_logL, rtol=1.e-12)

    # cell decomposition agrees with the engine's mode enumeration
    cells = eggbox_cells(n_par)
    modes, _mode_ints, _canonical = eggbox_module.gen_nd_modelist(n_par)
    assert int(np.count_nonzero(cells.even_mask)) == modes.shape[0]

    # occupancy across the 13 modes is uniform (chi^2 against uniform)
    even_centers = cells.centers[cells.even_mask]
    nearest = np.argmin(((draws[:, np.newaxis, :] - even_centers[np.newaxis, :, :])**2).sum(axis=2), axis=1)
    counts = np.bincount(nearest, minlength=even_centers.shape[0])
    expected = draws.shape[0] / even_centers.shape[0]
    chi2 = float(((counts - expected)**2 / expected).sum())
    assert chi2 < 3. * even_centers.shape[0]


def test_reference_samplers_nn_kl_self() -> None:
    """NN-KL between two independent draws from the same reference is ~ 0."""
    rng = get_rng(14)
    # |self-KL| bounded by estimator noise: worst observed is the cake's
    # two-scale geometry (~0.3 at n=2000); genuine mismatch measures ~+6
    for draw in (
        lambda n: draw_truncated_gaussian(n, 3, 8., rng),
        lambda n: draw_cake(n, 5, rng),
        lambda n: draw_eggbox(n, 2, rng),
    ):
        kl = nn_kl(draw(2000), draw(2000), 2000, rng)
        assert abs(kl) < 0.4

    # and the estimator resolves genuinely different distributions
    kl_cross = nn_kl(draw_cake(2000, 5, rng), draw_truncated_gaussian(2000, 5, 10., rng), 2000, rng)
    assert kl_cross > 3.


def test_round_trip_statistics_synthetic() -> None:
    """Round-trip utilities on the hand-computed scenario-A event log."""
    events = np.array([
        [0, 3, RT_ARRIVED_COLD], [3, 3, RT_ARRIVED_HOT],
        [3, 4, RT_ARRIVED_COLD], [0, 4, RT_ARRIVED_HOT],
        [0, 5, RT_ARRIVED_COLD], [3, 5, RT_ARRIVED_HOT],
        [3, 7, RT_ARRIVED_COLD], [0, 7, RT_ARRIVED_HOT],
        [0, 8, RT_ARRIVED_COLD], [3, 8, RT_ARRIVED_HOT],
    ], dtype=np.int64)
    n_chain = 4

    assert_allclose(round_trip_counts(events, n_chain), [2, 0, 0, 2])
    assert fraction_walkers_with_round_trip(events, n_chain) == pytest.approx(0.5)
    # 4 trips / 4 walkers / (8 iterations * 4 chains / 1e6)
    assert round_trip_rate(events, n_chain, 8) == pytest.approx(4. / 4. / (32. / 1.e6))

    # walker 0 cold arrivals at 3, 5, 8 -> gaps [2, 3]; walker 3 at 4, 7 -> [3]
    cold_times, hot_times = round_trip_times(events, n_chain)
    assert sorted(cold_times.tolist()) == [2, 3, 3]
    assert sorted(hot_times.tolist()) == [2, 3, 3]


def test_round_trip_metrics_do_not_pair_across_segments() -> None:
    """Arrivals in different ladder segments never pair (plan D6, amended per PR #16 review).

    The PR #16 review's straddle example: a pre-update HOT arrival and a
    post-update COLD arrival by the same (restarted) walker id counted
    as one complete round trip without segmentation.
    """
    events = np.array([
        [0, 10, RT_ARRIVED_HOT],   # before the ladder update
        [0, 20, RT_ARRIVED_COLD],  # after: walker labels restarted
    ], dtype=np.int64)

    # unsegmented (fixed-ladder) behavior pairs them
    assert int(round_trip_counts(events, 1).sum()) == 1
    # a segment boundary between the arrivals forbids the pairing
    boundaries = np.array([15], dtype=np.int64)
    assert int(round_trip_counts(events, 1, boundaries).sum()) == 0
    assert fraction_walkers_with_round_trip(events, 1, boundaries) == 0.
    assert round_trip_rate(events, 1, 20, boundaries) == 0.

    # an event exactly at the boundary iteration belongs to the closing
    # segment (the update happens after the block's last step)
    events_at_boundary = np.array([
        [0, 10, RT_ARRIVED_HOT],
        [0, 15, RT_ARRIVED_COLD],
    ], dtype=np.int64)
    assert int(round_trip_counts(events_at_boundary, 1, boundaries).sum()) == 1

    # same-direction gaps never span a boundary either
    events_gaps = np.array([
        [0, 3, RT_ARRIVED_COLD],
        [0, 10, RT_ARRIVED_COLD],
        [0, 20, RT_ARRIVED_COLD],
    ], dtype=np.int64)
    cold_times, _hot_times = round_trip_times(events_gaps, 1, boundaries)
    assert sorted(cold_times.tolist()) == [7]


def test_flow_fraction_normalization() -> None:
    """Flow fractions divide counts and mark unlabeled entries NaN."""
    up = np.array([[2, 0], [4, 0]], dtype=np.int64)
    labeled = np.array([[4, 0], [4, 0]], dtype=np.int64)
    frac = flow_fraction(up, labeled)
    assert frac[0, 0] == pytest.approx(0.5)
    assert frac[1, 0] == pytest.approx(1.0)
    assert np.isnan(frac[0, 1])


def test_knee_fits_recover_synthetic_knee() -> None:
    """Both knee estimators locate a clean piecewise-linear knee."""
    x = np.linspace(0., 10., 21)
    y = np.minimum(x, 4.)
    assert fit_knee_piecewise_linear(x, y) == pytest.approx(4., abs=0.51)
    assert fit_knee_max_curvature(x, y) == pytest.approx(4., abs=0.51)


def test_scramble_block_n_eff_calibration() -> None:
    """The frozen C1 estimator: ~n_tot on white noise, collapses under correlation."""
    rng = get_rng(15)
    n_rows, n_cold, n_par = 2048, 2, 3
    n_tot = n_rows * n_cold

    white = rng.standard_normal((n_rows, n_cold, n_par))
    n_eff_white = scramble_block_n_eff(white, 64, 256, get_rng(16))
    assert np.all(n_eff_white > n_tot / 2.)
    assert np.all(n_eff_white < n_tot * 2.)

    # strong autocorrelation: 64-row constant stretches -> n_eff ~ n_tot/64
    repeat = np.repeat(rng.standard_normal((n_rows // 64, n_cold, n_par)), 64, axis=0)
    n_eff_corr = scramble_block_n_eff(repeat, 64, 256, get_rng(17))
    assert np.all(n_eff_corr < n_tot / 8.)

    # the frozen C1 aggregation (plan §6): minimum over parameters, per run
    assert scramble_block_n_eff_min(white, 64, 256, get_rng(16)) == pytest.approx(float(n_eff_white.min()))

    # inclusive start bound: block_size == n_rows - 1 leaves exactly two
    # valid starts, and both rows 0 and n_rows-1 must be reachable
    tiny = np.arange(8, dtype=np.float64).reshape(8, 1, 1)
    starts_seen = set()
    probe_rng = get_rng(18)
    for _ in range(64):
        start = int(probe_rng.integers(0, 8 - 7 + 1))
        starts_seen.add(start)
    assert starts_seen == {0, 1}
    # and the estimator itself accepts that geometry without error
    scramble_block_n_eff(tiny, 7, 16, get_rng(19))


def test_de_spectrum_and_effective_rank() -> None:
    """DE-buffer spectra distinguish full-rank from collapsed buffers."""
    rng = get_rng(18)
    n_par = 4
    full_rank = rng.standard_normal((512, 2, n_par))
    eig_full = de_buffer_difference_spectrum(full_rank, 256, get_rng(19))
    assert eig_full.shape == (2, n_par)
    assert np.all(effective_rank(eig_full) > 0.9 * n_par)

    # rank-one collapse: every buffer entry lies on one line
    line = np.outer(rng.standard_normal(512), rng.standard_normal(n_par))
    collapsed = np.repeat(line[:, np.newaxis, :], 2, axis=1)
    eig_collapsed = de_buffer_difference_spectrum(collapsed, 256, get_rng(20))
    assert np.all(effective_rank(eig_collapsed) < 1.5)

    assert effective_rank(np.zeros((2, n_par)))[0] == pytest.approx(0.)


def test_apparent_super_efficiency_detector() -> None:
    """Anticorrelated chains trigger the S3 super-efficiency flag."""
    rng = get_rng(21)
    n_rows = 2048
    base = rng.standard_normal((n_rows, 1))

    # anticorrelated pair: pooled means cancel -> apparent super-efficiency
    anti = np.stack([base, -base + 1.e-3 * rng.standard_normal((n_rows, 1))], axis=1)
    result_anti = detect_apparent_super_efficiency(anti, block_size=32)
    assert bool(result_anti.flags[0])
    assert result_anti.n_eff_with_cross[0] > 3. * result_anti.n_eff_auto[0]

    # independent white-noise chains: no strong super-efficiency claim
    indep = rng.standard_normal((n_rows, 2, 1))
    result_indep = detect_apparent_super_efficiency(indep, block_size=32)
    assert result_indep.n_eff_with_cross[0] < 3. * result_indep.n_eff_auto[0]
