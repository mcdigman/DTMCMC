"""Benchmark registry and likelihood wiring tests.

Every harness likelihood must run end to end through the experiments
architecture, and every benchmark entry's ground truth must be
self-consistent: reference draws reproduce the analytic per-coordinate
moments, respect the likelihood's own prior bounds, and split mass
across modes at the registered weights. The symmetric NN divergence and
the segmented round-trip loader are covered here too.
"""

from typing import Any

import h5py
import numpy as np
import pytest

from DTMCMC.rng_helpers import get_rng, reset_seed_guard_for_tests, seed_run
from DTMCMC.tracker_manager import RT_ARRIVED_COLD, RT_ARRIVED_HOT
from experiments.benchmarks import BENCHMARKS, mode_occupancy
from experiments.harness.artifact import validate
from experiments.harness.runner import LIKELIHOOD_BUILDERS, build_likelihood, run_from_spec
from experiments.harness.spec import LIKELIHOOD_NAMES, RunSpec
from experiments.metrics import nn_divergence_symmetric, nn_kl
from experiments.pilots.common import load_run_metrics
from experiments.pilots.family_compare import summarize_arms

# tiny-but-representative dimensionality per likelihood for smoke runs:
# fixed-dimension targets pin their n_par, flexible ones shrink for speed
SMOKE_PARAMS: dict[str, dict[str, Any]] = {
    'gaussian': {'n_par': 3, 'cutoff': 5},
    'cake': {'n_par': 3, 'cutoff': 10},
    'constant_rectangular': {'n_par': 3},
    'eggbox': {'n_par': 2},
    'hawaii': {},
    'ar1': {'n_par': 6},
    'banana': {'n_par': 4},
    'gaussian_mixture': {'n_par': 3},
    'gaussian_shell': {'n_par': 2},
    'hyperpyramid': {'n_par': 2},
    'random_wheel': {'n_par': 2},
    'rosenbrock': {'n_par': 4},
    'spoke_wheel': {'n_par': 2},
    'uniform_gaussian_prior': {'n_par': 3, 'prior_mean': 1.5, 'prior_std': 0.75},
}


@pytest.fixture
def fresh_seed_guard():
    """Allow one seed_run call in a test that legitimately reseeds."""
    reset_seed_guard_for_tests()
    yield
    reset_seed_guard_for_tests()


def test_benchmark_registry_matches_likelihood_names() -> None:
    """Every harness likelihood has a benchmark entry and vice versa."""
    assert set(BENCHMARKS) == set(LIKELIHOOD_NAMES) == set(LIKELIHOOD_BUILDERS)
    assert set(SMOKE_PARAMS) == set(LIKELIHOOD_NAMES)
    for name, target in BENCHMARKS.items():
        assert target.likelihood_name == name


def test_only_hawaii_lacks_reference_draws() -> None:
    """Only hawaii, the deliberate no-ground-truth entry, lacks reference draws."""
    for name, target in BENCHMARKS.items():
        if name == 'hawaii':
            assert target.draw_reference is None
        else:
            assert target.draw_reference is not None


@pytest.mark.usefixtures('fresh_seed_guard')
def test_reference_draws_match_analytic_moments() -> None:
    """Reference draws reproduce registered per-coordinate means and variances.

    One seed_run covers the numba-stream gen_draws samplers (plan D1
    allows one seeding per process-run; the Generator-based samplers use
    their own explicit rng). Tolerances are wide relative to Monte Carlo
    error at 4000 draws but far tighter than any wrong-density failure.
    """
    seed_run(20260712)
    rng = get_rng(1234)
    for name, target in BENCHMARKS.items():
        if target.draw_reference is None or target.reference_moments is None:
            continue
        n_par = int(target.default_params.get('n_par', 2))
        draws = target.draw_reference(4000, n_par, rng)
        assert draws.shape == (4000, n_par), name
        means_ref, vars_ref = target.reference_moments(n_par)
        scale = np.sqrt(vars_ref)
        assert np.all(np.abs(draws.mean(axis=0) - means_ref) < 0.4 * scale), name
        var_ratio = draws.var(axis=0) / vars_ref
        assert np.all((var_ratio > 0.7) & (var_ratio < 1.4)), (name, var_ratio)


def _smoke_spec_data(name: str) -> dict[str, Any]:
    """Tiny end-to-end spec for one likelihood."""
    return {
        'name': f'smoke_{name}',
        'seed': 4242,
        'likelihood': {'name': name, **SMOKE_PARAMS[name]},
        'ladder': {
            'kind': 'geometric',
            'n_chain': 5,
            'n_cold': 1,
            'T_cold': 1.0,
            'T_min': 1.0,
            'T_max': 64.0,
            'n_inf_final': 1,
        },
        'run': {'n_steps': 128, 'block_size': 64, 'store_thin': 1, 'checkpoint_every_blocks': 2},
        'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
        'proposals': {
            'FisherJumpManager': {'verbose_fisher': False},
            'DEJumpManager': {'de_size': 256},
        },
    }


@pytest.mark.usefixtures('fresh_seed_guard')
def test_reference_draws_respect_likelihood_bounds() -> None:
    """Every reference draw lies inside its own likelihood's prior volume."""
    seed_run(20260713)
    rng = get_rng(4321)
    for name, target in BENCHMARKS.items():
        if target.draw_reference is None:
            continue
        params = dict(target.default_params)
        spec_data = _smoke_spec_data(name)
        spec_data['likelihood'] = {'name': name, **params}
        like_obj = build_likelihood(RunSpec.from_dict(spec_data))
        n_par = int(params.get('n_par', like_obj.n_par))
        draws = target.draw_reference(500, n_par, rng)
        assert all(like_obj.check_bounds(np.ascontiguousarray(draw)) for draw in draws), name


@pytest.mark.usefixtures('fresh_seed_guard')
def test_mode_weights_match_reference_draws() -> None:
    """Registered mode weights agree with reference-draw occupancy."""
    seed_run(20260714)
    rng = get_rng(999)
    for name, target in BENCHMARKS.items():
        if target.mode_centers is None or target.mode_weights is None or target.draw_reference is None:
            continue
        n_par = target.mode_centers.shape[1]
        draws = target.draw_reference(4000, n_par, rng)
        occupancy = mode_occupancy(draws, target.mode_centers)
        assert np.all(np.abs(occupancy - target.mode_weights) < 0.05), (name, occupancy)


@pytest.mark.parametrize('name', sorted(LIKELIHOOD_NAMES))
@pytest.mark.usefixtures('fresh_seed_guard')
def test_every_likelihood_runs_end_to_end(name, tmp_path) -> None:
    """Each likelihood runs through run_from_spec and yields a valid artifact."""
    spec = RunSpec.from_dict(_smoke_spec_data(name))
    artifact_path = run_from_spec(spec, tmp_path)
    assert validate(artifact_path, mode='complete') == []


def test_symmetric_nn_divergence_catches_both_failure_signs() -> None:
    """Overconcentration and support-missing both drive the symmetric NN up.

    The signed nn_kl goes negative for an overconcentrated test sample
    and positive for a support-missing sample; the symmetric form is
    positive for both and small for a matched pair.
    """
    rng = get_rng(31415)
    reference = rng.standard_normal((3000, 3))
    matched = rng.standard_normal((3000, 3))
    collapsed = 0.05 * rng.standard_normal((3000, 3))
    overdispersed = 3.0 * rng.standard_normal((3000, 3))

    assert abs(nn_divergence_symmetric(reference, matched, 2000, get_rng(1))) < 0.4
    assert nn_divergence_symmetric(reference, collapsed, 2000, get_rng(2)) > 1.5
    assert nn_divergence_symmetric(reference, overdispersed, 2000, get_rng(3)) > 0.8
    # the trap itself: signed divergence of the collapsed sample is negative
    assert nn_kl(reference, collapsed, 2000, get_rng(4)) < 0.0


def _write_synthetic_artifact(path, events: np.ndarray, segment_itrns: np.ndarray, n_iterations: int) -> None:
    """Write the minimal dataset set load_run_metrics reads."""
    rng = get_rng(7)
    with h5py.File(str(path), 'w') as hf:
        hf.attrs['n_iterations'] = n_iterations
        hf.attrs['n_likelihood_evals'] = 1000
        hf.attrs['wall_seconds'] = 1.0
        hf.create_dataset('ladder/Ts', data=np.array([1.0, 2.0, 4.0]))
        hf.create_dataset('events/rt_events', data=events.astype(np.int64))
        hf.create_dataset('events/rt_segment_itrns', data=segment_itrns.astype(np.int64))
        hf.create_dataset('store/samples', data=rng.standard_normal((n_iterations // 2, 1, 2)))
        hf['store'].attrs['store_thin'] = 2


def test_family_compare_summarize_ranks_pass_rate_before_efficiency() -> None:
    """Reliability outranks efficiency, and failing arms stay unranked.

    A partially failing arm must never outrank a fully passing one on
    efficiency alone, and an all-failing arm with spectacular n_eff/eval
    must stay unranked entirely (the issue-19 gate hierarchy: efficiency
    is a metric for the wrong distribution until posterior recovery
    passes).
    """

    def run_result(
        passed: bool, n_eff_per_eval: float, violations: list[str] | None = None
    ) -> dict[str, list[str] | float | bool | str]:
        return {
            'passed': passed,
            'n_eff_per_eval': n_eff_per_eval,
            'frozen_by': 'criterion',
            'violations': violations or [],
        }

    results = {
        'entropy': [run_result(True, 1.0e-3), run_result(True, 2.0e-3)],
        'length': [run_result(True, 4.0e-3), run_result(False, 5.0e-3, ['nn: too far'])],
        'acceptance': [
            run_result(False, 9.0e-3, ['tiers: collapsed']),
            run_result(False, 8.0e-3, ['tiers: collapsed']),
        ],
    }
    summary = summarize_arms(results)

    # entropy (2/2 passing, median 1.5e-3) outranks length (1/2 passing,
    # median 4.0e-3): pass rate is the primary key, efficiency the tiebreaker
    assert summary['ranking_by_pass_rate_then_efficiency'] == ['entropy', 'length']
    assert summary['unranked_failing_arms'] == ['acceptance']
    assert summary['arms']['entropy']['pass_rate'] == pytest.approx(1.0)
    assert summary['arms']['length']['pass_rate'] == pytest.approx(0.5)
    assert summary['arms']['entropy']['median_n_eff_per_eval'] == pytest.approx(1.5e-3)
    # the failing run's efficiency does not contaminate the passing median
    assert summary['arms']['length']['median_n_eff_per_eval'] == pytest.approx(4.0e-3)
    assert summary['arms']['acceptance']['median_n_eff_per_eval'] is None
    assert summary['arms']['acceptance']['violations'] == ['tiers: collapsed']

    # efficiency still breaks ties among equal pass rates
    tie = summarize_arms(
        {
            'entropy': [run_result(True, 1.0e-3)],
            'length': [run_result(True, 4.0e-3)],
        }
    )
    assert tie['ranking_by_pass_rate_then_efficiency'] == ['length', 'entropy']


def test_load_run_metrics_respects_segment_boundaries(tmp_path) -> None:
    """The pilot loader must not pair round-trip arrivals across a ladder update.

    Walker 0's cold/hot arrivals straddle the boundary (must not pair);
    walker 1's sit inside one segment (must pair).
    """
    events = np.array(
        [
            [0, 600, RT_ARRIVED_COLD],
            [0, 700, RT_ARRIVED_HOT],
            [1, 800, RT_ARRIVED_COLD],
            [1, 900, RT_ARRIVED_HOT],
        ]
    )
    boundary_after_walker0_cold = np.array([650])
    artifact_path = tmp_path / 'synthetic.h5'
    _write_synthetic_artifact(artifact_path, events, boundary_after_walker0_cold, n_iterations=1000)

    metrics = load_run_metrics(artifact_path, burn_fraction=0.5)
    assert metrics['total_round_trips'] == 1.0

    # control: with no boundary the same events pair into two trips
    control_path = tmp_path / 'control.h5'
    _write_synthetic_artifact(control_path, events, np.zeros(0), n_iterations=1000)
    assert load_run_metrics(control_path, burn_fraction=0.5)['total_round_trips'] == 2.0
