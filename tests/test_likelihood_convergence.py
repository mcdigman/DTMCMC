"""Per-likelihood adaptive convergence batteries (issue #19; slow suite).

Every wired likelihood with ground truth gets an end-to-end adaptive
run gated in sample space against its benchmark registry entry:
reference-draw NN divergence, analytic per-coordinate moments, and mode
occupancy where the target is a mixture — plus the structural gates
(freeze, T=1 readout pin, sub-readout rungs). Thresholds were
calibrated against known-good runs at exactly these configurations
(2x-3x the observed deviations, with reference-vs-reference noise
floors well below every threshold); the calibration table lives in the
PR that introduced this file.

Honest-scope notes, per target class:
- NN thresholds for the low-dimensional targets sit near the estimator
  floor; for ar1/banana/rosenbrock the NN statistic inflates on
  autocorrelated high-dimensional chains, so there it is a
  collapse-only guard and the load-bearing gates are moments plus (for
  ar1) the analytic adjacent-coordinate correlation.
- rosenbrock runs at 8d with coarse bands: valley traversal at 20d does
  not equilibrate second moments at any unit-test budget (the 20d
  structural test lives in test_adaptive.py's battery module scope).
"""

from typing import Any

import numpy as np
import pytest

from DTMCMC.rng_helpers import get_rng, reset_seed_guard_for_tests
from experiments.benchmarks import BENCHMARKS, mixture_mode_centers, mode_occupancy
from experiments.gates import GateReport, dedup_rows, moment_gates, nn_gate, occupancy_gates
from experiments.harness.runner import run_from_spec
from experiments.harness.spec import RunSpec
from experiments.reference_samplers import eggbox_cells
from tests.battery_common import adaptive_spec_data, assert_readout_structure, load_post_freeze

pytestmark = pytest.mark.slow

# per-target run configuration and calibrated gate thresholds
TARGETS: dict[str, dict[str, Any]] = {
    'gaussian': {
        'params': {'n_par': 4, 'cutoff': 5}, 'n_chain': 8, 'block': 256, 'blocks': 160, 'budget': 120,
        'nn': 0.8, 'mean_sigmas': 0.3, 'var_band': (0.85, 1.2),
    },
    'gaussian_fisher': {
        'like_name': 'gaussian',
        'params': {'n_par': 4, 'cutoff': 5}, 'n_chain': 8, 'block': 256, 'blocks': 160, 'budget': 120,
        'nn': 0.8, 'mean_sigmas': 0.3, 'var_band': (0.85, 1.2),
        # finite Fisher weights (all current specs run them at 0): the mixture
        # change must not drastically affect convergence
        'proposals_extra': {'FisherJumpManager': {'cold_fisher_weight': 0.333, 'hot_fisher_weight': 0.333}},
    },
    'gaussian_shell': {
        'params': {'n_par': 2}, 'n_chain': 8, 'block': 256, 'blocks': 160, 'budget': 120,
        'nn': 0.8, 'occupancy_tol': 0.08,
    },
    'hyperpyramid': {
        'params': {'n_par': 2}, 'n_chain': 8, 'block': 256, 'blocks': 160, 'budget': 120,
        'nn': 0.6, 'mean_sigmas': 0.3, 'var_band': (0.85, 1.2),
    },
    'eggbox': {
        'params': {'n_par': 2}, 'n_chain': 10, 'block': 256, 'blocks': 192, 'budget': 144,
        'nn': 0.9, 'occupancy_tol': 0.08,
    },
    'random_wheel': {
        'params': {'n_par': 2}, 'n_chain': 10, 'block': 256, 'blocks': 192, 'budget': 144,
        'nn': 1.2, 'mean_sigmas': 0.4, 'var_band': (0.8, 1.25), 'occupancy_tol': 0.07,
    },
    'spoke_wheel': {
        'params': {'n_par': 2}, 'n_chain': 10, 'block': 256, 'blocks': 192, 'budget': 144,
        'nn': 1.2, 'mean_sigmas': 0.6, 'var_band': (0.75, 1.25), 'occupancy_tol': 0.12,
    },
    'gaussian_mixture': {
        'params': {'n_par': 4}, 'n_chain': 10, 'block': 256, 'blocks': 192, 'budget': 144,
        'nn': 1.3, 'mean_sigmas': 0.3, 'var_band': (0.85, 1.2), 'occupancy_tol': 0.07,
    },
    'ar1': {
        'params': {'n_par': 12}, 'n_chain': 14, 'block': 512, 'blocks': 320, 'budget': 240,
        'nn': 8.0, 'mean_sigmas': 0.3, 'var_band': (0.8, 1.2), 'adjacent_corr': (0.9, 0.05),
    },
    'banana': {
        'params': {'n_par': 4}, 'n_chain': 20, 'block': 512, 'blocks': 448, 'budget': 352,
        'nn': 15.0, 'mean_sigmas': 0.5, 'var_band': (0.6, 1.3),
    },
    'rosenbrock': {
        'params': {'n_par': 8}, 'n_chain': 24, 'block': 512, 'blocks': 320, 'budget': 240,
        'nn': 40.0, 'mean_sigmas': 1.0, 'var_band': (0.2, 2.0),
    },
}

BATTERY_SEED = 777


@pytest.fixture
def fresh_seed_guard():
    """Allow one seed_run call in a test that legitimately reseeds."""
    reset_seed_guard_for_tests()
    yield
    reset_seed_guard_for_tests()


def _target_mode_centers(like_name: str, n_par: int):
    """Occupancy centers for a target, at the test dimensionality."""
    if like_name == 'gaussian_mixture':
        return mixture_mode_centers(n_par)
    if like_name == 'eggbox':
        cells = eggbox_cells(n_par)
        return cells.centers[cells.even_mask]
    return BENCHMARKS[like_name].mode_centers


@pytest.mark.parametrize('name', sorted(TARGETS))
@pytest.mark.usefixtures('fresh_seed_guard')
def test_adaptive_convergence_recovers_posterior(name, tmp_path) -> None:
    """Adaptive run freezes with a T=1 readout and recovers the target posterior."""
    cfg = TARGETS[name]
    like_name = str(cfg.get('like_name', name))
    target = BENCHMARKS[like_name]
    n_par = int(cfg['params'].get('n_par', 2))

    data = adaptive_spec_data(
        f'conv_{name}', BATTERY_SEED, {'name': like_name, **cfg['params']},
        n_chain=cfg['n_chain'], block_size=cfg['block'], n_blocks=cfg['blocks'],
        budget_blocks=cfg['budget'], proposals_extra=cfg.get('proposals_extra'),
    )
    spec = RunSpec.from_dict(data)
    artifact_path = run_from_spec(spec, tmp_path)

    run = load_post_freeze(artifact_path, block_size=cfg['block'], store_thin=4, budget_blocks=cfg['budget'])
    assert_readout_structure(run)

    cold = run['cold']
    cold_unique = dedup_rows(cold)
    assert cold_unique.shape[0] >= 200, 'too few unique post-freeze readout samples to gate'

    # reference draws after the run: the numba-stream samplers continue the
    # run's seeded stream, so this stays deterministic per seed/platform
    rng = get_rng(90000 + BATTERY_SEED)
    assert target.draw_reference is not None, 'convergence targets require reference draws'
    reference = target.draw_reference(8000, n_par, rng)

    report = GateReport()
    report.merge(nn_gate(reference, cold_unique, threshold=cfg['nn'], n_use=2000, rng=get_rng(2), label=name))
    if 'mean_sigmas' in cfg:
        assert target.reference_moments is not None
        means_ref, vars_ref = target.reference_moments(n_par)
        report.merge(moment_gates(cold, means_ref, vars_ref, mean_tol_sigmas=cfg['mean_sigmas'], var_ratio_bounds=cfg['var_band'], label=name))
    if 'occupancy_tol' in cfg:
        centers = _target_mode_centers(like_name, n_par)
        assert centers is not None
        weights_ref = target.mode_weights if target.mode_weights is not None else mode_occupancy(reference, centers)
        report.merge(occupancy_gates(cold, centers, weights_ref, tol=cfg['occupancy_tol'], label=name))
    if 'adjacent_corr' in cfg:
        corr_ref, corr_tol = cfg['adjacent_corr']
        adjacent = np.array([np.corrcoef(cold[:, itrp], cold[:, itrp + 1])[0, 1] for itrp in range(n_par - 1)])
        if float(np.abs(adjacent - corr_ref).max()) > corr_tol:
            report.violations.append(f'{name}: adjacent-coordinate correlations {np.round(adjacent, 3).tolist()} deviate from {corr_ref} by more than {corr_tol}')

    assert report.passed, report.violations


@pytest.mark.usefixtures('fresh_seed_guard')
def test_rosenbrock_20d_structural(tmp_path) -> None:
    """20d rosenbrock: adaptation completes its descent at production-like depth.

    The 20d target carries ~70 nats from the prior box to the posterior
    (~1.8 nats/link at 40 chains), so the annealing schedule must chain
    dozens of throttled extensions without stalling — the capacity
    regime none of the smaller batteries reach. Valley traversal does
    NOT equilibrate 20d second moments at any unit-test budget
    (calibration measured per-coordinate variance ratios spanning
    [0.03, 3.5] after 448 blocks), so posterior moments are gated only
    at the 8d configuration above; here the assertions are structural:
    full descent to a T=1 readout with sub-readout rungs, applied
    updates throughout, and a recorded freeze.
    """
    cfg = {'params': {'n_par': 20}, 'n_chain': 40, 'block': 512, 'blocks': 448, 'budget': 352}
    data = adaptive_spec_data(
        'conv_rosenbrock20', BATTERY_SEED, {'name': 'rosenbrock', **cfg['params']},
        n_chain=cfg['n_chain'], block_size=cfg['block'], n_blocks=cfg['blocks'], budget_blocks=cfg['budget'],
    )
    spec = RunSpec.from_dict(data)
    artifact_path = run_from_spec(spec, tmp_path)

    run = load_post_freeze(artifact_path, block_size=cfg['block'], store_thin=4, budget_blocks=cfg['budget'])
    assert_readout_structure(run)
    assert run['n_applied'] >= 20, 'the deep descent requires many applied extensions'
    finite_Ts = np.sort(run['final_Ts'][np.isfinite(run['final_Ts'])])
    assert np.all(np.diff(finite_Ts) >= 0.)
    # the readout stream must be alive and inside the prior volume
    assert np.all(np.abs(run['cold']) <= 10.)
    assert np.isfinite(run['cold_logLs']).all()
