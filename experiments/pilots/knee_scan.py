"""Pilot: coarse knee scan on cake 5D (plan §4 Phase 4).

Sweeps chain count with the gold-data entropy ladder, measures the
post-burn round-trip rate, and fits the knee with both candidate
estimators against both axes (raw n_chain and dS per link) with a
seed-bootstrap stability comparison — the data that freezes the knee
estimator and places the production n_chain grid.
"""

import numpy as np

from DTMCMC.rng_helpers import get_rng
from DTMCMC.temperature_ladder_helpers import Ts_to_betas, get_spacing_integrated, standardize_input_vars
from experiments.harness.paths import resolve
from experiments.metrics import fit_knee_max_curvature, fit_knee_piecewise_linear
from experiments.pilots.common import (
    PILOT_ROOT,
    cake5_likelihood,
    entropy_gold_ladder,
    load_run_metrics,
    make_spec,
    run_spec_files,
    save_summary,
    write_specs,
)

N_CHAIN_GRID = [6, 8, 10, 12, 16, 24, 32]
SEEDS = [1101, 1102, 1103]
N_STEPS = 262144
BOOTSTRAP_SEED = 314
N_BOOT = 500


def gold_total_entropy() -> float:
    """Total entropy span of the gold cake inputs (the equal-dS budget)."""
    Ts_in = np.load(resolve('data/Ts_cake_gold.npy'))
    vars_in = np.load(resolve('data/vars_cake_gold.npy'))
    keep = Ts_in >= 1.0
    betas_use, vars_use = standardize_input_vars(Ts_to_betas(Ts_in[keep]), vars_in[keep])
    return float(get_spacing_integrated(vars_use, betas_use, False)[-1])


def bootstrap_knee_sd(x: np.ndarray, rates: np.ndarray, fit, rng: np.random.Generator, n_boot: int = 500) -> float:
    """Seed-bootstrap the per-point rates and return the knee-location sd."""
    _n_points, n_seeds = rates.shape
    knees = np.zeros(n_boot)
    for itrb in range(n_boot):
        resampled = rates[:, rng.integers(0, n_seeds, size=n_seeds)].mean(axis=1)
        knees[itrb] = fit(x, resampled)
    return float(knees.std())


def main() -> int:
    """Run the knee scan and freeze-candidate analysis."""
    specs = [
        make_spec(f'knee_nc{n_chain}', seed, cake5_likelihood(), entropy_gold_ladder(n_chain), N_STEPS)
        for n_chain in N_CHAIN_GRID
        for seed in SEEDS
    ]
    spec_paths = write_specs(specs, PILOT_ROOT / 'knee' / 'specs')
    artifact_paths = run_spec_files(spec_paths, PILOT_ROOT / 'knee')

    rates = np.zeros((len(N_CHAIN_GRID), len(SEEDS)))
    for itrc, n_chain in enumerate(N_CHAIN_GRID):
        for itrs, seed in enumerate(SEEDS):
            path = next(p for p in artifact_paths if p.name == f'knee_nc{n_chain}_seed{seed}.h5')
            rates[itrc, itrs] = load_run_metrics(path)['rt_rate']
        print(
            f'n_chain={n_chain:>3}: rt_rate = {rates[itrc].mean():.3f} +- {rates[itrc].std():.3f} per walker per 1e6 chain-steps'
        )

    s_total = gold_total_entropy()
    # dS/link convention (matches entropy_spacing + _plug_cold_and_inf): the
    # gold span is divided into n_chain-1 equal-entropy links and the hottest
    # rung is then overwritten with T=inf, so adjacent finite rungs are spaced
    # by exactly s_total/(n_chain-1) and the final finite->inf link carries the
    # remaining slice of the gold span. The denominator is the link count
    # n_chain-1, not the finite-rung count.
    ds_per_link = s_total / (np.asarray(N_CHAIN_GRID, dtype=np.float64) - 1.0)
    mean_rates = rates.mean(axis=1)
    n_chain_arr = np.asarray(N_CHAIN_GRID, dtype=np.float64)

    rng = get_rng(BOOTSTRAP_SEED)
    results: dict[str, object] = {
        'n_chain_grid': N_CHAIN_GRID,
        'seeds': SEEDS,
        'n_steps': N_STEPS,
        'rt_rates': rates.tolist(),
        's_total_gold': s_total,
        'ds_per_link': ds_per_link.tolist(),
        'bootstrap_seed': BOOTSTRAP_SEED,
        'n_boot': N_BOOT,
    }
    # the two axes are fit independently; a piecewise-linear breakpoint is not
    # invariant under the nonlinear n_chain -> dS/link reparameterization, so
    # the two knees are reported side by side and must not be conflated
    for fit_name, fit in (('piecewise_linear', fit_knee_piecewise_linear), ('max_curvature', fit_knee_max_curvature)):
        knee_nc = fit(n_chain_arr, mean_rates)
        knee_ds = fit(ds_per_link[::-1].copy(), mean_rates[::-1].copy())
        sd_nc = bootstrap_knee_sd(n_chain_arr, rates, fit, rng, n_boot=N_BOOT)
        sd_ds = bootstrap_knee_sd(ds_per_link[::-1].copy(), rates[::-1].copy(), fit, rng, n_boot=N_BOOT)
        results[fit_name] = {
            'knee_n_chain': knee_nc,
            'bootstrap_sd_n_chain': sd_nc,
            'knee_ds_per_link': knee_ds,
            'bootstrap_sd_ds_per_link': sd_ds,
        }
        print(
            f'{fit_name}: n_chain-axis knee = {knee_nc:.1f} (bootstrap sd {sd_nc:.2f}); '
            f'dS/link-axis knee = {knee_ds:.3f} (bootstrap sd {sd_ds:.3f})'
        )

    save_summary('knee_scan', results)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
