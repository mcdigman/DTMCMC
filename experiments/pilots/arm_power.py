"""Pilot: paired-arm effect sizes and seed-count power analysis (plan §4 Phase 4).

Runs the five E1 ladder arms on cake 5D at a fixed chain count over a
shared seed list (plan D3 pairing), computes paired per-seed differences
of both C1 co-primary metrics against the entropy arm, and sizes the
production seed count for 95% CI half-widths <= 1/3 of the observed
effects. Also fixes the E1 geometric-tuned arm definition: T_max matched
to the gold entropy ladder's hottest finite rung at the same n_chain.
"""

from typing import Any

import numpy as np

from DTMCMC.temperature_ladder_helpers import EntropyTemperatureLadder
from experiments.harness.paths import resolve
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

N_CHAIN = 16
SEEDS = [2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208]
N_STEPS = 262144

CAKE1_FILES = {'Ts_file': 'data/Ts_cake1.npy', 'means_file': 'data/means_cake1.npy', 'vars_file': 'data/vars_cake1.npy'}


def tuned_geometric_t_max(n_chain: int) -> float:
    """E1 geometric-tuned definition: T_max of the gold entropy ladder's hottest finite rung."""
    Ts_in = np.load(resolve('data/Ts_cake_gold.npy'))
    vars_in = np.load(resolve('data/vars_cake_gold.npy'))
    keep = Ts_in >= 1.0
    ladder = EntropyTemperatureLadder(n_chain, Ts_in[keep], vars_in[keep], n_cold=1, T_cold=1.0, n_inf_final=1)
    return float(ladder.Ts[np.isfinite(ladder.Ts)].max())


def arm_ladders(n_chain: int) -> dict[str, dict[str, Any]]:
    """The five E1 ladder arms at the requested chain count."""
    t_max_tuned = tuned_geometric_t_max(n_chain)
    return {
        'entropy': entropy_gold_ladder(n_chain),
        'length': {
            'kind': 'length_file',
            'n_chain': n_chain,
            'n_cold': 1,
            'Ts_file': 'data/Ts_cake_gold.npy',
            'vars_file': 'data/vars_cake_gold.npy',
        },
        'acceptance': {'kind': 'acceptance_file', 'n_chain': n_chain, 'n_cold': 1, **CAKE1_FILES},
        'geometric_default': {'kind': 'geometric', 'n_chain': n_chain, 'n_cold': 1},
        'geometric_tuned': {'kind': 'geometric', 'n_chain': n_chain, 'n_cold': 1, 'T_max': t_max_tuned},
    }


def seeds_for_power(paired_diffs: np.ndarray) -> float:
    """Seeds needed so the 95% CI half-width is <= |median effect|/3."""
    effect = float(np.abs(np.median(paired_diffs)))
    sd = float(paired_diffs.std(ddof=1))
    if effect == 0.0:
        return float('inf')
    return (1.96 * sd / (effect / 3.0)) ** 2


def main() -> int:
    """Run the arm battery, print effects, and size the seed counts."""
    ladders = arm_ladders(N_CHAIN)
    specs = [
        make_spec(f'arm_{arm_name}', seed, cake5_likelihood(), ladder, N_STEPS)
        for arm_name, ladder in ladders.items()
        for seed in SEEDS
    ]
    spec_paths = write_specs(specs, PILOT_ROOT / 'arms' / 'specs')
    artifact_paths = run_spec_files(spec_paths, PILOT_ROOT / 'arms')

    metrics: dict[str, dict[str, np.ndarray]] = {}
    for arm_name in ladders:
        per_seed = [
            load_run_metrics(next(p for p in artifact_paths if p.name == f'arm_{arm_name}_seed{seed}.h5'))
            for seed in SEEDS
        ]
        metrics[arm_name] = {
            'rt_rate': np.array([m['rt_rate'] for m in per_seed]),
            'n_eff_per_eval': np.array([m['n_eff_per_eval'] for m in per_seed]),
            'total_round_trips': np.array([m['total_round_trips'] for m in per_seed]),
        }
        print(
            f'{arm_name:>18}: rt_rate {metrics[arm_name]["rt_rate"].mean():8.3f} +- {metrics[arm_name]["rt_rate"].std():6.3f}   '
            f'n_eff/eval {metrics[arm_name]["n_eff_per_eval"].mean():.3e}   '
            f'RTs/run {metrics[arm_name]["total_round_trips"].mean():.0f}'
        )

    results: dict[str, Any] = {
        'n_chain': N_CHAIN,
        'seeds': SEEDS,
        'n_steps': N_STEPS,
        'tuned_geometric_t_max': tuned_geometric_t_max(N_CHAIN),
        'arms': {
            arm: {key: values.tolist() for key, values in arm_metrics.items()} for arm, arm_metrics in metrics.items()
        },
        'power': {},
    }
    power: dict[str, Any] = results['power']
    for arm_name in ladders:
        if arm_name == 'entropy':
            continue
        for metric_name in ('rt_rate', 'n_eff_per_eval'):
            diffs = metrics['entropy'][metric_name] - metrics[arm_name][metric_name]
            needed = seeds_for_power(diffs)
            power[f'entropy_vs_{arm_name}_{metric_name}'] = {
                'median_effect': float(np.median(diffs)),
                'sd_paired': float(diffs.std(ddof=1)),
                'seeds_needed': needed,
            }
            print(
                f'entropy vs {arm_name:>18} [{metric_name:>14}]: effect {np.median(diffs):+.3e}, paired sd {diffs.std(ddof=1):.3e}, seeds needed {needed:.1f}'
            )

    save_summary('arm_power', results)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
