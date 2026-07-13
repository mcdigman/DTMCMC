"""Ladder-family comparison driver on the posterior-anchored gates (issue #19).

Runs the adaptive spacing rules (entropy | length | acceptance) as arms on
one benchmark target, seeds x modes, and scores them with the gate
hierarchy the issue-19 audits prescribed: an arm's efficiency (post-freeze
n_eff per likelihood evaluation) is reported and ranked ONLY when its
posterior-recovery gate passes — round trips, exchange overlap, and n_eff
are metrics for the wrong distribution on a run that has not recovered
the target, so failing arms are listed with their violations instead of a
rank.

Reference data comes from the benchmark registry at the target's default
parameters; a target run at non-default parameters needs its own ground
truth (see experiments.benchmarks.BenchmarkTarget.default_params). Gate
thresholds are target-dependent and must be calibrated the way the
convergence batteries were (tests/test_likelihood_convergence.py quotes
its calibration in the TARGETS table); the defaults here are collapse
guards, not tight gates.

One process per run (D1), via the harness CLI like every other pilot.
"""

import argparse
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from DTMCMC.rng_helpers import get_rng
from experiments.benchmarks import BENCHMARKS
from experiments.gates import GateReport, dedup_rows, moment_gates, nn_gate
from experiments.harness.postfreeze import load_post_freeze, readout_structure_violations
from experiments.pilots.common import PILOT_ROOT, load_run_metrics, make_adaptive_spec, run_spec_files, save_summary, write_specs

MODES = ('entropy', 'length', 'acceptance')


def evaluate_arm_run(
        artifact_path: Path,
        target_name: str,
        n_par: int,
        *,
        nn_threshold: float,
        mean_tol_sigmas: float,
        var_ratio_bounds: tuple[float, float],
        n_reference: int = 8000,
        gate_seed: int = 2,
) -> dict[str, Any]:
    """Gate one arm run and, if it passes, attach its efficiency metrics.

    Gate 1 is posterior recovery (structure + reference NN + analytic
    moments where the registry has them); the efficiency numbers are
    computed for every run but are only meaningful — and only ranked by
    summarize_arms — for passing runs.
    """
    target = BENCHMARKS[target_name]
    run = load_post_freeze(artifact_path)
    report = GateReport()
    report.violations.extend(readout_structure_violations(run))

    if target.draw_reference is not None:
        reference = target.draw_reference(n_reference, n_par, get_rng(90000 + gate_seed))
        report.merge(nn_gate(reference, dedup_rows(run['cold']), threshold=nn_threshold, n_use=2000, rng=get_rng(gate_seed), label=target_name))
    if target.reference_moments is not None:
        means_ref, vars_ref = target.reference_moments(n_par)
        report.merge(moment_gates(run['cold'], means_ref, vars_ref, mean_tol_sigmas=mean_tol_sigmas, var_ratio_bounds=var_ratio_bounds, label=target_name))

    metrics = load_run_metrics(artifact_path)
    return {
        'passed': report.passed,
        'violations': list(report.violations),
        'gate_stats': dict(report.stats),
        'frozen_by': run['frozen_by'],
        'freeze_block': run['freeze_block'],
        'n_applied': run['n_applied'],
        'n_eff_per_eval': metrics['n_eff_per_eval'],
        'n_eff_min': metrics['n_eff_min'],
        'total_round_trips': metrics['total_round_trips'],
        'wall_seconds': metrics['wall_seconds'],
    }


def summarize_arms(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Aggregate per-arm runs into the gate-then-efficiency ranking.

    Pure aggregation (unit-tested without MCMC runs): each arm reports its
    pass count and, over its PASSING runs only, the median efficiency;
    the ranking orders fully or partially passing arms by that median and
    lists arms with zero passes separately with their violations.
    """
    arms: dict[str, Any] = {}
    for arm, runs in results.items():
        passing = [run for run in runs if run['passed']]
        arms[arm] = {
            'n_runs': len(runs),
            'n_passed': len(passing),
            'median_n_eff_per_eval': float(np.median([run['n_eff_per_eval'] for run in passing])) if passing else None,
            'frozen_by': [run['frozen_by'] for run in runs],
            'violations': sorted({violation for run in runs for violation in run['violations']}),
        }
    ranked = sorted(
        (arm for arm, summary in arms.items() if summary['n_passed'] > 0),
        key=lambda arm: arms[arm]['median_n_eff_per_eval'],
        reverse=True,
    )
    return {
        'arms': arms,
        'ranking_by_efficiency_among_passing': ranked,
        'unranked_failing_arms': sorted(arm for arm, summary in arms.items() if summary['n_passed'] == 0),
    }


def run_family_comparison(
        target_name: str,
        seeds: tuple[int, ...],
        *,
        n_par: int,
        n_chain: int,
        block_size: int,
        n_blocks: int,
        budget_blocks: int,
        nn_threshold: float,
        mean_tol_sigmas: float = 0.5,
        var_ratio_bounds: tuple[float, float] = (0.5, 1.5),
        out_root: Path | None = None,
        jobs: int = 4,
) -> dict[str, Any]:
    """Run modes x seeds on one target and write the comparison summary JSON."""
    likelihood: dict[str, Any] = {'name': target_name, **BENCHMARKS[target_name].default_params}
    likelihood['n_par'] = n_par
    root = (out_root if out_root is not None else PILOT_ROOT) / f'family_{target_name}'

    specs = [
        make_adaptive_spec(
            f'family_{target_name}_{mode}', seed, likelihood,
            n_chain=n_chain, block_size=block_size, n_blocks=n_blocks,
            budget_blocks=budget_blocks, mode=mode,
        )
        for mode in MODES for seed in seeds
    ]
    spec_paths = write_specs(specs, root / 'specs')
    artifact_paths = run_spec_files(spec_paths, root / 'runs', jobs=jobs)

    results: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for spec_data, artifact_path in zip(specs, artifact_paths, strict=True):
        mode = str(spec_data['adaptive']['mode'])
        results[mode].append(evaluate_arm_run(
            artifact_path, target_name, n_par,
            nn_threshold=nn_threshold, mean_tol_sigmas=mean_tol_sigmas, var_ratio_bounds=var_ratio_bounds,
        ))

    summary = {
        'target': target_name,
        'n_par': n_par,
        'seeds': list(seeds),
        'config': {'n_chain': n_chain, 'block_size': block_size, 'n_blocks': n_blocks, 'budget_blocks': budget_blocks},
        'thresholds': {'nn': nn_threshold, 'mean_sigmas': mean_tol_sigmas, 'var_band': list(var_ratio_bounds)},
        **summarize_arms(results),
        'runs': dict(results),
    }
    save_summary(f'family_compare_{target_name}', summary)
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', choices=sorted(name for name, target in BENCHMARKS.items() if target.draw_reference is not None))
    parser.add_argument('--n-par', type=int, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[555, 556, 557])
    parser.add_argument('--n-chain', type=int, default=12)
    parser.add_argument('--block-size', type=int, default=512)
    parser.add_argument('--n-blocks', type=int, default=320)
    parser.add_argument('--budget-blocks', type=int, default=240)
    parser.add_argument('--nn-threshold', type=float, default=2.0)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    summary = run_family_comparison(
        args.target, tuple(args.seeds), n_par=args.n_par, n_chain=args.n_chain,
        block_size=args.block_size, n_blocks=args.n_blocks, budget_blocks=args.budget_blocks,
        nn_threshold=args.nn_threshold, jobs=args.jobs,
    )
    print(summary['ranking_by_efficiency_among_passing'], summary['unranked_failing_arms'])


if __name__ == '__main__':
    main()
