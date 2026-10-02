"""Pilot: DE-buffer remap A/B for the Phase 5 ladder-update default (plan D6).

Simulates a mid-run ladder update without the (not yet existing) engine
hook: run a sampler on a deliberately coarse geometric ladder, then
transplant its state into a fresh sampler on the gold entropy ladder
under one of three DE-buffer remap rules, and measure recovery. Each
(rule, seed) combination runs in its own process (one seeding per
process, plan D1); the orchestrator aggregates the JSON results.

Rules:
- at_or_hotter (the D6 default candidate): each new temperature inherits
  the buffer column of the nearest old temperature at-or-hotter.
- nearest: nearest old temperature in log T.
- partial_reset: nearest-column copy for a random half of the buffer
  rows, fresh prior draws (already in the new buffer) for the rest.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path  # noqa: TC003 — runtime Path construction
from typing import Any

import numpy as np

from DTMCMC.de_manager import DEJumpManager
from DTMCMC.dtmcmc_sampler import DTMCMCSampler
from DTMCMC.proposal_manager_helper import get_default_proposal_manager
from DTMCMC.rng_helpers import get_rng, seed_run
from DTMCMC.temperature_ladder_helpers import GeometricTemperatureLadder
from experiments.harness.paths import atomic_write_text, chdir_repo_root, repo_root
from experiments.harness.runner import build_ladder, build_likelihood
from experiments.harness.spec import RunSpec
from experiments.metrics import round_trip_counts
from experiments.pilots.common import PILOT_ROOT, cake5_likelihood, entropy_gold_ladder, make_spec, save_summary

RULES = ('at_or_hotter', 'nearest', 'partial_reset')
SEEDS = (3301, 3302, 3303, 3304)
N_CHAIN = 12
BLOCK_SIZE = 1024
PRE_BLOCKS = 32
POST_BLOCKS = 64


def remap_source_columns(Ts_old: np.ndarray, Ts_new: np.ndarray, rule: str) -> np.ndarray:
    """Old-ladder column index feeding each new temperature slot."""
    sources = np.zeros(Ts_new.size, dtype=np.int64)
    log_old = np.log(np.minimum(Ts_old, 1.0e300))
    for itrt, T_new in enumerate(Ts_new):
        if rule == 'at_or_hotter':
            hotter = np.flatnonzero(Ts_old >= T_new)
            sources[itrt] = int(hotter[np.argmin(Ts_old[hotter])]) if hotter.size else int(np.argmax(Ts_old))
        else:
            sources[itrt] = int(np.argmin(np.abs(log_old - np.log(min(T_new, 1.0e300)))))
    return sources


def run_single(rule: str, seed: int, out_path: Path) -> None:
    """One (rule, seed) transplant experiment; writes a JSON result."""
    chdir_repo_root()
    seed_run(seed)

    spec: RunSpec[Any] = RunSpec.from_dict(
        make_spec('remap', seed, cake5_likelihood(), entropy_gold_ladder(N_CHAIN), BLOCK_SIZE * 2)
    )
    like_obj = build_likelihood(spec)
    config = spec.build_proposal_config()

    # phase A: settle on a deliberately coarse geometric ladder
    ladder_a = GeometricTemperatureLadder(N_CHAIN, n_cold=1, T_cold=1.0, T_min=1.0, T_max=1000.0, n_inf_final=1)
    starting = np.array([like_obj.prior_draw() for _ in range(N_CHAIN)])
    pm_a = get_default_proposal_manager(ladder_a, like_obj, starting_samples=starting, config=config)
    sampler_a = DTMCMCSampler(
        ladder_a,
        like_obj,
        BLOCK_SIZE,
        PRE_BLOCKS * BLOCK_SIZE,
        proposal_manager=pm_a,
        starting_samples=starting,
        store_thin=64,
    )
    for _ in range(PRE_BLOCKS):
        sampler_a.advance_block()

    # ladder update to the gold entropy ladder: remap chain states to the
    # nearest new temperature (D6), carry logLs, remap DE buffer per rule
    ladder_b = build_ladder(spec)
    Ts_old = np.asarray(sampler_a.Ts)
    Ts_new = np.asarray(ladder_b.Ts)
    state_sources = remap_source_columns(Ts_old, Ts_new, 'nearest')
    carried_states = sampler_a.samples[0][state_sources].copy()

    de_a = next(m for m in pm_a.managers if isinstance(m, DEJumpManager))
    pm_b = get_default_proposal_manager(ladder_b, like_obj, starting_samples=carried_states, config=config)
    de_b = next(m for m in pm_b.managers if isinstance(m, DEJumpManager))
    buffer_sources = remap_source_columns(Ts_old, Ts_new, 'nearest' if rule == 'partial_reset' else rule)
    if rule == 'partial_reset':
        # copy for a random half of the rows; the rest keep fresh prior draws
        keep_rows = get_rng(seed).random(de_b.de_size) < 0.5
        for itrt, src in enumerate(buffer_sources):
            de_b.de_buffer[keep_rows, itrt, :] = de_a.de_buffer[keep_rows, src, :]
    else:
        for itrt, src in enumerate(buffer_sources):
            de_b.de_buffer[:, itrt, :] = de_a.de_buffer[:, src, :]

    sampler_b = DTMCMCSampler(
        ladder_b,
        like_obj,
        BLOCK_SIZE,
        POST_BLOCKS * BLOCK_SIZE,
        proposal_manager=pm_b,
        starting_samples=carried_states,
        store_thin=64,
    )

    accepted_swaps = np.zeros(POST_BLOCKS)
    last_total = 0
    for itrb in range(POST_BLOCKS):
        sampler_b.advance_block()
        # with track_full_exchanges=False, execute_swaps increments both
        # participants' [0, 0, :] slots per accepted swap, so the plane
        # sum is exactly 2x the accepted-swap count (always even)
        total = int(sampler_b.tracker_manager.exchange_tracker[0, 0].sum()) // 2
        accepted_swaps[itrb] = total - last_total
        last_total = total

    cold_logL_means = np.array([block_means[0] for block_means in sampler_b.logL_means])

    # settling time: first block whose value enters the final-16-block band
    # (mean +- 2 sd) and stays there for 4 consecutive blocks
    tail = cold_logL_means[-16:]
    band_lo, band_hi = tail.mean() - 2.0 * tail.std(), tail.mean() + 2.0 * tail.std()
    in_band = (cold_logL_means >= band_lo) & (cold_logL_means <= band_hi)
    settle_block = POST_BLOCKS
    for itrb in range(POST_BLOCKS - 3):
        if np.all(in_band[itrb : itrb + 4]):
            settle_block = itrb
            break

    rt_events = sampler_b.tracker_manager.get_rt_events()
    result = {
        'rule': rule,
        'seed': seed,
        'settle_block': int(settle_block),
        'post_round_trips': int(round_trip_counts(rt_events, N_CHAIN).sum()),
        'late_swap_acceptance_per_block': float(accepted_swaps[-16:].mean()),
        'early_swap_acceptance_per_block': float(accepted_swaps[:8].mean()),
    }
    atomic_write_text(out_path, json.dumps(result, indent=2))


def main(argv: list[str] | None = None) -> int:
    """Orchestrate the (rule x seed) grid in subprocesses and aggregate."""
    parser = argparse.ArgumentParser(description='DE-buffer remap A/B pilot')
    parser.add_argument('--single', nargs=2, metavar=('RULE', 'SEED'), default=None)
    args = parser.parse_args(argv)

    out_dir = PILOT_ROOT / 'remap'
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.single is not None:
        rule, seed = args.single[0], int(args.single[1])
        run_single(rule, seed, out_dir / f'remap_{rule}_seed{seed}.json')
        return 0

    def launch(rule_seed: tuple[str, int]) -> None:
        rule, seed = rule_seed
        result = subprocess.run(  # noqa: S603 — static module entry, pilot-local args
            [sys.executable, '-m', 'experiments.pilots.remap_ab', '--single', rule, str(seed)],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            msg = f'remap pilot failed for {rule} seed {seed}:\n{result.stderr[-2000:]}'
            raise RuntimeError(msg)

    combos = [(rule, seed) for rule in RULES for seed in SEEDS]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(launch, combos))

    summary: dict[str, object] = {
        'seeds': list(SEEDS),
        'pre_blocks': PRE_BLOCKS,
        'post_blocks': POST_BLOCKS,
        'n_chain': N_CHAIN,
    }
    for rule in RULES:
        results = [json.loads((out_dir / f'remap_{rule}_seed{seed}.json').read_text()) for seed in SEEDS]
        rule_summary: dict[str, list[float]] = {
            'settle_blocks': [r['settle_block'] for r in results],
            'post_round_trips': [r['post_round_trips'] for r in results],
            'early_swap_acceptance': [r['early_swap_acceptance_per_block'] for r in results],
            'late_swap_acceptance': [r['late_swap_acceptance_per_block'] for r in results],
        }
        summary[rule] = rule_summary
        print(
            f'{rule:>14}: settle {np.mean(rule_summary["settle_blocks"]):5.1f} blocks, '
            f'post RTs {np.mean(rule_summary["post_round_trips"]):6.1f}, '
            f'early swaps/block {np.mean(rule_summary["early_swap_acceptance"]):7.1f}'
        )

    save_summary('remap_ab', summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
