"""Shared pilot plumbing: spec construction, parallel execution, artifact loaders."""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path  # noqa: TC003 — runtime Path construction
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from DTMCMC.rng_helpers import get_rng
from experiments.harness.paths import repo_root
from experiments.harness.spec import RunSpec, dumps_toml
from experiments.metrics import round_trip_counts, round_trip_rate, scramble_block_n_eff_min

if TYPE_CHECKING:
    from numpy.typing import NDArray

PILOT_ROOT = repo_root() / 'artifacts' / 'pilots'

# pilot-standard proposal overrides: moderate DE buffer and quiet Fisher updates
PILOT_PROPOSALS: dict[str, dict[str, object]] = {
    'FisherJumpManager': {'verbose_fisher': False},
    'DEJumpManager': {'de_size': 10000},
}

GOLD_LADDER_FILES = {'Ts_file': 'data/Ts_cake_gold.npy', 'vars_file': 'data/vars_cake_gold.npy'}

# standard adaptive DE ring-buffer span, in blocks: with the standard
# cadence (update_every_blocks = 8) this is eight adaptation windows —
# long enough to bridge rebuilds, short enough that the buffer has fully
# turned over (and forgotten adaptation burn-in) well before the run ends.
# Whole-run buffers never forget burn-in and are reserved for explicit
# old-behavior controls.
ADAPTIVE_DE_WINDOW_BLOCKS = 64


def make_adaptive_spec(
        name: str,
        seed: int,
        likelihood: dict[str, Any],
        *,
        n_chain: int,
        block_size: int,
        n_blocks: int,
        budget_blocks: int,
        store_thin: int = 4,
        t_min_factor: float = 0.9,
        remap_rule: str = 'no_remap',
        mode: str = 'entropy',
        de_window_blocks: int | None = ADAPTIVE_DE_WINDOW_BLOCKS,
        proposals_extra: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the shared adaptive spec dict (batteries and family arms).

    de_window_blocks sizes the DE ring buffer in blocks (None = whole-run,
    reserved for old-behavior controls).
    """
    de_blocks = n_blocks if de_window_blocks is None else min(de_window_blocks, n_blocks)
    proposals: dict[str, dict[str, Any]] = {
        'FisherJumpManager': {'verbose_fisher': False},
        'DEJumpManager': {'de_size': block_size * de_blocks},
    }
    for section, entries in (proposals_extra or {}).items():
        proposals.setdefault(section, {}).update(entries)
    return {
        'name': name,
        'seed': seed,
        'likelihood': likelihood,
        'ladder': {'kind': 'geometric', 'n_chain': n_chain, 'n_cold': 1},
        'run': {'n_steps': block_size * n_blocks, 'block_size': block_size, 'store_thin': store_thin,
                'checkpoint_every_blocks': n_blocks},
        'adaptive': {'mode': mode, 'update_every_blocks': 8, 'forgetting': 0.15,
                     'freeze_dlog': 0.05, 'freeze_consecutive': 3, 'budget_blocks': budget_blocks,
                     'remap_rule': remap_rule, 'T_min_factor': t_min_factor},
        'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
        'proposals': proposals,
    }


def make_spec(name: str, seed: int, likelihood: dict[str, Any], ladder: dict[str, Any], n_steps: int, block_size: int = 1024, store_thin: int = 16) -> dict[str, Any]:
    """Build a pilot spec dict with the shared conventions."""
    return {
        'name': name,
        'seed': seed,
        'likelihood': likelihood,
        'ladder': ladder,
        'run': {
            'n_steps': n_steps,
            'block_size': block_size,
            'store_thin': store_thin,
            'checkpoint_every_blocks': max(n_steps // block_size // 4, 1),
        },
        'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
        'proposals': dict(PILOT_PROPOSALS),
    }


def cake5_likelihood() -> dict[str, Any]:
    """The shared cake 5D likelihood table."""
    return {'name': 'cake', 'n_par': 5, 'cutoff': 10}


def entropy_gold_ladder(n_chain: int) -> dict[str, Any]:
    """Entropy ladder from the gold cake data at the requested chain count."""
    return {'kind': 'entropy_file', 'n_chain': n_chain, 'n_cold': 1, **GOLD_LADDER_FILES}


def write_specs(specs: list[dict[str, Any]], spec_dir: Path) -> list[Path]:
    """Validate and write spec dicts as TOML files; returns the paths."""
    spec_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec_data in specs:
        RunSpec.from_dict(spec_data)  # validate before writing
        path = spec_dir / f'{spec_data["name"]}_seed{spec_data["seed"]}.toml'
        path.write_text(dumps_toml(spec_data))
        paths.append(path)
    return paths


def run_spec_files(spec_paths: list[Path], out_dir: Path, jobs: int = 8) -> list[Path]:
    """Execute specs one process per run via the harness CLI; returns artifact paths.

    One process per run keeps the D1 once-per-process seeding contract;
    parallelism comes from independent single-core processes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_one(spec_path: Path) -> Path:
        # sys.executable with a static module entry point; pilot-local paths
        result = subprocess.run(  # noqa: S603 — static module entry, pilot-local args
            [sys.executable, '-m', 'experiments.harness.run', str(spec_path), '--out', str(out_dir)],
            cwd=repo_root(), capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            msg = f'pilot run failed for {spec_path.name}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}'
            raise RuntimeError(msg)
        spec = RunSpec.from_toml(spec_path)
        return out_dir / f'{spec.name}_seed{spec.seed}.h5'

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(run_one, spec_paths))


def load_run_metrics(artifact_path: Path, burn_fraction: float = 0.5, n_eff_block: int = 64, n_eff_blocks: int = 256, n_eff_seed: int = 271828) -> dict[str, float]:
    """Load the C1 co-primary metrics and supporting counts from an artifact.

    Round trips use the full event log; n_eff uses the post-burn stored
    cold samples with the frozen scramble-block estimator (minimum over
    parameters) and a fixed recorded analysis seed.

    The burn boundary is the LATER of the burn_fraction heuristic and, on
    adaptive artifacts, the recorded freeze iteration: everything before
    the freeze was spent tuning the ladder rather than sampling a fixed
    target, so a fractional burn alone would let adaptation-era samples
    into the metrics whenever the freeze lands past the fraction.
    """
    with h5py.File(str(artifact_path), 'r') as hf:
        events: NDArray[np.int64] = np.asarray(hf['events/rt_events'])
        segment_itrns: NDArray[np.int64] = np.asarray(hf['events/rt_segment_itrns'])
        samples: NDArray[np.floating] = np.asarray(hf['store/samples'])
        n_iterations = int(np.asarray(hf.attrs['n_iterations']).item())
        n_evals = int(np.asarray(hf.attrs['n_likelihood_evals']).item())
        n_chain = int(np.asarray(hf['ladder/Ts']).shape[0])
        wall_seconds = float(np.asarray(hf.attrs['wall_seconds']).item())
        store_thin = int(np.asarray(hf['store'].attrs['store_thin']).item())
        freeze_itrn = 0
        if 'ladder/history' in hf:
            frozen_block = int(np.asarray(hf['ladder/history'].attrs['frozen_block']).item())
            if frozen_block >= 0:
                freeze_itrn = frozen_block * int(np.asarray(hf.attrs['block_size']).item())

    burn_itrn = max(int(n_iterations * burn_fraction), freeze_itrn)
    n_burn_rows = burn_itrn // store_thin
    post_burn = samples[n_burn_rows:]
    # exclude burn-in round trips: keep events after the burn iteration
    post_events = events[events[:, 1] > burn_itrn]

    # Ladder-segment boundaries must ride along so adaptive artifacts do
    # not pair arrivals across ladder updates. Fixed-ladder artifacts
    # store an empty boundary array, for which segmentation is a no-op.
    rt_rate = round_trip_rate(post_events, n_chain, n_iterations - burn_itrn, segment_itrns=segment_itrns)
    n_eff = scramble_block_n_eff_min(post_burn, n_eff_block, n_eff_blocks, get_rng(n_eff_seed))
    total_trips = float(round_trip_counts(post_events, n_chain, segment_itrns=segment_itrns).sum())

    return {
        'rt_rate': float(rt_rate),
        'total_round_trips': total_trips,
        'n_eff_min': float(n_eff),
        'n_eff_per_eval': float(n_eff / n_evals),
        'n_likelihood_evals': float(n_evals),
        'n_chain': float(n_chain),
        'n_iterations': float(n_iterations),
        'wall_seconds': wall_seconds,
    }


def save_summary(name: str, payload: dict[str, Any]) -> Path:
    """Write a pilot summary JSON under the pilot root."""
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)
    path = PILOT_ROOT / f'{name}.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path
