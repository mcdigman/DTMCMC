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
from experiments.metrics import round_trip_rate, scramble_block_n_eff_min

if TYPE_CHECKING:
    from numpy.typing import NDArray

PILOT_ROOT = repo_root() / 'artifacts' / 'pilots'

# pilot-standard proposal overrides: production-plausible DE buffer that
# still initializes quickly, quiet Fisher updates
PILOT_PROPOSALS: dict[str, dict[str, object]] = {
    'FisherJumpManager': {'verbose_fisher': False},
    'DEJumpManager': {'de_size': 10000},
}

GOLD_LADDER_FILES = {'Ts_file': 'data/Ts_cake_gold.npy', 'vars_file': 'data/vars_cake_gold.npy'}


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
            'n_record': -1,
            'checkpoint_every_blocks': max(n_steps // block_size // 4, 1),
        },
        'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
        'proposals': dict(PILOT_PROPOSALS),
    }


def cake5_likelihood() -> dict[str, Any]:
    """The production cake 5D likelihood table."""
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
    """
    with h5py.File(str(artifact_path), 'r') as hf:
        events: NDArray[np.int64] = np.asarray(hf['events/rt_events'])
        samples: NDArray[np.floating] = np.asarray(hf['store/samples'])
        n_iterations = int(np.asarray(hf.attrs['n_iterations']).item())
        n_evals = int(np.asarray(hf.attrs['n_likelihood_evals']).item())
        n_chain = int(np.asarray(hf['ladder/Ts']).shape[0])
        wall_seconds = float(np.asarray(hf.attrs['wall_seconds']).item())

    n_burn_rows = int(samples.shape[0] * burn_fraction)
    post_burn = samples[n_burn_rows:]
    # exclude burn-in round trips proportionally: keep events after the burn iteration
    burn_itrn = int(n_iterations * burn_fraction)
    post_events = events[events[:, 1] > burn_itrn]

    rt_rate = round_trip_rate(post_events, n_chain, n_iterations - burn_itrn)
    n_eff = scramble_block_n_eff_min(post_burn, n_eff_block, n_eff_blocks, get_rng(n_eff_seed))
    total_trips = float(np.sum(np.minimum(
        np.bincount(post_events[post_events[:, 2] == 0][:, 0], minlength=n_chain),
        np.bincount(post_events[post_events[:, 2] == 1][:, 0], minlength=n_chain),
    )))

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
