"""Pilot: throughput table across likelihood x n_par x n_chain (plan §4 Phase 4).

Each configuration runs the best-of-3 benchmark in its own process,
sequentially (single-core measurements must not compete for cores).
"""

import re
import subprocess
import sys
from typing import Any

from experiments.harness.paths import repo_root
from experiments.pilots.common import PILOT_ROOT, make_spec, save_summary, write_specs

THROUGHPUT_CONFIGS: list[tuple[str, dict[str, Any]]] = [
    *[
        (f'gaussian{n_par}d_nc{n_chain}', {'name': 'gaussian', 'n_par': n_par, 'cutoff': 8})
        for n_par in (5, 20)
        for n_chain in (8, 16, 48)
    ],
    *[
        (f'cake{n_par}d_nc{n_chain}', {'name': 'cake', 'n_par': n_par, 'cutoff': 10})
        for n_par in (5, 8)
        for n_chain in (8, 16, 48)
    ],
    *[(f'eggbox5d_nc{n_chain}', {'name': 'eggbox', 'n_par': 5}) for n_chain in (8, 16, 48)],
]


def config_n_chain(config_name: str) -> int:
    """Extract the chain count from a throughput config name."""
    match = re.search(r'nc(\d+)$', config_name)
    assert match is not None
    return int(match.group(1))


def main() -> int:
    """Run the throughput battery and write the summary table."""
    spec_dir = PILOT_ROOT / 'throughput' / 'specs'
    rates: dict[str, float] = {}
    for config_name, likelihood in THROUGHPUT_CONFIGS:
        n_chain = config_n_chain(config_name)
        ladder = {'kind': 'geometric', 'n_chain': n_chain, 'n_cold': 1, 'T_max': 1000.0}
        spec_data = make_spec(f'bench_{config_name}', 977, likelihood, ladder, n_steps=32768, store_thin=64)
        spec_path = write_specs([spec_data], spec_dir)[0]
        result = subprocess.run(  # noqa: S603 — static module entry, pilot-local args
            [sys.executable, '-m', 'experiments.benchmark_throughput', str(spec_path)],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        best_line = [line for line in result.stdout.splitlines() if line.startswith('best-of-')][-1]
        rate = float(best_line.split(':')[1].split('chain-steps')[0])
        rates[config_name] = rate
        print(f'{config_name:>20}: {rate:.4e} chain-steps/s')

    save_summary('throughput', {'chain_steps_per_second': rates})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
