"""Fixed-spec throughput benchmark for the Phase 2 <5% regression criterion.

Runs the benchmark spec once per process: 2 warm-up blocks (JIT), then
three timed segments, reporting the best (plan §4 Phase 2 acceptance 5:
best-of-3, measured locally; CI timing is never used). Usage::

    python -m experiments.benchmark_throughput [spec.toml]
"""

import sys
import time

from DTMCMC.rng_helpers import seed_run

from .harness.paths import chdir_repo_root, repo_root, resolve
from .harness.runner import build_sampler
from .harness.spec import RunSpec

WARMUP_BLOCKS = 2
SEGMENTS = 3


def run_benchmark(spec_path: str) -> float:
    """Run the benchmark and return the best chain-steps/s over the segments."""
    chdir_repo_root()
    spec = RunSpec.from_toml(resolve(spec_path))
    seed_run(spec.seed)
    sampler, _like_obj = build_sampler(spec)

    for _ in range(WARMUP_BLOCKS):
        sampler.advance_block()

    blocks_per_segment = (spec.n_blocks - WARMUP_BLOCKS) // SEGMENTS
    best_rate = 0.0
    for _ in range(SEGMENTS):
        start = time.perf_counter()
        for _ in range(blocks_per_segment):
            sampler.advance_block()
        elapsed = time.perf_counter() - start
        rate = blocks_per_segment * spec.block_size * spec.n_chain / elapsed
        best_rate = max(best_rate, rate)
        print(f'segment: {rate:.4e} chain-steps/s ({elapsed:.3f} s for {blocks_per_segment} blocks)')

    print(f'best-of-{SEGMENTS}: {best_rate:.4e} chain-steps/s')
    return best_rate


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = argv if argv is not None else sys.argv[1:]
    spec_path = args[0] if args else str(repo_root() / 'experiments' / 'specs' / 'bench_cake16.toml')
    run_benchmark(spec_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
