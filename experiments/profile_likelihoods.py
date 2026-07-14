"""Relative performance profile of the registered benchmark likelihoods.

For every target in ``experiments.benchmarks.BENCHMARKS`` this builds the
likelihood object at its canonical ``default_params`` and times the hot
per-step calls the sampler actually makes: ``prior_draw``, ``get_loglike``,
and ``correct_bounds``. Several likelihoods JIT-compile these through numba,
so every function is warmed (compiled and run a few times) before any timing
segment starts — the reported numbers are steady-state per-call costs, never
one-shot compilation.

Each measurement uses ``timeit`` autoranging for a stable iteration count and
reports the best of several repeats (best-of, not mean, to reject scheduler
noise). Results print as a table: absolute microseconds per call and, per
column, the cost relative to the fastest likelihood, so the numbers are
comparable across targets rather than absolute wall-clock promises. Usage::

    python -m experiments.profile_likelihoods [--repeats N] [--seed S]
"""

import argparse
import timeit
from typing import TYPE_CHECKING, Any

import numpy as np

from DTMCMC.rng_helpers import get_rng, seed_run

from .benchmarks import BENCHMARKS
from .harness.paths import chdir_repo_root
from .harness.runner import LIKELIHOOD_BUILDERS

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# functions profiled, in table-column order
FUNCTIONS = ('prior_draw', 'get_loglike', 'correct_bounds')

WARMUP_CALLS = 5
DEFAULT_REPEATS = 7
DEFAULT_SEED = 314159
# size of the pre-built pool of perturbed points fed to correct_bounds, so
# each timed call sees a distinct (often out-of-bounds) input rather than a
# single point that the first call reflects into range and every later call
# then hits on the trivial in-bounds path
POOL_SIZE = 512
# Gaussian step scale used to knock prior draws off their valid point; large
# enough that a realistic fraction of the pool lands outside the prior box
PERTURB_SCALE = 3.0


def _best_seconds_per_call(func: Callable[[], Any], repeats: int) -> float:
    """Return the best per-call time (seconds) for a zero-arg callable.

    Uses ``timeit`` autoranging to pick an iteration count worth at least
    ~0.2 s, then takes the minimum over ``repeats`` to reject noise.
    """
    timer = timeit.Timer(func)
    number, _ = timer.autorange()
    best = min(timer.repeat(repeat=repeats, number=number))
    return best / number


def _build_perturb_pool(like: Any, seed: int) -> NDArray[np.floating]:
    """Build a pool of prior draws perturbed off their valid points.

    Fresh points are needed because ``correct_bounds`` reflects in place, so
    reusing one array would measure only the trivial already-in-bounds path.
    """
    rng = get_rng(seed)
    base = np.array([like.prior_draw() for _ in range(POOL_SIZE)], dtype=np.float64)
    base += rng.normal(scale=PERTURB_SCALE, size=base.shape)
    return base


def profile_likelihood(name: str, repeats: int, seed: int) -> tuple[dict[str, float | None], int]:
    """Time the profiled functions for one benchmark likelihood.

    Returns a per-function map of best microseconds per call (a value is
    ``None`` when the likelihood does not support that call) and the object's
    parameter dimension. The caller must have already ``seed_run`` once.
    """
    target = BENCHMARKS[name]
    builder = LIKELIHOOD_BUILDERS[name]
    like = builder(**target.default_params)

    # a valid point for get_loglike, and a fresh pool for correct_bounds
    point = np.asarray(like.prior_draw(), dtype=np.float64)
    pool = _build_perturb_pool(like, seed)
    pool_len = pool.shape[0]

    # warm every JIT path before timing anything
    for _ in range(WARMUP_CALLS):
        like.prior_draw()
        like.get_loglike(point)
        like.correct_bounds(pool[0].copy())

    # correct_bounds mutates its argument, so cycle distinct copies; the copy
    # cost is measured separately below and subtracted out
    counter = {'i': 0}

    def call_correct_bounds() -> None:
        idx = counter['i'] % pool_len
        counter['i'] += 1
        like.correct_bounds(pool[idx].copy())

    def copy_baseline() -> None:
        idx = counter['i'] % pool_len
        counter['i'] += 1
        _ = pool[idx].copy()

    results: dict[str, float | None] = {}
    results['prior_draw'] = _best_seconds_per_call(like.prior_draw, repeats) * 1e6
    results['get_loglike'] = _best_seconds_per_call(lambda: like.get_loglike(point), repeats) * 1e6
    net = _best_seconds_per_call(call_correct_bounds, repeats) - _best_seconds_per_call(copy_baseline, repeats)
    results['correct_bounds'] = max(net, 0.0) * 1e6
    return results, int(like.n_par)


def _format_table(rows: dict[str, dict[str, float | None]], n_pars: dict[str, int]) -> str:
    """Render the results as an aligned text table with per-column relatives."""
    # per-column minimum over the likelihoods that support the call
    col_min: dict[str, float] = {}
    for func in FUNCTIONS:
        vals = [v for r in rows.values() if (v := r[func]) is not None]
        col_min[func] = min(vals) if vals else float('nan')

    name_w = max(len('likelihood'), *(len(n) for n in rows))
    cell_w = 20  # "  1234.567 (12.3x)"
    header = f'{"likelihood":<{name_w}}  {"n_par":>5}'
    for func in FUNCTIONS:
        header += f'  {func:>{cell_w}}'
    lines = [header, '-' * len(header)]

    for name, res in rows.items():
        line = f'{name:<{name_w}}  {n_pars[name]:>5}'
        for func in FUNCTIONS:
            val = res[func]
            if val is None:
                cell = 'n/a'
            else:
                rel = val / col_min[func] if col_min[func] else float('nan')
                cell = f'{val:9.3f} us ({rel:5.1f}x)'
            line += f'  {cell:>{cell_w}}'
        lines.append(line)

    footer = '\nunits: microseconds per call (best of {repeats}); (Nx) = cost relative to fastest in column'
    return '\n'.join(lines) + footer


def run(repeats: int, seed: int) -> dict[str, dict[str, float | None]]:
    """Profile every registered benchmark likelihood and print the table."""
    chdir_repo_root()
    # seed_run may run only once per process; it makes the numba-global-stream
    # prior_draw reproducible across invocations
    seed_run(seed)
    rows: dict[str, dict[str, float | None]] = {}
    n_pars: dict[str, int] = {}
    for name in BENCHMARKS:
        print(f'profiling {name} ...', flush=True)
        res, n_par = profile_likelihood(name, repeats, seed)
        rows[name] = res
        n_pars[name] = n_par

    print()
    print(_format_table(rows, n_pars).replace('{repeats}', str(repeats)))
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=DEFAULT_REPEATS, help='best-of timing repeats per measurement')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='run seed for reproducible prior draws')
    args = parser.parse_args(argv)
    run(args.repeats, args.seed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
