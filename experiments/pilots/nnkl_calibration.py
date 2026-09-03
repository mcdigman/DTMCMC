"""Pilot: NN-KL snapshot size and checkpoint cadence (plan §4 Phase 4).

Measures the self-KL noise floor and wall cost of the NN-KL estimator
versus snapshot size on the production likelihoods' exact references —
the estimator is O(n^2), so the chosen size must resolve the quality
thresholds without dominating checkpoint cost.
"""

import time
from typing import TYPE_CHECKING

import numpy as np

from DTMCMC.rng_helpers import get_rng
from experiments.metrics import nn_kl
from experiments.pilots.common import save_summary
from experiments.reference_samplers import draw_cake, draw_eggbox, draw_truncated_gaussian

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

SNAPSHOT_SIZES = [1000, 2000, 5000, 10000]
N_REPEATS = 16
RNG_SEED = 41


def main() -> int:
    """Measure self-KL noise and timing across snapshot sizes."""
    rng = get_rng(RNG_SEED)
    references: dict[str, Callable[[int], NDArray[np.floating]]] = {
        'gaussian5d': lambda n: draw_truncated_gaussian(n, 5, 10.0, rng),
        'cake5d': lambda n: draw_cake(n, 5, rng),
        'eggbox5d': lambda n: draw_eggbox(n, 5, rng),
    }

    # pay nn_kl's one-time numba compilation before any timed call
    nn_kl(draw_truncated_gaussian(256, 5, 10.0, rng), draw_truncated_gaussian(256, 5, 10.0, rng), 256, rng)

    results: dict[str, object] = {'snapshot_sizes': SNAPSHOT_SIZES, 'n_repeats': N_REPEATS, 'rng_seed': RNG_SEED}
    for ref_name, draw in references.items():
        noise_by_size = {}
        cost_by_size = {}
        for size in SNAPSHOT_SIZES:
            values = np.zeros(N_REPEATS)
            elapsed = 0.0
            for itrr in range(N_REPEATS):
                # draw outside the timed section: the cost column is the
                # per-checkpoint nn_kl evaluation alone
                ref_draw, test_draw = draw(size), draw(size)
                start = time.perf_counter()
                values[itrr] = nn_kl(ref_draw, test_draw, size, rng)
                elapsed += time.perf_counter() - start
            elapsed /= N_REPEATS
            noise_by_size[str(size)] = {
                'mean': float(values.mean()),
                'sd': float(values.std()),
                'max_abs': float(np.abs(values).max()),
            }
            cost_by_size[str(size)] = elapsed
            print(
                f'{ref_name:>10} n={size:>6}: self-KL {values.mean():+.3f} +- {values.std():.3f} (max |.| {np.abs(values).max():.3f}), {elapsed:.2f} s/eval'
            )
        results[ref_name] = {'noise': noise_by_size, 'seconds_per_eval': cost_by_size}

    save_summary('nnkl_calibration', results)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
