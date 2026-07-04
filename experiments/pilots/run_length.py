"""Pilot: run-length calibration for round-trip-rate precision (plan §4 Phase 4).

The round-trip rate's relative error scales ~1/sqrt(total round trips),
so 100 post-burn round trips per run gives ~10%. This module runs one
long reference run to verify rate stationarity (the extrapolation is
only valid if the rate is stable in time), then combines the arm-pilot
counts into required E1/E2 run lengths per arm.
"""

import json

import h5py
import numpy as np

from experiments.metrics import round_trip_rate
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

LONG_STEPS = 1048576
TARGET_ROUND_TRIPS = 100.


def main() -> int:
    """Run the long stationarity check and derive per-arm run lengths."""
    spec = make_spec('runlen_long', 4401, cake5_likelihood(), entropy_gold_ladder(16), LONG_STEPS)
    spec_paths = write_specs([spec], PILOT_ROOT / 'runlen' / 'specs')
    artifact_path = run_spec_files(spec_paths, PILOT_ROOT / 'runlen')[0]

    with h5py.File(str(artifact_path), 'r') as hf:
        events = np.asarray(hf['events/rt_events'])
        n_iterations = int(np.asarray(hf.attrs['n_iterations']).item())
    half = n_iterations // 2
    rate_first = round_trip_rate(events[(events[:, 1] > n_iterations // 4) & (events[:, 1] <= half)], 16, half - n_iterations // 4)
    rate_second = round_trip_rate(events[events[:, 1] > half], 16, n_iterations - half)
    long_metrics = load_run_metrics(artifact_path)
    print(f'long run: rate {long_metrics["rt_rate"]:.3f}; first half (post 25% burn) {rate_first:.3f} vs second half {rate_second:.3f}')

    arms = json.loads((PILOT_ROOT / 'arm_power.json').read_text())
    n_steps_pilot = arms['n_steps']
    lengths: dict[str, object] = {}
    for arm_name, arm_metrics in arms['arms'].items():
        mean_rts = float(np.mean(arm_metrics['total_round_trips']))
        if mean_rts > 0.:
            # post-burn half of the pilot produced mean_rts round trips
            required = TARGET_ROUND_TRIPS / mean_rts * (n_steps_pilot / 2.)
        else:
            required = float('inf')
        lengths[arm_name] = {'pilot_round_trips': mean_rts, 'steps_for_100_rts_post_burn': required}
        print(f'{arm_name:>18}: pilot RTs {mean_rts:7.1f} -> steps for 100 post-burn RTs: {required:.3e}')

    save_summary('run_length', {
        'long_run': {
            'n_steps': LONG_STEPS,
            'rate_overall_post_burn': long_metrics['rt_rate'],
            'rate_first_half': float(rate_first),
            'rate_second_half': float(rate_second),
        },
        'per_arm': lengths,
        'target_round_trips': TARGET_ROUND_TRIPS,
    })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
