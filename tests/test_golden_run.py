"""Golden-run regression test (methods-paper plan §8, Phase 2 acceptance 1).

A short fixed-seed cake run must produce a bit-exact digest of the stored
logLs/samples and the final chain state. The digest guards the RNG-stream
discipline (plan D5): any engine change that adds, removes, or reorders a
random draw in the sampling path changes the digest and fails this test.

Digests are per-platform (numba/libm codegen is not bit-identical across
architectures): the CI digest is blessed from a CI run, local-dev digests
are blessed locally. Re-blessing requires explicit justification in the
commit message. The run-twice check needs no blessed digest and guards
in-process determinism unconditionally.
"""

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from DTMCMC.rng_helpers import reset_seed_guard_for_tests, seed_run
from experiments.harness.runner import build_sampler
from experiments.harness.spec import RunSpec

DIGEST_FILE = Path(__file__).parent / 'test_data' / 'golden_digests.json'

GOLDEN_SPEC: dict[str, Any] = {
    'name': 'golden_cake',
    'seed': 20260704,
    'likelihood': {'name': 'cake', 'n_par': 5, 'cutoff': 10},
    'ladder': {'kind': 'geometric', 'n_chain': 8, 'n_cold': 1, 'T_cold': 1.0, 'T_min': 1.0, 'T_max': 1000.0, 'n_inf_final': 1},
    'run': {'n_steps': 256, 'block_size': 64, 'store_thin': 1, 'checkpoint_every_blocks': 4},
    'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
    'proposals': {
        'FisherJumpManager': {'verbose_fisher': False},
        'DEJumpManager': {'de_size': 128},
    },
}


def platform_key() -> str:
    """Get the digest key for this platform (OS + CPU architecture)."""
    return f'{sys.platform}-{platform.machine()}'


def run_golden_digest() -> str:
    """Run the golden spec from a fresh seed and digest its outputs.

    The digest covers the stored logLs and samples plus the full final
    state (samples, logLs, walker identity track) of the last block edge.
    """
    reset_seed_guard_for_tests()
    spec = RunSpec.from_dict(GOLDEN_SPEC)
    seed_run(spec.seed)
    sampler, _like_obj = build_sampler(spec)
    for _ in range(spec.n_blocks):
        sampler.advance_block()
    reset_seed_guard_for_tests()

    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(sampler.logLs_store).tobytes())
    hasher.update(np.ascontiguousarray(sampler.samples_store).tobytes())
    hasher.update(np.ascontiguousarray(sampler.samples[0]).tobytes())
    hasher.update(np.ascontiguousarray(sampler.logLs[0]).tobytes())
    hasher.update(np.ascontiguousarray(sampler.chain_track[0]).tobytes())
    return hasher.hexdigest()


@pytest.fixture(scope='module')
def golden_digests() -> tuple[str, str]:
    """Run the golden spec twice in this process and return both digests."""
    return run_golden_digest(), run_golden_digest()


def test_golden_run_twice_deterministic(golden_digests: tuple[str, str]) -> None:
    """Two golden runs in one process are bit-exact (no blessed digest needed)."""
    assert golden_digests[0] == golden_digests[1]


def test_golden_digest_matches_blessed(golden_digests: tuple[str, str]) -> None:
    """The golden digest matches the blessed digest for this platform."""
    key = platform_key()
    blessed: dict[str, str] = json.loads(DIGEST_FILE.read_text()) if DIGEST_FILE.is_file() else {}
    if key not in blessed:
        pytest.fail(
            f'no blessed golden digest for platform {key!r}; computed digest is\n'
            f'    "{key}": "{golden_digests[0]}"\n'
            f'add it to {DIGEST_FILE} (re-blessing an EXISTING digest requires justification in the commit message)'
        )
    assert golden_digests[0] == blessed[key], (
        f'golden digest changed on {key}: an engine change altered the RNG stream or sampling arithmetic '
        f'(expected {blessed[key]}, got {golden_digests[0]}); '
        'if intentional, re-bless with justification in the commit message (plan §8)'
    )
