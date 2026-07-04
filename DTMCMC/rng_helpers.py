"""C 2026 Matthew C. Digman
Run-level seeding helpers for reproducible sampling runs.

The sampler draws from two independent RNG streams: numba's njit-internal
stream (all hot-loop draws inside @njit helpers) and numpy's legacy global
stream (pure-Python draws, e.g. Fisher jumps). Reproducibility requires
seeding both once at run start, which is the only supported way to seed
anything: reseeding APIs are lint-banned (ruff TID251) outside this module.
"""

import numpy as np
from numba import njit

# guard state so a run can only be seeded once (D1); a dict avoids
# rebinding a module global from inside seed_run
_seed_guard: dict[str, bool] = {'seeded': False}


@njit()
def seed_numba(seed: int) -> None:
    """Seed numba's njit-internal RNG stream.

    Seeding must happen inside an @njit function: numba's stream is
    independent of np.random.seed called from Python. The stream is
    per-thread, but the sampler hot loop is single-threaded, so one
    call at run start suffices.
    """
    np.random.seed(seed)


def derive_child_seeds(run_seed: int) -> tuple[int, int]:
    """Deterministically derive the two child seeds for a run seed.

    Returns
    -------
    child_python: int
        Seed for the numpy legacy global stream
    child_numba: int
        Seed for the numba njit-internal stream
    """
    child_states = np.random.SeedSequence(run_seed).generate_state(2)
    return int(child_states[0]), int(child_states[1])


def seed_run(run_seed: int) -> tuple[int, int]:
    """Seed both RNG streams from a single run seed, once per run.

    Derives two child seeds from run_seed, seeds the numpy legacy global
    stream with the first and the numba njit-internal stream with the
    second, and returns both so they can be recorded in the run artifact.

    Raises
    ------
    RuntimeError
        If called more than once in a process (nothing may reseed after
        run start; tests may use reset_seed_guard_for_tests)
    """
    if _seed_guard['seeded']:
        msg = 'seed_run may only be called once per run; reseeding after run start is forbidden'
        raise RuntimeError(msg)

    child_python, child_numba = derive_child_seeds(run_seed)
    np.random.seed(child_python)
    seed_numba(child_numba)
    _seed_guard['seeded'] = True
    return child_python, child_numba


def reset_seed_guard_for_tests() -> None:
    """Reset the once-per-run guard so seed_run can be called again.

    Exists only for tests that legitimately reseed (golden-run,
    determinism, and post-freeze equivalence tests); production code
    must never call it.
    """
    _seed_guard['seeded'] = False


def get_rng(seed: int) -> np.random.Generator:
    """Get an independent Generator for analysis code.

    Analysis code (e.g. bootstrap resampling) must not touch the run
    streams; it obtains Generators here with the seed recorded in the
    analysis outputs, so analyses stay reproducible without adding
    lint-whitelist entries.
    """
    return np.random.default_rng(seed)
