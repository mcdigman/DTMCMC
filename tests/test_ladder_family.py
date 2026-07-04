"""Phase 3 acceptance tests: generalized spacing, ladder family, tunable cake.

The regression fixture (entropy_ladder_regression.npz) was generated from
the pre-Phase-3 EntropyTemperatureLadder on data/*_gold inputs, so the
exact-reproduction test is a genuine behavior guard across the
generalized-machinery refactor, not a tautology.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from DTMCMC.temperature_ladder_helpers import EntropyTemperatureLadder

FIXTURE_PATH = Path(__file__).parent / 'test_data' / 'entropy_ladder_regression.npz'
DATA_DIR = Path(__file__).parents[1] / 'data'

ENTROPY_REGRESSION_CONFIGS = [
    (8, 1, 1, 1.0, False),
    (16, 2, 1, 1.0, False),
    (12, 1, 0, 1.0, False),
    (16, 1, 2, 1.0, True),
    (9, 3, 1, 0.9, False),
]


@pytest.fixture(scope='module')
def gold_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Load the cake gold ladder inputs."""
    return np.load(DATA_DIR / 'Ts_cake_gold.npy'), np.load(DATA_DIR / 'vars_cake_gold.npy')


@pytest.mark.parametrize(('n_chain', 'n_cold', 'n_inf_final', 'T_cold', 'correct_last'), ENTROPY_REGRESSION_CONFIGS)
def test_entropy_ladder_exact_reproduction(gold_inputs, n_chain, n_cold, n_inf_final, T_cold, correct_last) -> None:
    """Acceptance 1: the refactored machinery reproduces the frozen fixture exactly."""
    Ts_in, vars_in = gold_inputs
    fixture = np.load(FIXTURE_PATH)
    key = f'{n_chain}_{n_cold}_{n_inf_final}_{T_cold}_{correct_last}'
    ladder = EntropyTemperatureLadder(n_chain, Ts_in, vars_in, n_cold=n_cold, T_cold=T_cold, n_inf_final=n_inf_final, correct_last=correct_last)
    assert_array_equal(ladder.Ts, fixture[key])
