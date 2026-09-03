"""Unit tests for construction of temperature ladders"""

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import DTMCMC.temperature_ladder_helpers as th

if TYPE_CHECKING:
    from numpy.typing import NDArray

TEST_DATA_DIR = 'tests/test_data/'
INVALID_N_COLD_PATTERN = r'(n cold cannot be more than total number of chain|n_cold \d+ not in \[0,)'

# set of parameters to use for several tests
test_set1 = [
    (1, 2, 1.0),
    (1, 3, 1.0),
    (2, 4, 1.0),
    (2, 32, 1.0),
    (1, 32, 1.0),
    (1, 32, 0.9),
    (2, 32, 0.9),
    (1, 32, 1.1),
    (2, 32, 1.1),
    (1, 32, 9.9),
    (2, 32, 9.9),
    (8, 8, 1.0),
    (7, 8, 1.0),
    (6, 8, 1.0),
    (5, 8, 1.0),
    (4, 8, 1.0),
    (3, 8, 1.0),
    (2, 8, 1.0),
    (1, 8, 1.0),
    (1, 1, 1.0),
    (1, 1, np.inf),
    (8, 8, np.inf),
    (4, 8, np.inf),
    (1, 8, np.inf),
    (2, 1, 1.0),
    (4, 3, 1.0),
    (4, 2, 1.0),
]


def unique_check_helper(
    Ts_in: NDArray[np.floating], T_cold: float, n_chain: int, n_cold: int, n_inf_final: int
) -> None:
    uniq, counts = np.unique(Ts_in, return_counts=True)
    print(Ts_in)
    print(T_cold, n_cold, n_inf_final)
    print(uniq)
    print(counts)

    if n_cold > 0:
        assert np.any(uniq == T_cold)
        arg_cold = int(np.argmax(uniq == T_cold))
        assert_array_equal(counts[(uniq != T_cold) & np.isfinite(uniq)], 1)
        if T_cold != np.inf:
            assert_array_equal(counts[arg_cold], n_cold)
            assert_array_equal(counts[~np.isfinite(uniq)], min(n_chain - n_cold, n_inf_final))
            assert uniq.size == n_chain - n_cold + 1 - max(min(n_chain - n_cold, n_inf_final) - 1, 0)
        else:
            assert_array_equal(counts[arg_cold], min(n_cold + n_inf_final, n_chain))
            assert uniq.size == n_chain + 1 - min(n_cold + n_inf_final, n_chain)
    else:
        assert uniq.size == n_chain

    if T_cold <= 1.0:
        assert np.all(Ts_in[n_cold:] > T_cold)  # check all higher temperatures warmer than cold chain


@pytest.mark.parametrize(('n_cold', 'n_chain', 'T_cold'), test_set1)
@pytest.mark.parametrize('n_inf_final', [0, 1, 2, 3, 4])
def test_entropy_spacing_fromfile_inf(n_cold: int, n_chain: int, T_cold: float, n_inf_final: int) -> None:
    """Test the entropy based spacing produces results that makes sense"""
    if n_cold > n_chain:
        with pytest.raises(ValueError, match=INVALID_N_COLD_PATTERN):
            T_ladder = th.entropy_ladder_fromfile(
                n_chain,
                n_cold,
                TEST_DATA_DIR + 'gal1_Ts_resample.npy',
                TEST_DATA_DIR + 'gal1_logL_var_resample.npy',
                n_inf_final=n_inf_final,
                T_cold=T_cold,
            )

        return

    if (n_cold + n_inf_final == n_chain and n_inf_final > 0) or n_cold + n_inf_final > n_chain:
        with pytest.warns(UserWarning):
            T_ladder = th.entropy_ladder_fromfile(
                n_chain,
                n_cold,
                TEST_DATA_DIR + 'gal1_Ts_resample.npy',
                TEST_DATA_DIR + 'gal1_logL_var_resample.npy',
                n_inf_final=n_inf_final,
                T_cold=T_cold,
            )
    else:
        T_ladder = th.entropy_ladder_fromfile(
            n_chain,
            n_cold,
            TEST_DATA_DIR + 'gal1_Ts_resample.npy',
            TEST_DATA_DIR + 'gal1_logL_var_resample.npy',
            n_inf_final=n_inf_final,
            T_cold=T_cold,
        )
    Ts_in = T_ladder.Ts

    finite_mask = np.isfinite(Ts_in)
    positive_mask = Ts_in > 0
    joint_mask = finite_mask & positive_mask

    # Strict technical requirements for non-negative temperatures
    assert Ts_in.size == n_chain  # check correct number of chains
    assert np.sum(Ts_in == T_cold) >= n_cold  # check correct number of cold chains
    assert not np.any(Ts_in < 0.0)  # check no negative temperature chains
    assert_array_equal(T_ladder.Ts, Ts_in)  # check object matches
    assert_array_equal(T_ladder.betas[joint_mask], 1.0 / Ts_in[joint_mask])  # check inverses match
    assert_array_equal(Ts_in[T_ladder.betas == 0.0], np.inf)  # check inverses match
    assert_array_equal(T_ladder.betas[Ts_in == 0.0], np.inf)  # check inverses match

    # Not technically required, but expected in this test case
    if T_cold != np.inf:
        assert np.sum(Ts_in == T_cold) == n_cold  # check correct number of cold chains
        if n_chain > n_cold:
            assert np.sum(~finite_mask) == min(
                n_inf_final, n_chain - n_cold
            )  # check 1 infinite temperature chain exists
        n_nonfinite = np.sum(~finite_mask)
        assert np.all(
            np.diff(Ts_in[n_cold : min(Ts_in.size, Ts_in.size - n_nonfinite + 1)]) >= 0.0
        )  # check non-cold chains are sorted
    else:
        assert np.sum(Ts_in == T_cold) == min(n_cold + n_inf_final, n_chain)

    unique_check_helper(Ts_in, T_cold, n_chain, n_cold, n_inf_final)


@pytest.mark.parametrize(('n_cold', 'n_chain', 'T_cold'), test_set1)
@pytest.mark.parametrize('n_inf_final', [0, 1, 2, 3, 4])
def test_geometric_spacing_inf(n_cold: int, n_chain: int, T_cold: float, n_inf_final: int) -> None:
    """Test the entropy based spacing produces results that makes sense"""
    T_min = 1.0
    T_max = 1000.0

    if n_cold > n_chain:
        with pytest.raises(ValueError, match=INVALID_N_COLD_PATTERN):
            betas_in, Ts_in = th.geometric_spaced_betas(n_chain, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final)

        return

    if T_cold == np.inf:
        with pytest.raises(AssertionError):
            betas_in, Ts_in = th.geometric_spaced_betas(n_chain, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final)
        return
    if (n_chain == n_cold + n_inf_final and n_inf_final > 0) or n_chain < n_cold + n_inf_final:
        with pytest.warns(UserWarning):
            betas_in, Ts_in = th.geometric_spaced_betas(n_chain, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final)
        with pytest.warns(UserWarning):
            T_ladder = th.GeometricTemperatureLadder(n_chain, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final)
    else:
        betas_in, Ts_in = th.geometric_spaced_betas(n_chain, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final)
        T_ladder = th.GeometricTemperatureLadder(n_chain, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final)

    finite_mask = np.isfinite(Ts_in)
    positive_mask = Ts_in > 0
    joint_mask = finite_mask & positive_mask

    # Strict technical requirements for non-negative temperatures
    assert Ts_in.size == n_chain  # check correct number of chains
    assert np.sum(Ts_in == T_cold) >= n_cold  # check correct number of cold chains
    assert not np.any(Ts_in < 0.0)  # check no negative temperature chains
    assert_array_equal(T_ladder.Ts, Ts_in)  # check object matches
    assert_array_equal(T_ladder.betas[joint_mask], 1.0 / Ts_in[joint_mask])  # check inverses match
    assert_array_equal(Ts_in[T_ladder.betas == 0.0], np.inf)  # check inverses match
    assert_array_equal(T_ladder.betas[Ts_in == 0.0], np.inf)  # check inverses match
    assert_array_equal(T_ladder.Ts, Ts_in)  # check object matches
    assert_array_equal(T_ladder.betas, betas_in)  # check object matches

    # Not technically required, but expected in this test case
    if T_cold != np.inf:
        assert np.sum(Ts_in == T_cold) == n_cold  # check correct number of cold chains
        if n_chain > n_cold:
            assert np.sum(~finite_mask) == min(
                n_inf_final, n_chain - n_cold
            )  # check 1 infinite temperature chain exists
        n_nonfinite = np.sum(~finite_mask)
        assert np.all(
            np.diff(Ts_in[n_cold : min(Ts_in.size, Ts_in.size - n_nonfinite + 1)]) >= 0.0
        )  # check non-cold chains are sorted
    else:
        assert np.sum(Ts_in == T_cold) == min(n_cold + n_inf_final, n_chain)

    unique_check_helper(Ts_in, T_cold, n_chain, n_cold, n_inf_final)


if __name__ == '__main__':
    pytest.cmdline.main(['tests/temperature_ladder_tests.py'])
