"""Regression guards for raw TemperatureLadder constructor validation."""

import numpy as np
import pytest

from DTMCMC.temperature_ladder_helpers import TemperatureLadder


def test_temperature_ladder_rejects_non_1d_temperatures() -> None:
    with pytest.raises(ValueError, match='Ts_in must be 1D'):
        TemperatureLadder(np.array([[1.0, 2.0]]), n_cold=1)


def test_temperature_ladder_rejects_empty_temperatures() -> None:
    with pytest.raises(ValueError, match='zero temperatures specified'):
        TemperatureLadder(np.array([], dtype=np.float64), n_cold=0)


def test_temperature_ladder_rejects_nan_temperatures() -> None:
    with pytest.raises(ValueError, match='input temperatures were nan'):
        TemperatureLadder(np.array([1.0, np.nan, np.inf]), n_cold=1)


def test_temperature_ladder_rejects_nan_t_cold() -> None:
    with pytest.raises(ValueError, match='T_cold of nan'):
        TemperatureLadder(np.array([1.0, 2.0]), T_cold=np.nan, n_cold=1)


@pytest.mark.parametrize('n_cold', [-1, 3])
def test_temperature_ladder_rejects_n_cold_out_of_range(n_cold: int) -> None:
    with pytest.raises(ValueError, match=r'n_cold .* not in \[0, 2\]'):
        TemperatureLadder(np.array([1.0, 2.0]), n_cold=n_cold)


@pytest.mark.parametrize('n_cold', [0.5, True, False, np.bool_(True)])
def test_temperature_ladder_rejects_non_integer_n_cold(n_cold: object) -> None:
    with pytest.raises(TypeError, match='n_cold must be an integer'):
        TemperatureLadder(np.array([1.0, 2.0]), n_cold=n_cold)  # type: ignore[arg-type]


def test_temperature_ladder_accepts_numpy_integer_n_cold() -> None:
    ladder = TemperatureLadder(np.array([1.0, 2.0]), n_cold=np.int64(1))  # type: ignore[arg-type]

    assert ladder.n_cold == 1
    assert np.array_equal(ladder.get_arg_cold(), [0])


def test_temperature_ladder_accepts_all_cold_rungs() -> None:
    ladder = TemperatureLadder(np.array([1.0, 1.0]), T_cold=1.0, n_cold=2)

    assert ladder.n_cold == 2
    assert np.array_equal(ladder.get_arg_cold(), [0, 1])


def test_temperature_ladder_rejects_missing_t_cold_rungs() -> None:
    with pytest.raises(ValueError, match='at least n_cold=1 values of T_cold=3\\.0'):
        TemperatureLadder(np.array([1.0, 2.0]), T_cold=3.0, n_cold=1)


def test_temperature_ladder_rejects_too_few_t_cold_rungs() -> None:
    with pytest.raises(ValueError, match='at least n_cold=2 values of T_cold=1\\.0'):
        TemperatureLadder(np.array([1.0, 2.0, 3.0]), T_cold=1.0, n_cold=2)


def test_temperature_ladder_rejects_unknown_sort_mode() -> None:
    with pytest.raises(ValueError, match='Unrecognized option sort_mode 2'):
        TemperatureLadder(np.array([1.0, 2.0]), sort_mode=2, n_cold=1)


def test_temperature_ladder_allows_nonfinite_non_nan_temperatures() -> None:
    ladder = TemperatureLadder(np.array([1.0, np.inf]), T_cold=np.inf, n_cold=1)

    assert np.array_equal(ladder.Ts, [1.0, np.inf])
    assert np.array_equal(ladder.get_arg_cold(), [1])
