"""Try to replicate and test for a glitch in temperature ladder generation"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import DTMCMC.temperature_ladder_helpers as tlh


def T_prediction_sensibility_check(Ts, T_min=1.0, T_cold=1.0, n_cold=1) -> None:
    assert np.all(Ts >= 0.0)  # negative temperatures make no sense here
    assert np.all(Ts != 0.0)  # zero temperatures could be meaningful, but the code should not be generating them
    # assert Ts[0] == T_cold
    assert_array_equal(Ts[:n_cold], T_cold)
    if n_cold == 0 and np.all(~np.isfinite(Ts)):  # exit the case where all chains are infinite
        return
    n_cut = max(n_cold, np.argmax(Ts != T_cold))

    if np.any(Ts != T_cold):
        assert np.all(
            Ts[n_cut:] >= T_min - 1.0e-13
        )  # if there is a specified minimum temperature, check no temperatures are below it
        assert np.all(Ts[:n_cut] == T_cold)  # assume starts with a block of cold temperatures
    if T_cold > T_min:
        assert np.all(np.diff(Ts[n_cut:]) >= 0.0)  # handle case where first several values may not be in order
    else:
        assert np.all(
            np.diff(Ts) >= 0.0
        )  # we expect the temperatures to be sorted, although it is not actually a requirement

    assert np.sum(~np.isfinite(Ts)) <= 1  # we expect at most one infinite temperature chain to be generated
    if np.any(Ts != T_cold):
        assert np.all(np.diff(Ts[n_cut:]) > 0.0)  # with the current code all Ts > 1 should be unique


def test_known_result() -> None:
    """Test a grid where we know the correct result analytically"""
    n_chain_in = 100
    n_chain_need = 100
    n_cold = 1
    T_cold = 1.0
    T_min = 1.0
    T_max = 1000.0
    n_inf_final_in = 0
    n_inf_final_out = 0
    beta_grid_in, T_grid_in = tlh.geometric_spaced_betas(
        n_chain_in, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final_in, sort_mode=0
    )

    T_grid_in = tlh.betas_to_Ts(beta_grid_in)

    vars_in = np.full(n_chain_in, 1.0) / beta_grid_in**2

    _beta_grid_got, T_grid_got = tlh.entropy_spaced_betas(
        n_chain_need,
        n_cold,
        T_grid_in,
        vars_in,
        n_inf_final=n_inf_final_out,
        T_cold=T_cold,
        correct_last=False,
        sort_mode=0,
    )

    T_prediction_sensibility_check(T_grid_got, 0.0, T_cold, n_cold)
    assert np.allclose(T_grid_in, T_grid_got, atol=1.0e-10, rtol=1.0e-10)


def test_ignore_invalid1() -> None:
    """Test a grid where we know the correct result analytically ignores invalid zero temperature chains"""
    n_chain_in = 100
    n_chain_need = 100
    n_cold = 1
    T_cold = 1.0
    T_min = 1.0
    T_max = 1000.0
    n_inf_final_in = False
    n_inf_final_out = 0
    beta_grid_in, T_grid_in = tlh.geometric_spaced_betas(
        n_chain_in, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final_in, sort_mode=0
    )

    T_grid_in = tlh.betas_to_Ts(beta_grid_in)

    vars_in = np.full(n_chain_in, 1.0) / beta_grid_in**2

    T_grid_in = np.hstack([T_grid_in, np.full(50, 0.0)])
    vars_in = np.hstack([vars_in, np.full(50, 1.0)])

    _beta_grid_got, T_grid_got = tlh.entropy_spaced_betas(
        n_chain_need,
        n_cold,
        T_grid_in,
        vars_in,
        n_inf_final=n_inf_final_out,
        T_cold=T_cold,
        correct_last=False,
        sort_mode=0,
    )

    T_prediction_sensibility_check(T_grid_got, 0.0, T_cold, n_cold)
    assert np.allclose(T_grid_in[:n_chain_in], T_grid_got, atol=1.0e-10, rtol=1.0e-10)


def test_ignore_invalid2() -> None:
    """Test a grid where we know the correct result analytically ignores invalid infinite variance chains"""
    n_chain_in = 100
    n_chain_need = 100
    n_cold = 1
    T_cold = 1.0
    T_min = 1.0
    T_max = 1000.0
    n_inf_final_in = 0
    n_inf_final_out = 0
    beta_grid_in, T_grid_in = tlh.geometric_spaced_betas(
        n_chain_in, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final_in, sort_mode=0
    )

    T_grid_in = tlh.betas_to_Ts(beta_grid_in)

    vars_in = np.full(n_chain_in, 1.0) / beta_grid_in**2

    T_grid_in = np.hstack([T_grid_in, np.full(50, 0.1)])
    vars_in = np.hstack([vars_in, np.full(50, np.inf)])

    with pytest.warns(UserWarning):
        _beta_grid_got, T_grid_got = tlh.entropy_spaced_betas(
            n_chain_need,
            n_cold,
            T_grid_in,
            vars_in,
            n_inf_final=n_inf_final_out,
            T_cold=T_cold,
            correct_last=False,
            sort_mode=0,
        )

    T_prediction_sensibility_check(T_grid_got, 0.0, T_cold, n_cold)
    assert np.allclose(T_grid_in[:n_chain_in], T_grid_got, atol=1.0e-10, rtol=1.0e-10)


def test_ignore_invalid3() -> None:
    """Test a grid where we know the correct result analytically ignores invalid zero T and infinite variance chains"""
    n_chain_in = 100
    n_chain_need = 100
    n_cold = 1
    T_cold = 1.0
    T_min = 1.0
    T_max = 1000.0
    n_inf_final_in = 0
    n_inf_final_out = 0
    beta_grid_in, T_grid_in = tlh.geometric_spaced_betas(
        n_chain_in, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final_in, sort_mode=0
    )

    T_grid_in = tlh.betas_to_Ts(beta_grid_in)

    vars_in = np.full(n_chain_in, 1.0) / beta_grid_in**2

    T_grid_in = np.hstack([T_grid_in, np.full(50, 0.0), np.full(50, 0.1)])
    vars_in = np.hstack([vars_in, np.full(50, 1.0), np.full(50, np.inf)])

    with pytest.warns(UserWarning):
        _beta_grid_got, T_grid_got = tlh.entropy_spaced_betas(
            n_chain_need,
            n_cold,
            T_grid_in,
            vars_in,
            n_inf_final=n_inf_final_out,
            T_cold=T_cold,
            correct_last=False,
            sort_mode=0,
        )

    T_prediction_sensibility_check(T_grid_got, 0.0, T_cold, n_cold)
    assert np.allclose(T_grid_in[:n_chain_in], T_grid_got, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.parametrize(
    ('n_chain_need', 'correct_last'),
    [
        (1, False),
        (2, False),
        (3, False),
        (4, False),
        (8, False),
        (15, False),
        (16, False),
        (32, False),
        (64, False),
        (128, False),
        (2048, False),
        (10000, False),
        (1, True),
        (2, True),
        (3, True),
        (4, True),
        (8, True),
        (15, True),
        (16, True),
        (32, True),
        (64, True),
        (128, True),
        (2048, True),
        (10000, True),
    ],
)
def test_interpolation_case(n_chain_need, correct_last) -> None:
    """This is a test case encountered from a real run that broke the use of cubic spline interpolation"""
    vars_break = np.array(
        [
            5.06922253,
            5.05661274,
            5.0814881,
            5.1003227,
            5.10862914,
            5.1422349,
            5.15319122,
            5.14869325,
            5.13853285,
            5.12498107,
            70.56735103,
            129.31212024,
            138.67556162,
            138.03120093,
            139.28396213,
        ]
    )

    betas_break = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.17782794, 0.03162278, 0.00562341, 0.001, 0.0]
    )

    assert vars_break.size == betas_break.size

    T_grid_got = tlh.entropy_spacing(n_chain_need, betas_break, vars_break, correct_last=correct_last)

    T_prediction_sensibility_check(T_grid_got, 1.0, 1.0, 1)


@pytest.mark.parametrize(
    ('n_chain_in', 'n_chain_need'),
    [
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 100),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 100),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
        (4, 100),
        (100, 100),
    ],
)
def test_zero_T_handling(n_chain_in, n_chain_need) -> None:
    """Test handling if there is a zero temperature chain included"""
    n_cold = 1
    T_cold = 1.0
    T_min = 1.0e-5
    T_max = 10.0
    n_inf_final_in = 1
    n_inf_final_out = 0

    if n_chain_in - 1 == n_cold + n_inf_final_in and n_inf_final_in > 0:
        with pytest.warns(UserWarning):
            beta_grid_in, T_grid_in = tlh.geometric_spaced_betas(
                n_chain_in - 1, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final_in, sort_mode=0
            )
    elif n_chain_in - 1 < n_cold + n_inf_final_in and n_inf_final_in > 0:
        with pytest.warns(UserWarning):
            beta_grid_in, T_grid_in = tlh.geometric_spaced_betas(
                n_chain_in - 1, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final_in, sort_mode=0
            )
    else:
        beta_grid_in, T_grid_in = tlh.geometric_spaced_betas(
            n_chain_in - 1, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final_in, sort_mode=0
        )

    beta_grid_in = np.hstack([beta_grid_in[:n_cold], np.inf, beta_grid_in[n_cold:]])
    T_grid_in = tlh.betas_to_Ts(beta_grid_in)

    vars_in = np.random.normal(0.0, 1.0, n_chain_in) ** 2
    vars_in[np.isfinite(T_grid_in)] *= T_grid_in[np.isfinite(T_grid_in)] ** 2

    if (n_chain_need == n_cold and n_inf_final_out > 0) or n_chain_in == 2:
        with pytest.warns(UserWarning):
            _beta_grid_got, T_grid_got = tlh.entropy_spaced_betas(
                n_chain_need,
                n_cold,
                T_grid_in,
                vars_in,
                n_inf_final=n_inf_final_out,
                T_cold=T_cold,
                correct_last=False,
                sort_mode=0,
            )
    else:
        _beta_grid_got, T_grid_got = tlh.entropy_spaced_betas(
            n_chain_need,
            n_cold,
            T_grid_in,
            vars_in,
            n_inf_final=n_inf_final_out,
            T_cold=T_cold,
            correct_last=False,
            sort_mode=0,
        )

    T_prediction_sensibility_check(T_grid_got, 0.0, T_cold, n_cold)


def test_zero_raises1() -> None:
    """Test that if no chains are valid appropriate errors are raised"""
    n_chain_need = 10
    n_cold = 1
    n_inf_final_out = 1
    T_cold = 1.0

    T_grid_in = np.zeros(10)
    vars_in = np.zeros(10)
    with pytest.raises(ValueError, match='No valid points available to construct ladder'):
        _beta_grid_got, _T_grid_got = tlh.entropy_spaced_betas(
            n_chain_need,
            n_cold,
            T_grid_in,
            vars_in,
            n_inf_final=n_inf_final_out,
            T_cold=T_cold,
            correct_last=False,
            sort_mode=0,
        )


def test_zero_raises2() -> None:
    """Test that if no chains are valid appropriate errors are raised"""
    n_chain_need = 10
    n_cold = 1
    n_inf_final_out = 1
    T_cold = 1.0

    T_grid_in = np.full(10, T_cold)
    vars_in = np.full(10, np.inf)
    with pytest.raises(ValueError, match='No valid points available to construct ladder'), pytest.warns(UserWarning):
        _beta_grid_got, _T_grid_got = tlh.entropy_spaced_betas(
            n_chain_need,
            n_cold,
            T_grid_in,
            vars_in,
            n_inf_final=n_inf_final_out,
            T_cold=T_cold,
            correct_last=False,
            sort_mode=0,
        )


def test_zero_raises3() -> None:
    """Test that if no chains are valid appropriate errors are raised"""
    n_chain_need = 10
    n_cold = 1
    n_inf_final_out = 1
    T_cold = 1.0

    T_grid_in = np.full(10, T_cold)
    vars_in = np.full(10, np.inf)
    T_grid_in[5:] = 0.0
    vars_in[5:] = 0.0
    with pytest.raises(ValueError, match='No valid points available to construct ladder'), pytest.warns(UserWarning):
        _beta_grid_got, _T_grid_got = tlh.entropy_spaced_betas(
            n_chain_need,
            n_cold,
            T_grid_in,
            vars_in,
            n_inf_final=n_inf_final_out,
            T_cold=T_cold,
            correct_last=False,
            sort_mode=0,
        )


def gen_combos_entropy1():
    """Generate a bunch of combinations of variables to test the counting logic for errors"""
    combos_loc = []
    for n_chain_in in [1, 2, 3, 4, 5, 8]:
        for n_chain_need in [1, 2, 3, 4, 5, 8]:
            for n_cold in [0, 1, 2, 3]:
                if n_cold <= n_chain_in:
                    for T_min in [1.0]:
                        for T_cold in [1.0]:
                            for T_max in [10.0]:
                                if T_max > T_min:
                                    for power_law_exp in [0.0]:
                                        for n_inf_final_in in [1, 0]:
                                            for correct_last in [True, False]:
                                                for n_inf_final_out in [1, 0]:
                                                    combos_loc.append(
                                                        (
                                                            n_cold,
                                                            n_chain_in,
                                                            n_chain_need,
                                                            T_cold,
                                                            T_min,
                                                            T_max,
                                                            power_law_exp,
                                                            n_inf_final_in,
                                                            correct_last,
                                                            n_inf_final_out,
                                                        )
                                                    )
    return combos_loc


def gen_combos_entropy2():
    """Generate a bunch of combinations of variables to test the spacing logic for errors"""
    combos_loc = []
    for n_chain_in in [4, 8, 100]:
        for n_chain_need in [4, 8, 100]:
            for n_cold in [1]:
                if n_cold <= n_chain_in:
                    for T_min in [1.0e-5, 0.9, 1.0, 1.1, 10.0]:
                        for T_cold in [0.9, 1.0, 1.1]:
                            for T_max in [1.0, 1.1, 10.0, 100000.0]:
                                if T_max > T_min:
                                    for power_law_exp in [-4.0, 0.0, 4.0]:
                                        for n_inf_final_in in [1, 0]:
                                            for correct_last in [True, False]:
                                                for n_inf_final_out in [1]:
                                                    combos_loc.append(
                                                        (
                                                            n_cold,
                                                            n_chain_in,
                                                            n_chain_need,
                                                            T_cold,
                                                            T_min,
                                                            T_max,
                                                            power_law_exp,
                                                            n_inf_final_in,
                                                            correct_last,
                                                            n_inf_final_out,
                                                        )
                                                    )
    return combos_loc


def gen_combos_entropy3():
    """Generate a fairly exhaustive bunch of combinations of variables to test with random data"""
    combos_loc = []
    for n_chain_in in [1, 2, 3, 4, 8, 100]:
        for n_chain_need in [1, 2, 3, 4, 8, 100]:
            for n_cold in [1, 2, 3, 8]:
                if n_cold <= n_chain_in:
                    for T_min in [1.0e-5, 0.9, 1.0, 1.1]:
                        for T_cold in [0.9, 1.0, 1.1]:
                            for T_max in [1.0, 1.1, 10.0, 10000]:
                                if T_max > T_min:
                                    for power_law_exp in [-4.0, 0.0, 4.0]:
                                        for n_inf_final_in in [1, 0]:
                                            for correct_last in [True, False]:
                                                for n_inf_final_out in [1, 0]:
                                                    combos_loc.append(
                                                        (
                                                            n_cold,
                                                            n_chain_in,
                                                            n_chain_need,
                                                            T_cold,
                                                            T_min,
                                                            T_max,
                                                            power_law_exp,
                                                            n_inf_final_in,
                                                            correct_last,
                                                            n_inf_final_out,
                                                        )
                                                    )
    return combos_loc


def gen_combos_geo():
    """Generate a bunch of combinations of variables to test with random data"""
    combos_loc = []
    for n_chain_in in [1, 2, 3, 4, 8]:
        for n_cold in [1, 2, 3, 8]:
            for T_min in [0.9, 1.0, 1.1]:
                for T_cold in [0.9, 1.0, 1.1]:
                    for T_max in [0.9, 1.0, 1.1]:
                        if T_max > T_min:
                            for n_inf_final_in in [True, False]:
                                combos_loc.append((n_cold, n_chain_in, T_cold, T_min, T_max, n_inf_final_in))
    return combos_loc


combos_entropy = gen_combos_entropy1()
combos_entropy.extend(gen_combos_entropy2())
# combos_entropy = []

combos_geo = gen_combos_geo()


@pytest.mark.parametrize(('n_cold', 'n_chain_in', 'T_cold', 'T_min', 'T_max', 'n_inf_final_in'), combos_geo)
def test_random_data_geo(n_cold, n_chain_in, T_cold, T_min, T_max, n_inf_final_in) -> None:
    """Test some random grids with power law variances"""
    if n_cold > n_chain_in:
        with pytest.raises(ValueError, match='n cold cannot be more than total number of chains'):
            tlh.geometric_spaced_betas(
                n_chain_in, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final_in, sort_mode=0
            )
        return
    if n_cold + n_inf_final_in >= n_chain_in and n_inf_final_in > 0:
        with pytest.warns(UserWarning):
            beta_grid_in, T_grid_in = tlh.geometric_spaced_betas(
                n_chain_in, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final_in, sort_mode=0
            )
    else:
        beta_grid_in, T_grid_in = tlh.geometric_spaced_betas(
            n_chain_in, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final_in, sort_mode=0
        )
        if n_inf_final_in:
            assert beta_grid_in[-1] == 0.0

    assert np.allclose(beta_grid_in, tlh.Ts_to_betas(T_grid_in), atol=1.0e-10, rtol=1.0e-10)

    assert not np.all(beta_grid_in == 0.0)

    assert beta_grid_in.size == T_grid_in.size
    assert beta_grid_in.size == n_chain_in

    assert np.all(T_grid_in[:n_cold] == T_cold)

    T_prediction_sensibility_check(T_grid_in, T_min, T_cold, n_cold)


@pytest.mark.parametrize(
    (
        'n_cold',
        'n_chain_in',
        'n_chain_need',
        'T_cold',
        'T_min',
        'T_max',
        'power_law_exp',
        'n_inf_final_in',
        'correct_last',
        'n_inf_final_out',
    ),
    combos_entropy,
)
def test_random_data_entropy(
    n_cold, n_chain_in, n_chain_need, T_cold, T_min, T_max, power_law_exp, n_inf_final_in, correct_last, n_inf_final_out
) -> None:
    """Test some random grids with power law variances"""
    if n_cold + n_inf_final_in >= n_chain_in and n_inf_final_in > 0:
        with pytest.warns(UserWarning):
            beta_grid_in, T_grid_in = tlh.geometric_spaced_betas(
                n_chain_in, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final_in, sort_mode=0
            )
    else:
        beta_grid_in, T_grid_in = tlh.geometric_spaced_betas(
            n_chain_in, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final_in, sort_mode=0
        )
        if n_inf_final_in:
            assert beta_grid_in[-1] == 0.0

    assert beta_grid_in.size == T_grid_in.size
    assert beta_grid_in.size == n_chain_in

    assert_array_equal(T_grid_in[:n_cold], T_cold)

    T_prediction_sensibility_check(T_grid_in, T_min, T_cold, n_cold)

    # generate variances according to a power law
    T_grid_temp = T_grid_in.copy()
    T_grid_temp[~np.isfinite(T_grid_temp)] = T_max

    vars_in = np.random.normal(0.0, 1.0, n_chain_in) ** 2 * T_grid_temp ** (power_law_exp / 2.0)

    print('Inputs: ', n_chain_in, n_chain_need, n_cold, T_cold, T_min, T_max, power_law_exp, n_inf_final_in)
    print('Betas: ', beta_grid_in)
    print('Variances: ', vars_in)

    if n_cold > n_chain_need:
        with pytest.raises(ValueError, match='n cold cannot be more than total number of chains'):
            beta_grid_got, T_grid_got = tlh.entropy_spaced_betas(
                n_chain_need,
                n_cold,
                T_grid_in,
                vars_in,
                n_inf_final=n_inf_final_out,
                T_cold=T_cold,
                correct_last=correct_last,
                sort_mode=0,
            )
        return
    if n_cold == n_chain_in:
        with pytest.warns(UserWarning):
            beta_grid_got, T_grid_got = tlh.entropy_spaced_betas(
                n_chain_need,
                n_cold,
                T_grid_in,
                vars_in,
                n_inf_final=n_inf_final_out,
                T_cold=T_cold,
                correct_last=correct_last,
                sort_mode=0,
            )
        if n_inf_final_out:
            if n_chain_need > n_cold:
                assert np.all(
                    T_grid_got[n_cold : n_chain_need - 1] == T_grid_got[n_cold]
                )  # if we only have one value, then the results all must be the same
                assert ~np.isfinite(T_grid_got[-1])

    else:
        if (n_cold + n_inf_final_out >= n_chain_need and n_inf_final_out > 0) or (n_cold == 0 and n_chain_in == 1):
            with pytest.warns(UserWarning):
                beta_grid_got, T_grid_got = tlh.entropy_spaced_betas(
                    n_chain_need,
                    n_cold,
                    T_grid_in,
                    vars_in,
                    n_inf_final=n_inf_final_out,
                    T_cold=T_cold,
                    correct_last=correct_last,
                    sort_mode=0,
                )
        else:
            beta_grid_got, T_grid_got = tlh.entropy_spaced_betas(
                n_chain_need,
                n_cold,
                T_grid_in,
                vars_in,
                n_inf_final=n_inf_final_out,
                T_cold=T_cold,
                correct_last=correct_last,
                sort_mode=0,
            )
        T_prediction_sensibility_check(T_grid_got, min(T_min, T_cold), T_cold, n_cold)

    assert np.allclose(beta_grid_got, tlh.Ts_to_betas(T_grid_got), atol=1.0e-10, rtol=1.0e-10)

    assert T_grid_got.size == n_chain_need


def test_prominences() -> None:
    """Test that prominence finding has at least basic functionality"""
    Ts_dummy = np.linspace(1.0, 100, 1000)
    betas_dummy = tlh.Ts_to_betas(Ts_dummy)
    heat_capacity_dummy = np.zeros(Ts_dummy.size)

    maxima_truths = np.array([10.0, 30.0, 50.0])
    ampl_truths = np.array([0.5, 30.0, 0.25])
    sigma_truths = np.array([1.0, 1.0, 1.0])
    for itrp in range(maxima_truths.size):
        heat_capacity_dummy += ampl_truths[itrp] * np.exp(
            -((maxima_truths[itrp] - Ts_dummy) ** 2) / (2 * sigma_truths[itrp] ** 2)
        )

    vars_dummy = heat_capacity_dummy / betas_dummy**2
    maxima_Ts, maxima_vals, prominences = tlh.find_potential_phase_transitions(betas_dummy, vars_dummy)

    print(maxima_vals)
    print(prominences)
    assert np.allclose(maxima_vals, prominences, atol=1.0e-14, rtol=1.0e-2)

    # insist that the proposed maxima is the closest bin possible to the truth
    for itrp in range(maxima_truths.size):
        arg_closest = np.argmin(np.abs(maxima_Ts[itrp] - Ts_dummy))
        arg_best = np.argmin(np.abs(maxima_truths[itrp] - Ts_dummy))
        assert arg_closest == arg_best

        if 1 < arg_best < Ts_dummy.size - 1:
            # insist that the predicted heat capacity is closer to the truth than to nearby values
            if np.abs(maxima_vals[itrp] - heat_capacity_dummy[arg_best]) > 0.01:
                assert np.abs(maxima_vals[itrp] - heat_capacity_dummy[arg_best]) < np.abs(
                    maxima_vals[itrp] - heat_capacity_dummy[arg_best + 1]
                )
                assert np.abs(maxima_vals[itrp] - heat_capacity_dummy[arg_best]) < np.abs(
                    maxima_vals[itrp] - heat_capacity_dummy[arg_best - 1]
                )


if __name__ == '__main__':
    pytest.cmdline.main(['tests/test_temperature_construction.py'])
