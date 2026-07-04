"""Phase 3 acceptance tests: generalized spacing, ladder family, tunable cake.

The regression fixture (entropy_ladder_regression.npz) was generated from
the pre-Phase-3 EntropyTemperatureLadder on data/*_gold inputs, so the
exact-reproduction test is a genuine behavior guard across the
generalized-machinery refactor, not a tautology.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.interpolate import InterpolatedUnivariateSpline

from DTMCMC.likelihoods.cake_likelihood import CakeLikelihood
from DTMCMC.likelihoods.cake_likelihood import get_loglike as cake_default_loglike
from DTMCMC.rng_helpers import get_rng
from DTMCMC.temperature_ladder_helpers import (
    AcceptanceTemperatureLadder,
    EntropyTemperatureLadder,
    GeometricTemperatureLadder,
    LengthTemperatureLadder,
    Ts_to_betas,
    predicted_swap_acceptance,
    standardize_input_stats,
)
from experiments.harness.runner import build_ladder
from experiments.harness.spec import RunSpec
from experiments.reference_samplers import cake_moment_r2, draw_cake
from tests.test_harness import TINY_GAUSSIAN_SPEC

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


def test_gaussian_null_case_ladders_coincide() -> None:
    """Acceptance 2: on analytic constant-C data, entropy == length == geometric.

    For a Gaussian target Var(logL) = n*T^2/2, so both the entropy
    integrand Var*beta = (n/2)/beta and the length integrand
    sqrt(Var) = sqrt(n/2)/beta integrate to log(beta): equal spacing in
    log T, i.e. the geometric ladder, up to trapezoid interpolation error
    on the input grid.
    """
    n_par = 4
    Ts_in = np.geomspace(1., 64., 33)
    vars_in = n_par * Ts_in**2 / 2.
    n_chain = 9

    entropy_ladder = EntropyTemperatureLadder(n_chain, Ts_in, vars_in, n_cold=1, T_cold=1., n_inf_final=0)
    length_ladder = LengthTemperatureLadder(n_chain, Ts_in, vars_in, n_cold=1, T_cold=1., n_inf_final=0)
    geometric_ladder = GeometricTemperatureLadder(n_chain, 1, 1., 1., 64., n_inf_final=0)

    assert_allclose(np.log(entropy_ladder.Ts), np.log(geometric_ladder.Ts), atol=0.02)
    assert_allclose(np.log(length_ladder.Ts), np.log(geometric_ladder.Ts), atol=0.02)


@pytest.mark.parametrize('beta1', [1.0, 0.5, 0.2])
@pytest.mark.parametrize('delta_beta', [0.01, 0.05, 0.1, 0.3])
def test_acceptance_predictor_vs_brute_force(beta1, delta_beta) -> None:
    """Acceptance 3: closed form within 1% absolute of swap Monte Carlo."""
    n_par = 4
    beta2 = beta1 - delta_beta
    if beta2 <= 0.:
        pytest.skip('beta2 not positive for this grid point')

    # Gaussian-target logL statistics at each rung
    mean1, mean2 = -n_par / (2. * beta1), -n_par / (2. * beta2)
    var1, var2 = n_par / (2. * beta1**2), n_par / (2. * beta2**2)

    predicted = predicted_swap_acceptance(beta1, beta2, mean1, mean2, var1, var2)

    rng = get_rng(hash((round(beta1 * 100), round(delta_beta * 100))) % 2**32)
    n_mc = 400000
    logL1 = mean1 + np.sqrt(var1) * rng.standard_normal(n_mc)
    logL2 = mean2 + np.sqrt(var2) * rng.standard_normal(n_mc)
    ratio = (beta1 - beta2) * (logL2 - logL1)
    # min(1, e^r) evaluated stably: exp clamped at r=0 where the min is 1
    brute_force = float(np.where(ratio >= 0., 1., np.exp(np.minimum(ratio, 0.))).mean())

    assert abs(predicted - brute_force) < 0.01


@pytest.fixture(scope='module')
def cake1_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the matched cake1 (Ts, means, vars) reference arrays."""
    Ts_in = np.load(DATA_DIR / 'Ts_cake1.npy')
    means_in = np.load(DATA_DIR / 'means_cake1.npy')
    vars_in = np.load(DATA_DIR / 'vars_cake1.npy')
    keep = Ts_in >= 1.
    return Ts_in[keep], means_in[keep], vars_in[keep]


def test_acceptance_ladder_equal_acceptance(cake1_inputs) -> None:
    """The acceptance ladder's defining property: equal predicted swap acceptance.

    Rebuild the interpolants the ladder used and check every adjacent
    finite pair sits at the achieved target.
    """
    Ts_in, means_in, vars_in = cake1_inputs
    n_chain = 12
    ladder = AcceptanceTemperatureLadder(n_chain, Ts_in, means_in, vars_in, n_cold=1, T_cold=1., n_inf_final=1)

    assert ladder.Ts.size == n_chain
    assert ladder.n_cold == 1
    assert np.sum(~np.isfinite(ladder.Ts)) == 1
    assert 0. < ladder.achieved_acceptance < 1.

    betas_use, means_use, vars_use = standardize_input_stats(Ts_to_betas(Ts_in), means_in, vars_in)
    mean_interp = InterpolatedUnivariateSpline(betas_use[::-1], means_use[::-1], k=1, ext=3)
    var_interp = InterpolatedUnivariateSpline(betas_use[::-1], vars_use[::-1], k=1, ext=3)

    def interp_scalar(spline: InterpolatedUnivariateSpline, beta_loc: float) -> float:
        return float(spline(np.asarray([beta_loc]))[0])

    finite_Ts = ladder.Ts[np.isfinite(ladder.Ts)]
    betas = 1. / finite_Ts
    acceptances = np.array([
        predicted_swap_acceptance(
            betas[itrt], betas[itrt + 1],
            interp_scalar(mean_interp, betas[itrt]), interp_scalar(mean_interp, betas[itrt + 1]),
            interp_scalar(var_interp, betas[itrt]), interp_scalar(var_interp, betas[itrt + 1]),
        )
        for itrt in range(finite_Ts.size - 1)
    ])
    assert_allclose(acceptances, ladder.achieved_acceptance, atol=0.02)


@pytest.mark.parametrize('ladder_table', [
    {'kind': 'geometric', 'n_chain': 8, 'n_cold': 1, 'T_max': 100.0},
    {'kind': 'explicit', 'n_chain': 3, 'n_cold': 1, 'Ts': [1.0, 2.0, 8.0]},
    {'kind': 'entropy_file', 'n_chain': 8, 'n_cold': 1, 'Ts_file': 'data/Ts_cake_gold.npy', 'vars_file': 'data/vars_cake_gold.npy'},
    {'kind': 'length_file', 'n_chain': 8, 'n_cold': 1, 'Ts_file': 'data/Ts_cake_gold.npy', 'vars_file': 'data/vars_cake_gold.npy'},
    {'kind': 'acceptance_file', 'n_chain': 8, 'n_cold': 1, 'Ts_file': 'data/Ts_cake1.npy', 'means_file': 'data/means_cake1.npy', 'vars_file': 'data/vars_cake1.npy'},
])
def test_all_ladder_kinds_constructible_from_spec(ladder_table) -> None:
    """Acceptance 4: every ladder kind builds from a harness spec."""
    data: dict[str, Any] = {key: dict(value) if isinstance(value, dict) else value for key, value in TINY_GAUSSIAN_SPEC.items()}
    data['ladder'] = ladder_table
    spec = RunSpec.from_dict(data)
    ladder = build_ladder(spec)
    assert ladder.n_chain == ladder_table['n_chain']
    assert np.all(ladder.Ts[np.isfinite(ladder.Ts)] > 0.)


def test_tunable_cake_family() -> None:
    """Tunable cake: identical defaults, distinct custom tiers, exact reference."""
    n_par = 5
    rng = get_rng(29)
    points = rng.uniform(-9., 9., size=(32, n_par))

    default_cake = CakeLikelihood(n_par=n_par, cutoff=10)
    for point in points:
        assert default_cake.get_loglike(point) == cake_default_loglike(point)

    amps, widths, exponents = (0.7, 0.3), (2., 0.5), (4., 2.)
    custom_cake = CakeLikelihood(n_par=n_par, cutoff=10, amps=amps, widths=widths, exponents=exponents)
    assert any(custom_cake.get_loglike(point) != default_cake.get_loglike(point) for point in points)

    # the reference sampler generalizes with it: analytic E[r^2] matches
    draws = draw_cake(20000, n_par, get_rng(30), amps=amps, widths=widths, exponents=exponents)
    assert_allclose((draws**2).sum(axis=1).mean(), cake_moment_r2(n_par, amps=amps, widths=widths, exponents=exponents), rtol=0.05)
