"""Regression guards for the rejection-sampling retry cap in the likelihood gen_draws.

``normal_nd.gen_draws`` initialised its attempt counter and compared it against
``attempt_lim`` but never incremented it, so impossible bounds spun forever
inside an njit loop — uninterruptible — instead of raising. The cap is also
compared with ``>=`` rather than ``==`` so a non-positive ``attempt_lim`` cannot
walk past it and reintroduce the same hang.

Only ``normal_nd.gen_draws`` takes its bounds as arguments, so it is the only
one whose exhaustion path is reachable through the public signature; the
siblings bake their own ``low_lim``/``high_lim`` module constants. They share
the idiom, so they are covered here by a smoke test that the cap does not fire
spuriously on draws that should succeed.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from DTMCMC.likelihoods import ar1, gaussian_mixture, gaussian_shell, normal_nd, random_wheel, rosenbrock, spoke_wheel

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

_N_PAR = 2
# a box the unit normal can never reach, so every draw is rejected
_UNREACHABLE_LOW = np.full(_N_PAR, 10.0)
_UNREACHABLE_HIGH = np.full(_N_PAR, 11.0)

_SIBLING_GEN_DRAWS: list[Callable[[int, int], NDArray[np.floating]]] = [
    ar1.gen_draws,
    gaussian_mixture.gen_draws,
    gaussian_shell.gen_draws,
    random_wheel.gen_draws,
    rosenbrock.gen_draws,
    spoke_wheel.gen_draws,
]


@pytest.mark.parametrize('attempt_lim', [1, 2, 10])
def test_gen_draws_raises_once_attempts_are_exhausted(attempt_lim: int) -> None:
    """Guards the counter increment: without it these cases never terminate.

    A regression here hangs rather than failing, since the loop it guards is an
    uninterruptible njit loop — an unexplained CI timeout in this test means the
    increment went missing again.
    """
    with pytest.raises(RuntimeError, match='failed to find valid posterior point'):
        normal_nd.gen_draws(1, _N_PAR, _UNREACHABLE_LOW, _UNREACHABLE_HIGH, attempt_lim=attempt_lim)


@pytest.mark.parametrize('attempt_lim', [-1, 0])
def test_gen_draws_non_positive_attempt_lim_raises_immediately(attempt_lim: int) -> None:
    """Guards the ``>=`` comparison: the counter starts at 1, so ``== 0`` never matched."""
    with pytest.raises(RuntimeError, match='failed to find valid posterior point'):
        normal_nd.gen_draws(1, _N_PAR, _UNREACHABLE_LOW, _UNREACHABLE_HIGH, attempt_lim=attempt_lim)


def test_gen_draws_returns_in_bounds_draws_when_bounds_are_reachable() -> None:
    """The cap does not fire on draws that should succeed."""
    draws = normal_nd.gen_draws(8, 3, np.full(3, -5.0), np.full(3, 5.0))

    assert draws.shape == (8, 3)
    assert np.all(draws >= -5.0)
    assert np.all(draws <= 5.0)


@pytest.mark.parametrize('gen_draws', _SIBLING_GEN_DRAWS, ids=lambda fn: fn.__module__.rsplit('.', 1)[-1])
def test_sibling_gen_draws_do_not_trip_their_cap(
    gen_draws: Callable[[int, int], NDArray[np.floating]],
) -> None:
    """Every sibling shares the counter idiom; none of them exhausts it in normal use."""
    draws = gen_draws(4, _N_PAR)

    assert draws.shape == (4, _N_PAR)
    assert np.all(np.isfinite(draws))
