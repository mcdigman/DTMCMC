"""Prior-recovery review mode: zero_loglike wrapping and the constant likelihood.

Scientific sampler reviews rerun with the likelihood forced to a constant so
the sampler must reproduce the prior. These tests pin the contract: a run
with any likelihood and ``zero_loglike=True`` is bit-for-bit identical to a
ConstantRectangularLikelihood run over the same bounds and settings, for
every built-in jump type on both kernel backends, and the wrapper replaces
only the log likelihood — priors and bounds handling stay untouched.
"""

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from DTMCMC.de_manager import DEJumpManager
from DTMCMC.likelihood import ZeroedLoglikeLikelihood, zero_loglike_native
from DTMCMC.likelihoods.banana import BananaLikelihood
from DTMCMC.likelihoods.constant_rectangular import ConstantRectangularLikelihood, high_lim, low_lim
from DTMCMC.likelihoods.uniform_gaussian_prior import UniformGaussianPriorLikelihood
from DTMCMC.numba_backend import NativeBackendUnsupportedError
from DTMCMC.rng_helpers import reset_seed_guard_for_tests, seed_run
from experiments.harness.runner import build_sampler
from experiments.harness.spec import RunSpec

N_PAR = 4
SEED = 20260720

# banana's rectangular bounds at N_PAR = 4, reused verbatim for the
# constant-likelihood reference run
_BANANA_BOUNDS = BananaLikelihood(N_PAR)

# every built-in local proposal (Cholesky mode enables Fisher All-D), plus
# None for the default mixed-jump weighting
JUMP_LABELS = [
    None,
    'Fisher All-D',
    'Std All-D',
    'Std Random-D',
    'DE Std All-D',
    'DE Std Random-D',
    'DE Big All-D',
    'DE Big Random-D',
    'Prior All-D',
    'Blank Jump',
]


def _make_spec(likelihood_table: dict[str, Any]) -> RunSpec:
    return RunSpec.from_dict(
        {
            'name': 'zero_loglike_equivalence',
            'seed': SEED,
            'likelihood': likelihood_table,
            'ladder': {
                'kind': 'geometric',
                'n_chain': 6,
                'n_cold': 1,
                'T_cold': 1.0,
                'T_min': 1.0,
                'T_max': 100.0,
                'n_inf_final': 1,
            },
            'run': {'n_steps': 8, 'block_size': 4, 'store_thin': 1, 'checkpoint_every_blocks': 1},
            'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
            'proposals': {
                'FisherJumpManager': {'verbose_fisher': False, 'use_chol_fishers': True},
                'DEJumpManager': {'de_size': 16},
            },
        }
    )


def _snapshot(sampler: Any) -> dict[str, object]:
    de_manager = next(manager for manager in sampler.proposal_manager.managers if isinstance(manager, DEJumpManager))
    tracker = sampler.tracker_manager
    return {
        'samples': sampler.samples.copy(),
        'logLs': sampler.logLs.copy(),
        'chain_track': sampler.chain_track.copy(),
        'samples_store': sampler.samples_store.copy(),
        'logLs_store': sampler.logLs_store.copy(),
        'accept_record': tracker.accept_record.copy(),
        'esd_record': tracker.esd_record.copy(),
        'exchange_tracker': tracker.exchange_tracker.copy(),
        'esd_exchange': tracker.esd_exchange.copy(),
        'de_buffer': de_manager.de_buffer.copy(),
        'de_write': de_manager.itrde_write,
        'de_count': de_manager.itrde_count,
        'eval_accounting': asdict(sampler.eval_accounting),
        'itrn': sampler.itrn,
    }


def _run(
    likelihood_table: dict[str, Any], backend: str, jump_label: str | None, *, zero_loglike: bool
) -> tuple[dict[str, object], Any]:
    reset_seed_guard_for_tests()
    try:
        spec = _make_spec(likelihood_table)
        seed_run(spec.seed)
        sampler, _like_obj = build_sampler(spec, kernel_backend=backend, zero_loglike=zero_loglike)
        if jump_label is not None:
            jump_idx = sampler.proposal_manager.jump_labels_array.index(jump_label)
            sampler.proposal_manager.jump_probs.fill(0.0)
            sampler.proposal_manager.jump_probs[:, jump_idx] = 1.0
        for _ in range(spec.n_blocks):
            sampler.advance_block()
        assert sampler.last_kernel_backend == backend
        return _snapshot(sampler), sampler
    finally:
        reset_seed_guard_for_tests()


@pytest.mark.parametrize('backend', ['python', 'numba'])
@pytest.mark.parametrize('jump_label', JUMP_LABELS)
def test_zeroed_likelihood_matches_constant_likelihood(backend: str, jump_label: str | None) -> None:
    """zero_loglike over any likelihood equals the constant likelihood bit-for-bit.

    Same seed, bounds, ladder, and proposal settings: the banana run with
    its log likelihood zeroed must reproduce the constant-rectangular run
    exactly — every sample, tracker, DE-buffer entry, and evaluation count.
    """
    constant_table = {
        'name': 'constant_rectangular',
        'n_par': N_PAR,
        'low_lims': _BANANA_BOUNDS.low_lims.tolist(),
        'high_lims': _BANANA_BOUNDS.high_lims.tolist(),
    }
    constant_state, constant_sampler = _run(constant_table, backend, jump_label, zero_loglike=False)
    zeroed_state, zeroed_sampler = _run({'name': 'banana', 'n_par': N_PAR}, backend, jump_label, zero_loglike=True)

    assert isinstance(constant_sampler.like_obj, ConstantRectangularLikelihood)
    assert isinstance(zeroed_sampler.like_obj, ZeroedLoglikeLikelihood)
    assert isinstance(zeroed_sampler.like_obj.inner, BananaLikelihood)

    assert constant_state.keys() == zeroed_state.keys()
    for key, constant_value in constant_state.items():
        zeroed_value = zeroed_state[key]
        if isinstance(constant_value, np.ndarray):
            np.testing.assert_array_equal(zeroed_value, constant_value, err_msg=key)
        else:
            assert zeroed_value == constant_value, key


def test_constant_rectangular_defaults() -> None:
    """The constant likelihood defaults to the borrowed banana box and zero loglike."""
    like = ConstantRectangularLikelihood(n_par=3)
    np.testing.assert_array_equal(like.low_lims, np.full(3, low_lim))
    np.testing.assert_array_equal(like.high_lims, np.full(3, high_lim))
    assert like.get_loglike(np.zeros(3)) == 0.0
    assert like.prior_factor(np.zeros(3)) == 0.0
    assert like.bind_native().loglike is zero_loglike_native


def test_zeroed_wrapper_replaces_only_the_loglike() -> None:
    """The wrapper zeroes the log likelihood but delegates priors and bounds."""
    inner = UniformGaussianPriorLikelihood(n_par=3, prior_mean=1.5, prior_std=0.75)
    wrapped = ZeroedLoglikeLikelihood(inner)
    point = np.full(3, 0.25)

    assert wrapped.get_loglike(point) == 0.0
    assert wrapped.prior_factor(point) == inner.prior_factor(point) != 0.0
    corrected, ok = wrapped.validate_bounds(point.copy())
    np.testing.assert_array_equal(corrected, inner.validate_bounds(point.copy())[0])
    assert ok

    inner_natives = inner.bind_native()
    wrapped_natives = wrapped.bind_native()
    assert wrapped_natives.loglike is zero_loglike_native
    assert wrapped_natives.prior_draw is inner_natives.prior_draw
    assert wrapped_natives.prior_factor is inner_natives.prior_factor
    assert wrapped_natives.validate_bounds is inner_natives.validate_bounds
    assert wrapped.native_state() == inner.native_state()


def test_zeroed_wrapper_rejects_unbindable_inner() -> None:
    """Wrapping does not launder a missing or stale native binding."""

    class _HooklessLikelihood:
        n_par = 2

        def get_loglike(self, params_in: np.ndarray) -> float:
            return -float(np.sum(params_in * params_in))

        def prior_draw(self) -> np.ndarray:
            return np.zeros(2)

        def prior_factor(self, params_in: np.ndarray) -> float:
            del params_in
            return 0.0

        def validate_bounds(self, params_in: np.ndarray) -> tuple[np.ndarray, bool]:
            return params_in, True

    wrapped = ZeroedLoglikeLikelihood(_HooklessLikelihood())  # type: ignore[arg-type]
    with pytest.raises(NativeBackendUnsupportedError, match='bind_native'):
        wrapped.bind_native()


def test_zero_loglike_kwarg_wraps_before_graph_assembly() -> None:
    """The kwarg wraps the likelihood before the default managers are built.

    Every sampler-built manager must hold the wrapped object, so Fisher
    refreshes, DE initialization, and prior-jump dispatch all see the
    zeroed log likelihood.
    """
    _state, sampler = _run({'name': 'banana', 'n_par': N_PAR}, 'python', None, zero_loglike=True)
    assert isinstance(sampler.like_obj, ZeroedLoglikeLikelihood)
    for manager in sampler.proposal_manager.managers:
        assert manager.like_obj is sampler.like_obj
    assert np.all(sampler.logLs == 0.0)
