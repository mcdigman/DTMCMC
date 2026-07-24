"""Spy-verified tests for the declared likelihood-evaluation accounting.

Every declared cost is checked against DTMCMC.eval_accounting.LoglikeCallSpy,
which independently counts actual get_loglike calls — the declarations are
never compared against themselves. LoglikeCallSpy is the same conformance
helper third-party managers and jumps should use to verify their own
declarations. The integrated whole-run spy test (initialization + proposal
targets + declared internal costs + scheduled post-block costs equals the
actual call total, artifact included) lives in
tests/test_harness.py::test_counting_matches_independent_spy.
"""

import configparser
from typing import TYPE_CHECKING, Any, NamedTuple, override

import numpy as np
import pytest

from DTMCMC.auxilliary_manager import AuxilliaryJumpManager
from DTMCMC.de_manager import DEJumpManager
from DTMCMC.dtmcmc_sampler import DTMCMCSampler
from DTMCMC.eval_accounting import EvalAccounting, LoglikeCallSpy
from DTMCMC.exchange_manager import NULL_TARGETS, ExchangeManager
from DTMCMC.fisher_manager import FisherJumpManager, declared_fisher_stencil_evals
from DTMCMC.history_jump_manager import LadderHistoryJumpManager
from DTMCMC.jump_manager import AbstractNativeJump, JumpManager
from DTMCMC.likelihood import AbstractLikelihood
from DTMCMC.likelihoods.normal_nd import GaussianLikelihood
from DTMCMC.prior_manager import PriorManager
from DTMCMC.proposal_manager import ProposalManager
from DTMCMC.rng_helpers import reset_seed_guard_for_tests, seed_run
from DTMCMC.temperature_ladder_helpers import GeometricTemperatureLadder, TemperatureLadder

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _config(**fisher_overrides: object) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read('default_config.ini')
    config['FisherJumpManager']['verbose_fisher'] = 'False'
    for key, value in fisher_overrides.items():
        config['FisherJumpManager'][key] = str(value)
    return config


def _ladder(n_chain: int) -> GeometricTemperatureLadder:
    return GeometricTemperatureLadder(n_chain, n_cold=1, T_max=50.0, n_inf_final=1)


@pytest.fixture
def fresh_seed_guard():
    """Allow one seed_run call in a test that legitimately reseeds."""
    reset_seed_guard_for_tests()
    yield
    reset_seed_guard_for_tests()


@pytest.mark.parametrize(('n_chain', 'n_par'), [(3, 2), (4, 3), (6, 5)])
@pytest.mark.parametrize('use_chol_fishers', [False, True])
def test_fisher_declared_costs_match_spy(n_chain: int, n_par: int, use_chol_fishers: bool) -> None:
    """The Fisher stencil declarations equal independently observed call counts.

    Covers construction, a scheduled refresh block, and a non-refresh block,
    across dimensions, chain counts, and diagonal/Cholesky modes.
    """
    like_obj = GaussianLikelihood(n_par=n_par)
    ladder = _ladder(n_chain)
    config = _config(use_chol_fishers=use_chol_fishers, fisher_downsample=2)
    starting = np.zeros((n_chain, n_par))
    expected = declared_fisher_stencil_evals(n_chain, n_par, use_chol_fishers)

    with LoglikeCallSpy(like_obj) as construction_spy:
        manager = FisherJumpManager(ladder, like_obj, starting, config)
    assert manager.declared_construction_evals == expected
    assert construction_spy.n_calls == expected

    block_size = 4
    samples = np.zeros((block_size + 1, n_chain, n_par))
    logLs = np.zeros((block_size + 1, n_chain))

    # itrn // block_size < 4 always refreshes
    with LoglikeCallSpy(like_obj) as refresh_spy:
        declared_refresh = manager.post_block_update(0, block_size, samples, logLs)
    assert declared_refresh == expected
    assert refresh_spy.n_calls == expected

    # past the initial refreshes and off the downsample cadence: no stencil
    with LoglikeCallSpy(like_obj) as idle_spy:
        declared_idle = manager.post_block_update(5 * block_size, block_size, samples, logLs)
    assert declared_idle == 0
    assert idle_spy.n_calls == 0


@pytest.mark.usefixtures('fresh_seed_guard')
def test_builtin_jump_internal_costs_match_spy() -> None:
    """Every built-in jump's declared per-dispatch internal cost is observed.

    All local proposals declare zero internal evaluations; the ladder
    history jump declares exactly one (its current-point evaluation).
    """
    seed_run(20260716)
    n_chain, n_par = 3, 2
    like_obj = GaussianLikelihood(n_par=n_par)
    ladder = _ladder(n_chain)
    # Cholesky mode so the full Fisher jump is dispatchable directly
    config = _config(use_chol_fishers=True)

    starting = np.zeros((n_chain, n_par))
    managers: list[JumpManager[AbstractLikelihood[Any], Any]] = [
        FisherJumpManager(ladder, like_obj, starting, config),
        DEJumpManager(ladder, like_obj, config),
        AuxilliaryJumpManager(ladder, like_obj, config),
        PriorManager(ladder, like_obj, config),
        LadderHistoryJumpManager(
            ladder, like_obj, config, ladder, np.zeros((4, n_chain)), np.zeros((4, n_chain, n_par))
        ),
    ]

    sample_point = np.zeros(n_par)
    checked = 0
    for manager in managers:
        for jump in manager.jumps:
            declared = getattr(jump, 'declared_internal_evals', None)
            assert declared is not None, jump.print_name
            with LoglikeCallSpy(like_obj) as spy:
                jump(sample_point.copy(), 0)
            assert spy.n_calls == declared, jump.print_name
            checked += 1
    assert checked == 10  # 3 fisher + 4 de + 1 blank + 1 prior + 1 history


class _UndeclaredNativeState(NamedTuple): ...


def _dummy_call(sample_point: NDArray[np.floating], _itrt: int, _state: _UndeclaredNativeState):
    return sample_point.copy(), 0.0, True


class _UndeclaredJump[LikelihoodType: AbstractLikelihood[Any]](
    AbstractNativeJump[LikelihoodType, _UndeclaredNativeState]
):
    """Extension jump without a declared_internal_evals attribute."""

    def __init__(self, manager: _ExtensionManager[LikelihoodType], print_name: str = 'Undeclared') -> None:
        super().__init__(_dummy_call, manager, print_name)


class _DeclaredJump[LikelihoodType: AbstractLikelihood[Any]](_UndeclaredJump[LikelihoodType]):
    def __init__(self, manager: _ExtensionManager[LikelihoodType]) -> None:
        super().__init__(manager, 'Declared')

    @property
    @override
    def declared_internal_evals(self) -> int:
        return 0


class _ExtensionManager[LikelihoodType: AbstractLikelihood[Any]](JumpManager[LikelihoodType, _UndeclaredNativeState]):
    def __init__(self, T_ladder: TemperatureLadder, like_obj: LikelihoodType, jump_sel: int = 0) -> None:
        if jump_sel == 0:
            jump: _UndeclaredJump[LikelihoodType] = _DeclaredJump(self)
        else:
            jump = _UndeclaredJump(self)

        super().__init__(T_ladder, like_obj, [jump])

    @override
    def set_jump_weights(self) -> None:
        self.jump_weights = np.ones((self.n_chain, self.n_jump_types))

    @override
    def record_config(self, config_in: configparser.ConfigParser) -> None:
        del config_in

    @property
    @override
    def native_state(self) -> _UndeclaredNativeState:
        return _UndeclaredNativeState()


class _UndeclaredPostBlockManager[LikelihoodType: AbstractLikelihood[Any]](_ExtensionManager[LikelihoodType]):
    """Extension manager whose post-block cost cannot be declared."""

    @override
    def post_block_update(self, itrn: int, block_size: int, samples: NDArray[np.floating], logLs: NDArray[np.floating]):
        del itrn, block_size, samples, logLs
        return


def _run_tiny_sampler(
    manager: JumpManager[GaussianLikelihood, Any], like_obj: GaussianLikelihood, ladder: TemperatureLadder
) -> EvalAccounting[GaussianLikelihood]:
    config = _config()
    config['ProposalManager']['only_prior_hot'] = 'False'
    proposal = ProposalManager(ladder, like_obj, (manager,), ExchangeManager(NULL_TARGETS, False), config)
    sampler = DTMCMCSampler(
        ladder,
        like_obj,
        block_size=4,
        store_size=4,
        proposal_manager=proposal,
        starting_samples=np.zeros((ladder.n_chain, like_obj.n_par)),
        kernel_backend='python',
    )
    sampler.advance_block()
    return sampler.eval_accounting


@pytest.mark.usefixtures('fresh_seed_guard')
def test_unknown_internal_cost_marks_accounting_incomplete() -> None:
    """A jump without a declaration must not be silently counted as zero."""
    seed_run(20260717)
    like_obj = GaussianLikelihood(n_par=2)
    ladder = _ladder(3)
    accounting = _run_tiny_sampler(_ExtensionManager(ladder, like_obj, 1), like_obj, ladder)
    assert not accounting.complete
    assert accounting.proposal_targets > 0


@pytest.mark.usefixtures('fresh_seed_guard')
def test_unknown_post_block_cost_marks_accounting_incomplete() -> None:
    """A post-block update returning None must not be silently counted as zero."""
    seed_run(20260718)
    like_obj = GaussianLikelihood(n_par=2)
    ladder = _ladder(3)
    accounting = _run_tiny_sampler(_UndeclaredPostBlockManager(ladder, like_obj, 0), like_obj, ladder)
    assert not accounting.complete
    assert accounting.post_block == 0


@pytest.mark.usefixtures('fresh_seed_guard')
def test_declared_extension_graph_stays_complete() -> None:
    """A fully declared extension graph keeps the accounting exact."""
    seed_run(20260719)
    like_obj = GaussianLikelihood(n_par=2)
    ladder = _ladder(3)
    with LoglikeCallSpy(like_obj) as spy:
        accounting = _run_tiny_sampler(_ExtensionManager(ladder, like_obj, 0), like_obj, ladder)
    assert accounting.complete
    assert accounting.total == spy.n_calls
