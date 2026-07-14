"""Regression guards for the HarnessSampler extension-API wiring (PR #20).

The end-to-end harness tests and golden digests only cover the resulting
run outputs, so they would keep passing if the harness quietly reverted
to an external driver loop that reproduced the same digests. These tests
lock the intended contract directly (PR #20 review F002): the subclass
builds its proposal manager around the base-class-drawn starting samples
in initialize_jumps, drives the adaptive controller from
postblock_operations, flushes checkpoints from post_Nblock_teardown, and
gates the base tracker summary through sampler_verbosity; the run CLI
threads its --sampler-verbosity flag into run_from_spec.
"""

from typing import Any

import numpy as np
import pytest

from DTMCMC.rng_helpers import reset_seed_guard_for_tests, seed_run
from experiments.harness import run as run_mod
from experiments.harness import runner as runner_mod
from experiments.harness.runner import build_sampler, run_from_spec
from experiments.harness.spec import RunSpec, dumps_toml

TINY_SPEC: dict[str, Any] = {
    'name': 'ext_api_test',
    'seed': 42,
    'likelihood': {'name': 'gaussian', 'n_par': 3, 'cutoff': 5},
    'ladder': {
        'kind': 'geometric',
        'n_chain': 6,
        'n_cold': 1,
        'T_cold': 1.0,
        'T_min': 1.0,
        'T_max': 100.0,
        'n_inf_final': 1,
    },
    'run': {'n_steps': 256, 'block_size': 64, 'store_thin': 1, 'checkpoint_every_blocks': 2},
    'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
    'proposals': {
        'FisherJumpManager': {'verbose_fisher': False},
        'DEJumpManager': {'de_size': 256},
    },
}


def make_spec(**run_overrides: object) -> RunSpec:
    """Build the tiny extension-API test spec, overriding [run] entries."""
    data = dict(TINY_SPEC)
    run_table = dict(TINY_SPEC['run'])
    run_table.update(run_overrides)
    data['run'] = run_table
    return RunSpec.from_dict(data)


@pytest.fixture
def fresh_seed_guard():
    """Allow reseeding within a test that drives runs directly."""
    reset_seed_guard_for_tests()
    yield
    reset_seed_guard_for_tests()


@pytest.mark.usefixtures('fresh_seed_guard')
def test_initialize_jumps_builds_manager_from_base_drawn_samples(monkeypatch) -> None:
    """initialize_jumps builds the manager around the base-class-drawn samples[0].

    Locks the de-dup contract: the subclass does not re-draw starting
    samples, it constructs the spec-configured manager around the array
    the base initialize_state filled (so the per-stream RNG order and the
    golden digest are preserved).
    """
    captured: dict[str, np.ndarray] = {}
    real = runner_mod.get_default_proposal_manager  # type: ignore[attr-defined]

    def spy(T_ladder, like_obj, *, starting_samples, **kwargs):
        captured['starting_samples'] = starting_samples
        return real(T_ladder, like_obj, starting_samples=starting_samples, **kwargs)

    monkeypatch.setattr(runner_mod, 'get_default_proposal_manager', spy)

    seed_run(TINY_SPEC['seed'])
    sampler, _like_obj = build_sampler(make_spec())

    assert 'starting_samples' in captured, 'initialize_jumps did not build the default manager'
    # the manager was built around the sampler's own state array (a view
    # into self.samples), not a fresh copy or a re-drawn array
    assert np.shares_memory(captured['starting_samples'], sampler.samples)
    assert np.array_equal(captured['starting_samples'], sampler.samples[0])
    # the base class had already drawn the starting samples before the
    # manager was constructed, so they are not the zero-initialized state
    assert np.any(captured['starting_samples'] != 0.0)


@pytest.mark.usefixtures('fresh_seed_guard')
def test_initialize_jumps_delegates_explicit_manager() -> None:
    """A supplied manager is adopted as-is (delegates to the base hook)."""
    seed_run(TINY_SPEC['seed'])
    sampler, _like_obj = build_sampler(make_spec())

    sentinel = object()
    sampler.initialize_jumps(sentinel)  # type: ignore[arg-type]
    assert sampler.proposal_manager is sentinel


class _RecordingController:
    """Adaptive-controller stub that records post_block invocations."""

    def __init__(self) -> None:
        self.post_block_samplers: list[object] = []

    def post_block(self, sampler: object) -> None:
        """Record the sampler passed at each block boundary."""
        self.post_block_samplers.append(sampler)


@pytest.mark.usefixtures('fresh_seed_guard')
def test_postblock_operations_invokes_controller() -> None:
    """postblock_operations delegates the block boundary to the controller."""
    controller = _RecordingController()
    seed_run(TINY_SPEC['seed'])
    sampler, _like_obj = build_sampler(make_spec(), controller=controller)  # type: ignore[arg-type]

    assert controller.post_block_samplers == []
    sampler.postblock_operations()
    assert controller.post_block_samplers == [sampler]


@pytest.mark.usefixtures('fresh_seed_guard')
def test_postblock_operations_without_controller_is_noop() -> None:
    """Without a controller the block-boundary hook is a safe no-op."""
    seed_run(TINY_SPEC['seed'])
    sampler, _like_obj = build_sampler(make_spec())
    assert sampler.controller is None
    sampler.postblock_operations()  # must not raise


@pytest.mark.usefixtures('fresh_seed_guard')
def test_teardown_flushes_each_segment_and_marks_major_reports(monkeypatch, tmp_path) -> None:
    """post_Nblock_teardown flushes once per segment; finalized tracks the report interval.

    Also guards the periodic major-report contract (PR #20 inline
    review): with the default interval the run finalizes only at the
    final teardown; with a shorter interval it finalizes at each
    boundary, and the sampler never compares itrn against a run total.
    """
    finalized_flags: list[bool] = []
    real = runner_mod.write_artifact  # type: ignore[attr-defined]

    def spy(*args, **kwargs):
        finalized_flags.append(bool(kwargs['finalized']))
        return real(*args, **kwargs)

    monkeypatch.setattr(runner_mod, 'write_artifact', spy)

    # n_steps=256, block_size=64, checkpoint_every_blocks=2 -> teardowns at
    # itrn 128 and 256; the default report interval is n_steps (=256)
    reset_seed_guard_for_tests()
    run_from_spec(make_spec(), tmp_path / 'default')
    assert finalized_flags == [False, True]

    # a half-length interval finalizes at every teardown boundary
    finalized_flags.clear()
    reset_seed_guard_for_tests()
    run_from_spec(make_spec(n_steps_per_major_report=128), tmp_path / 'periodic')
    assert finalized_flags == [True, True]


@pytest.mark.usefixtures('fresh_seed_guard')
@pytest.mark.parametrize(('verbosity', 'expected_summaries'), [(0, 0), (1, 1), (2, 2)])
def test_sampler_verbosity_gates_tracker_summary(monkeypatch, tmp_path, verbosity, expected_summaries) -> None:
    """sampler_verbosity gates the base tracker-summary print.

    0 stays silent, 1 prints once at the single major report, 2 prints at
    every checkpoint teardown (two for the tiny spec).
    """
    summary_calls: list[int] = []
    monkeypatch.setattr(
        'DTMCMC.tracker_manager.TrackerManager.print_tracker_summary',
        lambda _self, *_args, **_kwargs: summary_calls.append(1),
    )

    reset_seed_guard_for_tests()
    run_from_spec(make_spec(), tmp_path / f'v{verbosity}', sampler_verbosity=verbosity)
    assert len(summary_calls) == expected_summaries


class _StubCorrelationSummary:
    """Stand-in for CorrelationSummary that records n_burnin without computing."""

    instances: list[_StubCorrelationSummary] = []  # noqa: RUF012

    def __init__(self) -> None:
        _StubCorrelationSummary.instances.append(self)
        self.n_burnins: list[int] = []

    def summarize_blocks(self, _sampler, _tracker_manager, n_burnin: int) -> None:
        """Record the burn-in passed to the block summary."""
        self.n_burnins.append(n_burnin)

    def final_prints(self, _sampler, n_burnin: int) -> None:
        """Record the burn-in passed to the final prints."""
        self.n_burnins.append(n_burnin)


@pytest.mark.usefixtures('fresh_seed_guard')
@pytest.mark.parametrize(('verbosity', 'exp_commentary', 'exp_corr'), [(0, 0, 0), (1, 1, 0), (2, 1, 1)])
def test_verbosity_gates_major_report_diagnostics(monkeypatch, tmp_path, verbosity, exp_commentary, exp_corr) -> None:
    """Major-report diagnostics fire per verbosity: commentary at >=1, correlation summary at 2.

    Both are gated on the major-report boundary, so for the tiny spec's
    default interval they fire at most once (the final teardown), unlike
    the every-checkpoint tracker summary.
    """
    commentary_calls: list[int] = []
    _StubCorrelationSummary.instances = []
    monkeypatch.setattr(runner_mod, 'print_diagnostic_commentary', lambda _sampler: commentary_calls.append(1))
    monkeypatch.setattr(runner_mod, 'CorrelationSummary', _StubCorrelationSummary)
    # silence the base tracker summary so the test only measures the new paths
    monkeypatch.setattr('DTMCMC.tracker_manager.TrackerManager.print_tracker_summary', lambda _self, *_a, **_k: None)

    reset_seed_guard_for_tests()
    run_from_spec(make_spec(), tmp_path / f'v{verbosity}', sampler_verbosity=verbosity)

    assert len(commentary_calls) == exp_commentary
    assert len(_StubCorrelationSummary.instances) == exp_corr


class _FrozenController:
    """Minimal controller exposing a fixed freeze block for the burn-in helper."""

    def __init__(self, frozen_block_index: int | None) -> None:
        self.frozen_block_index = frozen_block_index


@pytest.mark.usefixtures('fresh_seed_guard')
@pytest.mark.parametrize(('frozen_block', 'expected_factor'), [(None, 0), (0, 0), (5, 5)])
def test_adaptive_burnin_iterations_from_freeze(frozen_block, expected_factor) -> None:
    """adaptive_burnin_iterations is the freeze block times the block size (0 while adapting)."""
    seed_run(TINY_SPEC['seed'])
    controller = _FrozenController(frozen_block)
    sampler, _like_obj = build_sampler(make_spec(), controller=controller)  # type: ignore[arg-type]
    assert sampler.adaptive_burnin_iterations() == expected_factor * sampler.block_size


@pytest.mark.usefixtures('fresh_seed_guard')
def test_adaptive_burnin_iterations_no_controller() -> None:
    """A fixed-ladder run (no controller) reports zero adaptive burn-in."""
    seed_run(TINY_SPEC['seed'])
    sampler, _like_obj = build_sampler(make_spec())
    assert sampler.controller is None
    assert sampler.adaptive_burnin_iterations() == 0


def test_run_cli_forwards_sampler_verbosity(monkeypatch, tmp_path) -> None:
    """The run CLI threads --sampler-verbosity into run_from_spec."""
    spec_path = tmp_path / 'spec.toml'
    spec_path.write_text(dumps_toml(dict(TINY_SPEC)))

    captured: dict[str, object] = {}

    def fake_run(_spec, _out, _artifact_name=None, sampler_verbosity=0):
        captured['sampler_verbosity'] = sampler_verbosity
        return tmp_path / 'artifact.h5'

    monkeypatch.setattr(run_mod, 'run_from_spec', fake_run)
    monkeypatch.setattr(run_mod, 'validate', lambda *_args, **_kwargs: [])

    assert run_mod.main([str(spec_path), '--out', str(tmp_path), '--sampler-verbosity', '2']) == 0
    assert captured['sampler_verbosity'] == 2
