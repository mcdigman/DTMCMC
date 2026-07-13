"""Dashboard acceptance tests: artifact reader, diagnostics, figures, app.

Two tiny Gaussian runs (one fixed-ladder, one adaptive) are generated once
per module through the real harness, so the reader is exercised against
artifacts the writer actually produces, including the dashboard-support
additions (trackers jump_labels attr, ladder/initial_Ts dataset).
"""

import dataclasses
import itertools
import os
import tomllib
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import plotly.graph_objects as go
import pytest
from setuptools import find_packages

if TYPE_CHECKING:
    from pathlib import Path

from dashboard.core import checks
from dashboard.core import diagnostics as diag
from dashboard.core.reader import ArtifactWatcher, list_artifacts, load_snapshot
from dashboard.figures.options import ViewOptions
from dashboard.figures.registry import LAYOUTS, PLOTS, build_figure
from dashboard.themes import THEMES, get_theme
from DTMCMC.rng_helpers import get_rng, reset_seed_guard_for_tests, seed_run
from experiments.harness.artifact import collect_provenance
from experiments.harness.paths import repo_root
from experiments.harness.runner import build_sampler, run_from_spec
from experiments.harness.spec import RunSpec, config_to_text

TINY_FIXED_SPEC: dict[str, Any] = {
    'name': 'dash_tiny_fixed',
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
    'run': {'n_steps': 256, 'block_size': 64, 'store_thin': 1, 'arg_record': [3], 'checkpoint_every_blocks': 2},
    'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
    'proposals': {'FisherJumpManager': {'verbose_fisher': False}, 'DEJumpManager': {'de_size': 256}},
}

TINY_ADAPTIVE_SPEC: dict[str, Any] = {
    **TINY_FIXED_SPEC,
    'name': 'dash_tiny_adaptive',
    'seed': 43,
    'run': {'n_steps': 512, 'block_size': 64, 'store_thin': 1, 'checkpoint_every_blocks': 4},
    'adaptive': {
        'mode': 'entropy',
        'update_every_blocks': 2,
        'budget_blocks': 4,
        'forgetting': 0.15,
        'freeze_dlog': 0.05,
        'freeze_consecutive': 3,
        'n_prior_draws': 64,
    },
}


def _run_tiny(spec_data: dict[str, Any], out_dir: Path) -> Path:
    """Run one tiny spec through the real harness, guarding the seed state."""
    reset_seed_guard_for_tests()
    try:
        return run_from_spec(RunSpec.from_dict(spec_data), out_dir)
    finally:
        reset_seed_guard_for_tests()


@pytest.fixture(scope='module')
def fixed_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A completed fixed-ladder run artifact."""
    return _run_tiny(TINY_FIXED_SPEC, tmp_path_factory.mktemp('dash_fixed'))


@pytest.fixture(scope='module')
def adaptive_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A completed adaptive run artifact (has ladder history)."""
    return _run_tiny(TINY_ADAPTIVE_SPEC, tmp_path_factory.mktemp('dash_adaptive'))


@pytest.fixture(scope='module')
def midrun_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An artifact flushed at a mid-run major report (finalized flag set).

    Advances 2 of 4 blocks with n_steps_per_major_report at the halfway
    point, then stops: the on-disk flush carries finalized=True while the
    run is only half done, exactly the state a live dashboard polls.
    """
    spec_data = dict(TINY_FIXED_SPEC)
    spec_data['name'] = 'dash_midrun'
    spec_data['run'] = {'n_steps': 256, 'n_steps_per_major_report': 128, 'block_size': 64, 'checkpoint_every_blocks': 2}
    spec = RunSpec.from_dict(spec_data)
    out_path = tmp_path_factory.mktemp('dash_midrun') / 'midrun.h5'
    reset_seed_guard_for_tests()
    try:
        seeds = seed_run(spec.seed)
        config = spec.build_proposal_config()
        provenance = collect_provenance(
            spec.seed, *seeds, spec_toml=spec.to_toml_text(), proposal_config_ini=config_to_text(config)
        )
        sampler, _like_obj = build_sampler(spec, config=config, artifact_path=out_path, provenance=provenance)
        sampler.advance_N_blocks(2)
    finally:
        reset_seed_guard_for_tests()
    return out_path


def test_snapshot_shapes_and_labels(fixed_artifact: Path) -> None:
    """The reader loads a coherent snapshot with labeled jump types."""
    snapshot = load_snapshot(fixed_artifact)
    n_chain = snapshot.n_chain
    assert snapshot.logL_means.shape == (snapshot.n_blocks, n_chain)
    assert snapshot.logL2_means.shape == snapshot.logL_means.shape
    assert snapshot.accept_record.shape[0] == 2
    assert snapshot.accept_record.shape[1] == n_chain
    assert len(snapshot.jump_labels) == snapshot.accept_record.shape[2]
    # the artifact additions for the dashboard are present and consistent
    assert snapshot.initial_Ts.shape == snapshot.Ts.shape
    assert snapshot.finalized
    assert snapshot.likelihood_name == 'gaussian'
    assert snapshot.n_par == 3
    assert snapshot.block_size == 64
    # store columns: the n_cold readout chains, then the arg_record extras
    assert snapshot.record_indices.tolist() == [0, 3]
    assert snapshot.n_recorded == 2
    assert snapshot.samples.shape[1] == snapshot.n_recorded
    assert snapshot.logLs.shape[1] == snapshot.n_recorded
    assert snapshot.record_history_indices.shape == (snapshot.n_blocks, snapshot.n_recorded)


def test_artifact_records_dashboard_metadata(fixed_artifact: Path) -> None:
    """The writer records jump labels and the run-start ladder."""
    with h5py.File(str(fixed_artifact), 'r') as hf:
        trackers = hf['trackers']
        assert isinstance(trackers, h5py.Group)
        labels = [str(label) for label in np.asarray(trackers.attrs['jump_labels'])]
        accept_record = hf['trackers/accept_record']
        assert isinstance(accept_record, h5py.Dataset)
        accept_shape = accept_record.shape
        assert 'ladder/initial_Ts' in hf
    assert len(labels) == accept_shape[2]
    assert all(labels)


def test_watcher_reloads_only_on_change(fixed_artifact: Path) -> None:
    """poll() caches on the stat token and reloads when the flush changes."""
    watcher = ArtifactWatcher(fixed_artifact)
    first = watcher.poll()
    assert first is not None
    assert watcher.poll() is first
    # a new flush is an atomic replace; simulate with a touch
    stat = fixed_artifact.stat()
    os.utime(fixed_artifact, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = watcher.poll()
    assert second is not None
    assert second is not first
    assert second.n_iterations == first.n_iterations


def test_watcher_tolerates_missing_file(tmp_path: Path) -> None:
    """A vanished artifact reports an error instead of raising."""
    watcher = ArtifactWatcher(tmp_path / 'missing.h5')
    assert watcher.poll() is None
    assert 'not readable' in watcher.last_error


def test_list_artifacts_skips_tmp(tmp_path: Path, fixed_artifact: Path) -> None:
    """Directory listing finds .h5 files but never in-flight .tmp files."""
    target = tmp_path / 'run.h5'
    target.write_bytes(fixed_artifact.read_bytes())
    (tmp_path / 'run.h5.tmp').write_bytes(b'partial')
    found = list_artifacts(tmp_path)
    assert found == [target]
    assert list_artifacts(target) == [target]


def test_ladder_segments_fixed(fixed_artifact: Path) -> None:
    """A fixed-ladder run is one current segment covering every block."""
    snapshot = load_snapshot(fixed_artifact)
    segments = diag.ladder_segments(snapshot)
    assert len(segments) == 1
    assert segments[0].start_block == 0
    assert segments[0].stop_block == snapshot.n_blocks
    assert segments[0].is_current


def test_ladder_segments_adaptive(adaptive_artifact: Path) -> None:
    """Adaptive segments tile the run and start from the initial ladder."""
    snapshot = load_snapshot(adaptive_artifact)
    assert snapshot.history is not None
    assert snapshot.history.block_index.size > 0
    segments = diag.ladder_segments(snapshot)
    assert segments[0].start_block == 0
    assert np.array_equal(segments[0].Ts, snapshot.initial_Ts)
    for previous, following in itertools.pairwise(segments):
        assert previous.stop_block == following.start_block
    assert segments[-1].stop_block == snapshot.n_blocks


def test_rate_tables_are_rates(fixed_artifact: Path) -> None:
    """Acceptance and exchange values are probabilities (or NaN)."""
    snapshot = load_snapshot(fixed_artifact)
    for window in ('total', 'latest'):
        table = diag.acceptance_by_temperature(snapshot, window)
        finite = np.isfinite(table.values)
        assert np.all(table.values[finite] >= 0.0)
        assert np.all(table.values[finite] <= 1.0)
        assert set(table.labels) <= set(snapshot.jump_labels)
        rates = diag.exchange_rates(snapshot, window)
        finite_nn = np.isfinite(rates.nn_rate)
        assert np.all(rates.nn_rate[finite_nn] >= 0.0)
        assert np.all(rates.nn_rate[finite_nn] <= 1.0)
    history = diag.exchange_history(snapshot)
    assert history.itrns.size == history.nn_rates.shape[0]
    assert int(history.itrns[-1]) == snapshot.n_iterations


def test_esd_normalizations(fixed_artifact: Path) -> None:
    """The accepted-only |Δx|² total never exceeds the all-proposal total.

    The means carry no such ordering (accepted jumps are typically smaller
    than proposals), but the sums recovered as value * trials must nest.
    """
    snapshot = load_snapshot(fixed_artifact)
    per_proposal = diag.esd_by_temperature(snapshot)
    per_accepted = diag.esd_by_temperature(snapshot, accepted_only=True)
    assert per_proposal.labels == per_accepted.labels
    both = np.isfinite(per_proposal.values) & np.isfinite(per_accepted.values)
    assert np.all(per_accepted.values[both] >= 0.0)
    sums_accepted = per_accepted.values[both] * per_accepted.trials[both]
    sums_all = per_proposal.values[both] * per_proposal.trials[both]
    assert np.all(sums_accepted <= sums_all + 1e-9)


def test_entropy_curves_nondecreasing(fixed_artifact: Path) -> None:
    """S(T) is an integral of a nonnegative integrand, so it never decreases."""
    snapshot = load_snapshot(fixed_artifact)
    for curve in diag.entropy_curves(snapshot):
        assert np.all(np.diff(curve.y) >= -1e-9)


def test_flow_and_round_trips(fixed_artifact: Path) -> None:
    """Flow fractions are probabilities and arrival curves are cumulative."""
    snapshot = load_snapshot(fixed_artifact)
    flow = diag.flow_fraction(snapshot)
    finite = np.isfinite(flow.f_latest)
    assert np.all(flow.f_latest[finite] >= 0.0)
    assert np.all(flow.f_latest[finite] <= 1.0)
    trips = diag.round_trip_summary(snapshot)
    assert np.all(np.diff(trips.itrns_cold) >= 0)
    assert np.all(np.diff(trips.cumulative_cold) == 1)
    assert trips.cold_arrivals_per_walker.sum() == trips.cumulative_cold.size


def test_acf_basics(fixed_artifact: Path) -> None:
    """ACF is 1 at lag zero and iid noise has a short integrated time."""
    snapshot = load_snapshot(fixed_artifact)
    results = diag.logl_acf(snapshot, [0], max_lag=64)
    assert len(results) == 1
    assert results[0].rho[0] == pytest.approx(1.0)
    noise = get_rng(7).standard_normal(4096)
    rho = diag.normalized_acf(noise, 128)
    assert diag.integrated_autocorr_time(rho) < 3.0
    # out-of-range chains are skipped rather than raising
    assert diag.logl_acf(snapshot, [99], max_lag=64) == []


def test_corner_and_trace_selection(fixed_artifact: Path) -> None:
    """Corner/trace subsetting respects dims, burn-in, and the point cap."""
    snapshot = load_snapshot(fixed_artifact)
    matrix, labels = diag.corner_matrix(snapshot, [0, 2, 99], chain=0, burnin_rows=10, max_points=100)
    assert labels == ['par 0', 'par 2']
    assert matrix.shape[1] == 2
    assert matrix.shape[0] <= 100
    curves = diag.parameter_trace(snapshot, 0, [0, 1], burnin_rows=0, max_points=50)
    assert [curve.label for curve in curves] == ['par 0', 'par 1']
    assert all(curve.x.size <= 50 for curve in curves)


def test_header_items_cover_key_config(adaptive_artifact: Path) -> None:
    """The header summarizes the essential run configuration."""
    snapshot = load_snapshot(adaptive_artifact)
    items = dict(diag.header_items(snapshot))
    assert items['run'] == 'dash_tiny_adaptive'
    assert 'gaussian' in items['likelihood']
    assert 'n_chain=6' in items['ladder']
    assert items['adaptive'].startswith('entropy')
    assert items['block size'] == '64'
    assert items['status'] == 'finalized'


def test_rate_tables_use_segment_ladders(adaptive_artifact: Path) -> None:
    """Tracker windows are grouped by the ladder in effect during them.

    Archives are taken before a ladder update mutates Ts, so counts from
    the first window belong to the initial ladder's temperatures, and the
    whole-run table must keep old-ladder temperatures as their own bins
    rather than relabeling those counts with the final ladder.
    """
    snapshot = load_snapshot(adaptive_artifact)
    assert snapshot.history is not None
    applied_Ts = snapshot.history.Ts[snapshot.history.applied]
    assert applied_Ts.shape[0] > 0, 'fixture must apply at least one ladder update'
    assert not np.array_equal(np.unique(snapshot.initial_Ts), np.unique(snapshot.Ts)), (
        'fixture ladder must actually move'
    )

    first_window = diag.acceptance_by_temperature(snapshot, 0)
    assert np.array_equal(first_window.Ts, np.unique(snapshot.initial_Ts))
    assert np.array_equal(diag.exchange_rates(snapshot, 0).Ts, np.unique(snapshot.initial_Ts))
    assert np.array_equal(diag.esd_by_temperature(snapshot, 0).Ts, np.unique(snapshot.initial_Ts))

    total = diag.acceptance_by_temperature(snapshot, 'total')
    initial_only = set(np.unique(snapshot.initial_Ts)) - set(np.unique(snapshot.Ts))
    assert initial_only <= set(total.Ts)
    # rebinning must conserve counts: every trial lands in exactly one bin
    record = snapshot.accept_record
    assert total.trials.sum() == pytest.approx(float((record[0] + record[1]).sum()))

    # 'segment' keeps only post-last-update counts: every bin on the current ladder
    segment = diag.acceptance_by_temperature(snapshot, 'segment')
    assert set(segment.Ts) <= set(np.unique(snapshot.Ts))
    assert segment.trials.sum() < total.trials.sum()


def test_segment_window_equals_total_on_fixed_ladder(fixed_artifact: Path) -> None:
    """Without ladder updates the 'segment' and 'total' windows coincide."""
    snapshot = load_snapshot(fixed_artifact)
    segment = diag.acceptance_by_temperature(snapshot, 'segment')
    total = diag.acceptance_by_temperature(snapshot, 'total')
    assert np.array_equal(segment.Ts, total.Ts)
    assert np.array_equal(segment.trials, total.trials)
    assert np.array_equal(segment.values, total.values, equal_nan=True)


def test_run_complete_vs_major_report(midrun_artifact: Path, fixed_artifact: Path) -> None:
    """A mid-run major-report flush is not displayed as a finalized run."""
    midrun = load_snapshot(midrun_artifact)
    assert midrun.finalized, 'writer marks major-report flushes finalized'
    assert midrun.n_iterations < midrun.n_steps
    assert not midrun.run_complete
    assert dict(diag.header_items(midrun))['status'] == 'in progress (major report)'
    completed = load_snapshot(fixed_artifact)
    assert completed.run_complete
    assert dict(diag.header_items(completed))['status'] == 'finalized'


def test_wheel_packaging_includes_dashboard() -> None:
    """Package discovery ships dashboard so python -m dashboard works installed."""
    with (repo_root() / 'pyproject.toml').open('rb') as handle:
        include = tomllib.load(handle)['tool']['setuptools']['packages']['find']['include']
    found = find_packages(where=str(repo_root()), include=include)
    assert {'dashboard', 'dashboard.core', 'dashboard.figures', 'dashboard.themes', 'dashboard.app'} <= set(found)


def test_artifact_selection_allowlist(tmp_path: Path, fixed_artifact: Path) -> None:
    """Client-supplied artifact paths outside the served root are rejected."""
    pytest.importorskip('dash')
    # imported lazily so the reader/diagnostics tests run without dash installed
    from dashboard.app.dash_app import DashboardConfig, _allowed_artifact  # noqa: PLC0415

    served = tmp_path / 'served.h5'
    served.write_bytes(fixed_artifact.read_bytes())
    config = DashboardConfig(artifact=tmp_path)
    assert _allowed_artifact(config, str(served)) == str(served)
    assert _allowed_artifact(config, str(fixed_artifact)) is None
    assert _allowed_artifact(config, str(tmp_path / '..' / 'served.h5')) is None
    assert _allowed_artifact(config, None) is None


def _single_check(check_id: str, snapshot) -> checks.CheckResult:
    """Evaluate one status light by id."""
    results = checks.evaluate_checks(snapshot, [check_id])
    assert len(results) == 1
    return results[0]


def test_checks_registry_and_silencing(fixed_artifact: Path) -> None:
    """Every registered light evaluates; silencing filters by id."""
    snapshot = load_snapshot(fixed_artifact)
    results = checks.evaluate_checks(snapshot)
    assert [result.check_id for result in results] == list(checks.CHECKS)
    valid = {checks.STATUS_OK, checks.STATUS_WARN, checks.STATUS_ALERT, checks.STATUS_NA}
    assert all(result.status in valid for result in results)
    assert all(result.message for result in results)
    subset = checks.evaluate_checks(snapshot, ['finite_moments'])
    assert [result.check_id for result in subset] == ['finite_moments']
    assert checks.evaluate_checks(snapshot, []) == []
    assert checks.worst_status([]) == checks.STATUS_NA
    counts = checks.status_counts(results)
    assert sum(counts.values()) == len(results)


def test_checks_flag_synthetic_pathologies(adaptive_artifact: Path) -> None:
    """Each light trips on a snapshot doctored to show its failure mode."""
    snapshot = load_snapshot(adaptive_artifact)

    # DE rank collapse: only one nonzero eigenvalue at the latest checkpoint
    eigvals = snapshot.de_spectrum_eigvals.copy()
    eigvals[-1, :, 1:] = 0.0
    collapsed = dataclasses.replace(snapshot, de_spectrum_eigvals=eigvals)
    assert _single_check('de_rank', collapsed).status == checks.STATUS_ALERT

    # numerical trouble: a NaN in a recent block moment
    means = snapshot.logL_means.copy()
    means[-1, 0] = np.nan
    poisoned = dataclasses.replace(snapshot, logL_means=means)
    assert _single_check('finite_moments', poisoned).status == checks.STATUS_ALERT

    # no round trips despite a long-enough run
    quiet = dataclasses.replace(snapshot, rt_events=np.zeros((0, 3), dtype=np.int64))
    assert _single_check('round_trips', quiet).status == checks.STATUS_ALERT

    # the healthy artifact itself does not alert on these lights
    assert _single_check('de_rank', snapshot).status in (checks.STATUS_OK, checks.STATUS_WARN)
    assert _single_check('finite_moments', snapshot).status == checks.STATUS_OK
    assert _single_check('round_trips', snapshot).status in (checks.STATUS_OK, checks.STATUS_WARN)


def test_checks_na_paths(fixed_artifact: Path) -> None:
    """Lights report n/a with a reason when their data is absent."""
    snapshot = load_snapshot(fixed_artifact)
    no_checkpoints = dataclasses.replace(
        snapshot,
        checkpoint_itrns=np.zeros(0),
        de_spectrum_eigvals=np.zeros((0, snapshot.n_chain, max(snapshot.n_par, 1))),
    )
    assert _single_check('de_rank', no_checkpoints).status == checks.STATUS_NA
    # fixed-ladder runs have no adaptation state to grade
    assert _single_check('ladder_freeze', snapshot).status == checks.STATUS_NA


def test_status_tab_children(fixed_artifact: Path) -> None:
    """The Status tab renders one card per enabled light plus silencing notes."""
    pytest.importorskip('dash')
    # imported lazily so the reader/diagnostics tests run without dash installed
    from dashboard.app.dash_app import status_tab_children  # noqa: PLC0415

    snapshot = load_snapshot(fixed_artifact)
    all_cards = status_tab_children(snapshot, list(checks.CHECKS))
    assert len(all_cards) == len(checks.CHECKS)
    one_enabled = status_tab_children(snapshot, ['finite_moments'])
    assert len(one_enabled) == 2  # the card plus the silenced-count note
    none_enabled = status_tab_children(snapshot, [])
    assert len(none_enabled) == 2  # silenced-count note plus the re-enable hint
    assert status_tab_children(None, list(checks.CHECKS))


def test_registry_layouts_reference_known_plots() -> None:
    """Every layout entry points at a registered plot."""
    for layout in LAYOUTS.values():
        for plot_ids in layout.values():
            for plot_id in plot_ids:
                assert plot_id in PLOTS


@pytest.mark.parametrize('artifact_fixture', ['fixed_artifact', 'adaptive_artifact'])
def test_every_figure_builds(artifact_fixture: str, request: pytest.FixtureRequest) -> None:
    """Every registered figure builds a plotly Figure on real artifacts."""
    snapshot = load_snapshot(request.getfixturevalue(artifact_fixture))
    opts = ViewOptions(dims=[0, 1], chains=[0], max_lag=32)
    for theme_name in THEMES:
        theme = get_theme(theme_name)
        for plot_id in PLOTS:
            figure = build_figure(plot_id, snapshot, opts, theme)
            assert isinstance(figure, go.Figure)


def test_dash_app_builds(fixed_artifact: Path) -> None:
    """The Dash app constructs against a real artifact without serving."""
    dash = pytest.importorskip('dash')
    # imported lazily so the reader/diagnostics tests run without dash installed
    from dashboard.app.dash_app import DashboardConfig, create_app  # noqa: PLC0415

    app = create_app(DashboardConfig(artifact=fixed_artifact))
    assert isinstance(app, dash.Dash)
    layout_ids = {component.id for component in app.layout._traverse() if getattr(component, 'id', None)}
    assert {
        'poll',
        'snapshot-token',
        'header',
        'tab-select',
        'tab-content',
        'artifact-select',
        'theme-select',
        'status-checks',
    } <= layout_ids
