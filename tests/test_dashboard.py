"""Dashboard acceptance tests: artifact reader, diagnostics, figures, app.

Two tiny Gaussian runs (one fixed-ladder, one adaptive) are generated once
per module through the real harness, so the reader is exercised against
artifacts the writer actually produces, including the dashboard-support
additions (trackers jump_labels attr, ladder/initial_Ts dataset).
"""

import itertools
import os
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import plotly.graph_objects as go
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from dashboard.core import diagnostics as diag
from dashboard.core.reader import ArtifactWatcher, list_artifacts, load_snapshot
from dashboard.figures.options import ViewOptions
from dashboard.figures.registry import LAYOUTS, PLOTS, build_figure
from dashboard.themes import THEMES, get_theme
from DTMCMC.rng_helpers import get_rng, reset_seed_guard_for_tests
from experiments.harness.runner import run_from_spec
from experiments.harness.spec import RunSpec

TINY_FIXED_SPEC: dict[str, Any] = {
    'name': 'dash_tiny_fixed',
    'seed': 42,
    'likelihood': {'name': 'gaussian', 'n_par': 3, 'cutoff': 5},
    'ladder': {'kind': 'geometric', 'n_chain': 6, 'n_cold': 1, 'T_cold': 1.0, 'T_min': 1.0, 'T_max': 100.0, 'n_inf_final': 1},
    'run': {'n_steps': 256, 'block_size': 64, 'store_thin': 1, 'n_record': -1, 'checkpoint_every_blocks': 2},
    'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
    'proposals': {'FisherJumpManager': {'verbose_fisher': False}, 'DEJumpManager': {'de_size': 256}},
}

TINY_ADAPTIVE_SPEC: dict[str, Any] = {
    **TINY_FIXED_SPEC,
    'name': 'dash_tiny_adaptive',
    'seed': 43,
    'run': {'n_steps': 512, 'block_size': 64, 'store_thin': 1, 'n_record': -1, 'checkpoint_every_blocks': 4},
    'adaptive': {'mode': 'entropy', 'update_every_blocks': 2, 'budget_blocks': 4, 'forgetting': 0.15, 'freeze_dlog': 0.05, 'freeze_consecutive': 3, 'n_prior_draws': 64},
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
        assert np.all(table.values[finite] >= 0.)
        assert np.all(table.values[finite] <= 1.)
        assert set(table.labels) <= set(snapshot.jump_labels)
        rates = diag.exchange_rates(snapshot, window)
        finite_nn = np.isfinite(rates.nn_rate)
        assert np.all(rates.nn_rate[finite_nn] >= 0.)
        assert np.all(rates.nn_rate[finite_nn] <= 1.)
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
    assert np.all(per_accepted.values[both] >= 0.)
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
    assert np.all(flow.f_latest[finite] >= 0.)
    assert np.all(flow.f_latest[finite] <= 1.)
    trips = diag.round_trip_summary(snapshot)
    assert np.all(np.diff(trips.itrns_cold) >= 0)
    assert np.all(np.diff(trips.cumulative_cold) == 1)
    assert trips.cold_arrivals_per_walker.sum() == trips.cumulative_cold.size


def test_acf_basics(fixed_artifact: Path) -> None:
    """ACF is 1 at lag zero and iid noise has a short integrated time."""
    snapshot = load_snapshot(fixed_artifact)
    results = diag.logl_acf(snapshot, [0], max_lag=64)
    assert len(results) == 1
    assert results[0].rho[0] == pytest.approx(1.)
    noise = get_rng(7).standard_normal(4096)
    rho = diag.normalized_acf(noise, 128)
    assert diag.integrated_autocorr_time(rho) < 3.
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
    layout_ids = {component.id for component in app.layout._traverse() if getattr(component, 'id', None)}  # noqa: SLF001
    assert {'poll', 'snapshot-token', 'header', 'tab-select', 'tab-content', 'artifact-select', 'theme-select'} <= layout_ids
