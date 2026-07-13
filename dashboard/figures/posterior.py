"""Posterior-convergence diagnostics: corner, traces, autocorrelation."""

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.core import diagnostics as diag
from dashboard.figures.base import annotate_empty, base_layout, curve_traces, sequential_colorscale

if TYPE_CHECKING:
    from dashboard.core.reader import RunSnapshot
    from dashboard.figures.options import ViewOptions
    from dashboard.themes import Theme


def fig_corner(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Corner plot of a dimension subset of one recorded chain.

    Diagonal panels are 1-D histograms, lower-triangle panels are filled
    2-D histogram contours. The dimension list must stay a small subset:
    high-dimensional likelihoods cannot be cornered whole.
    """
    matrix, labels = diag.corner_matrix(snapshot, opts.dims, opts.chain, opts.burnin_rows, opts.corner_max_points)
    n_dims = len(labels)
    if n_dims == 0 or matrix.shape[0] < 2:
        fig = go.Figure(layout=base_layout(theme))
        return annotate_empty(fig, theme, 'no stored samples for the selected chain/dims')

    fig = make_subplots(rows=n_dims, cols=n_dims, horizontal_spacing=0.02, vertical_spacing=0.02, shared_xaxes=True)
    for row in range(n_dims):
        for col in range(row + 1):
            if row == col:
                fig.add_trace(go.Histogram(
                    x=matrix[:, col], nbinsx=48, marker={'color': theme.categorical[0]},
                    hovertemplate=f'{labels[col]}: %{{x:.4g}}<br>count %{{y}}<extra></extra>',
                ), row=row + 1, col=col + 1)
            else:
                fig.add_trace(go.Histogram2dContour(
                    x=matrix[:, col], y=matrix[:, row], ncontours=10,
                    colorscale=sequential_colorscale(theme), showscale=False,
                    contours={'coloring': 'fill', 'showlines': False},
                    hovertemplate=f'{labels[col]}: %{{x:.4g}}<br>{labels[row]}: %{{y:.4g}}<extra></extra>',
                ), row=row + 1, col=col + 1)
    base = base_layout(theme)
    fig.update_layout(base.to_plotly_json(), showlegend=False, height=max(180 * n_dims, 360))
    fig.update_xaxes(gridcolor=theme.gridline, linecolor=theme.baseline, tickfont={'color': theme.ink_muted, 'size': 10}, showticklabels=False)
    fig.update_yaxes(gridcolor=theme.gridline, linecolor=theme.baseline, tickfont={'color': theme.ink_muted, 'size': 10}, showticklabels=False)
    for col in range(n_dims):
        fig.update_xaxes(title={'text': labels[col], 'font': {'color': theme.ink_secondary, 'size': 11}}, showticklabels=True, row=n_dims, col=col + 1)
    for row in range(1, n_dims):
        fig.update_yaxes(title={'text': labels[row], 'font': {'color': theme.ink_secondary, 'size': 11}}, showticklabels=True, row=row + 1, col=1)
    n_points = matrix.shape[0]
    fig.update_layout(title={'text': f'{diag.store_column_label(snapshot, opts.chain)}, {n_points} stored points', 'font': {'size': 12, 'color': theme.ink_secondary}})
    return fig


def fig_parameter_trace(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Thinned parameter traces against iteration for one recorded chain."""
    curves = diag.parameter_trace(snapshot, opts.chain, opts.dims, opts.burnin_rows)
    traces: list[go.Scatter] = []
    for itrd, curve in enumerate(curves):
        color = theme.categorical[itrd % len(theme.categorical)]
        traces.append(go.Scatter(
            x=curve.x, y=curve.y, name=curve.label, mode='lines',
            line={'color': color, 'width': 1.4},
            hovertemplate=f'itrn %{{x}}<br>%{{y:.6g}}<extra>{curve.label}</extra>',
        ))
    fig = go.Figure(data=traces, layout=base_layout(theme, x_title='iteration', y_title='parameter value'))
    if not curves:
        annotate_empty(fig, theme, 'no stored samples for the selected chain')
    return fig


def _acf_traces(results: list[diag.AcfResult], snapshot: RunSnapshot, theme: Theme) -> list[go.Scatter]:
    """ACF results as themed traces with tau in the legend label."""
    traces: list[go.Scatter] = []
    for itrc, result in enumerate(results):
        color = theme.categorical[itrc % len(theme.categorical)]
        label = result.label if not np.isfinite(result.tau_int) else f'{result.label} (τ≈{result.tau_int * snapshot.store_thin:.0f} itrn)'
        traces.append(go.Scatter(
            x=result.lags * snapshot.store_thin, y=result.rho, name=label, mode='lines',
            line={'color': color, 'width': 2.},
            hovertemplate='lag %{x} itrn<br>rho=%{y:.4f}<extra>' + label + '</extra>',
        ))
    return traces


def fig_logl_acf(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Autocorrelation of stored logL for the selected recorded chains."""
    results = diag.logl_acf(snapshot, opts.chains, opts.max_lag, opts.burnin_rows)
    fig = go.Figure(data=_acf_traces(results, snapshot, theme), layout=base_layout(theme, x_title='lag (iterations)', y_title='logL autocorrelation'))
    fig.add_hline(y=0., line={'color': theme.baseline, 'width': 1.})
    if not results:
        annotate_empty(fig, theme, 'no stored logL for the selected chains')
    return fig


def fig_parameter_acf(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Autocorrelation of stored parameters for one recorded chain."""
    results = diag.parameter_acf(snapshot, opts.chain, opts.dims, opts.max_lag, opts.burnin_rows)
    fig = go.Figure(data=_acf_traces(results, snapshot, theme), layout=base_layout(theme, x_title='lag (iterations)', y_title='parameter autocorrelation'))
    fig.add_hline(y=0., line={'color': theme.baseline, 'width': 1.})
    if not results:
        annotate_empty(fig, theme, 'no stored samples for the selected chain/dims')
    return fig


def fig_logl_cross_correlation(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Cross-correlation of stored logL between two recorded chains."""
    result = diag.logl_cross_correlation(snapshot, opts.cross_a, opts.cross_b, opts.max_lag, opts.burnin_rows)
    fig = go.Figure(layout=base_layout(theme, x_title='lag (iterations)', y_title='logL cross-correlation'))
    if result is None:
        return annotate_empty(fig, theme, 'need two recorded chains with stored logL')
    fig.add_trace(go.Scatter(
        x=result.lags * snapshot.store_thin, y=result.rho, name=result.label, mode='lines',
        line={'color': theme.categorical[0], 'width': 2.},
        hovertemplate='lag %{x} itrn<br>rho=%{y:.4f}<extra>' + result.label + '</extra>',
    ))
    fig.add_hline(y=0., line={'color': theme.baseline, 'width': 1.})
    return fig


def fig_logl_trace(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Stored logL traces against iteration for the selected recorded chains."""
    start = max(0, min(int(opts.burnin_rows), int(snapshot.logLs.shape[0])))
    rows = diag.downsample_rows(snapshot.logLs.shape[0] - start, 5000) + start
    curves = [
        diag.Curve(f'logL {diag.store_column_label(snapshot, chain)}', rows * snapshot.store_thin, snapshot.logLs[rows, chain], 'applied')
        for chain in opts.chains
        if 0 <= chain < snapshot.logLs.shape[1]
    ]
    fig = go.Figure(data=curve_traces(curves, theme), layout=base_layout(theme, x_title='iteration', y_title='stored logL'))
    if not curves:
        annotate_empty(fig, theme, 'no stored logL for the selected chains')
    return fig
