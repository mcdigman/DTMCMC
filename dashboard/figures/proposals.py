"""Proposal-performance diagnostics: acceptance, ESD, exchange rates."""

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from dashboard.core import diagnostics as diag
from dashboard.figures.base import annotate_empty, base_layout, sequential_colorscale

if TYPE_CHECKING:
    from dashboard.core.reader import RunSnapshot
    from dashboard.figures.options import ViewOptions
    from dashboard.themes import Theme


def _finite_T_axis(Ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mask to finite temperatures for log-x plotting."""
    Ts_arr = np.asarray(Ts)
    finite = np.isfinite(Ts_arr) & (Ts_arr > 0.0)
    return Ts_arr[finite], finite


def _rate_table_traces(table: diag.RateTable, theme: Theme, *, value_name: str) -> list[go.Scatter]:
    """Lines+markers per jump type over temperature, fixed categorical slots."""
    Ts_plot, finite = _finite_T_axis(table.Ts)
    traces: list[go.Scatter] = []
    for itrj, label in enumerate(table.labels):
        color = theme.categorical[itrj % len(theme.categorical)]
        traces.append(
            go.Scatter(
                x=Ts_plot,
                y=table.values[finite, itrj],
                name=label,
                mode='lines+markers',
                line={'color': color, 'width': 2.0},
                marker={'size': 7, 'color': color},
                customdata=table.trials[finite, itrj],
                hovertemplate=f'T=%{{x:.4g}}<br>{value_name}=%{{y:.4g}}<br>n=%{{customdata:.0f}}<extra>{label}</extra>',
            )
        )
    return traces


def fig_acceptance_vs_temperature(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Acceptance rate per tracked proposal against temperature."""
    table = diag.acceptance_by_temperature(snapshot, opts.window)
    fig = go.Figure(
        data=_rate_table_traces(table, theme, value_name='acceptance'),
        layout=base_layout(theme, x_title='temperature T', y_title='acceptance rate', log_x=True),
    )
    fig.update_yaxes(range=[0.0, 1.0])
    if not table.labels:
        annotate_empty(fig, theme, 'no proposals recorded in this window')
    return fig


def fig_esd_vs_temperature(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Expected squared jump displacement per proposal against temperature."""
    table = diag.esd_by_temperature(snapshot, opts.window, accepted_only=opts.accepted_only)
    normalization = 'per accepted jump' if opts.accepted_only else 'per proposal'
    fig = go.Figure(
        data=_rate_table_traces(table, theme, value_name='E[|Δx|²]'),
        layout=base_layout(theme, x_title='temperature T', y_title=f'E[|Δx|²] {normalization}', log_x=True, log_y=True),
    )
    if not table.labels:
        annotate_empty(fig, theme, 'no proposals recorded in this window')
    return fig


def fig_exchange_vs_temperature(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Exchange acceptance against temperature slot.

    For a constant-acceptance ladder the nearest-neighbor curve should be
    flat; the overall nn rate is drawn as a muted guide.
    """
    rates = diag.exchange_rates(snapshot, opts.window)
    Ts_plot, finite = _finite_T_axis(rates.Ts)
    fig = go.Figure(layout=base_layout(theme, x_title='temperature T', y_title='exchange acceptance', log_x=True))
    fig.add_trace(
        go.Scatter(
            x=Ts_plot,
            y=rates.nn_rate[finite],
            name='nearest neighbor',
            mode='lines+markers',
            line={'color': theme.categorical[0], 'width': 2.0},
            marker={'size': 7},
            customdata=rates.nn_trials[finite],
            hovertemplate='T=%{x:.4g}<br>rate=%{y:.4g}<br>n=%{customdata:.0f}<extra>nearest neighbor</extra>',
        )
    )
    if not np.allclose(rates.all_rate[finite], rates.nn_rate[finite], equal_nan=True):
        fig.add_trace(
            go.Scatter(
                x=Ts_plot,
                y=rates.all_rate[finite],
                name='all exchanges',
                mode='lines+markers',
                line={'color': theme.categorical[1], 'width': 2.0},
                marker={'size': 7},
                customdata=rates.all_trials[finite],
                hovertemplate='T=%{x:.4g}<br>rate=%{y:.4g}<br>n=%{customdata:.0f}<extra>all exchanges</extra>',
            )
        )
    if np.isfinite(rates.overall_nn_rate):
        fig.add_hline(
            y=rates.overall_nn_rate,
            line={'color': theme.ink_muted, 'dash': 'dot', 'width': 2.0},
            annotation_text=f'overall {rates.overall_nn_rate:.3f}',
            annotation_font={'color': theme.ink_muted},
        )
    fig.update_yaxes(range=[0.0, 1.0])
    return fig


def fig_exchange_history(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Nearest-neighbor exchange acceptance per temperature slot over the run.

    Each column is one tracker-archive window; a constant-acceptance ladder
    working as designed shows columns converging to a uniform color.
    """
    del opts
    history = diag.exchange_history(snapshot)
    fig = go.Figure(
        layout=base_layout(
            theme, x_title='archive window (ends at hovered iteration)', y_title='temperature slot (cold → hot)'
        )
    )
    if history.itrns.size == 0:
        return annotate_empty(fig, theme, 'no archive windows yet')
    # windows are unevenly spaced in iteration (block archives plus ladder
    # updates), so index the x axis by window and carry the iteration in hover
    itrn_columns = np.broadcast_to(history.itrns[:, np.newaxis], history.nn_rates.shape).T
    fig.add_trace(
        go.Heatmap(
            x=np.arange(history.itrns.size).tolist(),
            y=np.arange(snapshot.n_chain).tolist(),
            z=history.nn_rates.T,
            zmin=0.0,
            zmax=1.0,
            colorscale=sequential_colorscale(theme),
            customdata=itrn_columns,
            colorbar={
                'title': {'text': 'nn rate', 'font': {'color': theme.ink_secondary}},
                'tickfont': {'color': theme.ink_muted},
            },
            hovertemplate='window %{x} (itrn %{customdata})<br>slot %{y}<br>rate %{z:.3f}<extra></extra>',
        )
    )
    return fig
