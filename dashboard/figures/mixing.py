"""Mixing diagnostics: walker flow, round trips, logL history, DE spectra."""

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from dashboard.core import diagnostics as diag
from dashboard.figures.base import annotate_empty, base_layout, ordinal_colors, sequential_colorscale

if TYPE_CHECKING:
    from dashboard.core.reader import RunSnapshot
    from dashboard.figures.options import ViewOptions
    from dashboard.themes import Theme


def _temperature_hover(Ts: np.ndarray) -> list[str]:
    """Readable temperature tags per slot for hover text."""
    return [f'T={T:.4g}' if np.isfinite(T) else 'T=inf' for T in np.asarray(Ts)]


def fig_flow_fraction(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Walker up-flow fraction f against rung index, with the linear ideal.

    f is the fraction of residents whose last extreme visit was cold; the
    ideal constant-round-trip-flow profile is the dotted straight line.
    """
    flow = diag.flow_fraction(snapshot, opts.trailing_blocks)
    rungs = np.arange(snapshot.n_chain)
    window_label = f'last {opts.trailing_blocks} blocks' if opts.trailing_blocks > 0 else 'all blocks'
    fig = go.Figure(layout=base_layout(theme, x_title='temperature slot (cold → hot)', y_title='up-flow fraction f'))
    fig.add_trace(go.Scatter(
        x=rungs, y=flow.f_ideal, name='ideal (linear)', mode='lines',
        line={'color': theme.ink_muted, 'dash': 'dot', 'width': 2.},
        hovertemplate='slot %{x}<br>f=%{y:.3f}<extra>ideal</extra>',
    ))
    fig.add_trace(go.Scatter(
        x=rungs, y=flow.f_latest, name=f'measured ({window_label})', mode='lines+markers',
        line={'color': theme.categorical[0], 'width': 2.}, marker={'size': 7},
        text=_temperature_hover(flow.Ts),
        hovertemplate='slot %{x} (%{text})<br>f=%{y:.3f}<extra>measured</extra>',
    ))
    fig.update_yaxes(range=[-0.02, 1.02])
    return fig


def fig_flow_history(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Per-block up-flow fraction heatmap: temperature slot against block."""
    flow = diag.flow_fraction(snapshot, opts.trailing_blocks)
    fig = go.Figure(layout=base_layout(theme, x_title='block', y_title='temperature slot (cold → hot)'))
    if flow.f_per_block.shape[0] == 0:
        return annotate_empty(fig, theme, 'no completed blocks yet')
    fig.add_trace(go.Heatmap(
        x=flow.blocks, y=np.arange(snapshot.n_chain), z=flow.f_per_block.T,
        zmin=0., zmax=1., colorscale=sequential_colorscale(theme),
        colorbar={'title': {'text': 'f', 'font': {'color': theme.ink_secondary}}, 'tickfont': {'color': theme.ink_muted}},
        hovertemplate='block %{x}<br>slot %{y}<br>f=%{z:.3f}<extra></extra>',
    ))
    return fig


def fig_round_trips(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Cumulative extreme-arrival counts against iteration.

    Cold arrivals complete hot→cold trips and hot arrivals cold→hot trips;
    steady mixing gives straight lines. Vertical guides mark ladder-update
    segment boundaries (arrivals are never paired across them).
    """
    del opts
    trips = diag.round_trip_summary(snapshot)
    fig = go.Figure(layout=base_layout(theme, x_title='iteration', y_title='cumulative arrivals'))
    if trips.itrns_cold.size == 0 and trips.itrns_hot.size == 0:
        return annotate_empty(fig, theme, 'no round-trip events yet')
    fig.add_trace(go.Scatter(
        x=trips.itrns_cold, y=trips.cumulative_cold, name='arrived cold (hot→cold)', mode='lines',
        line={'color': theme.categorical[0], 'width': 2.},
        hovertemplate='itrn %{x}<br>%{y} arrivals<extra>arrived cold</extra>',
    ))
    fig.add_trace(go.Scatter(
        x=trips.itrns_hot, y=trips.cumulative_hot, name='arrived hot (cold→hot)', mode='lines',
        line={'color': theme.categorical[1], 'width': 2.},
        hovertemplate='itrn %{x}<br>%{y} arrivals<extra>arrived hot</extra>',
    ))
    for segment_itrn in trips.segment_itrns:
        fig.add_vline(x=int(segment_itrn), line={'color': theme.gridline, 'width': 1.})
    return fig


def fig_walker_arrivals(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Cold arrivals per walker over the whole run (are all walkers cycling?)."""
    del opts
    trips = diag.round_trip_summary(snapshot)
    fig = go.Figure(layout=base_layout(theme, x_title='walker id', y_title='cold arrivals'))
    fig.add_trace(go.Bar(
        x=trips.walker_ids, y=trips.cold_arrivals_per_walker, name='cold arrivals',
        marker={'color': theme.categorical[0]},
        customdata=trips.n_cycles_current_segment,
        hovertemplate='walker %{x}<br>%{y} cold arrivals<br>%{customdata} full cycles (current segment)<extra></extra>',
    ))
    fig.update_layout(bargap=0.35, showlegend=False)
    return fig


def fig_logl_block_history(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Per-block mean logL for every temperature slot against block number.

    The classic burn-in diagnostic: early blocks dominate the range, so the
    first burn-in blocks are trimmed (configurable). Cold slots are the
    darkest lines.
    """
    history = diag.logl_block_history(snapshot, opts.burnin_blocks)
    fig = go.Figure(layout=base_layout(theme, x_title='block', y_title='block mean logL'))
    if history.blocks.size == 0:
        return annotate_empty(fig, theme, 'no blocks past the burn-in trim')
    colors = ordinal_colors(theme, snapshot.n_chain)[::-1]
    hover_tags = _temperature_hover(history.Ts)
    for slot in range(snapshot.n_chain):
        fig.add_trace(go.Scatter(
            x=history.blocks, y=history.values[:, slot], name=f'slot {slot} ({hover_tags[slot]})', mode='lines',
            line={'color': colors[slot], 'width': 1.6},
            showlegend=slot in (0, snapshot.n_chain - 1),
            hovertemplate=f'block %{{x}}<br>E[logL]=%{{y:.6g}}<extra>slot {slot} {hover_tags[slot]}</extra>',
        ))
    return fig


def fig_de_eigvals(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Latest DE-buffer difference eigenspectrum per temperature slot.

    Trailing eigenvalues pinned near zero indicate rank collapse of the
    difference span the DE proposals draw from.
    """
    del opts
    spectrum = diag.de_spectrum_summary(snapshot)
    fig = go.Figure(layout=base_layout(theme, x_title='eigenvalue index', y_title='eigenvalue', log_y=True))
    if spectrum.itrns.size == 0:
        return annotate_empty(fig, theme, 'no DE-spectrum checkpoints recorded')
    latest = spectrum.eigvals[-1]
    colors = ordinal_colors(theme, snapshot.n_chain)[::-1]
    hover_tags = _temperature_hover(snapshot.Ts)
    for slot in range(latest.shape[0]):
        fig.add_trace(go.Scatter(
            x=np.arange(latest.shape[1]), y=latest[slot], name=f'slot {slot} ({hover_tags[slot]})', mode='lines+markers',
            line={'color': colors[slot], 'width': 1.6}, marker={'size': 5},
            showlegend=slot in (0, latest.shape[0] - 1),
            hovertemplate=f'index %{{x}}<br>λ=%{{y:.4g}}<extra>slot {slot} {hover_tags[slot]}</extra>',
        ))
    fig.update_layout(title={'text': f'checkpoint at iteration {int(spectrum.itrns[-1])}', 'font': {'size': 12, 'color': theme.ink_secondary}})
    return fig


def fig_de_effective_rank(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """DE-buffer effective rank against temperature, one curve per checkpoint."""
    del opts
    spectrum = diag.de_spectrum_summary(snapshot)
    fig = go.Figure(layout=base_layout(theme, x_title='temperature T', y_title='effective rank (participation ratio)', log_x=True))
    if spectrum.itrns.size == 0:
        return annotate_empty(fig, theme, 'no DE-spectrum checkpoints recorded')
    Ts_arr = np.asarray(snapshot.Ts)
    finite = np.isfinite(Ts_arr) & (Ts_arr > 0.)
    n_checkpoints = spectrum.itrns.size
    colors = ordinal_colors(theme, max(n_checkpoints - 1, 1))
    for idx in range(n_checkpoints):
        is_latest = idx == n_checkpoints - 1
        fig.add_trace(go.Scatter(
            x=Ts_arr[finite], y=spectrum.eff_rank[idx][finite], name=f'itrn {int(spectrum.itrns[idx])}', mode='lines',
            line={'color': theme.highlight if is_latest else colors[min(idx, len(colors) - 1)], 'width': 2.5 if is_latest else 1.4},
            hovertemplate=f'T=%{{x:.4g}}<br>rank=%{{y:.3g}}<extra>itrn {int(spectrum.itrns[idx])}</extra>',
        ))
    n_par = snapshot.n_par
    if n_par > 0:
        fig.add_hline(y=n_par, line={'color': theme.ink_muted, 'dash': 'dot', 'width': 2.}, annotation_text=f'full rank {n_par}', annotation_font={'color': theme.ink_muted})
    return fig
