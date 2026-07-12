"""Thermodynamic diagnostics: E[logL], heat capacity, entropy, ladder spacing."""

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from dashboard.core import diagnostics as diag
from dashboard.figures.base import annotate_empty, base_layout, curve_traces

if TYPE_CHECKING:
    from dashboard.core.reader import RunSnapshot
    from dashboard.figures.options import ViewOptions
    from dashboard.themes import Theme


def stride_history(curves: list[diag.Curve], stride: int) -> list[diag.Curve]:
    """Thin history curves to every nth, always keeping current/reference."""
    if stride <= 1:
        return curves
    history = [curve for curve in curves if curve.emphasis in ('applied', 'held')]
    keep = {id(curve) for curve in history[::stride]}
    # the last history curve is the most recent context; always keep it
    if history:
        keep.add(id(history[-1]))
    return [curve for curve in curves if curve.emphasis not in ('applied', 'held') or id(curve) in keep]


def fig_mean_logl(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Segment mean log likelihood against temperature (semilog-x)."""
    curves = stride_history(diag.mean_logl_curves(snapshot), opts.segment_stride)
    fig = go.Figure(data=curve_traces(curves, theme), layout=base_layout(theme, x_title='temperature T', y_title='E[logL]', log_x=True))
    if not curves:
        annotate_empty(fig, theme, 'no completed blocks yet')
    return fig


def fig_heat_capacity(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Segment-inferred heat capacity C(T) = Var(logL)/T^2 (log-log)."""
    curves = stride_history(diag.heat_capacity_curves(snapshot), opts.segment_stride)
    fig = go.Figure(data=curve_traces(curves, theme), layout=base_layout(theme, x_title='temperature T', y_title='C(T) = Var(logL) / T²', log_x=True, log_y=True))
    if not curves:
        annotate_empty(fig, theme, 'no completed blocks yet')
    return fig


def fig_entropy(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Integrated heat capacity S(T) against temperature (semilog-x)."""
    curves = stride_history(diag.entropy_curves(snapshot), opts.segment_stride)
    fig = go.Figure(data=curve_traces(curves, theme), layout=base_layout(theme, x_title='temperature T', y_title='entropy S(T) (integrated C, hot-referenced)', log_x=True))
    if not curves:
        annotate_empty(fig, theme, 'no completed blocks yet')
    return fig


def _spacing_curve(Ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Adjacent finite-rung spacing in log(T) against link index."""
    finite_Ts = np.sort(np.asarray(Ts)[np.isfinite(Ts) & (np.asarray(Ts) > 0.)])
    spacing = np.diff(np.log(finite_Ts))
    return np.arange(spacing.size), spacing


def fig_ladder_spacing(snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Adjacent rung spacing in log(T), one curve per ladder-history state.

    For a working equal-entropy adaptive ladder the spacing profile should
    converge; the current ladder is highlighted.
    """
    curves: list[diag.Curve] = []
    history = snapshot.history
    if history is not None and history.block_index.size:
        for idx in range(history.Ts.shape[0]):
            x_link, spacing = _spacing_curve(history.Ts[idx])
            emphasis = 'applied' if history.applied[idx] else 'held'
            curves.append(diag.Curve(f'block {int(history.block_index[idx])} ({emphasis})', x_link, spacing, emphasis))
    x_link, spacing = _spacing_curve(snapshot.Ts)
    curves.append(diag.Curve('current ladder', x_link, spacing, 'current'))
    curves = stride_history(curves, opts.segment_stride)
    fig = go.Figure(data=curve_traces(curves, theme), layout=base_layout(theme, x_title='adjacent finite link index (cold → hot)', y_title='Δlog(T) between rungs'))
    if x_link.size == 0:
        annotate_empty(fig, theme, 'fewer than two finite rungs')
    return fig
