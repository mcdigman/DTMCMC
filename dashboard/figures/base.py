"""Shared plotly styling helpers: theme application and series coloring.

This is the only place figure code touches Theme internals, so swapping
the plotting backend means reimplementing this module plus the small
per-figure factories — the diagnostics layer is untouched.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.colors as pcolors
import plotly.graph_objects as go

if TYPE_CHECKING:
    from dashboard.core.diagnostics import Curve
    from dashboard.themes import Theme


def base_layout(theme: Theme, *, x_title: str = '', y_title: str = '', log_x: bool = False, log_y: bool = False) -> go.Layout:
    """Build the standard themed layout: recessive hairline grid, thin chrome."""
    axis_common: dict[str, Any] = {
        'gridcolor': theme.gridline,
        'linecolor': theme.baseline,
        'zerolinecolor': theme.gridline,
        'tickcolor': theme.baseline,
        'tickfont': {'color': theme.ink_muted, 'size': 11},
        'title': {'font': {'color': theme.ink_secondary, 'size': 12}},
        'gridwidth': 1,
        'linewidth': 1,
    }
    return go.Layout(
        paper_bgcolor=theme.surface,
        plot_bgcolor=theme.surface,
        font={'family': theme.font_family, 'color': theme.ink_primary, 'size': 12},
        margin={'l': 60, 'r': 16, 't': 36, 'b': 44},
        xaxis={**axis_common, 'title': {**axis_common['title'], 'text': x_title}, 'type': 'log' if log_x else 'linear'},
        yaxis={**axis_common, 'title': {**axis_common['title'], 'text': y_title}, 'type': 'log' if log_y else 'linear'},
        legend={'font': {'color': theme.ink_secondary, 'size': 11}, 'bgcolor': 'rgba(0,0,0,0)'},
        hoverlabel={'font': {'family': theme.font_family, 'size': 12}},
        colorway=list(theme.categorical),
    )


def ordinal_colors(theme: Theme, n_steps: int) -> list[str]:
    """Sample n readable steps from the theme's ordinal single-hue ramp."""
    ramp = list(theme.ordinal_ramp)
    if n_steps <= 0:
        return []
    if n_steps == 1:
        return [ramp[len(ramp) // 2]]
    positions = np.linspace(0., 1., n_steps)
    return [str(pcolors.sample_colorscale(ramp, float(pos), colortype='rgb')[0]) for pos in positions]


def sequential_colorscale(theme: Theme) -> list[tuple[float, str]]:
    """The theme's sequential ramp as a plotly colorscale (low recedes)."""
    ramp = list(theme.sequential_ramp)
    return [(idx / (len(ramp) - 1), color) for idx, color in enumerate(ramp)]


def curve_traces(curves: list[Curve], theme: Theme, *, hover_suffix: str = '') -> list[go.Scatter]:
    """Render diagnostic Curves as themed line traces.

    History curves ('applied'/'held') walk the ordinal ramp in order (held
    updates dashed — a candidate that was not adopted), the 'current'
    segment gets the highlight color and a heavier line, and 'reference'
    curves are dotted muted-ink guides.
    """
    history = [curve for curve in curves if curve.emphasis in ('applied', 'held')]
    ramp = ordinal_colors(theme, len(history))
    history_color = dict(zip((id(curve) for curve in history), ramp, strict=True))

    traces: list[go.Scatter] = []
    for curve in curves:
        if curve.emphasis == 'current':
            color, dash, width = theme.highlight, 'solid', 2.5
        elif curve.emphasis == 'reference':
            color, dash, width = theme.ink_muted, 'dot', 2.
        else:
            color = history_color[id(curve)]
            dash = 'solid' if curve.emphasis == 'applied' else 'dash'
            width = 1.6
        traces.append(go.Scatter(
            x=curve.x, y=curve.y, name=curve.label, mode='lines',
            line={'color': color, 'dash': dash, 'width': width},
            hovertemplate=f'%{{x:.6g}}, %{{y:.6g}}{hover_suffix}<extra>{curve.label}</extra>',
        ))
    return traces


def annotate_empty(fig: go.Figure, theme: Theme, message: str) -> go.Figure:
    """Put a centered explanatory note on a figure with no data."""
    fig.add_annotation(
        text=message, showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5,
        font={'color': theme.ink_muted, 'size': 13},
    )
    return fig
