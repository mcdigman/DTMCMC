"""The Dash application: header, controls, tabbed figure grid, live polling.

The app is a thin shell over dashboard.core and dashboard.figures: one
interval callback polls the artifact watcher and bumps a token store only
when the file actually changed; one render callback rebuilds the active
tab's figures from the in-memory snapshot whenever the token, tab, theme,
or a control changes. All figure payloads travel to the browser as plotly
JSON — the artifact itself never leaves the server, so remote viewing
(e.g. an ssh tunnel to a cluster head node) transmits only plot data.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from dash import Dash, Input, Output, State, dcc, html, no_update

if TYPE_CHECKING:
    from pathlib import Path

from dashboard.core.diagnostics import header_items, store_column_label
from dashboard.core.reader import ArtifactWatcher, RunSnapshot, list_artifacts
from dashboard.figures.options import ViewOptions
from dashboard.figures.registry import LAYOUTS, PLOTS, build_figure
from dashboard.themes import THEMES, Theme, get_theme

# watchers persist across callbacks (one per artifact file per process)
_WATCHERS: dict[tuple[str, bool], ArtifactWatcher] = {}

GRAPH_CONFIG: Any = {'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']}


@dataclass
class DashboardConfig:
    """Server-level configuration (everything the CLI can set)."""

    artifact: Path
    host: str = '127.0.0.1'
    port: int = 8050
    poll_seconds: float = 5.
    theme: str = 'light'
    layout: str = 'default'
    stale_after_seconds: float = 600.
    load_store: bool = True
    debug: bool = False


def _watcher(path: str, load_store: bool) -> ArtifactWatcher:
    """Get (or create) the persistent watcher for one artifact file."""
    key = (path, load_store)
    if key not in _WATCHERS:
        _WATCHERS[key] = ArtifactWatcher(path, load_store=load_store)
    return _WATCHERS[key]


def _served_artifacts(config: DashboardConfig) -> list[str]:
    """The artifact paths this server is willing to open."""
    return [str(path) for path in list_artifacts(config.artifact)]


def _allowed_artifact(config: DashboardConfig, value: str | None, served: list[str] | None = None) -> str | None:
    """Validate a client-supplied artifact selection against the served set.

    Callback inputs are attacker-controllable in the remote-viewing model:
    only paths the server itself enumerated under its configured root may
    reach the HDF5 reader, so a crafted request cannot disclose an
    arbitrary server-side .h5 file.
    """
    served_set = set(_served_artifacts(config) if served is None else served)
    if value and value in served_set:
        return value
    return None


def _theme_css_block(theme: Theme) -> str:
    """CSS custom properties for one theme, keyed by data-theme."""
    return (
        f'[data-theme="{theme.name}"] {{'
        f'--page:{theme.page};--surface:{theme.surface};--ink:{theme.ink_primary};'
        f'--ink-secondary:{theme.ink_secondary};--ink-muted:{theme.ink_muted};'
        f'--gridline:{theme.gridline};--border:{theme.border};'
        f'--good:{theme.status_good};--warning:{theme.status_warning};'
        f'--serious:{theme.status_serious};--critical:{theme.status_critical};'
        f'--accent:{theme.categorical[0]};'
        f'}}'
    )


def _index_css() -> str:
    """Page CSS generated from the Theme definitions (single source of truth)."""
    theme_blocks = ''.join(_theme_css_block(theme) for theme in THEMES.values())
    return theme_blocks + """
    body { margin: 0; background: var(--page); color: var(--ink);
           font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
    .dashboard-root { min-height: 100vh; background: var(--page); }
    .header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 18px;
              padding: 10px 18px 6px; border-bottom: 1px solid var(--border); }
    .header h1 { font-size: 17px; margin: 0 6px 0 0; color: var(--ink); }
    .header-item { font-size: 12px; color: var(--ink-muted); white-space: nowrap; }
    .header-item b { color: var(--ink-secondary); font-weight: 600; }
    .status-chip { font-size: 12px; font-weight: 600; padding: 2px 10px; border-radius: 10px;
                   border: 1px solid var(--border); color: var(--ink); }
    .status-live { border-color: var(--good); color: var(--good); }
    .status-finalized { border-color: var(--accent); color: var(--accent); }
    .status-stale { border-color: var(--warning); color: var(--warning); }
    .status-error { border-color: var(--critical); color: var(--critical); }
    .controls { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: flex-end;
                padding: 8px 18px; border-bottom: 1px solid var(--border); }
    .control { display: flex; flex-direction: column; gap: 2px; min-width: 90px; }
    .control > label { font-size: 10px; color: var(--ink-muted); text-transform: uppercase;
                       letter-spacing: 0.04em; }
    .control input { background: var(--surface); color: var(--ink); border: 1px solid var(--border);
                     border-radius: 4px; padding: 3px 6px; font-size: 12px; width: 80px; }
    .control-wide { min-width: 220px; }
    .control .dash-dropdown { font-size: 12px; }
    .plot-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
                 gap: 14px; padding: 14px 18px; }
    .plot-card { background: var(--surface); border: 1px solid var(--border);
                 border-radius: 8px; padding: 6px 6px 2px; min-width: 0; }
    .plot-card.wide { grid-column: 1 / -1; }
    .plot-title { font-size: 13px; font-weight: 600; color: var(--ink); margin: 4px 8px 0; }
    .plot-desc { font-size: 11px; color: var(--ink-muted); margin: 1px 8px 5px; }
    .tab-bar .tab { background: var(--page) !important; color: var(--ink-secondary) !important;
                    border-color: var(--border) !important; font-size: 13px; padding: 8px 14px !important; }
    .tab-bar .tab--selected { background: var(--surface) !important; color: var(--ink) !important;
                              border-top: 2px solid var(--accent) !important; }
    .poll-note { font-size: 11px; color: var(--ink-muted); padding: 2px 18px 10px; }
    /* best-effort react-select theming for dcc.Dropdown in dark mode */
    .Select-control, .Select-menu-outer, .Select-value, .Select-input > input {
        background: var(--surface) !important; color: var(--ink) !important;
        border-color: var(--border) !important; font-size: 12px; }
    .Select-value-label, .Select-placeholder { color: var(--ink) !important; }
    .VirtualizedSelectOption { background: var(--surface); color: var(--ink); font-size: 12px; }
    .VirtualizedSelectFocusedOption { background: var(--page); }
    """


def _control(label: str, component: Any, *, wide: bool = False) -> html.Div:
    """One labeled control cell for the controls row."""
    return html.Div([html.Label(label), component], className='control control-wide' if wide else 'control')


def _controls_row(config: DashboardConfig, artifact_options: list[str]) -> html.Div:
    """The single filter row every tab shares."""
    return html.Div([
        _control('artifact', dcc.Dropdown(
            id='artifact-select', options=artifact_options, value=artifact_options[0] if artifact_options else None,
            clearable=False, className='dash-dropdown'), wide=True),
        _control('theme', dcc.RadioItems(
            id='theme-select', options=[{'label': name, 'value': name} for name in THEMES],
            value=config.theme if config.theme in THEMES else 'light',
            inline=True, style={'fontSize': '12px'})),
        _control('burn-in blocks', dcc.Input(id='burnin-blocks', type='number', value=1, min=0, step=1, debounce=True)),
        _control('rate window', dcc.Dropdown(
            id='rate-window',
            options=[
                # 'current ladder' is the default: every count sits on the
                # ladder now in effect, so adaptive runs stay uncluttered;
                # 'whole run' bins each slice at its own segment's
                # temperatures (union axis on adaptive runs)
                {'label': 'current ladder', 'value': 'segment'},
                {'label': 'whole run', 'value': 'total'},
                {'label': 'since last archive', 'value': 'latest'},
            ],
            value='segment', clearable=False, className='dash-dropdown')),
        _control('history stride', dcc.Input(id='segment-stride', type='number', value=1, min=1, step=1, debounce=True)),
        _control('chain', dcc.Dropdown(id='chain-select', options=[0], value=0, clearable=False, className='dash-dropdown')),
        _control('chains (ACF/trace)', dcc.Dropdown(id='chains-select', options=[0], value=[0], multi=True, className='dash-dropdown'), wide=True),
        _control('dims (corner/ACF)', dcc.Dropdown(id='dims-select', options=[0, 1, 2], value=[0, 1, 2], multi=True, className='dash-dropdown'), wide=True),
        _control('ESD normalization', dcc.RadioItems(
            id='esd-normalization', options=[{'label': 'per proposal', 'value': 'proposal'}, {'label': 'per accepted', 'value': 'accepted'}],
            value='proposal', inline=True, style={'fontSize': '12px'})),
        _control('max lag (rows)', dcc.Input(id='max-lag', type='number', value=256, min=8, step=8, debounce=True)),
    ], className='controls')


def _status_chip(snapshot: RunSnapshot | None, error: str, stale_after_seconds: float) -> html.Span:
    """Run-state chip: live / finalized / stale / error, icon + label."""
    if error and snapshot is None:
        return html.Span(f'✖ {error}', className='status-chip status-error')
    if snapshot is None:
        return html.Span('… waiting for artifact', className='status-chip status-stale')
    flush_age = ''
    try:
        flushed = datetime.fromisoformat(str(snapshot.attrs.get('flush_time_utc', '')))
        age_seconds = max(0., (datetime.now(tz=UTC) - flushed).total_seconds())
        flush_age = f' ({age_seconds:.0f}s ago)'
    except ValueError:
        age_seconds = 0.
    # the artifact's finalized flag marks any major-report flush; only a
    # run that reached its requested length is displayed as finalized
    if snapshot.run_complete:
        return html.Span('■ finalized' + flush_age, className='status-chip status-finalized')
    if age_seconds > stale_after_seconds:
        return html.Span('⚠ stale' + flush_age, className='status-chip status-stale')
    return html.Span('● live' + flush_age, className='status-chip status-live')


def _header_children(snapshot: RunSnapshot | None, error: str, stale_after_seconds: float) -> list[Any]:
    """Header contents: run name, status chip, config key-values."""
    title = snapshot.name if snapshot is not None else 'DTMCMC dashboard'
    children: list[Any] = [html.H1(title), _status_chip(snapshot, error, stale_after_seconds)]
    if snapshot is not None:
        children.extend(
            html.Span([html.B(f'{label}: '), value], className='header-item')
            for label, value in header_items(snapshot)
        )
    return children


def _tab_children(snapshot: RunSnapshot | None, plot_ids: tuple[str, ...], opts: ViewOptions, theme: Theme) -> list[html.Div]:
    """Figure cards for one tab."""
    if snapshot is None:
        return [html.Div('no artifact loaded yet', className='plot-desc')]
    cards: list[html.Div] = []
    for plot_id in plot_ids:
        spec = PLOTS[plot_id]
        figure = build_figure(plot_id, snapshot, opts, theme)
        # preserve zoom/pan across live updates of the same plot
        figure.update_layout(uirevision=plot_id)
        cards.append(html.Div([
            html.Div(spec.title, className='plot-title'),
            html.Div(spec.description, className='plot-desc'),
            dcc.Graph(figure=figure, config=GRAPH_CONFIG, style={'height': f'{getattr(figure.layout, "height", None) or 380}px'}),
        ], className='plot-card wide' if spec.wide else 'plot-card'))
    return cards


def create_app(config: DashboardConfig) -> Dash:
    """Build the Dash application for one artifact file or directory."""
    artifact_options = [str(path) for path in list_artifacts(config.artifact)]
    tabs_layout = LAYOUTS.get(config.layout, LAYOUTS['default'])

    app = Dash(__name__, title='DTMCMC dashboard')
    app.index_string = app.index_string.replace('</head>', f'<style>{_index_css()}</style></head>')

    app.layout = html.Div([
        dcc.Interval(id='poll', interval=int(config.poll_seconds * 1000)),
        dcc.Store(id='snapshot-token'),
        html.Div(id='header', className='header'),
        _controls_row(config, artifact_options),
        dcc.Tabs(
            id='tab-select', value=next(iter(tabs_layout)), className='tab-bar',
            children=[dcc.Tab(label=tab_name, value=tab_name, className='tab', selected_className='tab--selected') for tab_name in tabs_layout],
        ),
        html.Div(id='tab-content', className='plot-grid'),
        html.Div(f'polling every {config.poll_seconds:g}s — artifacts are read server-side; only plot data reaches the browser', className='poll-note'),
    ], id='root', className='dashboard-root', **cast('dict[str, Any]', {'data-theme': config.theme if config.theme in THEMES else 'light'}))

    @app.callback(
        Output('snapshot-token', 'data'),
        Output('header', 'children'),
        Output('chain-select', 'options'),
        Output('chains-select', 'options'),
        Output('dims-select', 'options'),
        Output('artifact-select', 'options'),
        Input('poll', 'n_intervals'),
        Input('artifact-select', 'value'),
        State('snapshot-token', 'data'),
        State('artifact-select', 'options'),
    )
    def poll_artifact(_n_intervals: int, artifact_value: str | None, previous_token: str | None, previous_options: list[str] | None):
        """Re-read the artifact only when its flush changed; bump the token."""
        # pick up run artifacts created after server start
        current_options = _served_artifacts(config)
        options_out = current_options if current_options != (previous_options or []) else no_update
        artifact_value = _allowed_artifact(config, artifact_value, current_options)
        if not artifact_value:
            return no_update, _header_children(None, 'no artifact selected (or selection not under the served root)', config.stale_after_seconds), no_update, no_update, no_update, options_out
        watcher = _watcher(artifact_value, config.load_store)
        snapshot = watcher.poll()
        token = f'{artifact_value}:{snapshot.stat_token if snapshot is not None else "none"}'
        header = _header_children(snapshot, watcher.last_error, config.stale_after_seconds)
        if token == previous_token or snapshot is None:
            return (no_update if token == previous_token else token), header, no_update, no_update, no_update, options_out
        # selector values are store columns; label each with the chain it
        # currently records (readout chains first, then arg_record extras)
        chain_options = [
            {'label': store_column_label(snapshot, column), 'value': column}
            for column in range(max(snapshot.n_recorded, 1))
        ]
        dim_options = list(range(max(snapshot.n_par, 1)))
        return token, header, chain_options, chain_options, dim_options, options_out

    @app.callback(
        Output('tab-content', 'children'),
        Output('root', 'data-theme'),
        Input('snapshot-token', 'data'),
        Input('tab-select', 'value'),
        Input('theme-select', 'value'),
        Input('burnin-blocks', 'value'),
        Input('rate-window', 'value'),
        Input('segment-stride', 'value'),
        Input('chain-select', 'value'),
        Input('chains-select', 'value'),
        Input('dims-select', 'value'),
        Input('esd-normalization', 'value'),
        Input('max-lag', 'value'),
        State('artifact-select', 'value'),
    )
    def render_tab(
        _token: str | None, tab_name: str, theme_name: str,
        burnin_blocks: int | None, rate_window: str, segment_stride: int | None,
        chain: int | None, chains: list[int] | None, dims: list[int] | None,
        esd_normalization: str, max_lag: int | None, artifact_value: str | None,
    ):
        """Rebuild the active tab's figures from the in-memory snapshot."""
        artifact_value = _allowed_artifact(config, artifact_value)
        snapshot = _watcher(artifact_value, config.load_store).snapshot if artifact_value else None
        theme = get_theme(theme_name)
        chains_use = [int(value) for value in chains] if chains else [0]
        opts = ViewOptions(
            burnin_blocks=int(burnin_blocks or 0),
            window=rate_window or 'segment',
            segment_stride=max(int(segment_stride or 1), 1),
            accepted_only=esd_normalization == 'accepted',
            chain=int(chain or 0),
            chains=chains_use,
            dims=[int(value) for value in dims] if dims else [0],
            max_lag=int(max_lag or 256),
            cross_a=chains_use[0],
            cross_b=chains_use[-1],
        )
        return _tab_children(snapshot, tabs_layout.get(tab_name, ()), opts, theme), theme.name

    return app


def run_dashboard(config: DashboardConfig) -> None:
    """Create the app and serve it (blocking)."""
    app = create_app(config)
    app.run(host=config.host, port=config.port, debug=config.debug)
