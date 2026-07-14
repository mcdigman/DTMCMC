"""Plot registry: every available figure and the default tab layout.

Front-ends render tabs by iterating LAYOUTS entries and calling
build_figure, so rearranging the dashboard (or defining an alternate
layout for a different audience) means editing data here, not UI code.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dashboard.figures import mixing, posterior, proposals, thermo

if TYPE_CHECKING:
    from collections.abc import Callable

    import plotly.graph_objects as go

    from dashboard.core.reader import RunSnapshot
    from dashboard.figures.options import ViewOptions
    from dashboard.themes import Theme


@dataclass(frozen=True)
class PlotSpec:
    """One registered figure: identity, prose, builder, and layout hint."""

    plot_id: str
    title: str
    description: str
    builder: Callable[[RunSnapshot, ViewOptions, Theme], go.Figure]
    wide: bool = False


_SPECS: tuple[PlotSpec, ...] = (
    PlotSpec(
        'mean_logl',
        'Mean logL vs temperature',
        'Segment mean E[logL](T), one curve per ladder segment.',
        thermo.fig_mean_logl,
    ),
    PlotSpec(
        'heat_capacity',
        'Heat capacity vs temperature',
        'Segment-inferred C(T) = Var(logL)/T² on log-log axes.',
        thermo.fig_heat_capacity,
    ),
    PlotSpec(
        'entropy',
        'Entropy vs temperature',
        'Integrated heat capacity S(T), the entropy-ladder spacing measure.',
        thermo.fig_entropy,
    ),
    PlotSpec(
        'ladder_spacing',
        'Ladder spacing evolution',
        'Adjacent Δlog(T) per rebuild evaluation; convergence check for adaptive ladders.',
        thermo.fig_ladder_spacing,
    ),
    PlotSpec(
        'acceptance',
        'Proposal acceptance vs temperature',
        'Acceptance rate of each tracked proposal against temperature.',
        proposals.fig_acceptance_vs_temperature,
    ),
    PlotSpec(
        'esd',
        'Jump distance vs temperature',
        'Expected squared displacement per proposal (or per accepted jump).',
        proposals.fig_esd_vs_temperature,
    ),
    PlotSpec(
        'exchange',
        'Exchange acceptance vs temperature',
        'Nearest-neighbor (and all-target) exchange acceptance; flat = constant-acceptance ladder working.',
        proposals.fig_exchange_vs_temperature,
    ),
    PlotSpec(
        'exchange_history',
        'Exchange acceptance history',
        'Nearest-neighbor exchange rate per slot per archive window.',
        proposals.fig_exchange_history,
        wide=True,
    ),
    PlotSpec(
        'logl_blocks',
        'Block mean logL history',
        'Per-block E[logL] per slot vs block number; the burn-in trim diagnostic.',
        mixing.fig_logl_block_history,
        wide=True,
    ),
    PlotSpec(
        'flow',
        'Walker up-flow fraction',
        'f(T) against rung index with the linear constant-flow ideal.',
        mixing.fig_flow_fraction,
    ),
    PlotSpec(
        'flow_history', 'Up-flow history', 'Per-block f(T) heatmap over the run.', mixing.fig_flow_history, wide=True
    ),
    PlotSpec(
        'round_trips',
        'Round-trip traffic',
        'Cumulative cold/hot arrivals vs iteration; straight lines = steady mixing.',
        mixing.fig_round_trips,
    ),
    PlotSpec(
        'walker_arrivals',
        'Arrivals per walker',
        'Cold arrivals per walker id over the run.',
        mixing.fig_walker_arrivals,
    ),
    PlotSpec(
        'de_eigvals',
        'DE buffer spectrum',
        'Latest DE-buffer difference eigenspectrum per slot; trailing zeros = rank collapse.',
        mixing.fig_de_eigvals,
    ),
    PlotSpec(
        'de_rank',
        'DE effective rank',
        'Participation-ratio rank of the DE difference span vs temperature per checkpoint.',
        mixing.fig_de_effective_rank,
    ),
    PlotSpec(
        'corner',
        'Corner (dimension subset)',
        'Histogram/contour corner plot of selected dimensions of one recorded chain.',
        posterior.fig_corner,
        wide=True,
    ),
    PlotSpec(
        'trace',
        'Parameter traces',
        'Stored parameter values vs iteration for one recorded chain.',
        posterior.fig_parameter_trace,
    ),
    PlotSpec(
        'logl_trace', 'logL traces', 'Stored logL vs iteration for selected recorded chains.', posterior.fig_logl_trace
    ),
    PlotSpec(
        'logl_acf',
        'logL autocorrelation',
        'Normalized logL ACF per selected recorded chain with integrated-time estimates.',
        posterior.fig_logl_acf,
    ),
    PlotSpec(
        'param_acf',
        'Parameter autocorrelation',
        'Normalized parameter ACF for one recorded chain.',
        posterior.fig_parameter_acf,
    ),
    PlotSpec(
        'cross_corr',
        'logL cross-correlation',
        'Cross-correlation of stored logL between two recorded chains.',
        posterior.fig_logl_cross_correlation,
    ),
)

PLOTS: dict[str, PlotSpec] = {spec.plot_id: spec for spec in _SPECS}

# default tab layout: tab title -> plot ids, rendered in order. Alternate
# UI configurations are additional entries here (selected via DashboardConfig).
DEFAULT_TABS: dict[str, tuple[str, ...]] = {
    'Thermodynamics': ('mean_logl', 'heat_capacity', 'entropy', 'ladder_spacing'),
    'Proposals': ('acceptance', 'esd', 'exchange', 'exchange_history'),
    'Mixing': ('logl_blocks', 'flow', 'round_trips', 'walker_arrivals', 'flow_history'),
    'DE buffer': ('de_eigvals', 'de_rank'),
    'Posterior': ('corner', 'trace', 'logl_trace', 'logl_acf', 'param_acf', 'cross_corr'),
}

LAYOUTS: dict[str, dict[str, tuple[str, ...]]] = {
    'default': DEFAULT_TABS,
    # single-tab overview for small runs / quick checks
    'overview': {
        'Overview': ('mean_logl', 'heat_capacity', 'acceptance', 'exchange', 'logl_blocks', 'round_trips'),
    },
}


def build_figure(plot_id: str, snapshot: RunSnapshot, opts: ViewOptions, theme: Theme) -> go.Figure:
    """Build one registered figure by id."""
    return PLOTS[plot_id].builder(snapshot, opts, theme)
