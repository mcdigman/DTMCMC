"""Plot-ready diagnostics computed from RunSnapshot arrays.

Every function here is a pure transformation: numpy arrays in, small
result dataclasses out. Nothing imports a plotting or UI library, so the
same diagnostics feed the Dash front-end, a future Panel front-end, or a
static report generator.

Thermodynamic conventions follow the engine: heat capacity is
C(T) = Var(logL)/T^2, entropy is the integrated heat capacity over beta
(DTMCMC.temperature_ladder_helpers.get_spacing_integrated with p=1, q=1),
and segment moments mirror AdaptiveLadderController._absorb_segment_stats
(block means of E[logL] and E[logL^2] over the segment).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from DTMCMC.temperature_ladder_helpers import Ts_to_betas, get_spacing_integrated, standardize_input_vars

if TYPE_CHECKING:
    from dashboard.core.reader import RunSnapshot


@dataclass(frozen=True)
class Curve:
    """One labeled line: x against y with an emphasis class for styling."""

    label: str
    x: np.ndarray
    y: np.ndarray
    emphasis: str  # 'current' | 'applied' | 'held' | 'reference'


@dataclass(frozen=True)
class LadderSegment:
    """A run interval [start_block, stop_block) sampled on one fixed ladder."""

    start_block: int
    stop_block: int
    Ts: np.ndarray
    applied: bool
    is_current: bool

    @property
    def label(self) -> str:
        """Short human-readable segment tag for legends."""
        if self.is_current:
            return f'blocks {self.start_block}-{self.stop_block} (current)'
        return f'blocks {self.start_block}-{self.stop_block} ({"applied" if self.applied else "held"})'


@dataclass(frozen=True)
class RateTable:
    """Per-temperature, per-jump-type rates with the trial counts behind them."""

    Ts: np.ndarray
    values: np.ndarray
    trials: np.ndarray
    labels: list[str]


@dataclass(frozen=True)
class ExchangeRates:
    """Per-temperature-slot exchange acceptance for one counting window."""

    Ts: np.ndarray
    nn_rate: np.ndarray
    nn_trials: np.ndarray
    all_rate: np.ndarray
    all_trials: np.ndarray
    overall_nn_rate: float


@dataclass(frozen=True)
class ExchangeHistory:
    """Nearest-neighbor exchange acceptance per archive window over time."""

    itrns: np.ndarray
    nn_rates: np.ndarray


@dataclass(frozen=True)
class BlockHistory:
    """A per-block, per-chain quantity against block number."""

    blocks: np.ndarray
    values: np.ndarray
    Ts: np.ndarray


@dataclass(frozen=True)
class DESpectrum:
    """DE-buffer difference eigenspectra at each checkpoint."""

    itrns: np.ndarray
    eigvals: np.ndarray
    eff_rank: np.ndarray


@dataclass(frozen=True)
class FlowSummary:
    """Walker up-flow fraction f(T) per temperature slot."""

    Ts: np.ndarray
    f_latest: np.ndarray
    f_per_block: np.ndarray
    blocks: np.ndarray
    f_ideal: np.ndarray


@dataclass(frozen=True)
class RoundTripSummary:
    """Round-trip traffic: cumulative arrival curves and per-walker counts."""

    itrns_cold: np.ndarray
    cumulative_cold: np.ndarray
    itrns_hot: np.ndarray
    cumulative_hot: np.ndarray
    walker_ids: np.ndarray
    cold_arrivals_per_walker: np.ndarray
    segment_itrns: np.ndarray
    n_cycles_current_segment: np.ndarray


@dataclass(frozen=True)
class AcfResult:
    """Normalized autocorrelation with an integrated-time estimate."""

    label: str
    lags: np.ndarray
    rho: np.ndarray
    tau_int: float


def ladder_segments(snapshot: RunSnapshot) -> list[LadderSegment]:
    """Split the completed blocks into intervals sampled on a fixed ladder.

    Fixed-ladder runs give a single 'current' segment. Adaptive runs give
    one segment per rebuild evaluation (using ladder/initial_Ts as the
    starting ladder and applying history rows in order, mirroring the
    controller), plus a trailing 'current' segment after the last
    evaluation when blocks continued past it.
    """
    n_blocks = snapshot.n_blocks
    history = snapshot.history
    if history is None or history.block_index.size == 0:
        return [LadderSegment(0, n_blocks, np.asarray(snapshot.Ts), applied=True, is_current=True)]

    segments: list[LadderSegment] = []
    current_Ts = np.asarray(snapshot.initial_Ts)
    start_block = 0
    for idx, stop_block in enumerate(history.block_index):
        stop = min(int(stop_block), n_blocks)
        if stop > start_block:
            segments.append(LadderSegment(start_block, stop, current_Ts, applied=bool(history.applied[idx]), is_current=False))
        if history.applied[idx]:
            current_Ts = np.asarray(history.Ts[idx])
        start_block = stop
    if start_block < n_blocks:
        segments.append(LadderSegment(start_block, n_blocks, current_Ts, applied=True, is_current=True))
    return segments


def segment_moments(snapshot: RunSnapshot, segment: LadderSegment) -> tuple[np.ndarray, np.ndarray]:
    """Segment-mean E[logL] and Var(logL) per chain over the segment's blocks.

    Matches AdaptiveLadderController._absorb_segment_stats: the variance is
    E[logL^2] - E[logL]^2 with block means averaged over the whole segment.
    """
    e1_blocks = snapshot.logL_means[segment.start_block:segment.stop_block]
    e2_blocks = snapshot.logL2_means[segment.start_block:segment.stop_block]
    if e1_blocks.shape[0] == 0:
        return np.zeros(snapshot.n_chain), np.zeros(snapshot.n_chain)
    seg_means = e1_blocks.mean(axis=0)
    seg_vars = np.maximum(e2_blocks.mean(axis=0) - seg_means**2, 0.)
    return seg_means, seg_vars


def _finite_sorted(Ts: np.ndarray, *values: np.ndarray) -> tuple[np.ndarray, ...]:
    """Restrict to finite positive temperatures and finite values, sorted by T."""
    Ts_arr = np.asarray(Ts)
    finite = np.isfinite(Ts_arr) & (Ts_arr > 0.)
    for value in values:
        finite &= np.isfinite(np.asarray(value))
    order = np.argsort(Ts_arr[finite])
    return (Ts_arr[finite][order], *tuple(np.asarray(value)[finite][order] for value in values))


def _segment_emphasis(segment: LadderSegment) -> str:
    """Map a segment onto a Curve emphasis class."""
    if segment.is_current:
        return 'current'
    return 'applied' if segment.applied else 'held'


def mean_logl_curves(snapshot: RunSnapshot, segments: list[LadderSegment] | None = None) -> list[Curve]:
    """E[logL] against temperature, one curve per ladder segment."""
    segments = ladder_segments(snapshot) if segments is None else segments
    curves: list[Curve] = []
    for segment in segments:
        seg_means, _seg_vars = segment_moments(snapshot, segment)
        Ts_use, means_use = _finite_sorted(segment.Ts, seg_means)
        if Ts_use.size:
            curves.append(Curve(segment.label, Ts_use, means_use, _segment_emphasis(segment)))
    return curves


def heat_capacity_curves(snapshot: RunSnapshot, segments: list[LadderSegment] | None = None) -> list[Curve]:
    """C(T) = Var(logL)/T^2 against temperature, one curve per ladder segment."""
    segments = ladder_segments(snapshot) if segments is None else segments
    curves: list[Curve] = []
    for segment in segments:
        _seg_means, seg_vars = segment_moments(snapshot, segment)
        Ts_use, vars_use = _finite_sorted(segment.Ts, seg_vars)
        positive = vars_use > 0.
        if np.any(positive):
            curves.append(Curve(segment.label, Ts_use[positive], vars_use[positive] / Ts_use[positive]**2, _segment_emphasis(segment)))
    return curves


def entropy_curves(snapshot: RunSnapshot, segments: list[LadderSegment] | None = None) -> list[Curve]:
    """Integrated heat capacity S(T) against temperature per ladder segment.

    Uses the engine's own entropy-ladder measure (get_spacing_integrated
    with p=1, q=1) on the segment-inferred Var(logL), so the curve is the
    quantity the entropy ladder equalizes between rungs, referenced to the
    hottest finite rung.
    """
    segments = ladder_segments(snapshot) if segments is None else segments
    curves: list[Curve] = []
    for segment in segments:
        _seg_means, seg_vars = segment_moments(snapshot, segment)
        Ts_use, vars_use = _finite_sorted(segment.Ts, seg_vars)
        if Ts_use.size < 2:
            continue
        betas_use, vars_std = standardize_input_vars(Ts_to_betas(Ts_use), vars_use)
        entropy = get_spacing_integrated(vars_std, betas_use, False)
        with np.errstate(divide='ignore'):
            Ts_std = np.where(betas_use > 0., 1. / betas_use, np.inf)
        Ts_plot, entropy_plot = _finite_sorted(Ts_std, entropy)
        if Ts_plot.size:
            curves.append(Curve(segment.label, Ts_plot, entropy_plot, _segment_emphasis(segment)))
    return curves


def window_counts(record: np.ndarray, archive: np.ndarray, window: str | int) -> np.ndarray:
    """Resolve one counting window from a cumulative record and its archive.

    'total' is the cumulative record; 'latest' is the record minus the last
    archived snapshot (counts since the most recent archive point); an
    integer index selects the window ending at archive entry ``window``
    (differenced against the previous entry).
    """
    record_arr = np.asarray(record)
    archive_arr = np.asarray(archive)
    if window == 'total':
        return record_arr
    if window == 'latest':
        if archive_arr.shape[0] == 0:
            return record_arr
        return record_arr - archive_arr[-1]
    idx = int(window)
    if idx == 0:
        return archive_arr[0]
    return archive_arr[idx] - archive_arr[idx - 1]


def _rates_by_unique_temperature(Ts: np.ndarray, yes: np.ndarray, no: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine per-chain yes/no counts over identical temperatures."""
    Ts_unique = np.unique(np.asarray(Ts))
    n_jump = yes.shape[-1]
    yes_unique = np.zeros((Ts_unique.size, n_jump))
    no_unique = np.zeros((Ts_unique.size, n_jump))
    for itrt, T_loc in enumerate(Ts_unique):
        mask = np.asarray(Ts) == T_loc
        yes_unique[itrt] = yes[mask].sum(axis=0)
        no_unique[itrt] = no[mask].sum(axis=0)
    return Ts_unique, yes_unique, no_unique


def acceptance_by_temperature(snapshot: RunSnapshot, window: str | int = 'total') -> RateTable:
    """Per-proposal acceptance rate against temperature for one window.

    Jump types never tried in the window are dropped; entries with no
    trials at a particular temperature are NaN.
    """
    counts = window_counts(snapshot.accept_record, snapshot.accept_archive, window)
    Ts_unique, yes_unique, no_unique = _rates_by_unique_temperature(snapshot.Ts, counts[0], counts[1])
    trials = yes_unique + no_unique
    tried = np.any(trials > 0, axis=0)
    labels = [label for label, keep in zip(snapshot.jump_labels, tried, strict=True) if keep]
    with np.errstate(invalid='ignore', divide='ignore'):
        rates = np.where(trials[:, tried] > 0, yes_unique[:, tried] / trials[:, tried], np.nan)
    return RateTable(Ts_unique, rates, trials[:, tried], labels)


def esd_by_temperature(snapshot: RunSnapshot, window: str | int = 'total', *, accepted_only: bool = False) -> RateTable:
    """Mean squared jump displacement against temperature per jump type.

    Per-proposal (default) divides the all-proposal |delta|^2 sums by the
    number of proposals; accepted_only divides the accepted-only sums by
    the number of acceptances.
    """
    esd_sums = window_counts(snapshot.esd_record, snapshot.esd_archive, window)
    accept_counts = window_counts(snapshot.accept_record, snapshot.accept_archive, window)
    sums = esd_sums[1] if accepted_only else esd_sums[0]
    counts = accept_counts[0] if accepted_only else accept_counts[0] + accept_counts[1]

    Ts_unique = np.unique(np.asarray(snapshot.Ts))
    n_jump = sums.shape[-1]
    sums_unique = np.zeros((Ts_unique.size, n_jump))
    counts_unique = np.zeros((Ts_unique.size, n_jump))
    for itrt, T_loc in enumerate(Ts_unique):
        mask = np.asarray(snapshot.Ts) == T_loc
        sums_unique[itrt] = sums[mask].sum(axis=0)
        counts_unique[itrt] = counts[mask].sum(axis=0)
    tried = np.any(counts_unique > 0, axis=0)
    labels = [label for label, keep in zip(snapshot.jump_labels, tried, strict=True) if keep]
    with np.errstate(invalid='ignore', divide='ignore'):
        values = np.where(counts_unique[:, tried] > 0, sums_unique[:, tried] / counts_unique[:, tried], np.nan)
    return RateTable(Ts_unique, values, counts_unique[:, tried], labels)


def _nn_exchange_counts(snapshot: RunSnapshot, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbor and all-exchange yes/no counts per temperature slot.

    Mirrors TrackerManager.get_exchange_rate_summary: in the limited layout
    row [1] holds nearest-neighbor yes/no per slot and row [0] holds
    all-exchange yes/no; in the full pairwise layout nearest-neighbor
    counts are symmetrized off-diagonals and 'all' sums each slot's row
    and column.
    """
    if snapshot.track_full_exchanges:
        n_chain = snapshot.n_chain
        a_yes = counts[0]
        a_no = counts[1]
        upper = (np.arange(0, n_chain - 1), np.arange(1, n_chain))
        nn_yes = np.zeros(n_chain)
        nn_no = np.zeros(n_chain)
        nn_yes[:-1] += a_yes[upper]
        nn_yes[1:] += a_yes[upper]
        nn_no[:-1] += a_no[upper]
        nn_no[1:] += a_no[upper]
        all_yes = a_yes.sum(axis=0) + a_yes.sum(axis=1)
        all_no = a_no.sum(axis=0) + a_no.sum(axis=1)
        return nn_yes, nn_no, all_yes, all_no
    return counts[1, 0].astype(float), counts[1, 1].astype(float), counts[0, 0].astype(float), counts[0, 1].astype(float)


def exchange_rates(snapshot: RunSnapshot, window: str | int = 'total') -> ExchangeRates:
    """Exchange acceptance against temperature slot for one counting window."""
    counts = window_counts(snapshot.exchange_tracker, snapshot.exchange_archive, window)
    nn_yes, nn_no, all_yes, all_no = _nn_exchange_counts(snapshot, counts)
    nn_trials = nn_yes + nn_no
    all_trials = all_yes + all_no
    with np.errstate(invalid='ignore', divide='ignore'):
        nn_rate = np.where(nn_trials > 0, nn_yes / nn_trials, np.nan)
        all_rate = np.where(all_trials > 0, all_yes / all_trials, np.nan)
    overall = float(nn_yes.sum() / nn_trials.sum()) if nn_trials.sum() > 0 else float('nan')
    return ExchangeRates(np.asarray(snapshot.Ts), nn_rate, nn_trials, all_rate, all_trials, overall)


def exchange_history(snapshot: RunSnapshot) -> ExchangeHistory:
    """Nearest-neighbor exchange acceptance per archive window over the run.

    Window k covers archive entry k differenced against entry k-1, with a
    final window from the last archive entry to the live record; the
    associated iteration is the window's end.
    """
    n_archived = int(snapshot.exchange_archive.shape[0])
    windows: list[np.ndarray] = [window_counts(snapshot.exchange_tracker, snapshot.exchange_archive, idx) for idx in range(n_archived)]
    itrns = [int(itrn) for itrn in snapshot.itrn_archive[:n_archived]]
    if n_archived == 0 or snapshot.n_iterations > itrns[-1]:
        windows.append(window_counts(snapshot.exchange_tracker, snapshot.exchange_archive, 'latest'))
        itrns.append(snapshot.n_iterations)
    nn_rates = np.full((len(windows), snapshot.n_chain), np.nan)
    for idx, counts in enumerate(windows):
        nn_yes, nn_no, _all_yes, _all_no = _nn_exchange_counts(snapshot, counts)
        nn_trials = nn_yes + nn_no
        with np.errstate(invalid='ignore', divide='ignore'):
            nn_rates[idx] = np.where(nn_trials > 0, nn_yes / nn_trials, np.nan)
    return ExchangeHistory(np.asarray(itrns, dtype=np.int64), nn_rates)


def logl_block_history(snapshot: RunSnapshot, burnin_blocks: int = 0) -> BlockHistory:
    """Per-block mean logL for every chain against block number.

    Columns are temperature slots (labeled with the current ladder); the
    first burnin_blocks blocks are trimmed to keep the range readable.
    """
    burnin = max(0, min(int(burnin_blocks), snapshot.n_blocks))
    blocks = np.arange(burnin, snapshot.n_blocks)
    return BlockHistory(blocks, snapshot.logL_means[burnin:], np.asarray(snapshot.Ts))


def effective_rank(eigvals: np.ndarray) -> np.ndarray:
    """Participation-ratio effective rank along the last axis.

    (sum lambda)^2 / sum lambda^2, as in experiments.metrics.effective_rank
    (reimplemented so the dashboard core does not import engine metrics).
    """
    total = np.asarray(eigvals).sum(axis=-1)
    total_sq = (np.asarray(eigvals)**2).sum(axis=-1)
    out = np.zeros(np.shape(total))
    nonzero = total_sq > 0.
    out[nonzero] = total[nonzero]**2 / total_sq[nonzero]
    return out


def de_spectrum_summary(snapshot: RunSnapshot) -> DESpectrum:
    """DE-buffer difference eigenspectra and effective ranks per checkpoint."""
    eigvals = np.asarray(snapshot.de_spectrum_eigvals)
    return DESpectrum(np.asarray(snapshot.checkpoint_itrns, dtype=np.int64), eigvals, effective_rank(eigvals))


def flow_fraction(snapshot: RunSnapshot, trailing_blocks: int = 0) -> FlowSummary:
    """Walker up-flow fraction f(T) per temperature slot.

    f is the fraction of resident walkers whose last extreme visit was the
    cold end; the ideal constant-round-trip-flow profile decreases linearly
    in rung index from 1 at the cold end to 0 at the hot end. f_latest
    averages the trailing ``trailing_blocks`` blocks (all blocks when 0).
    """
    up = np.asarray(snapshot.flow_up, dtype=float)
    labeled = np.asarray(snapshot.flow_labeled, dtype=float)
    n_blocks = up.shape[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        f_per_block = np.where(labeled > 0, up / labeled, np.nan)
    start = max(0, n_blocks - int(trailing_blocks)) if trailing_blocks > 0 else 0
    up_window = up[start:].sum(axis=0)
    labeled_window = labeled[start:].sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        f_latest = np.where(labeled_window > 0, up_window / labeled_window, np.nan)
    n_chain = snapshot.n_chain
    f_ideal = np.linspace(1., 0., n_chain) if n_chain > 1 else np.ones(1)
    return FlowSummary(np.asarray(snapshot.Ts), f_latest, f_per_block, np.arange(n_blocks), f_ideal)


def round_trip_summary(snapshot: RunSnapshot) -> RoundTripSummary:
    """Round-trip traffic from the artifact's arrival event log.

    Cumulative cold-arrival (direction 0) and hot-arrival (direction 1)
    counts against iteration, cold arrivals per walker, and the current
    tracker segment's completed-cycle counts per walker.
    """
    events = np.asarray(snapshot.rt_events, dtype=np.int64)
    n_chain = snapshot.n_chain
    if events.shape[0] == 0:
        empty = np.zeros(0, dtype=np.int64)
        return RoundTripSummary(
            empty, empty.copy(), empty.copy(), empty.copy(),
            np.arange(n_chain), np.zeros(n_chain, dtype=np.int64),
            np.asarray(snapshot.rt_segment_itrns, dtype=np.int64),
            np.min(snapshot.cycle_tracker[2:4], axis=0),
        )
    order = np.argsort(events[:, 1], kind='stable')
    events = events[order]
    cold = events[events[:, 2] == 0]
    hot = events[events[:, 2] == 1]
    per_walker = np.bincount(cold[:, 0], minlength=n_chain)
    return RoundTripSummary(
        cold[:, 1], np.arange(1, cold.shape[0] + 1),
        hot[:, 1], np.arange(1, hot.shape[0] + 1),
        np.arange(n_chain), per_walker,
        np.asarray(snapshot.rt_segment_itrns, dtype=np.int64),
        np.min(snapshot.cycle_tracker[2:4], axis=0),
    )


def normalized_acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized autocorrelation of a 1-D series out to max_lag via FFT."""
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    n_samples = values.size
    if n_samples < 2:
        return np.ones(1)
    values = values - values.mean()
    n_fft = int(2**np.ceil(np.log2(2 * n_samples)))
    spectrum = np.fft.rfft(values, n=n_fft)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft)[:n_samples].real
    if acov[0] <= 0.:
        return np.ones(1)
    lags_keep = min(int(max_lag) + 1, n_samples)
    return acov[:lags_keep] / acov[0]


def integrated_autocorr_time(rho: np.ndarray, window_factor: float = 5.) -> float:
    """Sokal self-consistent-window integrated autocorrelation time.

    tau = 1 + 2 sum(rho[1:M]) with M the smallest lag where
    M >= window_factor * tau; falls back to the full window when the
    criterion never triggers (series too short for the correlation).
    """
    rho_arr = np.asarray(rho, dtype=float)
    tau_running = 1. + 2. * np.cumsum(rho_arr[1:])
    if tau_running.size == 0:
        return 1.
    window_lags = np.arange(1, tau_running.size + 1)
    satisfied = window_lags >= window_factor * tau_running
    if np.any(satisfied):
        return float(tau_running[np.argmax(satisfied)])
    return float(tau_running[-1])


def _store_rows(snapshot: RunSnapshot, burnin_rows: int) -> int:
    """First stored row to use after trimming burn-in."""
    return max(0, min(int(burnin_rows), int(snapshot.logLs.shape[0])))


def store_column_label(snapshot: RunSnapshot, column: int) -> str:
    """Readable identity of one store column via the current chain mapping.

    Store columns are the ladder's readout chains first, then arg_record
    extras; a column that is currently a readout chain is just its chain
    index, an extra is tagged with the temperature it currently records.
    """
    if not 0 <= column < snapshot.record_indices.size:
        return f'col {column}'
    chain = int(snapshot.record_indices[column])
    if column < snapshot.n_cold:
        return f'chain {chain}'
    T_chain = snapshot.Ts[chain] if 0 <= chain < snapshot.n_chain else float('nan')
    T_tag = f'T={T_chain:.4g}' if np.isfinite(T_chain) else 'T=inf'
    return f'chain {chain} ({T_tag})'


def logl_acf(snapshot: RunSnapshot, chains: list[int], max_lag: int, burnin_rows: int = 0) -> list[AcfResult]:
    """Autocorrelation of stored logL for the selected recorded chains.

    Lags are in stored rows; multiply by store_thin for iterations. Chains
    index the store columns (0..n_recorded-1: the ladder's readout chains
    first, then the arg_record extras in spec order).
    """
    start = _store_rows(snapshot, burnin_rows)
    results: list[AcfResult] = []
    for chain in chains:
        if not 0 <= chain < snapshot.logLs.shape[1]:
            continue
        rho = normalized_acf(snapshot.logLs[start:, chain], max_lag)
        results.append(AcfResult(f'logL {store_column_label(snapshot, chain)}', np.arange(rho.size), rho, integrated_autocorr_time(rho)))
    return results


def parameter_acf(snapshot: RunSnapshot, chain: int, dims: list[int], max_lag: int, burnin_rows: int = 0) -> list[AcfResult]:
    """Autocorrelation of stored parameters for one recorded chain."""
    start = _store_rows(snapshot, burnin_rows)
    results: list[AcfResult] = []
    if not 0 <= chain < snapshot.samples.shape[1]:
        return results
    for dim in dims:
        if not 0 <= dim < snapshot.samples.shape[2]:
            continue
        rho = normalized_acf(snapshot.samples[start:, chain, dim], max_lag)
        results.append(AcfResult(f'par {dim} {store_column_label(snapshot, chain)}', np.arange(rho.size), rho, integrated_autocorr_time(rho)))
    return results


def logl_cross_correlation(snapshot: RunSnapshot, chain_a: int, chain_b: int, max_lag: int, burnin_rows: int = 0) -> AcfResult | None:
    """Normalized cross-correlation of stored logL between two recorded chains."""
    start = _store_rows(snapshot, burnin_rows)
    n_recorded = snapshot.logLs.shape[1]
    if not (0 <= chain_a < n_recorded and 0 <= chain_b < n_recorded):
        return None
    series_a = snapshot.logLs[start:, chain_a] - snapshot.logLs[start:, chain_a].mean()
    series_b = snapshot.logLs[start:, chain_b] - snapshot.logLs[start:, chain_b].mean()
    n_samples = series_a.size
    if n_samples < 2:
        return None
    norm = np.sqrt((series_a**2).sum() * (series_b**2).sum())
    if norm <= 0.:
        return None
    lags_keep = min(int(max_lag), n_samples - 1)
    lags = np.arange(-lags_keep, lags_keep + 1)
    cross = np.asarray([np.dot(series_a[max(0, -lag):n_samples - max(0, lag)], series_b[max(0, lag):n_samples - max(0, -lag)]) for lag in lags]) / norm
    return AcfResult(f'logL {store_column_label(snapshot, chain_a)} vs {store_column_label(snapshot, chain_b)}', lags, cross, float('nan'))


def downsample_rows(n_rows: int, max_points: int) -> np.ndarray:
    """Deterministic even-stride row selection (no RNG in the dashboard)."""
    if n_rows <= max_points:
        return np.arange(n_rows)
    stride = int(np.ceil(n_rows / max_points))
    return np.arange(0, n_rows, stride)


def corner_matrix(snapshot: RunSnapshot, dims: list[int], chain: int = 0, burnin_rows: int = 0, max_points: int = 20000) -> tuple[np.ndarray, list[str]]:
    """Stored samples for a dimension subset of one recorded chain.

    Returns an (n_points, len(dims)) array (evenly downsampled after
    burn-in trimming) and axis labels; corner plots must use a subset
    because full parameter spaces can be too high-dimensional.
    """
    start = _store_rows(snapshot, burnin_rows)
    if snapshot.samples.ndim != 3 or not 0 <= chain < snapshot.samples.shape[1]:
        return np.zeros((0, len(dims))), [f'par {dim}' for dim in dims]
    dims_use = [dim for dim in dims if 0 <= dim < snapshot.samples.shape[2]]
    block = snapshot.samples[start:, chain, :][:, dims_use]
    rows = downsample_rows(block.shape[0], max_points)
    return block[rows], [f'par {dim}' for dim in dims_use]


def parameter_trace(snapshot: RunSnapshot, chain: int, dims: list[int], burnin_rows: int = 0, max_points: int = 5000) -> list[Curve]:
    """Thinned parameter traces (against iteration) for one recorded chain."""
    start = _store_rows(snapshot, burnin_rows)
    curves: list[Curve] = []
    if snapshot.samples.ndim != 3 or not 0 <= chain < snapshot.samples.shape[1]:
        return curves
    n_rows = snapshot.samples.shape[0] - start
    rows = downsample_rows(n_rows, max_points) + start
    iterations = rows * snapshot.store_thin
    curves.extend(
        Curve(f'par {dim}', iterations, snapshot.samples[rows, chain, dim], 'current')
        for dim in dims
        if 0 <= dim < snapshot.samples.shape[2]
    )
    return curves


def _format_wall_seconds(wall_seconds: float) -> str:
    """Format a wall-clock duration as h:mm:ss."""
    total = int(wall_seconds)
    return f'{total // 3600}:{(total % 3600) // 60:02d}:{total % 60:02d}'


def _format_record_indices(snapshot: RunSnapshot) -> str:
    """Compact current store-column → chain mapping for the header."""
    indices = [int(idx) for idx in snapshot.record_indices]
    if len(indices) > 8:
        return f'[{", ".join(str(idx) for idx in indices[:8])}, …] ({len(indices)} cols)'
    return str(indices)


def header_items(snapshot: RunSnapshot) -> list[tuple[str, str]]:
    """Ordered (label, value) pairs summarizing the run configuration/state."""
    spec = snapshot.spec
    likelihood = spec.get('likelihood', {})
    ladder = spec.get('ladder', {})
    exchange = spec.get('exchange', {})
    adaptive = snapshot.adaptive

    de_size = snapshot.proposal_config.get('DEJumpManager', {}).get('de_size', 'default')
    progress = f'{snapshot.n_iterations}/{snapshot.n_steps}' if snapshot.n_steps else str(snapshot.n_iterations)
    if snapshot.n_steps:
        progress += f' ({100. * snapshot.n_iterations / snapshot.n_steps:.0f}%)'

    if adaptive is None:
        adaptive_desc = 'none (fixed ladder)'
    else:
        adaptive_desc = str(adaptive.get('mode', '?'))
        if snapshot.history is not None and snapshot.history.frozen:
            adaptive_desc += f' (frozen by {snapshot.history.frozen_by or "n/a"}'
            if snapshot.history.frozen_block >= 0:
                adaptive_desc += f' at block {snapshot.history.frozen_block}'
            adaptive_desc += ')'
        elif snapshot.history is not None:
            adaptive_desc += ' (adapting)'

    git_commit = str(snapshot.attrs.get('git_commit', 'unknown'))[:10]
    if bool(snapshot.attrs.get('git_dirty', False)):
        git_commit += ' (dirty)'

    return [
        ('run', snapshot.name),
        ('likelihood', f'{snapshot.likelihood_name} (n_par={likelihood.get("n_par", "?")})'),
        ('ladder', f'{ladder.get("kind", "?")} n_chain={snapshot.n_chain} n_cold={snapshot.n_cold}'),
        ('adaptive', adaptive_desc),
        ('exchange', str(exchange.get('strategy', 'sequential'))),
        ('block size', str(snapshot.block_size)),
        ('progress', progress),
        ('store', f'thin={snapshot.store_thin} recorded chains {_format_record_indices(snapshot)}'),
        ('DE buffer', str(de_size)),
        ('seed', str(snapshot.attrs.get('run_seed', spec.get('seed', '?')))),
        ('git', git_commit),
        ('host', str(snapshot.attrs.get('hostname', '?'))),
        ('wall time', _format_wall_seconds(float(snapshot.attrs.get('wall_seconds', 0.)))),
        ('logL evals', f'{int(snapshot.attrs.get("n_likelihood_evals", 0)):.4g}'.replace('e+0', 'e')),
        ('last flush', str(snapshot.attrs.get('flush_time_utc', '?'))[:19]),
        ('status', 'finalized' if snapshot.finalized else 'in progress'),
    ]
