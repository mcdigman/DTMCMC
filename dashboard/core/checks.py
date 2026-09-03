"""Computable run-health checks: the data behind the status lights.

Each check is a pure function RunSnapshot -> (status, message) registered
in CHECKS. Statuses are 'ok' / 'warn' / 'alert', plus 'na' when a check
does not apply or lacks data (the message says why). Adding a light means
writing one evaluator and one CheckSpec row; front-ends render whatever
the registry holds and can silence any subset, so the set can grow
without UI changes.

Thresholds are heuristic defaults chosen to be quiet on healthy tiny
runs; every evaluator takes its thresholds as keyword arguments so a
future configuration surface (or a caller with different tolerances) can
retune them without touching the logic.
"""

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dashboard.core import diagnostics as diag

if TYPE_CHECKING:
    from collections.abc import Callable

    from dashboard.core.reader import RunSnapshot

STATUS_OK = 'ok'
STATUS_WARN = 'warn'
STATUS_ALERT = 'alert'
STATUS_NA = 'na'

# severity order for summarizing many results into one badge
_STATUS_RANK = {STATUS_NA: 0, STATUS_OK: 1, STATUS_WARN: 2, STATUS_ALERT: 3}


@dataclass(frozen=True)
class CheckResult:
    """One evaluated status light."""

    check_id: str
    title: str
    status: str
    message: str
    description: str


@dataclass(frozen=True)
class CheckSpec:
    """One registered status light: identity, prose, and evaluator."""

    check_id: str
    title: str
    description: str
    evaluate: Callable[[RunSnapshot], tuple[str, str]]


def _grade(value: float, warn_at: float, alert_at: float, *, larger_is_worse: bool = True) -> str:
    """Map a scalar onto ok/warn/alert against two thresholds."""
    if larger_is_worse:
        if value >= alert_at:
            return STATUS_ALERT
        return STATUS_WARN if value >= warn_at else STATUS_OK
    if value <= alert_at:
        return STATUS_ALERT
    return STATUS_WARN if value <= warn_at else STATUS_OK


def _current_segment_blocks(snapshot: RunSnapshot) -> tuple[int, int]:
    """Block range [start, stop) of the current fixed-ladder segment."""
    segment = diag.ladder_segments(snapshot)[-1]
    return segment.start_block, segment.stop_block


def _temperature_tag(Ts: np.ndarray, index: int) -> str:
    """Short T=... tag for one ladder slot."""
    T_loc = float(np.asarray(Ts)[index])
    return f'T={T_loc:.4g}' if np.isfinite(T_loc) else 'T=inf'


def check_de_rank(snapshot: RunSnapshot, warn_ratio: float = 0.7, alert_ratio: float = 0.4) -> tuple[str, str]:
    """DE-buffer difference spectrum effective rank vs full rank."""
    spectrum = diag.de_spectrum_summary(snapshot)
    n_par = snapshot.n_par
    if spectrum.itrns.size == 0 or n_par == 0:
        return STATUS_NA, 'no DE-spectrum checkpoints recorded'
    latest = spectrum.eff_rank[-1]
    worst_slot = int(np.argmin(latest))
    ratio = float(latest[worst_slot]) / n_par
    message = f'min effective rank {latest[worst_slot]:.2f}/{n_par} at slot {worst_slot} ({_temperature_tag(snapshot.Ts, worst_slot)})'
    return _grade(ratio, warn_ratio, alert_ratio, larger_is_worse=False), message


def check_acceptance_roughness(
    snapshot: RunSnapshot, warn_jump: float = 0.25, alert_jump: float = 0.5, min_trials: int = 30
) -> tuple[str, str]:
    """Largest acceptance-rate jump between adjacent temperatures per proposal."""
    table = diag.acceptance_by_temperature(snapshot, 'segment')
    if not table.labels:
        return STATUS_NA, 'no proposals recorded yet'
    worst = 0.0
    worst_desc = ''
    found_pair = False
    finite_T = np.isfinite(table.Ts)
    for itrj, label in enumerate(table.labels):
        rates = table.values[finite_T, itrj]
        trials = table.trials[finite_T, itrj]
        Ts_use = table.Ts[finite_T]
        usable = np.flatnonzero(np.isfinite(rates) & (trials >= min_trials))
        # only genuinely adjacent temperature bins: a pair separated by a
        # filtered low-trial bin says nothing about local roughness
        for idx_cold, idx_hot in itertools.pairwise(usable):
            if idx_hot != idx_cold + 1:
                continue
            found_pair = True
            jump = abs(float(rates[idx_hot] - rates[idx_cold]))
            if jump > worst:
                worst = jump
                worst_desc = f'{label}: Δ={worst:.2f} between T={Ts_use[idx_cold]:.4g} and T={Ts_use[idx_hot]:.4g}'
    if not found_pair:
        return STATUS_NA, f'no adjacent temperature pair with ≥{min_trials} trials'
    return _grade(worst, warn_jump, alert_jump), worst_desc


def check_exchange_share(
    snapshot: RunSnapshot, warn_share: float = 0.95, alert_share: float = 0.995
) -> tuple[str, str]:
    """Fraction of accepted squared displacement carried by exchanges.

    Exchanges legitimately move a lot in parallel tempering; a share of
    nearly one means within-temperature proposals are barely moving
    anything and local exploration has stalled.
    """
    exchange_sums = diag.window_counts(snapshot.esd_exchange, snapshot.esd_exchange_archive, 'total')
    proposal_sums = diag.window_counts(snapshot.esd_record, snapshot.esd_archive, 'total')[1]
    exchange_total = float(np.asarray(exchange_sums).sum())
    proposal_total = float(np.asarray(proposal_sums).sum())
    if exchange_total + proposal_total <= 0.0:
        return STATUS_NA, 'no accepted displacement recorded yet'
    share = exchange_total / (exchange_total + proposal_total)
    message = f'exchanges carry {100.0 * share:.1f}% of accepted |Δx|² (proposals {100.0 * (1.0 - share):.1f}%)'
    return _grade(share, warn_share, alert_share), message


def check_exchange_uniformity(
    snapshot: RunSnapshot, warn_spread: float = 0.4, alert_spread: float = 0.7, min_trials: int = 50
) -> tuple[str, str]:
    """Spread of nearest-neighbor exchange acceptance across temperatures.

    A constant-acceptance ladder working as designed keeps this flat;
    a large spread means some links are far tighter than others.
    """
    rates = diag.exchange_rates(snapshot, 'segment')
    usable = np.isfinite(rates.nn_rate) & (rates.nn_trials >= min_trials)
    if int(usable.sum()) < 2:
        return STATUS_NA, f'fewer than two temperature bins with ≥{min_trials} exchange trials'
    spread = float(rates.nn_rate[usable].max() - rates.nn_rate[usable].min())
    message = (
        f'nn acceptance spans {rates.nn_rate[usable].min():.2f}-{rates.nn_rate[usable].max():.2f} (spread {spread:.2f})'
    )
    return _grade(spread, warn_spread, alert_spread), message


def check_exchange_bottleneck(
    snapshot: RunSnapshot, warn_rate: float = 0.05, alert_rate: float = 0.01, min_trials: int = 100
) -> tuple[str, str]:
    """Near-zero exchange link: the ladder splits into disconnected islands."""
    rates = diag.exchange_rates(snapshot, 'segment')
    usable = np.isfinite(rates.nn_rate) & (rates.nn_trials >= min_trials)
    if not np.any(usable):
        return STATUS_NA, f'no temperature bin with ≥{min_trials} exchange trials'
    rates_use = rates.nn_rate[usable]
    Ts_use = rates.Ts[usable]
    arg_min = int(np.argmin(rates_use))
    T_tag = f'T={Ts_use[arg_min]:.4g}' if np.isfinite(Ts_use[arg_min]) else 'T=inf'
    message = f'weakest nn link accepts {100.0 * rates_use[arg_min]:.2f}% at {T_tag}'
    return _grade(float(rates_use[arg_min]), warn_rate, alert_rate, larger_is_worse=False), message


def _half_block_means(snapshot: RunSnapshot, min_blocks_per_half: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Split the current segment's block means into (earlier, later) halves."""
    start, stop = _current_segment_blocks(snapshot)
    n_blocks = stop - start
    if n_blocks < 2 * min_blocks_per_half:
        return None
    mid = start + n_blocks // 2
    return snapshot.logL_means[start:mid], snapshot.logL_means[mid:stop]


def check_thermo_stability(
    snapshot: RunSnapshot, warn_log2: float = 1.5, alert_log2: float = 2.5, min_blocks_per_half: int = 6
) -> tuple[str, str]:
    """Var(logL) stability between the two halves of the current segment.

    The heat capacity C(T) = Var(logL)/T² is the thermodynamic profile the
    ladder is built from; a large half-to-half variance ratio at some
    temperature means the profile is still moving.
    """
    start, stop = _current_segment_blocks(snapshot)
    n_blocks = stop - start
    if n_blocks < 2 * min_blocks_per_half:
        return STATUS_NA, f'needs ≥{2 * min_blocks_per_half} blocks on the current ladder ({n_blocks} so far)'
    mid = start + n_blocks // 2
    finite_T = np.isfinite(np.asarray(snapshot.Ts))
    worst = 0.0
    worst_slot = -1
    for slot in np.flatnonzero(finite_T):
        e1_a = snapshot.logL_means[start:mid, slot]
        e2_a = snapshot.logL2_means[start:mid, slot]
        e1_b = snapshot.logL_means[mid:stop, slot]
        e2_b = snapshot.logL2_means[mid:stop, slot]
        var_a = max(float(e2_a.mean() - e1_a.mean() ** 2), 0.0)
        var_b = max(float(e2_b.mean() - e1_b.mean() ** 2), 0.0)
        if var_a <= 0.0 or var_b <= 0.0:
            continue
        log2_ratio = abs(float(np.log2(var_b / var_a)))
        if log2_ratio > worst:
            worst = log2_ratio
            worst_slot = int(slot)
    if worst_slot < 0:
        return STATUS_NA, 'no slot with positive Var(logL) in both halves'
    message = f'largest half-to-half Var(logL) change x{2.0**worst:.2f} at slot {worst_slot} ({_temperature_tag(snapshot.Ts, worst_slot)})'
    return _grade(worst, warn_log2, alert_log2), message


def check_logl_drift(
    snapshot: RunSnapshot, warn_z: float = 5.0, alert_z: float = 10.0, min_blocks_per_half: int = 4
) -> tuple[str, str]:
    """Drift of blockwise mean logL between halves of the current segment.

    A large drift z-score means the chain distribution is still moving
    (burn-in not over, or a late transition); block means are correlated,
    so the thresholds are deliberately loose.
    """
    halves = _half_block_means(snapshot, min_blocks_per_half)
    if halves is None:
        return STATUS_NA, f'needs ≥{2 * min_blocks_per_half} blocks on the current ladder'
    first, second = halves
    finite_T = np.isfinite(np.asarray(snapshot.Ts))
    with np.errstate(invalid='ignore', divide='ignore'):
        scale = np.sqrt(first.var(axis=0) / first.shape[0] + second.var(axis=0) / second.shape[0])
        z_scores = np.abs(second.mean(axis=0) - first.mean(axis=0)) / scale
    z_scores = np.where(finite_T & np.isfinite(z_scores), z_scores, 0.0)
    worst_slot = int(np.argmax(z_scores))
    worst = float(z_scores[worst_slot])
    if worst <= 0.0:
        return STATUS_NA, 'no slot with a finite drift estimate'
    message = f'largest block-mean logL drift z≈{worst:.1f} at slot {worst_slot} ({_temperature_tag(snapshot.Ts, worst_slot)})'
    return _grade(worst, warn_z, alert_z), message


def check_round_trips(snapshot: RunSnapshot, warn_per_walker: float = 1.0, min_blocks: int = 8) -> tuple[str, str]:
    """Round-trip traffic: walkers should complete hot-cold-hot cycles."""
    if snapshot.n_blocks < min_blocks:
        return STATUS_NA, f'needs ≥{min_blocks} blocks ({snapshot.n_blocks} so far)'
    trips = diag.round_trip_summary(snapshot)
    cold_arrivals = int(trips.cumulative_cold[-1]) if trips.cumulative_cold.size else 0
    per_walker = cold_arrivals / max(snapshot.n_chain, 1)
    message = f'{cold_arrivals} cold arrivals across {snapshot.n_chain} walkers ({per_walker:.1f} per walker)'
    if cold_arrivals == 0:
        return STATUS_ALERT, message + ' — no walker has completed a hot→cold trip'
    return (STATUS_WARN if per_walker < warn_per_walker else STATUS_OK), message


def check_cold_tau(
    snapshot: RunSnapshot, warn_taus: float = 50.0, alert_taus: float = 10.0, min_rows: int = 64
) -> tuple[str, str]:
    """Cold-chain logL autocorrelation time against the stored history length.

    Fewer than ~50 integrated times of data makes posterior summaries
    unreliable; fewer than ~10 means the store is essentially one sample.
    """
    n_rows = int(snapshot.logLs.shape[0])
    if n_rows < min_rows or snapshot.logLs.ndim != 2 or snapshot.logLs.shape[1] == 0:
        return STATUS_NA, f'needs ≥{min_rows} stored rows ({n_rows} so far)'
    burnin_rows = n_rows // 4
    results = diag.logl_acf(snapshot, [0], max_lag=max((n_rows - burnin_rows) // 4, 8), burnin_rows=burnin_rows)
    if not results:
        return STATUS_NA, 'no stored logL for the cold chain'
    tau_rows = max(results[0].tau_int, 1.0)
    n_taus = (n_rows - burnin_rows) / tau_rows
    message = f'store holds {n_taus:.0f} integrated times (τ≈{tau_rows * snapshot.store_thin:.0f} iterations)'
    return _grade(n_taus, warn_taus, alert_taus, larger_is_worse=False), message


def check_flow_linearity(
    snapshot: RunSnapshot, warn_dev: float = 0.3, alert_dev: float = 0.5, min_blocks: int = 8
) -> tuple[str, str]:
    """Deviation of the walker up-flow fraction from the linear ideal.

    The constant-round-trip-flow profile is linear in rung index; a large
    bulge marks where walker traffic piles up.
    """
    if snapshot.n_blocks < min_blocks:
        return STATUS_NA, f'needs ≥{min_blocks} blocks ({snapshot.n_blocks} so far)'
    flow = diag.flow_fraction(snapshot)
    deviation = np.abs(flow.f_latest - flow.f_ideal)
    usable = np.isfinite(deviation)
    if not np.any(usable):
        return STATUS_NA, 'no labeled walkers yet'
    worst_slot = int(np.flatnonzero(usable)[np.argmax(deviation[usable])])
    worst = float(deviation[worst_slot])
    message = f'max |f - ideal| = {worst:.2f} at slot {worst_slot} ({_temperature_tag(snapshot.Ts, worst_slot)})'
    return _grade(worst, warn_dev, alert_dev), message


def check_cold_acceptance(
    snapshot: RunSnapshot, warn_rate: float = 0.02, alert_rate: float = 0.005, min_trials: int = 200
) -> tuple[str, str]:
    """Overall acceptance at the coldest slot: is the cold chain frozen?

    Summed across every proposal type, so a legitimately cold-hostile
    proposal (e.g. prior draws) cannot trip it alone.
    """
    table = diag.acceptance_by_temperature(snapshot, 'segment')
    if not table.labels or not np.any(np.isfinite(table.Ts)):
        return STATUS_NA, 'no proposals recorded yet'
    cold_bin = int(np.nanargmin(np.where(np.isfinite(table.Ts), table.Ts, np.nan)))
    trials = float(table.trials[cold_bin].sum())
    if trials < min_trials:
        return STATUS_NA, f'needs ≥{min_trials} cold-slot proposals on the current ladder ({trials:.0f} so far)'
    with np.errstate(invalid='ignore'):
        accepted = float(np.nansum(table.values[cold_bin] * table.trials[cold_bin]))
    rate = accepted / trials
    message = f'coldest bin (T={table.Ts[cold_bin]:.4g}) accepted {100.0 * rate:.2f}% of {trials:.0f} proposals'
    return _grade(rate, warn_rate, alert_rate, larger_is_worse=False), message


def check_finite_moments(snapshot: RunSnapshot, recent_blocks: int = 4) -> tuple[str, str]:
    """Non-finite values in recent block moments signal numerical trouble."""
    if snapshot.n_blocks == 0:
        return STATUS_NA, 'no completed blocks yet'
    recent = slice(max(0, snapshot.n_blocks - recent_blocks), snapshot.n_blocks)
    bad_means = int((~np.isfinite(snapshot.logL_means[recent])).sum())
    bad_vars = int((~np.isfinite(snapshot.logL_vars[recent])).sum())
    if bad_means or bad_vars:
        return (
            STATUS_ALERT,
            f'{bad_means + bad_vars} non-finite entries in the last {recent_blocks} blocks of logL moments',
        )
    return STATUS_OK, f'all logL moments finite over the last {recent_blocks} blocks'


def check_ladder_freeze(snapshot: RunSnapshot) -> tuple[str, str]:
    """Adaptive-ladder state: adapting, budget-frozen, or criterion-frozen."""
    history = snapshot.history
    if history is None:
        return STATUS_NA, 'fixed ladder (no adaptation)'
    if not history.frozen:
        return STATUS_WARN, 'ladder still adapting — thermodynamic diagnostics are provisional'
    if history.frozen_by == 'budget':
        block_tag = f' at block {history.frozen_block}' if history.frozen_block >= 0 else ''
        return STATUS_WARN, f'freeze forced by budget{block_tag} — spacing may not have converged'
    block_tag = f' at block {history.frozen_block}' if history.frozen_block >= 0 else ''
    return STATUS_OK, f'frozen by {history.frozen_by or "criterion"}{block_tag}'


_SPECS: tuple[CheckSpec, ...] = (
    CheckSpec(
        'finite_moments', 'Finite moments', 'Recent block logL moments contain no NaN/inf.', check_finite_moments
    ),
    CheckSpec(
        'ladder_freeze',
        'Ladder freeze',
        'Adaptive ladder has frozen, and by criterion rather than budget.',
        check_ladder_freeze,
    ),
    CheckSpec(
        'logl_drift',
        'Mean logL drift',
        'Blockwise mean logL is not drifting between recent halves of the current segment.',
        check_logl_drift,
    ),
    CheckSpec(
        'thermo_stability',
        'Thermodynamic stability',
        'Var(logL) (the heat-capacity profile) is stable over recent iterations.',
        check_thermo_stability,
    ),
    CheckSpec(
        'round_trips', 'Round trips', 'Walkers complete hot-cold round trips at a healthy rate.', check_round_trips
    ),
    CheckSpec(
        'flow_linearity',
        'Walker flow',
        'The up-flow fraction stays near the linear constant-flow ideal.',
        check_flow_linearity,
    ),
    CheckSpec(
        'exchange_uniformity',
        'Exchange uniformity',
        'Nearest-neighbor exchange acceptance is even across temperatures.',
        check_exchange_uniformity,
    ),
    CheckSpec(
        'exchange_bottleneck',
        'Exchange bottleneck',
        'No temperature link has near-zero exchange acceptance.',
        check_exchange_bottleneck,
    ),
    CheckSpec(
        'exchange_share',
        'Exchange flow share',
        'Within-temperature proposals still contribute meaningful movement.',
        check_exchange_share,
    ),
    CheckSpec(
        'acceptance_roughness',
        'Acceptance smoothness',
        'Proposal acceptance does not jump sharply between adjacent temperatures.',
        check_acceptance_roughness,
    ),
    CheckSpec(
        'cold_acceptance', 'Cold-chain acceptance', 'The coldest chain is still accepting moves.', check_cold_acceptance
    ),
    CheckSpec(
        'cold_tau',
        'Cold-chain correlation',
        'The stored cold-chain history spans many autocorrelation times.',
        check_cold_tau,
    ),
    CheckSpec(
        'de_rank', 'DE buffer rank', 'The DE difference spectrum keeps (near) full effective rank.', check_de_rank
    ),
)

CHECKS: dict[str, CheckSpec] = {spec.check_id: spec for spec in _SPECS}


def evaluate_checks(snapshot: RunSnapshot, enabled: list[str] | None = None) -> list[CheckResult]:
    """Evaluate the enabled checks (all when enabled is None), in registry order.

    An evaluator that raises does not take the dashboard down: the light
    reports 'alert' with the error, since a check crashing on real data is
    itself a signal worth surfacing.
    """
    enabled_set = set(CHECKS) if enabled is None else set(enabled)
    results: list[CheckResult] = []
    for spec in _SPECS:
        if spec.check_id not in enabled_set:
            continue
        try:
            status, message = spec.evaluate(snapshot)
        except Exception as err:  # noqa: BLE001 - a broken check must not break the dashboard
            status, message = STATUS_ALERT, f'check failed: {type(err).__name__}: {err}'
        results.append(CheckResult(spec.check_id, spec.title, status, message, spec.description))
    return results


def worst_status(results: list[CheckResult]) -> str:
    """The most severe status among results ('na' when there are none)."""
    if not results:
        return STATUS_NA
    return max((result.status for result in results), key=lambda status: _STATUS_RANK.get(status, 0))


def status_counts(results: list[CheckResult]) -> dict[str, int]:
    """How many results landed at each status."""
    counts = {STATUS_OK: 0, STATUS_WARN: 0, STATUS_ALERT: 0, STATUS_NA: 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts
