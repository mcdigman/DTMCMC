"""Analysis-side metrics for the methods paper (plan §4 Phase 2).

Everything here is a pure observer. Functions intended for in-process
checkpoint use (the DE-buffer spectrum) take an explicit numpy Generator
from DTMCMC.rng_helpers.get_rng, which touches neither of the run's RNG
streams (plan D5) — the golden-run digest is unaffected by calling them
mid-run. Post-hoc functions operate on artifact contents.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.signal

from DTMCMC.corr_summary_helpers import autocorr_helper
from DTMCMC.tracker_manager import RT_ARRIVED_COLD, RT_ARRIVED_HOT
from entropy_process import NNEntropyK

if TYPE_CHECKING:
    from numpy.typing import NDArray


def de_buffer_difference_spectrum(de_buffer: NDArray[np.floating], n_pairs: int, rng: np.random.Generator) -> NDArray[np.floating]:
    """Eigenspectrum of the covariance of random DE-buffer differences.

    Draws n_pairs random buffer-entry pairs per temperature and returns
    the descending eigenvalues of the raw second moment of their
    differences, shape (n_chain, n_par). Rank collapse of the difference
    span (plan C4) shows up as trailing eigenvalues pinned near zero.
    Safe to call mid-run: draws only from the passed Generator.
    """
    de_size, n_chain, n_par = de_buffer.shape
    idx1 = rng.integers(0, de_size, size=n_pairs)
    idx2 = rng.integers(0, de_size, size=n_pairs)
    eigvals = np.zeros((n_chain, n_par))
    for itrt in range(n_chain):
        diffs = de_buffer[idx1, itrt, :] - de_buffer[idx2, itrt, :]
        # the difference distribution is symmetric about zero, so use the
        # raw second moment (what DE proposals actually sample)
        second_moment = diffs.T @ diffs / n_pairs
        eigvals[itrt] = np.linalg.eigvalsh(second_moment)[::-1]
    return eigvals


def effective_rank(eigvals: NDArray[np.floating]) -> NDArray[np.floating]:
    """Participation-ratio effective rank along the last axis.

    (sum lambda)^2 / sum lambda^2: n_par for isotropic spectra, 1 for
    rank-one collapse, 0 where the spectrum is identically zero.
    """
    total = eigvals.sum(axis=-1)
    total_sq = (eigvals**2).sum(axis=-1)
    out = np.zeros(np.shape(total))
    nonzero = total_sq > 0.
    out[nonzero] = total[nonzero]**2 / total_sq[nonzero]
    return out


def nn_kl(reference_samples: NDArray[np.floating], test_samples: NDArray[np.floating], n_use: int, rng: np.random.Generator) -> float:
    """Signed two-sample NN divergence of test against reference samples.

    Wraps entropy_process.NNEntropyK (O(n_use^2): keep n_use ~ 5-10k,
    checkpoint/post-hoc snapshots only), oriented so the result is ~0 when
    the distributions match and grows positive when test samples occupy
    regions where the reference is sparse (the direction quality
    thresholds care about; measured ~+6 for cake-vs-Gaussian, |value|
    < ~0.4 estimator noise at n_use = 2000 for matched 5D draws). Both
    inputs are subsampled without replacement to a common size with the
    passed Generator and copied, so callers' arrays are never mutated.
    """
    n_use = min(n_use, reference_samples.shape[0], test_samples.shape[0])
    ref_idx = rng.choice(reference_samples.shape[0], size=n_use, replace=False)
    test_idx = rng.choice(test_samples.shape[0], size=n_use, replace=False)
    ref_use = np.ascontiguousarray(reference_samples[ref_idx])
    test_use = np.ascontiguousarray(test_samples[test_idx])
    _kl, entropy_ref, entropy_cross = NNEntropyK(ref_use, test_use)
    return float(entropy_cross - entropy_ref)


def nn_divergence_symmetric(reference_samples: NDArray[np.floating], test_samples: NDArray[np.floating], n_use: int, rng: np.random.Generator) -> float:
    """Symmetric two-sample NN divergence: max of both nn_kl orientations.

    The signed nn_kl can have opposite signs for overconcentrated and
    support-missing samples. Taking the max over both orientations makes
    either mismatch a positive value, at the cost of doubling the
    O(n_use^2) work.
    """
    forward = nn_kl(reference_samples, test_samples, n_use, rng)
    backward = nn_kl(test_samples, reference_samples, n_use, rng)
    return max(forward, backward)


def scramble_block_n_eff(samples: NDArray[np.floating], block_size: int, n_blocks: int, rng: np.random.Generator) -> NDArray[np.floating]:
    """Frozen C1 effective-sample estimator (n_eff_preds_empirical, plan §6).

    Variance ratio of scrambled to sequential block means, per parameter,
    on cold-chain samples of shape (n_rows, n_cold, n_par). Conventions,
    pinned because this definition underlies the paper's primary metric:

    - Callers pass exactly the rows the estimate is over: post-burn-in,
      post-thinning cold-chain samples. n_tot = n_rows * n_cold of the
      passed array, nothing else.
    - Sequential blocks are contiguous runs of block_size rows across all
      cold chains at Generator-chosen starts uniform over
      [0, n_rows - block_size], inclusive: every row can appear.
    - Scrambled blocks draw block_size * n_cold entries independently
      and uniformly across (row, chain), with replacement.
    - Per-run aggregation over parameters is the minimum, provided by
      scramble_block_n_eff_min.

    Same estimator as the engine's n_eff_preds_empirical
    (DTMCMC.corr_summary_helpers), reimplemented Generator-first so
    analysis stays off the run RNG streams (plan D5). Two deliberate
    deviations from the engine original: the sequential-block start bound
    is inclusive (the original's np.random.randint high bound silently
    excludes the final row from every sequential block — noted on the
    issue #11 audit), and n_tot is defined on the passed array rather
    than via store bookkeeping.
    """
    n_rows, n_cold, n_par = samples.shape
    if n_rows <= block_size:
        msg = 'samples must be longer than block_size'
        raise ValueError(msg)
    n_tot = n_rows * n_cold

    seq_means = np.zeros((n_blocks, n_par))
    scr_means = np.zeros((n_blocks, n_par))
    for itrb in range(n_blocks):
        start = int(rng.integers(0, n_rows - block_size + 1))
        seq_means[itrb] = samples[start:start + block_size, :, :].mean(axis=(0, 1))
        row_idx = rng.integers(0, n_rows, size=block_size * n_cold)
        chain_idx = rng.integers(0, n_cold, size=block_size * n_cold)
        scr_means[itrb] = samples[row_idx, chain_idx, :].mean(axis=0)

    seq_var = seq_means.var(axis=0)
    scr_var = scr_means.var(axis=0)
    n_eff = np.full(n_par, np.inf)
    nonzero = seq_var > 0.
    n_eff[nonzero] = scr_var[nonzero] / seq_var[nonzero] * n_tot
    return n_eff


def scramble_block_n_eff_min(samples: NDArray[np.floating], block_size: int, n_blocks: int, rng: np.random.Generator) -> float:
    """The C1 primary statistic: minimum over parameters of the frozen estimator.

    Plan §6 freezes the aggregation rule — per-parameter n_eff on the
    cold chains, aggregated by minimum over parameters, evaluated per
    run — so the pre-registered quantity has a named function rather
    than a convention callers must remember.
    """
    return float(np.min(scramble_block_n_eff(samples, block_size, n_blocks, rng)))


@dataclass
class _StoreView:
    """Adapter exposing the sampler attrs corr_summary_helpers expects."""

    samples_store: NDArray[np.floating]
    store_size: int
    n_cold: int
    n_chain: int
    block_size: int
    store_thin: int


@dataclass(frozen=True)
class SuperEfficiencyResult:
    """Result of the apparent-super-efficiency detector (plan S3)."""

    flags: NDArray[np.bool_]
    n_eff_auto: NDArray[np.floating]
    n_eff_with_cross: NDArray[np.floating]


def _cross_covariance_estimate(samples_store: NDArray[np.floating], itrp: int, n_burnin_thin: int, autocorr_cut: int) -> float:
    """Lag-summed cross-chain covariance, truncated at the autocorrelation cut.

    Mirrors the folding in DTMCMC.corr_summary_helpers.get_crosscorr_sum
    with consistent lengths (that helper has a latent off-by-one in its
    buffer sizing, flagged for a separate engine fix).
    """
    n_rows, n_cold, _n_par = samples_store.shape
    n_use = n_rows - n_burnin_thin
    cross_sum = np.zeros(2 * n_use - 1)
    for itrt1 in range(n_cold):
        params_adj1 = samples_store[n_burnin_thin:, itrt1, itrp] - np.mean(samples_store[n_burnin_thin:, itrt1, itrp])
        for itrt2 in range(itrt1 + 1, n_cold):
            params_adj2 = samples_store[n_burnin_thin:, itrt2, itrp] - np.mean(samples_store[n_burnin_thin:, itrt2, itrp])
            corr_loc = scipy.signal.correlate(params_adj1, params_adj2, mode='full')
            cross_sum += corr_loc
            cross_sum += corr_loc[::-1]
    cross_lim = np.hstack([cross_sum[n_use - 1:n_use], cross_sum[n_use:2 * n_use - 2:2] + cross_sum[n_use + 1:2 * n_use - 1:2]])
    return float(cross_lim[0] + 2. * np.sum(cross_lim[1:autocorr_cut]))


def detect_apparent_super_efficiency(samples_store: NDArray[np.floating], block_size: int, store_thin: int = 1, n_burnin_thin: int = 0) -> SuperEfficiencyResult:
    """Flag parameters whose cross-chain terms claim n_eff above the autocorr estimate.

    Negative adjacent-chain cross-correlation makes the combined variance
    estimate smaller than the per-chain autocorrelation one — apparent
    "super-efficiency" (plan S3), diagnosed here per parameter using the
    autocorrelation machinery of DTMCMC.corr_summary_helpers plus a
    consistent-length cross term. Deterministic: no random draws.
    """
    n_rows, n_cold, n_par = samples_store.shape
    view = _StoreView(
        samples_store=samples_store,
        store_size=n_rows,
        n_cold=n_cold,
        n_chain=n_cold,
        block_size=block_size,
        store_thin=store_thin,
    )
    n_use = n_rows - n_burnin_thin
    n_tot = n_use * n_cold

    n_eff_auto = np.zeros(n_par)
    n_eff_with_cross = np.zeros(n_par)
    for itrp in range(n_par):
        autocorr_lim, autocorr_cut, est_var_auto = autocorr_helper(view, itrp, n_burnin_thin)
        n_eff_auto[itrp] = n_tot / (est_var_auto / autocorr_lim[0])
        est_var_cross = _cross_covariance_estimate(samples_store, itrp, n_burnin_thin, autocorr_cut)
        est_var_total = est_var_auto + est_var_cross
        if est_var_total <= 0.:
            n_eff_with_cross[itrp] = np.inf
        else:
            n_eff_with_cross[itrp] = n_tot / (est_var_total / autocorr_lim[0])

    return SuperEfficiencyResult(
        flags=n_eff_with_cross > n_eff_auto,
        n_eff_auto=n_eff_auto,
        n_eff_with_cross=n_eff_with_cross,
    )


def _rt_segment_ids(rt_events: NDArray[np.int64], segment_itrns: NDArray[np.int64] | None) -> NDArray[np.int64]:
    """Ladder-segment index of each event row.

    Boundaries are ladder-update iterations (plan D6); an event at or
    before a boundary belongs to the closing segment. No boundaries
    (fixed-ladder run) means one segment.
    """
    if segment_itrns is None or len(segment_itrns) == 0:
        return np.zeros(rt_events.shape[0], dtype=np.int64)
    return np.searchsorted(np.asarray(segment_itrns), rt_events[:, 1], side='left').astype(np.int64)


def round_trip_counts(rt_events: NDArray[np.int64], n_chain: int, segment_itrns: NDArray[np.int64] | None = None) -> NDArray[np.int64]:
    """Complete round trips per walker from the event log.

    A round trip needs one arrival in each direction; the count per
    walker is min(#cold arrivals, #hot arrivals), matching
    TrackerManager.get_n_cycles. With segment boundaries (adaptive
    runs), arrivals are paired within each ladder segment and the
    per-segment trips summed — never across an update (plan D6).
    """
    seg_ids = _rt_segment_ids(rt_events, segment_itrns)
    trips = np.zeros(n_chain, dtype=np.int64)
    for seg in np.unique(seg_ids):
        counts = np.zeros((2, n_chain), dtype=np.int64)
        for walker, _itrn, direction in rt_events[seg_ids == seg]:
            counts[direction, walker] += 1
        trips += np.minimum(counts[RT_ARRIVED_COLD], counts[RT_ARRIVED_HOT])
    return trips


def round_trip_rate(rt_events: NDArray[np.int64], n_chain: int, n_iterations: int, segment_itrns: NDArray[np.int64] | None = None) -> float:
    """Round trips per walker per 1e6 chain-steps (primary C1/C2 metric)."""
    total_trips = int(round_trip_counts(rt_events, n_chain, segment_itrns).sum())
    chain_steps = n_iterations * n_chain
    if chain_steps == 0:
        return 0.
    return total_trips / n_chain / (chain_steps / 1.e6)


def fraction_walkers_with_round_trip(rt_events: NDArray[np.int64], n_chain: int, segment_itrns: NDArray[np.int64] | None = None) -> float:
    """Fraction of walkers that completed at least one round trip."""
    return float(np.count_nonzero(round_trip_counts(rt_events, n_chain, segment_itrns))) / n_chain


def round_trip_times(rt_events: NDArray[np.int64], n_chain: int, segment_itrns: NDArray[np.int64] | None = None) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Full-cycle durations pooled over walkers, from the event log.

    Returns (cold-to-cold durations, hot-to-hot durations): iteration
    counts between a walker's consecutive same-direction arrivals.
    With segment boundaries, durations never span a ladder update
    (plan D6).
    """
    seg_ids = _rt_segment_ids(rt_events, segment_itrns)
    cold_times: list[int] = []
    hot_times: list[int] = []
    for seg in np.unique(seg_ids):
        seg_events = rt_events[seg_ids == seg]
        for walker in range(n_chain):
            walker_events = seg_events[seg_events[:, 0] == walker]
            for direction, out in ((RT_ARRIVED_COLD, cold_times), (RT_ARRIVED_HOT, hot_times)):
                arrivals = walker_events[walker_events[:, 2] == direction, 1]
                out.extend(np.diff(arrivals).tolist())
    return np.asarray(cold_times, dtype=np.int64), np.asarray(hot_times, dtype=np.int64)


def flow_fraction(up_counts: NDArray[np.int64], labeled_counts: NDArray[np.int64]) -> NDArray[np.floating]:
    """Up-mover fraction f(T) per (block, temperature) from stored flow counts.

    NaN where no resident walker had touched an extreme yet.
    """
    out = np.full(up_counts.shape, np.nan)
    nonzero = labeled_counts > 0
    out[nonzero] = up_counts[nonzero] / labeled_counts[nonzero]
    return out


def fit_knee_piecewise_linear(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Knee location from a continuous two-segment linear fit.

    Fits y ~ a + b*x + c*max(0, x - k) by least squares over interior
    breakpoint candidates k = x[1..n-2] and returns the k minimizing the
    residual. One of the two knee estimators frozen in Phase 4.
    """
    if x.size < 4:
        msg = 'piecewise-linear knee fit needs at least 4 points'
        raise ValueError(msg)
    best_k = float(x[1])
    best_sse = np.inf
    for candidate in x[1:-1]:
        design = np.column_stack([np.ones(x.size), x, np.maximum(0., x - candidate)])
        _coef, residual, _rank, _sv = np.linalg.lstsq(design, y)
        sse = float(residual[0]) if residual.size else 0.
        if sse < best_sse:
            best_sse = sse
            best_k = float(candidate)
    return best_k


def fit_knee_max_curvature(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Knee location as the point of maximum discrete curvature.

    Both axes are normalized to [0, 1] so the result is scale-invariant;
    curvature is estimated by second differences on the normalized curve.
    The other Phase 4 knee-estimator candidate.
    """
    if x.size < 3:
        msg = 'max-curvature knee fit needs at least 3 points'
        raise ValueError(msg)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_span = y.max() - y.min()
    y_norm = (y - y.min()) / y_span if y_span > 0. else np.zeros_like(y)
    curvature = np.zeros(x.size)
    for itrk in range(1, x.size - 1):
        h1 = x_norm[itrk] - x_norm[itrk - 1]
        h2 = x_norm[itrk + 1] - x_norm[itrk]
        curvature[itrk] = np.abs(2. * (h1 * y_norm[itrk + 1] - (h1 + h2) * y_norm[itrk] + h2 * y_norm[itrk - 1]) / (h1 * h2 * (h1 + h2)))
    return float(x[int(np.argmax(curvature))])
