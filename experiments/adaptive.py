"""Adaptive burn-in ladder controller.

The controller wraps a DTMCMCSampler and mutates engine state only
through the DTMCMCSampler.apply_ladder_update hook.

Annealing schedule: the initial ladder is anchored at the hot end from
prior-draw logL statistics; each update extends the cold edge of the
spacing window by a fixed factor toward the target (optionally below
T=1 via T_min_factor) and rebuilds the ladder from the pooled first two
logL cumulants measured so far. The rebuild is mode-agnostic, so
the entropy, length, and acceptance spacing rules share one schedule.
Rebuild variances default to the pessimistic estimator (max over each
temperature's recent segment estimates, var_estimator switch).
Sub-threshold rebuilds (within the freeze dlog threshold) are held,
not applied: no remap and no tracker segmentation.

Adaptation ends in a hard freeze gated by a coupling witness: at least
one completed cold<->hot round trip within each cadence window of the
stability streak. A run reaching budget_blocks unfrozen hard-freezes
with frozen_by='budget' recorded. After the freeze the hook is never
called again, matching the fixed-ladder path.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from DTMCMC.temperature_ladder_helpers import (
    AcceptanceTemperatureLadder,
    EntropyTemperatureLadder,
    GeometricTemperatureLadder,
    LengthTemperatureLadder,
    TemperatureLadder,
    Ts_to_betas,
    get_spacing_integrated,
    standardize_input_vars,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from DTMCMC.dtmcmc_sampler import DTMCMCSampler
    from DTMCMC.likelihood import AbstractLikelihood

ADAPTIVE_MODES: frozenset[str] = frozenset({'entropy', 'length', 'acceptance'})

# Defaults for controller geometry. Each is also available as an
# AdaptiveLadderController field wired to the [adaptive] spec table.

# cold-window extension factor per update: each rebuild reaches up to 4x
# colder until the target is hit, then refinement continues in place
WINDOW_EXTENSION_FACTOR = 4.0

# per-link dS budget throttling the extension: the window only descends
# while the pooled spacing integral stays within ds_link_cap nats per
# link
DS_LINK_CAP = 2.5

# candidate partial extension factors tried when the full factor blows
# the budget; factors above the configured window_extension_factor are
# skipped, so the config caps the schedule without editing this tuple
_EXTENSION_CANDIDATES = (4.0, 3.0, 2.0, 1.5, 1.25, 1.1)

# cold-edge cap: the coldest cold_cap_links links are pulled to at most
# exp(ds_link_cap / C) in temperature ratio, with C the measured local
# heat capacity beta^2 Var(logL) at the link's cold end, then clipped to
# cap_ratio_bounds. COLD_CAP_LINKS_AUTO resolves to a chain-count-scaled
# default.
COLD_CAP_LINKS_AUTO = -1
_MIN_AUTO_CAP_LINKS = 3
CAP_RATIO_BOUNDS = (1.05, 1.35)

# var_estimator switch values (integer to leave room for future rules)
VAR_ESTIMATOR_MEAN = 0  # forgetting-weighted mean of segment estimates
VAR_ESTIMATOR_PESSIMISTIC = 1  # max over the recent segment estimates (default)
_KNOWN_VAR_ESTIMATORS = frozenset({VAR_ESTIMATOR_MEAN, VAR_ESTIMATOR_PESSIMISTIC})

# rolling window (in absorbed segments) of per-temperature variance
# estimates kept for the pessimistic estimator
VAR_HISTORY_LENGTH = 4

# pooled-cumulant temperature matching tolerance in |dlog T|: pool rows
# within this of a measured rung merge instead of spawning fresh rows.
# This lets a rung's history survive small rebuild-to-rebuild
# temperature drift.
POOL_DLOG_TOL = 0.02

# blocks discarded from the head of each ladder segment before its
# statistics enter the pool: the remap leaves every chain at a state
# equilibrated to its PREVIOUS temperature, so the first post-update
# block measures a transient.
DISCARD_BLOCKS_AFTER_UPDATE = 1

# the readout temperature the T_min_factor target is expressed against
_T_READOUT = 1.0


@dataclass
class LadderUpdateRecord:
    """One rebuild-evaluation history entry (recorded in the artifact).

    One row per cadence evaluation: applied updates and held
    sub-threshold rebuilds both appear, distinguished by `applied`, so
    E3 can see the hold pattern and the freeze decision even when
    nothing was applied. Ts is the candidate ladder either way.
    """

    block_index: int
    Ts: NDArray[np.floating]
    applied: bool
    t_cold_window: float
    max_dlog_t: float
    n_pool_points: int
    frozen_after: bool


@dataclass
class AdaptiveLadderController[LikelihoodType: AbstractLikelihood[NamedTuple]]:
    """Annealing-style adaptive ladder controller around DTMCMCSampler.

    Parameters (interface fixed by the plan)
    ----------
    mode: str
        'entropy' | 'length' | 'acceptance' — the spacing rule; internals
        are mode-agnostic so all three share the schedule
    update_every_blocks: int
        Ladder rebuild cadence in blocks (block-boundary updates, D6)
    forgetting: float
        Per-evaluation multiplicative down-weighting of previously
        pooled cumulants (0 = cumulative, the pilot default); applies
        to the weighted-mean merges — under the pessimistic
        var_estimator the rebuild variances use the rolling
        VAR_HISTORY_LENGTH window instead
    freeze_criterion: tuple[float, int]
        (max |dlog T| threshold, consecutive updates) — freeze
        eligibility once the rebuilt ladder moves less than the
        threshold this many evaluations in a row with the cold window at
        its target and at least one completed cold<->hot round trip in
        each of those cadence windows (the coupling witness, per-window
        form — strictly stronger than the plan's open-segment floor)
    T_min_factor: float
        Final cold-edge target as a multiple of the T=1 readout
        temperature. Values < 1 extend the ladder below the
        readout: the n_cold readout chains stay pinned at exactly T=1
        (located by index at every update via the arg_record machinery)
        while the sub-readout rungs participate in the same spacing rule
        as every other rung
    budget_blocks: int
        Hard adaptation cap in blocks, spec-owned and required: a run
        reaching it unfrozen hard-freezes with frozen_by='budget' so a
        post-freeze fixed-ladder segment still runs (plan Phase 5)
    var_estimator: int
        Pooled-variance rule feeding the rebuilds (integer switch to
        leave room for future rules): VAR_ESTIMATOR_PESSIMISTIC (1, the
        default) takes the max over the last VAR_HISTORY_LENGTH segment
        estimates at each pooled temperature; VAR_ESTIMATOR_MEAN (0) is
        the forgetting-weighted mean of all segment estimates
    window_extension_factor: float
        Cold-window extension factor per update (module default 4)
    ds_link_cap: float
        Per-link dS budget throttling window extension and setting the
        cold-edge cap ratio scale (module default 2.5 nats)
    cold_cap_links: int
        Number of coldest links the coupling cap applies to;
        COLD_CAP_LINKS_AUTO (-1, the default) scales with the ladder:
        max(3, n_links // 4)
    cap_ratio_bounds: tuple[float, float]
        (min, max) clip on the capped link temperature ratio
    var_history_length: int
        Rolling window (absorbed segments) of the pessimistic estimator
    pool_dlog_tol: float
        |dlog T| within which measured rungs merge into an existing pool
        row instead of spawning a fresh one, preserving variance history
        across rebuild-to-rebuild temperature drift
    discard_blocks_after_update: int
        Blocks discarded from the head of each ladder segment before its
        statistics pool (post-remap chains are transients of their old
        temperatures; must be < update_every_blocks)
    """

    mode: str = 'entropy'
    update_every_blocks: int = 8
    forgetting: float = 0.0
    freeze_criterion: tuple[float, int] = (0.02, 3)
    # no_remap preserves DE-buffer columns by slot on ladder updates and
    # lets each column re-burn-in under its new temperature; the cloning
    # rules ('at_or_hotter', 'nearest') are retained for tests of the old
    # behavior and for pilot A/Bs
    remap_rule: str = 'no_remap'
    T_min_factor: float = 1.0
    budget_blocks: int = -1
    var_estimator: int = VAR_ESTIMATOR_PESSIMISTIC
    n_prior_draws: int = 256
    n_inf_final: int = 1
    # evaluations with the window at target before freeze counting begins:
    # cold-end statistics need dwell time before stability is meaningful
    min_updates_at_target: int = 6
    window_extension_factor: float = WINDOW_EXTENSION_FACTOR
    ds_link_cap: float = DS_LINK_CAP
    cold_cap_links: int = COLD_CAP_LINKS_AUTO
    cap_ratio_bounds: tuple[float, float] = CAP_RATIO_BOUNDS
    var_history_length: int = VAR_HISTORY_LENGTH
    pool_dlog_tol: float = POOL_DLOG_TOL
    discard_blocks_after_update: int = DISCARD_BLOCKS_AFTER_UPDATE

    frozen: bool = field(default=False, init=False)
    frozen_by: str = field(default='', init=False)
    # get_loglike calls initial_ladder performed before the sampler (and its
    # EvalAccounting) existed; the harness folds them into the run counter
    prior_draw_evals: int = field(default=0, init=False)
    _updates_at_target: int = field(default=0, init=False)
    _pending_discard: int = field(default=0, init=False)
    history: list[LadderUpdateRecord] = field(default_factory=list, init=False)
    _pool_Ts: list[float] = field(default_factory=list, init=False)
    _pool_means: list[float] = field(default_factory=list, init=False)
    _pool_vars: list[float] = field(default_factory=list, init=False)
    _pool_weights: list[float] = field(default_factory=list, init=False)
    _pool_var_history: list[list[float]] = field(default_factory=list, init=False)
    _prior_anchor_retired: bool = field(default=False, init=False)
    _blocks_seen: int = field(default=0, init=False)
    _blocks_since_update: int = field(default=0, init=False)
    _t_cold_window: float = field(default=np.inf, init=False)
    _consecutive_small: int = field(default=0, init=False)
    _trips_at_prev_eval: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate the fixed interface."""
        if self.mode not in ADAPTIVE_MODES:
            msg = f'unknown adaptive mode {self.mode!r}; known: {sorted(ADAPTIVE_MODES)}'
            raise ValueError(msg)
        if self.update_every_blocks < 1 or not 0.0 <= self.forgetting < 1.0:
            msg = 'invalid adaptive controller parameters'
            raise ValueError(msg)
        if not 0.0 < self.T_min_factor <= 1.0:
            msg = 'T_min_factor must be in (0, 1]: the cold-edge target is a multiple of the T=1 readout'
            raise ValueError(msg)
        if self.budget_blocks < 1:
            msg = 'budget_blocks is required (spec-owned, plan Phase 5) and must be >= 1'
            raise ValueError(msg)
        if self.var_estimator not in _KNOWN_VAR_ESTIMATORS:
            msg = f'unknown var_estimator {self.var_estimator!r}; known: {sorted(_KNOWN_VAR_ESTIMATORS)}'
            raise ValueError(msg)
        if self.remap_rule not in {'at_or_hotter', 'nearest', 'no_remap'}:
            msg = f'unknown remap_rule {self.remap_rule!r}'
            raise ValueError(msg)
        if self.window_extension_factor <= 1.0:
            msg = 'window_extension_factor must be > 1'
            raise ValueError(msg)
        if self.ds_link_cap <= 0.0 or self.var_history_length < 1 or self.pool_dlog_tol < 0.0:
            msg = 'invalid adaptive controller parameters'
            raise ValueError(msg)
        if self.cold_cap_links < COLD_CAP_LINKS_AUTO:
            msg = 'cold_cap_links must be >= 0, or COLD_CAP_LINKS_AUTO (-1) for the chain-scaled default'
            raise ValueError(msg)
        if not 1.0 < self.cap_ratio_bounds[0] <= self.cap_ratio_bounds[1]:
            msg = 'cap_ratio_bounds must satisfy 1 < min <= max'
            raise ValueError(msg)
        if not 0 <= self.discard_blocks_after_update < self.update_every_blocks:
            msg = 'discard_blocks_after_update must be in [0, update_every_blocks): a segment must keep at least one block of statistics'
            raise ValueError(msg)

    @property
    def t_cold_target(self) -> float:
        """Final cold anchor of the annealing schedule."""
        return self.T_min_factor

    @property
    def frozen_block_index(self) -> int | None:
        """Block index at which adaptation froze, or None while still adapting.

        A criterion freeze stamps frozen_after on its history row; a
        budget freeze returns before recording a row, so it maps to
        budget_blocks (the block at which the hard cap tripped). Callers
        turn this into an adaptive-burn-in iteration count by multiplying
        by the block size — every iteration up to the freeze was spent
        tuning the ladder rather than sampling a fixed target.
        """
        if not self.frozen:
            return None
        for record in self.history:
            if record.frozen_after:
                return record.block_index
        return self.budget_blocks

    def initial_ladder(self, like_obj: LikelihoodType, n_chain: int, n_cold: int) -> TemperatureLadder:
        """Build the hot-anchored initial ladder from prior-draw logL statistics.

        Draws from the run streams (the controller runs inside the run,
        after seed_run) and evaluates the likelihood at each draw — the
        evals are charged to the run's counter like any other (C3 charges
        adaptive burn-in fully). The hot anchor is T_hot = 2 sd(logL
        under the prior); the initial cold edge sits three octaves below
        it, so the first ladder reaches only partway toward T=1 and the
        schedule extends it as data accumulates.
        """
        prior_logLs = np.array([like_obj.get_loglike(like_obj.prior_draw()) for _ in range(self.n_prior_draws)])
        self.prior_draw_evals += int(prior_logLs.size)
        prior_mean = float(prior_logLs.mean())
        prior_var = float(prior_logLs.var())

        t_hot = max(2.0 * np.sqrt(prior_var), 10.0)
        self._t_cold_window = max(t_hot / 8.0, self.t_cold_target)
        # the first segment starts from prior draws, a transient like any
        # post-update segment: discard its head blocks from the pool too
        self._pending_discard = self.discard_blocks_after_update

        # the prior statistics are the beta ~ 0 anchor of every rebuild
        self._pool_Ts.append(np.inf)
        self._pool_means.append(prior_mean)
        self._pool_vars.append(prior_var)
        self._pool_weights.append(1.0)
        self._pool_var_history.append([prior_var])

        return GeometricTemperatureLadder(
            n_chain=n_chain,
            n_cold=n_cold,
            T_cold=self._t_cold_window,
            T_min=self._t_cold_window,
            T_max=t_hot,
            n_inf_final=self.n_inf_final,
        )

    def _pool_match(self, T_loc: float) -> int | None:
        """Index of the pool row within pool_dlog_tol of T_loc in |dlog T|, or None.

        Tolerance matching is the pool's memory across rebuilds: a rung
        that drifts a few percent between applied updates keeps feeding
        the same row (and its variance history) instead of spawning a
        fresh length-1 row. Non-finite temperatures match only each
        other.
        """
        Ts_pool = np.asarray(self._pool_Ts)
        if Ts_pool.size == 0:
            return None
        finite_pool = np.isfinite(Ts_pool)
        if not np.isfinite(T_loc):
            matches = np.flatnonzero(~finite_pool)
            return int(matches[0]) if matches.size else None
        dlogs = np.full(Ts_pool.size, np.inf)
        dlogs[finite_pool] = np.abs(np.log(Ts_pool[finite_pool]) - np.log(T_loc))
        idx = int(np.argmin(dlogs))
        return idx if dlogs[idx] <= self.pool_dlog_tol else None

    def _absorb_segment_stats(self, sampler: DTMCMCSampler[LikelihoodType]) -> None:
        """Pool the current segment's per-chain logL cumulants.

        Uses the stationary estimator E[logL^2] - E[logL]^2 over the
        segment's blocks (all run on the current ladder, since absorption
        happens exactly once per segment, at update time). The head of a
        post-update segment is discarded (discard_blocks_after_update):
        the remap leaves chains equilibrated to their previous
        temperatures, and pooling that transient feeds non-stabilized
        statistics into the next rebuild.
        """
        start = self._blocks_seen + self._pending_discard
        e1_blocks = np.asarray(sampler.logL_means[start:])
        e2_blocks = np.asarray(sampler.logL2_means[start:])
        if e1_blocks.shape[0] == 0:
            return
        self._pending_discard = 0

        # once real measurements exist, retire the prior pseudo-anchor —
        # exactly once: its huge variance otherwise dominates the
        # hot-segment trapezoid and over-spaces the hottest link relative
        # to the measured profile. Measured beta=0 rows added below pool
        # across segments like any other temperature.
        if not self._prior_anchor_retired:
            idx_prior = self._pool_Ts.index(np.inf)
            for pool in (self._pool_Ts, self._pool_means, self._pool_vars, self._pool_weights, self._pool_var_history):
                pool.pop(idx_prior)
            self._prior_anchor_retired = True

        weight = float(e1_blocks.shape[0])
        seg_means = e1_blocks.mean(axis=0)
        seg_vars = np.maximum(e2_blocks.mean(axis=0) - seg_means**2, 0.0)
        self._blocks_seen = len(sampler.logL_means)

        if self.forgetting > 0.0:
            self._pool_weights = [w * (1.0 - self.forgetting) for w in self._pool_weights]

        for itrt in range(seg_means.size):
            T_loc = float(sampler.Ts[itrt])
            idx = self._pool_match(T_loc)
            if idx is not None:
                w_old, w_new = self._pool_weights[idx], weight
                total = w_old + w_new
                self._pool_means[idx] = (w_old * self._pool_means[idx] + w_new * float(seg_means[itrt])) / total
                self._pool_vars[idx] = (w_old * self._pool_vars[idx] + w_new * float(seg_vars[itrt])) / total
                self._pool_weights[idx] = total
                self._pool_var_history[idx].append(float(seg_vars[itrt]))
                del self._pool_var_history[idx][: -self.var_history_length]
            else:
                self._pool_Ts.append(T_loc)
                self._pool_means.append(float(seg_means[itrt]))
                self._pool_vars.append(float(seg_vars[itrt]))
                self._pool_weights.append(weight)
                self._pool_var_history.append([float(seg_vars[itrt])])

    def _effective_pool_vars(self) -> NDArray[np.floating]:
        """Pooled variances as the configured var_estimator reads them.

        Pessimistic (the default): max over each temperature's recent
        segment estimates. Mean: the forgetting-weighted running mean.
        """
        if self.var_estimator == VAR_ESTIMATOR_PESSIMISTIC:
            return np.asarray([max(history) for history in self._pool_var_history])
        return np.asarray(self._pool_vars)

    def _pool_keep_mask(self, window: float) -> NDArray[np.bool_]:
        """Pool rows entering a rebuild clipped at a cold window.

        Length mode additionally excludes beta=0 rows, matching the
        file-driven length-arm convention: the sqrt(Var) integrand is
        nonzero at beta=0, so an inf row would shift rungs hotter
        (see LengthTemperatureLadder), while the file arms never see one.
        """
        Ts_pool = np.asarray(self._pool_Ts)
        keep = Ts_pool >= window
        if self.mode == 'length':
            keep &= np.isfinite(Ts_pool)
        return keep

    def _pooled_ds_per_link(self, window: float, n_links: int) -> float:
        """Per-link dS of the pooled spacing integral clipped at a window."""
        Ts_pool = np.asarray(self._pool_Ts)
        vars_pool = self._effective_pool_vars()
        keep = self._pool_keep_mask(window)
        if int(np.count_nonzero(keep)) < 2:
            return 0.0
        p_exp, q_exp = (0.5, 0.0) if self.mode == 'length' else (1.0, 1.0)
        betas_use, vars_use = standardize_input_vars(Ts_to_betas(Ts_pool[keep]), vars_pool[keep])
        total = float(get_spacing_integrated(vars_use, betas_use, False, p=p_exp, q=q_exp)[-1])
        return total / n_links

    def _extend_window(self, n_chain: int, n_cold: int) -> None:
        """Descend the cold window as far as the per-link dS budget allows."""
        n_links = max(n_chain - n_cold - self.n_inf_final, 1)
        candidates = (
            self.window_extension_factor,
            *(c for c in _EXTENSION_CANDIDATES if c < self.window_extension_factor),
        )
        for factor in candidates:
            candidate = max(self.t_cold_target, self._t_cold_window / factor)
            if self._pooled_ds_per_link(candidate, n_links) <= self.ds_link_cap:
                self._t_cold_window = candidate
                return
        # even the smallest step blows the budget: hold and keep refining

    def _resolve_cap_links(self, n_chain: int, n_cold: int) -> int:
        """Resolve cold_cap_links, scaling the auto default with the ladder size."""
        if self.cold_cap_links != COLD_CAP_LINKS_AUTO:
            return self.cold_cap_links
        n_links = max(n_chain - n_cold - self.n_inf_final, 1)
        return max(_MIN_AUTO_CAP_LINKS, n_links // 4)

    def _cap_cold_links(self, ladder: TemperatureLadder, n_cold: int) -> TemperatureLadder:
        """Enforce the cold-edge coupling cap on a rebuilt ladder.

        Only the coldest resolved cold_cap_links non-trivial links are
        capped. The ratio for each capped link comes from the measured
        local heat capacity at its cold end and is clipped to
        cap_ratio_bounds. Duplicate rungs are zero-width links the cap
        skips, and a rung pinned at the ladder's T_cold anchor is never
        moved.
        """
        cap_links = self._resolve_cap_links(ladder.n_chain, n_cold)
        if cap_links == 0:
            return ladder
        Ts_pool = np.asarray(self._pool_Ts)
        vars_pool = self._effective_pool_vars()
        finite_pool = np.isfinite(Ts_pool)
        pool_order = np.argsort(Ts_pool[finite_pool])
        pool_T_sorted = Ts_pool[finite_pool][pool_order]
        pool_var_sorted = vars_pool[finite_pool][pool_order]

        def local_ratio_cap(T_at: float) -> float:
            var_local = float(np.interp(T_at, pool_T_sorted, pool_var_sorted))
            c_local = var_local / T_at**2
            # clip the exponent before exponentiating: a tiny measured C
            # would otherwise overflow exp() on its way to the ratio clip
            log_ratio = min(self.ds_link_cap / max(c_local, 1.0e-8), np.log(self.cap_ratio_bounds[1]) + 1.0)
            return float(np.clip(np.exp(log_ratio), *self.cap_ratio_bounds))

        Ts = np.sort(np.asarray(ladder.Ts).copy())
        finite = np.isfinite(Ts)
        finite_Ts = Ts[finite]
        capped = finite_Ts.copy()
        n_capped = 0
        for itrt in range(1, finite_Ts.size):
            if n_capped >= cap_links:
                break
            if capped[itrt] == capped[itrt - 1]:
                # zero-width duplicate link (e.g. the pinned cold block)
                continue
            if ladder.T_cold is not None and finite_Ts[itrt] == ladder.T_cold:
                # never move the readout pin — and the pinned link does not
                # consume a cap slot, so cold_cap_links counts capped links
                continue
            n_capped += 1
            max_allowed = capped[itrt - 1] * local_ratio_cap(float(capped[itrt - 1]))
            if capped[itrt] > max_allowed:
                capped[itrt] = max_allowed
        if np.allclose(capped, finite_Ts, rtol=1.0e-12):
            return ladder
        Ts[finite] = capped
        return TemperatureLadder(Ts, n_cold=n_cold, T_cold=ladder.T_cold)

    def _build_ladder(self, n_chain: int, n_cold: int) -> TemperatureLadder:
        """Rebuild the ladder from pooled cumulants over the extended window."""
        self._extend_window(n_chain, n_cold)

        Ts_pool = np.asarray(self._pool_Ts)
        means_pool = np.asarray(self._pool_means)
        vars_pool = self._effective_pool_vars()

        keep = self._pool_keep_mask(self._t_cold_window)
        Ts_use, means_use, vars_use = Ts_pool[keep], means_pool[keep], vars_pool[keep]
        if float(np.min(Ts_use[np.isfinite(Ts_use)], initial=np.inf)) > self._t_cold_window:
            # extend the cold edge with the coldest measured statistics so
            # rungs place there and the next segment measures it for real
            idx_coldest = int(np.argmin(np.where(np.isfinite(Ts_use), Ts_use, np.inf)))
            Ts_use = np.append(Ts_use, self._t_cold_window)
            means_use = np.append(means_use, means_use[idx_coldest])
            vars_use = np.append(vars_use, vars_use[idx_coldest])

        # The readout chains pin at T=1 once the window descends past it.
        # Colder rungs then extend to the window edge under the same
        # spacing rule as every other rung.
        t_cold_build = max(self._t_cold_window, _T_READOUT)
        # snap_mode 1: the cold plug must not consume the sole sub-readout
        # rung; identical to nearest whenever no rung sits below T_cold.
        ladder: TemperatureLadder
        if self.mode == 'entropy':
            ladder = EntropyTemperatureLadder(
                n_chain, Ts_use, vars_use, n_cold=n_cold, T_cold=t_cold_build, n_inf_final=self.n_inf_final, snap_mode=1
            )
        elif self.mode == 'length':
            ladder = LengthTemperatureLadder(
                n_chain, Ts_use, vars_use, n_cold=n_cold, T_cold=t_cold_build, n_inf_final=self.n_inf_final, snap_mode=1
            )
        else:
            ladder = AcceptanceTemperatureLadder(
                n_chain,
                Ts_use,
                means_use,
                vars_use,
                n_cold=n_cold,
                T_cold=t_cold_build,
                n_inf_final=self.n_inf_final,
                snap_mode=1,
            )
        return self._cap_cold_links(ladder, n_cold)

    def post_block(self, sampler: DTMCMCSampler[LikelihoodType]) -> bool:
        """Advance the schedule after a block; returns True when the ladder updated.

        Sub-threshold rebuilds (max |dlog T| within the freeze
        threshold) are held, not applied — no remap, no tracker
        segmentation — so segments lengthen as adaptation converges and
        the coupling-witness clock is never reset by a rebuild that
        changed nothing (plan D6/Phase 5). The hard freeze requires the
        stability criterion with a completed round trip in each of its
        cadence windows (which implies the plan's open-segment witness
        floor); exhausting budget_blocks freezes unconditionally with
        the reason recorded.
        """
        if self.frozen:
            return False
        blocks_done = sampler.itrn // sampler.block_size
        if blocks_done >= self.budget_blocks:
            self.frozen = True
            self.frozen_by = 'budget'
            return False
        self._blocks_since_update += 1
        if self._blocks_since_update < self.update_every_blocks:
            return False
        self._blocks_since_update = 0

        self._absorb_segment_stats(sampler)
        at_target_before = self._t_cold_window <= self.t_cold_target
        new_ladder = self._build_ladder(sampler.n_chain, sampler.n_cold)

        old_finite = np.asarray(sampler.Ts)[np.isfinite(sampler.Ts)]
        new_finite = np.asarray(new_ladder.Ts)[np.isfinite(new_ladder.Ts)]
        if old_finite.size == new_finite.size:
            max_dlog = float(np.max(np.abs(np.log(new_finite) - np.log(old_finite))))
        else:
            max_dlog = np.inf

        # Coupling witness: read segment-reset cycle counters before any
        # ladder update resets them. The freeze streak requires new
        # completed trips in every cadence window it spans.
        n_trips_open = int(np.sum(sampler.tracker_manager.n_cycles))
        window_trips = n_trips_open - self._trips_at_prev_eval

        dlog_thresh, n_consecutive = self.freeze_criterion
        applied = max_dlog >= dlog_thresh
        if applied:
            sampler.apply_ladder_update(new_ladder, self.remap_rule)
            # segmentation reset the cycle counters with the segment
            self._trips_at_prev_eval = 0
            # the new segment starts as a remap transient: discard its
            # head blocks from the next absorption
            self._pending_discard = self.discard_blocks_after_update
        else:
            self._trips_at_prev_eval = n_trips_open

        # Freeze bookkeeping: only witnessed holds with the window already
        # at its target, and after a minimum dwell there, count toward the
        # stability criterion.
        if at_target_before:
            self._updates_at_target += 1
        if (
            at_target_before
            and self._updates_at_target > self.min_updates_at_target
            and not applied
            and window_trips >= 1
        ):
            self._consecutive_small += 1
        else:
            self._consecutive_small = 0

        if self._consecutive_small >= n_consecutive and n_trips_open >= 1:
            self.frozen = True
            self.frozen_by = 'criterion'

        self.history.append(
            LadderUpdateRecord(
                block_index=blocks_done,
                Ts=np.asarray(new_ladder.Ts).copy(),
                applied=applied,
                t_cold_window=self._t_cold_window,
                max_dlog_t=max_dlog,
                n_pool_points=len(self._pool_Ts),
                frozen_after=self.frozen,
            )
        )
        return applied
