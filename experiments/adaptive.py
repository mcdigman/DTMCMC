"""Adaptive burn-in ladder controller (methods-paper plan §4 Phase 5).

The controller wraps a DTMCMCSampler and mutates engine state only
through the DTMCMCSampler.apply_ladder_update hook. Interface fixed by
the plan; internals are exploratory and start from the Phase 4 pilot
knobs (update every 8 blocks, no forgetting, freeze at max |dlog T| <
2% over 3 consecutive updates, at-or-hotter DE remap).

Annealing schedule: the initial ladder is anchored at the hot end from
prior-draw logL statistics; each update extends the cold edge of the
spacing window by a fixed factor toward the target (optionally below
T=1 via T_min_factor, plan S2) and rebuilds the ladder from the pooled
first two logL cumulants measured so far — mode-agnostic, so the
entropy, length, and acceptance spacing rules share one schedule.
Adaptation ends in a hard freeze; afterwards the code path is identical
to a fixed-ladder sampler (the hook is never called again).
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    from experiments.harness.runner import LikelihoodLike

ADAPTIVE_MODES: frozenset[str] = frozenset({'entropy', 'length', 'acceptance'})

# cold-window extension factor per update: each rebuild reaches up to 4x
# colder until the target is hit, then refinement continues in place
WINDOW_EXTENSION_FACTOR = 4.

# per-link dS budget throttling the extension: the window only descends
# while the pooled spacing integral stays within ds_link_cap nats per
# link, so the cold edge is always swap-coupled to the ladder above it.
# Without the throttle the schedule is self-blocking on phase
# transitions: an over-wide coldest link blocks walker descent, the
# transition variance is never measured, and the coarse ladder
# self-perpetuates (observed on cake 5D: a ~7-nat coldest link froze a
# log-uniform ladder that never resolved the transition cluster)
DS_LINK_CAP = 2.5

# candidate partial extensions tried when the full factor blows the budget
_EXTENSION_CANDIDATES = (4., 3., 2., 1.5, 1.25, 1.1)

# coupling insurance at the cold edge: pooled statistics cannot know the
# entropy of a region the cold chain has never mixed through, so an
# equal-dS rebuild can starve its own coldest link (the walkers stop
# descending and the missing variance is never measured; observed as a
# self-perpetuating log-uniform ladder on cake 5D). The coldest
# N_COLD_CAPPED_LINKS links are pulled to at most exp(DS_LINK_CAP / C)
# in temperature ratio, with C the measured local heat capacity
# beta^2 Var(logL) at the link's cold end — clipped to never exceed the
# pessimistic default (a loose cap when C is under-measured would defeat
# the insurance exactly when it is needed: the cap forces coupling, the
# coupling produces the variance data, and the data then tightens the
# cap into a discovered transition
N_COLD_CAPPED_LINKS = 3
_CAP_RATIO_BOUNDS = (1.05, 1.35)


@dataclass
class LadderUpdateRecord:
    """One ladder-update history entry (recorded in the artifact)."""

    block_index: int
    Ts: NDArray[np.floating]
    t_cold_window: float
    max_dlog_t: float
    n_pool_points: int
    frozen_after: bool


@dataclass
class AdaptiveLadderController:
    """Annealing-style adaptive ladder controller around DTMCMCSampler.

    Parameters (interface fixed by the plan)
    ----------
    mode: str
        'entropy' | 'length' | 'acceptance' — the spacing rule; internals
        are mode-agnostic so all three share the schedule
    update_every_blocks: int
        Ladder rebuild cadence in blocks (block-boundary updates, D6)
    forgetting: float
        Per-update multiplicative down-weighting of previously pooled
        cumulants (0 = cumulative, the pilot default)
    freeze_criterion: tuple[float, int]
        (max |dlog T| threshold, consecutive updates) — hard freeze once
        the rebuilt ladder moves less than the threshold this many
        updates in a row with the cold window at its target
    T_min_factor: float
        Final cold-edge target as a multiple of T=1 (values < 1 place
        rungs slightly below the readout temperature, plan S2)
    """

    mode: str = 'entropy'
    update_every_blocks: int = 8
    forgetting: float = 0.
    freeze_criterion: tuple[float, int] = (0.02, 3)
    T_min_factor: float = 1.
    n_prior_draws: int = 256
    n_inf_final: int = 1
    # updates with the window at target before freeze counting begins:
    # cold-end statistics need dwell time before stability is meaningful
    min_updates_at_target: int = 6

    frozen: bool = field(default=False, init=False)
    _updates_at_target: int = field(default=0, init=False)
    history: list[LadderUpdateRecord] = field(default_factory=list, init=False)
    _pool_Ts: list[float] = field(default_factory=list, init=False)
    _pool_means: list[float] = field(default_factory=list, init=False)
    _pool_vars: list[float] = field(default_factory=list, init=False)
    _pool_weights: list[float] = field(default_factory=list, init=False)
    _blocks_seen: int = field(default=0, init=False)
    _blocks_since_update: int = field(default=0, init=False)
    _t_cold_window: float = field(default=np.inf, init=False)
    _consecutive_small: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        """Validate the fixed interface."""
        if self.mode not in ADAPTIVE_MODES:
            msg = f'unknown adaptive mode {self.mode!r}; known: {sorted(ADAPTIVE_MODES)}'
            raise ValueError(msg)
        if self.update_every_blocks < 1 or not 0. <= self.forgetting < 1. or self.T_min_factor <= 0.:
            msg = 'invalid adaptive controller parameters'
            raise ValueError(msg)

    @property
    def t_cold_target(self) -> float:
        """Final cold anchor of the annealing schedule."""
        return self.T_min_factor

    def initial_ladder(self, like_obj: LikelihoodLike, n_chain: int, n_cold: int) -> TemperatureLadder:
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
        prior_mean = float(prior_logLs.mean())
        prior_var = float(prior_logLs.var())

        t_hot = max(2. * np.sqrt(prior_var), 10.)
        self._t_cold_window = max(t_hot / 8., self.t_cold_target)

        # the prior statistics are the beta ~ 0 anchor of every rebuild
        self._pool_Ts.append(np.inf)
        self._pool_means.append(prior_mean)
        self._pool_vars.append(prior_var)
        self._pool_weights.append(1.)

        return GeometricTemperatureLadder(
            n_chain, n_cold=n_cold, T_cold=self._t_cold_window, T_min=self._t_cold_window,
            T_max=t_hot, n_inf_final=self.n_inf_final,
        )

    def _absorb_segment_stats(self, sampler: DTMCMCSampler) -> None:
        """Pool the current segment's per-chain logL cumulants.

        Uses the stationary estimator E[logL^2] - E[logL]^2 over the
        segment's blocks (all run on the current ladder, since absorption
        happens exactly once per segment, at update time).
        """
        e1_blocks = np.asarray(sampler.logL_means[self._blocks_seen:])
        e2_blocks = np.asarray(sampler.logL2_means[self._blocks_seen:])
        if e1_blocks.shape[0] == 0:
            return

        # once real measurements exist, retire the prior pseudo-anchor: its
        # huge variance otherwise dominates the hot-segment trapezoid and
        # over-spaces the hottest link relative to the measured profile
        if np.inf in self._pool_Ts:
            idx_prior = self._pool_Ts.index(np.inf)
            for pool in (self._pool_Ts, self._pool_means, self._pool_vars, self._pool_weights):
                pool.pop(idx_prior)

        weight = float(e1_blocks.shape[0])
        seg_means = e1_blocks.mean(axis=0)
        seg_vars = np.maximum(e2_blocks.mean(axis=0) - seg_means**2, 0.)
        self._blocks_seen = len(sampler.logL_means)

        if self.forgetting > 0.:
            self._pool_weights = [w * (1. - self.forgetting) for w in self._pool_weights]

        for itrt in range(seg_means.size):
            T_loc = float(sampler.Ts[itrt])
            if T_loc in self._pool_Ts:
                idx = self._pool_Ts.index(T_loc)
                w_old, w_new = self._pool_weights[idx], weight
                total = w_old + w_new
                self._pool_means[idx] = (w_old * self._pool_means[idx] + w_new * float(seg_means[itrt])) / total
                self._pool_vars[idx] = (w_old * self._pool_vars[idx] + w_new * float(seg_vars[itrt])) / total
                self._pool_weights[idx] = total
            else:
                self._pool_Ts.append(T_loc)
                self._pool_means.append(float(seg_means[itrt]))
                self._pool_vars.append(float(seg_vars[itrt]))
                self._pool_weights.append(weight)

    def _pooled_ds_per_link(self, window: float, n_links: int) -> float:
        """Per-link dS of the pooled spacing integral clipped at a window."""
        Ts_pool = np.asarray(self._pool_Ts)
        vars_pool = np.asarray(self._pool_vars)
        keep = Ts_pool >= window
        if int(np.count_nonzero(keep)) < 2:
            return 0.
        p_exp, q_exp = (0.5, 0.) if self.mode == 'length' else (1., 1.)
        betas_use, vars_use = standardize_input_vars(Ts_to_betas(Ts_pool[keep]), vars_pool[keep])
        total = float(get_spacing_integrated(vars_use, betas_use, False, p=p_exp, q=q_exp)[-1])
        return total / n_links

    def _extend_window(self, n_chain: int, n_cold: int) -> None:
        """Descend the cold window as far as the per-link dS budget allows."""
        n_links = max(n_chain - n_cold - self.n_inf_final, 1)
        for factor in _EXTENSION_CANDIDATES:
            candidate = max(self.t_cold_target, self._t_cold_window / factor)
            if self._pooled_ds_per_link(candidate, n_links) <= DS_LINK_CAP:
                self._t_cold_window = candidate
                return

    def _cap_cold_links(self, ladder: TemperatureLadder, n_cold: int) -> TemperatureLadder:
        """Enforce the cold-edge coupling cap on a rebuilt ladder.

        Only the coldest N_COLD_CAPPED_LINKS links are capped (capping
        every link would flatten the ladder and destroy hot coverage);
        the ratio for each capped link comes from the measured local heat
        capacity at its cold end, so the cap squeezes progressively as
        the transition variance is discovered. No-op once measured
        statistics pack the cold links tighter than the budget.
        """
        Ts_pool = np.asarray(self._pool_Ts)
        vars_pool = np.asarray(self._pool_vars)
        finite_pool = np.isfinite(Ts_pool)
        pool_order = np.argsort(Ts_pool[finite_pool])
        pool_T_sorted = Ts_pool[finite_pool][pool_order]
        pool_var_sorted = vars_pool[finite_pool][pool_order]

        def local_ratio_cap(T_at: float) -> float:
            var_local = float(np.interp(T_at, pool_T_sorted, pool_var_sorted))
            c_local = var_local / T_at**2
            ratio = float(np.exp(DS_LINK_CAP / max(c_local, 1.e-8)))
            return float(np.clip(ratio, *_CAP_RATIO_BOUNDS))

        Ts = np.sort(np.asarray(ladder.Ts).copy())
        finite = np.isfinite(Ts)
        finite_Ts = Ts[finite]
        capped = finite_Ts.copy()
        for itrt in range(n_cold, min(n_cold + N_COLD_CAPPED_LINKS, finite_Ts.size)):
            max_allowed = capped[itrt - 1] * local_ratio_cap(float(capped[itrt - 1]))
            if capped[itrt] > max_allowed:
                capped[itrt] = max_allowed
        if np.allclose(capped, finite_Ts, rtol=1.e-12):
            return ladder
        Ts[finite] = capped
        return TemperatureLadder(n_cold, Ts)
        # even the smallest step blows the budget: hold and keep refining

    def _build_ladder(self, n_chain: int, n_cold: int) -> TemperatureLadder:
        """Rebuild the ladder from pooled cumulants over the extended window."""
        self._extend_window(n_chain, n_cold)

        Ts_pool = np.asarray(self._pool_Ts)
        means_pool = np.asarray(self._pool_means)
        vars_pool = np.asarray(self._pool_vars)

        keep = Ts_pool >= self._t_cold_window
        Ts_use, means_use, vars_use = Ts_pool[keep], means_pool[keep], vars_pool[keep]
        if float(np.min(Ts_use[np.isfinite(Ts_use)], initial=np.inf)) > self._t_cold_window:
            # extend the cold edge with the coldest measured statistics so
            # rungs place there and the next segment measures it for real
            idx_coldest = int(np.argmin(np.where(np.isfinite(Ts_use), Ts_use, np.inf)))
            Ts_use = np.append(Ts_use, self._t_cold_window)
            means_use = np.append(means_use, means_use[idx_coldest])
            vars_use = np.append(vars_use, vars_use[idx_coldest])

        t_cold_build = self._t_cold_window
        ladder: TemperatureLadder
        if self.mode == 'entropy':
            ladder = EntropyTemperatureLadder(n_chain, Ts_use, vars_use, n_cold=n_cold, T_cold=t_cold_build, n_inf_final=self.n_inf_final)
        elif self.mode == 'length':
            ladder = LengthTemperatureLadder(n_chain, Ts_use, vars_use, n_cold=n_cold, T_cold=t_cold_build, n_inf_final=self.n_inf_final)
        else:
            ladder = AcceptanceTemperatureLadder(n_chain, Ts_use, means_use, vars_use, n_cold=n_cold, T_cold=t_cold_build, n_inf_final=self.n_inf_final)
        return self._cap_cold_links(ladder, n_cold)

    def post_block(self, sampler: DTMCMCSampler) -> bool:
        """Advance the schedule after a block; returns True when the ladder updated."""
        if self.frozen:
            return False
        self._blocks_since_update += 1
        if self._blocks_since_update < self.update_every_blocks:
            return False

        self._absorb_segment_stats(sampler)
        at_target_before = self._t_cold_window <= self.t_cold_target
        new_ladder = self._build_ladder(sampler.n_chain, sampler.n_cold)

        old_finite = np.asarray(sampler.Ts)[np.isfinite(sampler.Ts)]
        new_finite = np.asarray(new_ladder.Ts)[np.isfinite(new_ladder.Ts)]
        if old_finite.size == new_finite.size:
            max_dlog = float(np.max(np.abs(np.log(new_finite) - np.log(old_finite))))
        else:
            max_dlog = np.inf

        sampler.apply_ladder_update(new_ladder, 'at_or_hotter')
        self._blocks_since_update = 0

        # hard-freeze bookkeeping: only refinements with the window already
        # at its target, and after a minimum dwell there, count toward the
        # criterion — cold-end statistics need time before stability means
        # convergence rather than starvation
        if at_target_before:
            self._updates_at_target += 1
        dlog_thresh, n_consecutive = self.freeze_criterion
        if at_target_before and self._updates_at_target > self.min_updates_at_target and max_dlog < dlog_thresh:
            self._consecutive_small += 1
        else:
            self._consecutive_small = 0
        if self._consecutive_small >= n_consecutive:
            self.frozen = True

        self.history.append(LadderUpdateRecord(
            block_index=sampler.itrn // sampler.block_size,
            Ts=np.asarray(new_ladder.Ts).copy(),
            t_cold_window=self._t_cold_window,
            max_dlog_t=max_dlog,
            n_pool_points=len(self._pool_Ts),
            frozen_after=self.frozen,
        ))
        return True
