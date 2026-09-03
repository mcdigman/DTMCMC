# Phase 4 Pilot Report — production calibration

- **Date:** 2026-07-04 (revised 2026-07-06 per PR #15 review; knee-scan and remap runs
  reproduced bit-identically on regeneration, NN-KL recalibrated at 16 repeats with
  eggbox 5D)
- **Machine:** Apple silicon (14 cores), single-core runs, `DTMCMC-dev` env
- **Pilot code:** `experiments/pilots/` (this PR); every number below regenerates via
  `python -m experiments.pilots.<module>` (summaries land in `artifacts/pilots/*.json`).
- **Doc role:** fixes every number the Phase 6 production battery needs (plan §4 Phase 4,
  §10). Frozen items are marked **[FROZEN]**; they may not change after this report merges
  (plan D8).

## 1. Throughput (best-of-3, single core, chain-steps/s)

| likelihood | n_chain=8 | 16 | 48 |
|---|---|---|---|
| gaussian 5D  | 4.97e5 | 5.10e5 | 5.23e5 |
| gaussian 20D | 3.14e5 | 3.15e5 | 3.22e5 |
| cake 5D      | 4.67e5 | 4.81e5 | 4.95e5 |
| cake 8D      | 4.31e5 | 4.40e5 | 4.55e5 |
| eggbox 5D    | 5.05e5 | 4.96e5 | 5.18e5 |

Per-step cost is flat in `n_chain` (per-chain-step normalization holds). Quote arithmetic
below uses the per-likelihood n_chain=16 column.

**Bug found and fixed en route:** `eggbox.prior_factor`'s `del`-based body raised a numba
`TypingError` on every call — the eggbox jitclass had *never* run end-to-end through the
default proposal mixture (prior jumps evaluate `prior_factor` per proposal). Two-line fix +
end-to-end regression test in this PR; E2/E3 depend on it.

## 2. Knee scan (cake 5D, gold entropy ladder, 3 seeds × 2.6e5 iterations)

Total gold entropy span over the gold input range (T ≥ 1): **S_total = 11.72 nats**.

ΔS/link convention: ΔS/link = S_total/(n_chain − 1). The spaced-ladder construction
(`entropy_spacing` + `_plug_cold_and_inf`) divides the gold span into n_chain − 1
equal-entropy links and then overwrites the hottest rung with T=∞, so adjacent *finite*
rungs are spaced by exactly this increment and the final finite→∞ link carries the last
slice of the gold span (verified against the constructed ladders: at n_chain=16 the 15
finite rungs sit at exactly k × 0.782 nats, k = 0…14). The denominator is the link count
n_chain − 1, not the finite-rung count.

| n_chain | ΔS/link | RT rate (per walker per 1e6 chain-steps) |
|---|---|---|
| 6  | 2.34 | 197.5 ± 14.7 |
| 8  | 1.67 | 146.3 ± 3.5 |
| 10 | 1.30 | 137.8 ± 26.8 |
| 12 | 1.07 | 115.7 ± 12.6 |
| 16 | 0.78 | 77.2 ± 4.6 |
| 24 | 0.51 | 41.9 ± 2.1 |
| 32 | 0.38 | 29.1 ± 2.1 |

The rate is monotone decreasing (more rungs to traverse). Each candidate estimator is fit
*independently on each axis* — a fitted breakpoint is not invariant under the nonlinear
n_chain → ΔS/link reparameterization, so the two axes' knees are separate numbers, not
restatements of one location (seed-bootstrap sds, 500 resamples, seed 314, recorded in
`knee_scan.json`):

| fit | n_chain axis | ΔS/link axis |
|---|---|---|
| piecewise-linear | 16.0 (sd 2.8) | 1.07 (sd 0.16) |
| max-curvature    |  8.0 (sd 0.8) | 1.30 (sd 0.32) |

On the collapse axis — the C2 variable — the two estimators agree within ~1σ that the
knee sits at **ΔS/link ≈ 1.1–1.3**, consistent with the plan's expected ΔS/link ≲ 1 (C2).
The n_chain-axis piecewise-linear knee of 16.0 (which corresponds to ΔS/link 0.78) is the
same estimator on a different axis, reported for production grid placement only; an
earlier revision of this report incorrectly presented the two fits as one location.

**[FROZEN] Knee estimator: piecewise-linear** (`fit_knee_piecewise_linear`), applied
uniformly per axis; C2's collapse statistic compares between-likelihood knee dispersion
under the ΔS/link vs raw n_chain axes (plan §7). Rationale: it lands mid-grid rather than
at the second grid point (max-curvature's n_chain-axis 8.0 is a grid-edge affinity on a
monotone-decaying curve), it has a direct two-regime interpretation matching C2's
regime-statement framing, and its ΔS/link-axis knee 1.07 ± 0.16 agrees with theory.
Bootstrap sd 2.8 in n_chain (~18%) is adequate for grid placement. Production n_chain
grid {8, 12, 16, 24, 32, 48} brackets the knee well — unchanged from §5.

## 3. Arm pilot (cake 5D, n_chain=16, 8 paired seeds × 2.6e5 iterations)

| arm | RT rate | n_eff/eval | RTs/run (post-burn) |
|---|---|---|---|
| entropy           | 75.8 ± 5.9   | 2.88e-3 | 2542 |
| length            | 108.1 ± 21.9 | 2.61e-3 | 3626 |
| acceptance        | 109.8 ± 17.8 | 2.55e-3 | 3684 |
| geometric-default | 37.7 ± 25.9  | 1.33e-3 | 1264 |
| geometric-tuned   | 106.5 ± 35.0 | 2.43e-3 | 3575 |

**[FROZEN] E1 geometric-tuned arm definition:** geometric spacing from T=1 to the gold
entropy ladder's hottest finite rung at the same n_chain (T_max = 2058.7 at n_chain=16),
n_inf_final=1. Data-driven, reproducible, and clearly stronger than the default (RT 106.5
vs 37.7) — a fair non-straw comparison arm.

Substantive preview for C1: **entropy wins n_eff-per-eval against every arm** at this grid
point, but length/acceptance/tuned-geometric beat it on raw RT rate. Under the §7
conjunction rule, full C1 superiority at n_chain=16 holds only against geometric-default —
the production sweep must establish *where* (in n_chain / ΔS-per-link) the conjunction
holds, which is precisely the §7 regime-statement design. This is a feature of the
pre-registration, not a problem: each co-primary is also reported separately.

The acceptance arm's pilot inputs are the `cake1` (Ts, means, vars) triple (the gold set
lacks a means file); E9 regenerates matched gold cumulants for production so all arms
share identical inputs.

## 4. Run lengths

Long-run stationarity (entropy, n_chain=16, 1.05e6 iterations): post-burn rate 75.7;
window rates 72.3 (25–50%) vs 75.7 (50–100%) — stationary within ~5%.

Steps for ≥100 post-burn round trips (→ ≤10% relative RT-rate error), from pilot rates at
n_chain=16: acceptance 3.6e3, length 3.6e3, geometric-tuned 3.7e3, entropy 5.2e3,
geometric-default 1.0e4. Even scaled to n_chain=48 (rate ∝ ~1/n_chain² per walker,
compensated by more walkers) every arm clears 100 RTs within ~2e5 iterations.

**[FROZEN] Production run lengths:** E1/E2 at **2×10⁶ iterations** (down from the §5
placeholder 5×10⁶: the RT-precision criterion is met with ≥20× margin, and the n_eff and
NN-KL precision at 2e6 already exceed what the seed counts resolve). E3 unchanged at
2×10⁶; E9 gold references unchanged at 5×10⁷ (they serve evidence precision, not RT
counting).

**[FROZEN] Burn-in fraction: 25%**, identical across arms (long-run window agreement above
justifies it; pilots used 50% conservatively).

## 5. Seed counts (power analysis from paired differences)

95% CI half-width ≤ effect/3 requires (worst metric per pair): entropy-vs-default 18,
entropy-vs-length 27, entropy-vs-tuned 28, entropy-vs-acceptance **62** (its n_eff effect
is the smallest observed).

**[FROZEN] E1: 32 paired seeds** (covers every pair except entropy-vs-acceptance's n_eff
co-primary, which lands at ~2.2× the target half-width). Rather than tripling the whole
battery for one pair, the entropy and acceptance arms get a **top-up to 64 seeds** (those
two arms only). E2: 16 seeds (entropy + geometric arms only, larger effects); E3/E7: 16;
E4/E8: 10, unchanged from §5 placeholders.

## 6. DE-buffer remap A/B (D6 default for Phase 5)

Transplant experiment (coarse geometric → gold entropy ladder mid-run, n_chain=12, 4
seeds): blocks to settle the cold-chain logL band, higher is worse:

| rule | settle blocks | post-switch RTs | early accepted swaps/block |
|---|---|---|---|
| **at-or-hotter (D6)** | **5.5** | 1231 | 4066 |
| nearest               | 18.2    | 1167 | 4013 |
| partial reset         | 35.5    | 1207 | 4046 |

(An earlier revision reported 2× the swap counts: with `track_full_exchanges=False` the
engine increments both participants' tracker slots per accepted swap, and the pilot
summed the plane without halving. Rule comparisons were unaffected — all arms doubled
equally.)

**[FROZEN] Remap default: at-or-hotter** — 3.3× faster settling than nearest and 6.5×
faster than partial reset, directly confirming D6's "DE recovers much faster from
overdispersion than underdispersion". Phase 5 initial schedule knobs (to iterate, not
frozen): update every 8 blocks; no forgetting initially (cumulative moments); freeze when
max |Δlog T| < 2% across 3 consecutive updates.

## 7. NN-KL snapshot size and cadence

Self-KL noise (16 repeats, exact references; `nn_kl` timed alone after a JIT warm-up
call, reference-draw cost excluded — an earlier revision timed draws + first-call
compilation, which is why its cost column was non-monotonic; eggbox is now measured at
the production dimension, 5D, rather than proxied by 2D):

| n | gaussian 5D sd | cake 5D sd | eggbox 5D sd | nn_kl s/eval (worst) |
|---|---|---|---|---|
| 1000  | 0.08 | 0.36 | 0.38 | 0.005 |
| 2000  | 0.05 | 0.22 | 0.24 | 0.02 |
| 5000  | 0.05 | 0.17 | 0.07 | 0.11 |
| 10000 | 0.03 | **0.08** | 0.02 | 0.44 |

**[FROZEN] Snapshot size 10⁴, evaluated post-hoc at checkpoint cadence (4 per run).**
Cake's two-scale geometry sets the noise floor at the frozen snapshot size (eggbox 5D is
noisier at small n but drops below cake by n=5000); C3's NN-KL threshold must sit well
above it: **threshold 0.5** (≈6.5σ of the worst n=10⁴ noise floor, and 3.3× the largest
observed |self-KL| at that size, 0.15) for "quality reached". Cost ~1.8 s per run of
analysis (4 checkpoints × 0.44 s) — negligible.

## 8. Go/no-go decisions (§10 residuals)

- **Feedback-ladder arm: NO-GO.** The predicted-acceptance ladder is demonstrably not a
  strawman — it *beat* entropy on raw RT rate at the pilot point. A measured-acceptance
  variant adds engine surface without sharpening any claim.
- **S5/E10 (duplicate temperatures): NO-GO for the production battery.** Requires hot-set
  cycle semantics (engine change) for a secondary demonstration; the battery is already
  rich, and the pilot arm structure gives C1–C4 everything pre-registered. Revisit
  post-paper.
- **S2 cold-edge (T<1 rungs) promotion into E1: NO.** Stays in E7 as exploratory; E1's
  five arms are already the paper's core comparison.

## 9. Production run matrix and CPU-hour quote

Using measured throughput, 2×10⁶-iteration E1/E2 runs, and the frozen seed counts
(Σ n_chain over the grid = 140):

| ID | design | chain-steps | CPU-h |
|---|---|---|---|
| E1 | cake 5D, 5 arms × grid × 32 seeds ×2e6 | 4.5e10 | 26 |
| E1b | acceptance-pair top-up: 2 arms × grid × 32 extra seeds | 1.8e10 | 10 |
| E2 | {cake 8D, eggbox 5D, gauss 5D, gauss 20D} × 2 arms × grid × 16 seeds ×2e6 | 3.6e10 | 23 |
| E3 | 5 arms × {cake 5D, eggbox} × 16 seeds ×2e6, n_chain 16 | 5.1e9 | 3 |
| E4 | ESD/acceptance overlays, 3 likelihoods × 3 densities × 10 seeds | 3.1e9 | 2 |
| E5 | mixture ablations + collapse/recovery | ~2e9 | 1.5 |
| E6 | gaussian n_par {2,8,32,64} sweeps | ~3e9 | 3 |
| E7 | 6 exchange strategies × 2 ladders × 16 seeds | 6.1e9 | 3.5 |
| E8 | de_size × de_thin grid | ~4e9 | 2.5 |
| E9 | gold references (5e7 iterations, cake 5D/8D) | 1.3e10 | 8 |
| E10 | — (no-go) | — | 0 |

**Total ≈ 82 CPU-h central; ≤ 125 CPU-h with 1.5× contingency** — comfortably inside the
plan's 300 CPU-h worst case; an overnight job on a 16-core node, embarrassingly parallel
via the Phase 1 batch manifests.

## 10. Reproduction

```
python -m experiments.pilots.throughput
python -m experiments.pilots.knee_scan
python -m experiments.pilots.arm_power
python -m experiments.pilots.run_length     # needs arm_power.json
python -m experiments.pilots.remap_ab
python -m experiments.pilots.nnkl_calibration
```

Each module writes `artifacts/pilots/<name>.json`; artifacts are gitignored, the numbers
above are transcribed from the 2026-07-06 session summaries (knee-scan and remap sampler
runs reproduced bit-identically against the 2026-07-04 originals; analysis seeds are
recorded in the summary JSONs).
