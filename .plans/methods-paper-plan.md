# DTMCMC Methods Paper: Implementation & Experiment Plan

- **Date:** 2026-07-03
- **Branch:** `experimental`
- **Status:** Final approved draft (incorporates PR #8 review rounds 1–2) — Approved to begin Phase 1.
- **Doc role:** the contract for phased implementation. Each phase is executed in its own
  session/PR against this document. Items marked **[EXPLORATORY]** fix an interface and a
  success criterion but deliberately leave internals to be iterated against pilot runs;
  everything else is a hard contract with acceptance criteria.

## 1. Scope: claims and demonstrations

Primary claims (each gets a dedicated experiment and pre-registered statistic, §7):

- **C1 — Ladder optimality.** The variance-based equal-entropy ladder outperforms both a
  power-law (geometric) ladder and an equal-acceptance ladder on round-trip rate and
  effective-samples-per-likelihood-evaluation, at matched chain count and compute budget,
  on likelihoods with phase transitions.
- **C2 — Adiabatic transition.** Sampler quality undergoes a nonlinear transition as a
  function of chain density per nat of entropy (ΔS per ladder link); the transition curves
  for different likelihoods/dimensions collapse when plotted against ΔS/link (expected knee
  at ΔS/link ≲ 1), and do not collapse against raw `n_chain`. Geometric ladders reach the
  transition later because their *worst* link controls it.
- **C3 — Adaptive burn-in.** An annealing-style adaptive ladder (anchor the hot end from
  prior draws, refine downward, hard freeze) burns in faster — in likelihood evaluations to
  fixed quality — than both a pilot-run-then-fixed-ladder workflow and fixed geometric
  ladders. Corollary: the entropy ladder adapts faster than an acceptance-based ladder
  because it needs only second cumulants of logL, while true acceptance depends on the full
  logL distribution shape (which is estimated slowly, and is most non-Gaussian exactly at
  phase transitions).
- **C4 — Differential tempering.** DE-proposal value is temperature-dependent and peaks
  near phase transitions (where Fisher-type jumps collapse); DE's mechanism is sampling the
  posterior's two-point difference distribution; DE alone is vulnerable to buffer collapse
  (rank deficiency of the difference span; soft form = effective-temperature drift), which
  the non-DE mixture components repair.

Secondary demonstrations (exploratory, run if pilot budgets allow; no pre-registered test):

- **S1** Effective-temperature drift vs dimensionality (soft buffer collapse; curse of
  dimensionality; measured Var(logL)(T) and mean logL(T) against gold curves + NN-entropy offset).
- **S2** Exchange-targeting strategy A/B (the 6 implemented strategies in
  `DTMCMC/exchange_manager.py`), including cold-edge effects and value of T<1 rungs.
- **S3** Local-metric pathologies: negative adjacent-chain cross-correlation producing
  apparent "super-efficiency"; per-chain autocorrelation looking healthy while mode weights
  are wrong. Motivates round-trip time as the architecture-level autocorrelation.
- **S4** Asymptotic-bias vs `de_size`/`de_thin` (referee-proofing the "asymptotic detailed
  balance with self-inclusive buffer" claim).
- **S5** Multiple-walkers-per-temperature vs more-temperatures. Duplicate entries in the
  `Ts` array already work for proposals and exchanges (equal-T swaps always accept), but
  cycle tracking treats only slot `-1` as the hot extreme, so the round-trip metric needs
  hot-*set* semantics (all slots at T_max) plus fixtures — budgeted in the E10 go/no-go,
  not zero cost. Run only if time permits.

## 2. Verified starting state (2026-07-03)

- Repo reorganized (PR #7): package code in `DTMCMC/`, likelihoods promoted to
  `DTMCMC/likelihoods/` (cake, hawaii, eggbox, normal_nd), exploratory scripts frozen in
  `explorations/` (CI-excluded), reference `.npy` ladders/cumulants and `hawaii_map.hdf5`
  restored under `data/`, empty `experiments/` and `notebooks/` packages exist. CI: lint
  (prek/ruff), test (pytest on `tests/`), typecheck (mypy + pyright), build.
- **Two runtime bugs found during verification and fixed in commit `1ee32a2`** (sampler was
  unrunnable on this branch while CI stayed green, because tests only exercise ladder
  construction): (1) `NDArray` imports gated behind `if TYPE_CHECKING` in modules with
  `@njit` signatures → `NameError` under Python 3.14 PEP 649 when numba introspects;
  (2) `print_tracker_summary` indexed a `list[str]` with a boolean mask. Consequence: the
  golden-run regression test (§8) is mandatory, not optional.
- **RNG facts (measured):** numba's njit-internal RNG stream is independent of
  `np.random.seed()` called from Python; seeding inside an `@njit` function works. All hot-loop
  draws are inside `@njit` helpers (decision, DE, exchange, Fisher, rectangular prior draws);
  some likelihood/prior paths run in pure Python. Reproducibility therefore requires seeding
  **both** streams via a dedicated helper (§3 D1). The numba stream is per-thread; the
  sampler hot loop is single-threaded, so one helper call suffices per run.
- **Throughput (measured, single core, Apple silicon, n_chain=16):** cake 5D
  ≈ 3.45×10⁵ chain-steps/s (46 µs/step incl. exchanges); Gaussian 5D ≈ 5.5×10⁵ chain-steps/s.
  Per-step cost scales ≈ linearly with `n_chain`.
- **Path fragility:** `HawaiiLikelihood` opens `data/hawaii_map.hdf5` and
  `get_default_proposal_manager` reads `default_config.ini` relative to CWD. Harness must
  resolve paths against the repo root (Phase 1 task); existing behavior untouched.
- Existing infrastructure inventory relevant to this plan:
  - Ladders: `GeometricTemperatureLadder`, `EntropyTemperatureLadder`,
    `entropy_ladder_fromfile`, `find_potential_phase_transitions`
    (`DTMCMC/temperature_ladder_helpers.py`).
  - Thermodynamics: per-block logL moments to 6th order + adjacent-chain products
    (`DTMCMCSampler.block_end`), cumulants (`moment_helpers.py`), higher-order
    thermodynamic integration (`integral_heat_estimator.py`).
  - Trackers: per-(T, jump-type) accept counts, exchange matrices, cycle counting with
    block-indexed archives (`DTMCMC/tracker_manager.py`).
  - Efficiency: autocorr/cross-corr/scramble-block n_eff (`DTMCMC/corr_summary_helpers.py`,
    `DTMCMC/chain_analysis_helpers.py`).
  - Quality: NN entropy / two-sample KL estimators (`entropy_process.py`) — O(n²), use at
    checkpoints on thinned snapshots only.
  - Diagnostics: disconnect/imbalance/phase-transition/cycle commentary
    (`diagnostic_commentary_helpers.py`).
  - Test problems: cake 5D (two-tier, phase transition, semi-analytic radial density via
    `integ_box_filt.py`, gold ladder data in `data/`), eggbox (multi-modal with mode
    identification in `DTMCMC/likelihoods/eggbox.py`), Gaussian n-D (fully analytic),
    hawaii 2D (figure-friendly).

## 3. Locked design decisions

- **D1 — Seeding.** A run seed `s` deterministically derives two child seeds:
  `np.random.seed(child_a)` for the Python stream and a new `@njit` helper
  (`DTMCMC/rng_helpers.py::_seed_numba(child_b)`) for the numba stream, called once at run
  start. Both child seeds recorded in the artifact. **Nothing may reseed after run start:**
  the seed helper raises on a second call (an explicit test-only reset exists for the tests
  that legitimately reseed), and lint bans reseeding APIs outside `rng_helpers.py` (D5).
  The jitted seeder is private — numba cannot type a read of the guard state, so the
  once-per-run guard lives in `seed_run`, the only public seeding entry point; direct
  `_seed_numba` calls are TID251-banned outside `rng_helpers.py`/tests (amended per PR #9
  review).
- **D2 — Artifacts.** One HDF5 file per run. Root attrs (provenance): git commit hash +
  dirty flag, run-spec dump (full resolved config, INI/TOML text), run seed + child seeds,
  package versions (python, numpy, scipy, numba, h5py), hostname, start/end timestamps,
  wall-clock seconds, iteration/chain-step/likelihood-evaluation counters (§5 glossary;
  evals are counted where they happen at runtime, never derived from iteration counts via an
  assumed exchange cadence — the cadence is owned by `ExchangeManager.is_exchange_step` and
  may change in future A/B tests), and a `finalized` flag (complete vs partial artifact).
  Datasets: temperature
  ladder(s) (incl. history if adaptive), per-block logL moment arrays, tracker archives
  (accept/exchange/cycle + block indices), round-trip event log, thinned cold-chain samples,
  checkpoint metrics (buffer spectra recorded in-process, since the DE buffer is transient
  and never persisted; NN-KL computed post-hoc from the stored thinned samples at checkpoint
  cadence, per D7 — amended per PR #10 review), and anything a figure needs. Figures are
  generated **only** from artifacts (D7).
- **D3 — Paired seeds.** A/B arms within an experiment share the same run-seed list and the
  same initial states wherever the arm definition permits; comparisons are made on paired
  per-seed differences (§7). (Streams diverge after the first differing accept, so CRN mainly
  buys identical initialization and reduced between-seed variance; still worthwhile.)
- **D4 — Engine/experiment separation.** Changes inside `DTMCMC/` are minimal and surgical
  (rng helper, tracker extensions, ladder-family functions, the Phase 5
  `apply_ladder_update` hook). Everything else lives in
  `experiments/`. The adaptive controller starts in `experiments/` and is promoted into the
  package only after the paper's needs stabilize.
- **D5 — RNG-stream discipline.** No engine change may add, remove, or reorder random draws
  in the sampling path. Metrics are pure observers. Enforced by (a) the golden-run test
  (§8) — digest blessed on the CI platform, since numba/libm codegen is not bit-identical
  across architectures, plus a run-twice-in-job determinism check — and (b) ruff banned-API
  (TID251) rules against `numpy.random.seed`/`default_rng`/`RandomState` and
  `numba.prange`, plus a prek grep hook for `parallel=True` (a decorator kwarg, not an API
  path, so TID251 cannot match it) — both guarding per-thread streams — with per-file
  whitelists: `rng_helpers.py` (seeding), `entropy_process.py` (prange; currently serial
  under plain `@njit`), and `tests/` (reseeding is the point of the golden/determinism/freeze
  tests). Analysis code (e.g., §7 bootstrap) obtains Generators via
  `rng_helpers.get_rng(seed)` with the seed recorded in analysis outputs — reproducible
  analysis, no new whitelist. Any intentional re-blessing of the golden digest must be
  justified in the commit message.
- **D6 — Adaptive-ladder semantics.** Ladder updates happen only at block boundaries.
  On update: chain states remap to nearest new temperature; logLs carried over (they are
  T-independent); DE buffer columns are **not reset** — each new temperature inherits the
  buffer column of the nearest old temperature **at-or-hotter** (empirical rule: DE recovers
  much faster from overdispersion than underdispersion). Cycle/exchange trackers are
  segmented at each update (counts must not straddle a ladder change). Adaptation ends in a
  **hard freeze**; after freeze the code path is identical to a fixed-ladder sampler.
  Alternative remap/reset rules are a pilot A/B **[EXPLORATORY]**, not a production variable.
- **D7 — Artifact-first figures.** Every paper figure has a standalone script
  `experiments/figures/fig_<name>.py` reading only HDF5 artifacts.
- **D8 — Pre-registration.** The comparison statistics in §7 are fixed before production
  runs. Pilots may refine run lengths, seed counts, and grids — not the statistics or the
  primary metrics.

## 4. Phases

Each phase = one PR, reviewed against its acceptance criteria. Estimated engine diff per
phase ≤ ~300 lines; everything larger goes in `experiments/`.

### Phase 0 — Reorg and branch hygiene [DONE]

PR #7 (user) + commit `1ee32a2` (runtime fixes, ruff `runtime-evaluated-decorators` config).

### Phase 1 — Experiment harness

New package `experiments/harness/`:

- `spec.py`: `RunSpec` — likelihood name + params (n_par, cake tier params, …), ladder spec
  (type + params or input-data reference), proposal mixture (maps onto the existing
  ConfigParser sections), exchange strategy, `n_chain`, `n_cold`, `block_size`, total steps,
  storage/thinning, seed. Serializable to/from TOML; the resolved spec text is embedded in
  the artifact.
- `paths.py`: repo-root-anchored resolution for `data/` and config files. Mechanism for
  engine-internal relative paths (`hawaii_map.hdf5`): the runner chdirs to the repo root at
  startup (safe under one-process-per-run); the proposal-manager config is read by the
  harness from a resolved path and passed as an explicit `ConfigParser` (the parameter
  already exists), not via CWD.
- `artifact.py`: HDF5 writer/reader implementing D2, with a `validate(path)` function
  (schema + provenance completeness check).
- `runner.py`: single-run entry point (`python -m experiments.harness.run spec.toml --seed N
  --out dir/`): builds objects from spec, seeds both streams (D1), advances, flushes the
  artifact per checkpoint. No crash-tolerance guarantees beyond that — a run that dies is
  simply rerun (runs cost ≤ ~1 CPU-h); `validate(mode='partial'|'complete')` distinguishes
  via the `finalized` flag (D2). The likelihood object is wrapped in a counting proxy, so
  every `get_loglike` call — initialization, proposals, Fisher refreshes, history jumps
  (which evaluate inside proposal generation, so a history proposal costs two evals) —
  increments the artifact eval counter by construction rather than by enumerating call sites.
- `batch.py`: expands a sweep file (grid × seeds) into independent single-run invocations
  (one process per run; GNU-parallel/cluster-array friendly manifest output).
- Engine change: `DTMCMC/rng_helpers.py` (@njit seed helper + child-seed derivation +
  once-per-run guard with test-only reset). ~30 lines.

Acceptance criteria:
1. Tiny Gaussian spec runs end-to-end in <180 s **including JIT compilation** and produces a
   validating artifact.
2. Same spec + same seed twice → identical `logLs_store` (bit-exact, same platform);
   different seed → different; a second seed call within one run raises (D1 guard).
3. Artifact provenance attrs complete per D2 (`validate(mode='complete')`).
4. Existing tests, lint, typecheck stay green and their CI jobs cover `experiments/`
   (packaging continues to exclude it); no engine diff outside `DTMCMC/rng_helpers.py` and
   the ruff TID251 config.
5. TID251 behavior confirmed on our ruff version (it must flag attribute usage like
   `np.random.seed(...)`, not only imports); if it does not, an equivalent prek hook substitutes.
6. Counting-proxy test: the artifact eval counter equals the proxy count on a run exercising
   initialization, exchange and non-exchange iterations, and a Fisher refresh (history jumps
   added to the fixture if/when enabled).

### Phase 2 — Metrics and trackers

Engine extensions (all RNG-neutral observers, guarded by the golden test introduced here):

- Round-trip event log in `TrackerManager`: preallocated int64 buffer of
  (walker id, itrn, direction) events for T=1↔T=max touches, flushed per block (extends
  `process_chain_cycles`, which already detects the transitions). Enables round-trip time
  *distributions* and faster/lower-variance rate estimation than end-of-run counts.
- Flow fraction f(T): per-block, per temperature index, the fraction of resident walkers
  whose last extreme visit was cold (up-movers) — computable from the existing
  `cycle_tracker` state + `chain_track`; stored per block.
- Expected squared displacement per (T, jump type): accumulate proposed and accepted |δ|²
  alongside the existing accept counts (touches `mcmc_decision_helper` and callers;
  signature change, no RNG change).

Analysis-side (`experiments/metrics.py`, no engine change):

- DE-buffer difference-spectrum probe: eigenspectrum/effective-rank of the covariance of K
  random buffer differences per temperature, at checkpoints.
- NN-KL checkpoint evaluator wrapping `entropy_process` (thinned to ~5–10k samples).
- Cross-correlation "apparent super-efficiency" detector (n_eff_cross > n_eff_auto flag)
  wrapping `corr_summary_helpers`.
- Round-trip statistics: rate per walker per 10⁶ chain-steps, time distribution, fraction of
  walkers with ≥1 round trip, knee-fit utilities (knee definition finalized in pilots).

Reference samplers (`experiments/reference_samplers.py`): exact Gaussian draws (exists in
`normal_nd`); eggbox via mode enumeration + per-mode rejection; cake via radial inverse-CDF/
rejection from `integ_box_filt` density + isotropic angles. Needed by NN-KL and CDF-error
metrics.

Acceptance criteria:
1. **Golden-run test added first**: short fixed-seed cake run; stored digest of
   `logLs_store` + final state; digest blessed from a CI run (per-platform digests permitted
   for local dev) plus a run-twice-in-job determinism check. All subsequent Phase 2 commits
   keep it green.
2. Cycle/round-trip logic unit-tested on synthetic `chain_track` sequences with hand-computed
   answers (incl. n_cold>1 and duplicate-temperature cases — the S5/E10 prerequisite).
3. Gaussian invariants test: heat capacity C(T) = β²·Var(logL) ≈ n_par/2 at every tested T
   within tolerance (Var(logL) itself grows as n_par·T²/2), with the prior cutoff chosen so
   truncation is negligible over the tested range (cutoff/√T ≥ 4); entropy ladder built from
   measured vars ≈ geometric ladder.
4. Reference samplers validated (moments + NN-KL(self) ≈ 0 within estimator noise).
5. Throughput regression <5% vs Phase 1, measured locally on a fixed benchmark spec
   (best-of-3); CI timing is not used for this criterion.

### Phase 3 — Ladder family

In `DTMCMC/temperature_ladder_helpers.py`:

- Generalize the integrated-spacing machinery to integrand Var(logL)^p·β^q:
  (p=1, q=1) = current entropy ladder (behavior unchanged, verified by test);
  (p=½, q=0) = thermodynamic-length ladder (second-order equal-acceptance).
- `AcceptanceTemperatureLadder`: Gaussian closed-form predicted swap acceptance
  a(β₁,β₂) from interpolated cum1/cum2 (E[min(1,eʳ)] with m = −(β₁−β₂)(μ₁−μ₂),
  s² = (β₁−β₂)²(σ₁²+σ₂²)); sequential root-find from the hot end with outer bisection on the
  target a\* so exactly `n_chain` rungs span the ladder. Same inputs (Ts, logL vars/means) as
  the entropy ladder — deliberately, so adaptation-speed comparisons are apples-to-apples.
- **[EXPLORATORY, stretch]** measured-acceptance feedback ladder (Katzgraber-style) as a
  third arm, only if pilots show the predicted-acceptance ladder is too easy a strawman.
- Tunable cake family: promote hardcoded tier params in
  `DTMCMC/likelihoods/cake_likelihood.py` to constructor args with identical defaults
  (golden test guards behavior).

Acceptance criteria:
1. (p=1,q=1) reproduces current `EntropyTemperatureLadder` exactly on `data/*_gold` inputs.
2. On analytic Gaussian logL-var data, entropy, length, and geometric ladders coincide
   (within interpolation tolerance) — the built-in null case.
3. Acceptance predictor validated against brute-force swap Monte Carlo on synthetic Gaussian
   logL distributions (≤1% absolute error over a grid of Δβ).
4. All ladder types constructible from a harness spec.

### Phase 4 — Pilot calibration **[EXPLORATORY]**

Purpose: fix every number the production battery needs. Deliverable:
`.plans/pilot-report.md` containing the final production run matrix and a CPU-hour quote
(the artifact used to request external compute).

- Throughput table across likelihood × n_par × n_chain.
- Run-length calibration: steps needed for round-trip-rate estimates with ≤10% relative
  error per arm (drives E1/E2 lengths).
- Seed-count power analysis from pilot paired-difference variance, targeting 95% CI
  half-widths ≤ ~⅓ of observed effect sizes for C1's primary metric.
- Coarse knee scan on cake 5D to place the production n_chain grid around the transition.
- Knee estimator choice (piecewise-linear vs max-curvature) frozen here.
- DE-buffer remap A/B (D6 default vs nearest vs partial reset) — pick the default for Phase 5.
- NN-KL snapshot size / checkpoint cadence.
- Decision point for S5 and the feedback-ladder arm.

### Phase 5 — Adaptive burn-in controller **[EXPLORATORY internals, fixed interface]**

`experiments/adaptive.py`: controller around `DTMCMCSampler`, mutating engine state only
through one new engine hook `DTMCMCSampler.apply_ladder_update(new_ladder, remap_rule)`
(~30 lines, RNG-neutral, golden-guarded). The hook rebinds the ladder **and the
`betas`/`Ts` aliases** (`self.betas = self.T_ladder.betas` in `DTMCMCSampler.__init__` makes
external ladder swaps a stale-alias footgun), remaps chain states/logLs and DE-buffer
columns per D6, and segments trackers: archive-flush, then reset `cycle_tracker` to its
initialized state (in-flight extreme-visit records refer to the old ladder and must not
straddle an update).

- Interface (fixed): `AdaptiveLadderController(mode='entropy'|'length'|'acceptance',
  update_every_blocks, forgetting, freeze_criterion, T_min_factor)` with
  `mode`-agnostic internals so all three spacing rules share the schedule.
- Behavior per D6 (block-boundary updates, at-or-hotter buffer remap, tracker segmentation,
  hard freeze). Annealing schedule: initial ladder anchored at the hot end from prior-draw
  logL statistics (reaches ΔS ~2–3 immediately), extended/refined toward cold (optionally
  slightly below T=1 to suppress cold-edge effects, per S2) as data accumulates. Schedule
  internals, forgetting factor, and freeze criterion are iterated against pilots.
- Ladder history (every update: Ts, trigger stats, block index) recorded in the artifact.

Acceptance criteria:
1. Post-freeze equivalence is **bit-exact**: copy the frozen sampler's full state (samples,
   logLs, DE buffer, trackers, and proposal-manager state incl. Fisher matrices/scales and
   jump weights — Fisher state can be stale by up to `fisher_downsample` blocks, so
   recomputing it from the copied samples would not match) into a fresh fixed-ladder
   sampler, reseed both streams identically (test-only reset), advance one block, require
   identical output.
2. On cake 5D, the adaptive entropy ladder converges to within tolerance of the gold ladder
   (interpolated ΔS profile comparison), without human input.
3. Golden test still green (adaptive code must not touch fixed-ladder paths).

### Phase 6 — Production battery + analysis

- Execute the §5 registry (locally or on external compute from the Phase 4 manifest).
- `experiments/figures/fig_*.py` per figure; `experiments/analysis/` for the pre-registered
  statistics; results summary in `.plans/results.md` (tables feeding the paper).

## 5. Experiment registry

Units glossary: an **iteration** advances all chains by one step (regular or exchange); a
**chain-step** = iteration × `n_chain`; **likelihood evaluations** are counted at runtime
(exchange iterations evaluate none, so evals < chain-steps; the ratio is owned by
`ExchangeManager.is_exchange_step` and is never hard-coded into analysis — proposal/exchange
cadence is itself a legitimate future A/B variable). Compute estimates use measured
throughput (§2), scale ∝ n_chain, single-core runs, embarrassingly parallel. Seed counts
marked * are placeholders finalized in Phase 4.

| ID | Claim | Design | Est. CPU-h |
|----|-------|--------|-----------|
| E1 | C1 | cake 5D; arms = {geometric-default, geometric-tuned, length, acceptance, entropy}; n_chain ∈ {8,12,16,24,32,48}; 5×10⁶ iterations; 20* paired seeds | ~60 |
| E2 | C2 | E1 runs reanalyzed + same sweep on cake 8D, eggbox 5D, Gaussian {5,20}D (entropy + geometric arms only) | ~120 |
| E3 | C3 | arms = {adaptive-entropy, adaptive-acceptance, pilot+fixed-entropy (pilot evals charged), fixed-geometric ×2}; cake 5D + eggbox; ~2×10⁶ iterations; 20* paired seeds | ~15 |
| E4 | C4 | per-(T, jump-type) expected squared displacement + acceptance overlaid on C(T); cake 5D/8D + eggbox; entropy ladder at 2–3 chain densities; 10* seeds | ~15 |
| E5 | C4 | mixture ablations {full, no-DE, DE-only, no-prior} at fixed budget; plus rank-deficient-buffer collapse/recovery on Gaussian (DE-only + ε-Fisher, recovery time vs ε); buffer spectra + NN-KL checkpoints | ~25 |
| E6 | S1 | Gaussian n_par ∈ {2,8,32,64}: measured ⟨logL⟩(T), Var(logL)(T) vs analytic; NN-entropy offset; DE-heavy vs balanced mixtures | ~15 |
| E7 | S2 | exchange strategies × {entropy ladder, geometric}; cake 5D; round-trip + edge-effect metrics; 10* seeds | ~15 |
| E8 | S4 | bias vs de_size ∈ {10³,10⁴,10⁵,6×10⁵} × de_thin on Gaussian/cake; NN-KL + moment errors | ~15 |
| E9 | support | gold references: long cake 5D/8D runs (~5×10⁷ iterations, few seeds) for ladder inputs + high-precision evidence; exact-sampler references for Gaussian/eggbox/cake | ~10 |
| E10 | S5 | duplicate-temperature arms (requires hot-set cycle semantics per S5; deferred; decision in Phase 4) | ~15 |

Worst-case total ≈ **300 CPU-h** — a weekend on a 16-core workstation, or trivial as a
cluster array job. E1/E2 dominate and parallelize perfectly.

## 6. Primary metrics

- **C1 (co-primary):** round trips per walker per 10⁶ chain-steps (Phase 2 event log) **and**
  n_eff per likelihood evaluation; superiority claimed only where both hold (conjunction
  rule, §7). The n_eff estimator is frozen as the scramble-block empirical estimator
  (`n_eff_preds_empirical`: variance ratio of scrambled to sequential block means), computed
  per parameter on the cold chains, aggregated by minimum over parameters, evaluated per run
  (so §7 pairing applies); spectral/autocorrelation estimators are diagnostics only (S3).
  The burn-in fraction is a Phase 4 calibration constant, fixed before production and
  identical across arms.
- **C2:** round trips per walker per 10⁶ chain-steps.
  Secondary for both: min link swap acceptance, radial-CDF error (cake), mode-occupancy χ²
  and time-to-all-modes (eggbox), evidence error vs E9 reference, NN-KL.
- **C3:** likelihood evaluations to reach (a) first completed round trip, (b) ladder ΔS
  profile within ε of gold, (c) NN-KL below threshold. Thresholds frozen in Phase 4.
- **C4:** ratio of DE to Fisher expected squared displacement as a function of |T − T_c|;
  frozen-eigendirection variance vs time (collapse demo); round-trip rate across ablations.

## 7. Pre-registered comparison statistics

For each claim: paired per-seed differences of the primary metric between the named arms at
each sweep point; report the median paired difference with bootstrap 95% CI (10⁴ resamples)
and a Wilcoxon signed-rank test as robustness check. Superiority is claimed only where the
CI excludes zero, and claimed *as a regime statement* over the sweep region where it holds
(e.g., "for ΔS/link < X"). C1's two co-primaries use a conjunction rule: full superiority is
claimed only where both paired CIs exclude zero (conservative under multiplicity; each
metric is also reported separately). Primary claims are tested on primary metrics only; all
secondary metrics are reported as exploratory. Curve-collapse (C2) is quantified by comparing
between-likelihood dispersion of knee locations under ΔS/link vs raw n_chain axes.

## 8. Reviewability contract

1. One PR per phase; engine diffs ≤ ~300 lines; experiments code fully type-annotated
   (jitted internals may use loose array types where numba requires).
2. **Golden-run test** (from Phase 2 on): fixed-seed short run, bit-exact digest of
   `logLs_store` + final state, in CI. Re-blessing requires explicit justification.
3. **RNG discipline** (D5): metrics/trackers are observers; no draws added/removed/reordered.
4. Physics-invariant tests: Gaussian heat capacity C(T)=β²·Var(logL)=n_par/2; ladder
   coincidence on constant-C inputs; acceptance predictor vs brute force; synthetic
   cycle-count fixtures.
5. Artifact-first figures (D7); artifacts carry full provenance (D2).
6. Pre-registration (D8): §6–7 frozen before Phase 6.

## 9. Risks and fallbacks

- **Knee is mushy** → knee estimator frozen in Phase 4 before production; if no sharp knee,
  C2 becomes a scaling-collapse claim (still novel) rather than a threshold claim.
- **Entropy ≈ acceptance ladder in practice** → report equivalence plus the estimation-cost
  asymmetry (second-cumulant sufficiency vs distribution-shape dependence, worst at
  transitions); C1 vs geometric still stands.
- **Cake reference density accuracy** (rejection sampler tails) → validate against E9 long
  runs before use in NN-KL.
- **Memory at large n_chain × de_size** (6×10⁵ × 48 × 5 doubles ≈ 1.2 GB) → cap de_size or
  raise de_thin in sweep specs; record in spec.
- **Python 3.14/numba annotation regressions** → ruff `runtime-evaluated-decorators` config
  + golden test now guard this class of breakage.

## 10. Questions deferred to pilots (Phase 4)

Forgetting factor and update cadence; freeze criterion; DE-buffer remap A/B; knee estimator;
final seed counts and run lengths; E1's geometric-tuned arm definition (T_max/spacing
hand-tuned in Phase 4, documented in the pilot report); feedback-ladder arm go/no-go; NN-KL
snapshot size; E10 go/no-go; whether S2 cold-edge (T<1 rungs) merits promotion into E1 arms.
