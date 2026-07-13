# Contract amendment: issue #19 remediation (DRAFT — each item marked for ratification)

Status: **draft for review**. The issue-19 stack (PRs #27/#28/#29) changed
acceptance semantics and invariants that previously lived only in test
docstrings and PR prose. This document collects them as explicit contract
items so they can be ratified or rejected in one place. Items marked
`RATIFY?` need an explicit decision; items marked `codified` merely record
behavior already implemented and verified on the branch.

## A. DE buffer-memory invariant (supersedes plan §9 memory notes)

1. **The DE proposal buffer is a ring buffer whose purpose includes
   forgetting burn-in.** Production runs MUST NOT use whole-run buffers
   (`de_size × de_thin ≥ n_steps`): proposal support then retains
   prior-fill and adaptation-era states for the entire run, so post-freeze
   samples stay conditioned on burn-in forever, and memory scales with run
   length. `codified` — harness warns on whole-run buffers for adaptive
   runs.
2. **Sizing rule:** memory span `de_size × de_thin` covers at least
   ~2 adaptation windows (`update_every_blocks × block_size`; harness
   warning floor) and recommended ~8 windows (the battery standard,
   `ADAPTIVE_DE_WINDOW_BLOCKS = 64` blocks at cadence 8), while remaining
   well under the run so the buffer fully turns over — and forgets the
   adaptation — before the post-freeze readout window. `codified` in the
   batteries and the example spec; exact production numbers per-target.
   `RATIFY?` the 2-window floor and 8-window recommendation.
3. **Per-column support note:** DE difference vectors pair rows within a
   single chain column; a column's support diversity is fed by swap flux
   through its temperature slot and therefore scales with chain count and
   ladder health. Small-chain-count probes of "buffer bias" conflate
   memory span with column-refresh diversity; provisioning is
   buffer-span × chain-count jointly (consistent with the issue-19 sweep's
   ~independent contributions). `codified` as interpretation; no code
   change.
4. **Old behavior retained as control:** the whole-run configuration runs
   as an explicit old-behavior control battery (same gates, expects the
   warning), and the tiny-buffer failure stays a negative control.
   `codified`.

## B. Buffer remap on ladder updates (amends D6)

5. **`remap_rule = 'no_remap'` is the default**: columns keep their slot
   across equal-size ladder updates and re-burn-in under their new
   temperatures; the cloning rules (`at_or_hotter`, `nearest`) are
   retained for old-behavior tests and pilot A/Bs. `RATIFY?` — flipped in
   this PR based on local testing; D6's default candidate was
   `at_or_hotter`.

## C. Criterion 2 (adaptive acceptance battery) — redefinition

6. **Success = reference-anchored sample-space recovery on the post-freeze
   readout**, not tracker statistics and not ladder shape alone. Gates:
   symmetric NN vs exact reference draws, analytic moments, tier/mode
   occupancy and interconversion, logL mean, plus ladder-structure gates
   against an analytic entropy profile. `codified` (PR #29).
7. **Budget freezes are acceptable outcomes** provided the sample-space
   gates pass on the post-freeze segment; requiring a criterion freeze
   anti-selects for coarse fixed points. `RATIFY?` — note the current
   batteries would not fail if every seed budget-froze with a still-moving
   ladder; if "predictable convergence" is a paper claim it needs its own
   gate (freeze-block dispersion / criterion-freeze rate across seeds),
   which is NOT yet implemented. The post-freeze reader now surfaces
   `frozen_by`, `freeze_block`, and `max_dlog_last` so such a gate can be
   added without schema changes.
8. **The battery cake is `widths = (4, 0.15)`**, not the default
   `(4, 0.1)`: the default cake's 1e-8-volume spike makes discovery a rare
   event at unit-test budgets under honest dynamics (it remains a
   deliberate stress target). Default-cake full certification moves to the
   production-scale E-series; a structural battery guards it at unit
   scale. `RATIFY?` the width and the E-series deferral.

## D. Burn-in discard in reported results

9. **Production results report only post-freeze history**: everything
   before the adaptive freeze (at minimum the entire temperature-ladder
   adaptation) is burn-in and is discarded. The convention is owned by
   `experiments/harness/postfreeze.load_post_freeze`, which reads the
   freeze block from the artifact (`ladder/history` attrs, schema v4);
   the pilots' metric loader burns `max(burn_fraction, freeze)`.
   `codified`.
10. **Residual (open):** with any finite buffer, the first post-freeze
    buffer-turnover interval is still proposal-conditioned on pre-freeze
    support; a stricter convention would discard `freeze +
    de_size × de_thin` iterations. `RATIFY?` whether E-series adopts the
    stricter boundary (cheap: one parameter on the reader).
11. **Open for E-series:** `RunSpec` pins the sample store to whole-run
    (`store_size = ceil(n_steps / store_thin)`); the engine's wrapping
    ring store is unreachable from the harness and artifact row↔iteration
    mapping assumes no wrap. Long production runs need either
    freeze-anchored trimming at artifact-write time or a spec-exposed
    bounded store with wrap-aware artifacts. Decision deferred; not
    blocking at unit/pilot scale.

## E. Metrics

12. **`nn_divergence_symmetric` (max over both orientations) replaces the
    one-sided NN-KL in every gate** (C3(c) amendment): spike collapse
    drives the signed statistic negative and under any one-sided
    threshold. `codified` (PR #28).
13. **Round-trip metrics are demoted to coupling/throughput diagnostics**;
    they gate nothing by themselves. The freeze coupling witness (new
    trips per cadence window) remains as a liveness requirement, not a
    quality claim. `codified`.
14. **Pilot-derived numbers (knee, n_chain grid, run-length calibration)
    were measured under biased-buffer dynamics and RT-rate saturation and
    are NOT to be carried into Phase 6.** Re-derive on the
    posterior-anchored gates using the family-comparison driver
    (`experiments/pilots/family_compare.py`: gate-1 posterior recovery,
    efficiency ranked among passing arms only). `RATIFY?` scope and
    schedule of the re-derivation.

## F. Ladder families

15. **All three adaptive spacing rules are contract surfaces**: the
    non-entropy modes (`length`, `acceptance`) now carry end-to-end
    batteries on the gaussian benchmark, and the adaptive acceptance mode
    carries a realized exchange-rate flatness gate (cap disabled, spread
    < 0.12 vs the predicted target). The cold-edge cap deliberately
    overrides equal-acceptance spacing on the coldest links whenever it
    binds; flatness claims therefore apply to uncapped links only.
    `codified`.
16. **Realized-acceptance flatness on a structured-C(T) target** (battery
    cake) and non-gaussian family batteries remain open — the gaussian's
    constant heat capacity makes every family near-geometric there, so
    the current batteries certify wiring and realized targets, not family
    discrimination. Family discrimination happens through the
    family-comparison driver at pilot scale. `RATIFY?` whether a
    structured-C(T) flatness gate is required before Phase 6.

## G. Test-configuration hygiene (audit remediation 3)

17. Adaptive acceptance tests never inherit tiny harness proposal
    overrides; the battery spec builder (`make_adaptive_spec`) owns the
    ring-buffer default. Small-buffer configurations are negative controls
    only. `codified`.
18. The eight PR #18 likelihoods are contract targets with
    reference-anchored batteries (registry: `experiments/benchmarks.py`);
    `hawaii` is the deliberate no-ground-truth entry whose gates must
    degrade explicitly. `codified` (PR #28/#29).
