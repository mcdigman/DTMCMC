# DTMCMC telemetry dashboard

Mission-control view for DTMCMC runs: live monitoring of an in-progress
run, or post-hoc diagnostic analysis of a completed one, in a web browser.

```bash
# monitor one run artifact (or point at a directory to browse runs)
python -m dashboard /path/to/artifacts --port 8050

# viewing a run on a remote machine from a laptop:
#   (server) python -m dashboard /scratch/runs --port 8050
#   (laptop) ssh -N -L 8050:localhost:8050 user@server
# then open http://localhost:8050 locally
```

## Process isolation and data flow

The dashboard runs in a **completely separate process** from the sampler
and communicates only through the HDF5 run artifacts the harness already
writes (`experiments/harness/artifact.py`). A dashboard crash, lag, or
misconfiguration cannot affect a live run.

Concurrency is safe **by construction, not by locking**: the artifact
writer flushes atomically (write to a `.tmp` name, then `os.replace`), so
a reader that reopens the file at each poll sees either the previous
checkpoint or the new one, never a torn file. The reader
(`dashboard.core.reader.ArtifactWatcher`) keys reloads on the file's
`(mtime_ns, size)` stat pair and re-reads only when a new flush landed.
Live-update latency therefore equals the run's `checkpoint_every_blocks`
cadence, which is already a spec knob.

Artifacts are read **server-side**; only plotly figure JSON crosses the
network to the browser. Watching a multi-GB run from a laptop over an ssh
tunnel transmits kilobytes per update, never the archived data volume.
Artifact selection is allowlisted: a request can only open files the
server itself enumerated under its configured root, so a crafted browser
request cannot read arbitrary server-side HDF5 paths.

## Layering (what swaps out, what stays)

```
dashboard/
├── core/            framework-agnostic data layer (numpy + h5py only)
│   ├── reader.py      artifact -> RunSnapshot; polling watcher; run listing
│   ├── diagnostics.py pure functions: RunSnapshot -> plot-ready series
│   └── checks.py      status lights: RunSnapshot -> ok/warn/alert/na results
├── themes/          style tokens (light/dark), no plotting imports
├── figures/         thin plotly factories over core outputs
│   ├── base.py        the ONLY module that maps Theme -> plotly styling
│   ├── options.py     ViewOptions: every user-adjustable knob, one place
│   ├── thermo.py      E[logL], heat capacity, entropy, ladder spacing
│   ├── proposals.py   acceptance, ESD, exchange rates + history
│   ├── mixing.py      flow fraction, round trips, logL history, DE spectra
│   ├── posterior.py   corner, traces, ACF, cross-correlation
│   └── registry.py    PlotSpec table + named tab LAYOUTS
├── app/dash_app.py  the Dash front-end (layout, callbacks, polling)
└── cli.py           python -m dashboard entry point
```

The dependency arrows only point left: `core` imports nothing above it
(and no plotting library), `figures` imports `core` + `themes` + plotly,
`app` imports everything. Swapping the front-end (e.g. a Panel/bokeh live
app, or a Dash site showing post-analysis highlights) means rewriting only
the `app` layer — Panel renders plotly figures directly, so `figures`
is reusable as-is. Swapping the plot library (matplotlib/seaborn for
paper figures) means reimplementing `figures` against the same `core`
outputs; the diagnostics never change. A `corner`-based matplotlib corner
plot, for example, would consume `diagnostics.corner_matrix` unchanged.

Adding a plot = one factory function + one `PlotSpec` line in
`registry.py`. Rearranging the dashboard = editing the `LAYOUTS` dict
(tab name -> plot ids); alternate layouts are selected with `--layout`.

## Themes

`dashboard/themes` defines interchangeable style token sets (`light`,
`dark`) with no UI imports. The same `Theme` object drives the plotly
styling (via `figures/base.py`) and the page CSS (generated at app start
in `dash_app.py`), so adding a theme is one dataclass instance. Palette
discipline: fixed-order categorical hues for identity (jump types), a
single-hue ordinal ramp for ordered series (temperatures, ladder-history
snapshots), a reserved highlight for the current/live entity, and status
colors used only for run-state chips.

## Generality

Nothing in the dashboard is specific to a likelihood, ladder kind, or
proposal set: labels (likelihood name, jump types, ladder kind, adaptive
mode) come from the artifact itself (embedded `spec_toml`,
`proposal_config_ini`, and the `trackers` `jump_labels` attr), and every
figure degrades gracefully when its data is absent (fixed-ladder runs
have no ladder history; `--no-store` runs have no posterior tab data;
pre-dashboard artifacts without `jump_labels` fall back to numbered jump
types).

## Sampler-side changes (kept minimal)

Two additive, schema-compatible artifact changes support the dashboard
(both optional for the reader — older artifacts still load):

1. `trackers` attr `jump_labels`: proposal names aligned with the
   accept/ESD record axis, so acceptance plots are labeled without
   reconstructing engine objects.
2. `ladder/initial_Ts`: the run-start ladder, so adaptive runs' moment
   blocks can be mapped to the ladder active during them without
   re-running controller code (previously required re-seeding and
   rebuilding the controller, as in the old scratch plotting script).

## Dependencies

Beyond the core project requirements (numpy, h5py) and the in-repo
`DTMCMC` package (entropy integration helpers), the dashboard needs:

- `plotly` (figure layer) — installed in DTMCMC-dev (6.9)
- `dash` (front-end layer only) — installed in DTMCMC-dev (4.4)

Both are declared as the `dashboard` optional extra in `pyproject.toml`.
The `core` and `themes` layers deliberately run without either installed.

## Status lights

The Status tab (the landing tab) evaluates a registry of run-health
checks against the current snapshot and renders one card per light,
most severe first; a header badge summarizes the worst enabled status
next to the run-state chip. Statuses are ok / warning / alert, plus n/a
with a reason when a check lacks data (young runs) or does not apply
(fixed-ladder runs). Shipped lights:

| light | watches |
|---|---|
| Finite moments | NaN/inf in recent block logL moments |
| Ladder freeze | adaptation finished, and by criterion rather than budget |
| Mean logL drift | blockwise mean logL drifting between recent halves |
| Thermodynamic stability | Var(logL) (the C(T) profile) moving between recent halves |
| Round trips | walkers completing hot-cold cycles at a healthy rate |
| Walker flow | up-flow fraction far from the linear constant-flow ideal |
| Exchange uniformity | nn exchange acceptance spread across temperatures |
| Exchange bottleneck | a near-zero exchange link splitting the ladder |
| Exchange flow share | exchanges carrying nearly all accepted movement |
| Acceptance smoothness | acceptance jumping between adjacent temperatures |
| Cold-chain acceptance | the coldest chain accepting almost nothing |
| Cold-chain correlation | stored history spanning too few autocorrelation times |
| DE buffer rank | DE difference spectrum losing effective rank |

Adding a light is one evaluator function `RunSnapshot -> (status,
message)` plus one `CheckSpec` row in `core/checks.py` — the tab, the
badge, and the silencing control pick it up automatically. Thresholds
are keyword arguments with heuristic defaults, ready for a future
configuration surface. Any subset of lights can be silenced with the
"status checks" control (persisted per browser session); silenced
lights leave both the tab and the badge. A check that raises reports
itself as an alert rather than taking the dashboard down.

## Notable view options

- **burn-in blocks** trims the start of block-history plots (the block
  mean logL plot's range is otherwise dominated by the first block or two).
- **rate window** selects the counts behind the tracker-rate plots:
  'current ladder' (default) uses everything since the last applied
  ladder update, so every count sits on the ladder now in effect; 'whole
  run' bins each archive window at the temperatures its own ladder held
  (on adaptive runs the temperature axis is the union over segments);
  'since last archive' shows only the most recent window. Windows are
  never attributed to a ladder they did not run under — tracker archives
  are snapshotted before an update mutates the ladder.
- **history stride** thins ladder-history curves on long adaptive runs.
- **chains / dims** select the recorded-chain and dimension subsets for
  the posterior plots (corner plots must use a subset — some likelihoods
  are too high-dimensional to corner whole).
- Zoom/pan state persists across live updates (plotly `uirevision`).

## Known limitations / future work

- Update granularity is the artifact checkpoint cadence; sub-checkpoint
  streaming would need a sampler-side push channel (deliberately avoided:
  isolation beats latency here).
- The per-block moment record and sample store are re-read whole when a
  flush changes; for very long production runs an incremental reader
  (h5py partial reads keyed on block count) slots into `reader.py`
  without touching diagnostics.
- Cross-correlation currently pairs the first and last of the selected
  chains; a dedicated pair picker is trivial to add if it earns its
  place in the controls row.
- No per-figure table-view/CSV export yet (accessibility twin); hover
  tooltips and legends carry the values meanwhile.
