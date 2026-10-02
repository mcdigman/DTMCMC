# DTMCMC

Differential Tempering Markov Chain Monte Carlo: a parallel-tempering sampler
that combines differential-evolution proposals with entropy-spaced temperature
ladders. The hot loop is compiled with numba, so a full block of chain updates
(proposals, accept/reject, and inter-chain exchanges) runs as a single native
program.

Key features:

- **Temperature ladders** built geometrically, by equal entropy spacing, by
  equal swap acceptance, or by thermodynamic length, plus an adaptive burn-in
  controller that refines the ladder and then freezes it.
- **Proposal mixture** of differential-evolution jumps from a history buffer,
  Fisher-matrix jumps, prior draws, and pluggable auxiliary jumps, with
  temperature-dependent weights.
- **Diagnostics** for round trips, swap rates, autocorrelation, and per-chain
  logL cumulants, and a separate-process web dashboard for live runs.
- **Reproducible experiments** driven by TOML specs that write validated HDF5
  artifacts.

## Installation

Requires Python 3.14 or newer. From a clone of the repository:

```bash
pip install -e .
```

Optional extras:

| Extra         | Adds                                   |
| ------------- | -------------------------------------- |
| `plots`       | matplotlib and corner for diagnostics  |
| `dashboard`   | plotly and dash for the web dashboard  |
| `dev`         | pytest, type stubs, linters, checkers  |

For example, `pip install -e ".[dev,plots,dashboard]"`.

## Quickstart

Sample a 5-dimensional Gaussian with the bundled likelihood and default
proposal mixture. Run this from the repository root, since the default
proposal configuration is read from `default_config.ini` there.

```python
import numpy as np

from DTMCMC.dtmcmc_sampler import DTMCMCSampler
from DTMCMC.likelihoods.normal_nd import GaussianLikelihood
from DTMCMC.rng_helpers import seed_run
from DTMCMC.temperature_ladder_helpers import GeometricTemperatureLadder

seed_run(42)  # seeds both the numpy and numba RNG streams

like_obj = GaussianLikelihood(n_par=5, cutoff=5)
T_ladder = GeometricTemperatureLadder(n_chain=8, n_cold=2, T_max=100.0)

block_size = 200
store_size = 4000
sampler = DTMCMCSampler(T_ladder, like_obj, block_size, store_size)
sampler.advance_N_blocks(store_size // block_size)

# the first n_cold store columns are always the T = T_cold readout chains
samples, logLs = sampler.get_stored_flattened(n_burnin=1000, n_chain_out=T_ladder.n_cold)
print(samples.mean(axis=0), samples.std(axis=0))
```

The sampler prints an acceptance table per jump type and temperature at the
end of the run. Very high acceptance at cold temperatures usually means poor
exploration; very low acceptance means poorly scaled jumps.

### Writing your own likelihood

Subclass `RectangularLikelihood` from `DTMCMC/likelihood.py` and supply a
numba-compiled log likelihood. The base class provides uniform prior draws,
bounds handling, and Fisher-matrix support for box-shaped parameter spaces.
The files under `DTMCMC/likelihoods/` are short, complete examples, including
Rosenbrock, eggbox, Gaussian shell, and mixture targets.

### Configuring proposals

Proposal weights and buffer sizes come from an INI-style config. Copy
`default_config.ini`, edit the `[DEJumpManager]`, `[FisherJumpManager]`, and
`[PriorManager]` sections, and pass a `ProposalManager` built from it to the
sampler. The notebooks in `notebooks/` walk through this in full.

## Spec-driven runs

For experiments, describe a run in TOML and let the harness build the engine,
seed both RNG streams, checkpoint, and validate the output artifact:

```bash
python -m experiments.harness.run experiments/specs/tiny_gaussian.toml --out artifacts
```

Specs set the likelihood, ladder, run length, exchange strategy, and proposal
overrides. See `experiments/specs/` for examples, including an adaptive-ladder
run. Artifacts are HDF5 files flushed atomically at every checkpoint.

## Dashboard

The dashboard monitors a live run or inspects a finished one from a browser.
It runs in its own process and reads only the HDF5 artifacts, so it cannot
disturb the sampler.

```bash
python -m dashboard artifacts --port 8050
```

Point it at a single artifact or a directory of runs. See
`dashboard/README.md` for the design and the remote-viewing recipe.

## Repository layout

| Path            | Contents                                                        |
| --------------- | --------------------------------------------------------------- |
| `DTMCMC/`       | The sampler package: ladders, jump managers, kernel, likelihoods |
| `experiments/`  | TOML-spec harness, adaptive controller, metrics, benchmarks      |
| `dashboard/`    | Telemetry dashboard (core, figures, Dash app)                    |
| `notebooks/`    | Annotated demo notebooks                                         |
| `tests/`        | pytest suite; `-m slow` selects the heavy convergence batteries  |
| `data/`         | Reference ladders, cumulants, and the Hawaii map likelihood data |
| `explorations/` | Frozen exploratory scripts, excluded from CI                     |

## Development

Install the dev extras, then run the checks the CI workflows run:

```bash
pytest -m "not slow"
```

```bash
pre-commit run --all-files
```

Linting uses ruff and pyrefly through pre-commit; mypy and pyright run in the
type-check workflow. The hot loop must stay single-threaded so RNG streams
remain reproducible, and a pre-commit hook rejects `parallel=True` outside the
allowed modules.

## License

Apache License 2.0. See [LICENSE](LICENSE).
