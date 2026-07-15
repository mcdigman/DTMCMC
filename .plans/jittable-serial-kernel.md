# Jittable serial PTMCMC kernel plan

## Objective

Move the complete per-block PTMCMC transition loop into Numba nopython code
without converting the public sampler, likelihood, proposal-manager, tracker,
or adaptive-controller object graphs into jitclasses.

The first production milestone is a serial native kernel that preserves the
current seeded evolution for supported configurations. Parallel and GPU
backends are follow-ons; the design must leave room for hundreds or thousands
of chains, explicit per-chain RNG state, batched proposal/likelihood passes,
and configurable regular-to-exchange cadence.

## Agreed constraints

- The existing Python sampler remains the public facade and continues to own
  initialization, storage, block-boundary updates, adaptive hooks, reporting,
  artifacts, and user-facing extension behavior.
- The serial native backend should preserve the existing golden evolution.
  Guaranteed identical evolution may remain a serial-only contract.
- The serial core proposal set is DE, prior, and diagonal Fisher/sigma jumps.
  Full-Cholesky Fisher, ladder history, and arbitrary auxiliary proposals may
  silently fall back to the existing Python kernel.
- A custom native likelihood supplies a scalar log likelihood, bounds, and,
  for a non-uniform prior, native prior draw and log-density functions.
- Unsupported likelihoods or proposal configurations fall back silently.
- Hawaii may fall back initially. HDF5 loading is host-side and a future
  native implementation can replace scipy interpolation with a grid kernel.
- GPU execution is not part of this milestone. No NVIDIA hardware is
  currently available, so GPU-specific code must not be introduced without a
  verification path.

## Architecture

### Native likelihood descriptor

A class decorator attaches a native descriptor to an ordinary likelihood
class. The descriptor contains:

- a standardized scalar log-likelihood kernel;
- a state extractor returning only Numba-compatible scalars, tuples, and
  arrays;
- a bounds extractor;
- rectangular uniform-prior defaults; and
- optional standardized prior draw, prior log-density, and validation
  kernels.

The decorator does not replace or jitclass the Python class. The original
methods remain the debug/fallback implementation.

### Proposal translation

At each block boundary, the Python proposal manager is translated to native
inputs:

- integer proposal codes aligned with `jump_probs`;
- diagonal sigma scales and subspace configuration;
- the DE buffer, subspace configuration, and ring-buffer counters;
- exchange strategy and tracking mode; and
- tracker arrays.

Eligibility is conservative. Unsupported active proposals or managers with
unmodelled per-step side effects select the Python fallback. Unsupported
proposal types with exactly zero probability do not prevent native execution.

### Generated block kernel

One kernel is generated per native likelihood descriptor and cached in the
Python process. It performs, in the existing serial order:

1. exchange-vs-regular step selection;
2. jump selection and integer-coded proposal dispatch;
3. bounds validation;
4. likelihood evaluation;
5. Metropolis decision and tracker updates;
6. walker-identity propagation; and
7. DE ring-buffer updates.

It returns updated scalar DE counters and the exact number of likelihood calls
performed in the native block. The facade writes those values back to the
ordinary Python objects.

### Backend selection

The sampler accepts `kernel_backend="python" | "numba" | "auto"`:

- `python` always uses the existing implementation;
- `numba` attempts native execution and silently falls back if ineligible;
- `auto` has the same fallback behavior and is the default once no-drift
  coverage establishes compatibility.

This switch does not use `NUMBA_DISABLE_JIT`, so existing jitted numerical
helpers remain enabled in Python/debug mode.

## Verification

- Preserve the existing platform golden digest for the supported default Cake
  configuration.
- Add direct Python-vs-native state equivalence tests from identical seeds,
  covering samples, log likelihoods, chain identities, DE state/counters,
  acceptance/exchange/ESD trackers, and `n_evals`.
- Cover all core native proposal codes through forced jump probabilities.
- Verify silent fallback for Hawaii, undecorated extension subclasses, and
  active unsupported proposals such as full-Cholesky Fisher.
- Verify the decorator contract with a small external-style likelihood,
  including a non-uniform-prior example.
- Run the full non-slow test suite and the fixed throughput benchmark.

## Follow-ons

1. Add configurable regular/exchange cadence, including schedules such as 5:1
   and 5:5, with mixing and walltime-to-convergence benchmarks.
2. Add explicit per-chain RNG states and a separate parallel reproducibility
   contract/golden digest.
3. Add `prange` across chains, optional fast-math subsets, and benchmarks at
   `n_chain` from 48 through the thousands and `n_par` at 20 and 100.
4. Reduce DE memory pressure with shared/subsampled reservoirs, lower-precision
   storage, or temperature-grouped buffers if large-chain/high-dimensional
   runs require it.
5. Add an optional proposal-batch and likelihood-batch interface suitable for
   CPU vectorization and a future device-resident backend.
