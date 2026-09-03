# Registry-driven serial Numba proposal pass

## Goal

Replace the hard-coded proposal and exchange cases in `DTMCMC/numba_backend.py`
with an opt-in registry and generated monomorphic kernel, while retaining the
ordinary Python object model and behavior. Extend the serial-native milestone
to all built-in proposal types except ladder-history jumps, strengthen backend
selection semantics, and add behavior-equivalence coverage for the cases
identified in PR review.

## Agreed contracts

- `kernel_backend='python'` always uses the Python block implementation.
- `kernel_backend='numba'` is strict: an unsupported graph, missing
  registration, or Numba compilation failure raises a descriptive error.
- `kernel_backend='auto'` silently falls back only when the entire graph is
  undecorated. A graph mixing registered and unregistered components warns
  once and falls back. Once any registered component reaches native code, a
  compilation failure is an extension error and raises rather than falling
  back.
- Built-in DE, prior, auxiliary, Fisher diagonal/subspace, and Fisher Cholesky
  jumps are registered. `LadderHistoryJumpManager` remains an intentional
  Python fallback in this pass.
- Proposal selection uses the shared `choose_prob_helper`, and exchange cadence
  and execution are supplied by the registered exchange manager instead of
  being reimplemented in the generated kernel.
- The public Python classes remain ordinary classes. Registration is keyed by
  exact concrete type so an undecorated subclass cannot accidentally inherit a
  stale native contract.
- Module-level `NamedTuple` types are the native state ABI. Arrays stored in a
  state may be mutated in place. DE's write index and thinning counter live in
  a small mutable `int64` counter array created at block entry, avoiding
  `NamedTuple` reconstruction on every sampler step; its registered setter
  copies the final scalar values back to the Python manager once per block.
  Other manager state is immutable or array-backed in this milestone.
- Strategy-parameter containers become dataclasses while preserving their
  existing ConfigParser construction and `record_config` behavior.
- A uniform-likelihood/Gaussian-prior example becomes a library likelihood and
  a harness-selectable test case, exercising nonzero prior proposal factors and
  a known target distribution.

## Architecture

1. Define structural protocols for the aggregate proposal manager, component
   managers, exchange manager, and rectangular-bounds likelihood surface. Keep
   `JumpManager` and `ProposalManager` as useful concrete implementations.
2. Define typed native specifications and registries:
   - `jittable_jump(native_function)` registers a jump class. The native
     function receives `(sample_point, itrt, manager_state, likelihood_state)`;
     likelihood-aware jumps such as prior draws use a generated factory.
   - `jittable_jump_manager(state_getter, state_setter,
     post_step_function)` registers a component-manager class and its one native
     state value.
   - `jittable_exchange_manager(state_getter, is_exchange_step,
     exchange_function)` registers exchange scheduling and execution.
3. Generate one Python source function for the concrete ordered proposal graph.
   The graph is flattened from the manager/jump objects, each jump branch is
   emitted from registry metadata in exact list order, and per-manager
   post-step calls are emitted once per sampler iteration. Compile that function
   with Numba and cache it by a structural signature including concrete
   likelihood, manager, jump, and exchange types.
4. Bundle kernel inputs into module-level `NamedTuple` records for sampler,
   tracker, likelihood, proposal-manager states, and exchange state so the
   generated function does not expose the current long positional argument
   list.
5. Make likelihood adaptation generic via generated source rather than the
   current zero-through-four-state cascade. Move likelihood metadata into an
   exact-type registry with callable annotations and a rectangular-bounds
   protocol.
6. Resolve backend capability once per sampler graph/signature and cache the
   generated program. Re-extract runtime state at each block because Fisher and
   DE arrays/counters may change. Invalidate/re-resolve if the structural graph
   signature changes.

## Built-in registrations

- DE manager state: DE buffer, subspace fraction, thinning interval, write
  index, and thinning counter. All four DE jumps call `apply_de_helper`.
  Post-step calls one shared jitted DE state-transition helper used by both the
  Python manager and native program.
- Fisher manager state: `n_par`, subspace fraction, sigma scales, Cholesky
  factors, and gamma multipliers. Register diagonal full/subspace and Cholesky
  jumps; Cholesky uses the existing jitted triangular solver.
- Prior manager state: no manager-specific mutable data. Its jump is generated
  against the likelihood contract's prior draw/factor functions.
- Auxiliary manager state: empty state. Blank jump returns a copy and succeeds.
- Exchange manager state: strategy and full-tracking flag. Scheduling calls the
  registered cadence function; execution calls the shared
  `do_ptmcmc_exchange` helper.

## Strategy dataclasses

Convert `DEStrategyParameters`, `FisherStrategyParameters`,
`PriorStrategyParameters`, `AuxilliaryStrategyParameters`,
`HistoryStrategyParameters`, and the empty `ProposalStrategyParameters` shell
to dataclasses with explicit fields and ConfigParser-aware custom constructors.
Preserve `copy()` values without retaining a live ConfigParser dependency, and
preserve serialized key spelling where compatibility requires it. Strategy
parameters remain outside the native ABI; manager state getters expose only the
values needed by proposal kernels.

## Behavior fixes and examples

1. Remove Eggbox-only likelihood adapters by making its public likelihood
   interface conform to the shared rectangular likelihood contract.
2. Add `UniformGaussianPriorLikelihood`: constant scalar log likelihood on an
   unbounded domain, Gaussian prior draw and log-density factor, jittable
   registration, and exact analytic Gaussian posterior moments.
   Apply the likelihood's untempered prior-density change to every local MH
   decision; the prior-draw proposal's reverse/forward density factor then
   cancels it exactly, while other jumps target the intended non-uniform prior.
3. Add the new likelihood to harness specification/build routing and a small
   checked-in TOML case.
4. Remove the unnecessary `mcmc_decision_helper` compatibility re-export if no
   tracked caller requires it; otherwise update the actual tracked caller and
   retain no backend-specific dependency on the re-export.

## Verification

1. Unit-test registry behavior with a custom jump and manager that require no
   edit to `numba_backend.py`.
2. Test strict `numba`, undecorated `auto`, mixed-graph warning, and decorated
   compile-failure behavior, including warn-once/cached resolution.
3. Compare Python and generated serial snapshots/digests from identical seeds
   across every exchange targeting strategy, random targeting with
   `track_full_exchanges` both true and false, Fisher Cholesky enabled, DE
   thinning greater than one, multiple cold chains, and odd/even block sizes
   that do not divide storage thinning.
4. Exercise `store_thin > 1`, `de_thin > 1`, and their counter state across
   multiple blocks.
5. Test strategy dataclass construction, copy independence, mutation where
   existing callers rely on it, and config round-tripping.
6. Test the Gaussian-prior likelihood's prior factor, native/Python parity,
   harness construction, and sampled moments with appropriately loose
   tolerances.
7. Run formatting, mypy, the focused backend/harness tests, the full fast test
   suite, and a serial throughput smoke benchmark. Preserve the local user
   commit and untracked profiling artifacts.

## Delivery

Implement as follow-up commits on `codex/jittable-serial-kernel`, push the PR
branch, and update PR #37 with a self-contained body/comment describing the
registry protocol, fallback semantics, built-in coverage, tests, and remaining
ladder-history limitation.
