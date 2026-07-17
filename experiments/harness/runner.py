"""Single-run execution: build engine objects from a RunSpec and advance.

The likelihood object is wrapped in a counting proxy, so every get_loglike
call — initialization, proposals, Fisher refreshes, and future history
jumps — increments the eval counter by construction rather than by
enumerating call sites (plan §4 Phase 1). Both RNG streams are seeded once
at run start (plan D1); the artifact is flushed per checkpoint and
finalized at the end (plan D2).

Harness behavior rides the DTMCMCSampler extension API rather than an
external driver loop: HarnessSampler builds its proposal manager in
initialize_jumps, runs the adaptive controller in postblock_operations,
and checkpoints in post_Nblock_teardown, so run_from_spec just advances
the run as checkpoint-sized advance_N_blocks segments.
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast, override
from warnings import warn

import numpy as np

if TYPE_CHECKING:
    import configparser
    from collections.abc import Callable

    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.proposal_manager import AbstractProposalManager

from diagnostic_commentary_helpers import print_diagnostic_commentary
from DTMCMC.corr_summary_helpers import CorrelationSummary
from DTMCMC.de_manager import DEJumpManager
from DTMCMC.dtmcmc_sampler import DTMCMCSampler
from DTMCMC.exchange_manager import ExchangeManager
from DTMCMC.likelihoods.ar1 import Ar1Likelihood
from DTMCMC.likelihoods.banana import BananaLikelihood
from DTMCMC.likelihoods.cake_likelihood import CakeLikelihood
from DTMCMC.likelihoods.constant_rectangular import ConstantRectangularLikelihood
from DTMCMC.likelihoods.eggbox import EggboxLikelihood
from DTMCMC.likelihoods.gaussian_mixture import GaussianMixtureLikelihood
from DTMCMC.likelihoods.gaussian_shell import GaussianShellLikelihood
from DTMCMC.likelihoods.hawaii_likelihood import HawaiiLikelihood
from DTMCMC.likelihoods.hyperpyramid import HyperpyramidLikelihood
from DTMCMC.likelihoods.normal_nd import GaussianLikelihood
from DTMCMC.likelihoods.random_wheel import RandomWheelLikelihood
from DTMCMC.likelihoods.rosenbrock import RosenbrockLikelihood
from DTMCMC.likelihoods.spoke_wheel import SpokeWheelLikelihood
from DTMCMC.likelihoods.uniform_gaussian_prior import UniformGaussianPriorLikelihood
from DTMCMC.proposal_manager_helper import get_default_proposal_manager
from DTMCMC.rng_helpers import get_rng, seed_run
from DTMCMC.temperature_ladder_helpers import (
    AcceptanceTemperatureLadder,
    GeometricTemperatureLadder,
    LengthTemperatureLadder,
    TemperatureLadder,
    entropy_ladder_fromfile,
    filter_ladder_inputs,
)
from experiments import adaptive
from experiments.adaptive import AdaptiveLadderController
from experiments.metrics import de_buffer_difference_spectrum

from .artifact import CheckpointLog, RunProvenance, collect_provenance, write_artifact
from .paths import chdir_repo_root, resolve
from .spec import EXCHANGE_STRATEGY_CODES, RunSpec, config_to_text

# random buffer-difference pairs per temperature in the checkpoint DE spectrum
DE_SPECTRUM_PAIRS = 256

# DE buffer-memory floors (in adaptation windows / blocks) below which the
# harness warns: shorter spans cannot bridge ladder rebuilds (adaptive) or
# a block's own decorrelation scale (fixed ladders)
DE_MEMORY_MIN_WINDOWS = 2
DE_MEMORY_MIN_BLOCKS_FIXED = 4


# one constructor per spec likelihood name; a test asserts the keys stay in
# sync with spec.LIKELIHOOD_NAMES (spec.py cannot import these back without a
# circular import, so drift is caught by CI instead)
LIKELIHOOD_BUILDERS: dict[str, Callable[..., AbstractLikelihood[NamedTuple]]] = {
    'gaussian': GaussianLikelihood,
    'cake': CakeLikelihood,
    'constant_rectangular': ConstantRectangularLikelihood,
    'eggbox': EggboxLikelihood,
    'hawaii': HawaiiLikelihood,
    'ar1': Ar1Likelihood,
    'banana': BananaLikelihood,
    'gaussian_mixture': GaussianMixtureLikelihood,
    'gaussian_shell': GaussianShellLikelihood,
    'hyperpyramid': HyperpyramidLikelihood,
    'random_wheel': RandomWheelLikelihood,
    'rosenbrock': RosenbrockLikelihood,
    'spoke_wheel': SpokeWheelLikelihood,
    'uniform_gaussian_prior': UniformGaussianPriorLikelihood,
}


def build_likelihood(spec: RunSpec[AbstractLikelihood[NamedTuple]]) -> AbstractLikelihood[NamedTuple]:
    """Construct the likelihood object named by the spec."""
    params: dict[str, Any] = dict(spec.likelihood_params)
    builder = LIKELIHOOD_BUILDERS.get(spec.likelihood_name)
    if builder is None:
        msg = f'unknown likelihood {spec.likelihood_name!r}'
        raise ValueError(msg)
    return builder(**params)


def _scalar(value: object) -> float:
    """Narrow a spec scalar to float-convertible, rejecting lists."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f'expected a numeric scalar, got {value!r}'
        raise TypeError(msg)
    return float(value)


def _build_geometric_ladder[LikelihoodType: AbstractLikelihood[NamedTuple]](
    spec: RunSpec[LikelihoodType],
) -> TemperatureLadder:
    """Construct a geometric ladder from the spec's ladder table."""
    ladder = spec.ladder
    return GeometricTemperatureLadder(
        spec.n_chain,
        n_cold=spec.n_cold,
        T_cold=_scalar(ladder.get('T_cold', 1.0)),
        T_min=_scalar(ladder.get('T_min', 1.0)),
        T_max=_scalar(ladder.get('T_max', 1.0e15)),
        n_inf_final=int(_scalar(ladder.get('n_inf_final', 1))),
    )


def _build_entropy_file_ladder[LikelihoodType: AbstractLikelihood[NamedTuple]](
    spec: RunSpec[LikelihoodType],
) -> TemperatureLadder:
    """Construct an entropy ladder from reference data files named by the spec."""
    ladder = spec.ladder
    return entropy_ladder_fromfile(
        spec.n_chain,
        spec.n_cold,
        str(resolve(str(ladder['Ts_file']))),
        str(resolve(str(ladder['vars_file']))),
        n_inf_final=int(_scalar(ladder.get('n_inf_final', 1))),
        T_cold=_scalar(ladder.get('T_cold', 1.0)),
        correct_last=bool(ladder.get('correct_last', False)),
    )


def _load_ladder_inputs[LikelihoodType: AbstractLikelihood[NamedTuple]](
    spec: RunSpec[LikelihoodType], *stat_file_keys: str
) -> tuple[np.ndarray, ...]:
    """Load Ts plus stat arrays named by the spec, with the from-file filter.

    The Ts array is always loaded from 'Ts_file' explicitly (no
    positional first-key contract) and the shared engine helper
    filter_ladder_inputs applies the Ts >= 1 from-file convention, so
    every file-driven ladder arm filters identically.
    """
    Ts_in = np.load(resolve(str(spec.ladder['Ts_file'])))
    stats = [np.load(resolve(str(spec.ladder[key]))) for key in stat_file_keys]
    return filter_ladder_inputs(Ts_in, *stats)


def _build_length_file_ladder[LikelihoodType: AbstractLikelihood[NamedTuple]](
    spec: RunSpec[LikelihoodType],
) -> TemperatureLadder:
    """Construct a thermodynamic-length ladder from reference data files."""
    ladder = spec.ladder
    Ts_in, vars_in = _load_ladder_inputs(spec, 'vars_file')
    return LengthTemperatureLadder(
        spec.n_chain,
        Ts_in,
        vars_in,
        n_cold=spec.n_cold,
        T_cold=_scalar(ladder.get('T_cold', 1.0)),
        n_inf_final=int(_scalar(ladder.get('n_inf_final', 1))),
        correct_last=bool(ladder.get('correct_last', False)),
    )


def _build_acceptance_file_ladder[LikelihoodType: AbstractLikelihood[NamedTuple]](
    spec: RunSpec[LikelihoodType],
) -> TemperatureLadder:
    """Construct a predicted-acceptance ladder from reference data files."""
    ladder = spec.ladder
    Ts_in, means_in, vars_in = _load_ladder_inputs(spec, 'means_file', 'vars_file')
    return AcceptanceTemperatureLadder(
        spec.n_chain,
        Ts_in,
        means_in,
        vars_in,
        n_cold=spec.n_cold,
        T_cold=_scalar(ladder.get('T_cold', 1.0)),
        n_inf_final=int(_scalar(ladder.get('n_inf_final', 1))),
    )


def _build_explicit_ladder[LikelihoodType: AbstractLikelihood[NamedTuple]](
    spec: RunSpec[LikelihoodType],
) -> TemperatureLadder:
    """Construct a ladder directly from the spec's Ts list.

    RunSpec validation guarantees Ts is a numeric list of length n_chain.
    """
    Ts_raw = spec.ladder['Ts']
    if not isinstance(Ts_raw, list):
        msg = 'explicit ladder requires a Ts list'
        raise TypeError(msg)
    return TemperatureLadder(np.asarray(Ts_raw, dtype=np.float64), n_cold=spec.n_cold)


# one builder per spec ladder kind; a test asserts the keys stay in sync with
# spec.LADDER_KINDS (see LIKELIHOOD_BUILDERS note)
LADDER_BUILDERS: dict[str, Callable[[RunSpec[AbstractLikelihood[NamedTuple]]], TemperatureLadder]] = {
    'geometric': _build_geometric_ladder,
    'entropy_file': _build_entropy_file_ladder,
    'length_file': _build_length_file_ladder,
    'acceptance_file': _build_acceptance_file_ladder,
    'explicit': _build_explicit_ladder,
}


def build_ladder[LikelihoodType: AbstractLikelihood[NamedTuple]](spec: RunSpec[LikelihoodType]) -> TemperatureLadder:
    """Construct the temperature ladder described by the spec."""
    kind = spec.ladder['kind']
    builder = LADDER_BUILDERS.get(str(kind))
    if builder is None:
        msg = f'unknown ladder kind {kind!r}'
        raise ValueError(msg)
    return builder(spec)


class HarnessSampler[LikelihoodType: AbstractLikelihood[NamedTuple]](DTMCMCSampler[LikelihoodType]):
    """DTMCMCSampler wired to the harness through the extension API.

    The subclass carries the run-level context its hooks need (spec,
    proposal config, adaptive controller, artifact destination) and
    implements the harness behaviors as overrides of the pre-existing
    extension points rather than as an external driver loop:
    initialize_jumps builds the spec-configured proposal manager around
    the starting samples the base class draws; postblock_operations
    advances the adaptive controller schedule; post_Nblock_teardown
    checkpoints, so run_from_spec drives the run as checkpoint-sized
    advance_N_blocks segments.

    sampler_verbosity gates the pretty-printed tracker diagnostics that
    the base teardown emits unconditionally: 0 is silent (the default,
    so automated runners keep their logs clean), 1 prints the summary
    at each major-report boundary only, 2 prints at every checkpoint
    teardown. With the default report interval (spec.n_steps) the single
    major report lands at the last teardown of a full run.
    """

    def __init__(
        self,
        spec: RunSpec[LikelihoodType],
        T_ladder: TemperatureLadder,
        like_obj: LikelihoodType,
        config: configparser.ConfigParser,
        *,
        controller: AdaptiveLadderController[LikelihoodType] | None = None,
        artifact_path: Path | None = None,
        provenance: RunProvenance | None = None,
        start_monotonic: float | None = None,
        sampler_verbosity: int = 0,
        kernel_backend: str = 'auto',
    ) -> None:
        if artifact_path is not None and provenance is None:
            msg = 'artifact_path requires provenance'
            raise ValueError(msg)
        # hook-consumed attributes must exist before super().__init__,
        # which runs the overridable initialization chain
        # (initialize_jumps reads spec and config)
        self.spec = spec
        self.config = config
        self.controller = controller
        self.artifact_path = artifact_path
        self.provenance = provenance
        self.start_monotonic = time.monotonic() if start_monotonic is None else start_monotonic
        self.sampler_verbosity = sampler_verbosity
        # periodic "major report" bookkeeping: rather than comparing itrn
        # against a run total (the sampler must not know how many iterations
        # it will be run for), the teardown accumulates steps since the last
        # major report and wraps when the configured interval is reached
        self.steps_since_major_report = 0
        self.itrn_prev_teardown = 0
        self.checkpoints = CheckpointLog()
        # checkpoint metrics draw from a dedicated Generator seeded by the
        # run seed: reproducible, recorded, and independent of both run RNG
        # streams (plan D5) — the golden digest is unaffected
        self.metrics_rng = get_rng(spec.seed)
        super().__init__(
            T_ladder,
            like_obj,
            spec.block_size,
            spec.store_size,
            store_thin=spec.store_thin,
            arg_record=spec.arg_record,
            kernel_backend=kernel_backend,
            zero_loglike=spec.zero_loglike,
        )
        self.de_manager = next(
            (manager for manager in self.proposal_manager.managers if isinstance(manager, DEJumpManager)), None
        )
        # DE buffer-memory hygiene: the ring buffer's memory span is sized
        # to the ADAPTATION timescale, not the run. Too short and the
        # buffer cannot bridge ladder rebuilds; spanning the whole run and
        # the proposal support never forgets burn-in — post-freeze samples
        # stay conditioned on prior-fill and adaptation-era states, which
        # is not a production configuration.
        if self.de_manager is not None:
            memory_span = self.de_manager.de_size * self.de_manager.de_thin
            if spec.adaptive is not None:
                window = int(_scalar(spec.adaptive.get('update_every_blocks', 8))) * spec.block_size
                if memory_span < DE_MEMORY_MIN_WINDOWS * window:
                    warn(
                        f'DE buffer memory de_size*de_thin = {memory_span} < {DE_MEMORY_MIN_WINDOWS} adaptation '
                        f'windows ({DE_MEMORY_MIN_WINDOWS * window}): too short to bridge ladder rebuilds',
                        stacklevel=3,
                    )
                elif memory_span >= spec.n_steps:
                    warn(
                        f'DE buffer memory de_size*de_thin = {memory_span} spans the whole run ({spec.n_steps}): '
                        'proposal support never forgets burn-in (not a production configuration)',
                        stacklevel=3,
                    )
            elif memory_span < DE_MEMORY_MIN_BLOCKS_FIXED * spec.block_size:
                warn(
                    f'DE buffer memory de_size*de_thin = {memory_span} < {DE_MEMORY_MIN_BLOCKS_FIXED} blocks '
                    f'({DE_MEMORY_MIN_BLOCKS_FIXED * spec.block_size}): short DE buffers can change proposal behavior',
                    stacklevel=3,
                )
        # run-start ladder snapshot for the artifact: adaptive updates mutate
        # Ts in place, so the copy must be taken before the first block runs
        self.initial_Ts = self.Ts.copy()

    @override
    def initialize_jumps(self, proposal_manager_in: AbstractProposalManager[LikelihoodType] | None = None) -> None:
        """Build the spec-configured proposal manager around the base-drawn starting samples.

        Runs inside super().__init__ after initialize_state has filled
        samples[0] from prior draws, so the per-stream RNG order matches
        the previous external construction (starting draws first, then
        manager-construction draws) and the golden digest is unchanged.
        """
        if proposal_manager_in is not None:
            super().initialize_jumps(proposal_manager_in)
            return
        exchange_manager = ExchangeManager(
            EXCHANGE_STRATEGY_CODES[self.spec.exchange_strategy],
            track_full_exchanges=self.spec.track_full_exchanges,
        )
        self.proposal_manager = get_default_proposal_manager(
            self.T_ladder,
            self.like_obj,
            starting_samples=self.samples[0, :, :],
            config=self.config,
            exchange_manager_loc=exchange_manager,
        )

    @override
    def postblock_operations(self) -> None:
        """Advance the adaptive controller's schedule at the block boundary."""
        if self.controller is not None:
            self.controller.post_block(self)

    def adaptive_burnin_iterations(self) -> int:
        """Iterations to treat as adaptive burn-in for the correlation summary.

        Derived from the adaptive controller's freeze point (the block at
        which adaptation stopped, times the block size): everything up to
        the freeze was spent tuning the ladder rather than sampling a
        fixed target. Fixed-ladder runs (no controller) or a controller
        still adapting yield 0. CorrelationSummary clamps the value to the
        stored ring-buffer window, so an over-long burn-in is safe.
        """
        if self.controller is None:
            return 0
        frozen_block = self.controller.frozen_block_index
        return 0 if frozen_block is None else frozen_block * self.block_size

    def record_checkpoint_metrics(self) -> None:
        """Record the checkpoint DE-buffer difference spectrum."""
        if self.de_manager is not None:
            self.checkpoints.itrns.append(self.itrn)
            self.checkpoints.de_spectrum_eigvals.append(
                de_buffer_difference_spectrum(self.de_manager.de_buffer, DE_SPECTRUM_PAIRS, self.metrics_rng)
            )

    @override
    def post_Nblock_teardown(self) -> None:
        """Checkpoint at the end of each advance_N_blocks segment.

        Records metrics, flushes the artifact when a destination is
        configured (marked finalized at each major-report boundary), and
        prints diagnostics per sampler_verbosity: the tracker summary
        (verbosity 1 at each major report, verbosity 2 every checkpoint),
        plus, at each major report, the descriptive commentary (verbosity
        >= 1) and the full correlation summary (verbosity 2).

        Major-report boundaries are periodic: the teardown tracks steps
        elapsed since the last report and wraps when they reach the
        configured interval, rather than comparing itrn against a run
        total. This keeps the sampler agnostic to how many iterations it
        will ultimately run (parent-sampler design principle) — advancing
        past the initially requested count yields further periodic
        reports instead of re-emitting a one-shot "final" report every
        segment. With the default interval (spec.n_steps) exactly one
        report lands at the end of a full run.
        """
        self.record_checkpoint_metrics()
        self.steps_since_major_report += self.itrn - self.itrn_prev_teardown
        self.itrn_prev_teardown = self.itrn
        major_report = self.steps_since_major_report >= self.spec.n_steps_per_major_report
        if major_report:
            # preserve the overflow remainder so the report cadence does
            # not drift when the interval is not a whole number of segments
            self.steps_since_major_report %= self.spec.n_steps_per_major_report
        if self.artifact_path is not None and self.provenance is not None:
            write_artifact(
                self.artifact_path,
                self.spec,
                self,
                self.eval_accounting,
                self.provenance,
                finalized=major_report,
                wall_seconds=time.monotonic() - self.start_monotonic,
                checkpoints=self.checkpoints,
                adaptive_state=self.controller,
            )
        if self.sampler_verbosity >= 2 or (self.sampler_verbosity == 1 and major_report):
            self.tracker_manager.print_tracker_summary(self.n_cold, self.Ts, self.proposal_manager)
        if major_report and self.sampler_verbosity >= 1:
            print_diagnostic_commentary(self)
            if self.sampler_verbosity >= 2:
                # burn-in from the adaptive freeze; 0 for fixed-ladder runs
                n_burnin = self.adaptive_burnin_iterations()
                corr_sum: CorrelationSummary[LikelihoodType] = CorrelationSummary()
                corr_sum.summarize_blocks(self, self.tracker_manager, n_burnin)
                corr_sum.final_prints(self, n_burnin)


def build_sampler[LikelihoodType: AbstractLikelihood[NamedTuple]](
    spec: RunSpec[LikelihoodType],
    config: configparser.ConfigParser | None = None,
    like_obj: LikelihoodType | None = None,
    T_ladder: TemperatureLadder | None = None,
    *,
    controller: AdaptiveLadderController[LikelihoodType] | None = None,
    artifact_path: Path | None = None,
    provenance: RunProvenance | None = None,
    start_monotonic: float | None = None,
    sampler_verbosity: int = 0,
    kernel_backend: str = 'auto',
) -> tuple[HarnessSampler[LikelihoodType], LikelihoodType]:
    """Build the harness sampler and counting-proxy likelihood for a spec.

    Assumes both RNG streams are already seeded (see run_from_spec):
    starting samples, DE-buffer fills, and Fisher initialization all draw
    from the run streams. Pass the config explicitly to share one instance
    between the sampler and the artifact provenance (run_from_spec does);
    pass like_obj/T_ladder to override the spec-built ones (the adaptive
    path supplies its prior-anchored initial ladder). The keyword-only
    arguments configure the extension hooks: without an artifact_path the
    teardown records checkpoint metrics but writes nothing. Scientific run
    modes, including zero_loglike, come from the serializable RunSpec.
    """
    if like_obj is None:
        like_obj = cast('LikelihoodType', build_likelihood(spec))
    if T_ladder is None:
        T_ladder = build_ladder(spec)
    if config is None:
        config = spec.build_proposal_config()

    sampler = HarnessSampler(
        spec,
        T_ladder,
        like_obj,
        config,
        controller=controller,
        artifact_path=artifact_path,
        provenance=provenance,
        start_monotonic=start_monotonic,
        sampler_verbosity=sampler_verbosity,
        kernel_backend=kernel_backend,
    )
    return sampler, like_obj


def build_adaptive_controller(
    adaptive_table: dict[str, Any],
) -> AdaptiveLadderController[AbstractLikelihood[NamedTuple]]:
    """Construct the adaptive controller from a spec [adaptive] table.

    Keys (validated against ADAPTIVE_KEYS at spec load; see
    experiments/specs/adaptive_cake12.toml for a worked example):
    `mode` ('entropy'|'length'|'acceptance', required), `budget_blocks`
    (hard adaptation cap in blocks, required, plan Phase 5),
    `update_every_blocks` (rebuild cadence, default 8), `forgetting`
    (pool down-weighting per evaluation, default 0), `freeze_dlog` /
    `freeze_consecutive` (stability criterion, defaults 0.02 / 3),
    `remap_rule` (DE-buffer remap applied on ladder updates, default
    'no_remap': columns keep their slot and re-burn-in under the new
    temperature; 'at_or_hotter'/'nearest' clone columns and are retained
    for old-behavior tests and pilot A/Bs),
    `T_min_factor` (cold-edge target in (0, 1] as a multiple of the T=1
    readout; sub-unit values extend the ladder below the readout),
    `var_estimator` (rebuild-variance rule: 1 = pessimistic max over
    recent segment estimates, the default; 0 = forgetting-weighted
    mean), `n_prior_draws` (hot-anchor prior sample size, default 256),
    `min_updates_at_target` (dwell evaluations before freeze counting,
    default 6), plus optional controller-geometry fields:
    `window_extension_factor`, `ds_link_cap`, `cold_cap_links`,
    `cap_ratio_min` / `cap_ratio_max`, `var_history_length`,
    `pool_dlog_tol`, and `discard_blocks_after_update` — defaults are
    the module constants in experiments.adaptive.
    """
    return AdaptiveLadderController(
        mode=str(adaptive_table['mode']),
        update_every_blocks=int(_scalar(adaptive_table.get('update_every_blocks', 8))),
        forgetting=_scalar(adaptive_table.get('forgetting', 0.0)),
        freeze_criterion=(
            _scalar(adaptive_table.get('freeze_dlog', 0.02)),
            int(_scalar(adaptive_table.get('freeze_consecutive', 3))),
        ),
        remap_rule=str(adaptive_table.get('remap_rule', 'no_remap')),
        T_min_factor=_scalar(adaptive_table.get('T_min_factor', 1.0)),
        budget_blocks=int(_scalar(adaptive_table['budget_blocks'])),
        var_estimator=int(_scalar(adaptive_table.get('var_estimator', 1))),
        n_prior_draws=int(_scalar(adaptive_table.get('n_prior_draws', 256))),
        min_updates_at_target=int(_scalar(adaptive_table.get('min_updates_at_target', 6))),
        window_extension_factor=_scalar(
            adaptive_table.get('window_extension_factor', adaptive.WINDOW_EXTENSION_FACTOR)
        ),
        ds_link_cap=_scalar(adaptive_table.get('ds_link_cap', adaptive.DS_LINK_CAP)),
        cold_cap_links=int(_scalar(adaptive_table.get('cold_cap_links', adaptive.COLD_CAP_LINKS_AUTO))),
        cap_ratio_bounds=(
            _scalar(adaptive_table.get('cap_ratio_min', adaptive.CAP_RATIO_BOUNDS[0])),
            _scalar(adaptive_table.get('cap_ratio_max', adaptive.CAP_RATIO_BOUNDS[1])),
        ),
        var_history_length=int(_scalar(adaptive_table.get('var_history_length', adaptive.VAR_HISTORY_LENGTH))),
        pool_dlog_tol=_scalar(adaptive_table.get('pool_dlog_tol', adaptive.POOL_DLOG_TOL)),
        discard_blocks_after_update=int(
            _scalar(adaptive_table.get('discard_blocks_after_update', adaptive.DISCARD_BLOCKS_AFTER_UPDATE))
        ),
    )


def run_from_spec[LikelihoodType: AbstractLikelihood[NamedTuple]](
    spec: RunSpec[LikelihoodType], out_dir: str | Path, artifact_name: str | None = None, sampler_verbosity: int = 0
) -> Path:
    """Execute one run end to end and return the artifact path.

    Chdirs to the repo root (engine-internal relative paths), seeds both
    RNG streams from the spec seed (once per process, plan D1), and
    advances the run as checkpoint-sized advance_N_blocks segments: the
    sampler's own teardown hook flushes the artifact at every checkpoint
    and finalizes at the last (plan D2).
    """
    chdir_repo_root()
    start_monotonic = time.monotonic()

    child_seed_python, child_seed_numba = seed_run(spec.seed)

    # one ConfigParser instance is shared between the sampler and the
    # artifact provenance, captured as text once at run start, so the
    # artifact records exactly the config the run used even if
    # default_config.ini changes mid-run (PR #9 review)
    config = spec.build_proposal_config()
    provenance: RunProvenance = collect_provenance(
        spec.seed,
        child_seed_python,
        child_seed_numba,
        spec_toml=spec.to_toml_text(),
        proposal_config_ini=config_to_text(config),
    )

    controller: AdaptiveLadderController[LikelihoodType] | None = None
    like_obj: LikelihoodType | None = None
    initial_ladder: TemperatureLadder | None = None
    if spec.adaptive is not None:
        controller_temp = build_adaptive_controller(spec.adaptive)
        like_obj = cast('LikelihoodType', build_likelihood(spec))
        # prior-draw anchoring consumes run-stream draws and counted evals,
        # deliberately: adaptive burn-in is charged in full (plan C3)
        initial_ladder = controller_temp.initial_ladder(like_obj, spec.n_chain, spec.n_cold)
        controller = cast('AdaptiveLadderController[LikelihoodType]', controller_temp)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    artifact_path = out_path / (artifact_name if artifact_name is not None else f'{spec.name}_seed{spec.seed}.h5')

    # build_sampler consumes scientific run modes from the effective spec,
    # including zero_loglike, so artifact provenance and execution agree.
    sampler, _like_obj = build_sampler(
        spec,
        config=config,
        controller=controller,
        like_obj=like_obj,
        T_ladder=initial_ladder,
        artifact_path=artifact_path,
        provenance=provenance,
        start_monotonic=start_monotonic,
        sampler_verbosity=sampler_verbosity,
    )

    n_full_segments, blocks_remainder = divmod(spec.n_blocks, spec.checkpoint_every_blocks)
    for _ in range(n_full_segments):
        sampler.advance_N_blocks(spec.checkpoint_every_blocks)
    if blocks_remainder:
        sampler.advance_N_blocks(blocks_remainder)
    return artifact_path
