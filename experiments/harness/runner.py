"""Single-run execution: build engine objects from a RunSpec and advance.

The likelihood object is wrapped in a counting proxy, so every get_loglike
call — initialization, proposals, Fisher refreshes, and future history
jumps — increments the eval counter by construction rather than by
enumerating call sites (plan §4 Phase 1). Both RNG streams are seeded once
at run start (plan D1); the artifact is flushed per checkpoint and
finalized at the end (plan D2).
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    import configparser
    from collections.abc import Callable

    from numpy.typing import NDArray

from DTMCMC.de_manager import DEJumpManager
from DTMCMC.dtmcmc_sampler import DTMCMCSampler
from DTMCMC.exchange_manager import ExchangeManager
from DTMCMC.likelihood import AbstractLikelihood
from DTMCMC.likelihoods.cake_likelihood import CakeLikelihood
from DTMCMC.likelihoods.eggbox import Likelihood as EggboxLikelihood
from DTMCMC.likelihoods.hawaii_likelihood import HawaiiLikelihood
from DTMCMC.likelihoods.normal_nd import GaussianLikelihood
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
from experiments.metrics import de_buffer_difference_spectrum

from .artifact import CheckpointLog, RunProvenance, collect_provenance, write_artifact
from .paths import chdir_repo_root, resolve
from .spec import EXCHANGE_STRATEGY_CODES, RunSpec, config_to_text

# random buffer-difference pairs per temperature in the checkpoint DE spectrum
DE_SPECTRUM_PAIRS = 256


class LikelihoodLike(Protocol):
    """Structural interface the engine requires of a likelihood object.

    Matches AbstractLikelihood but also fits duck-typed likelihoods such
    as the eggbox numba jitclass, which cannot subclass an ABC.
    """

    n_par: int

    def get_loglike(self, params_in: NDArray[np.floating], /) -> float:
        """Get the log likelihood at the specified parameters."""
        ...

    def prior_draw(self) -> NDArray[np.floating]:
        """Get a draw from the priors for this likelihood."""
        ...

    def prior_factor(self, params_in: NDArray[np.floating], /) -> float:
        """Get the prior density factor for the input parameters."""
        ...

    def correct_bounds(self, params_in: NDArray[np.floating], /) -> NDArray[np.floating]:
        """Correct the input parameters to be within the prior range."""
        ...

    def check_bounds(self, params_in: NDArray[np.floating], /) -> bool:
        """Check if the specified point is within the prior volume."""
        ...


class CountingLikelihood(AbstractLikelihood):
    """Proxy that counts get_loglike calls and delegates everything else.

    Wrapping the likelihood before any engine object sees it guarantees
    the artifact eval counter covers every call site (initialization,
    proposal evaluation, Fisher refreshes, history jumps) by construction.
    The proxy is a pure observer: it adds no random draws (plan D5).
    """

    def __init__(self, wrapped: LikelihoodLike) -> None:
        self._wrapped: LikelihoodLike = wrapped
        self.n_evals: int = 0
        AbstractLikelihood.__init__(self, int(wrapped.n_par))

    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Count and delegate a log-likelihood evaluation."""
        self.n_evals += 1
        return self._wrapped.get_loglike(params_in)

    def prior_draw(self) -> NDArray[np.floating]:
        """Delegate a prior draw (a draw is not an evaluation)."""
        return self._wrapped.prior_draw()

    def prior_factor(self, params_in: NDArray[np.floating]) -> float:
        """Delegate the prior density factor."""
        return self._wrapped.prior_factor(params_in)

    def correct_bounds(self, params_in: NDArray[np.floating]) -> NDArray[np.floating]:
        """Delegate bounds correction."""
        return self._wrapped.correct_bounds(params_in)

    def check_bounds(self, params_in: NDArray[np.floating]) -> bool:
        """Delegate the bounds check."""
        return self._wrapped.check_bounds(params_in)

    def get_epsilons(self) -> NDArray[np.floating]:
        """Delegate Fisher epsilons, bridging duck-typed likelihoods.

        The eggbox jitclass exposes an epsilons attribute instead of the
        get_epsilons method; fall back to zeros like AbstractLikelihood.
        """
        get_eps = getattr(self._wrapped, 'get_epsilons', None)
        if callable(get_eps):
            return np.asarray(get_eps(), dtype=np.float64)
        eps_attr = getattr(self._wrapped, 'epsilons', None)
        if eps_attr is not None:
            return np.asarray(eps_attr, dtype=np.float64).copy()
        return np.zeros(self.n_par)


# one constructor per spec likelihood name; a test asserts the keys stay in
# sync with spec.LIKELIHOOD_NAMES (spec.py cannot import these back without a
# circular import, so drift is caught by CI instead)
LIKELIHOOD_BUILDERS: dict[str, Callable[..., LikelihoodLike]] = {
    'gaussian': GaussianLikelihood,
    'cake': CakeLikelihood,
    'eggbox': EggboxLikelihood,
    'hawaii': HawaiiLikelihood,
}


def build_likelihood(spec: RunSpec) -> LikelihoodLike:
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


def _build_geometric_ladder(spec: RunSpec) -> TemperatureLadder:
    """Construct a geometric ladder from the spec's ladder table."""
    ladder = spec.ladder
    return GeometricTemperatureLadder(
        spec.n_chain,
        n_cold=spec.n_cold,
        T_cold=_scalar(ladder.get('T_cold', 1.)),
        T_min=_scalar(ladder.get('T_min', 1.)),
        T_max=_scalar(ladder.get('T_max', 1.e15)),
        n_inf_final=int(_scalar(ladder.get('n_inf_final', 1))),
    )


def _build_entropy_file_ladder(spec: RunSpec) -> TemperatureLadder:
    """Construct an entropy ladder from reference data files named by the spec."""
    ladder = spec.ladder
    return entropy_ladder_fromfile(
        spec.n_chain,
        spec.n_cold,
        str(resolve(str(ladder['Ts_file']))),
        str(resolve(str(ladder['vars_file']))),
        n_inf_final=int(_scalar(ladder.get('n_inf_final', 1))),
        T_cold=_scalar(ladder.get('T_cold', 1.)),
        correct_last=bool(ladder.get('correct_last', False)),
    )


def _load_ladder_inputs(spec: RunSpec, *stat_file_keys: str) -> tuple[np.ndarray, ...]:
    """Load Ts plus stat arrays named by the spec, with the from-file filter.

    The Ts array is always loaded from 'Ts_file' explicitly (no
    positional first-key contract) and the shared engine helper
    filter_ladder_inputs applies the Ts >= 1 from-file convention, so
    every file-driven ladder arm filters identically.
    """
    Ts_in = np.load(resolve(str(spec.ladder['Ts_file'])))
    stats = [np.load(resolve(str(spec.ladder[key]))) for key in stat_file_keys]
    return filter_ladder_inputs(Ts_in, *stats)


def _build_length_file_ladder(spec: RunSpec) -> TemperatureLadder:
    """Construct a thermodynamic-length ladder from reference data files."""
    ladder = spec.ladder
    Ts_in, vars_in = _load_ladder_inputs(spec, 'vars_file')
    return LengthTemperatureLadder(
        spec.n_chain,
        Ts_in,
        vars_in,
        n_cold=spec.n_cold,
        T_cold=_scalar(ladder.get('T_cold', 1.)),
        n_inf_final=int(_scalar(ladder.get('n_inf_final', 1))),
        correct_last=bool(ladder.get('correct_last', False)),
    )


def _build_acceptance_file_ladder(spec: RunSpec) -> TemperatureLadder:
    """Construct a predicted-acceptance ladder from reference data files."""
    ladder = spec.ladder
    Ts_in, means_in, vars_in = _load_ladder_inputs(spec, 'means_file', 'vars_file')
    return AcceptanceTemperatureLadder(
        spec.n_chain,
        Ts_in,
        means_in,
        vars_in,
        n_cold=spec.n_cold,
        T_cold=_scalar(ladder.get('T_cold', 1.)),
        n_inf_final=int(_scalar(ladder.get('n_inf_final', 1))),
    )


def _build_explicit_ladder(spec: RunSpec) -> TemperatureLadder:
    """Construct a ladder directly from the spec's Ts list.

    RunSpec validation guarantees Ts is a numeric list of length n_chain.
    """
    Ts_raw = spec.ladder['Ts']
    if not isinstance(Ts_raw, list):
        msg = 'explicit ladder requires a Ts list'
        raise TypeError(msg)
    return TemperatureLadder(spec.n_cold, np.asarray(Ts_raw, dtype=np.float64))


# one builder per spec ladder kind; a test asserts the keys stay in sync with
# spec.LADDER_KINDS (see LIKELIHOOD_BUILDERS note)
LADDER_BUILDERS: dict[str, Callable[[RunSpec], TemperatureLadder]] = {
    'geometric': _build_geometric_ladder,
    'entropy_file': _build_entropy_file_ladder,
    'length_file': _build_length_file_ladder,
    'acceptance_file': _build_acceptance_file_ladder,
    'explicit': _build_explicit_ladder,
}


def build_ladder(spec: RunSpec) -> TemperatureLadder:
    """Construct the temperature ladder described by the spec."""
    kind = spec.ladder['kind']
    builder = LADDER_BUILDERS.get(str(kind))
    if builder is None:
        msg = f'unknown ladder kind {kind!r}'
        raise ValueError(msg)
    return builder(spec)


def build_sampler(spec: RunSpec, config: configparser.ConfigParser | None = None) -> tuple[DTMCMCSampler, CountingLikelihood]:
    """Build the sampler and counting-proxy likelihood for a spec.

    Assumes both RNG streams are already seeded (see run_from_spec):
    starting samples, DE-buffer fills, and Fisher initialization all draw
    from the run streams. Pass the config explicitly to share one instance
    between the sampler and the artifact provenance (run_from_spec does).
    """
    like_obj = CountingLikelihood(build_likelihood(spec))
    T_ladder = build_ladder(spec)
    if config is None:
        config = spec.build_proposal_config()

    starting_samples = np.zeros((T_ladder.n_chain, like_obj.n_par))
    for itrt in range(T_ladder.n_chain):
        starting_samples[itrt, :] = like_obj.prior_draw()

    exchange_manager = ExchangeManager(
        EXCHANGE_STRATEGY_CODES[spec.exchange_strategy],
        track_full_exchanges=spec.track_full_exchanges,
    )
    proposal_manager = get_default_proposal_manager(
        T_ladder,
        like_obj,
        starting_samples=starting_samples,
        config=config,
        exchange_manager_loc=exchange_manager,
    )

    sampler = DTMCMCSampler(
        T_ladder,
        like_obj,
        spec.block_size,
        spec.store_size,
        proposal_manager=proposal_manager,
        starting_samples=starting_samples,
        store_thin=spec.store_thin,
        n_record=spec.n_record,
    )
    return sampler, like_obj


def run_from_spec(spec: RunSpec, out_dir: str | Path, artifact_name: str | None = None) -> Path:
    """Execute one run end to end and return the artifact path.

    Chdirs to the repo root (engine-internal relative paths), seeds both
    RNG streams from the spec seed (once per process, plan D1), advances
    block by block, flushes the artifact every checkpoint, and finalizes.
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
        spec.seed, child_seed_python, child_seed_numba,
        spec_toml=spec.to_toml_text(), proposal_config_ini=config_to_text(config),
    )

    sampler, like_obj = build_sampler(spec, config=config)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    artifact_path = out_path / (artifact_name if artifact_name is not None else f'{spec.name}_seed{spec.seed}.h5')

    # checkpoint metrics draw from a dedicated Generator seeded by the run
    # seed: reproducible, recorded, and independent of both run RNG
    # streams (plan D5) — the golden digest is unaffected
    metrics_rng = get_rng(spec.seed)
    checkpoints = CheckpointLog()
    de_manager = next((manager for manager in sampler.proposal_manager.managers if isinstance(manager, DEJumpManager)), None)

    def record_checkpoint_metrics() -> None:
        if de_manager is not None:
            checkpoints.itrns.append(sampler.itrn)
            checkpoints.de_spectrum_eigvals.append(de_buffer_difference_spectrum(de_manager.de_buffer, DE_SPECTRUM_PAIRS, metrics_rng))

    for itr_block in range(spec.n_blocks):
        sampler.advance_block()
        blocks_done = itr_block + 1
        if blocks_done % spec.checkpoint_every_blocks == 0 and blocks_done < spec.n_blocks:
            record_checkpoint_metrics()
            write_artifact(
                artifact_path, spec, sampler, like_obj.n_evals, provenance,
                finalized=False, wall_seconds=time.monotonic() - start_monotonic, checkpoints=checkpoints,
            )

    record_checkpoint_metrics()
    write_artifact(
        artifact_path, spec, sampler, like_obj.n_evals, provenance,
        finalized=True, wall_seconds=time.monotonic() - start_monotonic, checkpoints=checkpoints,
    )
    return artifact_path
