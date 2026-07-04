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
    from numpy.typing import NDArray

from DTMCMC.dtmcmc_sampler import DTMCMCSampler
from DTMCMC.exchange_manager import ExchangeManager
from DTMCMC.likelihood import AbstractLikelihood
from DTMCMC.likelihoods.cake_likelihood import CakeLikelihood
from DTMCMC.likelihoods.eggbox import Likelihood as EggboxLikelihood
from DTMCMC.likelihoods.hawaii_likelihood import HawaiiLikelihood
from DTMCMC.likelihoods.normal_nd import GaussianLikelihood
from DTMCMC.proposal_manager_helper import get_default_proposal_manager
from DTMCMC.rng_helpers import seed_run
from DTMCMC.temperature_ladder_helpers import (
    GeometricTemperatureLadder,
    TemperatureLadder,
    entropy_ladder_fromfile,
)

from .artifact import RunProvenance, collect_provenance, write_artifact
from .paths import chdir_repo_root, resolve
from .spec import EXCHANGE_STRATEGY_CODES, RunSpec


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


def build_likelihood(spec: RunSpec) -> LikelihoodLike:
    """Construct the likelihood object named by the spec."""
    params: dict[str, Any] = dict(spec.likelihood_params)
    if spec.likelihood_name == 'gaussian':
        return GaussianLikelihood(**params)
    if spec.likelihood_name == 'cake':
        return CakeLikelihood(**params)
    if spec.likelihood_name == 'eggbox':
        return EggboxLikelihood(**params)
    if spec.likelihood_name == 'hawaii':
        return HawaiiLikelihood(**params)
    msg = f'unknown likelihood {spec.likelihood_name!r}'
    raise ValueError(msg)


def _scalar(value: object) -> float:
    """Narrow a spec scalar to float-convertible, rejecting lists."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f'expected a numeric scalar, got {value!r}'
        raise TypeError(msg)
    return float(value)


def build_ladder(spec: RunSpec) -> TemperatureLadder:
    """Construct the temperature ladder described by the spec."""
    ladder = spec.ladder
    kind = ladder['kind']
    n_chain = spec.n_chain
    n_cold = spec.n_cold

    if kind == 'geometric':
        return GeometricTemperatureLadder(
            n_chain,
            n_cold=n_cold,
            T_cold=float(_scalar(ladder.get('T_cold', 1.))),
            T_min=float(_scalar(ladder.get('T_min', 1.))),
            T_max=float(_scalar(ladder.get('T_max', 1.e15))),
            n_inf_final=int(_scalar(ladder.get('n_inf_final', 1))),
        )
    if kind == 'entropy_file':
        return entropy_ladder_fromfile(
            n_chain,
            n_cold,
            str(resolve(str(ladder['Ts_file']))),
            str(resolve(str(ladder['vars_file']))),
            n_inf_final=int(_scalar(ladder.get('n_inf_final', 1))),
            T_cold=float(_scalar(ladder.get('T_cold', 1.))),
            correct_last=bool(ladder.get('correct_last', False)),
        )
    if kind == 'explicit':
        Ts_raw = ladder['Ts']
        if not isinstance(Ts_raw, list):
            msg = 'explicit ladder requires a Ts list'
            raise TypeError(msg)
        return TemperatureLadder(n_cold, np.asarray(Ts_raw, dtype=np.float64))
    msg = f'unknown ladder kind {kind!r}'
    raise ValueError(msg)


def build_sampler(spec: RunSpec) -> tuple[DTMCMCSampler, CountingLikelihood]:
    """Build the sampler and counting-proxy likelihood for a spec.

    Assumes both RNG streams are already seeded (see run_from_spec):
    starting samples, DE-buffer fills, and Fisher initialization all draw
    from the run streams.
    """
    like_obj = CountingLikelihood(build_likelihood(spec))
    T_ladder = build_ladder(spec)
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
    provenance: RunProvenance = collect_provenance(spec.seed, child_seed_python, child_seed_numba)

    sampler, like_obj = build_sampler(spec)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    artifact_path = out_path / (artifact_name if artifact_name is not None else f'{spec.name}_seed{spec.seed}.h5')

    for itr_block in range(spec.n_blocks):
        sampler.advance_block()
        blocks_done = itr_block + 1
        if blocks_done % spec.checkpoint_every_blocks == 0 and blocks_done < spec.n_blocks:
            write_artifact(
                artifact_path, spec, sampler, like_obj.n_evals, provenance,
                finalized=False, wall_seconds=time.monotonic() - start_monotonic,
            )

    write_artifact(
        artifact_path, spec, sampler, like_obj.n_evals, provenance,
        finalized=True, wall_seconds=time.monotonic() - start_monotonic,
    )
    return artifact_path
