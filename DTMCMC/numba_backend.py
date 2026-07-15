"""Generated nopython backend for the serial PTMCMC block transition.

The public sampler and extension objects remain ordinary Python objects. A
likelihood class may opt into this backend with :func:`jittable_likelihood`,
which attaches native callables and a state extractor without replacing the
class or changing its Python fallback behavior.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numba import njit
from numba.core.errors import NumbaError
from numpy.typing import NDArray

from DTMCMC.auxilliary_manager import AuxilliaryJumpManager, BlankJump
from DTMCMC.de_manager import (
    DEBigFullJump,
    DEBigRandomSubspaceJump,
    DEJumpManager,
    DEStandardFullJump,
    DEStandardRandomSubspaceJump,
    apply_de_helper,
)
from DTMCMC.exchange_manager import do_ptmcmc_exchange
from DTMCMC.fisher_manager import (
    FisherJumpManager,
    SigmaFullJump,
    SigmaRandomSubspaceJump,
    sigma_subspace_jump_helper,
)
from DTMCMC.likelihood import prior_draw_rectangular, validate_bounds_rectangular
from DTMCMC.mcmc_kernel_helpers import mcmc_decision_helper
from DTMCMC.prior_manager import PriorFullJump, PriorManager
from DTMCMC.proposal_manager import ProposalManager

if TYPE_CHECKING:
    from collections.abc import Callable

    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder
    from DTMCMC.tracker_manager import TrackerManager


JUMP_UNSUPPORTED = -1
JUMP_SIGMA_FULL = 0
JUMP_SIGMA_SUBSPACE = 1
JUMP_DE_STANDARD_FULL = 2
JUMP_DE_STANDARD_SUBSPACE = 3
JUMP_DE_BIG_FULL = 4
JUMP_DE_BIG_SUBSPACE = 5
JUMP_PRIOR = 6
JUMP_BLANK = 7

_EMPTY_SIGMA_SCALES = np.zeros((1, 1))
_EMPTY_DE_BUFFER = np.zeros((1, 1, 1))


@njit(inline='always')
def _uniform_prior_draw(
    n_par: int,
    low_lims: NDArray[np.floating],
    high_lims: NDArray[np.floating],
    _state: tuple[Any, ...],
) -> NDArray[np.floating]:
    return prior_draw_rectangular(n_par, low_lims, high_lims)


@njit(inline='always')
def _uniform_prior_factor(_params: NDArray[np.floating], _state: tuple[Any, ...]) -> float:
    return 0.0


@njit(inline='always')
def _rectangular_validate_bounds(
    params: NDArray[np.floating],
    low_lims: NDArray[np.floating],
    high_lims: NDArray[np.floating],
    _state: tuple[Any, ...],
) -> tuple[NDArray[np.floating], bool]:
    return validate_bounds_rectangular(params, low_lims, high_lims)


def _ensure_njit(func: Callable[..., Any]) -> Any:
    """Return a Numba dispatcher for ``func`` without double-wrapping one."""
    if hasattr(func, 'py_func'):
        return func
    return njit(func)


def _adapt_loglike(loglike: Callable[..., float], n_state: int) -> Any:
    """Adapt ``loglike(params, *state)`` to ``loglike(params, state)``."""
    native_loglike: Callable[..., float] = _ensure_njit(loglike)
    if n_state == 0:

        @njit(inline='always')
        def adapted0(params: NDArray[np.floating], _state: tuple[Any, ...]) -> float:
            return native_loglike(params)

        return adapted0

    if n_state == 1:

        @njit(inline='always')
        def adapted1(params: NDArray[np.floating], state: tuple[Any, ...]) -> float:
            return native_loglike(params, state[0])

        return adapted1

    if n_state == 2:

        @njit(inline='always')
        def adapted2(params: NDArray[np.floating], state: tuple[Any, ...]) -> float:
            return native_loglike(params, state[0], state[1])

        return adapted2

    if n_state == 3:

        @njit(inline='always')
        def adapted3(params: NDArray[np.floating], state: tuple[Any, ...]) -> float:
            return native_loglike(params, state[0], state[1], state[2])

        return adapted3

    if n_state == 4:

        @njit(inline='always')
        def adapted4(params: NDArray[np.floating], state: tuple[Any, ...]) -> float:
            return native_loglike(params, state[0], state[1], state[2], state[3])

        return adapted4

    msg = 'jittable_likelihood currently supports at most four state attributes'
    raise ValueError(msg)


def _default_bounds_getter(like_obj: AbstractLikelihood) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Get rectangular bounds without copying them."""
    return np.asarray(like_obj.low_lims), np.asarray(like_obj.high_lims)  # type: ignore[attr-defined]


@dataclass
class NativeLikelihoodSpec:
    """Native functions and Python-side state extraction for a likelihood."""

    loglike: Any
    state_getter: Callable[[AbstractLikelihood], tuple[Any, ...]]
    bounds_getter: Callable[[AbstractLikelihood], tuple[NDArray[np.floating], NDArray[np.floating]]]
    prior_draw: Any
    prior_factor: Any
    validate_bounds: Any
    serial_kernel: Any = None


def jittable_likelihood(
    loglike: Callable[..., float],
    *,
    state_attrs: tuple[str, ...] = (),
    prior_draw: Callable[..., NDArray[np.floating]] | None = None,
    prior_factor: Callable[..., float] | None = None,
    validate_bounds: Callable[..., tuple[NDArray[np.floating], bool]] | None = None,
    bounds_getter: Callable[[AbstractLikelihood], tuple[NDArray[np.floating], NDArray[np.floating]]] | None = None,
) -> Callable[[type], type]:
    """Attach a serial-native likelihood contract to an ordinary class.

    ``loglike`` has the user-facing native signature ``(params, *state)``;
    values named by ``state_attrs`` are read from each instance at block
    entry. Rectangular uniform-prior behavior is generated by default.

    Non-uniform priors may supply standardized native functions with these
    signatures::

        prior_draw(n_par, low_lims, high_lims, state) -> params
        prior_factor(params, state) -> log_density

    A custom validator, if supplied, has signature
    ``validate_bounds(params, low_lims, high_lims, state)``.
    """
    adapted_loglike = _adapt_loglike(loglike, len(state_attrs))
    native_prior_draw = _uniform_prior_draw if prior_draw is None else _ensure_njit(prior_draw)
    native_prior_factor = _uniform_prior_factor if prior_factor is None else _ensure_njit(prior_factor)
    native_validate = _rectangular_validate_bounds if validate_bounds is None else _ensure_njit(validate_bounds)
    get_bounds = _default_bounds_getter if bounds_getter is None else bounds_getter

    def decorate(cls: type) -> type:
        def get_state(like_obj: AbstractLikelihood) -> tuple[Any, ...]:
            return tuple(getattr(like_obj, name) for name in state_attrs)

        cls.__dtmcmc_numba_likelihood__ = NativeLikelihoodSpec(  # type: ignore[attr-defined]
            loglike=adapted_loglike,
            state_getter=get_state,
            bounds_getter=get_bounds,
            prior_draw=native_prior_draw,
            prior_factor=native_prior_factor,
            validate_bounds=native_validate,
        )
        return cls

    return decorate


@dataclass
class _SerialProposalInputs:
    jump_codes: NDArray[np.int64]
    sigma_scales: NDArray[np.floating]
    fisher_subspace_frac: float
    de_buffer: NDArray[np.floating]
    de_subspace_frac: float
    de_thin: int
    de_write: int
    de_count: int
    de_manager: DEJumpManager | None


_ALLOWED_MANAGER_TYPES = {FisherJumpManager, DEJumpManager, AuxilliaryJumpManager, PriorManager}


def _proposal_code(jump: object) -> int:
    """Translate a built-in jump object to its native integer code."""
    jump_type = type(jump)
    if jump_type is SigmaFullJump:
        return JUMP_SIGMA_FULL
    if jump_type is SigmaRandomSubspaceJump:
        return JUMP_SIGMA_SUBSPACE
    if jump_type is DEStandardFullJump:
        return JUMP_DE_STANDARD_FULL
    if jump_type is DEStandardRandomSubspaceJump:
        return JUMP_DE_STANDARD_SUBSPACE
    if jump_type is DEBigFullJump:
        return JUMP_DE_BIG_FULL
    if jump_type is DEBigRandomSubspaceJump:
        return JUMP_DE_BIG_SUBSPACE
    if jump_type is PriorFullJump:
        return JUMP_PRIOR
    if jump_type is BlankJump:
        return JUMP_BLANK
    return JUMP_UNSUPPORTED


def _get_serial_proposal_inputs(proposal_manager: ProposalManager) -> _SerialProposalInputs | None:
    """Translate an eligible built-in proposal graph to native inputs."""
    if type(proposal_manager) is not ProposalManager:
        return None
    if any(type(manager) not in _ALLOWED_MANAGER_TYPES for manager in proposal_manager.managers):
        return None

    jump_codes = np.asarray([_proposal_code(jump) for jump in proposal_manager.jumps], dtype=np.int64)
    active = np.any(proposal_manager.jump_probs > 0.0, axis=0)
    if np.any(active & (jump_codes == JUMP_UNSUPPORTED)):
        return None

    fisher_managers = [manager for manager in proposal_manager.managers if type(manager) is FisherJumpManager]
    de_managers = [manager for manager in proposal_manager.managers if type(manager) is DEJumpManager]
    if len(fisher_managers) > 1 or len(de_managers) > 1:
        return None

    if fisher_managers:
        fisher_manager = fisher_managers[0]
        sigma_scales = fisher_manager.sigma_scales
        fisher_subspace_frac = float(fisher_manager.strategy_params.fisher_subspace_frac)
    else:
        sigma_scales = _EMPTY_SIGMA_SCALES
        fisher_subspace_frac = 1.0

    if de_managers:
        de_manager = de_managers[0]
        de_buffer = de_manager.de_buffer
        de_subspace_frac = float(de_manager.de_subspace_frac)
        de_thin = de_manager.de_thin
        de_write = de_manager.itrde_write
        de_count = de_manager.itrde_count
    else:
        de_manager = None
        de_buffer = _EMPTY_DE_BUFFER
        de_subspace_frac = 1.0
        de_thin = 1
        de_write = 0
        de_count = 0

    return _SerialProposalInputs(
        jump_codes,
        sigma_scales,
        fisher_subspace_frac,
        de_buffer,
        de_subspace_frac,
        de_thin,
        de_write,
        de_count,
        de_manager,
    )


def _make_serial_kernel(spec: NativeLikelihoodSpec) -> Any:
    """Generate the monomorphic block kernel for one likelihood contract."""
    loglike = spec.loglike
    prior_draw = spec.prior_draw
    prior_factor = spec.prior_factor
    validate_bounds = spec.validate_bounds

    @njit()
    def advance_block_numba_serial(
        betas: NDArray[np.floating],
        logLs: NDArray[np.floating],
        samples: NDArray[np.floating],
        chain_track: NDArray[np.int64],
        jump_probs: NDArray[np.floating],
        jump_codes: NDArray[np.int64],
        sigma_scales: NDArray[np.floating],
        fisher_subspace_frac: float,
        de_buffer: NDArray[np.floating],
        de_subspace_frac: float,
        de_thin: int,
        de_write_in: int,
        de_count_in: int,
        de_enabled: bool,
        exchange_strategy: int,
        track_full_exchanges: bool,
        exchange_tracker: NDArray[np.int64],
        esd_exchange: NDArray[np.floating],
        accept_record: NDArray[np.int64],
        esd_record: NDArray[np.floating],
        low_lims: NDArray[np.floating],
        high_lims: NDArray[np.floating],
        like_state: tuple[Any, ...],
    ) -> tuple[int, int, int]:
        block_size = samples.shape[0] - 1
        n_chain = samples.shape[1]
        n_par = samples.shape[2]
        de_write = de_write_in
        de_count = de_count_in
        n_evals = 0

        for itrb in range(1, block_size + 1):
            if itrb % 2 == 0:
                do_ptmcmc_exchange(
                    itrb - 1,
                    samples,
                    logLs,
                    n_chain,
                    betas,
                    exchange_tracker,
                    esd_exchange,
                    chain_track,
                    exchange_strategy,
                    track_full_exchanges,
                )
            else:
                for itrt in range(n_chain):
                    choose_val = np.random.uniform(0.0, 1.0)
                    choose_sum = jump_probs[itrt, 0]
                    idx_jump = jump_probs.shape[1] - 1
                    for itrj in range(1, jump_probs.shape[1]):
                        if choose_val < choose_sum:
                            idx_jump = itrj - 1
                            break
                        choose_sum += jump_probs[itrt, itrj]

                    jump_code = jump_codes[idx_jump]
                    sample_point = samples[itrb - 1, itrt]
                    if jump_code == JUMP_SIGMA_FULL:
                        new_point, density_fac, success = sigma_subspace_jump_helper(
                            sample_point,
                            itrt,
                            n_par,
                            fisher_subspace_frac,
                            sigma_scales,
                            True,
                        )
                    elif jump_code == JUMP_SIGMA_SUBSPACE:
                        new_point, density_fac, success = sigma_subspace_jump_helper(
                            sample_point,
                            itrt,
                            n_par,
                            fisher_subspace_frac,
                            sigma_scales,
                            False,
                        )
                    elif jump_code == JUMP_DE_STANDARD_FULL:
                        new_point, density_fac, success = apply_de_helper(
                            de_buffer, de_subspace_frac, itrt, sample_point, False, False
                        )
                    elif jump_code == JUMP_DE_STANDARD_SUBSPACE:
                        new_point, density_fac, success = apply_de_helper(
                            de_buffer, de_subspace_frac, itrt, sample_point, True, False
                        )
                    elif jump_code == JUMP_DE_BIG_FULL:
                        new_point, density_fac, success = apply_de_helper(
                            de_buffer, de_subspace_frac, itrt, sample_point, False, True
                        )
                    elif jump_code == JUMP_DE_BIG_SUBSPACE:
                        new_point, density_fac, success = apply_de_helper(
                            de_buffer, de_subspace_frac, itrt, sample_point, True, True
                        )
                    elif jump_code == JUMP_PRIOR:
                        new_point = prior_draw(n_par, low_lims, high_lims, like_state)
                        density_fac = prior_factor(sample_point, like_state) - prior_factor(new_point, like_state)
                        success = True
                    else:
                        new_point = sample_point.copy()
                        density_fac = 0.0
                        success = jump_code == JUMP_BLANK

                    if success:
                        new_point, success = validate_bounds(new_point, low_lims, high_lims, like_state)
                    if success:
                        logL_new = loglike(new_point, like_state)
                        n_evals += 1
                    else:
                        logL_new = -np.inf

                    mcmc_decision_helper(
                        itrb,
                        samples,
                        logLs,
                        betas,
                        accept_record,
                        esd_record,
                        itrt,
                        new_point,
                        logL_new,
                        density_fac,
                        idx_jump,
                    )
                chain_track[itrb, :] = chain_track[itrb - 1, :]

            if de_enabled:
                if de_count == 0:
                    de_buffer[de_write, :, :] = samples[itrb]
                    de_write += 1
                    if de_write == de_buffer.shape[0]:
                        de_write = 0
                de_count += 1
                if de_count >= de_thin:
                    de_count = 0

        return de_write, de_count, n_evals

    return advance_block_numba_serial


def try_advance_block_numba_serial(
    T_ladder: TemperatureLadder,
    logLs: NDArray[np.floating],
    samples: NDArray[np.floating],
    chain_track: NDArray[np.int64],
    proposal_manager: ProposalManager,
    like_obj: AbstractLikelihood,
    tracker_manager: TrackerManager,
) -> bool:
    """Try the serial native kernel, returning false for silent fallback."""
    # Opt-in is per concrete class.  An undecorated subclass may override
    # methods or add state that invalidates its parent's generated kernel.
    spec = type(like_obj).__dict__.get('__dtmcmc_numba_likelihood__')
    if not isinstance(spec, NativeLikelihoodSpec):
        return False
    proposal_inputs = _get_serial_proposal_inputs(proposal_manager)
    if proposal_inputs is None:
        return False

    if spec.serial_kernel is None:
        spec.serial_kernel = _make_serial_kernel(spec)

    try:
        low_lims, high_lims = spec.bounds_getter(like_obj)
        like_state = spec.state_getter(like_obj)
        de_write, de_count, n_evals = spec.serial_kernel(
            T_ladder.betas,
            logLs,
            samples,
            chain_track,
            proposal_manager.jump_probs,
            proposal_inputs.jump_codes,
            proposal_inputs.sigma_scales,
            proposal_inputs.fisher_subspace_frac,
            proposal_inputs.de_buffer,
            proposal_inputs.de_subspace_frac,
            proposal_inputs.de_thin,
            proposal_inputs.de_write,
            proposal_inputs.de_count,
            proposal_inputs.de_manager is not None,
            proposal_manager.exchange_manager.strategy,
            proposal_manager.exchange_manager.track_full_exchanges,
            tracker_manager.exchange_tracker,
            tracker_manager.esd_exchange,
            tracker_manager.accept_record,
            tracker_manager.esd_record,
            low_lims,
            high_lims,
            like_state,
        )
    except NumbaError:  # compilation failures select the Python fallback
        return False

    if proposal_inputs.de_manager is not None:
        proposal_inputs.de_manager.itrde_write = de_write
        proposal_inputs.de_manager.itrde_count = de_count
    like_obj.n_evals += n_evals
    return True
