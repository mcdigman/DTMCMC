"""C 2023 Matthew C. Digman
Module to manage differential evoultion jumps
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.jump_manager import AbstractJump, JumpManager

if TYPE_CHECKING:
    from configparser import ConfigParser

    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.numba_backend import NativeJumpCall, NativeLikelihoodFunctions, NativePostStepCall
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder


@njit()
def apply_de_helper(
    de_buffer: NDArray[np.floating],
    de_subspace_frac: float,
    itrt: int,
    sample_point: NDArray[np.floating],
    do_subspace: bool,
    do_big: bool,
) -> tuple[NDArray[np.floating], float, bool]:
    """Apply the differential evolution jump"""
    de_size: int = de_buffer.shape[0]
    n_par: int = de_buffer.shape[2]

    itrd1: int = np.random.randint(0, de_size)
    itrd2: int = np.random.randint(0, de_size)

    if do_big:
        alpha: float = 1.0
    else:
        alpha = 1.68 / np.sqrt(n_par) * np.random.normal(0.0, 1.0)

    delta: NDArray[np.floating] = de_buffer[itrd1, itrt, :] - de_buffer[itrd2, itrt, :]
    count: int = n_par
    if do_subspace:
        safe_itrp = np.random.randint(n_par)
        for itrp in range(n_par):
            if np.random.uniform(0.0, 1.0) > de_subspace_frac and itrp != safe_itrp:
                delta[itrp] = 0.0
                count -= 1
        assert count > 0

    # calculate the new point based on the shifts above
    new_point: NDArray[np.floating] = sample_point + alpha * delta

    # make sure something changed or else flag the jump as trivial
    nontrivial: bool = not np.all(delta == 0.0) and alpha != 0.0

    # density factor is 0 for differential evolution jumps
    return new_point, 0.0, nontrivial


def _bind_de_native(manager: DEJumpManager, do_subspace: bool, do_big: bool) -> NativeJumpCall:
    """Bind a differential-evolution jump reading the manager's runtime state.

    The ring buffer is mutated inside the block by the per-step update, so
    it must arrive through the manager's runtime native state rather than
    being baked (Numba treats captured arrays as read-only non-aliasing
    constants); only the immutable subspace fraction is baked.
    """
    de_subspace_frac = manager.de_subspace_frac

    @njit(inline='always')
    def native_call(
        sample_point: NDArray[np.floating], itrt: int, state: tuple[NDArray[np.floating], NDArray[np.int64]]
    ) -> tuple[NDArray[np.floating], float, bool]:
        de_buffer = state[0]
        return apply_de_helper(de_buffer, de_subspace_frac, itrt, sample_point, do_subspace, do_big)

    return native_call


class DEStandardFullJump(AbstractJump):
    """apply a jump with standard random size in all dimensions
    null proposals are marked as failures
    """

    def __init__(self, manager: DEJumpManager) -> None:
        self.manager: DEJumpManager = manager
        self.print_name = 'DE Std All-D'

    def bind_native(self, likelihood_natives: NativeLikelihoodFunctions) -> NativeJumpCall:
        """Bind the jump as a jitted closure reading the manager runtime state."""
        del likelihood_natives
        return _bind_de_native(self.manager, False, False)

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        return apply_de_helper(self.manager.de_buffer, self.manager.de_subspace_frac, itrt, sample_point, False, False)


class DEStandardRandomSubspaceJump(AbstractJump):
    """apply a jump with standard random size in a random subspace
    null proposals are marked as failures
    """

    def __init__(self, manager: DEJumpManager) -> None:
        self.manager: DEJumpManager = manager
        self.print_name = 'DE Std Random-D'

    def bind_native(self, likelihood_natives: NativeLikelihoodFunctions) -> NativeJumpCall:
        """Bind the jump as a jitted closure reading the manager runtime state."""
        del likelihood_natives
        return _bind_de_native(self.manager, True, False)

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        return apply_de_helper(self.manager.de_buffer, self.manager.de_subspace_frac, itrt, sample_point, True, False)


class DEBigFullJump(AbstractJump):
    """apply the full length differential evolution jump in all dimensions
    null proposals are marked as failures
    """

    def __init__(self, manager: DEJumpManager) -> None:
        self.manager: DEJumpManager = manager
        self.print_name = 'DE Big All-D'

    def bind_native(self, likelihood_natives: NativeLikelihoodFunctions) -> NativeJumpCall:
        """Bind the jump as a jitted closure reading the manager runtime state."""
        del likelihood_natives
        return _bind_de_native(self.manager, False, True)

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        return apply_de_helper(self.manager.de_buffer, self.manager.de_subspace_frac, itrt, sample_point, False, True)


class DEBigRandomSubspaceJump(AbstractJump):
    """apply the full length differential evolution jump in a random subspace
    null proposals are marked as failures
    """

    def __init__(self, manager: DEJumpManager) -> None:
        self.manager: DEJumpManager = manager
        self.print_name = 'DE Big Random-D'

    def bind_native(self, likelihood_natives: NativeLikelihoodFunctions) -> NativeJumpCall:
        """Bind the jump as a jitted closure reading the manager runtime state."""
        del likelihood_natives
        return _bind_de_native(self.manager, True, True)

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        return apply_de_helper(self.manager.de_buffer, self.manager.de_subspace_frac, itrt, sample_point, True, True)


def initialize_de_helper(
    de_buffer: NDArray[np.floating], de_size: int, n_chain: int, like_obj: AbstractLikelihood
) -> None:
    """Helper to initialize the differential evolution buffer with prior draws"""
    for itrd in range(de_size):
        for itrt in range(n_chain):
            de_buffer[itrd, itrt, :] = like_obj.prior_draw()


@njit()
def advance_de_state_helper(
    itrde_count: int,
    itrde_write: int,
    de_thin: int,
    de_buffer: NDArray[np.floating],
    samples: NDArray[np.floating],
) -> tuple[int, int]:
    """Write one DE state when due and advance its write/thinning counters."""
    if itrde_count == 0:
        de_buffer[itrde_write, :] = samples
        itrde_write += 1
        if itrde_write == de_buffer.shape[0]:
            itrde_write = 0
    itrde_count += 1
    if itrde_count >= de_thin:
        itrde_count = 0
    return itrde_write, itrde_count


@dataclass(init=False)
class DEStrategyParameters:
    """container to store some parameters related to the strategy of differential evolution proposal generation"""

    cold_de_weight: float
    hot_de_weight: float
    big_de_prob: float
    de_subspace_frac: float
    de_full_d_frac: float
    de_size: int
    de_thin: int

    def __init__(self, config: ConfigParser) -> None:
        """Initialize the object with the prescribed parameters"""
        config_de = config['DEJumpManager']

        # how often to do de draws in the cold chains
        self.cold_de_weight = config_de.getfloat('cold_de_weight', 0.333)
        # how often to do de draws in the hottest finite temperature chain
        self.hot_de_weight = config_de.getfloat('hot_de_weight', 0.333)
        # how often to do the big differential evolution jump
        self.big_de_prob = config_de.getfloat('big_de_prob', 0.5)
        # what fraction of dimensions to include in de subspace jumps
        self.de_subspace_frac = config_de.getfloat('de_subspace_frac', 1.0)
        # how often to not do subspace jumps when doing a de jump
        self.de_full_d_frac = config_de.getfloat('de_full_d_frac', 1.0)
        # size of differential evolution buffer
        self.de_size = config_de.getint('de_size', 1000)
        # how much to thin the differential evolution buffer by
        self.de_thin = config_de.getint('de_thin', 1)

    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to the requested configuration object
        inputs:
            config_in: ConfigParser object
        """
        config_de = config_in['DEJumpManager']
        config_de['cold_de_weight'] = str(self.cold_de_weight)
        config_de['hot_de_weight'] = str(self.hot_de_weight)
        config_de['big_de_prob'] = str(self.big_de_prob)
        config_de['de_subspace_frac'] = str(self.de_subspace_frac)
        config_de['de_full_d_frac'] = str(self.de_full_d_frac)
        config_de['de_size'] = str(self.de_size)
        config_de['de_thin'] = str(self.de_thin)


# TODO apply a global default jump weight
# TODO fix name lengths


class DEJumpManager(JumpManager):
    """manage the differential evolution jumps, subclass of DTMCMC.jump_manager.JumpManager

    The ring buffer and the write/thinning counter array are allocated once
    and mutated in place: native bindings bake them into compiled closures by
    reference, so their identity must be stable for the sampler's lifetime.
    """

    def __init__(self, T_ladder: TemperatureLadder, like_obj: AbstractLikelihood, config: ConfigParser) -> None:
        """Create the manager object"""
        self.strategy_params = DEStrategyParameters(config)

        self.de_thin: int = self.strategy_params.de_thin
        self.de_size: int = self.strategy_params.de_size
        self.de_subspace_frac: float = self.strategy_params.de_subspace_frac

        jumps: list[AbstractJump] = [
            DEStandardFullJump(self),
            DEStandardRandomSubspaceJump(self),
            DEBigFullJump(self),
            DEBigRandomSubspaceJump(self),
        ]

        JumpManager.__init__(self, T_ladder, like_obj, jumps)

        self.de_buffer = np.zeros((self.de_size, self.n_chain, self.n_par))
        initialize_de_helper(self.de_buffer, self.de_size, self.n_chain, self.like_obj)

        # identity-stable counter storage as (write, count); the int-valued
        # itrde_write/itrde_count properties are views into this array
        self._de_counters: NDArray[np.int64] = np.zeros(2, dtype=np.int64)
        self.itrde_write = 1
        self.itrde_count = 1

    @property
    def itrde_write(self) -> int:
        """Next ring-buffer row to write, backed by the counter array."""
        return int(self._de_counters[0])

    @itrde_write.setter
    def itrde_write(self, value: int) -> None:
        self._de_counters[0] = value

    @property
    def itrde_count(self) -> int:
        """Thinning countdown state, backed by the counter array."""
        return int(self._de_counters[1])

    @itrde_count.setter
    def itrde_count(self, value: int) -> None:
        self._de_counters[1] = value

    def bind_native_state(self) -> tuple[NDArray[np.floating], NDArray[np.int64]]:
        """Return the identity-stable mutable state shared by jumps and post-step."""
        return (self.de_buffer, self._de_counters)

    def bind_native_post_step(self) -> NativePostStepCall:
        """Bind the ring-buffer write as a jitted per-step closure.

        The buffer and counters are written per step, so they arrive through
        the runtime state; only the immutable thinning factor is baked.
        """
        de_thin = self.de_thin

        @njit(inline='always')
        def post_step_native(
            state: tuple[NDArray[np.floating], NDArray[np.int64]], samples: NDArray[np.floating]
        ) -> None:
            de_buffer, counters = state
            itrde_write, itrde_count = advance_de_state_helper(counters[1], counters[0], de_thin, de_buffer, samples)
            counters[0] = itrde_write
            counters[1] = itrde_count

        return post_step_native

    def write_de(self, samples: NDArray[np.floating]) -> None:
        """Write to the differential evolution buffer"""
        self.itrde_write, self.itrde_count = advance_de_state_helper(
            self.itrde_count,
            self.itrde_write,
            self.de_thin,
            self.de_buffer,
            samples,
        )

    def set_jump_weights(self) -> None:
        """Set the conditional probabilities of the different jump types"""
        n_chain: int = self.T_ladder.n_chain
        n_cold: int = self.T_ladder.n_cold

        cold_de_weight: float = self.strategy_params.cold_de_weight  # weight of de in cold proposals
        hot_de_weight: float = self.strategy_params.hot_de_weight  # weight of de in hot proposals
        de_full_frac: float = self.strategy_params.de_full_d_frac  # fraction of time not to do a subspace jump
        big_de_prob: float = self.strategy_params.big_de_prob  # probability of doing a full length de jump

        jump_weights = np.zeros((n_chain, self.n_jump_types))
        # just a default equal weight
        jump_weights[:] = 0.333

        standard_prob = 1 - self.strategy_params.big_de_prob  # probability of doing a standard jump
        subspace_prob = 1.0 - de_full_frac  # probability of doing a subspace jump

        standard_full_prob = standard_prob * de_full_frac  # probability of doing a standard full jump
        standard_subspace_prob = standard_prob * subspace_prob  # probability of doing a standard subspace jump

        big_subspace_prob = big_de_prob * subspace_prob  # probability of doing a full length jump in a subspace
        big_full_prob = big_de_prob * de_full_frac  # probability of doing a full length jump in a subspace

        # get the indices of the jump types we need to assign probabilities for
        de_standard_full_idx = -1
        de_standard_subspace_idx = -1
        de_big_full_idx = -1
        de_big_subspace_idx = -1
        for itrp, jump in enumerate(self.jumps):
            if isinstance(jump, DEStandardFullJump):
                de_standard_full_idx = itrp
            if isinstance(jump, DEStandardRandomSubspaceJump):
                de_standard_subspace_idx = itrp
            if isinstance(jump, DEBigFullJump):
                de_big_full_idx = itrp
            if isinstance(jump, DEBigRandomSubspaceJump):
                de_big_subspace_idx = itrp

        assert de_standard_full_idx >= 0
        assert de_standard_subspace_idx >= 0
        assert de_big_full_idx >= 0
        assert de_big_subspace_idx >= 0

        jump_weights[:n_cold, de_standard_full_idx] = cold_de_weight * standard_full_prob
        jump_weights[n_cold:, de_standard_full_idx] = hot_de_weight * standard_full_prob

        jump_weights[:n_cold, de_standard_subspace_idx] = cold_de_weight * standard_subspace_prob
        jump_weights[n_cold:, de_standard_subspace_idx] = hot_de_weight * standard_subspace_prob

        jump_weights[:n_cold, de_big_full_idx] = cold_de_weight * big_full_prob
        jump_weights[n_cold:, de_big_full_idx] = hot_de_weight * big_full_prob

        jump_weights[:n_cold, de_big_subspace_idx] = cold_de_weight * big_subspace_prob
        jump_weights[n_cold:, de_big_subspace_idx] = hot_de_weight * big_subspace_prob

        self.jump_weights = jump_weights
        assert np.all(self.jump_weights >= 0.0)

    def post_step_update(self, samples: NDArray[np.floating]) -> None:
        """Do any needed internal processing after an individual step of all temperatures;
        mainly intended to be used to write to differential evolution buffer
        """
        self.write_de(samples)

    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)
