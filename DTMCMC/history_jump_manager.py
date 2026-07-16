"""C 2023 Matthew C. Digman
blank manager to serve as template for adding more draw types
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from DTMCMC.jump_manager import AbstractJump, JumpManager

if TYPE_CHECKING:
    from configparser import ConfigParser

    from numpy.typing import NDArray

    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder

# TODO test if this works
# TODO create a specialized likelihood object that makes this useful


@dataclass(init=False)
class HistoryStrategyParameters:
    """container to store some parameters related to the strategy of proposal generation"""

    history_jump_weight: float

    def __init__(self, config: ConfigParser) -> None:
        """Initialize the object with the prescribed parameters"""
        config_h = config['LadderHistoryJumpManager']
        self.history_jump_weight = config_h.getfloat('ladder_history_jump_weight', 0.0)

    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to the requested configuration object
        inputs:
            config_in: ConfigParser object
        """
        config_h = config_in['LadderHistoryJumpManager']
        config_h['history_jump_weight'] = str(self.history_jump_weight)


class LadderHistoryJumpManager(JumpManager):
    """manager for a jump that proposes jumps to historical states
    at different temperatures
    """

    def __init__(
        self,
        T_ladder: TemperatureLadder,
        like_obj: AbstractLikelihood,
        config: ConfigParser,
        T_ladder_old: TemperatureLadder,
        logLs_old: NDArray[np.floating],
        states_old: NDArray[np.floating],
    ) -> None:
        """A blank"""
        self.T_ladder_old: TemperatureLadder = T_ladder_old
        self.states_old: NDArray[np.floating] = states_old
        self.logLs_old: NDArray[np.floating] = logLs_old

        self.strategy_params: HistoryStrategyParameters = HistoryStrategyParameters(config)

        jumps: list[AbstractJump] = [LadderHistoryJump(self)]

        JumpManager.__init__(self, T_ladder, like_obj, jumps)

    def set_jump_weights(self) -> None:
        """Set the relative probabilities of the different jump types"""
        n_chain: int = self.T_ladder.n_chain
        jump_weights: NDArray[np.floating] = np.zeros((n_chain, self.n_jump_types))
        # default to equal weight
        jump_weights[:] = self.strategy_params.history_jump_weight

        self.jump_weights = jump_weights
        assert np.all(self.jump_weights >= 0.0)

    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)


class LadderHistoryJump(AbstractJump):
    """Get a proposal from a random draw from the recorded historical points"""

    def __init__(self, manager: LadderHistoryJumpManager) -> None:
        """Get the object to propose ladder history draws"""
        self.manager: LadderHistoryJumpManager = manager
        self.print_name = 'Ladder History'

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        """Draw a future point from the stored set of past points"""
        del itrt
        itrt_target = np.random.randint(0, self.manager.T_ladder_old.n_chain)
        idx_target = np.random.randint(0, self.manager.logLs_old.shape[0])

        # this jump-internal evaluation is invisible to the sampler's
        # LikelihoodEvalTracker, which only counts orchestrated call sites
        logL_cur = self.manager.like_obj.get_loglike(sample_point)
        logL_new = self.manager.logLs_old[idx_target, itrt_target]

        new_point = self.manager.states_old[idx_target, itrt_target].copy()

        # The density factor is the same as the density factor for an exchange proposal
        density_fac = self.manager.T_ladder_old.betas[itrt_target] * (logL_cur - logL_new)
        return new_point, density_fac, True
