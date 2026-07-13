"""C 2023 Matthew C. Digman
blank manager to serve as template for adding more draw types
"""

from typing import TYPE_CHECKING

import numpy as np

from DTMCMC.jump_manager import AbstractJump, JumpManager

if TYPE_CHECKING:
    from configparser import ConfigParser

    from numpy.typing import NDArray

    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder


class BlankJump(AbstractJump):
    """Template jump for future extensions"""

    def __init__(self, manager: JumpManager) -> None:
        self.manager: JumpManager = manager
        AbstractJump.__init__(self, 'Blank Jump')

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        """Call the jump"""
        del itrt
        return sample_point.copy(), 0.0, True


class AuxilliaryStrategyParameters:
    """container to store some parameters related to the strategy of proposal generation"""

    def __init__(self, config: ConfigParser) -> None:
        """Initialize the object with the prescribed parameters"""
        self.config: ConfigParser = config
        config_a = self.config['AuxilliaryJumpManager']
        self.auxilliary_jump_weight = config_a.getfloat('auxilliary_jump_weight', 0.0)

    def copy(self) -> AuxilliaryStrategyParameters:
        """Copy the object"""
        return AuxilliaryStrategyParameters(self.config)

    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to the requested configuration object
        inputs:
            config_in: ConfigParser object
        """
        config_a = config_in['AuxilliaryJumpManager']
        config_a['auxilliary_jump_weight'] = str(self.auxilliary_jump_weight)


class AuxilliaryJumpManager(JumpManager):
    """template manager for an extra jump type,
    subclass of DTMCMC.jump_manager.JumpManager
    """

    def __init__(self, T_ladder: TemperatureLadder, like_obj: AbstractLikelihood, config: ConfigParser) -> None:
        """A blank proposal as a template"""
        self.strategy_params = AuxilliaryStrategyParameters(config)

        jumps: list[AbstractJump] = [BlankJump(self)]

        JumpManager.__init__(self, T_ladder, like_obj, jumps)

    def set_jump_weights(self) -> None:
        """Set the relative probabilities of the different jump types"""
        n_chain: int = self.T_ladder.n_chain
        jump_weights: NDArray[np.floating] = np.zeros((n_chain, self.n_jump_types))

        # default to equal weight
        jump_weights[:] = self.strategy_params.auxilliary_jump_weight

        self.jump_weights = jump_weights

        assert np.all(self.jump_weights >= 0.0)

    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)
