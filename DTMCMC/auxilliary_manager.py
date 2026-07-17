"""C 2023 Matthew C. Digman
blank manager to serve as template for adding more draw types
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.jump_manager import AbstractJump, JumpManager
from DTMCMC.numba_backend import NativeJumpCall, NativeLikelihoodFunctions

if TYPE_CHECKING:
    from configparser import ConfigParser

    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder


@njit(inline='always')
def _blank_jump_native(
    sample_point: NDArray[np.floating], _itrt: int, _state: None, _like_state: object
) -> tuple[NDArray[np.floating], float, bool]:
    return sample_point.copy(), 0.0, True


class BlankJump[LikelihoodType: AbstractLikelihood[NamedTuple]](AbstractJump[LikelihoodType]):
    """Template jump for future extensions"""

    declared_internal_evals = 0

    def __init__(self, manager: JumpManager[LikelihoodType]) -> None:
        self.manager: JumpManager[LikelihoodType] = manager
        self.print_name = 'Blank Jump'

    def bind_native(self, likelihood_natives: NativeLikelihoodFunctions[object]) -> NativeJumpCall[None, object]:
        """The blank jump is stateless, so the per-class module function suffices."""
        del likelihood_natives
        return _blank_jump_native

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        """Call the jump"""
        del itrt
        return sample_point.copy(), 0.0, True


@dataclass(init=False)
class AuxilliaryStrategyParameters:
    """container to store some parameters related to the strategy of proposal generation"""

    auxilliary_jump_weight: float

    def __init__(self, config: ConfigParser) -> None:
        """Initialize the object with the prescribed parameters"""
        config_a = config['AuxilliaryJumpManager']
        self.auxilliary_jump_weight = config_a.getfloat('auxilliary_jump_weight', 0.0)

    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to the requested configuration object
        inputs:
            config_in: ConfigParser object
        """
        config_a = config_in['AuxilliaryJumpManager']
        config_a['auxilliary_jump_weight'] = str(self.auxilliary_jump_weight)


class AuxilliaryJumpManager[LikelihoodType: AbstractLikelihood[NamedTuple]](JumpManager[LikelihoodType]):
    """template manager for an extra jump type,
    subclass of DTMCMC.jump_manager.JumpManager
    """

    def __init__(self, T_ladder: TemperatureLadder, like_obj: LikelihoodType, config: ConfigParser) -> None:
        """A blank proposal as a template"""
        self.strategy_params = AuxilliaryStrategyParameters(config)

        jumps: list[AbstractJump[LikelihoodType]] = [BlankJump(self)]

        JumpManager.__init__(self, T_ladder, like_obj, jumps)

    @override
    def set_jump_weights(self) -> None:
        """Set the relative probabilities of the different jump types"""
        n_chain: int = self.T_ladder.n_chain
        jump_weights: NDArray[np.floating] = np.zeros((n_chain, self.n_jump_types))

        # default to equal weight
        jump_weights[:] = self.strategy_params.auxilliary_jump_weight

        self._jump_weights = jump_weights

        assert np.all(self._jump_weights >= 0.0)

    @override
    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)
