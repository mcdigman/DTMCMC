"""C 2023 Matthew C. Digman
blank manager to serve as template for adding more draw types
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.jump_manager import AbstractNativeJump, JumpManager

if TYPE_CHECKING:
    from configparser import ConfigParser

    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder


class AuxilliaryNativeState(NamedTuple): ...


@njit(inline='always')
def _blank_jump_native(
    sample_point: NDArray[np.floating],
    _itrt: int,
    _state: AuxilliaryNativeState,
) -> tuple[NDArray[np.floating], float, bool]:
    return sample_point.copy(), 0.0, True


class BlankJump[LikelihoodType: AbstractLikelihood[Any]](AbstractNativeJump[LikelihoodType, AuxilliaryNativeState]):
    """Template jump for future extensions"""

    def __init__(self, manager: JumpManager[LikelihoodType, AuxilliaryNativeState]) -> None:
        print_name = 'Blank Jump'
        handle = _blank_jump_native
        super().__init__(handle, manager, print_name)

    @property
    @override
    def declared_internal_evals(self) -> int:
        return 0


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


class AuxilliaryJumpManager[LikelihoodType: AbstractLikelihood[Any]](
    JumpManager[LikelihoodType, AuxilliaryNativeState]
):
    """template manager for an extra jump type,
    subclass of DTMCMC.jump_manager.JumpManager
    """

    def __init__(self, T_ladder: TemperatureLadder, like_obj: LikelihoodType, config: ConfigParser) -> None:
        """A blank proposal as a template"""
        self._strategy_params: AuxilliaryStrategyParameters = AuxilliaryStrategyParameters(config)

        jumps: list[AbstractNativeJump[LikelihoodType, AuxilliaryNativeState]] = [BlankJump(self)]

        self._native_state: AuxilliaryNativeState = AuxilliaryNativeState()

        super().__init__(T_ladder, like_obj, jumps)

    @property
    def strategy_params(self) -> AuxilliaryStrategyParameters:
        return self._strategy_params

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

    @property
    @override
    def native_state(self) -> AuxilliaryNativeState:
        return self._native_state
