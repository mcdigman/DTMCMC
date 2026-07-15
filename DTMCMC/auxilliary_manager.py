"""C 2023 Matthew C. Digman
blank manager to serve as template for adding more draw types
"""

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.jump_manager import AbstractJump, JumpManager
from DTMCMC.numba_backend import NativeLikelihoodState, jittable_jump, jittable_jump_manager

if TYPE_CHECKING:
    from configparser import ConfigParser

    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder


class AuxilliaryNativeState(NamedTuple):
    """Empty native state for the built-in blank proposal."""


@njit(inline='always')
def _blank_jump_native(
    sample_point: NDArray[np.floating],
    _itrt: int,
    _manager_state: AuxilliaryNativeState,
    _likelihood: NativeLikelihoodState,
) -> tuple[NDArray[np.floating], float, bool]:
    return sample_point.copy(), 0.0, True


@jittable_jump(_blank_jump_native)
class BlankJump(AbstractJump):
    """Template jump for future extensions"""

    def __init__(self, manager: JumpManager) -> None:
        self.manager: JumpManager = manager
        self.print_name = 'Blank Jump'

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

    def copy(self) -> AuxilliaryStrategyParameters:
        """Copy the object"""
        return copy(self)

    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to the requested configuration object
        inputs:
            config_in: ConfigParser object
        """
        config_a = config_in['AuxilliaryJumpManager']
        config_a['auxilliary_jump_weight'] = str(self.auxilliary_jump_weight)


def _get_auxilliary_native_state(_manager: Any) -> AuxilliaryNativeState:
    return AuxilliaryNativeState()


def _set_auxilliary_native_state(_manager: Any, _state: AuxilliaryNativeState) -> None:
    """Auxiliary manager native state is empty."""


@njit(inline='always')
def _post_auxilliary_native_state(
    state: AuxilliaryNativeState, _samples: NDArray[np.floating]
) -> AuxilliaryNativeState:
    return state


@jittable_jump_manager(
    state_getter=_get_auxilliary_native_state,
    state_setter=_set_auxilliary_native_state,
    post_step=_post_auxilliary_native_state,
)
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
