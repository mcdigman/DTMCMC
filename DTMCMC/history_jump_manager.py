"""C 2023 Matthew C. Digman
blank manager to serve as template for adding more draw types
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, override

import numpy as np

from DTMCMC.jump_manager import AbstractNativeJump, JumpManager

if TYPE_CHECKING:
    from configparser import ConfigParser

    from numpy.typing import NDArray

    from DTMCMC.likelihood import AbstractLikelihood, LoglikeFn
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder

# TODO test if this works
# TODO create a specialized likelihood object that makes this useful


class HistoryNativeState[LikelihoodType: AbstractLikelihood[Any]](NamedTuple):
    T_ladder: TemperatureLadder
    T_ladder_old: TemperatureLadder
    loglike_fn: LoglikeFn
    logLs_old: NDArray[np.floating]
    states_old: NDArray[np.floating]


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


class LadderHistoryJumpManager[LikelihoodType: AbstractLikelihood[Any]](
    JumpManager[LikelihoodType, HistoryNativeState[LikelihoodType]]
):
    """manager for a jump that proposes jumps to historical states
    at different temperatures
    """

    def __init__(
        self,
        T_ladder: TemperatureLadder,
        like_obj: LikelihoodType,
        config: ConfigParser,
        T_ladder_old: TemperatureLadder,
        logLs_old: NDArray[np.floating],
        states_old: NDArray[np.floating],
    ) -> None:
        """A blank"""
        self.strategy_params: HistoryStrategyParameters = HistoryStrategyParameters(config)

        jumps: list[AbstractNativeJump[LikelihoodType, HistoryNativeState[LikelihoodType]]] = [LadderHistoryJump(self)]
        self._native_state: HistoryNativeState[LikelihoodType] = HistoryNativeState(
            T_ladder, T_ladder_old, like_obj.loglike_fn_baked, logLs_old, states_old
        )

        super().__init__(T_ladder, like_obj, jumps)

    @property
    @override
    def native_state(self) -> HistoryNativeState[LikelihoodType]:
        return self._native_state

    @override
    def set_jump_weights(self) -> None:
        """Set the relative probabilities of the different jump types"""
        n_chain: int = self.T_ladder.n_chain
        jump_weights: NDArray[np.floating] = np.zeros((n_chain, self.n_jump_types))
        # default to equal weight
        jump_weights[:] = self.strategy_params.history_jump_weight

        self._jump_weights = jump_weights
        assert np.all(self._jump_weights >= 0.0)

    @override
    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)


def _ladder_history_native[LikelihoodType: AbstractLikelihood[Any]](
    sample_point: NDArray[np.floating], itrt: int, state: HistoryNativeState[LikelihoodType]
) -> tuple[NDArray[np.floating], float, bool]:
    """Draw a future point from the stored set of past points"""
    del itrt
    itrt_target = np.random.randint(0, state.T_ladder_old.n_chain)
    idx_target = np.random.randint(0, state.logLs_old.shape[0])

    # this jump-internal evaluation is reported to the sampler's
    # eval accounting through declared_internal_evals above
    logL_cur = state.loglike_fn(sample_point)
    logL_new = state.logLs_old[idx_target, itrt_target]

    new_point = state.states_old[idx_target, itrt_target].copy()

    # The density factor is the same as the density factor for an exchange proposal
    density_fac = state.T_ladder_old.betas[itrt_target] * (logL_cur - logL_new)
    return new_point, density_fac, True


class LadderHistoryJump[LikelihoodType: AbstractLikelihood[NamedTuple]](
    AbstractNativeJump[LikelihoodType, HistoryNativeState[LikelihoodType]]
):
    """Get a proposal from a random draw from the recorded historical points"""

    def __init__(self, manager: LadderHistoryJumpManager[LikelihoodType]) -> None:
        """Get the object to propose ladder history draws"""
        print_name = 'Ladder History'
        super().__init__(_ladder_history_native, manager, print_name)

    @property
    @override
    def declared_internal_evals(self) -> int:
        # each dispatch evaluates the likelihood once at the current point
        return 1
