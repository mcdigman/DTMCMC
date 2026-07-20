"""C 2023 Matthew C. Digman
manager to manage prior-draw based jumps
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.jump_manager import AbstractJump, JumpManager
from DTMCMC.likelihood import PriorDrawFn, PriorFactorFn, prior_draw_handle, prior_factor_handle
from DTMCMC.numba_backend import NativeJumpCall

if TYPE_CHECKING:
    from configparser import ConfigParser

    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder

# composition memo keyed on the likelihood's handles: repeated binds return
# the same jitted function, so samplers whose likelihoods resolve to the
# same handles share one compiled program (holding the keys alive keeps
# ids stable)
_PRIOR_NATIVE_MEMO: dict[tuple[object, object], NativeJumpCall[None]] = {}


def _make_prior_native(prior_draw: PriorDrawFn, prior_factor: PriorFactorFn) -> NativeJumpCall[None]:
    """Compose the likelihood's prior draw and density handles into a jump."""
    key = (prior_draw, prior_factor)
    cached = _PRIOR_NATIVE_MEMO.get(key)
    if cached is not None:
        return cached

    @njit(inline='always')
    def prior_native(
        sample_point: NDArray[np.floating], _itrt: int, _manager_state: None
    ) -> tuple[NDArray[np.floating], float, bool]:
        new_point = prior_draw()
        density_fac = prior_factor(sample_point) - prior_factor(new_point)
        return new_point, density_fac, True

    _PRIOR_NATIVE_MEMO[key] = prior_native
    return prior_native


class PriorFullJump[LikelihoodType: AbstractLikelihood](AbstractJump[LikelihoodType]):
    declared_internal_evals = 0

    def __init__(self, manager: JumpManager[LikelihoodType]) -> None:
        self.manager: JumpManager[LikelihoodType] = manager
        self.print_name = 'Prior All-D'

    def bind_native(self) -> NativeJumpCall[None]:
        """Compose the likelihood's own prior draw and density handles."""
        like_obj = self.manager.like_obj
        return _make_prior_native(prior_draw_handle(like_obj), prior_factor_handle(like_obj))

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        del itrt
        new_point = self.manager.like_obj.prior_draw()
        density_fac = self.manager.like_obj.prior_factor(sample_point) - self.manager.like_obj.prior_factor(new_point)
        return new_point, density_fac, True


@dataclass(init=False)
class PriorStrategyParameters:
    """container to store some parameters related to the strategy of proposal generation"""

    cold_prior_weight: float
    hot_prior_target_weight: float

    def __init__(self, config: ConfigParser) -> None:
        """Initialize the object with the prescribed parameters"""
        config_p = config['PriorManager']
        # how often to do prior draws in the cold chains
        self.cold_prior_weight = config_p.getfloat('cold_prior_weight', 0.333)
        # how often to do prior draws in the hottest finite temperature chain
        self.hot_prior_target_weight = config_p.getfloat('hot_prior_target_weight', 0.333)

    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to the requested configuration object
        inputs:
            config_in: ConfigParser object
        """
        config_prior = config_in['PriorManager']
        config_prior['cold_prior_weight'] = str(self.cold_prior_weight)
        config_prior['hot_prior_target_weight'] = str(self.hot_prior_target_weight)


class PriorManager[LikelihoodType: AbstractLikelihood](JumpManager[LikelihoodType]):
    """manage prior draw-based jumps, subclass of DTMCMC.jump_manager.JumpManager"""

    def __init__(self, T_ladder: TemperatureLadder, like_obj: LikelihoodType, config: ConfigParser) -> None:
        """Take a likelihood object and create an object that can propose prior draws"""
        self.strategy_params = PriorStrategyParameters(config)

        jumps: list[AbstractJump[LikelihoodType]] = [PriorFullJump(self)]

        JumpManager.__init__(self, T_ladder, like_obj, jumps)

    @override
    def set_jump_weights(self) -> None:
        """Set the relative probabilities of the different jump types"""
        n_cold = self.T_ladder.n_cold
        n_chain = self.T_ladder.n_chain
        jump_weights = np.zeros((n_chain, self.n_jump_types))
        # just a default equal weight
        jump_weights[:] = 0.333

        cold_prior_weight = self.strategy_params.cold_prior_weight
        hot_prior_weight = self.strategy_params.hot_prior_target_weight

        idx_prior_full = -1
        for itrp, jump in enumerate(self.jumps):
            if isinstance(jump, PriorFullJump):
                idx_prior_full = itrp

        assert idx_prior_full >= -1

        jump_weights[n_chain - 1, :] = 0.0
        jump_weights[n_chain - 1, idx_prior_full] = 1.0
        jump_weights[n_cold : n_chain - 1, idx_prior_full] = np.linspace(
            cold_prior_weight, hot_prior_weight, n_chain - n_cold
        )[1:]
        jump_weights[:n_cold, idx_prior_full] = cold_prior_weight

        self._jump_weights = jump_weights
        assert np.all(self._jump_weights >= 0.0)

    @override
    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)
