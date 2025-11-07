"""C 2023 Matthew C. Digman
manager to manage prior-draw based jumps
"""

import numpy as np

from DTMCMC.jump_manager import AbstractJump, JumpManager


class PriorFullJump(AbstractJump):

    def __init__(self, manager) -> None:
        self.manager = manager
        AbstractJump.__init__(self, 'Prior All-D')

    def __call__(self, sample_point, itrt):
        del itrt
        new_point = self.manager.like_obj.prior_draw()
        density_fac = self.manager.like_obj.prior_factor(sample_point) - self.manager.like_obj.prior_factor(new_point)
        return new_point, density_fac, True


class PriorStrategyParameters():
    """container to store some parameters related to the strategy of proposal generation"""

    def __init__(self, config) -> None:
        """Initialize the object with the prescribed parameters"""
        self.config = config
        config_p = config['PriorManager']
        # how often to do prior draws in the cold chains
        self.cold_prior_weight = config_p.getfloat('cold_prior_weight', 0.333)
        # how often to do prior draws in the hottest finite temperature chain
        self.hot_prior_target_weight = config_p.getfloat('hot_prior_target_weight', 0.333)

    def copy(self):
        """Copy the object"""
        return PriorStrategyParameters(self.config)

    def record_config(self, config_in) -> None:
        """Record the current configuration to the requested configuration object
            inputs:
                config_in: ConfigParser object
        """
        config_prior = config_in['PriorManager']
        config_prior['cold_prior_weight'] = str(self.cold_prior_weight)
        config_prior['hot_prior_target_weight'] = str(self.hot_prior_target_weight)


class PriorManager(JumpManager):
    """manage prior draw-based jumps, subclass of DTMCMC.jump_manager.JumpManager"""

    def __init__(self, T_ladder, like_obj, config) -> None:
        """Take a likelihood object and create an object that can propose prior draws"""
        self.strategy_params = PriorStrategyParameters(config)

        jumps = [PriorFullJump(self)]

        JumpManager.__init__(self, T_ladder, like_obj, jumps)

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

        jump_weights[n_chain - 1, :] = 0.
        jump_weights[n_chain - 1, idx_prior_full] = 1.
        jump_weights[n_cold:n_chain - 1, idx_prior_full] = np.linspace(cold_prior_weight, hot_prior_weight, n_chain - n_cold)[1:]
        jump_weights[:n_cold, idx_prior_full] = cold_prior_weight

        self.jump_weights = jump_weights
        assert np.all(self.jump_weights >= 0.)

    def record_config(self, config_in) -> None:
        """Record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)
