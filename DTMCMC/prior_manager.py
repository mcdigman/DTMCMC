"""C 2023 Matthew C. Digman
manager to manage prior-draw based jumps"""

import numpy as np

from DTMCMC.jump_manager import JumpManager

JUMP_NAMES = ['PRIOR_FULL']
# dictionary of display names for the jumps
JUMP_LABELS_DICT = {'PRIOR_FULL': 'prior full'}


class PriorManager(JumpManager):
    """manage prior draw-based jumps, subclass of DTMCMC.jump_manager.JumpManager"""

    def __init__(self, T_ladder, like_obj, config):
        """take a likelihood object and create an object that can propose prior draws"""
        self.strategy_params = PriorStrategyParameters(config)

        JumpManager.__init__(self, T_ladder, like_obj, JUMP_NAMES, JUMP_LABELS_DICT)


    def dispatch_jump(self, sample_point, itrt, choose):
        """dispatch the prior draw jumps"""
        if choose == 0:
            return self.like_obj.prior_proposal(sample_point)
        else:
            assert False


    def set_jump_weights(self):
        """set the relative probabilities of the different jump types"""
        n_cold = self.T_ladder.n_cold
        n_chain = self.T_ladder.n_chain
        jump_weights = np.zeros((n_chain, self.n_jump_types))
        jump_weights[:] = 1./3.  # just a default equal weight

        cold_prior_weight = self.strategy_params.cold_prior_weight
        hot_prior_weight = self.strategy_params.hot_prior_target_weight

        idx_prior_full = self.name_to_idx['PRIOR_FULL']

        jump_weights[n_chain-1, :] = 0.
        jump_weights[n_chain-1, idx_prior_full] = 1.
        jump_weights[n_cold:n_chain-1, idx_prior_full] = np.linspace(cold_prior_weight, hot_prior_weight, n_chain-n_cold)[1:]
        jump_weights[:n_cold, idx_prior_full] = cold_prior_weight

        self.jump_weights = jump_weights
        assert np.all(self.jump_weights >= 0.)

    def record_config(self,config_in):
        """record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)


class PriorStrategyParameters():
    """container to store some parameters related to the strategy of proposal generation"""

    def __init__(self,config):
        """initialize the object with the prescribed parameters"""
        self.config = config
        config_p = config['PriorManager']
        # how often to do prior draws in the cold chains
        self.cold_prior_weight = config_p.getfloat('cold_prior_weight',0.333)
        # how often to do prior draws in the hottest finite temperature chain
        self.hot_prior_target_weight = config_p.getfloat('hot_prior_target_weight',0.333)


    def copy(self):
        """copy the object"""
        return PriorStrategyParameters(self.config)


    def record_config(self,config_in):
        """record the current configuration to the requested configuration object 
            inputs:
                config_in: ConfigParser object"""
        config_prior = config_in['PriorManager']
        config_prior['cold_prior_weight'] = str(self.cold_prior_weight)
        config_prior['hot_prior_target_weight'] = str(self.hot_prior_target_weight)
