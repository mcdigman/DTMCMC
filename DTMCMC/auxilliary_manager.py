"""C 2023 Matthew C. Digman
blank manager to serve as template for adding more draw types"""

import numpy as np

from DTMCMC.jump_manager import JumpManager

JUMP_NAMES = ["BLANK_JUMP"]

# dictionary of display names for the jumps
JUMP_LABELS_DICT = {"BLANK_JUMP": "blank jump"}

class AuxilliaryJumpManager(JumpManager):
    """template manager for an extra jump type,
    subclass of DTMCMC.jump_manager.JumpManager"""

    def __init__(self, T_ladder, like_obj, config):
        """a blank proposal as a template"""
        self.strategy_params = AuxilliaryStrategyParameters(config)

        JumpManager.__init__(self, T_ladder, like_obj, JUMP_NAMES, JUMP_LABELS_DICT)

    def dispatch_jump(self, sample_point, itrt, choose):
        """dispatch the auxilliary jumps"""
        if choose == 0:
            # A blank jump as an example, fill in your own jump here!
            return sample_point.copy(), 0., True
        else:
            assert False

        return sample_point.copy(), 0., True

    def set_jump_weights(self):
        """set the relative probabilities of the different jump types"""
        n_chain = self.T_ladder.n_chain
        jump_weights = np.zeros((n_chain, self.n_jump_types))
        # default to equal weight
        jump_weights[:] = self.strategy_params.auxilliary_jump_weight

        self.jump_weights = jump_weights

        assert np.all(self.jump_weights >= 0.)

    def record_config(self, config_in):
        """record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)


class AuxilliaryStrategyParameters():
    """container to store some parameters related to the strategy of proposal generation"""

    def __init__(self, config):
        """initialize the object with the prescribed parameters"""
        self.config = config
        config_a = self.config['AuxilliaryJumpManager']
        self.auxilliary_jump_weight = config_a.getfloat('auxilliary_jump_weight', 0.)

    def copy(self):
        """copy the object"""
        return AuxilliaryStrategyParameters(self.config)

    def record_config(self, config_in):
        """record the current configuration to the requested configuration object
            inputs:
                config_in: ConfigParser object"""
        config_a = config_in['AuxilliaryJumpManager']
        config_a['auxilliary_jump_weight'] = str(self.auxilliary_jump_weight)
