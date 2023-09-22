"""C 2023 Matthew C. Digman
blank manager to serve as template for adding more draw types"""

import numpy as np

from DTMCMC.jump_manager import JumpManager

# define unique codes for each jump type
AUXILLIARY_JUMPS = np.array([],dtype=np.int64)

# dictionary of display names for the jumps
JUMP_LABELS = {}
JUMP_LABELS_ARRAY = np.array([JUMP_LABELS[code] for code in AUXILLIARY_JUMPS])


class AuxilliaryJumpManager(JumpManager):
    """template manager for an extra jump type, subclass of DTMCMC.jump_manager.JumpManager"""

    def __init__(self, T_ladder, strategy_params, like_obj):
        """a blank """
        self.like_obj = like_obj
        self.n_chain = T_ladder.n_chain
        self.T_ladder = T_ladder
        self.strategy_params = strategy_params

        self.n_jump_types = AUXILLIARY_JUMPS.size
        self.jump_probs = np.zeros((self.n_chain, self.n_jump_types))
        self.jump_weights = np.zeros((self.n_chain, self.n_jump_types))
        self.jumps_need = AUXILLIARY_JUMPS.copy()
        self.jump_labels_array = JUMP_LABELS_ARRAY.copy()

        # map the codes that exist to indices in jumps_need
        if self.jumps_need.size > 0:
            self.code_to_idx = np.zeros(self.jumps_need.max()+1, dtype=np.int64)-1
        else:
            self.code_to_idx = np.array([])

        for idx, code in enumerate(self.jumps_need):
            self.code_to_idx[code] = idx

        self.set_jump_weights()

    def dispatch_jump(self, sample_point, itrt, choose_code):
        """dispatch the auxilliary jumps"""
        raise NotImplementedError('No jumps can be dispatched for this jump type')

    def set_jump_weights(self):
        """set the relative probabilities of the different jump types"""
        n_chain = self.T_ladder.n_chain
        jump_weights = np.zeros((n_chain, self.n_jump_types))
        jump_weights[:] = 1./3.  # just a default equal weight

        self.jump_weights = jump_weights
        self.jump_probs = (self.jump_weights.T/self.jump_weights.sum(axis=1)).T  # the normalized conditional jump probabilities
        self.jump_probs[~np.isfinite(self.jump_probs)]=0.

    def get_jump_weights(self):
        """get the desired weights of this jump type as a function of temperature"""
        return self.jump_weights

    def get_jump_codes(self):
        """return the internal codes the manager object uses to index its respective jump types"""
        return AUXILLIARY_JUMPS.copy()

    def get_jump_labels(self):
        """get text labels for the different jump types"""
        return self.jump_labels_array.copy()

    def post_step_update(self, samples):
        """do any needed internal processing after an individual step of all temperatures;
        mainly intended to be used to write to differential evolution buffer"""
        return

    def post_block_update(self, itrn, block_size, samples, logLs):
        """do any needed internal processing after an individual block of size block_size:
        ie, fisher matrix updates"""
        return
