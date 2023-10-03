"""C 2023 Matthew C. Digman
Abstract class for the interface a proposal manager must export
in order to be properly recognized by the framework"""

from abc import ABC, abstractmethod

import numpy as np

class JumpManager(ABC):
    """mcmc proposals should be dispatched from extensions of this class"""

    def __init__(self, T_ladder, like_obj, jump_names, jump_labels_dict):
        """Default constructor that handles all the common actions we expect to need"""

        self.T_ladder = T_ladder
        self.like_obj = like_obj
        self.n_chain = self.T_ladder.n_chain
        self.n_par = self.like_obj.n_par

        self.jump_names = jump_names
        self.n_jump_types = len(jump_names)

        self.jump_probs = np.zeros((self.n_chain, self.n_jump_types))
        self.jump_weights = np.zeros((self.n_chain, self.n_jump_types))

        self.jump_labels_array = np.array([jump_labels_dict.get(name, name) for name in jump_names])

        self.name_to_idx = {}
        for itrm, name in enumerate(self.jump_names):
            self.name_to_idx[name] = itrm

        self.set_jump_probs()

    @abstractmethod
    def dispatch_jump(self, sample_point, itrt, choose):
        """dispatch the specified proposal
            inputs:
                sample_point: 1D float array, the parameters of the current point
                itrt: scalar integer, the index of the temperature chain for which to dispatch a proposal
                choose: scalar int, an index that the dispatcher may use to select which proposal to try
            returns:
                new_point: 1D float array, the parameter of the new point
                density_fac: a scalar float for the density factor of the proposal,
                                will be added to the log likelihood to modify the acceptance probability
                success: scalar boolean, whether or not generating the proposal succeeded
                            (if not, the proposal will automatically be marked rejected)"""

    @abstractmethod
    def set_jump_weights(self):
        """set the relative jump probabilities as a function of temperature for each jump type the manager exports
        based on a given strategy parameter object"""

    @abstractmethod
    def record_config(self,config_in):
        """do any necessary steps to record the current configuration of the manager
        to the input ConfigParser object config_in"""

    def set_jump_probs(self):
        """set the normalized probabilities of the jump subtypes
        as a function of temperature, relying on the set_jump_weights
        methods which must be provided in subclasses"""
         
        #unnormalized jump weights must be provided for in a subclass
        self.set_jump_weights()

        assert np.all(self.jump_weights >= 0.)

        if np.any(self.jump_weights!=0.):
            # get the normalized conditional jump probabilities
            self.jump_probs = (self.jump_weights.T/self.jump_weights.sum(axis=1)).T
            self.jump_probs[~np.isfinite(self.jump_probs)] = 0.
        else:
            self.jump_probs = np.zeros((self.n_chain, self.n_jump_types))

        assert np.all(self.jump_probs >= 0.)

        for itrt in range(0,self.jump_probs.shape[0]):
            # sanity check that all rows are either normalized  to 1 or sum to 0
            sum_check = np.sum(self.jump_probs[itrt])
            assert sum_check == 0. or sum_check == 1.

    def get_jump_weights(self):
        """get the desired weights of this jump type as a function of temperature"""
        return self.jump_weights

    def get_jump_labels(self):
        """get text labels for the different jump types"""
        return self.jump_labels_array.copy()

    def get_jump_codes(self):
        """return the internal names the manager object uses
        to reference its respective jump types"""
        return self.jump_names.copy()

    def post_step_update(self, samples):
        """do any needed internal processing after an individual step of all temperatures;
        mainly intended to be used to write to differential evolution buffer
        inputs:
            samples: 2D float array of samples"""
        return

    def post_block_update(self, itrn, block_size, samples, logLs):
        """do any needed internal processing after an individual block of size block_size:
        ie, fisher matrix updates
        inputs:
            itrn: int, the current index of the chain state
            block_size: int, the number of steps in this block
            samples: 3D float array of samples
            logLs: 2D float array of likelihoods"""
        """do any needed internal processing after an individual block of size block_size:
        ie, fisher matrix updates"""
        return


#class DefaultJumpManager(JumpManager):
#    """manage some common logistics for default jump types to avoid repetitive code"""
#
#
#    def set_jump_probs(self):
#        """set the normalized probabilities of the jump subtypes
#        as a function of temperature, relying on the set_jump_weights
#        methods which must be provided in subclasses"""
#         
#        #unnormalized jump weights must be provided for in a subclass
#        self.set_jump_weights()
#
#        assert np.all(self.jump_weights >= 0.)
#
#        if np.any(self.jump_weights!=0.):
#            # get the normalized conditional jump probabilities
#            self.jump_probs = (self.jump_weights.T/self.jump_weights.sum(axis=1)).T
#            self.jump_probs[~np.isfinite(self.jump_probs)] = 0.
#        else:
#            self.jump_probs = np.zeros((self.n_chain, self.n_jump_types))
#
#        assert np.all(self.jump_probs >= 0.)
#
#    def get_jump_weights(self):
#        """get the desired weights of this jump type as a function of temperature"""
#        return self.jump_weights
#
#    def get_jump_labels(self):
#        """get text labels for the different jump types"""
#        return self.jump_labels_array.copy()
#
#    def get_jump_codes(self):
#        """return the internal names the manager object uses
#        to reference its respective jump types"""
#        return self.jump_names.copy()
#
#    def post_step_update(self, samples):
#        """do any needed internal processing after an individual step of all temperatures;
#        mainly intended to be used to write to differential evolution buffer"""
#        return
#
#    def post_block_update(self, itrn, block_size, samples, logLs):
#        """do any needed internal processing after an individual block of size block_size:
#        ie, fisher matrix updates"""
#        return
