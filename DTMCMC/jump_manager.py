"""C 2023 Matthew C. Digman
Abstract class for the interface a proposal manager must export
in order to be properly recognized by the framework"""

from abc import ABC, abstractmethod


class JumpManager(ABC):
    """mcmc proposals should be dispatched from extensions of this class"""

    @abstractmethod
    def dispatch_jump(self, sample_point, itrt, choose_code):
        """dispatch the specified proposal
            inputs:
                sample_point: 1D float array, the parameters of the current point
                itrt: scalar integer, the index of the temperature chain for which to dispatch a proposal
                choose_code: an ID that the dispatcher may use to select which proposal to try
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
    def get_jump_weights(self):
        """get the desired weights of this jump type as a function of temperature
            returns:
                a 2d float array of relative weights"""
        return

    @abstractmethod
    def get_jump_codes(self):
        """return the internal codes the manager object uses to index its respective jump types"""

    @abstractmethod
    def get_jump_labels(self):
        """get text labels for the different jump types as numpy array of strings"""

    @abstractmethod
    def post_step_update(self, samples):
        """do any needed internal processing after an individual step of all temperatures;
        mainly intended to be used to write to differential evolution buffer
        inputs:
            samples: 2D float array of samples"""
        return

    @abstractmethod
    def post_block_update(self, itrn, block_size, samples, logLs):
        """do any needed internal processing after an individual block of size block_size:
        ie, fisher matrix updates
        inputs:
            itrn: int, the current index of the chain state
            block_size: int, the number of steps in this block
            samples: 3D float array of samples
            logLs: 2D float array of likelihoods"""
