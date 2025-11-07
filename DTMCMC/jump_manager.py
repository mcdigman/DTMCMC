"""C 2023 Matthew C. Digman
Abstract class for the interface a proposal manager must export
in order to be properly recognized by the framework
"""

from abc import ABC, abstractmethod

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.likelihood import AbstractLikelihood
from DTMCMC.temperature_ladder_helpers import TemperatureLadder

# TODO update docs
# TODO jump name length check


class AbstractJump(ABC):
    """An object that performs a single proposal from its __call__ method"""

    def __init__(self, print_name: str) -> None:
        """Create the jump object:
            inputs:
                print_name: a string to print as the formatted name of this jump
        """
        self.print_name: str = print_name

    def get_print_name(self) -> str:
        """Retrieve the formatted name of the jump
        Outputs:
            print_name: a string to print as the formatted name of this jump
        """
        return self.print_name

    @abstractmethod
    def __call__(self, sample_point, itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        """Perform and MCMC proposal
            inputs:
                sample_point: a numpy array with the current point
                itrt: the index of the requested temperature chain
            outputs:
                new_point: a numpy array with the proposed new point
                density_factor: a scalar float, proposal density factor
                    if no proposal density factor is needed can just be set to 0.
                success: a boolean, whether generating the proposal succeeded
        """
        return np.zeros(sample_point.size), 0., True


@njit()
def choose_prob_helper(jump_probs: NDArray[np.floating]) -> int:
    """Helper that picks a random integer with the given input probabilities"""
    choose_val: float = np.random.uniform(0., 1)
    choose_sum: float = jump_probs[0]
    choose: int = jump_probs.size - 1
    for itrp in range(1, jump_probs.size):
        if choose_val < choose_sum:
            choose = itrp - 1
            break
        choose_sum += jump_probs[itrp]
    return choose


class JumpManager(ABC):
    """mcmc proposals should be dispatched from extensions of this class"""

    def __init__(self, T_ladder: TemperatureLadder, like_obj: AbstractLikelihood, jumps: list[AbstractJump]) -> None:
        """Default constructor that handles all the common actions we expect to need"""
        self.T_ladder: TemperatureLadder = T_ladder
        self.like_obj: AbstractLikelihood = like_obj
        self.n_chain: int = self.T_ladder.n_chain
        self.n_par: int = self.like_obj.n_par

        # self.jump_names = jump_names
        self.jumps: list[AbstractJump] = jumps
        self.n_jump_types = len(jumps)

        self.jump_probs: NDArray[np.floating] = np.zeros((self.n_chain, self.n_jump_types))
        self.jump_weights: NDArray[np.floating] = np.zeros((self.n_chain, self.n_jump_types))

        # self.jump_labels_array = np.array([jump_labels_dict.get(name, name) for name in jump_names])
        self.jump_labels_array: list[str] = [jump.get_print_name() for jump in self.jumps]

        self.name_to_idx: dict[str, int] = {}
        for itrm, name in enumerate(self.jump_labels_array):
            self.name_to_idx[name] = itrm

        self.set_jump_probs()

    def dispatch_jump(self, sample_point, itrt: int, choose: int = -1):
        """Dispatch the specified proposal
            inputs:
                sample_point: 1D float array, the parameters of the current point
                itrt: scalar integer, the index of the temperature chain for which to dispatch a proposal
                choose: scalar int, optional, an index that the dispatcher may use to select which proposal to try
                        if choose is not set, then try to jump according to the specified probability matrix

            returns:
                new_point: 1D float array, the parameter of the new point
                density_fac: a scalar float for the density factor of the proposal,
                                will be added to the log likelihood to modify the acceptance probability
                success: scalar boolean, whether or not generating the proposal succeeded
                            (if not, the proposal will automatically be marked rejected)
                choose: scalar int, index of the chosen jump type
        """
        if choose == -1:
            # choose the jump
            choose = choose_prob_helper(self.jump_probs[itrt])
        else:
            # validate the input choice if it is forced
            assert 0 <= choose < self.n_jump_types

        new_point, density_fac, success = self.jumps[choose](sample_point, itrt)
        return new_point, density_fac, success, choose

    def set_jump_weights(self) -> None:
        """Set the relative jump probabilities as a function of temperature for each jump type the manager exports
        based on a given strategy parameter object
        """
        jump_weights = np.zeros((self.n_chain, self.n_jump_types))
        # just a default equal weight
        jump_weights[:] = 0.333
        self.jump_weights = jump_weights

    @abstractmethod
    def record_config(self, config_in) -> None:
        """Do any necessary steps to record the current configuration of the manager
        to the input ConfigParser object config_in
        """
        return

    def set_jump_probs(self) -> None:
        """Set the normalized probabilities of the jump subtypes
        as a function of temperature, relying on the set_jump_weights
        methods which must be provided in subclasses
        """
        # unnormalized jump weights must be provided for in a subclass
        self.set_jump_weights()

        assert np.all(self.jump_weights >= 0.)

        if np.any(self.jump_weights != 0.):
            # get the normalized conditional jump probabilities
            self.jump_probs = (self.jump_weights.T / self.jump_weights.sum(axis=1)).T
            self.jump_probs[~np.isfinite(self.jump_probs)] = 0.
        else:
            self.jump_probs = np.zeros((self.n_chain, self.n_jump_types))

        assert np.all(self.jump_probs >= 0.)

        for itrt in range(self.jump_probs.shape[0]):
            # sanity check that all rows are either normalized  to 1 or sum to 0
            sum_check: float = float(np.sum(self.jump_probs[itrt]))
            assert sum_check == 0. or sum_check == 1.

    def get_jump_weights(self) -> NDArray[np.floating]:
        """Get the desired weights of this jump type as a function of temperature"""
        return self.jump_weights

    def get_jump_labels(self) -> list[str]:
        """Get text labels for the different jump types"""
        return self.jump_labels_array.copy()

    def get_jumps(self):
        """Return the list of available jumps"""
        return self.jumps

    def post_step_update(self, samples) -> None:
        """Do any needed internal processing after an individual step of all temperatures;
        mainly intended to be used to write to differential evolution buffer
        inputs:
            samples: 2D float array of samples
        """
        del samples
        return

    def post_block_update(self, itrn: int, block_size: int, samples, logLs) -> None:
        """Do any needed internal processing after an individual block of size block_size:
        ie, fisher matrix updates
        inputs:
            itrn: int, the current index of the chain state
            block_size: int, the number of steps in this block
            samples: 3D float array of samples
            logLs: 2D float array of likelihoods
        """
        del itrn
        del block_size
        del samples
        del logLs
