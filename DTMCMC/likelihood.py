"""C 2023 Matthew C. Digman
abstract class to hold a likelihood object"""
from abc import ABC, abstractmethod
import numpy as np


class Likelihood(ABC):
    """abstract likelihood object"""

    @abstractmethod
    def get_loglike(self, params_in):
        """get the log likelihood at the specified parameters:
            input:
                params_in: a 1D float array of parameters
            output:
                logL: a scalar float likelihood"""
        return 0.

    @abstractmethod
    def prior_draw(self):
        """get a draw from the priors for this likelihood
            output:
                params: a 1D float array of parameters"""
        return np.zeros(1)

    @abstractmethod
    def prior_proposal(self, params_in):
        """get a proposal from the priors for this likelihood
            input:
                params_in: a 1D float array of parameters for the old point
            output:
                params_out: a 1D float array of parameters
                density_fac: a scalar density factor for the prior draw, connecting the old point to the new point"""
        return np.zeros(params_in.size), 0.

    @abstractmethod
    def prior_factor(self, params_in):
        """get the prior density factor for the input parameters
            input:
                params_in: the parameters to consider
            output:
                prior_factor: a scalar density factor for the prior draw"""
        return 0.

    @abstractmethod
    def correct_bounds(self, params_in):
        """correct the bounds for the input parameters to be within the prior range, if possible:
            input:
                params_in: the point with possibly incorrect parameters
            output:
                params_out: the point with corrected parameters"""
        return np.zeros(params_in.size)

    @abstractmethod
    def check_bounds(self, params_in):
        """check if the specified point is within the prior volume
            input:
                params_in: the point to be checkout
            output:
                valid: a scalar boolean which is True is the point is valid in the prior volume and false otherwise"""
        return True
