"""C 2023 Matthew C. Digman
abstract class to hold a likelihood object
"""

from abc import ABC, abstractmethod

import numpy as np
from numba import njit

from DTMCMC.correction_helpers import reflect_into_range


class AbstractLikelihood(ABC):
    """abstract likelihood object"""

    def __init__(self, n_par):
        """Initialize the likelihood
            input: n_par integer, how many dimensions in the parameter space
        """
        self.n_par = n_par

    @abstractmethod
    def get_loglike(self, params_in):
        """Get the log likelihood at the specified parameters:
            input:
                params_in: a 1D float array of parameters
            output:
                logL: a scalar float likelihood
        """
        del params_in
        return 0.

    @abstractmethod
    def prior_draw(self):
        """Get a draw from the priors for this likelihood
            output:
                params: a 1D float array of parameters
        """
        return np.zeros(1)

    @abstractmethod
    def prior_factor(self, params_in):
        """Get the prior density factor for the input parameters
            input:
                params_in: the parameters to consider
            output:
                prior_factor: a scalar density factor for the prior draw
        """
        del params_in
        return 0.

    @abstractmethod
    def correct_bounds(self, params_in):
        """Correct the bounds for the input parameters to be within the prior range, if possible:
            input:
                params_in: the point with possibly incorrect parameters
            output:
                params_out: the point with corrected parameters
        """
        return np.zeros(params_in.size)

    @abstractmethod
    def check_bounds(self, params_in):
        """Check if the specified point is within the prior volume
            input:
                params_in: the point to be checkout
            output:
                valid: a scalar boolean which is True is the point is valid in the prior volume and false otherwise
        """
        del params_in
        return True

    def get_epsilons(self):
        """Special helper for FisherJumpManager
        if this likelihood has special epsilons specified for fisher matrix jumps, get them here,
        otherwise just return zeros
        """
        return np.zeros(self.n_par)

    def get_labels(self):
        """Get formatted axis labels for corner plots"""
        return [r'$v_' + str(itrp) + '$' for itrp in range(self.n_par)]

    def format_samples_output(self, samples_store, params_fid):
        """Purely a convenience function for making corner plots:
        if we desire to do any adjustments to input samples to make corner plots
        look nice, for example converting some dimension the raw parameter
        to Delta that parameter, or changing the units, we can do that here
        """
        return samples_store.copy(), params_fid.copy()


@njit()
def correct_bounds_rectangular(v, low_lims, high_lims):
    """Wrap parameters into range"""
    for itrp in range(v.size):
        v[itrp] = reflect_into_range(v[itrp], low_lims[itrp], high_lims[itrp])
    return v


@njit()
def prior_draw_rectangular(n_par, low_lims, high_lims):
    """Get a uniform prior draw with rectangular walls"""
    draw = np.zeros(n_par)
    for itrp in range(n_par):
        draw[itrp] = np.random.uniform(low_lims[itrp], high_lims[itrp])

    return draw


# TODO handle trivial bounds correctly
@njit()
def check_bounds_rectangular(v, low_lims, high_lims):
    """Check if a sample is within the prior range"""
    for itrp in range(v.size):
        if not low_lims[itrp] <= v[itrp] <= high_lims[itrp]:
            return False
    return True


class RectangularLikelihood(AbstractLikelihood):
    """Handle a likelihood with rectangular bounds
    by default assume a uniform prior
    """

    def __init__(self, n_par, low_lims, high_lims):
        self.low_lims = low_lims
        self.high_lims = high_lims

        assert self.low_lims.size == n_par
        assert self.high_lims.size == n_par

        AbstractLikelihood.__init__(self, n_par)

    def correct_bounds(self, params_in):
        """Correct bounds for rectangular walls"""
        return correct_bounds_rectangular(params_in, self.low_lims, self.high_lims)

    def check_bounds(self, params_in):
        """Check bounds for rectangular walls"""
        return check_bounds_rectangular(params_in, self.low_lims, self.high_lims)

    def prior_factor(self, v):
        """Get the density factor for prior draws assuming a uniform prior"""
        del v
        return 0.

    def prior_draw(self):
        """Get a draw from the prior"""
        return prior_draw_rectangular(self.n_par, self.low_lims, self.high_lims)
