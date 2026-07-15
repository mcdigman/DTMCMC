"""C 2023 Matthew C. Digman
abstract class to hold a likelihood object
"""

from typing import Protocol

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.correction_helpers import reflect_into_range


class AbstractLikelihood(Protocol):
    """Structural interface the engine requires of a likelihood object."""

    n_par: int
    n_evals: int

    def __init__(self, n_par: int) -> None:
        """Initialize the likelihood.
        input: n_par integer, how many dimensions in the parameter space
        """
        ...

    def get_loglike(self, params_in: NDArray[np.floating], /) -> float:
        """Get the log likelihood at the specified parameters.
        input:
            params_in: a 1D float array of parameters
        output:
            logL: a scalar float likelihood
        """
        ...

    def prior_draw(self) -> NDArray[np.floating]:
        """Get a draw from the priors for this likelihood.
        output:
            params: a 1D float array of parameters
        """
        ...

    def prior_factor(self, params_in: NDArray[np.floating], /) -> float:
        """Get the untempered log prior density for the input parameters.
        input:
            params_in: the parameters to consider
        output:
            prior_factor: log prior density up to an additive constant
        """
        ...

    def correct_bounds(self, params_in: NDArray[np.floating], /) -> NDArray[np.floating]:
        """Correct the bounds for the input parameters to be within the prior range.
        input:
            params_in: the point with possibly incorrect parameters
        output:
            params_out: the point with corrected parameters
        """
        ...

    def check_bounds(self, params_in: NDArray[np.floating], /) -> bool:
        """Check if the specified point is within the prior volume
        input:
            params_in: the point to be checkout
        output:
            valid: a scalar boolean which is True is the point is valid in the prior volume and false otherwise
        """
        ...

    def validate_bounds(self, params_in: NDArray[np.floating], /) -> tuple[NDArray[np.floating], bool]:
        """Check if the specified point is within the prior volume, try to correct if not."""
        ...

    def get_epsilons(self, /) -> NDArray[np.floating]:
        """Get epsilons by dimension for fisher matrix computation."""
        ...

    def get_labels(self) -> list[str]:
        """Get formatted axis labels for corner plots"""
        ...

    def format_samples_output(
        self, samples_store: NDArray[np.floating], params_fid: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Purely a convenience function for making corner plots:
        if we desire to do any adjustments to input samples to make corner plots
        look nice, for example converting some dimension the raw parameter
        to Delta that parameter, or changing the units, we can do that here
        """
        ...


@njit()
def correct_bounds_rectangular(
    v: NDArray[np.floating], low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Wrap parameters into range"""
    for itrp in range(v.size):
        v[itrp] = reflect_into_range(v[itrp], low_lims[itrp], high_lims[itrp])
    return v


@njit()
def prior_draw_rectangular(
    n_par: int, low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Get a uniform prior draw with rectangular walls"""
    draw = np.zeros(n_par)
    for itrp in range(n_par):
        draw[itrp] = np.random.uniform(low_lims[itrp], high_lims[itrp])

    return draw


@njit()
def check_bounds_rectangular(
    v: NDArray[np.floating], low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]
) -> bool:
    """Check if a sample is within the prior range"""
    for itrp in range(v.size):
        if not low_lims[itrp] <= v[itrp] <= high_lims[itrp]:
            return False
    return True


@njit()
def validate_bounds_rectangular(
    params_in: NDArray[np.floating], low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]
) -> tuple[NDArray[np.floating], bool]:
    success: bool = check_bounds_rectangular(params_in, low_lims, high_lims)
    if not success:
        # try to make the point in bounds and fail if unsuccesful
        new_point = correct_bounds_rectangular(params_in, low_lims, high_lims)
        success = check_bounds_rectangular(params_in, low_lims, high_lims)
    else:
        new_point = params_in
    return new_point, success


class RectangularLikelihood(AbstractLikelihood):
    """Handle a likelihood with rectangular bounds
    by default assume a uniform prior
    """

    def __init__(self, n_par: int, low_lims: NDArray[np.floating], high_lims: NDArray[np.floating]) -> None:
        self.low_lims: NDArray[np.floating] = low_lims
        self.high_lims: NDArray[np.floating] = high_lims

        assert self.low_lims.size == n_par
        assert self.high_lims.size == n_par

        self.n_par = n_par
        self.n_evals = 0

    def correct_bounds(self, params_in: NDArray[np.floating]) -> NDArray[np.floating]:
        """Correct bounds for rectangular walls"""
        return correct_bounds_rectangular(params_in, self.low_lims, self.high_lims)

    def check_bounds(self, params_in: NDArray[np.floating]) -> bool:
        """Check bounds for rectangular walls"""
        return check_bounds_rectangular(params_in, self.low_lims, self.high_lims)

    def prior_factor(self, params_in: NDArray[np.floating]) -> float:
        """Get the density factor for prior draws assuming a uniform prior"""
        del params_in
        return 0.0

    def prior_draw(self) -> NDArray[np.floating]:
        """Get a draw from the prior"""
        return prior_draw_rectangular(self.n_par, self.low_lims, self.high_lims)

    def validate_bounds(self, params_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], bool]:
        """Check the parameters and correct if required."""
        return validate_bounds_rectangular(params_in, self.low_lims, self.high_lims)

    def get_epsilons(self) -> NDArray[np.floating]:
        """Special helper for FisherJumpManager
        if this likelihood has special epsilons specified for fisher matrix jumps, get them here,
        otherwise just return zeros
        """
        return np.zeros(self.n_par)

    def get_labels(self) -> list[str]:
        """Get formatted axis labels for corner plots"""
        return [r'$v_' + str(itrp) + '$' for itrp in range(self.n_par)]

    def format_samples_output(
        self, samples_store: NDArray[np.floating], params_fid: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Purely a convenience function for making corner plots:
        if we desire to do any adjustments to input samples to make corner plots
        look nice, for example converting some dimension the raw parameter
        to Delta that parameter, or changing the units, we can do that here
        """
        return samples_store.copy(), params_fid.copy()
