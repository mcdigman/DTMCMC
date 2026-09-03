"""Numba-compatible helpers shared by PTMCMC transition backends."""

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit()
def mcmc_decision_helper(
    itrb: int,
    samples: NDArray[np.floating],
    logLs: NDArray[np.floating],
    betas: NDArray[np.floating],
    accept_record: NDArray[np.int64],
    esd_record: NDArray[np.floating],
    itrt: int,
    new_point: NDArray[np.floating],
    logL_new: float,
    density_fac: float,
    idx_jump: int,
) -> None:
    """Decide whether an MCMC proposal is accepted and update observers."""
    # draw to determine if we will accept
    test: float = np.log(np.random.uniform(0.0, 1.0))

    # squared displacement of the proposal, accumulated per (T, jump type)
    # for the expected-squared-displacement tracker; pure observer, no draws
    delta_sq: float = 0.0
    for itrp in range(new_point.size):
        diff: float = new_point[itrp] - samples[itrb - 1, itrt, itrp]
        delta_sq += diff * diff
    esd_record[0, itrt, idx_jump] += delta_sq

    # process acceptance or rejection
    if betas[itrt] * (logL_new - logLs[itrb - 1, itrt]) + density_fac > test:
        # the draw was accepted, assign its parameters
        samples[itrb, itrt] = new_point
        logLs[itrb, itrt] = logL_new
        accept_record[0, itrt, idx_jump] += 1
        esd_record[1, itrt, idx_jump] += delta_sq
    else:
        # the draw was rejected, assign the old parameters
        samples[itrb, itrt] = samples[itrb - 1, itrt]
        logLs[itrb, itrt] = logLs[itrb - 1, itrt]
        accept_record[1, itrt, idx_jump] += 1
