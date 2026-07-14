"""the eggbox likelihood in n dimensions"""

# see MNRAS 455, 1919-1937 (2016) doi:10.1093/mnras/stv2422 for 5D extension
import numba as nb
import numpy as np
from numba import njit
from numba.experimental import jitclass  # pyright: ignore[reportPrivateImportUsage]
from numpy.typing import NDArray

from DTMCMC.correction_helpers import reflect_into_range

tmax = 5.0 * np.pi

Tp = 2.0e-1
betap = 1.0 / Tp

low_lim = -(5 * np.pi / 2.0)
high_lim = 5 * np.pi / 2.0

n_pi_range = np.int64((high_lim - low_lim) / np.pi)


@njit()
def get_loglike(x: NDArray[np.floating], n_par: int) -> float:
    """Get the eggbox likelihood in n dimensions"""
    prod = 1.0
    for itrp in range(n_par):
        prod *= np.cos(x[itrp])
    return (prod + 1.0) ** betap


@njit()
def prior_draw(n_par: int) -> NDArray[np.floating]:
    """Get a prior draw"""
    return np.random.uniform(low_lim, high_lim, n_par)


@njit()
def prior_factor(_v: NDArray[np.floating], _n_par: int) -> float:
    """Get the denstiy factor for prior draws

    numba cannot type `del` of function arguments, so the unused inputs
    are marked by naming: the previous del-based body made every call to
    prior_factor raise a TypingError, which left the eggbox jitclass
    unable to run through the default proposal mixture at all
    """
    return 0.0


@njit()
def correct_bounds(v: NDArray[np.floating], n_par: int) -> NDArray[np.floating]:
    """Correct parameters to be in boundaries"""
    for itrp in range(n_par):
        v[itrp] = reflect_into_range(v[itrp], low_lim, high_lim)
    return v


@njit()
def check_bounds(v: NDArray[np.floating]) -> bool:
    """Check if a sample is within the prior range"""
    for itrp in range(v.size):
        if not low_lim < v[itrp] < high_lim:
            return False
    return True


@jitclass([('n_par', nb.int64), ('epsilons', nb.float64[:])])  # pyright: ignore[reportCallIssue]
class Likelihood:
    """class to manage the likelihood-specific essential functions for the sampler"""

    def __init__(self, n_par: int = 5, eps_default: float = 1.0e-3) -> None:
        """Create the class and store any object specific variables"""
        self.n_par = n_par
        self.epsilons = np.zeros(n_par) + eps_default

    def get_loglike(self, v: NDArray[np.floating]) -> float:
        """Get the log likelihood given a set of parameters v"""
        return get_loglike(v, self.n_par)

    def prior_draw(self) -> NDArray[np.floating]:
        """Get a draw from the prior"""
        return prior_draw(self.n_par)

    def prior_proposal(self, v_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], float, bool]:
        """Get a proposal from the prior"""
        v_out = prior_draw(self.n_par)
        return v_out, prior_factor(v_in, self.n_par) - prior_factor(v_out, self.n_par), True

    def prior_factor(self, v: NDArray[np.floating]) -> float:
        """Get the density factor for prior draws, if the prior draws are not uniform"""
        return prior_factor(v, self.n_par)

    def correct_bounds(self, v: NDArray[np.floating]) -> NDArray[np.floating]:
        """Correct the bounds of a draw to be in range, if allowed for this likelihood"""
        return correct_bounds(v, self.n_par)

    def check_bounds(self, v: NDArray[np.floating]) -> bool:
        """Check if the bounds of a draw are in the prior range but do not change them"""
        return check_bounds(v)


def get_labels(n_par: int) -> list[str]:
    """Get useful labels for corner plots"""
    return [r'$v_' + str(itrp) + '$' for itrp in range(n_par)]


def format_samples_output(
    samples: NDArray[np.floating], params_fid: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating], list[str]]:
    labels_loc = get_labels(params_fid.size)
    return samples.copy(), params_fid.copy(), labels_loc


@njit()
def mode_matcher(
    n_cold: int,
    samples_store: NDArray[np.floating],
    itrn: int,
    mode_first_idx: NDArray[np.int64],
    mode_last_idx: NDArray[np.int64],
    n_par: int,
    modes_canonical: NDArray[np.int64],
) -> NDArray[np.bool_]:
    """Guess which mode each sample is in"""
    mode_all = np.full(samples_store.shape[0], True, dtype=np.bool_)
    for itrl in range(samples_store.shape[0]):
        for itrt in range(n_cold):
            point_int_res = np.zeros(n_par, dtype=np.int64)
            failed = False
            for itrp in range(samples_store.shape[2]):
                point_loc_div = samples_store[itrl, itrt, itrp] / np.pi
                point_loc_mod = point_loc_div % 1
                if point_loc_mod < 0.2:
                    point_int_res[itrp] = np.int64(np.floor(point_loc_div)) + 2
                elif point_loc_mod > 0.8:
                    point_int_res[itrp] = np.int64(np.ceil(point_loc_div)) + 2
                else:
                    failed = True
                    break

            if failed:
                mode_all[itrl] = False
                continue

            canonical_build = 0
            for itrp in range(n_par):
                canonical_build += point_int_res[itrp] * n_pi_range**itrp

            itrm = modes_canonical[canonical_build]

            if itrm >= -1:
                if mode_first_idx[itrm, itrt] == -1:
                    mode_first_idx[itrm, itrt] = itrn + itrl
                mode_last_idx[itrm, itrt] = itrn + itrl
            else:
                mode_all[itrl] = False
    return mode_all


def gen_nd_modelist(n_par: int = 5) -> tuple[NDArray[np.floating], NDArray[np.int64], NDArray[np.int64]]:
    """Get the full list of modes for nd eggbox"""
    mode_loc: list[NDArray[np.floating]] = []
    mode_int_loc: list[NDArray[np.int64]] = []
    idx_max_float: float = (high_lim - low_lim) / (np.pi / 2)
    if idx_max_float % 1 > 0.999:
        idx_max: np.int64 = np.int64(idx_max_float) + 2
    else:
        idx_max = np.int64(idx_max_float) + 1
    targ_like = get_loglike(np.zeros(n_par), n_par)
    modes_canonical_got = np.zeros(n_pi_range**n_par, dtype=np.int64) - 1
    mode_idxs = np.zeros(n_par, dtype=np.int64)
    for _itrm in range(idx_max**n_par):
        pos_loc = np.zeros(n_par)
        for itrp in range(n_par):
            pos_loc[itrp] = low_lim + mode_idxs[itrp] * np.pi / 2
        res = get_loglike(pos_loc, n_par)
        # check mode is almost as likely as the mode at (0,0)
        if res > 0.9 * targ_like:
            mode_loc.append(pos_loc.copy())
            for itrp in range(n_par):
                assert mode_idxs[itrp] % 2 == 1
            mode_canonical = 0
            for itrp in range(n_par):
                mode_canonical += ((mode_idxs[itrp] + np.int64(low_lim / (np.pi / 2))) // 2 + 2) * n_pi_range**itrp
            modes_canonical_got[mode_canonical] = len(mode_int_loc)
            mode_int_loc.append((mode_idxs + np.int64(low_lim / (np.pi / 2))) // 2)

        # scan through potential modes
        for itrp in range(n_par - 1, -1, -1):
            mode_idxs[itrp] += 1
            if mode_idxs[itrp] < idx_max:
                break
            mode_idxs[itrp] = 0

    return np.array(mode_loc), np.array(mode_int_loc), modes_canonical_got
