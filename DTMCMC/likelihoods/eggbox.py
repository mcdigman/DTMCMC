"""the eggbox likelihood in n dimensions"""

# see MNRAS 455, 1919-1937 (2016) doi:10.1093/mnras/stv2422 for 5D extension
import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.correction_helpers import reflect_into_range
from DTMCMC.likelihood import RectangularLikelihood, RectangularNativeState
from DTMCMC.numba_backend import NativeLoglikeCall

tmax: float = 5.0 * np.pi

Tp: float = 2.0e-1
betap: float = 1.0 / Tp

low_lim: float = -(5 * np.pi / 2.0)
high_lim: float = 5 * np.pi / 2.0

n_pi_range: np.int64 = np.int64((high_lim - low_lim) / np.pi)


@njit()
def get_loglike(x: NDArray[np.floating], n_par: int) -> float:
    """Get the eggbox likelihood in n dimensions"""
    prod: float = 1.0
    for itrp in range(n_par):
        prod *= np.cos(x[itrp])
    # note prod can never be <-1.0, so res will be a float
    res: float = (prod + 1.0) ** betap
    return res


@njit(inline='always')
def _loglike_native(params_in: NDArray[np.floating], state: RectangularNativeState) -> float:
    """Per-class native log likelihood reading n_par from the state bundle."""
    return get_loglike(params_in, state.n_par)


@njit()
def prior_draw(n_par: int) -> NDArray[np.floating]:
    """Get a prior draw"""
    return np.random.uniform(low_lim, high_lim, n_par)


@njit()
def prior_factor(_v: NDArray[np.floating], _n_par: int) -> float:
    """Get the denstiy factor for prior draws."""
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


@njit()
def validate_bounds(params_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], bool]:
    success: bool = check_bounds(params_in)
    if not success:
        # try to make the point in bounds and fail if unsuccesful
        new_point = correct_bounds(params_in, params_in.size)
        success = check_bounds(params_in)
    else:
        new_point = params_in
    return new_point, success


# @jitclass([('n_par', nb.int64), ('epsilons', nb.float64[:])])  # type: ignore[no-untyped-call] # pyright: ignore[reportCallIssue]
class EggboxLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""

    def __init__(self, n_par: int = 5, eps_default: float = 1.0e-3) -> None:
        """Create the class and store any object specific variables"""
        super().__init__(n_par, np.full(n_par, low_lim), np.full(n_par, high_lim))
        self.epsilons = np.zeros(n_par) + eps_default

    def get_loglike(self, v: NDArray[np.floating]) -> float:
        """Get the log likelihood given a set of parameters v"""
        return get_loglike(v, self.n_par)

    def bind_native_loglike(self) -> NativeLoglikeCall[RectangularNativeState]:
        """Return the per-class native log likelihood."""
        return _loglike_native

    def get_epsilons(self) -> NDArray[np.floating]:
        """Get epsilons for fisher matrix calculation."""
        return self.epsilons


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
