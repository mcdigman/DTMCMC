"""C 2023 Matthew C. Digman
module to store objects related to fisher matrix jumps
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.jump_manager import AbstractJump, JumpManager
from DTMCMC.lapack_wrappers import solve_triangular

if TYPE_CHECKING:
    from configparser import ConfigParser

    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.numba_backend import NativeJumpCall, NativeLikelihoodFunctions
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder
    from DTMCMC.tracker_manager import LikelihoodEvalTracker


@njit()
def sigma_subspace_jump_helper(
    sample_point: NDArray[np.floating],
    itrt: int,
    n_par: int,
    fisher_subspace_frac: float,
    sigma_scales: NDArray[np.floating],
    do_full: int,
) -> tuple[NDArray[np.floating], float, bool]:
    """Helper to compute a standard deviation jump in random subspaces"""
    mult = np.random.normal(0.0, 1.0, n_par)
    count = n_par
    # average fraction of dimensions to use in subspace
    subspace_frac = fisher_subspace_frac

    if not do_full:
        # ensure at least one random direction is protected so we aren't making null proposals
        safe_itrp = np.random.randint(n_par)
        for itrp in range(n_par):
            if np.random.uniform(0.0, 1.0) > subspace_frac and itrp != safe_itrp:
                mult[itrp] = 0.0
                count -= 1

    assert count > 0

    new_point = sample_point + sigma_scales[itrt] * np.sqrt(n_par / count) * mult
    return new_point, 0.0, True


FisherNativeState = tuple[
    'NDArray[np.floating]', 'NDArray[np.floating]', 'NDArray[np.floating]'
]  # (sigma_scales, chol_fishers, gamma_mults)


def _bind_sigma_native(manager: FisherJumpManager, do_full: bool) -> NativeJumpCall:
    """Bind a standard-deviation jump reading the manager's runtime state.

    The scale arrays are refreshed between blocks, so they arrive through
    the manager's runtime native state; only the immutable configuration
    (n_par, subspace fraction) is baked.
    """
    n_par = manager.n_par
    fisher_subspace_frac = manager.strategy_params.fisher_subspace_frac

    @njit(inline='always')
    def native_call(
        sample_point: NDArray[np.floating], itrt: int, state: FisherNativeState
    ) -> tuple[NDArray[np.floating], float, bool]:
        sigma_scales = state[0]
        return sigma_subspace_jump_helper(sample_point, itrt, n_par, fisher_subspace_frac, sigma_scales, do_full)

    return native_call


@njit(inline='always')
def fisher_full_jump_helper(
    sample_point: NDArray[np.floating],
    itrt: int,
    chol_fishers: NDArray[np.floating],
    gamma_mults: NDArray[np.floating],
) -> tuple[NDArray[np.floating], float, bool]:
    """Apply a full Cholesky Fisher jump."""
    n_par = sample_point.size
    new_point = sample_point + solve_triangular(
        chol_fishers[itrt],
        gamma_mults[itrt] * np.random.normal(0.0, 1.0, n_par),
        trans_a=True,
    )
    return new_point, 0.0, True


class SigmaFullJump(AbstractJump):
    """Standard Deviation Jump in Full Dimensions"""

    def __init__(self, manager: FisherJumpManager) -> None:
        """Create the jump"""
        self.manager: FisherJumpManager = manager
        self.print_name = 'Std All-D'
        # n_par = self.manager.n_par
        # mult = np.random.normal(0., 1., n_par)
        # new_point = sample_point+self.manager.sigma_scales[itrt]*mult
        # return new_point, 0., True

    def bind_native(self, likelihood_natives: NativeLikelihoodFunctions) -> NativeJumpCall:
        """Bind the jump as a jitted closure reading the manager runtime state."""
        del likelihood_natives
        return _bind_sigma_native(self.manager, True)

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        """Apply a standard deviation jump"""
        return sigma_subspace_jump_helper(
            sample_point,
            itrt,
            self.manager.n_par,
            self.manager.strategy_params.fisher_subspace_frac,
            self.manager.sigma_scales,
            True,
        )


class SigmaRandomSubspaceJump(AbstractJump):
    """Standard deviation jump in random subspaces"""

    def __init__(self, manager: FisherJumpManager) -> None:
        self.manager: FisherJumpManager = manager
        self.print_name = 'Std Random-D'

    def bind_native(self, likelihood_natives: NativeLikelihoodFunctions) -> NativeJumpCall:
        """Bind the jump as a jitted closure reading the manager runtime state."""
        del likelihood_natives
        return _bind_sigma_native(self.manager, False)

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        """Apply a standard deviation jump in random subspaces"""
        return sigma_subspace_jump_helper(
            sample_point,
            itrt,
            self.manager.n_par,
            self.manager.strategy_params.fisher_subspace_frac,
            self.manager.sigma_scales,
            False,
        )


@dataclass(init=False)
class FisherStrategyParameters:
    """container to store some parameters related to the strategy of
    fisher matrix proposal generation
    """

    use_chol_fishers: bool
    cold_fisher_weight: float
    hot_fisher_weight: float
    fisher_subspace_frac: float
    fisher_full_d_frac: float
    fisher_downsample: int
    sigma_default: float
    max_fisher_el: float
    eps_default: float
    verbose_fisher: bool

    def __init__(self, config: ConfigParser) -> None:
        """Initialize the object with the prescribed parameters"""
        config_f = config['FisherJumpManager']

        # whether to do fisher jumps using the cholesky decomposition
        self.use_chol_fishers = config_f.getboolean('use_chol_fishers', False)
        # how often to do fisher draws in the cold chains
        self.cold_fisher_weight = config_f.getfloat('cold_fisher_weight', 0.333)
        # how often to do fisher draws in the hottest finite temperature chain
        self.hot_fisher_weight = config_f.getfloat('hot_fisher_weight', 0.333)
        # what fraction of dimensions to include in fisher subspace jumps
        self.fisher_subspace_frac = config_f.getfloat('fisher_subspace_frac', 1.0)
        # how often to not do subspace jumps when doing a fisher jump
        self.fisher_full_d_frac = config_f.getfloat('fisher_full_d_frac', 1.0)
        # how many blocks to skip between fisher matrix updates
        self.fisher_downsample = config_f.getint('fisher_downsample', 1)
        # default sigma for fisher matrix jumps
        self.sigma_default = config_f.getfloat('sigma_default', 100.0)
        # maximum element of fisher matrix
        self.max_fisher_el = config_f.getfloat('max_fisher_el', np.inf)
        # default epsilon of fisher matrix
        self.eps_default = config_f.getfloat('eps_default', 1.0e-4)
        # whether to print a notification every time the fisher matrix is update
        self.verbose_fisher = config_f.getboolean('verbose_fisher', True)

    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to the requested configuration object
        inputs:
            config_in: ConfigParser object
        """
        config_f = config_in['FisherJumpManager']
        config_f['use_chol_fishers'] = str(self.use_chol_fishers)
        config_f['cold_fisher_weight'] = str(self.cold_fisher_weight)
        config_f['hot_fisher_weight'] = str(self.hot_fisher_weight)
        config_f['fisher_subspace_frac'] = str(self.fisher_subspace_frac)
        config_f['fisher_full_d_frac'] = str(self.fisher_full_d_frac)
        config_f['fisher_downsample'] = str(self.fisher_downsample)
        config_f['sigma_default'] = str(self.sigma_default)
        config_f['max_fisher_el'] = str(self.max_fisher_el)
        config_f['eps_default'] = str(self.eps_default)
        config_f['verbose_Fisher'] = str(self.verbose_fisher)


def set_fishers(
    sample_set: NDArray[np.floating],
    strategy_params: FisherStrategyParameters,
    n_chain: int,
    like_obj: AbstractLikelihood,
    eval_tracker: LikelihoodEvalTracker | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Set up the fisher matrices"""
    n_evals = 0
    use_chol_fishers = strategy_params.use_chol_fishers
    sigma_default = strategy_params.sigma_default
    max_fisher_el = strategy_params.max_fisher_el

    n_par = sample_set.shape[1]
    sigma_diags = np.zeros((n_chain, n_par))
    if use_chol_fishers:
        fishers = np.zeros((n_chain, n_par, n_par))
        chol_fishers = np.zeros((n_chain, n_par, n_par))
    else:
        fishers = np.zeros((0, 0, 0))
        chol_fishers = np.zeros((0, 0, 0))

    epsilons = like_obj.get_epsilons()
    for itrp, eps in enumerate(epsilons):
        if eps == 0.0:
            epsilons[itrp] = strategy_params.eps_default

    for itrt in range(n_chain):
        new_point = sample_set[itrt]
        new_point_alt = like_obj.correct_bounds(new_point.copy())
        assert np.all(new_point == new_point_alt)
        nn = like_obj.get_loglike(new_point)
        n_evals += 1
        for itrp in range(n_par):
            eps = epsilons[itrp]
            pointp = new_point.copy()
            pointp[itrp] += 2 * eps
            pointmp = like_obj.correct_bounds(pointp)
            pp = like_obj.get_loglike(pointp)

            pointm = new_point.copy()
            pointm[itrp] -= 2 * eps
            pointm = like_obj.correct_bounds(pointm)
            mm = like_obj.get_loglike(pointm)
            n_evals += 2

            fisher_loc = -(pp - 2.0 * nn + mm) / (4 * eps * eps) + 1.0 / sigma_default**2

            if use_chol_fishers:
                fishers[itrt, itrp, itrp] = fisher_loc
            if not np.isfinite(fisher_loc) or fisher_loc <= 0.0 or fisher_loc > max_fisher_el:
                if use_chol_fishers:
                    fishers[itrt, itrp, itrp] = 1.0 / sigma_default**2
                sigma_diags[itrt, itrp] = sigma_default
            else:
                sigma_diags[itrt, itrp] = 1.0 / np.sqrt(fisher_loc)

        if use_chol_fishers:
            for itrp1 in range(n_par):
                eps1 = epsilons[itrp1]
                for itrp2 in range(itrp1 + 1, n_par):
                    eps2 = epsilons[itrp2]
                    pointpp = new_point.copy()
                    pointpp[itrp1] += eps1
                    pointpp[itrp2] += eps2
                    pointpp = like_obj.correct_bounds(pointpp)
                    pp = like_obj.get_loglike(pointpp)

                    pointpm = new_point.copy()
                    pointpm[itrp1] += eps1
                    pointpm[itrp2] -= eps2
                    pointpm = like_obj.correct_bounds(pointpm)
                    pm = like_obj.get_loglike(pointpm)

                    pointmp = new_point.copy()
                    pointmp[itrp1] -= eps1
                    pointmp[itrp2] += eps2
                    pointmp = like_obj.correct_bounds(pointmp)
                    mp = like_obj.get_loglike(pointmp)

                    pointmm = new_point.copy()
                    pointmm[itrp1] -= eps1
                    pointmm[itrp2] -= eps2
                    pointmm = like_obj.correct_bounds(pointmm)
                    mm = like_obj.get_loglike(pointmm)
                    n_evals += 4

                    res = -(pp - mp - pm + mm) / (4.0 * eps1 * eps2)
                    if not np.isfinite(res) or np.abs(res) > max_fisher_el:
                        res = 0.0

                    fishers[itrt, itrp1, itrp2] = res
                    fishers[itrt, itrp2, itrp1] = fishers[itrt, itrp1, itrp2]

            det_fisher = np.linalg.det(fishers[itrt])
            if not np.isfinite(det_fisher) or det_fisher <= 0.0 or np.any(np.linalg.eigh(fishers[itrt])[0] <= 0.0):
                for itrp1 in range(n_par):
                    for itrp2 in range(itrp1 + 1, n_par):
                        fishers[itrt, itrp1, itrp2] = 0.0
                        fishers[itrt, itrp2, itrp1] = 0.0

            chol_fishers[itrt] = np.linalg.cholesky(fishers[itrt])

    if eval_tracker is not None:
        eval_tracker.count(n_evals)
    return sigma_diags, fishers, chol_fishers


def set_scales(
    n_par: int, T_ladder: TemperatureLadder, sigma_diags: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Helper to get several scaling parameters for fisher matrix jumps"""
    n_chain = T_ladder.n_chain
    betas = T_ladder.betas

    sigma_scales = np.zeros((n_chain, n_par))
    gamma_mults = np.zeros(n_chain)

    if np.any((betas > 0.0) & (np.isfinite(betas))):
        # use the smallest postive beta if beta is 0 or non-finite for scaling
        small_default = np.min(betas[betas > 0.0])
    else:
        small_default = 1.0

    # set the scale factors for the sigma and fisher jumps as a function of temperature
    for itrj in range(n_chain):
        if np.isfinite(betas[itrj]):
            beta_loc = max(betas[itrj], small_default)
        else:
            beta_loc = small_default

        gamma_mults[itrj] = 2.38 / np.sqrt(beta_loc)
        for itrp in range(n_par):
            sigma_scales[itrj, itrp] = 2.38 * (sigma_diags[itrj, itrp] / np.sqrt(beta_loc) / np.sqrt(n_par))
    return sigma_scales, gamma_mults


class FisherJumpManager(JumpManager):
    """manage everything related to fisher matrix jumps, subclass of DTMCMC.jump_manager.JumpManager

    The fisher arrays are allocated once and refreshed in place: native jump
    bindings bake them into compiled closures by reference, so their identity
    must be stable for the lifetime of the sampler.
    """

    def __init__(
        self,
        T_ladder: TemperatureLadder,
        like_obj: AbstractLikelihood,
        sample_set: NDArray[np.floating],
        config: ConfigParser,
        eval_tracker: LikelihoodEvalTracker | None = None,
    ) -> None:
        """Create the object"""
        self.strategy_params = FisherStrategyParameters(config)
        self.eval_tracker = eval_tracker

        jumps: list[AbstractJump] = [FisherFullJump(self), SigmaFullJump(self), SigmaRandomSubspaceJump(self)]

        JumpManager.__init__(self, T_ladder, like_obj, jumps)

        self.sample_set = sample_set
        self.sigma_diags: NDArray[np.floating] = np.zeros((self.n_chain, self.n_par))
        if self.strategy_params.use_chol_fishers:
            self.fishers: NDArray[np.floating] = np.zeros((self.n_chain, self.n_par, self.n_par))
            self.chol_fishers: NDArray[np.floating] = np.zeros((self.n_chain, self.n_par, self.n_par))
        else:
            self.fishers = np.zeros((0, 0, 0))
            self.chol_fishers = np.zeros((0, 0, 0))
        self.sigma_scales: NDArray[np.floating] = np.zeros((self.n_chain, self.n_par))
        self.gamma_mults: NDArray[np.floating] = np.zeros(self.n_chain)
        self.reset_fishers_from_point(self.sample_set)

    def bind_native_state(self) -> FisherNativeState:
        """Return the identity-stable scale arrays read by this manager's jumps."""
        return (self.sigma_scales, self.chol_fishers, self.gamma_mults)

    def set_jump_weights(self) -> None:
        """Set the relative probabilities of the different jump types"""
        n_cold = self.T_ladder.n_cold
        n_chain = self.T_ladder.n_chain
        jump_weights = np.zeros((n_chain, self.n_jump_types))
        # just a default equal weight
        jump_weights[:] = 0.333

        cold_weight = self.strategy_params.cold_fisher_weight
        hot_weight = self.strategy_params.hot_fisher_weight

        subspace_weight = 1.0 - self.strategy_params.fisher_full_d_frac
        full_weight = self.strategy_params.fisher_full_d_frac

        # get the indices of the jump types we need to assign probabilities for
        sigma_full_idx = -1
        sigma_random_idx = -1
        fisher_full_idx = -1
        for itrp, jump in enumerate(self.jumps):
            if isinstance(jump, SigmaFullJump):
                sigma_full_idx = itrp
            if isinstance(jump, SigmaRandomSubspaceJump):
                sigma_random_idx = itrp
            if isinstance(jump, FisherFullJump):
                fisher_full_idx = itrp

        assert sigma_full_idx >= 0
        assert sigma_random_idx >= 0
        assert fisher_full_idx >= 0

        if self.strategy_params.use_chol_fishers:
            jump_weights[:n_cold, sigma_full_idx] = 0.0
            jump_weights[:n_cold, sigma_random_idx] = 0.0
            jump_weights[:n_cold, fisher_full_idx] = cold_weight
        else:
            jump_weights[:n_cold, sigma_full_idx] = cold_weight * full_weight
            jump_weights[:n_cold, sigma_random_idx] = cold_weight * subspace_weight
            jump_weights[:n_cold, fisher_full_idx] = 0.0

        if self.strategy_params.use_chol_fishers:
            jump_weights[n_cold:, fisher_full_idx] = hot_weight
            jump_weights[n_cold:, sigma_full_idx] = 0.0
            jump_weights[n_cold:, sigma_random_idx] = 0.0
        else:
            jump_weights[n_cold:, fisher_full_idx] = 0.0
            jump_weights[n_cold:, sigma_full_idx] = hot_weight * full_weight
            jump_weights[n_cold:, sigma_random_idx] = hot_weight * subspace_weight

        self.jump_weights = jump_weights
        assert np.all(self.jump_weights >= 0.0)

    def post_block_update(
        self, itrn: int, block_size: int, samples: NDArray[np.floating], logLs: NDArray[np.floating]
    ) -> None:
        """Do any needed internal processing after an individual block of size block_size:
        ie, fisher matrix updates
        """
        return self.reset_fishers(itrn, block_size, samples, logLs)

    def reset_fishers_from_point(self, sample_set: NDArray[np.floating]) -> None:
        """Set the fisher matrix object at the specified point, updating in place"""
        sigma_diags, fishers, chol_fishers = set_fishers(
            sample_set, self.strategy_params, self.n_chain, self.like_obj, eval_tracker=self.eval_tracker
        )
        self.sigma_diags[:] = sigma_diags
        self.fishers[:] = fishers
        self.chol_fishers[:] = chol_fishers
        sigma_scales, gamma_mults = set_scales(self.n_par, self.T_ladder, self.sigma_diags)
        self.sigma_scales[:] = sigma_scales
        self.gamma_mults[:] = gamma_mults

    def reset_fishers(
        self, itrn: int, block_size: int, samples: NDArray[np.floating], logLs: NDArray[np.floating]
    ) -> None:
        """Reset the fisher matrices from input samples"""
        if itrn // block_size < 4 or itrn % (block_size * self.strategy_params.fisher_downsample) == 0:
            samples_fisher = np.zeros((self.n_chain, self.n_par))
            if self.strategy_params.verbose_fisher:
                print('fisher update', itrn)
            for itrt in range(self.n_chain):
                # TODO fishers should not all be the same,
                # but try making them so for now because of fisher calculation instability
                index_select = np.unravel_index(np.argmax(logLs[:, :]), logLs.shape)
                # index_select = (np.random.randint(0,self.block_size+1),np.random.randint(0,self.n_cold))
                samples_fisher[itrt] = samples[index_select]
            self.reset_fishers_from_point(samples_fisher)

    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)


class FisherFullJump(AbstractJump):
    def __init__(self, manager: FisherJumpManager) -> None:
        self.manager: FisherJumpManager = manager
        self.print_name = 'Fisher All-D'

    def bind_native(self, likelihood_natives: NativeLikelihoodFunctions) -> NativeJumpCall:
        """Bind the jump over the manager's runtime Cholesky state."""
        del likelihood_natives

        @njit(inline='always')
        def native_call(
            sample_point: NDArray[np.floating], itrt: int, state: FisherNativeState
        ) -> tuple[NDArray[np.floating], float, bool]:
            chol_fishers = state[1]
            gamma_mults = state[2]
            return fisher_full_jump_helper(sample_point, itrt, chol_fishers, gamma_mults)

        return native_call

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        """Apply a fisher matrix jump"""
        return fisher_full_jump_helper(
            sample_point,
            itrt,
            self.manager.chol_fishers,
            self.manager.gamma_mults,
        )
