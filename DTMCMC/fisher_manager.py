"""C 2023 Matthew C. Digman
module to store objects related to fisher matrix jumps
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, override

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.jump_manager import AbstractNativeJump, JumpManager
from DTMCMC.lapack_wrappers import solve_triangular
from DTMCMC.likelihood import AbstractLikelihood

if TYPE_CHECKING:
    from configparser import ConfigParser

    from DTMCMC.temperature_ladder_helpers import TemperatureLadder


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


class FisherNativeState(NamedTuple):
    """Runtime state bundle for the Fisher-derived native jumps.

    Re-read at every block entry, so the between-block scale refreshes are
    picked up exactly as on the Python path.
    """

    n_par: int
    fisher_subspace_frac: float
    sigma_scales: NDArray[np.floating]
    chol_fishers: NDArray[np.floating]
    gamma_mults: NDArray[np.floating]


@njit(inline='always')
def _sigma_full_native(
    sample_point: NDArray[np.floating],
    itrt: int,
    state: FisherNativeState,
) -> tuple[NDArray[np.floating], float, bool]:
    """Per-class native standard-deviation jump in all dimensions."""
    return sigma_subspace_jump_helper(
        sample_point, itrt, state.n_par, state.fisher_subspace_frac, state.sigma_scales, True
    )


@njit(inline='always')
def _sigma_subspace_native(
    sample_point: NDArray[np.floating],
    itrt: int,
    state: FisherNativeState,
) -> tuple[NDArray[np.floating], float, bool]:
    """Per-class native standard-deviation jump in random subspaces."""
    return sigma_subspace_jump_helper(
        sample_point, itrt, state.n_par, state.fisher_subspace_frac, state.sigma_scales, False
    )


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


@njit(inline='always')
def _fisher_full_native(
    sample_point: NDArray[np.floating],
    itrt: int,
    state: FisherNativeState,
) -> tuple[NDArray[np.floating], float, bool]:
    """Per-class native full Cholesky Fisher jump."""
    return fisher_full_jump_helper(sample_point, itrt, state.chol_fishers, state.gamma_mults)


class SigmaFullJump[LikelihoodType: AbstractLikelihood[NamedTuple]](
    AbstractNativeJump[LikelihoodType, FisherNativeState]
):
    """Standard Deviation Jump in Full Dimensions"""

    def __init__(self, manager: FisherJumpManager[LikelihoodType]) -> None:
        """Create the jump"""
        print_name = 'Std All-D'
        super().__init__(_sigma_full_native, manager, print_name)

    @property
    @override
    def declared_internal_evals(self) -> int:
        return 0


class SigmaRandomSubspaceJump[LikelihoodType: AbstractLikelihood[NamedTuple]](
    AbstractNativeJump[LikelihoodType, FisherNativeState]
):
    """Standard deviation jump in random subspaces"""

    def __init__(self, manager: FisherJumpManager[LikelihoodType]) -> None:
        print_name = 'Std Random-D'
        super().__init__(_sigma_subspace_native, manager, print_name)

    @property
    @override
    def declared_internal_evals(self) -> int:
        return 0


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


def declared_fisher_stencil_evals(n_chain: int, n_par: int, use_chol_fishers: bool) -> int:
    """Deterministic likelihood-evaluation count of one ``set_fishers`` stencil.

    Per chain: one center evaluation, two per diagonal element, and, with
    Cholesky Fisher matrices, four per off-diagonal pair. Verified against
    an independent call spy in the eval-accounting tests.
    """
    if use_chol_fishers:
        return n_chain * (1 + 2 * n_par**2)
    return n_chain * (1 + 2 * n_par)


def set_fishers(
    sample_set: NDArray[np.floating],
    strategy_params: FisherStrategyParameters,
    n_chain: int,
    like_obj: AbstractLikelihood[NamedTuple],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Set up the fisher matrices"""
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


class FisherJumpManager[LikelihoodType: AbstractLikelihood[Any]](JumpManager[LikelihoodType, FisherNativeState]):
    """manage everything related to fisher matrix jumps, subclass of DTMCMC.jump_manager.JumpManager

    The fisher arrays are allocated once and refreshed in place; the native
    state bundle is re-read at every block entry, so between-block refreshes
    reach the native jumps exactly as they reach the Python path.
    """

    def __init__(
        self,
        T_ladder: TemperatureLadder,
        like_obj: LikelihoodType,
        sample_set: NDArray[np.floating],
        config: ConfigParser,
    ) -> None:
        """Create the object"""
        if not isinstance(like_obj, AbstractLikelihood):
            msg = (
                f'likelihood {type(like_obj).__qualname__} does not implement AbstractLikelihood '
                '(correct_bounds and get_epsilons), which FisherJumpManager requires'
            )
            raise TypeError(msg)
        self.strategy_params = FisherStrategyParameters(config)

        jumps: list[AbstractNativeJump[LikelihoodType, FisherNativeState]] = [
            FisherFullJump(self),
            SigmaFullJump(self),
            SigmaRandomSubspaceJump(self),
        ]

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
        # the construction-time stencil below is a declared deterministic
        # cost read by the sampler's eval accounting
        self.declared_construction_evals: int = self.declared_refresh_evals()
        self.reset_fishers_from_point(self.sample_set)

    def declared_refresh_evals(self) -> int:
        """Deterministic likelihood-evaluation cost of one fisher refresh."""
        return declared_fisher_stencil_evals(self.n_chain, self.n_par, self.strategy_params.use_chol_fishers)

    @property
    @override
    def native_state(self) -> FisherNativeState:
        """Return the runtime state bundle read by this manager's native jumps."""
        return FisherNativeState(
            self.n_par,
            self.strategy_params.fisher_subspace_frac,
            self.sigma_scales,
            self.chol_fishers,
            self.gamma_mults,
        )

    @override
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

        self._jump_weights = jump_weights
        assert np.all(self._jump_weights >= 0.0)

    @override
    def post_block_update(
        self, itrn: int, block_size: int, samples: NDArray[np.floating], logLs: NDArray[np.floating]
    ) -> int:
        """Do any needed internal processing after an individual block of size block_size:
        ie, fisher matrix updates. Returns the deterministic likelihood-evaluation cost.
        """
        return self.reset_fishers(itrn, block_size, samples, logLs)

    def reset_fishers_from_point(self, sample_set: NDArray[np.floating]) -> None:
        """Set the fisher matrix object at the specified point, updating in place"""
        sigma_diags, fishers, chol_fishers = set_fishers(sample_set, self.strategy_params, self.n_chain, self.like_obj)
        self.sigma_diags[:] = sigma_diags
        self.fishers[:] = fishers
        self.chol_fishers[:] = chol_fishers
        sigma_scales, gamma_mults = set_scales(self.n_par, self.T_ladder, self.sigma_diags)
        self.sigma_scales[:] = sigma_scales
        self.gamma_mults[:] = gamma_mults

    def reset_fishers(
        self, itrn: int, block_size: int, samples: NDArray[np.floating], logLs: NDArray[np.floating]
    ) -> int:
        """Reset the fisher matrices from input samples when a refresh is scheduled.

        Returns the number of likelihood evaluations performed (the declared
        stencil cost on a refresh block, 0 otherwise).
        """
        if itrn // block_size < 4 or itrn % (block_size * self.strategy_params.fisher_downsample) == 0:
            samples_fisher = np.zeros((self.n_chain, self.n_par))
            if self.strategy_params.verbose_fisher:
                print('fisher update', itrn)
            # TODO fishers should not all be the same,
            # but try making them so for now because of fisher calculation instability
            index_select = np.unravel_index(np.argmax(logLs[:, :]), logLs.shape)
            # index_select = (np.random.randint(0,self.block_size+1),np.random.randint(0,self.n_cold))
            for itrt in range(self.n_chain):
                samples_fisher[itrt] = samples[index_select]
            self.reset_fishers_from_point(samples_fisher)
            return self.declared_refresh_evals()
        return 0

    @override
    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)


class FisherFullJump[LikelihoodType: AbstractLikelihood[NamedTuple]](
    AbstractNativeJump[LikelihoodType, FisherNativeState]
):
    def __init__(self, manager: FisherJumpManager[LikelihoodType]) -> None:
        print_name = 'Fisher All-D'
        super().__init__(_fisher_full_native, manager, print_name)

    @property
    @override
    def declared_internal_evals(self) -> int:
        return 0
