"""C 2023 Matthew C. Digman
module to store objects related to fisher matrix jumps"""
import numpy as np

from DTMCMC.lapack_wrappers import solve_triangular
from DTMCMC.jump_manager import JumpManager

# dictionary of display names for the jumps
JUMP_LABELS_DICT = {
    'FISHER_FULL': 'fisher full',
    'SIGMA_FULL': 'std full',
    'SIGMA_RANDOM_SUBSPACE': 'std subspace',
}

JUMP_NAMES = ['FISHER_FULL', 'SIGMA_FULL', 'SIGMA_RANDOM_SUBSPACE']

class FisherJumpManager(JumpManager):
    """manage everything related to fisher matrix jumps, subclass of DTMCMC.jump_manager.JumpManager"""

    def __init__(self, T_ladder, like_obj, sample_set, config):
        """create the object"""

        self.betas = T_ladder.betas

        self.strategy_params = FisherStrategyParameters(config)
        self.sample_set = sample_set

        JumpManager.__init__(self, T_ladder, like_obj, JUMP_NAMES, JUMP_LABELS_DICT)

        self.reset_fishers_from_point(self.sample_set)


    def dispatch_jump(self, sample_point, itrt, choose):
        """dispatch a fisher matrix jump based on the idx selected"""
        if choose == 0:
            return self.apply_fisher_full(sample_point, itrt)
        elif choose == 1:
            return self.apply_sigma_full(sample_point, itrt, do_subspace=False)
        elif choose == 2:
            return self.apply_sigma_full(sample_point, itrt, do_subspace=True)
        else:
            assert False

    def set_jump_weights(self):
        """set the relative probabilities of the different jump types"""
        n_cold = self.T_ladder.n_cold
        n_chain = self.T_ladder.n_chain
        jump_weights = np.zeros((n_chain, self.n_jump_types))
        jump_weights[:] = 1./3.  # just a default equal weight

        cold_weight = self.strategy_params.cold_fisher_weight
        hot_weight = self.strategy_params.hot_fisher_weight

        subspace_weight = 1.-self.strategy_params.fisher_full_d_frac
        full_weight = self.strategy_params.fisher_full_d_frac

        name_map = self.name_to_idx

        if self.strategy_params.use_chol_fishers:
            jump_weights[:n_cold, name_map['SIGMA_FULL']] = 0.
            jump_weights[:n_cold, name_map['SIGMA_RANDOM_SUBSPACE']] = 0.
            jump_weights[:n_cold, name_map['FISHER_FULL']] = cold_weight
        else:
            jump_weights[:n_cold, name_map['SIGMA_FULL']] = cold_weight*full_weight
            jump_weights[:n_cold, name_map['SIGMA_RANDOM_SUBSPACE']] = cold_weight*subspace_weight
            jump_weights[:n_cold, name_map['FISHER_FULL']] = 0.

        if self.strategy_params.use_chol_fishers:
            jump_weights[n_cold:, name_map['FISHER_FULL']] = hot_weight
            jump_weights[n_cold:, name_map['SIGMA_FULL']] = 0.
            jump_weights[n_cold:, name_map['SIGMA_RANDOM_SUBSPACE']] = 0.
        else:
            jump_weights[n_cold:, name_map['FISHER_FULL']] = 0.
            jump_weights[n_cold:, name_map['SIGMA_FULL']] = hot_weight*full_weight
            jump_weights[n_cold:, name_map['SIGMA_RANDOM_SUBSPACE']] = hot_weight*subspace_weight

        self.jump_weights = jump_weights
        assert np.all(self.jump_weights >= 0.)

    def post_block_update(self, itrn, block_size, samples, logLs):
        """do any needed internal processing after an individual block of size block_size:
        ie, fisher matrix updates"""
        return self.reset_fishers(itrn, block_size, samples, logLs)

    def apply_fisher_full(self, sample_point, itrt):
        """apply a fisher matrix jump"""
        n_par = sample_point.size
        new_point = sample_point + solve_triangular(
            self.chol_fishers[itrt],
            self.gamma_mults[itrt]*np.random.normal(0., 1., n_par),
            trans_a=True
        )
        return new_point, 0., True

    def apply_sigma_full(self, sample_point, itrt, do_subspace=False):
        """apply a standard deviation jump"""
        n_par = self.n_par
        mult = np.random.normal(0., 1., n_par)
        count = n_par
        if do_subspace:
            # ensure at least one random direction is protected so we aren't making null proposals
            safe_itrp = np.random.randint(n_par)
            for itrp in range(0, n_par):
                if (
                    np.random.uniform(0., 1.,) > self.strategy_params.fisher_subspace_frac
                    and itrp != safe_itrp
                ):
                    mult[itrp] = 0.
                    count -= 1
            assert count > 0

        new_point = sample_point+self.sigma_scales[itrt]*np.sqrt(n_par/count)*mult
        return new_point, 0., True

    def reset_fishers_from_point(self, sample_set):
        """set the fisher matrix object at the specified point"""
        self.sigma_diags, self.fishers, self.chol_fishers = set_fishers(
            sample_set, self.strategy_params, self.n_chain, self.like_obj
        )
        self.sigma_scales, self.gamma_mults = set_scales(self.n_par, self.T_ladder, self.sigma_diags)

    def reset_fishers(self, itrn, block_size, samples, logLs):
        """reset the fisher matrices from input samples"""
        if itrn//block_size < 4 or itrn % (block_size*self.strategy_params.fisher_downsample) == 0:
            samples_fisher = np.zeros((self.n_chain, self.n_par))
            print("fisher update", itrn)
            for itrt in range(self.n_chain):
                # TODO fishers should not all be the same,
                # but try making them so for now because of fisher calculation instability
                index_select = np.unravel_index(np.argmax(logLs[:, :]), logLs.shape)
                # index_select = (np.random.randint(0,self.block_size+1),np.random.randint(0,self.n_cold))
                samples_fisher[itrt] = samples[index_select]
            self.reset_fishers_from_point(samples_fisher)

    def record_config(self, config_in):
        """record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)


def set_fishers(sample_set, strategy_params, n_chain, like_obj):
    """set up the fisher matrices"""
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

    epsilons = like_obj.epsilons

    for itrt in range(n_chain):
        new_point = sample_set[itrt]
        new_point_alt = like_obj.correct_bounds(new_point.copy())
        assert np.all(new_point == new_point_alt)
        nn = like_obj.get_loglike(new_point)
        for itrp in range(n_par):
            eps = epsilons[itrp]
            pointp = new_point.copy()
            pointp[itrp] += 2*eps
            pointmp = like_obj.correct_bounds(pointp)
            pp = like_obj.get_loglike(pointp)

            pointm = new_point.copy()
            pointm[itrp] -= 2*eps
            pointm = like_obj.correct_bounds(pointm)
            mm = like_obj.get_loglike(pointm)

            fisher_loc = -(pp - 2.0*nn + mm)/(4*eps*eps)+1./sigma_default**2

            if use_chol_fishers:
                fishers[itrt, itrp, itrp] = fisher_loc
            if not np.isfinite(fisher_loc) or fisher_loc <= 0. or fisher_loc > max_fisher_el:
                if use_chol_fishers:
                    fishers[itrt, itrp, itrp] = 1./sigma_default**2
                sigma_diags[itrt, itrp] = sigma_default
            else:
                sigma_diags[itrt, itrp] = 1./np.sqrt(fisher_loc)

        if use_chol_fishers:
            for itrp1 in range(n_par):
                eps1 = epsilons[itrp1]
                for itrp2 in range(itrp1+1, n_par):
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

                    res = -(pp - mp - pm + mm)/(4.0*eps1*eps2)
                    if not np.isfinite(res) or np.abs(res) > max_fisher_el:
                        res = 0.

                    fishers[itrt, itrp1, itrp2] = res
                    fishers[itrt, itrp2, itrp1] = fishers[itrt, itrp1, itrp2]

            det_fisher = np.linalg.det(fishers[itrt])
            if (
                not np.isfinite(det_fisher) or
                det_fisher <= 0. or
                np.any(np.linalg.eigh(fishers[itrt])[0] <= 0.)
            ):
                for itrp1 in range(n_par):
                    for itrp2 in range(itrp1+1, n_par):
                        fishers[itrt, itrp1, itrp2] = 0.
                        fishers[itrt, itrp2, itrp1] = 0.

            chol_fishers[itrt] = np.linalg.cholesky(fishers[itrt])
    return sigma_diags, fishers, chol_fishers


def set_scales(n_par, T_ladder, sigma_diags):
    """helper to get several scaling parameters for fisher matrix jumps"""
    n_chain = T_ladder.n_chain
    betas = T_ladder.betas

    sigma_scales = np.zeros((n_chain, n_par))
    gamma_mults = np.zeros(n_chain)

    if np.any((betas > 0.) & (np.isfinite(betas))):
        # use the smallest postive beta if beta is 0 or non-finite for scaling
        small_default = np.min(betas[betas > 0.])
    else:
        small_default = 1.

    # set the scale factors for the sigma and fisher jumps as a function of temperature
    for itrj in range(n_chain):
        if np.isfinite(betas[itrj]):
            beta_loc = max(betas[itrj], small_default)
        else:
            beta_loc = small_default

        gamma_mults[itrj] = 2.38/np.sqrt(beta_loc)
        for itrp in range(n_par):
            sigma_scales[itrj, itrp] = 2.38*(sigma_diags[itrj, itrp]/np.sqrt(beta_loc)/np.sqrt(n_par))
    return sigma_scales, gamma_mults


class FisherStrategyParameters():
    """container to store some parameters related to the strategy of
    fisher matrix proposal generation"""

    def __init__(self, config):
        """initialize the object with the prescribed parameters"""
        self.config = config

        config_f = self.config['FisherJumpManager']

        # whether to do fisher jumps using the cholesky decomposition
        self.use_chol_fishers = config_f.getboolean('use_chol_fishers', False)
        # how often to do fisher draws in the cold chains
        self.cold_fisher_weight = config_f.getfloat('cold_fisher_weight', 0.333)
        # how often to do fisher draws in the hottest finite temperature chain
        self.hot_fisher_weight = config_f.getfloat('hot_fisher_weight', 0.333)
        # what fraction of dimensions to include in fisher subspace jumps
        self.fisher_subspace_frac = config_f.getfloat('fisher_subspace_frac', 1.)
        # how often to not do subspace jumps when doing a fisher jump
        self.fisher_full_d_frac = config_f.getfloat('fisher_full_d_frac', 1.)
        # how many blocks to skip between fisher matrix updates
        self.fisher_downsample = config_f.getint('fisher_downsample', 1)
        # default sigma for fisher matrix jumps
        self.sigma_default = config_f.getfloat('sigma_default', 100.)
        # maximum element of fisher matrix
        self.max_fisher_el = config_f.getfloat('max_fisher_el', np.inf)

    def copy(self):
        """copy the object"""
        return FisherStrategyParameters(self.config)

    def record_config(self, config_in):
        """record the current configuration to the requested configuration object
            inputs:
                config_in: ConfigParser object"""
        config_f = config_in['FisherJumpManager']
        config_f['use_chol_fishers'] = str(self.use_chol_fishers)
        config_f['cold_fisher_weight'] = str(self.cold_fisher_weight)
        config_f['hot_fisher_weight'] = str(self.hot_fisher_weight)
        config_f['fisher_subspace_frac'] = str(self.fisher_subspace_frac)
        config_f['fisher_full_d_frac'] = str(self.fisher_full_d_frac)
        config_f['fisher_downsample'] = str(self.fisher_downsample)
        config_f['sigma_default'] = str(self.sigma_default)
        config_f['max_fisher_el'] = str(self.max_fisher_el)
