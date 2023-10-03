"""C 2023 Matthew C. Digman
Module to manage differential evoultion jumps"""
import numpy as np
from numba import njit

from DTMCMC.jump_manager import JumpManager,AbstractJump

# TODO apply a global default jump weight
# TODO fix name lengths

class DEJumpManager(JumpManager):
    """manage the differential evolution jumps, subclass of DTMCMC.jump_manager.JumpManager"""

    def __init__(self, T_ladder, like_obj, config):
        """create the manager object"""
        self.strategy_params = DEStrategyParameters(config)

        self.de_thin = self.strategy_params.de_thin
        self.de_size = self.strategy_params.de_size
        self.de_subspace_frac = self.strategy_params.de_subspace_frac

        jumps = [DEStandardFullJump(self),DEStandardRandomSubspaceJump(self),DEBigFullJump(self),DEBigRandomSubspaceJump(self)]

        JumpManager.__init__(self, T_ladder, like_obj, jumps)

        self.de_buffer = np.zeros((self.de_size, self.n_chain, self.n_par))
        initialize_de_helper(self.de_buffer, self.de_size, self.n_chain, self.like_obj)

        self.itrde_write = 1
        self.itrde_count = 1


    def write_de(self, samples):
        """write to the differential evolution buffer"""
        if self.itrde_count == 0:
            write_de_helper(self.itrde_count, self.itrde_write, self.de_buffer, samples)
            # wrap write index
            self.itrde_write += 1
            if self.itrde_write == self.de_size:
                self.itrde_write = 0
        self.itrde_count += 1
        # wrap counter for whether to write
        if self.itrde_count >= self.de_thin:
            self.itrde_count = 0

    def set_jump_weights(self):
        """set the conditional probabilities of the different jump types"""
        n_chain = self.T_ladder.n_chain
        n_cold = self.T_ladder.n_cold

        cold_de_weight = self.strategy_params.cold_de_weight  # weight of de in cold proposals
        hot_de_weight = self.strategy_params.hot_de_weight    # weight of de in hot proposals
        de_full_frac = self.strategy_params.de_full_d_frac    # fraction of time not to do a subspace jump
        big_de_prob = self.strategy_params.big_de_prob        # probability of doing a full length de jump

        jump_weights = np.zeros((n_chain, self.n_jump_types))
        # just a default equal weight
        jump_weights[:] = 0.333

        standard_prob = 1-self.strategy_params.big_de_prob    # probability of doing a standard jump
        subspace_prob = 1.-de_full_frac                       # probability of doing a subspace jump

        standard_full_prob = standard_prob*de_full_frac       # probability of doing a standard full jump
        standard_subspace_prob = standard_prob*subspace_prob  # probability of doing a standard subspace jump

        big_subspace_prob = big_de_prob*subspace_prob         # probability of doing a full length jump in a subspace
        big_full_prob = big_de_prob*de_full_frac              # probability of doing a full length jump in a subspace



        # get the indices of the jump types we need to assign probabilities for
        de_standard_full_idx = -1
        de_standard_subspace_idx = -1
        de_big_full_idx = -1
        de_big_subspace_idx = -1
        for itrp,jump in enumerate(self.jumps):
            if isinstance(jump,DEStandardFullJump):
                de_standard_full_idx = itrp
            if isinstance(jump,DEStandardRandomSubspaceJump):
                de_standard_subspace_idx = itrp
            if isinstance(jump,DEBigFullJump):
                de_big_full_idx = itrp
            if isinstance(jump,DEBigRandomSubspaceJump):
                de_big_subspace_idx = itrp

        assert de_standard_full_idx >= 0
        assert de_standard_subspace_idx >= 0
        assert de_big_full_idx >= 0
        assert de_big_subspace_idx >= 0

        jump_weights[:n_cold, de_standard_full_idx] = cold_de_weight*standard_full_prob
        jump_weights[n_cold:, de_standard_full_idx] = hot_de_weight*standard_full_prob

        jump_weights[:n_cold, de_standard_subspace_idx] = cold_de_weight*standard_subspace_prob
        jump_weights[n_cold:, de_standard_subspace_idx] = hot_de_weight*standard_subspace_prob

        jump_weights[:n_cold, de_big_full_idx] = cold_de_weight*big_full_prob
        jump_weights[n_cold:, de_big_full_idx] = hot_de_weight*big_full_prob

        jump_weights[:n_cold, de_big_subspace_idx] = cold_de_weight*big_subspace_prob
        jump_weights[n_cold:, de_big_subspace_idx] = hot_de_weight*big_subspace_prob

        self.jump_weights = jump_weights
        assert np.all(self.jump_weights >= 0.)

    def post_step_update(self, samples):
        """do any needed internal processing after an individual step of all temperatures;
        mainly intended to be used to write to differential evolution buffer"""
        self.write_de(samples)

    def record_config(self, config_in):
        """record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)


class DEStandardFullJump(AbstractJump):
    """apply a jump with standard random size in all dimensions
    null proposals are marked as failures"""

    def __init__(self,manager):
        self.manager = manager
        AbstractJump.__init__(self,'DE Std All-D')

    def __call__(self,sample_point,itrt):
        return apply_de_helper(self.manager.de_buffer, self.manager.de_subspace_frac, itrt, sample_point, False, False)

class DEStandardRandomSubspaceJump(AbstractJump):
    """apply a jump with standard random size in a random subspace
    null proposals are marked as failures"""

    def __init__(self,manager):
        self.manager = manager
        AbstractJump.__init__(self,'DE Std Random-D')

    def __call__(self,sample_point,itrt):
        return apply_de_helper(self.manager.de_buffer, self.manager.de_subspace_frac, itrt, sample_point, True, False)

class DEBigFullJump(AbstractJump):
    """apply the full length differential evolution jump in all dimensions
    null proposals are marked as failures"""

    def __init__(self,manager):
        self.manager = manager
        AbstractJump.__init__(self,'DE Big All-D')

    def __call__(self,sample_point,itrt):
        return apply_de_helper(self.manager.de_buffer, self.manager.de_subspace_frac, itrt, sample_point, False, True)

class DEBigRandomSubspaceJump(AbstractJump):
    """apply the full length differential evolution jump in a random subspace
    null proposals are marked as failures"""

    def __init__(self,manager):
        self.manager = manager
        AbstractJump.__init__(self,'DE Big Random-D')

    def __call__(self,sample_point,itrt):
        return apply_de_helper(self.manager.de_buffer, self.manager.de_subspace_frac, itrt, sample_point, True, True)

@njit()
def apply_de_helper(de_buffer, de_subspace_frac, itrt, sample_point, do_subspace, do_big):
    """apply the differential evolution jump"""
    de_size = de_buffer.shape[0]
    n_par = de_buffer.shape[2]

    itrd1 = np.random.randint(0, de_size)
    itrd2 = np.random.randint(0, de_size)

    if do_big:
        alpha = 1.
    else:
        alpha = 1.68/np.sqrt(n_par)*np.random.normal(0., 1.)

    delta = de_buffer[itrd1, itrt, :]-de_buffer[itrd2, itrt, :]
    count = n_par
    if do_subspace:
        safe_itrp = np.random.randint(n_par)
        for itrp in range(0, n_par):
            if np.random.uniform(0., 1.) > de_subspace_frac and itrp != safe_itrp:
                delta[itrp] = 0.
                count -= 1
        assert count > 0

    # calculate the new point based on the shifts above
    new_point = sample_point+alpha*delta

    # make sure something changed or else flag the jump as trivial
    nontrivial = not np.all(delta == 0.) and not alpha == 0.

    # density factor is 0 for differential evolution jumps
    return new_point, 0., nontrivial


def initialize_de_helper(de_buffer, de_size, n_chain, like_obj):
    """helper to initialize the differential evolution buffer with prior draws"""
    for itrd in range(de_size):
        for itrt in range(n_chain):
            de_buffer[itrd, itrt, :] = like_obj.prior_draw()


@njit()
def write_de_helper(itrde_count, itrde_write, de_buffer, samples):
    """helper to write to the differential evolution buffer"""
    if itrde_count == 0:
        de_buffer[itrde_write, :] = samples


class DEStrategyParameters():
    """container to store some parameters related to the strategy of differential evolution proposal generation"""

    def __init__(self, config):
        """initialize the object with the prescribed parameters"""
        self.config = config
        config_de = self.config['DEJumpManager']

        # how often to do de draws in the cold chains
        self.cold_de_weight = config_de.getfloat('cold_de_weight', 0.333)
        # how often to do de draws in the hottest finite temperature chain
        self.hot_de_weight = config_de.getfloat('hot_de_weight', 0.333)
        # how often to do the big differential evolution jump
        self.big_de_prob = config_de.getfloat('big_de_prob', 0.5)
        # what fraction of dimensions to include in de subspace jumps
        self.de_subspace_frac = config_de.getfloat('de_subspace_frac', 1.)
        # how often to not do subspace jumps when doing a de jump
        self.de_full_d_frac = config_de.getfloat('de_full_d_frac', 1.)
        # size of differential evolution buffer
        self.de_size = config_de.getint('de_size', 1000)
        # how much to thin the differential evolution buffer by
        self.de_thin = config_de.getint('de_thin', 1)

    def copy(self):
        """copy the object"""
        return DEStrategyParameters(self.config)

    def record_config(self, config_in):
        """record the current configuration to the requested configuration object
            inputs:
                config_in: ConfigParser object"""
        config_de = config_in['DEJumpManager']
        config_de['cold_de_weight'] = str(self.cold_de_weight)
        config_de['hot_de_weight'] = str(self.hot_de_weight)
        config_de['big_de_prob'] = str(self.big_de_prob)
        config_de['de_subspace_frac'] = str(self.de_subspace_frac)
        config_de['de_full_d_frac'] = str(self.de_full_d_frac)
        config_de['de_size'] = str(self.de_size)
        config_de['de_thin'] = str(self.de_thin)
