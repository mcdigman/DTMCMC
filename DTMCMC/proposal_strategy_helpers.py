"""C 2023 Matthew C. Digman
hold some helpers to help determine the proposal strategy"""
import numpy as np

import DTMCMC.fisher_manager as fm
import DTMCMC.de_manager as dm
import DTMCMC.prior_manager as pm

# TODO make proposal strategy hierarchical

class ProposalStrategyParameters():
    """container to store some parameters related to the strategy of proposal generation"""

    def __init__(self,
                 use_chol_fishers=False,            # whether to do fisher jumps using the cholesky decomposition
                 cold_prior_weight=1./3.,           # how often to do prior draws in the cold chains
                 cold_de_weight=1./3.,              # how often to do de draws in the cold chains
                 hot_de_weight=1./3.,               # how often to do de draws in the hottest finite temperature chain
                 cold_fisher_weight=1./3.,          # how often to do fisher draws in the cold chains
                 hot_fisher_weight=1./3.,           # how often to do fisher draws in the hottest finite temperature chain
                 hot_prior_target_weight=1./3.,     # how often to do prior draws in the hottest finite temperature chain
                 big_de_prob=0.5,                   # how often to do the big differential evolution jump
                 de_subspace_frac=1.,               # what fraction of dimensions to include in de subspace jumps
                 de_subspace_override_frac=1.,      # how often to not do subspace jumps when doing a de jump
                 fisher_subspace_frac=1.,           # what fraction of dimensions to include in fisher subspace jumps
                 fisher_subspace_override_frac=1.,  # how often to not do subspace jumps when doing a fisher jump
                 fisher_downsample=1,               # how many blocks to skip between fisher matrix updates
                 sigma_default=0.25,                # default sigma for fisher matrix jumps
                 max_fisher_el=np.inf,              # maximum element of fisher matrix
                 de_size=1000,                      # size of differential evolution buffer
                 de_thin=1):                        # how much to thin the differential evolution buffer by
        """initialize the object with the prescribed parameters"""
        self.fisher_strategy = fm.FisherStrategyParameters(use_chol_fishers, cold_fisher_weight, hot_fisher_weight, fisher_subspace_frac, fisher_subspace_override_frac, sigma_default, max_fisher_el)
        self.de_strategy = dm.DEStrategyParameters(cold_de_weight, hot_de_weight, big_de_prob, de_subspace_frac, de_subspace_override_frac, de_size, de_thin)
        self.prior_strategy = pm.PriorStrategyParameters(cold_prior_weight, hot_prior_target_weight)

    def copy(self):
        """copy the object"""
        return ProposalStrategyParameters(self.use_chol_fishers, self.cold_prior_weight, self.cold_de_weight, self.hot_de_weight, self.cold_fisher_weight, self.hot_fisher_weight, self.hot_prior_target_weight, self.big_de_prob, self.de_subspace_frac, self.de_subspace_override_frac, self.fisher_subspace_frac, self.sigma_default, self.max_fisher_el, self.de_size, self.de_thin)
