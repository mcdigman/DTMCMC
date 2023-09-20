"""C 2023 Matthew C. Digman
hold some helpers to help determine the proposal strategy"""
import numpy as np


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
        self.use_chol_fishers = use_chol_fishers
        self.cold_prior_weight = cold_prior_weight
        self.cold_de_weight = cold_de_weight
        self.hot_de_weight = hot_de_weight
        self.cold_fisher_weight = cold_fisher_weight
        self.hot_fisher_weight = hot_fisher_weight
        self.hot_prior_target_weight = hot_prior_target_weight
        self.big_de_prob = big_de_prob
        self.de_subspace_frac = de_subspace_frac
        self.de_subspace_override_frac = de_subspace_override_frac
        self.fisher_subspace_frac = fisher_subspace_frac
        self.fisher_subspace_override_frac = fisher_subspace_override_frac
        self.fisher_downsample = fisher_downsample
        self.sigma_default = sigma_default
        self.max_fisher_el = max_fisher_el
        self.de_size = de_size
        self.de_thin = de_thin

    def copy(self):
        """copy the object"""
        return ProposalStrategyParameters(self.use_chol_fishers, self.cold_prior_weight, self.cold_de_weight, self.hot_de_weight, self.cold_fisher_weight, self.hot_fisher_weight, self.hot_prior_target_weight, self.big_de_prob, self.de_subspace_frac, self.de_subspace_override_frac, self.fisher_subspace_frac, self.sigma_default, self.max_fisher_el, self.de_size, self.de_thin)
