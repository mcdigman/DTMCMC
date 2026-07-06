"""the 2D hyper-pyramid lkelihood, adapted from https://github.com/joshspeagle/dynesty/blob/master/demos/"""
import numba as nb
import numpy as np
from correction_helpers import reflect_into_range
from numba import njit
from numba.experimental import jitclass
from proposal_strategy_helpers import ProposalStrategyParameters

strategy_default = ProposalStrategyParameters(use_chol_fishers=True,use_de=True,cold_prior_weight=0.,cold_de_weight=1./3,hot_de_weight=1./3,cold_fisher_weight=1./3.,hot_fisher_weight=1./3.,hot_prior_target_weight=1.,big_de_prob=0.1,de_subspace_frac=1.,de_subspace_override_frac=1.,fisher_subspace_frac=1.,eps_default=1.e-1,sigma_default=4.,max_fisher_el=np.inf)
strategy_default.use_chol_fishers = True
strategy_default.big_de_prob = 0.1
strategy_default.eps_default = 1.e-1
strategy_default.sigma_default = 4.
strategy_default.hot_prior_target_weight = 1.

#constants
low_lim = -15.
high_lim = 15.
s, sigma = 0.5, 1.
center = 0.


@njit()
def loglike(x):
    """Get the likelihood"""
    return -max(np.abs((x - center) / sigma))**(1. / s)

@njit()
def prior_draw():
    """Do a prior draw"""
    return np.random.uniform(low_lim,high_lim,2)

@njit()
def correct_bounds(v):
    """Wrap the parameter into bounds"""
    v[0] = reflect_into_range(v[0],low_lim,high_lim)
    v[1] = reflect_into_range(v[1],low_lim,high_lim)
    return v

@njit()
def check_bounds(v):
    """Check if a sample is within the prior range"""
    for itrp in range(v.size):
        if not low_lim<v[itrp]<high_lim:
            return False
    return True

@njit()
def prior_factor(v):
    """Get the prior factor, if one is needed"""
    return 0.

@jitclass([('n_par',nb.int64)])
class Likelihood():
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self):
        """Create the class and store any object specific variables"""
        self.n_par = 2

    def loglike(self,v):
        """Get the log likelihood given a set of parameters v"""
        return loglike(v)

    def prior_draw(self):
        """Get a draw from the prior"""
        return prior_draw()

    def prior_proposal(self,v_in):
        """Get a proposal from the prior"""
        v_out = prior_draw()
        return v_out,prior_factor(v_in)-prior_factor(v_out)

    def prior_factor(self,v):
        """Get the density factor for prior draws, if the prior draws are not uniform"""
        return prior_factor(v)

    def correct_bounds(self,v):
        """Correct the bounds of a draw to be in range, if allowed for this likelihood"""
        return correct_bounds(v)

    def check_bounds(self,v):
        """Check if the bounds of a draw are in the prior range but do not change them"""
        return check_bounds(v)

def gen_draws(n_draws,n_par,attempt_lim=10000):
    """Get posterior draws"""
    raise NotImplementedError('Posterior draws not implemented for the likelihood')

def get_labels(n_par):
    """Get useful labels for corner plots"""
    labels = []
    for itrp in range(n_par):
        labels.append(r'$v_'+str(itrp)+'$')
    return labels
