"""the banana likelihood in n dimensions"""
import numpy as np
import numba as nb
from numba.experimental import jitclass
from numba import njit
from correction_helpers import reflect_into_range
from proposal_strategy_helpers import ProposalStrategyParameters
strategy_default = ProposalStrategyParameters(use_chol_fishers=True,use_de=True,cold_prior_weight=0.,cold_de_weight=1./3,hot_de_weight=1./3,cold_fisher_weight=1./3.,hot_fisher_weight=1./3.,hot_prior_target_weight=1.,big_de_prob=0.1,de_subspace_frac=1.,de_subspace_override_frac=1.,fisher_subspace_frac=1.,eps_default=1.e-1,sigma_default=4.,max_fisher_el=np.inf)
strategy_default.use_chol_fishers = False
strategy_default.cold_prior_weight = 0.#1./12.#1./6.#1./12.
strategy_default.big_de_prob = 0.1
strategy_default.eps_default = 1.e-1
strategy_default.sigma_default = 100.
strategy_default.hot_prior_target_weight = 1.
strategy_default.max_fisher_el = 1.e7

#constants
#limits for first two parameters
low_lim01 = -10000.
high_lim01 = -10000.
limits for n>=2 parameters
low_limn = -100.
high_limn = 100.
#see https://link.springer.com/content/pdf/10.1007/s001800050022.pdf
#and https://www.tandfonline.com/doi/pdf/10.1198/jcgs.2009.06134?needAccess=true
#20d banana is default
B = 0.1 #bananacity parameter

@jitclass([('n_par',nb.int64)])
class Likelihood():
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self,n_par=20):
        """create the class and store any object specific variables"""
        assert n_par >= 2
        self.n_par = n_par

    def loglike(self,v):
        """get the log likelihood given a set of parameters v"""
        return loglike(v,self.n_par)

    def prior_draw(self):
        """get a draw from the prior"""
        return prior_draw(self.n_par)

    def prior_proposal(self,v_in):
        """get a proposal from the prior"""
        v_out = prior_draw(self.n_par)
        return v_out,prior_factor(v_in,self.n_par)-prior_factor(v_out,self.n_par)

    def prior_factor(self,v):
        """get the density factor for prior draws, if the prior draws are not uniform"""
        return prior_factor(v,self.n_par)

    def correct_bounds(self,v):
        """correct the bounds of a draw to be in range, if allowed for this likelihood"""
        return correct_bounds(v,self.n_par)

    def check_bounds(self,v):
        """check if the bounds of a draw are in the prior range but do not change them"""
        return check_bounds(v)


@njit()
def loglike(v,n_par):
    """get the log likelihood for the 'banana' test case"""
    res = -v[0]**2/200-1/2*(v[1]+B*v[0]**2-100*B)**2
    for itrp in range(2,n_par):
        res += -1/2*v[itrp]**2
    return res

@njit()
def prior_draw(n_par):
    """draw from our priors"""
    draw = np.zeros(n_par)
    draw[0] = np.random.uniform(low_lim01,high_lim01)
    draw[1] = np.random.uniform(low_lim01,high_lim01)
    for itrp in range(2,n_par):
        draw[itrp] = np.random.uniform(low_limn,high_limn)
    return draw

@njit()
def prior_factor(v,n_par):
    """the density factor for prior draws, 0 if all uniform"""
    return 0.

@njit()
def correct_bounds(v,n_par):
    """wrap the parameters back into bounds"""
    v[0] = reflect_into_range(v[0],low_lim01,high_lim01)
    v[1] = reflect_into_range(v[1],low_lim01,high_lim01)
    for itrp in range(2,n_par):
        v[itrp] = reflect_into_range(v[itrp],low_limn,high_limn)
    return v

@njit()
def check_bounds(v):
    """check if a sample is within the prior range"""
    if not low_lim01<v[0]<high_lim01:
        return False
    if not low_lim01<v[1]<high_lim01:
        return False
    for itrp in range(2,v.size):
        if not low_limn<v[itrp]<high_limn:
            return False
    return True

def gen_draws(n_draws,n_par,attempt_lim=10000):
    """get posterior draws"""
    raise NotImplementedError('Posterior draws not implemented for the likelihood')

def get_labels(n_par):
    """get useful labels for corner plots"""
    labels = [r'$b_1',r'$b_2']
    for itrp in range(2,n_par):
        labels.append(r'$v_'+str(itrp)+'$')
    return labels
