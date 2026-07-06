"""The ar(1) likelihood in n dimensions"""
import numba as nb
import numpy as np
from correction_helpers import reflect_into_range
from numba import njit
from numba.experimental import jitclass
from proposal_strategy_helpers import ProposalStrategyParameters

strategy_default = ProposalStrategyParameters(use_chol_fishers=True,use_de=True,cold_prior_weight=0.,cold_de_weight=1./3,hot_de_weight=1./3,cold_fisher_weight=1./3.,hot_fisher_weight=1./3.,hot_prior_target_weight=1.,big_de_prob=0.1,de_subspace_frac=1.,de_subspace_override_frac=1.,fisher_subspace_frac=1.,eps_default=1.e-1,sigma_default=4.,max_fisher_el=np.inf)
strategy_default.use_chol_fishers = False
strategy_default.cold_prior_weight = 0.#1./12.#1./6.#1./12.
strategy_default.cold_fisher_weight = 1./3.
strategy_default.hot_fisher_weight = 1./3
strategy_default.big_de_prob = 0.1
strategy_default.eps_default = 1.e-2
strategy_default.sigma_default = 100.
strategy_default.hot_prior_target_weight = 1.
strategy_default.max_fisher_el = 1.e7

#constants
low_lim = -10.
high_lim = 10.
alpha = 0.9
beta = np.sqrt(1-alpha**2)
const = np.log(1. / np.sqrt(2. * np.pi*beta**2))  # normalization constant


@njit()
def loglike(v,n_par):
    """Get the likelihood for ar1"""
    res = 0.
    x_next = 0.
    res += const-1/2*(v[0]-x_next)**2
    x_next = alpha*v[0]

    for itrp in range(1,n_par):
        res += const-1/2/beta**2*(v[itrp]-x_next)**2
        x_next = alpha*v[itrp]
    return res

@njit()
def prior_draw(n_par):
    """Do a prior draw"""
    draw = np.zeros(n_par)
    for itrp in range(n_par):
        draw[itrp] = np.random.uniform(low_lim,high_lim)
    return draw

@njit()
def prior_factor(v,n_par):
    """Density factor for prior draws, if needed"""
    return 0.

@njit()
def correct_bounds(v,n_par):
    """Reflect all parameter back into boundaries"""
    for itrp in range(n_par):
        v[itrp] = reflect_into_range(v[itrp],low_lim,high_lim)
    return v

@njit()
def check_bounds(v):
    """Check if a sample is within the prior range"""
    for itrp in range(v.size):
        if not low_lim<v[itrp]<high_lim:
            return False
    return True

@jitclass([('n_par',nb.int64)])
class Likelihood():
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self,n_par=50):
        """Create the class and store any object specific variables"""
        self.n_par = n_par

    def loglike(self,v):
        """Get the log likelihood given a set of parameters v"""
        return loglike(v,self.n_par)

    def prior_draw(self):
        """Get a draw from the prior"""
        return prior_draw(self.n_par)

    def prior_proposal(self,v_in):
        """Get a proposal from the prior"""
        v_out = prior_draw(self.n_par)
        return v_out,prior_factor(v_in,self.n_par)-prior_factor(v_out,self.n_par)

    def prior_factor(self,v):
        """Get the density factor for prior draws, if the prior draws are not uniform"""
        return prior_factor(v,self.n_par)

    def correct_bounds(self,v):
        """Correct the bounds of a draw to be in range, if allowed for this likelihood"""
        return correct_bounds(v,self.n_par)

    def check_bounds(self,v):
        """Check if the bounds of a draw are in the prior range but do not change them"""
        return check_bounds(v)

@njit()
def gen_draws(n_draws,n_par,attempt_lim=10000):
    """Get posterior draws"""
    draws = np.zeros((n_draws,n_par))
    for itrk in range(n_draws):
        itra = 0
        draw_loc = np.zeros(n_par)
        draw_loc[0] = np.random.normal(0.,1.)
        for itrp in range(1,n_par):
            n1 = np.random.normal(alpha*draw_loc[itrp-1],beta)
            draw_loc[itrp] = n1

        while not check_bounds(draw_loc):
            if itra==attempt_lim:
                print('failed to find valid posterior point')
                assert False

            draw_loc = np.zeros(n_par)
            draw_loc[0] = np.random.normal(0.,1.)
            for itrp in range(1,n_par):
                n1 = np.random.normal(alpha*draw_loc[itrp-1],beta)
                draw_loc[itrp] = n1
        draws[itrk] = draw_loc
    return draws

def get_labels(n_par):
    """Get useful labels for corner plots"""
    labels = []
    for itrp in range(n_par):
        labels.append(r'$v_'+str(itrp)+'$')
    return labels
