"""a gaussian mixture likelihood in n dimensions with 2 unequal modes at +/-5"""
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
strategy_default.de_subspace_frac = 1.
strategy_default.de_subspace_override_frac = 1.
strategy_default.fisher_subspace_frac = 1.
strategy_default.big_de_prob = 0.9
strategy_default.eps_default = 1.e-2
strategy_default.sigma_default = 100.
strategy_default.hot_prior_target_weight = 1.
strategy_default.max_fisher_el = 1.e7

#constants
low_lim = -10
high_lim = 10


@njit()
def loglike(v,n_par):
    """Get likelihood for gaussian mixture"""
    res1 = np.log(1. /(3*(2. * np.pi)**(n_par/2)))
    res2 = np.log(2. /(3*(2. * np.pi)**(n_par/2)))
    for itrp in range(n_par):
        res1 += -(v[itrp]-5)**2/2
        res2 += -(v[itrp]+5)**2/2

    res = np.logaddexp(res1,res2)

    return res

@njit()
def prior_draw(n_par):
    """Get a prior draw"""
    draw = np.zeros(n_par)
    for itrp in range(n_par):
        draw[itrp] = np.random.uniform(low_lim,high_lim)
    return draw

@njit()
def prior_factor(v,n_par):
    """The prior factor, if one is needed"""
    return 0.

@njit()
def correct_bounds(v,n_par):
    """Wrap parameters into range"""
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
        mode_choose = np.random.uniform(0.,1.)
        if mode_choose<1./3.:
            draw_loc = np.random.normal(5,1,n_par)
        else:
            draw_loc = np.random.normal(-5,1,n_par)

        while not check_bounds(draw_loc):
            if itra==attempt_lim:
                print('failed to find valid posterior point')
                assert False

            #redraw if it doesn't fit
            mode_choose = np.random.uniform(0.,1.)
            if mode_choose<1./3.:
                draw_loc = np.random.normal(5,1,n_par)
            else:
                draw_loc = np.random.normal(-5,1,n_par)
            itra += 1

    return draws

def get_labels(n_par):
    """Get useful labels for corner plots"""
    labels = []
    for itrp in range(n_par):
        labels.append(r'$v_'+str(itrp)+'$')
    return labels
