"""the spoke wheel likelihood"""
import numba as nb
import numpy as np
from correction_helpers import reflect_into_range
from numba import njit
from numba.experimental import jitclass
from proposal_strategy_helpers import ProposalStrategyParameters

strategy_default = ProposalStrategyParameters(use_chol_fishers=True,use_de=True,cold_prior_weight=0.,cold_de_weight=1./3,hot_de_weight=1./3,cold_fisher_weight=1./3.,hot_fisher_weight=1./3.,hot_prior_target_weight=1.,big_de_prob=0.1,de_subspace_frac=1.,de_subspace_override_frac=1.,fisher_subspace_frac=1.,eps_default=1.e-1,sigma_default=4.,max_fisher_el=np.inf)
strategy_default.use_chol_fishers = False
strategy_default.big_de_prob = 0.9
strategy_default.eps_default = 1.e-2
strategy_default.sigma_default = 2.
strategy_default.hot_prior_target_weight = 1.

#constants
low_lim = -40.
high_lim = 40.

w = 0.1  # width
r1 = 30.
c1 = np.array([-r1, 0.])  # location of mode 1
c2 = np.array([r1, 0.])  # location of mode 2
c3 = np.array([0., -r1])  # location of mode 3
c4 = np.array([0., r1])  # location of mode 4
c5 = np.sqrt(2)/2*np.array([-r1, -r1])  # location of mode 5
c6 = np.sqrt(2)/2*np.array([-r1, r1])  # location of mode 6
c7 = np.sqrt(2)/2*np.array([r1, -r1])  # location of mode 7
c8 = np.sqrt(2)/2*np.array([r1, r1])  # location of mode 8
c9 = np.array([0, 0])  # location of mode 9

cs = np.array([c1,c2,c3,c4,c5,c6,c7,c8,c9])
const = np.log(1. / np.sqrt(2. * np.pi * w**2))  # normalization constant
n_par = 2


@njit()
def gaussian(v, c):
    """Helper for log likelihood of a gaussian"""
    res = 0.
    for itrp in range(n_par):
        res += const-1/(2*w**2)*(v[itrp]-c[itrp])**2
    return res

@njit()
def loglike(v):
    """Get the likelihood for our wheel potential"""
    res = np.logaddexp(gaussian(v, c1), gaussian(v, c2))
    res = np.logaddexp(res, gaussian(v, c3))
    res = np.logaddexp(res, gaussian(v, c4))
    res = np.logaddexp(res, gaussian(v, c5))
    res = np.logaddexp(res, gaussian(v, c6))
    res = np.logaddexp(res, gaussian(v, c7))
    res = np.logaddexp(res, gaussian(v, c8))
    res = np.logaddexp(res, gaussian(v, c9))
    return res

@njit()
def prior_draw():
    """Get a prior draw"""
    return np.random.uniform(low_lim,high_lim,n_par)

@njit()
def correct_bounds(v):
    """Wrap into range"""
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

@njit()
def prior_factor(v):
    """Get a prior factor, if one is needed"""
    return 0.

@jitclass([('n_par',nb.int64)])
class Likelihood():
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self):
        """Create the class and store any object specific variables"""
        self.n_par = n_par

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

@njit()
def gen_draws(n_draws,n_par,attempt_lim=10000):
    """Get posterior draws"""
    draws = np.zeros((n_draws,n_par))
    for itrk in range(n_draws):
        itra = 0
        mode_choose = np.random.randint(0,9)
        draw_loc = cs[mode_choose]+np.random.normal(0.,w,2)
        while not check_bounds(draw_loc):
            if itra==attempt_lim:
                print('failed to find valid posterior point')
                assert False

            mode_choose = np.random.randint(0,9)
            draw_loc = cs[mode_choose]+np.random.normal(0.,w,2)

        draws[itrk] = draw_loc
    return draws

def get_labels():
    """Get useful labels for corner plots"""
    labels = []
    for itrp in range(2):
        labels.append(r'$v_'+str(itrp)+'$')
    return labels
