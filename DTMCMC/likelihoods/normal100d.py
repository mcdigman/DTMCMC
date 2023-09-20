"""an n dimensional normal distribution"""
import numpy as np
import numba as nb
from numba.experimental import jitclass
from numba import njit
from DTMCMC.correction_helpers import reflect_into_range

#100d unit normal motivated by https://statmodeling.stat.columbia.edu/2017/03/15/ensemble-methods-doomed-fail-high-dimensions/

#global constants
low_lim = -5
high_lim = 5
const = np.log(1. / np.sqrt(2. * np.pi))  # normalization constant
eps_default = 1.e-4

@jitclass([('n_par',nb.int64),('epsilons',nb.float64[:])])
class Likelihood():
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self,n_par=100):
        """create the class and store any object specific variables"""
        self.n_par = n_par
        self.epsilons = np.zeros(n_par)+eps_default

    def get_loglike(self,v):
        """get the log likelihood given a set of parameters v"""
        return get_loglike(v)

    def get_snr(self,v):
        """get the snr given a set of parameters v"""
        #TODO check if this is needed
        return get_loglike(v)

    def prior_draw(self):
        """get a draw from the prior"""
        return prior_draw(self.n_par)

    def prior_proposal(self,v_in):
        """get a proposal from the prior"""
        v_out = prior_draw(self.n_par)
        return v_out,prior_factor(v_in,self.n_par)-prior_factor(v_out,self.n_par),True

    def prior_factor(self,v):
        """get the density factor for prior draws, if the prior draws are not uniform"""
        return prior_factor(v,self.n_par)

    def correct_bounds(self,v):
        """correct the bounds of a draw to be in range, if allowed for this likelihood"""
        return correct_bounds(v)

    def check_bounds(self,v):
        """check if the bounds of a draw are in the prior range but do not change them"""
        return check_bounds(v)

@njit()
def get_loglike(v):
    """get an n dimensional gaussian likelihood"""
    res = 0.
    for itrp in range(0,v.shape[0]):
        res += const-1/2*v[itrp]**2
    return res

@njit()
def prior_draw(n_par):
    """get a prior draw"""
    draw = np.zeros(n_par)
    for itrp in range(0,n_par):
        draw[itrp] = np.random.uniform(low_lim,high_lim)
    return draw

@njit()
def prior_factor(v,n_par):
    """prior draw density factor, if we need one"""
    return 0.

@njit()
def correct_bounds(v):
    """wrap parameters into range"""
    for itrp in range(0,v.size):
        v[itrp] = reflect_into_range(v[itrp],low_lim,high_lim)
    return v

@njit()
def check_bounds(v):
    """check if a sample is within the prior range"""
    for itrp in range(v.size):
        if not low_lim<v[itrp]<high_lim:
            return False
    return True


@njit()
def gen_draws(n_draws,n_par,attempt_lim=10000):
    """get posterior draws"""
    draws = np.zeros((n_draws,n_par))
    for itrk in range(n_draws):
        itra = 0
        draw_loc = np.random.normal(0.,1,n_par)
        while not check_bounds(draw_loc):
            if itra==attempt_lim:
                print('failed to find valid posterior point')
                assert False

            draw_loc = np.random.normal(0.,1,n_par)
        draws[itrk] = draw_loc
    return draws

def get_labels(n_par):
    """get useful labels for corner plots"""
    labels = []
    for itrp in range(n_par):
        labels.append(r"$v_"+str(itrp)+"$")
    return labels

def format_samples_output(samples, params_fid):
    labels_loc = get_labels(params_fid.size)
    return samples.copy(), params_fid.copy(), labels_loc
