"""the banana likelihood in n dimensions"""
import numpy as np
from numba import njit

from DTMCMC.likelihood import RectangularLikelihood

#constants
#limits for first two parameters
low_lim01 = -10000.
high_lim01 = 10000.
# limits for n>=2 parameters
low_limn = -100.
high_limn = 100.
#see https://link.springer.com/content/pdf/10.1007/s001800050022.pdf
#and https://www.tandfonline.com/doi/pdf/10.1198/jcgs.2009.06134?needAccess=true
#20d banana is default
B = 0.1 #bananacity parameter


@njit()
def get_loglike(v,n_par):
    """Get the log likelihood for the 'banana' test case"""
    res = -v[0]**2/200-1/2*(v[1]+B*v[0]**2-100*B)**2
    for itrp in range(2,n_par):
        res += -1/2*v[itrp]**2
    return res


class BananaLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self,n_par=20) -> None:
        """Create the class and store any object specific variables"""
        assert n_par >= 2
        low_lims = np.full(n_par, low_limn)
        high_lims = np.full(n_par, high_limn)
        low_lims[:2] = low_lim01
        high_lims[:2] = high_lim01

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    def get_loglike(self,params_in):
        """Get the log likelihood given a set of parameters v"""
        return get_loglike(params_in,self.n_par)
