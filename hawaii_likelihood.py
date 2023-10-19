"""an n dimensional normal distribution"""
import numpy as np
import numba as nb

from numba.experimental import jitclass
from numba import njit

from scipy.interpolate import RegularGridInterpolator

import h5py

from DTMCMC.correction_helpers import reflect_into_range
from DTMCMC.likelihood import RectangularLikelihood

class HawaiiLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler"""
    def __init__(self,rescale_like=1.,default_like=5.e-1,normalize_like=True):
        """create the class and store any object specific variables"""

        self.rescale_like = rescale_like 
        self.default_like = default_like
        self.normalize_like = normalize_like

        n_par = 2

        hf_out = h5py.File('hawaii_map.hdf5','r')

        # an input map of hawaii with elevations in meters
        self.hawaii_grid = np.asarray(hf_out['map']['hawaii'],dtype=np.float64)

        # rescale the input map as requested
        self.hawaii_grid *= rescale_like

        self.hawaii_grid[self.hawaii_grid<=default_like] = default_like
        self.hawaii_grid /= np.sum(self.hawaii_grid) 
        
        self.log_hawaii_grid = np.log(self.hawaii_grid)

        self.xs_grid = np.linspace(-1.,1.,self.hawaii_grid.shape[0])
        self.ys_grid = np.linspace(-1.,1.,self.hawaii_grid.shape[1])*self.hawaii_grid.shape[1]/self.hawaii_grid.shape[0]

        self.hawaii_interp = RegularGridInterpolator((self.xs_grid,self.ys_grid),self.log_hawaii_grid,method='linear')

        low_lims = np.array([self.xs_grid.min(),self.ys_grid.min()])
        high_lims = np.array([self.xs_grid.max(),self.ys_grid.max()])

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    def get_loglike(self,v):
        """get the log likelihood given a set of parameters v"""
        return self.hawaii_interp(v)[0]

