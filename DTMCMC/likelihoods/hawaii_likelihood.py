"""an n dimensional normal distribution"""

from typing import TYPE_CHECKING, override

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from DTMCMC.likelihood import RectangularLikelihood

if TYPE_CHECKING:
    from numpy.typing import NDArray


class HawaiiLikelihood(RectangularLikelihood):
    """class to manage the likelihood-specific essential functions for the sampler

    The scipy interpolator cannot compile to nopython, so this class
    overrides ``get_loglike`` directly and the sampler routes it through
    the Python kernel path.
    """

    def __init__(self, rescale_like: float = 1.0, default_like: float = 5.0e-1, normalize_like: bool = True) -> None:
        """Create the class and store any object specific variables"""
        self.rescale_like = rescale_like
        self.default_like = default_like
        self.normalize_like = normalize_like

        n_par = 2

        hf_out = h5py.File('data/hawaii_map.hdf5', 'r')

        # an input map of hawaii with elevations in meters
        hawaii_map = hf_out['map']
        if not isinstance(hawaii_map, h5py.Group):
            msg = 'expected "map" in hawaii_map.hdf5 to be an HDF5 group'
            raise TypeError(msg)

        hawaii_dataset = hawaii_map['hawaii']
        if not isinstance(hawaii_dataset, h5py.Dataset):
            msg = 'expected "map/hawaii" in hawaii_map.hdf5 to be an HDF5 dataset'
            raise TypeError(msg)

        self.hawaii_grid = np.asarray(hawaii_dataset, dtype=np.float64)

        # rescale the input map as requested
        self.hawaii_grid *= rescale_like

        self.hawaii_grid[self.hawaii_grid <= default_like] = default_like
        self.hawaii_grid /= np.sum(self.hawaii_grid)

        self.log_hawaii_grid = np.log(self.hawaii_grid)

        self.xs_grid = np.linspace(-1.0, 1.0, self.hawaii_grid.shape[0])
        self.ys_grid = (
            np.linspace(-1.0, 1.0, self.hawaii_grid.shape[1]) * self.hawaii_grid.shape[1] / self.hawaii_grid.shape[0]
        )

        self.hawaii_interp = RegularGridInterpolator(
            (self.xs_grid, self.ys_grid), self.log_hawaii_grid, method='linear'
        )

        low_lims = np.array([self.xs_grid.min(), self.ys_grid.min()])
        high_lims = np.array([self.xs_grid.max(), self.ys_grid.max()])

        RectangularLikelihood.__init__(self, n_par, low_lims, high_lims)

    @override
    def get_loglike(self, params_in: NDArray[np.floating]) -> float:
        """Get the log likelihood given a set of parameters v"""
        res: float = self.hawaii_interp(params_in)[0]
        return res
