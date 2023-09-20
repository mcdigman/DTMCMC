"""C 2023 Matthew C. Digman
helpers for correcting parameters"""
from numba import njit
import numpy as np


@njit()
def reflect_into_range(x, x_low, x_high):
    """reflect an arbitrary parameter into a nominal range"""
    # ensure always returns something in range (i.e. do an arbitrary number of reflections) similar to reflect_cosines but does not need to track angles
    x_range = x_high-x_low
    res = x
    if res < x_low:
        res = x_low+(-(res-x_low)) % (2*x_range)      # 2*x_low - x
    if res > x_high:
        res = x_high-(res-x_high) % (2*x_range)       # 2*x_high - x
        if res < x_low:
            res = x_low+(-(res-x_low)) % (2*x_range)  # 2*x_low - x
    return res


@njit()
def reflect_cosines(cos_in, angle_in, rotfac=np.pi, modfac=2*np.pi):
    """helper to reflect cosines of coordinates around poles  to get them between -1 and 1,
        which requires also rotating the signal by rotfac each time, then mod the angle by modfac"""
    # reflect at poles, requires shifting corresponding azimuth as well
    # needs to be mod 4 instead of 2 because a very large jump could have gotten reflected even or odd number of times
    # if so, params_in[1] is guaranteed to be between 1 and 3 so next step will correct if that happened
    if cos_in < -1.:
        cos_in = -1.+(-(cos_in+1.)) % 4
        angle_in += rotfac
        # if this reflects even number of times, params_in[1] after is guaranteed to be between -1 and -3, so one more correction attempt will suffice
    if cos_in > 1.:
        cos_in = 1.-(cos_in-1.) % 4
        angle_in += rotfac
    if cos_in < -1.:
        cos_in = -1.+(-(cos_in+1.)) % 4
        angle_in += rotfac
    angle_in = angle_in % modfac
    return cos_in, angle_in
