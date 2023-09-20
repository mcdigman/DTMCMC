"""C 2023 Matthew C. Digman
helpers for computing the temperature ladder for parallel tempering"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import cumtrapz


class TemperatureLadder():
    """store a temperature ladder for parallel tempering"""

    def __init__(self, n_chain, n_cold=1, T_cold=1., T_min=1., T_max=1.e15, Ts_in=None, use_inf_final=True):
        """create the temperature ladder object:
            inputs:
                n_chain: scalar integer, total number of parallel tempering chains
                n_cold: scalar integer<=n_chain, total number of T=T_cold chains
                T_cold: scalar float, temperature of 'cold' chain for readout, 1 by default
                T_min: scalar float, minimum temperature of temperature ladder, permitted to be less than T_cold for annealing
                T_max: scalar float, maximum temperature of finite temperature chains
                use_inf_final: scalar boolean, whether the last temperature should be infinity"""
        self.n_chain = n_chain
        self.n_cold = n_cold
        self.T_cold = T_cold
        self.T_min = T_min
        self.Ts_in = Ts_in
        self.T_max = T_max
        self.use_inf_final = use_inf_final
        if Ts_in is not None:
            self.Ts = Ts_in
            self.betas = np.zeros(n_chain)
            for itrj in range(0, n_chain):
                if np.isfinite(Ts_in[itrj]):
                    self.betas[itrj] = 1./Ts_in[itrj]
                else:
                    self.betas[itrj] = 0.

        else:
            self.betas, self.Ts = geometric_spaced_betas(n_chain, n_cold, T_cold, T_min, T_max, use_inf_final=use_inf_final)


def geometric_spaced_betas(n_chain, n_cold, T_cold, T_min, T_max, use_inf_final=True):
    """temeperatures spaced geometrically in a range
        inputs:
            n_chain: scalar integer, total number of chains
            n_cold: scalar integer, number of T=T_cold chains
                    (cold chains are separate from geometric ladder, unless T_cold=T_min
                     in which case 1 of the cold chains is considered part of the geometric ladder)
            T_cold: scalar float, temperature of 'cold' chains
            T_min: scalar float, minimum temperature of geometric ladder
            T_max: scalar float, maximum temperature of finite part of geometric ladder
            use_inf_final: scalar boolean, whether to include an infinite temperature chain separate from the geometric ladder"""
    betas = np.zeros(n_chain)
    Ts = np.zeros(n_chain)
    betas[:n_cold] = 1./T_cold

    if use_inf_final:
        # if T_cold==T_min then include geometric ladder is pinned to n_cold-1 element.
        # otherwise, ladder needs to be pinned to n_cold element, or it will not include an element at T_min
        if T_cold == T_min:
            n_geo = n_chain-n_cold
        else:
            n_geo = n_chain-n_cold-1
        beta_loc = 10**np.linspace(-np.log10(T_min), -np.log10(T_max), n_geo)
        if T_cold == T_min:
            betas[n_cold:n_chain-1] = beta_loc[1:]
        else:
            betas[n_cold:n_chain-1] = beta_loc

        Ts[n_cold:n_chain-1] = 1./betas[n_cold:n_chain-1]

        betas[-1] = 0.
        Ts[-1] = np.inf
    else:
        if T_cold == T_min:
            n_geo = n_chain-n_cold+1
        else:
            n_geo = n_chain-n_cold
        beta_loc = 10**np.linspace(-np.log10(T_min), -np.log10(T_max), n_geo)
        if T_cold == T_min:
            betas[n_cold:n_chain] = beta_loc[1:]
        else:
            betas[n_cold:n_chain] = beta_loc

        Ts[n_cold:] = 1./betas[n_cold:]

    Ts[:n_cold] = T_cold

    return betas, Ts


def entropy_spacing(n_chain_need, betas_in, logLs_in):
    """estimate constant entropy increase spaced chain from an input file of betas and logLs"""
    # as implemented, can't interpolate temperature ladder at non-finite Ts, so remove them
    logLs_in = logLs_in[betas_in > 0.]
    betas_in = betas_in[betas_in > 0.]

    Ts_in = 1./betas_in

    # need to sort the input temperatures and get only unique ones so we can interpolate
    Ts_use = np.unique(Ts_in)
    logLs_use = np.zeros(Ts_use.size)

    for itrf in range(0, Ts_use.size):
        # if there were duplicate temps, average the likelihoods
        logLs_use[itrf] = np.mean(logLs_in[Ts_use[itrf] == Ts_in])

    heat_capacities2 = np.abs(-np.gradient(logLs_use, Ts_use))
    heat_capacity_integ = cumtrapz(heat_capacities2/Ts_use, Ts_use, initial=0.)
    space_heat_need = heat_capacity_integ[Ts_use.size-1]/n_chain_need
    heat_grid_need = np.arange(0, n_chain_need)*space_heat_need
    T_grid_got = 10**InterpolatedUnivariateSpline(heat_capacity_integ, np.log10(Ts_use))(heat_grid_need)

    return T_grid_got

def entropy_spacing_var(n_chain_need, betas_in, logL_vars_in,correct_last=False,beta_low_mult=1.e-3):
    """estimate constant entropy increase spaced chain from an input file of betas and logLs"""
    # as implemented, can't interpolate temperature ladder at non-finite Ts, so remove them
    #logL_vars_in = logL_vars_in[betas_in > 0.]
    #betas_in = betas_in[betas_in > 0.]

    # need to sort the input temperatures and get only unique ones so we can interpolate
    betas_use = np.unique(betas_in)[::-1]
    betas_use[~np.isfinite(betas_use)] = 0.
    logL_vars_use = np.zeros(betas_use.size)

    for itrf in range(0, betas_use.size):
        # if there were duplicate temps, average the likelihood variances
        logL_vars_use[itrf] = np.mean(logL_vars_in[betas_use[itrf] == betas_in])

    heat_capacity_integrand = -np.abs(logL_vars_use)*betas_use
    heat_capacity_integ = cumtrapz(heat_capacity_integrand, betas_use, initial=0.)
    if correct_last and betas_use[-1]==0.:
        # this should be a more accurate approximation of the integrand from the last finitie temperature to infinite temperature,
        # assuming the last finite temperature was already sufficiently high
        heat_capacity_integ[-1] += betas_use[-2]**2/2*logL_vars_use[-1]

    #if betas_use[-1]==0.:
    #    # if last value is not finite temperature just set it much larger than the largest finite value so that interpolation will work
    #    betas_use[-1] = np.min(betas_use[np.isfinite(betas_use)&(betas_use!=0.)])*beta_low_mult

    space_heat_need = heat_capacity_integ[-1]/(n_chain_need-1)
    heat_grid_need = np.arange(0, n_chain_need)*space_heat_need
    #TODO would be better to do log interpolation in some cases, but not others
    beta_grid_got = InterpolatedUnivariateSpline(heat_capacity_integ, betas_use)(heat_grid_need)
    assert np.all(np.diff(beta_grid_got)<0.)
    T_grid_got = 1./beta_grid_got
    T_grid_got[beta_grid_got==0.] = np.inf

    return T_grid_got


def entropy_spacing_fromfile_var(n_chain_need, n_cold, T_file_in, logL_var_file_in,use_inf_final=True):
    """estimate constant entropy increase spaced chain from an input file of betas and logLs"""
    Ts_in = np.load(T_file_in)
    logL_vars_in = np.load(logL_var_file_in)
    betas_in = 1./Ts_in
    betas_in[~np.isfinite(Ts_in)] = 0.
    n_chain_space = n_chain_need-n_cold+1
    Ts_got = entropy_spacing_var(n_chain_space, betas_in, logL_vars_in)
    Ts_got[0] = 1.
    if n_cold>1:
        Ts_got = np.hstack([np.full(n_cold-1,1.),Ts_got])
    if use_inf_final:
        Ts_got[-1] = np.inf
    assert Ts_got.size==n_chain_need
    return Ts_got

def entropy_spacing_fromfile(n_chain_need, n_cold, T_file_in, logL_file_in,use_inf_final=True):
    """estimate constant entropy increase spaced chain from an input file of betas and logLs"""
    Ts_in = np.load(T_file_in)
    logLs_in = np.load(logL_file_in)
    betas_in = 1./Ts_in
    betas_in[~np.isfinite(Ts_in)] = 0.
    if use_inf_final:
        n_chain_space =  n_chain_need-n_cold
    else:
        n_chain_space = n_chain_need-n_cold+1
    Ts_got = entropy_spacing(n_chain_space, betas_in, logLs_in)
    Ts_got[0] = 1.
    if n_cold>1:
        Ts_got = np.hstack([np.full(n_cold-1,1.),Ts_got])
    if use_inf_final:
        Ts_got = np.hstack([Ts_got,np.inf])
    assert Ts_got.size==n_chain_need
    return Ts_got
