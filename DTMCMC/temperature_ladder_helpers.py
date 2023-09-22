"""C 2023 Matthew C. Digman
helpers for computing the temperature ladder for parallel tempering"""

from warnings import warn
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
                use_inf_final: scalar boolean, whether the last temperature should be infinity
                Ts_in: array float, override everything else and replace Ts with this if it is not None"""
        self.n_chain = n_chain
        self.n_cold = n_cold
        self.T_cold = T_cold
        self.T_min = T_min
        self.Ts_in = Ts_in
        self.T_max = T_max
        self.use_inf_final = use_inf_final
        if Ts_in is not None:
            assert Ts_in.size == self.n_chain
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

    if n_cold > n_chain:
        raise ValueError('n cold cannot be more than total number of chains')

    assert T_min>0. and np.isfinite(T_min)
    assert T_max>0. and np.isfinite(T_max)
    assert T_cold>0. and np.isfinite(T_cold)
    assert T_max>T_min

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
        if n_geo<=0:
            warn("all chains are cold, infinite temperature chain will be overwritten")
        else:
            beta_loc = 10**np.linspace(-np.log10(T_min), -np.log10(T_max), n_geo)
            if T_cold == T_min:
                betas[n_cold:n_chain-1] = beta_loc[1:]
            else:
                betas[n_cold:n_chain-1] = beta_loc

            Ts[n_cold:n_chain-1] = 1./betas[n_cold:n_chain-1]
            betas[n_cold:n_chain-1] = 1./Ts[n_cold:n_chain-1]  # recalculate this way for internal test consistency

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
        betas[n_cold:] = 1./Ts[n_cold:]  # recalculate this way for internal test consistency

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

def standardize_input_vars(betas_in,logL_vars_in):
    """standardize the input betas and variances"""
    # need to sort the input temperatures and get only unique ones so we can interpolate
    betas_use = np.unique(betas_in)[::-1]
    betas_use[~np.isfinite(betas_use)] = 0.
    logL_vars_use = np.zeros(betas_use.size)

    for itrf in range(0, betas_use.size):
        # if there were duplicate temps, average the likelihood variances
        logL_vars_use[itrf] = np.mean(logL_vars_in[betas_use[itrf] == betas_in])

    return betas_use,logL_vars_use

def entropy_spacing_var(n_chain_need, betas_in, logL_vars_in,correct_last=False):
    """estimate constant entropy increase spaced chain from an input file of betas and logLs"""
    assert n_chain_need > 0

    betas_use,logL_vars_use = standardize_input_vars(betas_in,logL_vars_in)
    heat_capacity_integ = get_heat_capacity_integrated(logL_vars_use,betas_use,correct_last)

    if n_chain_need == 1:
        # unsure what to do in this case, but don't divide by zero
        space_heat_need = heat_capacity_integ[-1]
    else:
        space_heat_need = heat_capacity_integ[-1]/(n_chain_need-1)
    heat_grid_need = np.arange(0, n_chain_need)*space_heat_need

    #TODO would be better to do log interpolation in some cases, but not others
    beta_grid_got = InterpolatedUnivariateSpline(heat_capacity_integ, betas_use)(heat_grid_need)
    beta_grid_got[beta_grid_got<0.] = 0.
    beta_grid_got = np.sort(beta_grid_got)[::-1]
    #assert np.all(np.diff(beta_grid_got)<0.)

    # Avoid dividing by zero
    T_grid_got = np.zeros(beta_grid_got.size)
    T_grid_got[beta_grid_got!=0.] = 1./beta_grid_got[beta_grid_got!=0.]
    T_grid_got[beta_grid_got==0.] = np.inf

    return T_grid_got

def get_heat_capacity_integrated(logL_vars_use,betas_use,correct_last):
    """get the integral of the heat capacity"""
    heat_capacity_integrand = -np.abs(logL_vars_use)*betas_use
    heat_capacity_integ = cumtrapz(heat_capacity_integrand, betas_use, initial=0.)
    if correct_last and betas_use[-1]==0.:
        # this should be a more accurate approximation of the integrand from the last finitie temperature to infinite temperature,
        # assuming the last finite temperature was already sufficiently high
        heat_capacity_integ[-1] += betas_use[-2]**2/2*logL_vars_use[-1]

    return heat_capacity_integ




def entropy_spacing_fromfile_var(n_chain_need, n_cold, T_file_in, logL_var_file_in,use_inf_final=True,T_cold=1.,correct_last=False):
    """estimate constant entropy increase spaced chain from an input file of betas and logLs"""
    if n_cold > n_chain_need:
        raise ValueError('cannot have more cold chains than total chains')

    Ts_in = np.load(T_file_in)
    logL_vars_in = np.load(logL_var_file_in)

    assert np.all(logL_vars_in >= 0.)
    assert np.all(Ts_in >= 0.)

    betas_in = 1./Ts_in
    betas_in[~np.isfinite(Ts_in)] = 0.
    n_chain_space = n_chain_need-n_cold+1
    Ts_got = entropy_spacing_var(n_chain_space, betas_in, logL_vars_in,correct_last=correct_last)

    if use_inf_final:
        Ts_got[-1] = np.inf
        if n_chain_need == n_cold:
            warn("all chains are cold, infinite temperature chain will be overwritten")

    Ts_got[0] = T_cold
    if n_cold>1:
        Ts_got = np.hstack([np.full(n_cold-1,T_cold),Ts_got])
    assert Ts_got.size==n_chain_need
    return Ts_got

def find_potential_phase_transitions(betas_in,logL_vars_in,correct_last=True,n_chain_need=2048):
    """find the best estimates for temperatures of potential phase transitions by interpolating the integrated heat capacity"""
    maxima = []
    minima = []

    #get a spacing that is predicted to be good
    Ts_got = entropy_spacing_var(n_chain_need, betas_in, logL_vars_in,correct_last=correct_last)

    betas_got = np.zeros(n_chain_need)
    betas_got[(Ts_got>0.)&(np.isfinite(Ts_got))] = 1./Ts_got[(Ts_got>0.)&(np.isfinite(Ts_got))]
    betas_got[Ts_got==0.] = np.inf
    betas_got[~np.isfinite(Ts_got)] = 0.

    #get the integrated heat capacity
    betas_use,logL_vars_use = standardize_input_vars(betas_in,logL_vars_in)
    heat_capacity_integ = get_heat_capacity_integrated(logL_vars_use,betas_use,correct_last)

    #Interpolate the integrate heat capacity and get the derivative
    heat_capacity_got = -InterpolatedUnivariateSpline(betas_use[::-1],heat_capacity_integ[::-1]).derivative(1)(betas_got)*betas_got
    #import matplotlib.pyplot as plt

    #find local maxima that may represent a phase transition
    itrt_last = 0
    itrt = 1

    while heat_capacity_got[itrt_last] == heat_capacity_got[itrt] and itrt<n_chain_need-2:
        # while loops instead of for loops to handle the unlikely case where some heat capacities are exactly equal
        itrt = itrt + 1

    itrt_next = itrt+1

    # handle starting boundary
    if heat_capacity_got[0] <  heat_capacity_got[itrt]:
        minima.append(0)
    elif heat_capacity_got[0] >  heat_capacity_got[itrt]:
        maxima.append(0)

    while itrt_next < n_chain_need:
        # while loops instead of for loops to handle the unlikely case where some heat capacities are exactly equal

        while heat_capacity_got[itrt] == heat_capacity_got[itrt_next] and itrt_next<n_chain_need-1:
            itrt_next = itrt_next + 1

        if heat_capacity_got[itrt_last]<heat_capacity_got[itrt] and heat_capacity_got[itrt_next]<=heat_capacity_got[itrt]:
            maxima.append(itrt)
            #plt.scatter(betas_got[itrt],heat_capacity_got[itrt])

        elif heat_capacity_got[itrt_last]>heat_capacity_got[itrt] and heat_capacity_got[itrt_next]>=heat_capacity_got[itrt]:
            minima.append(itrt)
            #plt.scatter(betas_got[itrt],heat_capacity_got[itrt])

        itrt_last = itrt
        itrt = itrt_next
        itrt_next = itrt+1

    # handle ending boundary 
    if heat_capacity_got[itrt_last] >  heat_capacity_got[-1]:
        minima.append(n_chain_need-1)
    elif heat_capacity_got[itrt_last] <  heat_capacity_got[-1]:
        maxima.append(n_chain_need-1)

    minima = np.array(minima)
    maxima = np.array(maxima)

    minima_vals = heat_capacity_got[minima]
    maxima_vals = heat_capacity_got[maxima]

    minima_Ts = Ts_got[minima]
    maxima_Ts = Ts_got[maxima]

    #default end values 
    maxima_Ts[maxima==0] = 1./betas_use[0]
    maxima_Ts[maxima==n_chain_need-1] = 0.
    
    # calculate the prominence of each maxima, in the same sense as topographic prominence
    # prominence is difference between maxima and key col, where key col is lowest point between that maxima and a higher maxima
    prominences = np.zeros(maxima.size)
    for itrp,itrt in enumerate(maxima):
        cur_max_val = maxima_vals[itrp]

        key_col1 = 0.
        key_col2 = 0.
        
        val_last = 0.
        val_next = 0.

        
        itrt_last = 0
        itrt_next = n_chain_need-1

        itrp_last = itrp-1
        if itrp_last >= 0:
            while maxima_vals[itrp_last] < cur_max_val:
                itrp_last -= 1
                if itrp_last<0:
                    break

        if itrp_last >= 0:
            itrt_last = maxima[itrp_last]
            val_last = maxima_vals[itrp_last]

            if np.any((minima>=itrt_last)&(minima<=itrt)):
                key_col1 = np.min(minima_vals[(minima>=itrt_last)&(minima<=itrt)])
            else:
                key_col1 = cur_max_val
        else:
            key_col1 = heat_capacity_got[0]

        itrp_next = itrp+1
        if itrp_next <= maxima.size-1:
            while maxima_vals[itrp_next] < cur_max_val:
                itrp_next += 1
                if itrp_next>=maxima.size:
                    break

        if itrp_next < maxima.size:
            itrt_next = maxima[itrp_next]
            val_next = maxima_vals[itrp_next]

            if np.any((minima<=itrt_next)&(minima>=itrt)):
                key_col2 = np.min(minima_vals[(minima<=itrt_next)&(minima>=itrt)])
            else:
                key_col2 = cur_max_val
        else:
            key_col2 = 0.
        
        key_col = max(key_col1,key_col2)

        prominences[itrp] = cur_max_val - key_col

#    print(minima)
#    print(maxima)
#    print(heat_capacity_got[minima])
#    print(heat_capacity_got[maxima])
#    print(prominences)
#
#    plt.plot(betas_got,heat_capacity_got)
#    #plt.plot(betas_use,-np.gradient(heat_capacity_integ,betas_use)*betas_use)
#    plt.plot(betas_use,logL_vars_use*betas_use**2)
#    plt.show()


    return maxima_Ts,maxima_vals,prominences





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
