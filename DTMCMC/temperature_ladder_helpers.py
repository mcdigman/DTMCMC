"""C 2023 Matthew C. Digman
helpers for computing the temperature ladder for parallel tempering"""

from warnings import warn
import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import cumtrapz


class TemperatureLadder():
    """store a temperature ladder for parallel tempering"""

    def __init__(self, n_cold, Ts_in):
        """create the temperature ladder object:
            inputs:
                n_cold: scalar integer<=n_chain, total number of T=T_cold chains
                Ts_in: array float, override everything else and replace Ts with
                    this if it is not None"""

        self.Ts = Ts_in
        self.betas = Ts_to_betas(self.Ts)

        self.n_chain = Ts_in.size
        self.n_cold = n_cold


class GeometricTemperatureLadder(TemperatureLadder):
    """store a geometrically spaced temperature ladder for parallel tempering"""

    def __init__(self, n_chain, n_cold=1, T_cold=1., T_min=1., T_max=1.e15, use_inf_final=True):
        """create the temperature ladder object:
            inputs:
                n_chain: scalar integer, total number of parallel tempering chains
                n_cold: scalar integer<=n_chain, total number of T=T_cold chains
                T_cold: scalar float, temperature of 'cold' chain for readout, 1 by default
                T_min: scalar float, minimum temperature of temperature ladder,
                        permitted to be less than T_cold for annealing
                T_max: scalar float, maximum temperature of finite temperature chains
                use_inf_final: scalar boolean, whether the last temperature should be infinity"""

        self.T_cold = T_cold
        self.T_min = T_min
        self.T_max = T_max
        self.use_inf_final = use_inf_final

        _, Ts = geometric_spaced_betas(
            n_chain, n_cold, T_cold, T_min, T_max, use_inf_final=use_inf_final
        )
        TemperatureLadder.__init__(self, n_cold, Ts)


class EntropyTemperatureLadder(TemperatureLadder):
    """store a constant entropy increase spaced temperature ladder for parallel tempering"""

    def __init__(
            self,
            n_chain,
            Ts_in,
            logL_vars_in,
            n_cold=1,
            T_cold=1.,
            use_inf_final=True,
            correct_last=False
    ):
        """create the temperature ladder object:
            inputs:
                n_chain: scalar integer, total number of parallel tempering chains
                Ts_in: float array of input temperatures to use for building heat capacity
                logL_vars_in: float array of input variances to use for building heat capacity
                n_cold: scalar integer<=n_chain, total number of T=T_cold chains
                T_cold: scalar float, temperature of 'cold' chain for readout, 1 by default
                use_inf_final: scalar boolean, whether the last temperature should be infinity
                correct_last: scalar boolean, whether to use a taylor series estimate to
                    extrapolate the heat capacity integral out to infinity; only works well
                    if maximum T in input ladder is well above any phase transitions"""

        self.T_cold = T_cold
        self.use_inf_final = use_inf_final

        _, Ts = entropy_spaced_betas(
            n_chain,
            n_cold,
            Ts_in,
            logL_vars_in,
            use_inf_final=use_inf_final,
            T_cold=T_cold,
            correct_last=correct_last,
        )

        TemperatureLadder.__init__(self, n_cold, Ts)


def geometric_spaced_betas(n_chain, n_cold, T_cold, T_min, T_max, use_inf_final=True):
    """temperatures spaced geometrically in a range
        inputs:
            n_chain: scalar integer, total number of chains
            n_cold: scalar integer, number of T=T_cold chains
                    (cold chains are separate from geometric ladder, unless T_cold=T_min
                     in which case 1 of the cold chains is considered part of the geometric ladder)
            T_cold: scalar float, temperature of 'cold' chains
            T_min: scalar float, minimum temperature of geometric ladder
            T_max: scalar float, maximum temperature of finite part of geometric ladder
            use_inf_final: scalar boolean, whether to include an infinite temperature chain
                separate from the geometric ladder"""

    if n_cold > n_chain:
        raise ValueError('n cold cannot be more than total number of chains')

    assert T_min > 0. and np.isfinite(T_min)
    assert T_max > 0. and np.isfinite(T_max)
    assert T_cold > 0. and np.isfinite(T_cold)
    assert T_max > T_min
    assert n_cold >= 0
    assert n_chain > 0

    betas = np.zeros(n_chain)
    Ts = np.zeros(n_chain)
    betas[:n_cold] = 1./T_cold

    if use_inf_final:
        # if T_cold==T_min then include geometric ladder is pinned to n_cold-1 element.
        # otherwise, ladder needs to be pinned to n_cold element,
        # or it will not include an element at T_min
        if n_chain == n_cold:
            warn("all chains are cold, infinite temperature chain will be overwritten")
        else:
            if T_cold == T_min and n_cold != 0:
                n_geo = n_chain-n_cold
            else:
                n_geo = n_chain-n_cold-1

            beta_loc = 10**np.linspace(-np.log10(T_min), -np.log10(T_max), n_geo)
            if T_cold == T_min and n_cold != 0:
                betas[n_cold:n_chain-1] = beta_loc[1:]
            else:
                betas[n_cold:n_chain-1] = beta_loc

            Ts[n_cold:n_chain-1] = betas_to_Ts(betas[n_cold:n_chain-1])
            # recalculate beta this way for internal test consistency
            betas[n_cold:n_chain-1] = Ts_to_betas(Ts[n_cold:n_chain-1])

            betas[-1] = 0.
            Ts[-1] = np.inf

    else:
        if T_cold == T_min and n_cold != 0:
            n_geo = n_chain-n_cold+1
        else:
            n_geo = n_chain-n_cold
        beta_loc = 10**np.linspace(-np.log10(T_min), -np.log10(T_max), n_geo)
        if T_cold == T_min and n_cold != 0:
            betas[n_cold:n_chain] = beta_loc[1:]
        else:
            betas[n_cold:n_chain] = beta_loc

        Ts[n_cold:] = betas_to_Ts(betas[n_cold:])
        # recalculate beta this way for internal test consistency
        betas[n_cold:] = Ts_to_betas(Ts[n_cold:])

    Ts[:n_cold] = T_cold

    return betas, Ts


def entropy_spaced_betas(
    n_chain_need,
    n_cold,
    Ts_in,
    logL_vars_in,
    use_inf_final=True,
    T_cold=1.,
    correct_last=False,
):
    """estimate constant entropy increase spaced chain from an input file of betas and logLs"""
    if n_cold > n_chain_need:
        raise ValueError('cannot have more cold chains than total chains')

    assert T_cold >= 0.
    assert np.all(logL_vars_in >= 0.)
    assert np.all(Ts_in >= 0.)
    assert Ts_in.size == logL_vars_in.size
    assert n_cold >= 0
    assert n_chain_need > 0

    betas_in = Ts_to_betas(Ts_in)

    if n_cold == 0:
        n_chain_space = n_chain_need
    else:
        n_chain_space = n_chain_need-n_cold+1

    Ts_got = entropy_spacing(n_chain_space, betas_in, logL_vars_in, correct_last=correct_last)

    assert np.all(Ts_got >= 0.)

    if use_inf_final:
        Ts_got[-1] = np.inf

        if n_chain_need == n_cold:
            warn("all chains are cold, infinite temperature chain will be overwritten")

    # TODO add option to do include cold spacing adaptively or not
    if n_cold > 0:
        Ts_got[0] = T_cold

    if n_cold > 1:
        Ts_got = np.hstack([np.full(n_cold-1, T_cold), Ts_got])

    assert Ts_got.size == n_chain_need

    betas_got = Ts_to_betas(Ts_got)

    return betas_got, Ts_got


def standardize_input_vars(betas_in, logL_vars_in):
    """helper to standardize the input betas and variances
    so that the heat capacity integration can work correctly"""
    # need to sort the input temperatures and get only unique ones so we can interpolate
    betas_use = np.unique(betas_in)[::-1]
    # Note that using partition functions the lowest order taylor approximation
    # of the heat capacity integral from T=zero to T1 is 0,
    # And we cannot compute the next order correction knowing only
    # the variances of the log likelihoods,
    # So the simplest thing to do is to just cut out any zero temperature (infinite beta)
    # chains from the heat capacity integrals
    betas_use = betas_use[np.isfinite(betas_use)]

    logL_vars_use = np.zeros(betas_use.size)

    for itrf in range(0, betas_use.size):
        # if there were duplicate temps, average the likelihood variances
        logL_vars_use[itrf] = np.mean(logL_vars_in[betas_use[itrf] == betas_in])

    if np.any(~np.isfinite(logL_vars_use)):
        # Handle non-finite variance just by excising those points
        warn('Nonfinite variance requested, results may not be meaningful')
        betas_use = betas_use[np.isfinite(logL_vars_use)]
        logL_vars_use = logL_vars_use[np.isfinite(logL_vars_use)]

    return betas_use, logL_vars_use


def entropy_spacing(n_chain_need, betas_in, logL_vars_in, correct_last=False):
    """helper to estimate constant entropy increase spaced chain
    from an input file of betas and logLs"""
    assert n_chain_need > 0
    assert betas_in.size > 0
    assert betas_in.size == logL_vars_in.size

    betas_use, logL_vars_use = standardize_input_vars(betas_in, logL_vars_in)

    if betas_use.size == 0:
        # Somehow there are no valid betas; there is really nothing we can do to recover
        raise ValueError('No valid points available to construct ladder')

    assert betas_use.size > 0

    heat_capacity_integ = get_heat_capacity_integrated(logL_vars_use, betas_use, correct_last)

    if n_chain_need == 1:
        # unsure what to do in this case, but don't divide by zero
        space_heat_need = heat_capacity_integ[-1]
    else:
        space_heat_need = heat_capacity_integ[-1]/(n_chain_need-1)

    heat_grid_need = np.arange(0, n_chain_need)*space_heat_need

    # TODO cubic splines or log interpolation might work better in some cases,
    # but we would need to enforce monotonicity on the splines
    # If the splines are non-monotonic it could produce nonsense negative temperatures
    if betas_use.size == 1:
        beta_grid_got = np.full(n_chain_need, betas_use[0])
        warn('Only one unique input temperature: cannot generate a meaningful grid')
    else:
        beta_interp = InterpolatedUnivariateSpline(heat_capacity_integ, betas_use, k=1, ext=3)
        beta_grid_got = beta_interp(heat_grid_need)

    if np.any(beta_grid_got < 0.):
        warn('Unexpected negative temperatures: defaulting to abs')
        beta_grid_got[beta_grid_got < 0.] = np.abs(beta_grid_got[beta_grid_got < 0.])
        beta_grid_got = np.sort(beta_grid_got)[::-1]

    if not np.all(np.diff(beta_grid_got) <= 0.):
        warn('Temperature grid is not sorted correctly')
        beta_grid_got = np.sort(beta_grid_got)[::-1]

    assert np.all(beta_grid_got >= 0.)
    assert np.all(np.diff(beta_grid_got) <= 0.)

    # Avoid dividing by zero
    T_grid_got = betas_to_Ts(beta_grid_got)

    return T_grid_got


def get_heat_capacity_integrated(logL_vars_use, betas_use, correct_last):
    """helper to get the integral of the heat capacity"""
    heat_capacity_integrand = -np.abs(logL_vars_use)*betas_use
    # cannot handle non finite beta case correctly
    heat_capacity_integrand[~np.isfinite(heat_capacity_integrand)] = 0.

    heat_capacity_integ = cumtrapz(heat_capacity_integrand, betas_use, initial=0.)

    if correct_last and betas_use[-1] == 0. and betas_use.size > 1:
        # this should be a more accurate approximation of the integrand
        # from the last finite temperature to infinite temperature,
        # assuming the last finite temperature was already sufficiently high
        heat_capacity_integ[-1] += betas_use[-2]**2/2*logL_vars_use[-1]

    # We need to enforce that the heat capacity integral is strictly increasing
    # Integral is strictly increasing if we add tiny increments
    # each time we see two identical values

    # Handle if first value is zero
    if heat_capacity_integ[0] == 0.:
        if np.any(heat_capacity_integ > 0.):
            heat_capacity_integ[0] = 1.e-14*np.min(heat_capacity_integ)
        else:
            heat_capacity_integ[0] = 1.e-15

    for itrn in range(1, heat_capacity_integ.size):
        if heat_capacity_integ[itrn] == heat_capacity_integ[itrn-1]:
            if heat_capacity_integ[itrn-1] == 0.:
                heat_capacity_integ[itrn:] += 1.e-15
            else:
                heat_capacity_integ[itrn:] += 1.e-14*heat_capacity_integ[itrn-1]

    assert np.all(np.diff(heat_capacity_integ) > 0.)

    return heat_capacity_integ


def Ts_to_betas(Ts_in):
    """convert Ts to betas=1/Ts safely handling infinities and zeros:
            input:
                Ts_in: float array, temperatures
            output:
                betas_got: float array, 1/temperatures"""

    assert np.all(Ts_in >= 0.)
    betas_got = np.zeros(Ts_in.size)
    Ts_finite = np.isfinite(Ts_in) & (Ts_in != 0.)
    betas_got[Ts_finite] = 1./Ts_in[Ts_finite]
    betas_got[~np.isfinite(Ts_in)] = 0.
    betas_got[Ts_in == 0.] = np.inf
    return betas_got


def betas_to_Ts(betas_in):
    """convert Ts to Ts=1/betas safely handling infinities and zeros:
            input:
                betas_in: float array, 1/temperatures
            output:
                Ts_got: float array, temperatures"""

    assert np.all(betas_in >= 0.)
    Ts_got = np.zeros(betas_in.size)
    beta_finite = np.isfinite(betas_in) & (betas_in != 0.)
    Ts_got[beta_finite] = 1./betas_in[beta_finite]
    Ts_got[~np.isfinite(betas_in)] = 0.
    Ts_got[betas_in == 0.] = np.inf
    return Ts_got


def entropy_ladder_fromfile(
    n_chain_need,
    n_cold,
    T_file_in,
    logL_var_file_in,
    use_inf_final=True,
    T_cold=1.,
    correct_last=False,
):
    """get a constant entropy increase spaced temperature ladder
    from an input file of betas and logL variances"""

    Ts_in = np.load(T_file_in)
    logL_vars_in = np.load(logL_var_file_in)

    #logL_vars_in = logL_vars_in[Ts_in>=1.]
    #Ts_in = Ts_in[Ts_in>=1.]

    assert np.all(logL_vars_in >= 0.)
    assert np.all(Ts_in >= 0.)
    assert Ts_in.size == logL_vars_in.size

    return EntropyTemperatureLadder(
        n_chain_need,
        Ts_in,
        logL_vars_in,
        n_cold=n_cold,
        T_cold=T_cold,
        use_inf_final=use_inf_final,
        correct_last=correct_last,
    )


def find_potential_phase_transitions(
    betas_in, logL_vars_in, correct_last=True, n_chain_need=2048, micro_thresh=1.e-5
):
    """find the best estimates for temperatures of potential phase transitions
    by interpolating the integrated heat capacity"""

    maxima = []
    minima = []

    # get a spacing that is predicted to be good
    Ts_in = Ts_to_betas(betas_in)

    # get the integrated heat capacity
    betas_use, logL_vars_use = standardize_input_vars(betas_in, logL_vars_in)
    heat_capacity_integ = get_heat_capacity_integrated(logL_vars_use, betas_use, correct_last)

    # Sample the grid with increased density around predicted maxima,
    # but also include and all input points
    betas_got, Ts_got = entropy_spaced_betas(
        n_chain_need,
        0,
        Ts_in,
        logL_vars_in,
        use_inf_final=False,
        T_cold=1.,
        correct_last=correct_last,
    )
    betas_got = np.unique(np.hstack([betas_use, betas_got]))[::-1]
    Ts_got = betas_to_Ts(betas_got)
    n_chain_got = betas_got.size

    # Interpolate the integrate heat capacity and get the derivative
    # Note that because we actually want to interpolate the derivative
    # we need to use k=3 splines, despite possible dangers

    integral_interp = InterpolatedUnivariateSpline(
        betas_use[::-1],
        -heat_capacity_integ[::-1],
        k=3,
        ext=3
    )
    heat_capacity_interp = integral_interp.derivative(1)
    heat_capacity_got = heat_capacity_interp(betas_got)*betas_got

    # remove spurious negative heat capacities
    heat_capacity_got[heat_capacity_got < 0.] = 0.

    assert np.all(heat_capacity_got >= 0.)

    # find local maxima that may represent a phase transition
    itrt_last = 0
    itrt = 1

    while heat_capacity_got[itrt_last] == heat_capacity_got[itrt] and itrt < n_chain_got-2:
        # while loops instead of for loops to handle the unlikely case
        # where some heat capacities are exactly equal
        itrt = itrt + 1

    itrt_next = itrt+1

    # handle starting boundary
    if heat_capacity_got[0] < heat_capacity_got[itrt]:
        minima.append(0)
    elif heat_capacity_got[0] > heat_capacity_got[itrt]:
        maxima.append(0)

    while itrt_next < n_chain_got:
        # while loops instead of for loops to handle the unlikely case
        # where some heat capacities are exactly equal

        while (
            heat_capacity_got[itrt] == heat_capacity_got[itrt_next] and
            itrt_next < n_chain_got-1
        ):
            itrt_next = itrt_next + 1

        if (
            heat_capacity_got[itrt_last] < heat_capacity_got[itrt] and
            heat_capacity_got[itrt_next] <= heat_capacity_got[itrt]
        ):
            maxima.append(itrt)

        elif (
            heat_capacity_got[itrt_last] > heat_capacity_got[itrt] and
            heat_capacity_got[itrt_next] >= heat_capacity_got[itrt]
        ):
            minima.append(itrt)

        itrt_last = itrt
        itrt = itrt_next
        itrt_next = itrt + 1

    # handle ending boundary
    if heat_capacity_got[itrt_last] > heat_capacity_got[-1]:
        minima.append(n_chain_got-1)
    elif heat_capacity_got[itrt_last] < heat_capacity_got[-1]:
        maxima.append(n_chain_got-1)

    minima = np.array(minima)
    maxima = np.array(maxima)

    # minima_Ts = Ts_got[minima]
    maxima_Ts = Ts_got[maxima]

    minima_vals = heat_capacity_got[minima]
    maxima_vals = heat_capacity_got[maxima]

    # default end values
    maxima_Ts[maxima == 0] = 1./betas_use[0]
    maxima_Ts[maxima == n_chain_got-1] = 0.

    # calculate the prominence of each maxima, in the same sense as topographic prominence
    # prominence is difference between maxima and key col, where key col is
    # lowest point between that maxima and a higher maxima
    prominences = np.zeros(maxima.size)
    for itrp, itrt in enumerate(maxima):
        cur_max_val = maxima_vals[itrp]

        key_col1 = 0.
        key_col2 = 0.

        itrt_last = 0
        itrt_next = n_chain_got - 1

        itrp_last = itrp - 1
        if itrp_last >= 0:
            while maxima_vals[itrp_last] < cur_max_val:
                itrp_last -= 1
                if itrp_last < 0:
                    break

        if itrp_last >= 0:
            itrt_last = maxima[itrp_last]

            if np.any((minima >= itrt_last) & (minima <= itrt)):
                key_col1 = np.min(minima_vals[(minima >= itrt_last) & (minima <= itrt)])
            else:
                key_col1 = cur_max_val
        else:
            key_col1 = heat_capacity_got[0]

        itrp_next = itrp + 1
        if itrp_next <= maxima.size - 1:
            while maxima_vals[itrp_next] < cur_max_val:
                itrp_next += 1
                if itrp_next >= maxima.size:
                    break

        if itrp_next < maxima.size:
            itrt_next = maxima[itrp_next]

            if np.any((minima <= itrt_next) & (minima >= itrt)):
                key_col2 = np.min(minima_vals[(minima <= itrt_next) & (minima >= itrt)])
            else:
                key_col2 = cur_max_val
        else:
            key_col2 = 0.

        key_col = max(key_col1, key_col2)

        prominences[itrp] = cur_max_val - key_col

    # cut out micro-prominent maxima, which are probably just noise
    if prominences.size > 0:
        # make sure we keep at least one peak if there are any
        micro_thresh_loc = min(micro_thresh, np.max(prominences))
    else:
        micro_thresh_loc = micro_thresh

    maxima_Ts = maxima_Ts[prominences > micro_thresh_loc]
    maxima_vals = maxima_vals[prominences > micro_thresh_loc]
    prominences = prominences[prominences > micro_thresh_loc]

    return maxima_Ts, maxima_vals, prominences
