"""
Helpersfor computing the temperature ladder for parallel tempering.

C 2023 Matthew C. Digman
"""

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import brentq
from scipy.special import log_ndtr, ndtr

if TYPE_CHECKING:
    from numpy.typing import NDArray


def Ts_to_betas(Ts_in: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert Ts to betas=1/Ts safely handling infinities and zeros.

    Parameters
    ----------
    Ts_in: NDArray[np.floating]
        Temperatures in units of boltzmann constant

    betas_got: NDArray[float]
        Inverse temperatures
    """
    assert np.all(Ts_in >= 0.)
    betas_got: NDArray[np.floating] = np.zeros(Ts_in.size)
    Ts_finite: NDArray[np.bool] = np.isfinite(Ts_in) & (Ts_in != 0.)
    betas_got[Ts_finite] = 1. / Ts_in[Ts_finite]
    betas_got[~np.isfinite(Ts_in)] = 0.
    betas_got[Ts_in == 0.] = np.inf
    return betas_got


class TemperatureLadder:
    """Store a temperature ladder for parallel tempering."""

    def __init__(self, n_cold: int, Ts_in: NDArray[np.floating], sort_mode: int = 1, T_cold: float | None = None) -> None:
        """Create the temperature ladder object.

        Parameters
        ----------
        n_cold: int
            n_cold<=n_chain, total number of T=T_cold chains
        Ts_in: NDArray[np.floating]
            Ts to store
        sort_mode: int
            Selector for how to sort the input temperatures
        T_cold: float | None
            temperature of the n_cold readout chains; None (default) keeps
            the historical convention that the first n_cold rungs are the
            readout chains. Ladders that may extend below the readout
            temperature must set T_cold so get_arg_cold can locate the
            readout rungs by temperature instead of by position.

        Raises
        ------
        ValueError
            If the sort mode is not recognized
        """
        if sort_mode == 0:
            self.Ts: NDArray[np.floating] = Ts_in.copy()
        elif sort_mode == 1:
            self.Ts = np.sort(Ts_in)
        else:
            msg = f'Unrecognized option sort_mode {sort_mode}'
            raise ValueError(msg)

        # self.Ts = Ts_in
        self.sort_mode: int = sort_mode

        self.betas: NDArray[np.floating] = Ts_to_betas(self.Ts)

        self.n_chain: int = Ts_in.size
        self.n_cold: int = n_cold
        self.T_cold: float | None = T_cold

    def get_arg_cold(self) -> NDArray[np.int64]:
        """Get the indices of the n_cold readout chains in this ladder.

        With T_cold set, the readout chains are the n_cold rungs pinned at
        exactly T_cold (every ladder family pins them there by
        construction) — not necessarily the coldest rungs, since a ladder
        may extend below T_cold (T_min < T_cold, sort_mode=1). Without
        T_cold the first n_cold rungs are the readout chains, preserving
        the historical positional convention for raw ladders.
        """
        if self.T_cold is None:
            return np.arange(self.n_cold, dtype=np.int64)
        matches = np.flatnonzero(self.Ts == self.T_cold)
        # every ladder family pins exactly n_cold rungs at T_cold; a spaced
        # rung landing there exactly only adds interchangeable duplicates
        assert matches.size >= self.n_cold
        return matches[:self.n_cold].astype(np.int64)


def betas_to_Ts(betas_in: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert Ts to Ts=1/betas safely handling infinities and zeros.

    Parameters
    ----------
    betas_in: NDArray[np.floating]
        inverse temperatures

    Returns
    -------
    Ts_got: NDarray[np.floating]
        Temperatures in units of boltzmann constant
    """
    assert np.all(betas_in >= 0.)
    Ts_got: NDArray[np.floating] = np.zeros(betas_in.size)
    beta_finite: NDArray[np.bool] = np.isfinite(betas_in) & (betas_in != 0.)
    Ts_got[beta_finite] = 1. / betas_in[beta_finite]
    Ts_got[~np.isfinite(betas_in)] = 0.
    Ts_got[betas_in == 0.] = np.inf
    return Ts_got


def geometric_spaced_betas(n_chain: int, n_cold: int, T_cold: float, T_min: float, T_max: float, n_inf_final: int = 1, sort_mode: int = 1) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Space temperatures geometrically in a range.

    Parameters
    ----------
    n_chain: int
        total number of chains
    n_cold: int
        number of T=T_cold chains.
        cold chains are separate from geometric ladder, unless T_cold=T_min,
        In which case 1 of the cold chains is also considered part of the geometric ladder.
    T_cold: float
        temperature of 'cold' chains
    T_min: float
        minimum temperature of geometric ladder
    T_max: float
        maximum temperature of finite part of geometric ladder
    n_inf_final: int
        How many infinite temperature chains to insert at the end
    """
    if n_cold > n_chain:
        msg = 'n cold cannot be more than total number of chains'
        raise ValueError(msg)

    assert T_min > 0.
    assert np.isfinite(T_min)
    assert T_max > 0.
    assert np.isfinite(T_max)
    assert T_cold > 0.
    assert np.isfinite(T_cold)
    assert T_max > T_min
    assert n_cold >= 0
    assert n_chain > 0

    betas: NDArray[np.floating] = np.zeros(n_chain)
    Ts: NDArray[np.floating] = np.zeros(n_chain)
    assert n_inf_final >= 0

    # if T_cold==T_min then include geometric ladder is pinned to n_cold-1 element.
    # otherwise, ladder needs to be pinned to n_cold element,
    # or it will not include an element at T_min
    if n_inf_final > 0 and n_chain - n_cold == n_inf_final:
        warn('No finite temperature chains will be created', stacklevel=2)
    elif n_chain - n_cold < n_inf_final:
        warn('Some infinite temperature chains will be overwritten', stacklevel=2)
        n_inf_final = n_chain - n_cold

    if T_cold == T_min and n_cold != 0:
        n_geo = n_chain - n_cold - n_inf_final + 1
    else:
        n_geo = n_chain - n_cold - n_inf_final

    beta_loc = 10**np.linspace(-np.log10(T_min), -np.log10(T_max), n_geo)
    if T_cold == T_min and n_cold != 0:
        betas[n_cold:n_chain - n_inf_final] = beta_loc[1:]
    else:
        betas[n_cold:n_chain - n_inf_final] = beta_loc

    Ts[n_cold:n_chain - n_inf_final] = betas_to_Ts(betas[n_cold:n_chain - n_inf_final])
    # recalculate beta this way for internal test consistency
    betas[n_cold:n_chain - n_inf_final] = Ts_to_betas(Ts[n_cold:n_chain - n_inf_final])

    betas[n_chain - n_inf_final:] = 0.
    Ts[n_chain - n_inf_final:] = np.inf

    betas[:n_cold] = 1. / T_cold
    Ts[:n_cold] = T_cold

    if sort_mode == 0:
        pass
    elif sort_mode == 1:
        idx_sort = np.argsort(Ts)
        Ts = Ts[idx_sort].copy()
        betas = betas[idx_sort].copy()
    else:
        msg = f'Unrecognized option sort_mode {sort_mode}'
        raise ValueError(msg)

    return betas, Ts


class GeometricTemperatureLadder(TemperatureLadder):
    """store a geometrically spaced temperature ladder for parallel tempering."""

    def __init__(self, n_chain: int, n_cold: int = 1, T_cold: float = 1., T_min: float = 1., T_max: float = 1.e15, n_inf_final: int = 1, sort_mode: int = 1) -> None:
        """Create the temperature ladder object.

        Parameters
        ----------
        n_chain: int
            total number of parallel tempering chains
        n_cold: int
            n_cold<=n_chain, total number of T=T_cold chains
        T_cold: float
            temperature of 'cold' chain for readout, 1 by default
        T_min: float
            minimum temperature of temperature ladder,
            permitted to be less than T_cold for annealing
        T_max: float
            maximum temperature of finite temperature chains
        n_inf_final: int
            How many infinite temperature chains to insert at the end
        """
        self.T_min: float = T_min
        self.T_max: float = T_max
        self.n_inf_final: int = n_inf_final

        _, Ts = geometric_spaced_betas(
            n_chain, n_cold, T_cold, T_min, T_max, n_inf_final=n_inf_final, sort_mode=sort_mode
        )
        TemperatureLadder.__init__(self, n_cold, Ts, sort_mode=sort_mode, T_cold=T_cold)


def _standardize_stats_core(betas_in: NDArray[np.floating], stats_in: list[NDArray[np.floating]], nonfinite_msg: str) -> tuple[NDArray[np.floating], list[NDArray[np.floating]]]:
    """Shared standardization core: unique finite betas, averaged duplicate stats.

    Sorts betas descending and keeps only finite ones (a zero-temperature
    chain cannot enter the spacing integrals), averages every stat over
    duplicate temperatures, and excises entries where any stat is
    non-finite (with a caller-supplied warning).
    """
    # need to sort the input temperatures and get only unique ones so we can interpolate
    betas_use = np.unique(betas_in)[::-1]
    # Note that using partition functions the lowest order taylor approximation
    # of the heat capacity integral from T=zero to T1 is 0,
    # And we cannot compute the next order correction knowing only
    # the variances of the log likelihoods,
    # So the simplest thing to do is to just cut out any zero temperature (infinite beta)
    # chains from the heat capacity integrals
    betas_use = betas_use[np.isfinite(betas_use)]

    stats_use: list[NDArray[np.floating]] = [np.zeros(betas_use.size) for _ in stats_in]
    for itrf in range(betas_use.size):
        # if there were duplicate temps, average the statistics
        duplicate_sel = betas_use[itrf] == betas_in
        for itrs, stat_in in enumerate(stats_in):
            stats_use[itrs][itrf] = np.mean(stat_in[duplicate_sel])

    finite_mask = np.full(betas_use.size, True)
    for stat_use in stats_use:
        finite_mask &= np.isfinite(stat_use)
    if np.any(~finite_mask):
        # Handle non-finite statistics just by excising those points
        warn(nonfinite_msg, stacklevel=3)
        betas_use = betas_use[finite_mask]
        stats_use = [stat_use[finite_mask] for stat_use in stats_use]

    return betas_use, stats_use


def standardize_input_vars(betas_in: NDArray[np.floating], logL_vars_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Convert the input betas and variances to a standardized form.

    Needed so that the heat capacity integration can work correctly.
    """
    assert len(betas_in.shape) == 1
    assert len(logL_vars_in.shape) == 1
    assert betas_in.shape == logL_vars_in.shape
    betas_use, stats_use = _standardize_stats_core(betas_in, [logL_vars_in], 'Nonfinite variance requested, results may not be meaningful')
    return betas_use, stats_use[0]


def get_spacing_integrated(logL_vars_use: NDArray[np.floating], betas_use: NDArray[np.floating], correct_last: bool, p: float = 1., q: float = 1.) -> NDArray[np.floating]:
    """Get the integrated spacing integrand Var(logL)^p * beta^q over beta.

    (p=1, q=1) is the integrated heat capacity of the entropy ladder;
    (p=1/2, q=0) is the thermodynamic length (second-order
    equal-acceptance) spacing.
    """
    assert len(logL_vars_use.shape) == 1
    assert len(betas_use.shape) == 1
    assert logL_vars_use.shape == betas_use.shape

    heat_capacity_integrand: NDArray[np.floating] = -np.abs(logL_vars_use)**p * betas_use**q
    # cannot handle non finite beta case correctly
    heat_capacity_integrand[~np.isfinite(heat_capacity_integrand)] = 0.

    heat_capacity_integ = cumulative_trapezoid(heat_capacity_integrand[::-1], betas_use[::-1], initial=0)[::-1]

    if correct_last and betas_use[-1] == 0. and betas_use.size > 1:
        # this should be a more accurate approximation of the integrand
        # from the last finite temperature to infinite temperature,
        # assuming the last finite temperature was already sufficiently high:
        # int_0^b Var^p x^q dx = Var^p * b^(q+1)/(q+1) with Var held at its
        # hottest measured value
        heat_capacity_integ[:heat_capacity_integ.size - 1] -= betas_use[-2]**(q + 1.) / (q + 1.) * logL_vars_use[-1]**p

    heat_capacity_integ -= heat_capacity_integ[0]

    # We need to enforce that the heat capacity integral is strictly increasing
    # Integral is strictly increasing if we add tiny increments
    # each time we see two identical values

    # Handle if first value is zero
    # if heat_capacity_integ[0] == 0.:
    #    if np.any(heat_capacity_integ > 0.):
    #        heat_capacity_integ[0] = 1.e-14*np.min(heat_capacity_integ)
    #    else:
    #        heat_capacity_integ[0] = 1.e-15

    for itrn in range(1, heat_capacity_integ.size):
        if heat_capacity_integ[itrn] < heat_capacity_integ[itrn - 1]:
            heat_capacity_integ[itrn:] += heat_capacity_integ[itrn - 1] - heat_capacity_integ[itrn]

        if heat_capacity_integ[itrn] <= heat_capacity_integ[itrn - 1]:
            if heat_capacity_integ[itrn - 1] == 0.:
                heat_capacity_integ[itrn:] += 1.e-15
            else:
                heat_capacity_integ[itrn:] += 1.e-14 * heat_capacity_integ[itrn - 1]

    assert np.all(np.diff(heat_capacity_integ) > 0.)

    return heat_capacity_integ


def get_heat_capacity_integrated(logL_vars_use: NDArray[np.floating], betas_use: NDArray[np.floating], correct_last: bool) -> NDArray[np.floating]:
    """Get the integrated heat capacity: the (p=1, q=1) spacing integral."""
    return get_spacing_integrated(logL_vars_use, betas_use, correct_last, p=1., q=1.)


def entropy_spacing(n_chain_need: int, betas_in: NDArray[np.floating], logL_vars_in: NDArray[np.floating], correct_last: bool = False, p: float = 1., q: float = 1.) -> NDArray[np.floating]:
    """Help estimate constant entropy increase spaced chain.

    Takes an input file of betas and logLs. The integrand generalizes to
    Var(logL)^p * beta^q: (1, 1) is the entropy ladder, (1/2, 0) the
    thermodynamic-length ladder.
    """
    assert n_chain_need > 0
    assert betas_in.size > 0
    assert len(betas_in.shape) == 1
    assert len(logL_vars_in.shape) == 1
    assert betas_in.shape == logL_vars_in.shape

    betas_use, logL_vars_use = standardize_input_vars(betas_in, logL_vars_in)

    if betas_use.size == 0:
        # Somehow there are no valid betas; there is really nothing we can do to recover
        msg = 'No valid points available to construct ladder'
        raise ValueError(msg)

    assert betas_use.size > 0

    heat_capacity_integ = get_spacing_integrated(logL_vars_use, betas_use, correct_last, p=p, q=q)

    if n_chain_need == 1:
        # unsure what to do in this case, but don't divide by zero
        space_heat_need = heat_capacity_integ[-1]
        heat_grid_need = np.arange(0, n_chain_need) * space_heat_need
    else:
        space_heat_need = heat_capacity_integ[-1] / (n_chain_need - 1)
        heat_grid_need = np.arange(0, n_chain_need) * space_heat_need
        assert np.isclose(heat_grid_need[-1], heat_capacity_integ[-1], atol=1.e-14, rtol=1.e-14)
        heat_grid_need[-1] = heat_capacity_integ[-1]  # this should be true anyway, but enforce for numerical stability

    # TODO cubic splines or log interpolation might work better in some cases,
    # but we would need to enforce monotonicity on the splines
    # If the splines are non-monotonic it could produce undesired negative temperatures
    if betas_use.size == 1:
        beta_grid_got: NDArray[np.floating] = np.full(n_chain_need, betas_use[0])
        warn('Only one unique input temperature: cannot generate a meaningful grid', stacklevel=2)
    else:
        beta_interp = InterpolatedUnivariateSpline(heat_capacity_integ, betas_use, k=1, ext=3)
        beta_grid_got = beta_interp(heat_grid_need)

    if np.any(beta_grid_got < 0.):
        warn('Unexpected negative temperatures: defaulting to abs', stacklevel=2)
        beta_grid_got[beta_grid_got < 0.] = np.abs(beta_grid_got[beta_grid_got < 0.])
        beta_grid_got = np.sort(beta_grid_got)[::-1]

    if not np.all(np.diff(beta_grid_got) <= 0.):
        warn('Temperature grid is not sorted correctly', stacklevel=2)
        beta_grid_got = np.sort(beta_grid_got)[::-1]

    assert np.all(beta_grid_got >= 0.)
    assert np.all(np.diff(beta_grid_got) <= 0.)

    # Avoid dividing by zero
    T_grid_got: NDArray[np.floating] = betas_to_Ts(beta_grid_got)

    return T_grid_got


def _plug_cold_and_inf(
        Ts_got: NDArray[np.floating],
        betas_in: NDArray[np.floating],
        n_chain_space: int,
        n_chain_need: int,
        n_cold: int,
        n_inf_final: int,
        T_cold: float,
        sort_mode: int,
        snap_mode: int = 0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Shared cold/infinite-rung plugging and sorting for spaced ladders.

    Every spaced ladder family (entropy, length, acceptance) applies the
    same conventions after spacing its rungs: overwrite the hottest
    n_inf_final rungs with infinity, snap one spaced rung to T_cold
    exactly (this deliberately distorts the adjacent spacing interval
    when T_cold sits far from the input range — the documented cost of
    the cold plug, identical across families so arm comparisons stay
    apples-to-apples), pad n_cold-1 duplicate cold rungs, and apply
    sort_mode.

    snap_mode selects which rung the plug consumes: 0 (default) the
    rung nearest T_cold; 1 the coolest rung at or above T_cold, falling
    back to nearest when none sits above. The two coincide whenever the
    spaced rungs all lie at or above T_cold; they differ only for
    ladders extending below T_cold (T_min < T_cold).
    """
    if n_inf_final > 0:
        Ts_got[n_chain_space - n_inf_final:] = np.inf
        assert np.all(np.isfinite(Ts_got[:n_chain_space - n_inf_final])) or np.all(betas_in == 0.)

    # TODO add option to do include cold spacing adaptively or not
    if n_cold > 0:
        # shift the generated value that is closest to the original cold value
        # need special handling if the 'cold' temperature is infinity
        if ~np.isfinite(T_cold):
            if Ts_got.size > 1:
                arg_cold = max(Ts_got.size - n_inf_final - 1, 0)
            else:
                arg_cold = 0
            assert arg_cold >= 0
        elif snap_mode == 1:
            at_or_above = np.flatnonzero(Ts_got >= T_cold)
            if at_or_above.size:
                arg_cold = int(at_or_above[np.argmin(Ts_got[at_or_above] - T_cold)])
            else:
                arg_cold = int(np.argmin(np.abs(Ts_got - T_cold)))
        elif snap_mode == 0:
            arg_cold = int(np.argmin(np.abs(Ts_got - T_cold)))
        else:
            msg = f'Unrecognized option snap_mode {snap_mode}'
            raise ValueError(msg)
        Ts_got[arg_cold] = T_cold
        if arg_cold != 0:
            # put cold values first for now
            Ts_got = np.hstack([Ts_got[arg_cold], Ts_got[:arg_cold], Ts_got[arg_cold + 1:]])

    if n_cold > 1:
        Ts_got = np.hstack([np.full(n_cold - 1, T_cold), Ts_got])

    assert Ts_got.size == n_chain_need

    betas_got = Ts_to_betas(Ts_got)

    if sort_mode == 0:
        pass
    elif sort_mode == 1:
        idx_sort = np.argsort(Ts_got)
        Ts_got = Ts_got[idx_sort].copy()
        betas_got = betas_got[idx_sort].copy()
    else:
        msg = f'Unrecognized option sort_mode {sort_mode}'
        raise ValueError(msg)

    return betas_got, Ts_got


def entropy_spaced_betas(
        n_chain_need: int,
        n_cold: int,
        Ts_in: NDArray[np.floating],
        logL_vars_in: NDArray[np.floating],
        n_inf_final: int = 1,
        T_cold: float = 1.,
        correct_last: bool = False,
        sort_mode: int = 1,
        p: float = 1.,
        q: float = 1.,
        snap_mode: int = 0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Estimate constant entropy increase spaced chain from an input file of betas and logLs.

    The (p, q) exponents select the spacing integrand Var(logL)^p * beta^q:
    the defaults give the entropy ladder, (1/2, 0) the thermodynamic-length
    ladder.
    """
    if n_cold > n_chain_need:
        msg = 'n cold cannot be more than total number of chains'
        raise ValueError(msg)

    assert len(Ts_in.shape) == 1
    assert len(logL_vars_in.shape) == 1
    assert Ts_in.shape == logL_vars_in.shape
    assert T_cold >= 0.
    assert np.all(logL_vars_in >= 0.)
    assert np.all(Ts_in >= 0.)
    assert n_cold >= 0
    assert n_chain_need > 0
    assert n_inf_final >= 0
    if n_inf_final > 0 and n_chain_need - n_cold == n_inf_final:
        warn('No finite temperature chains will be created', stacklevel=2)
    elif n_inf_final > 0 and n_chain_need - n_cold < n_inf_final:
        warn('Some infinite temperature chains will be overwritten', stacklevel=2)
        n_inf_final = n_chain_need - n_cold

    if n_cold == 0:
        n_chain_space = n_chain_need
    else:
        n_chain_space = n_chain_need - n_cold + 1

    assert n_chain_space >= 0

    betas_in = Ts_to_betas(Ts_in)

    # if any betas_in are zero, there will be a nonfinite temperature, so prune it if we don't want it
    if n_inf_final == 0 and n_chain_space > 1 and np.any(betas_in == 0.) and not correct_last:
        needs_prune = True
        n_chain_space = n_chain_space + 1
    else:
        needs_prune = False

    Ts_got = entropy_spacing(n_chain_space, betas_in, logL_vars_in, correct_last=correct_last, p=p, q=q)
    if needs_prune:
        # trim the non-finite beta if we requested none
        assert Ts_got[-1] == np.inf
        n_chain_space = n_chain_space - 1
        assert n_chain_space >= 0
        Ts_got = Ts_got[:n_chain_space]

    assert Ts_got.shape == (n_chain_space,)

    assert np.all(Ts_got >= 0.)

    return _plug_cold_and_inf(Ts_got, betas_in, n_chain_space, n_chain_need, n_cold, n_inf_final, T_cold, sort_mode, snap_mode=snap_mode)


class EntropyTemperatureLadder(TemperatureLadder):
    """Store a constant entropy increase spaced temperature ladder for parallel tempering."""

    def __init__(
            self,
            n_chain: int,
            Ts_in: NDArray[np.floating],
            logL_vars_in: NDArray[np.floating],
            n_cold: int = 1,
            T_cold: float = 1.,
            n_inf_final: int = 1,
            correct_last: bool = False,
            sort_mode: int = 1,
            snap_mode: int = 0,
    ) -> None:
        """
        Create the temperature ladder object.

        Parameters
        ----------
        n_chain: int
            total number of parallel tempering chains
        Ts_in: NDArray[np.floating]
            input temperatures to use for building heat capacity
        logL_vars_in: NDArray[np.floating]
            input variances to use for building heat capacity
        n_cold: int
            n_cold<=n_chain
            total number of T=T_cold chains
        T_cold: float
            temperature of 'cold' chain for readout, 1 by default
        n_inf_final: int
            How many infinite temperature chains to insert at the end
        correct_last: bool
            whether to use a taylor series estimate to
            extrapolate the heat capacity integral out to infinity; only works well
            if maximum T in input ladder is well above any phase transitions
        sort_mode: int
            Select mode for how temperatures are sorted.
        snap_mode: int
            Which spaced rung the cold plug consumes (see _plug_cold_and_inf):
            0 nearest to T_cold, 1 coolest at or above T_cold (preserves
            sub-T_cold rungs when the inputs extend below the readout)
        """
        self.n_inf_final: int = n_inf_final

        _, Ts = entropy_spaced_betas(
            n_chain,
            n_cold,
            Ts_in,
            logL_vars_in,
            n_inf_final=n_inf_final,
            T_cold=T_cold,
            correct_last=correct_last,
            sort_mode=sort_mode,
            snap_mode=snap_mode,
        )
        TemperatureLadder.__init__(self, n_cold, Ts, sort_mode=sort_mode, T_cold=T_cold)


class LengthTemperatureLadder(TemperatureLadder):
    """Store a thermodynamic-length spaced temperature ladder for parallel tempering.

    Equal increments of the (p=1/2, q=0) spacing integral
    int sqrt(Var(logL)) dbeta — the second-order equal-acceptance
    (thermodynamic length) rule. Same inputs and conventions as
    EntropyTemperatureLadder, deliberately, so ladder-family comparisons
    are apples-to-apples.

    Unlike the entropy integrand (whose beta^1 factor suppresses the
    hot end), sqrt(Var) is nonzero at beta = 0: if the inputs include an
    infinite-temperature entry, its statistics contribute finite length
    to the (0, beta_min] segment and rungs legitimately place hotter
    than the hottest finite input. Exclude infinite-temperature rows
    from the inputs (e.g. via filter_ladder_inputs, as the harness's
    file-driven paths do) when the prior rung should not influence the
    spacing.
    """

    def __init__(
            self,
            n_chain: int,
            Ts_in: NDArray[np.floating],
            logL_vars_in: NDArray[np.floating],
            n_cold: int = 1,
            T_cold: float = 1.,
            n_inf_final: int = 1,
            correct_last: bool = False,
            sort_mode: int = 1,
            snap_mode: int = 0,
    ) -> None:
        """Create the temperature ladder object; parameters as EntropyTemperatureLadder."""
        self.n_inf_final: int = n_inf_final

        _, Ts = entropy_spaced_betas(
            n_chain,
            n_cold,
            Ts_in,
            logL_vars_in,
            n_inf_final=n_inf_final,
            T_cold=T_cold,
            correct_last=correct_last,
            sort_mode=sort_mode,
            p=0.5,
            q=0.,
            snap_mode=snap_mode,
        )
        TemperatureLadder.__init__(self, n_cold, Ts, sort_mode=sort_mode, T_cold=T_cold)


def standardize_input_stats(betas_in: NDArray[np.floating], logL_means_in: NDArray[np.floating], logL_vars_in: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Convert input betas with logL means and variances to a standardized form.

    Mirrors standardize_input_vars (unique finite betas, descending;
    duplicate-beta stats averaged; entries with non-finite stats excised)
    but carries the first two cumulants together, as the acceptance
    ladder needs both.
    """
    assert len(betas_in.shape) == 1
    assert betas_in.shape == logL_means_in.shape
    assert betas_in.shape == logL_vars_in.shape
    betas_use, stats_use = _standardize_stats_core(betas_in, [logL_means_in, logL_vars_in], 'Nonfinite logL statistics requested, results may not be meaningful')
    return betas_use, stats_use[0], stats_use[1]


def predicted_swap_acceptance(beta1: float, beta2: float, mean1: float, mean2: float, var1: float, var2: float) -> float:
    """Gaussian closed-form predicted swap acceptance between two rungs.

    The swap Metropolis-Hastings log-ratio is r = (beta1-beta2)*(logL2-logL1);
    with Gaussian logL marginals it has mean m = -(beta1-beta2)*(mean1-mean2)
    and variance s^2 = (beta1-beta2)^2*(var1+var2), giving
    E[min(1, e^r)] = Phi(m/s) + exp(m + s^2/2)*Phi(-(m+s^2)/s),
    evaluated log-stably. For large z = m/s + s the exact expression
    suffers catastrophic cancellation (s^2/2 and log_ndtr(-z) are equal
    huge magnitudes: at s ~ 1e9 both round to +-1e18 and their float sum
    is 0, driving the result above 1), so that regime uses the
    well-conditioned asymptotic exp(m + s^2/2)*Phi(-z) ->
    exp(-(m/s)^2/2)/(z*sqrt(2*pi)), with relative error O(1/z^2). The
    result is clamped into [0, 1].
    """
    m: float = -(beta1 - beta2) * (mean1 - mean2)
    s: float = float(np.sqrt((beta1 - beta2)**2 * (var1 + var2)))
    if s < 1.e-150:
        return float(min(1., np.exp(min(m, 0.))))
    u: float = m / s
    z: float = u + s
    if z > 30.:
        second_term: float = float(np.exp(-u * u / 2.) / (z * np.sqrt(2. * np.pi)))
    else:
        # z bounded above means m <= s*(30 - s) stays moderate: no cancellation
        second_term = float(np.exp(min(m + s * s / 2. + log_ndtr(-z), 0.)))
    return float(min(1., ndtr(u) + second_term))


def acceptance_spaced_betas(
        n_chain_need: int,
        n_cold: int,
        Ts_in: NDArray[np.floating],
        logL_means_in: NDArray[np.floating],
        logL_vars_in: NDArray[np.floating],
        n_inf_final: int = 1,
        T_cold: float = 1.,
        sort_mode: int = 1,
        snap_mode: int = 0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """Space rungs for constant predicted swap acceptance between neighbors.

    Interpolates logL means/variances in beta from the same inputs the
    entropy ladder uses (deliberately, so adaptation-speed comparisons
    are apples-to-apples), then places rungs by sequential root-find from
    the hot end — each next rung is where the Gaussian closed-form
    acceptance to its predecessor equals the target a* — with an outer
    bisection on a* so exactly the required number of rungs spans the
    input range. Returns (betas, Ts, achieved a*).

    Contract (mirrors the entropy ladder's conventions exactly): the
    equal-acceptance property holds *among the finite spaced rungs*.
    Infinite rungs are plugged afterwards, outside the spacing contract
    (as the entropy ladder's inf rungs sit outside its equal-dS
    contract), so the hottest-finite-to-infinite edge is not constrained
    to a*. Likewise the rung nearest T_cold is snapped to T_cold, which
    distorts its adjacent interval when T_cold sits far from the input
    range — the shared, documented cost of the cold plug (see
    _plug_cold_and_inf). If the inputs include a beta = 0 (infinite
    temperature) entry but no infinite output rung is requested, the
    walk anchors there and the anchor rung is pruned, mirroring the
    entropy machinery's prune handling.
    """
    if n_cold > n_chain_need:
        msg = 'n cold cannot be more than total number of chains'
        raise ValueError(msg)

    assert T_cold >= 0.
    assert np.all(logL_vars_in >= 0.)
    assert np.all(Ts_in >= 0.)
    assert n_cold >= 0
    assert n_chain_need > 0
    assert n_inf_final >= 0
    if n_inf_final > 0 and n_chain_need - n_cold == n_inf_final:
        warn('No finite temperature chains will be created', stacklevel=2)
    elif n_inf_final > 0 and n_chain_need - n_cold < n_inf_final:
        warn('Some infinite temperature chains will be overwritten', stacklevel=2)
        n_inf_final = n_chain_need - n_cold

    if n_cold == 0:
        n_chain_space = n_chain_need
    else:
        n_chain_space = n_chain_need - n_cold + 1

    if n_chain_space < 2:
        msg = 'acceptance ladder needs at least 2 rungs to space'
        raise ValueError(msg)

    betas_in = Ts_to_betas(Ts_in)
    betas_use, means_use, vars_use = standardize_input_stats(betas_in, logL_means_in, logL_vars_in)
    if betas_use.size < 2:
        msg = 'acceptance ladder needs at least 2 distinct finite input temperatures'
        raise ValueError(msg)

    # interpolants in ascending beta, clamped beyond the input range
    betas_asc = betas_use[::-1]
    mean_interp = InterpolatedUnivariateSpline(betas_asc, means_use[::-1], k=1, ext=3)
    var_interp = InterpolatedUnivariateSpline(betas_asc, vars_use[::-1], k=1, ext=3)

    beta_hot: float = float(betas_asc[0])
    beta_cold_end: float = float(betas_asc[-1])
    # rungs may transiently overshoot the cold end during bisection; the
    # clamped interpolants keep the acceptance well-defined out there
    beta_upper: float = beta_cold_end * 10. + 1.

    # if the input includes beta = 0 but no infinite output rung is
    # requested, walk one extra rung anchored at beta = 0 and trim it
    # afterwards — mirroring the entropy machinery's prune handling
    needs_prune: bool = n_inf_final == 0 and beta_hot == 0.
    n_walk: int = n_chain_space + 1 if needs_prune else n_chain_space

    def interp_scalar(spline: InterpolatedUnivariateSpline, beta_loc: float) -> float:
        return float(spline(np.asarray([beta_loc]))[0])

    def acceptance_from(beta_lo: float, beta_hi_loc: float) -> float:
        return predicted_swap_acceptance(
            beta_hi_loc, beta_lo,
            interp_scalar(mean_interp, beta_hi_loc), interp_scalar(mean_interp, beta_lo),
            interp_scalar(var_interp, beta_hi_loc), interp_scalar(var_interp, beta_lo),
        )

    def walk_positions(a_target: float) -> NDArray[np.floating]:
        positions = np.zeros(n_walk)
        positions[0] = beta_hot
        for itrs in range(1, n_walk):
            beta_prev = positions[itrs - 1]
            if acceptance_from(beta_prev, beta_upper) >= a_target:
                # even the largest permitted step stays above target
                positions[itrs] = beta_upper
            else:
                start = beta_prev + 1.e-14 * (1. + beta_prev)
                positions[itrs] = brentq(lambda x: acceptance_from(beta_prev, x) - a_target, start, beta_upper)  # noqa: B023
        return positions

    # outer bisection: higher targets give tighter rungs, so the walk end
    # is monotone decreasing in a*; find a* landing rung n on the cold end
    a_lo, a_hi = 1.e-4, 1. - 1.e-4
    if walk_positions(a_lo)[-1] < beta_cold_end or walk_positions(a_hi)[-1] > beta_cold_end:
        msg = 'acceptance ladder target is not bracketed; inputs may span too little range for the requested rung count'
        raise ValueError(msg)
    for _ in range(60):
        a_mid = 0.5 * (a_lo + a_hi)
        if walk_positions(a_mid)[-1] > beta_cold_end:
            a_lo = a_mid
        else:
            a_hi = a_mid
    a_star = 0.5 * (a_lo + a_hi)

    positions = walk_positions(a_star)
    positions[-1] = beta_cold_end
    assert np.all(np.diff(positions) > 0.)

    if needs_prune:
        # trim the beta = 0 anchor rung: no infinite rung was requested
        assert positions[0] == 0.
        positions = positions[1:]

    # coldest first, mirroring the entropy machinery's conventions
    Ts_got = betas_to_Ts(positions[::-1].copy())
    assert Ts_got.size == n_chain_space

    betas_got, Ts_got = _plug_cold_and_inf(Ts_got, betas_in, n_chain_space, n_chain_need, n_cold, n_inf_final, T_cold, sort_mode, snap_mode=snap_mode)
    return betas_got, Ts_got, a_star


class AcceptanceTemperatureLadder(TemperatureLadder):
    """Store a constant predicted-swap-acceptance ladder for parallel tempering.

    Takes the same inputs as EntropyTemperatureLadder plus the logL means
    (the Gaussian acceptance prediction needs both first cumulants); the
    achieved target acceptance is kept as achieved_acceptance.
    """

    def __init__(
            self,
            n_chain: int,
            Ts_in: NDArray[np.floating],
            logL_means_in: NDArray[np.floating],
            logL_vars_in: NDArray[np.floating],
            n_cold: int = 1,
            T_cold: float = 1.,
            n_inf_final: int = 1,
            sort_mode: int = 1,
            snap_mode: int = 0,
    ) -> None:
        """Create the temperature ladder object.

        Parameters as EntropyTemperatureLadder, plus logL_means_in: the
        per-input-temperature mean log likelihoods.
        """
        self.n_inf_final: int = n_inf_final

        _, Ts, a_star = acceptance_spaced_betas(
            n_chain,
            n_cold,
            Ts_in,
            logL_means_in,
            logL_vars_in,
            n_inf_final=n_inf_final,
            T_cold=T_cold,
            sort_mode=sort_mode,
            snap_mode=snap_mode,
        )
        self.achieved_acceptance: float = a_star
        TemperatureLadder.__init__(self, n_cold, Ts, sort_mode=sort_mode, T_cold=T_cold)


def remap_ladder_indices(Ts_old: NDArray[np.floating], Ts_new: NDArray[np.floating], remap_rule: str) -> NDArray[np.int64]:
    """Old-ladder source column feeding each new-ladder slot on a ladder update.

    The apply_ladder_update hook and pilot code share this definition.
    Chain states do not use these rules: they remap by temperature rank,
    which for equal-size sorted ladders is the identity.

    - 'at_or_hotter': the coolest old temperature at-or-hotter than the
      new one, falling back to the hottest old rung. Under cold support
      extension this rule can be many-to-one.
    - 'nearest': nearest old temperature in log T.
    - 'no_remap': preserve DE-buffer columns by slot. This is bijective
      for equal-size ladder updates.

    Exact-temperature ties resolve slot-preservingly, else to the lowest
    tied slot, so an identical-ladder update — including duplicate
    temperatures — maps every slot to itself. Infinite temperatures are
    clipped to 1e300 for the log comparison.
    """
    if remap_rule == 'no_remap':
        if Ts_old.size != Ts_new.size:
            msg = 'no_remap requires equal-size old and new ladders'
            raise ValueError(msg)
        return np.arange(Ts_new.size, dtype=np.int64)

    sources = np.zeros(Ts_new.size, dtype=np.int64)
    Ts_old_clip = np.minimum(Ts_old, 1.e300)
    log_old = np.log(Ts_old_clip)
    for itrt in range(Ts_new.size):
        T_new_clip = min(Ts_new[itrt], 1.e300)
        if remap_rule == 'at_or_hotter':
            hotter = np.flatnonzero(Ts_old_clip >= T_new_clip)
            if hotter.size:
                tied = hotter[Ts_old_clip[hotter] == Ts_old_clip[hotter].min()]
            else:
                tied = np.flatnonzero(Ts_old_clip == Ts_old_clip.max())
        elif remap_rule == 'nearest':
            log_dist = np.abs(log_old - np.log(T_new_clip))
            tied = np.flatnonzero(log_dist == log_dist.min())
        else:
            msg = f'unknown remap rule {remap_rule!r}'
            raise ValueError(msg)
        sources[itrt] = itrt if itrt in tied else int(tied[0])
    return sources


def filter_ladder_inputs(Ts_in: NDArray[np.floating], *stats_in: NDArray[np.floating], T_min: float = 1.) -> tuple[NDArray[np.floating], ...]:
    """Apply the from-file input convention: keep only rungs with Ts >= T_min.

    The single source of the Ts >= 1 filter used by every file-driven
    ladder path (entropy_ladder_fromfile and the harness builders). The
    ladder classes themselves deliberately do not filter: sub-T=1 inputs
    are legitimate for cold-edge rungs and annealing anchors.
    """
    keep = Ts_in >= T_min
    return tuple(array[keep].copy() for array in (Ts_in, *stats_in))


def entropy_ladder_fromfile(
        n_chain_need: int,
        n_cold: int,
        T_file_in: str,
        logL_var_file_in: str,
        n_inf_final: int = 1,
        T_cold: float = 1.,
        correct_last: bool = False,
        sort_mode: int = 1,
) -> EntropyTemperatureLadder:
    """Get a constant entropy increase spaced temperature ladder.

    Takes an input file of betas and logL variances.
    """
    Ts_in: NDArray[np.floating] = np.load(T_file_in)
    logL_vars_in: NDArray[np.floating] = np.load(logL_var_file_in)

    assert n_chain_need >= 0
    assert n_cold >= 0
    assert n_inf_final >= 0
    assert len(Ts_in.shape) == 1
    assert len(logL_vars_in.shape) == 1
    assert Ts_in.shape == logL_vars_in.shape
    assert Ts_in.size > 0
    assert logL_vars_in.size > 0

    Ts_in, logL_vars_in = filter_ladder_inputs(Ts_in, logL_vars_in)

    assert np.all(logL_vars_in >= 0.)
    assert np.all(Ts_in >= 0.)

    return EntropyTemperatureLadder(
        n_chain_need,
        Ts_in,
        logL_vars_in,
        n_cold=n_cold,
        T_cold=T_cold,
        n_inf_final=n_inf_final,
        correct_last=correct_last,
        sort_mode=sort_mode,
    )


def find_potential_phase_transitions(
        betas_in: NDArray[np.floating], logL_vars_in: NDArray[np.floating], correct_last: bool = True, n_chain_need: int = 2048, micro_thresh: float = 1.e-5, sort_mode: int = 1
        ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Find the best estimates for temperatures of potential phase transitions.

    Interpolates the integrated heat capacity.
    """
    assert len(betas_in.shape) == 1
    assert len(logL_vars_in.shape) == 1
    assert logL_vars_in.shape == betas_in.shape

    maxima_list: list[int] = []
    minima_list: list[int] = []

    # get a spacing that is predicted to be good
    Ts_in: NDArray[np.floating] = Ts_to_betas(betas_in)

    # get the integrated heat capacity
    betas_use, logL_vars_use = standardize_input_vars(betas_in, logL_vars_in)
    heat_capacity_integ: NDArray[np.floating] = get_heat_capacity_integrated(logL_vars_use, betas_use, correct_last)

    # Sample the grid with increased density around predicted maxima,
    # but also include and all input points
    betas_got, Ts_got = entropy_spaced_betas(
        n_chain_need,
        0,
        Ts_in,
        logL_vars_in,
        n_inf_final=0,
        T_cold=1.,
        correct_last=correct_last,
        sort_mode=sort_mode,
    )
    betas_got = np.unique(np.hstack([betas_use, betas_got]))[::-1]
    Ts_got = betas_to_Ts(betas_got)
    n_chain_got: int = betas_got.size

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
    heat_capacity_got: NDArray[np.floating] = heat_capacity_interp(betas_got) * betas_got

    # remove spurious negative heat capacities
    heat_capacity_got[heat_capacity_got < 0.] = 0.

    assert np.all(heat_capacity_got >= 0.)

    # find local maxima that may represent a phase transition
    itrt_last: int = 0
    itrt: int = 1

    while heat_capacity_got[itrt_last] == heat_capacity_got[itrt] and itrt < n_chain_got - 2:
        # while loops instead of for loops to handle the unlikely case
        # where some heat capacities are exactly equal
        itrt = itrt + 1

    itrt_next: int = itrt + 1

    # handle starting boundary
    if heat_capacity_got[0] < heat_capacity_got[itrt]:
        minima_list.append(0)
    elif heat_capacity_got[0] > heat_capacity_got[itrt]:
        maxima_list.append(0)

    while itrt_next < n_chain_got:
        # while loops instead of for loops to handle the unlikely case
        # where some heat capacities are exactly equal

        while (
            heat_capacity_got[itrt] == heat_capacity_got[itrt_next] and
            itrt_next < n_chain_got - 1
        ):
            itrt_next = itrt_next + 1

        if (
            heat_capacity_got[itrt_last] < heat_capacity_got[itrt] and
            heat_capacity_got[itrt_next] <= heat_capacity_got[itrt]
        ):
            maxima_list.append(itrt)

        elif (
            heat_capacity_got[itrt_last] > heat_capacity_got[itrt] and
            heat_capacity_got[itrt_next] >= heat_capacity_got[itrt]
        ):
            minima_list.append(itrt)

        itrt_last = itrt
        itrt = itrt_next
        itrt_next = itrt + 1

    # handle ending boundary
    if heat_capacity_got[itrt_last] > heat_capacity_got[-1]:
        minima_list.append(n_chain_got - 1)
    elif heat_capacity_got[itrt_last] < heat_capacity_got[-1]:
        maxima_list.append(n_chain_got - 1)

    minima: NDArray[np.int64] = np.array(minima_list, dtype=np.int64)
    maxima: NDArray[np.int64] = np.array(maxima_list, dtype=np.int64)

    # minima_Ts = Ts_got[minima]
    maxima_Ts: NDArray[np.floating] = Ts_got[maxima]

    minima_vals: NDArray[np.floating] = heat_capacity_got[minima]
    maxima_vals: NDArray[np.floating] = heat_capacity_got[maxima]

    # default end values
    maxima_Ts[maxima == 0] = 1. / betas_use[0]
    maxima_Ts[maxima == n_chain_got - 1] = 0.

    # calculate the prominence of each maxima, in the same sense as topographic prominence
    # prominence is difference between maxima and key col, where key col is
    # lowest point between that maxima and a higher maxima
    prominences: NDArray[np.floating] = np.zeros(maxima.size)
    for itrp, itrt in enumerate(maxima):
        cur_max_val: float = maxima_vals[itrp]

        key_col1: float = 0.
        key_col2: float = 0.

        itrt_last = 0
        itrt_next = n_chain_got - 1

        itrp_last: int = itrp - 1
        if itrp_last >= 0:
            while maxima_vals[itrp_last] < cur_max_val:
                itrp_last -= 1
                if itrp_last < 0:
                    break

        if itrp_last >= 0:
            itrt_last = maxima[itrp_last]

            if np.any((minima >= itrt_last) & (minima <= itrt)):
                key_col1 = float(np.min(minima_vals[(minima >= itrt_last) & (minima <= itrt)]))
            else:
                key_col1 = cur_max_val
        else:
            key_col1 = float(heat_capacity_got[0])

        itrp_next: int = itrp + 1
        if itrp_next <= maxima.size - 1:
            while maxima_vals[itrp_next] < cur_max_val:
                itrp_next += 1
                if itrp_next >= maxima.size:
                    break

        if itrp_next < maxima.size:
            itrt_next = int(maxima[itrp_next])

            if np.any((minima <= itrt_next) & (minima >= itrt)):
                key_col2 = float(np.min(minima_vals[(minima <= itrt_next) & (minima >= itrt)]))
            else:
                key_col2 = cur_max_val
        else:
            key_col2 = 0.

        key_col: float = max(key_col1, key_col2)

        prominences[itrp] = cur_max_val - key_col

    # cut out micro-prominent maxima, which are probably just noise
    if prominences.size > 0:
        # make sure we keep at least one peak if there are any
        micro_thresh_loc: float = min(micro_thresh, float(np.max(prominences)))
    else:
        micro_thresh_loc = micro_thresh

    maxima_Ts = maxima_Ts[prominences > micro_thresh_loc]
    maxima_vals = maxima_vals[prominences > micro_thresh_loc]
    prominences = prominences[prominences > micro_thresh_loc]

    return maxima_Ts, maxima_vals, prominences
