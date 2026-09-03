"""helper to print some diagnostic commentary about chain performance"""

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy

import DTMCMC.temperature_ladder_helpers as th

if TYPE_CHECKING:
    from DTMCMC.dtmcmc_sampler import DTMCMCSampler
    from DTMCMC.likelihood import AbstractLikelihood


def print_diagnostic_commentary[LikelihoodType: AbstractLikelihood[Any]](
    mcc: DTMCMCSampler[LikelihoodType],
) -> None:
    print('==========Descriptive Summary===========')
    print('Sampler has %5d chains, of which %5d are cold' % (mcc.n_chain, mcc.n_cold))
    print('Sampler reports having run for %10d iterations' % (mcc.itrn))
    print('Temperature ladder is of type ', type(mcc.T_ladder))
    print('Likelihood has %5d dimensions and is of type ' % (mcc.n_par), type(mcc.like_obj))

    print('==========Qualitative Comments==========')

    max_logL_found = max(np.max(mcc.logLs_store), np.max(mcc.logLs))
    logL_means = np.array(mcc.logL_means)
    logL_vars = np.array(mcc.logL_vars)

    if np.any(mcc.logLs == 0.0):
        print('Some likelihoods in last block are identically 0: Could be a bug?')

    if np.any(mcc.logLs_store == 0.0):
        print('Some stored likelihoods are identically 0: Could be a bug?')

    # The maximum likelihood currently available indicates how search is going
    print('Maximum likelihood stored=%+.9e' % (max_logL_found))

    # If the cold chain is not near the maximum at end, that may be suspicious
    off_max = logL_means[:, 0 : mcc.n_cold] < max_logL_found - mcc.n_par - 5 * np.sqrt(logL_vars[:, 0 : mcc.n_cold])
    if np.any(off_max[-1]):
        print('Cold chains may be off maximum at end')
        print('Note: if cold chain is substantially off-maximum, it may be a sign of drifting')

    print('Cold chains have %6d potentially off max blocks out of %6d total' % (np.sum(off_max), off_max.size))
    if np.sum(off_max) > off_max.size // 2:
        print('Note: if many blocks are off maximum, it may be a sign of drifting or non-convergence')

    # may indicate limit on earliest time burned in
    if np.any(off_max):
        print('Last potentially off max block in cold chain is at block %6d' % (np.max(np.argmax(off_max, axis=1))))
        if np.max(np.argmax(off_max, axis=1)) > off_max.shape[0] // 2:
            print('Note: if off max blocks continue very late in evolution, it may be a sign of inadequate burn in')

    # TODO improve variance and burn in estimates

    # NOTE: this broadcasts a width-record_indices store against the full
    # betas vector, which is only numerically meaningful for full-width or
    # single-column storage; mapping store columns through record_indices is
    # deferred so this PR preserves the pre-arg_record diagnostic output
    # exactly (see issue #30).
    df_predict = np.var(-2 * mcc.logLs_store[mcc.logLs_store.shape[0] // 2 :] * mcc.betas, axis=0) / 2
    df_mean = np.var(-2 * mcc.logLs_store[mcc.logLs_store.shape[0] // 2 :, : mcc.n_cold] * mcc.betas[: mcc.n_cold]) / 2
    df_res, _loc_res, _scale_res = scipy.stats.chi2.fit(
        2 * max_logL_found - 2 * mcc.logLs_store[mcc.logLs_store.shape[0] // 2 :, 0 : mcc.n_cold].flatten(), df_mean
    )
    print(
        'Cold likelihood distribution estimate %8.5f effective dimensions (best fit: %8.5f): %5d expected if all dimensions are gaussian'
        % (df_mean, df_res, mcc.n_par)
    )

    print('Effective dimension hottest two chains: %8.5f %8.5f' % (df_predict[-2], df_predict[-1]))

    # Look for potential disconnections based on direct log-likelihood diagnostics
    print('=========Mean likelihood analysis==========')

    print('Searching for disconnects by mean likelihood')
    disconnect_up = np.zeros(mcc.n_chain, dtype=np.bool_)
    disconnect_down = np.zeros(mcc.n_chain, dtype=np.bool_)
    for itrt in range(mcc.n_chain - 1):
        logL_diff = logL_means[-1, itrt] - logL_means[-1, itrt + 1]
        if logL_diff > 2 * np.sqrt(logL_vars[-1, itrt]):
            print(
                'Probable likelihood disconnect at temperature %5d between T=%+.9e and T=%+.9e'
                % (itrt, mcc.Ts[itrt], mcc.Ts[itrt + 1])
            )
            disconnect_up[itrt] = True
            disconnect_down[itrt + 1] = True
            if disconnect_down[itrt]:
                print('Temperature %5d may be completely isolated in likelihood from rest of sampler' % (itrt))

    # disconnect_sym = disconnect_up & disconnect_down

    print('%5d disconnects found by mean likelihood' % (np.sum(disconnect_up)))
    if np.any(np.sum(disconnect_up)):
        print('Note: disconnects can indicate that the sampler is inefficient')

    print('Searching for imbalances between pairs of adjacent chains')

    imbalance = np.zeros(mcc.n_chain, dtype=np.bool_)
    for itrt in range(1, mcc.n_chain - 1):
        logL_diff1 = logL_means[-1, itrt - 1] - logL_means[-1, itrt]
        logL_diff2 = logL_means[-1, itrt] - logL_means[-1, itrt + 1]
        # TODO check standard that logL_diff is compared by
        if np.abs(logL_diff1 - logL_diff2) > 2 * np.sqrt(logL_vars[-1, itrt]):
            print(
                'Probable imbalanced spacing at temperature %5d between T=%+.9e, T=%+.9e and T=%+.9e'
                % (itrt, mcc.Ts[itrt], mcc.Ts[itrt], mcc.Ts[itrt + 1])
            )
            imbalance[itrt] = True

    print('%5d imbalances found by mean likelihood' % (np.sum(imbalance)))
    if np.any(imbalance):
        print('Note: imbalances can be a sign of sub-optimal chain spacing')

    print('=========Exchange Rate Analysis==========')
    # TODO could also check left and right exchanges separately
    nn_exchanges = mcc.tracker_manager.get_exchange_rate_summary(0)[1]
    nn_exchange_var = np.var(nn_exchanges)
    nn_exchange_mean = np.mean(nn_exchanges)
    print('Mean nearest neighbor exchange rate is %.5f' % nn_exchange_mean)

    exchange_variance = np.abs(nn_exchanges - nn_exchange_mean) / np.sqrt(nn_exchange_var) > 3
    if np.any(exchange_variance):
        print(
            'Nearest neighbor exchange rate looks inhomogeneous, with %5d chains showing discrepant exchange rates'
            % (np.sum(exchange_variance))
        )
        print(
            'Note: inhomeogenous nearest-neighbor exchange rates can be a sign of sub-optimal chain spacing or disconnects'
        )
    else:
        print('Nearest neighbor exchange rate looks fairly homogeneous')

    low_exchange = nn_exchanges < 0.1
    if np.any(low_exchange):
        print('Some chains have very low nearest-neighbor exchange rates')
        print('Note: low nearest-neighbor exchange rates are a sign of poor mixing')

    large_exchange_change = np.abs(np.diff(nn_exchanges)) > 0.1
    large_exchange_change_up = np.hstack([large_exchange_change, False])
    large_exchange_change_down = np.hstack([False, large_exchange_change])
    # large_exchange_change_sym = large_exchange_change_up & large_exchange_change_down

    if np.any(large_exchange_change):
        print('Some chains have large changes in their nearest neighbor exchange_rate')
        print('Note: large changes in nearest-neighbor exchange rate can indicate a poorly resolved phase transition')

    for itrt in range(mcc.n_chain):
        if (
            large_exchange_change_up[itrt]
            or large_exchange_change_down[itrt]
            or exchange_variance[itrt]
            or low_exchange[itrt]
        ):
            print(
                'Chain %5d at T=%+.9e with nearest-neighbor exchange rate %.5f had an indication of exchange rate problems'
                % (itrt, mcc.Ts[itrt], nn_exchanges[itrt])
            )

    print('===========Heat Capacity Analysis=========')

    maxima_Ts, maxima_vals, prominences = th.find_potential_phase_transitions(mcc.betas, logL_vars[-1])
    potential_phase_transitions = prominences > 1.0
    print(
        'Heat capacity exhibits %5d maxima, of which %5d are potentially phase transitions'
        % (maxima_Ts.size, potential_phase_transitions.sum())
    )
    for itrp in range(maxima_Ts.size):
        if potential_phase_transitions[itrp]:
            lower_T = 1.0
            higher_T = np.inf
            if np.any(mcc.Ts <= maxima_Ts[itrp]):
                lower_T = np.max(mcc.Ts[mcc.Ts <= maxima_Ts[itrp]])
            if np.any(mcc.Ts >= maxima_Ts[itrp]):
                higher_T = np.min(mcc.Ts[mcc.Ts >= maxima_Ts[itrp]])
            print(
                'Possible C=%+.9e phase transition with prominence=%.9e near T=%+.9e'
                % (maxima_vals[itrp], prominences[itrp], maxima_Ts[itrp])
            )
            print(
                'Nearest chains: %5d at T=%+.9e and %5d at T=%.9e'
                % (np.argmax(mcc.Ts == lower_T), lower_T, np.argmax(mcc.Ts == higher_T), higher_T)
            )

    # TODO add a commentary comparing the cycle lengths to the DE buffer

    print('===========Cycle Analysis=========')
    n_cycles = mcc.tracker_manager.n_cycles
    n_cycles_tot = np.sum(n_cycles)
    if n_cycles_tot > 0:
        print('The sampler executed %8d total hot->cold->hot temperature cycles' % (n_cycles_tot))
        print('Cycle-based chain correlation length %.9f ' % (mcc.itrn / n_cycles_tot))
        # expected number of iterations for all chains to complete a cycle per the coupon collector problem
        memory_length = mcc.itrn / n_cycles_tot * mcc.n_chain * np.sum(1.0 / np.arange(1, mcc.n_chain + 1))
        print('Estimated # iterations to completely forget previous sampler state: %.13f' % (memory_length))
        if np.min(n_cycles) > 0:
            memory_length2 = mcc.itrn / n_cycles.min()
            print('Alternative estimate # iterations to forget previous sampler state: %.13f' % (memory_length2))
            if memory_length2 > 5.0 * memory_length or memory_length > 5.0 * memory_length2:
                print('Estimates of number of iterations to forget state vary significantly; may indicate poor burn in')
            if np.min(n_cycles) >= 5:
                cycle_deviations = (n_cycles - np.mean(n_cycles)) / np.std(n_cycles)
                if np.any(np.abs(cycle_deviations) > 3.0):
                    print('Some chains have significantly different numbers of cycles: may indicate poor burn in')
            else:
                print(
                    'Some chains have few cycles: running for more iterations would improve estimates of cycle length'
                )
        else:
            print('Some chains have no cycles: running for more iterations would improve estimate of cycle length')

        print(
            'Ideally, differential evolution buffer length de_size*de_thin would be longer than number of iterations to forget previous state'
        )
        print('Number of burn in samples discarded should also ideally be longer than memory length')
        print(
            'Chain with minimum # of cycles %8d, maximum # cycles %8d mean # cycles %.8f, std # cycles %.8f'
            % (n_cycles.min(), n_cycles.max(), n_cycles.mean(), np.std(n_cycles))
        )
    else:
        print(
            'Sampler executed no hot->cold->hot cycles, cannot estimate correlation length; consider running for longer, or adjusting parameters'
        )
        print('Sampler may not be ergodic if full cycles are not possible')
