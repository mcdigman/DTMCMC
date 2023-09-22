"""C 2023 Matthew C. Digman
helpers to perform the parallel tempering exchanges"""
import numpy as np
from numba import njit


RANDOM_TARGETS = 0      # uniformlly random exchange targetting
SEQUENTIAL_TARGETS = 1  # target sequentially from back to front
ADJACENT_TARGETS = 2    # target alternating +/- 1 positions
NULL_TARGETS = 3        # do not do any exchanges


class ExchangeManager():
    """class to take a temperature ladder and state of a chain and define the strategy by which to propose exchanges"""

    def __init__(self, strategy=RANDOM_TARGETS, track_full_exchanges=True):
        """select the exchange targeting strategy"""
        self.strategy = strategy
        self.track_full_exchanges = track_full_exchanges

    def do_ptmcmc_exchange(self, itrb, samples, logLs, T_ladder, exchange_tracker, chain_track):
        """do the exchange step"""
        assert self.is_exchange_step(itrb)#itrb % 2 == 0
        return do_ptmcmc_exchange(itrb-1, samples, logLs, T_ladder.n_chain, T_ladder.betas, exchange_tracker, chain_track, self.strategy, self.track_full_exchanges)

    def is_exchange_step(self, itrb):
        """check whether the step with the given index should be an exchange step, currently based on alternating even and odd"""
        return itrb % 2 == 0
        #return itrb % 4 in (1,2)


@njit()
def exchange_step_helper(logLs_loc, betas, exchange_tracker, exchange_order, targets, no_repeat, track_full_exchanges):
    """actually execute the swaps for an exchange step"""
    n_chain = betas.shape[0]

    itrs_fin = np.arange(0, n_chain)
    for idxt in range(0, n_chain):
        itrt = exchange_order[idxt]
        itrt_target = targets[itrt]
        if no_repeat and itrt > itrt_target:
            # prevent random targetting from undoing the exchange it just proposed
            continue
        if itrt == itrt_target:
            # not a real proposal
            continue
        if not no_repeat:
            assert itrs_fin[itrt_target] == itrt_target
        else:
            assert targets[itrt_target] == itrt
            assert itrs_fin[itrt_target] == itrt_target
            assert itrs_fin[itrt] == itrt

        log_accept_prob_exchange = np.log(np.random.uniform(0., 1.))
        log_mh_ratio_exchange = betas[itrt]*(logLs_loc[itrt_target]-logLs_loc[itrt])+betas[itrt_target]*(logLs_loc[itrt]-logLs_loc[itrt_target])
        if log_mh_ratio_exchange > log_accept_prob_exchange:
            logLs_hold = logLs_loc[itrt_target]
            logLs_loc[itrt_target] = logLs_loc[itrt]
            logLs_loc[itrt] = logLs_hold

            itr_hold = itrs_fin[itrt]
            itrs_fin[itrt] = itrs_fin[itrt_target]
            itrs_fin[itrt_target] = itr_hold

            if track_full_exchanges:
                # track full exchange matrix
                exchange_tracker[0, itrt, itrt_target] += 1
            else:
                # track all exchanges for each individual chain
                exchange_tracker[0, 0, itrt] += 1
                exchange_tracker[0, 0, itrt_target] += 1
                # track nn exchanges
                if itrt_target == itrt+1 or itrt_target==itrt-1:
                    exchange_tracker[1, 0, itrt] += 1
                    exchange_tracker[1, 0, itrt_target] += 1
        else:
            if track_full_exchanges:
                # track full exchange matrix
                exchange_tracker[1, itrt, itrt_target] += 1
            else:
                # track all exchanges for each individual chain
                exchange_tracker[0, 1, itrt] += 1
                exchange_tracker[0, 1, itrt_target] += 1
                # track nn exchanges
                if itrt_target == itrt+1 or itrt_target==itrt-1:
                    exchange_tracker[1, 1, itrt] += 1
                    exchange_tracker[1, 1, itrt_target] += 1

    return itrs_fin


@njit()
def random_pair_generate(n_chain):
    """pairs are generated uniformally at random"""
    target_shuffle = np.random.permutation(np.arange(0, n_chain))
    target_shuffle = np.concatenate((target_shuffle[::2], target_shuffle[1::2]))

    targets = np.zeros(n_chain, dtype=np.int64)
    targets[target_shuffle[:n_chain//2]] = target_shuffle[n_chain//2:n_chain]
    targets[target_shuffle[n_chain//2:n_chain]] = target_shuffle[:n_chain//2]
    exchange_order = np.arange(0, n_chain)
    return targets, exchange_order


@njit()
def offset_pair_generate(n_chain, offset):
    """pairs are generated as [(0,offset+1),(2,offset+3),...(n_chain-2,offset+n_chain-1)]%n_chain, e.g.,
       offset = 0 corresponds to pairs [(0,1),(2,3),...(n_chain-2,n_chain-1)]
       offset = -1 corresponds to       [(n_chain-1,0),(1,2),...(n_chain-3,n_chain-1)]"""
    # can only handle offset pairs for integer divisors
    if offset >= 0:
        assert n_chain % (offset+1) == 0
    else:
        assert n_chain % (np.abs(offset)) == 0
    targets = np.zeros(n_chain, dtype=np.int64)-1
    if offset >= 0:
        ctr = (offset+1) % n_chain
    else:
        ctr = offset % n_chain
    for itrm in range(0, n_chain):
        if offset >= 0:
            check = (itrm//(offset+1) % 2) == 0
        else:
            check = ((itrm-np.abs(offset))//(np.abs(offset)) % 2) != 0

        if check and targets[itrm] == -1:
            targets[itrm] = ctr
            targets[ctr] = itrm
        ctr += 1
        ctr %= n_chain
    exchange_order = np.arange(0, n_chain)
    return targets, exchange_order


@njit()
def do_ptmcmc_exchange(itrb, samples, logLs, n_chain, betas, exchange_tracker, chain_track, target_select, track_full_exchanges):
    """chose and exchange strategy and do the exchange step"""

    no_repeat = True
    if target_select == RANDOM_TARGETS:
        # random exchange pairs
        targets, exchange_order = random_pair_generate(n_chain)
    elif target_select == SEQUENTIAL_TARGETS:
        # target from back to front, results in repeated exchanges
        targets = np.arange(-1, n_chain-1)
        exchange_order = np.arange(n_chain-1, -1, -1)
        targets[0] = 0
        no_repeat = False
    elif target_select == ADJACENT_TARGETS:
        # alternate targeting exchanges at distance +/-1
        if itrb % 4 == 1:
            targets, exchange_order = offset_pair_generate(n_chain, 0)
        else:
            targets, exchange_order = offset_pair_generate(n_chain, -1)
    elif target_select == NULL_TARGETS:
        # do not actually propose any exchanges
        targets = np.arange(0, n_chain)
        exchange_order = np.arange(0, n_chain)
    else:
        assert False

    logLs_cur = np.zeros(n_chain)
    logLs_cur[:] = logLs[itrb]

    itrs_fin = exchange_step_helper(logLs_cur, betas, exchange_tracker, exchange_order, targets, no_repeat, track_full_exchanges)

    for itrt in range(0, n_chain):
        logLs[itrb+1, itrt] = logLs[itrb, itrs_fin[itrt]]
        samples[itrb+1, itrt] = samples[itrb, itrs_fin[itrt]]
        chain_track[itrb+1, itrt] = chain_track[itrb, itrs_fin[itrt]]
