"""C 2023 Matthew C. Digman
helpers to analyze the results of an mcmc chain
"""
import numpy as np
import scipy.signal
from numba import njit


@njit()
def get_blockwise_vars(N_blocks, n_burnin, samples_store, block_size, itrr, itrp, blockwise_vars, blockwise_means):
    """Get the variances for sequential blocks of samples"""
    for itrb in range(N_blocks):
        start1 = np.random.randint(n_burnin, samples_store.shape[0] - block_size)
        blockwise_vars[itrr, itrp, itrb] = np.var(samples_store[start1:start1 + block_size, :, itrp])
        blockwise_means[itrr, itrp, itrb] = np.mean(samples_store[start1:start1 + block_size, :, itrp])
    return blockwise_vars, blockwise_means


@njit()
def get_blockwise_vars_scramble(N_blocks, n_cold, n_burnin, samples_store, block_size, itrr, itrp, blockwise_vars_scramble, blockwise_means_scramble):
    """Get the variances for random blocks of samples"""
    dim1 = samples_store.shape[0] - n_burnin
    dim2 = n_cold
    for itrb in range(N_blocks):
        sample_block_loc = np.zeros(block_size * n_cold)
        for itrk in range(block_size * n_cold):
            targ1 = n_burnin + np.random.randint(0, dim1)
            targ2 = np.random.randint(0, dim2)
            sample_block_loc[itrk] = samples_store[targ1, targ2, itrp]

        blockwise_vars_scramble[itrr, itrp, itrb] = np.var(sample_block_loc)
        blockwise_means_scramble[itrr, itrp, itrb] = np.mean(sample_block_loc)
    return blockwise_vars_scramble, blockwise_means_scramble


def get_autocorr_sum(n_burnin, mcc, itrp, autocorr_sum) -> None:
    """Get the sum of the autocorrelations of all the cold chains"""
    for itrt in range(mcc.n_cold):
        params_adj = mcc.samples_store[n_burnin:, itrt, itrp] - np.mean(mcc.samples_store[n_burnin:, itrt, itrp])
        autocorr_sum += scipy.signal.correlate(params_adj, params_adj, mode='full')
