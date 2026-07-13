"""Implementation of the 1 and 2 sample nearest neigbor approximations of sample entropy, and the corresponding divergence"""

import numpy as np
import scipy.special
from numba import njit, prange

# TODO make robust against number of parameters out of range
# TODO should use log gamma directly so does not have limit of 341 for overflow
n_par_max = 341
n_par_range = np.arange(0, n_par_max + 1)
ce = 0.57721566490153286060651209008240243104215933593992  # eulers constant
logc1 = np.log(np.pi ** (n_par_range / 2) / scipy.special.gamma(n_par_range / 2 + 1))
par_large = 1e9
dist_large = n_par_range * par_large**2


@njit()
def NNEntropy1(samples):
    """Get entropy of IID samples from the posterior based on eq. 5 of https://hal.archives-ouvertes.fr/hal-02774953/document"""
    n_par = samples.shape[1]
    assert n_par <= n_par_max
    n_samp = samples.shape[0]
    entropy_sum = 0.0
    for itrn in prange(n_samp):  # type: ignore[not-iterable]
        # find the distance to the nearest neighbor to sample itrn
        samples_cur = samples[itrn].copy()
        samples[itrn, :] = (
            par_large  # just something large to make sure this isn't the nearest neighbor without explicitly excising for efficiency
        )
        # dist_sq_min = np.min(np.sum((samples_cur-samples)**2,axis=1))
        dist_sq_min = dist_large[n_par]  # n_par*par_large**2
        for itrn2 in range(n_samp):
            dist_sq = 0.0
            for itrp in range(n_par):
                dist_sq += (samples_cur[itrp] - samples[itrn2, itrp]) ** 2

            if dist_sq < dist_sq_min:
                dist_sq_min = dist_sq
        # if itrn>0:
        #    distsq1 = np.min(np.sum((samples_cur-samples[:itrn])**2,axis=1))
        # else:
        #    distsq1 = np.inf
        # if itrn+1<n_samp:
        #    distsq2 = np.min(np.sum((samples_cur-samples[itrn+1:])**2,axis=1))
        # else:
        #    distsq2 = np.inf
        samples[itrn, :] = samples_cur
        # distsq_min = min(distsq1,distsq2)
        entropy_contrib_loc = np.log(dist_sq_min)
        entropy_sum += entropy_contrib_loc
    entropy = n_par / n_samp / 2 * entropy_sum + np.log(n_samp - 1) + logc1[n_par] + ce
    # entropy = 1/n_samp*entropy_sum+np.log(2)+ce
    return entropy


@njit()
def NNEntropy2(samples1, samples2):
    """Get entropy of arbitrary test sample samples2 given IID samples samples1 of the same size
    based on eq. 6 of https://hal.archives-ouvertes.fr/hal-02774953/document
    """
    n_par1 = samples1.shape[1]
    n_par2 = samples2.shape[1]
    assert n_par1 == n_par2
    assert n_par1 <= n_par_max
    assert samples1.shape[0] == samples2.shape[0]
    n_par = n_par1
    n_samp = samples2.shape[0]
    entropy_sum = 0.0
    for itrn in prange(n_samp):  # type: ignore[not-iterable]
        # find the distance to the nearest neighbor to sample itrn
        samples_cur = samples2[itrn]
        dist_sq_min = dist_large[n_par]
        for itrn2 in range(n_samp):
            dist_sq = 0.0
            for itrp in range(n_par):
                dist_sq += (samples_cur[itrp] - samples1[itrn2, itrp]) ** 2
            if dist_sq < dist_sq_min:
                dist_sq_min = dist_sq
        # distsq_min = np.min(np.sum((samples2[itrn]-samples1)**2,axis=1))
        entropy_contrib_loc = np.log(dist_sq_min)
        entropy_sum += entropy_contrib_loc
    entropy = n_par / n_samp / 2 * entropy_sum + np.log(n_samp - 1) + logc1[n_par] + ce
    return entropy


@njit()
def NNEntropyK(samples1, samples2):
    """Get two sample estimate of relative entropy assuming samples1 contains IID posterior draws
    and samples2 is the distribution of samples to be tested,
    see eq. 7 of https://hal.archives-ouvertes.fr/hal-02774953/document
    """
    n_par1 = samples1.shape[1]
    n_par2 = samples2.shape[1]
    assert n_par1 == n_par2
    assert n_par1 <= n_par_max
    e1s = NNEntropy1(samples1)
    e2s = NNEntropy2(samples1, samples2)
    return e1s - e2s, e1s, e2s


def unit_normal_battery(signal, mult=1.0, sig_thresh=5.0, A2_cut=2.28, do_assert=True) -> bool:
    """Battery of tests for checking if signal is unit normal white noise"""
    # default anderson darling cutoff of 2.28 is hand selected to
    # give ~1 in 1e5 empirical probablity of false positive for n=64
    # calibration looks about same for n=32 could probably choose better way
    # with current defaults that should make it the most sensitive test
    n_sig = signal.size

    sig_adjust = signal / mult
    mean_wave = np.mean(sig_adjust)
    std_wave = np.std(sig_adjust)
    std_std_wave = np.std(sig_adjust) * np.sqrt(2 / n_sig)
    # check mean and variance

    # D'Agostino and Pearson's test for skew and kurtosis
    # doesn't seem to respect calibration of p value very well
    # p_skew = scipy.stats.normaltest(sig_adjust).pvalue
    # assert p_skew>p_thresh

    # anderson darling test statistic assuming true mean and variance are unknown
    sig_sort = np.sort((sig_adjust - mean_wave) / std_wave)
    phis = scipy.stats.norm.cdf(sig_sort)
    A2 = -n_sig - 1 / n_sig * np.sum(
        (2 * np.arange(1, n_sig + 1) - 1) * np.log(phis)
        + (2 * (n_sig - np.arange(1, n_sig + 1)) + 1) * np.log(1 - phis)
    )
    A2Star = A2 * (1 + 4 / n_sig - 25 / n_sig**2)
    if (
        np.any(np.isnan(phis))
        or A2Star >= A2_cut
        or np.abs(mean_wave) / std_wave >= sig_thresh
        or np.abs(std_wave - 1.0) / std_std_wave >= sig_thresh
    ):
        print('failed', A2Star, A2_cut, std_wave, np.abs(mean_wave) / std_wave, np.abs(std_wave - 1.0) / std_std_wave)
        if do_assert:
            assert A2Star < A2_cut  # should be less than cutoff value
            assert np.abs(mean_wave) / std_wave < sig_thresh
            assert np.abs(std_wave - 1.0) / std_std_wave < sig_thresh
        else:
            print(A2Star < A2_cut)  # should be less than cutoff value
            print(np.abs(mean_wave) / std_wave < sig_thresh)
            print(np.abs(std_wave - 1.0) / std_std_wave < sig_thresh)
        return False
    return True
