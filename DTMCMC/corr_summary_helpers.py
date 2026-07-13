"""C 2023 Matthew C. Digman
helpers to summarize the auto and cross-correlations and sampling efficiency for an mcmc run
"""

from typing import TYPE_CHECKING

import numpy as np
import scipy.signal

from DTMCMC.chain_analysis_helpers import StoreView, get_autocorr_sum, get_blockwise_vars, get_blockwise_vars_scramble

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from DTMCMC.dtmcmc_sampler import DTMCMCSampler
    from DTMCMC.tracker_manager import TrackerManager


def restrict_n_burnin(mcc: DTMCMCSampler | StoreView, n_burnin: int) -> int:
    """Helper to restrict n_burnin to last block"""
    if mcc.store_size * mcc.store_thin < n_burnin:
        # handle burning more than 1 entire storage block
        if mcc.itrn - mcc.store_size * mcc.store_thin < n_burnin:
            # amount of elements to burn from last storage block
            n_burnin = n_burnin - (mcc.itrn - mcc.store_size * mcc.store_thin)
        else:
            # no elements need to be burned from last storage block
            n_burnin = 0

    return n_burnin


def autocorr_helper(
    mcc: DTMCMCSampler | StoreView, itrp: int, n_burnin_thin: int
) -> tuple[NDArray[np.floating], int, float]:
    """Helper to get the autocorrleation functions for a particular parameter"""
    n_use: int = mcc.store_size - n_burnin_thin
    autocorr_sum: NDArray[np.floating] = np.zeros((n_use - 1) * 2 + 1)
    get_autocorr_sum(n_burnin_thin, mcc, itrp, autocorr_sum)
    autocorr_lim: NDArray[np.floating] = np.hstack(
        [
            autocorr_sum[n_use - 1 : n_use],
            autocorr_sum[n_use : 2 * n_use - 2 : 2] + autocorr_sum[n_use + 1 : 2 * n_use - 1 : 2],
        ]
    )
    autocorr_cut: int = 1 + int(np.argmax(autocorr_lim[1:] < 0.0))
    est_var_auto: float = autocorr_lim[0] + 2 * np.sum(autocorr_lim[1:autocorr_cut])
    return autocorr_lim, autocorr_cut, est_var_auto


def get_crosscorr_sum(
    mcc: DTMCMCSampler | StoreView,
    n_burnin_thin: int,
    itrp: int,
    autocorr_lim: NDArray[np.floating],
    autocorr_cut: int,
    obs_var: NDArray[np.floating],
    n_eff_pred_auto: NDArray[np.floating],
) -> tuple[NDArray[np.floating], int, float]:
    """Estimate the average cross correlations"""
    n_use: int = mcc.store_size - n_burnin_thin
    n_cold: int = mcc.n_cold
    n_cross_eval: int = min(64, n_cold)  # don't go too large or it takes a very long time
    n_chain: int = mcc.n_chain
    block_size: int = mcc.block_size
    n_tot: int = n_use * n_cold

    cov_cross_sum: NDArray[np.floating] = np.zeros((n_use - 1) * 2 + 1)

    for itrt1 in range(n_cross_eval):
        params_adj1: NDArray[np.floating] = mcc.samples_store[n_burnin_thin:, itrt1, itrp] - np.mean(
            mcc.samples_store[n_burnin_thin:, itrt1, itrp]
        )
        for itrt2 in range(itrt1 + 1, n_cross_eval):
            params_adj2: NDArray[np.floating] = mcc.samples_store[n_burnin_thin:, itrt2, itrp] - np.mean(
                mcc.samples_store[n_burnin_thin:, itrt2, itrp]
            )
            corr_loc: NDArray[np.floating] = scipy.signal.correlate(params_adj1, params_adj2, mode='full')
            cov_cross_sum += corr_loc
            cov_cross_sum += corr_loc[::-1]  # for the itrt2,itr1 correlation

    # TODO check
    if n_cross_eval < n_cold:
        cov_cross_sum *= (n_cold**2 - n_cold) / (n_cross_eval**2 - n_cross_eval)
    cov_cross_lim: NDArray[np.floating] = np.hstack(
        [
            cov_cross_sum[n_use - 1 : n_use],
            cov_cross_sum[n_use : 2 * n_use - 2 : 2] + cov_cross_sum[n_use + 1 : 2 * n_use - 1 : 2],
        ]
    )
    cov_cross_cut_last: int = n_use // 2 - 2 * block_size
    cov_cut_std_thresh: float = 10.0
    std_comp: NDArray[np.floating] = (
        np.sqrt(n_tot / n_eff_pred_auto[itrp])
        * obs_var[itrp]
        * np.sqrt(2 * (n_chain**2 - n_chain) * np.arange(n_use - 2, n_use - 2 * cov_cross_cut_last, -2))
    )

    if np.any(np.abs(cov_cross_lim[1:cov_cross_cut_last][::-1]) > cov_cut_std_thresh * std_comp[::-1]):
        cut_from_back1: int = int(
            np.argmax(np.abs(cov_cross_lim[1:cov_cross_cut_last][::-1]) > cov_cut_std_thresh * std_comp[::-1])
        )
    else:
        cut_from_back1 = 0

    if cut_from_back1 == 0:
        cut_from_back1 = cov_cross_cut_last

    cut_cond1: NDArray[np.bool] = np.abs(autocorr_lim[0]) * 1.0e-1 < np.abs(cov_cross_lim[1:cov_cross_cut_last][::-1])
    if (
        autocorr_cut > 10
        and float(np.max(np.abs(cov_cross_lim))) > 1.0e-1 * np.abs(autocorr_lim[0])
        and np.any(cut_cond1)
    ):
        cut_from_back2: int = int(np.argmax(cut_cond1))
    else:
        cut_from_back2 = cov_cross_cut_last
    cut_from_back = min(cut_from_back1, cut_from_back2)

    cov_cross_cut = 1 + cov_cross_cut_last - cut_from_back

    est_var_cross: float = cov_cross_lim[0] + 2 * float(np.sum(cov_cross_lim[1:cov_cross_cut]))
    return cov_cross_lim, cov_cross_cut, est_var_cross


def n_eff_summary_print(
    n_par: int,
    n_use: int,
    n_cold: int,
    n_chain: int,
    store_thin: int,
    n_eff_preds: NDArray[np.floating],
    n_eff_preds_empirical: NDArray[np.floating],
    obs_vars: NDArray[np.floating],
    obs_means: NDArray[np.floating],
) -> None:
    """Print salient information about the number of effective samples"""
    eff_empiricals = np.zeros(n_par)
    eff_preds = np.zeros(n_par)
    eff_overalls = np.zeros(n_par)

    eff_empirical_mean = 0.0
    eff_pred_mean = 0.0
    eff_overall_mean = 0.0

    # can't quote multi-run efficiencies if only one run was done
    overall_usable = obs_vars.shape[0] > 1

    for itrp in range(n_par):
        obs_varf = np.var(obs_means[:, itrp])
        if overall_usable:
            eff_overalls[itrp] = 2.0 * np.mean(obs_vars[:, itrp]) / obs_varf / (n_use * n_chain * store_thin)
            eff_overall_mean += eff_overalls[itrp]

        eff_preds[itrp] = 2 * np.mean(n_eff_preds[:, itrp]) / (n_use * n_chain * store_thin)
        eff_empiricals[itrp] = 2 * np.mean(n_eff_preds_empirical[:, itrp]) / (n_use * n_chain * store_thin)
        eff_pred_mean += eff_preds[itrp]
        eff_empirical_mean += eff_empiricals[itrp]

    print(n_cold, n_chain)
    eff_empirical_mean /= n_par
    eff_pred_mean /= n_par
    eff_overall_mean /= n_par

    eff_empirical_string = ''
    eff_overall_string = ''
    eff_preds_string = ''
    for itrp in range(n_par):
        eff_empirical_string = eff_empirical_string + ' %+.5e' % eff_empiricals[itrp]
        eff_overall_string = eff_overall_string + ' %+.5e' % eff_overalls[itrp]
        eff_preds_string = eff_preds_string + ' %+.5e' % eff_preds[itrp]

    print('correlation efficiencies' + eff_preds_string)
    print('empirical   efficiencies' + eff_empirical_string)
    if overall_usable:
        print('overall     efficiencies' + eff_overall_string)
    print('mean correlation efficiency %.5e' % (eff_pred_mean))
    print('mean empirical   efficiency %.5e' % (eff_empirical_mean))
    if overall_usable:
        print('mean overall     efficiency %.5e' % (eff_overall_mean))

    print('overall effective sample sizes:', n_eff_preds_empirical[0])


def autocorr_summary_print(n_par: int, autocorr_lims: list[NDArray[np.floating]], do_cross: bool) -> None:
    """Print useful information about the autocorrelations"""
    autocorr_lim_array: NDArray[np.floating] = np.array(autocorr_lims)
    crosscorr_lim_array: NDArray[np.floating] = np.array(autocorr_lims)

    autocorr_lim_means = np.zeros((n_par, autocorr_lims[0].size))
    crosscorr_lim_means = np.zeros((n_par, autocorr_lims[0].size))
    autocorr_cut_means = np.zeros(n_par, dtype=np.int64)
    autocorr_len_means = np.zeros(n_par)

    autocorr_len_str = ''
    for itrp in range(n_par):
        autocorr_lim_means[itrp] = autocorr_lim_array[itrp::n_par].mean(axis=0)
        if do_cross:
            crosscorr_lim_means[itrp] = crosscorr_lim_array[itrp::n_par].mean(axis=0)
        autocorr_cut_means[itrp] = 1 + int(np.argmax(autocorr_lim_means[itrp, 1:] < 0.0))
        autocorr_len_means[itrp] = (
            autocorr_lim_means[itrp, 0] + 2 * np.sum(autocorr_lim_means[itrp, 1 : autocorr_cut_means[itrp]])
        ) / autocorr_lim_means[itrp, 0]
        autocorr_len_str = autocorr_len_str + ' %.8e' % autocorr_len_means[itrp]

    print('best estimate of autocorrelation lengths:', autocorr_len_str)


def summarize_logLs(mcc: DTMCMCSampler | StoreView, N_blocks: int) -> tuple[NDArray[np.floating], int, int]:
    """Get useful summary statistics about the likelihoods"""
    block_size: int = mcc.block_size // mcc.store_thin
    logL_block_mean: NDArray[np.floating] = np.zeros(N_blocks)
    for itrk in range(N_blocks):
        logL_block_mean[itrk] = np.mean(mcc.logLs_store[itrk * block_size : (itrk + 1) * block_size])

    logL_mean: float = float(np.mean(logL_block_mean[-10:]))
    logL_std: float = float(np.std(logL_block_mean[-10:]))
    if np.any(logL_block_mean > logL_mean - logL_std):
        arg_logL_burn: int = int(np.argmax(logL_block_mean > logL_mean - logL_std))
    else:
        print('logL never burned in')
        arg_logL_burn = -1
    if np.any(logL_block_mean < logL_mean - 5 * logL_std):
        arg_logL_deviant: int = (
            logL_block_mean.size - int(np.argmax(logL_block_mean[::-1] < logL_mean - 6 * logL_std)) - 1
        )
    else:
        arg_logL_deviant = -1
    return logL_block_mean, arg_logL_burn, arg_logL_deviant


def summarize_vars(
    mcc: DTMCMCSampler | StoreView, n_burnin_thin: int
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Get the means and variances for the samples"""
    n_par = mcc.n_par
    obs_means = np.mean(mcc.samples_store[n_burnin_thin:, :, :], axis=0).mean(axis=0)
    obs_vars = np.zeros(n_par)
    for itrp in range(n_par):
        obs_vars[itrp] = np.var(mcc.samples_store[n_burnin_thin:, :, itrp])

    return obs_means, obs_vars


class CorrelationSummary:
    """class to store various attributes memorializing the correlations of a chain across multiple runs"""

    def __init__(self, do_corr_summary: bool = True, do_autocorr: bool = True, do_cross: bool = True) -> None:
        """Create the class instance"""
        self.do_corr_summary: bool = do_corr_summary
        self.do_cross: bool = do_cross
        self.do_autocorr: bool = do_autocorr
        self.blockwise_vars: list[NDArray[np.floating]] = []
        self.blockwise_means: list[NDArray[np.floating]] = []
        self.blockwise_vars_scramble: list[NDArray[np.floating]] = []
        self.blockwise_means_scramble: list[NDArray[np.floating]] = []
        self.n_eff_preds: list[NDArray[np.floating]] = []
        self.n_eff_preds_auto: list[NDArray[np.floating]] = []
        self.n_eff_preds_empirical: list[NDArray[np.floating]] = []
        self.est_vars_cross: list[NDArray[np.floating]] = []
        self.est_vars_auto: list[NDArray[np.floating]] = []
        self.est_vars: list[NDArray[np.floating]] = []
        self.autocorr_lims: list[NDArray[np.floating]] = []
        self.cov_cross_lims: list[NDArray[np.floating]] = []
        self.obs_means: list[NDArray[np.floating]] = []
        self.obs_vars: list[NDArray[np.floating]] = []
        self.logL_block_means: list[NDArray[np.floating]] = []
        self.arg_logL_burns: list[int] = []
        self.arg_logL_deviant: list[int] = []

    def final_prints(self, mcc: DTMCMCSampler | StoreView, n_burnin: int) -> None:
        """Printouts to do after all the runs have been done"""
        if self.do_corr_summary:
            self.n_eff_summary_print(mcc, n_burnin)
            self.autocorr_summary_print(mcc)

    def summarize_blocks(self, mcc: DTMCMCSampler | StoreView, tracker_manager: TrackerManager, n_burnin: int) -> None:
        """Summary functions that can be printed after a run has been executed"""
        self.summarize_logLs(mcc)
        self.summarize_vars(mcc, n_burnin)

        if self.do_corr_summary:
            self.corr_summary(mcc, n_burnin)

        print('last two Ts', mcc.Ts[-2], mcc.Ts[-1])
        print('logL burns', np.array(self.arg_logL_burns))
        n_complete_hc_cycles = tracker_manager.n_cycles
        print(n_complete_hc_cycles.sum())

    def corr_summary(self, mcc: DTMCMCSampler | StoreView, n_burnin: int) -> None:
        """The summaries of correlations that need to be computed after every run"""
        n_par: int = mcc.n_par
        n_cold: int = mcc.n_cold
        n_burnin_thin: int = restrict_n_burnin(mcc, n_burnin) // mcc.store_thin
        n_use: int = mcc.store_size - n_burnin_thin
        n_tot: int = n_use * n_cold
        block_size: int = mcc.block_size // mcc.store_thin
        N_blocks: int = mcc.store_size // block_size

        blockwise_vars: NDArray[np.floating] = np.zeros((1, n_par, N_blocks))
        blockwise_means: NDArray[np.floating] = np.zeros((1, n_par, N_blocks))
        blockwise_vars_scramble: NDArray[np.floating] = np.zeros((1, n_par, N_blocks))
        blockwise_means_scramble: NDArray[np.floating] = np.zeros((1, n_par, N_blocks))
        n_eff_preds: NDArray[np.floating] = np.zeros(n_par)
        n_eff_preds_auto: NDArray[np.floating] = np.zeros(n_par)
        n_eff_preds_empirical: NDArray[np.floating] = np.zeros(n_par)

        obs_var_loc = self.obs_vars[-1]

        est_vars_cross: NDArray[np.floating] = np.zeros(n_par)
        est_vars_auto: NDArray[np.floating] = np.zeros(n_par)
        est_vars: NDArray[np.floating] = np.zeros(n_par)

        for itrp in range(n_par):
            est_vars_cross[itrp] = 0.0

            get_blockwise_vars(
                N_blocks, n_burnin_thin, mcc.samples_store, block_size, 0, itrp, blockwise_vars, blockwise_means
            )
            get_blockwise_vars_scramble(
                N_blocks,
                n_cold,
                n_burnin_thin,
                mcc.samples_store,
                block_size,
                0,
                itrp,
                blockwise_vars_scramble,
                blockwise_means_scramble,
            )

            n_eff_preds_empirical[itrp] = (
                np.var(blockwise_means_scramble[0, itrp, :]) / np.var(blockwise_means[0, itrp, :]) * n_tot
            )

            if self.do_autocorr:
                autocorr_lim, autocorr_cut, est_vars_auto[itrp] = autocorr_helper(mcc, itrp, n_burnin_thin)
                self.autocorr_lims.append(autocorr_lim)

                n_eff_preds_auto[itrp] = n_tot / (est_vars_auto[itrp] / autocorr_lim[0])

                if self.do_cross:
                    cov_cross_lim, _cov_cross_cut, est_vars_cross[itrp] = get_crosscorr_sum(
                        mcc, n_burnin_thin, itrp, autocorr_lim, autocorr_cut, obs_var_loc, n_eff_preds_auto
                    )
                    self.cov_cross_lims.append(cov_cross_lim)

                est_vars[itrp] = est_vars_auto[itrp] + est_vars_cross[itrp]

                n_eff_preds[itrp] = n_tot / (est_vars[itrp] / autocorr_lim[0])  # TODO double check factor of two

        self.blockwise_vars.append(blockwise_vars[0])
        self.blockwise_means.append(blockwise_means[0])
        self.blockwise_vars_scramble.append(blockwise_vars_scramble[0])
        self.blockwise_means_scramble.append(blockwise_means_scramble[0])
        self.n_eff_preds.append(n_eff_preds)
        self.n_eff_preds_auto.append(n_eff_preds_auto)
        self.n_eff_preds_empirical.append(n_eff_preds_empirical)
        self.est_vars_cross.append(est_vars_cross)
        self.est_vars_auto.append(est_vars_auto)
        self.est_vars.append(est_vars)

    def summarize_vars(self, mcc: DTMCMCSampler | StoreView, n_burnin: int) -> None:
        """Get the means and vars for the whole run"""
        n_burnin_thin: int = restrict_n_burnin(mcc, n_burnin) // mcc.store_thin
        obs_mean, obs_var = summarize_vars(mcc, n_burnin_thin)
        self.obs_means.append(obs_mean)
        self.obs_vars.append(obs_var)

    def n_eff_summary_print(self, mcc: DTMCMCSampler | StoreView, n_burnin: int) -> None:
        """Print salient information about the number of effective samples"""
        n_burnin_thin: int = restrict_n_burnin(mcc, n_burnin) // mcc.store_thin
        n_par: int = mcc.n_par
        n_cold: int = mcc.n_cold
        n_chain: int = mcc.n_chain
        n_use: int = mcc.store_size - n_burnin_thin
        n_eff_summary_print(
            n_par,
            n_use,
            n_cold,
            n_chain,
            mcc.store_thin,
            np.array(self.n_eff_preds),
            np.array(self.n_eff_preds_empirical),
            np.array(self.obs_vars),
            np.array(self.obs_means),
        )

    def autocorr_summary_print(self, mcc: DTMCMCSampler | StoreView) -> None:
        """Print salient information about autocorrelation functions"""
        autocorr_summary_print(mcc.n_par, self.autocorr_lims, self.do_cross)

    def summarize_logLs(self, mcc: DTMCMCSampler | StoreView) -> None:
        """Save some summary statistics related to the likelihoods"""
        N_blocks = mcc.store_size // (mcc.block_size // mcc.store_thin)
        logL_block_mean, arg_logL_burn, arg_logL_deviant = summarize_logLs(mcc, N_blocks)
        self.logL_block_means.append(logL_block_mean)
        self.arg_logL_burns.append(arg_logL_burn)
        self.arg_logL_deviant.append(arg_logL_deviant)

    def restrict_n_burnin(self, mcc: DTMCMCSampler | StoreView, n_burnin: int) -> int:
        """Restrict n_burnin based on storage size"""
        return restrict_n_burnin(mcc, n_burnin)
