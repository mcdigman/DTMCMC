"""C 2023 Matthew C. Digman
helpers to summarize the auto and cross-correlations and sampling efficiency for an mcmc run"""
import numpy as np
import scipy.signal
from DTMCMC.chain_analysis_helpers import get_blockwise_vars, get_blockwise_vars_scramble, get_autocorr_sum


class CorrelationSummary():
    """class to store various attributes memorializing the correlations of a chain across multiple runs"""

    def __init__(self, do_corr_summary=True, do_autocorr=True, do_cross=True):
        """create the class instance"""
        self.do_corr_summary = do_corr_summary
        self.do_cross = do_cross
        self.do_autocorr = do_autocorr
        self.blockwise_vars = []
        self.blockwise_means = []
        self.blockwise_vars_scramble = []
        self.blockwise_means_scramble = []
        self.n_eff_preds = []
        self.n_eff_preds_auto = []
        self.n_eff_preds_empirical = []
        self.est_vars_cross = []
        self.est_vars_auto = []
        self.est_vars = []
        self.autocorr_lims = []
        self.cov_cross_lims = []
        self.obs_means = []
        self.obs_vars = []
        self.logL_block_means = []
        self.arg_logL_burns = []
        self.arg_logL_deviant = []

    def final_prints(self, mcc, n_burnin):
        """printouts to do after all the runs have been done"""
        if self.do_corr_summary:
            self.n_eff_summary_print(mcc, n_burnin)
            self.autocorr_summary_print(mcc)

    def summarize_blocks(self, mcc, n_burnin):
        """summary functions that can be printed after a run has been executed"""
        self.summarize_logLs(mcc)
        self.summarize_vars(mcc, n_burnin)

        if self.do_corr_summary:
            self.corr_summary(mcc, n_burnin)

        print("last two Ts", mcc.Ts[-2], mcc.Ts[-1])
        print("logL burns", np.array(self.arg_logL_burns))
        n_complete_hc_cycles = mcc.tracker_manager.get_n_cycles()
        print(n_complete_hc_cycles.sum())

    def corr_summary(self, mcc, n_burnin):
        """the summaries of correlations that need to be computed after every run"""
        n_par = mcc.n_par
        n_cold = mcc.n_cold
        n_burnin_thin = restrict_n_burnin(mcc, n_burnin)//mcc.store_thin
        n_use = mcc.store_size+1-n_burnin_thin
        n_tot = n_use*n_cold
        block_size = mcc.block_size//mcc.store_thin
        N_blocks = mcc.store_size//block_size

        blockwise_vars = np.zeros((1, n_par, N_blocks))
        blockwise_means = np.zeros((1, n_par, N_blocks))
        blockwise_vars_scramble = np.zeros((1, n_par, N_blocks))
        blockwise_means_scramble = np.zeros((1, n_par, N_blocks))
        n_eff_preds = np.zeros(n_par)
        n_eff_preds_auto = np.zeros(n_par)
        n_eff_preds_empirical = np.zeros(n_par)

        obs_var_loc = self.obs_vars[-1]

        est_vars_cross = np.zeros(n_par)
        est_vars_auto = np.zeros(n_par)
        est_vars = np.zeros(n_par)

        for itrp in range(n_par):
            est_vars_cross[itrp] = 0.
            cov_cross_cut = 0

            get_blockwise_vars(N_blocks, n_burnin_thin, mcc.samples_store, block_size, 0, itrp, blockwise_vars, blockwise_means)
            get_blockwise_vars_scramble(N_blocks, n_cold, n_burnin_thin, mcc.samples_store, block_size, 0, itrp, blockwise_vars_scramble, blockwise_means_scramble)

            n_eff_preds_empirical[itrp] = np.var(blockwise_means_scramble[0, itrp, :])/np.var(blockwise_means[0, itrp, :])*n_tot

            if self.do_autocorr:
                autocorr_lim, autocorr_cut, est_vars_auto[itrp] = autocorr_helper(mcc, itrp, n_burnin_thin)
                self.autocorr_lims.append(autocorr_lim)

                n_eff_preds_auto[itrp] = n_tot/(est_vars_auto[itrp]/autocorr_lim[0])

                if self.do_cross:
                    cov_cross_lim, cov_cross_cut, est_vars_cross[itrp] = get_crosscorr_sum(mcc, n_burnin_thin, itrp, autocorr_lim, autocorr_cut, obs_var_loc, n_eff_preds_auto)
                    self.cov_cross_lims.append(cov_cross_lim)

                est_vars[itrp] = est_vars_auto[itrp]+est_vars_cross[itrp]

                n_eff_preds[itrp] = n_tot/(est_vars[itrp]/autocorr_lim[0])  # TODO double check factor of two

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

    def summarize_vars(self, mcc, n_burnin):
        """get the means and vars for the whole run"""
        n_burnin_thin = restrict_n_burnin(mcc, n_burnin)//mcc.store_thin
        obs_mean, obs_var = summarize_vars(mcc, n_burnin_thin)
        self.obs_means.append(obs_mean)
        self.obs_vars.append(obs_var)

    def n_eff_summary_print(self, mcc, n_burnin):
        """print salient information about the number of effective samples"""
        n_burnin_thin = restrict_n_burnin(mcc, n_burnin)//mcc.store_thin
        n_par = mcc.n_par
        n_cold = mcc.n_cold
        n_chain = mcc.n_chain
        n_use = mcc.store_size+1-n_burnin_thin
        return n_eff_summary_print(n_par, n_use, n_cold, n_chain, mcc.store_thin, np.array(self.n_eff_preds),
                                   np.array(self.n_eff_preds_empirical), np.array(self.obs_vars), np.array(self.obs_means))

    def autocorr_summary_print(self, mcc):
        """print salient information about autocorrelation functions"""
        return autocorr_summary_print(mcc.n_par, self.autocorr_lims, self.do_cross)

    def summarize_logLs(self, mcc):
        """save some summary statistics related to the likelihoods"""
        N_blocks = mcc.store_size//(mcc.block_size//mcc.store_thin)
        logL_block_mean, arg_logL_burn, arg_logL_deviant = summarize_logLs(mcc, N_blocks)
        self.logL_block_means.append(logL_block_mean)
        self.arg_logL_burns.append(arg_logL_burn)
        self.arg_logL_deviant.append(arg_logL_deviant)

    def restrict_n_burnin(self, mcc, n_burnin):
        """restrict n_burnin based on storage size"""
        return restrict_n_burnin(mcc, n_burnin)


def restrict_n_burnin(mcc, n_burnin):
    """helper to restrict n_burnin to last block"""
    if mcc.store_size*mcc.store_thin < n_burnin:
        # handle burning more than 1 entire storage block
        if mcc.itrn-mcc.store_size*mcc.store_thin < n_burnin:
            # amount of elements to burn from last storage block
            n_burnin = n_burnin-(mcc.itrn-mcc.store_size*mcc.store_thin)
        else:
            # no elements need to be burned from last storage block
            n_burnin = 0

    return n_burnin


def autocorr_helper(mcc, itrp, n_burnin_thin):
    """helper to get the autocorrleation functions for a particular parameter"""
    n_use = mcc.store_size+1-n_burnin_thin
    autocorr_sum = np.zeros((n_use-1)*2+1)
    get_autocorr_sum(n_burnin_thin, mcc, itrp, autocorr_sum)
    autocorr_lim = np.hstack([autocorr_sum[n_use-1:n_use], autocorr_sum[n_use:2*n_use-2:2]+autocorr_sum[n_use+1:2*n_use-1:2]])
    autocorr_cut = 1+np.argmax(autocorr_lim[1:] < 0.)
    est_var_auto = autocorr_lim[0]+2*np.sum(autocorr_lim[1:autocorr_cut])
    return autocorr_lim, autocorr_cut, est_var_auto


def get_crosscorr_sum(mcc, n_burnin_thin, itrp, autocorr_lim, autocorr_cut, obs_var, n_eff_pred_auto):
    """estimate the average cross correlations"""
    n_use = mcc.store_size+1-n_burnin_thin
    n_cold = mcc.n_cold
    n_cross_eval = min(64, n_cold)  # don't go too large or it takes a very long time
    n_chain = mcc.n_chain
    block_size = mcc.block_size
    n_tot = n_use*n_cold

    cov_cross_sum = np.zeros((n_use-1)*2+1)

    for itrt1 in range(n_cross_eval):
        params_adj1 = mcc.samples_store[n_burnin_thin:, itrt1, itrp]-np.mean(mcc.samples_store[n_burnin_thin:, itrt1, itrp])
        for itrt2 in range(itrt1+1, n_cross_eval):
            params_adj2 = mcc.samples_store[n_burnin_thin:, itrt2, itrp]-np.mean(mcc.samples_store[n_burnin_thin:, itrt2, itrp])
            corr_loc = scipy.signal.correlate(params_adj1, params_adj2, mode='full')
            cov_cross_sum += corr_loc
            cov_cross_sum += corr_loc[::-1]  # for the itrt2,itr1 correlation

    # TODO check
    if n_cross_eval < n_cold:
        cov_cross_sum *= (n_cold**2-n_cold)/(n_cross_eval**2-n_cross_eval)
    cov_cross_lim = np.hstack([cov_cross_sum[n_use-1:n_use], cov_cross_sum[n_use:2*n_use-2:2]+cov_cross_sum[n_use+1:2*n_use-1:2]])
    cov_cross_cut_last = n_use//2-2*block_size
    cov_cut_std_thresh = 10
    std_comp = np.sqrt(n_tot/n_eff_pred_auto[itrp])*obs_var[itrp]*np.sqrt(2*(n_chain**2-n_chain)*np.arange(n_use-2, n_use-2*cov_cross_cut_last, -2))

    if np.any(np.abs(cov_cross_lim[1:cov_cross_cut_last][::-1]) > cov_cut_std_thresh*std_comp[::-1]):
        cut_from_back1 = np.argmax(np.abs(cov_cross_lim[1:cov_cross_cut_last][::-1]) > cov_cut_std_thresh*std_comp[::-1])
    else:
        cut_from_back1 = 0

    if cut_from_back1 == 0:
        cut_from_back1 = cov_cross_cut_last

    cut_cond1 = np.abs(autocorr_lim[0])*1.e-1 < np.abs(cov_cross_lim[1:cov_cross_cut_last][::-1])
    if autocorr_cut > 10 and np.max(np.abs(cov_cross_lim)) > 1.e-1*np.abs(autocorr_lim[0]) and np.any(cut_cond1):
        cut_from_back2 = np.argmax(cut_cond1)
    else:
        cut_from_back2 = cov_cross_cut_last
    cut_from_back = min(cut_from_back1, cut_from_back2)

    cov_cross_cut = 1+cov_cross_cut_last-cut_from_back

    est_var_cross = cov_cross_lim[0]+2*np.sum(cov_cross_lim[1:cov_cross_cut])
    return cov_cross_lim, cov_cross_cut, est_var_cross


def n_eff_summary_print(n_par, n_use, n_cold, n_chain, store_thin, n_eff_preds, n_eff_preds_empirical, obs_vars, obs_means):
    """print salient information about the number of effective samples"""
    eff_empiricals = np.zeros(n_par)
    eff_preds = np.zeros(n_par)
    eff_overalls = np.zeros(n_par)

    eff_empirical_mean = 0.
    eff_pred_mean = 0.
    eff_overall_mean = 0.

    # can't quote multi-run efficiencies if only one run was done
    overall_usable = obs_vars.shape[0] > 1

    for itrp in range(n_par):
        obs_varf = np.var(obs_means[:, itrp])
        if overall_usable:
            eff_overalls[itrp] = 2.*np.mean(obs_vars[:, itrp])/obs_varf/(n_use*n_chain*store_thin)
            eff_overall_mean += eff_overalls[itrp]

        eff_preds[itrp] = 2*np.mean(n_eff_preds[:, itrp])/(n_use*n_chain*store_thin)
        eff_empiricals[itrp] = 2*np.mean(n_eff_preds_empirical[:, itrp])/(n_use*n_chain*store_thin)
        eff_pred_mean += eff_preds[itrp]
        eff_empirical_mean += eff_empiricals[itrp]

    print(n_cold, n_chain)
    eff_empirical_mean /= n_par
    eff_pred_mean /= n_par
    eff_overall_mean /= n_par

    eff_empirical_string = ""
    eff_overall_string = ""
    eff_preds_string = ""
    for itrp in range(0, n_par):
        eff_empirical_string = eff_empirical_string + " %+.5e" % eff_empiricals[itrp]
        eff_overall_string = eff_overall_string + " %+.5e" % eff_overalls[itrp]
        eff_preds_string = eff_preds_string + " %+.5e" % eff_preds[itrp]

    print('correlation efficiencies'+eff_preds_string)
    print('empirical   efficiencies'+eff_empirical_string)
    if overall_usable:
        print('overall     efficiencies'+eff_overall_string)
    print('mean correlation efficiency %.5e' % (eff_pred_mean))
    print('mean empirical   efficiency %.5e' % (eff_empirical_mean))
    if overall_usable:
        print('mean overall     efficiency %.5e' % (eff_overall_mean))

    print('overall effective sample sizes:', n_eff_preds_empirical[0])


def autocorr_summary_print(n_par, autocorr_lims, do_cross):
    """print useful information about the autocorrelations"""
    autocorr_lim_array = np.array(autocorr_lims)
    crosscorr_lim_array = np.array(autocorr_lims)

    autocorr_lim_means = np.zeros((n_par, autocorr_lims[0].size))
    crosscorr_lim_means = np.zeros((n_par, autocorr_lims[0].size))
    autocorr_cut_means = np.zeros(n_par, dtype=np.int64)
    autocorr_len_means = np.zeros(n_par)

    autocorr_len_str = ""
    for itrp in range(n_par):
        autocorr_lim_means[itrp] = autocorr_lim_array[itrp::n_par].mean(axis=0)
        if do_cross:
            crosscorr_lim_means[itrp] = crosscorr_lim_array[itrp::n_par].mean(axis=0)
        autocorr_cut_means[itrp] = 1+np.argmax(autocorr_lim_means[itrp, 1:] < 0.)
        autocorr_len_means[itrp] = (autocorr_lim_means[itrp, 0]+2*np.sum(autocorr_lim_means[itrp, 1:autocorr_cut_means[itrp]]))/autocorr_lim_means[itrp, 0]
        autocorr_len_str = autocorr_len_str + " %.8e" % autocorr_len_means[itrp]

    print("best estimate of autocorrelation lengths:", autocorr_len_str)


def summarize_logLs(mcc, N_blocks):
    """get useful summary statistics about the likelihoods"""
    block_size = mcc.block_size//mcc.store_thin
    logL_block_mean = np.zeros(N_blocks)
    for itrk in range(0, N_blocks):
        logL_block_mean[itrk] = np.mean(mcc.logLs_store[itrk*block_size:(itrk+1)*block_size])

    logL_mean = np.mean(logL_block_mean[-10:])
    logL_std = np.std(logL_block_mean[-10:])
    if np.any(logL_block_mean > logL_mean-logL_std):
        arg_logL_burn = np.argmax(logL_block_mean > logL_mean-logL_std)
    else:
        print("logL never burned in")
        arg_logL_burn = -1
    if np.any(logL_block_mean < logL_mean-5*logL_std):
        arg_logL_deviant = logL_block_mean.size-np.argmax(logL_block_mean[::-1] < logL_mean-6*logL_std)-1
    else:
        arg_logL_deviant = -1
    return logL_block_mean, arg_logL_burn, arg_logL_deviant


def summarize_vars(mcc, n_burnin_thin):
    """get the means and variances for the samples"""
    n_par = mcc.n_par
    obs_means = np.mean(mcc.samples_store[n_burnin_thin:, :, :], axis=0).mean(axis=0)
    obs_vars = np.zeros(n_par)
    for itrp in range(n_par):
        obs_vars[itrp] = np.var(mcc.samples_store[n_burnin_thin:, :, itrp])

    return obs_means, obs_vars
