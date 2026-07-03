"""C 2023 Matthew C. Digman
code example to run the galactic binary parameter estimation pipeline
and plot results
"""

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from time import perf_counter

import numpy as np
import scipy.signal
from numba import njit
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline

# import likelihood_gb as trial_likelihood
import cake_likelihood as trial_likelihood
import diagnostic_commentary_helpers as dch
import DTMCMC.exchange_manager as eh
import DTMCMC.temperature_ladder_helpers as th
import integ_box_filt as ibf
import moment_helpers
from DTMCMC.corr_summary_helpers import CorrelationSummary
from DTMCMC.dtmcmc_sampler import DTMCMCSampler
from DTMCMC.proposal_manager_helper import get_default_proposal_manager

# TODO reduce exposure of block_size parameter

if __name__ == '__main__':
    t0 = perf_counter()

    # starting variables
    n_chain = 16                       # number of total chains for parallel tempering
    n_cold = 1                         # number of T=1 chains for parallel tempering
    n_burnin = 0                    # number of iterations to discard as burn in
    block_size = 10000                  # number of iterations per block when advancing the chain state
    store_size = 50000                # number of samples to store total
    N_blocks = store_size // block_size  # number of blocks the sampler must iterate through
    n_par = 5

    T_max = 1.e3                       # maximum temperature for geometric part of temperature ladder

    params_true = np.zeros(n_par)      # true parameters for search

    # create needed objects
    # T_ladder = GeometricTemperatureLadder(n_chain, n_cold=n_cold, T_max=T_max,T_min=80.,T_cold=80.)  # get the temperature ladder object
    # T_ladder = th.entropy_ladder_fromfile(n_chain,n_cold,'Ts_cake_combo2.npy','vars_cake_combo2.npy',n_inf_final=1,T_cold=1.,correct_last=False)
    T_ladder = th.entropy_ladder_fromfile(n_chain, n_cold, 'data/Ts_cake_gold.npy', 'data/vars_cake_gold.npy', n_inf_final=1, T_cold=1., correct_last=False)

    like_obj = trial_likelihood.CakeLikelihood(n_par)
    params_true = like_obj.correct_bounds(params_true)                 # make sure the conventions on the parameters match

    # create the starting samples
    starting_samples = np.zeros((T_ladder.n_chain, like_obj.n_par))
    for itrt in range(n_chain):
        # start from prior draws
        starting_samples[itrt] = like_obj.prior_draw()

    # create the overarching proposal manager object
    exchange_manager = eh.ExchangeManager(strategy=eh.SEQUENTIAL_TARGETS, track_full_exchanges=True)
    proposal_manager = get_default_proposal_manager(T_ladder, like_obj, starting_samples, exchange_manager_loc=exchange_manager)

    print('Chain parameters', n_cold, n_chain, n_burnin, block_size, store_size, T_max)

    # create the chain object
    mcc = DTMCMCSampler(T_ladder, like_obj, block_size, store_size, starting_samples=starting_samples, n_record=n_chain, proposal_manager=proposal_manager)

    t_init_end = perf_counter()
    print('all objects initialized in ', t_init_end - t0, 's')

    t_advance_begin = perf_counter()

    argT_1 = np.argmax(T_ladder.Ts == T_ladder.T_cold)
    rs_save = []

    # the main loop which actually advances the MCMC state
    for itrb in range(1080):
        mcc.advance_N_blocks(N_blocks)
        rs_save.append(np.sqrt(np.sum(mcc.samples_store[:, argT_1, :]**2, axis=1)))
    # mcc.advance_N_blocks(N_blocks)
    # mcc.advance_N_blocks(N_blocks)
    # mcc.advance_N_blocks(N_blocks)
    # mcc.advance_N_blocks(N_blocks)
    # mcc.advance_N_blocks(N_blocks)
    # mcc.advance_N_blocks(N_blocks)
    # mcc.advance_N_blocks(N_blocks)
    # mcc.advance_N_blocks(N_blocks)
    # mcc.advance_N_blocks(N_blocks)
    # mcc.advance_N_blocks(N_blocks)
    # mcc.advance_N_blocks(N_blocks)
    # mcc.advance_N_blocks(N_blocks)
    # mcc.advance_N_blocks(N_blocks)
    # mcc.advance_N_blocks(N_blocks)

    t_advance_end = perf_counter()
    print('advanced in ', t_advance_end - t_advance_begin, 's')

    # generate some summary information
    corr_sum = CorrelationSummary()
    corr_sum.summarize_blocks(mcc, n_burnin)
    corr_sum.final_prints(mcc, n_burnin)

    # get flattened samples for plotting
    # samples_flattened, logLs_flattened = mcc.get_stored_flattened(corr_sum.restrict_n_burnin(mcc, n_burnin),n_chain_out=n_cold)

    tf = perf_counter()

    print('full search time ', str(tf - t0) + 's')


    argT_1 = np.argmax(T_ladder.Ts == T_ladder.T_cold)

    block_burnin = min(60, len(mcc.logL_means) // 2)

    argTs = np.argsort(T_ladder.Ts)

    import matplotlib.pyplot as plt

    do_corner_plot = False
    if do_corner_plot:
        # generate a corner plot
        import corner

        # reformat the samples to make the plots look nicer
        labels = like_obj.get_labels()
        samples_format, params_true_format = like_obj.format_samples_output(mcc.samples_store[n_burnin:, argT_1, :].copy(), params_true)

        # create the corner plot figure
        fig = plt.figure(figsize=(10, 7.5))
        figure = corner.corner(samples_format, fig=fig, bins=25, hist_kwargs={'density': True}, show_titles=True, title_fmt=None,
                               title_kwargs={'fontsize': 12}, labels=labels, max_n_ticks=3, label_kwargs={'fontsize': 12}, labelpad=0.15,
                               smooth=0.25, levels=[0.682, 0.954])

        # overplot the true parameters
        corner.overplot_points(figure, params_true_format[None], marker='s', color='tab:blue', markersize=4)
        corner.overplot_lines(figure, params_true_format, color='tab:blue')

        # adjust the figure to fit the box better
        fig.subplots_adjust(wspace=0., hspace=0., left=0.05, top=0.95, right=0.99, bottom=0.05)
        for ax in figure.get_axes():
            ax.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True, labelsize=6)
        plt.show()

    # plt.semilogx(T_ladder.Ts[argTs],np.gradient(np.mean(np.array(mcc.logL_means[block_burnin:]),axis=0)[argTs],T_ladder.betas[argTs])*T_ladder.betas[argTs]**2)
    # plt.semilogx(T_ladder.Ts[argTs],(np.mean(np.array(mcc.logL2_means[block_burnin:]),axis=0)-np.mean(np.array(mcc.logL_means[block_burnin:]),axis=0)**2)[argTs]*T_ladder.betas[argTs]**2)
    # plt.semilogx(T_ladder.Ts[argTs],np.var(mcc.logLs_store[n_burnin:],axis=0)[argTs]*T_ladder.betas[argTs]**2)
    # plt.show()

    do_corr_plots = False
    if do_corr_plots:
        n_use = mcc.store_size + 1
        for itrt in range(max(0, argT_1 - 5), min(n_chain, argT_1 + 5)):
            logL_diff = mcc.logLs_store[:, itrt] - np.mean(mcc.logLs_store[:, itrt])
            autocorr_logL = scipy.signal.correlate(logL_diff, logL_diff, mode='full')
            autocorr_logL_lim = np.hstack([autocorr_logL[n_use - 1:n_use], autocorr_logL[n_use:2 * n_use - 2:2] + autocorr_logL[n_use + 1:2 * n_use - 1:2]])
            plt.plot(autocorr_logL_lim / autocorr_logL_lim[0])

        plt.show()

        logL_means_store = np.array(mcc.logL_means[block_burnin:])

        n_use = logL_means_store.shape[0] + 1
        for itrt in range(max(0, argT_1 - 10), min(n_chain, argT_1 + 10)):
            logL_diff = logL_means_store[:, itrt] - np.mean(logL_means_store[:, itrt])
            autocorr_logL = scipy.signal.correlate(logL_diff, logL_diff, mode='full')
            plt.plot(autocorr_logL[n_use - 2:2 * n_use - 1] / autocorr_logL[n_use - 2])

        plt.show()

        logL_means_store = np.array(mcc.logL_means[block_burnin:])
        logL_diff0 = logL_means_store[:, argT_1] - np.mean(logL_means_store[:, argT_1])

        n_use = logL_means_store.shape[0] + 1
        for itrt in range(max(0, argT_1 - 20), min(n_chain, argT_1 + 20)):
            logL_diff = logL_means_store[:, itrt] - np.mean(logL_means_store[:, itrt])
            crosscorr_logL = scipy.signal.correlate(logL_diff, logL_diff0, mode='full')
            plt.plot(crosscorr_logL[n_use - 2:2 * n_use - 1])

        plt.show()

    logL_diff0 = mcc.logLs_store[:, argT_1] - np.mean(mcc.logLs_store[:, argT_1])

    n_use = mcc.logLs_store.shape[0] + 1
    for itrt in range(max(0, argT_1 - 10), min(n_chain, argT_1)):
        logL_diff = mcc.logLs_store[:, itrt] - np.mean(mcc.logLs_store[:, itrt])
        crosscorr_logL = scipy.signal.correlate(logL_diff, logL_diff0, mode='full')
        # plt.plot(crosscorr_logL[n_use-1:2*n_use-2:2]+crosscorr_logL[n_use:2*n_use-1:2])
        b1 = n_use - 1
        b2 = 2 * n_use
        plt.plot(crosscorr_logL[b1:b2 - 5:4] + crosscorr_logL[b1 + 1:b2 - 4:4] + crosscorr_logL[b1 + 2:b2 - 3:4] + crosscorr_logL[b1 + 3:b2 - 2:4])


    # plt.xlim(-1,500)
        plt.show()

    # argT_sort = np.argsort(T_ladder.Ts)
    # a_ex_yes = (mcc.tracker_manager.exchange_tracker[0]+mcc.tracker_manager.exchange_tracker[0].T)[argT_sort,:][:,argT_sort]
    # a_ex_no = (mcc.tracker_manager.exchange_tracker[1]+mcc.tracker_manager.exchange_tracker[1].T)[argT_sort,:][:,argT_sort]
    #
    # accept_exchange = a_ex_yes/(a_ex_yes+a_ex_no)
    # accept_exchange_nn_left = np.zeros(n_chain)
    # accept_exchange_nn_right = np.zeros(n_chain)
    # accept_exchange_nn = np.zeros(n_chain)
    # accept_exchange_nn_left[n_chain-1] = accept_exchange[n_chain-2,n_chain-1]
    # accept_exchange_nn_right[0] = accept_exchange[0,1]
    #
    # accept_exchange_nn[n_chain-1] = accept_exchange[n_chain-2,n_chain-1]
    # accept_exchange_nn[0] = accept_exchange[0,1]
    # for itrt in range(1,n_chain-1):
    #    accept_exchange_nn_right[itrt] = accept_exchange[itrt,itrt+1]
    #    accept_exchange_nn_left[itrt] = accept_exchange[itrt,itrt-1]
    #    accept_exchange_nn[itrt] =  (a_ex_yes[itrt,itrt+1]+a_ex_yes[itrt,itrt-1])/(a_ex_yes[itrt,itrt+1]+a_ex_no[itrt,itrt+1]+a_ex_yes[itrt,itrt-1]+a_ex_no[itrt,itrt-1])
    #
    #
    # plt.plot(T_ladder.betas[:n_chain-1],accept_exchange_nn_right[:n_chain-1])
    # plt.plot(T_ladder.betas[1:],accept_exchange_nn_left[1:])
    # plt.plot(T_ladder.betas,accept_exchange_nn)
    # plt.show()

    do_heat_plot_gold = False
    if do_heat_plot_gold:
    # Ts_old = np.load('Ts_cake_combo2.npy')
    # vars_old = np.load('vars_cake_combo2.npy')
        Ts_old = np.load('data/Ts_cake_gold.npy')
        vars_old = np.load('data/vars_cake_gold.npy')
        plt.loglog(Ts_old, vars_old)
        plt.loglog(T_ladder.Ts[argTs], (np.mean(np.array(mcc.logL2_means[block_burnin:]), axis=0) - np.mean(np.array(mcc.logL_means[block_burnin:]), axis=0)**2)[argTs])
        plt.show()

    # plt.semilogx(Ts_old,vars_old/Ts_old**2)
    # plt.semilogx(T_ladder.Ts[argTs],(np.mean(np.array(mcc.logL2_means[block_burnin:]),axis=0)-np.mean(np.array(mcc.logL_means[block_burnin:]),axis=0)**2)[argTs]*T_ladder.betas[argTs]**2)
    # plt.show()


    ####

    n_burnin = min(mcc.itrn // 2, 4000000)

    block_burnin = min(n_burnin // block_size, len(mcc.logL_means) // 2)

    cumulants = np.array(moment_helpers.get_cumulants(moment_helpers.get_averaged_means(mcc, len(mcc.logL_means) - block_burnin, cut=block_burnin)))[:, 0]


    rs_stack = np.hstack(rs_save)[n_burnin:]

    counts, bins = np.histogram(rs_stack, 10000, range=[0., np.sqrt(n_par) * like_obj.high_lims[1]], density=True)

    integ_true = cumulative_trapezoid(ibf.get_density_pred(1.)[::-1], ibf.rs[::-1], initial=0)[::-1] + 1
    integ_loc = cumulative_trapezoid(counts[::-1], bins[::-1][1:], initial=0)[::-1] + 1
    interp_true = InterpolatedUnivariateSpline(ibf.rs, integ_true, k=3, ext=2)(bins[:bins.size - 1])

    n_use = rs_stack.size
    rs_mean = np.mean(rs_stack)
    r2s_mean = np.mean(rs_stack**2)
    autocorr_rs = scipy.signal.correlate(rs_stack - rs_mean, rs_stack - rs_mean, mode='full')
    avg_len = 32
    autocorr_rs_lim = np.hstack([autocorr_rs[n_use - 1], autocorr_rs[n_use:2 * n_use - avg_len:avg_len]])
    for itrb in range(1, avg_len):
        autocorr_rs_lim[1:] += autocorr_rs[n_use + itrb:2 * n_use - avg_len + itrb:avg_len]

    autocorr_rs_lim[1:] /= avg_len

    autocorr_rs = None


    arg_cut_r = np.argmax(autocorr_rs_lim / autocorr_rs_lim[0] < 0.)

    autocorr_len_inferred = 1 + avg_len * np.sum(autocorr_rs_lim[1:arg_cut_r] / autocorr_rs_lim[0])

    integ_entropy = -cumulative_trapezoid(cumulants[1][::-1] * mcc.betas[::-1], mcc.betas[::-1], initial=0)[::-1]

    arg_cut_r = np.argmax(autocorr_rs_lim / autocorr_rs_lim[0] < 0.)

    res_even, binsx, binsy = np.histogram2d(rs_stack[0:rs_stack.size - 1:2], np.diff(rs_stack)[::2], 100, range=[[0, 6.3], [-6.3, 6.3]])
    res_odd, binsx, binsy = np.histogram2d(rs_stack[1:rs_stack.size - 1:2], np.diff(rs_stack)[1::2], 100, range=[[0, 6.3], [-6.3, 6.3]])

    res_shuffle = np.zeros_like(res_even)


    means_shuffle_100 = []

    empty_log = []

    rs_shuffle = rs_stack.copy()

    for itrb in range(10):
        np.random.shuffle(rs_shuffle)
        res_shuffle_loc, binsx, binsy = np.histogram2d(rs_shuffle[0:rs_shuffle.size - 1:], np.diff(rs_shuffle), 100, range=[[0, 6.3], [-6.3, 6.3]])
        res_shuffle += res_shuffle_loc
        empty_log.append(np.sum(res_shuffle == 0.))
        means_shuffle_100.append(np.mean(rs_shuffle.reshape((rs_shuffle.size // 100, 100)), axis=1))


    rs_shuffle = None


    @njit()
    def get_block_mean(n_block_in, block_length, rs_stack):
        means_got = np.zeros(n_block_in)
        for itrb in range(n_block_in):
            start1 = np.random.randint(0, rs_stack.size - block_length)
            means_got[itrb] = np.mean(rs_stack[start1:start1 + block_length])
        return means_got


    means_stack_100 = get_block_mean(10 * rs_stack.size // 100, 100, rs_stack)
    means_stack_1k = get_block_mean(10 * rs_stack.size // 1000, 1000, rs_stack)
    means_stack_10k = get_block_mean(100 * rs_stack.size // 10000, 10000, rs_stack)
    means_stack_100k = get_block_mean(1000 * rs_stack.size // 100000, 100000, rs_stack)
    means_stack_1m = get_block_mean(1000 * rs_stack.size // 1000000, 1000000, rs_stack)

    var_shuffle_100 = np.var(means_shuffle_100)
    var_shuffle_1k = var_shuffle_100 / 10
    var_shuffle_10k = var_shuffle_100 / 100
    var_shuffle_100k = var_shuffle_100 / 1000
    var_shuffle_1m = var_shuffle_100 / 10000

    autocorr_len_100 = np.var(means_stack_100) / var_shuffle_100
    autocorr_len_1k = np.var(means_stack_1k) / var_shuffle_1k
    autocorr_len_10k = np.var(means_stack_10k) / var_shuffle_10k
    autocorr_len_100k = np.var(means_stack_100k) / var_shuffle_100k
    autocorr_len_1m = np.var(means_stack_1m) / var_shuffle_1m

    cycle_burn_index = np.argmax(np.array(mcc.tracker_manager.itrn_archive) == n_burnin)

    tracker_archive = mcc.tracker_manager.exchange_archive[cycle_burn_index]

    argT_sort = np.argsort(T_ladder.Ts)
    a_ex_yes = (mcc.tracker_manager.exchange_tracker[0] - tracker_archive[0])[argT_sort, :][:, argT_sort]
    a_ex_no = (mcc.tracker_manager.exchange_tracker[1] - tracker_archive[1])[argT_sort, :][:, argT_sort]

    accept_exchange = a_ex_yes / (a_ex_yes + a_ex_no)
    accept_exchange_nn_left = np.zeros(n_chain)
    accept_exchange_nn_right = np.zeros(n_chain)
    accept_exchange_nn = np.zeros(n_chain)
    accept_exchange_nn_left[n_chain - 1] = accept_exchange[n_chain - 2, n_chain - 1]
    accept_exchange_nn_right[0] = accept_exchange[0, 1]

    accept_exchange_nn[n_chain - 1] = accept_exchange[n_chain - 2, n_chain - 1]
    accept_exchange_nn[0] = accept_exchange[0, 1]
    for itrt in range(1, n_chain - 1):
        accept_exchange_nn_right[itrt] = accept_exchange[itrt, itrt + 1]
        accept_exchange_nn_left[itrt] = accept_exchange[itrt, itrt - 1]
        accept_exchange_nn[itrt] = (a_ex_yes[itrt, itrt + 1] + a_ex_yes[itrt, itrt - 1]) / (a_ex_yes[itrt, itrt + 1] + a_ex_no[itrt, itrt + 1] + a_ex_yes[itrt, itrt - 1] + a_ex_no[itrt, itrt - 1])


    accept_record = mcc.tracker_manager.accept_record - mcc.tracker_manager.accept_archive[cycle_burn_index]
    accept = accept_record[0] / (accept_record[0] + accept_record[1])

    n_cycles = mcc.tracker_manager.get_n_cycles()

    cycle_min_log = np.zeros(len(mcc.tracker_manager.cycle_archive))
    cycle_zero_log = np.zeros(len(mcc.tracker_manager.cycle_archive))
    diff_ext_log = np.zeros(len(mcc.tracker_manager.cycle_archive))
    cycle_archive_old = np.zeros_like(mcc.tracker_manager.cycle_archive[0])

    for itri in range(0, len(mcc.tracker_manager.cycle_archive), 1):
        cycle_archive = mcc.tracker_manager.cycle_archive[itri]
        new_cycles = np.min(cycle_archive[2:4, :] - cycle_archive_old[2:4, :], axis=0)
        diff_cycles = mcc.tracker_manager.itrn_archive[itri] - np.min(cycle_archive[0:2, :], axis=0)
        cycle_min_log[itri] = np.min(new_cycles)
        diff_ext_log[itri] = np.max(diff_cycles)
        cycle_zero_log[itri] = np.sum(new_cycles == 0)
        cycle_archive_old = cycle_archive


    print(np.sum(n_cycles), np.min(n_cycles), np.max(n_cycles), np.std(n_cycles))
    print(cumulants[:, argT_1])
    print(cumulants[:, -1])
    print(np.max(interp_true - integ_loc), np.min(interp_true - integ_loc))
    print(rs_mean, r2s_mean - rs_mean**2)
    print(arg_cut_r * 16 + 1, autocorr_len_inferred)

    print(autocorr_len_100, autocorr_len_1k, autocorr_len_10k, autocorr_len_100k, autocorr_len_1m)

    print(accept_exchange_nn[argT_1], accept_exchange_nn[1], np.mean(accept_exchange_nn[1:n_chain - 1]))
    print(accept[argT_1][[1, 2, 3, 4, 5, 6, 8]])

    print(integ_entropy[argT_1])

    var_od2 = np.var(np.diff(rs_stack)[1::2]**2)
    var_ed2 = np.var(np.diff(rs_stack)[2::2]**2)

    var_ed1 = np.var(np.diff(rs_stack)[0::2])
    var_ev1 = np.var(rs_stack[0::2])

    var_ov1 = np.var(rs_stack[1:rs_stack.size - 1:2])
    var_od1 = np.var(np.diff(rs_stack)[1::2])

    mean_od2 = np.mean(np.diff(rs_stack)[1::2]**2)
    mean_ed2 = np.mean(np.diff(rs_stack)[2::2]**2)

    mean_ed1 = np.mean(np.diff(rs_stack)[0::2])
    mean_ev1 = np.mean(rs_stack[0::2])

    mean_ov1 = np.mean(rs_stack[1:rs_stack.size - 1:2])
    mean_od1 = np.mean(np.diff(rs_stack)[1::2])

    mean_od2_ed2 = np.mean(np.diff(rs_stack)[1::2]**2 * np.diff(rs_stack)[2::2]**2)
    mean_ed1_ev1 = np.mean(np.diff(rs_stack)[0::2] * rs_stack[0::2])
    mean_od1_ov1 = np.mean(np.diff(rs_stack)[1::2] * rs_stack[1:rs_stack.size - 1:2])
    mean_ed2_ev1 = np.mean(np.diff(rs_stack)[0::2]**2 * rs_stack[0::2])
    mean_od2_ov1 = np.mean(np.diff(rs_stack)[1::2]**2 * rs_stack[1:rs_stack.size - 1:2])


    print((mean_od2_ed2 - mean_od2 * mean_ed2) / np.sqrt(var_od2 * var_ed2))
    print((mean_ed1_ev1 - mean_ed1 * mean_ev1) / np.sqrt(var_ed1 * var_ev1))
    print((mean_od1_ov1 - mean_od1 * mean_ov1) / np.sqrt(var_od1 * var_ov1))
    print((mean_ed2_ev1 - mean_ed2 * mean_ev1) / np.sqrt(var_ed2 * var_ev1))
    print((mean_od2_ov1 - mean_od2 * mean_ov1) / np.sqrt(var_od2 * var_ov1))

    print(np.min(diff_ext_log[cycle_burn_index:]), np.max(diff_ext_log[cycle_burn_index:]), np.median(diff_ext_log[cycle_burn_index:]), np.mean(diff_ext_log[cycle_burn_index:]), np.std(diff_ext_log[cycle_burn_index:]))
    print(np.min(cycle_zero_log[cycle_burn_index:]), np.max(cycle_zero_log[cycle_burn_index:]), np.median(cycle_zero_log[cycle_burn_index:]), np.mean(cycle_zero_log[cycle_burn_index:]), np.std(cycle_zero_log[cycle_burn_index:]))
    print(np.max(cycle_min_log[cycle_burn_index:]), np.median(cycle_min_log[cycle_burn_index:]), np.mean(cycle_min_log[cycle_burn_index:]), np.std(cycle_min_log[cycle_burn_index:]))


    dch.print_diagnostic_commentary(mcc)

    counts, bins = np.histogram(rs_stack, 10000, range=[0., np.sqrt(n_par) * like_obj.high_lims[1]], density=True)

    rs_stack = None

    plt.plot(bins[1:], counts)
    plt.plot(ibf.rs, ibf.get_density_pred(1.))
    plt.show()

    plt.plot(ibf.rs, integ_true)
    plt.plot(bins[:bins.size - 1], integ_loc)
    plt.show()

    plt.plot(bins[:bins.size - 1], integ_loc - interp_true)
    plt.show()

    plt.plot(bins[:bins.size - 1], integ_loc - interp_true)
    plt.show()

    plt.plot(autocorr_rs_lim / autocorr_rs_lim[0])
    plt.show()

    plt.imshow(np.rot90(np.log(res_shuffle)))
    plt.show()

    plt.imshow(np.rot90((res_even) / np.sqrt(res_even)))
    plt.show()

    plt.imshow(np.rot90((res_odd) / np.sqrt(res_odd)))
    plt.show()

    plt.imshow(np.rot90((res_even - res_odd) / np.sqrt(res_even + res_odd)))
    plt.show()


    import sys

    sys.exit()

    unique_rs, args_forward, count_rs = np.unique(rs_stack, return_index=True, return_counts=True)
    unique_rs_reverse, args_reverse = np.unique(rs_stack[::-1], return_index=True)
    assert np.all(unique_rs == unique_rs_reverse)
    args_reverse = rs_stack.size - 1 - args_reverse
    recurr_lengths = args_reverse - args_forward + 1
    assert np.all(recurr_lengths >= 1)

    recurr_lengths_mask = (recurr_lengths >= 2)

    plt.hist(np.log10(recurr_lengths[recurr_lengths_mask]), 100)
    plt.show()

    plt.scatter(unique_rs[recurr_lengths_mask], np.log10(recurr_lengths[recurr_lengths_mask]), s=0.1, alpha=1., color='black')
    plt.show()

    plt.scatter(unique_rs[recurr_lengths_mask], count_rs[recurr_lengths_mask], s=0.1, alpha=1., color='black')
    plt.show()

    import sys

    sys.exit()

    Ts_high_high = np.load('Ts_cake_hot.npy')
    vars_high_high = np.load('vars_cake_hot.npy')
    means_high_high = np.load('means_cake_hot.npy')

    Ts_mid_high = np.load('Ts_cake_mid_hot1.npy')
    vars_mid_high = np.load('vars_cake_mid_hot1.npy')
    means_mid_high = np.load('means_cake_mid_hot1.npy')

    Ts_low_high = np.load('Ts_cake_low_hot1.npy')
    vars_low_high = np.load('vars_cake_low_hot1.npy')
    means_low_high = np.load('means_cake_low_hot1.npy')

    Ts_evolve = np.load('Ts_cake_evolve_entropy1.npy')
    vars_evolve = np.load('vars_cake_evolve_entropy1.npy')
    means_evolve = np.load('means_cake_evolve_entropy1.npy')

    Ts_high_combo = np.hstack([Ts_high_high, Ts_mid_high, Ts_low_high, Ts_evolve])
    vars_high_combo = np.hstack([vars_high_high, vars_mid_high, vars_low_high, vars_evolve])
    means_high_combo = np.hstack([means_high_high, means_mid_high, means_low_high, means_evolve])


    args_high = np.argsort(Ts_high_combo)
    vars_high_combo = vars_high_combo[args_high]
    means_high_combo = means_high_combo[args_high]
    Ts_high_combo = Ts_high_combo[args_high]

    # np.save('Ts_cake_combo2.npy',Ts_high_combo)
    # np.save('vars_cake_combo2.npy',vars_high_combo)
    # np.save('means_cake_combo2.npy',means_high_combo)

    plt.semilogx(Ts_old, vars_old / Ts_old**2)
    plt.plot(Ts_high_combo, vars_high_combo / Ts_high_combo**2)
    plt.show()

    import sys

    sys.exit()

    cold_save = [mcc.samples_store[:, argT_1].copy()]


    for itrb in range(2):
        mcc.advance_N_blocks(N_blocks)
        cold_save.append(mcc.samples_store[:, argT_1].copy())


    cumulants = np.array(moment_helpers.get_cumulants(moment_helpers.get_averaged_means(mcc, len(mcc.logL_means) - block_burnin, cut=block_burnin)))[:, 0]

    import sys

    sys.exit()

    rs_got = np.sqrt(np.sum(np.vstack(cold_save)[block_burnin * block_size::10]**2, axis=1))
    rs_got = rs_got[rs_got < 10.]
    counts, bins, _ = plt.hist(rs_got, 1000, density=True)
    bins_match = np.unique(np.hstack([np.linspace(0., 1., 100), bins, np.array([10.])]))
    bin_likes = np.zeros(len(bins_match))
    for itrb in range(bin_likes.size):
        bin_likes[itrb] = like_obj.get_loglike(np.array([bins_match[itrb], 0., 0., 0., 0.]))


    dens_pred = np.exp(T_ladder.betas[argT_1] * bin_likes) * bins_match**4 * 2 * np.pi
    dens_pred /= np.trapezoid(dens_pred, bins_match)

    plt.plot(bins_match, dens_pred)
    plt.show()

    import sys

    sys.exit()

    cumulants_gold = np.load('data/cumulants_cake_gold.npy')
    Ts_gold = np.load('data/Ts_cake_gold.npy')
    betas_gold = th.Ts_to_betas(Ts_gold)

    cumulants = np.array(moment_helpers.get_cumulants(moment_helpers.get_averaged_means(mcc, len(mcc.logL_means) - block_burnin, cut=block_burnin)))[:, 0]

    # plt.semilogx(mcc.Ts,(cumulative_trapezoid(cumulants[1],mcc.betas,initial=0)+cumulants[0][0])*mcc.betas**1)
    plt.semilogx(mcc.Ts, cumulants[0] * mcc.betas**1)
    plt.semilogx(Ts_gold, cumulants_gold[0] * betas_gold**1)
    plt.show()

    # plt.semilogx(mcc.Ts,np.gradient(cumulants[0],mcc.betas)*mcc.betas**2)
    # plt.semilogx(mcc.Ts,(cumulative_trapezoid(cumulants[2],mcc.betas,initial=0)+cumulants[1][0])*mcc.betas**2)
    plt.semilogx(mcc.Ts, cumulants[1] * mcc.betas**2)
    plt.semilogx(Ts_gold, cumulants_gold[1] * betas_gold**2)
    plt.show()


    # plt.semilogx(mcc.Ts,np.gradient(cumulants[1],mcc.betas)*mcc.betas**3)
    # plt.semilogx(mcc.Ts,(cumulative_trapezoid(cumulants[3],mcc.betas,initial=0)+cumulants[2][0])*mcc.betas**3)
    plt.semilogx(mcc.Ts, cumulants[2] * mcc.betas**3)
    plt.semilogx(Ts_gold, cumulants_gold[2] * betas_gold**3)
    plt.show()

    # plt.semilogx(mcc.Ts,np.gradient(cumulants[2],mcc.betas)*mcc.betas**4)
    # plt.semilogx(mcc.Ts,(cumulative_trapezoid(cumulants[4],mcc.betas,initial=0)+cumulants[3][0])*mcc.betas**4)
    plt.semilogx(mcc.Ts, cumulants[3] * mcc.betas**4)
    plt.semilogx(Ts_gold, cumulants_gold[3] * betas_gold**4)
    plt.show()

    # plt.semilogx(mcc.Ts,np.gradient(cumulants[3],mcc.betas)*mcc.betas**5)
    # plt.semilogx(mcc.Ts,(cumulative_trapezoid(cumulants[5],mcc.betas,initial=0)+cumulants[4][0])*mcc.betas**5)
    plt.semilogx(mcc.Ts, cumulants[4] * mcc.betas**5)
    plt.semilogx(Ts_gold, cumulants_gold[4] * betas_gold**5)
    plt.show()

    # plt.semilogx(mcc.Ts,np.gradient(cumulants[4],mcc.betas)*mcc.betas**6)
    plt.semilogx(mcc.Ts, cumulants[5] * mcc.betas**6)
    plt.semilogx(Ts_gold, cumulants_gold[5] * betas_gold**6)
    plt.show()

    plt.semilogx(mcc.Ts, np.gradient(cumulants[5], mcc.betas) * mcc.betas**7)
    plt.semilogx(Ts_gold, np.gradient(cumulants_gold[5], betas_gold) * betas_gold**7)
    plt.show()

    import sys

    sys.exit()

    cov_means = np.cov((moment_helpers.get_averaged_means(mcc, (len(mcc.logL_means) - block_burnin) // 10, cut=block_burnin))[0].T)
    var_means = np.diag(cov_means)
    corr_means = cov_means / np.sqrt(var_means)
    corr_means = (corr_means.T / np.sqrt(var_means)).T

    plt.plot(corr_means[argT_1])
    plt.show()


    import sys

    sys.exit()


    plt.plot(moment_helpers.get_corr_quantities(moment_helpers.get_averaged_means(mcc, len(mcc.logL_means) - block_burnin, cut=block_burnin), moment_helpers.get_averaged_adjacents(mcc, len(mcc.logL_means) - block_burnin, cut=block_burnin))[1][0][0])
    plt.show()

    plt.plot(moment_helpers.get_corr_quantities(moment_helpers.get_averaged_means(mcc, len(mcc.logL_means) - block_burnin, cut=block_burnin), moment_helpers.get_averaged_adjacents(mcc, len(mcc.logL_means) - block_burnin, cut=block_burnin))[1][1][0])
    plt.plot(moment_helpers.get_corr_quantities(moment_helpers.get_averaged_means(mcc, len(mcc.logL_means) - block_burnin, cut=block_burnin), moment_helpers.get_averaged_adjacents(mcc, len(mcc.logL_means) - block_burnin, cut=block_burnin))[1][2][0])
    plt.show()


    import sys

    sys.exit()

    import integral_heat_estimator

    cumulants = np.array(moment_helpers.get_cumulants(moment_helpers.get_averaged_means(mcc, len(mcc.logL_means) - block_burnin, cut=block_burnin)))[:, 0]
    integrand_left1, integrand_right1, integrand_avg1 = integral_heat_estimator.cumulant_integrand(cumulants[:2], mcc.betas)
    integrand_left2, integrand_right2, integrand_avg2 = integral_heat_estimator.cumulant_integrand(cumulants[:3], mcc.betas)
    integrand_left3, integrand_right3, integrand_avg3 = integral_heat_estimator.cumulant_integrand(cumulants[:4], mcc.betas)
    integrand_left4, integrand_right4, integrand_avg4 = integral_heat_estimator.cumulant_integrand(cumulants[:5], mcc.betas)
    integrand_left5, integrand_right5, integrand_avg5 = integral_heat_estimator.cumulant_integrand(cumulants[:6], mcc.betas)
    plt.plot(mcc.betas[1:], np.cumsum(integrand_avg1))
    plt.plot(mcc.betas[1:], np.cumsum(integrand_avg2))
    plt.plot(mcc.betas[1:], np.cumsum(integrand_avg3))
    plt.plot(mcc.betas[1:], np.cumsum(integrand_avg4))
    plt.plot(mcc.betas[1:], np.cumsum(integrand_avg5))
    plt.show()

    plt.plot(mcc.betas[1:], np.cumsum(integrand_left1))
    plt.plot(mcc.betas[1:], np.cumsum(integrand_left2))
    plt.plot(mcc.betas[1:], np.cumsum(integrand_left3))
    plt.plot(mcc.betas[1:], np.cumsum(integrand_left4))
    plt.plot(mcc.betas[1:], np.cumsum(integrand_left5))
    plt.show()

    plt.plot(mcc.betas[1:], np.cumsum(integrand_right1))
    plt.plot(mcc.betas[1:], np.cumsum(integrand_right2))
    plt.plot(mcc.betas[1:], np.cumsum(integrand_right3))
    plt.plot(mcc.betas[1:], np.cumsum(integrand_right4))
    plt.plot(mcc.betas[1:], np.cumsum(integrand_right5))
    plt.show()


    import sys

    sys.exit()

    # this test detects the correlations produced by the different exchange strategies
    # the chain translate version detects the autocorrelations in chain position
    # chain_track_loc = np.vstack(chain_track_hist)
    chain_track_loc = np.vstack(mcc.chain_track)

    # chain_translate = np.zeros((chain_track_loc.shape[0],n_chain))
    # for itrt in range(0,n_chain):
    #    chain_translate[:,itrt] = np.argmax(chain_track_loc==itrt,axis=1)


    n_use = chain_track_loc.shape[0]
    # chain_track_loc = None
    autocorr_chain_sum = np.zeros(2 * n_use - 1)
    chain_diff0 = chain_track_loc[:, argT_1] - np.mean(chain_track_loc[:, argT_1])
    for itrt in range(n_chain):
        chain_diff = chain_track_loc[:, itrt] - np.mean(chain_track_loc[:, itrt])
        # autocorr_chain_sum += scipy.signal.correlate(chain_diff,chain_diff, mode='full')
        autocorr_chain = scipy.signal.correlate(chain_diff, chain_diff0, mode='full')
        # plt.plot(autocorr_chain[n_use-1:n_use-1+2000]/autocorr_chain[n_use-1])
        plt.plot(autocorr_chain[n_use - 1:n_use - 1 + 2000])


    # plt.plot(autocorr_chain_sum[n_use-1:]/autocorr_chain_sum[n_use-1])
    plt.show()


    import sys

    sys.exit()

    # chain_track_hist = [mcc.chain_track[1:].copy()]
    #
    # for itrb in range(0,9):
    #    mcc.advance_N_blocks(1)
    #    chain_track_hist.append(mcc.chain_track[1:].copy())
    import integral_heat_estimator

    betas_old = th.Ts_to_betas(Ts_old)
    max_beta = max(np.max(mcc.betas), np.max(betas_old))
    betas_new = np.unique(np.hstack([mcc.betas, betas_old, np.linspace(0.5, 2., 10000), 10**np.linspace(np.log10(max_beta), -10., 10000), np.linspace(max_beta, 0., 10000)]))[::-1]
    Ts_new = th.betas_to_Ts(betas_new)

    cumulants = np.array(moment_helpers.get_cumulants(moment_helpers.get_averaged_means(mcc, len(mcc.logL_means) - block_burnin, cut=block_burnin)))[:, 0]

    estim_left, estim_right, estim_center = integral_heat_estimator.cumulant_heat_cap_interp(cumulants, mcc.betas, betas_new)
    estim_left1, estim_right1, estim_center1 = integral_heat_estimator.cumulant_heat_cap_interp(cumulants[:2], mcc.betas, betas_new)
    estim_left2, estim_right2, estim_center2 = integral_heat_estimator.cumulant_heat_cap_interp(cumulants[:3], mcc.betas, betas_new)
    estim_left3, estim_right3, estim_center3 = integral_heat_estimator.cumulant_heat_cap_interp(cumulants[:4], mcc.betas, betas_new)
    estim_left4, estim_right4, estim_center4 = integral_heat_estimator.cumulant_heat_cap_interp(cumulants[:5], mcc.betas, betas_new)

    plt.loglog(Ts_new, estim_left)
    plt.loglog(Ts_new, estim_right)
    plt.loglog(Ts_new, estim_center)
    plt.loglog(Ts_old, vars_old * betas_old**2, 'k--')
    plt.show()

    plt.semilogx(Ts_new, np.gradient(estim_center4, betas_new))
    plt.semilogx(Ts_new, np.gradient(estim_center, betas_new))
    plt.semilogx(Ts_old, np.gradient(vars_old * betas_old**2, betas_old), 'k--')
    plt.show()


    import sys

    sys.exit()


    cycle_burn_index = np.argmax(np.array(mcc.tracker_manager.itrn_archive) == 2000000)


    tracker_archive = mcc.tracker_manager.exchange_archive[cycle_burn_index]

    argT_sort = np.argsort(T_ladder.Ts)
    a_ex_yes = (mcc.tracker_manager.exchange_tracker[0] - tracker_archive[0])[argT_sort, :][:, argT_sort]
    a_ex_no = (mcc.tracker_manager.exchange_tracker[1] - tracker_archive[1])[argT_sort, :][:, argT_sort]

    accept_exchange = a_ex_yes / (a_ex_yes + a_ex_no)
    accept_exchange_nn_left = np.zeros(n_chain)
    accept_exchange_nn_right = np.zeros(n_chain)
    accept_exchange_nn = np.zeros(n_chain)
    accept_exchange_nn_left[n_chain - 1] = accept_exchange[n_chain - 2, n_chain - 1]
    accept_exchange_nn_right[0] = accept_exchange[0, 1]

    accept_exchange_nn[n_chain - 1] = accept_exchange[n_chain - 2, n_chain - 1]
    accept_exchange_nn[0] = accept_exchange[0, 1]
    for itrt in range(1, n_chain - 1):
        accept_exchange_nn_right[itrt] = accept_exchange[itrt, itrt + 1]
        accept_exchange_nn_left[itrt] = accept_exchange[itrt, itrt - 1]
        accept_exchange_nn[itrt] = (a_ex_yes[itrt, itrt + 1] + a_ex_yes[itrt, itrt - 1]) / (a_ex_yes[itrt, itrt + 1] + a_ex_no[itrt, itrt + 1] + a_ex_yes[itrt, itrt - 1] + a_ex_no[itrt, itrt - 1])

    accept_record = mcc.tracker_manager.accept_record - mcc.tracker_manager.accept_archive[cycle_burn_index]
    accept = accept_record[0] / (accept_record[0] + accept_record[1])
    print(accept_exchange_nn[argT_1], np.mean(accept_exchange_nn[1:n_chain - 1]))
    print(accept[argT_1][[1, 3, 4, 5, 6, 8]])


    plt.plot(T_ladder.betas[:n_chain - 1], accept_exchange_nn_right[:n_chain - 1])
    plt.plot(T_ladder.betas[1:], accept_exchange_nn_left[1:])
    plt.plot(T_ladder.betas, accept_exchange_nn)
    plt.show()


    cycle_count = np.array([np.sum((mcc.tracker_manager.cycle_archive[itrb][2] + mcc.tracker_manager.cycle_archive[itrb][3]) / 2) for itrb in range(len(mcc.tracker_manager.itrn_archive))])
    new_cycles = np.hstack([cycle_count[0], np.diff(cycle_count)])
    new_iterations = np.hstack([mcc.tracker_manager.itrn_archive[0], np.diff(np.array(mcc.tracker_manager.itrn_archive))])
    cycle_rate = new_cycles / new_iterations
    iterations_postburn = (mcc.tracker_manager.itrn_archive[-1] - mcc.tracker_manager.itrn_archive[cycle_burn_index])
    mean_rate = (cycle_count[-1] - cycle_count[cycle_burn_index]) / iterations_postburn
    plt.plot(np.array(mcc.tracker_manager.itrn_archive[cycle_burn_index:]), cycle_rate[cycle_burn_index:])
    plt.plot(np.array(mcc.tracker_manager.itrn_archive[cycle_burn_index:]), np.full(np.array(mcc.tracker_manager.itrn_archive[cycle_burn_index:]).size, mean_rate), 'k--')
    plt.ylim(0., 0.001)
    plt.show()

    cycle_record = mcc.tracker_manager.cycle_archive[-1][2:] - mcc.tracker_manager.cycle_archive[cycle_burn_index][2:]
    cycle_rate = np.vstack([cycle_record, np.sum(cycle_record, axis=0) / (2 * iterations_postburn)])


    accept_record = mcc.tracker_manager.accept_record - mcc.tracker_manager.accept_archive[cycle_burn_index]
    accept = accept_record[0] / (accept_record[0] + accept_record[1])
    print(accept_exchange_nn[argT_1], np.mean(accept_exchange_nn[1:n_chain - 1]))
    print(accept[argT_1][[3, 5, 8]])


    import sys

    sys.exit()


    cov_means = np.cov(mcc.logLs_store.T)
    var_means = np.diag(cov_means)
    corr_means = cov_means / np.sqrt(var_means)
    corr_means = (corr_means.T / np.sqrt(var_means)).T

    plt.plot(corr_means[argT_1])
    plt.show()


    plt.imshow(corr_means - np.eye(n_chain))
    plt.show()

    plt.plot(np.sum(cov_means, axis=0) / var_means)
    plt.show()


    import sys

    sys.exit()


    from scipy.interpolate import InterpolatedUnivariateSpline

    import integ_box_filt as ibf

    rs_stack = np.hstack(rs_save)[n_burnin:]

    counts, bins, _ = plt.hist(rs_stack, 10000, range=[0., np.sqrt(n_par) * like_obj.high_lims[1]], density=True)

    plt.plot(ibf.rs, ibf.get_density_pred(1.))
    plt.show()

    integ_true = cumulative_trapezoid(ibf.get_density_pred(1.)[::-1], ibf.rs[::-1], initial=0)[::-1] + 1
    integ_loc = cumulative_trapezoid(counts[::-1], bins[::-1][1:], initial=0)[::-1] + 1
    interp_true = InterpolatedUnivariateSpline(ibf.rs, integ_true, k=3, ext=2)(bins[:bins.size - 1])

    plt.plot(ibf.rs, integ_true)
    plt.plot(bins[:bins.size - 1], integ_loc)
    plt.show()

    plt.plot(bins[:bins.size - 1], integ_loc - interp_true)
    plt.show()

    n_use = rs_stack.size

    rs_mean = np.mean(rs_stack)
    r2s_mean = np.mean(rs_stack**2)
    autocorr_rs = scipy.signal.correlate(rs_stack - rs_mean, rs_stack - rs_mean, mode='full')
    avg_len = 32
    autocorr_rs_lim = np.hstack([autocorr_rs[n_use - 1], autocorr_rs[n_use:2 * n_use - avg_len:avg_len]])
    for itrb in range(1, avg_len):
        autocorr_rs_lim[1:] += autocorr_rs[n_use + itrb:2 * n_use - avg_len + itrb:avg_len]

    autocorr_rs_lim[1:] /= avg_len

    plt.plot(autocorr_rs_lim / autocorr_rs_lim[0])
    plt.show()

    arg_cut_r = np.argmax(autocorr_rs_lim / autocorr_rs_lim[0] < 0.)

    autocorr_len_inferred = 1 + avg_len * np.sum(autocorr_rs_lim[1:arg_cut_r] / autocorr_rs_lim[0])
    print(np.max(interp_true - integ_loc), np.min(interp_true - integ_loc))
    print(rs_mean, r2s_mean - rs_mean**2)
    print(arg_cut_r * 16 + 1, autocorr_len_inferred, (arg_cut_r * 16 + 1) / autocorr_len_inferred)

    integ_entropy = -cumulative_trapezoid(cumulants[1][::-1] * mcc.betas[::-1], mcc.betas[::-1], initial=0)[::-1]
    print(integ_entropy[argT_1])

    shuffle_length = 1000
    rs_shuffle = rs_stack.copy()
    np.random.shuffle(rs_shuffle)
    means_shuffle = np.mean(rs_shuffle.reshape((rs_shuffle.size // shuffle_length, shuffle_length)), axis=1)
    means_stack = np.mean(rs_stack.reshape((rs_shuffle.size // shuffle_length, shuffle_length)), axis=1)
    autocorr_len_1k = np.var(means_stack) / np.var(means_shuffle)

    shuffle_length = 10000
    rs_shuffle = rs_stack.copy()
    np.random.shuffle(rs_shuffle)
    means_shuffle = np.mean(rs_shuffle.reshape((rs_shuffle.size // shuffle_length, shuffle_length)), axis=1)
    means_stack = np.mean(rs_stack.reshape((rs_shuffle.size // shuffle_length, shuffle_length)), axis=1)
    autocorr_len_10k = np.var(means_stack) / np.var(means_shuffle)

    shuffle_length = 100000
    rs_shuffle = rs_stack.copy()
    np.random.shuffle(rs_shuffle)
    means_shuffle = np.mean(rs_shuffle.reshape((rs_shuffle.size // shuffle_length, shuffle_length)), axis=1)
    means_stack = np.mean(rs_stack.reshape((rs_shuffle.size // shuffle_length, shuffle_length)), axis=1)
    autocorr_len_100k = np.var(means_stack) / np.var(means_shuffle)

    shuffle_length = 1000000
    rs_shuffle = rs_stack.copy()
    np.random.shuffle(rs_shuffle)
    means_shuffle = np.mean(rs_shuffle.reshape((rs_shuffle.size // shuffle_length, shuffle_length)), axis=1)
    means_stack = np.mean(rs_stack.reshape((rs_shuffle.size // shuffle_length, shuffle_length)), axis=1)
    autocorr_len_1m = np.var(means_stack) / np.var(means_shuffle)

    print(autocorr_len_1k, autocorr_len_10k, autocorr_len_100k, autocorr_len_1m)


    import sys

    sys.exit()


    first_exchange_accept_log = np.zeros(len(mcc.tracker_manager.exchange_archive))
    acceptance_log = np.zeros((len(mcc.tracker_manager.exchange_archive), n_chain, mcc.tracker_manager.accept_archive[0].shape[2]))
    tracker_archive_old = np.zeros_like(mcc.tracker_manager.exchange_archive[0])
    accept_archive_old = np.zeros_like(mcc.tracker_manager.accept_archive[0])

    for itri in range(0, len(mcc.tracker_manager.exchange_archive), 1):
        tracker_archive = mcc.tracker_manager.exchange_archive[itri]
        accept_archive = mcc.tracker_manager.accept_archive[itri]
        a_ex_yes_loc = (tracker_archive[0] - tracker_archive_old[0])[argT_sort, :][:, argT_sort]
        a_ex_no_loc = (tracker_archive[1] - tracker_archive_old[1])[argT_sort, :][:, argT_sort]
        first_exchange_accept_log[itri] = a_ex_yes_loc[1, 0] / (a_ex_yes_loc[1, 0] + a_ex_no_loc[1, 0])
        tracker_archive_old = tracker_archive
        a_yes_loc = (accept_archive[0] - accept_archive_old[0])
        a_no_loc = (accept_archive[1] - accept_archive_old[1])
        acceptance_log[itri] = a_yes_loc / (a_yes_loc + a_no_loc)
        accept_archive_old = accept_archive


    import sys

    sys.exit()


    # acceptance_log = np.zeros((len(mcc.tracker_manager.cycle_archive),n_chain,mcc.tracker_manager.accept_archive[0].shape[2]))
    cycle_min_log = np.zeros(len(mcc.tracker_manager.cycle_archive))
    cycle_zero_log = np.zeros(len(mcc.tracker_manager.cycle_archive))
    diff_ext_log = np.zeros(len(mcc.tracker_manager.cycle_archive))
    cycle_archive_old = np.zeros_like(mcc.tracker_manager.cycle_archive[0])

    for itri in range(0, len(mcc.tracker_manager.cycle_archive), 1):
        cycle_archive = mcc.tracker_manager.cycle_archive[itri]
        new_cycles = np.min(cycle_archive[2:4, :] - cycle_archive_old[2:4, :], axis=0)
        diff_cycles = mcc.tracker_manager.itrn_archive[itri] - np.min(cycle_archive[0:2, :], axis=0)
        cycle_min_log[itri] = np.min(new_cycles)
        diff_ext_log[itri] = np.max(diff_cycles)
        cycle_zero_log[itri] = np.sum(new_cycles == 0)
        cycle_archive_old = cycle_archive


    print(np.min(diff_ext_log[cycle_burn_index:]), np.max(diff_ext_log[cycle_burn_index:]), np.median(diff_ext_log[cycle_burn_index:]), np.mean(diff_ext_log[cycle_burn_index:]), np.std(diff_ext_log[cycle_burn_index:]))
    print(np.min(cycle_zero_log[cycle_burn_index:]), np.max(cycle_zero_log[cycle_burn_index:]), np.median(cycle_zero_log[cycle_burn_index:]), np.mean(cycle_zero_log[cycle_burn_index:]), np.std(cycle_zero_log[cycle_burn_index:]))
    print(np.max(cycle_min_log[cycle_burn_index:]), np.median(cycle_min_log[cycle_burn_index:]), np.mean(cycle_min_log[cycle_burn_index:]), np.std(cycle_min_log[cycle_burn_index:]))
