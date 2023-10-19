"""C 2023 Matthew C. Digman
code example to run the galactic binary parameter estimation pipeline
and plot results"""

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from time import perf_counter

import numpy as np
from numba import njit

import configparser

#import likelihood_gb as trial_likelihood
import cake_likelihood as trial_likelihood

from DTMCMC.dtmcmc_sampler import DTMCMCSampler
from DTMCMC.temperature_ladder_helpers import GeometricTemperatureLadder
import DTMCMC.temperature_ladder_helpers as th
from DTMCMC.corr_summary_helpers import CorrelationSummary
from DTMCMC.proposal_strategy_helpers import ProposalStrategyParameters
from DTMCMC.proposal_manager_helper import get_default_proposal_manager
import DTMCMC.exchange_manager as eh
from entropy_process import unit_normal_battery
import diagnostic_commentary_helpers as dch

# TODO reduce exposure of block_size parameter

if __name__ == '__main__':
    t0 = perf_counter()

    # starting variables
    n_chain = 32                       # number of total chains for parallel tempering
    n_cold = 1                         # number of T=1 chains for parallel tempering
    n_burnin = 100000                    # number of iterations to discard as burn in
    block_size = 1000                  # number of iterations per block when advancing the chain state
    store_size = 400000                # number of samples to store total
    N_blocks = store_size//block_size  # number of blocks the sampler must iterate through
    n_par = 5

    T_max = 1.e8                       # maximum temperature for geometric part of temperature ladder

    params_true = np.zeros(n_par)      # true parameters for search

    # create needed objects
    T_ladder = GeometricTemperatureLadder(n_chain, n_cold=n_cold, T_max=T_max,T_min=1000.,T_cold=1000.)  # get the temperature ladder object
    #T_ladder = th.entropy_ladder_fromfile(n_chain,n_cold,'Ts_cake1.npy','vars_cake1.npy',use_inf_final=True,T_cold=1.,correct_last=False)

    like_obj = trial_likelihood.CakeLikelihood(n_par)
    params_true = like_obj.correct_bounds(params_true)                 # make sure the conventions on the parameters match

    # create the starting samples
    starting_samples = np.zeros((T_ladder.n_chain, like_obj.n_par))
    for itrt in range(0, n_chain):
        # start from prior draws
        starting_samples[itrt] = like_obj.prior_draw()

    # create the overarching proposal manager object
    exchange_manager = eh.ExchangeManager(strategy=eh.RANDOM_TARGETS,track_full_exchanges=True)
    proposal_manager = get_default_proposal_manager(T_ladder, like_obj, starting_samples,exchange_manager_loc=exchange_manager)

    print('Chain parameters', n_cold, n_chain, n_burnin, block_size, store_size, T_max)

    # create the chain object
    mcc = DTMCMCSampler(T_ladder, like_obj,  block_size, store_size, starting_samples=starting_samples, n_record=n_chain,proposal_manager=proposal_manager)

    t_init_end = perf_counter()
    print('all objects initialized in ', t_init_end-t0, 's')

    t_advance_begin = perf_counter()

    # the main loop which actually advances the MCMC state
    mcc.advance_N_blocks(N_blocks)
    mcc.advance_N_blocks(N_blocks)
    mcc.advance_N_blocks(N_blocks)
    #mcc.advance_N_blocks(N_blocks)
    #mcc.advance_N_blocks(N_blocks)
    #mcc.advance_N_blocks(N_blocks)
    #mcc.advance_N_blocks(N_blocks)
    #mcc.advance_N_blocks(N_blocks)

    t_advance_end = perf_counter()
    print('advanced in ', t_advance_end-t_advance_begin, 's')

    # generate some summary information
    corr_sum = CorrelationSummary()
    corr_sum.summarize_blocks(mcc, n_burnin)
    corr_sum.final_prints(mcc, n_burnin)

    # get flattened samples for plotting
    samples_flattened, logLs_flattened = mcc.get_stored_flattened(corr_sum.restrict_n_burnin(mcc, n_burnin),n_chain_out=n_cold)

    tf = perf_counter()

    print('full search time ', str(tf-t0)+'s')

do_sigma_plot = True
if do_sigma_plot:
    #samples_got = mcc.samples_store[n_burnin:]
    #samples_post = trial_likelihood.drawposterior(store_size-n_burnin,mcc.Ts,n_par,like_obj.cutoff)
    #sigma_got = np.std(samples_got,axis=(0,2))
    #sigma_post = np.std(samples_post,axis=(0,2))
    #import matplotlib.pyplot as plt
    #plt.plot(mcc.Ts,sigma_got)
    #plt.plot(mcc.Ts,sigma_post)
    #plt.plot(mcc.Ts,np.sqrt(mcc.Ts))
    #plt.show()

    #plt.semilogx(mcc.Ts,1-sigma_got/sigma_post)

    #plt.show()

    #print(unit_normal_battery(np.reshape(samples_post[:,0],samples_post[:,0].size),do_assert=False))
    #print(unit_normal_battery(np.reshape(samples_got[:,0],samples_got[:,0].size),do_assert=False))
    dch.print_diagnostic_commentary(mcc)

argT_1 = np.argmax(T_ladder.Ts==T_ladder.T_cold)

do_corner_plot = True
if do_corner_plot:
    # generate a corner plot
    import matplotlib.pyplot as plt
    import corner

    # reformat the samples to make the plots look nicer
    labels = like_obj.get_labels()
    samples_format, params_true_format = like_obj.format_samples_output(mcc.samples_store[n_burnin:,argT_1,:].copy(), params_true)

    # create the corner plot figure
    fig = plt.figure(figsize=(10, 7.5))
    figure = corner.corner(samples_format, fig=fig, bins=25, hist_kwargs={"density": True}, show_titles=True, title_fmt=None,
                           title_kwargs={"fontsize": 12}, labels=labels, max_n_ticks=3, label_kwargs={"fontsize": 12}, labelpad=0.15,
                           smooth=0.25, levels=[0.682, 0.954])

    # overplot the true parameters
    corner.overplot_points(figure, params_true_format[None], marker="s", color='tab:blue', markersize=4)
    corner.overplot_lines(figure, params_true_format, color='tab:blue')

    # adjust the figure to fit the box better
    fig.subplots_adjust(wspace=0., hspace=0., left=0.05, top=0.95, right=0.99, bottom=0.05)
    for ax in figure.get_axes():
        ax.tick_params(which='both', direction='in', bottom=True, top=True, left=True, right=True, labelsize=6)
    plt.show()

rs_got = np.sqrt(np.sum(mcc.samples_store[n_burnin:,argT_1,:]**2,axis=1))
rs_got = rs_got[rs_got<10.]
counts,bins,_ = plt.hist(rs_got,1000,density=True)
bins_match = np.unique(np.hstack([np.linspace(0.,1.,100),bins,np.array([10.])]))
bin_likes = np.zeros(len(bins_match))
for itrb in range(bin_likes.size):
    bin_likes[itrb] = like_obj.get_loglike(np.array([bins_match[itrb],0.,0.,0.,0.]))


dens_pred = np.exp(T_ladder.betas[argT_1]*bin_likes)*bins_match**4*2*np.pi
dens_pred /= np.trapz(dens_pred,bins_match)

plt.plot(bins_match,dens_pred)
plt.show()

block_burnin = 400

argTs = np.argsort(T_ladder.Ts)
#plt.semilogx(T_ladder.Ts[argTs],np.gradient(np.mean(mcc.logLs_store[n_burnin:],axis=0)[argTs],T_ladder.betas[argTs])*T_ladder.betas[argTs]**2)
plt.semilogx(T_ladder.Ts[argTs],np.gradient(np.mean(np.array(mcc.logL_means[block_burnin:]),axis=0)[argTs],T_ladder.betas[argTs])*T_ladder.betas[argTs]**2)
plt.semilogx(T_ladder.Ts[argTs],(np.mean(np.array(mcc.logL2_means[block_burnin:]),axis=0)-np.mean(np.array(mcc.logL_means[block_burnin:]),axis=0)**2)[argTs]*T_ladder.betas[argTs]**2)
plt.semilogx(T_ladder.Ts[argTs],np.var(mcc.logLs_store[n_burnin:],axis=0)[argTs]*T_ladder.betas[argTs]**2)
plt.show()

import scipy.signal

n_use = mcc.store_size+1
for itrt in range(0,10):
    logL_diff = mcc.logLs_store[:,itrt]-np.mean(mcc.logLs_store[:,itrt])
    autocorr_logL = scipy.signal.correlate(logL_diff,logL_diff, mode='full')
    autocorr_logL_lim = np.hstack([autocorr_logL[n_use-1:n_use], autocorr_logL[n_use:2*n_use-2:2]+autocorr_logL[n_use+1:2*n_use-1:2]])
    plt.plot(autocorr_logL_lim/autocorr_logL_lim[0])


plt.show()


logL_means_store = np.array(mcc.logL_means[block_burnin:])

n_use = logL_means_store.shape[0]+1
for itrt in range(120,135):
    logL_diff = logL_means_store[:,itrt]-np.mean(logL_means_store[:,itrt])
    autocorr_logL = scipy.signal.correlate(logL_diff,logL_diff, mode='full')
    plt.plot(autocorr_logL[n_use-1:2*n_use-1]/autocorr_logL[n_use-1])


plt.show()

argT_sort = np.argsort(T_ladder.Ts)
a_ex_yes = (mcc.tracker_manager.exchange_tracker[0]+mcc.tracker_manager.exchange_tracker[0].T)[argT_sort,:][:,argT_sort]
a_ex_no = (mcc.tracker_manager.exchange_tracker[1]+mcc.tracker_manager.exchange_tracker[1].T)[argT_sort,:][:,argT_sort]

accept_exchange = a_ex_yes/(a_ex_yes+a_ex_no)
accept_exchange_nn = np.zeros(n_chain)
accept_exchange_nn[n_chain-1] = accept_exchange[n_chain-2,n_chain-1]
accept_exchange_nn[0] = accept_exchange[0,1]
for itrt in range(1,n_chain-1):
    accept_exchange_nn[itrt] = (accept_exchange[itrt,itrt+1]+accept_exchange[itrt,itrt-1])/2.


Ts_old = np.load('Ts_cake1.npy')
vars_old = np.load('vars_cake1.npy')
plt.loglog(Ts_old,vars_old)
plt.loglog(T_ladder.Ts[argTs],(np.mean(np.array(mcc.logL2_means[block_burnin:]),axis=0)-np.mean(np.array(mcc.logL_means[block_burnin:]),axis=0)**2)[argTs])
plt.show()
