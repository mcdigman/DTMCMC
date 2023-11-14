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
import moment_helpers
from scipy.integrate import cumtrapz

# TODO reduce exposure of block_size parameter

if __name__ == '__main__':
    t0 = perf_counter()

    # starting variables
    n_chain = 16                       # number of total chains for parallel tempering
    n_cold = 1                         # number of T=1 chains for parallel tempering
    n_burnin = 5000                    # number of iterations to discard as burn in
    block_size = 10000                  # number of iterations per block when advancing the chain state
    store_size = 50000                # number of samples to store total
    N_blocks = store_size//block_size  # number of blocks the sampler must iterate through
    n_par = 5

    T_max = 1.e3                       # maximum temperature for geometric part of temperature ladder

    params_true = np.zeros(n_par)      # true parameters for search

    # create needed objects
    #T_ladder = GeometricTemperatureLadder(n_chain, n_cold=n_cold, T_max=T_max,T_min=80.,T_cold=80.)  # get the temperature ladder object
    #T_ladder = th.entropy_ladder_fromfile(n_chain,n_cold,'Ts_cake_combo2.npy','vars_cake_combo2.npy',use_inf_final=True,T_cold=1.,correct_last=False)
    T_ladder = th.entropy_ladder_fromfile(n_chain,n_cold,'Ts_interpolate_cake_alternate1.npy','vars_interpolate_cake_alternate1.npy',use_inf_final=True,T_cold=1.,correct_last=False)

    like_obj = trial_likelihood.CakeLikelihood(n_par)
    params_true = like_obj.correct_bounds(params_true)                 # make sure the conventions on the parameters match

    # create the starting samples
    starting_samples = np.zeros((T_ladder.n_chain, like_obj.n_par))
    for itrt in range(0, n_chain):
        # start from prior draws
        starting_samples[itrt] = like_obj.prior_draw()

    # create the overarching proposal manager object
    exchange_manager = eh.ExchangeManager(strategy=eh.ALTERNATE_SEQUENTIAL_TARGETS,track_full_exchanges=True)
    proposal_manager = get_default_proposal_manager(T_ladder, like_obj, starting_samples,exchange_manager_loc=exchange_manager)

    print('Chain parameters', n_cold, n_chain, n_burnin, block_size, store_size, T_max)

    # create the chain object
    mcc = DTMCMCSampler(T_ladder, like_obj,  block_size, store_size, starting_samples=starting_samples, n_record=n_chain,proposal_manager=proposal_manager)

    t_init_end = perf_counter()
    print('all objects initialized in ', t_init_end-t0, 's')

    t_advance_begin = perf_counter()

    argT_1 = np.argmax(T_ladder.Ts==T_ladder.T_cold)

    # the main loop which actually advances the MCMC state
    for itrb in range(5):
        mcc.advance_N_blocks(4*N_blocks)
    #mcc.advance_N_blocks(N_blocks)
    #mcc.advance_N_blocks(N_blocks)
    #mcc.advance_N_blocks(N_blocks)
    #mcc.advance_N_blocks(N_blocks)
    #mcc.advance_N_blocks(N_blocks)
    #mcc.advance_N_blocks(N_blocks)
    #mcc.advance_N_blocks(N_blocks)
    #mcc.advance_N_blocks(N_blocks)
    #mcc.advance_N_blocks(N_blocks)
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

block_burnin = 60

argTs = np.argsort(T_ladder.Ts)
#plt.semilogx(T_ladder.Ts[argTs],np.gradient(np.mean(mcc.logLs_store[n_burnin:],axis=0)[argTs],T_ladder.betas[argTs])*T_ladder.betas[argTs]**2)
plt.semilogx(T_ladder.Ts[argTs],np.gradient(np.mean(np.array(mcc.logL_means[block_burnin:]),axis=0)[argTs],T_ladder.betas[argTs])*T_ladder.betas[argTs]**2)
plt.semilogx(T_ladder.Ts[argTs],(np.mean(np.array(mcc.logL2_means[block_burnin:]),axis=0)-np.mean(np.array(mcc.logL_means[block_burnin:]),axis=0)**2)[argTs]*T_ladder.betas[argTs]**2)
#plt.semilogx(T_ladder.Ts[argTs],np.var(mcc.logLs_store[n_burnin:],axis=0)[argTs]*T_ladder.betas[argTs]**2)
plt.show()

import scipy.signal

n_use = mcc.store_size+1
for itrt in range(max(0,argT_1-5),min(n_chain,argT_1+5)):
    logL_diff = mcc.logLs_store[:,itrt]-np.mean(mcc.logLs_store[:,itrt])
    autocorr_logL = scipy.signal.correlate(logL_diff,logL_diff, mode='full')
    autocorr_logL_lim = np.hstack([autocorr_logL[n_use-1:n_use], autocorr_logL[n_use:2*n_use-2:2]+autocorr_logL[n_use+1:2*n_use-1:2]])
    plt.plot(autocorr_logL_lim/autocorr_logL_lim[0])


plt.show()


logL_means_store = np.array(mcc.logL_means[block_burnin:])

n_use = logL_means_store.shape[0]+1
for itrt in  range(max(0,argT_1-10),min(n_chain,argT_1+10)):
    logL_diff = logL_means_store[:,itrt]-np.mean(logL_means_store[:,itrt])
    autocorr_logL = scipy.signal.correlate(logL_diff,logL_diff, mode='full')
    plt.plot(autocorr_logL[n_use-2:2*n_use-1]/autocorr_logL[n_use-2])



plt.show()

logL_means_store = np.array(mcc.logL_means[block_burnin:])
logL_diff0 = logL_means_store[:,argT_1]-np.mean(logL_means_store[:,argT_1])

n_use = logL_means_store.shape[0]+1
for itrt in  range(max(0,argT_1-20),min(n_chain,argT_1+20)):
    logL_diff = logL_means_store[:,itrt]-np.mean(logL_means_store[:,itrt])
    crosscorr_logL = scipy.signal.correlate(logL_diff,logL_diff0, mode='full')
    plt.plot(crosscorr_logL[n_use-2:2*n_use-1])



plt.show()

logL_diff0 = mcc.logLs_store[:,argT_1]-np.mean(mcc.logLs_store[:,argT_1])

n_use = mcc.logLs_store.shape[0]+1
for itrt in  range(max(0,argT_1-10),min(n_chain,argT_1)):
    logL_diff = mcc.logLs_store[:,itrt]-np.mean(mcc.logLs_store[:,itrt])
    crosscorr_logL = scipy.signal.correlate(logL_diff,logL_diff0, mode='full')
    #plt.plot(crosscorr_logL[n_use-1:2*n_use-2:2]+crosscorr_logL[n_use:2*n_use-1:2])
    b1 = n_use-1
    b2 = 2*n_use
    plt.plot(crosscorr_logL[b1:b2-5:4]+crosscorr_logL[b1+1:b2-4:4]+crosscorr_logL[b1+2:b2-3:4]+crosscorr_logL[b1+3:b2-2:4])


#plt.xlim(-1,500)
plt.show()

argT_sort = np.argsort(T_ladder.Ts)
a_ex_yes = (mcc.tracker_manager.exchange_tracker[0]+mcc.tracker_manager.exchange_tracker[0].T)[argT_sort,:][:,argT_sort]
a_ex_no = (mcc.tracker_manager.exchange_tracker[1]+mcc.tracker_manager.exchange_tracker[1].T)[argT_sort,:][:,argT_sort]

accept_exchange = a_ex_yes/(a_ex_yes+a_ex_no)
accept_exchange_nn_left = np.zeros(n_chain)
accept_exchange_nn_right = np.zeros(n_chain)
accept_exchange_nn = np.zeros(n_chain)
accept_exchange_nn_left[n_chain-1] = accept_exchange[n_chain-2,n_chain-1]
accept_exchange_nn_right[0] = accept_exchange[0,1]

accept_exchange_nn[n_chain-1] = accept_exchange[n_chain-2,n_chain-1]
accept_exchange_nn[0] = accept_exchange[0,1]
for itrt in range(1,n_chain-1):
    accept_exchange_nn_right[itrt] = accept_exchange[itrt,itrt+1]
    accept_exchange_nn_left[itrt] = accept_exchange[itrt,itrt-1]
    accept_exchange_nn[itrt] =  (a_ex_yes[itrt,itrt+1]+a_ex_yes[itrt,itrt-1])/(a_ex_yes[itrt,itrt+1]+a_ex_no[itrt,itrt+1]+a_ex_yes[itrt,itrt-1]+a_ex_no[itrt,itrt-1])


plt.plot(T_ladder.betas[:n_chain-1],accept_exchange_nn_right[:n_chain-1])
plt.plot(T_ladder.betas[1:],accept_exchange_nn_left[1:])
plt.plot(T_ladder.betas,accept_exchange_nn)
plt.show()

Ts_old = np.load('Ts_cake_combo2.npy')
vars_old = np.load('vars_cake_combo2.npy')
plt.loglog(Ts_old,vars_old)
plt.loglog(T_ladder.Ts[argTs],(np.mean(np.array(mcc.logL2_means[block_burnin:]),axis=0)-np.mean(np.array(mcc.logL_means[block_burnin:]),axis=0)**2)[argTs])
plt.show()

plt.semilogx(Ts_old,vars_old/Ts_old**2)
plt.semilogx(T_ladder.Ts[argTs],(np.mean(np.array(mcc.logL2_means[block_burnin:]),axis=0)-np.mean(np.array(mcc.logL_means[block_burnin:]),axis=0)**2)[argTs]*T_ladder.betas[argTs]**2)
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

Ts_high_combo = np.hstack([Ts_high_high,Ts_mid_high,Ts_low_high,Ts_evolve])
vars_high_combo = np.hstack([vars_high_high,vars_mid_high,vars_low_high,vars_evolve])
means_high_combo = np.hstack([means_high_high,means_mid_high,means_low_high,means_evolve])


args_high = np.argsort(Ts_high_combo)
vars_high_combo = vars_high_combo[args_high]
means_high_combo = means_high_combo[args_high]
Ts_high_combo = Ts_high_combo[args_high]

#np.save('Ts_cake_combo2.npy',Ts_high_combo)
#np.save('vars_cake_combo2.npy',vars_high_combo)
#np.save('means_cake_combo2.npy',means_high_combo)

plt.semilogx(Ts_old,vars_old/Ts_old**2)
plt.plot(Ts_high_combo,vars_high_combo/Ts_high_combo**2)
plt.show()

import sys
sys.exit()

cold_save = [mcc.samples_store[:,argT_1].copy()]



for itrb in range(2):
    mcc.advance_N_blocks(N_blocks)
    cold_save.append(mcc.samples_store[:,argT_1].copy())



cumulants = np.array(moment_helpers.get_cumulants(moment_helpers.get_averaged_means(mcc,len(mcc.logL_means)-block_burnin,cut=block_burnin)))[:,0]

import sys
sys.exit()

rs_got = np.sqrt(np.sum(np.vstack(cold_save)[block_burnin*block_size::10]**2,axis=1))
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

import sys
sys.exit()

cumulants = np.array(moment_helpers.get_cumulants(moment_helpers.get_averaged_means(mcc,len(mcc.logL_means)-block_burnin,cut=block_burnin)))[:,0]

plt.plot((cumtrapz(cumulants[1],mcc.betas,initial=0.)+cumulants[0][0])*mcc.betas**1)
plt.plot(cumulants[0]*mcc.betas**1)
plt.show()

plt.plot(np.gradient(cumulants[0],mcc.betas)*mcc.betas**2)
plt.plot((cumtrapz(cumulants[2],mcc.betas,initial=0.)+cumulants[1][0])*mcc.betas**2)
plt.plot(cumulants[1]*mcc.betas**2)
plt.show()


plt.plot(np.gradient(cumulants[1],mcc.betas)*mcc.betas**3)
plt.plot((cumtrapz(cumulants[3],mcc.betas,initial=0.)+cumulants[2][0])*mcc.betas**3)
plt.plot(cumulants[2]*mcc.betas**3)
plt.show()

plt.plot(np.gradient(cumulants[2],mcc.betas)*mcc.betas**4)
plt.plot((cumtrapz(cumulants[4],mcc.betas,initial=0.)+cumulants[3][0])*mcc.betas**4)
plt.plot(cumulants[3]*mcc.betas**4)
plt.show()

plt.plot(np.gradient(cumulants[3],mcc.betas)*mcc.betas**5)
plt.plot((cumtrapz(cumulants[5],mcc.betas,initial=0.)+cumulants[4][0])*mcc.betas**5)
plt.plot(cumulants[4]*mcc.betas**5)
plt.show()

plt.plot(np.gradient(cumulants[4],mcc.betas)*mcc.betas**6)
plt.plot(cumulants[5]*mcc.betas**6)
plt.show()

plt.plot(np.gradient(cumulants[5],mcc.betas)*mcc.betas**7)
plt.show()

import sys
sys.exit()

cov_means = np.cov((moment_helpers.get_averaged_means(mcc,(len(mcc.logL_means)-block_burnin)//10,cut=block_burnin))[0].T)
var_means = np.diag(cov_means)
corr_means = cov_means/np.sqrt(var_means)
corr_means = (corr_means.T/np.sqrt(var_means)).T

plt.plot(corr_means[argT_1])
plt.show()


import sys
sys.exit()


plt.plot(moment_helpers.get_corr_quantities(moment_helpers.get_averaged_means(mcc,len(mcc.logL_means)-block_burnin,cut=block_burnin),moment_helpers.get_averaged_adjacents(mcc,len(mcc.logL_means)-block_burnin,cut=block_burnin))[1][0][0])
plt.show()

plt.plot(moment_helpers.get_corr_quantities(moment_helpers.get_averaged_means(mcc,len(mcc.logL_means)-block_burnin,cut=block_burnin),moment_helpers.get_averaged_adjacents(mcc,len(mcc.logL_means)-block_burnin,cut=block_burnin))[1][1][0])
plt.plot(moment_helpers.get_corr_quantities(moment_helpers.get_averaged_means(mcc,len(mcc.logL_means)-block_burnin,cut=block_burnin),moment_helpers.get_averaged_adjacents(mcc,len(mcc.logL_means)-block_burnin,cut=block_burnin))[1][2][0])
plt.show()


import sys
sys.exit()

import integral_heat_estimator
cumulants = np.array(moment_helpers.get_cumulants(moment_helpers.get_averaged_means(mcc,len(mcc.logL_means)-block_burnin,cut=block_burnin)))[:,0]
integrand_left1,integrand_right1,integrand_avg1 = integral_heat_estimator.cumulant_integrand(cumulants[:2],mcc.betas)
integrand_left2,integrand_right2,integrand_avg2 = integral_heat_estimator.cumulant_integrand(cumulants[:3],mcc.betas)
integrand_left3,integrand_right3,integrand_avg3 = integral_heat_estimator.cumulant_integrand(cumulants[:4],mcc.betas)
integrand_left4,integrand_right4,integrand_avg4 = integral_heat_estimator.cumulant_integrand(cumulants[:5],mcc.betas)
integrand_left5,integrand_right5,integrand_avg5 = integral_heat_estimator.cumulant_integrand(cumulants[:6],mcc.betas)
plt.plot(mcc.betas[1:],np.cumsum(integrand_avg1))
plt.plot(mcc.betas[1:],np.cumsum(integrand_avg2))
plt.plot(mcc.betas[1:],np.cumsum(integrand_avg3))
plt.plot(mcc.betas[1:],np.cumsum(integrand_avg4))
plt.plot(mcc.betas[1:],np.cumsum(integrand_avg5))
plt.show()

plt.plot(mcc.betas[1:],np.cumsum(integrand_left1))
plt.plot(mcc.betas[1:],np.cumsum(integrand_left2))
plt.plot(mcc.betas[1:],np.cumsum(integrand_left3))
plt.plot(mcc.betas[1:],np.cumsum(integrand_left4))
plt.plot(mcc.betas[1:],np.cumsum(integrand_left5))
plt.show()

plt.plot(mcc.betas[1:],np.cumsum(integrand_right1))
plt.plot(mcc.betas[1:],np.cumsum(integrand_right2))
plt.plot(mcc.betas[1:],np.cumsum(integrand_right3))
plt.plot(mcc.betas[1:],np.cumsum(integrand_right4))
plt.plot(mcc.betas[1:],np.cumsum(integrand_right5))
plt.show()



import sys
sys.exit()

#this test detects the correlations produced by the different exchange strategies
#the chain translate version detects the autocorrelations in chain position
#chain_track_loc = np.vstack(chain_track_hist)
chain_track_loc = np.vstack(mcc.chain_track)

#chain_translate = np.zeros((chain_track_loc.shape[0],n_chain))
#for itrt in range(0,n_chain):
#    chain_translate[:,itrt] = np.argmax(chain_track_loc==itrt,axis=1)


n_use = chain_track_loc.shape[0]
#chain_track_loc = None
autocorr_chain_sum = np.zeros(2*n_use-1)
chain_diff0 = chain_track_loc[:,argT_1]-np.mean(chain_track_loc[:,argT_1])
for itrt in  range(0,n_chain):
    chain_diff = chain_track_loc[:,itrt]-np.mean(chain_track_loc[:,itrt])
    #autocorr_chain_sum += scipy.signal.correlate(chain_diff,chain_diff, mode='full')
    autocorr_chain = scipy.signal.correlate(chain_diff,chain_diff0, mode='full')
    #plt.plot(autocorr_chain[n_use-1:n_use-1+2000]/autocorr_chain[n_use-1])
    plt.plot(autocorr_chain[n_use-1:n_use-1+2000])



#plt.plot(autocorr_chain_sum[n_use-1:]/autocorr_chain_sum[n_use-1])
plt.show()


import sys
sys.exit()

#chain_track_hist = [mcc.chain_track[1:].copy()]
#
#for itrb in range(0,9):
#    mcc.advance_N_blocks(1)
#    chain_track_hist.append(mcc.chain_track[1:].copy())
import integral_heat_estimator
betas_old = th.Ts_to_betas(Ts_old)
max_beta = max(np.max(mcc.betas),np.max(betas_old))
betas_new = np.unique(np.hstack([mcc.betas,betas_old,np.linspace(0.5,2.,10000),10**np.linspace(np.log10(max_beta),-10.,10000),np.linspace(max_beta,0.,10000)]))[::-1]
Ts_new = th.betas_to_Ts(betas_new)

cumulants = np.array(moment_helpers.get_cumulants(moment_helpers.get_averaged_means(mcc,len(mcc.logL_means)-block_burnin,cut=block_burnin)))[:,0]

estim_left,estim_right,estim_center = integral_heat_estimator.cumulant_heat_cap_interp(cumulants,mcc.betas,betas_new)
estim_left1,estim_right1,estim_center1 = integral_heat_estimator.cumulant_heat_cap_interp(cumulants[:2],mcc.betas,betas_new)
estim_left2,estim_right2,estim_center2 = integral_heat_estimator.cumulant_heat_cap_interp(cumulants[:3],mcc.betas,betas_new)
estim_left3,estim_right3,estim_center3 = integral_heat_estimator.cumulant_heat_cap_interp(cumulants[:4],mcc.betas,betas_new)
estim_left4,estim_right4,estim_center4 = integral_heat_estimator.cumulant_heat_cap_interp(cumulants[:5],mcc.betas,betas_new)

plt.loglog(Ts_new,estim_left)
plt.loglog(Ts_new,estim_right)
plt.loglog(Ts_new,estim_center)
plt.loglog(Ts_old,vars_old*betas_old**2,'k--')
plt.show()

plt.semilogx(Ts_new,np.gradient(estim_center4,betas_new))
plt.semilogx(Ts_new,np.gradient(estim_center,betas_new))
plt.semilogx(Ts_old,np.gradient(vars_old*betas_old**2,betas_old),'k--')
plt.show()


import sys 
sys.exit()


cycle_burn_index = np.argmax(np.array(mcc.tracker_manager.itrn_archive)==2000000)

tracker_archive = mcc.tracker_manager.exchange_archive[cycle_burn_index]

argT_sort = np.argsort(T_ladder.Ts)
a_ex_yes = (mcc.tracker_manager.exchange_tracker[0]-tracker_archive[0])[argT_sort,:][:,argT_sort]
a_ex_no =  (mcc.tracker_manager.exchange_tracker[1]-tracker_archive[1])[argT_sort,:][:,argT_sort]

accept_exchange = a_ex_yes/(a_ex_yes+a_ex_no)
accept_exchange_nn_left = np.zeros(n_chain)
accept_exchange_nn_right = np.zeros(n_chain)
accept_exchange_nn = np.zeros(n_chain)
accept_exchange_nn_left[n_chain-1] = accept_exchange[n_chain-2,n_chain-1]
accept_exchange_nn_right[0] = accept_exchange[0,1]

accept_exchange_nn[n_chain-1] = accept_exchange[n_chain-2,n_chain-1]
accept_exchange_nn[0] = accept_exchange[0,1]
for itrt in range(1,n_chain-1):
    accept_exchange_nn_right[itrt] = accept_exchange[itrt,itrt+1]
    accept_exchange_nn_left[itrt] = accept_exchange[itrt,itrt-1]
    accept_exchange_nn[itrt] =  (a_ex_yes[itrt,itrt+1]+a_ex_yes[itrt,itrt-1])/(a_ex_yes[itrt,itrt+1]+a_ex_no[itrt,itrt+1]+a_ex_yes[itrt,itrt-1]+a_ex_no[itrt,itrt-1])


plt.plot(T_ladder.betas[:n_chain-1],accept_exchange_nn_right[:n_chain-1])
plt.plot(T_ladder.betas[1:],accept_exchange_nn_left[1:])
plt.plot(T_ladder.betas,accept_exchange_nn)
plt.show()


plt.plot(T_ladder.betas,accept_exchange_nn)
plt.plot(T_ladder.betas,accept_exchange_nn_old)
plt.show()

cycle_count = np.array([np.sum((mcc.tracker_manager.cycle_archive[itrb][2]+mcc.tracker_manager.cycle_archive[itrb][3])/2) for itrb in range(0,len(mcc.tracker_manager.itrn_archive))])
new_cycles = np.hstack([cycle_count[0],np.diff(cycle_count)])
new_iterations = np.hstack([mcc.tracker_manager.itrn_archive[0],np.diff(np.array(mcc.tracker_manager.itrn_archive))])
cycle_rate = new_cycles/new_iterations
iterations_postburn = (mcc.tracker_manager.itrn_archive[-1]-mcc.tracker_manager.itrn_archive[cycle_burn_index])
mean_rate = (cycle_count[-1]-cycle_count[cycle_burn_index])/iterations_postburn
plt.plot(np.array(mcc.tracker_manager.itrn_archive[cycle_burn_index:]),cycle_rate[cycle_burn_index:])
plt.plot(np.array(mcc.tracker_manager.itrn_archive[cycle_burn_index:]),np.full(np.array(mcc.tracker_manager.itrn_archive[cycle_burn_index:]).size,mean_rate),'k--')
plt.ylim(0.,0.001)
plt.show()

cycle_record = mcc.tracker_manager.cycle_archive[-1][2:]-mcc.tracker_manager.cycle_archive[cycle_burn_index][2:]
cycle_rate = np.vstack([cycle_record,np.sum(cycle_record,axis=0)/(2*iterations_postburn)])


import sys
sys.exit()




cov_means = np.cov(mcc.logLs_store.T)
var_means = np.diag(cov_means)
corr_means = cov_means/np.sqrt(var_means)
corr_means = (corr_means.T/np.sqrt(var_means)).T

plt.plot(corr_means[argT_1])
plt.show()


plt.imshow(corr_means-np.eye(n_chain))
plt.show()

plt.plot(np.sum(cov_means,axis=0)/var_means)
plt.show()
