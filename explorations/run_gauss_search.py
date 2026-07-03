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

import diagnostic_commentary_helpers as dch
import DTMCMC.exchange_manager as eh

# import likelihood_gb as trial_likelihood
import DTMCMC.likelihoods.normal_nd as trial_likelihood
from DTMCMC.corr_summary_helpers import CorrelationSummary
from DTMCMC.dtmcmc_sampler import DTMCMCSampler
from DTMCMC.proposal_manager_helper import get_default_proposal_manager
from DTMCMC.temperature_ladder_helpers import GeometricTemperatureLadder

# TODO reduce exposure of block_size parameter

if __name__ == '__main__':
    t0 = perf_counter()

    # starting variables
    n_chain = 8                       # number of total chains for parallel tempering
    n_cold = 1                         # number of T=1 chains for parallel tempering
    n_burnin = 5000                    # number of iterations to discard as burn in
    block_size = 1000                  # number of iterations per block when advancing the chain state
    store_size = 100000                # number of samples to store total
    N_blocks = store_size // block_size  # number of blocks the sampler must iterate through
    n_par = 5

    T_max = 100.                       # maximum temperature for geometric part of temperature ladder

    params_true = np.zeros(n_par)      # true parameters for search

    # create needed objects
    T_ladder = GeometricTemperatureLadder(n_chain, n_cold=n_cold, T_max=T_max)  # get the temperature ladder object

    like_obj = trial_likelihood.GaussianLikelihood(n_par)
    params_true = like_obj.correct_bounds(params_true)                 # make sure the conventions on the parameters match

    # create the starting samples
    starting_samples = np.zeros((T_ladder.n_chain, like_obj.n_par))
    for itrt in range(n_chain):
        # start from prior draws
        starting_samples[itrt] = like_obj.prior_draw()

    # create the overarching proposal manager object
    exchange_manager = eh.ExchangeManager(strategy=eh.RANDOM_TARGETS, track_full_exchanges=True)
    proposal_manager = get_default_proposal_manager(T_ladder, like_obj, starting_samples, exchange_manager_loc=exchange_manager)

    print('Chain parameters', n_cold, n_chain, n_burnin, block_size, store_size, T_max)

    # create the chain object
    mcc = DTMCMCSampler(T_ladder, like_obj, block_size, store_size, starting_samples=starting_samples, n_record=n_chain, proposal_manager=proposal_manager)

    t_init_end = perf_counter()
    print('all objects initialized in ', t_init_end - t0, 's')

    t_advance_begin = perf_counter()

    # the main loop which actually advances the MCMC state
    mcc.advance_N_blocks(N_blocks)

    t_advance_end = perf_counter()
    print('advanced in ', t_advance_end - t_advance_begin, 's')

    # generate some summary information
    corr_sum = CorrelationSummary()
    corr_sum.summarize_blocks(mcc, n_burnin)
    corr_sum.final_prints(mcc, n_burnin)

    # get flattened samples for plotting
    samples_flattened, logLs_flattened = mcc.get_stored_flattened(corr_sum.restrict_n_burnin(mcc, n_burnin), n_chain_out=n_cold)

    tf = perf_counter()

    print('full search time ', str(tf - t0) + 's')

    do_sigma_plot = True
    if do_sigma_plot:
        # samples_got = mcc.samples_store[n_burnin:]
        # samples_post = trial_likelihood.drawposterior(store_size-n_burnin,mcc.Ts,n_par,like_obj.cutoff)
        # sigma_got = np.std(samples_got,axis=(0,2))
        # sigma_post = np.std(samples_post,axis=(0,2))
        # import matplotlib.pyplot as plt
        # plt.plot(mcc.Ts,sigma_got)
        # plt.plot(mcc.Ts,sigma_post)
        # plt.plot(mcc.Ts,np.sqrt(mcc.Ts))
        # plt.show()

        # plt.semilogx(mcc.Ts,1-sigma_got/sigma_post)

        # plt.show()

        # print(unit_normal_battery(np.reshape(samples_post[:,0],samples_post[:,0].size),do_assert=False))
        # print(unit_normal_battery(np.reshape(samples_got[:,0],samples_got[:,0].size),do_assert=False))
        dch.print_diagnostic_commentary(mcc)

    do_corner_plot = True
    if do_corner_plot:
        # generate a corner plot
        import corner
        import matplotlib.pyplot as plt

        # reformat the samples to make the plots look nicer
        labels = like_obj.get_labels()
        samples_format, params_true_format = like_obj.format_samples_output(mcc.samples_store[:, 0, :].copy(), params_true)

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
