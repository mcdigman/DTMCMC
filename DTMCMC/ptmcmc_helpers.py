"""C 2023 Matthew C. Digman
Module with the overall PTMCMC Chain object"""
import numpy as np
from numba import njit

from DTMCMC.proposal_manager_helper import get_default_proposal_manager

from DTMCMC.tracker_manager import TrackerManager


class PTMCMCChain():
    """object to manage the overall chain evolution"""

    def __init__(self, T_ladder_in, like_obj, strategy_params, block_size, store_size,
                 tracker_manager=None, proposal_manager=None, starting_samples=None,
                 store_thin=1, n_record=-1):
        """create the chain object
        inputs:
            block_size: scalar integer, the number of MCMC iterations to do per block
            store_size: scalar integer, the number of MCMC states to store
            like_obj: a subclass of the abstract DTMCMC.Likelihood object, that gets likelihoods for a given set of parameters
            T_ladder_in: a DTMCMC.temperature_helpers.TemperatureLadder object (or suitable replacement)
            strategy_params: a DTMCMC.strategy_helpers.StrategyParams object (or suitable replacement)
            tracker_manager: a DTMCMC.tracker_manager.TrackerManager object (or suitable replacement)
            proposal_manager: a DTCMCM.proposal_manager.ProposalManager object
            starting_samples: a (n_chain, n_par) float array of starting samples
            store_thin: scalar integer, how much to thin the stored samples by (default 1)
            n_record: scalar integer, how many chains to store the results of (default n_cold)"""
        self.block_size = block_size
        self.n_par = like_obj.n_par
        self.store_size = store_size
        self.store_thin = store_thin
        self.store_idx = 0
        self.store_counter = 0
        self.like_obj = like_obj
        self.strategy_params = strategy_params
        self.tracker_manager = tracker_manager
        self.proposal_manager = proposal_manager
        self.starting_samples = starting_samples

        self.T_ladder = T_ladder_in

        self.betas = self.T_ladder.betas
        self.Ts = self.T_ladder.Ts
        self.n_chain = self.T_ladder.n_chain
        self.n_cold = self.T_ladder.n_cold

        # how many chains to save in the stored block, default is n_cold
        if n_record == -1:
            self.n_record = self.n_cold
        else:
            self.n_record = n_record

        self.instantiate_state()

        self.initialize_iterators()
        self.initialize_state()
        self.initialize_jumps()
        self.initialize_trackers()

    def initialize_trackers(self):
        """initialize the various trackers like acceptance rate and cycle times"""
        if self.tracker_manager is None:
            track_full_exchanges = self.proposal_manager.exchange_manager.track_full_exchanges
            self.tracker_manager = TrackerManager(self.n_cold, self.n_chain, self.block_size, self.n_par, track_full_exchanges, self.proposal_manager.n_jump_types)
        self.logL_means = []
        self.logL_vars = []

    def initialize_iterators(self):
        """initialize needed iterators"""
        self.itrn = 0

    def instantiate_state(self):
        """instantiate the state of the sampler"""
        self.logLs = np.zeros((self.block_size+1, self.n_chain))
        self.samples = np.zeros((self.block_size+1, self.n_chain, self.n_par))
        self.chain_track = np.zeros((self.block_size+1, self.n_chain), dtype=np.int64)
        self.chain_track[0] = np.arange(0, self.n_chain)
        self.samples_store = np.zeros((self.store_size+1, self.n_record, self.n_par))
        self.logLs_store = np.zeros((self.store_size+1, self.n_record))

    def initialize_jumps(self):
        """anything that needs to be done to initialize the various jumps"""
        if self.proposal_manager is None:
            self.proposal_manager = get_default_proposal_manager(self.T_ladder, self.like_obj, self.strategy_params, self.samples[0, :, :])

    def initialize_state(self):
        """initialize the samples"""
        for itrt in range(self.n_chain):
            if self.starting_samples is None:
                self.samples[0, itrt, :] = self.like_obj.prior_draw()
            else:
                self.samples[0, itrt, :] = self.starting_samples[itrt]
            self.logLs[0, itrt] = self.like_obj.get_loglike(self.samples[0, itrt, :])

        # initialize the storage with just the first element
        self.store_idx, self.store_counter = store_sample_helper(self.samples_store, self.logLs_store, self.samples, self.logLs,
                                                                 self.store_idx, self.store_counter, self.n_record, 1, self.store_thin, 0)

    def get_stored_flattened(self, n_burnin, n_chain_out=-1, thin=1):
        """get the stored samples flattened, with additional thinning if desired and only the first n_chain_out chains"""
        if n_chain_out == -1:
            n_chain_out = self.n_record

        n_burnin_thin = n_burnin//self.store_thin

        flat_shape = ((self.samples_store.shape[0]-n_burnin_thin-1)//thin+1)*n_chain_out
        samples_flattened = self.samples_store[n_burnin_thin::thin, :n_chain_out, :].reshape(flat_shape, self.n_par)
        logLs_flattened = self.logLs_store[n_burnin_thin::thin, :n_chain_out].reshape(flat_shape)
        return samples_flattened, logLs_flattened

    def store_samples(self):
        """store the samples from the current block in the memory block"""
        self.store_idx, self.store_counter = store_sample_helper(self.samples_store, self.logLs_store, self.samples, self.logLs,
                                                                 self.store_idx, self.store_counter, self.n_record, self.block_size, self.store_thin, 1)

    def reset_block(self):
        """blank all but the first sample"""
        self.samples[1:, :, :] = 0.
        self.logLs[1:, :] = 0.
        self.chain_track[1:, :] = 0

    def loop_block(self):
        """loop the final values of the previous block back to the next block's starting parameters"""
        self.samples[0, :, :] = self.samples[self.block_size, :, :]
        self.logLs[0, :] = self.logLs[self.block_size, :]
        self.chain_track[0, :] = self.chain_track[self.block_size, :]

    def block_start(self):
        """things to execute before the main body of the block to prepare for the mcmc step"""
        self.reset_block()

    def block_main(self):
        """the main body of the block with the mcmc step"""
        advance_block_ptmcmc(self.T_ladder, self.logLs, self.samples, self.chain_track, self.proposal_manager, self.like_obj, self.tracker_manager)

    def block_end(self):
        """things to execute after the main mcmc body of the block, like clean up recalculating fisher matrices, and storing results
        as well as perhaps non-legal burn in steps"""
        self.store_samples()
        self.proposal_manager.post_block_update(self.itrn, self.block_size, self.samples, self.logLs)
        self.tracker_manager.process_chain_cycles(self.itrn, self.chain_track)
        # track the block mean and std of the likelihoods by chain
        self.logL_means.append(self.logLs[1:].mean(axis=0))
        self.logL_vars.append(self.logLs[1:].var(axis=0))
        self.loop_block()

    def block_advance_iterators(self):
        """iterators to be advanced at the end of every block"""
        self.itrn += self.block_size

    def advance_block(self):
        """advance the state of the mcmc chain 1 full block"""
        self.block_start()
        self.block_main()
        self.block_end()
        self.block_advance_iterators()

    def preblock_operations(self):
        """any operations to be done before each block even starts, like resetting acceptance rate trackers at the end of burn in"""
        return

    def postblock_operations(self):
        """any operations to be done after the block finishes completely, perhaps printing acceptances"""
        return

    def pre_Nblock_setup(self):
        """any operations to be done before advance_N_blocks starts, maybe rearranging file outputs"""
        return

    def post_Nblock_teardown(self):
        """any operations to be done after advance_N_blocks ends, maybe finishing file outputs"""
        self.tracker_manager.print_tracker_summary(self.n_cold, self.Ts, self.proposal_manager)

    def advance_N_blocks(self, Nblocks):
        """advance the current state of the chain forward Nblocks blocks"""
        self.pre_Nblock_setup()
        for itrk in range(Nblocks):
            self.preblock_operations()
            self.advance_block()
            self.postblock_operations()

        self.post_Nblock_teardown()


@njit()
def store_sample_helper(samples_store, logLs_store, samples_block, logLs_block, store_idx_in, store_counter_in, n_record, block_size, store_thin, read_offset):
    """write the samples from n_record chains to be stored using store_thin thinning,
    store_idx and store_counter are counters for the index to write into and the thinning respectively
    read offset needs to be zero for first write and 1 otherwise to prevent duplicate writes due to wrapping"""
    store_idx = store_idx_in
    store_counter = store_counter_in
    for itrk in range(read_offset, block_size+read_offset):
        if store_counter == 0:
            # write the sample if the thinning counter is 0
            samples_store[store_idx, :n_record, :] = samples_block[itrk, :n_record, :]
            logLs_store[store_idx, :n_record] = logLs_block[itrk, :n_record]
            store_idx += 1
        store_counter += 1

        if store_counter >= store_thin:
            # wrap the thinning counter
            store_counter = 0
        if store_idx >= samples_store.shape[0]:
            # wrap the writing counter
            store_idx = 0

    return store_idx, store_counter


def advance_block_ptmcmc(T_ladder, logLs, samples, chain_track, proposal_manager, like_obj, tracker_manager):
    """advance an entire block in the ptmcmc chain, alternating regular and exchange proposals"""
    block_size = samples.shape[0]-1

    for itrb in range(1, block_size+1):
        if proposal_manager.exchange_manager.is_exchange_step(itrb):
            # if the index requests an exchange, do that
            proposal_manager.exchange_manager.do_ptmcmc_exchange(itrb, samples, logLs, T_ladder, tracker_manager.exchange_tracker, chain_track)
        else:
            # if the index is a normal jump
            advance_step_ptmcmc(itrb, samples, logLs, T_ladder, tracker_manager.accept_record, proposal_manager, like_obj)
            chain_track[itrb, :] = chain_track[itrb-1, :]         # track the indexes of the chains, which only change on exchange steps

        proposal_manager.post_step_update(samples[itrb])        # record differential evolution buffer

    return samples


def advance_step_ptmcmc(itrb, samples, logLs, T_ladder, accept_record, proposal_manager, like_obj):
    """advance a single step step in the ptmcmc chain"""
    n_chain = T_ladder.n_chain
    betas = T_ladder.betas

    for itrt in range(0, n_chain):
        new_point, density_fac, idx_jump, success = proposal_manager.dispatch_jump(samples[itrb-1, itrt], itrt)

        if success:
            # skip likelihood evaluation if proposal is marked as a failure
            new_point = like_obj.correct_bounds(new_point)   # make sure the point is legal if possible
            success = like_obj.check_bounds(new_point)       # check that the point was correctly made legal

        if success:
            # if the point passes, get the likelihood
            logL_new = like_obj.get_loglike(new_point)       # get the likelihood
        else:
            logL_new = -np.inf                               # if the point failed, just set the likelihood to negative infinity so it won't be accepted

        test = np.log(np.random.uniform(0., 1.))             # get the test draw to determine if we accept the point
        if betas[itrt]*(logL_new-logLs[itrb-1, itrt])+density_fac > test:
            # the draw was accepted, assign its parameters
            samples[itrb, itrt] = new_point
            logLs[itrb, itrt] = logL_new
            accept_record[0, itrt, idx_jump] += 1
        else:
            # the draw was rejected, assign the old parameters
            samples[itrb, itrt] = samples[itrb-1, itrt]
            logLs[itrb, itrt] = logLs[itrb-1, itrt]
            accept_record[1, itrt, idx_jump] += 1
