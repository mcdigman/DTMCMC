"""C 2023 Matthew C. Digman
Module with the overall PTMCMC Chain object
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from DTMCMC.proposal_manager_helper import ProposalManager, get_default_proposal_manager
from DTMCMC.tracker_manager import TrackerManager

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder


@njit()
def store_sample_helper(
    samples_store,
    logLs_store,
    samples_block,
    logLs_block,
    store_idx_in: int,
    store_counter_in: int,
    n_record: int,
    block_size: int,
    store_thin: int,
    read_offset: int) -> tuple[int, int]:
    """Write the samples from n_record chains to be stored using store_thin thinning,
    store_idx and store_counter are counters for the index to write into and
    the thinning respectively read offset needs to be zero for first write
    and 1 otherwise to prevent duplicate writes due to wrapping
    """
    store_idx: int = store_idx_in
    store_counter: int = store_counter_in
    for itrk in range(read_offset, block_size + read_offset):
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


@njit()
def mcmc_decision_helper(itrb: int, samples, logLs, betas, accept_record, itrt: int, new_point, logL_new, density_fac, idx_jump: int) -> None:
    """Helper to decide whether mcmc point is accepted or not and process accordingly"""
    # draw to determine if we will accept
    test: float = np.log(np.random.uniform(0., 1.))

    # process acceptance or rejection
    if betas[itrt] * (logL_new - logLs[itrb - 1, itrt]) + density_fac > test:
        # the draw was accepted, assign its parameters
        samples[itrb, itrt] = new_point
        logLs[itrb, itrt] = logL_new
        accept_record[0, itrt, idx_jump] += 1
    else:
        # the draw was rejected, assign the old parameters
        samples[itrb, itrt] = samples[itrb - 1, itrt]
        logLs[itrb, itrt] = logLs[itrb - 1, itrt]
        accept_record[1, itrt, idx_jump] += 1


def advance_step_ptmcmc(itrb: int, samples, logLs, T_ladder: TemperatureLadder, accept_record, proposal_manager: ProposalManager, like_obj: AbstractLikelihood) -> None:
    """Advance a single step step in the ptmcmc chain"""
    n_chain: int = T_ladder.n_chain
    betas: NDArray[np.floating] = T_ladder.betas

    for itrt in range(n_chain):
        new_point, density_fac, success, idx_jump = proposal_manager.dispatch_jump(samples[itrb - 1, itrt], itrt)

        if success:
            # see if the point is in bounds, if not try to make it legal
            check_success: bool = like_obj.check_bounds(new_point)
            if not check_success:
                # try to make the point in bounds and fail if unsuccesful
                new_point = like_obj.correct_bounds(new_point)
                success = like_obj.check_bounds(new_point)

        # skip likelihood evaluation if proposal is marked as a failure
        if success:
            # if the point passes, get the likelihood
            logL_new: float = like_obj.get_loglike(new_point)
        else:
            # Failed, ensure the point will not be accepted
            logL_new = -np.inf

        mcmc_decision_helper(itrb, samples, logLs, betas, accept_record, itrt, new_point, logL_new, density_fac, idx_jump)


def advance_block_ptmcmc(
        T_ladder: TemperatureLadder, logLs, samples, chain_track, proposal_manager: ProposalManager, like_obj: AbstractLikelihood, tracker_manager: TrackerManager
):
    """Advance an entire block in the ptmcmc chain, alternating regular and exchange proposals"""
    block_size: int = samples.shape[0] - 1

    for itrb in range(1, block_size + 1):
        if proposal_manager.exchange_manager.is_exchange_step(itrb):
            # if the index requests an exchange, do that
            proposal_manager.exchange_manager.do_ptmcmc_exchange(
                itrb,
                samples,
                logLs,
                T_ladder,
                tracker_manager.exchange_tracker,
                chain_track,
            )
        else:
            # if the index is a normal jump
            advance_step_ptmcmc(
                itrb,
                samples,
                logLs,
                T_ladder,
                tracker_manager.accept_record,
                proposal_manager,
                like_obj,
            )
            # track the indexes of the chains, which only change on exchange steps
            chain_track[itrb, :] = chain_track[itrb - 1, :]

        # record the differential evolution buffer
        proposal_manager.post_step_update(samples[itrb])

    return samples

# TODO rename this module
# TODO add any necessary handlers for block length


class DTMCMCSampler:
    """object to manage the overall chain evolution"""

    def __init__(self, T_ladder_in: TemperatureLadder, like_obj: AbstractLikelihood, block_size, store_size,
                 tracker_manager: TrackerManager | None = None, proposal_manager: ProposalManager | None = None, starting_samples=None,
                 store_thin: int = 1, n_record: int = -1) -> None:
        """Create the chain object

        Parameters
        ----------
        block_size: int
            the number of MCMC iterations to do per block
        store_size: int
            the number of MCMC states to store
        like_obj: DTMCMC.Likelihood
            Object that gets likelihoods for a given set of parameters
        T_ladder_in: TemperatureLadder
        tracker_manager: TrackerManager
        proposal_manager: ProposalManager
        starting_samples: a (n_chain, n_par) float array of starting samples
        store_thin: scalar integer, how much to thin the stored samples by (default 1)
        n_record: scalar integer, how many chains to store the results of (default n_cold)
        """
        self.block_size: int = block_size
        self.n_par: int = like_obj.n_par
        self.store_size: int = store_size
        self.store_thin: int = store_thin
        self.store_idx: int = 0
        self.store_counter: int = 0
        self.itrn: int = 0
        self.like_obj: AbstractLikelihood = like_obj
        self.tracker_manager: TrackerManager
        self.proposal_manager: ProposalManager
        self.starting_samples = starting_samples

        self.T_ladder: TemperatureLadder = T_ladder_in

        self.betas = self.T_ladder.betas
        self.Ts = self.T_ladder.Ts
        self.n_chain: int = self.T_ladder.n_chain
        self.n_cold: int = self.T_ladder.n_cold

        # how many chains to save in the stored block, default is n_cold
        if n_record == -1:
            self.n_record = self.n_cold
        else:
            self.n_record = n_record

        self.instantiate_state()

        self.initialize_iterators()
        self.initialize_state()
        self.initialize_jumps(proposal_manager)
        self.initialize_trackers(tracker_manager)

    def initialize_trackers(self, tracker_manager_in: TrackerManager | None = None) -> None:
        """Initialize the various trackers like acceptance rate and cycle times"""
        if tracker_manager_in is None:
            track_full_exchanges = self.proposal_manager.exchange_manager.track_full_exchanges
            self.tracker_manager = TrackerManager(
                self.n_cold,
                self.n_chain,
                self.block_size,
                self.n_par,
                track_full_exchanges,
                self.proposal_manager.n_jump_types,
                max(self.store_size // self.block_size, 1),
            )
        else:
            self.tracker_manager = tracker_manager_in
        self.logL_means: list[NDArray[np.floating]] = []
        self.logL2_means: list[NDArray[np.floating]] = []
        self.logL3_means: list[NDArray[np.floating]] = []
        self.logL4_means: list[NDArray[np.floating]] = []
        self.logL5_means: list[NDArray[np.floating]] = []
        self.logL6_means: list[NDArray[np.floating]] = []
        self.logL_prod11_means: list[NDArray[np.floating]] = []
        self.logL_prod21_means: list[NDArray[np.floating]] = []
        self.logL_prod12_means: list[NDArray[np.floating]] = []
        self.logL_vars: list[NDArray[np.floating]] = []

    def initialize_iterators(self) -> None:
        """Initialize needed iterators"""
        self.itrn = 0

    def instantiate_state(self) -> None:
        """Instantiate the state of the sampler"""
        self.logLs = np.zeros((self.block_size + 1, self.n_chain))
        self.samples = np.zeros((self.block_size + 1, self.n_chain, self.n_par))
        self.chain_track = np.zeros((self.block_size + 1, self.n_chain), dtype=np.int64)
        self.chain_track[0] = np.arange(0, self.n_chain)
        # TODO fix non-required plus one
        self.samples_store = np.zeros((self.store_size, self.n_record, self.n_par))
        self.logLs_store = np.zeros((self.store_size, self.n_record))

    def initialize_jumps(self, proposal_manager_in: ProposalManager | None = None) -> None:
        """Anything that needs to be done to initialize the various jumps"""
        if proposal_manager_in is None:
            self.proposal_manager = get_default_proposal_manager(self.T_ladder, self.like_obj, self.samples[0, :, :])
        else:
            self.proposal_manager = proposal_manager_in

    def initialize_state(self) -> None:
        """Initialize the samples"""
        if self.starting_samples is None:
            self.starting_samples = np.zeros((self.n_chain, self.n_par))
            for itrt in range(self.n_chain):
                self.starting_samples[itrt, :] = self.like_obj.prior_draw()

        self.starting_logLs = np.zeros(self.n_chain)
        for itrt in range(self.n_chain):
            self.starting_logLs[itrt] = self.like_obj.get_loglike(self.starting_samples[itrt, :])

        for itrt in range(self.n_chain):
            self.samples[0, itrt, :] = self.starting_samples[itrt, :]
            self.logLs[0, itrt] = self.starting_logLs[itrt]

        # initialize the storage with just the first element
        # TODO initialize storage without breaking first block
        self.store_idx = 0
        self.store_counter = 0
        # self.store_idx, self.store_counter = store_sample_helper(
        #    self.samples_store,
        #    self.logLs_store,
        #    self.samples,
        #    self.logLs,
        #    self.store_idx,
        #    self.store_counter,
        #    self.n_record,
        #    1,
        #    self.store_thin,
        #    0,
        # )

    def get_stored_flattened(self, n_burnin, n_chain_out=-1, thin=1):
        """Get the stored samples flattened, with additional thinning if desired and only the first n_chain_out chains"""
        if n_chain_out == -1:
            n_chain_out = self.n_record

        n_burnin_thin = n_burnin // self.store_thin

        flat_shape = ((self.samples_store.shape[0] - n_burnin_thin - 1) // thin + 1) * n_chain_out
        samples_flattened = self.samples_store[n_burnin_thin::thin, :n_chain_out, :].reshape(flat_shape, self.n_par)
        logLs_flattened = self.logLs_store[n_burnin_thin::thin, :n_chain_out].reshape(flat_shape)
        return samples_flattened, logLs_flattened

    def store_samples(self) -> None:
        """Store the samples from the current block in the memory block"""
        # make sure the very first value gets written to storage correctly
        self.store_idx, self.store_counter = store_sample_helper(
            self.samples_store,
            self.logLs_store,
            self.samples,
            self.logLs,
            self.store_idx,
            self.store_counter,
            self.n_record,
            self.block_size,
            self.store_thin,
            1,
        )

    def reset_block(self) -> None:
        """Blank all but the first sample"""
        self.samples[1:, :, :] = 0.
        self.logLs[1:, :] = 0.
        self.chain_track[1:, :] = 0

    def loop_block(self) -> None:
        """Loop the final values of the previous block back to
        the next block's starting parameters
        """
        self.samples[0, :, :] = self.samples[self.block_size, :, :]
        self.logLs[0, :] = self.logLs[self.block_size, :]
        self.chain_track[0, :] = self.chain_track[self.block_size, :]

    def block_start(self) -> None:
        """Things to execute before the main body of the block to prepare for the mcmc step"""
        self.reset_block()

    def block_main(self) -> None:
        """The main body of the block with the mcmc step"""
        advance_block_ptmcmc(
            self.T_ladder,
            self.logLs,
            self.samples,
            self.chain_track,
            self.proposal_manager,
            self.like_obj,
            self.tracker_manager,
        )

    def block_end(self) -> None:
        """Things to execute after the main mcmc body of the block,
        like clean up recalculating fisher matrices, and storing results
        as well as perhaps non-legal burn in steps
        """
        self.store_samples()
        self.proposal_manager.post_block_update(self.itrn, self.block_size, self.samples, self.logLs)
        self.tracker_manager.post_block_update(self.itrn, self.chain_track)
        # track the block mean and std of the likelihoods by chain
        self.logL_means.append(self.logLs[1:].mean(axis=0))
        self.logL_vars.append(self.logLs[1:].var(axis=0))
        # also track some higher powers of the likelihood distribution
        # storing them directly as powers allows moments to be calculated later
        # averaging over an arbitrarily long window in a stable way

        self.logL2_means.append((self.logLs[1:]**2).mean(axis=0))
        self.logL3_means.append((self.logLs[1:]**3).mean(axis=0))
        self.logL4_means.append((self.logLs[1:]**4).mean(axis=0))
        self.logL5_means.append((self.logLs[1:]**5).mean(axis=0))
        self.logL6_means.append((self.logLs[1:]**6).mean(axis=0))
        self.logL_prod11_means.append((self.logLs[1:, :self.n_chain - 1] * self.logLs[1:, 1:self.n_chain]).mean(axis=0))
        self.logL_prod21_means.append((self.logLs[1:, :self.n_chain - 1]**2 * self.logLs[1:, 1:self.n_chain]).mean(axis=0))
        self.logL_prod12_means.append((self.logLs[1:, :self.n_chain - 1] * self.logLs[1:, 1:self.n_chain]**2).mean(axis=0))
        self.loop_block()

    def block_advance_iterators(self) -> None:
        """Iterators to be advanced at the end of every block"""
        self.itrn += self.block_size

    def advance_block(self) -> None:
        """Advance the state of the mcmc chain 1 full block"""
        self.block_start()
        self.block_main()
        self.block_end()
        self.block_advance_iterators()

    def preblock_operations(self) -> None:
        """Any operations to be done before each block even starts, like resetting acceptance rate trackers at the end of burn in"""
        return

    def postblock_operations(self) -> None:
        """Any operations to be done after the block finishes completely, perhaps printing acceptances"""
        return

    def pre_Nblock_setup(self) -> None:
        """Any operations to be done before advance_N_blocks starts, maybe rearranging file outputs"""
        return

    def post_Nblock_teardown(self) -> None:
        """Any operations to be done after advance_N_blocks ends, maybe finishing file outputs"""
        self.tracker_manager.print_tracker_summary(self.n_cold, self.Ts, self.proposal_manager)

    def advance_N_blocks(self, Nblocks: int) -> None:
        """Advance the current state of the chain forward Nblocks blocks"""
        self.pre_Nblock_setup()
        for _itrk in range(Nblocks):
            self.preblock_operations()
            self.advance_block()
            self.postblock_operations()

        self.post_Nblock_teardown()
