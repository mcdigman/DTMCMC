"""C 2023 Matthew C. Digman
Module with the overall PTMCMC Chain object
"""

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from DTMCMC.de_manager import DEJumpManager
from DTMCMC.fisher_manager import FisherJumpManager, set_scales
from DTMCMC.proposal_manager_helper import ProposalManager, get_default_proposal_manager
from DTMCMC.temperature_ladder_helpers import remap_ladder_indices
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
    record_indices,
    block_size: int,
    store_thin: int,
    read_offset: int) -> tuple[int, int]:
    """Write the samples of the record_indices chains to be stored using store_thin
    thinning: store column j holds chain record_indices[j] (duplicates permitted).
    store_idx and store_counter are counters for the index to write into and
    the thinning respectively read offset needs to be zero for first write
    and 1 otherwise to prevent duplicate writes due to wrapping
    """
    store_idx: int = store_idx_in
    store_counter: int = store_counter_in
    for itrk in range(read_offset, block_size + read_offset):
        if store_counter == 0:
            # write the sample if the thinning counter is 0
            for itrr in range(record_indices.size):
                itrt = record_indices[itrr]
                samples_store[store_idx, itrr, :] = samples_block[itrk, itrt, :]
                logLs_store[store_idx, itrr] = logLs_block[itrk, itrt]
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
def mcmc_decision_helper(itrb: int, samples, logLs, betas, accept_record, esd_record, itrt: int, new_point, logL_new, density_fac, idx_jump: int) -> None:
    """Helper to decide whether mcmc point is accepted or not and process accordingly"""
    # draw to determine if we will accept
    test: float = np.log(np.random.uniform(0., 1.))

    # squared displacement of the proposal, accumulated per (T, jump type)
    # for the expected-squared-displacement tracker; pure observer, no draws
    delta_sq: float = 0.
    for itrp in range(new_point.size):
        diff: float = new_point[itrp] - samples[itrb - 1, itrt, itrp]
        delta_sq += diff * diff
    esd_record[0, itrt, idx_jump] += delta_sq

    # process acceptance or rejection
    if betas[itrt] * (logL_new - logLs[itrb - 1, itrt]) + density_fac > test:
        # the draw was accepted, assign its parameters
        samples[itrb, itrt] = new_point
        logLs[itrb, itrt] = logL_new
        accept_record[0, itrt, idx_jump] += 1
        esd_record[1, itrt, idx_jump] += delta_sq
    else:
        # the draw was rejected, assign the old parameters
        samples[itrb, itrt] = samples[itrb - 1, itrt]
        logLs[itrb, itrt] = logLs[itrb - 1, itrt]
        accept_record[1, itrt, idx_jump] += 1


def advance_step_ptmcmc(itrb: int, samples, logLs, T_ladder: TemperatureLadder, accept_record, esd_record, proposal_manager: ProposalManager, like_obj: AbstractLikelihood) -> None:
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

        mcmc_decision_helper(itrb, samples, logLs, betas, accept_record, esd_record, itrt, new_point, logL_new, density_fac, idx_jump)


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
                tracker_manager.esd_exchange,
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
                tracker_manager.esd_record,
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
                 store_thin: int = 1, arg_record=None) -> None:
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
        arg_record: optional sequence of chain indices to record in storage in
            addition to the ladder's n_cold readout chains (default none).
            The readout chains always occupy the first n_cold store columns
            and their indices are recomputed at every ladder update; the
            arg_record columns follow in the given order, and an index that
            duplicates a readout chain is simply stored twice
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

        # recorded chains: the ladder's readout (cold) chains first — their
        # indices are recomputed at every ladder update — then the extras
        if arg_record is None:
            self.arg_record: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        else:
            self.arg_record = np.asarray(arg_record, dtype=np.int64)
        assert np.all(self.arg_record >= 0)
        assert np.all(self.arg_record < self.n_chain)
        self.record_indices: NDArray[np.int64] = np.concatenate([self.T_ladder.get_arg_cold(), self.arg_record])
        # (itrn, record_indices) pairs recording which chain each store
        # column held from which iteration on; ladder updates that move the
        # readout chains append a new entry
        self.record_history: list[tuple[int, NDArray[np.int64]]] = [(0, self.record_indices.copy())]

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
        self.samples_store = np.zeros((self.store_size, self.record_indices.size, self.n_par))
        self.logLs_store = np.zeros((self.store_size, self.record_indices.size))

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
        #    self.record_indices,
        #    1,
        #    self.store_thin,
        #    0,
        # )

    def get_stored_flattened(self, n_burnin, n_chain_out=-1, thin=1):
        """Get the stored samples flattened, with additional thinning if desired and only the first n_chain_out store columns.

        The first n_cold columns are always the ladder's readout chains, so
        n_chain_out=n_cold selects exactly the readout posterior samples.
        """
        if n_chain_out == -1:
            n_chain_out = self.record_indices.size

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
            self.record_indices,
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

    def apply_ladder_update(self, new_ladder: TemperatureLadder, remap_rule: str = 'at_or_hotter') -> None:
        """Swap the temperature ladder at a block boundary.

        RNG-neutral: pure deterministic state remapping, no draws. The
        hook rebinds the ladder, the betas/Ts aliases, and every jump
        manager's ladder reference. Chain states remap by temperature
        rank; for equal-size sorted ladders this is the identity, so no
        walker state is cloned or discarded. logLs carry over because
        they are T-independent. DE-buffer columns remap per remap_rule;
        any rung whose column was resourced then gets its current state
        written at the buffer's most recently written row. Fisher scales
        refresh from the existing diagonals, and trackers are segmented
        with a round-trip event-log boundary. Walker identities restart
        with the new segment, matching the cycle-tracker reset.
        """
        assert new_ladder.n_chain == self.n_chain
        assert new_ladder.n_cold == self.n_cold
        # the rank remap relies on the sorted-ladder convention (D6);
        # clip infs so duplicate hot anchors do not produce inf - inf
        assert np.all(np.diff(np.minimum(self.Ts, 1.e300)) >= 0.)
        assert np.all(np.diff(np.minimum(new_ladder.Ts, 1.e300)) >= 0.)

        self.tracker_manager.segment_for_ladder_update(self.itrn)

        # state remap by temperature rank: identity for equal-size sorted
        # ladders, so samples/logLs stay in their slots; only the walker
        # labels restart, consistent with the cycle-tracker reset
        self.chain_track[0] = np.arange(0, self.n_chain)

        buffer_sources = remap_ladder_indices(self.Ts, new_ladder.Ts, remap_rule)
        resourced = np.flatnonzero(buffer_sources != np.arange(self.n_chain))
        for manager in self.proposal_manager.managers:
            if isinstance(manager, DEJumpManager):
                manager.de_buffer[:, :, :] = manager.de_buffer[:, buffer_sources, :]
                # conditional, so an identical-ladder update stays bit-exact
                row_newest = (manager.itrde_write - 1) % manager.de_size
                for itrt in resourced:
                    manager.de_buffer[row_newest, itrt, :] = self.samples[0][itrt]

        self.T_ladder = new_ladder
        self.betas = new_ladder.betas
        self.Ts = new_ladder.Ts
        self.proposal_manager.T_ladder = new_ladder
        for manager in self.proposal_manager.managers:
            manager.T_ladder = new_ladder
            if isinstance(manager, FisherJumpManager):
                manager.sigma_scales, manager.gamma_mults = set_scales(self.n_par, new_ladder, manager.sigma_diags)

        # the readout chains may sit at different indices in the new ladder
        # (e.g. rungs added below T_cold): recompute the recorded set and log
        # the change so stored columns stay interpretable per iteration
        record_indices_new = np.concatenate([new_ladder.get_arg_cold(), self.arg_record])
        assert record_indices_new.size == self.record_indices.size
        if not np.array_equal(record_indices_new, self.record_indices):
            self.record_indices = record_indices_new
            self.record_history.append((self.itrn, record_indices_new.copy()))

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
