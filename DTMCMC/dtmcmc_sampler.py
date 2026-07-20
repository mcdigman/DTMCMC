"""C 2023 Matthew C. Digman
Module with the overall PTMCMC Chain object
"""

from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray

from DTMCMC.de_manager import DEJumpManager
from DTMCMC.eval_accounting import EvalAccounting
from DTMCMC.exchange_manager import AbstractExchangeManager
from DTMCMC.fisher_manager import FisherJumpManager, set_scales
from DTMCMC.jump_manager import AbstractJump, AbstractJumpManager
from DTMCMC.likelihood import AbstractLikelihood
from DTMCMC.mcmc_kernel_helpers import mcmc_decision_helper
from DTMCMC.numba_backend import NativeSerialBackend
from DTMCMC.proposal_manager import AbstractProposalManager
from DTMCMC.proposal_manager_helper import get_default_proposal_manager
from DTMCMC.temperature_ladder_helpers import remap_ladder_indices
from DTMCMC.tracker_manager import TrackerManager

if TYPE_CHECKING:
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder


@njit()
def store_sample_helper(
    samples_store: NDArray[np.floating],
    logLs_store: NDArray[np.floating],
    samples_block: NDArray[np.floating],
    logLs_block: NDArray[np.floating],
    store_idx_in: int,
    store_counter_in: int,
    record_indices: NDArray[np.int64],
    block_size: int,
    store_thin: int,
    read_offset: int,
) -> tuple[int, int]:
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
            for itrr, itrt in enumerate(record_indices):
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


def advance_step_ptmcmc[LikelihoodType: AbstractLikelihood](
    itrb: int,
    samples: NDArray[np.floating],
    logLs: NDArray[np.floating],
    T_ladder: TemperatureLadder,
    accept_record: NDArray[np.int64],
    esd_record: NDArray[np.floating],
    proposal_manager: AbstractProposalManager[LikelihoodType],
    like_obj: LikelihoodType,
    jump_internal_evals: NDArray[np.int64],
    zero_loglike: bool,
) -> tuple[int, int]:
    """Advance a single step in the ptmcmc chain.

    Returns the number of target likelihood evaluations performed and the
    declared jump-internal evaluations incurred (``jump_internal_evals``
    holds the per-jump declared cost in flattened jump order).
    """
    n_chain: int = T_ladder.n_chain
    betas: NDArray[np.floating] = T_ladder.betas
    n_target_evals = 0
    n_internal_evals = 0

    for itrt in range(n_chain):
        new_point, density_fac, success, idx_jump = proposal_manager.dispatch_jump(samples[itrb - 1, itrt], itrt)
        n_internal_evals += int(jump_internal_evals[idx_jump])

        if success:
            # see if the point is in bounds, if not try to make it legal
            new_point, success = like_obj.validate_bounds(new_point)

        if success:
            # The prior is not tempered. For a prior-draw proposal this target
            # factor cancels the reverse/forward proposal factor supplied by
            # PriorFullJump; other jumps retain the target-prior contribution.
            density_fac += like_obj.prior_factor(new_point) - like_obj.prior_factor(samples[itrb - 1, itrt])

        # skip likelihood evaluation if proposal is marked as a failure
        if success:
            # if the point passes, get the likelihood
            if zero_loglike:
                logL_new: float = 0.0
            else:
                logL_new = like_obj.get_loglike(new_point)
            n_target_evals += 1
        else:
            # Failed, ensure the point will not be accepted
            logL_new = -np.inf

        mcmc_decision_helper(
            itrb, samples, logLs, betas, accept_record, esd_record, itrt, new_point, logL_new, density_fac, idx_jump
        )

    return n_target_evals, n_internal_evals


def declared_jump_internal_evals[LikelihoodType: AbstractLikelihood](
    proposal_manager: AbstractProposalManager[LikelihoodType],
) -> tuple[NDArray[np.int64], bool]:
    """Collect the per-jump declared internal evaluation costs in flattened order.

    Returns the cost array and whether every jump declared one; a missing
    declaration contributes 0 to the array but marks the accounting
    incomplete — an unknown cost is never silently treated as zero.
    """
    declared = [getattr(jump, 'declared_internal_evals', None) for jump in proposal_manager.jumps]
    known = all(value is not None for value in declared)
    return np.array([0 if value is None else value for value in declared], dtype=np.int64), known


def advance_block_ptmcmc[LikelihoodType: AbstractLikelihood](
    T_ladder: TemperatureLadder,
    logLs: NDArray[np.floating],
    samples: NDArray[np.floating],
    chain_track: NDArray[np.int64],
    proposal_manager: AbstractProposalManager[LikelihoodType],
    like_obj: LikelihoodType,
    tracker_manager: TrackerManager[LikelihoodType],
    zero_loglike: bool,
) -> tuple[int, int, bool]:
    """Advance an entire block in the ptmcmc chain, alternating regular and exchange proposals.

    Returns the target evaluation count, the declared jump-internal
    evaluation count, and whether every jump declared its internal cost.
    """
    block_size: int = samples.shape[0] - 1
    jump_internal_evals, internal_known = declared_jump_internal_evals(proposal_manager)
    n_target_evals = 0
    n_internal_evals = 0

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
            step_targets, step_internal = advance_step_ptmcmc(
                itrb,
                samples,
                logLs,
                T_ladder,
                tracker_manager.accept_record,
                tracker_manager.esd_record,
                proposal_manager,
                like_obj,
                jump_internal_evals,
                zero_loglike,
            )
            n_target_evals += step_targets
            n_internal_evals += step_internal
            # track the indexes of the chains, which only change on exchange steps
            chain_track[itrb, :] = chain_track[itrb - 1, :]

        # record the differential evolution buffer
        proposal_manager.post_step_update(samples[itrb])

    return n_target_evals, n_internal_evals, internal_known


# TODO rename this module
# TODO add any necessary handlers for block length


class DTMCMCSampler[LikelihoodType: AbstractLikelihood]:
    """object to manage the overall chain evolution"""

    def __init__(
        self,
        T_ladder_in: TemperatureLadder,
        like_obj: LikelihoodType,
        block_size: int,
        store_size: int,
        tracker_manager: TrackerManager[LikelihoodType] | None = None,
        proposal_manager: AbstractProposalManager[LikelihoodType] | None = None,
        starting_samples: NDArray[np.floating] | None = None,
        store_thin: int = 1,
        arg_record: list[int] | None = None,
        kernel_backend: str = 'auto',
        zero_loglike: bool = False,
    ) -> None:
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
        proposal_manager: AbstractProposalManager
        starting_samples: a (n_chain, n_par) float array of starting samples
        store_thin: scalar integer, how much to thin the stored samples by (default 1)
        arg_record: optional sequence of chain indices to record in storage in
            addition to the ladder's n_cold readout chains (default none).
            The readout chains always occupy the first n_cold store columns
            and their indices are recomputed at every ladder update; the
            arg_record columns follow in the given order, and an index that
            duplicates a readout chain is simply stored twice
        kernel_backend: {'auto', 'numba', 'python'}
            Select the native serial block kernel. 'numba' requires every
            concrete graph component to provide native bindings and raises
            on binding or compilation failure. 'auto' silently falls back
            for a graph with no native bindings and warns once for a mixed
            graph or a compilation failure before using Python. 'python'
            disables only the native block kernel, leaving existing jitted
            helpers enabled.
        zero_loglike: bool
            Prior-recovery review mode: set the sampler's target log
            likelihood values to zero without evaluating them. The original
            likelihood remains attached to the proposal graph, so internal
            calculations such as Fisher matrix stencils continue to use it.
        """
        # fail fast before the likelihood is used: initialization below
        # draws from the prior and evaluates the likelihood
        if not isinstance(like_obj, AbstractLikelihood):
            msg = (
                f'likelihood {type(like_obj).__qualname__} does not implement AbstractLikelihood '
                '(n_par, get_loglike, prior_draw, prior_factor, validate_bounds)'
            )
            raise TypeError(msg)
        self.eval_accounting: EvalAccounting[LikelihoodType] = EvalAccounting()
        self.block_size: int = block_size
        self.n_par: int = like_obj.n_par
        self.store_size: int = store_size
        self.store_thin: int = store_thin
        self.store_idx: int = 0
        self.store_counter: int = 0
        self.itrn: int = 0
        self.like_obj: LikelihoodType = like_obj
        self.zero_loglike: bool = zero_loglike
        self.kernel_backend: str = kernel_backend
        # the backend validates kernel_backend, raising ValueError
        self._native_serial_backend = NativeSerialBackend(kernel_backend)
        self.last_kernel_backend: str = 'python'
        self.tracker_manager: TrackerManager[LikelihoodType]
        self.proposal_manager: AbstractProposalManager[LikelihoodType]
        self.starting_samples = starting_samples

        self.T_ladder: TemperatureLadder = T_ladder_in

        self.betas = self.T_ladder.betas
        self.Ts = self.T_ladder.Ts
        self.n_chain: int = self.T_ladder.n_chain
        self.n_cold: int = self.T_ladder.n_cold

        # recorded chains: the ladder's readout (cold) chains first — their
        # indices are recomputed at every ladder update — then the extras
        if arg_record is None:
            self.arg_record: list[int] = []
        else:
            self.arg_record = list(arg_record)
        assert np.all(np.asarray(self.arg_record) >= 0)
        assert np.all(np.asarray(self.arg_record) < self.n_chain)
        self.record_indices: list[int] = list(self.T_ladder.get_arg_cold()) + self.arg_record

        self.instantiate_state()

        self.initialize_iterators()
        self.initialize_state()
        self.initialize_jumps(proposal_manager)
        # validate the proposal graph before anything consumes it
        # (initialize_trackers reads the exchange manager)
        self.validate_protocol_conformance()
        self.count_construction_evals()
        self.initialize_trackers(tracker_manager)

    def count_construction_evals(self) -> None:
        """Fold the managers' declared construction costs into the accounting.

        A manager without a ``declared_construction_evals`` attribute makes
        the accounting incomplete rather than silently contributing zero.
        """
        for manager in self.proposal_manager.managers:
            declared = getattr(manager, 'declared_construction_evals', None)
            if declared is None:
                self.eval_accounting.complete = False
            else:
                self.eval_accounting.initialization += declared

    def validate_protocol_conformance(self) -> None:
        """Fail fast when a component is missing structural protocol members.

        runtime_checkable protocols verify member existence (not
        signatures), which catches incomplete extensions at construction
        in every kernel backend instead of failing obscurely mid-run. The
        likelihood's core interface is checked before any use at the top
        of ``__init__``; this hook validates the assembled proposal graph.
        """
        problems: list[str] = []
        if not isinstance(self.proposal_manager, AbstractProposalManager):
            problems.append(
                f'proposal manager {type(self.proposal_manager).__qualname__} does not implement AbstractProposalManager'
            )
        else:
            if not isinstance(self.proposal_manager.exchange_manager, AbstractExchangeManager):
                problems.append(
                    f'exchange manager {type(self.proposal_manager.exchange_manager).__qualname__} '
                    'does not implement AbstractExchangeManager'
                )
            for idx, manager in enumerate(self.proposal_manager.managers):
                if not isinstance(manager, AbstractJumpManager):
                    problems.append(
                        f'proposal manager {idx} {type(manager).__qualname__} does not implement AbstractJumpManager'
                    )
            for idx, jump in enumerate(self.proposal_manager.jumps):
                if not isinstance(jump, AbstractJump):
                    problems.append(f'jump {idx} {type(jump).__qualname__} does not implement AbstractJump')
        if problems:
            msg = '; '.join(problems)
            raise TypeError(msg)

    def initialize_trackers(self, tracker_manager_in: TrackerManager[LikelihoodType] | None = None) -> None:
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

        self.record_history: list[list[int]] = []
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
        self.samples_store = np.zeros((self.store_size, len(self.record_indices), self.n_par))
        self.logLs_store = np.zeros((self.store_size, len(self.record_indices)))

    def initialize_jumps(self, proposal_manager_in: AbstractProposalManager[LikelihoodType] | None = None) -> None:
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
            if self.zero_loglike:
                self.starting_logLs[itrt] = 0.0
            else:
                self.starting_logLs[itrt] = self.like_obj.get_loglike(self.starting_samples[itrt, :])
            self.eval_accounting.initialization += 1

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

    def get_stored_flattened(
        self, n_burnin: int, n_chain_out: int = -1, thin: int = 1
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Get the stored samples flattened, with additional thinning if desired and only the first n_chain_out store columns.

        The first n_cold columns are always the ladder's readout chains, so
        n_chain_out=n_cold selects exactly the readout posterior samples.
        """
        if n_chain_out == -1:
            n_chain_out = len(self.record_indices)

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
            np.asarray(self.record_indices),
            self.block_size,
            self.store_thin,
            1,
        )

    def reset_block(self) -> None:
        """Blank all but the first sample"""
        self.samples[1:, :, :] = 0.0
        self.logLs[1:, :] = 0.0
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
        if self._native_serial_backend.try_advance_block(
            self.T_ladder,
            self.logLs,
            self.samples,
            self.chain_track,
            self.proposal_manager,
            self.like_obj,
            self.tracker_manager,
            self.eval_accounting,
            self.zero_loglike,
        ):
            self.last_kernel_backend = 'numba'
            return
        self.last_kernel_backend = 'python'
        n_target_evals, n_internal_evals, internal_known = advance_block_ptmcmc(
            self.T_ladder,
            self.logLs,
            self.samples,
            self.chain_track,
            self.proposal_manager,
            self.like_obj,
            self.tracker_manager,
            self.zero_loglike,
        )
        self.eval_accounting.proposal_targets += n_target_evals
        self.eval_accounting.proposal_internal += n_internal_evals
        if not internal_known:
            self.eval_accounting.complete = False

    def block_end(self) -> None:
        """Things to execute after the main mcmc body of the block,
        like clean up recalculating fisher matrices, and storing results
        as well as perhaps non-legal burn in steps
        """
        self.record_history.append(self.record_indices.copy())

        self.store_samples()
        post_block_evals = self.proposal_manager.post_block_update(self.itrn, self.block_size, self.samples, self.logLs)
        if post_block_evals is None:
            self.eval_accounting.complete = False
        else:
            self.eval_accounting.post_block += post_block_evals
        self.tracker_manager.post_block_update(self.itrn, self.chain_track)
        # track the block mean and std of the likelihoods by chain
        self.logL_means.append(self.logLs[1:].mean(axis=0))
        self.logL_vars.append(self.logLs[1:].var(axis=0))
        # also track some higher powers of the likelihood distribution
        # storing them directly as powers allows moments to be calculated later
        # averaging over an arbitrarily long window in a stable way

        self.logL2_means.append((self.logLs[1:] ** 2).mean(axis=0))
        self.logL3_means.append((self.logLs[1:] ** 3).mean(axis=0))
        self.logL4_means.append((self.logLs[1:] ** 4).mean(axis=0))
        self.logL5_means.append((self.logLs[1:] ** 5).mean(axis=0))
        self.logL6_means.append((self.logLs[1:] ** 6).mean(axis=0))
        self.logL_prod11_means.append(
            (self.logLs[1:, : self.n_chain - 1] * self.logLs[1:, 1 : self.n_chain]).mean(axis=0)
        )
        self.logL_prod21_means.append(
            (self.logLs[1:, : self.n_chain - 1] ** 2 * self.logLs[1:, 1 : self.n_chain]).mean(axis=0)
        )
        self.logL_prod12_means.append(
            (self.logLs[1:, : self.n_chain - 1] * self.logLs[1:, 1 : self.n_chain] ** 2).mean(axis=0)
        )
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

    def apply_ladder_update(self, new_ladder: TemperatureLadder, remap_rule: str = 'no_remap') -> None:
        """Swap the temperature ladder at a block boundary.

        RNG-neutral: pure deterministic state remapping, no draws. The
        hook rebinds the ladder, the betas/Ts aliases, and every jump
        manager's ladder reference. Chain states remap by temperature
        rank; for equal-size sorted ladders this is the identity, so no
        walker state is cloned or discarded. logLs carry over because
        they are T-independent. DE-buffer columns remap per remap_rule —
        default 'no_remap' (columns keep their slot and re-burn-in under
        their new temperatures, matching the adaptive controller/spec
        default); under the cloning rules ('at_or_hotter', 'nearest') any
        rung whose column was resourced then gets its current state
        written at the buffer's most recently written row. Fisher scales
        refresh from the existing diagonals, and trackers are segmented
        with a round-trip event-log boundary. Walker identities restart
        with the new segment, matching the cycle-tracker reset.
        """
        assert new_ladder.n_chain == self.n_chain
        assert new_ladder.n_cold == self.n_cold
        # the rank remap relies on the sorted-ladder convention (D6);
        # clip infs so duplicate hot anchors do not produce inf - inf
        assert np.all(np.diff(np.minimum(self.Ts, 1.0e300)) >= 0.0)
        assert np.all(np.diff(np.minimum(new_ladder.Ts, 1.0e300)) >= 0.0)

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
                # in place: native jump bindings bake the scale arrays by reference
                sigma_scales, gamma_mults = set_scales(self.n_par, new_ladder, manager.sigma_diags)
                manager.sigma_scales[:] = sigma_scales
                manager.gamma_mults[:] = gamma_mults

        # the readout chains may sit at different indices in the new ladder
        # (e.g. rungs added below T_cold): recompute the recorded set and log
        # the change so stored columns stay interpretable per iteration
        record_indices_new = list(new_ladder.get_arg_cold()) + self.arg_record
        assert len(record_indices_new) == len(self.record_indices)
        self.record_indices = record_indices_new

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
