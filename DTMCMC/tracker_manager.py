"""C 2023 Matthew C. Digman
module to store various trackers about the state of chains
"""

from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from numpy.typing import NDArray

if TYPE_CHECKING:
    from DTMCMC.proposal_manager import AbstractProposalManager


# direction codes for the round-trip event log
RT_ARRIVED_COLD = 0  # walker touched the cold extreme having last been at the hot extreme
RT_ARRIVED_HOT = 1  # walker touched the hot extreme having last been at the cold extreme


# TODO fix cycle and exchange tracking if not sorted
@njit()
def process_chain_cycles(
    cycle_tracker: NDArray[np.int64],
    itrn: int,
    block_size: int,
    chain_track: NDArray[np.int64],
    n_cold: int,
    rt_event_buffer: NDArray[np.int64],
    flow_up_count: NDArray[np.int64],
    flow_labeled_count: NDArray[np.int64],
) -> int:
    """Process whether the sampler has undergone any partial cold-hot cycles.

    Also logs each extreme-touch transition as a (walker id, iteration,
    direction) event into rt_event_buffer (returning the number written)
    and accumulates per-temperature flow counts: for every step, each
    resident walker whose last extreme visit was cold counts as an
    up-mover at its temperature index. Pure observer: no random draws.
    """
    n_chain = chain_track.shape[1]
    n_events = 0
    for itrb in range(1, block_size + 1, 1):
        # check if any current cold chains have been hot more recently than it was last cold
        # if so, a hot->cold cycle has occurred
        for itrj in range(n_cold):
            chain_idx = chain_track[itrb, itrj]
            if cycle_tracker[0][chain_idx] < cycle_tracker[1][chain_idx] and cycle_tracker[0][chain_idx] > -1:
                cycle_tracker[2][chain_idx] += 1
                rt_event_buffer[n_events, 0] = chain_idx
                rt_event_buffer[n_events, 1] = itrn + itrb
                rt_event_buffer[n_events, 2] = RT_ARRIVED_COLD
                n_events += 1

        # check if the current hot chain has been cold more recently than it was last hot
        # if so, a cold->hot cycle has occurred
        chain_idx = chain_track[itrb, -1]
        if cycle_tracker[1][chain_idx] < cycle_tracker[0][chain_idx] and cycle_tracker[1][chain_idx] > -1:
            cycle_tracker[3][chain_idx] += 1
            rt_event_buffer[n_events, 0] = chain_idx
            rt_event_buffer[n_events, 1] = itrn + itrb
            rt_event_buffer[n_events, 2] = RT_ARRIVED_HOT
            n_events += 1

        # track which chain is currently hot
        cycle_tracker[1][chain_track[itrb, -1]] = itrn + itrb

        # track which chains are currently one of the cold chains
        for itrj in range(n_cold):
            cycle_tracker[0][chain_track[itrb, itrj]] = itrn + itrb

        # flow fraction f(T): with this step's timestamps in place, count
        # resident walkers by the direction of their last extreme visit
        for itrt in range(n_chain):
            walker_idx = chain_track[itrb, itrt]
            last_cold = cycle_tracker[0][walker_idx]
            last_hot = cycle_tracker[1][walker_idx]
            if last_cold > -1 or last_hot > -1:
                flow_labeled_count[itrt] += 1
                if last_cold > last_hot:
                    flow_up_count[itrt] += 1

    return n_events


# TODO clean up tracker reporting


class TrackerManager:
    """track various things about chains like acceptance rates and cycle times."""

    def __init__(
        self,
        n_cold: int,
        n_chain: int,
        block_size: int,
        n_par: int,
        track_full_exchanges: int,
        n_jump_types: int,
        n_block_archive: int,
    ) -> None:
        self.n_cold: int = n_cold
        self.n_chain: int = n_chain
        self.block_size: int = block_size
        self.n_par: int = n_par
        self.track_full_exchanges: int = track_full_exchanges
        self.n_jump_types: int = n_jump_types
        self.initialize_trackers()

        self.n_block_archive: int = n_block_archive
        self.itrb: int = 0

        self.cycle_archive: list[NDArray[np.int64]] = []
        self.accept_archive: list[NDArray[np.int64]] = []
        self.exchange_archive: list[NDArray[np.int64]] = []
        self.esd_archive: list[NDArray[np.floating]] = []
        self.esd_exchange_archive: list[NDArray[np.floating]] = []
        self.itrn_archive: list[int] = []

        # round-trip event log: (walker id, iteration, direction) rows,
        # detected per step in process_chain_cycles and flushed per block
        self.rt_event_buffer: NDArray[np.int64] = np.zeros((block_size * (n_cold + 1), 3), dtype=np.int64)
        self.rt_event_log: list[NDArray[np.int64]] = []
        # ladder-segment boundaries for the event log: the iteration of
        # each ladder update; events at or before a boundary belong to
        # the closing segment, and round-trip metrics must never pair
        # arrivals across segments (plan D6)
        self.rt_segment_itrns: list[int] = []

        # per-block flow counts: entry [itrt] counts (step, resident walker)
        # pairs at temperature index itrt whose last extreme visit was cold
        # (up-movers), and pairs with any extreme label at all
        self.flow_up_archive: list[NDArray[np.int64]] = []
        self.flow_labeled_archive: list[NDArray[np.int64]] = []

    def initialize_trackers(self) -> None:
        """Initialize the various trackers like acceptance rate and cycle times."""
        # cycle_tracker stores 4 integer variables related to tracking the number of cycles
        # the time the chain was last at T=T_cold, the time the chain was last at T=maximum index
        # the number of cycles hot to cold, and number of cycles cold to hot
        # the layout is: cycle_tracker = [chain_last_cold,chain_last_hot,chain_hc_cycles,chain_ch_cyles]
        # the hot to cold and cold to hot trackers should be within 1 of each other
        self.cycle_tracker = np.zeros((4, self.n_chain), dtype=np.int64)
        self.cycle_tracker[0][self.n_cold :] = -1
        self.cycle_tracker[1][: self.n_chain - 1] = -1
        self.cycle_tracker[3] = np.zeros(self.n_chain, dtype=np.int64)

        self.accept_record: NDArray[np.int64] = np.zeros((2, self.n_chain, self.n_jump_types), dtype=np.int64)

        # expected squared displacement sums per (temperature, jump type):
        # [0] accumulates |delta|^2 over all proposals, [1] over accepted ones
        self.esd_record: NDArray[np.floating] = np.zeros((2, self.n_chain, self.n_jump_types))

        # squared state displacement accepted exchanges produce per slot;
        # rejected swaps move nothing, so only accepted swaps accumulate
        self.esd_exchange: NDArray[np.floating] = np.zeros(self.n_chain)

        if self.track_full_exchanges:
            self.exchange_tracker: NDArray[np.int64] = np.zeros((2, self.n_chain, self.n_chain), dtype=np.int64)
        else:
            # track limited exchange information
            self.exchange_tracker = np.zeros((2, 2, self.n_chain), dtype=np.int64)

    def post_block_update(self, itrn: int, chain_track: NDArray[np.int64]) -> None:
        """Process anything the tracker needs to do after every block."""
        self.process_chain_cycles(itrn, chain_track)

        self.itrb += 1

        # occasionally archive the current states of the trackers, to track changes over time
        if self.itrb % self.n_block_archive == 0:
            self.cycle_archive.append(self.cycle_tracker.copy())
            self.accept_archive.append(self.accept_record.copy())
            self.exchange_archive.append(self.exchange_tracker.copy())
            self.esd_archive.append(self.esd_record.copy())
            self.esd_exchange_archive.append(self.esd_exchange.copy())
            self.itrn_archive.append(itrn + self.block_size)

    def process_chain_cycles(self, itrn: int, chain_track: NDArray[np.int64]) -> None:
        """Process whether the sampler has undergone any partial cold-hot cycles."""
        flow_up_count = np.zeros(self.n_chain, dtype=np.int64)
        flow_labeled_count = np.zeros(self.n_chain, dtype=np.int64)
        n_events = process_chain_cycles(
            self.cycle_tracker,
            itrn,
            self.block_size,
            chain_track,
            self.n_cold,
            self.rt_event_buffer,
            flow_up_count,
            flow_labeled_count,
        )
        if n_events > 0:
            self.rt_event_log.append(self.rt_event_buffer[:n_events].copy())
        self.flow_up_archive.append(flow_up_count)
        self.flow_labeled_archive.append(flow_labeled_count)

    def segment_for_ladder_update(self, itrn: int) -> None:
        """Archive tracker state and reset cycle tracking at a ladder update.

        Counts must not straddle a ladder change (plan D6): the current
        tracker snapshots are archived with the update iteration, then
        the cycle tracker returns to its initialized state — in-flight
        extreme-visit records refer to the old ladder. A segment
        boundary is recorded for the round-trip event log so metrics
        never pair arrivals across ladder updates. Cumulative
        accept/exchange/ESD records stay cumulative; segmentation for
        them is recovered by differencing archive entries.
        """
        self.cycle_archive.append(self.cycle_tracker.copy())
        self.accept_archive.append(self.accept_record.copy())
        self.exchange_archive.append(self.exchange_tracker.copy())
        self.esd_archive.append(self.esd_record.copy())
        self.esd_exchange_archive.append(self.esd_exchange.copy())
        self.itrn_archive.append(itrn)
        self.rt_segment_itrns.append(itrn)

        self.cycle_tracker[:] = 0
        self.cycle_tracker[0][self.n_cold :] = -1
        self.cycle_tracker[1][: self.n_chain - 1] = -1

    def get_rt_segment_itrns(self) -> NDArray[np.int64]:
        """Get the round-trip event-log segment boundaries as an array."""
        return np.asarray(self.rt_segment_itrns, dtype=np.int64)

    def get_rt_events(self) -> NDArray[np.int64]:
        """Get the full round-trip event log as an (n_events, 3) array."""
        if len(self.rt_event_log) == 0:
            return np.zeros((0, 3), dtype=np.int64)
        return np.concatenate(self.rt_event_log, axis=0)

    def get_flow_counts(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Get per-block flow counts as (n_blocks, n_chain) arrays (up-movers, labeled)."""
        if len(self.flow_up_archive) == 0:
            empty = np.zeros((0, self.n_chain), dtype=np.int64)
            return empty, empty.copy()
        return np.asarray(self.flow_up_archive), np.asarray(self.flow_labeled_archive)

    def get_exchange_rate_summary(
        self, itrt_start: int = 0, itrt_end: int = -1, last_itrn: int = -1
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Get nn exchange rate summary."""
        if last_itrn == -1 and len(self.itrn_archive) >= 2:
            exchange_tracker_loc = self.exchange_tracker - self.exchange_archive[-2]
        else:
            exchange_tracker_loc = self.exchange_tracker

        itrt_start = min(itrt_start, self.n_chain - 1)

        itrt_start = max(itrt_start, 0)

        if itrt_end == -1 or itrt_end > self.n_chain:
            itrt_end = self.n_chain

        if itrt_end < itrt_start:
            itrt_end = min(itrt_start + 1, self.n_chain)

        if self.track_full_exchanges:
            a_yes = exchange_tracker_loc[0, itrt_start:itrt_end, itrt_start:itrt_end]
            a_no = exchange_tracker_loc[1, itrt_start:itrt_end, itrt_start:itrt_end]

            a_yes_nn_right = np.hstack(
                [a_yes[np.arange(0, itrt_end - itrt_start - 1), np.arange(1, itrt_end - itrt_start)], np.array([0])]
            )
            a_no_nn_right = np.hstack(
                [a_no[np.arange(0, itrt_end - itrt_start - 1), np.arange(1, itrt_end - itrt_start)], np.array([0])]
            )

            a_yes_nn_left = np.hstack(
                [np.array([0]), a_yes[np.arange(0, itrt_end - itrt_start - 1), np.arange(1, itrt_end - itrt_start)]]
            )
            a_no_nn_left = np.hstack(
                [np.array([0]), a_no[np.arange(0, itrt_end - itrt_start - 1), np.arange(1, itrt_end - itrt_start)]]
            )

            a_yes_nn_sym = a_yes_nn_right + a_yes_nn_left
            a_no_nn_sym = a_no_nn_right + a_no_nn_left

            exchange_vec_nn_sym: NDArray[np.floating] = a_yes_nn_sym / (a_yes_nn_sym + a_no_nn_sym)
            exchange_tot_nn: NDArray[np.floating] = a_yes.sum() / (a_yes.sum() + a_no.sum())
            exchange_full: NDArray[np.floating] = a_yes / (a_yes + a_no)

        else:
            a_yes_nn_sym = exchange_tracker_loc[1, 0, itrt_start:]
            a_no_nn_sym = exchange_tracker_loc[1, 1, itrt_start:]

            exchange_vec_nn_sym = a_yes_nn_sym / (a_yes_nn_sym + a_no_nn_sym)
            exchange_tot_nn = a_yes_nn_sym.sum() / (a_yes_nn_sym.sum() + a_no_nn_sym.sum())
            exchange_full = exchange_vec_nn_sym.copy()

        return exchange_full, exchange_vec_nn_sym, exchange_tot_nn

    @property
    def n_cycles(self) -> NDArray[np.int64]:
        """Get number of complete hot to cold to hot (or vice versa) cycles each chain has undergone."""
        res: NDArray[np.int64] = np.min([self.cycle_tracker[3], self.cycle_tracker[2]], axis=0)
        return res

    def print_tracker_summary(
        self, n_cold: int, Ts: NDArray[np.floating], proposal_manager: AbstractProposalManager, last_itrn: int = -1
    ) -> None:
        """Print a summmary of results from this tracker object."""
        with np.errstate(invalid='ignore', divide='ignore'):
            if last_itrn == -1 and len(self.itrn_archive) >= 2:
                accept_record_loc = self.accept_record - self.accept_archive[-2]
            else:
                accept_record_loc = self.accept_record

            # combine acceptances from identical temperatures
            Ts_unique = np.unique(Ts)
            a_yes_unique = np.zeros((Ts_unique.size, accept_record_loc[0].shape[1]))
            a_no_unique = np.zeros((Ts_unique.size, accept_record_loc[0].shape[1]))
            for itrt, T_loc in enumerate(Ts_unique):
                a_yes_unique[itrt] = accept_record_loc[0, Ts == T_loc].sum(axis=0)
                a_no_unique[itrt] = accept_record_loc[1, Ts == T_loc].sum(axis=0)

            a_tot_unique = a_yes_unique + a_no_unique  # get total number of trials
            a_any_mask = np.any(a_tot_unique > 0, axis=0)  # filter to only jumps that are actually tried
            a_yes_unique = a_yes_unique[:, a_any_mask]
            a_no_unique = a_no_unique[:, a_any_mask]
            a_tot_unique = a_tot_unique[:, a_any_mask]
            jump_labels_need = [
                label for label, keep in zip(proposal_manager.get_jump_labels(), a_any_mask, strict=True) if keep
            ]

            # build the print string for jump labels
            print('Acceptance Summary:')
            label_str = '%12s ' % 'Temperature'
            print(jump_labels_need)
            for label_got in jump_labels_need:
                # find the label if it is recorded somewhere
                # label_got = "#%-9d"%code

                label_loc = ' %-15s' % label_got

                label_str = label_str + label_loc

            print(label_str)
            for itrt, T_loc in enumerate(Ts_unique):
                label_T = '%12e ' % T_loc
                for itrj in range(len(jump_labels_need)):
                    if a_tot_unique[itrt, itrj] == 0:
                        # no trials so print something useful instead of nan
                        label_loc = '%-15s ' % ' No Trials'
                    elif a_yes_unique[itrt, itrj] == 0:
                        # print an upper limit on the acceptance instead of zero if no trials were accepted
                        label_loc = '<%-.3e      ' % (1.0 / a_tot_unique[itrt, itrj])
                    else:
                        # print the actual acceptance
                        label_loc = ' %-9.7f      ' % (a_yes_unique[itrt, itrj] / a_tot_unique[itrt, itrj])
                    label_T = label_T + label_loc
                print(label_T)

            _exchange_full, _exchange_nn, exchange_overall = self.get_exchange_rate_summary(0)
            _exchange_full_no_cold, exchange_nn_no_cold, exchange_overall_no_cold = self.get_exchange_rate_summary(
                n_cold
            )
            # TODO maybe need option to use actual nearest neighbors, not just in the ladder
            print(
                'overall exchange rate, no cold exchange rate, no cold nearest neighbor exchange rate',
                exchange_overall,
                exchange_overall_no_cold,
                np.mean(exchange_nn_no_cold),
            )
