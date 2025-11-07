"""C 2023 Matthew C. Digman
module to store various trackers about the state of chains
"""
import numpy as np
from numba import njit


# TODO fix cycle and exchange tracking if not sorted
@njit()
def process_chain_cycles(cycle_tracker, itrn, block_size, chain_track, n_cold) -> None:
    """Process whether the sampler has undergone any partial cold-hot cycles"""
    for itrb in range(1, block_size + 1, 1):

        # check if any current cold chains have been hot more recently than it was last cold
        # if so, a hot->cold cycle has occurred
        for itrj in range(n_cold):
            chain_idx = chain_track[itrb, itrj]
            if cycle_tracker[0][chain_idx] < cycle_tracker[1][chain_idx] and cycle_tracker[0][chain_idx] > -1:
                cycle_tracker[2][chain_idx] += 1

        # check if the current hot chain has been cold more recently than it was last hot
        # if so, a cold->hot cycle has occurred
        chain_idx = chain_track[itrb, -1]
        if cycle_tracker[1][chain_idx] < cycle_tracker[0][chain_idx] and cycle_tracker[1][chain_idx] > -1:
            cycle_tracker[3][chain_idx] += 1

        # track which chain is currently hot
        cycle_tracker[1][chain_track[itrb, -1]] = itrn + itrb

        # track which chains are currently one of the cold chains
        for itrj in range(n_cold):
            cycle_tracker[0][chain_track[itrb, itrj]] = itrn + itrb

# TODO clean up tracker reporting


class TrackerManager():
    """track various things about chains like acceptance rates and cycle times"""

    def __init__(self, n_cold, n_chain, block_size, n_par, track_full_exchanges, n_jump_types, n_block_archive) -> None:
        self.n_cold = n_cold
        self.n_chain = n_chain
        self.block_size = block_size
        self.n_par = n_par
        self.track_full_exchanges = track_full_exchanges
        self.n_jump_types = n_jump_types
        self.initialize_trackers()

        self.n_block_archive = n_block_archive
        self.itrb = 0

        self.cycle_archive = []
        self.accept_archive = []
        self.exchange_archive = []
        self.itrn_archive = []

    def initialize_trackers(self) -> None:
        """Initialize the various trackers like acceptance rate and cycle times"""
        # cycle_tracker stores 4 integer variables related to tracking the number of cycles
        # the time the chain was last at T=T_cold, the time the chain was last at T=maximum index
        # the number of cycles hot to cold, and number of cycles cold to hot
        # the layout is: cycle_tracker = [chain_last_cold,chain_last_hot,chain_hc_cycles,chain_ch_cyles]
        # the hot to cold and cold to hot trackers should be within 1 of each other
        self.cycle_tracker = np.zeros((4, self.n_chain), dtype=np.int64)
        self.cycle_tracker[0][self.n_cold:] = -1
        self.cycle_tracker[1][:self.n_chain - 1] = -1
        self.cycle_tracker[3] = np.zeros(self.n_chain, dtype=np.int64)

        self.accept_record = np.zeros((2, self.n_chain, self.n_jump_types), dtype=np.int64)

        if self.track_full_exchanges:
            self.exchange_tracker = np.zeros((2, self.n_chain, self.n_chain), dtype=np.int64)
        else:
            # track limited exchange information
            self.exchange_tracker = np.zeros((2, 2, self.n_chain), dtype=np.int64)

    def post_block_update(self, itrn, chain_track) -> None:
        """Process anything the tracker needs to do after every block"""
        self.process_chain_cycles(itrn, chain_track)

        self.itrb += 1

        # occasionally archive the current states of the trackers, to track changes over time
        if self.itrb % self.n_block_archive == 0:
            self.cycle_archive.append(self.cycle_tracker.copy())
            self.accept_archive.append(self.accept_record.copy())
            self.exchange_archive.append(self.exchange_tracker.copy())
            self.itrn_archive.append(itrn + self.block_size)

    def process_chain_cycles(self, itrn, chain_track) -> None:
        """Process whether the sampler has undergone any partial cold-hot cycles"""
        process_chain_cycles(self.cycle_tracker, itrn, self.block_size, chain_track, self.n_cold)

    def get_exchange_rate_summary(self, itrt_start=0, itrt_end=-1, last_itrn=-1):
        """Get nn exchange rate summary"""
        if last_itrn == -1 and len(self.itrn_archive) >= 2:
            exchange_tracker_loc = self.exchange_tracker - self.exchange_archive[-2]
        else:
            exchange_tracker_loc = self.exchange_tracker

        if itrt_start > self.n_chain - 1:
            itrt_start = self.n_chain - 1

        if itrt_start < 0:
            itrt_start = 0

        if itrt_end == -1 or itrt_end > self.n_chain:
            itrt_end = self.n_chain

        if itrt_end < itrt_start:
            itrt_end = min(itrt_start + 1, self.n_chain)

        if self.track_full_exchanges:

            a_yes = exchange_tracker_loc[0, itrt_start:itrt_end, itrt_start:itrt_end]
            a_no = exchange_tracker_loc[1, itrt_start:itrt_end, itrt_start:itrt_end]

            a_yes_nn_right = np.hstack([a_yes[np.arange(0, itrt_end - itrt_start - 1), np.arange(1, itrt_end - itrt_start)], np.array([0])])
            a_no_nn_right = np.hstack([a_no[np.arange(0, itrt_end - itrt_start - 1), np.arange(1, itrt_end - itrt_start)], np.array([0])])

            a_yes_nn_left = np.hstack([np.array([0]), a_yes[np.arange(0, itrt_end - itrt_start - 1), np.arange(1, itrt_end - itrt_start)]])
            a_no_nn_left = np.hstack([np.array([0]), a_no[np.arange(0, itrt_end - itrt_start - 1), np.arange(1, itrt_end - itrt_start)]])

            a_yes_nn_sym = a_yes_nn_right + a_yes_nn_left
            a_no_nn_sym = a_no_nn_right + a_no_nn_left

            exchange_vec_nn_sym = a_yes_nn_sym / (a_yes_nn_sym + a_no_nn_sym)
            exchange_tot_nn = a_yes.sum() / (a_yes.sum() + a_no.sum())
            exchange_full = a_yes / (a_yes + a_no)

        else:
            a_yes_nn_sym = exchange_tracker_loc[1, 0, itrt_start:]
            a_no_nn_sym = exchange_tracker_loc[1, 1, itrt_start:]

            exchange_vec_nn_sym = a_yes_nn_sym / (a_yes_nn_sym + a_no_nn_sym)
            exchange_tot_nn = a_yes_nn_sym.sum() / (a_yes_nn_sym.sum() + a_no_nn_sym.sum())
            exchange_full = exchange_vec_nn_sym.copy()

        return exchange_full, exchange_vec_nn_sym, exchange_tot_nn

    def get_n_cycles(self):
        """Get number of complete hot to cold to hot (or vice versa) cycles each chain has undergone"""
        return np.min([self.cycle_tracker[3], self.cycle_tracker[2]], axis=0)

    def print_tracker_summary(self, n_cold, Ts, proposal_manager, last_itrn=-1) -> None:
        """Print a summmary of results from this tracker object"""
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

            a_tot_unique = a_yes_unique + a_no_unique    # get total number of trials
            a_any_mask = np.any(a_tot_unique > 0, axis=0)  # filter to only jumps that are actually tried
            a_yes_unique = a_yes_unique[:, a_any_mask]
            a_no_unique = a_no_unique[:, a_any_mask]
            a_tot_unique = a_tot_unique[:, a_any_mask]
            jump_labels_need = proposal_manager.get_jump_labels()[a_any_mask]

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
                for itrj in range(jump_labels_need.size):
                    if a_tot_unique[itrt, itrj] == 0:
                        # no trials so print something useful instead of nan
                        label_loc = '%-15s ' % ' No Trials'
                    elif a_yes_unique[itrt, itrj] == 0:
                        # print an upper limit on the acceptance instead of zero if no trials were accepted
                        label_loc = '<%-.3e      ' % (1. / a_tot_unique[itrt, itrj])
                    else:
                        # print the actual acceptance
                        label_loc = ' %-9.7f      ' % (a_yes_unique[itrt, itrj] / a_tot_unique[itrt, itrj])
                    label_T = label_T + label_loc
                print(label_T)

            _exchange_full, _exchange_nn, exchange_overall = self.get_exchange_rate_summary(0)
            _exchange_full_no_cold, exchange_nn_no_cold, exchange_overall_no_cold = self.get_exchange_rate_summary(n_cold)
            # TODO maybe need option to use actual nearest neighbors, not just in the ladder
            print('overall exchange rate, no cold exchange rate, no cold nearest neighbor exchange rate', exchange_overall, exchange_overall_no_cold, np.mean(exchange_nn_no_cold))
