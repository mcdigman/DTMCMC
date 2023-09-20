"""C 2023 Matthew C. Digman
module to store various trackers about the state of chains"""
import numpy as np
from numba import njit


class TrackerManager():
    """track various things about chains like acceptance rates and cycle times"""

    def __init__(self, n_cold, n_chain, block_size, n_par, track_full_exchanges, n_jump_types):
        self.n_cold = n_cold
        self.n_chain = n_chain
        self.block_size = block_size
        self.n_par = n_par
        self.track_full_exchanges = track_full_exchanges
        self.n_jump_types = n_jump_types
        self.initialize_trackers()

    def initialize_trackers(self):
        """initialize the various trackers like acceptance rate and cycle times"""

        # cycle_tracker stores 4 integer variables related to tracking the number of cycles
        # the time the chain was last at T=T_cold, the time the chain was last at T=maximum index
        # the number of cycles hot to cold, and number of cycles cold to hot
        # the layout is: cycle_tracker = [chain_last_cold,chain_last_hot,chain_hc_cycles,chain_ch_cyles]
        # the hot to cold and cold to hot trackers should be within 1 of each other
        self.cycle_tracker = np.zeros((4, self.n_chain), dtype=np.int64)
        self.cycle_tracker[0][self.n_cold:] = -1
        self.cycle_tracker[1][:self.n_chain-1] = -1
        self.cycle_tracker[3] = np.zeros(self.n_chain, dtype=np.int64)

        self.accept_record = np.zeros((2, self.n_chain, self.n_jump_types), dtype=np.int64)

        if self.track_full_exchanges:
            self.exchange_tracker = np.zeros((2, self.n_chain, self.n_chain), dtype=np.int64)
        else:
            # track limited exchange information
            self.exchange_tracker = np.zeros((2, 2, self.n_chain), dtype=np.int64)

    def process_chain_cycles(self, itrn, chain_track):
        """process whether the sampler has undergone any partial cold-hot cycles"""
        process_chain_cycles(self.cycle_tracker, itrn, self.block_size, chain_track, self.n_cold)

    def get_total_exchange_rate(self, idx_start):
        """get the full excange rate summary"""
        if self.track_full_exchanges:
            a_yes = np.zeros(self.n_chain-idx_start, dtype=np.int64)
            a_no = np.zeros(self.n_chain-idx_start, dtype=np.int64)
            for itrm in range(0, self.n_chain-idx_start):
                idxm = itrm+idx_start
                a_yes[itrm] = np.sum(self.exchange_tracker[0, :, idxm])+np.sum(self.exchange_tracker[0, idxm, :])
                a_no[itrm] = np.sum(self.exchange_tracker[1, :, idxm])+np.sum(self.exchange_tracker[1, idxm, :])
        else:
            a_yes = self.exchange_tracker[0, 0, idx_start:]
            a_no = self.exchange_tracker[0, 1, idx_start:]

        exchange_vec = a_yes/(a_yes+a_no)
        exchange_tot = a_yes.sum()/(a_yes.sum()+a_no.sum())
        return exchange_vec, exchange_tot

    def get_nn_exchange_rate(self, idx_start):
        """get nn exchange rate summary"""
        if self.track_full_exchanges:
            a_yes = np.zeros(self.n_chain-idx_start, dtype=np.int64)
            a_no = np.zeros(self.n_chain-idx_start, dtype=np.int64)
            for itrm in range(0, self.n_chain-idx_start):
                idxm = itrm+idx_start
                if idxm+1 < self.n_chain:
                    a_yes[itrm] += self.exchange_tracker[0, idxm, idxm+1]
                    a_no[itrm] += self.exchange_tracker[1, idxm, idxm+1]
                if idxm > 0:
                    a_yes[itrm] += self.exchange_tracker[0, idxm-1, idxm]
                    a_no[itrm] += self.exchange_tracker[1, idxm-1, idxm]
        else:
            a_yes = self.exchange_tracker[1, 0, idx_start:]
            a_no = self.exchange_tracker[1, 1, idx_start:]

        exchange_vec = a_yes/(a_yes+a_no)
        exchange_tot = a_yes.sum()/(a_yes.sum()+a_no.sum())
        return exchange_vec, exchange_tot

    def get_n_cycles(self):
        """get number of complete hot to cold to hot (or vice versa) cycles each chain has undergone"""
        return np.min([self.cycle_tracker[3], self.cycle_tracker[2]], axis=0)

    def print_tracker_summary(self, n_cold, Ts, proposal_manager):
        """print a summmary of results from this tracker object"""
        with np.errstate(invalid='ignore', divide='ignore'):
            # combine acceptances from identical temperatures
            Ts_unique = np.unique(Ts)
            a_yes_unique = np.zeros((Ts_unique.size, self.accept_record[0].shape[1]))
            a_no_unique = np.zeros((Ts_unique.size, self.accept_record[0].shape[1]))
            for itrt, T_loc in enumerate(Ts_unique):
                a_yes_unique[itrt] = self.accept_record[0, Ts == T_loc].sum(axis=0)
                a_no_unique[itrt] = self.accept_record[1, Ts == T_loc].sum(axis=0)

            a_tot_unique = a_yes_unique+a_no_unique    # get total number of trials
            a_any_mask = np.any(a_tot_unique > 0, axis=0)  # filter to only jumps that are actually tried
            a_yes_unique = a_yes_unique[:, a_any_mask]
            a_no_unique = a_no_unique[:, a_any_mask]
            a_tot_unique = a_tot_unique[:, a_any_mask]
            jump_codes_need = proposal_manager.get_jump_codes()[a_any_mask]
            jump_labels_need = proposal_manager.get_jump_labels()[a_any_mask]

            # build the print string for jump labels
            print("Acceptance Summary:")
            label_str = "%12s " % "Temperature"
            for itrj, label_got in enumerate(jump_labels_need):
                # find the label if it is recorded somewhere
                label_got = jump_labels_need[itrj]
                # label_got = "#%-9d"%code

                label_loc = " %-15s" % label_got

                label_str = label_str+label_loc

            print(label_str)
            for itrt, T_loc in enumerate(Ts_unique):
                label_T = "%12e " % T_loc
                for itrj in range(jump_codes_need.size):
                    if a_tot_unique[itrt, itrj] == 0:
                        # no trials so print something useful instead of nan
                        label_loc = "%-15s " % " No Trials"
                    elif a_yes_unique[itrt, itrj] == 0:
                        # print an upper limit on the acceptance instead of zero if no trials were accepted
                        label_loc = "<%-.3e      " % (1./a_tot_unique[itrt, itrj])
                    else:
                        # print the actual acceptance
                        label_loc = " %-9.7f      " % (a_yes_unique[itrt, itrj]/a_tot_unique[itrt, itrj])
                    label_T = label_T+label_loc
                print(label_T)

            _, exchange_overall = self.get_total_exchange_rate(0)
            _, exchange_no_cold = self.get_total_exchange_rate(n_cold)
            _, exchange_nn = self.get_nn_exchange_rate(n_cold)
            print("overall exchange rate, no cold exchange rate, no cold nearest neighbor exchange rate", exchange_overall, exchange_no_cold, exchange_nn)


@njit()
def process_chain_cycles(cycle_tracker, itrn, block_size, chain_track, n_cold):
    """process whether the sampler has undergone any partial cold-hot cycles"""
    for itrb in range(1, block_size+1, 1):
        # track which chain is currently hot
        cycle_tracker[1][chain_track[itrb, -1]] = itrn+itrb

        # track which chains are currently one of the cold chains
        for itrj in range(n_cold):
            cycle_tracker[0][chain_track[itrb, itrj]] = itrn+itrb

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
