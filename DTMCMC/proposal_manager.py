"""C 2023 Matthew C. Digman
Manager object to handle all dispatching of proposals"""
import numpy as np
from DTMCMC.jump_manager import JumpManager
import DTMCMC.prior_manager as ph


class ProposalManager(JumpManager):
    """manage generation of proposals, handles all dispatching of jumps"""

    def __init__(self, T_ladder, strategy_params, managers, exchange_manager):
        """create the core proposal manager object, subclass of DTMCMC.jump_manager.JumpManager
            inputs:
                T_ladder: a DTMCMC.temperature_helpers.TemperatureLadder object (or suitable replacement)
                strategy_params: a DTMCMC.strategy_helper.StrategyParams object (or suitable replacement)
                managers: a tuple of objects extending DTMCMC.jump_manager.JumpManager, managers to dispatch jumps too
                exchange_manager: a DTMCMC.exchange_manager.ExchangeManager object (or suitable replacement)
                                  exchanges are a separate manager from the other jump types because
                                  they are fundamentally different in how the temperatures interact"""
        self.T_ladder = T_ladder
        self.strategy_params = strategy_params

        self.managers = managers
        self.n_managers = len(self.managers)
        self.n_jumps_managers = np.zeros(self.n_managers, dtype=np.int64)

        self.exchange_manager = exchange_manager

        jumps_need_temp = []
        jump_labels_temp = []
        for itrm, manager in enumerate(self.managers):
            jumps_need_loc = manager.get_jump_codes()
            jump_labels_loc = manager.get_jump_labels()
            self.n_jumps_managers[itrm] = jumps_need_loc.size
            jumps_need_temp.append(jumps_need_loc)
            jump_labels_temp.append(jump_labels_loc)

        self.jumps_need = np.hstack(jumps_need_temp)
        self.jump_labels_array = np.hstack(jump_labels_temp)
        self.n_jump_types = self.jumps_need.size
        print(self.jumps_need)

        # map the codes that exist to indices in jumps_need
        self.code_to_idx = np.zeros(self.jumps_need.max()+1, dtype=np.int64)-1
        for idx, code in enumerate(self.jumps_need):
            self.code_to_idx[code] = idx

        self.n_chain = self.T_ladder.n_chain

        self.jump_probs = np.zeros((self.n_chain, self.n_jump_types))
        self.jump_weights = np.zeros((self.n_chain, self.n_jump_types))

        self.set_jump_weights()

    def dispatch_jump(self, sample_point, itrt, choose_code=-1):
        """generate a proposal"""

        # choose the jump
        choose_val = np.random.uniform(0., 1)
        choose_sum = self.jump_probs[itrt][0]
        choose = self.jump_probs[itrt].size-1
        for itrp in range(1, self.jump_probs[itrt].size):
            if choose_val < choose_sum:
                choose = itrp-1
                break
            else:
                choose_sum += self.jump_probs[itrt][itrp]

        new_point = sample_point.copy()
        density_fac = 0.
        success = False

        found = False

        # figure out which jump we chose and dispatch it
        if choose_code == -1:
            choose_code = self.jumps_need[choose]

        itrj1 = 0
        for itrm in range(0, self.n_managers):
            itrj2 = itrj1+self.n_jumps_managers[itrm]
            if itrj1 <= choose < itrj2:
                # found the correct manager, dispatch the jump
                new_point, density_fac, success = self.managers[itrm].dispatch_jump(sample_point, itrt, choose_code)
                found = True
                break
            itrj1 = itrj2

        assert found  # make sure we actually tried a jump

        return new_point, density_fac, choose, success

    def get_jump_weights(self):
        """return the unnormalized jump weights for each jump type the manager knows"""
        return self.jump_weights

    def set_jump_weights(self):
        """set the jump probabilities for everything combined"""
        n_chain = self.T_ladder.n_chain
        jump_weights = np.zeros((n_chain, self.n_jump_types))

        itrj1 = 0
        for itrm in range(0, self.n_managers):
            itrj2 = itrj1+self.n_jumps_managers[itrm]
            jump_weights[:, itrj1:itrj2] = self.managers[itrm].get_jump_weights()
            if not isinstance(self.managers[itrm], ph.PriorManager):
                jump_weights[n_chain-1, itrj1:itrj2] = 0.  # override specified weights and only allow prior-type draws to contribute to the last chain
            itrj1 = itrj2

        jump_probs = (jump_weights.T/jump_weights.sum(axis=1)).T
        jump_probs[~np.isfinite(jump_probs)]=0.

        self.jump_weights = jump_weights
        self.jump_probs = jump_probs

    def get_jump_codes(self):
        """return the internal codes the manager object uses to index its respective jump types"""
        return self.jumps_need.copy()

    def get_jump_labels(self):
        """get text labels for the different jump types"""
        return self.jump_labels_array.copy()

    def post_step_update(self, samples):
        """do any needed internal processing after an individual step of all temperatures;
        mainly intended to be used to write to differential evolution buffer"""
        for itrm in range(0, self.n_managers):
            self.managers[itrm].post_step_update(samples)

    def post_block_update(self, itrn, block_size, samples, logLs):
        """do any needed internal processing after an individual block of size block_size:
        ie, fisher matrix updates"""
        for itrm in range(0, self.n_managers):
            self.managers[itrm].post_block_update(itrn, block_size, samples, logLs)
