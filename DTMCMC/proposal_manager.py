"""C 2023 Matthew C. Digman
Manager object to handle all dispatching of proposals"""
import numpy as np
from DTMCMC.jump_manager import JumpManager
import DTMCMC.prior_manager as ph


class ProposalManager(JumpManager):
    """manage generation of proposals, handles all dispatching of jumps"""

    def __init__(self, T_ladder, like_obj, managers, exchange_manager, config):
        """create the core proposal manager object, subclass of DTMCMC.jump_manager.JumpManager
            inputs:
                T_ladder: a DTMCMC.temperature_helpers.TemperatureLadder object (or suitable replacement)
                managers: a tuple of objects extending DTMCMC.jump_manager.JumpManager, managers to dispatch jumps too
                exchange_manager: a DTMCMC.exchange_manager.ExchangeManager object (or suitable replacement)
                                  exchanges are a separate manager from the other jump types because
                                  they are fundamentally different in how the temperatures interact
                config: a ConfigParser object storing any relevant configuration variables"""

        self.config = config

        self.only_prior_hot = config['ProposalManager'].getboolean('only_prior_hot',True)

        #self.T_ladder = T_ladder

        self.managers = managers
        self.n_managers = len(self.managers)
        self.n_jumps_managers = np.zeros(self.n_managers, dtype=np.int64)

        self.exchange_manager = exchange_manager

        jumps_need_temp = []
        jump_labels_temp = []
        for itrm, manager in enumerate(self.managers):
            jumps_need_loc = manager.get_jump_codes()
            jump_labels_loc = manager.get_jump_labels()
            self.n_jumps_managers[itrm] = len(jumps_need_loc)
            jumps_need_temp.append(jumps_need_loc)
            jump_labels_temp.append(jump_labels_loc)

        self.jumps_need = np.hstack(jumps_need_temp)
        self.jump_labels_array = np.hstack(jump_labels_temp)
        self.n_jump_types = np.sum(self.n_jumps_managers)

        self.jump_labels_dict = {}
        for itrp in range(self.jumps_need.size):
            self.jump_labels_dict[self.jumps_need[itrp]] = self.jump_labels_array[itrp]


        self.choose_idx_modifiers = np.zeros(self.n_managers,dtype=np.int64)
        if self.n_managers>1:
            # If there is more than one manager we will need to adjust the jump indexes when dispatching
            for itrm in range(1,self.n_managers):
                self.choose_idx_modifiers[itrm:] += self.n_jumps_managers[itrm-1]

        print(self.jumps_need)
        print(self.choose_idx_modifiers)

        #self.n_chain = self.T_ladder.n_chain

        #self.jump_probs = np.zeros((self.n_chain, self.n_jump_types))
        #self.jump_weights = np.zeros((self.n_chain, self.n_jump_types))

        #self.set_jump_weights()

        JumpManager.__init__(self, T_ladder, like_obj, self.jumps_need, self.jump_labels_dict)

    def dispatch_jump(self, sample_point, itrt, choose=-1):
        """generate a proposal"""

        if choose == -1:
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

        found = False

        itrj1 = 0
        for itrm in range(self.n_managers):
            itrj2 = itrj1+self.n_jumps_managers[itrm]
            if itrj1 <= choose < itrj2:
                # found the correct manager, dispatch the jump
                choose_loc = choose-self.choose_idx_modifiers[itrm]
                new_point, density_fac, success = self.managers[itrm].dispatch_jump(sample_point, itrt, choose_loc)
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
        for itrm in range(self.n_managers):
            itrj2 = itrj1+self.n_jumps_managers[itrm]
            jump_weights[:, itrj1:itrj2] = self.managers[itrm].get_jump_weights()
            if self.only_prior_hot and not isinstance(self.managers[itrm], ph.PriorManager):
                # override and only allow prior-type draws to contribute to the last chain
                jump_weights[n_chain-1, itrj1:itrj2] = 0.
            itrj1 = itrj2

        self.jump_weights = jump_weights
        assert np.all(self.jump_weights >= 0.)

    def set_jump_probs(self):
        """set the normalized probabilities of the jump subtypes
        as a function of temperature, relying on set_jump_weights.
        The overall manager cannot have any rows with 0 total probability,
        so assert that """
        super().set_jump_probs()
        # individual proposals can have temps where they do not suggest proposals
        # but the overarching proposal manager must make proposals for all temps 
        assert np.all(np.sum(self.jump_probs,axis=1)==1.)

    def post_step_update(self, samples):
        """do any needed internal processing after an individual step of all temperatures;
        mainly intended to be used to write to differential evolution buffer"""
        for itrm in range(self.n_managers):
            self.managers[itrm].post_step_update(samples)

    def post_block_update(self, itrn, block_size, samples, logLs):
        """do any needed internal processing after an individual block of size block_size:
        ie, fisher matrix updates"""
        for itrm in range(self.n_managers):
            self.managers[itrm].post_block_update(itrn, block_size, samples, logLs)

    def record_config(self,config_in):
        """record the current configuration to an input ConfigParser object config_in"""
        for itrm in range(self.n_managers):
            self.managers[itrm].record_config(config_in)

        config_in['ProposalManager']['only_prior_hot'] = self.only_prior_hot
