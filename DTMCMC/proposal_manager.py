"""
Manager object to handle all dispatching of proposals.

C 2023 Matthew C. Digman
"""
from typing import TYPE_CHECKING

import numpy as np

import DTMCMC.prior_manager as ph
from DTMCMC.jump_manager import AbstractJump, JumpManager

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from DTMCMC.exchange_manager import ExchangeManager
    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder


class ProposalManager(JumpManager):
    """Manage generation of proposals, handles all dispatching of jumps."""

    def __init__(self, T_ladder: TemperatureLadder, like_obj: AbstractLikelihood, managers: tuple[JumpManager, ...], exchange_manager: ExchangeManager, config) -> None:
        """Create the core proposal manager object.

        Parameters
        ----------
        T_ladder: TemperatureLadder
            Temperatures for the sampler
        managers: tuple[JumpManager, ...]
            Managers to dispatch jumps between
        exchange_manager: ExchangeManager
            Manager for exchange proposals.
            Exchanges are a separate manager from the other jump types because
            the temperatures intereact differenctly
        config: ConfigParser
            object storing any relevant configuration variables
        """
        self.config = config

        self.only_prior_hot: bool = config['ProposalManager'].getboolean('only_prior_hot', True)

        # self.T_ladder = T_ladder

        self.managers: tuple[JumpManager, ...] = managers
        self.n_managers: int = len(self.managers)
        self.n_jumps_managers: NDArray[np.int64] = np.zeros(self.n_managers, dtype=np.int64)

        self.exchange_manager: ExchangeManager = exchange_manager

        jump_labels_temp: list[str] = []
        jumps_temp: list[AbstractJump] = []
        for itrm, manager in enumerate(self.managers):
            jump_labels_loc: list[str] = manager.get_jump_labels()
            self.n_jumps_managers[itrm] = len(manager.jumps)
            jump_labels_temp.extend(jump_labels_loc)
            jumps_temp.extend(manager.jumps)

        self.jumps: list[AbstractJump] = jumps_temp
        self.jump_labels_array: list[str] = jump_labels_temp
        self.n_jump_types: int = int(np.sum(self.n_jumps_managers))

        self.choose_idx_modifiers: NDArray[np.int64] = np.zeros(self.n_managers, dtype=np.int64)
        if self.n_managers > 1:
            # If there is more than one manager we will need to adjust the jump indexes when dispatching
            for itrm in range(1, self.n_managers):
                self.choose_idx_modifiers[itrm:] += self.n_jumps_managers[itrm - 1]

        print(self.jumps)
        print(self.choose_idx_modifiers)

        JumpManager.__init__(self, T_ladder, like_obj, self.jumps)

    def get_jump_weights(self):
        """Return the unnormalized jump weights for each jump type the manager knows."""
        return self.jump_weights

    def set_jump_weights(self) -> None:
        """Set the jump probabilities for everything combined."""
        n_chain: int = self.T_ladder.n_chain
        jump_weights: NDArray[np.floating] = np.zeros((n_chain, self.n_jump_types))

        itrj1: int = 0
        for itrm in range(self.n_managers):
            itrj2: int = itrj1 + self.n_jumps_managers[itrm]
            jump_weights[:, itrj1:itrj2] = self.managers[itrm].get_jump_weights()
            if self.only_prior_hot and not isinstance(self.managers[itrm], ph.PriorManager):
                # override and only allow prior-type draws to contribute to the last chain
                jump_weights[n_chain - 1, itrj1:itrj2] = 0.
            itrj1 = itrj2

        self.jump_weights = jump_weights
        assert np.all(self.jump_weights >= 0.)

    def set_jump_probs(self) -> None:
        """Set the normalized probabilities of the jump subtypes.

        Probabilities are a function of temperature, relying on set_jump_weights.
        The overall manager cannot have any rows with 0 total probability,
        """
        super().set_jump_probs()
        # individual proposals can have temps where they do not suggest proposals
        # but the overarching proposal manager must make proposals for all temps
        assert np.all(np.sum(self.jump_probs, axis=1) == 1.)

    def post_step_update(self, samples) -> None:
        """Do any needed internal processing after an individual step of all temperatures.

        Mainly intended to be used to write to e.g. differential evolution buffer.
        """
        for itrm in range(self.n_managers):
            self.managers[itrm].post_step_update(samples)

    def post_block_update(self, itrn: int, block_size: int, samples, logLs) -> None:
        """Do any needed internal processing after an individual block of size block_size.

        E.g. fisher matrix updates.
        """
        for itrm in range(self.n_managers):
            self.managers[itrm].post_block_update(itrn, block_size, samples, logLs)

    def record_config(self, config_in) -> None:
        """Record the current configuration to an input ConfigParser object config_in."""
        for itrm in range(self.n_managers):
            self.managers[itrm].record_config(config_in)

        config_in['ProposalManager']['only_prior_hot'] = self.only_prior_hot
