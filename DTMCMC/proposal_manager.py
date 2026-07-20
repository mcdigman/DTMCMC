"""
Manager object to handle all dispatching of proposals.

C 2023 Matthew C. Digman
"""

from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

import numpy as np

import DTMCMC.prior_manager as ph
from DTMCMC.jump_manager import AbstractJump, AbstractJumpManager, JumpManager

if TYPE_CHECKING:
    from configparser import ConfigParser

    from numpy.typing import NDArray

    from DTMCMC.exchange_manager import AbstractExchangeManager
    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder


@runtime_checkable
class AbstractProposalManager[LikelihoodType: AbstractLikelihood = AbstractLikelihood](
    AbstractJumpManager[LikelihoodType], Protocol
):
    """Structural aggregate proposal interface required by sampler kernels."""

    @property
    def managers(self) -> tuple[AbstractJumpManager[LikelihoodType], ...]:
        """Ordered component proposal managers."""
        ...

    @property
    def exchange_manager(self) -> AbstractExchangeManager:
        """Exchange scheduler and executor."""
        ...


class ProposalManager[LikelihoodType: AbstractLikelihood = AbstractLikelihood](
    JumpManager[LikelihoodType], AbstractProposalManager[LikelihoodType]
):
    """Manage generation of proposals, handles all dispatching of jumps."""

    def __init__(
        self,
        T_ladder: TemperatureLadder,
        like_obj: LikelihoodType,
        managers: tuple[AbstractJumpManager[LikelihoodType], ...],
        exchange_manager: AbstractExchangeManager,
        config: ConfigParser,
    ) -> None:
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
        self._only_prior_hot: bool = config['ProposalManager'].getboolean('only_prior_hot', True)

        # self.T_ladder = T_ladder

        self._managers: tuple[AbstractJumpManager[LikelihoodType], ...] = managers
        self._n_managers: int = len(self._managers)
        self._n_jumps_managers: NDArray[np.int64] = np.zeros(self._n_managers, dtype=np.int64)

        self._exchange_manager: AbstractExchangeManager = exchange_manager

        jump_labels_temp: list[str] = []
        jumps_temp: list[AbstractJump[LikelihoodType]] = []
        for itrm, manager in enumerate(self._managers):
            jump_labels_loc: list[str] = manager.jump_labels
            self._n_jumps_managers[itrm] = len(manager.jumps)
            jump_labels_temp.extend(jump_labels_loc)
            jumps_temp.extend(manager.jumps)

        self._jumps: list[AbstractJump[LikelihoodType]] = jumps_temp
        self._jump_labels_array: list[str] = jump_labels_temp
        self._n_jump_types: int = int(np.sum(self._n_jumps_managers))

        self._choose_idx_modifiers: NDArray[np.int64] = np.zeros(self._n_managers, dtype=np.int64)
        if self._n_managers > 1:
            # If there is more than one manager we will need to adjust the jump indexes when dispatching
            for itrm in range(1, self._n_managers):
                self._choose_idx_modifiers[itrm:] += self._n_jumps_managers[itrm - 1]

        super().__init__(T_ladder, like_obj, self._jumps)

    @property
    @override
    def managers(self) -> tuple[AbstractJumpManager[LikelihoodType], ...]:
        return self._managers

    @property
    @override
    def exchange_manager(self) -> AbstractExchangeManager:
        return self._exchange_manager

    @override
    def set_jump_weights(self) -> None:
        """Set the jump probabilities for everything combined."""
        n_chain: int = self._T_ladder.n_chain
        jump_weights: NDArray[np.floating] = np.zeros((n_chain, self._n_jump_types))

        itrj1: int = 0
        for itrm in range(self._n_managers):
            itrj2: int = itrj1 + self._n_jumps_managers[itrm]
            jump_weights[:, itrj1:itrj2] = self._managers[itrm].jump_weights
            if self._only_prior_hot and not isinstance(self._managers[itrm], ph.PriorManager):
                # override and only allow prior-type draws to contribute to the last chain
                jump_weights[n_chain - 1, itrj1:itrj2] = 0.0
            itrj1 = itrj2

        self._jump_weights = jump_weights
        assert np.all(self._jump_weights >= 0.0)

    @override
    def set_jump_probs(self) -> None:
        """Set the normalized probabilities of the jump subtypes.

        Probabilities are a function of temperature, relying on set_jump_weights.
        The overall manager cannot have any rows with 0 total probability,
        """
        super().set_jump_probs()
        # individual proposals can have temps where they do not suggest proposals
        # but the overarching proposal manager must make proposals for all temps
        assert np.all(np.sum(self._jump_probs, axis=1) == 1.0)

    @override
    def post_step_update(self, samples: NDArray[np.floating]) -> None:
        """Do any needed internal processing after an individual step of all temperatures.

        Mainly intended to be used to write to e.g. differential evolution buffer.
        """
        for itrm in range(self._n_managers):
            self._managers[itrm].post_step_update(samples)

    @override
    def post_block_update(
        self, itrn: int, block_size: int, samples: NDArray[np.floating], logLs: NDArray[np.floating]
    ) -> int | None:
        """Do any needed internal processing after an individual block of size block_size.

        E.g. fisher matrix updates. Returns the summed deterministic
        likelihood-evaluation cost, or None when any component manager
        cannot declare its cost (every manager is still updated).
        """
        n_evals: int | None = 0
        for itrm in range(self._n_managers):
            got = self._managers[itrm].post_block_update(itrn, block_size, samples, logLs)
            if got is None or n_evals is None:
                n_evals = None
            else:
                n_evals += got
        return n_evals

    @override
    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to an input ConfigParser object config_in."""
        for itrm in range(self._n_managers):
            self._managers[itrm].record_config(config_in)

        config_in['ProposalManager']['only_prior_hot'] = str(self._only_prior_hot)
