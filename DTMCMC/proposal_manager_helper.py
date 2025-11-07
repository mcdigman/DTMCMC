"""C 2023 Matthew C. Digman
get a default proposal manager object
"""

from __future__ import annotations

import configparser
from typing import TYPE_CHECKING

import numpy as np

import DTMCMC.exchange_manager as em
from DTMCMC.auxilliary_manager import AuxilliaryJumpManager
from DTMCMC.de_manager import DEJumpManager
from DTMCMC.exchange_manager import ExchangeManager
from DTMCMC.fisher_manager import FisherJumpManager
from DTMCMC.prior_manager import PriorManager
from DTMCMC.proposal_manager import ProposalManager

if TYPE_CHECKING:
    from DTMCMC.jump_manager import JumpManager
    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder


def get_default_proposal_manager(
    T_ladder: TemperatureLadder,
    like_obj: AbstractLikelihood,
    starting_samples=None,
    config=None,
    fisher_manager_loc: FisherJumpManager | None = None,
    de_manager_loc: DEJumpManager | None = None,
    auxilliary_manager_loc: AuxilliaryJumpManager | None = None,
    prior_manager_loc: PriorManager | None = None,
    exchange_manager_loc: ExchangeManager | None = None,
) -> ProposalManager:
    """Get a default proposal manager object, or allow any individual part
    of the default fisher_manager_loc, de_manager_loc, prior_manager_loc to be replaced separately
    auxilliary_manager_loc is a blank template manager to make it easy to substitute in a new manager type
    """
    if config is None:
        config = configparser.ConfigParser()
        config.read('default_config.ini')

    if starting_samples is None:
        starting_samples = np.zeros((T_ladder.n_chain, like_obj.n_par))
        for itrt in range(T_ladder.n_chain):
            starting_samples[itrt] = like_obj.prior_draw()

    if fisher_manager_loc is None:
        fisher_manager_loc = FisherJumpManager(T_ladder, like_obj, starting_samples, config)

    if de_manager_loc is None:
        de_manager_loc = DEJumpManager(T_ladder, like_obj, config)

    if auxilliary_manager_loc is None:
        auxilliary_manager_loc = AuxilliaryJumpManager(T_ladder, like_obj, config)

    if prior_manager_loc is None:
        prior_manager_loc = PriorManager(T_ladder, like_obj, config)

    if exchange_manager_loc is None:
        exchange_manager_loc = ExchangeManager(em.SEQUENTIAL_TARGETS, track_full_exchanges=False)

    managers: tuple[JumpManager, ...] = (fisher_manager_loc, de_manager_loc, auxilliary_manager_loc, prior_manager_loc)
    return ProposalManager(T_ladder, like_obj, managers, exchange_manager_loc, config)
