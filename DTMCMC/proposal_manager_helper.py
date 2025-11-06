"""C 2023 Matthew C. Digman
get a default proposal manager object
"""

import configparser

import numpy as np

import DTMCMC.auxilliary_manager as am
import DTMCMC.de_manager as dm
import DTMCMC.exchange_manager as em
import DTMCMC.fisher_manager as fm
import DTMCMC.prior_manager as pm
from DTMCMC.proposal_manager import ProposalManager


def get_default_proposal_manager(T_ladder, like_obj, starting_samples=None, config=None,
                                 fisher_manager_loc=None, de_manager_loc=None, auxilliary_manager_loc=None, prior_manager_loc=None,
                                 exchange_manager_loc=None):
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
            starting_samples[itrt] = like_obj.prior_draw

    if fisher_manager_loc is None:
        fisher_manager_loc = fm.FisherJumpManager(T_ladder, like_obj, starting_samples, config)

    if de_manager_loc is None:
        de_manager_loc = dm.DEJumpManager(T_ladder, like_obj, config)

    if auxilliary_manager_loc is None:
        auxilliary_manager_loc = am.AuxilliaryJumpManager(T_ladder, like_obj, config)

    if prior_manager_loc is None:
        prior_manager_loc = pm.PriorManager(T_ladder, like_obj, config)

    if exchange_manager_loc is None:
        exchange_manager_loc = em.ExchangeManager(em.SEQUENTIAL_TARGETS, track_full_exchanges=False)

    managers = (fisher_manager_loc, de_manager_loc, auxilliary_manager_loc, prior_manager_loc)
    proposal_manager = ProposalManager(T_ladder, like_obj, managers, exchange_manager_loc, config)
    return proposal_manager
