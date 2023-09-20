"""C 2023 Matthew C. Digman
get a default proposal manager object"""

import DTMCMC.fisher_manager as fm
import DTMCMC.de_manager as dm
import DTMCMC.prior_manager as pm
import DTMCMC.auxilliary_manager as am
import DTMCMC.exchange_manager as em

from DTMCMC.proposal_manager import ProposalManager


def get_default_proposal_manager(T_ladder, like_obj, strategy_params, starting_samples,
                                 fisher_manager_loc=None, de_manager_loc=None, auxilliary_manager_loc=None, prior_manager_loc=None,
                                 exchange_manager_loc=None):
    """get a default proposal manager object, or allow any individual part
    of the default fisher_manager_loc, de_manager_loc, prior_manager_loc to be replaced separately
    auxilliary_manager_loc is a blank template manager to make it easy to substitute in a new manager type"""

    if fisher_manager_loc is None:
        fisher_manager_loc = fm.FisherJumpManager(T_ladder, strategy_params, like_obj, starting_samples)

    if de_manager_loc is None:
        de_manager_loc = dm.DEJumpManager(T_ladder, strategy_params, like_obj)

    if auxilliary_manager_loc is None:
        auxilliary_manager_loc = am.AuxilliaryJumpManager(T_ladder, strategy_params, like_obj)

    if prior_manager_loc is None:
        prior_manager_loc = pm.PriorManager(T_ladder, strategy_params, like_obj)

    if exchange_manager_loc is None:
        exchange_manager_loc = em.ExchangeManager(em.SEQUENTIAL_TARGETS,track_full_exchanges=False)

    managers = (fisher_manager_loc, de_manager_loc, prior_manager_loc)
    proposal_manager = ProposalManager(T_ladder, strategy_params, managers, exchange_manager_loc)
    return proposal_manager
