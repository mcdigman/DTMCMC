"""C 2023 Matthew C. Digman
manager to manage prior-draw based jumps
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, override
from warnings import warn

import numba.core.types as nb_types
import numpy as np
from numba import typeof
from numpy.typing import NDArray  # noqa: TC002

from DTMCMC.jump_manager import AbstractNativeJump, JumpManager, NativeJumpCall
from DTMCMC.likelihood import CompilationFallbackWarning, compile_handle

if TYPE_CHECKING:
    from configparser import ConfigParser

    from DTMCMC.likelihood import AbstractLikelihood, PriorProposalFn
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder


class PriorNativeState(NamedTuple):
    """Runtime state bundle for the prior proposal jumps."""


# composition memo keyed on the likelihood's per-class functions: repeated
# binds return the same jitted function, so structurally identical samplers
# share one compiled program (holding the keys alive keeps ids stable)
_PRIOR_NATIVE_MEMO: dict[tuple[object], tuple[NativeJumpCall[PriorNativeState], str | None]] = {}

_JUMP_ARGS: tuple[nb_types.Type, ...] = (
    nb_types.Array(nb_types.float64, 1, 'C'),  # type: ignore[no-untyped-call]
    nb_types.int64,
    typeof(PriorNativeState()),  # type: ignore[no-untyped-call]
)


def _make_prior_native(
    prior_proposal: PriorProposalFn,
) -> tuple[NativeJumpCall[PriorNativeState], str | None]:
    """Compose the likelihood's native prior draw and density into a jump."""
    key = (prior_proposal,)
    got = _PRIOR_NATIVE_MEMO.get(key)
    if got is not None:
        return got

    # @njit(inline='always')
    def prior_native(
        sample_point: NDArray[np.floating], _itrt: int, _manager_state: PriorNativeState, /
    ) -> tuple[NDArray[np.floating], float, bool]:
        return prior_proposal(sample_point)

    got = compile_handle(prior_native, _JUMP_ARGS)

    _PRIOR_NATIVE_MEMO[key] = got
    return got


class PriorFullJump[LikelihoodType: AbstractLikelihood[NamedTuple]](
    AbstractNativeJump[LikelihoodType, PriorNativeState]
):
    def __init__(self, manager: JumpManager[LikelihoodType, PriorNativeState]) -> None:
        print_name = 'Prior All-D'

        handle, failure = _make_prior_native(manager.like_obj.prior_proposal_fn_baked)
        if failure is not None:
            warn(
                f'{print_name} failed nopython compilation and will run as plain Python:\n{failure}',
                CompilationFallbackWarning,
                stacklevel=2,
            )
        super().__init__(handle, manager, print_name)

    @property
    @override
    def declared_internal_evals(self) -> int:
        return 0


@dataclass(init=False)
class PriorStrategyParameters:
    """container to store some parameters related to the strategy of proposal generation"""

    cold_prior_weight: float
    hot_prior_target_weight: float

    def __init__(self, config: ConfigParser) -> None:
        """Initialize the object with the prescribed parameters"""
        config_p = config['PriorManager']
        # how often to do prior draws in the cold chains
        self.cold_prior_weight = config_p.getfloat('cold_prior_weight', 0.333)
        # how often to do prior draws in the hottest finite temperature chain
        self.hot_prior_target_weight = config_p.getfloat('hot_prior_target_weight', 0.333)

    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to the requested configuration object
        inputs:
            config_in: ConfigParser object
        """
        config_prior = config_in['PriorManager']
        config_prior['cold_prior_weight'] = str(self.cold_prior_weight)
        config_prior['hot_prior_target_weight'] = str(self.hot_prior_target_weight)


class PriorManager[LikelihoodType: AbstractLikelihood[Any]](JumpManager[LikelihoodType, PriorNativeState]):
    """manage prior draw-based jumps, subclass of DTMCMC.jump_manager.JumpManager"""

    def __init__(self, T_ladder: TemperatureLadder, like_obj: LikelihoodType, config: ConfigParser) -> None:
        """Take a likelihood object and create an object that can propose prior draws"""
        self.strategy_params = PriorStrategyParameters(config)

        self._like_obj = like_obj

        jumps: list[AbstractNativeJump[LikelihoodType, PriorNativeState]] = [PriorFullJump(self)]

        self._state = PriorNativeState()

        super().__init__(T_ladder, like_obj, jumps)

    @override
    def set_jump_weights(self) -> None:
        """Set the relative probabilities of the different jump types"""
        n_cold = self.T_ladder.n_cold
        n_chain = self.T_ladder.n_chain
        jump_weights = np.zeros((n_chain, self.n_jump_types))
        # just a default equal weight
        jump_weights[:] = 0.333

        cold_prior_weight = self.strategy_params.cold_prior_weight
        hot_prior_weight = self.strategy_params.hot_prior_target_weight

        idx_prior_full = -1
        for itrp, jump in enumerate(self.jumps):
            if isinstance(jump, PriorFullJump):
                idx_prior_full = itrp

        assert idx_prior_full >= -1

        jump_weights[n_chain - 1, :] = 0.0
        jump_weights[n_chain - 1, idx_prior_full] = 1.0
        jump_weights[n_cold : n_chain - 1, idx_prior_full] = np.linspace(
            cold_prior_weight, hot_prior_weight, n_chain - n_cold
        )[1:]
        jump_weights[:n_cold, idx_prior_full] = cold_prior_weight

        self._jump_weights = jump_weights
        assert np.all(self._jump_weights >= 0.0)

    @override
    def record_config(self, config_in: ConfigParser) -> None:
        """Record the current configuration to an input ConfigParser object config_in"""
        self.strategy_params.record_config(config_in)

    @property
    @override
    def native_state(self) -> PriorNativeState:
        return self._state
