"""C 2023 Matthew C. Digman
Abstract class for the interface a proposal manager must export
in order to be properly recognized by the framework
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeVar, final, runtime_checkable

import numpy as np
from numba import njit
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence
    from configparser import ConfigParser

    from DTMCMC.likelihood import AbstractLikelihood
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder


@runtime_checkable
class AbstractJump[LikelihoodType: AbstractLikelihood[Any]](Protocol):
    """An object that performs a single proposal from its __call__ method.

    A jump may additionally define ``bind_native(likelihood_natives)``
    returning a per-class jitted function with this ``__call__``
    signature plus the manager and likelihood runtime states (see DTMCMC.numba_backend).

    A jump should also declare ``declared_internal_evals``: the fixed
    number of target-likelihood evaluations one dispatch performs
    internally (0 for every proposal that does not evaluate the
    likelihood itself). A jump without the attribute makes the sampler's
    evaluation accounting incomplete — an unknown cost is never silently
    treated as zero.
    """

    print_name: str
    declared_internal_evals: int

    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        """Generate the MCMC proposal.
        inputs:
            sample_point: a numpy array with the current point
            itrt: the index of the requested temperature chain
        outputs:
            new_point: a numpy array with the proposed new point
            density_factor: a scalar float, proposal density factor
                if no proposal density factor is needed can just be set to 0.
            success: a boolean, whether generating the proposal succeeded
        """
        ...


ManagerStateT_contra = TypeVar('ManagerStateT_contra', contravariant=True)


type NativeJumpCall[ManagerType] = Callable[
    [NDArray[np.floating], int, ManagerType], tuple[NDArray[np.floating], float, bool]
]


type NativePostStepCall[ManagerType] = Callable[[ManagerType, NDArray[np.floating]], None]


@runtime_checkable
class AbstractJumpManager[LikelihoodType: AbstractLikelihood[Any]](Protocol):
    """Structural component-manager interface used by aggregate dispatchers."""

    @property
    def like_obj(self) -> LikelihoodType:
        """Likelihood object that the jump manager creates jumps for."""
        ...

    @property
    def T_ladder(self) -> TemperatureLadder:
        """TemperatureLadder object."""
        ...

    @T_ladder.setter
    def T_ladder(self, T_ladder_in: TemperatureLadder) -> None:
        """Set the T_ladder."""
        ...

    @property
    def jumps(self) -> Sequence[AbstractJump[LikelihoodType]]:
        """Ordered jump objects exported by this manager."""
        ...

    @property
    def n_jump_types(self) -> int:
        """Number of jump objects exported by this manager."""
        ...

    @property
    def jump_probs(self) -> NDArray[np.floating]:
        """Conditional jump probabilities by temperature."""
        ...

    @property
    def jump_weights(self) -> NDArray[np.floating]:
        """Return unnormalized jump weights by temperature and jump type."""
        ...

    @jump_weights.setter
    def jump_weights(self, jump_weights_in: NDArray[np.floating]) -> None:
        """Override the default jump weights."""
        ...

    def dispatch_jump(
        self, sample_point: NDArray[np.floating], itrt: int, choose: int = -1
    ) -> tuple[NDArray[np.floating], float, bool, int]:
        """Dispatch the specified proposal."""
        ...

    @property
    def jump_labels(self) -> list[str]:
        """Return labels in the same order as ``jumps``."""
        ...

    def post_step_update(self, samples: NDArray[np.floating]) -> None:
        """Update manager state after one sampler step across all chains."""
        ...

    def post_block_update(
        self, itrn: int, block_size: int, samples: NDArray[np.floating], logLs: NDArray[np.floating]
    ) -> int | None:
        """Update manager state after one completed block.

        Returns the deterministic number of target-likelihood evaluations
        the update performed, or None when the cost cannot be declared
        (which makes the sampler's evaluation accounting incomplete).
        """
        ...

    def record_config(self, config_in: ConfigParser) -> None:
        """Record manager configuration."""
        ...


@njit()
def choose_prob_helper(jump_probs: NDArray[np.floating]) -> int:
    """Helper that picks a random integer with the given input probabilities"""
    choose_val: float = np.random.uniform(0.0, 1)
    choose_sum: float = jump_probs[0]
    choose: int = jump_probs.size - 1
    for itrp in range(1, jump_probs.size):
        if choose_val < choose_sum:
            choose = itrp - 1
            break
        choose_sum += jump_probs[itrp]
    return choose


@njit()
def _null_bind_post_step(_state: NamedTuple, _samples_row: NDArray[np.floating], /) -> None:
    """Placeholder post step binding that does nothing."""
    return


class JumpManager[LikelihoodType: AbstractLikelihood[Any], StateType: Any]:
    """Extensions of this class dispatch MCMC proposals."""

    # deterministic likelihood-evaluation cost of constructing the manager;
    # subclasses that evaluate the likelihood at construction must override
    declared_construction_evals: int = 0

    def __init__(
        self,
        T_ladder: TemperatureLadder,
        like_obj: LikelihoodType,
        jumps: list[AbstractNativeJump[LikelihoodType, StateType]],
    ) -> None:
        """Default constructor that handles all the common actions we expect to need"""
        self._T_ladder: TemperatureLadder = T_ladder
        self._like_obj: LikelihoodType = like_obj
        self.n_chain: int = T_ladder.n_chain
        self.n_par: int = like_obj.n_par

        # self.jump_names = jump_names
        self._jumps: list[AbstractNativeJump[LikelihoodType, StateType]] = jumps
        self._n_jump_types = len(jumps)

        self._jump_probs: NDArray[np.floating] = np.zeros((self.n_chain, self._n_jump_types))
        self._jump_weights: NDArray[np.floating] = np.zeros((self.n_chain, self._n_jump_types))

        # self.jump_labels_array = np.array([jump_labels_dict.get(name, name) for name in jump_names])
        self._jump_labels_array: list[str] = [jump.print_name for jump in self._jumps]

        self.name_to_idx: dict[str, int] = {}
        for itrm, name in enumerate(self._jump_labels_array):
            self.name_to_idx[name] = itrm

        self.set_jump_probs()

    @property
    @abstractmethod
    def native_state(self) -> StateType: ...

    @final
    @property
    def like_obj(self) -> LikelihoodType:
        return self._like_obj

    @property
    def T_ladder(self) -> TemperatureLadder:
        return self._T_ladder

    @T_ladder.setter
    def T_ladder(self, T_ladder_in: TemperatureLadder) -> None:
        self._T_ladder = T_ladder_in

    @final
    @property
    def jumps(self) -> list[AbstractNativeJump[LikelihoodType, StateType]]:
        return self._jumps

    @final
    @property
    def n_jump_types(self) -> int:
        return self._n_jump_types

    @final
    @property
    def jump_probs(self) -> NDArray[np.floating]:
        return self._jump_probs

    @final
    def dispatch_jump(
        self, sample_point: NDArray[np.floating], itrt: int, choose: int = -1
    ) -> tuple[NDArray[np.floating], float, bool, int]:
        """Dispatch the specified proposal
        inputs:
            sample_point: 1D float array, the parameters of the current point
            itrt: scalar integer, the index of the temperature chain for which to dispatch a proposal
            choose: scalar int, optional, an index that the dispatcher may use to select which proposal to try
                    if choose is not set, then try to jump according to the specified probability matrix

        Returns
        -------
            new_point: 1D float array, the parameter of the new point
            density_fac: a scalar float for the density factor of the proposal,
                            will be added to the log likelihood to modify the acceptance probability
            success: scalar boolean, whether or not generating the proposal succeeded
                        (if not, the proposal will automatically be marked rejected)
            choose: scalar int, index of the chosen jump type
        """
        if choose == -1:
            # choose the jump
            choose = choose_prob_helper(self._jump_probs[itrt])
        else:
            # validate the input choice if it is forced
            assert 0 <= choose < self._n_jump_types

        new_point, density_fac, success = self._jumps[choose](sample_point, itrt)
        return new_point, density_fac, success, choose

    def set_jump_weights(self) -> None:
        """Set the relative jump probabilities as a function of temperature for each jump type the manager exports
        based on a given strategy parameter object
        """
        jump_weights = np.zeros((self.n_chain, self._n_jump_types))
        # just a default equal weight
        jump_weights[:] = 0.333
        self._jump_weights = jump_weights

    def set_jump_probs(self) -> None:
        """Set the normalized probabilities of the jump subtypes
        as a function of temperature, relying on the set_jump_weights
        methods which must be provided in subclasses
        """
        # unnormalized jump weights must be provided for in a subclass
        self.set_jump_weights()
        self.normalize_jump_probs()

    @final
    def normalize_jump_probs(self) -> None:
        """Normalize the jump probabilities."""
        assert np.all(self._jump_weights >= 0.0)

        if np.any(self._jump_weights != 0.0):
            # get the normalized conditional jump probabilities
            self._jump_probs = (self._jump_weights.T / self._jump_weights.sum(axis=1)).T
            self._jump_probs[~np.isfinite(self._jump_probs)] = 0.0
        else:
            self._jump_probs = np.zeros((self.n_chain, self._n_jump_types))

        assert np.all(self._jump_probs >= 0.0)

        for itrt in range(self._jump_probs.shape[0]):
            # sanity check that all rows are either normalized  to 1 or sum to 0
            sum_check: float = float(np.sum(self._jump_probs[itrt]))
            assert sum_check in {0.0, 1.0}

    @property
    def jump_weights(self) -> NDArray[np.floating]:
        """Get the desired weights of this jump type as a function of temperature"""
        return self._jump_weights

    @jump_weights.setter
    def jump_weights(self, jump_weights_in: NDArray[np.floating]) -> None:
        """Override the default jump weights."""
        self._jump_weights = jump_weights_in
        self.normalize_jump_probs()
        # self.set_jump_probs() # TODO need to enforce jump weight normalization

    @final
    @property
    def jump_labels(self) -> list[str]:
        """Get text labels for the different jump types"""
        return self._jump_labels_array.copy()

    @final
    def post_step_update(self, samples: NDArray[np.floating]) -> None:
        """Do any needed internal processing after an individual step of all temperatures;
        mainly intended to be used to write to differential evolution buffer
        inputs:
            samples: 2D float array of samples
        """
        self.bind_native_post_step(self.native_state, samples)

    @property
    def bind_native_post_step(self) -> NativePostStepCall[StateType]:
        return _null_bind_post_step

    def post_block_update(
        self, itrn: int, block_size: int, samples: NDArray[np.floating], logLs: NDArray[np.floating]
    ) -> int | None:
        """Do any needed internal processing after an individual block of size block_size:
        ie, fisher matrix updates
        inputs:
            itrn: int, the current index of the chain state
            block_size: int, the number of steps in this block
            samples: 3D float array of samples
            logLs: 2D float array of likelihoods
        output:
            the deterministic number of target-likelihood evaluations the
            update performed (0 for the base no-op), or None when the cost
            cannot be declared
        """
        del itrn
        del block_size
        del samples
        del logLs
        return 0

    @abstractmethod
    def record_config(self, config_in: ConfigParser) -> None: ...


class AbstractNativeJump[LikelihoodType: AbstractLikelihood[Any], StateType: Any](ABC):
    declared_internal_evals: int
    handle: NativeJumpCall[StateType]
    manager: JumpManager[LikelihoodType, StateType]
    print_name: str

    def __init__(
        self, handle: NativeJumpCall[StateType], manager: JumpManager[LikelihoodType, StateType], print_name: str
    ) -> None:
        self.handle = handle
        self.manager = manager
        self.print_name = print_name

    @final
    def __call__(self, sample_point: NDArray[np.floating], itrt: int) -> tuple[NDArray[np.floating], float, bool]:
        return self.handle(sample_point, itrt, self.manager.native_state)
