"""C 2023 Matthew C. Digman
helpers to perform the parallel tempering exchanges
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, Protocol, final, runtime_checkable

import numpy as np
from numba import njit
from numpy.typing import NDArray

if TYPE_CHECKING:
    from DTMCMC.temperature_ladder_helpers import TemperatureLadder

# TODO implement option to not do exchanges at all

RANDOM_TARGETS = 0  # uniform random exchange targetting
SEQUENTIAL_TARGETS = 1  # target sequentially from back to front
ADJACENT_TARGETS = 2  # target alternating +/- 1 positions
NULL_TARGETS = 3  # do not do any exchanges
REVERSE_SEQUENTIAL_TARGETS = 4  # target sequentially from front to back
ALTERNATE_SEQUENTIAL_TARGETS = 5  # target sequentially from front to back and back to front alternating


type NativeExchangeStepCall[ExchangeInputType] = Callable[[int, ExchangeInputType], bool]


class NativeExchangeCall[ExchangeInputType](Protocol):
    """Exchange executor over the exchange manager's inputs."""

    def __call__(
        self,
        itrb: int,
        samples: NDArray[np.floating],
        logLs: NDArray[np.floating],
        n_chain: int,
        betas: NDArray[np.floating],
        exchange_tracker: NDArray[np.int64],
        esd_exchange: NDArray[np.floating],
        chain_track: NDArray[np.int64],
        inputs: ExchangeInputType,
        /,
    ) -> None:
        """Execute one exchange step."""
        ...


class NativeExchangeFunctions[ExchangeInputType](NamedTuple):
    """Store function handles for native binding"""

    is_exchange_step: NativeExchangeStepCall[ExchangeInputType]
    exchange: NativeExchangeCall[ExchangeInputType]


class ExchangeNativeInputs(NamedTuple):
    """Compile-time inputs for the built-in native exchange functions."""

    strategy: int
    track_full_exchanges: int


@runtime_checkable
class AbstractExchangeManager[ExchangeInputType](Protocol):
    """Structural exchange-manager interface used by sampler kernels."""

    @property
    def track_full_exchanges(self) -> int: ...

    @property
    def inputs(self) -> ExchangeInputType: ...

    def do_ptmcmc_exchange(
        self,
        itrb: int,
        samples: NDArray[np.floating],
        logLs: NDArray[np.floating],
        T_ladder: TemperatureLadder,
        exchange_tracker: NDArray[np.int64],
        esd_exchange: NDArray[np.floating],
        chain_track: NDArray[np.int64],
    ) -> None:
        """Execute one scheduled exchange step."""
        ...

    def is_exchange_step(self, itrb: int) -> bool:
        """Return whether a sampler step is an exchange step."""
        ...

    @property
    def bind_native(self) -> NativeExchangeFunctions[ExchangeInputType]:
        """Return the per-class native exchange schedule and executor."""
        ...


@njit()
def exchange_step_helper(
    logLs_loc: NDArray[np.floating],
    betas: NDArray[np.floating],
    exchange_tracker: NDArray[np.int64],
    exchange_order: NDArray[np.int64],
    targets: NDArray[np.int64],
    no_repeat: bool,
    track_full_exchanges: int,
) -> NDArray[np.int64]:
    """Actually execute the swaps for an exchange step"""
    n_chain = betas.shape[0]

    itrs_fin = np.arange(0, n_chain, dtype=np.int64)
    for idxt in range(n_chain):
        itrt = exchange_order[idxt]
        itrt_target = targets[itrt]
        if no_repeat and itrt > itrt_target:
            # prevent random targetting from undoing itself
            continue
        if itrt == itrt_target:
            # not a real proposal
            continue
        if not no_repeat:
            assert itrs_fin[itrt_target] == itrt_target
        else:
            assert targets[itrt_target] == itrt
            assert itrs_fin[itrt_target] == itrt_target
            assert itrs_fin[itrt] == itrt

        log_accept_prob_exchange = np.log(np.random.uniform(0.0, 1.0))
        log_mh_ratio_exchange = betas[itrt] * (logLs_loc[itrt_target] - logLs_loc[itrt]) + betas[itrt_target] * (
            logLs_loc[itrt] - logLs_loc[itrt_target]
        )
        if log_mh_ratio_exchange > log_accept_prob_exchange:
            logLs_hold = logLs_loc[itrt_target]
            logLs_loc[itrt_target] = logLs_loc[itrt]
            logLs_loc[itrt] = logLs_hold

            itr_hold = itrs_fin[itrt]
            itrs_fin[itrt] = itrs_fin[itrt_target]
            itrs_fin[itrt_target] = itr_hold

            if track_full_exchanges:
                # track full exchange matrix
                exchange_tracker[0, itrt, itrt_target] += 1
            else:
                # track all exchanges for each individual chain
                exchange_tracker[0, 0, itrt] += 1
                exchange_tracker[0, 0, itrt_target] += 1
                # track nn exchanges
                if itrt_target == itrt + 1 or itrt_target == itrt - 1:
                    exchange_tracker[1, 0, itrt] += 1
                    exchange_tracker[1, 0, itrt_target] += 1
        elif track_full_exchanges:
            # track full exchange matrix
            exchange_tracker[1, itrt, itrt_target] += 1
        else:
            # track all exchanges for each individual chain
            exchange_tracker[0, 1, itrt] += 1
            exchange_tracker[0, 1, itrt_target] += 1
            # track nn exchanges
            if itrt_target == itrt + 1 or itrt_target == itrt - 1:
                exchange_tracker[1, 1, itrt] += 1
                exchange_tracker[1, 1, itrt_target] += 1

    return itrs_fin


@njit()
def random_pair_generate(n_chain: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Pairs are generated uniformally at random"""
    target_shuffle = np.random.permutation(np.arange(0, n_chain))
    target_shuffle = np.concatenate((target_shuffle[::2], target_shuffle[1::2]))

    targets = np.zeros(n_chain, dtype=np.int64)
    targets[target_shuffle[: n_chain // 2]] = target_shuffle[n_chain // 2 : n_chain]
    targets[target_shuffle[n_chain // 2 : n_chain]] = target_shuffle[: n_chain // 2]
    exchange_order = np.arange(0, n_chain)
    return targets, exchange_order


@njit()
def offset_pair_generate(n_chain: int, offset: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Pairs are generated as
    [(0,offset+1),(2,offset+3),...(n_chain-2,offset+n_chain-1)]%n_chain,
    e.g.,
    offset = 0 corresponds to pairs [(0,1),(2,3),...(n_chain-2,n_chain-1)]
    offset = -1 corresponds to       [(n_chain-1,0),(1,2),...(n_chain-3,n_chain-1)]
    """
    # can only handle offset pairs for integer divisors
    if offset >= 0:
        assert n_chain % (offset + 1) == 0
    else:
        assert n_chain % (np.abs(offset)) == 0
    targets = np.zeros(n_chain, dtype=np.int64) - 1
    if offset >= 0:
        ctr = (offset + 1) % n_chain
    else:
        ctr = offset % n_chain
    for itrm in range(n_chain):
        if offset >= 0:
            check = (itrm // (offset + 1) % 2) == 0
        else:
            check = ((itrm - np.abs(offset)) // (np.abs(offset)) % 2) != 0

        if check and targets[itrm] == -1:
            targets[itrm] = ctr
            targets[ctr] = itrm
        ctr += 1
        ctr %= n_chain
    exchange_order = np.arange(0, n_chain)
    return targets, exchange_order


@njit()
def do_ptmcmc_exchange(
    itrb: int,
    samples: NDArray[np.floating],
    logLs: NDArray[np.floating],
    n_chain: int,
    betas: NDArray[np.floating],
    exchange_tracker: NDArray[np.int64],
    esd_exchange: NDArray[np.floating],
    chain_track: NDArray[np.int64],
    target_select: int,
    track_full_exchanges: int,
) -> None:
    """Chose and exchange strategy and do the exchange step"""
    no_repeat = True
    if target_select == RANDOM_TARGETS:
        # random exchange pairs
        targets, exchange_order = random_pair_generate(n_chain)
    elif target_select == SEQUENTIAL_TARGETS:
        # target from back to front, results in repeated exchanges
        targets = np.arange(-1, n_chain - 1)
        exchange_order = np.arange(n_chain - 1, -1, -1)
        targets[0] = 0
        no_repeat = False
    elif target_select == ADJACENT_TARGETS:
        # alternate targeting exchanges at distance +/-1
        if itrb % 4 == 1:
            targets, exchange_order = offset_pair_generate(n_chain, 0)
        else:
            targets, exchange_order = offset_pair_generate(n_chain, -1)
    elif target_select == NULL_TARGETS:
        # do not actually propose any exchanges
        targets = np.arange(0, n_chain)
        exchange_order = np.arange(0, n_chain)
    elif target_select == REVERSE_SEQUENTIAL_TARGETS:
        # target from front to back, results in repeated exchanges
        targets = np.arange(1, n_chain + 1)
        exchange_order = np.arange(0, n_chain)
        targets[n_chain - 1] = n_chain - 1
        no_repeat = False
    elif target_select == ALTERNATE_SEQUENTIAL_TARGETS:
        if itrb % 4 == 1:
            targets = np.arange(1, n_chain + 1)
            exchange_order = np.arange(0, n_chain)
            targets[n_chain - 1] = n_chain - 1
        else:
            targets = np.arange(-1, n_chain - 1)
            exchange_order = np.arange(n_chain - 1, -1, -1)
            targets[0] = 0
        no_repeat = False
    else:
        msg = f'Unrecognized target_select: {target_select}'
        raise ValueError(msg)

    logLs_cur = np.zeros(n_chain)
    logLs_cur[:] = logLs[itrb]

    itrs_fin = exchange_step_helper(
        logLs_cur, betas, exchange_tracker, exchange_order, targets, no_repeat, track_full_exchanges
    )

    for itrt in range(n_chain):
        logLs[itrb + 1, itrt] = logLs[itrb, itrs_fin[itrt]]
        samples[itrb + 1, itrt] = samples[itrb, itrs_fin[itrt]]
        chain_track[itrb + 1, itrt] = chain_track[itrb, itrs_fin[itrt]]

        # accumulate the squared state displacement accepted swaps produce
        # at each temperature slot; near phase transitions exchange flow can
        # dominate state motion where local acceptance craters, so per-slot
        # displacement accounting needs this term alongside the per-jump-type
        # esd_record. Pure observer: no draws (D5)
        if itrs_fin[itrt] != itrt:
            delta_sq = 0.0
            for itrp in range(samples.shape[2]):
                diff = samples[itrb, itrs_fin[itrt], itrp] - samples[itrb, itrt, itrp]
                delta_sq += diff * diff
            esd_exchange[itrt] += delta_sq


@njit(inline='always')
def _exchange_is_step_inputs_native(itrb: int, _input: ExchangeNativeInputs) -> bool:
    """Default per-class native exchange cadence."""
    return itrb % 2 == 0


@njit(inline='always')
def _exchange_native(
    itrb: int,
    samples: NDArray[np.floating],
    logLs: NDArray[np.floating],
    n_chain: int,
    betas: NDArray[np.floating],
    exchange_tracker: NDArray[np.int64],
    esd_exchange: NDArray[np.floating],
    chain_track: NDArray[np.int64],
    inputs: ExchangeNativeInputs,
) -> None:
    """Per-class native exchange executor reading the strategy from the inputs bundle."""
    do_ptmcmc_exchange(
        itrb - 1,
        samples,
        logLs,
        n_chain,
        betas,
        exchange_tracker,
        esd_exchange,
        chain_track,
        inputs.strategy,
        inputs.track_full_exchanges,
    )


class ExchangeManager:
    """class to take a temperature ladder and state of a chain
    and define the strategy by which to propose exchanges
    """

    def __init__(self, strategy: int = RANDOM_TARGETS, track_full_exchanges: int = 1) -> None:
        """Select the exchange targeting strategy"""
        self._inputs: ExchangeNativeInputs = ExchangeNativeInputs(strategy, track_full_exchanges)

    @property
    def inputs(self) -> ExchangeNativeInputs:
        return self._inputs

    @property
    @final
    def track_full_exchanges(self) -> int:
        return self.inputs.track_full_exchanges

    @final
    def do_ptmcmc_exchange(
        self,
        itrb: int,
        samples: NDArray[np.floating],
        logLs: NDArray[np.floating],
        T_ladder: TemperatureLadder,
        exchange_tracker: NDArray[np.int64],
        esd_exchange: NDArray[np.floating],
        chain_track: NDArray[np.int64],
    ) -> None:
        """Do the exchange step"""
        assert self.is_exchange_step(itrb)
        self.exchange_native(
            itrb,
            samples,
            logLs,
            T_ladder.n_chain,
            T_ladder.betas,
            exchange_tracker,
            esd_exchange,
            chain_track,
            self.inputs,
        )

    @property
    def is_exchange_step_native(self) -> NativeExchangeStepCall[ExchangeNativeInputs]:
        return _exchange_is_step_inputs_native

    @property
    def exchange_native(self) -> NativeExchangeCall[ExchangeNativeInputs]:
        return _exchange_native

    @final
    def is_exchange_step(self, itrb: int) -> bool:
        """Check whether the step with the given index should be an exchange,
        currently based on alternating even and odd
        """
        return self.is_exchange_step_native(itrb, self.inputs)

    @final
    @property
    def bind_native(self) -> NativeExchangeFunctions[ExchangeNativeInputs]:
        """Return the per-class native exchange schedule and executor."""
        return NativeExchangeFunctions(is_exchange_step=self.is_exchange_step_native, exchange=self.exchange_native)
