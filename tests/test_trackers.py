"""Phase 2 acceptance 2: cycle/round-trip/flow/ESD trackers on synthetic data.

Every scenario feeds hand-constructed chain_track sequences through the
real TrackerManager and asserts hand-computed cycle counts, round-trip
events, and flow counts — including n_cold > 1 and the duplicate-
temperature (hot-set, S5/E10 prerequisite) case, which documents that
only slot -1 counts as the hot extreme today.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import DTMCMC.exchange_manager as em
from DTMCMC.dtmcmc_sampler import mcmc_decision_helper
from DTMCMC.rng_helpers import reset_seed_guard_for_tests, seed_run
from DTMCMC.tracker_manager import RT_ARRIVED_COLD, RT_ARRIVED_HOT, TrackerManager
from experiments.harness.runner import build_sampler
from experiments.harness.spec import RunSpec
from tests.test_harness import TINY_GAUSSIAN_SPEC


def make_tracker(n_cold: int, n_chain: int, block_size: int) -> TrackerManager:
    """Build a TrackerManager with trivial jump bookkeeping for synthetic feeds."""
    return TrackerManager(n_cold, n_chain, block_size, 1, False, 1, 1)


def feed_blocks(tracker: TrackerManager, blocks: list[np.ndarray]) -> None:
    """Feed synthetic chain_track blocks (each (block_size+1, n_chain)) in sequence."""
    itrn = 0
    for chain_track in blocks:
        tracker.post_block_update(itrn, chain_track)
        itrn += tracker.block_size


def test_single_cold_round_trips() -> None:
    """n_cold=1: walker 0 shuttles cold<->hot; every count hand-computed."""
    # rows are the chain state after each iteration; row 0 is the block start
    chain_track = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],   # step 1: no moves
        [3, 1, 2, 0],   # step 2: walkers 0 and 3 swap extremes (first arrivals: no events)
        [0, 1, 2, 3],   # step 3: swap back (both complete transits)
        [3, 1, 2, 0],   # step 4: again
        [0, 1, 2, 3],   # step 5: again
        [0, 1, 2, 3],   # step 6: resident (no repeat events while staying)
        [3, 1, 2, 0],   # step 7
        [0, 1, 2, 3],   # step 8
    ], dtype=np.int64)

    tracker = make_tracker(1, 4, 8)
    feed_blocks(tracker, [chain_track])

    assert_array_equal(tracker.cycle_tracker[2], [3, 0, 0, 2])  # hot->cold arrivals
    assert_array_equal(tracker.cycle_tracker[3], [2, 0, 0, 3])  # cold->hot arrivals
    assert_array_equal(tracker.n_cycles, [2, 0, 0, 2])

    expected_events = np.array([
        [0, 3, RT_ARRIVED_COLD], [3, 3, RT_ARRIVED_HOT],
        [3, 4, RT_ARRIVED_COLD], [0, 4, RT_ARRIVED_HOT],
        [0, 5, RT_ARRIVED_COLD], [3, 5, RT_ARRIVED_HOT],
        [3, 7, RT_ARRIVED_COLD], [0, 7, RT_ARRIVED_HOT],
        [0, 8, RT_ARRIVED_COLD], [3, 8, RT_ARRIVED_HOT],
    ], dtype=np.int64)
    assert_array_equal(tracker.get_rt_events(), expected_events)

    flow_up, flow_labeled = tracker.get_flow_counts()
    assert_array_equal(flow_up, [[8, 0, 0, 0]])
    assert_array_equal(flow_labeled, [[8, 0, 0, 8]])


def test_two_cold_chains() -> None:
    """n_cold=2: the second cold slot detects transits independently."""
    chain_track = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],   # step 1
        [0, 3, 2, 1],   # step 2: walker 1 to hot, walker 3 to cold slot 1 (first arrivals)
        [0, 1, 2, 3],   # step 3: both complete transits
        [0, 1, 2, 3],   # step 4
    ], dtype=np.int64)

    tracker = make_tracker(2, 4, 4)
    feed_blocks(tracker, [chain_track])

    assert_array_equal(tracker.cycle_tracker[2], [0, 1, 0, 0])
    assert_array_equal(tracker.cycle_tracker[3], [0, 0, 0, 1])
    expected_events = np.array([
        [1, 3, RT_ARRIVED_COLD], [3, 3, RT_ARRIVED_HOT],
    ], dtype=np.int64)
    assert_array_equal(tracker.get_rt_events(), expected_events)

    flow_up, flow_labeled = tracker.get_flow_counts()
    assert_array_equal(flow_up, [[4, 4, 0, 0]])
    assert_array_equal(flow_labeled, [[4, 4, 0, 4]])


def test_duplicate_temperature_hot_set_semantics() -> None:
    """S5/E10 prerequisite: only slot -1 is the hot extreme today.

    With duplicate hot temperatures (slots 2 and 3 at T_max), a walker
    visiting slot 2 and returning cold completes no cycle and logs no
    event — the documented limitation that hot-*set* semantics (plan S5)
    would lift.
    """
    chain_track = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],   # step 1
        [2, 1, 0, 3],   # step 2: walker 0 visits slot 2 (duplicate-T, not the extreme)
        [0, 1, 2, 3],   # step 3: returns cold — no completed cycle
        [0, 1, 2, 3],   # step 4
    ], dtype=np.int64)

    tracker = make_tracker(1, 4, 4)
    feed_blocks(tracker, [chain_track])

    assert_array_equal(tracker.cycle_tracker[2], [0, 0, 0, 0])
    assert_array_equal(tracker.cycle_tracker[3], [0, 0, 0, 0])
    assert tracker.get_rt_events().shape == (0, 3)


def test_multi_block_event_offsets() -> None:
    """Events in later blocks carry global iteration indices."""
    block1 = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],   # step 1
        [3, 1, 2, 0],   # step 2 (first arrivals)
        [0, 1, 2, 3],   # step 3: events at global iteration 3
        [0, 1, 2, 3],   # step 4
    ], dtype=np.int64)
    block2 = np.array([
        [0, 1, 2, 3],
        [3, 1, 2, 0],   # step 5: completed transits at global iteration 5
        [0, 1, 2, 3],   # step 6: and again at 6
        [0, 1, 2, 3],   # step 7
        [0, 1, 2, 3],   # step 8
    ], dtype=np.int64)

    tracker = make_tracker(1, 4, 4)
    feed_blocks(tracker, [block1, block2])

    expected_events = np.array([
        [0, 3, RT_ARRIVED_COLD], [3, 3, RT_ARRIVED_HOT],
        [3, 5, RT_ARRIVED_COLD], [0, 5, RT_ARRIVED_HOT],
        [0, 6, RT_ARRIVED_COLD], [3, 6, RT_ARRIVED_HOT],
    ], dtype=np.int64)
    assert_array_equal(tracker.get_rt_events(), expected_events)
    flow_up, _flow_labeled = tracker.get_flow_counts()
    assert flow_up.shape == (2, 4)


def test_esd_accumulation_direct() -> None:
    """mcmc_decision_helper accumulates |delta|^2 for proposed and accepted."""
    samples = np.zeros((2, 1, 2))
    logLs = np.zeros((2, 1))
    betas = np.ones(1)
    accept_record = np.zeros((2, 1, 1), dtype=np.int64)
    esd_record = np.zeros((2, 1, 1))

    # forced accept: enormous likelihood gain
    mcmc_decision_helper(1, samples, logLs, betas, accept_record, esd_record, 0, np.array([3., 4.]), 1.e6, 0., 0)
    assert esd_record[0, 0, 0] == pytest.approx(25.)
    assert esd_record[1, 0, 0] == pytest.approx(25.)
    assert accept_record[0, 0, 0] == 1

    # forced reject: -inf likelihood; proposed sum grows, accepted does not
    samples[1] = 0.
    mcmc_decision_helper(1, samples, logLs, betas, accept_record, esd_record, 0, np.array([0., 2.]), -np.inf, 0., 0)
    assert esd_record[0, 0, 0] == pytest.approx(29.)
    assert esd_record[1, 0, 0] == pytest.approx(25.)
    assert accept_record[1, 0, 0] == 1


def test_esd_exchange_accumulation_direct() -> None:
    """Accepted swaps accumulate |delta|^2 per slot; hand-computed answer.

    Sequential targeting on two chains with a large likelihood gap makes
    the swap certain (the Metropolis-Hastings log-ratio is +4.5, and
    log(u) <= 0 always), so both slots record the squared distance
    between the swapped states.
    """
    n_chain = 2
    samples = np.zeros((3, n_chain, 2))
    samples[1] = [[0., 0.], [3., 4.]]
    logLs = np.zeros((3, n_chain))
    logLs[1] = [0., 5.]
    betas = np.array([1., 0.1])
    chain_track = np.zeros((3, n_chain), dtype=np.int64)
    chain_track[1] = [0, 1]
    exchange_tracker = np.zeros((2, 2, n_chain), dtype=np.int64)
    esd_exchange = np.zeros(n_chain)

    em.do_ptmcmc_exchange(1, samples, logLs, n_chain, betas, exchange_tracker, esd_exchange, chain_track, em.SEQUENTIAL_TARGETS, False)

    assert_array_equal(chain_track[2], [1, 0])
    assert_array_equal(samples[2], [[3., 4.], [0., 0.]])
    assert esd_exchange[0] == pytest.approx(25.)
    assert esd_exchange[1] == pytest.approx(25.)

    # null targeting proposes nothing: no displacement accumulates
    esd_null = np.zeros(n_chain)
    em.do_ptmcmc_exchange(1, samples, logLs, n_chain, betas, exchange_tracker, esd_null, chain_track, em.NULL_TARGETS, False)
    assert_array_equal(esd_null, [0., 0.])


def test_trackers_on_real_run() -> None:
    """Integration: event log and counters agree on a real sampler run."""
    reset_seed_guard_for_tests()
    spec = RunSpec.from_dict(TINY_GAUSSIAN_SPEC)
    seed_run(spec.seed)
    sampler, _like_obj = build_sampler(spec)
    for _ in range(spec.n_blocks):
        sampler.advance_block()
    reset_seed_guard_for_tests()

    tracker = sampler.tracker_manager
    events = tracker.get_rt_events()
    n_cold_arrivals = int(np.count_nonzero(events[:, 2] == RT_ARRIVED_COLD))
    n_hot_arrivals = int(np.count_nonzero(events[:, 2] == RT_ARRIVED_HOT))
    assert n_cold_arrivals == int(tracker.cycle_tracker[2].sum())
    assert n_hot_arrivals == int(tracker.cycle_tracker[3].sum())

    # ESD is a nonnegative observer: accepted sums never exceed proposed
    assert np.all(tracker.esd_record >= 0.)
    assert np.all(tracker.esd_record[1] <= tracker.esd_record[0] + 1.e-12)
    # jump types that were proposed have positive proposed-ESD
    proposed_counts = tracker.accept_record.sum(axis=0)
    assert np.all((tracker.esd_record[0] > 0.) == (proposed_counts > 0))

    flow_up, flow_labeled = tracker.get_flow_counts()
    assert flow_up.shape == (spec.n_blocks, spec.n_chain)
    assert np.all(flow_up <= flow_labeled)
    assert np.all(flow_labeled <= spec.block_size)

    # exchange displacement accumulates exactly where swaps were accepted
    # (exchange_tracker[0, 0, t] counts accepted exchanges per slot)
    accepted_exchanges = tracker.exchange_tracker[0, 0]
    assert np.all(tracker.esd_exchange >= 0.)
    assert_array_equal(tracker.esd_exchange > 0., accepted_exchanges > 0)
