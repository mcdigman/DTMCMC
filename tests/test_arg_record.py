"""Tests for readout-chain recording via record indices.

The static n_record prefix convention could not express T_min < T_cold
ladders, where the readout (T = T_cold) chains are interior rungs of the
sorted ladder. Recording is now index-based: the ladder locates its
readout chains (get_arg_cold), the sampler stores those plus the
arg_record extras, recomputes the set at every ladder update, and the
artifact carries the column-to-chain map per iteration range.
"""

import tomllib
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
import pytest

from DTMCMC.dtmcmc_sampler import DTMCMCSampler, store_sample_helper
from DTMCMC.likelihoods.normal_nd import GaussianLikelihood
from DTMCMC.rng_helpers import reset_seed_guard_for_tests, seed_run
from DTMCMC.temperature_ladder_helpers import (
    EntropyTemperatureLadder,
    GeometricTemperatureLadder,
    TemperatureLadder,
)
from experiments.harness.artifact import validate
from experiments.harness.runner import run_from_spec
from experiments.harness.spec import RunSpec, SpecError

if TYPE_CHECKING:
    from numpy.typing import NDArray

TINY_ARG_RECORD_SPEC: dict[str, Any] = {
    'name': 'tiny_arg_record_test',
    'seed': 42,
    'likelihood': {'name': 'gaussian', 'n_par': 3, 'cutoff': 5},
    'ladder': {'kind': 'geometric', 'n_chain': 6, 'n_cold': 1, 'T_cold': 1.0, 'T_min': 1.0, 'T_max': 100.0, 'n_inf_final': 1},
    'run': {'n_steps': 256, 'block_size': 64, 'store_thin': 1, 'arg_record': [0, 5], 'checkpoint_every_blocks': 2},
    'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
    'proposals': {
        'FisherJumpManager': {'verbose_fisher': False},
        'DEJumpManager': {'de_size': 256},
    },
}


@pytest.fixture
def fresh_seed_guard():
    """Allow one seed_run call in a test that legitimately reseeds."""
    reset_seed_guard_for_tests()
    yield
    reset_seed_guard_for_tests()


def test_arg_cold_positional_without_t_cold() -> None:
    """A raw ladder without T_cold keeps the historical first-n_cold convention."""
    ladder = TemperatureLadder(np.array([1.0, 1.0, 4.0, np.inf]), n_cold=2)
    assert ladder.T_cold == 1.0
    assert np.array_equal(ladder.get_arg_cold(), [0, 1])


def test_arg_cold_geometric_t_min_below_t_cold() -> None:
    """With T_min < T_cold the readout chains are interior sorted-ladder rungs."""
    ladder = GeometricTemperatureLadder(n_chain=8, T_cold=1.0, T_min=0.9, T_max=100.0, n_inf_final=1, n_cold=2)
    arg_cold = ladder.get_arg_cold()
    assert arg_cold.size == 2
    assert np.all(ladder.Ts[arg_cold] == 1.0)
    # the sub-cold rung sorts ahead of the readout chains
    assert ladder.Ts[0] < 1.0
    assert arg_cold[0] > 0


def test_arg_cold_geometric_default_matches_positional() -> None:
    """With T_min == T_cold the readout chains are the first n_cold rungs."""
    ladder = GeometricTemperatureLadder(n_chain=6, n_cold=1, T_cold=1.0, T_min=1.0, T_max=100.0, n_inf_final=1)
    assert np.array_equal(ladder.get_arg_cold(), [0])


def test_arg_cold_entropy_ladder_with_subcold_inputs() -> None:
    """Spaced ladders snap a rung to exactly T_cold even with sub-cold inputs."""
    Ts_in = np.array([0.8, 1.5, 3.0, 10.0, 40.0])
    vars_in = np.array([6.0, 5.0, 4.0, 2.0, 1.0])
    ladder = EntropyTemperatureLadder(10, Ts_in, vars_in, n_cold=1, T_cold=1.0, n_inf_final=1)
    arg_cold = ladder.get_arg_cold()
    assert arg_cold.size == 1
    assert ladder.Ts[arg_cold[0]] == 1.0
    # the ladder really extends below the readout temperature
    assert np.min(ladder.Ts) < 1.0


def test_store_sample_helper_maps_columns_with_duplicates() -> None:
    """Store column j holds chain record_indices[j], duplicates included."""
    block_size = 4
    n_chain = 5
    n_par = 2
    record_indices = [3, 0, 3]
    samples_block = np.arange((block_size + 1) * n_chain * n_par, dtype=np.float64).reshape(block_size + 1, n_chain, n_par)
    logLs_block = np.arange((block_size + 1) * n_chain, dtype=np.float64).reshape(block_size + 1, n_chain)
    samples_store = np.zeros((block_size, len(record_indices), n_par))
    logLs_store = np.zeros((block_size, len(record_indices)))

    store_idx, store_counter = store_sample_helper(
        samples_store, logLs_store, samples_block, logLs_block, 0, 0, np.asarray(record_indices), block_size, 1, 1,
    )
    assert (store_idx, store_counter) == (0, 0)
    for row, itrk in enumerate(range(1, block_size + 1)):
        assert np.array_equal(samples_store[row], samples_block[itrk][record_indices])
        assert np.array_equal(logLs_store[row], logLs_block[itrk][record_indices])
    assert np.array_equal(samples_store[:, 0], samples_store[:, 2])


@pytest.mark.usefixtures('fresh_seed_guard')
def test_sampler_records_readout_plus_extras_and_tracks_updates() -> None:
    """The sampler stores readout chains first, extras after, and re-locates
    the readout chains when a ladder update moves them.
    """
    seed_run(987)
    ladder = GeometricTemperatureLadder(n_chain=8, n_cold=2, T_cold=1.0, T_min=0.9, T_max=100.0, n_inf_final=1)
    like_obj = GaussianLikelihood(n_par=3, cutoff=5)
    sampler = DTMCMCSampler(ladder, like_obj, 32, 64, arg_record=[0, 7, 1])

    arg_cold = ladder.get_arg_cold()
    assert np.array_equal(sampler.record_indices, np.concatenate([arg_cold, [0, 7, 1]]))
    assert sampler.samples_store.shape == (64, 5, 3)
    assert len(sampler.record_history) == 0

    for _ in range(2):
        sampler.advance_block()

    # the last stored row is the final chain state at the recorded indices
    assert np.allclose(sampler.logLs_store[63], sampler.logLs[0, sampler.record_indices])
    assert np.allclose(sampler.samples_store[63], sampler.samples[0, sampler.record_indices])
    # columns 0 and 4 both hold chain 1 (readout copy and requested extra)
    assert sampler.record_indices[0] == sampler.record_indices[4] == 1
    assert np.allclose(sampler.samples_store[:, 4], sampler.samples_store[:, 0])

    # a ladder update that adds a second sub-cold rung shifts the readout indices
    new_ladder = TemperatureLadder(np.array([0.85, 0.9, 1.0, 1.0, 2.0, 5.0, 20.0, np.inf]), T_cold=1.0, n_cold=2)
    old_indices = sampler.record_indices.copy()
    sampler.apply_ladder_update(new_ladder)
    assert np.array_equal(sampler.record_indices, np.concatenate([[2, 3], [0, 7, 1]]))
    assert not np.array_equal(sampler.record_indices, old_indices)
    assert len(sampler.record_history) == 2
    # storage width is invariant across updates
    assert len(sampler.record_indices) == len(old_indices)


@pytest.mark.usefixtures('fresh_seed_guard')
def test_sampler_accepts_tuple_arg_record_at_runtime() -> None:
    """Tuple inputs are runtime-permitted even though the typed API is list-only."""
    seed_run(989)
    ladder = GeometricTemperatureLadder(n_chain=6, n_cold=1, T_cold=1.0, T_min=1.0, T_max=100.0, n_inf_final=1)
    like_obj = GaussianLikelihood(n_par=3, cutoff=5)
    arg_record: tuple[int, ...] = (0, 5)

    sampler = DTMCMCSampler(ladder, like_obj, 32, 64, arg_record=arg_record)  # type: ignore[arg-type]

    assert sampler.arg_record == [0, 5]
    assert np.array_equal(sampler.record_indices, [0, 0, 5])
    assert sampler.samples_store.shape == (64, 3, 3)


@pytest.mark.usefixtures('fresh_seed_guard')
def test_sampler_accepts_int64_ndarray_arg_record_at_runtime() -> None:
    """NDArray inputs are runtime-permitted even though the typed API is list-only."""
    seed_run(990)
    ladder = GeometricTemperatureLadder(n_chain=6, n_cold=1, T_cold=1.0, T_min=1.0, T_max=100.0, n_inf_final=1)
    like_obj = GaussianLikelihood(n_par=3, cutoff=5)
    arg_record: NDArray[np.int64] = np.array([0, 5], dtype=np.int64)

    sampler = DTMCMCSampler(ladder, like_obj, 32, 64, arg_record=arg_record)  # type: ignore[arg-type]

    assert sampler.arg_record == [0, 5]
    assert np.array_equal(sampler.record_indices, [0, 0, 5])
    assert sampler.samples_store.shape == (64, 3, 3)


@pytest.mark.usefixtures('fresh_seed_guard')
def test_identical_ladder_update_keeps_record_history() -> None:
    """An update that does not move the readout chains appends no history row."""
    seed_run(988)
    ladder = GeometricTemperatureLadder(n_chain=6, n_cold=1, T_cold=1.0, T_min=1.0, T_max=100.0, n_inf_final=1)
    like_obj = GaussianLikelihood(n_par=3, cutoff=5)
    sampler = DTMCMCSampler(ladder, like_obj, 32, 32)
    sampler.advance_block()
    sampler.apply_ladder_update(TemperatureLadder(ladder.Ts.copy(), T_cold=1.0, n_cold=1))
    assert len(sampler.record_history) == 1


def test_spec_arg_record_roundtrip_and_validation() -> None:
    """arg_record round-trips through TOML; bad entries fail loudly."""
    spec = RunSpec.from_dict(dict(TINY_ARG_RECORD_SPEC))
    assert spec.arg_record == [0, 5]
    round_tripped = RunSpec.from_dict(tomllib.loads(spec.to_toml_text()))
    assert round_tripped == spec

    for bad_value in ([6], [-1], [1.5], [True], 3):
        data: dict[str, Any] = {key: dict(value) if isinstance(value, dict) else value for key, value in TINY_ARG_RECORD_SPEC.items()}
        data['run'] = dict(data['run'])
        data['run']['arg_record'] = bad_value
        with pytest.raises(SpecError, match='arg_record'):
            RunSpec.from_dict(data)


def test_spec_rejects_legacy_n_record() -> None:
    """A legacy run.n_record spec fails loudly instead of silently dropping it."""
    data: dict[str, Any] = {key: dict(value) if isinstance(value, dict) else value for key, value in TINY_ARG_RECORD_SPEC.items()}
    data['run'] = dict(data['run'])
    del data['run']['arg_record']
    data['run']['n_record'] = 4
    with pytest.raises(SpecError, match='n_record'):
        RunSpec.from_dict(data)


def test_spec_rejects_unknown_run_key() -> None:
    """An unknown [run] key fails loudly rather than being silently ignored."""
    data: dict[str, Any] = {key: dict(value) if isinstance(value, dict) else value for key, value in TINY_ARG_RECORD_SPEC.items()}
    data['run'] = dict(data['run'])
    data['run']['n_steps_typo'] = 128
    with pytest.raises(SpecError, match='unknown \\[run\\] keys'):
        RunSpec.from_dict(data)


def test_spec_arg_record_defaults_empty() -> None:
    """Omitting arg_record records only the readout chains."""
    data: dict[str, Any] = {key: dict(value) if isinstance(value, dict) else value for key, value in TINY_ARG_RECORD_SPEC.items()}
    data['run'] = dict(data['run'])
    del data['run']['arg_record']
    spec = RunSpec.from_dict(data)
    assert spec.arg_record == []


@pytest.mark.usefixtures('fresh_seed_guard')
def test_artifact_records_indices_end_to_end(tmp_path) -> None:
    """A harness run stores the recorded columns and their chain map."""
    spec = RunSpec.from_dict(dict(TINY_ARG_RECORD_SPEC))
    artifact_path = run_from_spec(spec, tmp_path)
    assert validate(artifact_path, mode='complete') == []

    with h5py.File(str(artifact_path), 'r') as hf:
        record_indices = np.asarray(hf['store/record_indices'])
        history_indices = np.asarray(hf['store/record_history_indices'])
        logLs = np.asarray(hf['store/logLs'])
        samples = np.asarray(hf['store/samples'])

    # readout chain 0 first, then the requested extras (0 duplicated, 5 hot)
    assert np.array_equal(record_indices, [0, 0, 5])
    for itr in range(history_indices.shape[0]):
        assert np.array_equal(history_indices[itr], [0, 0, 5])
    assert logLs.shape[1] == 3
    assert samples.shape[1] == 3
    # the duplicate columns are identical; the hot column is not
    assert np.array_equal(logLs[:, 0], logLs[:, 1])
    assert not np.array_equal(logLs[:, 0], logLs[:, 2])
