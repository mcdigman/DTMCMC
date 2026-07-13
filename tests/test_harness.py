"""Phase 1 acceptance tests for the experiment harness (methods-paper plan §4).

Covers: end-to-end tiny Gaussian run with validating artifact; bit-exact
same-seed determinism and different-seed divergence; the once-per-run seed
guard (D1); artifact provenance completeness (D2); spec TOML round-trip;
counting-proxy eval accounting; and batch sweep expansion.
"""

import shlex
import tomllib
from typing import Any

import h5py
import numpy as np
import pytest

import DTMCMC.rng_helpers as rng_helpers
from DTMCMC.rng_helpers import derive_child_seeds, get_rng, reset_seed_guard_for_tests, seed_run
from experiments.harness.artifact import collect_provenance, read_attrs, validate, write_artifact
from experiments.harness.batch import write_batch
from experiments.harness.paths import default_config_path, repo_root
from experiments.harness.runner import LADDER_BUILDERS, LIKELIHOOD_BUILDERS, build_ladder, build_sampler, run_from_spec
from experiments.harness.spec import LADDER_KINDS, LIKELIHOOD_NAMES, RunSpec, SpecError, dumps_toml

TINY_GAUSSIAN_SPEC: dict[str, Any] = {
    'name': 'tiny_gaussian_test',
    'seed': 42,
    'likelihood': {'name': 'gaussian', 'n_par': 3, 'cutoff': 5},
    'ladder': {'kind': 'geometric', 'n_chain': 6, 'n_cold': 1, 'T_cold': 1.0, 'T_min': 1.0, 'T_max': 100.0, 'n_inf_final': 1},
    'run': {'n_steps': 256, 'block_size': 64, 'store_thin': 1, 'n_record': -1, 'checkpoint_every_blocks': 2},
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


def make_tiny_spec(**run_overrides) -> RunSpec:
    """Build the tiny Gaussian test spec, optionally overriding [run] entries."""
    data = dict(TINY_GAUSSIAN_SPEC)
    run_table = dict(TINY_GAUSSIAN_SPEC['run'])
    run_table.update(run_overrides)
    data['run'] = run_table
    return RunSpec.from_dict(data)


def test_rng_child_seeds_deterministic() -> None:
    """Child-seed derivation is a pure function of the run seed."""
    assert derive_child_seeds(1234) == derive_child_seeds(1234)
    assert derive_child_seeds(1234) != derive_child_seeds(1235)
    rng = get_rng(7)
    assert isinstance(rng, np.random.Generator)


@pytest.mark.usefixtures('fresh_seed_guard')
def test_seed_run_guard() -> None:
    """A second seed_run call within one run raises (plan D1)."""
    children = seed_run(1234)
    assert children == derive_child_seeds(1234)
    with pytest.raises(RuntimeError, match='once per run'):
        seed_run(1234)


def test_spec_toml_roundtrip_example_file() -> None:
    """The checked-in example spec parses, and its TOML round-trip is exact."""
    spec = RunSpec.from_toml(repo_root() / 'experiments' / 'specs' / 'tiny_gaussian.toml')
    assert spec.likelihood_name == 'gaussian'
    round_tripped = RunSpec.from_dict(tomllib.loads(spec.to_toml_text()))
    assert round_tripped == spec


def test_dumps_toml_roundtrip_tricky_values() -> None:
    """The minimal TOML emitter round-trips awkward scalars through tomllib."""
    data: dict[str, object] = {
        'a_str': "it's got 'quotes' and \\slashes\\ and \"doubles\"",
        'big_float': 1.0e15,
        'small_float': 1.2345678901234567e-8,
        'plain_float': 4.0,
        'neg_inf': float('-inf'),
        'pos_inf': float('inf'),
        'truthy': True,
        'ints': [1, 2, 3],
        'floats': [1.5, 2.5],
        'nested': {'deeper': {'value': 12, 'weird key': 'ok'}},
    }
    assert tomllib.loads(dumps_toml(data)) == data


@pytest.mark.parametrize(('field_path', 'bad_value', 'match'), [
    (('likelihood', 'name'), 'nonsense', 'unknown likelihood'),
    (('ladder', 'kind'), 'nonsense', 'unknown ladder kind'),
    (('ladder', 'n_cold'), 0, 'n_cold must be'),
    (('run', 'n_steps'), 100, 'multiple of'),
    (('exchange', 'strategy'), 'nonsense', 'unknown exchange strategy'),
    (('proposals', 'NoSuchManager'), {'x': 1}, 'unknown proposal section'),
    (('ladder', 'kind'), 'explicit', 'non-empty numeric ladder.Ts list'),
])
def test_spec_validation_errors(field_path, bad_value, match) -> None:
    """Malformed specs raise SpecError with a pointed message."""
    data: dict[str, Any] = {key: dict(value) if isinstance(value, dict) else value for key, value in TINY_GAUSSIAN_SPEC.items()}
    table: dict[str, Any] = data
    for key in field_path[:-1]:
        table = table[key]
    table[field_path[-1]] = bad_value
    with pytest.raises(SpecError, match=match):
        RunSpec.from_dict(data)


@pytest.mark.usefixtures('fresh_seed_guard')
def test_end_to_end_tiny_gaussian(tmp_path) -> None:
    """Acceptance 1+3: tiny spec runs end to end and produces a validating artifact."""
    spec = make_tiny_spec()
    artifact_path = run_from_spec(spec, tmp_path)

    assert artifact_path.is_file()
    assert validate(artifact_path, mode='complete') == []
    assert validate(artifact_path, mode='partial') == []

    attrs = read_attrs(artifact_path)
    assert int(np.asarray(attrs['run_seed']).item()) == spec.seed
    child_python, child_numba = derive_child_seeds(spec.seed)
    assert int(np.asarray(attrs['child_seed_python']).item()) == child_python
    assert int(np.asarray(attrs['child_seed_numba']).item()) == child_numba
    assert int(np.asarray(attrs['n_iterations']).item()) == spec.n_steps
    assert int(np.asarray(attrs['n_chain_steps']).item()) == spec.n_steps * spec.n_chain

    # the embedded spec text reproduces the resolved spec exactly
    embedded = RunSpec.from_dict(tomllib.loads(str(attrs['spec_toml'])))
    assert embedded == spec

    with h5py.File(str(artifact_path), 'r') as hf:
        logLs_ds = hf['store/logLs']
        assert isinstance(logLs_ds, h5py.Dataset)
        assert logLs_ds.shape == (spec.store_size, spec.n_cold)
        moments_ds = hf['moments/logL_means']
        assert isinstance(moments_ds, h5py.Dataset)
        assert moments_ds.shape == (spec.n_blocks, spec.n_chain)


def test_determinism_same_seed(tmp_path) -> None:
    """Acceptance 2: same spec + seed twice is bit-exact; different seed differs."""
    spec = make_tiny_spec()

    reset_seed_guard_for_tests()
    path_a = run_from_spec(spec, tmp_path / 'a')
    reset_seed_guard_for_tests()
    path_b = run_from_spec(spec, tmp_path / 'b')
    reset_seed_guard_for_tests()
    path_c = run_from_spec(spec.with_seed(43), tmp_path / 'c')
    reset_seed_guard_for_tests()

    def load(path):
        with h5py.File(str(path), 'r') as hf:
            logLs_ds = hf['store/logLs']
            samples_ds = hf['store/samples']
            assert isinstance(logLs_ds, h5py.Dataset)
            assert isinstance(samples_ds, h5py.Dataset)
            return logLs_ds[...], samples_ds[...]

    logLs_a, samples_a = load(path_a)
    logLs_b, samples_b = load(path_b)
    logLs_c, _samples_c = load(path_c)

    assert np.array_equal(logLs_a, logLs_b)
    assert np.array_equal(samples_a, samples_b)
    assert not np.array_equal(logLs_a, logLs_c)


@pytest.mark.usefixtures('fresh_seed_guard')
def test_counting_proxy_matches_artifact(tmp_path) -> None:
    """Acceptance 6: the artifact eval counter equals the proxy count.

    The tiny run exercises every current call site: initialization,
    exchange and non-exchange iterations, and Fisher refreshes (the first
    four blocks always refresh).
    """
    spec = make_tiny_spec()
    seed_children = seed_run(spec.seed)
    provenance = collect_provenance(spec.seed, *seed_children, spec_toml=spec.to_toml_text(), proposal_config_ini=spec.resolved_config_text())

    sampler, like_obj = build_sampler(spec)
    evals_after_init = like_obj.n_evals
    # initialization evaluates each starting sample once plus the Fisher
    # stencil: n_chain * (1 + 2 * n_par) evaluations at minimum
    assert evals_after_init >= spec.n_chain * (1 + 1 + 2 * 3)

    sampler.advance_N_blocks(spec.n_blocks)

    artifact_path = tmp_path / 'counting.h5'
    write_artifact(artifact_path, spec, sampler, like_obj.n_evals, provenance, finalized=True, wall_seconds=0.0)

    attrs = read_attrs(artifact_path)
    assert int(np.asarray(attrs['n_likelihood_evals']).item()) == like_obj.n_evals

    # exchange iterations evaluate nothing, so evals stay strictly below
    # chain-steps even after adding initialization and Fisher refreshes
    n_chain_steps = spec.n_steps * spec.n_chain
    assert evals_after_init < like_obj.n_evals < n_chain_steps
    assert validate(artifact_path, mode='complete') == []


@pytest.mark.usefixtures('fresh_seed_guard')
def test_partial_artifact_validates_as_partial_only(tmp_path) -> None:
    """A non-finalized artifact passes partial validation but not complete."""
    spec = make_tiny_spec()
    seed_children = seed_run(spec.seed)
    provenance = collect_provenance(spec.seed, *seed_children, spec_toml=spec.to_toml_text(), proposal_config_ini=spec.resolved_config_text())

    sampler, like_obj = build_sampler(spec)
    sampler.advance_block()

    artifact_path = tmp_path / 'partial.h5'
    write_artifact(artifact_path, spec, sampler, like_obj.n_evals, provenance, finalized=False, wall_seconds=0.0)

    assert validate(artifact_path, mode='partial') == []
    problems = validate(artifact_path, mode='complete')
    assert any('not finalized' in problem for problem in problems)


def test_batch_expansion(tmp_path) -> None:
    """Sweep expansion writes one valid spec per grid point x seed plus a manifest."""
    base_path = tmp_path / 'base.toml'
    base_path.write_text(dumps_toml(dict(TINY_GAUSSIAN_SPEC)))

    sweep_path = tmp_path / 'sweep.toml'
    sweep_path.write_text(dumps_toml({
        'name': 'test_sweep',
        'base_spec': str(base_path),
        'out': str(tmp_path / 'out'),
        'seeds': [101, 102, 103],
        'grid': {'ladder.n_chain': [6, 8], 'run.n_steps': [128]},
    }))

    manifest_path = write_batch(sweep_path)
    manifest_lines = manifest_path.read_text().strip().splitlines()
    assert len(manifest_lines) == 2 * 1 * 3

    seen: set[tuple[int, int]] = set()
    for line in manifest_lines:
        spec_file = shlex.split(line)[3]
        spec = RunSpec.from_toml(spec_file)
        assert spec.n_steps == 128
        seen.add((spec.n_chain, spec.seed))
    assert seen == {(n_chain, seed) for n_chain in (6, 8) for seed in (101, 102, 103)}


def test_paths_anchored_to_repo_root() -> None:
    """Path resolution is CWD-independent and finds the shipped config."""
    assert default_config_path().is_file()
    assert (repo_root() / 'DTMCMC').is_dir()


def test_numba_seeder_is_private() -> None:
    """PR #9 review: the raw numba seeder must not be a public entry point.

    The once-per-run guard cannot live inside the jitted body (numba
    cannot type the guard dict), so seed_run must be the only public
    seeding API; the helper is private and TID251-banned elsewhere.
    """
    public_names = {name for name in dir(rng_helpers) if not name.startswith('_')}
    assert 'seed_numba' not in public_names
    assert hasattr(rng_helpers, '_seed_numba')


def test_builder_registries_match_spec_names() -> None:
    """PR #9 review: runner builder registries stay in sync with spec constants."""
    assert set(LIKELIHOOD_BUILDERS) == set(LIKELIHOOD_NAMES)
    assert set(LADDER_BUILDERS) == set(LADDER_KINDS)


def _explicit_ladder_data(n_chain: int, Ts: list[float]) -> dict[str, Any]:
    """Copy the tiny spec with an explicit ladder of the given geometry."""
    data: dict[str, Any] = {key: dict(value) if isinstance(value, dict) else value for key, value in TINY_GAUSSIAN_SPEC.items()}
    data['ladder'] = {'kind': 'explicit', 'n_chain': n_chain, 'n_cold': 1, 'Ts': Ts}
    return data


def test_explicit_ladder_length_mismatch_raises() -> None:
    """PR #9 review: a Ts list contradicting ladder.n_chain fails at spec time."""
    with pytest.raises(SpecError, match=r'3 entries but ladder\.n_chain is 6'):
        RunSpec.from_dict(_explicit_ladder_data(6, [1.0, 2.0, 10.0]))


def test_explicit_ladder_matching_length_builds() -> None:
    """An explicit ladder with consistent geometry builds the declared chain count."""
    spec = RunSpec.from_dict(_explicit_ladder_data(3, [1.0, 2.0, 10.0]))
    ladder = build_ladder(spec)
    assert ladder.n_chain == spec.n_chain == 3


@pytest.mark.usefixtures('fresh_seed_guard')
def test_artifact_ladder_mismatch_detected(tmp_path) -> None:
    """PR #9 review: validate() flags a ladder that contradicts the embedded spec."""
    spec = make_tiny_spec()
    seed_children = seed_run(spec.seed)
    provenance = collect_provenance(spec.seed, *seed_children, spec_toml=spec.to_toml_text(), proposal_config_ini=spec.resolved_config_text())
    sampler, like_obj = build_sampler(spec)

    artifact_path = tmp_path / 'tampered.h5'
    write_artifact(artifact_path, spec, sampler, like_obj.n_evals, provenance, finalized=False, wall_seconds=0.0)
    assert validate(artifact_path, mode='partial') == []

    with h5py.File(str(artifact_path), 'a') as hf:
        ladder_grp = hf['ladder']
        assert isinstance(ladder_grp, h5py.Group)
        del ladder_grp['Ts']
        ladder_grp.create_dataset('Ts', data=np.array([1.0, 2.0, 10.0]))

    problems = validate(artifact_path, mode='partial')
    assert any('ladder.n_chain' in problem for problem in problems)
