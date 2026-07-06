"""HDF5 run artifacts: writer, reader, and schema/provenance validation.

One HDF5 file per run (plan D2). Root attrs carry full provenance (git
state, resolved spec text, seeds, package versions, host, timestamps,
counters, finalized flag); datasets carry everything a figure needs
(ladder, per-block logL moments, tracker archives, thinned stored
samples). Figures are generated only from artifacts (plan D7), so nothing
may be written anywhere else.

Flushes rewrite the whole file to a temp name and atomically replace the
artifact, so a killed run leaves either the previous checkpoint or nothing,
never a truncated file. Runs are cheap (≤ ~1 CPU-h): a run that dies is
simply rerun; `finalized` distinguishes partial from complete artifacts.
"""

import platform
import socket
import subprocess
import time
import tomllib
from dataclasses import dataclass, fields
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, SupportsInt, cast

import h5py
import numpy as np

from .paths import repo_root

if TYPE_CHECKING:
    from DTMCMC.dtmcmc_sampler import DTMCMCSampler
    from experiments.adaptive import AdaptiveLadderController

    from .spec import RunSpec

SCHEMA_VERSION = 2

# root attrs written at flush time rather than carried by RunProvenance
_FLUSH_ATTRS: tuple[str, ...] = (
    'schema_version',
    'finalized',
    'name',
    'flush_time_utc',
    'wall_seconds',
    'n_iterations',
    'n_chain_steps',
    'n_likelihood_evals',
)

REQUIRED_DATASETS: tuple[str, ...] = (
    'ladder/Ts',
    'ladder/betas',
    'moments/logL_means',
    'moments/logL2_means',
    'moments/logL3_means',
    'moments/logL4_means',
    'moments/logL5_means',
    'moments/logL6_means',
    'moments/logL_prod11_means',
    'moments/logL_prod21_means',
    'moments/logL_prod12_means',
    'moments/logL_vars',
    'moments/block_end_itrn',
    'trackers/accept_record',
    'trackers/cycle_tracker',
    'trackers/exchange_tracker',
    'trackers/accept_archive',
    'trackers/cycle_archive',
    'trackers/exchange_archive',
    'trackers/esd_record',
    'trackers/esd_archive',
    'trackers/esd_exchange',
    'trackers/esd_exchange_archive',
    'trackers/itrn_archive',
    'events/rt_events',
    'events/rt_segment_itrns',
    'flow/up_counts',
    'flow/labeled_counts',
    'store/samples',
    'store/logLs',
)


@dataclass(frozen=True)
class RunProvenance:
    """Run-start provenance recorded in every artifact flush.

    Field names double as the artifact attr keys: write_artifact writes
    every field and REQUIRED_ATTRS is derived from the field list, so a
    new field cannot be silently dropped from either side. spec_toml and
    proposal_config_ini are captured once at run start — the INI text from
    the very ConfigParser instance the sampler was built with — so a
    mid-run edit of default_config.ini cannot corrupt provenance.
    """

    run_seed: int
    child_seed_python: int
    child_seed_numba: int
    git_commit: str
    git_dirty: bool
    hostname: str
    version_python: str
    version_numpy: str
    version_scipy: str
    version_numba: str
    version_h5py: str
    start_time_utc: str
    spec_toml: str
    proposal_config_ini: str


# root attrs that every artifact must carry (plan D2)
REQUIRED_ATTRS: tuple[str, ...] = tuple(prov_field.name for prov_field in fields(RunProvenance)) + _FLUSH_ATTRS


def _git_state() -> tuple[str, bool]:
    """Get the current git commit hash and dirty flag, tolerating failure."""
    try:
        # static arg list, repo-root cwd; provenance only, no untrusted input
        commit_res = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],  # noqa: S607
            cwd=repo_root(), capture_output=True, text=True, check=False, timeout=10,
        )
        status_res = subprocess.run(
            ['git', 'status', '--porcelain'],  # noqa: S607
            cwd=repo_root(), capture_output=True, text=True, check=False, timeout=10,
        )
    except OSError:
        return 'unknown', True
    if commit_res.returncode != 0 or status_res.returncode != 0:
        return 'unknown', True
    return commit_res.stdout.strip(), status_res.stdout.strip() != ''


def collect_provenance(run_seed: int, child_seed_python: int, child_seed_numba: int, spec_toml: str, proposal_config_ini: str) -> RunProvenance:
    """Collect run-start provenance (git state, host, versions, resolved config texts)."""
    git_commit, git_dirty = _git_state()
    return RunProvenance(
        run_seed=run_seed,
        child_seed_python=child_seed_python,
        child_seed_numba=child_seed_numba,
        git_commit=git_commit,
        git_dirty=git_dirty,
        hostname=socket.gethostname(),
        version_python=platform.python_version(),
        version_numpy=package_version('numpy'),
        version_scipy=package_version('scipy'),
        version_numba=package_version('numba'),
        version_h5py=package_version('h5py'),
        start_time_utc=datetime.now(tz=UTC).isoformat(),
        spec_toml=spec_toml,
        proposal_config_ini=proposal_config_ini,
    )


def _stack_blocks(arrays: list[np.ndarray], n_chain: int) -> np.ndarray:
    """Stack a list of per-block arrays, tolerating zero completed blocks."""
    if not arrays:
        return np.zeros((0, n_chain))
    return np.asarray(arrays)


def _stack_archive(arrays: list[np.ndarray], element_shape: tuple[int, ...]) -> np.ndarray:
    """Stack a list of archived tracker snapshots, tolerating an empty archive."""
    if not arrays:
        return np.zeros((0, *element_shape), dtype=np.int64)
    return np.asarray(arrays)


@dataclass
class CheckpointLog:
    """In-process checkpoint metrics accumulated by the runner.

    The DE buffer is transient (never persisted), so its difference
    spectra can only be measured in-process; everything else in the
    artifact supports post-hoc analysis (plan D7).
    """

    itrns: list[int] = dataclass_field(default_factory=list)
    de_spectrum_eigvals: list[np.ndarray] = dataclass_field(default_factory=list)


def write_artifact(
    path: Path,
    spec: RunSpec,
    sampler: DTMCMCSampler,
    n_likelihood_evals: int,
    provenance: RunProvenance,
    finalized: bool,
    wall_seconds: float,
    checkpoints: CheckpointLog | None = None,
    adaptive_state: AdaptiveLadderController | None = None,
) -> None:
    """Write the full run artifact, atomically replacing any previous flush."""
    tracker = sampler.tracker_manager
    n_chain = sampler.n_chain
    n_blocks_done = sampler.itrn // sampler.block_size

    # the store ring buffer is sized to be exactly filled at run end and
    # never wraps mid-run, so the written rows are a prefix of the buffer
    rows_written = min(-(-sampler.itrn // sampler.store_thin), sampler.store_size)

    tmp_path = path.with_name(path.name + '.tmp')
    with h5py.File(str(tmp_path), 'w') as hf:
        for prov_field in fields(provenance):
            hf.attrs[prov_field.name] = getattr(provenance, prov_field.name)
        hf.attrs['schema_version'] = SCHEMA_VERSION
        hf.attrs['finalized'] = finalized
        hf.attrs['name'] = spec.name
        hf.attrs['flush_time_utc'] = datetime.now(tz=UTC).isoformat()
        hf.attrs['wall_seconds'] = wall_seconds
        hf.attrs['n_iterations'] = sampler.itrn
        hf.attrs['n_chain_steps'] = sampler.itrn * n_chain
        hf.attrs['n_likelihood_evals'] = n_likelihood_evals

        ladder_grp = hf.create_group('ladder')
        ladder_grp.attrs['n_cold'] = sampler.n_cold
        ladder_grp.create_dataset('Ts', data=sampler.Ts)
        ladder_grp.create_dataset('betas', data=sampler.betas)

        if adaptive_state is not None:
            # adaptive runs: one row per rebuild evaluation, applied or
            # held (plan D2/Phase 5); frozen_by is '' until the freeze
            # fires, then 'criterion' or 'budget' so E3 can segregate
            # budget-frozen runs
            ladder_history = adaptive_state.history
            history_grp = ladder_grp.create_group('history')
            history_grp.attrs['frozen'] = adaptive_state.frozen
            history_grp.attrs['frozen_by'] = adaptive_state.frozen_by
            history_grp.create_dataset('Ts', data=np.asarray([record.Ts for record in ladder_history]))
            history_grp.create_dataset('block_index', data=np.asarray([record.block_index for record in ladder_history], dtype=np.int64))
            history_grp.create_dataset('applied', data=np.asarray([record.applied for record in ladder_history], dtype=np.bool_))
            history_grp.create_dataset('t_cold_window', data=np.asarray([record.t_cold_window for record in ladder_history]))
            history_grp.create_dataset('max_dlog_t', data=np.asarray([record.max_dlog_t for record in ladder_history]))
            history_grp.create_dataset('n_pool_points', data=np.asarray([record.n_pool_points for record in ladder_history], dtype=np.int64))

        moments_grp = hf.create_group('moments')
        moments_grp.create_dataset('logL_means', data=_stack_blocks(sampler.logL_means, n_chain))
        moments_grp.create_dataset('logL2_means', data=_stack_blocks(sampler.logL2_means, n_chain))
        moments_grp.create_dataset('logL3_means', data=_stack_blocks(sampler.logL3_means, n_chain))
        moments_grp.create_dataset('logL4_means', data=_stack_blocks(sampler.logL4_means, n_chain))
        moments_grp.create_dataset('logL5_means', data=_stack_blocks(sampler.logL5_means, n_chain))
        moments_grp.create_dataset('logL6_means', data=_stack_blocks(sampler.logL6_means, n_chain))
        moments_grp.create_dataset('logL_prod11_means', data=_stack_blocks(sampler.logL_prod11_means, n_chain - 1))
        moments_grp.create_dataset('logL_prod21_means', data=_stack_blocks(sampler.logL_prod21_means, n_chain - 1))
        moments_grp.create_dataset('logL_prod12_means', data=_stack_blocks(sampler.logL_prod12_means, n_chain - 1))
        moments_grp.create_dataset('logL_vars', data=_stack_blocks(sampler.logL_vars, n_chain))
        moments_grp.create_dataset('block_end_itrn', data=np.arange(1, n_blocks_done + 1, dtype=np.int64) * sampler.block_size)

        trackers_grp = hf.create_group('trackers')
        trackers_grp.create_dataset('accept_record', data=tracker.accept_record)
        trackers_grp.create_dataset('cycle_tracker', data=tracker.cycle_tracker)
        trackers_grp.create_dataset('exchange_tracker', data=tracker.exchange_tracker)
        trackers_grp.create_dataset('accept_archive', data=_stack_archive(tracker.accept_archive, tracker.accept_record.shape))
        trackers_grp.create_dataset('cycle_archive', data=_stack_archive(tracker.cycle_archive, tracker.cycle_tracker.shape))
        trackers_grp.create_dataset('exchange_archive', data=_stack_archive(tracker.exchange_archive, tracker.exchange_tracker.shape))
        trackers_grp.create_dataset('esd_record', data=tracker.esd_record)
        trackers_grp.create_dataset('esd_archive', data=np.asarray(tracker.esd_archive) if tracker.esd_archive else np.zeros((0, *tracker.esd_record.shape)))
        trackers_grp.create_dataset('esd_exchange', data=tracker.esd_exchange)
        trackers_grp.create_dataset('esd_exchange_archive', data=np.asarray(tracker.esd_exchange_archive) if tracker.esd_exchange_archive else np.zeros((0, tracker.esd_exchange.shape[0])))
        trackers_grp.create_dataset('itrn_archive', data=np.asarray(tracker.itrn_archive, dtype=np.int64))

        # round-trip event log: rows of (walker id, iteration, direction)
        # with direction 0 = arrived cold from hot, 1 = arrived hot from cold,
        # plus the ladder-segment boundaries (empty for fixed-ladder runs)
        # that round-trip metrics must not pair arrivals across (plan D6)
        events_grp = hf.create_group('events')
        events_grp.create_dataset('rt_events', data=tracker.get_rt_events())
        events_grp.create_dataset('rt_segment_itrns', data=tracker.get_rt_segment_itrns())

        flow_up, flow_labeled = tracker.get_flow_counts()
        flow_grp = hf.create_group('flow')
        flow_grp.create_dataset('up_counts', data=flow_up)
        flow_grp.create_dataset('labeled_counts', data=flow_labeled)

        if checkpoints is not None and checkpoints.itrns:
            ckpt_grp = hf.create_group('checkpoints')
            ckpt_grp.create_dataset('itrn', data=np.asarray(checkpoints.itrns, dtype=np.int64))
            ckpt_grp.create_dataset('de_spectrum_eigvals', data=np.asarray(checkpoints.de_spectrum_eigvals))

        store_grp = hf.create_group('store')
        store_grp.attrs['store_thin'] = sampler.store_thin
        store_grp.attrs['n_record'] = sampler.n_record
        store_grp.create_dataset('samples', data=sampler.samples_store[:rows_written])
        store_grp.create_dataset('logLs', data=sampler.logLs_store[:rows_written])

    tmp_path.replace(path)


def _attr_int(hf: h5py.File, key: str) -> int:
    """Read a root attr as a plain int."""
    return int(cast('SupportsInt', hf.attrs[key]))


def validate(path: str | Path, mode: str = 'complete') -> list[str]:
    """Validate artifact schema and provenance completeness.

    Parameters
    ----------
    path: str | Path
        Artifact file to check
    mode: str
        'complete' additionally requires the finalized flag and that the
        counters match the embedded spec; 'partial' checks schema and
        provenance only

    Returns
    -------
    problems: list[str]
        Empty when the artifact is valid
    """
    if mode not in ('complete', 'partial'):
        msg = f"validate mode must be 'complete' or 'partial', got {mode!r}"
        raise ValueError(msg)

    artifact_path = Path(path)
    if not artifact_path.is_file():
        return [f'artifact file {artifact_path} does not exist']

    problems: list[str] = []
    with h5py.File(str(artifact_path), 'r') as hf:
        # a schema mismatch explains every downstream difference at once, so
        # report it alone instead of a pile of missing-dataset messages
        if 'schema_version' not in hf.attrs:
            return ["missing root attr 'schema_version'"]
        found_schema = _attr_int(hf, 'schema_version')
        if found_schema != SCHEMA_VERSION:
            return [f'artifact schema version {found_schema} != supported {SCHEMA_VERSION}']

        problems.extend(f'missing root attr {attr!r}' for attr in REQUIRED_ATTRS if attr not in hf.attrs)
        problems.extend(f'missing dataset {dataset!r}' for dataset in REQUIRED_DATASETS if dataset not in hf)

        if problems:
            return problems

        spec_text = str(hf.attrs['spec_toml'])
        try:
            spec_data = tomllib.loads(spec_text)
        except tomllib.TOMLDecodeError as err:
            problems.append(f'embedded spec_toml does not parse: {err}')
            return problems

        Ts_dataset = hf['ladder/Ts']
        if not isinstance(Ts_dataset, h5py.Dataset):
            problems.append('ladder/Ts is not a dataset')
            return problems

        n_iterations = _attr_int(hf, 'n_iterations')
        n_chain = int(Ts_dataset.shape[0])
        if _attr_int(hf, 'n_chain_steps') != n_iterations * n_chain:
            problems.append('n_chain_steps attr inconsistent with n_iterations and ladder size')
        if _attr_int(hf, 'n_likelihood_evals') < 0:
            problems.append('n_likelihood_evals attr is negative')

        # the run's actual geometry must match the embedded spec (PR #9
        # review): a mismatch means the artifact contradicts its own
        # provenance, e.g. an explicit ladder built from a wrong-length Ts
        ladder_table = spec_data.get('ladder')
        declared_n_chain = ladder_table.get('n_chain') if isinstance(ladder_table, dict) else None
        if not isinstance(declared_n_chain, int) or isinstance(declared_n_chain, bool):
            problems.append('embedded spec_toml lacks an integer ladder.n_chain')
        elif declared_n_chain != n_chain:
            problems.append(f'ladder/Ts length {n_chain} != embedded spec ladder.n_chain {declared_n_chain}')

        if mode == 'complete':
            if not bool(hf.attrs['finalized']):
                problems.append('artifact is not finalized')
            run_table = spec_data.get('run')
            n_steps = run_table.get('n_steps') if isinstance(run_table, dict) else None
            if not isinstance(n_steps, int):
                problems.append('embedded spec_toml lacks an integer run.n_steps')
            elif n_iterations != n_steps:
                problems.append(f'n_iterations {n_iterations} != spec run.n_steps {n_steps}')

    return problems


def read_attrs(path: str | Path) -> dict[str, object]:
    """Read all root attrs of an artifact as a plain dict."""
    with h5py.File(str(Path(path)), 'r') as hf:
        return dict(hf.attrs.items())


def wall_seconds_since(start_monotonic: float) -> float:
    """Get elapsed wall-clock seconds since a time.monotonic() reference."""
    return time.monotonic() - start_monotonic
