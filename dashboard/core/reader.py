"""Read DTMCMC run artifacts into immutable in-memory snapshots.

The harness writer flushes atomically (write to a temp name, then rename),
so a reader that opens the file fresh at every poll can never observe a
torn file: it sees either the previous checkpoint or the new one. The
watcher exploits that by keying reloads on the (mtime_ns, size) stat pair
and re-reading the whole file only when it changes.

Everything is loaded eagerly into plain numpy arrays and stdlib types so
downstream diagnostics and figures never touch h5py handles.
"""

import configparser
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsInt, cast

import h5py
import numpy as np

# artifact schema this reader understands (experiments/harness/artifact.py)
SUPPORTED_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class LadderHistory:
    """Adaptive-ladder rebuild history (one row per rebuild evaluation)."""

    Ts: np.ndarray
    block_index: np.ndarray
    applied: np.ndarray
    t_cold_window: np.ndarray
    max_dlog_t: np.ndarray
    n_pool_points: np.ndarray
    frozen: bool
    frozen_by: str
    # burn-in boundary in blocks: -1 while still adapting (schema v4 stores
    # this so every reader shares one convention)
    frozen_block: int
    budget_blocks: int


@dataclass(frozen=True)
class RunSnapshot:
    """One artifact checkpoint loaded into plain numpy arrays.

    Field layouts follow the artifact schema: moments are per-block
    (n_blocks, n_chain); accept/esd records are (2, n_chain, n_jump_types)
    with [0] = accepted (for esd: all-proposal sums) and [1] = rejected
    (for esd: accepted-only sums); archives prepend a snapshot axis and
    are cumulative snapshots taken at ``itrn_archive`` iterations.
    """

    path: Path
    stat_token: tuple[int, int]

    # root attrs and embedded configuration
    attrs: dict[str, Any]
    spec: dict[str, Any]
    proposal_config: dict[str, dict[str, str]]

    # ladder
    n_cold: int
    Ts: np.ndarray
    betas: np.ndarray
    initial_Ts: np.ndarray
    history: LadderHistory | None

    # per-block moments
    logL_means: np.ndarray
    logL2_means: np.ndarray
    logL_vars: np.ndarray
    block_end_itrn: np.ndarray

    # trackers
    jump_labels: list[str]
    accept_record: np.ndarray
    accept_archive: np.ndarray
    esd_record: np.ndarray
    esd_archive: np.ndarray
    esd_exchange: np.ndarray
    esd_exchange_archive: np.ndarray
    exchange_tracker: np.ndarray
    exchange_archive: np.ndarray
    cycle_tracker: np.ndarray
    cycle_archive: np.ndarray
    itrn_archive: np.ndarray

    # round-trip events and walker flow
    rt_events: np.ndarray
    rt_segment_itrns: np.ndarray
    flow_up: np.ndarray
    flow_labeled: np.ndarray

    # checkpoint-only metrics (empty arrays when the run recorded none)
    checkpoint_itrns: np.ndarray
    de_spectrum_eigvals: np.ndarray

    # thinned sample store (recorded chains only): store column j holds
    # chain record_indices[j]; record_history_indices maps each completed
    # block to the recorded set active during it (readout-chain indices
    # move when a ladder update adds or removes rungs below T_cold)
    samples: np.ndarray
    logLs: np.ndarray
    store_thin: int
    record_indices: np.ndarray
    record_history_indices: np.ndarray

    @property
    def n_recorded(self) -> int:
        """Number of recorded store columns (readout chains + arg_record extras)."""
        return int(self.record_indices.size)

    @property
    def n_chain(self) -> int:
        """Number of chains in the ladder."""
        return int(self.Ts.size)

    @property
    def n_blocks(self) -> int:
        """Number of completed blocks in the moment record."""
        return int(self.logL_means.shape[0])

    @property
    def n_iterations(self) -> int:
        """Iterations completed at this checkpoint."""
        return int(self.attrs.get('n_iterations', 0))

    @property
    def block_size(self) -> int:
        """Iterations per block (schema v4 root attr, spec fallback)."""
        if 'block_size' in self.attrs:
            return int(cast('SupportsInt', self.attrs['block_size']))
        return int(self.spec.get('run', {}).get('block_size', 0))

    @property
    def n_steps(self) -> int:
        """Requested total iterations, from the embedded spec."""
        return int(self.spec.get('run', {}).get('n_steps', 0))

    @property
    def finalized(self) -> bool:
        """Whether the artifact was flushed at a major-report boundary."""
        return bool(self.attrs.get('finalized', False))

    @property
    def name(self) -> str:
        """Run name recorded by the harness."""
        return str(self.attrs.get('name', self.path.stem))

    @property
    def likelihood_name(self) -> str:
        """Likelihood name from the embedded spec."""
        return str(self.spec.get('likelihood', {}).get('name', 'unknown'))

    @property
    def n_par(self) -> int:
        """Parameter-space dimension of the recorded samples."""
        if self.samples.ndim == 3 and self.samples.shape[2] > 0:
            return int(self.samples.shape[2])
        return int(self.spec.get('likelihood', {}).get('n_par', 0))

    @property
    def track_full_exchanges(self) -> bool:
        """Whether the exchange tracker holds the full pairwise matrix."""
        return bool(self.spec.get('exchange', {}).get('track_full_exchanges', False))

    @property
    def adaptive(self) -> dict[str, Any] | None:
        """The [adaptive] spec table, or None for fixed-ladder runs."""
        table = self.spec.get('adaptive')
        return dict(table) if isinstance(table, dict) else None


def _stat_token(path: Path) -> tuple[int, int]:
    """Get the (mtime_ns, size) pair that identifies a flush of the file."""
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)


def _array(hf: h5py.File, key: str, fallback_shape: tuple[int, ...]) -> np.ndarray:
    """Read a dataset as a numpy array, tolerating its absence."""
    if key not in hf:
        return np.zeros(fallback_shape)
    return np.asarray(hf[key])


def _parse_ini(text: str) -> dict[str, dict[str, str]]:
    """Parse embedded INI text into plain nested dicts."""
    parser = configparser.ConfigParser()
    parser.read_string(text)
    return {section: dict(parser[section]) for section in parser.sections()}


def _load_history(hf: h5py.File) -> LadderHistory | None:
    """Load the adaptive ladder history group when present."""
    if 'ladder/history' not in hf:
        return None
    history = hf['ladder/history']
    assert isinstance(history, h5py.Group)
    return LadderHistory(
        Ts=np.asarray(history['Ts']),
        block_index=np.asarray(history['block_index'], dtype=np.int64),
        applied=np.asarray(history['applied'], dtype=bool),
        t_cold_window=np.asarray(history['t_cold_window']),
        max_dlog_t=np.asarray(history['max_dlog_t']),
        n_pool_points=np.asarray(history['n_pool_points'], dtype=np.int64),
        frozen=bool(history.attrs['frozen']),
        frozen_by=str(history.attrs['frozen_by']),
        frozen_block=int(cast('SupportsInt', history.attrs.get('frozen_block', -1))),
        budget_blocks=int(cast('SupportsInt', history.attrs.get('budget_blocks', -1))),
    )


def load_snapshot(path: str | Path, *, load_store: bool = True) -> RunSnapshot:
    """Load one artifact checkpoint into a RunSnapshot.

    Parameters
    ----------
    path: str | Path
        Artifact file to read
    load_store: bool
        Skip the (potentially large) thinned sample store when False;
        corner/trace/autocorrelation diagnostics then see empty arrays

    Returns
    -------
    snapshot: RunSnapshot
        Immutable in-memory copy of the checkpoint
    """
    artifact_path = Path(path)
    token = _stat_token(artifact_path)
    with h5py.File(str(artifact_path), 'r') as hf:
        attrs = dict(hf.attrs.items())
        schema = int(cast('SupportsInt', attrs.get('schema_version', -1)))
        if schema != SUPPORTED_SCHEMA_VERSION:
            msg = f'artifact schema version {schema} != supported {SUPPORTED_SCHEMA_VERSION}'
            raise ValueError(msg)

        spec = tomllib.loads(str(attrs.get('spec_toml', '')))
        proposal_config = _parse_ini(str(attrs.get('proposal_config_ini', '')))

        Ts = np.asarray(hf['ladder/Ts'])
        n_chain = int(Ts.size)
        ladder_grp = hf['ladder']
        assert isinstance(ladder_grp, h5py.Group)
        trackers_grp = hf['trackers']
        assert isinstance(trackers_grp, h5py.Group)

        accept_record = np.asarray(hf['trackers/accept_record'])
        n_jump_types = int(accept_record.shape[-1])
        raw_labels = trackers_grp.attrs.get('jump_labels')
        if raw_labels is None:
            # artifacts written before the labels were recorded
            jump_labels = [f'jump {itrj}' for itrj in range(n_jump_types)]
        else:
            jump_labels = [str(label) for label in np.asarray(raw_labels)]

        store_grp = hf['store']
        assert isinstance(store_grp, h5py.Group)
        n_par = int(spec.get('likelihood', {}).get('n_par', 0))
        if load_store:
            samples = np.asarray(hf['store/samples'])
            logLs = np.asarray(hf['store/logLs'])
        else:
            samples = np.zeros((0, 0, n_par))
            logLs = np.zeros((0, 0))

        return RunSnapshot(
            path=artifact_path,
            stat_token=token,
            attrs=attrs,
            spec=spec,
            proposal_config=proposal_config,
            n_cold=int(np.asarray(ladder_grp.attrs['n_cold']).item()),
            Ts=Ts,
            betas=np.asarray(hf['ladder/betas']),
            initial_Ts=np.asarray(hf['ladder/initial_Ts']) if 'ladder/initial_Ts' in hf else Ts.copy(),
            history=_load_history(hf),
            logL_means=np.asarray(hf['moments/logL_means']),
            logL2_means=np.asarray(hf['moments/logL2_means']),
            logL_vars=np.asarray(hf['moments/logL_vars']),
            block_end_itrn=np.asarray(hf['moments/block_end_itrn'], dtype=np.int64),
            jump_labels=jump_labels,
            accept_record=accept_record,
            accept_archive=np.asarray(hf['trackers/accept_archive']),
            esd_record=np.asarray(hf['trackers/esd_record']),
            esd_archive=np.asarray(hf['trackers/esd_archive']),
            esd_exchange=np.asarray(hf['trackers/esd_exchange']),
            esd_exchange_archive=np.asarray(hf['trackers/esd_exchange_archive']),
            exchange_tracker=np.asarray(hf['trackers/exchange_tracker']),
            exchange_archive=np.asarray(hf['trackers/exchange_archive']),
            cycle_tracker=np.asarray(hf['trackers/cycle_tracker']),
            cycle_archive=np.asarray(hf['trackers/cycle_archive']),
            itrn_archive=np.asarray(hf['trackers/itrn_archive'], dtype=np.int64),
            rt_events=np.asarray(hf['events/rt_events'], dtype=np.int64),
            rt_segment_itrns=np.asarray(hf['events/rt_segment_itrns'], dtype=np.int64),
            flow_up=np.asarray(hf['flow/up_counts'], dtype=np.int64),
            flow_labeled=np.asarray(hf['flow/labeled_counts'], dtype=np.int64),
            checkpoint_itrns=_array(hf, 'checkpoints/itrn', (0,)),
            de_spectrum_eigvals=_array(hf, 'checkpoints/de_spectrum_eigvals', (0, n_chain, max(n_par, 1))),
            samples=samples,
            logLs=logLs,
            store_thin=int(np.asarray(store_grp.attrs['store_thin']).item()),
            record_indices=np.asarray(hf['store/record_indices'], dtype=np.int64),
            record_history_indices=np.asarray(hf['store/record_history_indices'], dtype=np.int64),
        )


class ArtifactWatcher:
    """Poll one artifact file and reload it only when its flush changes.

    The watcher never raises out of poll(): a vanished file or a transient
    read failure (e.g. the artifact is being replaced by a run on a
    filesystem without atomic rename semantics) leaves the previous
    snapshot in place and reports the error string instead.
    """

    def __init__(self, path: str | Path, *, load_store: bool = True) -> None:
        self.path = Path(path)
        self.load_store = load_store
        self.snapshot: RunSnapshot | None = None
        self.last_error: str = ''

    def poll(self) -> RunSnapshot | None:
        """Reload the artifact if it changed; return the current snapshot."""
        try:
            token = _stat_token(self.path)
        except OSError as err:
            self.last_error = f'artifact not readable: {err}'
            return self.snapshot
        if self.snapshot is not None and token == self.snapshot.stat_token:
            self.last_error = ''
            return self.snapshot
        try:
            self.snapshot = load_snapshot(self.path, load_store=self.load_store)
            self.last_error = ''
        except (OSError, KeyError, ValueError) as err:
            self.last_error = f'artifact read failed: {err}'
        return self.snapshot


def list_artifacts(root: str | Path) -> list[Path]:
    """List candidate artifact files under a directory (or the file itself)."""
    root_path = Path(root)
    if root_path.is_file():
        return [root_path]
    if not root_path.is_dir():
        return []
    return sorted(candidate for candidate in root_path.rglob('*.h5') if not candidate.name.endswith('.tmp'))
