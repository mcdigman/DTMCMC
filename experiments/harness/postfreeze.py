"""Post-freeze readout loading: the shared burn-in-discard convention.

Every consumer of an adaptive artifact's posterior stream — acceptance
batteries, pilots, production analyses — must discard at least the entire
temperature-ladder adaptation from the reported history. This module owns
that convention so it is applied identically everywhere instead of being
re-derived per caller: the freeze block comes from the artifact itself
(schema v4 stores it in ladder/history attrs), rows before it are
adaptation burn-in and are dropped, and the readout columns are located
through the arg_record record map.
"""

from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def load_post_freeze(artifact_path: str | Path) -> dict[str, Any]:
    """Load freeze metadata and the post-freeze readout-column samples.

    The readout chains always occupy the first store columns (arg_record
    convention), so column 0 is the T = 1 posterior stream; rows before
    the freeze block are adaptation burn-in and are discarded. All
    geometry (block size, store thinning, freeze block) is read from the
    artifact, so callers cannot drift from the run's actual configuration.

    Raises
    ------
    ValueError
        If the artifact has no adaptive history or its controller never
        froze: the post-freeze posterior stream is undefined for such a
        run and gating it would silently include adaptation burn-in.
    """
    with h5py.File(str(artifact_path), 'r') as hf:
        if 'ladder/history' not in hf:
            msg = 'artifact has no ladder/history group: not an adaptive run'
            raise ValueError(msg)
        history = hf['ladder/history']
        frozen = bool(history.attrs['frozen'])
        frozen_by = str(history.attrs['frozen_by'])
        freeze_block = int(np.asarray(history.attrs['frozen_block']).item())
        block_size = int(np.asarray(hf.attrs['block_size']).item())
        store_thin = int(np.asarray(hf['store'].attrs['store_thin']).item())
        applied = np.asarray(hf['ladder/history/applied'])
        max_dlog_t = np.asarray(hf['ladder/history/max_dlog_t'])
        final_Ts = np.asarray(hf['ladder/Ts'])
        samples = np.asarray(hf['store/samples'])
        logLs = np.asarray(hf['store/logLs'])
        record_indices = np.asarray(hf['store/record_indices'])

    if not frozen or freeze_block < 0:
        msg = f'artifact controller never froze (frozen={frozen}, frozen_block={freeze_block}); post-freeze loading is undefined'
        raise ValueError(msg)

    row_start = freeze_block * block_size // store_thin
    return {
        'frozen': frozen,
        'frozen_by': frozen_by,
        'freeze_block': freeze_block,
        'n_applied': int(applied.sum()),
        'max_dlog_last': float(max_dlog_t[-1]) if max_dlog_t.size else np.inf,
        'final_Ts': final_Ts,
        'record_indices': record_indices,
        'cold': samples[row_start:, 0, :],
        'cold_logLs': logLs[row_start:, 0],
    }


def readout_structure_violations(run: dict[str, Any], *, require_below_readout: bool = True) -> list[str]:
    """Structural problems with a post-freeze run dict; empty when sound.

    Budget freezes are acceptable outcomes (the run still ends on a fixed
    ladder with the reason recorded); a run that never froze, never
    applied an update, or reads out anywhere but T = 1 is broken
    regardless of posterior quality.
    """
    violations: list[str] = []
    if not run['frozen']:
        violations.append('adaptive run must end frozen')
    if run['frozen_by'] not in ('criterion', 'budget'):
        violations.append(f'unknown freeze reason {run["frozen_by"]!r}')
    if run['n_applied'] < 2:
        violations.append('adaptation never applied an update')
    readout_T = float(run['final_Ts'][int(run['record_indices'][0])])
    if readout_T != 1.:
        violations.append(f'readout chain sits at T={readout_T}, not the T=1 target')
    if require_below_readout:
        if int((run['final_Ts'] < 1.).sum()) < 1:
            violations.append('no sub-readout rungs despite T_min_factor < 1')
        if int(run['record_indices'][0]) <= 0:
            violations.append('readout should sit interior to the sorted ladder')
    return violations
