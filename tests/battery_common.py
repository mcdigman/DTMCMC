"""Shared plumbing for the adaptive convergence batteries (not a test module).

Builds the shared adaptive test spec and loads the post-freeze readout
column plus freeze metadata from an artifact, so every battery reads
runs the same way.
"""

from typing import Any

import h5py
import numpy as np


def adaptive_spec_data(
        name: str,
        seed: int,
        likelihood: dict[str, Any],
        *,
        n_chain: int,
        block_size: int,
        n_blocks: int,
        budget_blocks: int,
        store_thin: int = 4,
        t_min_factor: float = 0.9,
        remap_rule: str = 'no_remap',
        proposals_extra: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the shared adaptive battery spec."""
    proposals: dict[str, dict[str, Any]] = {
        'FisherJumpManager': {'verbose_fisher': False},
        'DEJumpManager': {'de_size': block_size * n_blocks},
    }
    for section, entries in (proposals_extra or {}).items():
        proposals.setdefault(section, {}).update(entries)
    return {
        'name': name,
        'seed': seed,
        'likelihood': likelihood,
        'ladder': {'kind': 'geometric', 'n_chain': n_chain, 'n_cold': 1},
        'run': {'n_steps': block_size * n_blocks, 'block_size': block_size, 'store_thin': store_thin,
                'checkpoint_every_blocks': n_blocks},
        'adaptive': {'mode': 'entropy', 'update_every_blocks': 8, 'forgetting': 0.15,
                     'freeze_dlog': 0.05, 'freeze_consecutive': 3, 'budget_blocks': budget_blocks,
                     'remap_rule': remap_rule, 'T_min_factor': t_min_factor},
        'exchange': {'strategy': 'sequential', 'track_full_exchanges': False},
        'proposals': proposals,
    }


def load_post_freeze(artifact_path, *, block_size: int, store_thin: int, budget_blocks: int) -> dict[str, Any]:
    """Load freeze metadata and the post-freeze readout-column samples.

    The readout chains always occupy the first store columns (arg_record
    convention), so column 0 is the T = 1 posterior stream; rows before
    the freeze block are adaptation burn-in and are discarded.
    """
    with h5py.File(str(artifact_path), 'r') as hf:
        frozen = bool(hf['ladder/history'].attrs['frozen'])
        frozen_by = str(hf['ladder/history'].attrs['frozen_by'])
        block_index = np.asarray(hf['ladder/history/block_index'])
        applied = np.asarray(hf['ladder/history/applied'])
        final_Ts = np.asarray(hf['ladder/Ts'])
        samples = np.asarray(hf['store/samples'])
        logLs = np.asarray(hf['store/logLs'])
        record_indices = np.asarray(hf['store/record_indices'])

    freeze_block = int(block_index[-1]) if frozen_by == 'criterion' else budget_blocks
    row_start = freeze_block * block_size // store_thin
    return {
        'frozen': frozen,
        'frozen_by': frozen_by,
        'freeze_block': freeze_block,
        'n_applied': int(applied.sum()),
        'final_Ts': final_Ts,
        'record_indices': record_indices,
        'cold': samples[row_start:, 0, :],
        'cold_logLs': logLs[row_start:, 0],
    }


def assert_readout_structure(run: dict[str, Any], *, require_below_readout: bool = True) -> None:
    """Common structural assertions: frozen, readout pinned at T=1, sub-readout rungs.

    Budget freezes are acceptable outcomes (the run still ends on a fixed
    ladder with the reason recorded); a run that never froze, never
    applied an update, or reads out anywhere but T = 1 is broken
    regardless of posterior quality.
    """
    assert run['frozen'], 'adaptive run must end frozen'
    assert run['frozen_by'] in ('criterion', 'budget')
    assert run['n_applied'] >= 2, 'adaptation never applied an update'
    readout_T = float(run['final_Ts'][int(run['record_indices'][0])])
    assert readout_T == 1., f'readout chain sits at T={readout_T}, not the T=1 target'
    if require_below_readout:
        assert int((run['final_Ts'] < 1.).sum()) >= 1, 'no sub-readout rungs despite T_min_factor < 1'
        assert int(run['record_indices'][0]) > 0, 'readout should sit interior to the sorted ladder'
