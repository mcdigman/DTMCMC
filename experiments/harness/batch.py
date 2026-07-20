"""Expand a sweep file into independent single-run invocations.

A sweep TOML names a base spec, a seed list, and a grid of dotted-key
overrides. Expansion writes one spec file per (grid point x seed) plus a
manifest of one command per line: GNU-parallel/cluster-array friendly,
one process per run (plan §4 Phase 1). Paired-seed comparisons (plan D3)
fall out of every grid point sharing the same seed list.

Example sweep file::

    name = 'chain_sweep'
    base_spec = 'experiments/specs/tiny_gaussian.toml'
    out = 'artifacts/chain_sweep'
    seeds = [101, 102, 103]

    [grid]
    'ladder.n_chain' = [8, 16, 32]
"""

import argparse
import itertools
import shlex
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from .paths import resolve
from .spec import RunSpec, dumps_toml

if TYPE_CHECKING:
    from DTMCMC.likelihood import AbstractLikelihood


def _apply_dotted_override(data: dict[str, object], dotted_key: str, value: object) -> None:
    """Set a nested dict entry addressed by a dotted key like 'ladder.n_chain'."""
    keys = dotted_key.split('.')
    table: dict[str, object] = data
    for key in keys[:-1]:
        nested = table.setdefault(key, {})
        if not isinstance(nested, dict):
            msg = f'grid key {dotted_key!r} descends through non-table {key!r}'
            raise TypeError(msg)
        table = nested
    table[keys[-1]] = value


def expand_sweep(sweep_path: str | Path) -> tuple[str, Path, list[RunSpec[AbstractLikelihood]]]:
    """Expand a sweep file into fully validated per-run specs.

    Returns
    -------
    name: str
        Sweep name
    out_dir: Path
        Output directory named by the sweep file
    specs: list[RunSpec]
        One spec per (grid point x seed), each with a unique name and seed
    """
    with Path(resolve(sweep_path)).open('rb') as sweep_file:
        sweep = tomllib.load(sweep_file)

    name = sweep.get('name')
    base_spec_path = sweep.get('base_spec')
    out_dir_raw = sweep.get('out')
    seeds = sweep.get('seeds')
    grid = sweep.get('grid', {})

    if not isinstance(name, str) or not isinstance(base_spec_path, str) or not isinstance(out_dir_raw, str):
        msg = 'sweep file requires string entries: name, base_spec, out'
        raise TypeError(msg)
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
    ):
        msg = 'sweep file requires a non-empty integer list: seeds'
        raise TypeError(msg)
    if not isinstance(grid, dict) or not all(isinstance(values, list) and values for values in grid.values()):
        msg = 'sweep [grid] must map dotted keys to non-empty lists'
        raise TypeError(msg)

    base_data = RunSpec.from_toml(resolve(base_spec_path)).to_dict()

    grid_keys = sorted(grid)
    grid_values = [grid[key] for key in grid_keys]

    specs: list[RunSpec[AbstractLikelihood]] = []
    for point_index, combo in enumerate(itertools.product(*grid_values)):
        for seed in seeds:
            data = RunSpec.from_dict(base_data).to_dict()
            for dotted_key, value in zip(grid_keys, combo, strict=True):
                _apply_dotted_override(data, dotted_key, value)
            data['name'] = f'{name}_pt{point_index:04d}'
            data['seed'] = seed
            specs.append(RunSpec.from_dict(data))

    return name, Path(resolve(out_dir_raw)), specs


def write_batch(sweep_path: str | Path) -> Path:
    """Write expanded spec files and the run manifest; return the manifest path."""
    specs: list[RunSpec[AbstractLikelihood]]
    _name, out_dir, specs = expand_sweep(sweep_path)

    specs_dir = out_dir / 'specs'
    specs_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines: list[str] = []
    for spec in specs:
        spec_path = specs_dir / f'{spec.name}_seed{spec.seed}.toml'
        spec_path.write_text(dumps_toml(spec.to_dict()))
        manifest_lines.append(
            f'python -m experiments.harness.run {shlex.quote(str(spec_path))} --out {shlex.quote(str(out_dir))}'
        )

    manifest_path = out_dir / 'manifest.txt'
    manifest_path.write_text('\n'.join(manifest_lines) + '\n')
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    """Expand a sweep file and print the manifest path."""
    parser = argparse.ArgumentParser(description='Expand a sweep TOML into per-run specs and a manifest')
    parser.add_argument('sweep', help='path to the sweep TOML (repo-root relative or absolute)')
    args = parser.parse_args(argv)

    manifest_path = write_batch(args.sweep)
    print(f'wrote {manifest_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
