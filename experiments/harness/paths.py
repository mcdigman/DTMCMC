"""Repo-root-anchored path resolution for the experiment harness.

The engine has two path fragilities (methods-paper plan §2): HawaiiLikelihood
opens data/hawaii_map.hdf5 relative to the CWD, and get_default_proposal_manager
reads default_config.ini from the CWD when no config is passed. The harness
resolves everything against the repo root: the runner chdirs there at startup
(safe under one-process-per-run), and the proposal config is always passed as
an explicit, resolved ConfigParser.
"""

import os
from pathlib import Path


def repo_root() -> Path:
    """Get the repository root, anchored to this file's location."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Get the repo data directory holding reference ladders and maps."""
    return repo_root() / 'data'


def default_config_path() -> Path:
    """Get the resolved path of the default proposal-manager config."""
    return repo_root() / 'default_config.ini'


def resolve(path_in: str | Path) -> Path:
    """Resolve a possibly repo-relative path against the repo root.

    Absolute paths pass through unchanged; relative paths are interpreted
    relative to the repo root regardless of the caller's CWD.
    """
    path = Path(path_in)
    if path.is_absolute():
        return path
    return repo_root() / path


def chdir_repo_root() -> None:
    """Change the working directory to the repo root.

    Engine-internal relative paths (data/hawaii_map.hdf5) then resolve
    correctly; called once at runner startup, one process per run.
    """
    os.chdir(repo_root())
