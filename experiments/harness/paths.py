"""Repo-root-anchored path resolution for the experiment harness.

The engine has two path fragilities (methods-paper plan §2): HawaiiLikelihood
opens data/hawaii_map.hdf5 relative to the CWD, and get_default_proposal_manager
reads default_config.ini from the CWD when no config is passed. The harness
resolves everything against the repo root: the runner chdirs there at startup
(safe under one-process-per-run), and the proposal config is always passed as
an explicit, resolved ConfigParser.
"""

import os
import stat
import tempfile
import tomllib
from pathlib import Path

MAX_TOML_FILE_BYTES = 1024 * 1024


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


def is_filename_component(value: str) -> bool:
    """Return whether *value* is one portable, non-special path component."""
    return bool(value) and value not in {'.', '..'} and value.isprintable() and '/' not in value and '\\' not in value


def read_regular_file_bytes(path_in: str | Path, *, max_bytes: int) -> bytes:
    """Read a bounded regular file without following a final-component symlink.

    The harness intentionally accepts both absolute and repo-relative local
    input paths.  This protects that contract's actual trust boundary: the
    selected object must remain the same regular file from validation through
    open, and its contents must fit within the caller's explicit size bound.
    """
    if max_bytes < 0:
        msg = 'max_bytes must be non-negative'
        raise ValueError(msg)

    path = Path(path_in)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        msg = f'{path} must be a regular file, not a symlink or special file'
        raise OSError(msg)
    if before.st_size > max_bytes:
        msg = f'{path} is larger than the {max_bytes}-byte read limit'
        raise ValueError(msg)

    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(  # skylos: ignore[SKY-D215] caller-selected local file is no-follow, regular, identity-checked, and bounded
        path, flags
    )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            msg = f'{path} must be a regular file'
            raise OSError(msg)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        identity_opened = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        if identity_opened != identity_before:
            msg = f'{path} changed while it was being opened'
            raise OSError(msg)
        if opened.st_size > max_bytes:
            msg = f'{path} is larger than the {max_bytes}-byte read limit'
            raise ValueError(msg)

        with os.fdopen(descriptor, 'rb') as handle:
            descriptor = -1
            payload = handle.read(max_bytes + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if len(payload) > max_bytes:
        msg = f'{path} grew beyond the {max_bytes}-byte read limit'
        raise ValueError(msg)
    return payload


def load_toml(path_in: str | Path) -> dict[str, object]:
    """Load a small TOML input through the harness's safe local-file reader."""
    payload = read_regular_file_bytes(path_in, max_bytes=MAX_TOML_FILE_BYTES)
    return tomllib.loads(payload.decode())


def atomic_write_bytes(path_in: str | Path, payload: bytes) -> None:
    """Atomically replace a file without following an existing destination symlink."""
    destination = Path(path_in)
    descriptor, temporary_name = tempfile.mkstemp(prefix='.dtmcmc-', suffix='.tmp', dir=destination.parent)
    temporary_path = Path(temporary_name)
    temporary_exists = True
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
        temporary_exists = False
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_exists:
            temporary_path.unlink(missing_ok=True)


def atomic_write_text(path_in: str | Path, text: str) -> None:
    """UTF-8 encode and atomically replace a text file."""
    atomic_write_bytes(path_in, text.encode())


def chdir_repo_root() -> None:
    """Change the working directory to the repo root.

    Engine-internal relative paths (data/hawaii_map.hdf5) then resolve
    correctly; called once at runner startup, one process per run.
    """
    os.chdir(repo_root())
