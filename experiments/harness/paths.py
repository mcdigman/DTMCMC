"""Repo-root-anchored path resolution for the experiment harness.

The engine has two path fragilities (methods-paper plan §2): HawaiiLikelihood
opens data/hawaii_map.hdf5 relative to the CWD, and get_default_proposal_manager
reads default_config.ini from the CWD when no config is passed. The harness
resolves everything against the repo root: the runner chdirs there at startup
(safe under one-process-per-run), and the proposal config is always passed as
an explicit, resolved ConfigParser.
"""

import ntpath
import os
import stat
import tempfile
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def is_printable_line(value: str) -> bool:
    """Return whether *value* is a non-empty, printable, single-line string.

    Newlines, tabs and other control characters are not printable, so this is
    the containment test for any string interpolated into a line-oriented
    generated file (the batch manifest) or into a generated filename.
    """
    return bool(value) and value.isprintable()


def is_filename_component(value: str) -> bool:
    """Return whether *value* is one portable, non-special path component.

    ntpath.isreserved carries the Windows half of "portable": device names
    (CON, NUL, COM1), alternate data streams (name:stream), wildcards, and
    trailing dots or spaces, none of which name a plain file there.  It reads
    a leading 'x:' as a drive instead, so the colon is rejected here as well;
    'a:b' names a file on another drive rather than a component of this path.
    """
    return (
        is_printable_line(value)
        and value not in {'.', '..'}
        and '/' not in value
        and '\\' not in value
        and ':' not in value
        and not ntpath.isreserved(value)
    )


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


def _carry_destination_mode(replacement: Path, destination: Path) -> None:
    """Give the staged replacement the permissions of the regular file it replaces.

    Without this a replacement keeps whatever mode it was created with and
    silently narrows (or widens) a destination whose permissions were set
    deliberately.  A symlink or special-file destination is left alone: this
    module never follows one, so its mode is not the mode being replaced.
    """
    try:
        existing = destination.lstat()
    except OSError:
        return
    if stat.S_ISREG(existing.st_mode):
        replacement.chmod(stat.S_IMODE(existing.st_mode))


@contextmanager
def staged_replacement(destination: str | Path) -> Iterator[Path]:
    """Yield a staging path that atomically replaces *destination* once the body succeeds.

    The staging file lives in a fresh 0o700 directory beside the destination,
    so no other user can substitute a symlink for it between its creation and a
    writer opening it by name -- the window a bare temporary filename leaves
    open for writers that take a path rather than a descriptor (h5py).  An
    ordinary open() inside that directory lets the umask pick the permissions
    exactly as writing the destination directly would, unless there is already
    a regular file there whose mode should be carried over instead.  The
    directory and anything still in it are removed on every exit, so a failed
    body or a failed replace never leaves a partial file behind.
    """
    destination_path = Path(destination)
    with tempfile.TemporaryDirectory(
        prefix='.dtmcmc-', dir=destination_path.parent, ignore_cleanup_errors=True
    ) as staging_dir:
        # a fixed name is fine inside a directory nobody else can enter, and the
        # .tmp suffix keeps an in-flight artifact out of the dashboard's *.h5 scan
        staging_path = Path(staging_dir) / 'staged.tmp'
        yield staging_path
        _carry_destination_mode(staging_path, destination_path)
        staging_path.replace(destination_path)


def atomic_write_bytes(path_in: str | Path, payload: bytes) -> None:
    """Atomically replace a file without following an existing destination symlink."""
    with staged_replacement(path_in) as staging_path, staging_path.open('wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_text(path_in: str | Path, text: str) -> None:
    """UTF-8 encode and atomically replace a text file."""
    atomic_write_bytes(path_in, text.encode())


def chdir_repo_root() -> None:
    """Change the working directory to the repo root.

    Engine-internal relative paths (data/hawaii_map.hdf5) then resolve
    correctly; called once at runner startup, one process per run.
    """
    os.chdir(repo_root())
