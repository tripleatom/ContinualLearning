"""Shared server routing for scripts reading/writing the lab file server.

The OLD server (\\\\10.129.151.108\\xieluanlabs) is FULL, so ALL new output goes
to the NEW server (\\\\10.129.151.88\\xieluanlabs2), keeping the same subpath
after the share name. Nothing is written to the old server any more - it stays
readable for the raw recordings and previously-computed data that live there.

Reads resolve on either server, NEW first: every write now lands on the new
server, so when a file exists on both, the new-server copy is the current one.

Usage:
    from server_fallback import resolve_output_folder, resolve_existing_file, mirror_on_backup_server

    out_dir = resolve_output_folder(some_folder)   # redirected to the new server
    with open(out_dir / "file.pkl", "wb") as f:
        pickle.dump(data, f)

The per-script `except OSError (ENOSPC) -> mirror_on_backup_server(...)` handlers
still work, but they are now a genuine last resort: once output is already on
the new server there is nowhere left to spill to, so ENOSPC re-raises.

Set WRITE_TO_NEW_SERVER = False to restore the previous behaviour (write on the
old server, spilling over to the new one only when free space runs low).
"""
import shutil
from pathlib import Path

OLD_SERVER_ROOT = r"\\10.129.151.108\xieluanlabs"    # full - read only
NEW_SERVER_ROOT = r"\\10.129.151.88\xieluanlabs2"    # write everything here

# Send every output to the new server, regardless of free space on the old one.
WRITE_TO_NEW_SERVER = True

# Only consulted when WRITE_TO_NEW_SERVER is False.
MIN_FREE_BYTES = 5 * 1024 ** 3


def mirror_on_backup_server(path):
    """Map a path under OLD_SERVER_ROOT to the equivalent path under NEW_SERVER_ROOT.

    Returns None when `path` is not on the old server - either it is already on
    the new server or it is local - i.e. "there is nowhere else to put this".
    """
    path_str = str(path)
    if path_str.lower().startswith(OLD_SERVER_ROOT.lower()):
        return Path(NEW_SERVER_ROOT + path_str[len(OLD_SERVER_ROOT):])
    return None


def _existing_ancestor(path):
    """Walk up from path until an existing directory is found (for disk_usage)."""
    path = Path(path)
    while not path.exists():
        parent = path.parent
        if parent == path:
            return path
        path = parent
    return path


def resolve_output_folder(folder, min_free_bytes=MIN_FREE_BYTES):
    """Return the folder to write into, created: the new-server mirror of `folder`.

    Old-server paths are redirected unconditionally - no disk_usage probe, which
    also skips a slow network stat call. Paths that are already on the new server,
    or that live on a local disk, are used as given.
    """
    folder = Path(folder)

    if WRITE_TO_NEW_SERVER:
        target = mirror_on_backup_server(folder) or folder
        target.mkdir(parents=True, exist_ok=True)
        if target != folder:
            print(f"Output redirected to the new server: {target}")
        return target

    # Legacy behaviour: write in place until the disk gets tight, then spill over.
    try:
        free = shutil.disk_usage(_existing_ancestor(folder)).free
    except OSError:
        free = None

    if free is not None and free >= min_free_bytes:
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    backup_folder = mirror_on_backup_server(folder)
    free_str = "unknown" if free is None else f"{free / 1e9:.1f} GB"
    if backup_folder is None:
        print(f"Warning: low space on {folder} ({free_str} free), "
              f"but no backup mapping exists for this path - using it anyway.")
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    backup_folder.mkdir(parents=True, exist_ok=True)
    print(f"Low space on {folder} ({free_str} free) - "
          f"using backup server instead: {backup_folder}")
    return backup_folder


def resolve_existing_file(path):
    """Return the path to read from: the new-server copy if it exists, else `path`.

    New server first, since that is where everything is written now - if a file
    exists on both servers, the old-server copy is the stale one. Falls back to
    `path` unchanged (even when missing) so "file not found" errors still name
    the path the caller asked for.
    """
    path = Path(path)
    mirrored = mirror_on_backup_server(path)
    if mirrored is not None and mirrored.exists():
        return mirrored
    return path
