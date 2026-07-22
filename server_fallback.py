"""Shared no-space fallback for scripts writing to the lab file server.

When the primary server (\\\\10.129.151.108\\xieluanlabs) is low on space,
outputs are redirected to the backup server (\\\\10.129.151.88\\xieluanlabs2),
keeping the same subpath after the share name. Reads fall back the same way
if a file is missing from the primary but present on the backup.

Usage:
    from server_fallback import resolve_output_folder, resolve_existing_file, mirror_on_backup_server

    out_dir = resolve_output_folder(some_folder)   # creates it; may be the backup mirror
    ...
    try:
        with open(out_dir / "file.pkl", "wb") as f:
            pickle.dump(data, f)
    except OSError as e:
        if e.errno != errno.ENOSPC:
            raise
        backup_dir = mirror_on_backup_server(out_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        with open(backup_dir / "file.pkl", "wb") as f:
            pickle.dump(data, f)
"""
import shutil
from pathlib import Path

PRIMARY_SERVER_ROOT = r"\\10.129.151.108\xieluanlabs"
BACKUP_SERVER_ROOT  = r"\\10.129.151.88\xieluanlabs2"
MIN_FREE_BYTES      = 5 * 1024 ** 3  # switch to backup server once less than this remains free


def mirror_on_backup_server(path):
    """Map a path under PRIMARY_SERVER_ROOT to the equivalent path under BACKUP_SERVER_ROOT."""
    path_str = str(path)
    if path_str.lower().startswith(PRIMARY_SERVER_ROOT.lower()):
        return Path(BACKUP_SERVER_ROOT + path_str[len(PRIMARY_SERVER_ROOT):])
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
    """Return folder (created), or its mirror on the backup server if low on space."""
    folder = Path(folder)
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
    """Return path if it exists, else its mirror on the backup server if that exists."""
    path = Path(path)
    if path.exists():
        return path
    backup = mirror_on_backup_server(path)
    if backup is not None and backup.exists():
        return backup
    return path
