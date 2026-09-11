"""Shared server routing for scripts reading/writing the lab file server.

The OLD server (\\\\10.129.151.108\\xieluanlabs) is FULL, so ALL new output goes
to the NEW server (\\\\10.129.151.88\\xieluanlabs2), keeping the same subpath
after the share name. Nothing is written to the old server any more - it stays
readable for the raw recordings and previously-computed data that live there.

Recordings are also commonly PROCESSED from a fast local disk (G:\\CnL46\\
CnL46_20260727) while the day's permanent home is its experiment_data folder on
the share. Derived output written under such a local session folder is likewise
redirected onto the share, so the local disk only ever holds the raw recording:

    G:\\CnL46\\CnL46_20260727\\low_freq\\...
      -> \\\\10.129.151.88\\...\\experiment_data\\CnL46\\260727\\CnL46_20260727\\low_freq\\...

See LOCAL SESSION FOLDERS below for how the animal/date folders are derived and
how to override the mapping for a day that does not follow the convention.

Reads resolve on either server and on the local disk, server first: every write
now lands on the server, so when a file exists in both places the server copy is
the current one.

Usage:
    from server_fallback import resolve_output_folder, resolve_existing_file, mirror_on_backup_server

    out_dir = resolve_output_folder(some_folder)   # redirected to the new server
    with open(out_dir / "file.pkl", "wb") as f:
        pickle.dump(data, f)

The per-script `except OSError (ENOSPC) -> mirror_on_backup_server(...)` handlers
still work, but they are now a genuine last resort: once output is already on
the new server there is nowhere left to spill to, so ENOSPC re-raises.

Set WRITE_TO_NEW_SERVER = False to restore the previous behaviour (write on the
old server, spilling over to the new one only when free space runs low), and
WRITE_OUTPUT_TO_SERVER = False to keep local-disk output beside the recording.
"""
import re
import shutil
from pathlib import Path

OLD_SERVER_ROOT = r"\\10.129.151.108\xieluanlabs"    # full - read only
NEW_SERVER_ROOT = r"\\10.129.151.88\xieluanlabs2"    # write everything here

# Send every output to the new server, regardless of free space on the old one.
WRITE_TO_NEW_SERVER = True

# Only consulted when WRITE_TO_NEW_SERVER is False.
MIN_FREE_BYTES = 5 * 1024 ** 3


# =====================================================
# LOCAL SESSION FOLDERS -> SERVER
# =====================================================
# Every recording folder on a local disk has a matching folder on the share:
#
#   local :  G:\CnL46\CnL46_20260727
#   server:  \\10.129.151.88\xieluanlabs2\xl_cl\experiment_data\CnL46\260727\CnL46_20260727
#
# The animal and date folders are derived from the session folder's own name
# ("<animal>_<YYYYMMDD>"), then matched case-insensitively against what the
# share actually holds - a local folder is occasionally spelled differently
# ("Cnl45_20260811" -> "CnL45\260811\CnL45_20260811") or filed under another
# animal's directory, in which case the parent folder name is tried as the
# animal too. A date folder that does not exist yet is created; an animal
# folder that cannot be matched is NOT invented, since that usually means a
# typo or an unreachable share - the local path is then used unchanged.
EXPERIMENT_DATA_ROOT = Path(NEW_SERVER_ROOT) / "xl_cl" / "experiment_data"

# Set False to keep output from local-disk sessions on the local disk.
WRITE_OUTPUT_TO_SERVER = True

# Explicit local -> server session folders, for days the derivation above
# cannot work out (an unusual name, or a session stored outside
# experiment_data). Sub-folders follow automatically: giving the SESSION
# folder here routes its low_freq/, MUA/, sync pickles and everything else.
#   SESSION_SERVER_FOLDERS = {
#       r"G:\CnL46\Cnl45_20260811":
#           r"\\10.129.151.88\xieluanlabs2\xl_cl\experiment_data\CnL45\260811\CnL45_20260811",
#   }
SESSION_SERVER_FOLDERS = {}

# "<animal>_<YYYYMMDD>", the name every recording folder uses.
_SESSION_FOLDER_RE = re.compile(r"^(?P<animal>.+)_(?P<date>\d{8})$")

# Each lookup listing a share directory costs a network round trip, and these
# run once per file, so both levels are memoised for the life of the process.
_child_cache = {}
_session_cache = {}


def _is_server_path(path):
    text = str(path).lower()
    return (text.startswith(OLD_SERVER_ROOT.lower())
            or text.startswith(NEW_SERVER_ROOT.lower()))


def _child_named(folder, name):
    """Child of `folder` called `name` ignoring case, or None if there is none.

    The whole listing is cached per folder, so checking several candidate names
    against the share costs one round trip rather than one each.
    """
    key = str(folder).lower()
    if key not in _child_cache:
        try:
            _child_cache[key] = {child.name.lower(): child
                                 for child in Path(folder).iterdir()}
        except OSError:              # share unreachable, or not a directory
            _child_cache[key] = {}
    return _child_cache[key].get(name.lower())


def _split_session_path(path):
    """Split into (session folder, subpath below it), or (None, None)."""
    path = Path(path)
    for candidate in (path, *path.parents):
        if _SESSION_FOLDER_RE.match(candidate.name):
            return candidate, path.relative_to(candidate)
    return None, None


def _session_override(session_folder):
    wanted = str(session_folder).rstrip("\\/").lower()
    for local, server in SESSION_SERVER_FOLDERS.items():
        if str(local).rstrip("\\/").lower() == wanted:
            return Path(server)
    return None


def server_session_folder(session_folder):
    """Server experiment_data folder for a recording folder, or None.

    None means "no server home could be identified" - the caller should then
    use the path it already has.
    """
    session_folder = Path(session_folder)
    key = str(session_folder).lower()
    if key in _session_cache:
        return _session_cache[key]

    resolved = _session_override(session_folder)
    if resolved is None:
        match = _SESSION_FOLDER_RE.match(session_folder.name)
        if match is not None:
            date6 = match.group("date")[2:]          # YYYYMMDD -> YYMMDD
            # The animal is normally the session-name prefix; the folder the
            # session sits in is the fallback, for one filed elsewhere.
            for animal in (match.group("animal"), session_folder.parent.name):
                animal_dir = _child_named(EXPERIMENT_DATA_ROOT, animal)
                if animal_dir is None:
                    continue
                date_dir = _child_named(animal_dir, date6) or animal_dir / date6
                resolved = (_child_named(date_dir, session_folder.name)
                            or date_dir / session_folder.name)
                break

    _session_cache[key] = resolved
    return resolved


def server_twin(path):
    """The server path for something inside a local session folder, or None."""
    if not WRITE_OUTPUT_TO_SERVER or _is_server_path(path):
        return None
    session_folder, tail = _split_session_path(path)
    if session_folder is None:
        return None
    server_folder = server_session_folder(session_folder)
    if server_folder is None:
        return None
    return server_folder if str(tail) == "." else server_folder / tail


def mirror_on_backup_server(path):
    """Map `path` onto the server, keeping its subpath.

    Two mappings, in order: OLD_SERVER_ROOT -> NEW_SERVER_ROOT, and a local
    session folder -> its experiment_data folder on the share.

    Returns None when neither applies - the path is already on the new server,
    or it is local but has no identifiable server home - i.e. "there is nowhere
    else to put this".
    """
    path_str = str(path)
    if path_str.lower().startswith(OLD_SERVER_ROOT.lower()):
        return Path(NEW_SERVER_ROOT + path_str[len(OLD_SERVER_ROOT):])
    return server_twin(path)


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
    """Return the folder to write into, created: the server mirror of `folder`.

    Old-server paths and local session folders are redirected unconditionally -
    no disk_usage probe, which also skips a slow network stat call. Paths that
    are already on the new server, and local paths with no identifiable server
    home, are used as given.
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
    """Return the path to read from: the server copy if it exists, else `path`.

    The server first, since that is where everything is written now - if a file
    exists both there and on the old server or a local disk, the server copy is
    the current one. Falls back to `path` unchanged (even when missing) so "file
    not found" errors still name the path the caller asked for, and so output
    computed before this routing existed is still found where it sits.
    """
    path = Path(path)
    mirrored = mirror_on_backup_server(path)
    if mirrored is not None and mirrored.exists():
        return mirrored
    return path


if __name__ == "__main__":
    # `python server_fallback.py <path> [...]` answers "where would this be
    # written, and where would it be read from?" without writing anything -
    # the quickest way to check a day's routing before running a stage.
    import sys

    for argument in sys.argv[1:]:
        target = mirror_on_backup_server(argument) or Path(argument)
        source = resolve_existing_file(argument)
        print(f"\n{argument}")
        print(f"  write -> {target}"
              + ("" if str(target) != str(argument) else "   (unchanged)"))
        print(f"  read  <- {source}"
              + ("" if Path(source).exists() else "   (does not exist)"))
