"""
stage_sessions.py
=================
Copy unconverted session folders from the big slow disks onto the fast local
one, ready for the pipeline, without filling it up.

Recordings accumulate on F: and G:, but converting and sorting them is much
faster from D:. This picks the oldest sessions that still need processing and
copies them over with their contents verified.

The original is kept by default, so a session sits on two disks until it is
archived to the server. ``--remove-source`` deletes it instead, once every file
of the session has been read back from the destination and re-hashed -- which
frees F:/G: immediately but leaves the destination holding the only copy of the
raw recording until the archive step puts its NWBs on the server.

How much it takes is bounded rather than fixed. The pipeline turns each raw
session into per-shank NWBs on the same disk (roughly ``--reserve`` GB), and a
session is normally converted and archived before the next is started, so the
budget keeps room for two sessions' NWBs plus ``--keep-free`` GB of slack:

    room for copies = free - keep_free - 2 x reserve

Sessions are skipped when they are already converted (an NWB is present),
already staged at the destination, already on the server, or when the same
session name exists under more than one source root -- a split session, where
neither copy is the whole recording and staging half of it would convert into
an NWB missing an epoch.

Every file is hashed as it is read and the copy is read back and hashed again;
a file that does not match is reported and removed, leaving the original alone.

Usage
-----
  python stage_sessions.py --animal CnL43                      # plan only
  python stage_sessions.py --animal CnL43 --apply
  python stage_sessions.py --animal CnL43 --sessions 20260913,20260914 --apply
  python stage_sessions.py --animal CnL43 --keep-free 400 --apply
  python stage_sessions.py --animal CnL43 --apply --remove-source
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from archive_nwb_sessions import SHARE_ROOT, gb
from migrate_local_session import _copy_hashed, _hash_file, wait_while_recording

DEFAULT_SOURCES = ("F:", "G:")
DEFAULT_DEST = Path(r"D:\cl\ephys")
KEEP_FREE_GB = 250          # slack left on the fast disk
RESERVE_GB = 110            # NWBs one converted session adds back


def sessions_under(root: Path, animal: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for entry in sorted(root.iterdir()):
        stem, _, date8 = entry.name.rpartition("_")
        if entry.is_dir() and stem.lower() == animal.lower() and date8.isdigit():
            out[entry.name] = entry
    return out


def folder_size(folder: Path) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def _file_map(folder: Path) -> dict[str, int]:
    return {str(p.relative_to(folder)): p.stat().st_size
            for p in folder.rglob("*") if p.is_file()}


def _same_contents(a: Path, b: Path) -> bool:
    """Same files at the same sizes -- a cheap 'looks fully copied' check."""
    return bool(_file_map(a)) and _file_map(a) == _file_map(b)


def _bytes_to_copy(src: Path, dst: Path) -> int:
    """Bytes still missing from `dst`.

    A part-copied session must be charged only for what is left: its copied
    bytes are already out of the destination's free space, so counting the
    whole session again would make a resumed run drop sessions that do fit.
    """
    have = _file_map(dst) if dst.is_dir() else {}
    return sum(size for rel, size in _file_map(src).items() if have.get(rel) != size)


def plan(sources: list[Path], dest: Path, animal: str, only: set[str] | None,
         keep_free_gb: float, reserve_gb: float, max_sessions: int | None,
         remove_source: bool = False):
    """Return (chosen, skipped) -- chosen is what fits the budget, oldest first."""
    found: dict[str, list[Path]] = {}
    for root in sources:
        for name, folder in sessions_under(root, animal).items():
            found.setdefault(name, []).append(folder)

    candidates, skipped = [], []
    for name in sorted(found):
        places = found[name]
        date8 = name.rpartition("_")[2]
        if only and date8 not in only and date8[2:] not in only:
            continue
        if len(places) > 1:
            skipped.append((name, f"split across {len(places)} disks "
                                  f"({', '.join(str(p.drive) for p in places)}) -- "
                                  f"neither copy is the whole session"))
            continue
        folder = places[0]
        if any(folder.glob("*.nwb")):
            skipped.append((name, "already converted (has NWBs)"))
            continue
        # Only a complete copy counts as staged. A half-finished one -- or the
        # empty folder a discarded copy leaves behind -- is staged again, and
        # stage() verifies what is already there rather than re-copying it.
        staged = dest / name
        if staged.is_dir() and _same_contents(folder, staged):
            # With --remove-source it is still work: this run has no record of
            # the earlier copy's hashes, so stage() re-reads both sides before
            # the original can be deleted. It copies nothing.
            if not remove_source:
                skipped.append((name, f"already staged at {staged}"))
                continue
        on_server = SHARE_ROOT / animal / date8[2:] / name
        if on_server.is_dir():
            skipped.append((name, f"already on the server ({on_server})"))
            continue
        candidates.append((name, folder, folder_size(folder),
                           _bytes_to_copy(folder, staged)))

    free = shutil.disk_usage(dest).free
    budget = free - (keep_free_gb + 2 * reserve_gb) * 1024 ** 3
    chosen, used = [], 0
    for name, folder, size, todo in candidates:
        if max_sessions is not None and len(chosen) >= max_sessions:
            skipped.append((name, "beyond --max-sessions"))
            continue
        if used + todo > budget:
            skipped.append((name, f"would not fit the space budget "
                                  f"({gb(todo)} still to copy, "
                                  f"{gb(int(budget - used))} left)"))
            continue
        chosen.append((name, folder, size))
        used += todo
    return chosen, skipped, free, budget


def drop_source(folder: Path, target_root: Path, log=print) -> bool:
    """Delete a staged session's original, once the copy accounts for all of it.

    stage() has already hashed every file both ways, so this re-checks only the
    shape of the copy -- same relative paths, same sizes -- as a last guard
    against deleting a source whose destination lost a file in between.
    """
    src_map, dst_map = _file_map(folder), _file_map(target_root)
    missing = {rel for rel in src_map if dst_map.get(rel) != src_map[rel]}
    if missing:
        log(f"      NOT deleting {folder}: {len(missing)} file(s) do not match at "
            f"{target_root}, e.g. {sorted(missing)[0]}")
        return False
    wait_while_recording(log, exclude=[target_root])
    shutil.rmtree(folder)
    log(f"      deleted the original {folder}  ({gb(sum(src_map.values()))} freed "
        f"on {folder.drive})")
    parent = folder.parent
    try:
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
            log(f"      removed the empty {parent}")
    except OSError as exc:
        log(f"      left {parent} in place: {exc}")
    return True


def stage(name: str, folder: Path, dest_root: Path, log=print,
          remove_source: bool = False) -> bool:
    """Copy one session to dest_root/name, verifying every file.

    The original is deleted only with `remove_source`, and only if every file
    verified -- one mismatch anywhere leaves the whole source alone.
    """
    target_root = dest_root / name
    ok = True
    for src in sorted((p for p in folder.rglob("*") if p.is_file()),
                      key=lambda p: str(p.relative_to(folder))):
        # Exclude what this copy is writing: a .rec it has just written is
        # fresh on disk and would otherwise look like a recording in progress.
        wait_while_recording(log, exclude=[target_root])
        rel = src.relative_to(folder)
        target = target_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        size = src.stat().st_size
        if target.exists() and target.stat().st_size == size \
                and _hash_file(target) == _hash_file(src):
            log(f"      already staged, verified: {rel}")
            continue
        log(f"      copying {rel}  ({gb(size)}) ...")
        src_hash = _copy_hashed(src, target)
        if _hash_file(target) != src_hash:
            target.unlink(missing_ok=True)      # the original is untouched
            log(f"      MISMATCH {rel}: copy discarded, original left on "
                f"{src.drive}")
            ok = False
            continue
        log(f"      staged {rel}  ({gb(size)}), content verified")
    if remove_source:
        if ok:
            ok &= drop_source(folder, target_root, log)
        else:
            log(f"      keeping the original {folder}: not every file verified")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--animal", required=True)
    parser.add_argument("--source", type=Path, action="append", dest="sources",
                        help=f"root to look in; repeatable. Default: "
                             f"{'/'.join(DEFAULT_SOURCES)}\\<animal>")
    parser.add_argument("--dest", type=Path, default=None,
                        help=f"default: {DEFAULT_DEST}\\<animal>")
    parser.add_argument("--sessions", default="", help="comma-separated YYYYMMDD")
    parser.add_argument("--keep-free", type=float, default=KEEP_FREE_GB,
                        help=f"GB to leave free on the destination (default {KEEP_FREE_GB})")
    parser.add_argument("--reserve", type=float, default=RESERVE_GB,
                        help=f"GB one session's NWBs add back (default {RESERVE_GB})")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="copy (default: plan only)")
    parser.add_argument("--remove-source", action="store_true",
                        help="delete the source session once every file of it has "
                             "been read back from the destination and re-hashed")
    args = parser.parse_args(argv)

    sources = args.sources or [Path(f"{d}\\{args.animal}") for d in DEFAULT_SOURCES]
    dest = args.dest or (DEFAULT_DEST / args.animal)
    dest.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.sessions.split(",") if s.strip()} or None

    chosen, skipped, free, budget = plan(sources, dest, args.animal, only,
                                         args.keep_free, args.reserve, args.max_sessions,
                                         args.remove_source)
    print(f"destination : {dest}   {gb(free)} free")
    print(f"budget      : {gb(int(budget))} for copies "
          f"(keeping {args.keep_free:.0f} GB free + {2 * args.reserve:.0f} GB "
          f"for two sessions' NWBs)")
    if args.remove_source:
        print("sources     : deleted once the copy verifies -- until each session "
              "is archived, the destination holds the only copy")
    total = sum(s for _n, _f, s in chosen)
    for name, folder, size in chosen:
        print(f"  stage {name:22} {gb(size):>10}   from {folder}")
    for name, why in skipped:
        print(f"  skip  {name:22} {why}")
    print(f"\n{len(chosen)} session(s), {gb(total)}; "
          f"{gb(int(free - total))} would remain free")
    if not args.apply:
        print("(plan only -- add --apply to copy)")
        return 0

    ok = True
    for name, folder, size in chosen:
        print(f"\n=== {name}  ({gb(size)}) ===")
        ok &= stage(name, folder, dest, remove_source=args.remove_source)
    print(f"\ndone; {gb(shutil.disk_usage(dest).free)} free on {dest.drive}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
