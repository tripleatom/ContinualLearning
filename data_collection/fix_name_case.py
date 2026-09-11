"""
fix_name_case.py
================
Rename files and folders whose name spells an animal id in the wrong case.

Recordings are occasionally saved under a miscased animal id -- ``Cnl45`` where
every other file says ``CnL45`` -- and the misspelling propagates into the
session folder, the ``.rec`` folder, the trace inside it, the ``.DIO`` and
``.timestampoffset`` folders and every ``.dat`` beneath them. Windows and SMB
fold case, so nothing breaks day to day, but the names sort oddly, any
case-sensitive tool (or a Linux box reading the share) sees two animals, and
``conversion_list.txt`` ends up naming a ``.rec`` that a strict match cannot
find.

This walks the given roots and renames every path component containing the
animal id to the canonical spelling, deepest path first so a parent is never
renamed out from under a child still to be done. It also rewrites the miscased
name inside any ``conversion_list.txt``, so the list keeps matching the files it
names.

Case-only renames need care on a case-insensitive filesystem: ``os.rename``
usually handles them, but where it silently no-ops the rename is done via a
temporary name instead, and the result is read back from the directory listing
to confirm the on-disk spelling actually changed.

Usage
-----
  python fix_name_case.py --animal CnL45                    # dry run
  python fix_name_case.py --animal CnL45 --apply
  python fix_name_case.py --animal CnL45 --root G:\\CnL45 --apply
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

SHARE_ROOT = Path(r"\\10.129.151.88\xieluanlabs2\xl_cl\experiment_data")
DEFAULT_LOCAL_DRIVES = ("F:", "G:")


def actual_name(path: Path) -> str | None:
    """The name as the directory actually spells it, or None if absent."""
    try:
        with os.scandir(path.parent) as it:
            for entry in it:
                if entry.name.lower() == path.name.lower():
                    return entry.name
    except OSError:
        return None
    return None


def rename_exact(path: Path, new_name: str) -> str:
    """Rename `path` to `new_name`, which may differ only in case.

    Returns "renamed", "already", or raises. A plain ``os.rename`` to a
    case-variant of the same name is a no-op on some filesystems, so the result
    is verified against the directory listing and retried through a temporary
    name if the spelling did not actually change.
    """
    if actual_name(path) == new_name:
        return "already"
    target = path.with_name(new_name)

    if path.name.lower() != new_name.lower() and target.exists():
        raise FileExistsError(f"{target} already exists and is a different file")

    os.rename(path, target)
    if actual_name(target) == new_name:
        return "renamed"

    # Case-only rename did not take: go via a name nothing else can collide with.
    staging = path.with_name(f"{new_name}.case-fix-tmp")
    os.rename(target, staging)
    os.rename(staging, target)
    if actual_name(target) != new_name:
        raise OSError(f"could not change the on-disk spelling of {path}")
    return "renamed"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fix miscased animal ids in file and folder names.")
    parser.add_argument("--animal", required=True,
                        help="canonical spelling, e.g. CnL45")
    parser.add_argument("--root", type=Path, action="append", dest="roots",
                        help="root to walk; repeatable. Default: the animal's "
                             "folder on each local drive plus the share.")
    parser.add_argument("--apply", action="store_true",
                        help="perform the renames (default is a dry run)")
    args = parser.parse_args(argv)

    canon = args.animal
    roots = args.roots or ([Path(f"{d}\\{canon}") for d in DEFAULT_LOCAL_DRIVES]
                           + [SHARE_ROOT / canon])
    bad = re.compile(re.escape(canon), re.IGNORECASE)

    def fixed(name: str) -> str:
        return bad.sub(canon, name)

    targets: list[Path] = []
    lists: list[Path] = []
    for root in roots:
        if not root.is_dir():
            print(f"(skipping absent root {root})")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for name in dirnames + filenames:
                if fixed(name) != name:
                    targets.append(Path(dirpath) / name)
            for name in filenames:
                if name == "conversion_list.txt":
                    lists.append(Path(dirpath) / name)

    # Deepest first: renaming a child never invalidates its parent's path,
    # but renaming a parent would invalidate every child path already found.
    targets.sort(key=lambda p: (len(p.parts), str(p)), reverse=True)

    print(f"\n{len(targets)} path(s) to rename "
          f"({'APPLYING' if args.apply else 'dry run'}):\n")
    done = failed = 0
    for path in targets:
        new_name = fixed(path.name)
        kind = "dir " if path.is_dir() else "file"
        if not args.apply:
            print(f"  [{kind}] {path.parent}\\\n           {path.name}  ->  {new_name}")
            continue
        try:
            result = rename_exact(path, new_name)
            done += result == "renamed"
            print(f"  [{result:8}] {path.parent}\\{new_name}")
        except OSError as exc:
            failed += 1
            print(f"  [FAILED  ] {path}: {exc}")

    print(f"\n{len(lists)} conversion_list.txt checked for miscased references:")
    for path in lists:
        try:
            text = path.read_text(errors="replace")
        except OSError as exc:
            print(f"  [FAILED ] {path}: {exc}")
            failed += 1
            continue
        new_text = bad.sub(canon, text)
        if new_text == text:
            continue
        changed = [(a, b) for a, b in zip(text.splitlines(), new_text.splitlines())
                   if a != b]
        print(f"  {path}")
        for a, b in changed:
            print(f"      - {a.strip()}")
            print(f"      + {b.strip()}")
        if args.apply:
            path.write_text(new_text)
            print("      (written)")

    if args.apply:
        print(f"\nrenamed {done}, failed {failed}")
    else:
        print("\n(dry run -- nothing changed. Add --apply.)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
