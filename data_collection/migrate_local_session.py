"""
migrate_local_session.py
========================
Move a whole session folder from a local disk onto the share, leaving the raw
``.rec`` traces behind to be deleted.

This is the sibling of ``archive_nwb_sessions.py``, for the other arrangement a
session can be in. There, the NWBs sat on a local disk and the ``.rec`` folders
were already on the share, so only the NWBs travelled. Here the whole day is
still local -- NWBs, ``.rec`` folders, ``bad_channels.txt``, the conversion list
-- and nothing is on the share yet::

    D:\\cl\\ephys\\CnL45_20260816\\
        CnL45_20260816sh0..7.nwb
        bad_channels.txt, conversion_list.txt
        CnL45_passive_20260816_101804.rec\\
            CnL45_passive_20260816_101804.rec          <- raw trace, dropped
            CnL45_passive_20260816_101804.DIO\\         <- kept
            CnL45_passive_20260816_101804.timestampoffset\\
            params.json, CnL45_passive_20260816_101804.rec.txt

Everything except the raw traces is copied to
``<share>\\experiment_data\\<animal>\\<YYMMDD>\\<folder name>``, keeping its layout,
so the ``.rec`` folders arrive complete apart from the trace itself. That
matters: ``trodes_io.DIO`` and the sleep pipeline read the ``.rec`` folder for
its DIO and never touch the trace, so the day stays fully usable.

The traces are deleted only after every other file is on the share AND the NWBs
have been re-read there and matched against the conversion list -- the trace is
the only other copy of that signal, so it goes last and only once its
replacement is proven.

A session is refused unless all of these hold:

  * ``conversion_list.txt`` is present;
  * every ``.rec`` it names exists in the folder, and no extra ``.rec`` sits
    there unnamed by it (an unconverted recording must never be deleted);
  * all expected shanks are present (default 8, less ``--ignore-shanks``);
  * every shank NWB opens, has an ``ElectricalSeries``, and its sample count
    equals the conversion list's total.

Usage
-----
  python migrate_local_session.py D:\\cl\\ephys\\CnL45_20260816
  python migrate_local_session.py D:\\cl\\ephys\\CnL45_20260816 --apply
  python migrate_local_session.py D:\\cl\\ephys\\* --apply --i-have-backups
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from archive_nwb_sessions import (
    SHARE_ROOT,
    check_nwb_set,
    copy_verified,
    gb,
    nwb_shank_files,
    parse_conversion_list,
    raw_rec_files,
)

EXPECTED_SHANKS = 8


def describe(folder: Path, expect_shanks: int, ignore: frozenset[int]):
    """Work out where this folder belongs and whether it may be migrated.

    Returns (dest, traces, payload, expect_samples, problems) where `traces` are
    the raw files to delete and `payload` is every other file, each as a
    (source, path relative to the folder) pair.
    """
    problems: list[str] = []
    animal, _, date8 = folder.name.rpartition("_")
    if not (len(date8) == 8 and date8.isdigit() and animal):
        return None, {}, [], 0, [f"folder name is not <animal>_<YYYYMMDD>: {folder.name}"]

    dest = SHARE_ROOT / animal / date8[2:] / folder.name

    # The list travels with everything else, so on a second pass -- traces kept
    # the first time, or an interrupted run -- it is already on the share. Look
    # there too, or the session could never be finished.
    listing = folder / "conversion_list.txt"
    if not listing.is_file():
        listing = dest / "conversion_list.txt"
    listed = parse_conversion_list(listing) if listing.is_file() else {}
    if not listed:
        problems.append("no conversion_list.txt, locally or on the share "
                        "(or it names no .rec)")
    expect_samples = sum(listed.values())

    traces = raw_rec_files(folder)
    if listed:
        absent = sorted(set(listed) - set(traces))
        extra = sorted(set(traces) - set(listed))
        if absent:
            problems.append(f"conversion list names .rec not in the folder: {absent}")
        if extra:
            problems.append(f".rec in the folder is NOT in the conversion list, so "
                            f"it was never exported: {extra}")

    # Likewise the NWBs: judge the union of both sides, the share winning, so a
    # half-finished move is neither mistaken for complete nor stuck.
    nwbs = dict(nwb_shank_files(folder, folder.name))
    nwbs.update(nwb_shank_files(dest, folder.name))
    if listed:
        found, unusable = check_nwb_set(nwbs, expect_samples, expect_shanks, ignore)
        problems += found
    else:
        unusable = set()

    trace_paths = {p.resolve() for p in traces.values()}
    payload = []
    for root, _dirs, files in os.walk(folder):
        for name in files:
            src = Path(root) / name
            if src.resolve() in trace_paths:
                continue
            payload.append((src, src.relative_to(folder)))
    payload.sort(key=lambda pair: str(pair[1]))
    return dest, traces, payload, expect_samples, problems


def migrate(folder: Path, expect_shanks: int, ignore: frozenset[int],
            apply: bool, allow_delete: bool, log=print) -> bool:
    dest, traces, payload, expect_samples, problems = describe(
        folder, expect_shanks, ignore)

    log(f"\n{'=' * 74}\n{folder}")
    if dest:
        log(f"  -> {dest}")
    if problems:
        log("  REFUSED:")
        for p in problems:
            log(f"      - {p}")
        return False

    payload_bytes = sum(s.stat().st_size for s, _r in payload)
    trace_bytes = sum(p.stat().st_size for p in traces.values())
    log(f"  moves  : {len(payload)} file(s)  {gb(payload_bytes)}")
    log(f"  deletes: {len(traces)} raw trace(s)  {gb(trace_bytes)}")
    log(f"  expects: {expect_samples:,} samples per shank")

    if not apply:
        for src, rel in payload[:12]:
            log(f"      would move {rel}  ({gb(src.stat().st_size)})")
        if len(payload) > 12:
            log(f"      ... and {len(payload) - 12} more")
        for name in sorted(traces):
            log(f"      would delete {traces[name].relative_to(folder)}"
                f"  ({gb(traces[name].stat().st_size)})")
        return True

    # 1. everything but the traces, each verified before its source is dropped
    for src, rel in payload:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == src.stat().st_size:
            log(f"      already on share, dropping local: {rel}")
            src.unlink()
            continue
        expect = expect_samples if src.suffix.lower() == ".nwb" else None
        if expect is None:
            shutil.copy2(src, target)
            if target.stat().st_size != src.stat().st_size:
                raise IOError(f"size mismatch after copy: {target}")
        else:
            log(f"      copying {rel}  ({gb(src.stat().st_size)}) ...")
            copy_verified(src, target, expect, log=log)
        src.unlink()

    # 2. re-verify on the share before anything irreversible
    problems, _unusable = check_nwb_set(nwb_shank_files(dest, folder.name),
                                        expect_samples, expect_shanks, ignore)
    if problems:
        log("  NOT deleting traces -- share NWBs failed re-verification:")
        for p in problems:
            log(f"      - {p}")
        return False
    log(f"  {len(nwb_shank_files(dest, folder.name))} shank(s) re-verified on the share")

    # 3. now the traces
    if not allow_delete:
        log("  traces kept: --i-have-backups was not given")
        return True
    for name in sorted(traces):
        path = traces[name]
        size = path.stat().st_size
        path.unlink()
        log(f"      deleted {path.relative_to(folder)}  ({gb(size)})")

    # 4. tidy up whatever is now empty. Emptiness is re-read from disk rather
    # than taken from os.walk's lists, which still name the children removed
    # earlier in this same bottom-up pass.
    for root, _dirs, _files in os.walk(folder, topdown=False):
        try:
            if not os.listdir(root):
                os.rmdir(root)
        except OSError:
            pass
    # Best-effort, and deliberately never fatal: by this point the NWBs are on
    # the share, verified there, and the traces are gone -- the migration has
    # succeeded. Windows also reports a just-removed directory as "access
    # denied" while a handle from the walk above lingers, so failing here would
    # report a completed migration as broken and invite a needless re-run.
    try:
        if folder.exists() and not any(folder.iterdir()):
            folder.rmdir()
            log(f"  removed empty {folder}")
    except OSError as exc:
        log(f"  (left {folder} in place: {exc})")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Move a local session folder to the share, dropping raw .rec traces.")
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--expect-shanks", type=int, default=EXPECTED_SHANKS)
    parser.add_argument("--ignore-shanks", default="")
    parser.add_argument("--apply", action="store_true",
                        help="do it (default is a dry run)")
    parser.add_argument("--i-have-backups", action="store_true",
                        help="also delete the raw traces once everything else is "
                             "on the share and the NWBs re-verify there")
    args = parser.parse_args(argv)

    ignore = frozenset(int(s) for s in args.ignore_shanks.split(",") if s.strip())
    folders = [f for f in args.folders if f.is_dir()]
    missing = [f for f in args.folders if not f.is_dir()]
    for f in missing:
        print(f"not a folder, skipping: {f}")

    ok = True
    for folder in folders:
        ok &= migrate(folder, args.expect_shanks, ignore,
                      args.apply, args.i_have_backups)

    if not args.apply:
        print("\n(dry run -- nothing changed. Add --apply, and "
              "--i-have-backups to also drop the traces.)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
