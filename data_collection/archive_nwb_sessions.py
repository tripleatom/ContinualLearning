"""
archive_nwb_sessions.py
=======================
Move exported NWBs off the fast local disk onto the share, then drop the raw
``.rec`` traces they replace -- freeing space on both.

A CnL46 day currently lives in two places::

    G:\\CnL46\\CnL46_20260817\\                 CnL46_20260817sh0..7.nwb, conversion_list.txt
    \\\\10.129.151.88\\...\\CnL46\\260817\\CnL46_20260817\\
                                             CnL46_passive_20260817_144853.rec\\
                                                 CnL46_passive_20260817_144853.rec   <- raw trace
                                                 CnL46_passive_20260817_144853.DIO\\
                                                 CnL46_passive_20260817_144853.timestampoffset\\
                                                 params.json
                                             low_freq\\, MUA\\, bad_channels.txt, ...

The NWB is the same signal as the ``.rec``: ``acquisition/ElectricalSeries/data``
is the full-band int16 trace at the acquisition rate, one file per shank, with
that day's bad channels dropped. Once every shank is on the share and verified,
the raw ``.rec`` is redundant.

What this does, per session, in three stages that must be run in order:

1. ``--move``   copies ``*.nwb`` + ``conversion_list.txt`` from the local folder
                to the session's folder on the share, re-opens each copy there
                to confirm it reads back at the right length, deletes the local
                copy, and repoints ``sleep_day_configs.json`` at the share.
2. ``--delete-raw``  deletes the raw ``.rec`` FILES on the share, re-verifying
                the NWBs *on the share* first. Only the trace file goes: the
                ``.DIO`` and ``.timestampoffset`` folders, ``params.json`` and
                ``*.rec.txt`` stay, because ``trodes_io.DIO`` and the sleep
                pipeline read the ``.rec`` folder for its DIO, never the trace.
3. ``--drop-duplicates``  deletes leftover local files that already exist on the
                share at the same size (stale ``low_freq`` output, sync pickles).

A session is only eligible when ALL of these hold -- anything else is reported
and skipped, never half-done:

  * a ``conversion_list.txt`` is found (locally, or on the share);
  * every ``.rec`` it names exists on the share, and no *extra* ``.rec`` sits
    there unnamed by it (an unconverted recording must never be deleted);
  * all expected shanks are present (default 8, less any ``--ignore-shanks``);
  * every shank NWB opens, has an ``ElectricalSeries``, and its sample count
    equals the conversion list's total.

That last check is what makes the deletion safe: a truncated, empty or
still-running export fails it. Empty 800-byte NWB shells and missing shanks are
caught this way, not by trusting the file name.

Not every day has every shank -- CnL46 has no shank 7 from 20260806 on. Pass
``--ignore-shanks 7`` so those days archive from the shanks they do have. An
ignored shank that exists and verifies is still archived; one that is only an
empty shell is left on the local disk, since a shell on the share would fail
confusingly in ``read_nwb_recording`` later.

Deleting raw traces is irreversible, so ``--delete-raw`` additionally requires
``--i-have-backups``. Nothing is deleted without it.

Usage
-----
  python archive_nwb_sessions.py                       # plan only, touches nothing
  python archive_nwb_sessions.py --move
  python archive_nwb_sessions.py --delete-raw --i-have-backups
  python archive_nwb_sessions.py --drop-duplicates
  python archive_nwb_sessions.py --sessions 260817,260816 --move
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import h5py
except ImportError:  # pragma: no cover - environment problem, not a data problem
    sys.exit("h5py is required. Run this from the `ms10` conda environment:\n"
             r"  C:\Users\Windows\.conda\envs\ms10\python.exe archive_nwb_sessions.py")

DEFAULT_LOCAL_DRIVES = ("F:", "G:")   # recordings sit on whichever has room
SHARE_ROOT = Path(r"\\10.129.151.88\xieluanlabs2\xl_cl\experiment_data")
DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / "sleep" / "sleep_day_configs.json"
DEFAULT_ANIMAL = "CnL46"
EXPECTED_SHANKS = 8

# The dataset every real export has; an empty NWB shell has no such node.
DATA_PATH = "acquisition/ElectricalSeries/data"

# Files inside a .rec folder that are NOT the raw trace and must be kept.
# (Everything except "<name>.rec" itself, but spelled out so the intent is clear.)
KEEP_IN_REC_FOLDER = (".dio", ".timestampoffset", ".json", ".txt")

# "   1. CnL46_passive_20260817_144853.rec: 129350860 samples  [nwb 0-129350859]"
# Older lists say "timestamps" instead of "samples".
_ENTRY_RE = re.compile(
    r"^\s*\d+\.\s*(?P<rec>\S+\.rec)\s*:\s*(?P<n>[\d,]+)\s*(?:samples|timestamps)\b",
    re.MULTILINE,
)
_SHANK_RE = re.compile(r"sh(\d+)\.nwb$", re.IGNORECASE)


def gb(n_bytes: int) -> str:
    return f"{n_bytes / 1024 ** 3:,.1f} GB"


def tree_bytes(folder: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(folder):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------------
# Reading what a session claims to contain
# ---------------------------------------------------------------------------

def parse_conversion_list(path: Path) -> dict[str, int]:
    """``{"CnL46_passive_..._144853.rec": 129350860, ...}`` from a conversion list."""
    text = path.read_text(errors="replace")
    return {m.group("rec"): int(m.group("n").replace(",", ""))
            for m in _ENTRY_RE.finditer(text)}


def raw_rec_files(inner: Path) -> dict[str, Path]:
    """Raw ``.rec`` trace files on the share, keyed by file name.

    A ``.rec`` entry is normally a folder holding the trace of the same name,
    but the two names do not always agree (``CnL46_passive2_..._150252.rec/``
    holds ``CnL46_passive_..._150252.rec``), so the folder is searched rather
    than assumed. A bare ``.rec`` file is accepted too.
    """
    found: dict[str, Path] = {}
    if not inner.is_dir():
        return found
    for entry in sorted(inner.glob("*.rec")):
        if entry.is_dir():
            for trace in sorted(entry.glob("*.rec")):
                if trace.is_file():
                    found[trace.name] = trace
        elif entry.is_file():
            found[entry.name] = entry
    return found


def archived_trace_stems(inner: Path) -> set[str]:
    """Traces this tool has already retired, by stem.

    A ``.rec`` folder that still holds its ``.DIO`` but no longer holds a trace
    is a finished session, not a broken one. Telling the two apart matters: once
    a day is archived its conversion list necessarily names traces that are no
    longer there, and without this every completed day would report as though
    its recordings had gone missing -- hiding a day where one really had.
    """
    stems: set[str] = set()
    if not inner.is_dir():
        return stems
    for rec in sorted(inner.glob("*.rec")):
        if not rec.is_dir() or any(rec.glob("*.rec")):
            continue
        for child in rec.iterdir():
            if child.is_dir() and child.name.lower().endswith(".dio"):
                stems.add(child.name[:-4])
    return stems


def nwb_shank_files(folder: Path, prefix: str) -> dict[int, Path]:
    """``{shank index: path}`` for ``<prefix>sh<N>.nwb`` in folder.

    Matched case-insensitively and spelled out rather than left to ``glob``:
    folders and files are occasionally miscased (``Cnl45_20260811sh0.nwb`` under
    ``CnL45``), and relying on the filesystem to fold case would make this work
    on Windows and quietly miss those files anywhere else.
    """
    out: dict[int, Path] = {}
    if not folder.is_dir():
        return out
    pattern = re.compile(re.escape(prefix) + r"sh(\d+)\.nwb$", re.IGNORECASE)
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        m = pattern.fullmatch(path.name)
        if m:
            out[int(m.group(1))] = path
    return out


def local_session_dirs(local_roots: list[Path], animal: str) -> dict[str, list[Path]]:
    """``{YYYYMMDD: [folder, ...]}`` for this animal across every local disk.

    A date is returned with every folder that claims it, so the caller can
    refuse to guess when the same day somehow exists on two disks.
    """
    found: dict[str, list[Path]] = {}
    for root in local_roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            stem, _, date8 = entry.name.rpartition("_")
            if (len(date8) == 8 and date8.isdigit()
                    and stem.lower() == animal.lower()):
                found.setdefault(date8, []).append(entry)
    return found


def check_nwb_set(files: dict[int, Path], expect_samples: int, expect_shanks: int,
                  ignore: frozenset[int] = frozenset()) -> tuple[list[str], set[int]]:
    """Inspect a set of shank NWBs. Returns ``(problems, unusable)``.

    Every file must open, carry an ElectricalSeries, and hold exactly
    ``expect_samples`` samples -- the sum the conversion list says went in.

    ``problems`` are blockers: a required shank absent, or a required shank that
    failed. An empty list means the session may be archived.

    ``unusable`` is every shank that is absent, empty or the wrong length,
    whether required or ignored. A shank in ``ignore`` lands here instead of in
    ``problems``, and is left where it is rather than copied -- an 800-byte
    empty shell must never reach the share looking like a real export.
    """
    problems: list[str] = []
    unusable: set[int] = set()
    required = set(range(expect_shanks)) - set(ignore)

    missing = sorted(required - set(files))
    if missing:
        problems.append(f"missing shank(s) {missing}")
    unusable |= set(range(expect_shanks)) - set(files)

    for shank in sorted(files):
        path = files[shank]
        fault = None
        try:
            with h5py.File(path, "r") as handle:
                data = handle.get(DATA_PATH)
                if data is None:
                    fault = (f"no {DATA_PATH} -- empty export "
                             f"({path.stat().st_size:,} bytes)")
                elif data.shape[0] != expect_samples:
                    fault = (f"{data.shape[0]:,} samples, conversion list says "
                             f"{expect_samples:,}")
        except OSError as exc:
            fault = f"unreadable ({exc})"
        if fault:
            unusable.add(shank)
            if shank not in ignore:
                problems.append(f"sh{shank}: {fault}")
    return problems, unusable


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class SessionPlan:
    date8: str                      # 20260817
    local: Path                     # G:\CnL46\CnL46_20260817
    inner: Path                     # <share>\260817\CnL46_20260817
    prefix: str                     # CnL46_20260817
    expect_samples: int = 0
    conversion_list: Path | None = None
    local_nwb: dict[int, Path] = field(default_factory=dict)
    server_nwb: dict[int, Path] = field(default_factory=dict)
    raw_recs: dict[str, Path] = field(default_factory=dict)   # name -> trace on share
    local_raw: dict[str, Path] = field(default_factory=dict)  # name -> trace on a local disk
    listed_recs: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    unusable: set[int] = field(default_factory=set)   # absent/empty/short shanks
    archived: set[str] = field(default_factory=set)   # trace stems already retired

    @property
    def date6(self) -> str:
        return self.date8[2:]

    @property
    def eligible(self) -> bool:
        return not self.problems

    @property
    def local_nwb_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.local_nwb.values())

    @property
    def movable(self) -> dict[int, Path]:
        """Local NWBs that verified, and so are worth copying across.

        Excludes held-back shanks -- an empty shell left on the local disk is
        not work still to do, and must not read as though it were.
        """
        return {s: p for s, p in self.local_nwb.items() if s not in self.unusable}

    @property
    def movable_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.movable.values())

    @property
    def server_nwb_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.server_nwb.values())

    @property
    def raw_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.raw_recs.values())

    @property
    def local_raw_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.local_raw.values())

    @property
    def done(self) -> bool:
        """Fully archived: NWBs on the share and every trace already retired."""
        return (bool(self.server_nwb) and not self.raw_recs and bool(self.listed_recs)
                and all(name.rsplit(".rec", 1)[0] in self.archived
                        for name in self.listed_recs))

    @property
    def unbacked_raw(self) -> bool:
        """Raw traces sitting on a local disk with nothing matching on the share.

        Not this script's job to fix -- it only ever deletes a trace the share
        holds -- but it must be visible, since such a recording has no second
        copy anywhere.
        """
        return bool(self.local_raw) and not self.raw_recs

    @property
    def moved(self) -> bool:
        """Every usable NWB is on the share, with none still to go across.

        A held-back shank (empty shell, wrong length) is left on the local disk
        deliberately, so it must not keep the session looking unmoved.
        """
        return bool(self.server_nwb) and not (set(self.local_nwb) - self.unusable)


def session_dates(local_dirs: dict[str, list[Path]], server_root: Path,
                  animal: str) -> list[str]:
    """Every ``YYYYMMDD`` this animal has, on a local disk or on the share.

    A day whose NWBs were archived on an earlier run no longer exists locally,
    so the share has to be enumerated too or its raw traces would never come up
    for deletion.
    """
    dates = set(local_dirs)
    if server_root.is_dir():
        for day in server_root.iterdir():
            if not (day.is_dir() and len(day.name) == 6 and day.name.isdigit()):
                continue
            for inner in day.iterdir():
                stem, _, date8 = inner.name.rpartition("_")
                if (inner.is_dir() and len(date8) == 8 and date8.isdigit()
                        and stem.lower() == animal.lower()):
                    dates.add(date8)
    return sorted(dates)


def build_plan(local_roots: list[Path], server_root: Path, animal: str,
               expect_shanks: int, only: set[str] | None,
               ignore: frozenset[int] = frozenset()) -> list[SessionPlan]:
    plans = []
    local_dirs = local_session_dirs(local_roots, animal)
    for date8 in session_dates(local_dirs, server_root, animal):
        if only and date8 not in only and date8[2:] not in only:
            continue

        prefix = f"{animal}_{date8}"
        candidates = local_dirs.get(date8, [])
        # Keep the animal's canonical spelling for the share; the local folder
        # keeps whatever it is actually called.
        local = candidates[0] if candidates else (local_roots[0] / prefix)
        inner = server_root / date8[2:] / prefix
        plan = SessionPlan(date8=date8, local=local, inner=inner, prefix=prefix)
        if len(candidates) > 1:
            plan.problems.append(
                f"this day exists on more than one local disk, so which copy is "
                f"current cannot be decided here: {[str(c) for c in candidates]}")
        plan.local_nwb = nwb_shank_files(local, prefix)
        plan.server_nwb = nwb_shank_files(inner, prefix)
        plan.raw_recs = raw_rec_files(inner)
        plan.local_raw = raw_rec_files(local)
        # Stands in until the real check runs below, so a session that bails out
        # early (no share folder, no conversion list) still reports its shanks
        # honestly rather than looking complete.
        plan.unusable = (set(range(expect_shanks))
                         - (set(plan.local_nwb) | set(plan.server_nwb)))

        if not inner.is_dir():
            plan.problems.append(f"no session folder on the share: {inner}")
            plans.append(plan)
            continue

        for candidate in (local / "conversion_list.txt", inner / "conversion_list.txt"):
            if candidate.is_file():
                plan.conversion_list = candidate
                plan.listed_recs = parse_conversion_list(candidate)
                break
        if not plan.listed_recs:
            plan.problems.append("no conversion_list.txt found (local or share)")
            plans.append(plan)
            continue
        plan.expect_samples = sum(plan.listed_recs.values())

        plan.archived = archived_trace_stems(inner)
        absent = sorted(name for name in plan.listed_recs
                        if name not in plan.raw_recs
                        and name.rsplit(".rec", 1)[0] not in plan.archived)
        extra = sorted(set(plan.raw_recs) - set(plan.listed_recs))
        if absent:
            plan.problems.append(f"conversion list names .rec that is neither on "
                                 f"the share nor already archived: {absent}")
        if extra:
            plan.problems.append(
                f"raw .rec on the share is NOT in the conversion list, so it was "
                f"never exported: {extra}")

        # Verify the union of both sides, the share winning where a shank is in
        # both. Judging either side alone misreads a half-finished move: the
        # share would look short of shanks and the session would be stuck,
        # unable to move the rest because it is no longer eligible.
        authoritative = dict(plan.local_nwb)
        authoritative.update(plan.server_nwb)
        problems, plan.unusable = check_nwb_set(authoritative, plan.expect_samples,
                                                expect_shanks, ignore)
        plan.problems += problems
        plans.append(plan)
    return plans


def render_plan(plans: list[SessionPlan], expect_shanks: int, log=print) -> None:
    ready = [p for p in plans if p.eligible]
    blocked = [p for p in plans if not p.eligible and not p.done]

    log(f"{'session':10} {'shanks':>7} {'nwb local':>12} {'nwb share':>12} "
        f"{'raw .rec':>12}  state")
    log("-" * 88)
    for p in plans:
        where = "on share" if p.moved else ("local" if p.local_nwb else "none")
        state = "DONE" if p.done else ("READY" if p.eligible else "BLOCKED")
        # Shanks that actually verified -- not the file count, which would
        # count a held-back empty shell as though it were a real export.
        good = expect_shanks - len({s for s in p.unusable if s < expect_shanks})
        # Show local raw where the share has none, so a day holding the only
        # copy of a recording can never read as an empty 0.0 GB row.
        raw = f"{gb(p.raw_bytes)} " if p.raw_recs else f"{gb(p.local_raw_bytes)}*"
        log(f"{p.date8:10} {good:3d}/{expect_shanks:<3d} "
            f"{gb(p.movable_bytes):>12} {gb(p.server_nwb_bytes):>12} "
            f"{raw:>13}  {state} ({where})")
    log("-" * 88)
    if any(p.unbacked_raw for p in plans):
        log("* raw .rec on a local disk only -- see below")

    if blocked:
        log("\nBLOCKED -- nothing will be moved or deleted for these:")
        for p in blocked:
            log(f"  {p.date8}:")
            for problem in p.problems:
                log(f"      - {problem}")

    orphans = [p for p in plans if p.unbacked_raw]
    if orphans:
        log(f"\nNOT BACKED UP -- {len(orphans)} day(s) whose raw .rec exists only on "
            f"a local disk, with no copy on the share. This script never touches "
            f"them; they are listed because nothing else holds that data:")
        for p in orphans:
            log(f"  {p.date8}  {len(p.local_raw):2d} rec  "
                f"{gb(p.local_raw_bytes):>11}   {p.local}")
        log(f"  total at risk: {gb(sum(p.local_raw_bytes for p in orphans))}")

    to_move = [p for p in ready if p.movable]
    to_delete = [p for p in ready if p.moved and p.raw_recs]
    log(f"\nready to move   : {len(to_move)} session(s), "
        f"{gb(sum(p.movable_bytes for p in to_move))} off the local disk")
    log(f"ready to delete : {len(to_delete)} session(s), "
        f"{gb(sum(p.raw_bytes for p in to_delete))} of raw .rec on the share")
    pending = [p for p in to_move if p not in to_delete and p.raw_recs]
    if pending:
        log(f"(after --move, {len(pending)} more session(s) become ready to "
            f"delete: {gb(sum(p.raw_bytes for p in pending))})")
    held = [p for p in plans if p.local_nwb and not p.movable]
    if held:
        log(f"left on the local disk: "
            f"{', '.join(f'{p.date8} sh{sorted(p.local_nwb)}' for p in held)} "
            f"(empty export(s), not copied to the share)")


# ---------------------------------------------------------------------------
# Stage 1 -- move the NWBs onto the share
# ---------------------------------------------------------------------------

def copy_verified(src: Path, dest: Path, expect_samples: int | None,
                  log=print) -> None:
    """Copy via a .part staging file, size-check, rename, then re-open the result.

    Raises rather than leaving a partial file in place, so an interrupted copy
    never looks like a finished one and the source is still there to retry from.
    """
    staging = dest.parent / (dest.name + ".part")
    try:
        shutil.copy2(src, staging)
        copied, expected = staging.stat().st_size, src.stat().st_size
        if copied != expected:
            raise IOError(f"size mismatch after copy: {copied:,} != {expected:,} bytes")
        os.replace(staging, dest)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise

    if expect_samples is not None:
        with h5py.File(dest, "r") as handle:
            data = handle.get(DATA_PATH)
            if data is None:
                raise IOError(f"copy has no {DATA_PATH}: {dest}")
            if data.shape[0] != expect_samples:
                raise IOError(f"copy reads back {data.shape[0]:,} samples, "
                              f"expected {expect_samples:,}: {dest}")
    log(f"      verified on share: {dest.name}")


def do_move(plans: list[SessionPlan], registry: Path | None,
            dry_run: bool, log=print) -> None:
    targets = [p for p in plans if p.eligible and p.movable]

    # Sessions that went across on an earlier run -- one interrupted before it
    # reached the registry, say -- have no local NWBs left, so they are not in
    # `targets` and would never be repointed. Reconcile them first: the update
    # is idempotent, so a day already pointing at the share is left alone.
    if registry and not dry_run:
        update_registry(registry, [p for p in plans if p.eligible and p.moved], log=log)

    if not targets:
        log("nothing to move.")
        return

    for plan in targets:
        movable = plan.movable
        held = sorted(set(plan.local_nwb) - set(movable))
        log(f"\n{plan.date8}: {len(movable)} NWB "
            f"({gb(plan.movable_bytes)}) -> {plan.inner}")
        for shank in held:
            log(f"    LEFT LOCAL sh{shank}: failed verification, not copied "
                f"({plan.local_nwb[shank].stat().st_size:,} bytes)")

        if dry_run:
            for shank in sorted(movable):
                log(f"    would move {movable[shank].name}")
            if plan.conversion_list and plan.conversion_list.parent == plan.local:
                log("    would move conversion_list.txt")
            log("    would repoint rec_folder in the registry")
            continue

        plan.inner.mkdir(parents=True, exist_ok=True)
        for shank in sorted(movable):
            src = movable[shank]
            dest = plan.inner / src.name
            if dest.exists() and dest.stat().st_size == src.stat().st_size:
                log(f"    already on share, dropping local copy: {src.name}")
                src.unlink()
                plan.server_nwb[shank] = dest
                continue
            log(f"    copying {src.name} ({gb(src.stat().st_size)}) ...")
            copy_verified(src, dest, plan.expect_samples, log=log)
            src.unlink()
            plan.server_nwb[shank] = dest

        local_list = plan.local / "conversion_list.txt"
        if local_list.is_file():
            dest = plan.inner / "conversion_list.txt"
            if not dest.exists():
                shutil.copy2(local_list, dest)
            local_list.unlink()
            plan.conversion_list = dest
            log("    moved conversion_list.txt")

        # Held-back shanks stay put; everything usable has gone across.
        plan.local_nwb = {s: p for s, p in plan.local_nwb.items() if s in held}

        # Repoint this day as soon as it lands, not after the whole run: a
        # multi-hour move that is interrupted must still leave the registry
        # agreeing with where the files actually are.
        if registry:
            update_registry(registry, [plan], log=log)


def update_registry(registry: Path, plans: list[SessionPlan], log=print) -> None:
    """Repoint each moved day's ``rec_folder`` at the share.

    ``extract_sleep_lfp.resolve_nwb_path`` and ``find_sleep_mua`` look for the
    NWB inside ``rec_folder`` literally -- unlike derived output, that lookup
    does not fall back to the share -- so the registry has to follow the files.
    ``session_name`` is the folder's own stem either way, so output filenames
    are unaffected.
    """
    if not plans or not registry.is_file():
        return
    config = json.loads(registry.read_text())
    changed = []
    for plan in plans:
        for key, entry in config.items():
            if not isinstance(entry, dict):
                continue
            if Path(str(entry.get("rec_folder", ""))) == plan.local:
                entry["rec_folder"] = str(plan.inner)
                changed.append(f"{key}: {plan.local} -> {plan.inner}")
    if changed:
        registry.write_text(json.dumps(config, indent=2) + "\n")
        log(f"\nupdated {registry}:")
        for line in changed:
            log(f"    {line}")


# ---------------------------------------------------------------------------
# Stage 2 -- delete the raw traces
# ---------------------------------------------------------------------------

def do_delete_raw(plans: list[SessionPlan], expect_shanks: int, dry_run: bool,
                  ignore: frozenset[int] = frozenset(), log=print) -> None:
    """Delete raw ``.rec`` traces, re-verifying the share's NWBs immediately first."""
    freed = 0
    for plan in plans:
        if not plan.eligible or not plan.raw_recs:
            continue
        if not plan.moved:
            log(f"{plan.date8}: skipped -- NWBs are not on the share yet "
                f"(run --move first)")
            continue

        # Verify the copies that are about to become the only copy.
        problems, unusable = check_nwb_set(nwb_shank_files(plan.inner, plan.prefix),
                                           plan.expect_samples, expect_shanks, ignore)
        absent = sorted(unusable & set(ignore))
        if absent:
            log(f"\n{plan.date8}: no shank {absent} on this day; archived from "
                f"the remaining shanks.")
        if problems:
            log(f"{plan.date8}: NOT deleting -- share NWBs failed re-verification:")
            for problem in problems:
                log(f"      - {problem}")
            continue

        log(f"\n{plan.date8}: {len(plan.raw_recs)} raw trace(s), "
            f"{gb(plan.raw_bytes)}  [{len(plan.server_nwb)} shanks verified on share]")
        for name in sorted(plan.raw_recs):
            trace = plan.raw_recs[name]
            size = trace.stat().st_size
            # Siblings inside the .rec folder -- .DIO, .timestampoffset,
            # params.json, *.rec.txt -- all stay; only the trace goes.
            kept = ([q.name for q in trace.parent.iterdir() if q.name != trace.name]
                    if trace.parent.name.endswith(".rec") else [])
            freed += size
            if dry_run:
                log(f"    would delete {trace.relative_to(plan.inner)} ({gb(size)})")
                log(f"        keeping: {', '.join(sorted(kept)) or '(nothing else)'}")
                continue
            trace.unlink()
            log(f"    deleted {trace.relative_to(plan.inner)} ({gb(size)}); "
                f"kept {', '.join(sorted(kept)) or '(nothing else)'}")
    log(f"\n{'would free' if dry_run else 'freed'} on the share: {gb(freed)}")


# ---------------------------------------------------------------------------
# Stage 3 -- drop local leftovers that the share already has
# ---------------------------------------------------------------------------

def do_drop_duplicates(plans: list[SessionPlan], dry_run: bool, log=print) -> None:
    """Delete local files whose share counterpart exists at the same size.

    Only exact size matches go. Anything that differs is reported and kept --
    per ``server_fallback``, the share copy is the current one, but a local file
    that is genuinely different is not this script's call to resolve.
    """
    freed = differing = 0
    for plan in plans:
        if not plan.local.is_dir():
            continue
        pairs, mismatched = [], []
        for root, _dirs, files in os.walk(plan.local):
            rel_root = Path(root).relative_to(plan.local)
            for name in files:
                local_file = Path(root) / name
                share_file = plan.inner / rel_root / name
                if not share_file.is_file():
                    continue
                if local_file.stat().st_size == share_file.stat().st_size:
                    pairs.append(local_file)
                else:
                    mismatched.append((local_file, share_file))
        if not pairs and not mismatched:
            continue
        log(f"\n{plan.date8}: {len(pairs)} duplicate file(s) "
            f"({gb(sum(p.stat().st_size for p in pairs))}), "
            f"{len(mismatched)} differing")
        for local_file in pairs:
            size = local_file.stat().st_size
            if dry_run:
                log(f"    would delete {local_file.relative_to(plan.local)} ({gb(size)})")
            else:
                local_file.unlink()
                freed += size
        for local_file, share_file in mismatched:
            differing += 1
            log(f"    KEPT (differs) {local_file.relative_to(plan.local)}: "
                f"local {gb(local_file.stat().st_size)} vs "
                f"share {gb(share_file.stat().st_size)}")
    if not dry_run:
        log(f"\nfreed on the local disk: {gb(freed)}; "
            f"{differing} differing file(s) left for you to look at")


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Archive exported NWBs to the share and drop the raw .rec traces.")
    parser.add_argument("--animal", default=DEFAULT_ANIMAL)
    parser.add_argument("--local-root", type=Path, action="append", dest="local_roots",
                        help="local folder holding this animal's sessions; repeatable. "
                             f"Default: <{'|'.join(DEFAULT_LOCAL_DRIVES)}>\\<animal>")
    parser.add_argument("--server-root", type=Path, default=None,
                        help=f"default: {SHARE_ROOT}\\<animal>")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY,
                        help="sleep_day_configs.json to repoint after moving")
    parser.add_argument("--no-registry-update", action="store_true")
    parser.add_argument("--expect-shanks", type=int, default=EXPECTED_SHANKS)
    parser.add_argument("--ignore-shanks", default="",
                        help="comma-separated shanks that some days do not have, "
                             "e.g. 7. Their absence stops blocking a session: the "
                             "day is archived from its remaining shanks, and where "
                             "one exists and verifies it is still archived too.")
    parser.add_argument("--sessions", default="",
                        help="comma-separated YYYYMMDD or YYMMDD; default all")
    parser.add_argument("--move", action="store_true",
                        help="stage 1: copy NWBs to the share, verify, delete local")
    parser.add_argument("--delete-raw", action="store_true",
                        help="stage 2: delete raw .rec traces on the share")
    parser.add_argument("--drop-duplicates", action="store_true",
                        help="stage 3: delete local files the share already has")
    parser.add_argument("--i-have-backups", action="store_true",
                        help="required alongside --delete-raw; raw traces are "
                             "not recoverable once deleted")
    parser.add_argument("--dry-run", action="store_true",
                        help="with a stage flag, show what it would do")
    args = parser.parse_args(argv)

    only = {s.strip() for s in args.sessions.split(",") if s.strip()} or None
    ignore = frozenset(int(s) for s in args.ignore_shanks.split(",") if s.strip())
    local_roots = args.local_roots or [Path(f"{d}\\{args.animal}")
                                       for d in DEFAULT_LOCAL_DRIVES]
    server_root = args.server_root or (SHARE_ROOT / args.animal)

    print(f"animal      : {args.animal}")
    for root in local_roots:
        print(f"local root  : {root}" + ("" if root.is_dir() else "   (absent)"))
    print(f"share root  : {server_root}"
          + ("" if server_root.is_dir() else "   (absent)"))
    print()

    plans = build_plan(local_roots, server_root, args.animal,
                       args.expect_shanks, only, ignore)
    if not plans:
        print(f"no {args.animal} sessions found on "
              f"{', '.join(str(r) for r in local_roots)} or {server_root}")
        return 1

    if ignore:
        print(f"shank(s) {sorted(ignore)} are optional: a day without one is "
              f"archived from its remaining shanks. Where such a shank has only "
              f"an empty NWB, that file stays on the local disk rather than "
              f"joining the export on the share.\n")
    render_plan(plans, args.expect_shanks)

    if args.delete_raw and not args.i_have_backups and not args.dry_run:
        print("\nRefusing to delete raw traces without --i-have-backups.")
        return 2

    if args.move:
        print("\n=== stage 1: move NWBs to the share ===")
        do_move(plans, None if args.no_registry_update else args.registry,
                args.dry_run)
    if args.delete_raw:
        print("\n=== stage 2: delete raw .rec traces ===")
        do_delete_raw(plans, args.expect_shanks, args.dry_run, ignore)
    if args.drop_duplicates:
        print("\n=== stage 3: drop local duplicates ===")
        do_drop_duplicates(plans, args.dry_run)

    if not (args.move or args.delete_raw or args.drop_duplicates):
        print("\n(plan only -- nothing was changed. Add --move / --delete-raw "
              "--i-have-backups / --drop-duplicates to act.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
