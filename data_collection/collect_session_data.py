"""
collect_session_data.py
========================
Move behaviour videos and behaviour log files out of their local capture
folders into each session's folder on the server. Four sources feed each
session, grouped into three columns:

- **video**   -- the camera capture folder (default ``D:\\cl\\video``).
- **passive** -- the drifting-grating disk sessions: the Unity engine log
  folder (default ``D:\\cl\\grating_disk_representation\\driftingGratingBuildNew\\
  DriftingGratingData``). No task or reward is involved.
- **task**    -- two SphereTask behaviour-summary folders, one per task
  variant: the grating task (default ``D:\\cl\\SphereTask\\sphereGratingBuild\\
  BehaviorData``) and the contrast task (default ``D:\\cl\\SphereTask\\
  sphereTaskBuild\\BehaviorData``).

Given a destination animal root such as::

    \\\\10.129.151.88\\xieluanlabs2\\xl_cl\\experiment_data\\CnL46

the animal id is taken from the folder name (``CnL46``) and every ``YYMMDD``
session subfolder underneath it is scanned. For each session:

- The video capture folder is searched for files whose name contains
  ``_<animal>_<YYYY-MM-DD>_`` -- this picks up every prefix in use
  (``Imaging_source_``, ``front_camera_``, ...) and every companion type
  (``_VIDEO.avi``, ``_TS.npy``, ``_PROC``, ``_DLC.hdf5``). Matches are moved
  into ``<session>/video/``, which is created if absent.
- The Unity log, grating-task, and contrast-task folders are each searched for
  files named ``<animal>_drifting_grating_exp_<YYYYMMDD>_<HHMMSS>...`` (Unity)
  or ``<animal>_<YYYY-MM-DD>_Session<NNN>_...`` (the two SphereTask
  BehaviorData folders). Matches are moved straight into the session folder
  itself, alongside the ``.rec`` files -- no subfolder, no grouping.

Nothing is required to come as a complete set: a video recording moves whether
or not its ``_TS`` / ``_PROC`` / ``_DLC`` companions were ever produced, and
companion files whose video has already been moved away travel on their own
too. The preview groups video files by recording and names any missing
companions, so a video going across without its DLC output is visible rather
than silent. Anything else carrying the animal and date --
``_VIDEODLC_resnet50_*.h5``, ``_filtered.csv``, ``_meta.pickle``, ``_trim.mp4``
-- is swept along as well. Any source can be turned off (``--no-video`` /
``--no-unity`` / ``--no-grating-task`` / ``--no-contrast-task``, or by leaving
the corresponding GUI field blank).

Files are copied to a ``.part`` file first, checked for size, renamed into
place, and only then removed from the source. An interrupted transfer therefore
never destroys the local copy nor leaves a truncated file looking complete.

For a window that lets you pick the destination folders each time, run
``collect_session_data_gui.py`` -- it drives the same planning and moving code.

Usage
-----
  python collect_session_data.py <dest animal root> [<dest animal root> ...]
  python collect_session_data.py \\\\10.129.151.88\\...\\CnL46 --dry-run
  python collect_session_data.py \\\\10.129.151.88\\...\\CnL46 --video-root D:\\cl\\video
  python collect_session_data.py \\\\10.129.151.88\\...\\CnL46 --no-video
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_VIDEO_ROOT = Path(r"D:\cl\video")
DEFAULT_UNITY_ROOT = Path(r"D:\cl\grating_disk_representation\driftingGratingBuildNew\DriftingGratingData")
DEFAULT_GRATING_TASK_ROOT = Path(r"D:\cl\SphereTask\sphereGratingBuild\BehaviorData")
DEFAULT_CONTRAST_TASK_ROOT = Path(r"D:\cl\SphereTask\sphereTaskBuild\BehaviorData")
VIDEO_SUBDIR = "video"

# Session folders are YYMMDD (260719) with YYYYMMDD (20260719) also accepted.
SESSION_PATTERNS = (("%y%m%d", re.compile(r"^\d{6}$")),
                    ("%Y%m%d", re.compile(r"^\d{8}$")))


# ---------------------------------------------------------------------------
# Naming rules -- video
# ---------------------------------------------------------------------------

def animal_id_from_root(dest_root: Path) -> str:
    """``CnL46`` -> ``CnL46``; ``CnL42SG_1`` -> ``CnL42``."""
    return re.sub(r"SG(_.*)?$", "", dest_root.name)


def session_date(folder: Path) -> str | None:
    """Return the folder's date as ``YYYY-MM-DD``, or None if it isn't a session."""
    for fmt, pattern in SESSION_PATTERNS:
        if pattern.match(folder.name):
            try:
                return datetime.strptime(folder.name, fmt).strftime("%Y-%m-%d")
            except ValueError:
                return None
    return None


def find_source_files(video_root: Path, animal: str, date: str) -> list[Path]:
    """Files in video_root tagged with this animal and date, any prefix or suffix."""
    tag = f"_{animal}_{date}_".lower()
    return sorted(f for f in video_root.iterdir()
                  if f.is_file() and tag in f.name.lower())


# A recording is <prefix>_<animal>_<date>_<index>; the tail names what the file is.
COMPANION_KINDS = ("VIDEO", "TS", "PROC", "DLC")
_KIND_RE = re.compile(r"_(VIDEO|TS|PROC|DLC)(\.[A-Za-z0-9]+)?$", re.IGNORECASE)
_GROUP_RE = re.compile(r"^(.*_\d{4}-\d{2}-\d{2}_\d+)")


def file_kind(name: str) -> str:
    """``..._2_TS.npy`` -> ``TS``. Anything unrecognised is ``other``.

    ``other`` still moves -- it is the DLC side products (``_VIDEODLC_*.h5``,
    ``_filtered.csv``, ``_meta.pickle``) and one-off names like ``_trim.mp4``.
    """
    match = _KIND_RE.search(name)
    return match.group(1).upper() if match else "other"


def recording_key(name: str) -> str:
    """Group name shared by one recording's files, or "" if the name has no index."""
    match = _GROUP_RE.match(name)
    return match.group(1) if match else ""


# Known capture prefixes, so the animal id can be read back out of a filename.
FILENAME_PREFIXES = ("front_camera_", "Imaging_source_")
_ANIMAL_RE = re.compile(r"^(?P<head>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_")


def animal_in_filename(name: str) -> tuple[str, str] | None:
    """``front_camera_CnL43_2026-07-30_3_VIDEO.avi`` -> ``("CnL43", "2026-07-30")``."""
    match = _ANIMAL_RE.match(name)
    if not match:
        return None
    head = match.group("head")
    for prefix in FILENAME_PREFIXES:
        if head.lower().startswith(prefix.lower()):
            return head[len(prefix):], match.group("date")
    return head.rsplit("_", 1)[-1], match.group("date")


def other_animals_on(video_root: Path, animal: str, date: str) -> dict[str, int]:
    """Other animal ids that also have files on this date, with file counts.

    A session's own recordings landing under someone else's id is the signature
    of the animal not being changed in the camera software, which would
    otherwise look exactly like "no videos for that session".
    """
    counts: dict[str, int] = {}
    for f in video_root.iterdir():
        if not f.is_file():
            continue
        parsed = animal_in_filename(f.name)
        if parsed and parsed[1] == date and parsed[0].lower() != animal.lower():
            counts[parsed[0]] = counts.get(parsed[0], 0) + 1
    return counts


def dates_available(video_root: Path, animal: str) -> set[str]:
    """Every date this animal has files for in video_root."""
    pattern = re.compile(rf"_{re.escape(animal)}_(\d{{4}}-\d{{2}}-\d{{2}})_", re.IGNORECASE)
    found = set()
    for f in video_root.iterdir():
        match = pattern.search(f.name)
        if f.is_file() and match:
            found.add(match.group(1))
    return found


def _merge_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, n in src.items():
        dst[key] = dst.get(key, 0) + n


# ---------------------------------------------------------------------------
# Naming rules -- flat behaviour logs (Unity, grating task, contrast task)
#
# Each of these names itself ``<animal>_<date>_...`` at the top of a flat
# folder (no subfolders, no companion files), so one class covers all three:
# only the destination folder and the animal/date pattern differ. Files land
# straight in the session folder -- no subfolder, unlike video.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlatSource:
    key: str             # stable id, e.g. "unity" -- used as a dict key elsewhere
    label: str            # for messages, e.g. "unity log"
    pattern: re.Pattern   # matches a filename; captures named groups "animal" and "date"
    date_is_compact: bool = False   # True if "date" is YYYYMMDD rather than YYYY-MM-DD

    def _iso(self, raw: str) -> str:
        return raw if not self.date_is_compact else f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"

    def scan(self, root: Path) -> dict[Path, tuple[str, str]]:
        """Every top-level file in root mapped to (animal, date_iso)."""
        out: dict[Path, tuple[str, str]] = {}
        for f in root.iterdir():
            if not f.is_file():
                continue
            match = self.pattern.match(f.name)
            if match:
                out[f] = (match.group("animal"), self._iso(match.group("date")))
        return out

    def find(self, root: Path, animal: str, date: str) -> list[Path]:
        return sorted(f for f, (a, d) in self.scan(root).items()
                      if d == date and a.lower() == animal.lower())

    def other_animals_on(self, root: Path, animal: str, date: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for a, d in self.scan(root).values():
            if d == date and a.lower() != animal.lower():
                counts[a] = counts.get(a, 0) + 1
        return counts

    def dates_available(self, root: Path, animal: str) -> set[str]:
        return {d for a, d in self.scan(root).values() if a.lower() == animal.lower()}


# ``<animal>_drifting_grating_exp_<YYYYMMDD>_<HHMMSS>....txt``
UNITY_SOURCE = FlatSource(
    "unity", "unity log",
    re.compile(r"^(?P<animal>.+?)_drifting_grating_exp_(?P<date>\d{8})_"),
    date_is_compact=True)

# ``<animal>_<YYYY-MM-DD>_Session<NNN>_Data.json`` / ``..._Grating.json``
_BEHAVIOR_SESSION_RE = re.compile(r"^(?P<animal>.+?)_(?P<date>\d{4}-\d{2}-\d{2})_Session\d+_")
GRATING_TASK_SOURCE = FlatSource("grating_task", "grating task file", _BEHAVIOR_SESSION_RE)
CONTRAST_TASK_SOURCE = FlatSource("contrast_task", "contrast task file", _BEHAVIOR_SESSION_RE)

# All flat sources, keyed for lookup from a SessionPlan.flat_files dict.
FLAT_SOURCES: dict[str, FlatSource] = {
    "unity": UNITY_SOURCE,
    "grating_task": GRATING_TASK_SOURCE,
    "contrast_task": CONTRAST_TASK_SOURCE,
}


# ---------------------------------------------------------------------------
# Planning: work out what would move, without moving anything
# ---------------------------------------------------------------------------

@dataclass
class Recording:
    """One recording's files, however many of them exist."""
    key: str                # <prefix>_<animal>_<date>_<index>, or "" if unindexed
    files: list[Path]

    @property
    def kinds(self) -> set[str]:
        return {file_kind(f.name) for f in self.files}

    @property
    def has_video(self) -> bool:
        return "VIDEO" in self.kinds

    @property
    def missing(self) -> list[str]:
        """Companion kinds this recording never got. Empty for a complete set.

        Loose files (no ``key``) are not a recording, so nothing is "missing".
        """
        if not self.key:
            return []
        return [k for k in COMPANION_KINDS if k not in self.kinds]

    @property
    def n_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.files)


def group_recordings(files: list[Path]) -> list[Recording]:
    """Bucket files by recording. Unindexed names land in a trailing "" bucket."""
    buckets: dict[str, list[Path]] = {}
    for f in files:
        buckets.setdefault(recording_key(f.name), []).append(f)
    ordered = sorted(buckets.items(), key=lambda kv: (kv[0] == "", kv[0]))
    return [Recording(key, sorted(group)) for key, group in ordered]


@dataclass
class SessionPlan:
    folder: Path            # the YYMMDD session folder
    date: str               # YYYY-MM-DD
    video_files: list[Path]           # video-side source files to move (may be empty)
    flat_files: dict[str, list[Path]] = field(default_factory=dict)  # FlatSource.key -> files
    other_animals: dict[str, int] = field(default_factory=dict)  # possible mislabels

    @property
    def video_dir(self) -> Path:
        return self.folder / VIDEO_SUBDIR

    @property
    def needs_video_dir(self) -> bool:
        return bool(self.video_files) and not self.video_dir.exists()

    @property
    def all_flat_files(self) -> list[Path]:
        return [f for files in self.flat_files.values() for f in files]

    @property
    def n_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.video_files + self.all_flat_files)

    @property
    def recordings(self) -> list[Recording]:
        return group_recordings(self.video_files)


@dataclass
class AnimalPlan:
    dest_root: Path
    animal: str
    sessions: list[SessionPlan] = field(default_factory=list)
    leftover_dates: list[str] = field(default_factory=list)  # files with no session folder
    error: str = ""

    @property
    def n_files(self) -> int:
        return sum(len(s.video_files) + len(s.all_flat_files) for s in self.sessions)

    @property
    def n_bytes(self) -> int:
        return sum(s.n_bytes for s in self.sessions)

    @property
    def n_recordings(self) -> int:
        return sum(1 for s in self.sessions for r in s.recordings if r.key)

    @property
    def n_incomplete(self) -> int:
        """Recordings moving without a full set of companions."""
        return sum(1 for s in self.sessions for r in s.recordings if r.missing)


def plan_animal(dest_root: Path, video_root: Path | None = None,
                unity_root: Path | None = None, grating_task_root: Path | None = None,
                contrast_task_root: Path | None = None,
                animal: str | None = None) -> AnimalPlan:
    """Scan one animal root and return what a move would do. Reads only.

    Any source can be disabled by passing None for its root -- it is then
    simply left out of the plan rather than treated as an error.
    """
    animal = animal or animal_id_from_root(dest_root)
    plan = AnimalPlan(dest_root=dest_root, animal=animal)

    flat_roots: dict[str, Path | None] = {
        "unity": unity_root, "grating_task": grating_task_root,
        "contrast_task": contrast_task_root,
    }

    if not dest_root.is_dir():
        plan.error = "destination folder not found"
        return plan
    if video_root is not None and not video_root.is_dir():
        plan.error = f"video root not found: {video_root}"
        return plan
    for key, root in flat_roots.items():
        if root is not None and not root.is_dir():
            plan.error = f"{FLAT_SOURCES[key].label} root not found: {root}"
            return plan

    dated = []
    for folder in sorted(p for p in dest_root.iterdir() if p.is_dir()):
        date = session_date(folder)
        if date:
            dated.append((folder, date))
    if not dated:
        plan.error = "no YYMMDD session folders found"
        return plan

    for folder, date in dated:
        video_files = find_source_files(video_root, animal, date) if video_root else []
        flat_files: dict[str, list[Path]] = {}
        other: dict[str, int] = {}
        if video_root:
            _merge_counts(other, other_animals_on(video_root, animal, date))
        for key, root in flat_roots.items():
            if root is None:
                continue
            source = FLAT_SOURCES[key]
            files = source.find(root, animal, date)
            if files:
                flat_files[key] = files
            _merge_counts(other, source.other_animals_on(root, animal, date))
        plan.sessions.append(SessionPlan(folder, date, video_files, flat_files, other))

    claimed_video = {s.date for s in plan.sessions if s.video_files}
    leftover: set[str] = set()
    if video_root:
        leftover |= dates_available(video_root, animal) - claimed_video
    for key, root in flat_roots.items():
        if root is None:
            continue
        claimed = {s.date for s in plan.sessions if key in s.flat_files}
        leftover |= FLAT_SOURCES[key].dates_available(root, animal) - claimed
    plan.leftover_dates = sorted(leftover)
    return plan


# ---------------------------------------------------------------------------
# Preview text (shared by the CLI's --dry-run and the GUI's Preview button)
# ---------------------------------------------------------------------------

def describe_recording(rec: Recording) -> str:
    """One line per recording: what it is, how big, and what it is missing."""
    kinds = " ".join(k if k in rec.kinds else "-" * len(k) for k in COMPANION_KINDS)
    extra = sum(1 for f in rec.files if file_kind(f.name) == "other")
    line = f"    {rec.key}  [{kinds}]  {rec.n_bytes / 1e6:.1f} MB"
    if extra:
        line += f"  +{extra} other"
    if not rec.has_video:
        line += "   (companions only, no video - moving anyway)"
    elif rec.missing:
        line += f"   (no {', '.join(rec.missing)} - moving anyway)"
    return line


def preview_lines(plan: AnimalPlan) -> list[str]:
    """The full read-only report for one animal, grouped by recording."""
    out = [f"{plan.dest_root}  (animal {plan.animal})"]
    if plan.error:
        out.append(f"  skipped: {plan.error}")
        return out

    for session in plan.sessions:
        if session.video_files or session.flat_files:
            parts = []
            if session.video_files:
                parts.append(f"{len(session.video_files)} video file(s)")
            for key, files in session.flat_files.items():
                parts.append(f"{len(files)} {FLAT_SOURCES[key].label}(s)")
            note = f"  (will create {VIDEO_SUBDIR}/)" if session.needs_video_dir else ""
            out.append(f"  {session.folder.name} ({session.date}): "
                       f"{', '.join(parts)}{note}")
            for rec in session.recordings:
                if rec.key:
                    out.append(describe_recording(rec))
                else:  # no _<index>_: not part of a recording, still moved
                    for f in rec.files:
                        out.append(f"    {f.name}  {f.stat().st_size / 1e6:.1f} MB"
                                   f"   (loose file - moving anyway)")
            for key, files in session.flat_files.items():
                for f in files:
                    out.append(f"    {f.name}  {f.stat().st_size / 1e6:.2f} MB"
                               f"   ({FLAT_SOURCES[key].label} -> straight into "
                               f"{session.folder.name}/)")
        else:
            out.append(f"  {session.folder.name} ({session.date}): no files found")

        if session.other_animals:
            tally = ", ".join(f"{other} ({n} file(s))"
                              for other, n in sorted(session.other_animals.items()))
            out.append(f"    ** {session.date} also has files under another animal id: "
                       f"{tally}")
            out.append(f"    ** not moved - check whether the animal was left wrong "
                       f"in the recording software")

    if plan.n_incomplete:
        out.append(f"  {plan.n_incomplete} of {plan.n_recordings} recording(s) are "
                   f"missing companions; all are included.")
    if plan.leftover_dates:
        out.append(f"  note: {plan.animal} files with no session folder: "
                   f"{', '.join(plan.leftover_dates)}")
    return out


# ---------------------------------------------------------------------------
# Moving
# ---------------------------------------------------------------------------

def move_file(src: Path, dest: Path, overwrite: bool = False) -> str:
    """Copy src to dest via a .part staging file, then delete src.

    Returns one of "moved", "skipped-identical", "skipped-exists".
    """
    if dest.exists():
        if dest.stat().st_size == src.stat().st_size and not overwrite:
            src.unlink()
            return "skipped-identical"
        if not overwrite:
            return "skipped-exists"

    staging = dest.parent / (dest.name + ".part")
    try:
        shutil.copy2(src, staging)
        copied, expected = staging.stat().st_size, src.stat().st_size
        if copied != expected:
            raise IOError(f"size mismatch after copy: {copied} != {expected} bytes")
        os.replace(staging, dest)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    src.unlink()
    return "moved"


def run_plan(plan: AnimalPlan, dry_run: bool = False, overwrite: bool = False,
             log=print, on_file=None, should_stop=None) -> tuple[int, int]:
    """Execute one AnimalPlan. Returns (files moved, bytes moved).

    ``log`` receives progress lines, ``on_file`` (if given) is called after every
    attempted file, and ``should_stop`` (if given) is polled between files so a
    GUI can interrupt cleanly.
    """
    n_moved = 0
    bytes_moved = 0

    for session in plan.sessions:
        flat_all = session.all_flat_files
        if not session.video_files and not flat_all:
            log(f"  {session.folder.name} ({session.date}): no files found")
            continue
        if should_stop and should_stop():
            log("  stopped by user")
            break

        if session.needs_video_dir:
            log(f"  {session.folder.name} ({session.date}): creating {VIDEO_SUBDIR}/")
            if not dry_run:
                session.video_dir.mkdir()

        parts = []
        if session.video_files:
            parts.append(f"{len(session.video_files)} video file(s)")
        for key, files in session.flat_files.items():
            parts.append(f"{len(files)} {FLAT_SOURCES[key].label}(s)")
        log(f"  {session.folder.name} ({session.date}): {', '.join(parts)}")

        # Video files land in <session>/video/; flat sources land in <session>/ itself.
        moves = ([(f, session.video_dir) for f in session.video_files] +
                 [(f, session.folder) for f in flat_all])
        for src, dest_dir in moves:
            if should_stop and should_stop():
                log("  stopped by user")
                return n_moved, bytes_moved
            size = src.stat().st_size
            if dry_run:
                log(f"    would move {src.name} ({size / 1e6:.1f} MB)")
                n_moved += 1
                bytes_moved += size
            else:
                try:
                    result = move_file(src, dest_dir / src.name, overwrite)
                except Exception as exc:
                    log(f"    FAILED {src.name}: {exc}")
                    result = "failed"
                if result == "skipped-exists":
                    log(f"    skipped {src.name}: already at destination with a "
                        f"different size (tick 'Overwrite' to replace)")
                elif result == "skipped-identical":
                    log(f"    removed local {src.name}: already at destination")
                elif result == "moved":
                    log(f"    moved {src.name} ({size / 1e6:.1f} MB)")
                    n_moved += 1
                    bytes_moved += size
            if on_file:
                on_file()

    if plan.leftover_dates:
        log(f"  note: {plan.animal} files with no session folder: "
            f"{', '.join(plan.leftover_dates)}")
    return n_moved, bytes_moved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dest_roots", nargs="+", type=Path,
                        help=r"destination animal folder(s), e.g. \\10.129.151.88\...\CnL46")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT,
                        help=f"video capture folder (default {DEFAULT_VIDEO_ROOT})")
    parser.add_argument("--unity-root", type=Path, default=DEFAULT_UNITY_ROOT,
                        help=f"Unity drifting-grating log folder (default {DEFAULT_UNITY_ROOT})")
    parser.add_argument("--grating-task-root", type=Path, default=DEFAULT_GRATING_TASK_ROOT,
                        help=f"grating task behavior folder (default {DEFAULT_GRATING_TASK_ROOT})")
    parser.add_argument("--contrast-task-root", type=Path, default=DEFAULT_CONTRAST_TASK_ROOT,
                        help=f"contrast task behavior folder (default {DEFAULT_CONTRAST_TASK_ROOT})")
    parser.add_argument("--no-video", action="store_true", help="skip moving video files")
    parser.add_argument("--no-unity", action="store_true", help="skip moving Unity log files")
    parser.add_argument("--no-grating-task", action="store_true",
                        help="skip moving grating task behavior files")
    parser.add_argument("--no-contrast-task", action="store_true",
                        help="skip moving contrast task behavior files")
    parser.add_argument("--animal-id",
                        help="override the animal id inferred from the destination folder name")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would move without touching anything")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace destination files that differ in size")
    args = parser.parse_args()

    roots = {
        "video": (None if args.no_video else args.video_root),
        "unity": (None if args.no_unity else args.unity_root),
        "grating_task": (None if args.no_grating_task else args.grating_task_root),
        "contrast_task": (None if args.no_contrast_task else args.contrast_task_root),
    }
    for name, root in roots.items():
        if root is not None and not root.is_dir():
            parser.error(f"{name} root not found: {root}")
    if not any(roots.values()):
        parser.error("all sources disabled - nothing to move")
    if args.animal_id and len(args.dest_roots) > 1:
        parser.error("--animal-id can only be used with a single destination root")

    total_files = 0
    total_bytes = 0
    for dest_root in args.dest_roots:
        plan = plan_animal(dest_root, roots["video"], roots["unity"], roots["grating_task"],
                           roots["contrast_task"], args.animal_id)
        if args.dry_run:
            print("\n".join(preview_lines(plan)))
            total_files += plan.n_files
            total_bytes += plan.n_bytes
            continue
        print(f"{dest_root}  (animal {plan.animal})")
        if plan.error:
            print(f"  skipped: {plan.error}", file=sys.stderr)
            continue
        n, nbytes = run_plan(plan, args.dry_run, args.overwrite)
        total_files += n
        total_bytes += nbytes

    verb = "would move" if args.dry_run else "moved"
    print(f"\n{verb} {total_files} file(s), {total_bytes / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
