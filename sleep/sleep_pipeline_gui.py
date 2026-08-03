"""
Tkinter control panel for the sleep LFP / spectrogram pipeline.

Drives the sequence normally run by hand for each recording day:

    1. video_ephys_sync.py           video PROC square wave vs .rec DIO -> sync_times_{pre,post}.pkl
    2. proc_func_velocity.py         tracking PROC -> *_velocity_advanced.pkl
    3. extract_sleep_lfp.py          NWB -> low_freq/*_lfp_traces.npz
    4. compute_sleep_spectrograms.py LFP traces -> *_spectrograms.npz
    5. compute_sleep_features.py     LFP + spectrograms -> *_all_shanks_band_powers.pkl
    6. plot_sleep_spectrograms.py    band powers -> per-channel figures + trace pkl

The GUI picks the animal and recording day (a pair registered in
sleep_day_configs.json), writes ACTIVE_ANIMAL / ACTIVE_DATE / SESSION_FILTER /
shanks / VELOCITY_SOURCE into sleep_pipeline_config.py - the same file the
scripts already read, so running them standalone afterwards stays consistent -
then runs each selected stage as a subprocess and streams its output into the
log pane.

An animal-day the registry doesn't know has no paths at all, so nothing can run
for it. "Set up this day..." (and the prompt shown when you hit Run on an
unregistered pair) launches set_sleep_day.py with that animal and date
preselected; when that window closes the registry is re-read and preflight
re-runs.

The NWB folder and its file prefix can also be re-pointed here - Browse... then
"Save path" writes both back to the registry, which is what every stage reads.
That covers the common case of a day's data being copied to the other server
after it was first registered. The other registry fields (sample windows, .rec
epochs, PROC files) stay in set_sleep_day.py.

Preflight inspects the day on disk first and reports, per sleep session and
shank, what each stage will find: .rec DIO export, PROC tracking file, per-shank
NWB, and every intermediate file already computed. That makes it obvious which
stages actually need re-running.

Stdlib only (tkinter + subprocess); the heavy imports (spikeinterface, scipy,
sklearn) happen in the subprocesses, so the panel starts instantly and a crashed
stage cannot take the GUI down with it.

Run with:  python sleep_pipeline_gui.py
"""
import ast
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

SLEEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = SLEEP_DIR.parent
for _path in (str(REPO_ROOT), str(SLEEP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Both stdlib-only, so importing them here cannot slow the GUI down or drag in
# scipy/spikeinterface (which this interpreter may not even have).
from server_fallback import mirror_on_backup_server
from sleep_day_registry import (REGISTRY_FILE, day_key, entry_animal,
                                load_registry, registered_animals,
                                registered_days, update_day_paths)

CONFIG_PATH = SLEEP_DIR / "sleep_pipeline_config.py"
SETUP_SCRIPT = SLEEP_DIR / "set_sleep_day.py"
SETTINGS_PATH = Path.home() / ".sleep_pipeline_gui.json"

# Conda envs preferred as the interpreter, best first. The pipeline needs
# spikeinterface + scipy + sklearn; whichever is picked is remembered in
# SETTINGS_PATH, and Browse... overrides it.
PREFERRED_ENVS = ("ms10", "kilosort", "phy2")

SESSIONS = ("pre", "post")

# Config scalars this GUI owns. Everything else in sleep_pipeline_config.py
# (preproc_params, spec_params, plot_params, artifact_params, ...) is left alone.
CONFIG_SCALARS = ("ACTIVE_ANIMAL", "ACTIVE_DATE", "SESSION_FILTER", "shanks",
                  "VELOCITY_SOURCE")
# Read but never written - only needed to turn sample indices into seconds and
# to name the velocity file each stage expects.
CONFIG_READONLY = ("original_fs", "VELOCITY_KEYPOINTS")

# Velocity sources, mirroring proc_func_velocity.VELOCITY_SOURCES (imported by
# name here rather than from that module, which needs scipy/matplotlib).
VELOCITY_FILE_STEMS = {"proc_center": "velocity_advanced", "dlc_body": "velocity_body"}

DEFAULT_DIO_CHANNEL = 2


# =============================================================================
# INTERPRETER
# =============================================================================

def find_interpreters():
    """Candidate python.exe paths: conda envs, then the one running this GUI."""
    found = []
    envs = Path.home() / ".conda" / "envs"
    exe = "python.exe" if os.name == "nt" else "python"
    try:
        for env in sorted(envs.iterdir()):
            candidate = env / exe
            if candidate.exists():
                found.append(str(candidate))
    except OSError:
        pass
    if sys.executable and sys.executable not in found:
        found.append(sys.executable)
    return found


def default_interpreter(candidates):
    for name in PREFERRED_ENVS:
        for c in candidates:
            if Path(c).parent.name == name:
                return c
    return candidates[0] if candidates else sys.executable


# =============================================================================
# sleep_pipeline_config.py READ / WRITE
# =============================================================================

def read_config_values():
    """Module-level values of the GUI-managed scalars, without importing the module.

    Importing sleep_pipeline_config would need the registry entry to exist (it
    calls load_day_config at import time) and would pull in server_fallback, so
    the file is parsed instead - the GUI must still open on a broken config.
    """
    values = {}
    wanted = CONFIG_SCALARS + CONFIG_READONLY
    try:
        tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return values
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    values[target.id] = ast.unparse(node.value)
    return values


def _assignment_pattern(name):
    # Captures: (1) "NAME = ", (2) the value, (3) any trailing inline comment.
    return re.compile(rf"^([ \t]*{name}[ \t]*=[ \t]*)([^#\n]*?)([ \t]*#.*)?$", re.MULTILINE)


def _literal_text(name, value):
    """Source text to write for a config value, matching the file's quoting style."""
    if isinstance(value, str):
        return json.dumps(value)  # double quotes, as sleep_pipeline_config.py uses
    return repr(value)


def _same_value(old_text, new_text):
    """
    True if two assignment texts mean the same thing.

    Compared as parsed literals so that re-writing an unchanged day does not
    churn the file over quoting or spacing style ([5,7] vs [5, 7]) alone.
    """
    try:
        return ast.literal_eval(old_text) == ast.literal_eval(new_text)
    except (ValueError, SyntaxError):
        return old_text == new_text


def write_config_values(values, backup=True):
    """
    Rewrite CONFIG_SCALARS in sleep_pipeline_config.py in place.

    Only the value of each assignment is replaced, so comments, layout and every
    other setting in the file survive.

    Returns a list of (name, old_text, new_text) for the assignments that changed.
    """
    raw = CONFIG_PATH.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")  # edit in LF, restore the file's own ending on write
    changes = []

    for name, value in values.items():
        if name not in CONFIG_SCALARS:
            raise KeyError(f"{name} is not a GUI-managed config scalar")
        new_text = _literal_text(name, value)
        pattern = _assignment_pattern(name)
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"No assignment for {name} found in {CONFIG_PATH.name}")
        old_text = match.group(2).strip()
        if _same_value(old_text, new_text):
            continue
        text = pattern.sub(
            lambda m: f"{m.group(1)}{new_text}{m.group(3) or ''}", text, count=1
        )
        changes.append((name, old_text, new_text))

    if changes:
        if backup:
            shutil.copy2(CONFIG_PATH, CONFIG_PATH.with_suffix(".py.bak"))
        CONFIG_PATH.write_text(text.replace("\n", newline), encoding="utf-8", newline="")
    return changes


# =============================================================================
# DISK HELPERS
#   Every stage reads through server_fallback: outputs go to the NEW server,
#   reads check the new server first and fall back to the old one. Preflight
#   has to look in both places for the same reason.
# =============================================================================

def _exists(path):
    try:
        return path is not None and Path(path).exists()
    except OSError:
        return False


def find_any(path):
    """First existing copy of `path` (new server preferred), or None."""
    if path is None:
        return None
    path = Path(path)
    mirrored = mirror_on_backup_server(path)
    if _exists(mirrored):
        return mirrored
    return path if _exists(path) else None


def output_target(path):
    """Where a stage would WRITE `path` (resolve_output_folder's mapping)."""
    path = Path(path)
    return mirror_on_backup_server(path) or path


def parse_int_list(text, label):
    """"5 7" / "5,7" -> [5, 7]. Empty -> []."""
    items = [chunk for chunk in re.split(r"[,\s]+", text.strip()) if chunk]
    try:
        return [int(item) for item in items]
    except ValueError:
        raise ValueError(f"{label} must be whole numbers, got {text!r}")


def proc_session_name(proc_file):
    """Session name from a front-camera *_PROC path (proc_func_velocity's rule)."""
    stem = Path(proc_file).name
    if stem.endswith("_PROC"):
        stem = stem[:-len("_PROC")]
    if stem.startswith("front_camera_"):
        stem = stem[len("front_camera_"):]
    return stem


def velocity_output_name(proc_file, source="proc_center"):
    """Velocity filename for a PROC file, per source (proc_func_velocity's rule)."""
    stem = VELOCITY_FILE_STEMS.get(source, VELOCITY_FILE_STEMS["proc_center"])
    return f"{proc_session_name(proc_file)}_{stem}.pkl"


def dlc_file_for_proc(proc_file):
    """Companion *_DLC.hdf5 for a PROC file - the keypoint source for 'dlc_body'."""
    proc_file = Path(proc_file)
    if not proc_file.name.endswith("_PROC"):
        return None
    return proc_file.with_name(f"{proc_file.name[:-len('_PROC')]}_DLC.hdf5")


def dio_folders(rec_epoch):
    """*.DIO export folders inside a .rec epoch folder (trodes_io.DIO's rule)."""
    try:
        return [f for f in Path(rec_epoch).iterdir()
                if f.is_dir() and f.name.endswith(".DIO")]
    except OSError:
        return []


def has_din(rec_epoch, channel):
    """True if any DIO folder actually holds the Din<channel>.dat that sync reads."""
    for folder in dio_folders(rec_epoch):
        try:
            if any(f.is_file() and f.name.endswith(f"Din{channel}.dat")
                   for f in folder.iterdir()):
                return True
        except OSError:
            continue
    return False


def resolve_rec_epoch(rec_epoch, channel):
    """Copy of a .rec epoch folder that has the DIO export, as video_ephys_sync picks it.

    Returns (resolved_path_or_None, has_dio). A .rec folder often exists on BOTH
    servers while Trodes DIO export only ran on one of them, so plain existence
    is not enough.
    """
    if rec_epoch is None:
        return None, False
    candidates = [Path(rec_epoch)]
    mirrored = mirror_on_backup_server(rec_epoch)
    if mirrored is not None:
        candidates.insert(0, mirrored)
    existing = [c for c in candidates if _exists(c)]
    for candidate in existing:
        if has_din(candidate, channel):
            return candidate, True
    return (existing[0] if existing else None), False


def nwb_prefixes_in(folder):
    """Distinct '<prefix>' of the <prefix>sh<N>.nwb files in a folder, both servers.

    Lets the GUI fill in nwb_session_name from whatever a browsed folder really
    holds, instead of assuming it matches the folder name.
    """
    prefixes = set()
    for candidate in (mirror_on_backup_server(folder), Path(folder)):
        if candidate is None:
            continue
        try:
            for path in candidate.glob("*sh*.nwb"):
                match = re.fullmatch(r"(.+)sh\d+\.nwb", path.name)
                if match:
                    prefixes.add(match.group(1))
        except OSError:
            continue
    return sorted(prefixes)


def find_nwb(rec_folder, session_prefix, shank):
    """Per-shank NWB as extract_sleep_lfp.resolve_nwb_path would find it.

    Returns (path_or_None, note). `note` explains a non-obvious match (date
    format differs, or a unique glob match) or why nothing was chosen.
    """
    suffix = f"sh{shank}.nwb"
    prefixes = [session_prefix]

    # NWB exports are inconsistent about using YYMMDD versus YYYYMMDD.
    date_match = re.fullmatch(r"(.+_)(\d{6}|\d{8})", session_prefix or "")
    if date_match:
        stem, date = date_match.groups()
        prefixes.append(f"{stem}{'20' + date if len(date) == 6 else date[2:]}")

    folders = [f for f in (mirror_on_backup_server(rec_folder), Path(rec_folder))
               if f is not None]
    for folder in folders:
        for prefix in prefixes:
            candidate = folder / f"{prefix}{suffix}"
            if _exists(candidate):
                note = None if prefix == session_prefix else "date-format match"
                return candidate, note

    # Last resort, same as the script: a UNIQUE file for this animal/shank.
    animal_prefix = (session_prefix or "").rsplit("_", 1)[0]
    matches = []
    for folder in folders:
        try:
            matches += [p for p in folder.glob(f"{animal_prefix}_*{suffix}") if p.is_file()]
        except OSError:
            continue
    unique = sorted({p.name: p for p in matches}.values(), key=lambda p: p.name)
    if len(unique) == 1:
        return unique[0], "unique shank match"
    if len(unique) > 1:
        return None, f"ambiguous: {len(unique)} files match {animal_prefix}_*{suffix}"
    return None, None


# =============================================================================
# DAY PREFLIGHT
# =============================================================================

def session_preflight(entry, session, ctx):
    """What every stage will find on disk for one sleep session (pre/post)."""
    cfg = entry.get(session) or {}
    start, end = cfg.get("start_sample"), cfg.get("end_sample")
    out = {
        "start": start,
        "end": end,
        "active": not (start is None and end is None),
        "duration_s": (end - start) / ctx["fs"] if (start is not None and end is not None)
                      else None,
        "suffix": f"_{session}",
    }
    label = f"{ctx['session_name']}_{session}"
    low_freq = ctx["low_freq"]

    # 1. sync: .rec epoch (DIO) + PROC -> sync_times_<session>.pkl beside the NWBs
    rec_epoch = cfg.get("rec_file_folder")
    resolved, dio = resolve_rec_epoch(rec_epoch, ctx["dio_channel"])
    out["rec_epoch"] = Path(rec_epoch) if rec_epoch else None
    out["rec_epoch_resolved"] = resolved
    out["has_dio"] = dio
    sync_name = f"sync_times_{session}.pkl"
    out["sync_target"] = output_target(ctx["rec_folder"]) / sync_name
    out["sync_pkl"] = find_any(Path(ctx["rec_folder"]) / sync_name)

    # 2. velocity: PROC (+ DLC keypoints) -> <session video name>_velocity_*.pkl
    proc_file = cfg.get("proc_file")
    out["proc_file"] = Path(proc_file) if proc_file else None
    out["proc_found"] = find_any(proc_file)
    out["dlc_found"] = find_any(dlc_file_for_proc(proc_file)) if proc_file else None
    if proc_file:
        vel_name = velocity_output_name(proc_file, ctx["velocity_source"])
        out["velocity_target"] = output_target(Path(proc_file).parent) / vel_name
        # The plotting stage looks in video_folder; the velocity stage writes
        # beside the PROC file. They are normally the same folder - check both.
        out["velocity_pkl"] = (find_any(Path(proc_file).parent / vel_name)
                               or find_any(ctx["video_folder"] / vel_name))
    else:
        out["velocity_target"] = None
        out["velocity_pkl"] = None

    # 3-4. per-shank LFP traces and spectrograms
    out["lfp"] = {}
    out["spectrograms"] = {}
    for shank in ctx["shanks"]:
        out["lfp"][shank] = find_any(low_freq / f"{label}_sh{shank}_lfp_traces.npz")
        out["spectrograms"][shank] = find_any(low_freq / f"{label}_sh{shank}_spectrograms.npz")

    # 5. band powers / PC1 pickle (all shanks in one file)
    out["bands"] = find_any(low_freq / f"{label}_all_shanks_band_powers.pkl")

    # 6. figures written next to whichever copy of the pkl was used
    fig_folder = output_target((out["bands"].parent if out["bands"] else low_freq)
                               / "spectrogram")
    out["fig_folder"] = fig_folder
    out["figures"] = {}
    for shank in ctx["shanks"]:
        try:
            out["figures"][shank] = len(list(
                fig_folder.glob(f"{label}_sh{shank}_ch*_full_recording*.png")))
        except OSError:
            out["figures"][shank] = 0
    return out


def find_day_entry(registry, animal, date):
    """Registry entry for an animal-day, accepting a pre-animal bare-date key."""
    entry = registry.get(day_key(animal, date))
    if entry is not None:
        return entry
    legacy = registry.get(str(date))
    if legacy is not None and (not animal or entry_animal(legacy) == animal):
        return legacy
    return None


def preflight(animal, date, shanks, dio_channel, velocity_source="proc_center"):
    """Inspect a registered animal-day and report what each stage will find."""
    info = {"animal": animal, "date": date, "registered": False, "error": None,
            "shanks": list(shanks), "dio_channel": dio_channel,
            "velocity_source": velocity_source, "sessions": {}, "nwb": {}}

    registry = load_registry()
    entry = find_day_entry(registry, animal, date)
    if entry is None:
        known = ", ".join(f"{a}/{d}" for a, d, _ in registered_days(registry)) or "none yet"
        info["error"] = (
            f"'{animal} {date}' is not registered in {REGISTRY_FILE.name}. "
            f"Use 'Set up this day...' to add its paths (set_sleep_day.py).\n"
            f"  Registered: {known}")
        return info
    info["animal"] = entry_animal(entry) or animal

    info["registered"] = True
    info["entry"] = entry
    rec_folder = Path(entry["rec_folder"])
    # Mirrors sleep_pipeline_config: session_name comes from the folder name,
    # low_freq / video sit beside it.
    info["rec_folder"] = rec_folder
    info["rec_folder_found"] = find_any(rec_folder)
    info["session_name"] = rec_folder.stem.split(".")[0]
    info["nwb_session_name"] = entry.get("nwb_session_name")
    info["low_freq"] = rec_folder / "low_freq"
    info["video_folder"] = rec_folder.parent / "video"

    for shank in shanks:
        path, note = find_nwb(rec_folder, info["nwb_session_name"], shank)
        info["nwb"][shank] = {"path": path, "note": note}

    values = read_config_values()
    ctx = {
        "fs": float(values.get("original_fs", 30000)),
        "session_name": info["session_name"],
        "rec_folder": rec_folder,
        "low_freq": info["low_freq"],
        "video_folder": info["video_folder"],
        "shanks": shanks,
        "dio_channel": dio_channel,
        "velocity_source": velocity_source,
    }
    for session in SESSIONS:
        info["sessions"][session] = session_preflight(entry, session, ctx)
    return info


# =============================================================================
# STAGES
# =============================================================================

# key, label, script
STAGES = [
    ("sync", "1. Video/DIO sync", "video_ephys_sync.py"),
    ("velocity", "2. Velocity", "proc_func_velocity.py"),
    ("lfp", "3. Extract LFP", "extract_sleep_lfp.py"),
    ("spectrograms", "4. Spectrograms", "compute_sleep_spectrograms.py"),
    ("features", "5. Band powers", "compute_sleep_features.py"),
    ("plots", "6. Plot spectrograms", "plot_sleep_spectrograms.py"),
]
STAGE_LABELS = {key: label for key, label, _ in STAGES}
STAGE_SCRIPTS = {key: script for key, _, script in STAGES}


def missing_prerequisites(info, selected, session_filter="both"):
    """Warnings for stages whose inputs are neither on disk nor produced earlier in this run.

    Advisory only - a stage may still be worth running (it prints its own
    per-session skip messages), so these are shown as a confirmation, not a block.
    """
    warnings = []
    if "lfp" in selected:
        for shank in info["shanks"]:
            if info["nwb"][shank]["path"] is None:
                warnings.append(f"stage 3: no NWB found for shank {shank}")

    sessions = {name: data for name, data in info["sessions"].items()
                if data["active"] and session_filter in ("both", name)}
    for name, data in sessions.items():
        if "sync" in selected and not data["has_dio"]:
            warnings.append(f"stage 1 ({name}): no Din{info['dio_channel']}.dat under the "
                            f".rec epoch - Trodes DIO export has not run")
        if "velocity" in selected and data["proc_found"] is None:
            warnings.append(f"stage 2 ({name}): video _PROC tracking file not found")
        if ("velocity" in selected and info["velocity_source"] == "dlc_body"
                and data["proc_found"] is not None and data["dlc_found"] is None):
            warnings.append(f"stage 2 ({name}): velocity source is dlc_body but the "
                            f"_DLC.hdf5 companion file is missing")
        if "spectrograms" in selected and "lfp" not in selected:
            missing = [s for s in info["shanks"] if data["lfp"][s] is None]
            if missing:
                warnings.append(f"stage 4 ({name}): no LFP traces for shank(s) "
                                f"{missing} - run stage 3 first")
        if "features" in selected and "spectrograms" not in selected:
            missing = [s for s in info["shanks"] if data["spectrograms"][s] is None]
            if missing:
                warnings.append(f"stage 5 ({name}): no spectrograms for shank(s) "
                                f"{missing} - run stage 4 first")
        if "plots" in selected and "features" not in selected and data["bands"] is None:
            warnings.append(f"stage 6 ({name}): no band-powers pkl - run stage 5 first")
        if "plots" in selected and data["sync_pkl"] is None and "sync" not in selected:
            warnings.append(f"stage 6 ({name}): no sync_times{data['suffix']}.pkl - "
                            f"figures will have no velocity panel")
        if "plots" in selected and data["velocity_pkl"] is None and "velocity" not in selected:
            warnings.append(f"stage 6 ({name}): no velocity pkl - "
                            f"figures will have no velocity panel")
    if not sessions:
        warnings.append(f"no active sleep session for this day with sessions="
                        f"{session_filter} - every stage will exit immediately")
    return warnings


class PipelineGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Sleep pipeline")
        self.geometry("1180x920")
        self.minsize(960, 720)

        self.settings = self._load_settings()
        cfg = read_config_values()

        interpreters = find_interpreters()
        self.interpreter_var = tk.StringVar(
            value=self.settings.get("interpreter") or default_interpreter(interpreters))

        self.animal_var = tk.StringVar(value=cfg.get("ACTIVE_ANIMAL", ""))
        self.date_var = tk.StringVar(value=cfg.get("ACTIVE_DATE", ""))
        self.session_var = tk.StringVar(value=cfg.get("SESSION_FILTER") or "both")
        shanks = cfg.get("shanks") or []
        self.shanks_var = tk.StringVar(value=" ".join(str(s) for s in shanks))
        # Editable copies of the two registry fields that move when data is
        # copied between servers. They follow the registry as you switch days
        # (see _preflight_done) until you edit them, and "Save path" writes
        # them back - every stage reads them from the registry, never from here.
        self.rec_folder_var = tk.StringVar()
        self.nwb_prefix_var = tk.StringVar()
        self._registry_paths = ("", "")
        self.velocity_source_var = tk.StringVar(
            value=cfg.get("VELOCITY_SOURCE", "proc_center"))
        self.velocity_keypoints = cfg.get("VELOCITY_KEYPOINTS") or ()
        self.write_config_var = tk.BooleanVar(value=True)

        self.stage_vars = {k: tk.BooleanVar(value=self.settings.get("stages", {}).get(k, True))
                           for k, _, _ in STAGES}
        self.dio_channel_var = tk.StringVar(
            value=str(self.settings.get("dio_channel", DEFAULT_DIO_CHANNEL)))
        self.force_sync_var = tk.BooleanVar(value=self.settings.get("force_sync", False))
        self.velocity_overwrite_var = tk.BooleanVar(
            value=self.settings.get("velocity_overwrite", False))
        self.channels_var = tk.StringVar(value=self.settings.get("channels", ""))
        self.max_channels_var = tk.StringVar(value=self.settings.get("max_channels", ""))
        self.output_suffix_var = tk.StringVar(value=self.settings.get("output_suffix", ""))
        self.continue_var = tk.BooleanVar(value=self.settings.get("continue_on_error", True))

        self.status_var = tk.StringVar(value="Ready.")
        self.stage_labels = {}
        self.info = None
        self._proc = None
        self._stop = False
        self._running = False
        # Worker threads never touch widgets or Tk variables directly; they push
        # output lines and UI callables onto these, drained on the UI thread.
        self._log_queue = queue.Queue()
        self._ui_queue = queue.Queue()

        self._build_ui(interpreters)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self._drain_log)
        self.after(200, self.run_preflight)

    # -- settings ----------------------------------------------------------

    @staticmethod
    def _load_settings():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_settings(self):
        data = {
            "interpreter": self.interpreter_var.get(),
            "stages": {k: v.get() for k, v in self.stage_vars.items()},
            "dio_channel": self.dio_channel_var.get(),
            "force_sync": self.force_sync_var.get(),
            "velocity_overwrite": self.velocity_overwrite_var.get(),
            "channels": self.channels_var.get(),
            "max_channels": self.max_channels_var.get(),
            "output_suffix": self.output_suffix_var.get(),
            "continue_on_error": self.continue_var.get(),
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    # -- construction ------------------------------------------------------

    def _build_ui(self, interpreters):
        # Rows: 0 day, 1 preflight, 2 stages, 3 run bar, 4 log.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(4, weight=2)

        # -- Recording day -------------------------------------------------
        day = ttk.LabelFrame(self, text="Recording day", padding=(8, 6))
        day.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        day.columnconfigure(6, weight=1)

        ttk.Label(day, text="Animal:").grid(row=0, column=0, sticky="w")
        self.animal_combo = ttk.Combobox(day, textvariable=self.animal_var, width=10,
                                         values=registered_animals())
        self.animal_combo.grid(row=0, column=1, sticky="w", padx=(4, 12))
        self.animal_combo.bind("<<ComboboxSelected>>", lambda _e: self._animal_changed())
        self.animal_combo.bind("<Return>", lambda _e: self._animal_changed())

        ttk.Label(day, text="Date:").grid(row=0, column=2, sticky="w")
        self.date_combo = ttk.Combobox(day, textvariable=self.date_var, width=10,
                                       values=self._dates_for_animal())
        self.date_combo.grid(row=0, column=3, sticky="w", padx=(4, 12))
        self.date_combo.bind("<<ComboboxSelected>>", lambda _e: self.run_preflight())
        self.date_combo.bind("<Return>", lambda _e: self.run_preflight())

        ttk.Button(day, text="Set up this day...", command=self.open_day_setup).grid(
            row=0, column=4, padx=(0, 12))

        ttk.Label(day, text="Sessions:").grid(row=0, column=5, sticky="w")
        ttk.Combobox(day, textvariable=self.session_var, width=8, state="readonly",
                     values=["both", "pre", "post"]).grid(row=0, column=6, sticky="w",
                                                          padx=(4, 12))

        ttk.Label(day, text="Shanks:").grid(row=0, column=7, sticky="w")
        shanks_entry = ttk.Entry(day, textvariable=self.shanks_var, width=12)
        shanks_entry.grid(row=0, column=8, sticky="w", padx=(4, 0))
        shanks_entry.bind("<Return>", lambda _e: self.run_preflight())

        # NWB folder (registry rec_folder) - editable so a day whose data moved
        # server can be re-pointed without reopening the day setup window.
        ttk.Label(day, text="NWB folder:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        paths = ttk.Frame(day)
        paths.grid(row=1, column=1, columnspan=8, sticky="ew", padx=(4, 0), pady=(6, 0))
        paths.columnconfigure(0, weight=1)
        ttk.Entry(paths, textvariable=self.rec_folder_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(paths, text="Browse...", command=self.browse_rec_folder).grid(
            row=0, column=1, padx=(6, 0))
        ttk.Button(paths, text="Save path", command=self.save_rec_folder).grid(
            row=0, column=2, padx=(6, 0))

        ttk.Label(day, text="NWB prefix:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        prefix = ttk.Frame(day)
        prefix.grid(row=2, column=1, columnspan=8, sticky="ew", padx=(4, 0), pady=(4, 0))
        ttk.Entry(prefix, textvariable=self.nwb_prefix_var, width=30).grid(row=0, column=0,
                                                                          sticky="w")
        ttk.Label(prefix, text="(everything before \"sh<N>.nwb\"; filled in from the "
                               "folder you browse to. Save path writes both to the "
                               "registry.)",
                  foreground="#666").grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(day, text="Python:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        interp = ttk.Frame(day)
        interp.grid(row=3, column=1, columnspan=8, sticky="ew", padx=(4, 0), pady=(6, 0))
        interp.columnconfigure(0, weight=1)
        ttk.Combobox(interp, textvariable=self.interpreter_var,
                     values=interpreters).grid(row=0, column=0, sticky="ew")
        ttk.Button(interp, text="Browse...", command=self.browse_interpreter).grid(
            row=0, column=1, padx=(6, 0))
        ttk.Button(interp, text="Refresh preflight", command=self.run_preflight).grid(
            row=0, column=2, padx=(6, 0))
        ttk.Button(interp, text="Write config now", command=self.write_config).grid(
            row=0, column=3, padx=(6, 0))

        ttk.Checkbutton(day, text=f"Write {CONFIG_PATH.name} before running "
                                  f"(ACTIVE_ANIMAL / ACTIVE_DATE / SESSION_FILTER / "
                                  f"shanks / VELOCITY_SOURCE)",
                        variable=self.write_config_var).grid(
            row=4, column=0, columnspan=6, sticky="w", pady=(6, 0))
        ttk.Label(day, text=f"registry: {REGISTRY_FILE}", foreground="#666").grid(
            row=4, column=6, columnspan=3, sticky="e", pady=(6, 0))

        # -- Preflight -----------------------------------------------------
        pre = ttk.LabelFrame(self, text="Preflight - what is on disk for this day",
                             padding=(8, 6))
        pre.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        pre.columnconfigure(0, weight=1)
        pre.rowconfigure(0, weight=1)
        self.pre_text = tk.Text(pre, height=14, wrap="none", font=("Consolas", 9))
        self.pre_text.grid(row=0, column=0, sticky="nsew")
        pre_scroll = ttk.Scrollbar(pre, orient="vertical", command=self.pre_text.yview)
        pre_scroll.grid(row=0, column=1, sticky="ns")
        pre_scroll_x = ttk.Scrollbar(pre, orient="horizontal", command=self.pre_text.xview)
        pre_scroll_x.grid(row=1, column=0, sticky="ew")
        self.pre_text.configure(yscrollcommand=pre_scroll.set,
                                xscrollcommand=pre_scroll_x.set, state="disabled")
        for tag, color in (("ok", "#1a7f37"), ("warn", "#b8860b"),
                           ("bad", "#c00000"), ("head", "#333333")):
            self.pre_text.tag_configure(tag, foreground=color)
        self.pre_text.tag_configure("head", font=("Consolas", 9, "bold"))

        # -- Stages --------------------------------------------------------
        stages = ttk.LabelFrame(self, text="Stages", padding=(8, 6))
        stages.grid(row=2, column=0, sticky="ew", padx=10, pady=4)

        row0 = ttk.Frame(stages)
        row0.pack(fill="x")
        for key, label, _ in STAGES:
            cell = ttk.Frame(row0)
            cell.pack(side="left", padx=(0, 14))
            ttk.Checkbutton(cell, text=label, variable=self.stage_vars[key]).pack(side="left")
            status = ttk.Label(cell, text="-", width=3, foreground="#888")
            status.pack(side="left")
            self.stage_labels[key] = status
        ttk.Button(row0, text="All", width=5,
                   command=lambda: self._set_all_stages(True)).pack(side="left")
        ttk.Button(row0, text="None", width=6,
                   command=lambda: self._set_all_stages(False)).pack(side="left", padx=(4, 0))

        row1 = ttk.Frame(stages)
        row1.pack(fill="x", pady=(8, 0))
        ttk.Label(row1, text="1. DIO channel:").pack(side="left")
        ttk.Entry(row1, textvariable=self.dio_channel_var, width=5).pack(side="left", padx=(4, 8))
        ttk.Checkbutton(row1, text="save sync even if edges don't match",
                        variable=self.force_sync_var).pack(side="left", padx=(0, 16))
        ttk.Label(row1, text="2. Velocity from:").pack(side="left")
        velocity_combo = ttk.Combobox(row1, textvariable=self.velocity_source_var,
                                      width=13, state="readonly",
                                      values=["proc_center", "dlc_body"])
        velocity_combo.pack(side="left", padx=(4, 4))
        velocity_combo.bind("<<ComboboxSelected>>", lambda _e: self.run_preflight())
        ttk.Label(row1, text=f"({', '.join(self.velocity_keypoints)})"
                             if self.velocity_keypoints else "",
                  foreground="#666").pack(side="left", padx=(0, 12))
        ttk.Checkbutton(row1, text="recompute if it already exists",
                        variable=self.velocity_overwrite_var).pack(side="left")

        row2 = ttk.Frame(stages)
        row2.pack(fill="x", pady=(8, 0))
        ttk.Label(row2, text="6. Channels:").pack(side="left")
        ttk.Entry(row2, textvariable=self.channels_var, width=18).pack(side="left", padx=(4, 8))
        ttk.Label(row2, text="max channels:").pack(side="left")
        ttk.Entry(row2, textvariable=self.max_channels_var, width=6).pack(side="left", padx=(4, 8))
        ttk.Label(row2, text="figure suffix:").pack(side="left")
        ttk.Entry(row2, textvariable=self.output_suffix_var, width=14).pack(side="left", padx=(4, 8))
        ttk.Label(row2, text="(channels blank = every channel on each shank)",
                  foreground="#666").pack(side="left")
        ttk.Checkbutton(row2, text="continue on error",
                        variable=self.continue_var).pack(side="right")

        # -- Run bar + log -------------------------------------------------
        bar = ttk.Frame(self, padding=(10, 4))
        bar.grid(row=3, column=0, sticky="ew")
        self.run_button = ttk.Button(bar, text="Run pipeline", command=self.start_pipeline)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(bar, text="Stop", command=self.stop_pipeline,
                                      state="disabled")
        self.stop_button.pack(side="left", padx=6)
        ttk.Button(bar, text="Save log...", command=self.save_log).pack(side="left")
        ttk.Button(bar, text="Clear log", command=self.clear_log).pack(side="left", padx=6)
        ttk.Button(bar, text="Open low_freq folder",
                   command=self.open_output_folder).pack(side="left")
        ttk.Label(bar, textvariable=self.status_var).pack(side="right")

        log = ttk.LabelFrame(self, text="Log", padding=(8, 6))
        log.grid(row=4, column=0, sticky="nsew", padx=10, pady=(4, 8))
        log.columnconfigure(0, weight=1)
        log.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log, wrap="none", font=("Consolas", 9),
                                background="#1e1e1e", foreground="#dcdcdc",
                                insertbackground="#dcdcdc")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_y = ttk.Scrollbar(log, orient="vertical", command=self.log_text.yview)
        log_y.grid(row=0, column=1, sticky="ns")
        log_x = ttk.Scrollbar(log, orient="horizontal", command=self.log_text.xview)
        log_x.grid(row=1, column=0, sticky="ew")
        self.log_text.configure(yscrollcommand=log_y.set, xscrollcommand=log_x.set,
                                state="disabled")
        self.log_text.tag_configure("cmd", foreground="#4ec9b0")
        self.log_text.tag_configure("banner", foreground="#dcdcaa")
        self.log_text.tag_configure("bad", foreground="#f48771")
        self.log_text.tag_configure("ok", foreground="#89d185")

    def _set_all_stages(self, value):
        for var in self.stage_vars.values():
            var.set(value)

    def browse_interpreter(self):
        path = filedialog.askopenfilename(title="Select python executable")
        if path:
            self.interpreter_var.set(path)

    # -- NWB folder --------------------------------------------------------

    def browse_rec_folder(self):
        """Pick the folder holding this day's per-shank NWB files."""
        current = self.rec_folder_var.get().strip()
        start = None
        for candidate in (current, str(Path(current).parent) if current else ""):
            if candidate and Path(candidate).is_dir():
                start = candidate
                break
        chosen = filedialog.askdirectory(
            title="Select the day's NWB folder (rec_folder)", initialdir=start)
        if not chosen:
            return

        chosen = str(Path(chosen))
        self.rec_folder_var.set(chosen)

        # Take the NWB prefix from what the folder actually holds; fall back to
        # the folder name, which is the convention on every day so far.
        prefixes = nwb_prefixes_in(chosen)
        if len(prefixes) == 1:
            self.nwb_prefix_var.set(prefixes[0])
            self.log(f"NWB prefix from {Path(chosen).name}: {prefixes[0]}\n", "ok")
        elif len(prefixes) > 1:
            self.log(f"Folder holds several NWB prefixes ({', '.join(prefixes)}) - "
                     f"leaving 'NWB prefix' as {self.nwb_prefix_var.get()!r}.\n", "bad")
        else:
            self.nwb_prefix_var.set(Path(chosen).name)
            self.log(f"No *sh<N>.nwb in {chosen} - guessing prefix from the folder "
                     f"name: {Path(chosen).name}\n", "bad")
        self.run_preflight()

    def save_rec_folder(self):
        """Write the NWB folder / prefix fields back into the registry."""
        animal = self.animal_var.get().strip()
        date = self.date_var.get().strip()
        rec_folder = self.rec_folder_var.get().strip()
        prefix = self.nwb_prefix_var.get().strip()
        if not animal or not date:
            messagebox.showerror("Save path", "Pick the animal and date first.")
            return
        if not rec_folder or not prefix:
            messagebox.showerror("Save path",
                                 "Both the NWB folder and the NWB prefix are required.")
            return

        entry = find_day_entry(load_registry(), animal, date)
        if entry is None:
            messagebox.showerror(
                "Save path",
                f"'{animal} {date}' is not registered yet, so there is nothing to "
                f"re-point. Use 'Set up this day...' to add it.")
            return

        pending = {field: (entry.get(field), value)
                   for field, value in (("rec_folder", rec_folder),
                                        ("nwb_session_name", prefix))
                   if value != entry.get(field)}
        if not pending:
            self.status_var.set("NWB folder already matches the registry.")
            return

        detail = "\n\n".join(f"{field}:\n  old: {old}\n  new: {new}"
                             for field, (old, new) in pending.items())
        if not messagebox.askyesno(
                "Save path",
                f"Update {REGISTRY_FILE.name} for {animal} {date}?\n\n{detail}\n\n"
                f"Sample windows, .rec epochs and PROC files are left unchanged."):
            return

        try:
            changed = update_day_paths(animal, date, rec_folder, prefix)
        except (KeyError, OSError) as exc:
            messagebox.showerror("Save path", str(exc))
            return

        self.log(f"Updated {REGISTRY_FILE.name} for {animal} {date}:\n", "banner")
        for field, (old, new) in changed.items():
            self.log(f"    {field}: {old} -> {new}\n")
        self._registry_paths = (rec_folder, prefix)
        self.run_preflight()

    # -- day setup ---------------------------------------------------------

    def _dates_for_animal(self, animal=None):
        """Registered dates for one animal (all of them when no animal is set)."""
        animal = (self.animal_var.get().strip() if animal is None else animal)
        return [date for day_animal, date, _ in registered_days()
                if not animal or day_animal == animal]

    def _animal_changed(self):
        """Narrow the date list to the chosen animal, then re-scan."""
        dates = self._dates_for_animal()
        self.date_combo["values"] = dates
        # Keep a date that this animal actually has; otherwise take its first.
        if self.date_var.get().strip() not in dates and dates:
            self.date_var.set(dates[0])
        self.run_preflight()

    def open_day_setup(self):
        """Launch set_sleep_day.py for the current animal-day, then re-read the registry.

        Run as its own process (not an embedded Toplevel) so a mistake in there
        cannot take this panel down, and so it keeps working standalone.
        """
        animal = self.animal_var.get().strip()
        date = self.date_var.get().strip()
        argv = [self.interpreter_var.get().strip() or sys.executable, str(SETUP_SCRIPT)]
        if animal:
            argv += ["--animal", animal]
        if date:
            argv += ["--date", date]
        self.log(f"\nOpening day setup: {subprocess.list2cmdline(argv)}\n", "cmd")
        try:
            proc = subprocess.Popen(argv, cwd=str(SLEEP_DIR), env=self._subprocess_env("setup"))
        except OSError as exc:
            messagebox.showerror("Day setup", f"Could not start set_sleep_day.py:\n{exc}")
            return
        self.status_var.set("Day setup window open - close it to refresh preflight.")

        def wait():
            proc.wait()
            self._post(self._day_setup_closed)

        threading.Thread(target=wait, daemon=True).start()

    def _day_setup_closed(self):
        self.animal_combo["values"] = registered_animals()
        self.date_combo["values"] = self._dates_for_animal()
        self.log("Day setup closed - re-reading "
                 f"{REGISTRY_FILE.name}.\n", "banner")
        self.run_preflight()

    def ensure_registered(self):
        """True if this animal-day is registered; otherwise offer the setup window."""
        animal = self.animal_var.get().strip()
        date = self.date_var.get().strip()
        if not animal or not date:
            messagebox.showerror(
                "Cannot run", "Enter the animal (e.g. CnL42) and recording date "
                              "(e.g. 260324).")
            return False
        if find_day_entry(load_registry(), animal, date) is not None:
            return True
        open_setup = messagebox.askyesno(
            "Day not registered",
            f"'{animal} {date}' has no entry in {REGISTRY_FILE.name}, so the pipeline "
            f"has no paths for it (rec_folder, sample windows, .rec epochs, PROC "
            f"files).\n\nOpen the day setup window (set_sleep_day.py) for "
            f"{animal} {date} now?")
        if open_setup:
            self.open_day_setup()
        return False

    # -- preflight ---------------------------------------------------------

    def run_preflight(self):
        animal = self.animal_var.get().strip()
        date = self.date_var.get().strip()
        self.animal_combo["values"] = registered_animals()
        self.date_combo["values"] = self._dates_for_animal(animal)
        if not animal or not date:
            self._set_pre_text([("Enter or pick an animal and a recording date.\n",
                                 "warn")])
            return
        try:
            shanks = parse_int_list(self.shanks_var.get(), "Shanks")
            dio_channel = int(self.dio_channel_var.get().strip() or DEFAULT_DIO_CHANNEL)
        except ValueError as exc:
            self._set_pre_text([(f"{exc}\n", "bad")])
            return

        self._set_pre_text([("Scanning disk...\n", None)])
        self.status_var.set("Preflight scanning...")
        velocity_source = self.velocity_source_var.get()

        def runner():
            result = self._safe(
                lambda: preflight(animal, date, shanks, dio_channel, velocity_source))
            self._post(lambda: self._preflight_done(result))

        threading.Thread(target=runner, daemon=True).start()

    def _preflight_done(self, info):
        if isinstance(info, Exception):
            self._set_pre_text([(f"Preflight failed: {info}\n", "bad")])
            self.status_var.set("Preflight failed.")
            return
        self.info = info
        self._sync_path_fields(info)
        self._render_preflight(info)
        self.status_var.set(
            "Preflight done." if info["registered"]
            else f"{info['animal']} {info['date']} is not registered.")

    def _sync_path_fields(self, info):
        """Follow the registry's paths, unless they have been edited by hand.

        Switching animal/date should show that day's paths, but an edit in
        progress (or a browsed folder not yet saved) must survive a re-scan.
        """
        if not info["registered"]:
            self._registry_paths = ("", "")
            return
        entry = info["entry"]
        registry_paths = (str(entry.get("rec_folder") or ""),
                          str(entry.get("nwb_session_name") or ""))
        fields = (self.rec_folder_var.get().strip(), self.nwb_prefix_var.get().strip())
        if fields == self._registry_paths or fields == ("", ""):
            self.rec_folder_var.set(registry_paths[0])
            self.nwb_prefix_var.set(registry_paths[1])
        self._registry_paths = registry_paths

    @staticmethod
    def _safe(fn):
        try:
            return fn()
        except Exception as exc:
            return exc

    def _set_pre_text(self, chunks):
        self.pre_text.configure(state="normal")
        self.pre_text.delete("1.0", "end")
        for text, tag in chunks:
            self.pre_text.insert("end", text, tag or "")
        self.pre_text.configure(state="disabled")

    def _render_preflight(self, info):
        chunks = []

        def add(text, tag=None):
            chunks.append((text, tag))

        def status(found, todo_is_fine=True):
            if found:
                return "[ok]   ", "ok"
            return ("[todo] ", "warn") if todo_is_fine else ("[MISS] ", "bad")

        if not info["registered"]:
            add(f"{info['animal']}  {info['date']}\n", "head")
            add(f"  {info['error']}\n", "bad")
            self._set_pre_text(chunks)
            return

        add(f"{info['animal']}  {info['date']}   "
            f"session_name={info['session_name']}\n", "head")
        mark, tag = status(info["rec_folder_found"], todo_is_fine=False)
        add(f"  rec_folder  {mark}{info['rec_folder']}\n", tag)
        # Everything below reflects the REGISTRY, which is what the stages read.
        edited = (self.rec_folder_var.get().strip(), self.nwb_prefix_var.get().strip())
        if edited != self._registry_paths and edited != ("", ""):
            add(f"  NOTE: the NWB folder/prefix fields differ from the registry - "
                f"click 'Save path' to make this scan and the run use them.\n", "warn")
        add(f"  low_freq    {info['low_freq']}\n")
        add(f"  video       {info['video_folder']}\n")

        add("\nNWB per shank (stage 3 input)\n", "head")
        if not info["shanks"]:
            add("  no shanks selected - stages 3-6 have nothing to process\n", "bad")
        for shank in info["shanks"]:
            entry = info["nwb"][shank]
            mark, tag = status(entry["path"] is not None, todo_is_fine=False)
            name = entry["path"].name if entry["path"] else \
                f"{info['nwb_session_name']}sh{shank}.nwb not found"
            note = f"   ({entry['note']})" if entry["note"] else ""
            add(f"  sh{shank}  {mark}{name}{note}\n", tag)

        session_filter = self.session_var.get()
        for name in SESSIONS:
            data = info["sessions"][name]
            skipped = session_filter != "both" and session_filter != name
            if not data["active"]:
                add(f"\nSESSION {name}: not recorded this day "
                    f"(start_sample and end_sample both blank)\n", "head")
                continue

            window = (f"samples {data['start']}-{data['end']}"
                      if data["duration_s"] is not None else
                      f"samples {data['start']}-{data['end']} (open bound)")
            duration = f"  =  {data['duration_s']:.0f} s ({data['duration_s'] / 60:.0f} min)" \
                if data["duration_s"] is not None else ""
            add(f"\nSESSION {name}   {window}{duration}"
                f"{'   [SKIPPED by session filter]' if skipped else ''}\n", "head")

            # 1. sync
            mark, tag = status(data["sync_pkl"] is not None)
            add(f"  1 sync      {mark}{data['sync_pkl'].name if data['sync_pkl'] else data['sync_target'].name}\n", tag)
            if data["rec_epoch"] is None:
                add("              [MISS] no .rec epoch folder registered for this session\n", "bad")
            elif data["rec_epoch_resolved"] is None:
                add(f"              [MISS] .rec epoch not found: {data['rec_epoch']}\n", "bad")
            elif not data["has_dio"]:
                add(f"              [MISS] no Din{info['dio_channel']}.dat under "
                    f"{data['rec_epoch_resolved'].name} - run the Trodes DIO export\n", "bad")
            else:
                add(f"              [ok]   DIO in {data['rec_epoch_resolved'].name}\n", "ok")

            # 2. velocity
            mark, tag = status(data["velocity_pkl"] is not None)
            vel_name = (data["velocity_pkl"].name if data["velocity_pkl"]
                        else (data["velocity_target"].name if data["velocity_target"]
                              else "no PROC file registered"))
            add(f"  2 velocity  {mark}{vel_name}\n", tag)
            if data["proc_file"] is None:
                add("              [MISS] no video _PROC file registered for this session\n", "bad")
            elif data["proc_found"] is None:
                add(f"              [MISS] PROC file not found: {data['proc_file']}\n", "bad")
            else:
                add(f"              [ok]   PROC {data['proc_found'].name}\n", "ok")
                if info["velocity_source"] == "dlc_body":
                    mark, tag = status(data["dlc_found"] is not None, todo_is_fine=False)
                    add(f"              {mark}DLC  "
                        f"{data['dlc_found'].name if data['dlc_found'] else 'companion _DLC.hdf5 not found'}"
                        f"\n", tag)

            # 3-4. per shank
            for shank in info["shanks"]:
                mark, tag = status(data["lfp"][shank] is not None)
                add(f"  3 lfp  sh{shank} {mark}"
                    f"{info['session_name']}_{name}_sh{shank}_lfp_traces.npz\n", tag)
                mark, tag = status(data["spectrograms"][shank] is not None)
                add(f"  4 spec sh{shank} {mark}"
                    f"{info['session_name']}_{name}_sh{shank}_spectrograms.npz\n", tag)

            # 5. band powers
            mark, tag = status(data["bands"] is not None)
            add(f"  5 bands     {mark}{info['session_name']}_{name}_all_shanks_band_powers.pkl\n", tag)

            # 6. figures
            total = sum(data["figures"].get(s, 0) for s in info["shanks"])
            mark, tag = status(total > 0)
            per_shank = ", ".join(f"sh{s}: {data['figures'].get(s, 0)}" for s in info["shanks"])
            add(f"  6 figures   {mark}{total} png in spectrogram/  ({per_shank})\n", tag)

        self._set_pre_text(chunks)

    # -- config ------------------------------------------------------------

    def collect_config(self):
        """GUI fields as config values, raising ValueError on bad input."""
        animal = self.animal_var.get().strip()
        if not animal:
            raise ValueError("Enter the animal id (e.g. CnL42).")
        date = self.date_var.get().strip()
        if not date:
            raise ValueError("Enter a recording date (e.g. 260324).")
        shanks = parse_int_list(self.shanks_var.get(), "Shanks")
        if not shanks:
            raise ValueError("At least one shank is required (e.g. '5 7').")
        session = self.session_var.get()
        return {
            "ACTIVE_ANIMAL": animal,
            "ACTIVE_DATE": date,
            "SESSION_FILTER": None if session == "both" else session,
            "shanks": shanks,
            "VELOCITY_SOURCE": self.velocity_source_var.get(),
        }

    def write_config(self):
        try:
            changes = write_config_values(self.collect_config())
        except (ValueError, KeyError, OSError) as exc:
            messagebox.showerror("Config", str(exc))
            return False
        if changes:
            self.log(f"Updated {CONFIG_PATH.name} "
                     f"(backup: {CONFIG_PATH.with_suffix('.py.bak').name}):\n", "banner")
            for name, old, new in changes:
                self.log(f"    {name}: {old} -> {new}\n")
        else:
            self.log(f"{CONFIG_PATH.name} already matches the GUI fields.\n")
        return True

    # -- command construction ----------------------------------------------

    def build_plan(self):
        """[(stage_key, argv)] for the selected stages, in pipeline order."""
        python = self.interpreter_var.get().strip()
        if not python or not Path(python).exists():
            raise ValueError(f"Python interpreter not found: {python!r}")

        selected = [k for k, _, _ in STAGES if self.stage_vars[k].get()]
        if not selected:
            raise ValueError("No stages selected.")

        try:
            dio_channel = int(self.dio_channel_var.get().strip() or DEFAULT_DIO_CHANNEL)
        except ValueError:
            raise ValueError(f"DIO channel must be an integer, got "
                             f"{self.dio_channel_var.get()!r}")
        channels = parse_int_list(self.channels_var.get(), "Channels")
        max_channels = self.max_channels_var.get().strip()
        if max_channels:
            try:
                int(max_channels)
            except ValueError:
                raise ValueError(f"Max channels must be an integer or blank, "
                                 f"got {max_channels!r}")

        plan = []
        for key in selected:
            argv = [python, STAGE_SCRIPTS[key]]
            if key == "sync":
                argv += ["--dio-channel", str(dio_channel)]
                if self.force_sync_var.get():
                    argv.append("--force-save-sync")
            elif key == "velocity":
                # Passed explicitly so the run matches the GUI even when
                # "write config before running" is off.
                argv += ["--source", self.velocity_source_var.get()]
                if self.velocity_overwrite_var.get():
                    argv.append("--overwrite")
            elif key == "plots":
                # shanks come from the config write; only the per-figure options
                # live in the GUI.
                if channels:
                    argv += ["--channels"] + [str(c) for c in channels]
                if max_channels:
                    argv += ["--max-channels", max_channels]
                if self.output_suffix_var.get().strip():
                    argv += ["--output-suffix", self.output_suffix_var.get().strip()]
            plan.append((key, argv))
        return plan

    def _subprocess_env(self, stage):
        env = os.environ.copy()
        # The sleep scripts import sleep_pipeline_config flat and reach the repo
        # root for trodes_io / server_fallback, so expose both roots.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT), str(SLEEP_DIR), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if stage != "setup":
            # Every stage only saves figures; a headless backend keeps figure
            # windows from stealing focus (and blocking) during long runs.
            env["MPLBACKEND"] = "Agg"
        return env

    # -- running -----------------------------------------------------------

    def start_pipeline(self):
        if self._running:
            return
        if not self.ensure_registered():
            return
        try:
            plan = self.build_plan()
        except ValueError as exc:
            messagebox.showerror("Cannot run", str(exc))
            return

        if (self.info is not None and self.info.get("registered")
                and self.info["date"] == self.date_var.get().strip()
                and self.info["animal"] == self.animal_var.get().strip()):
            warnings = missing_prerequisites(self.info, [key for key, _ in plan],
                                             self.session_var.get())
            if warnings and not messagebox.askyesno(
                    "Missing inputs",
                    "Preflight found missing inputs:\n\n  "
                    + "\n  ".join(warnings)
                    + "\n\nRun the selected stages anyway?"):
                return

        if self.write_config_var.get() and not self.write_config():
            return

        self._save_settings()
        self._stop = False
        self._running = True
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        for key, _, _ in STAGES:
            self._set_stage_status(key, "pending" if self.stage_vars[key].get() else "skip")

        opts = {"continue_on_error": self.continue_var.get()}
        threading.Thread(target=self._run_plan, args=(plan, opts), daemon=True).start()

    def _run_plan(self, plan, opts):
        started = time.time()
        failures = []
        for index, (key, argv) in enumerate(plan, start=1):
            if self._stop:
                self._post(lambda k=key: self.log(
                    f"\nStopped before {STAGE_LABELS[k]}.\n", "bad"))
                break

            self._post(lambda k=key: self._set_stage_status(k, "running"))
            self._post(lambda k=key, i=index, n=len(plan): self.status_var.set(
                f"Stage {i}/{n}: {STAGE_LABELS[k]} running..."))
            self._post(lambda k=key: self.log(
                f"\n{'=' * 72}\n{STAGE_LABELS[k]}\n{'=' * 72}\n", "banner"))
            self._post(lambda a=argv: self.log(
                f"$ cd {SLEEP_DIR}\n$ {subprocess.list2cmdline(a)}\n", "cmd"))

            stage_started = time.time()
            code = self._run_process(key, argv)
            elapsed = time.time() - stage_started

            if code == 0:
                self._post(lambda k=key: self._set_stage_status(k, "done"))
                self._post(lambda k=key, e=elapsed: self.log(
                    f"\n{STAGE_LABELS[k]} finished in {e:.1f}s.\n", "ok"))
            else:
                failures.append(key)
                self._post(lambda k=key: self._set_stage_status(k, "failed"))
                self._post(lambda k=key, c=code, e=elapsed: self.log(
                    f"\n{STAGE_LABELS[k]} FAILED (exit {c}) after {e:.1f}s.\n", "bad"))
                if not opts["continue_on_error"]:
                    self._post(lambda: self.log(
                        "Stopping: 'continue on error' is off.\n", "bad"))
                    break

        total = time.time() - started
        self._post(lambda: self._pipeline_finished(failures, total))

    def _run_process(self, stage, argv):
        try:
            self._proc = subprocess.Popen(
                argv, cwd=str(SLEEP_DIR), env=self._subprocess_env(stage),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                errors="replace", bufsize=1,
            )
        except OSError as exc:
            self._post(lambda: self.log(f"Could not start process: {exc}\n", "bad"))
            return -1

        for line in self._proc.stdout:
            self._log_queue.put((line, None))
        self._proc.wait()
        code = self._proc.returncode
        self._proc = None
        return code

    def _pipeline_finished(self, failures, total):
        self._running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if self._stop:
            self.status_var.set(f"Stopped after {total:.0f}s.")
        elif failures:
            self.status_var.set(
                f"Finished with {len(failures)} failed stage(s) in {total:.0f}s.")
        else:
            self.status_var.set(f"All stages completed in {total:.0f}s.")
        self.log(f"\n{'=' * 72}\n{self.status_var.get()}\n{'=' * 72}\n",
                 "bad" if failures else "ok")
        self.run_preflight()

    def stop_pipeline(self):
        self._stop = True
        proc = self._proc
        if proc is None:
            return
        self.log("\nStop requested - terminating current stage...\n", "bad")
        try:
            if os.name == "nt":
                # extract_sleep_lfp spawns worker processes; /T kills the tree.
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                               capture_output=True)
            else:
                proc.terminate()
        except OSError as exc:
            self.log(f"Could not terminate: {exc}\n", "bad")

    def _set_stage_status(self, key, state):
        text, color = {
            "pending": ("-", "#888"),
            "running": (">>", "#0066cc"),
            "done": ("OK", "#1a7f37"),
            "failed": ("X", "#c00000"),
            "skip": ("", "#888"),
        }[state]
        self.stage_labels[key].configure(text=text, foreground=color)

    # -- log ---------------------------------------------------------------

    def _post(self, fn):
        """Queue fn to run on the UI thread (safe to call from a worker thread)."""
        self._ui_queue.put(fn)

    def log(self, text, tag=None):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text, tag or "")
        # Keep the widget bounded; these stages print a lot.
        if int(self.log_text.index("end-1c").split(".")[0]) > 6000:
            self.log_text.delete("1.0", "2000.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _drain_log(self):
        chunks = []
        try:
            while True:
                chunks.append(self._log_queue.get_nowait())
        except queue.Empty:
            pass
        for text, tag in chunks:
            self.log(text, tag)

        callbacks = []
        try:
            while True:
                callbacks.append(self._ui_queue.get_nowait())
        except queue.Empty:
            pass
        for callback in callbacks:
            try:
                callback()
            except Exception as exc:  # a UI callback must not kill the drain loop
                self.log(f"[gui] callback failed: {exc}\n", "bad")

        self.after(100, self._drain_log)

    def clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def save_log(self):
        default = (f"sleep_pipeline_{self.animal_var.get().strip()}_"
                   f"{self.date_var.get().strip()}_"
                   f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        path = filedialog.asksaveasfilename(title="Save log", defaultextension=".log",
                                            initialfile=default)
        if not path:
            return
        header = (f"# Sleep pipeline log\n"
                  f"# saved {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"# animal={self.animal_var.get()} date={self.date_var.get()} "
                  f"sessions={self.session_var.get()} "
                  f"shanks={self.shanks_var.get()}\n"
                  f"# python={self.interpreter_var.get()}\n"
                  f"# velocity_source={self.velocity_source_var.get()}\n"
                  f"# dio_channel={self.dio_channel_var.get()} "
                  f"channels={self.channels_var.get() or 'all'} "
                  f"max_channels={self.max_channels_var.get() or 'all'} "
                  f"suffix={self.output_suffix_var.get()!r}\n\n")
        try:
            Path(path).write_text(header + self.log_text.get("1.0", "end"), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save log", str(exc))
            return
        self.status_var.set(f"Log saved to {path}")

    def open_output_folder(self):
        if not self.info or not self.info.get("registered"):
            messagebox.showinfo("Output folder", "Run preflight on a registered day first.")
            return
        folder = find_any(self.info["low_freq"]) or output_target(self.info["low_freq"])
        try:
            if os.name == "nt":
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(folder)])
            else:
                subprocess.run(["xdg-open", str(folder)])
        except OSError as exc:
            messagebox.showerror("Output folder", str(exc))

    def on_close(self):
        if self._running and not messagebox.askokcancel(
                "Pipeline running", "A stage is still running. Stop it and quit?"):
            return
        if self._running:
            self.stop_pipeline()
        self._save_settings()
        self.destroy()


def main():
    try:  # crisp text on high-DPI Windows displays; harmless if it fails
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    PipelineGUI().mainloop()


if __name__ == "__main__":
    main()
