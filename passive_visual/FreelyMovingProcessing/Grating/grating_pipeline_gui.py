"""
Tkinter control panel for the Grating processing pipeline.

Drives the sequence normally run by hand for each session:

    1. DIO_grating.py           fix/segment photodiode edges -> <task>_DIO.npz
    2. GratingExport.py         spikes x trials              -> <rec>_grating_data*.pkl
    3-6. batch_grating_analysis.py --stages ...  embedding / LDA / SVM / tuning curves,
                                all reading the pkl written by step 2

The GUI picks the session (animal + date from experiment_log/<animal>.csv), writes
those choices into grating_config.py - the same file the scripts already read, so
running them standalone afterwards stays consistent - then runs each selected stage
as a subprocess and streams its output into the log pane.

Preflight inspects the session on disk first and reports what each stage will find:
DIO .npz files, curated vs raw sorting, and any grating pkl already exported. That
makes it obvious which stages actually need re-running.

Stdlib only (tkinter + subprocess); the heavy imports happen in the subprocesses, so
the panel starts instantly and a crashed stage cannot take the GUI down with it.

Run with:  python grating_pipeline_gui.py
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

import experiment_log_gui as elg  # session CSV reading + experiment_data roots

GRATING_DIR = Path(__file__).resolve().parent
# Repo root = the ancestor holding the passive_visual package; GratingExport imports
# itself by that fully qualified name, so it must be on the subprocess PYTHONPATH.
REPO_ROOT = next(
    (p for p in GRATING_DIR.parents if (p / "passive_visual").is_dir()),
    GRATING_DIR.parents[2],
)
CONFIG_PATH = GRATING_DIR / "grating_config.py"
SETTINGS_PATH = Path.home() / ".grating_pipeline_gui.json"

# Conda envs preferred as the interpreter, best first. The pipeline needs
# spikeinterface + sklearn + seaborn; whichever is picked is remembered in
# SETTINGS_PATH, and Browse... overrides it.
PREFERRED_ENVS = ("ms10", "kilosort", "phy2")

SORTOUT_DRIVES = ("xieluanlabs", "xieluanlabs2")

# Config scalars this GUI owns. Everything else in grating_config.py is left alone.
CONFIG_SCALARS = (
    "ANIMAL_ID",
    "SORTOUT_ANIMAL_ID",
    "EXPERIMENT_DATE",
    "SORTOUT_DRIVE",
    "PASSIVE_START",
    "PASSIVE_END",
    "PASSIVE_WINDOWS",
)


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
# grating_config.py READ / WRITE
# =============================================================================

def read_config_values():
    """Module-level values of CONFIG_SCALARS, parsed without importing the module."""
    values = {}
    try:
        tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return values
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in CONFIG_SCALARS:
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
    if name == "PASSIVE_WINDOWS":
        return value  # already validated source text
    if isinstance(value, str):
        return json.dumps(value)  # double quotes, as grating_config.py uses
    return repr(value)


def _same_value(old_text, new_text):
    """
    True if two assignment texts mean the same thing.

    Compared as parsed literals so that re-writing an unchanged session does not
    churn the file over quoting style ("CnL46" vs 'CnL46') alone.
    """
    try:
        return ast.literal_eval(old_text) == ast.literal_eval(new_text)
    except (ValueError, SyntaxError):
        return old_text == new_text


def write_config_values(values, backup=True):
    """
    Rewrite CONFIG_SCALARS in grating_config.py in place.

    Only the value of each assignment is replaced, so comments, layout and every
    other setting in the file (SLEEP_BLOCKS, the sortout root table, ...) survive.
    Values are written as Python literals except PASSIVE_WINDOWS, which is passed
    through as already-validated source text.

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
# SESSION PREFLIGHT
# =============================================================================

def sortout_roots():
    """Sortout root per drive name, mirroring grating_config's table."""
    if sys.platform == "darwin":
        return {d: Path(f"/Volumes/{d}/xl_cl/sortout") for d in SORTOUT_DRIVES}
    return {
        "xieluanlabs": Path(r"\\10.129.151.108\xieluanlabs\xl_cl\sortout"),
        "xieluanlabs2": Path(r"\\10.129.151.88\xieluanlabs2\xl_cl\sortout"),
    }


def _exists(path):
    try:
        return path.exists()
    except OSError:
        return False


def has_sort_data(folder):
    """True if folder holds a curated_analyzer or any shank raw sorting_analyzer."""
    if not _exists(folder):
        return False
    if _exists(folder / "curated_analyzer"):
        return True
    try:
        for shank in folder.glob("shank*"):
            for results in shank.glob("sorting_results_*"):
                if _exists(results / "sorting_analyzer"):
                    return True
    except OSError:
        pass
    return False


def resolve_session(animal, date):
    """
    Resolve rec folders and task files exactly as grating_utils.load_session_paths
    does: the first experiment_data root where every referenced path exists.
    """
    out = {"ok": False, "error": None, "root": None, "base": None,
           "rec_folders": [], "task_files": [], "row": None}

    rows, _ = elg.read_log(animal)
    row = next((r for r in rows if r["date"] == date), None)
    if row is None:
        out["error"] = (f"No entry for date {date} in experiment_log/{animal}.csv - "
                        f"add it with experiment_log_gui.py first.")
        return out
    out["row"] = row

    tried = []
    for root in elg.experiment_data_roots():
        base = root / animal / date
        ephys = base / row["EphysFolder"]
        rec_folders = [ephys / f.strip() for f in row["PassiveFolder"].split(";") if f.strip()]
        task_files = [base / f.strip() for f in row["TaskFile"].split(";")
                      if f.strip().endswith(".txt")]
        if rec_folders and task_files and all(_exists(p) for p in rec_folders + task_files):
            out.update(ok=True, root=root, base=base,
                       rec_folders=rec_folders, task_files=task_files)
            return out
        tried.append(base)

    out["error"] = ("Session not found under any experiment_data root. Tried:\n  "
                    + "\n  ".join(str(t) for t in tried))
    return out


def preflight(animal, date, sortout_animal, drive):
    """
    Inspect a session and report what each pipeline stage will find on disk.

    Returns a dict with the resolved session, per-task DIO status, the sortout
    folder actually holding sorted data, and any exported grating pkl candidates.
    """
    info = resolve_session(animal, date)
    info["dio"] = []
    info["sortout"] = None
    info["sortout_resolved"] = None
    info["curation"] = None
    info["pkls"] = []

    if not info["ok"]:
        return info

    for task in info["task_files"]:
        npz = task.parent / f"{task.stem}_DIO.npz"
        info["dio"].append({"task": task, "npz": npz, "exists": _exists(npz)})

    root = sortout_roots()[drive]
    configured = root / sortout_animal / f"{sortout_animal}_20{date}"
    info["sortout"] = configured

    # GratingExport falls back to a sortout folder named after the rec timestamp
    # when the short <animal>_<date> form holds no sorted data.
    resolved = configured
    if not has_sort_data(configured) and info["rec_folders"]:
        rec_stem = info["rec_folders"][0].name
        if rec_stem.endswith(".rec"):
            rec_stem = rec_stem[:-4]
        candidate = configured.parent / rec_stem
        if has_sort_data(candidate):
            resolved = candidate
    info["sortout_resolved"] = resolved

    if _exists(resolved / "curated_analyzer"):
        info["curation"] = "curated"
    elif has_sort_data(resolved):
        info["curation"] = "raw (uncurated shank analyzers)"
    else:
        info["curation"] = None

    # Export writes beside the configured folder; a resolved fallback folder may
    # already hold output from an earlier run, so look in both.
    rec_name = info["rec_folders"][0].name.replace(".rec", "")
    seen = set()
    for folder in (configured, resolved):
        out_dir = folder / "passive_embedding_analysis"
        if not _exists(out_dir):
            continue
        candidates = []
        merged = out_dir / f"{rec_name}_grating_data_merged.pkl"
        if _exists(merged):
            candidates.append(merged)
        try:
            candidates += sorted(out_dir.glob(f"{rec_name}_*_grating_data.pkl"))
        except OSError:
            pass
        for pkl in candidates:
            if str(pkl) not in seen:
                seen.add(str(pkl))
                info["pkls"].append(pkl)

    return info


def detect_pkl(animal, date, sortout_animal, drive):
    """Newest exported grating pkl for a session, or "" if none exist yet."""
    try:
        info = preflight(animal, date, sortout_animal, drive)
    except Exception:
        return ""
    if not info["pkls"]:
        return ""
    return str(max(info["pkls"], key=lambda p: p.stat().st_mtime))


# =============================================================================
# STAGES
# =============================================================================

# key, label, whether it is one of batch_grating_analysis's downstream stages
STAGES = [
    ("dio", "1. DIO edges", False),
    ("export", "2. Export pkl", False),
    ("embedding", "3. Embedding", True),
    ("lda", "4. LDA", True),
    ("svm", "5. SVM", True),
    ("tuning", "6. Tuning curves", True),
]
STAGE_LABELS = dict((k, l) for k, l, _ in STAGES)
DOWNSTREAM = [k for k, _, d in STAGES if d]


class PipelineGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Grating pipeline")
        self.geometry("1180x900")
        self.minsize(940, 700)

        self.settings = self._load_settings()
        cfg = read_config_values()

        interpreters = find_interpreters()
        self.interpreter_var = tk.StringVar(
            value=self.settings.get("interpreter") or default_interpreter(interpreters))

        self.animal_var = tk.StringVar(value=cfg.get("ANIMAL_ID", ""))
        self.date_var = tk.StringVar(value=cfg.get("EXPERIMENT_DATE", ""))
        self.sortout_animal_var = tk.StringVar(value=cfg.get("SORTOUT_ANIMAL_ID", ""))
        self.drive_var = tk.StringVar(value=cfg.get("SORTOUT_DRIVE", SORTOUT_DRIVES[-1]))
        self.passive_start_var = tk.StringVar(value=str(cfg.get("PASSIVE_START", 0)))
        self.passive_end_var = tk.StringVar(
            value="" if cfg.get("PASSIVE_END") is None else str(cfg.get("PASSIVE_END")))
        self.passive_windows_var = tk.StringVar(
            value=self._config_source("PASSIVE_WINDOWS") or "None")
        self.write_config_var = tk.BooleanVar(value=True)

        self.stage_vars = {k: tk.BooleanVar(value=self.settings.get("stages", {}).get(k, True))
                           for k, _, _ in STAGES}
        self.t0_var = tk.StringVar(value=str(self.settings.get("t0", 0.05)))
        self.t1_var = tk.StringVar(value=str(self.settings.get("t1", 1.0)))
        self.svm_kernel_var = tk.StringVar(value=self.settings.get("svm_kernel", "rbf"))
        self.svm_c_var = tk.StringVar(value=str(self.settings.get("svm_c", 1.0)))
        self.svm_gamma_var = tk.StringVar(value=self.settings.get("svm_gamma", "scale"))
        self.tuning_summary_var = tk.BooleanVar(
            value=self.settings.get("tuning_summary", True))
        self.continue_var = tk.BooleanVar(value=self.settings.get("continue_on_error", True))
        self.dio_duration_var = tk.StringVar()
        self.dio_pieces_var = tk.StringVar()
        self.pkl_var = tk.StringVar()

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
            "t0": self.t0_var.get(),
            "t1": self.t1_var.get(),
            "svm_kernel": self.svm_kernel_var.get(),
            "svm_c": self.svm_c_var.get(),
            "svm_gamma": self.svm_gamma_var.get(),
            "tuning_summary": self.tuning_summary_var.get(),
            "continue_on_error": self.continue_var.get(),
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _config_source(name):
        """Raw source text of a config assignment (for list/tuple values)."""
        try:
            text = CONFIG_PATH.read_text(encoding="utf-8")
        except OSError:
            return None
        match = _assignment_pattern(name).search(text)
        return match.group(2).strip() if match else None

    # -- construction ------------------------------------------------------

    def _build_ui(self, interpreters):
        # Rows: 0 session, 1 preflight, 2 stages, 3 run bar, 4 log.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(4, weight=2)

        # ── Session ───────────────────────────────────────────────────────
        session = ttk.LabelFrame(self, text="Session", padding=(8, 6))
        session.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        for col in (1, 3, 5):
            session.columnconfigure(col, weight=1)

        ttk.Label(session, text="Animal:").grid(row=0, column=0, sticky="w")
        self.animal_combo = ttk.Combobox(session, textvariable=self.animal_var, width=12,
                                         values=sorted(p.stem for p in elg.LOG_DIR.glob("*.csv")))
        self.animal_combo.grid(row=0, column=1, sticky="w", padx=(4, 12))
        self.animal_combo.bind("<<ComboboxSelected>>", lambda _e: self._animal_changed())

        ttk.Label(session, text="Date:").grid(row=0, column=2, sticky="w")
        self.date_combo = ttk.Combobox(session, textvariable=self.date_var, width=12)
        self.date_combo.grid(row=0, column=3, sticky="w", padx=(4, 12))
        self.date_combo.bind("<<ComboboxSelected>>", lambda _e: self.run_preflight())

        ttk.Label(session, text="Sortout animal:").grid(row=0, column=4, sticky="w")
        ttk.Entry(session, textvariable=self.sortout_animal_var, width=12).grid(
            row=0, column=5, sticky="w", padx=(4, 12))

        ttk.Label(session, text="Sortout drive:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(session, textvariable=self.drive_var, width=12, state="readonly",
                     values=list(SORTOUT_DRIVES)).grid(row=1, column=1, sticky="w",
                                                       padx=(4, 12), pady=(6, 0))

        ttk.Label(session, text="PASSIVE_START:").grid(row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Entry(session, textvariable=self.passive_start_var, width=14).grid(
            row=1, column=3, sticky="w", padx=(4, 12), pady=(6, 0))

        ttk.Label(session, text="PASSIVE_END:").grid(row=1, column=4, sticky="w", pady=(6, 0))
        ttk.Entry(session, textvariable=self.passive_end_var, width=14).grid(
            row=1, column=5, sticky="w", padx=(4, 12), pady=(6, 0))

        ttk.Label(session, text="PASSIVE_WINDOWS:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(session, textvariable=self.passive_windows_var).grid(
            row=2, column=1, columnspan=3, sticky="ew", padx=(4, 12), pady=(6, 0))
        ttk.Checkbutton(session, text="Write grating_config.py before running",
                        variable=self.write_config_var).grid(
            row=2, column=4, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(session, text="Python:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        interp = ttk.Frame(session)
        interp.grid(row=3, column=1, columnspan=5, sticky="ew", padx=(4, 0), pady=(6, 0))
        interp.columnconfigure(0, weight=1)
        ttk.Combobox(interp, textvariable=self.interpreter_var,
                     values=interpreters).grid(row=0, column=0, sticky="ew")
        ttk.Button(interp, text="Browse...", command=self.browse_interpreter).grid(
            row=0, column=1, padx=(6, 0))
        ttk.Button(interp, text="Refresh preflight", command=self.run_preflight).grid(
            row=0, column=2, padx=(6, 0))
        ttk.Button(interp, text="Write config now", command=self.write_config).grid(
            row=0, column=3, padx=(6, 0))

        # ── Preflight ─────────────────────────────────────────────────────
        pre = ttk.LabelFrame(self, text="Preflight - what is on disk for this session",
                             padding=(8, 6))
        pre.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
        pre.columnconfigure(0, weight=1)
        pre.rowconfigure(0, weight=1)
        self.pre_text = tk.Text(pre, height=9, wrap="none", font=("Consolas", 9))
        self.pre_text.grid(row=0, column=0, sticky="nsew")
        pre_scroll = ttk.Scrollbar(pre, orient="vertical", command=self.pre_text.yview)
        pre_scroll.grid(row=0, column=1, sticky="ns")
        self.pre_text.configure(yscrollcommand=pre_scroll.set, state="disabled")
        for tag, color in (("ok", "#1a7f37"), ("warn", "#b8860b"),
                           ("bad", "#c00000"), ("head", "#333333")):
            self.pre_text.tag_configure(tag, foreground=color)
        self.pre_text.tag_configure("head", font=("Consolas", 9, "bold"))

        # ── Stages ────────────────────────────────────────────────────────
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
        ttk.Label(row1, text="Analysis window t0:").pack(side="left")
        ttk.Entry(row1, textvariable=self.t0_var, width=7).pack(side="left", padx=(4, 8))
        ttk.Label(row1, text="t1:").pack(side="left")
        ttk.Entry(row1, textvariable=self.t1_var, width=7).pack(side="left", padx=(4, 16))
        ttk.Label(row1, text="SVM kernel:").pack(side="left")
        ttk.Combobox(row1, textvariable=self.svm_kernel_var, width=8, state="readonly",
                     values=["rbf", "linear", "poly", "sigmoid"]).pack(side="left", padx=(4, 8))
        ttk.Label(row1, text="C:").pack(side="left")
        ttk.Entry(row1, textvariable=self.svm_c_var, width=7).pack(side="left", padx=(4, 8))
        ttk.Label(row1, text="gamma:").pack(side="left")
        ttk.Entry(row1, textvariable=self.svm_gamma_var, width=8).pack(side="left", padx=(4, 16))
        ttk.Checkbutton(row1, text="tuning summary figures",
                        variable=self.tuning_summary_var).pack(side="left")
        ttk.Checkbutton(row1, text="continue on error",
                        variable=self.continue_var).pack(side="left", padx=(12, 0))

        row2 = ttk.Frame(stages)
        row2.pack(fill="x", pady=(8, 0))
        ttk.Label(row2, text="DIO answers - trial duration (s):").pack(side="left")
        ttk.Entry(row2, textvariable=self.dio_duration_var, width=8).pack(side="left", padx=(4, 8))
        ttk.Label(row2, text="task pieces:").pack(side="left")
        ttk.Entry(row2, textvariable=self.dio_pieces_var, width=6).pack(side="left", padx=(4, 8))
        ttk.Label(row2, text="(blank = accept the script's default; DIO opens plot "
                             "windows you must close to continue)",
                  foreground="#666").pack(side="left")

        row3 = ttk.Frame(stages)
        row3.pack(fill="x", pady=(8, 0))
        ttk.Label(row3, text="Data pkl for stages 3-6:").pack(side="left")
        self.pkl_combo = ttk.Combobox(row3, textvariable=self.pkl_var)
        self.pkl_combo.pack(side="left", fill="x", expand=True, padx=(4, 6))
        ttk.Button(row3, text="Browse...", command=self.browse_pkl).pack(side="left")

        # ── Run bar + log ─────────────────────────────────────────────────
        bar = ttk.Frame(self, padding=(10, 4))
        bar.grid(row=3, column=0, sticky="ew")
        self.run_button = ttk.Button(bar, text="Run pipeline", command=self.start_pipeline)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(bar, text="Stop", command=self.stop_pipeline,
                                      state="disabled")
        self.stop_button.pack(side="left", padx=6)
        ttk.Button(bar, text="Save log...", command=self.save_log).pack(side="left")
        ttk.Button(bar, text="Clear log", command=self.clear_log).pack(side="left", padx=6)
        ttk.Button(bar, text="Open output folder",
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

    def browse_pkl(self):
        path = filedialog.askopenfilename(title="Select grating data pkl",
                                          filetypes=[("Pickle", "*.pkl"), ("All", "*.*")])
        if path:
            self.pkl_var.set(path)

    # -- preflight ---------------------------------------------------------

    def _animal_changed(self):
        animal = self.animal_var.get().strip()
        if animal and not self.sortout_animal_var.get().strip():
            self.sortout_animal_var.set(animal)
        rows, _ = elg.read_log(animal)
        self.date_combo["values"] = [r["date"] for r in rows]
        self.run_preflight()

    def run_preflight(self):
        animal = self.animal_var.get().strip()
        date = self.date_var.get().strip()
        sortout_animal = self.sortout_animal_var.get().strip() or animal
        drive = self.drive_var.get()
        if not animal or not date:
            return

        rows, _ = elg.read_log(animal)
        self.date_combo["values"] = [r["date"] for r in rows]
        self._set_pre_text([("Scanning disk...\n", None)])
        self.status_var.set("Preflight scanning...")

        def work():
            return preflight(animal, date, sortout_animal, drive)

        def done(info):
            if isinstance(info, Exception):
                self._set_pre_text([(f"Preflight failed: {info}\n", "bad")])
                self.status_var.set("Preflight failed.")
                return
            self.info = info
            self._render_preflight(info)
            pkls = [str(p) for p in info["pkls"]]
            self.pkl_combo["values"] = pkls
            if pkls and (self.pkl_var.get() not in pkls):
                self.pkl_var.set(pkls[0])
            elif not pkls:
                self.pkl_var.set("")
            self.status_var.set("Preflight done.")

        def runner():
            result = self._safe(work)
            self._post(lambda: done(result))

        threading.Thread(target=runner, daemon=True).start()

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
        mark = {True: "[ok]   ", False: "[MISS] "}

        if not info["ok"]:
            chunks.append(("Session\n", "head"))
            chunks.append((f"  {info['error']}\n", "bad"))
            self._set_pre_text(chunks)
            return

        chunks.append((f"Session   {info['base']}\n", "head"))
        for rec in info["rec_folders"]:
            chunks.append((f"  rec    {mark[True]}{rec.name}\n", "ok"))
        chunks.append(("\nStage 1 - DIO\n", "head"))
        for entry in info["dio"]:
            tag = "ok" if entry["exists"] else "warn"
            state = "[ok]   " if entry["exists"] else "[todo] "
            chunks.append((f"  {state}{entry['task'].name}\n", tag))
            chunks.append((f"         -> {entry['npz'].name}"
                           f"{'' if entry['exists'] else '  (not yet created)'}\n", tag))

        chunks.append(("\nStage 2 - Export\n", "head"))
        chunks.append((f"  sortout   {info['sortout']}\n", None))
        if info["sortout_resolved"] != info["sortout"]:
            chunks.append((f"  resolved  {info['sortout_resolved']}  "
                           f"(rec-timestamp named folder)\n", "warn"))
        if info["curation"] is None:
            chunks.append(("  [MISS] no curated_analyzer and no shank sorting_analyzer - "
                           "export will fail\n", "bad"))
        else:
            chunks.append((f"  [ok]   sorting: {info['curation']}\n", "ok"))

        chunks.append(("\nStages 3-6 - exported pkl\n", "head"))
        if not info["pkls"]:
            chunks.append(("  [todo] none yet - run stage 2 first\n", "warn"))
        else:
            for pkl in info["pkls"]:
                chunks.append((f"  [ok]   {pkl.name}\n", "ok"))
            if len(info["pkls"]) > 1:
                chunks.append(("  note: more than one pkl matches this session, so "
                               "grating_utils.resolve_data_path\n        cannot "
                               "auto-detect - pick the one to use below.\n", "warn"))
        self._set_pre_text(chunks)

    # -- config ------------------------------------------------------------

    def collect_config(self):
        """GUI fields as config values, raising ValueError on bad input."""
        animal = self.animal_var.get().strip()
        date = self.date_var.get().strip()
        if not animal or not date:
            raise ValueError("Animal and date are required.")

        start_text = self.passive_start_var.get().strip() or "0"
        try:
            passive_start = int(start_text)
        except ValueError:
            raise ValueError(f"PASSIVE_START must be an integer, got {start_text!r}")

        end_text = self.passive_end_var.get().strip()
        if end_text in ("", "None"):
            passive_end = None
        else:
            try:
                passive_end = int(end_text)
            except ValueError:
                raise ValueError(f"PASSIVE_END must be an integer or blank, got {end_text!r}")

        windows_text = self.passive_windows_var.get().strip() or "None"
        try:
            ast.literal_eval(windows_text)
        except (ValueError, SyntaxError):
            raise ValueError(f"PASSIVE_WINDOWS must be a Python literal, got {windows_text!r}")

        return {
            "ANIMAL_ID": animal,
            "SORTOUT_ANIMAL_ID": self.sortout_animal_var.get().strip() or animal,
            "EXPERIMENT_DATE": date,
            "SORTOUT_DRIVE": self.drive_var.get(),
            "PASSIVE_START": passive_start,
            "PASSIVE_END": passive_end,
            "PASSIVE_WINDOWS": windows_text,
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
        """[(stage_key, argv, cwd, stdin_text)] for the selected stages."""
        python = self.interpreter_var.get().strip()
        if not python or not Path(python).exists():
            raise ValueError(f"Python interpreter not found: {python!r}")

        selected = [k for k, _, _ in STAGES if self.stage_vars[k].get()]
        if not selected:
            raise ValueError("No stages selected.")

        downstream = [k for k in selected if k in DOWNSTREAM]
        if downstream and not self.pkl_var.get().strip() and "export" not in selected:
            raise ValueError(
                "Stages 3-6 need a data pkl. Select stage 2 to export one, or pick an "
                "existing pkl in 'Data pkl for stages 3-6'.")

        plan = []
        for key in selected:
            if key == "dio":
                stdin = f"{self.dio_duration_var.get().strip()}\n" \
                        f"{self.dio_pieces_var.get().strip()}\n"
                plan.append((key, [python, "DIO_grating.py"], GRATING_DIR, stdin))
            elif key == "export":
                plan.append((key, [python, "GratingExport.py"], GRATING_DIR, None))
            else:
                argv = [python, "batch_grating_analysis.py",
                        "--stages", key,
                        "--t0", self.t0_var.get().strip(),
                        "--t1", self.t1_var.get().strip(),
                        "--svm-kernel", self.svm_kernel_var.get(),
                        "--svm-c", self.svm_c_var.get().strip(),
                        "--svm-gamma", self.svm_gamma_var.get().strip()]
                if not self.tuning_summary_var.get():
                    argv.append("--no-tuning-summary")
                if self.continue_var.get():
                    argv.append("--continue-on-error")
                plan.append((key, argv, GRATING_DIR, None))
        return plan

    def _subprocess_env(self, stage):
        env = os.environ.copy()
        # Both import styles are used across these scripts: flat (DIO_grating) and
        # fully qualified passive_visual.* (GratingExport), so expose both roots.
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT), str(GRATING_DIR), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if stage == "dio":
            # DIO_grating needs a real window for its interactive save-version
            # picker, so make sure no inherited MPLBACKEND forces a headless one.
            env.pop("MPLBACKEND", None)
        else:
            env["MPLBACKEND"] = "Agg"
        return env

    # -- running -----------------------------------------------------------

    def start_pipeline(self):
        if self._running:
            return
        try:
            plan = self.build_plan()
        except ValueError as exc:
            messagebox.showerror("Cannot run", str(exc))
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

        # Snapshot everything the worker needs: Tk variables belong to the UI thread.
        animal = self.animal_var.get().strip()
        opts = {
            "continue_on_error": self.continue_var.get(),
            "pkl": self.pkl_var.get().strip(),
            "session": (animal, self.date_var.get().strip(),
                        self.sortout_animal_var.get().strip() or animal,
                        self.drive_var.get()),
        }
        threading.Thread(target=self._run_plan, args=(plan, opts), daemon=True).start()

    def _run_plan(self, plan, opts):
        started = time.time()
        failures = []
        for index, (key, argv, cwd, stdin_text) in enumerate(plan, start=1):
            if self._stop:
                self._post(lambda: self.log("\nStopped before "
                                            f"{STAGE_LABELS[key]}.\n", "bad"))
                break

            # A pkl produced by an earlier export in this same run is only known now.
            if key in DOWNSTREAM:
                pkl = opts["pkl"] or detect_pkl(*opts["session"])
                if not pkl:
                    self._post(lambda k=key: self.log(
                        f"\n{STAGE_LABELS[k]}: no exported pkl found, skipping.\n", "bad"))
                    self._post(lambda k=key: self._set_stage_status(k, "failed"))
                    failures.append(key)
                    if not opts["continue_on_error"]:
                        break
                    continue
                opts["pkl"] = pkl
                argv = argv + ["--pkl", pkl]

            self._post(lambda k=key, i=index: self._set_stage_status(k, "running"))
            self._post(lambda k=key, i=index, n=len(plan): self.status_var.set(
                f"Stage {i}/{n}: {STAGE_LABELS[k]} running..."))
            self._post(lambda a=argv, c=cwd, k=key: self.log(
                f"\n{'=' * 72}\n{STAGE_LABELS[k]}\n{'=' * 72}\n", "banner"))
            self._post(lambda a=argv, c=cwd: self.log(
                f"$ cd {c}\n$ {subprocess.list2cmdline(a)}\n", "cmd"))

            stage_started = time.time()
            code = self._run_process(key, argv, cwd, stdin_text)
            elapsed = time.time() - stage_started

            if code == 0:
                self._post(lambda k=key: self._set_stage_status(k, "done"))
                self._post(lambda k=key, e=elapsed: self.log(
                    f"\n{STAGE_LABELS[k]} finished in {e:.1f}s.\n", "ok"))
                if key == "export":
                    exported = detect_pkl(*opts["session"])
                    if exported:
                        opts["pkl"] = exported
                        self._post(lambda p=exported: self._use_pkl(p))
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

    def _run_process(self, stage, argv, cwd, stdin_text):
        try:
            self._proc = subprocess.Popen(
                argv, cwd=str(cwd), env=self._subprocess_env(stage),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace", bufsize=1,
            )
        except OSError as exc:
            self._post(lambda: self.log(f"Could not start process: {exc}\n", "bad"))
            return -1

        try:
            if stdin_text:
                self._proc.stdin.write(stdin_text)
            self._proc.stdin.close()
        except OSError:
            pass

        for line in self._proc.stdout:
            self._log_queue.put((line, None))
        self._proc.wait()
        code = self._proc.returncode
        self._proc = None
        return code

    def _use_pkl(self, pkl):
        """Adopt a freshly exported pkl for the remaining stages (UI thread)."""
        self.pkl_var.set(pkl)
        values = list(self.pkl_combo["values"])
        if pkl not in values:
            self.pkl_combo["values"] = values + [pkl]
        self.log(f"Using exported pkl for later stages:\n    {pkl}\n", "ok")

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
        default = (f"grating_pipeline_{self.animal_var.get().strip()}_"
                   f"{self.date_var.get().strip()}_"
                   f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        path = filedialog.asksaveasfilename(title="Save log", defaultextension=".log",
                                            initialfile=default)
        if not path:
            return
        header = (f"# Grating pipeline log\n"
                  f"# saved {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"# animal={self.animal_var.get()} date={self.date_var.get()} "
                  f"sortout_animal={self.sortout_animal_var.get()} "
                  f"drive={self.drive_var.get()}\n"
                  f"# python={self.interpreter_var.get()}\n"
                  f"# window=({self.t0_var.get()}, {self.t1_var.get()}) "
                  f"svm=({self.svm_kernel_var.get()}, C={self.svm_c_var.get()}, "
                  f"gamma={self.svm_gamma_var.get()})\n\n")
        try:
            Path(path).write_text(header + self.log_text.get("1.0", "end"), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save log", str(exc))
            return
        self.status_var.set(f"Log saved to {path}")

    def open_output_folder(self):
        if not self.info or not self.info.get("sortout_resolved"):
            messagebox.showinfo("Output folder", "Run preflight first.")
            return
        folder = self.info["sortout_resolved"] / "passive_embedding_analysis"
        if not _exists(folder):
            folder = self.info["sortout_resolved"]
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
    PipelineGUI().mainloop()


if __name__ == "__main__":
    main()
