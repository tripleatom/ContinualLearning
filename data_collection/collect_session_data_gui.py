"""
Collect videos and behavior logs into session folders  --  GUI
================================================================

A window around collect_session_data.py: pick the destination animal folders (as
many as you like, chosen fresh each time), preview exactly what would move, then
move it. All naming rules, the ``video`` subfolder creation and the safe
copy-verify-delete transfer live in collect_session_data.py -- this is only the
front end, so the GUI and the command line can never drift apart.

Four independent sources feed each session folder, shown as three columns:
    Video    -- camera captures; land in ``<session>/video/``.
    Passive  -- the drifting-grating disk sessions: the Unity engine log
                folder; lands straight in ``<session>/``. No task or reward
                is involved.
    Task     -- two SphereTask behavior-summary folders, one per task
                variant (grating task, contrast task); both land straight in
                ``<session>/``.
Any folder can be left blank to skip that source.

Typical use:
    1. "Add folder..." and pick  \\\\10.129.151.88\\...\\experiment_data\\CnL46
       (or "Add all animals under..." and pick experiment_data itself)
    2. "Preview"  -- lists what would move, grouped by recording, nothing touched
    3. "Move files" -- does it, with a progress bar, and can be stopped part-way

Preview shows a [VIDEO TS PROC DLC] column per recording with dashes for the
companions that were never produced. Incomplete recordings move regardless --
the dashes are there to tell you what is going across, not to hold anything back.

Requires only the Python standard library (tkinter).  Run:

    python collect_session_data_gui.py
"""

from __future__ import annotations

import json
import queue
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

if __package__ in (None, ""):
    # Running as a plain script (e.g. `python collect_session_data_gui.py`) rather
    # than `python -m data_collection.collect_session_data_gui` -- put the project
    # root on sys.path so the package-style import below can still find data_collection.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_collection.collect_session_data import (COMPANION_KINDS, DEFAULT_CONTRAST_TASK_ROOT,
                            DEFAULT_GRATING_TASK_ROOT, DEFAULT_UNITY_ROOT,
                            DEFAULT_VIDEO_ROOT, VIDEO_SUBDIR,
                            AnimalPlan, plan_animal, preview_lines, run_plan,
                            session_date)

SETTINGS_FILE = Path(__file__).with_name("collect_session_data_gui_settings.json")

# Source fields, grouped into the three GUI columns. Each entry is
# (key, entry label, default root); `key` doubles as the plan_animal() kwarg
# name (<key>_root) and the settings-file key (<key>_root).
SOURCE_COLUMNS: list[tuple[str, list[tuple[str, str, Path]]]] = [
    ("Video", [
        ("video", "Video folder:", DEFAULT_VIDEO_ROOT),
    ]),
    ("Passive (drifting grating)", [
        ("unity", "Unity log folder:", DEFAULT_UNITY_ROOT),
    ]),
    ("Task", [
        ("grating_task", "Grating task folder:", DEFAULT_GRATING_TASK_ROOT),
        ("contrast_task", "Contrast task folder:", DEFAULT_CONTRAST_TASK_ROOT),
    ]),
]


# ---------------------------------------------------------------------------
# Worker: run the plans in the background so the window stays responsive
# ---------------------------------------------------------------------------

class MoveWorker(threading.Thread):
    """Executes a list of AnimalPlans, reporting back through a queue.

    Queue messages are tuples:
        ("log",   str)    -- a line for the log pane
        ("step",  None)   -- one more file attempted (advances the progress bar)
        ("done",  dict)   -- finished or stopped; totals for the status line
    """

    def __init__(self, plans: list[AnimalPlan], dry_run: bool, overwrite: bool,
                 out_q: "queue.Queue"):
        super().__init__(daemon=True)
        self.plans = plans
        self.dry_run = dry_run
        self.overwrite = overwrite
        self.q = out_q
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        total_files = 0
        total_bytes = 0
        try:
            for plan in self.plans:
                if self._stop_evt.is_set():
                    break
                self.q.put(("log", f"{plan.dest_root}  (animal {plan.animal})"))
                n, nbytes = run_plan(
                    plan,
                    dry_run=self.dry_run,
                    overwrite=self.overwrite,
                    log=lambda m: self.q.put(("log", m)),
                    on_file=lambda: self.q.put(("step", None)),
                    should_stop=self._stop_evt.is_set,
                )
                total_files += n
                total_bytes += nbytes
        except Exception as exc:  # noqa: BLE001 - surface any failure to the log
            self.q.put(("log", f"ERROR: {exc}"))
        finally:
            self.q.put(("done", {
                "files": total_files,
                "bytes": total_bytes,
                "stopped": self._stop_evt.is_set(),
                "dry_run": self.dry_run,
            }))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Collect videos and behavior logs into session folders")
        self.geometry("1100x740")

        self.q: queue.Queue = queue.Queue()
        self.worker: MoveWorker | None = None
        self.plans: list[AnimalPlan] = []
        self.last_browse = ""
        self.source_vars: dict[str, tk.StringVar] = {}
        self.source_labels: dict[str, str] = {}

        self._build_ui()
        self._load_settings()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_queue)

    # -- UI construction ----------------------------------------------------

    def _build_ui(self):
        pad = dict(padx=6, pady=4)
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        # Destination list
        ttk.Label(frm, text="Destination animal folders:").grid(
            row=0, column=0, sticky="nw", **pad)
        list_frame = ttk.Frame(frm)
        list_frame.grid(row=0, column=1, sticky="nsew", **pad)
        self.dest_list = tk.Listbox(list_frame, height=5, selectmode="extended",
                                    font=("Consolas", 9))
        ys = ttk.Scrollbar(list_frame, orient="vertical", command=self.dest_list.yview)
        self.dest_list.configure(yscrollcommand=ys.set)
        self.dest_list.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        btns = ttk.Frame(frm)
        btns.grid(row=0, column=2, sticky="n", **pad)
        ttk.Button(btns, text="Add folder...", command=self._add_folder).pack(
            fill="x", pady=1)
        ttk.Button(btns, text="Add all animals under...",
                   command=self._add_all_animals).pack(fill="x", pady=1)
        ttk.Button(btns, text="Remove selected", command=self._remove_selected).pack(
            fill="x", pady=1)
        ttk.Button(btns, text="Clear", command=self._clear_list).pack(fill="x", pady=1)

        # Sources: three columns -- Video | Passive | Task
        src = ttk.Frame(frm)
        src.grid(row=1, column=0, columnspan=3, sticky="ew", **pad)
        for col_i, (title, fields) in enumerate(SOURCE_COLUMNS):
            src.columnconfigure(col_i, weight=1)
            box = ttk.LabelFrame(src, text=title, padding=6)
            box.grid(row=0, column=col_i, sticky="new", padx=4)
            for key, label, default in fields:
                self.source_labels[key] = label
                var = tk.StringVar(value=str(default))
                self.source_vars[key] = var
                ttk.Label(box, text=label).pack(anchor="w")
                field_row = ttk.Frame(box)
                field_row.pack(fill="x", pady=(0, 6))
                ttk.Entry(field_row, textvariable=var).pack(side="left", fill="x", expand=True)
                ttk.Button(field_row, text="...", width=3,
                          command=lambda k=key: self._browse_source(k)).pack(
                    side="left", padx=(3, 0))

        # Options
        opt = ttk.Frame(frm)
        opt.grid(row=2, column=0, columnspan=3, sticky="w", **pad)
        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="Overwrite destination files that differ in size",
                        variable=self.overwrite_var).pack(side="left")
        ttk.Label(opt, text=f"   Video files move into  <session>\\{VIDEO_SUBDIR}\\ ;  "
                            f"passive/task files move into  <session>\\  itself. "
                            f"Leave any folder blank to skip that source.",
                  foreground="#555").pack(side="left")

        # Actions
        act = ttk.Frame(frm)
        act.grid(row=3, column=0, columnspan=3, sticky="w", **pad)
        self.preview_btn = ttk.Button(act, text="Preview", command=self._preview)
        self.preview_btn.pack(side="left")
        self.move_btn = ttk.Button(act, text="Move files", command=self._move,
                                   state="disabled")
        self.move_btn.pack(side="left", padx=6)
        self.stop_btn = ttk.Button(act, text="Stop", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left")
        self.status_var = tk.StringVar(
            value="Add one or more animal folders, then click Preview.")
        ttk.Label(act, textvariable=self.status_var, foreground="#555").pack(
            side="left", padx=12)

        self.progress = ttk.Progressbar(frm, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew", **pad)

        # Log pane
        log_frame = ttk.Frame(frm)
        log_frame.grid(row=5, column=0, columnspan=3, sticky="nsew", **pad)
        self.log = tk.Text(log_frame, wrap="none", font=("Consolas", 9), height=20)
        ys2 = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        xs = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log.xview)
        self.log.configure(yscrollcommand=ys2.set, xscrollcommand=xs.set, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        ys2.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(5, weight=1)

    # -- Destination list helpers -------------------------------------------

    def _dest_roots(self) -> list[Path]:
        return [Path(self.dest_list.get(i)) for i in range(self.dest_list.size())]

    def _add_dest(self, path: Path):
        if str(path) not in {str(p) for p in self._dest_roots()}:
            self.dest_list.insert("end", str(path))

    def _add_folder(self):
        d = filedialog.askdirectory(title="Select an animal folder (e.g. ...\\CnL46)",
                                    initialdir=self.last_browse or None)
        if d:
            path = Path(d)
            self.last_browse = str(path.parent)
            self._add_dest(path)
            self._invalidate()

    def _add_all_animals(self):
        d = filedialog.askdirectory(
            title="Select the parent folder holding the animals (e.g. experiment_data)",
            initialdir=self.last_browse or None)
        if not d:
            return
        parent = Path(d)
        self.last_browse = str(parent)
        added = 0
        for child in sorted(p for p in parent.iterdir() if p.is_dir()):
            # Only folders that actually hold YYMMDD sessions.
            try:
                if any(session_date(s) for s in child.iterdir() if s.is_dir()):
                    self._add_dest(child)
                    added += 1
            except OSError:
                continue
        if added:
            self._invalidate()
        else:
            messagebox.showinfo("Nothing added",
                                f"No animal folders with YYMMDD session subfolders "
                                f"found under:\n{parent}")

    def _remove_selected(self):
        for i in reversed(self.dest_list.curselection()):
            self.dest_list.delete(i)
        self._invalidate()

    def _clear_list(self):
        self.dest_list.delete(0, "end")
        self._invalidate()

    def _invalidate(self):
        """Destinations changed, so any previous preview no longer applies."""
        self.plans = []
        self.move_btn.configure(state="disabled")
        self.status_var.set("Destinations changed - click Preview.")

    def _browse_source(self, key: str):
        var = self.source_vars[key]
        d = filedialog.askdirectory(
            title=f"Select the {self.source_labels[key].rstrip(':')} folder",
            initialdir=var.get() or None)
        if d:
            var.set(d)
            self._invalidate()

    # -- Log ----------------------------------------------------------------

    def _log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    # -- Actions ------------------------------------------------------------

    def _source_root(self, key: str) -> Path | None:
        """Blank field -> source disabled (None); otherwise the typed path."""
        text = self.source_vars[key].get().strip().strip('"')
        return Path(text) if text else None

    def _all_roots(self) -> dict[str, Path | None]:
        return {key: self._source_root(key) for key in self.source_vars}

    def _build_plans(self) -> bool:
        """Scan every destination. Returns True if anything is there to move."""
        dest_roots = self._dest_roots()
        if not dest_roots:
            messagebox.showerror("No destinations",
                                 "Add at least one animal folder first.")
            return False
        roots = self._all_roots()
        if not any(roots.values()):
            messagebox.showerror("No sources", "Set at least one source folder.")
            return False
        for key, root in roots.items():
            if root is not None and not root.is_dir():
                messagebox.showerror(f"{self.source_labels[key].rstrip(':')} not found",
                                     f"This folder does not exist:\n{root}")
                return False

        self.plans = [plan_animal(root, roots["video"], roots["unity"], roots["grating_task"],
                                  roots["contrast_task"])
                     for root in dest_roots]
        return any(p.n_files for p in self.plans)

    def _preview(self):
        self._clear_log()
        self.status_var.set("Scanning...")
        self.update_idletasks()
        has_work = self._build_plans()
        if not self.plans:
            self.status_var.set("Nothing to scan.")
            return

        for key, root in self._all_roots().items():
            if root:
                self._log(f"{self.source_labels[key].rstrip(':')} source: {root}")
        self._log(f"Columns after each recording are [{' '.join(COMPANION_KINDS)}]; "
                  f"a dashed entry means that companion was never made.")
        self._log("")
        for plan in self.plans:
            for line in preview_lines(plan):
                self._log(line)
            self._log("")

        n_files = sum(p.n_files for p in self.plans)
        n_bytes = sum(p.n_bytes for p in self.plans)
        n_incomplete = sum(p.n_incomplete for p in self.plans)
        self._log("-" * 60)
        self._log(f"{n_files} file(s), {n_bytes / 1e9:.2f} GB would move.")
        if n_incomplete:
            self._log(f"{n_incomplete} recording(s) are missing companion files "
                      f"and are included anyway.")
        if has_work:
            self.status_var.set(f"{n_files} file(s), {n_bytes / 1e9:.2f} GB ready. "
                                f"Click 'Move files'.")
            self.move_btn.configure(state="normal")
        else:
            self.status_var.set("Nothing to move.")
            self.move_btn.configure(state="disabled")

    def _move(self):
        if not self.plans and not self._build_plans():
            messagebox.showinfo("Nothing to move", "No matching files were found.")
            return

        n_files = sum(p.n_files for p in self.plans)
        n_bytes = sum(p.n_bytes for p in self.plans)
        sources = [str(r) for r in self._all_roots().values() if r]
        if not messagebox.askyesno(
                "Confirm move",
                f"Move {n_files} file(s) ({n_bytes / 1e9:.2f} GB) into the session "
                f"folders?\n\nThey will be DELETED from\n"
                f"{chr(10).join(sources)}\nonce each copy is verified."):
            return

        self._clear_log()
        self.progress.configure(maximum=max(n_files, 1), value=0)
        self.preview_btn.configure(state="disabled")
        self.move_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Moving...")
        self._log("=" * 60)
        self._log(f"Moving {n_files} file(s), {n_bytes / 1e9:.2f} GB")
        self._log("=" * 60)

        self.worker = MoveWorker(self.plans, dry_run=False,
                                 overwrite=self.overwrite_var.get(), out_q=self.q)
        self.worker.start()

    def _stop(self):
        if self.worker:
            self.worker.stop()
            self.status_var.set("Stopping after the current file...")
            self.stop_btn.configure(state="disabled")

    # -- Queue pump ---------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "step":
                    self.progress.step(1)
                elif kind == "done":
                    self._on_done(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _on_done(self, summary: dict):
        self.worker = None
        self.plans = []
        self.preview_btn.configure(state="normal")
        self.move_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")

        verb = "Stopped after" if summary["stopped"] else "Moved"
        msg = f"{verb} {summary['files']} file(s), {summary['bytes'] / 1e9:.2f} GB."
        self._log("")
        self._log(msg)
        self.status_var.set(msg + "  Click Preview to rescan.")
        messagebox.showinfo("Done", msg + "\n\nClick Preview to rescan.")

    # -- Settings persistence -----------------------------------------------

    def _load_settings(self):
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for key, var in self.source_vars.items():
            saved = data.get(f"{key}_root")
            if saved:
                var.set(saved)
        self.overwrite_var.set(bool(data.get("overwrite", False)))
        self.last_browse = data.get("last_browse", "")
        for d in data.get("dest_roots", []):
            self.dest_list.insert("end", d)

    def _save_settings(self):
        data = {f"{key}_root": var.get() for key, var in self.source_vars.items()}
        data.update({
            "overwrite": self.overwrite_var.get(),
            "last_browse": self.last_browse,
            "dest_roots": [str(p) for p in self._dest_roots()],
        })
        try:
            SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Still moving",
                                       "A transfer is running. Stop it and quit?"):
                return
            self.worker.stop()
        self._save_settings()
        self.destroy()


def main() -> int:
    App().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
