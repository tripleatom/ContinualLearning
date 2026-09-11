"""Standalone multi-day sleep-state scoring and channel-review panel.

This intentionally complements, rather than modifies, sleep_pipeline_gui.py.
It uses the same day registry and temporary config selection, runs the two
state-scoring scripts for one day at a time, and records the reviewer's chosen
channel for each session/shank in state_scoring/*_final_channel.json.

Run:  python sleep_state_scoring_gui.py
"""
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

SLEEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = SLEEP_DIR.parent
for folder in (str(REPO_ROOT), str(SLEEP_DIR)):
    if folder not in sys.path:
        sys.path.insert(0, folder)

from sleep_day_registry import (entry_animal, load_registry, registered_animals,
                                registered_days)
from sleep_pipeline_gui import (default_interpreter, find_any, find_interpreters,
                                read_config_values, write_config_values)

SESSIONS = ("pre", "post")
HISTOGRAM_SCRIPT = "plot_pc1_theta_velocity_histograms.py"
SCORER_SCRIPT = "score_states_from_thresholds.py"


def day_entry(animal, date):
    registry = load_registry()
    for key, entry in registry.items():
        if str(key).endswith(str(date)) and entry_animal(entry) == animal:
            return entry
    return None


def day_info(animal, date):
    entry = day_entry(animal, date)
    if entry is None:
        return None
    rec_folder = Path(entry["rec_folder"])
    sessions = [name for name in SESSIONS
                if any(entry.get(name, {}).get(k) is not None
                       for k in ("start_sample", "end_sample"))]
    return {"entry": entry, "rec_folder": rec_folder,
            "session_name": rec_folder.stem.split(".")[0],
            "low_freq": rec_folder / "low_freq", "sessions": sessions}


def score_summary_covers(path, sessions, shank):
    """True only when the existing summary contains every requested session."""
    if path is None:
        return False
    try:
        with open(path, newline="", encoding="utf-8") as file:
            present = {row["session"] for row in csv.DictReader(file)
                       if int(row["shank"]) == shank}
    except (OSError, KeyError, ValueError):
        return False
    return set(sessions).issubset(present)


class SleepStateScoringGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sleep state scoring")
        self.geometry("1000x760")
        self.minsize(820, 620)
        cfg = read_config_values()
        candidates = find_interpreters()
        self.python_var = tk.StringVar(value=default_interpreter(candidates))
        self.animal_var = tk.StringVar(value=cfg.get("ACTIVE_ANIMAL", ""))
        self.date_var = tk.StringVar(value=cfg.get("ACTIVE_DATE", ""))
        self.session_var = tk.StringVar(value=cfg.get("SESSION_FILTER") or "both")
        self.shank_var = tk.StringVar(value="0")
        self.plot_max_points_var = tk.StringVar(value="0")
        self.make_histograms_var = tk.BooleanVar(value=True)
        self.score_states_var = tk.BooleanVar(value=True)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready.")
        self._running = False
        self._stop = False
        self._queue_index = 0
        self._events = queue.Queue()
        self._build(candidates)
        self.after(100, self._drain_events)

    def _build(self, candidates):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        day = ttk.LabelFrame(self, text="Recording day", padding=8)
        day.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        ttk.Label(day, text="Animal:").grid(row=0, column=0, sticky="w")
        self.animal_box = ttk.Combobox(day, textvariable=self.animal_var,
                                       values=registered_animals(), width=12)
        self.animal_box.grid(row=0, column=1, padx=(4, 14))
        self.animal_box.bind("<<ComboboxSelected>>", self._animal_changed)
        ttk.Label(day, text="Date:").grid(row=0, column=2, sticky="w")
        self.date_box = ttk.Combobox(day, textvariable=self.date_var, width=12)
        self.date_box.grid(row=0, column=3, padx=(4, 14))
        ttk.Label(day, text="Sessions:").grid(row=0, column=4, sticky="w")
        ttk.Combobox(day, textvariable=self.session_var, values=("both", "pre", "post"),
                     state="readonly", width=8).grid(row=0, column=5, padx=(4, 14))
        ttk.Label(day, text="Shank:").grid(row=0, column=6, sticky="w")
        ttk.Entry(day, textvariable=self.shank_var, width=5).grid(row=0, column=7, padx=(4, 0))
        ttk.Label(day, text="Figure points:").grid(row=0, column=8, sticky="w", padx=(14, 0))
        ttk.Entry(day, textvariable=self.plot_max_points_var, width=7).grid(row=0, column=9, padx=(4, 0))
        ttk.Label(day, text="Python:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(day, textvariable=self.python_var, values=candidates, width=70).grid(
            row=1, column=1, columnspan=7, sticky="ew", padx=(4, 0), pady=(8, 0))
        day.columnconfigure(7, weight=1)

        actions = ttk.LabelFrame(self, text="Scoring", padding=8)
        actions.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        ttk.Checkbutton(actions, text="1. Build PC1/theta/velocity thresholds",
                        variable=self.make_histograms_var).pack(side="left")
        ttk.Checkbutton(actions, text="2. Score every channel", variable=self.score_states_var).pack(
            side="left", padx=(16, 0))
        ttk.Checkbutton(actions, text="overwrite existing outputs",
                        variable=self.overwrite_var).pack(side="left", padx=(16, 0))
        ttk.Button(actions, text="Run selected day", command=self.run_current).pack(
            side="right")
        ttk.Button(actions, text="Review channels / choose final", command=self.review).pack(
            side="right", padx=(0, 8))

        queued = ttk.LabelFrame(self, text="Queue — score multiple days", padding=8)
        queued.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        queued.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(queued, columns=("animal", "date", "status"),
                                 show="headings", height=4, selectmode="extended")
        for name, width in (("animal", 120), ("date", 120), ("status", 400)):
            self.tree.heading(name, text=name.capitalize())
            self.tree.column(name, width=width, stretch=(name == "status"))
        self.tree.grid(row=0, column=0, sticky="ew")
        row = ttk.Frame(queued)
        row.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(row, text="Add current day", command=self.add_current).pack(side="left")
        ttk.Button(row, text="Add all days for animal", command=self.add_animal_days).pack(
            side="left", padx=6)
        ttk.Button(row, text="Remove selected", command=lambda: self.tree.delete(*self.tree.selection())).pack(
            side="left")
        ttk.Button(row, text="Run queue", command=self.run_queue).pack(side="left", padx=(18, 0))
        ttk.Button(row, text="Stop", command=lambda: setattr(self, "_stop", True)).pack(side="left", padx=6)

        log_frame = ttk.LabelFrame(self, text="Log", padding=8)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=(4, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, wrap="none", font=("Consolas", 9),
                           background="#1e1e1e", foreground="#dddddd")
        self.log.grid(row=0, column=0, sticky="nsew")
        ttk.Label(self, textvariable=self.status_var).grid(row=4, column=0, sticky="e", padx=12)
        self._animal_changed()

    def _animal_changed(self, *_):
        animal = self.animal_var.get().strip()
        dates = [date for day_animal, date, _ in registered_days() if day_animal == animal]
        self.date_box["values"] = dates
        if dates and self.date_var.get() not in dates:
            self.date_var.set(dates[0])

    def _selected_sessions(self, info):
        return [session for session in info["sessions"]
                if self.session_var.get() in ("both", session)]

    def _append(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def _run_day(self, animal, date):
        info = day_info(animal, date)
        if info is None:
            return False, "not registered"
        try:
            shank = int(self.shank_var.get().strip())
            plot_max_points = int(self.plot_max_points_var.get().strip() or 0)
        except ValueError:
            return False, "shank and figure points must be integers"
        if plot_max_points < 0:
            return False, "figure points must be zero or positive"
        sessions = self._selected_sessions(info)
        if not sessions:
            return False, "no active selected sleep sessions"
        python = self.python_var.get().strip()
        if not Path(python).is_file():
            return False, "Python interpreter not found"
        # Exactly like the pipeline GUI, scripts receive their day through the
        # shared config so existing script entry points remain unchanged.
        write_config_values({"ACTIVE_ANIMAL": animal, "ACTIVE_DATE": date,
                             "SESSION_FILTER": self.session_var.get(), "shanks": [shank]})
        commands = []
        histograms_exist = all(find_any(
            info["low_freq"] / "histograms" /
            f"{info['session_name']}_{session}_sh{shank}_bimodality_summary.csv")
            is not None for session in sessions)
        score_summary = find_any(info["low_freq"] / "state_scoring" /
                                 f"{info['session_name']}_sh{shank}_state_scoring_summary.csv")
        scores_exist = score_summary_covers(score_summary, sessions, shank)
        if self.make_histograms_var.get() and (self.overwrite_var.get() or not histograms_exist):
            commands.append([python, HISTOGRAM_SCRIPT, "--shank", str(shank)])
        elif self.make_histograms_var.get():
            self._events.put(("log", "Threshold histograms already exist; skipped.\n"))
        if self.score_states_var.get() and (self.overwrite_var.get() or not scores_exist):
            commands.append([python, SCORER_SCRIPT, "--shank", str(shank), "--all-channels",
                             "--plot-max-points", str(plot_max_points)])
        elif self.score_states_var.get():
            self._events.put(("log", "State-scoring outputs already exist; skipped.\n"))
        if not commands:
            return True, "already exists"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(SLEEP_DIR), env.get("PYTHONPATH", "")])
        env["MPLBACKEND"] = "Agg"
        for command in commands:
            self._events.put(("log", "\n$ " + subprocess.list2cmdline(command) + "\n"))
            proc = subprocess.Popen(command, cwd=SLEEP_DIR, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            for line in proc.stdout:
                self._events.put(("log", line))
            if proc.wait() != 0:
                return False, f"{Path(command[1]).name} failed"
            if self._stop:
                return False, "stopped"
        return True, "done"

    def run_current(self):
        if self._running:
            return
        animal, date = self.animal_var.get().strip(), self.date_var.get().strip()
        if not animal or not date:
            messagebox.showerror("Scoring", "Choose an animal and date first.")
            return
        self._start([(animal, date)])

    def add_current(self):
        values = (self.animal_var.get().strip(), self.date_var.get().strip())
        if all(values) and values not in [self.tree.item(i, "values")[:2] for i in self.tree.get_children()]:
            self.tree.insert("", "end", values=(*values, "pending"))

    def add_animal_days(self):
        existing = {self.tree.item(i, "values")[:2] for i in self.tree.get_children()}
        animal = self.animal_var.get().strip()
        for day_animal, date, _ in registered_days():
            if day_animal == animal and (animal, date) not in existing:
                self.tree.insert("", "end", values=(animal, date, "pending"))

    def run_queue(self):
        days = [tuple(self.tree.item(i, "values")[:2]) for i in self.tree.get_children()]
        if not days:
            messagebox.showerror("Queue", "Add at least one day first.")
            return
        self._start(days)

    def _start(self, days):
        self._running, self._stop = True, False
        self.status_var.set(f"Scoring {len(days)} day(s)...")
        def worker():
            for index, (animal, date) in enumerate(days, 1):
                if self._stop:
                    break
                self._events.put(("status", f"{animal} {date}: running ({index}/{len(days)})"))
                ok, message = self._run_day(animal, date)
                self._events.put(("done", (animal, date, "done" if ok else message)))
            self._events.put(("finished", None))
        threading.Thread(target=worker, daemon=True).start()

    def _drain_events(self):
        try:
            while True:
                kind, value = self._events.get_nowait()
                if kind == "log": self._append(value)
                elif kind == "status": self.status_var.set(value)
                elif kind == "done":
                    animal, date, status = value
                    for item in self.tree.get_children():
                        if tuple(self.tree.item(item, "values")[:2]) == (animal, date):
                            self.tree.item(item, values=(animal, date, status))
                elif kind == "finished":
                    self._running = False
                    self.status_var.set("Stopped." if self._stop else "Scoring queue complete.")
        except queue.Empty:
            pass
        self.after(100, self._drain_events)

    def review(self):
        info = day_info(self.animal_var.get().strip(), self.date_var.get().strip())
        if info is None:
            messagebox.showerror("Review", "Choose a registered day first.")
            return
        try: shank = int(self.shank_var.get().strip())
        except ValueError:
            messagebox.showerror("Review", "Shank must be an integer."); return
        out_dir = find_any(info["low_freq"] / "state_scoring")
        if out_dir is None:
            messagebox.showinfo("Review", "No scoring outputs found. Run state scoring first."); return
        sessions = self._selected_sessions(info)
        if not sessions: return
        win = tk.Toplevel(self); win.title("Review sleep-score channels"); win.geometry("850x430")
        session = tk.StringVar(value=sessions[0]); rows = []
        top = ttk.Frame(win, padding=8); top.pack(fill="x")
        ttk.Label(top, text="Session:").pack(side="left")
        box = ttk.Combobox(top, textvariable=session, values=sessions, state="readonly", width=8)
        box.pack(side="left", padx=6)
        tree = ttk.Treeview(win, columns=("ch", "nrem", "rem", "wake", "quiet", "r", "figure"), show="headings")
        for col, title, width in (("ch", "Channel", 80), ("nrem", "NREM %", 80), ("rem", "REM %", 80),
                                  ("wake", "Wake %", 80), ("quiet", "Quiet %", 80), ("r", "PC1-delta r", 100),
                                  ("figure", "Figure", 300)):
            tree.heading(col, text=title); tree.column(col, width=width, stretch=(col == "figure"))
        tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        def load(*_):
            nonlocal rows
            summary = out_dir / f"{info['session_name']}_sh{shank}_state_scoring_summary.csv"
            try:
                with open(summary, newline="", encoding="utf-8") as f:
                    rows = [r for r in csv.DictReader(f) if r["session"] == session.get() and int(r["shank"]) == shank]
            except OSError: rows = []
            tree.delete(*tree.get_children())
            for r in rows:
                tree.insert("", "end", values=(r["channel"], f"{float(r['NREM_percent']):.1f}", f"{float(r['REM_percent']):.1f}", f"{float(r['WAKE_percent']):.1f}", f"{float(r['QUIET_percent']):.1f}", f"{float(r['pc1_orientation_r']):+.2f}", r["figure"]))
        def current():
            selected = tree.selection()
            if not selected: return None
            ch = int(tree.item(selected[0], "values")[0])
            return next((r for r in rows if int(r["channel"]) == ch), None)
        def open_figure():
            row = current()
            if row is None: return
            path = find_any(out_dir / row["figure"])
            if path is None: return
            if os.name == "nt": os.startfile(str(path))
            else: subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        def save_final():
            row = current()
            if row is None:
                messagebox.showinfo("Review", "Select a channel first.", parent=win); return
            record = {"animal": self.animal_var.get(), "date": self.date_var.get(), "session": session.get(),
                      "shank": shank, "channel": int(row["channel"]), "figure": row["figure"],
                      "selected_at": datetime.now().isoformat(timespec="seconds"), "selection_method": "manual review"}
            target = out_dir / f"{info['session_name']}_{session.get()}_sh{shank}_final_channel.json"
            if target.exists() and not messagebox.askyesno(
                    "Replace final channel?",
                    f"{target.name} already selects a final channel. Replace it?",
                    parent=win):
                return
            target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            messagebox.showinfo("Final channel", f"Channel {record['channel']} saved to\n{target.name}", parent=win)
        buttons = ttk.Frame(win, padding=8); buttons.pack(fill="x")
        ttk.Button(buttons, text="Open selected figure", command=open_figure).pack(side="left")
        ttk.Button(buttons, text="Choose selected channel as final", command=save_final).pack(side="left", padx=8)
        box.bind("<<ComboboxSelected>>", load); tree.bind("<Double-1>", lambda _event: open_figure()); load()


if __name__ == "__main__":
    SleepStateScoringGUI().mainloop()
