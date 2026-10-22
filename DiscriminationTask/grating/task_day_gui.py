"""Tkinter panel to organize each day's discrimination-task data and extract its spikes.

    python task_day_gui.py [--animal CnL46]

Pick an animal: every day in its experiment_data folder is listed with what
was found on disk (behavior JSONs, their paired task .rec, the sortout folder)
and whether the two outputs already exist:

    trial pkl   task_spikes_trial_<sortout>.pkl   per-trial spikes (readDIO_task.py schema)
    task pkl    task_spikes_<sortout>.pkl         whole task .rec (extract_session_spikes.py schema)

Selecting a day shows its config, auto-filled by task_day_discovery.py
(conversion_list.txt gives each task .rec's sample window in the sorter
space; the photodiode channel is picked from the DIO when left on "auto").
Anything can be corrected by hand; "Save config" stores the day in
task_day_configs.json, which extract_task_day.py then uses instead of
re-discovering. "Check DIO" runs only the DIO <-> JSON trial matching, a
few seconds per task. "Extract" runs the full extraction for the selected
day(s) in a subprocess and streams its log here.

Stdlib only; the spikeinterface work happens in the subprocess, using the
interpreter chosen at the top (ms10 by default - the env with the full deps).
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

GRATING_DIR = Path(__file__).resolve().parent
if str(GRATING_DIR) not in sys.path:
    sys.path.insert(0, str(GRATING_DIR))

from task_day_discovery import (REGISTRY_FILE, day_key, day_status, discover_task_day,
                                fill_output_paths, list_animals, list_days, load_registry,
                                save_day_config)

EXTRACT_SCRIPT = GRATING_DIR / "extract_task_day.py"
PREFERRED_ENVS = ("ms10", "kilosort", "phy2")   # same order as sleep/sleep_pipeline_gui.py
DIN_CHOICES = ["auto"] + [str(i) for i in range(1, 9)]
DAY_COLUMNS = (("date", "Date", 70), ("tasks", "Tasks", 50), ("ready", "Ready", 50),
               ("trial", "Trial pkl", 70), ("task", "Task pkl", 70), ("notes", "Notes", 320))
TASK_COLUMNS = (("json", "Behavior JSON", 250), ("trials", "Trials", 50), ("rec", "Task .rec", 230),
                ("window", "Sample window", 170), ("din", "DIN", 45), ("last", "Last run", 260))


def find_interpreters():
    found = []
    exe = "python.exe" if os.name == "nt" else "python"
    for envs in (Path.home() / ".conda" / "envs", Path.home() / "anaconda3" / "envs",
                 Path.home() / "miniconda3" / "envs"):
        try:
            for env in sorted(envs.iterdir()):
                if (env / exe).exists() and str(env / exe) not in found:
                    found.append(str(env / exe))
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


def _int_or_none(raw, label):
    raw = raw.strip()
    if raw == "" or raw.lower() == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{label} must be a whole number of samples, got {raw!r}")


class TaskDayGUI(tk.Tk):
    def __init__(self, animal=None):
        super().__init__()
        self.title("Task day extraction")
        self.configs = {}                  # date -> config dict shown in the table
        self.current_date = None
        self.current_task = None
        self.proc = None
        self.run_queue = []
        self._scan_token = 0
        self._ui_queue = queue.Queue()

        interpreters = find_interpreters()
        self.animal_var = tk.StringVar(value=animal or "")
        self.interp_var = tk.StringVar(value=default_interpreter(interpreters))
        self.overwrite_var = tk.BooleanVar(value=False)
        self.sortout_var = tk.StringVar()
        self.pre_var = tk.StringVar()
        self.post_var = tk.StringVar()
        self.notes_var = tk.StringVar()
        self.t_rec_var = tk.StringVar()
        self.t_start_var = tk.StringVar()
        self.t_end_var = tk.StringVar()
        self.t_din_var = tk.StringVar(value="auto")
        self.status_var = tk.StringVar(value=f"Configs saved to {REGISTRY_FILE}")

        self._build(interpreters)
        self.after(100, self._drain_ui_queue)
        threading.Thread(target=self._load_animals, daemon=True).start()
        if animal:
            self.after(200, self.scan_days)

    # ------------------------------------------------------------------ layout
    def _build(self, interpreters):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(3, weight=1)

        top = ttk.Frame(self, padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(5, weight=1)
        ttk.Label(top, text="Animal").grid(row=0, column=0, padx=(0, 6))
        self.animal_box = ttk.Combobox(top, textvariable=self.animal_var, width=12)
        self.animal_box.grid(row=0, column=1)
        self.animal_box.bind("<<ComboboxSelected>>", lambda _e: self.scan_days())
        self.animal_box.bind("<Return>", lambda _e: self.scan_days())
        ttk.Button(top, text="Scan days", command=self.scan_days).grid(row=0, column=2, padx=6)
        ttk.Label(top, text="Python").grid(row=0, column=4, padx=(20, 6))
        ttk.Combobox(top, textvariable=self.interp_var, values=interpreters).grid(
            row=0, column=5, sticky="ew")

        # ---- day table
        days = ttk.LabelFrame(self, text="Days  (select one or more; ctrl/shift-click)", padding=6)
        days.grid(row=1, column=0, sticky="nsew", padx=10)
        days.columnconfigure(0, weight=1)
        days.rowconfigure(0, weight=1)
        self.day_tree = ttk.Treeview(days, columns=[c[0] for c in DAY_COLUMNS], show="headings",
                                     height=10, selectmode="extended")
        for key, label, width in DAY_COLUMNS:
            self.day_tree.heading(key, text=label)
            self.day_tree.column(key, width=width, stretch=(key == "notes"))
        self.day_tree.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(days, orient="vertical", command=self.day_tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.day_tree.configure(yscrollcommand=sb.set)
        self.day_tree.bind("<<TreeviewSelect>>", lambda _e: self._day_selected())
        self.day_tree.tag_configure("done", foreground="#1b7a1b")
        self.day_tree.tag_configure("blocked", foreground="#999999")

        # ---- day detail
        detail = ttk.LabelFrame(self, text="Selected day", padding=6)
        detail.grid(row=2, column=0, sticky="ew", padx=10, pady=6)
        detail.columnconfigure(1, weight=1)
        ttk.Label(detail, text="Sortout folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(detail, textvariable=self.sortout_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(detail, text="Browse...", command=self.browse_sortout).grid(row=0, column=2)
        win = ttk.Frame(detail)
        win.grid(row=1, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(win, text="Trial window: pre (s)").pack(side="left")
        ttk.Entry(win, textvariable=self.pre_var, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(win, text="post (s)").pack(side="left")
        ttk.Entry(win, textvariable=self.post_var, width=6).pack(side="left", padx=4)
        ttk.Label(win, text="   (spikes kept in [stim on - pre, stim off + post])",
                  foreground="gray40").pack(side="left")

        self.task_tree = ttk.Treeview(detail, columns=[c[0] for c in TASK_COLUMNS],
                                      show="headings", height=4, selectmode="browse")
        for key, label, width in TASK_COLUMNS:
            self.task_tree.heading(key, text=label)
            self.task_tree.column(key, width=width, stretch=(key in ("json", "rec", "last")))
        self.task_tree.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.task_tree.bind("<<TreeviewSelect>>", lambda _e: self._task_selected())

        edit = ttk.Frame(detail)
        edit.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        edit.columnconfigure(1, weight=1)
        ttk.Label(edit, text="Task .rec folder").grid(row=0, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.t_rec_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(edit, text="Browse...", command=self.browse_rec).grid(row=0, column=2)
        row2 = ttk.Frame(edit)
        row2.grid(row=1, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(row2, text="task_start").pack(side="left")
        ttk.Entry(row2, textvariable=self.t_start_var, width=14).pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="task_end").pack(side="left")
        ttk.Entry(row2, textvariable=self.t_end_var, width=14).pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="Photodiode DIN").pack(side="left")
        ttk.Combobox(row2, textvariable=self.t_din_var, values=DIN_CHOICES, width=6,
                     state="readonly").pack(side="left", padx=(4, 12))
        ttk.Button(row2, text="Apply to task", command=self.apply_task_edit).pack(side="left")
        ttk.Label(detail, textvariable=self.notes_var, foreground="#8a5a00", wraplength=1000,
                  justify="left").grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))

        buttons = ttk.Frame(detail)
        buttons.grid(row=5, column=0, columnspan=3, sticky="e", pady=(6, 0))
        ttk.Button(buttons, text="Re-detect from disk", command=self.rediscover).pack(side="left", padx=3)
        ttk.Button(buttons, text="Save config", command=self.save_current).pack(side="left", padx=3)
        ttk.Separator(buttons, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Checkbutton(buttons, text="Overwrite existing pkl", variable=self.overwrite_var).pack(side="left", padx=3)
        ttk.Button(buttons, text="Check DIO", command=lambda: self.run_selected(check_only=True)).pack(side="left", padx=3)
        ttk.Button(buttons, text="Extract selected day(s)", command=self.run_selected).pack(side="left", padx=3)
        ttk.Button(buttons, text="Extract all unfinished", command=self.run_unfinished).pack(side="left", padx=3)
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=3)

        # ---- log
        logf = ttk.LabelFrame(self, text="Log", padding=4)
        logf.grid(row=3, column=0, sticky="nsew", padx=10)
        logf.columnconfigure(0, weight=1)
        logf.rowconfigure(0, weight=1)
        self.log_text = tk.Text(logf, height=12, wrap="none", font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")
        lsb = ttk.Scrollbar(logf, orient="vertical", command=self.log_text.yview)
        lsb.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=lsb.set)
        ttk.Label(self, textvariable=self.status_var, foreground="gray30").grid(
            row=4, column=0, sticky="w", padx=10, pady=(2, 8))

    # ------------------------------------------------------------------ threading helpers
    def _post(self, fn):
        self._ui_queue.put(fn)

    def _drain_ui_queue(self):
        try:
            while True:
                self._ui_queue.get_nowait()()
        except queue.Empty:
            pass
        self.after(100, self._drain_ui_queue)

    def log(self, text):
        self.log_text.insert("end", text.rstrip("\n") + "\n")
        self.log_text.see("end")

    # ------------------------------------------------------------------ scanning
    def _load_animals(self):
        animals = list_animals()
        self._post(lambda: self.animal_box.configure(values=animals))

    def scan_days(self):
        animal = self.animal_var.get().strip()
        if not animal:
            return
        self._scan_token += 1
        token = self._scan_token
        self.configs.clear()
        self.day_tree.delete(*self.day_tree.get_children())
        self._clear_detail()
        self.status_var.set(f"Scanning {animal} days on the server...")
        threading.Thread(target=self._scan_worker, args=(animal, token), daemon=True).start()

    def _scan_worker(self, animal, token):
        registry = load_registry()
        dates = list_days(animal)
        for i, date in enumerate(dates):
            if token != self._scan_token:
                return
            config = registry.get(day_key(animal, date))
            if config is None:
                try:
                    config = discover_task_day(animal, date)
                except Exception as exc:      # an unreadable day must not stop the scan
                    config = {"animal": animal, "date": date, "tasks": [],
                              "notes": [f"discovery failed: {exc}"]}
            else:
                config["_registered"] = True
                fill_output_paths(config)
            status = day_status(config)
            self._post(lambda d=date, c=config, s=status, n=i + 1, tot=len(dates), t=token:
                       self._add_day_row(d, c, s, n, tot, t))
        if not dates:
            self._post(lambda: self.status_var.set(f"No experiment_data days found for {animal}."))

    def _add_day_row(self, date, config, status, n, total, token):
        if token != self._scan_token:
            return
        self.configs[date] = config
        values = self._day_values(date, config, status)
        tag = self._day_tag(config, status)
        if self.day_tree.exists(date):
            self.day_tree.item(date, values=values, tags=(tag,))
        else:
            self.day_tree.insert("", "end", iid=date, values=values, tags=(tag,))
        self.status_var.set(f"Scanned {n}/{total} days" + ("" if n < total else
                            f" - {REGISTRY_FILE.name} entries shown as registered (*)"))

    @staticmethod
    def _day_values(date, config, status):
        notes = "; ".join(config.get("notes", []))
        label = date + (" *" if config.get("_registered") else "")
        ready = "yes" if status["ready"] else "no"
        if status["ready"] and status["n_ready"] < status["n_tasks"]:
            ready = f"{status['n_ready']}/{status['n_tasks']}"
        return (label, status["n_tasks"], ready, status["trial_done"], status["task_done"], notes)

    @staticmethod
    def _day_tag(config, status):
        if not status["ready"]:
            return "blocked"
        n = status["n_ready"]
        if status["trial_done"] == status["task_done"] == f"{n}/{status['n_tasks']}" and n:
            return "done"
        return ""

    def _refresh_day_row(self, date):
        config = self.configs.get(date)
        if config is None or not self.day_tree.exists(date):
            return
        status = day_status(config)
        self.day_tree.item(date, values=self._day_values(date, config, status),
                           tags=(self._day_tag(config, status),))

    # ------------------------------------------------------------------ detail pane
    def _clear_detail(self):
        self.current_date = self.current_task = None
        for var in (self.sortout_var, self.pre_var, self.post_var, self.notes_var,
                    self.t_rec_var, self.t_start_var, self.t_end_var):
            var.set("")
        self.t_din_var.set("auto")
        self.task_tree.delete(*self.task_tree.get_children())

    def _day_selected(self):
        sel = self.day_tree.selection()
        if not sel:
            return
        self._commit_day_fields()
        date = sel[0]
        config = self.configs.get(date)
        if config is None:
            return
        self._clear_detail()
        self.current_date = date
        self.sortout_var.set(config.get("sortout_folder") or "")
        self.pre_var.set(str(config.get("pre_sec", "")))
        self.post_var.set(str(config.get("post_sec", "")))
        self.notes_var.set("\n".join(config.get("notes", [])))
        self._fill_task_tree(config)

    def _fill_task_tree(self, config):
        self.task_tree.delete(*self.task_tree.get_children())
        for i, t in enumerate(config.get("tasks", [])):
            window = (f"{t['task_start']} - {t['task_end']}"
                      if t.get("task_start") is not None and t.get("task_end") is not None else "(unset)")
            last = t.get("last_run") or {}
            last_txt = last.get("status", "")
            if "n_matched" in last:
                last_txt += (f" | Din{last.get('din_channel_used')} "
                             f"{last['n_matched']}/{last.get('n_json_trials')} matched")
            self.task_tree.insert("", "end", iid=str(i), values=(
                Path(t["task_file"]).name, t.get("n_json_trials", ""), t.get("rec_name") or "(none)",
                window, t.get("din_channel", "auto"), last_txt))

    def _task_selected(self):
        sel = self.task_tree.selection()
        config = self.configs.get(self.current_date)
        if not sel or config is None:
            return
        self.current_task = int(sel[0])
        t = config["tasks"][self.current_task]
        self.t_rec_var.set(t.get("rec_folder") or "")
        self.t_start_var.set("" if t.get("task_start") is None else str(t["task_start"]))
        self.t_end_var.set("" if t.get("task_end") is None else str(t["task_end"]))
        self.t_din_var.set(str(t.get("din_channel", "auto")))

    def apply_task_edit(self):
        config = self.configs.get(self.current_date)
        if config is None or self.current_task is None:
            messagebox.showinfo("Apply", "Select a task row first.", parent=self)
            return
        t = config["tasks"][self.current_task]
        try:
            start = _int_or_none(self.t_start_var.get(), "task_start")
            end = _int_or_none(self.t_end_var.get(), "task_end")
        except ValueError as exc:
            messagebox.showerror("Apply", str(exc), parent=self)
            return
        if start is not None and end is not None and end <= start:
            messagebox.showerror("Apply", "task_end must be greater than task_start.", parent=self)
            return
        rec = self.t_rec_var.get().strip().strip('"') or None
        t.update(rec_folder=rec, rec_name=Path(rec).name if rec else None,
                 task_start=start, task_end=end, din_channel=self.t_din_var.get() or "auto")
        fill_output_paths(config)
        self._fill_task_tree(config)
        self.task_tree.selection_set(str(self.current_task))
        self._refresh_day_row(self.current_date)
        self.status_var.set("Task edited - Save config (or run) to keep it.")

    def _commit_day_fields(self):
        """Copy the day-level entry fields back into the shown config."""
        config = self.configs.get(self.current_date)
        if config is None:
            return
        config["sortout_folder"] = self.sortout_var.get().strip().strip('"') or None
        for key, var in (("pre_sec", self.pre_var), ("post_sec", self.post_var)):
            try:
                config[key] = float(var.get())
            except ValueError:
                pass
        fill_output_paths(config)

    def browse_sortout(self):
        chosen = filedialog.askdirectory(title="Select the day's sortout folder",
                                         initialdir=self.sortout_var.get() or None)
        if chosen:
            self.sortout_var.set(str(Path(chosen)))
            self._commit_day_fields()
            self._refresh_day_row(self.current_date)

    def browse_rec(self):
        config = self.configs.get(self.current_date) or {}
        chosen = filedialog.askdirectory(title="Select the task .rec folder",
                                         initialdir=self.t_rec_var.get() or config.get("ephys_folder"))
        if chosen:
            self.t_rec_var.set(str(Path(chosen)))

    def rediscover(self):
        if self.current_date is None:
            return
        animal = self.animal_var.get().strip()
        if self.configs[self.current_date].get("_registered") and not messagebox.askyesno(
                "Re-detect", "Replace this day's saved config with a fresh detection from disk?\n"
                             "(Hand edits are lost once you Save.)", parent=self):
            return
        config = discover_task_day(animal, self.current_date)
        self.configs[self.current_date] = config
        self._refresh_day_row(self.current_date)
        self._day_selected()
        self.status_var.set("Re-detected from disk (not saved yet).")

    def save_current(self, quiet=False):
        self._commit_day_fields()
        config = self.configs.get(self.current_date)
        if config is None:
            return False
        save_day_config({k: v for k, v in config.items() if not k.startswith("_")})
        config["_registered"] = True
        self._refresh_day_row(self.current_date)
        if not quiet:
            self.status_var.set(f"Saved {day_key(config['animal'], config['date'])} to {REGISTRY_FILE.name}")
        return True

    # ------------------------------------------------------------------ running
    def run_selected(self, check_only=False):
        dates = list(self.day_tree.selection())
        if not dates:
            messagebox.showinfo("Run", "Select one or more days first.", parent=self)
            return
        self._start_queue(dates, check_only)

    def run_unfinished(self):
        dates = [d for d, c in self.configs.items()
                 if self._day_tag(c, day_status(c)) == ""]
        if not dates:
            messagebox.showinfo("Run", "No ready day is missing outputs.", parent=self)
            return
        if messagebox.askyesno("Extract all unfinished",
                               f"Extract {len(dates)} day(s)?\n" + ", ".join(sorted(dates)),
                               parent=self):
            self._start_queue(sorted(dates), check_only=False)

    def _start_queue(self, dates, check_only):
        if self.proc is not None:
            messagebox.showinfo("Run", "A job is already running.", parent=self)
            return
        # the extractor reads task_day_configs.json, so the configs shown here are saved first
        self._commit_day_fields()
        for date in dates:
            config = self.configs.get(date)
            if config is not None:
                save_day_config({k: v for k, v in config.items() if not k.startswith("_")})
                config["_registered"] = True
        self.run_queue = [(d, check_only) for d in dates]
        self.stop_button.configure(state="normal")
        self._next_in_queue()

    def _next_in_queue(self):
        if not self.run_queue:
            self.proc = None
            self.stop_button.configure(state="disabled")
            self.status_var.set("Done.")
            return
        date, check_only = self.run_queue.pop(0)
        animal = self.animal_var.get().strip()
        argv = [self.interp_var.get().strip() or sys.executable, "-u", str(EXTRACT_SCRIPT), animal, date]
        if check_only:
            argv.append("--check-only")
        if self.overwrite_var.get():
            argv.append("--overwrite")
        self.log(f"\n>>> {' '.join(argv)}")
        self.status_var.set(f"Running {animal} {date}{' (check)' if check_only else ''}... "
                            f"{len(self.run_queue)} more queued")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        try:
            self.proc = subprocess.Popen(argv, cwd=str(GRATING_DIR), stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                         errors="replace", env=env,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            self.log(f"could not start: {exc}")
            self.proc = None
            self._next_in_queue()
            return
        threading.Thread(target=self._pump, args=(self.proc, date), daemon=True).start()

    def _pump(self, proc, date):
        for line in proc.stdout:
            self._post(lambda l=line: self.log(l))
        code = proc.wait()
        self._post(lambda: self._job_finished(date, code))

    def _job_finished(self, date, code):
        self.log(f"<<< {date}: {'ok' if code == 0 else f'exit code {code}'}")
        config = load_registry().get(day_key(self.animal_var.get().strip(), date))
        if config is not None:
            config["_registered"] = True
            fill_output_paths(config)
            self.configs[date] = config
            self._refresh_day_row(date)
            if date == self.current_date:
                self._day_selected()
        self._next_in_queue()

    def stop(self):
        self.run_queue.clear()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.log("... stopped by user")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--animal", default=None, help="preselect and scan this animal")
    args = parser.parse_args()
    try:  # crisp text on high-DPI Windows displays
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = TaskDayGUI(animal=args.animal)
    app.geometry("1200x860")
    app.mainloop()


if __name__ == "__main__":
    main()
