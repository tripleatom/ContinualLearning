"""
Tkinter front-end for decoder_workbench.py.

Pick the data context, tick the kinematic properties to send into the decoder
(firing rate is always included), tick the classifiers, set bins / CV, press Run.
The panel builds a decoder_workbench.py command line, runs it as a subprocess and
streams its output, so the heavy imports (numpy, sklearn) never load in the GUI and
a crashed run cannot take the panel down.

Stdlib only.  Run with the environment that has sklearn (ms10):
    python reactivation/decoders/decoder_gui.py
"""

import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decoder_workbench as wb  # light: only constants + params at import time

SCRIPT = HERE / "decoder_workbench.py"
PROGRESS_RE = re.compile(r"^\[progress\] (\d+)/(\d+)")
SAVED_RE = re.compile(r"^\[saved\] (.+)$")


def build_argv(state, python=sys.executable):
    """Command line for one run from a plain-dict GUI state (kept pure for testing)."""
    argv = [python, "-u", str(SCRIPT), "--context", state["context"]]
    if state["features"]:
        argv += ["--features", *state["features"]]
    argv += ["--classifiers", *state["classifiers"],
             "--bins", *[str(b) for b in state["bins"]],
             "--splits", str(state["splits"]), "--repeats", str(state["repeats"]),
             "--shuffles", str(state["shuffles"]), "--seed", str(state["seed"]),
             "--task-pkl", state["task_pkl"], "--passive-pkl", state["passive_pkl"]]
    if not state["baseline"]:
        argv.append("--no-baseline")
    if state["count_models"]:
        argv.append("--count-models-see-features")
    if state["speed_match"]:
        argv.append("--speed-match")
    if state["context"] == "merged" and not state["class_group"]:
        argv += ["--merged-balance", "class"]
    if state["out_dir"]:
        argv += ["--out-dir", state["out_dir"]]
    return argv


def parse_bins(text):
    values = [int(tok) for tok in re.split(r"[,\s]+", text.strip()) if tok]
    if not values or any(v <= 0 for v in values):
        raise ValueError("Bin sizes must be positive integers in ms, e.g. 50 100 200")
    return sorted(set(values))


class App:
    def __init__(self, root):
        self.root = root
        root.title("Decoder workbench")
        root.minsize(900, 700)
        d = wb.default_config()

        self.context = tk.StringVar(value=d["context"])
        self.feature_vars = {f: tk.BooleanVar(value=False) for f in wb.FEATURE_GROUPS}
        self.clf_vars = {c: tk.BooleanVar(value=True) for c in wb.CLASSIFIER_NAMES}
        self.bins_var = tk.StringVar(value=" ".join(map(str, d["bin_sizes_ms"])))
        self.splits = tk.IntVar(value=d["n_splits"])
        self.repeats = tk.IntVar(value=d["n_repeats"])
        self.shuffles = tk.IntVar(value=d["n_shuffles"])
        self.seed = tk.IntVar(value=d["random_state"])
        self.baseline = tk.BooleanVar(value=True)
        self.class_group = tk.BooleanVar(value=True)
        self.count_models = tk.BooleanVar(value=False)
        self.speed_match = tk.BooleanVar(value=False)
        self.task_pkl = tk.StringVar(value=d["task_pkl"])
        self.passive_pkl = tk.StringVar(value=d["passive_pkl"])
        self.out_dir = tk.StringVar(value="")
        self.python = tk.StringVar(value=sys.executable)

        self.proc = None
        self.lines = queue.Queue()
        self.saved = []
        self._build()
        for var in (self.context, self.bins_var, self.splits, self.repeats, self.shuffles,
                    self.seed, self.baseline, self.task_pkl, self.passive_pkl, self.out_dir,
                    self.class_group, self.count_models, self.speed_match, *self.feature_vars.values(), *self.clf_vars.values()):
            var.trace_add("write", lambda *_: self._refresh())
        self._refresh()
        root.after(100, self._pump)

    # -- layout ---------------------------------------------------------

    def _build(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        top = ttk.Frame(outer)
        top.grid(row=0, column=0, sticky="ew")
        for col in range(3):
            top.columnconfigure(col, weight=1, uniform="cols")

        # 1. data context
        box = ttk.LabelFrame(top, text="1. Data", padding=8)
        box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        for value, text in (("task", "Task (freely moving)"),
                            ("passive", "Passive (head-fixed)"),
                            ("merged", "Merged task + passive")):
            ttk.Radiobutton(box, text=text, value=value, variable=self.context).pack(anchor="w")
        self.class_group_cb = ttk.Checkbutton(box, text="Merged: balance training by\nclass x context (newer method)",
                                              variable=self.class_group)
        self.class_group_cb.pack(anchor="w", pady=(6, 0))
        ttk.Label(box, text="Labels: +1 / -1 / 0 (ITI) from\nparams.class_pos / class_neg",
                  foreground="gray30").pack(anchor="w", pady=(8, 0))

        # 2. features
        box = ttk.LabelFrame(top, text="2. Inputs to the decoder", padding=8)
        box.grid(row=0, column=1, sticky="nsew", padx=6)
        always = ttk.Checkbutton(box, text="Firing rate (all units)  - always on")
        always.state(["selected", "disabled"])
        always.pack(anchor="w")
        ttk.Separator(box).pack(fill="x", pady=4)
        self.feature_boxes = {}
        for name in wb.FEATURE_GROUPS:
            cb = ttk.Checkbutton(box, text=f"{name.replace('_', ' ')}  ({wb.FEATURE_HELP[name]})",
                                 variable=self.feature_vars[name])
            cb.pack(anchor="w")
            self.feature_boxes[name] = cb
        self.feature_note = ttk.Label(box, text="", foreground="gray30", wraplength=250)
        self.feature_note.pack(anchor="w", pady=(6, 0))
        ttk.Checkbutton(box, text="Also run rate-only baseline\n(paired comparison on the same bins)",
                        variable=self.baseline).pack(anchor="w", pady=(8, 0))
        self.speed_cb = ttk.Checkbutton(
            box, text="Speed-match the ITI to stim-on\n(task only; matched vs uniform ITI)",
            variable=self.speed_match)
        self.speed_cb.pack(anchor="w", pady=(8, 0))

        # 3. classifiers
        box = ttk.LabelFrame(top, text="3. Classifiers", padding=8)
        box.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        for name in wb.CLASSIFIER_NAMES:
            text = name + ("  (rate columns only)" if name in wb.RATE_ONLY_CLASSIFIERS else "")
            ttk.Checkbutton(box, text=text, variable=self.clf_vars[name]).pack(anchor="w")
        ttk.Checkbutton(box, text="Feed added inputs to Poisson NB / AODE\n(old scripts did; off = rate only)",
                        variable=self.count_models).pack(anchor="w", pady=(6, 0))
        row = ttk.Frame(box)
        row.pack(anchor="w", pady=(6, 0))
        ttk.Button(row, text="All", width=6, command=lambda: self._set_all_clf(True)).pack(side="left")
        ttk.Button(row, text="None", width=6, command=lambda: self._set_all_clf(False)).pack(side="left", padx=4)

        # sweep + paths
        mid = ttk.LabelFrame(outer, text="4. Sweep and paths", padding=8)
        mid.grid(row=1, column=0, sticky="ew", pady=8)
        mid.columnconfigure(1, weight=1)
        ttk.Label(mid, text="Bin sizes (ms)").grid(row=0, column=0, sticky="w")
        ttk.Entry(mid, textvariable=self.bins_var).grid(row=0, column=1, sticky="ew", padx=6)
        nums = ttk.Frame(mid)
        nums.grid(row=0, column=2, sticky="e")
        for label, var, hi in (("CV folds", self.splits, 20), ("repeats", self.repeats, 50),
                               ("shuffles", self.shuffles, 200), ("seed", self.seed, 10**6)):
            ttk.Label(nums, text=label).pack(side="left", padx=(8, 2))
            ttk.Spinbox(nums, textvariable=var, from_=0, to=hi, width=6).pack(side="left")
        for r, (label, var, browse) in enumerate((
                ("Task spikes pkl", self.task_pkl, "file"),
                ("Passive spikes pkl", self.passive_pkl, "file"),
                ("Output folder", self.out_dir, "dir"),
                ("Python", self.python, "file")), start=1):
            ttk.Label(mid, text=label).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(mid, textvariable=var).grid(row=r, column=1, columnspan=2, sticky="ew", padx=6)
            ttk.Button(mid, text="Browse", width=8,
                       command=lambda v=var, k=browse: self._browse(v, k)).grid(row=r, column=3)
        ttk.Label(mid, text="(output folder blank = <session folder>/reactivation)",
                  foreground="gray30").grid(row=3, column=1, sticky="e", columnspan=2)

        # command + run
        run = ttk.Frame(outer)
        run.grid(row=2, column=0, sticky="ew")
        run.columnconfigure(0, weight=1)
        self.cmd_text = tk.Text(run, height=3, wrap="word", font=("Consolas", 9),
                                background="#f4f4f4", relief="flat")
        self.cmd_text.grid(row=0, column=0, sticky="ew")
        self.summary = ttk.Label(run, text="", foreground="gray30")
        self.summary.grid(row=1, column=0, sticky="w", pady=(2, 0))
        buttons = ttk.Frame(run)
        buttons.grid(row=0, column=1, rowspan=2, padx=(8, 0))
        self.run_btn = ttk.Button(buttons, text="Run", width=12, command=self.start)
        self.run_btn.pack(pady=1)
        self.stop_btn = ttk.Button(buttons, text="Stop", width=12, command=self.stop, state="disabled")
        self.stop_btn.pack(pady=1)

        # log
        logbox = ttk.LabelFrame(outer, text="Log", padding=4)
        logbox.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        logbox.columnconfigure(0, weight=1)
        logbox.rowconfigure(0, weight=1)
        self.log = tk.Text(logbox, height=14, wrap="none", font=("Consolas", 9))
        scroll = ttk.Scrollbar(logbox, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        foot = ttk.Frame(outer)
        foot.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        foot.columnconfigure(0, weight=1)
        self.bar = ttk.Progressbar(foot, maximum=100)
        self.bar.grid(row=0, column=0, sticky="ew")
        self.open_fig = ttk.Button(foot, text="Open figure", command=lambda: self._open(".png"),
                                   state="disabled")
        self.open_fig.grid(row=0, column=1, padx=(8, 2))
        self.open_dir = ttk.Button(foot, text="Open folder", command=lambda: self._open("dir"),
                                   state="disabled")
        self.open_dir.grid(row=0, column=2)

    # -- state ----------------------------------------------------------

    def _set_all_clf(self, value):
        for var in self.clf_vars.values():
            var.set(value)

    def _browse(self, var, kind):
        current = var.get()
        start = str(Path(current).parent) if current and Path(current).parent.exists() else None
        path = (filedialog.askdirectory(initialdir=start) if kind == "dir"
                else filedialog.askopenfilename(initialdir=start))
        if path:
            var.set(path)

    def state(self):
        features = [f for f in wb.FEATURE_GROUPS if self.feature_vars[f].get()]
        if self.context.get() == "passive":
            features = []
        return {
            "context": self.context.get(), "features": features,
            "classifiers": [c for c in wb.CLASSIFIER_NAMES if self.clf_vars[c].get()],
            "bins": parse_bins(self.bins_var.get()),
            "splits": self.splits.get(), "repeats": self.repeats.get(),
            "shuffles": self.shuffles.get(), "seed": self.seed.get(),
            "baseline": self.baseline.get(), "class_group": self.class_group.get(), "count_models": self.count_models.get(),
            "speed_match": self.speed_match.get() and self.context.get() == "task",
            "task_pkl": self.task_pkl.get(), "passive_pkl": self.passive_pkl.get(),
            "out_dir": self.out_dir.get().strip(),
        }

    def _refresh(self):
        passive = self.context.get() == "passive"
        for name, cb in self.feature_boxes.items():
            cb.state(["disabled"] if passive else ["!disabled"])
        self.class_group_cb.state(["!disabled"] if self.context.get() == "merged" else ["disabled"])
        self.speed_cb.state(["!disabled"] if self.context.get() == "task" else ["disabled"])
        self.feature_note.configure(
            text=("Passive is head-fixed: no kinematics, rate only." if passive else
                  "Merged: passive rows get zeros in these columns." if self.context.get() == "merged"
                  else "Bins without clean tracking are dropped."))
        try:
            state = self.state()
            argv = build_argv(state, self.python.get() or sys.executable)
            text = subprocess.list2cmdline(argv)
            arms = 2 if state["speed_match"] or (state["features"] and state["baseline"]) else 1
            fits = (len(state["classifiers"]) * state["repeats"] * arms * len(state["bins"])
                    * state["splits"])
            summary = (f"{len(state['classifiers'])} classifiers x {len(state['bins'])} bins x "
                       f"{state['repeats']} repeats x {arms} arm(s) x {state['splits']} folds"
                       f" = {fits} fits"
                       + (f", plus {state['shuffles']} shuffles per cell" if state["shuffles"] else ""))
        except (ValueError, tk.TclError) as exc:
            text, summary = f"(fix inputs: {exc})", ""
        self.cmd_text.configure(state="normal")
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("1.0", text)
        self.cmd_text.configure(state="disabled")
        self.summary.configure(text=summary)

    # -- running --------------------------------------------------------

    def _append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        if self.proc is not None:
            return
        try:
            state = self.state()
            if not state["classifiers"]:
                raise ValueError("Select at least one classifier.")
            wb.validate({"context": state["context"], "features": state["features"],
                         "classifiers": state["classifiers"], "bin_sizes_ms": state["bins"],
                         "n_splits": state["splits"], "n_repeats": state["repeats"],
                         "speed_match": state["speed_match"]})
            argv = build_argv(state, self.python.get() or sys.executable)
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Cannot run", str(exc))
            return
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["MPLBACKEND"] = "Agg"
        self.saved = []
        self.bar.configure(value=0)
        self._append(f"\n$ {subprocess.list2cmdline(argv)}\n")
        try:
            self.proc = subprocess.Popen(argv, cwd=str(HERE), env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         text=True, encoding="utf-8", errors="replace", bufsize=1)
        except OSError as exc:
            messagebox.showerror("Cannot start", str(exc))
            return
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.open_fig.configure(state="disabled")
        self.open_dir.configure(state="disabled")
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        for line in proc.stdout:
            self.lines.put(line)
        proc.wait()
        self.lines.put(None)

    def _pump(self):
        try:
            while True:
                line = self.lines.get_nowait()
                if line is None:
                    self._finished()
                    continue
                match = PROGRESS_RE.match(line)
                if match:
                    done, total = int(match.group(1)), int(match.group(2))
                    self.bar.configure(value=100.0 * done / max(total, 1))
                    continue
                saved = SAVED_RE.match(line.strip())
                if saved:
                    self.saved.append(saved.group(1))
                self._append(line)
        except queue.Empty:
            pass
        self.root.after(100, self._pump)

    def _finished(self):
        code = self.proc.returncode if self.proc else None
        self.proc = None
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._append(f"\n[exit {code}]\n")
        if code == 0 and self.saved:
            self.bar.configure(value=100)
            self.open_fig.configure(state="normal")
            self.open_dir.configure(state="normal")

    def stop(self):
        if self.proc is not None:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           capture_output=True)
            self._append("\n[stopped by user]\n")

    def _open(self, what):
        if not self.saved:
            return
        if what == "dir":
            os.startfile(str(Path(self.saved[-1]).parent))
            return
        figs = [p for p in self.saved if p.lower().endswith(what)]
        if figs:
            os.startfile(figs[-1])


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
