"""
Tkinter GUI for maintaining the Grating experiment log CSVs.

Each animal has one CSV in experiment_log/ with the columns
``date, EphysFolder, PassiveFolder, TaskFile`` that
grating_utils.load_session_paths reads to resolve a session's .rec folder(s)
and drifting-grating task .txt file(s):

    <experiment_data root>/<animal>/<date>/            <- session base
        <EphysFolder>/
            <PassiveFolder>.rec                        <- one or more, ';'-joined
        <TaskFile>.txt                                 <- one or more, ';'-joined

This GUI scans the experiment_data roots for what is actually on disk, so the
columns get filled in by picking real folders instead of typing names by hand.
Multi-entry fields keep the semicolon order, which is what PASSIVE_WINDOWS in
grating_config.py indexes into.

Stdlib only (tkinter + csv), so it runs in any environment.

Run with:  python experiment_log_gui.py
"""
import csv
import io
import platform
import re
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

FIELDS = ["date", "EphysFolder", "PassiveFolder", "TaskFile"]
LOG_DIR = Path(__file__).resolve().parent / "experiment_log"
DATE_RE = re.compile(r"^\d{6}$")


# =============================================================================
# DISK LAYOUT
# =============================================================================

def experiment_data_roots():
    """Candidate experiment_data roots, in the order load_session_paths tries them."""
    if platform.system() == "Darwin":
        return [
            Path(r"/Volumes/xieluanlabs/xl_cl/experiment_data"),
            Path(r"/Volumes/xieluanlabs2/xl_cl/experiment_data"),
        ]
    return [
        Path(r"\\10.129.151.108\xieluanlabs\xl_cl\experiment_data"),
        Path(r"\\10.129.151.88\xieluanlabs2\xl_cl\experiment_data"),
    ]


def scan_animals():
    """Animal ids present under any experiment_data root."""
    animals = set()
    for root in experiment_data_roots():
        try:
            animals.update(c.name for c in root.iterdir() if c.is_dir())
        except OSError:
            continue
    return sorted(animals)


def scan_dates(animal):
    """YYMMDD session folders on disk for one animal."""
    dates = set()
    for root in experiment_data_roots():
        try:
            entries = list((root / animal).iterdir())
        except OSError:
            continue
        dates.update(c.name for c in entries if c.is_dir() and DATE_RE.match(c.name))
    return sorted(dates)


def find_session_base(animal, date):
    """First existing <root>/<animal>/<date>, or None."""
    for root in experiment_data_roots():
        base = root / animal / date
        try:
            if base.is_dir():
                return base
        except OSError:
            continue
    return None


def scan_session(animal, date):
    """
    Inventory one session folder.

    Returns
    -------
    dict with keys ``base`` (Path or None), ``ephys`` (candidate EphysFolder
    names), ``rec`` ({ephys folder name: [.rec names]}) and ``task`` (.txt names
    directly under the session base).
    """
    out = {"base": None, "ephys": [], "rec": {}, "task": []}
    base = find_session_base(animal, date)
    if base is None:
        return out
    out["base"] = base

    try:
        entries = sorted(base.iterdir(), key=lambda p: p.name)
    except OSError:
        return out

    for entry in entries:
        if entry.is_dir() and not entry.name.endswith(".rec"):
            out["ephys"].append(entry.name)
        elif entry.is_file() and entry.suffix == ".txt":
            out["task"].append(entry.name)

    for name in out["ephys"]:
        try:
            children = sorted((base / name).iterdir(), key=lambda p: p.name)
        except OSError:
            children = []
        out["rec"][name] = [c.name for c in children if c.name.endswith(".rec")]

    return out


# =============================================================================
# CSV I/O
# =============================================================================

def log_path(animal):
    return LOG_DIR / f"{animal}.csv"


def read_log(animal):
    """
    Read one animal's log.

    Returns ``(rows, sep)`` where rows is a list of dicts over FIELDS and sep is
    the column separator the file already uses - plain "," (CnL42) or a comma
    padded with whitespace such as ",\\t" (CnL43/45/46) - so the existing style
    survives a round trip.
    """
    path = log_path(animal)
    if not path.exists():
        return [], ","

    text = path.read_text(encoding="utf-8-sig")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    match = re.search(r",([ \t]*)", first_line)
    sep = "," + (match.group(1) if match else "")

    reader = csv.DictReader(io.StringIO(text))
    header = [k.strip() for k in (reader.fieldnames or [])]
    reader.fieldnames = header

    rows = []
    for raw in reader:
        row = {k: (raw.get(k) or "").strip() for k in FIELDS}
        if any(row.values()):
            rows.append(row)
    return rows, sep


def _format_value(value):
    """Quote a cell only when it would otherwise break the CSV."""
    if any(ch in value for ch in ',"\n'):
        return '"' + value.replace('"', '""') + '"'
    return value


def _line_style(path):
    """
    (newline, ends_with_newline) already used by a file.

    These CSVs are a mix of CRLF and LF, and CnL45.csv has no final newline;
    matching what is there keeps a one-row edit to a one-line diff.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return "\n", True
    return ("\r\n" if b"\r\n" in raw else "\n"), raw.endswith(b"\n")


def write_log(animal, rows, sep=",", backup=True):
    """Write rows back, sorted by date, keeping a single .bak of the previous file."""
    path = log_path(animal)
    path.parent.mkdir(parents=True, exist_ok=True)
    newline, trailing = _line_style(path)
    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(".csv.bak"))

    lines = [sep.join(FIELDS)]
    for row in sorted(rows, key=lambda r: r["date"]):
        lines.append(sep.join(_format_value(row[k]) for k in FIELDS))
    text = newline.join(lines) + (newline if trailing else "")
    path.write_text(text, encoding="utf-8", newline="")
    return path


# =============================================================================
# GUI
# =============================================================================

class ExperimentLogGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Grating experiment log")
        self.geometry("1120x780")
        self.minsize(880, 620)

        self.rows = []          # list of dicts, current in-memory log
        self.sep = ","          # column separator style of the loaded CSV
        self.dirty = False
        self.session = {"base": None, "ephys": [], "rec": {}, "task": []}
        self._scan_token = 0

        self.animal_var = tk.StringVar()
        self.date_var = tk.StringVar()
        self.ephys_var = tk.StringVar()
        self.passive_var = tk.StringVar()
        self.task_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready.")
        self.base_var = tk.StringVar(value="Session folder: (not scanned)")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.refresh_animals()
        self.after(100, self.scan_animals_async)

    # -- construction ------------------------------------------------------

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=3)
        self.rowconfigure(2, weight=4)

        bar = ttk.Frame(self, padding=(10, 8))
        bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(bar, text="Animal:").pack(side="left")
        self.animal_combo = ttk.Combobox(bar, textvariable=self.animal_var, width=18)
        self.animal_combo.pack(side="left", padx=(6, 6))
        self.animal_combo.bind("<<ComboboxSelected>>", lambda _e: self.load_animal())
        self.animal_combo.bind("<Return>", lambda _e: self.load_animal())
        ttk.Button(bar, text="Load", command=self.load_animal).pack(side="left")
        ttk.Button(bar, text="Save CSV", command=self.save_log).pack(side="left", padx=(12, 0))
        ttk.Label(bar, textvariable=self.base_var, foreground="#555").pack(side="right")

        # -- existing rows --
        table = ttk.LabelFrame(self, text="Logged sessions", padding=(8, 6))
        table.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 6))
        table.columnconfigure(0, weight=1)
        table.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(table, columns=FIELDS, show="headings", selectmode="browse")
        widths = {"date": 70, "EphysFolder": 170, "PassiveFolder": 330, "TaskFile": 430}
        for field in FIELDS:
            self.tree.heading(field, text=field)
            self.tree.column(field, width=widths[field], anchor="w", stretch=(field == "TaskFile"))
        self.tree.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(table, orient="horizontal", command=self.tree.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.tree.bind("<<TreeviewSelect>>", self.on_row_selected)

        # -- editor --
        editor = ttk.LabelFrame(self, text="Session details", padding=(8, 6))
        editor.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 6))
        editor.columnconfigure(1, weight=1)
        editor.columnconfigure(3, weight=1)
        editor.rowconfigure(3, weight=1)

        ttk.Label(editor, text="date (YYMMDD):").grid(row=0, column=0, sticky="w", pady=3)
        date_row = ttk.Frame(editor)
        date_row.grid(row=0, column=1, sticky="ew", pady=3)
        self.date_combo = ttk.Combobox(date_row, textvariable=self.date_var, width=14)
        self.date_combo.pack(side="left")
        self.date_combo.bind("<<ComboboxSelected>>", lambda _e: self.scan_session_async())
        self.date_combo.bind("<Return>", lambda _e: self.scan_session_async())
        ttk.Button(date_row, text="Scan disk", command=self.scan_session_async).pack(side="left", padx=6)

        ttk.Label(editor, text="EphysFolder:").grid(row=0, column=2, sticky="w", padx=(16, 0), pady=3)
        self.ephys_combo = ttk.Combobox(editor, textvariable=self.ephys_var)
        self.ephys_combo.grid(row=0, column=3, sticky="ew", pady=3)
        self.ephys_combo.bind("<<ComboboxSelected>>", lambda _e: self.fill_rec_list())

        ttk.Label(editor, text="PassiveFolder (.rec, order matters):").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(editor, text="TaskFile (.txt, order matters):").grid(
            row=1, column=2, columnspan=2, sticky="w", padx=(16, 0), pady=(8, 0))

        self.passive_list = tk.Listbox(editor, selectmode="extended", exportselection=False, height=6)
        self.passive_list.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(2, 2))
        self.passive_list.bind("<<ListboxSelect>>",
                               lambda _e: self.sync_from_list(self.passive_list, self.passive_var))
        self.task_list = tk.Listbox(editor, selectmode="extended", exportselection=False, height=6)
        self.task_list.grid(row=2, column=2, columnspan=2, sticky="nsew", padx=(16, 0), pady=(2, 2))
        self.task_list.bind("<<ListboxSelect>>",
                            lambda _e: self.sync_from_list(self.task_list, self.task_var))
        editor.rowconfigure(2, weight=1)

        ttk.Entry(editor, textvariable=self.passive_var).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Entry(editor, textvariable=self.task_var).grid(
            row=3, column=2, columnspan=4, sticky="ew", padx=(16, 0), pady=(0, 4))
        editor.rowconfigure(3, weight=0)

        hint = ("Ctrl/Shift-click to pick several; the text box below each list is what gets "
                "written (';'-joined) and can be edited directly to reorder.")
        ttk.Label(editor, text=hint, foreground="#555", wraplength=1040).grid(
            row=4, column=0, columnspan=4, sticky="w")

        buttons = ttk.Frame(self, padding=(10, 0))
        buttons.grid(row=3, column=0, sticky="ew")
        ttk.Button(buttons, text="Add / Update row", command=self.add_or_update_row).pack(side="left")
        ttk.Button(buttons, text="Delete row", command=self.delete_row).pack(side="left", padx=6)
        ttk.Button(buttons, text="Clear fields", command=self.clear_fields).pack(side="left")
        ttk.Button(buttons, text="Verify paths", command=self.verify_paths).pack(side="left", padx=6)
        ttk.Button(buttons, text="Save CSV", command=self.save_log).pack(side="right")

        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w",
                  padding=(6, 3)).grid(row=4, column=0, sticky="ew", padx=10, pady=(6, 8))

    # -- background scans --------------------------------------------------

    def _async(self, work, done):
        """Run `work()` off the UI thread, hand its result to `done` on the UI thread."""
        token = self._scan_token = self._scan_token + 1

        def runner():
            try:
                result = work()
            except Exception as exc:  # network hiccups shouldn't kill the GUI
                result = exc
            self.after(0, lambda: token == self._scan_token and done(result))

        threading.Thread(target=runner, daemon=True).start()

    def refresh_animals(self):
        """Populate the animal box from the CSVs already in experiment_log/."""
        existing = sorted(p.stem for p in LOG_DIR.glob("*.csv"))
        self.animal_combo["values"] = existing
        if existing and not self.animal_var.get():
            self.animal_var.set(existing[0])
            self.load_animal()

    def scan_animals_async(self):
        self.status_var.set("Scanning experiment_data roots for animals...")

        def done(result):
            if isinstance(result, Exception):
                self.status_var.set(f"Could not reach experiment_data roots: {result}")
                return
            merged = sorted(set(result) | set(self.animal_combo["values"]))
            self.animal_combo["values"] = merged
            self.status_var.set(f"Found {len(result)} animal folder(s) on disk.")

        self._async(scan_animals, done)

    def scan_session_async(self):
        animal, date = self.animal_var.get().strip(), self.date_var.get().strip()
        if not animal or not date:
            self.status_var.set("Pick an animal and a date first.")
            return
        self.status_var.set(f"Scanning {animal}/{date} ...")

        def done(result):
            if isinstance(result, Exception):
                self.status_var.set(f"Scan failed: {result}")
                return
            self.session = result
            if result["base"] is None:
                self.base_var.set(f"Session folder: {animal}/{date} not found on any root")
                self.status_var.set(
                    f"{animal}/{date} is not on disk - fields stay editable by hand.")
                self.ephys_combo["values"] = []
                self.passive_list.delete(0, "end")
                self.task_list.delete(0, "end")
                return

            self.base_var.set(f"Session folder: {result['base']}")
            self.ephys_combo["values"] = result["ephys"]
            if self.ephys_var.get() not in result["ephys"]:
                self.ephys_var.set(self._default_ephys(result, animal, date))
            self.fill_rec_list()
            self.fill_task_list()
            self.status_var.set(
                f"{result['base'].name}: {len(result['ephys'])} ephys folder(s), "
                f"{len(result['task'])} task .txt file(s).")

        self._async(lambda: scan_session(animal, date), done)

    @staticmethod
    def _default_ephys(session, animal, date):
        """
        Guess the EphysFolder: the conventional <animal>_20<date> name if it is
        there, else the first folder that actually holds .rec recordings (skips
        siblings like video/), else the first folder.
        """
        names = session["ephys"]
        if not names:
            return ""
        conventional = f"{animal}_20{date}"
        if conventional in names:
            return conventional
        with_rec = [n for n in names if session["rec"].get(n)]
        return with_rec[0] if with_rec else names[0]

    # -- list <-> text syncing ---------------------------------------------

    def fill_rec_list(self):
        names = self.session["rec"].get(self.ephys_var.get(), [])
        self._fill_list(self.passive_list, names, self.passive_var.get())

    def fill_task_list(self):
        self._fill_list(self.task_list, self.session["task"], self.task_var.get())

    @staticmethod
    def _fill_list(listbox, names, current):
        listbox.delete(0, "end")
        for name in names:
            listbox.insert("end", name)
        wanted = {v.strip() for v in current.split(";") if v.strip()}
        for i, name in enumerate(names):
            if name in wanted:
                listbox.selection_set(i)

    @staticmethod
    def sync_from_list(listbox, var):
        var.set(";".join(listbox.get(i) for i in listbox.curselection()))

    # -- row editing -------------------------------------------------------

    def load_animal(self):
        animal = self.animal_var.get().strip()
        if not animal:
            return
        if self.dirty and not messagebox.askokcancel(
                "Unsaved changes", "Discard unsaved changes and load another animal?"):
            return
        self.rows, self.sep = read_log(animal)
        self.dirty = False
        self.refresh_tree()
        self.clear_fields()
        path = log_path(animal)
        self.status_var.set(
            f"Loaded {len(self.rows)} row(s) from {path.name}" if path.exists()
            else f"{path.name} does not exist yet - it will be created on save.")
        self._update_title()

        def done(dates):
            if isinstance(dates, Exception):
                return
            logged = {r["date"] for r in self.rows}
            # Dates already logged first, then what is on disk but not yet logged.
            self.date_combo["values"] = sorted(logged | set(dates))
            new = sorted(set(dates) - logged)
            if new:
                self.status_var.set(
                    f"{self.status_var.get()}  |  {len(new)} unlogged date(s) on disk: "
                    + ", ".join(new[:8]) + ("..." if len(new) > 8 else ""))

        self._async(lambda: scan_dates(animal), done)

    def refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for row in sorted(self.rows, key=lambda r: r["date"]):
            self.tree.insert("", "end", values=[row[k] for k in FIELDS])

    def on_row_selected(self, _event=None):
        selection = self.tree.selection()
        if not selection:
            return
        values = self.tree.item(selection[0], "values")
        row = dict(zip(FIELDS, values))
        self.date_var.set(row["date"])
        self.ephys_var.set(row["EphysFolder"])
        self.passive_var.set(row["PassiveFolder"])
        self.task_var.set(row["TaskFile"])
        self.scan_session_async()

    def current_row(self):
        return {
            "date": self.date_var.get().strip(),
            "EphysFolder": self.ephys_var.get().strip(),
            "PassiveFolder": self._clean_multi(self.passive_var.get()),
            "TaskFile": self._clean_multi(self.task_var.get()),
        }

    @staticmethod
    def _clean_multi(value):
        return ";".join(v.strip() for v in value.split(";") if v.strip())

    def add_or_update_row(self):
        animal = self.animal_var.get().strip()
        if not animal:
            messagebox.showerror("No animal", "Pick or type an animal id first.")
            return
        row = self.current_row()
        if not DATE_RE.match(row["date"]):
            messagebox.showerror("Bad date", "date must be 6 digits, e.g. 260722.")
            return
        missing = [k for k in FIELDS if not row[k]]
        if missing and not messagebox.askokcancel(
                "Empty fields", "These are empty:\n  " + "\n  ".join(missing) + "\n\nSave anyway?"):
            return
        bad = self.missing_paths(row)
        if bad and not messagebox.askokcancel(
                "Paths not found",
                "These paths do not exist on disk:\n  " + "\n  ".join(str(p) for p in bad)
                + "\n\nAdd the row anyway?"):
            return

        for i, existing in enumerate(self.rows):
            if existing["date"] == row["date"]:
                self.rows[i] = row
                action = "Updated"
                break
        else:
            self.rows.append(row)
            action = "Added"

        self.dirty = True
        self.refresh_tree()
        self._select_date(row["date"])
        self.status_var.set(f"{action} {animal} {row['date']} (not saved yet - press Save CSV).")
        self._update_title()

    def delete_row(self):
        row = self.current_row()
        match = next((r for r in self.rows if r["date"] == row["date"]), None)
        if match is None:
            self.status_var.set(f"No row with date {row['date']} to delete.")
            return
        if not messagebox.askokcancel("Delete row", f"Remove the row for {row['date']}?"):
            return
        self.rows.remove(match)
        self.dirty = True
        self.refresh_tree()
        self.status_var.set(f"Deleted {row['date']} (not saved yet - press Save CSV).")
        self._update_title()

    def clear_fields(self):
        for var in (self.date_var, self.ephys_var, self.passive_var, self.task_var):
            var.set("")
        self.passive_list.delete(0, "end")
        self.task_list.delete(0, "end")
        self.base_var.set("Session folder: (not scanned)")

    def _select_date(self, date):
        for item in self.tree.get_children():
            if self.tree.item(item, "values")[0] == date:
                self.tree.selection_set(item)
                self.tree.see(item)
                return

    # -- validation / saving -----------------------------------------------

    def missing_paths(self, row):
        """Paths this row points at that are not on disk (empty list if the base is unreachable)."""
        base = self.session["base"]
        if base is None or base.name != row["date"]:
            base = find_session_base(self.animal_var.get().strip(), row["date"])
        if base is None:
            return []
        ephys = base / row["EphysFolder"]
        candidates = [ephys] if row["EphysFolder"] else []
        candidates += [ephys / p for p in row["PassiveFolder"].split(";") if p]
        candidates += [base / t for t in row["TaskFile"].split(";") if t]
        out = []
        for path in candidates:
            try:
                if not path.exists():
                    out.append(path)
            except OSError:
                out.append(path)
        return out

    def verify_paths(self):
        animal = self.animal_var.get().strip()
        if not animal:
            return
        self.status_var.set("Verifying every logged path...")

        def work():
            report = []
            for row in sorted(self.rows, key=lambda r: r["date"]):
                base = find_session_base(animal, row["date"])
                if base is None:
                    report.append((row["date"], [f"session folder {animal}/{row['date']} not found"]))
                    continue
                ephys = base / row["EphysFolder"]
                checks = [ephys] if row["EphysFolder"] else []
                checks += [ephys / p for p in row["PassiveFolder"].split(";") if p.strip()]
                checks += [base / t for t in row["TaskFile"].split(";") if t.strip()]
                bad = []
                for path in checks:
                    try:
                        ok = path.exists()
                    except OSError:
                        ok = False
                    if not ok:
                        bad.append(str(path))
                if bad:
                    report.append((row["date"], bad))
            return report

        def done(report):
            if isinstance(report, Exception):
                self.status_var.set(f"Verify failed: {report}")
                return
            if not report:
                self.status_var.set(f"All {len(self.rows)} row(s) resolve to existing paths.")
                messagebox.showinfo("Verify paths", "Every logged path exists on disk.")
                return
            text = "\n\n".join(f"{date}:\n  " + "\n  ".join(items) for date, items in report)
            self.status_var.set(f"{len(report)} row(s) have missing paths.")
            messagebox.showwarning("Missing paths", text[:3000])

        self._async(work, done)

    def save_log(self):
        animal = self.animal_var.get().strip()
        if not animal:
            messagebox.showerror("No animal", "Pick or type an animal id first.")
            return
        dates = [r["date"] for r in self.rows]
        duplicates = {d for d in dates if dates.count(d) > 1}
        if duplicates:
            messagebox.showerror("Duplicate dates", f"Duplicate date(s): {sorted(duplicates)}")
            return
        try:
            path = write_log(animal, self.rows, self.sep)
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.dirty = False
        self.refresh_animals_values()
        self.status_var.set(f"Saved {len(self.rows)} row(s) to {path}")
        self._update_title()

    def refresh_animals_values(self):
        existing = sorted(p.stem for p in LOG_DIR.glob("*.csv"))
        self.animal_combo["values"] = sorted(set(existing) | set(self.animal_combo["values"]))

    def _update_title(self):
        animal = self.animal_var.get().strip() or "-"
        self.title(f"Grating experiment log - {animal}{' *' if self.dirty else ''}")

    def on_close(self):
        if self.dirty and not messagebox.askokcancel(
                "Unsaved changes", "There are unsaved changes. Quit anyway?"):
            return
        self.destroy()


def main():
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    ExperimentLogGUI().mainloop()


if __name__ == "__main__":
    main()
