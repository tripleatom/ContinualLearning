"""Interactive one-time setup for a new recording day's sleep-pipeline config.

Run this by hand whenever you start working on a new date:

    python set_sleep_day.py

A small window collects the animal and date, the per-day paths (rec_folder,
nwb_session_name) and the per-session fields (sample window, .rec epoch folder
for video/DIO sync, video _PROC file). Every path field has a Browse button:
rec_folder and the .rec epoch are folder pickers (a SpikeGadgets .rec epoch is
a directory), the _PROC tracking file is a file picker. Everything is saved to
sleep_day_configs.json keyed by animal AND date, so two animals recorded on the
same date stay separate and sleep_pipeline_config.py can just look the pair up
(via ACTIVE_ANIMAL / ACTIVE_DATE) without ever prompting itself.

Choosing an animal-day that's already registered loads its current values into
the form for editing, and saving over it asks for confirmation first - it won't
clobber anything by accident.
"""
import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sleep_day_registry import (REGISTRY_FILE, day_key, entry_animal,
                                load_registry, register_day_config,
                                registered_animals, registered_days)

SESSIONS = ("pre", "post")
SESSION_LABELS = {"pre": "pre-task sleep session", "post": "post-task sleep session"}


def clean_path(raw):
    """Strip whitespace plus the quotes Windows' "Copy as path" wraps around."""
    return raw.strip().strip('"').strip("'")


def parse_sample(raw, label):
    """Blank -> None (meaning "recording start/end"), otherwise an int."""
    raw = raw.strip()
    if raw == "" or raw.lower() == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{label} must be a whole number of samples (or blank), got {raw!r}")


def format_entry(entry):
    """Registry entry -> readable block for the overwrite confirmation dialog."""
    lines = [f"  animal: {entry_animal(entry)}",
             f"  rec_folder: {entry.get('rec_folder')}",
             f"  nwb_session_name: {entry.get('nwb_session_name')}"]
    for name in SESSIONS:
        session = entry.get(name, {})
        lines.append(f"  {name}:")
        for key in ("start_sample", "end_sample", "rec_file_folder", "proc_file"):
            lines.append(f"    {key}: {session.get(key)}")
    return "\n".join(lines)


class SleepDayForm:
    """The whole window: one day's registry entry as a fill-in form."""

    def __init__(self, root):
        self.root = root
        self.registry = load_registry()

        self.animal_var = tk.StringVar()
        self.date_var = tk.StringVar()
        self.rec_folder_var = tk.StringVar()
        self.nwb_name_var = tk.StringVar()
        self.session_vars = {
            name: {field: tk.StringVar() for field in
                   ("start_sample", "end_sample", "rec_file_folder", "proc_file")}
            for name in SESSIONS
        }
        self.status_var = tk.StringVar(value=f"Saves to {REGISTRY_FILE}")

        self._build()

    # ---------- layout ----------

    def _build(self):
        self.root.title("Sleep day config")
        self.root.columnconfigure(0, weight=1)

        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._build_day_section(frame, row=0)
        for i, name in enumerate(SESSIONS):
            self._build_session_section(frame, name, row=1 + i)
        self._build_buttons(frame, row=1 + len(SESSIONS))

    def _build_day_section(self, parent, row):
        box = ttk.LabelFrame(parent, text="Recording day", padding=8)
        box.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        box.columnconfigure(1, weight=1)

        picker = ttk.Frame(box)
        picker.grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Label(picker, text="Animal (e.g. CnL42)").grid(row=0, column=0,
                                                           sticky="w", padx=(0, 8))
        self.animal_box = ttk.Combobox(picker, textvariable=self.animal_var,
                                       values=registered_animals(self.registry), width=14)
        self.animal_box.grid(row=0, column=1, sticky="w")
        self.animal_box.bind("<<ComboboxSelected>>", lambda _event: self.animal_changed())

        ttk.Label(picker, text="Date (e.g. 260324)").grid(row=0, column=2,
                                                          sticky="w", padx=(16, 8))
        self.date_box = ttk.Combobox(picker, textvariable=self.date_var,
                                     values=self.dates_for_animal(), width=14)
        self.date_box.grid(row=0, column=3, sticky="w")
        self.date_box.bind("<<ComboboxSelected>>", lambda _event: self.load_date())
        ttk.Button(picker, text="Load", width=9,
                   command=self.load_date).grid(row=0, column=4, padx=(8, 0))

        self._entry_row(box, 1, "rec_folder (day's NWB folder)", self.rec_folder_var,
                        self.browse_rec_folder)
        self._entry_row(box, 2, "nwb_session_name (NWB file prefix)", self.nwb_name_var)

    def _build_session_section(self, parent, name, row):
        box = ttk.LabelFrame(parent, text=SESSION_LABELS[name], padding=8)
        box.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        box.columnconfigure(1, weight=1)
        variables = self.session_vars[name]

        samples = ttk.Frame(box)
        samples.grid(row=0, column=0, columnspan=3, sticky="w", pady=2)
        ttk.Label(samples, text="start_sample").grid(row=0, column=0, padx=(0, 8))
        ttk.Entry(samples, textvariable=variables["start_sample"], width=16).grid(row=0, column=1)
        ttk.Label(samples, text="end_sample").grid(row=0, column=2, padx=(16, 8))
        ttk.Entry(samples, textvariable=variables["end_sample"], width=16).grid(row=0, column=3)
        ttk.Label(samples, text="(both blank = skip this session)",
                  foreground="gray40").grid(row=0, column=4, padx=(16, 0))

        self._entry_row(box, 1, ".rec epoch folder (video/DIO sync)",
                        variables["rec_file_folder"],
                        lambda: self.browse_rec_file_folder(name))
        self._entry_row(box, 2, "video _PROC file", variables["proc_file"],
                        lambda: self.browse_proc_file(name))

    def _entry_row(self, parent, row, label, variable, browse_command=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(parent, textvariable=variable, width=70).grid(row=row, column=1,
                                                                sticky="ew", pady=2)
        if browse_command is not None:
            ttk.Button(parent, text="Browse...", width=9,
                       command=browse_command).grid(row=row, column=2, padx=(6, 0), pady=2)

    def _build_buttons(self, parent, row):
        ttk.Label(parent, textvariable=self.status_var, foreground="gray30",
                  wraplength=760, justify="left").grid(row=row, column=0, sticky="w")
        buttons = ttk.Frame(parent)
        buttons.grid(row=row + 1, column=0, sticky="e", pady=(8, 0))
        ttk.Button(buttons, text="Clear form", command=self.clear).grid(row=0, column=0, padx=4)
        ttk.Button(buttons, text="Close", command=self.root.destroy).grid(row=0, column=1, padx=4)
        ttk.Button(buttons, text="Save", command=self.save).grid(row=0, column=2, padx=4)

    # ---------- browsing ----------

    def _start_dir(self, current, fallback=None):
        """Open the picker where the user is already working, not at C:\\."""
        for candidate in (clean_path(current), fallback):
            if not candidate:
                continue
            path = Path(candidate)
            if path.is_dir():
                return str(path)
            if path.parent.is_dir():
                return str(path.parent)
        return None

    def browse_rec_folder(self):
        chosen = filedialog.askdirectory(
            title="Select the day's NWB folder (rec_folder)",
            initialdir=self._start_dir(self.rec_folder_var.get()))
        if not chosen:
            return
        self.rec_folder_var.set(str(Path(chosen)))
        # The folder name is the NWB prefix on every day so far
        # (.../CnL42SG_20260324 -> CnL42SG_20260324sh0.nwb).
        if not self.nwb_name_var.get().strip():
            self.nwb_name_var.set(Path(chosen).name)

    def browse_rec_file_folder(self, name):
        variable = self.session_vars[name]["rec_file_folder"]
        chosen = filedialog.askdirectory(
            title=f"Select the {name}-sleep .rec epoch folder",
            initialdir=self._start_dir(variable.get(), clean_path(self.rec_folder_var.get())))
        if chosen:
            variable.set(str(Path(chosen)))

    def browse_proc_file(self, name):
        variable = self.session_vars[name]["proc_file"]
        rec_folder = clean_path(self.rec_folder_var.get())
        # Videos + tracking live in the day's sibling "video" folder
        # (same layout sleep_pipeline_config.video_folder assumes).
        video_folder = str(Path(rec_folder).parent / "video") if rec_folder else None
        chosen = filedialog.askopenfilename(
            title=f"Select the {name}-sleep video _PROC file",
            initialdir=self._start_dir(variable.get(), video_folder),
            filetypes=[("PROC tracking files", "*_PROC*"), ("All files", "*.*")])
        if chosen:
            variable.set(str(Path(chosen)))

    # ---------- form <-> registry ----------

    def clear(self):
        for variable in (self.rec_folder_var, self.nwb_name_var):
            variable.set("")
        for variables in self.session_vars.values():
            for variable in variables.values():
                variable.set("")
        self.status_var.set("Form cleared.")

    def dates_for_animal(self, animal=None):
        """Registered dates for one animal (all of them when no animal is set)."""
        animal = (self.animal_var.get().strip() if animal is None else animal)
        return [date for entry_animal_id, date, _ in registered_days(self.registry)
                if not animal or entry_animal_id == animal]

    def animal_changed(self):
        """Narrow the date list to the chosen animal, then load if it matches."""
        self.date_box["values"] = self.dates_for_animal()
        if self.date_var.get().strip():
            self.load_date()

    def lookup(self, animal, date_str):
        """Registry entry for an animal-day, accepting a pre-animal bare-date key."""
        entry = self.registry.get(day_key(animal, date_str))
        if entry is not None:
            return entry
        legacy = self.registry.get(date_str)
        if legacy is not None and (not animal or entry_animal(legacy) == animal):
            return legacy
        return None

    def load_date(self):
        """Pull an already-registered animal-day back into the form for editing."""
        animal = self.animal_var.get().strip()
        date_str = self.date_var.get().strip()
        entry = self.lookup(animal, date_str)
        if entry is None:
            self.clear()
            self.status_var.set(f"'{animal} {date_str}' isn't registered yet - fill in "
                                f"the fields and Save to add it.")
            return

        if not animal:
            self.animal_var.set(entry_animal(entry))
        self.rec_folder_var.set(entry.get("rec_folder") or "")
        self.nwb_name_var.set(entry.get("nwb_session_name") or "")
        for name in SESSIONS:
            session = entry.get(name, {})
            for field, variable in self.session_vars[name].items():
                value = session.get(field)
                variable.set("" if value is None else str(value))
        self.status_var.set(f"Loaded '{self.animal_var.get().strip()} {date_str}' from "
                            f"{REGISTRY_FILE.name}. Saving will overwrite it.")

    def collect(self):
        """Read the form into (animal, date_str, rec_folder, nwb_session_name, pre, post).

        Raises ValueError with a user-facing message if something's missing or
        a sample index isn't a number.
        """
        animal = self.animal_var.get().strip()
        if not animal:
            raise ValueError("Enter the animal id (e.g. CnL42).")
        date_str = self.date_var.get().strip()
        if not date_str:
            raise ValueError("Enter the recording date (e.g. 260324).")

        rec_folder = clean_path(self.rec_folder_var.get())
        if not rec_folder:
            raise ValueError("rec_folder is required.")
        nwb_session_name = clean_path(self.nwb_name_var.get())
        if not nwb_session_name:
            raise ValueError("nwb_session_name is required.")

        sessions = {}
        for name in SESSIONS:
            variables = self.session_vars[name]
            sessions[name] = {
                "start_sample": parse_sample(variables["start_sample"].get(),
                                             f"{name} start_sample"),
                "end_sample": parse_sample(variables["end_sample"].get(),
                                           f"{name} end_sample"),
                "rec_file_folder": clean_path(variables["rec_file_folder"].get()) or None,
                "proc_file": clean_path(variables["proc_file"].get()) or None,
            }
        return (animal, date_str, rec_folder, nwb_session_name,
                sessions["pre"], sessions["post"])

    def save(self):
        try:
            animal, date_str, rec_folder, nwb_session_name, pre, post = self.collect()
        except ValueError as err:
            messagebox.showerror("Can't save yet", str(err), parent=self.root)
            return

        # Re-read: another copy of this GUI (or a hand edit) may have touched
        # the file since we started.
        self.registry = load_registry()
        existing = self.lookup(animal, date_str)
        if existing is not None:
            confirm = messagebox.askyesno(
                "Overwrite?",
                f"'{animal} {date_str}' is already registered in "
                f"{REGISTRY_FILE.name}:\n\n{format_entry(existing)}\n\nOverwrite it?",
                parent=self.root)
            if not confirm:
                self.status_var.set("Left unchanged.")
                return

        register_day_config(date_str, rec_folder, nwb_session_name, pre, post,
                            animal=animal)
        self.registry = load_registry()
        self.animal_box["values"] = registered_animals(self.registry)
        self.date_box["values"] = self.dates_for_animal()
        self.status_var.set(
            f"Saved '{animal} {date_str}' to {REGISTRY_FILE}. Set ACTIVE_ANIMAL = "
            f"\"{animal}\" and ACTIVE_DATE = \"{date_str}\" in "
            f"sleep_pipeline_config.py (or pick them in sleep_pipeline_gui.py) "
            f"to run the pipeline on it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--animal", default=None,
        help="Preselect this animal id (e.g. CnL42).")
    parser.add_argument(
        "--date", default=None,
        help="Preselect this date (e.g. 260324): loads the animal-day for editing "
             "if it is already registered, otherwise just fills the boxes. "
             "sleep_pipeline_gui.py passes the animal/date you were about to run.")
    args = parser.parse_args()

    try:  # crisp text on high-DPI Windows displays; harmless if it fails
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    root = tk.Tk()
    form = SleepDayForm(root)
    if args.animal:
        form.animal_var.set(args.animal.strip())
        form.animal_changed()
    if args.date:
        form.date_var.set(args.date.strip())
        form.load_date()
    root.minsize(860, 520)
    root.mainloop()


if __name__ == "__main__":
    main()
