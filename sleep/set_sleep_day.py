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

Each per-shank NWB is the concatenation, in listed order, of that day's .rec
files, and rec2nwb records that order (with each file's timestamp count) in
conversion_list.txt beside the NWBs. As soon as rec_folder names a folder that
has one, it's parsed and the cumulative offsets of whichever .rec file names
contain "presleep"/"postsleep" are offered as start_sample/end_sample
candidates for that session - "Fill from conversion list" copies them in.
Mirrors MUA/find_sleep_mua.py's epoch_windows, which derives the same sleep
epoch's sample bounds from the same file for a different downstream use.

"Auto-fill from disk" goes further and guesses every path for the animal/date
already typed in, using the lab's standard layout (server_fallback.py):
rec_folder on the local G: fast disk, falling back to the animal's server
experiment_data folder if G: doesn't have it; each session's .rec epoch
folder as the same filename conversion_list.txt names, inside that server
day folder; and each session's video _PROC file by position - the video
recordings share the day's .rec recording order, so the Nth .rec file's
session (by presleep/postsleep) pairs with the Nth front_camera_..._PROC
file. Nothing this can't find on disk is guessed at - those fields are just
left blank.
"""
import argparse
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

SLEEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = SLEEP_DIR.parent
for _path in (str(REPO_ROOT), str(SLEEP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from server_fallback import server_session_folder
from sleep_day_registry import (REGISTRY_FILE, day_key, entry_animal,
                                load_registry, register_day_config,
                                registered_animals, registered_days)

SESSIONS = ("pre", "post")
SESSION_LABELS = {"pre": "pre-task sleep session", "post": "post-task sleep session"}

# The local fast disk this lab processes recordings from before their permanent
# home on the server (server_fallback.py's "LOCAL SESSION FOLDERS" convention).
LOCAL_FAST_DISK = Path("G:/")

# conversion_list.txt lines look like "   1. CnL46_presleep_20260728_141233.rec:
# 123627847 timestamps" (MUA/find_sleep_mua.py's _CONV_LINE). Newer exports (once
# rec2nwb started reporting gap-interpolation) instead say "153711550 samples
# [nwb 124181211-277892760] (153711403 recorded + 147 filled over 147 gap(s))" -
# the count is still first, so both are matched; the trailing detail is ignored
# since re.match only anchors the start of the line.
_CONV_LINE = re.compile(r"^\s*(\d+)\.\s*(\S+\.rec)\s*:\s*(\d+)\s+(?:timestamps|samples)\b",
                        re.I)
# .rec filename substrings identifying each session's recording block.
SESSION_KEYWORDS = (("postsleep", "post"), ("presleep", "pre"))


def parse_conversion_list(rec_folder):
    """[(rec_name, n_timestamps), ...] in concatenation order, or None if the
    file is missing.

    Raises ValueError if the file exists but no line matched _CONV_LINE - most
    likely its format changed again - rather than silently treating that the
    same as "missing".
    """
    path = Path(rec_folder) / "conversion_list.txt"
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    entries = [(m.group(2), int(m.group(3))) for m in
              (_CONV_LINE.match(line) for line in text.splitlines()) if m]
    if not entries:
        raise ValueError(f"{path} exists but no '<n>. <file>.rec: <count> "
                         f"timestamps/samples' line was recognized in it - "
                         f"its format may have changed")
    return entries


def conversion_offsets(entries):
    """[(rec_name, start_sample, end_sample), ...], cumulative in concatenation
    order - the exact sample bounds of each .rec file inside the NWB."""
    offsets, offset = [], 0
    for rec_name, n in entries:
        offsets.append((rec_name, offset, offset + n))
        offset += n
    return offsets


def session_for_rec_name(rec_name):
    """'pre'/'post' this .rec filename's recording block is, or None."""
    low = rec_name.lower()
    for keyword, session in SESSION_KEYWORDS:
        if keyword in low:
            return session
    return None


def conversion_session_windows(entries):
    """{'pre': (start, end), 'post': (start, end)} from parsed conversion_list
    entries, matched by session-recording filename keyword."""
    windows = {}
    for rec_name, start, end in conversion_offsets(entries):
        session = session_for_rec_name(rec_name)
        if session is not None:
            windows[session] = (start, end)
    return windows


def full_date(date_str):
    """'260728' -> '20260728' (YYMMDD -> YYYYMMDD, the NWB/server folder convention)."""
    date_str = date_str.strip()
    return "20" + date_str if len(date_str) == 6 else date_str


def find_case_insensitive(parent, name):
    """Child of `parent` named `name`, matched ignoring case, or None.

    A day's local folder is occasionally spelled differently from the
    convention (server_fallback.py's "Cnl45_20260811" example).
    """
    try:
        for child in Path(parent).iterdir():
            if child.name.lower() == name.lower():
                return child
    except OSError:
        pass
    return None


def find_proc_file(video_folder, animal, date8, index):
    """This session's front_camera_<animal>_<yyyy-mm-dd>_<index>_PROC file in
    video_folder, or None if it's missing or the match isn't unique.

    The video recordings share the day's .rec recording order, so the video
    file at the same position (1-based, matching conversion_list.txt's
    numbering) is this session's - e.g. if conversion_list.txt's 2nd .rec is
    "presleep", the day's *_2_PROC video is the presleep video (checked
    against every currently-registered day, CnL42 and CnL46 both).
    """
    yyyy, mm, dd = date8[:4], date8[4:6], date8[6:8]
    pattern = re.compile(
        rf"^front_camera_{re.escape(animal)}_{yyyy}-{mm}-{dd}_{index}_PROC.*$", re.I)
    try:
        matches = [f for f in Path(video_folder).iterdir() if pattern.match(f.name)]
    except OSError:
        return None
    return matches[0] if len(matches) == 1 else None


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

        # conversion_list.txt state: parsed entries for the current rec_folder,
        # and the pre/post windows derived from them (see conversion_session_windows).
        self.conversion_entries = None
        self.conversion_sessions = {}
        self.conversion_status_var = tk.StringVar(value="")
        self.session_candidate_vars = {name: tk.StringVar() for name in SESSIONS}
        self.session_candidate_buttons = {}

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
        ttk.Button(picker, text="Auto-fill from disk", width=18,
                   command=self.auto_fill_from_disk).grid(row=0, column=5, padx=(8, 0))

        rec_folder_entry = self._entry_row(box, 1, "rec_folder (day's NWB folder)",
                                           self.rec_folder_var, self.browse_rec_folder,
                                           width=115)
        rec_folder_entry.bind("<Return>", lambda _e: self.scan_conversion_list())
        rec_folder_entry.bind("<FocusOut>", lambda _e: self.scan_conversion_list())
        self._entry_row(box, 2, "nwb_session_name (NWB file prefix)", self.nwb_name_var)

        ttk.Label(box, textvariable=self.conversion_status_var, foreground="gray40",
                  wraplength=760, justify="left").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(2, 0))

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

        # A separate row spanning the same columns 0-2 as the entry rows below,
        # rather than a column off to their right (column 5) - that used to
        # reserve dead space on every OTHER row too, since a grid column's
        # width is shared across all rows in the same container regardless of
        # which rows actually put anything there. This way column 1 (the entry
        # column, via box.columnconfigure(1, weight=1) below) is free to
        # stretch across the box's full width on every row.
        candidate_row = ttk.Frame(box)
        candidate_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 4))
        ttk.Label(candidate_row, textvariable=self.session_candidate_vars[name],
                  foreground="gray30").pack(side="left")
        candidate_button = ttk.Button(
            candidate_row, text="Fill from conversion list", width=22,
            command=lambda n=name: self.fill_from_conversion_list(n))
        candidate_button.pack(side="left", padx=(8, 0))
        candidate_button.state(["disabled"])
        self.session_candidate_buttons[name] = candidate_button

        self._entry_row(box, 2, ".rec epoch folder (video/DIO sync)",
                        variables["rec_file_folder"],
                        lambda: self.browse_rec_file_folder(name), width=115)
        self._entry_row(box, 3, "video _PROC file", variables["proc_file"],
                        lambda: self.browse_proc_file(name), width=115)

    def _entry_row(self, parent, row, label, variable, browse_command=None, width=70):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        if browse_command is not None:
            ttk.Button(parent, text="Browse...", width=9,
                       command=browse_command).grid(row=row, column=2, padx=(6, 0), pady=2)
        return entry

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
        self.scan_conversion_list()

    # ---------- conversion_list.txt ----------

    def scan_conversion_list(self):
        """Parse conversion_list.txt in the current rec_folder and refresh the
        per-session start/end sample candidates derived from it."""
        rec_folder = clean_path(self.rec_folder_var.get())
        try:
            entries = parse_conversion_list(rec_folder) if rec_folder else None
            parse_error = None
        except ValueError as exc:
            entries, parse_error = None, exc
        self.conversion_entries = entries
        self.conversion_sessions = conversion_session_windows(entries) if entries else {}

        if not rec_folder:
            self.conversion_status_var.set("")
        elif parse_error is not None:
            self.conversion_status_var.set(str(parse_error))
        elif entries is None:
            self.conversion_status_var.set(
                f"No conversion_list.txt found in {rec_folder}")
        else:
            offsets = conversion_offsets(entries)
            total = offsets[-1][2] if offsets else 0
            listed = "\n".join(f"  {name}: {start}-{end}" for name, start, end in offsets)
            unmatched = [s for s in SESSIONS if s not in self.conversion_sessions]
            note = (f" (no {'/'.join(unmatched)} match)" if unmatched else "")
            self.conversion_status_var.set(
                f"conversion_list.txt: {len(entries)} file(s), {total} timestamps total"
                f"{note}\n{listed}")

        for name in SESSIONS:
            window = self.conversion_sessions.get(name)
            var = self.session_candidate_vars[name]
            button = self.session_candidate_buttons[name]
            if window:
                start, end = window
                var.set(f"candidate: {start}-{end} ({end - start} samples)")
                button.state(["!disabled"])
            else:
                var.set("")
                button.state(["disabled"])

    def fill_from_conversion_list(self, name):
        """Copy the conversion_list.txt candidate window into this session's
        start_sample/end_sample fields, confirming first if they're not blank."""
        window = self.conversion_sessions.get(name)
        if not window:
            return
        start, end = window
        variables = self.session_vars[name]
        old_start, old_end = variables["start_sample"].get().strip(), \
            variables["end_sample"].get().strip()
        if (old_start or old_end) and not messagebox.askyesno(
                "Replace sample window?",
                f"{SESSION_LABELS[name]} already has start_sample={old_start!r}, "
                f"end_sample={old_end!r}.\n\nReplace with the conversion_list.txt "
                f"candidate {start}-{end}?",
                parent=self.root):
            return
        variables["start_sample"].set(str(start))
        variables["end_sample"].set(str(end))
        self.status_var.set(f"Filled {name} start_sample/end_sample from "
                            f"conversion_list.txt: {start}-{end}.")

    # ---------- auto-fill from disk ----------

    def guess_rec_folder(self, animal, date8):
        """(path, source_label) for this animal-day's NWB folder: the local G:
        fast disk first, the server's experiment_data folder otherwise.
        (None, None) if neither is found."""
        wanted = f"{animal}_{date8}"
        local = find_case_insensitive(LOCAL_FAST_DISK / animal, wanted)
        if local is not None and local.is_dir():
            return local, "G:"
        server_folder = server_session_folder(LOCAL_FAST_DISK / animal / wanted)
        if server_folder is not None and server_folder.is_dir():
            return server_folder, "server (no local G: copy)"
        return None, None

    def auto_fill_from_disk(self):
        """Guess every path for the current animal/date from the lab's standard
        layout (see the module docstring). Anything not found on disk is left
        blank rather than guessed, and anything already filled in is left alone."""
        animal = self.animal_var.get().strip()
        date_str = self.date_var.get().strip()
        if not animal or not date_str:
            messagebox.showerror("Auto-fill from disk",
                                 "Enter the animal and date first.", parent=self.root)
            return
        date8 = full_date(date_str)
        notes = []

        rec_folder, source = self.guess_rec_folder(animal, date8)
        if rec_folder is None:
            notes.append(f"no NWB folder found for {animal}_{date8} on G: or the server")
        else:
            self.rec_folder_var.set(str(rec_folder))
            self.nwb_name_var.set(rec_folder.name)
            notes.append(f"rec_folder from {source}")
        self.scan_conversion_list()

        server_day_folder = server_session_folder(LOCAL_FAST_DISK / animal / f"{animal}_{date8}")
        video_folder = (server_day_folder.parent / "video") if server_day_folder else None
        if server_day_folder is None:
            notes.append(f"could not resolve a server folder for {animal} - "
                         ".rec epoch folders and video files left blank")

        entries = self.conversion_entries or []
        if not entries:
            notes.append("no conversion_list.txt - .rec/video paths and sample "
                         "windows could not be matched by session")
        for index, (rec_name, _n) in enumerate(entries, start=1):
            session = session_for_rec_name(rec_name)
            if session is None:
                continue
            variables = self.session_vars[session]

            window = self.conversion_sessions.get(session)
            if window and not variables["start_sample"].get().strip() \
                    and not variables["end_sample"].get().strip():
                variables["start_sample"].set(str(window[0]))
                variables["end_sample"].set(str(window[1]))

            if server_day_folder is not None:
                rec_epoch = server_day_folder / rec_name
                if rec_epoch.is_dir():
                    variables["rec_file_folder"].set(str(rec_epoch))
                else:
                    notes.append(f"{session}: .rec epoch not found on server: {rec_epoch}")

            if video_folder is not None:
                proc = find_proc_file(video_folder, animal, date8, index)
                if proc is not None:
                    variables["proc_file"].set(str(proc))
                else:
                    notes.append(f"{session}: no video _PROC file found (index "
                                 f"{index}) in {video_folder}")

        self.status_var.set(f"Auto-fill from disk for '{animal} {date_str}': "
                            + ("; ".join(notes) if notes else "everything found."))

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
        self.scan_conversion_list()
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
        self.scan_conversion_list()
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
