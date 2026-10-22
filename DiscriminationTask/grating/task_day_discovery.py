"""Find everything the task spike extraction needs for one animal-day, from disk.

Stdlib only, so task_day_gui.py can call it without the spikeinterface import.

Lab layout this relies on (same as sleep/set_sleep_day.py and
passive_visual/.../extract_sleep_session_spikes_multishank.py):

    experiment_data/<animal>/<YYMMDD>/
        <animal>_<YYYY-MM-DD>_SessionNNN_{Grating,Data}.json   behavior, one per task run
        <ephys folder>/                                        e.g. CnL46_20260728, CnL42SG_20260324
            conversion_list.txt                                .rec order + sample counts of the per-shank NWBs
            <animal>_task_<YYYYMMDD>_<HHMMSS>.rec/<...>.DIO/   photodiode DIO for that run
    sortout/<animal-ish>/<animal-ish>_<YYYYMMDD>/              curated_analyzer, or shank*/sorting_results_*/

One *task* = one behavior JSON with trials. It is paired with the .rec that was
recording when the JSON's startTime happened; that .rec's [start, end) window in
conversion_list.txt is where its spikes live in the sorter's concatenated sample
space. Pairing by time rather than by name handles days with two JSONs in one
.rec (CnL46 260811), three task .recs (CnL46 260812) and misspelled .rec names
("CnL45_taks2_20260830_132901.rec").

The photodiode DIO channel is left as "auto" here - picking it needs the DIO
files read, which extract_task_day.py does.
"""
import json
import re
from datetime import datetime
from pathlib import Path

NEW_SERVER_ROOT = Path("//10.129.151.88/xieluanlabs2/xl_cl")
OLD_SERVER_ROOT = Path("//10.129.151.108/xieluanlabs/xl_cl")
SERVER_ROOTS = (NEW_SERVER_ROOT, OLD_SERVER_ROOT)   # new server first: newest copy wins
LOCAL_FAST_DISK = Path("G:/")                        # days are processed here before archiving

REGISTRY_FILE = Path(__file__).resolve().parent / "task_day_configs.json"

DEFAULT_FS = 30000.0
DEFAULT_PRE_SEC = 2.0     # per-trial window before stim onset (readDIO_task.py's value)
DEFAULT_POST_SEC = 2.0    # per-trial window after stim offset

# A .rec's label is the part between the animal id and the _YYYYMMDD_HHMMSS
# stamp ("task", "task2", "taks2", "presleep", ...). Labels containing one of
# these are never task runs; typos seen so far ("preleep", "post") included.
NON_TASK_LABEL_KEYWORDS = ("passive", "sleep", "preleep", "zebra", "noise", "grating")
NON_TASK_LABELS = ("pre", "post")

# Both conversion_list.txt formats (see sleep/set_sleep_day.py's _CONV_LINE):
#   "   3. CnL46_task_20260728_155226.rec: 108594905 timestamps"
#   "   3. CnL46_task_20260812_171309.rec: 37467937 samples  [nwb 259118554-296586490]"
_CONV_LINE = re.compile(r"^\s*(\d+)\.\s*(\S+\.rec)\s*:\s*(\d+)\s+(?:timestamps|samples)\b", re.I)
_REC_TIME = re.compile(r"_(\d{8})_(\d{6})(?:\.rec)?$")
_JSON_NAME = re.compile(r"^(?P<animal>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<session>Session\d+)_(?P<kind>[A-Za-z]+)\.json$")

# JSON pairs with a .rec that started up to this long AFTER the JSON did
# (clock skew between the task PC and the Trodes PC's file timestamps).
PAIR_SLACK_SEC = 300.0


# ---------------------------------------------------------------- helpers

def full_date(date_str):
    """'260728' -> '20260728'."""
    date_str = date_str.strip()
    return "20" + date_str if len(date_str) == 6 else date_str


def short_date(date_str):
    """'20260728' -> '260728'."""
    date_str = date_str.strip()
    return date_str[2:] if len(date_str) == 8 else date_str


def _child_ci(parent, name):
    """Child of parent named `name` ignoring case (CnL45 vs Cnl45), or None."""
    try:
        for child in Path(parent).iterdir():
            if child.name.lower() == name.lower():
                return child
    except OSError:
        pass
    return None


def _safe_iterdir(folder):
    try:
        return sorted(Path(folder).iterdir())
    except OSError:
        return []


def rec_start_time(rec_name):
    """datetime from '<...>_YYYYMMDD_HHMMSS.rec', or None."""
    m = _REC_TIME.search(Path(rec_name).name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def rec_label(rec_name):
    """'CnL46_task2_20260729_182818.rec' -> 'task2' ('' when there is none)."""
    stem = _REC_TIME.sub("", Path(rec_name).name)
    return stem.split("_", 1)[1].lower() if "_" in stem else ""


def is_task_rec(rec_name):
    label = rec_label(rec_name)
    if not label or label in NON_TASK_LABELS:
        return False
    return not any(k in label for k in NON_TASK_LABEL_KEYWORDS)


# ---------------------------------------------------------------- listing

def list_animals():
    """Animal folders present in experiment_data on either server."""
    names = {}
    for root in SERVER_ROOTS:
        for child in _safe_iterdir(root / "experiment_data"):
            if child.is_dir() and not child.name.startswith((".", "_")):
                names.setdefault(child.name.lower(), child.name)
    return sorted(names.values())


def list_days(animal):
    """YYMMDD day folders for an animal on either server, newest last."""
    days = set()
    for root in SERVER_ROOTS:
        animal_dir = _child_ci(root / "experiment_data", animal)
        if animal_dir is None:
            continue
        for child in _safe_iterdir(animal_dir):
            if child.is_dir() and re.fullmatch(r"\d{6}", child.name):
                days.add(child.name)
    return sorted(days)


def find_day_folder(animal, date6):
    """experiment_data/<animal>/<YYMMDD>, new server first."""
    for root in SERVER_ROOTS:
        animal_dir = _child_ci(root / "experiment_data", animal)
        if animal_dir is not None and (animal_dir / date6).is_dir():
            return animal_dir / date6
    return None


def find_ephys_folder(animal, date6, day_folder):
    """The folder holding the day's .rec runs (+ conversion_list.txt).

    Prefers the server day folder's subfolder; falls back to the local G: copy
    for days not yet archived. Subfolder name varies (CnL42SG_20260324), so any
    subfolder with .rec dirs or a conversion_list.txt counts.
    """
    date8 = full_date(date6)
    candidates = []
    if day_folder is not None:
        for child in _safe_iterdir(day_folder):
            if child.is_dir() and date8 in child.name and child.name.lower() != "video":
                candidates.append(child)
    local = _child_ci(LOCAL_FAST_DISK / animal, f"{animal}_{date8}")
    if local is not None:
        candidates.append(local)
    for folder in candidates:
        if (folder / "conversion_list.txt").is_file():
            return folder
    for folder in candidates:
        if any(c.name.endswith(".rec") for c in _safe_iterdir(folder)):
            return folder
    return candidates[0] if candidates else None


def parse_conversion_list(ephys_folder):
    """[(rec_name, start_sample, end_sample)] in concatenation order, or None."""
    if ephys_folder is None:
        return None
    path = Path(ephys_folder) / "conversion_list.txt"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    rows, cursor = [], 0
    for line in text.splitlines():
        m = _CONV_LINE.match(line)
        if m:
            n = int(m.group(3))
            rows.append((m.group(2), cursor, cursor + n))
            cursor += n
    return rows or None


def list_behavior_jsons(day_folder):
    """Behavior JSONs in the day folder (macOS '._' shadows skipped)."""
    out = []
    for f in _safe_iterdir(day_folder) if day_folder else []:
        if f.is_file() and not f.name.startswith("._") and _JSON_NAME.match(f.name):
            out.append(f)
    return out


def read_json_summary(path):
    """Cheap facts about a behavior JSON: startTime, n_trials, sessionLabel."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        return {"error": str(exc), "start_time": None, "n_trials": 0, "label": None}
    return {
        "start_time": data.get("startTime"),
        "n_trials": len(data.get("trials", []) or []),
        "label": data.get("sessionLabel"),
    }


def find_sortout_folder(animal, date6):
    """sortout/<animal*>/<animal*>_<YYYYMMDD>[_HHMMSS] holding sort data.

    The sortout animal folder may carry a suffix (CnL42 -> CnL42SG), so every
    animal folder starting with the animal id is tried, new server first.
    """
    date8 = full_date(date6)
    fallback = None
    for root in SERVER_ROOTS:
        for animal_dir in _safe_iterdir(root / "sortout"):
            if not (animal_dir.is_dir() and animal_dir.name.lower().startswith(animal.lower())):
                continue
            for session in _safe_iterdir(animal_dir):
                if not session.is_dir() or not session.name.startswith(f"{animal_dir.name}_{date8}"):
                    continue
                if has_sort_data(session):
                    return session
                fallback = fallback or session
    return fallback


def has_sort_data(folder):
    folder = Path(folder)
    if (folder / "curated_analyzer").exists():
        return True
    return any((p / "sorting_analyzer").exists()
               for p in folder.glob("shank*/sorting_results_*"))


# ---------------------------------------------------------------- output names

def output_paths(sortout_folder, json_path, n_tasks_in_day):
    """(trial_pkl, task_pkl) for one task.

    A day with a single task keeps the names readDIO_task.py /
    extract_session_spikes.py already use, so existing readers find them.
    Days with several tasks append the JSON's SessionNNN_Kind (both of CnL45
    260902's JSONs are Session001: one _Data, one _Grating).
    """
    sortout_folder = Path(sortout_folder)
    tag = sortout_folder.name
    if n_tasks_in_day > 1:
        m = _JSON_NAME.match(Path(json_path).name)
        tag += f"_{m.group('session')}_{m.group('kind')}" if m else f"_{Path(json_path).stem}"
    return (sortout_folder / f"task_spikes_trial_{tag}.pkl",
            sortout_folder / f"task_spikes_{tag}.pkl")


def resolved_output(path):
    """Where an output actually lands/lives: the old server is read-only, so
    writes go to the new server mirror (server_fallback.resolve_output_folder)."""
    path = Path(path)
    s = str(path).replace("\\", "/")
    old = str(OLD_SERVER_ROOT).replace("\\", "/")
    if s.lower().startswith(old.lower()):
        return Path(str(NEW_SERVER_ROOT) + s[len(old):])
    return path


# ---------------------------------------------------------------- discovery

def discover_task_day(animal, date6):
    """Build a day config from disk. Never raises for missing pieces: they are
    left None and described in config['notes']."""
    date6 = short_date(date6)
    notes = []
    day_folder = find_day_folder(animal, date6)
    if day_folder is None:
        notes.append(f"no experiment_data/{animal}/{date6} folder on either server")
    ephys_folder = find_ephys_folder(animal, date6, day_folder)
    if ephys_folder is None:
        notes.append("no ephys folder (with .rec runs) found")
    conversion = parse_conversion_list(ephys_folder)
    if ephys_folder is not None and conversion is None:
        notes.append("no conversion_list.txt - task start/end samples must be typed in")
    sortout = find_sortout_folder(animal, date6)
    if sortout is None:
        notes.append("no sortout folder found")
    elif not has_sort_data(sortout):
        notes.append(f"sortout folder has no sort data yet: {sortout}")

    # candidate .rec runs with their sample windows and wall-clock start
    recs = []
    if conversion:
        for rec_name, start, end in conversion:
            recs.append({"rec_name": rec_name, "start": start, "end": end,
                         "t0": rec_start_time(rec_name), "is_task": is_task_rec(rec_name)})
    elif ephys_folder is not None:
        for child in _safe_iterdir(ephys_folder):
            if child.is_dir() and child.name.endswith(".rec"):
                recs.append({"rec_name": child.name, "start": None, "end": None,
                             "t0": rec_start_time(child.name), "is_task": is_task_rec(child.name)})

    json_files = list_behavior_jsons(day_folder)
    summaries = [(p, read_json_summary(p)) for p in json_files]
    with_trials = [(p, s) for p, s in summaries if s["n_trials"] > 0]
    for p, s in summaries:
        if s["n_trials"] == 0:
            notes.append(f"{p.name}: no trials - skipped")

    tasks = []
    for json_path, summary in with_trials:
        rec = _pair_rec(recs, summary["start_time"])
        rec_folder = (ephys_folder / rec["rec_name"]) if (rec and ephys_folder) else None
        if rec is None:
            notes.append(f"{json_path.name}: no .rec was recording at its startTime")
        elif rec_folder is not None and not rec_folder.is_dir():
            notes.append(f"{json_path.name}: paired .rec folder missing on disk: {rec_folder}")
        tasks.append({
            "task_file": str(json_path),
            "session_label": summary["label"],
            "n_json_trials": summary["n_trials"],
            "rec_name": rec["rec_name"] if rec else None,
            "rec_folder": str(rec_folder) if rec_folder else None,
            "task_start": rec["start"] if rec else None,
            "task_end": rec["end"] if rec else None,
            "din_channel": "auto",
        })

    for rec in recs:
        if rec["is_task"] and not any(t["rec_name"] == rec["rec_name"] for t in tasks):
            notes.append(f"{rec['rec_name']}: task-like .rec with no behavior JSON")

    config = {
        "animal": animal,
        "date": date6,
        "day_folder": str(day_folder) if day_folder else None,
        "ephys_folder": str(ephys_folder) if ephys_folder else None,
        "sortout_folder": str(sortout) if sortout else None,
        "pre_sec": DEFAULT_PRE_SEC,
        "post_sec": DEFAULT_POST_SEC,
        "tasks": tasks,
        "notes": notes,
        "discovered_at": datetime.now().isoformat(timespec="seconds"),
    }
    fill_output_paths(config)
    return config


def fill_output_paths(config):
    """(Re)compute each task's output paths from the sortout folder. Only tasks
    paired with a .rec count toward "several tasks today" naming."""
    n = sum(1 for t in config.get("tasks", []) if t.get("rec_name"))
    for task in config.get("tasks", []):
        if config.get("sortout_folder") and task.get("task_file"):
            trial_pkl, task_pkl = output_paths(config["sortout_folder"], task["task_file"], n)
            task["trial_pkl"] = str(resolved_output(trial_pkl))
            task["task_pkl"] = str(resolved_output(task_pkl))
        else:
            task["trial_pkl"] = task["task_pkl"] = None
    return config


def _pair_rec(recs, json_start_unix):
    """The task .rec that was running when the JSON started: the latest task
    .rec that began no later than PAIR_SLACK_SEC after it and (when its length
    is known from conversion_list.txt) had not yet ended."""
    if json_start_unix is None:
        return None
    js = datetime.fromtimestamp(float(json_start_unix))
    running = []
    for r in recs:
        if not r["is_task"] or r["t0"] is None:
            continue
        lead = (js - r["t0"]).total_seconds()     # how long after the .rec started the JSON did
        if lead < -PAIR_SLACK_SEC:
            continue
        if r["start"] is not None and lead > (r["end"] - r["start"]) / DEFAULT_FS + PAIR_SLACK_SEC:
            continue
        running.append(r)
    return max(running, key=lambda r: r["t0"]) if running else None


# ---------------------------------------------------------------- registry

def day_key(animal, date6):
    return f"{animal}_{short_date(date6)}"


def load_registry():
    try:
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_day_config(config):
    registry = load_registry()
    registry[day_key(config["animal"], config["date"])] = config
    tmp = REGISTRY_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(registry, f, indent=2)
    tmp.replace(REGISTRY_FILE)


def get_day_config(animal, date6, discover=True):
    """Registered config if any (user edits win), else a fresh discovery."""
    config = load_registry().get(day_key(animal, date6))
    if config is None and discover:
        config = discover_task_day(animal, date6)
    return config


def day_status(config):
    """Short per-day status strings for the GUI table."""
    tasks = config.get("tasks", [])
    n_trial = sum(1 for t in tasks if t.get("trial_pkl") and Path(t["trial_pkl"]).exists())
    n_task = sum(1 for t in tasks if t.get("task_pkl") and Path(t["task_pkl"]).exists())
    n_ready = sum(1 for t in tasks if t.get("rec_folder") and t.get("task_start") is not None
                  and t.get("task_end") is not None)
    ready = bool(config.get("sortout_folder")) and n_ready > 0
    return {
        "n_tasks": len(tasks),
        "n_ready": n_ready,
        "ready": bool(ready),
        "trial_done": f"{n_trial}/{len(tasks)}" if tasks else "-",
        "task_done": f"{n_task}/{len(tasks)}" if tasks else "-",
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Print the discovered task config for an animal-day.")
    parser.add_argument("animal")
    parser.add_argument("date", nargs="?", help="YYMMDD; omit to summarize every day")
    args = parser.parse_args()
    dates = [args.date] if args.date else list_days(args.animal)
    for d in dates:
        cfg = discover_task_day(args.animal, d)
        if args.date:
            print(json.dumps(cfg, indent=2))
        else:
            st = day_status(cfg)
            recs = ", ".join(f"{Path(t['task_file']).stem.split('_', 2)[-1]}->{t['rec_name']}" for t in cfg["tasks"])
            print(f"{d}: ready={st['ready']!s:5} {recs or '(no tasks)'}"
                  + (f"  | {'; '.join(cfg['notes'])}" if cfg["notes"] else ""))
