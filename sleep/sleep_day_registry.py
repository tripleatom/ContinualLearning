"""Per-animal, per-day path registry for the sleep pipeline.

Stores the fields that change every recording day: the NWB rec_folder, and
per pre/post sleep session the sample window plus the epoch-specific .rec
folder (needed for video/DIO sync) and video _PROC file.

Entries are keyed "<animal>_<date>" (e.g. "CnL42_260324") so two animals
recorded on the SAME date cannot overwrite each other. Entries written before
the key carried the animal (bare "260324") are still readable - their animal is
read off the rec_folder path - and are replaced by the composite key the next
time that day is saved.

Populate a new day with set_sleep_day.py (interactive, run manually) or the
Animal/Date fields in sleep_pipeline_gui.py. sleep_pipeline_config.py only ever
calls load_day_config() to look a day up - it never prompts - so it stays safe
to import from the worker processes that extract_sleep_lfp.py spawns for
parallel extraction.
"""
import json
from pathlib import Path

REGISTRY_FILE = Path(__file__).parent / "sleep_day_configs.json"

REQUIRED_SESSION_FIELDS = ("start_sample", "end_sample", "rec_file_folder", "proc_file")


def load_registry():
    if not REGISTRY_FILE.exists():
        return {}
    with open(REGISTRY_FILE, "r") as f:
        return json.load(f)


def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def day_key(animal, date_str):
    """Registry key for one animal's recording day."""
    animal = str(animal or "").strip()
    date_str = str(date_str).strip()
    return f"{animal}_{date_str}" if animal else date_str


def entry_animal(entry):
    """Animal for an entry: the stored field, else read off the rec_folder path.

    Every day is laid out as <...>/experiment_data/<animal>/<date>/<rec folder>,
    so the animal is recoverable for entries saved before it was stored.
    """
    if entry.get("animal"):
        return str(entry["animal"])
    parts = Path(entry.get("rec_folder", "")).parts
    return parts[-3] if len(parts) >= 3 else ""


def entry_date(entry, key=""):
    """Date for an entry: the stored field, else the rec_folder path, else the key."""
    if entry.get("date"):
        return str(entry["date"])
    parts = Path(entry.get("rec_folder", "")).parts
    if len(parts) >= 2:
        return parts[-2]
    return str(key).rsplit("_", 1)[-1]


def registered_days(registry=None):
    """[(animal, date, key)] for every registered day, sorted by animal then date."""
    registry = load_registry() if registry is None else registry
    days = [(entry_animal(entry), entry_date(entry, key), key)
            for key, entry in registry.items()]
    return sorted(days, key=lambda row: (row[0], row[1]))


def registered_animals(registry=None):
    """Animal ids that have at least one registered day."""
    return sorted({animal for animal, _, _ in registered_days(registry) if animal})


def load_day_config(date_str, animal=None):
    """Look up a previously-registered day's config.

    Raises KeyError with setup instructions if this animal/date hasn't been
    registered yet - run set_sleep_day.py to fix that, rather than prompting
    here (this module gets re-imported by spawned worker processes).
    """
    registry = load_registry()

    key = day_key(animal, date_str)
    if key in registry:
        return registry[key]

    # Pre-animal entry for the same day. Only accept it when it really is this
    # animal's - otherwise a date shared by two animals would silently load the
    # wrong one.
    legacy = registry.get(str(date_str))
    if legacy is not None and (animal is None or entry_animal(legacy) == str(animal)):
        return legacy

    known = ", ".join(f"{a}/{d}" for a, d, _ in registered_days(registry)) or "none yet"
    raise KeyError(
        f"No sleep-day config registered for {animal or '<no animal>'} {date_str!r}. "
        f"Run `python set_sleep_day.py` to register it first "
        f"(saved to {REGISTRY_FILE}). Registered: {known}."
    )


def find_entry_key(registry, animal, date_str):
    """Key holding an animal-day, accepting a pre-animal bare-date key, or None."""
    key = day_key(animal, date_str)
    if key in registry:
        return key
    legacy = str(date_str).strip()
    if legacy in registry and (not animal or entry_animal(registry[legacy]) == animal):
        return legacy
    return None


def update_day_paths(animal, date_str, rec_folder=None, nwb_session_name=None):
    """Re-point an existing animal-day's rec_folder / nwb_session_name.

    Only these two fields move when data is copied between servers, so they can
    be corrected without re-entering the sample windows and per-session paths.
    Returns {field: (old, new)} for what actually changed.
    """
    registry = load_registry()
    key = find_entry_key(registry, animal, date_str)
    if key is None:
        raise KeyError(
            f"{animal} {date_str} is not registered in {REGISTRY_FILE.name} - "
            f"add it with set_sleep_day.py before re-pointing its paths.")

    entry = registry[key]
    changed = {}
    for field, value in (("rec_folder", rec_folder),
                         ("nwb_session_name", nwb_session_name)):
        if value is None:
            continue
        value = str(value).strip()
        if value and value != entry.get(field):
            changed[field] = (entry.get(field), value)
            entry[field] = value

    if changed:
        save_registry(registry)
    return changed


def register_day_config(date_str, rec_folder, nwb_session_name, pre, post, animal=None):
    """Save (or overwrite) one animal-day's config.

    pre / post are dicts with keys: start_sample, end_sample,
    rec_file_folder, proc_file (any of the latter two may be None).
    """
    for label, session in (("pre", pre), ("post", post)):
        missing = [k for k in REQUIRED_SESSION_FIELDS if k not in session]
        if missing:
            raise ValueError(f"{label} session config missing fields: {missing}")

    registry = load_registry()
    key = day_key(animal, date_str)
    registry[key] = {
        "animal": str(animal).strip() if animal else entry_animal({"rec_folder": rec_folder}),
        "date": str(date_str).strip(),
        "rec_folder": str(rec_folder),
        "nwb_session_name": nwb_session_name,
        "pre": pre,
        "post": post,
    }

    # Retire this day's pre-animal entry so it is not registered twice.
    legacy_key = str(date_str).strip()
    if (legacy_key != key and legacy_key in registry
            and entry_animal(registry[legacy_key]) == registry[key]["animal"]):
        del registry[legacy_key]

    save_registry(registry)
    return registry[key]
