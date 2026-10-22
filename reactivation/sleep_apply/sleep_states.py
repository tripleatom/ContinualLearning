"""Sleep-block time base, NREM / channel-MUA ON intervals, and decoder-event state tags.

Joins decoder output to the completed sleep pipeline (sleep/staging/score_sleep_states.py
and sleep/mua_onoff/detect_off_states.py) without re-estimating state from the sorted
units being tested.  Everything is in block-relative seconds; whole-day sample
indices appear only in :class:`SleepBlock` metadata and the MUA ON pickle.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from server_fallback import mirror_on_backup_server, resolve_existing_file

DEFAULT_REGISTRY = REPO_ROOT / "sleep" / "sleep_day_configs.json"

# Which merged-decoder condition (a key of train_task_test_passive's results) is applied to sleep.
# 'merged_neural_cv' = neural-only merged decoder, no kinematic columns.
MERGED_CONDITION = "merged_neural_cv"


def decoder_results_dir(task_pkl, condition=MERGED_CONDITION):
    """The one folder every sleep-decoder script reads from and writes to:

        <session folder>/reactivation/sleep_<condition>_<session>/
            best_merged_decoder_model.pkl
            <block>/sleep_decoding_results.pkl
            template_diagnostics/
    """
    session_dir = Path(task_pkl).parent
    return session_dir / "reactivation" / f"sleep_{condition}_{session_dir.name}"


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SleepBlock:
    """One sleep epoch with an explicit local and whole-day time base."""

    label: str
    sampling_frequency_hz: float
    day_start_sample: int
    day_end_sample: int
    duration_sec: float
    source_path: str = ""
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        if not self.label:
            raise ValueError("SleepBlock.label cannot be empty")
        if not np.isfinite(self.sampling_frequency_hz) or self.sampling_frequency_hz <= 0:
            raise ValueError("sampling_frequency_hz must be positive")
        if self.day_end_sample <= self.day_start_sample:
            raise ValueError("day_end_sample must be greater than day_start_sample")
        expected = (self.day_end_sample - self.day_start_sample) / self.sampling_frequency_hz
        tolerance = max(1.0 / self.sampling_frequency_hz, 1e-6)
        if not np.isclose(self.duration_sec, expected, atol=tolerance, rtol=1e-6):
            raise ValueError(
                f"duration_sec={self.duration_sec:g} disagrees with the sample window "
                f"({expected:g} s)"
            )

    @property
    def local_start_sec(self) -> float:
        return 0.0

    @property
    def local_end_sec(self) -> float:
        return self.duration_sec

    def local_sec_to_day_sample(self, seconds):
        seconds = np.asarray(seconds, dtype=float)
        self.validate_local_seconds(seconds)
        values = self.day_start_sample + np.rint(seconds * self.sampling_frequency_hz)
        return values.astype(np.int64)

    def day_sample_to_local_sec(self, samples):
        samples = np.asarray(samples, dtype=np.int64)
        if np.any(samples < self.day_start_sample) or np.any(samples > self.day_end_sample):
            raise ValueError("whole-day samples fall outside this sleep block")
        return (samples - self.day_start_sample) / self.sampling_frequency_hz

    def validate_local_seconds(self, seconds):
        values = np.asarray(seconds, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("local seconds must be finite")
        tolerance = 1.0 / self.sampling_frequency_hz
        if np.any(values < -tolerance) or np.any(values > self.duration_sec + tolerance):
            raise ValueError(
                f"local seconds must lie in [0, {self.duration_sec:g}]. "
                "Whole-day sample indices must be converted before binning."
            )

    def to_dict(self):
        return asdict(self)


def load_sleep_block(path, label=None):
    """Load the standard sleep-spike pickle and return ``(SleepBlock, data)``."""
    path = Path(path)
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or "spike_data" not in data or "window" not in data:
        raise ValueError(f"{path} is not a standard sleep-spike pickle")

    window = data["window"]
    metadata = data.get("metadata", {})
    start = int(window["sleep_start_sample"])
    end = int(window["sleep_end_sample"])
    fs = float(metadata.get("sampling_frequency", data.get("sampling_frequency", 30000.0)))
    duration = float(window.get("window_duration_sec", (end - start) / fs))
    inferred_label = label or metadata.get("sleep_id") or path.stem
    block = SleepBlock(
        label=str(inferred_label),
        sampling_frequency_hz=fs,
        day_start_sample=start,
        day_end_sample=end,
        duration_sec=duration,
        source_path=str(path),
    )

    for unit, record in data["spike_data"].items():
        if "spike_times_sec" not in record:
            raise ValueError(f"unit {unit!r} has no spike_times_sec")
        block.validate_local_seconds(record["spike_times_sec"])
    return block, data


def normalize_intervals(intervals, *, duration_sec=None, name="intervals"):
    """Validate, sort, clip, and merge half-open intervals in local seconds."""
    array = np.asarray(intervals, dtype=float)
    if array.size == 0:
        return np.zeros((0, 2), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite (n, 2) array")
    if np.any(array[:, 1] <= array[:, 0]):
        raise ValueError(f"{name} contains an interval with end <= start")
    array = array[np.argsort(array[:, 0])].copy()
    if duration_sec is not None:
        array = array[(array[:, 1] > 0) & (array[:, 0] < duration_sec)]
        if not len(array):
            return np.zeros((0, 2), dtype=float)
        array[:, 0] = np.clip(array[:, 0], 0, duration_sec)
        array[:, 1] = np.clip(array[:, 1], 0, duration_sec)
    merged = [array[0].copy()]
    for start, end in array[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append(np.array([start, end]))
    return np.asarray(merged, dtype=float)


def interval_mask(times_sec, intervals):
    """Return whether each time lies in any sorted half-open interval."""
    times = np.asarray(times_sec, dtype=float)
    windows = normalize_intervals(intervals)
    if not len(windows):
        return np.zeros(times.shape, dtype=bool)
    index = np.searchsorted(windows[:, 0], times, side="right") - 1
    valid = index >= 0
    safe_index = np.clip(index, 0, len(windows) - 1)
    return valid & (times < windows[safe_index, 1])


def load_nrem_intervals(path, *, duration_sec=None, require_bouts=False):
    """Load all scored NREM bouts, in seconds relative to the sleep block.

    A block with no scored NREM gives an empty (0, 2) array.  Analyses that only make
    sense inside NREM pass ``require_bouts=True`` to raise instead.
    """
    path = Path(path)
    with path.open("rb") as stream:
        document = pickle.load(stream)
    keys = (
        "bout_intervals_s", "nrem_windows", "nrem_bouts", "bouts",
        "nrem_periods", "consolidated_windows_s", "consolidated_windows",
    )
    rows = next((document[key] for key in keys if document.get(key) is not None), None)
    if rows is None:
        raise KeyError(f"No NREM intervals in {path.name}; keys={list(document)[:15]}")
    intervals = []
    for row in rows:
        if isinstance(row, dict):
            intervals.append((row.get("start_sec", row.get("start")),
                              row.get("end_sec", row.get("end"))))
        else:
            intervals.append((row[0], row[1]))
    windows = normalize_intervals(intervals, duration_sec=duration_sec, name="NREM intervals")
    if require_bouts and not len(windows):
        raise ValueError(f"No scored NREM bouts in {path.name}")
    return windows


def load_mua_on_intervals(path, block: SleepBlock, *, min_shank_fraction=0.5,
                          min_duration_sec=0.05, max_duration_sec=2.0,
                          shanks=None):
    """Combine per-shank channel-MUA ON windows into probe-wide ON (UP) intervals.

    The input is the ``*_global_on_windows_*.pkl`` written by
    ``sleep/mua_onoff/detect_off_states.py``, which scores ON/OFF separately per shank.
    Its whole-day samples are converted to the supplied block's local seconds, then the
    shanks are combined: at each instant the fraction of scored shanks that are ON is
    computed, and an interval is a stretch where that fraction reaches
    ``min_shank_fraction``.  Intervals are kept if ``min_duration_sec`` <= length <=
    ``max_duration_sec``.

    ``shanks`` restricts the combination to a subset.  A single shank gives the LOCAL
    definition: that shank's own merged ON windows, with no requirement that the rest
    of the probe agree.  Shanks do not always share a regime, so demanding simultaneous
    ON across most of them can discard much of the UP time.  The cost of the local
    definition is circularity when the units being analysed sit on the same shank whose
    MUA defines their windows.
    """
    path = Path(path)
    with path.open("rb") as stream:
        document = pickle.load(stream)
    rows = document.get("rows", [])
    if not rows:
        raise ValueError(f"{path.name} contains no MUA ON windows")

    source_fs = float(document.get("sampling_frequency", block.sampling_frequency_hz))
    if not np.isclose(source_fs, block.sampling_frequency_hz, rtol=1e-6):
        raise ValueError(
            f"MUA sampling frequency {source_fs:g} does not match sleep block "
            f"{block.sampling_frequency_hz:g}"
        )
    wanted = None if shanks is None else {int(value) for value in np.atleast_1d(shanks)}
    per_shank = {}
    for row in rows:
        shank = int(row["shank"])
        if wanted is not None and shank not in wanted:
            continue
        samples = np.array([row["on_start_daysample"], row["on_end_daysample"]])
        local = block.day_sample_to_local_sec(samples)
        per_shank.setdefault(shank, []).append(tuple(local))
    if not per_shank:
        raise ValueError(f"No requested shanks found in {path.name}")

    threshold = max(1, int(np.ceil(min_shank_fraction * len(per_shank))))
    edges = []
    for windows in per_shank.values():
        for start, end in normalize_intervals(windows, duration_sec=block.duration_sec):
            edges.extend(((float(start), 1), (float(end), -1)))
    # End edges sort before start edges at identical times, preserving [start, end).
    edges.sort(key=lambda item: (item[0], item[1]))
    count = 0
    run_start = None
    combined = []
    for time, delta in edges:
        before = count >= threshold
        count += delta
        after = count >= threshold
        if after and not before:
            run_start = time
        elif before and not after and run_start is not None:
            combined.append((run_start, time))
            run_start = None
    windows = normalize_intervals(combined, duration_sec=block.duration_sec,
                                  name="MUA ON intervals")
    if len(windows):
        durations = windows[:, 1] - windows[:, 0]
        keep = durations >= min_duration_sec
        if max_duration_sec is not None:
            keep &= durations <= max_duration_sec
        windows = windows[keep]
    metadata = {
        "source": str(path),
        "n_shanks": len(per_shank),
        "required_shanks_on": threshold,
        "min_shank_fraction": float(min_shank_fraction),
    }
    return windows, metadata


def load_day(animal, date, registry_path=DEFAULT_REGISTRY):
    with Path(registry_path).open("r") as stream:
        registry = json.load(stream)
    key = f"{animal}_{date}"
    if key not in registry:
        raise KeyError(f"{key} is not registered in {registry_path}")
    return registry[key]


def state_artifact_candidates(animal, date, block, registry_path=DEFAULT_REGISTRY):
    """Return expected NREM and MUA-ON paths without requiring them to exist."""
    day = load_day(animal, date, registry_path)
    if block not in ("pre", "post"):
        raise ValueError("block must be 'pre' or 'post'")
    rec_folder = Path(day["rec_folder"])
    session = rec_folder.stem.split(".")[0]
    nrem = rec_folder / "low_freq" / "sleep_segmentation" / f"{session}_{block}_sleep_periods.pkl"
    on_dir = rec_folder / "MUA" / "off_states"
    return {
        "nrem": nrem,
        "on_directory": on_dir,
        "on_glob": f"{session}_{block}_global_on_windows_*.pkl",
    }


def resolve_state_artifacts(animal, date, block, registry_path=DEFAULT_REGISTRY):
    candidates = state_artifact_candidates(animal, date, block, registry_path)
    on_directories = [candidates["on_directory"]]
    mirrored = mirror_on_backup_server(candidates["on_directory"])
    if mirrored is not None and mirrored not in on_directories:
        on_directories.insert(0, mirrored)
    matches = []
    for directory in on_directories:
        try:
            matches.extend(directory.glob(candidates["on_glob"]))
        except OSError:
            continue
    try:
        nrem = resolve_existing_file(candidates["nrem"])
    except OSError:
        nrem = candidates["nrem"]
    try:
        nrem_exists = nrem.is_file()
    except OSError:
        nrem_exists = False
    return {
        "nrem": nrem if nrem_exists else None,
        "on": sorted(set(matches))[-1] if matches else None,
        "expected": candidates,
    }


def annotate_decoder_results(results, *, nrem_path, on_path,
                             min_shank_fraction=0.5):
    """Return a copy of decoder results with canonical state annotations."""
    output = deepcopy(results)
    block, _ = load_sleep_block(results["sleep_pkl"], label=results.get("sleep_label"))
    centers = np.asarray(results["sleep_centers_sec"], dtype=float)
    block.validate_local_seconds(centers)
    nrem = load_nrem_intervals(nrem_path, duration_sec=block.duration_sec)
    on, on_metadata = load_mua_on_intervals(
        on_path, block, min_shank_fraction=min_shank_fraction
    )
    is_nrem = interval_mask(centers, nrem)
    is_on = interval_mask(centers, on)
    is_primary = is_nrem & is_on

    event_counts = {"all": 0, "nrem": 0, "nrem_on": 0}
    for class_label, events in output.get("events", {}).items():
        for event in events:
            time_sec = float(event["time_sec"])
            event["is_nrem"] = bool(interval_mask([time_sec], nrem)[0])
            event["is_on"] = bool(interval_mask([time_sec], on)[0])
            event["is_primary_reactivation"] = event["is_nrem"] and event["is_on"]
            event["time_day_sample"] = int(block.local_sec_to_day_sample([time_sec])[0])
            event_counts["all"] += 1
            event_counts["nrem"] += int(event["is_nrem"])
            event_counts["nrem_on"] += int(event["is_primary_reactivation"])

    output["schema_version"] = 1
    output["sleep_block"] = block.to_dict()
    output["state_annotation"] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "nrem_source": str(nrem_path),
        "on_source": str(on_path),
        "on_metadata": on_metadata,
        "nrem_intervals_sec": nrem,
        "on_intervals_sec": on,
        "is_nrem": is_nrem,
        "is_on": is_on,
        "is_nrem_on": is_primary,
        "event_counts": event_counts,
        "primary_definition": "decoder event inside artifact-cleaned scored NREM and MUA ON",
    }
    return output
