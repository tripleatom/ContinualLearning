r"""Detect channel-level and global OFF states from an existing MUA-events pickle.

This is a pure downstream consumer: it never reads raw high-frequency traces and
never modifies ``find_sleep_mua.py`` or ``mua_detect.py``. Everything it needs is
already in ``<session>/MUA/<session>_<suffix>_mua_events.pkl``.

Criteria (Vyazovskiy-style OFF-period detection, as specified for this analysis)
-------------------------------------------------------------------------------
1. MUA spikes are detected independently per channel (done upstream).
2. A channel ON run is a maximal spike sequence whose interspike intervals are
   all < 50 ms; it is *valid* when it spans at least 30 ms.
3. A channel OFF interval is a spike-free gap of at least 50 ms ...
4. ... flanked by valid ON activity on both sides.
5. At each instant, count how many channels are simultaneously OFF.
6. A global OFF period is a run where that count reaches a fraction of the
   probe (the paper's 12-of-16 = 75%).
7. Global OFF periods are kept only when they last 50-400 ms.

Channel count is NOT assumed
---------------------------
The paper used 16 channels and a threshold of 12. These probes carry a different
number of channels per shank (32 here), and a shank can lose channels, so the
threshold is stored as a FRACTION (``GLOBAL_OFF_FRACTION``, default 0.75) and
resolved per shank against the channels that actually contribute. Channels whose
firing rate falls below ``MIN_CHANNEL_RATE_HZ`` can never form a valid ON run and
therefore can never be scored OFF, so they are dropped from the denominator
rather than silently making the criterion harder to reach. Pass an absolute
count with ``--global-min-channels`` to override the fraction.

Paths
-----
Session folders are discovered rather than hardcoded, because a day commonly
exists in more than one place (a local working copy plus the lab share, which
itself spans two servers). Candidates are tried in order:

  1. ``--session-folder`` if given
  2. ``sleep_pipeline_config.rec_folder`` (the registered ACTIVE_ANIMAL/DATE)
  3. ``find_sleep_mua.SESSION_FOLDER``
  4. the ``server_fallback`` mirror of each of the above

The first candidate that actually holds the requested MUA pickle wins, and each
input (MUA pkl, LFP npz, NREM pkl) is resolved independently -- so reading MUA
from the share while reading LFP from the local copy is fine and expected.

Usage
-----
    python detect_off_states.py --list            # what is discoverable
    python detect_off_states.py                   # every shank, pre + post
    python detect_off_states.py --shanks 5 --epochs post
    python detect_off_states.py --shanks 5 --start-sec 1200 --window-sec 4
    python detect_off_states.py --session-folder D:\somewhere\CnL46_20260727
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

_HERE = Path(__file__).resolve().parent          # .../sleep/MUA
_SLEEP = _HERE.parent                            # .../sleep
_ROOT = _SLEEP.parent                            # repo root (server_fallback.py)
for _p in (str(_HERE), str(_SLEEP), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# =====================================================
# PAPER CRITERIA
# =====================================================
DEFAULT_ISI_SEC = 0.050          # ON run: every ISI must be shorter than this
DEFAULT_MIN_ON_SEC = 0.030       # a valid ON run spans at least this
DEFAULT_MIN_OFF_SEC = 0.050      # channel OFF: spike-free for at least this
DEFAULT_GLOBAL_MIN_SEC = 0.050   # global OFF duration window, inclusive
DEFAULT_GLOBAL_MAX_SEC = 0.400

#: Fraction of contributing channels that must be simultaneously OFF. The paper's
#: 12-of-16 is 0.75; expressed as a fraction it transfers to any channel count.
GLOBAL_OFF_FRACTION = 0.75

#: A channel firing below this (Hz) over the epoch cannot form valid ON runs, so
#: it never registers as OFF. Counting it in the denominator would quietly raise
#: the bar for every global OFF period, so it is excluded and reported instead.
MIN_CHANNEL_RATE_HZ = 0.05

#: Comparison slack, seconds. Spike times are float64 seconds produced upstream
#: as ``sample_index / sampling_frequency``, so an interval that is exactly at a
#: threshold can land either side of it: 1500 samples at 30 kHz is "50 ms", but
#: differencing the two quotients can give 0.049999999999999996. That would make
#: the ON/OFF classification of a boundary interval depend on rounding. This
#: slack is far above float64 noise at epoch timescales (~2e-12 s at t = 5000 s)
#: and far below one acquisition sample (33 us), so it only ever resolves exact
#: ties, always in the direction the criteria specify.
COMPARE_TOL_SEC = 1e-9

#: How the flanking-ON criterion treats an isolated spike inside a long silence.
#:   'bridge' - the silence stays ONE OFF period spanning the isolated spike.
#:              This follows the criterion's stated purpose ("prevents isolated
#:              spikes from artificially splitting a long silent period into
#:              multiple OFF periods"): the intended result is one period, not
#:              none. The reported interval then contains that stray spike.
#:   'strict' - a gap is kept only when the groups immediately either side are
#:              both valid ON runs. Literal about "contains no spikes", but a
#:              single stray spike deletes the whole silence.
DEFAULT_FLANK_MODE = "bridge"


# =====================================================
# PATH RESOLUTION
# =====================================================

def _load_pipeline_config():
    """Import sleep_pipeline_config, or None if this day is not registered."""
    try:
        import sleep_pipeline_config as cfg
        return cfg
    except Exception as exc:                     # unregistered day, bad share, ...
        print(f"  note: sleep_pipeline_config unavailable ({exc})")
        return None


def _load_find_sleep_mua():
    """Import find_sleep_mua for its SESSION_FOLDER / NWB_BASE, or None."""
    try:
        import find_sleep_mua as fsm
        return fsm
    except Exception as exc:
        print(f"  note: find_sleep_mua unavailable ({exc})")
        return None


def _mirror(path):
    """server_fallback's old-server -> new-server mapping, or None."""
    try:
        from server_fallback import mirror_on_backup_server
        return mirror_on_backup_server(path)
    except Exception:
        return None


def candidate_session_folders(explicit=None):
    """Ordered, de-duplicated session folders to search for this day's files."""
    candidates = []

    def add(folder):
        if folder is None:
            return
        folder = Path(folder)
        if folder not in candidates:
            candidates.append(folder)
        mirrored = _mirror(folder)
        if mirrored is not None and mirrored not in candidates:
            candidates.append(mirrored)

    add(explicit)
    cfg = _load_pipeline_config()
    if cfg is not None:
        add(cfg.rec_folder)
    fsm = _load_find_sleep_mua()
    if fsm is not None:
        add(fsm.SESSION_FOLDER)
    return candidates


def candidate_session_names(explicit=None):
    """Ordered base names used in output filenames (``<name>_<suffix>_...``)."""
    names = []

    def add(name):
        if name and name not in names:
            names.append(name)

    add(explicit)
    cfg = _load_pipeline_config()
    if cfg is not None:
        add(cfg.session_name)
        add(cfg.nwb_session_name)
    fsm = _load_find_sleep_mua()
    if fsm is not None:
        add(fsm.NWB_BASE)
    return names


def _first_existing(paths):
    for path in paths:
        if path.is_file():
            return path
    return None


def resolve_mua_pkl(suffix, folders, names):
    """Locate ``<name>_<suffix>_mua_events.pkl`` across candidate folders."""
    return _first_existing([
        folder / "MUA" / f"{name}_{suffix}_mua_events.pkl"
        for folder in folders for name in names
    ])


def resolve_lfp_npz(suffix, shank, folders, names):
    """Locate ``low_freq/<name>_<suffix>_sh<N>_lfp_traces.npz``, or None."""
    return _first_existing([
        folder / "low_freq" / f"{name}_{suffix}_sh{shank}_lfp_traces.npz"
        for folder in folders for name in names
    ])


def resolve_sleep_periods(suffix, folders, names):
    """Locate score_nrem_epochs.py's sleep-periods pkl, or None.

    Both the suffixed and unsuffixed names are tried: the NREM stage currently
    writes ``<name>_sleep_periods.pkl`` without a pre/post suffix.
    """
    return _first_existing([
        folder / "low_freq" / candidate
        for folder in folders for name in names
        for candidate in (f"{name}_{suffix}_sleep_periods.pkl",
                          f"{name}_sleep_periods.pkl")
    ])


def resolve_output_dir(mua_pkl):
    """``<session>/MUA/off_states/``, redirected by server_fallback when needed."""
    target = mua_pkl.parent / "off_states"
    try:
        from server_fallback import resolve_output_folder
        return Path(resolve_output_folder(target))
    except Exception:
        target.mkdir(parents=True, exist_ok=True)
        return target


# =====================================================
# LOADING
# =====================================================

def _as_python_scalar(value):
    return value.item() if hasattr(value, "item") else value


def _lookup_channel(mapping, channel_id):
    """Look up a channel ID robustly across Python/NumPy scalar types."""
    if channel_id in mapping:
        return mapping[channel_id]
    wanted = str(_as_python_scalar(channel_id))
    for key, value in mapping.items():
        if str(_as_python_scalar(key)) == wanted:
            return value
    raise KeyError(f"Channel {channel_id!r} is absent")


def load_epoch_pkl(path):
    with Path(path).open("rb") as file:
        return pickle.load(file)


def merge_shank(data, shank):
    """Flatten one shank plus the epoch-level metadata into a single dict."""
    shanks = data.get("shanks", {})
    if shank not in shanks:
        raise KeyError(
            f"Shank {shank} absent; present: {sorted(shanks)}"
        )
    merged = {key: value for key, value in data.items() if key != "shanks"}
    merged.update(shanks[shank])
    return merged


def depth_sorted_channels(mua):
    """Channel IDs ordered superficial -> deep.

    ``channel_locations`` is stored in acquisition order, which is NOT depth
    order on these probes, so the raster rows would otherwise be scrambled.
    """
    channel_ids = [_as_python_scalar(value) for value in mua["channel_ids"]]
    locations = np.asarray(mua.get("channel_locations", []))
    if locations.ndim == 2 and locations.shape[0] == len(channel_ids):
        depth_column = 1 if locations.shape[1] > 1 else 0
        order = np.argsort(locations[:, depth_column], kind="stable")
        channel_ids = [channel_ids[index] for index in order]
        depths = locations[order, depth_column].astype(float)
    else:
        depths = np.full(len(channel_ids), np.nan)
    return channel_ids, depths


# =====================================================
# NREM RESTRICTION (optional)
# =====================================================

def nrem_windows_from_pkl(path, mua):
    """NREM bouts as (start, end) seconds relative to the MUA epoch start.

    The NREM stage works on the LFP file's own clock, which starts at
    ``sleep_start_sample``; MUA times start at ``epoch_start_sample``. Both are
    whole-day sample indices at the acquisition rate, so the difference converts
    one to the other.
    """
    with Path(path).open("rb") as file:
        periods = pickle.load(file)

    bouts = periods.get("bout_intervals_s")
    if not bouts:
        return np.empty((0, 2), dtype=np.float64), periods

    lfp_start = periods.get("sleep_start_sample")
    if lfp_start is None:
        # Not recorded by the NREM stage; assume the two clocks already agree.
        offset_sec = 0.0
    else:
        original_fs = float(mua.get("sampling_frequency", 30000.0))
        offset_sec = (int(lfp_start) - int(mua.get("epoch_start_sample", 0))) / original_fs

    windows = np.asarray(bouts, dtype=np.float64).reshape(-1, 2) + offset_sec
    duration = float(mua["duration_sec"])
    windows[:, 0] = np.clip(windows[:, 0], 0.0, duration)
    windows[:, 1] = np.clip(windows[:, 1], 0.0, duration)
    return windows[windows[:, 1] > windows[:, 0]], periods


def clip_intervals(intervals, windows):
    """Intersect ``intervals`` with ``windows``; both are (N, 2) second arrays."""
    intervals = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    if windows is None or len(windows) == 0 or len(intervals) == 0:
        return intervals
    kept = []
    for start, end in intervals:
        for win_start, win_end in windows:
            left = max(start, win_start)
            right = min(end, win_end)
            if right > left:
                kept.append((left, right))
    return np.asarray(kept, dtype=np.float64).reshape(-1, 2)


def mask_spikes(spike_times, windows):
    """Keep only spikes falling inside ``windows``."""
    spikes = np.asarray(spike_times, dtype=np.float64)
    if windows is None or len(windows) == 0:
        return spikes
    keep = np.zeros(spikes.shape, dtype=bool)
    for win_start, win_end in windows:
        keep |= (spikes >= win_start) & (spikes <= win_end)
    return spikes[keep]


# =====================================================
# CHANNEL-LEVEL ON / OFF
# =====================================================

def channel_on_off(spike_times, duration_sec, *, isi_sec=DEFAULT_ISI_SEC,
                   min_on_sec=DEFAULT_MIN_ON_SEC, min_off_sec=DEFAULT_MIN_OFF_SEC,
                   flank_mode=DEFAULT_FLANK_MODE):
    """Return ``(on_runs, off_intervals)`` for one channel, in seconds.

    Spikes are grouped into maximal runs at every ISI of ``isi_sec`` or more, so
    a run's internal ISIs are all strictly shorter -- the paper's "< 50 ms". A
    run is valid when it holds at least two spikes spanning ``min_on_sec``.

    See ``DEFAULT_FLANK_MODE`` for what ``flank_mode`` changes.
    """
    spikes = np.asarray(spike_times, dtype=np.float64)
    spikes = np.unique(spikes[np.isfinite(spikes)])
    spikes = spikes[(spikes >= 0.0) & (spikes <= duration_sec)]
    empty = np.empty((0, 2), dtype=np.float64)
    if spikes.size < 2:
        return empty, empty.copy()

    split = np.flatnonzero(np.diff(spikes) >= isi_sec - COMPARE_TOL_SEC) + 1
    groups = np.split(spikes, split)
    is_valid = np.asarray([
        group.size >= 2 and (group[-1] - group[0]) >= min_on_sec - COMPARE_TOL_SEC
        for group in groups
    ])
    on_runs = np.asarray(
        [(group[0], group[-1]) for group, valid in zip(groups, is_valid) if valid],
        dtype=np.float64,
    ).reshape(-1, 2)

    off = []
    if flank_mode == "strict":
        for index in range(len(groups) - 1):
            if not (is_valid[index] and is_valid[index + 1]):
                continue
            start = float(groups[index][-1])
            end = float(groups[index + 1][0])
            if end - start >= min_off_sec - COMPARE_TOL_SEC:
                off.append((start, end))
    elif flank_mode == "bridge":
        valid_index = np.flatnonzero(is_valid)
        for left, right in zip(valid_index[:-1], valid_index[1:]):
            start = float(groups[left][-1])
            end = float(groups[right][0])
            if end - start >= min_off_sec - COMPARE_TOL_SEC:
                off.append((start, end))
    else:
        raise ValueError(f"flank_mode must be 'bridge' or 'strict', got {flank_mode!r}")

    return on_runs, np.asarray(off, dtype=np.float64).reshape(-1, 2)


# =====================================================
# POPULATION OFF
# =====================================================

def simultaneous_off_intervals(channel_intervals, min_channels):
    """Exact intervals where at least ``min_channels`` channels are OFF.

    The OFF count only changes at a channel-OFF boundary, so sweeping those
    boundaries is exact and needs no time bin. Returns
    ``(segments, edges, counts)`` where ``counts[i]`` is the number of channels
    OFF throughout ``[edges[i], edges[i + 1])``.
    """
    deltas = {}
    for intervals in channel_intervals.values():
        for start, end in np.asarray(intervals, dtype=np.float64).reshape(-1, 2):
            if end <= start:
                continue
            deltas[float(start)] = deltas.get(float(start), 0) + 1
            deltas[float(end)] = deltas.get(float(end), 0) - 1

    edges = np.asarray(sorted(deltas), dtype=np.float64)
    if edges.size < 2:
        return (np.empty((0, 2), dtype=np.float64), edges,
                np.zeros(max(edges.size - 1, 0), dtype=int))

    count = 0
    counts = []
    segments = []
    for index in range(edges.size - 1):
        count += deltas[float(edges[index])]
        counts.append(count)
        start, end = edges[index], edges[index + 1]
        if count >= min_channels and end > start:
            # Exact equality is right here: both values are the same element of
            # `edges`. np.isclose would use a RELATIVE tolerance, which at an
            # epoch time of thousands of seconds is tens of milliseconds and
            # would fuse genuinely separate OFF periods.
            if segments and segments[-1][1] == start:
                segments[-1][1] = end
            else:
                segments.append([start, end])

    return (np.asarray(segments, dtype=np.float64).reshape(-1, 2),
            edges, np.asarray(counts, dtype=int))


def detect_off_states(mua, *, isi_sec=DEFAULT_ISI_SEC,
                      min_on_sec=DEFAULT_MIN_ON_SEC,
                      min_channel_off_sec=DEFAULT_MIN_OFF_SEC,
                      global_fraction=GLOBAL_OFF_FRACTION,
                      global_min_channels=None,
                      global_min_sec=DEFAULT_GLOBAL_MIN_SEC,
                      global_max_sec=DEFAULT_GLOBAL_MAX_SEC,
                      min_channel_rate_hz=MIN_CHANNEL_RATE_HZ,
                      flank_mode=DEFAULT_FLANK_MODE,
                      nrem_windows=None, verbose=True):
    """Apply the channel and population OFF criteria to one shank."""
    channel_ids, depths = depth_sorted_channels(mua)
    duration = float(mua["duration_sec"])
    spike_map = mua["channel_spike_times"]

    if nrem_windows is not None and len(nrem_windows):
        analysed_sec = float(np.sum(nrem_windows[:, 1] - nrem_windows[:, 0]))
    else:
        analysed_sec = duration

    channel_on, channel_off = {}, {}
    channel_rate, channel_off_fraction = {}, {}
    contributing = []
    for channel_id in channel_ids:
        spikes = mask_spikes(_lookup_channel(spike_map, channel_id), nrem_windows)
        rate = len(spikes) / analysed_sec if analysed_sec > 0 else 0.0
        channel_rate[channel_id] = float(rate)

        on, off = channel_on_off(
            spikes, duration, isi_sec=isi_sec, min_on_sec=min_on_sec,
            min_off_sec=min_channel_off_sec, flank_mode=flank_mode,
        )
        off = clip_intervals(off, nrem_windows)
        on = clip_intervals(on, nrem_windows)
        channel_on[channel_id] = on
        channel_off[channel_id] = off
        channel_off_fraction[channel_id] = (
            float(np.sum(off[:, 1] - off[:, 0]) / analysed_sec) if len(off) and analysed_sec > 0 else 0.0
        )
        if rate >= min_channel_rate_hz and len(off):
            contributing.append(channel_id)

    n_contributing = len(contributing)
    if global_min_channels is None:
        min_channels = int(math.ceil(global_fraction * n_contributing))
    else:
        min_channels = int(global_min_channels)
    min_channels = max(min_channels, 1)

    if verbose:
        excluded = [c for c in channel_ids if c not in contributing]
        print(f"    channels: {len(channel_ids)} total, {n_contributing} contributing"
              + (f", excluded {excluded}" if excluded else ""))
        print(f"    global OFF threshold: >= {min_channels} channels"
              + (f" ({global_fraction:.0%} of contributing)"
                 if global_min_channels is None else " (explicit)"))

    counted = {cid: channel_off[cid] for cid in contributing}
    candidates, edges, counts = simultaneous_off_intervals(counted, min_channels)

    if len(candidates):
        durations = candidates[:, 1] - candidates[:, 0]
        keep = ((durations >= global_min_sec - COMPARE_TOL_SEC)
                & (durations <= global_max_sec + COMPARE_TOL_SEC))
        global_off = candidates[keep]
    else:
        global_off = candidates

    off_durations = (global_off[:, 1] - global_off[:, 0]) if len(global_off) else np.array([])

    return {
        "session": mua.get("session"),
        "epoch": mua.get("epoch"),
        "suffix": mua.get("suffix"),
        "shank": int(mua["shank"]),
        "duration_sec": duration,
        "analysed_sec": analysed_sec,
        "sampling_frequency": float(mua.get("sampling_frequency", np.nan)),
        "epoch_start_sample": int(mua.get("epoch_start_sample", 0)),
        "epoch_end_sample": int(mua.get("epoch_end_sample", 0)),
        "channel_ids": np.asarray(channel_ids),
        "channel_depths_um": depths,
        "channel_rate_hz": channel_rate,
        "channel_off_fraction": channel_off_fraction,
        "contributing_channels": np.asarray(contributing),
        "n_channels": len(channel_ids),
        "n_contributing_channels": n_contributing,
        "global_min_channels": min_channels,
        "channel_on_intervals": channel_on,
        "channel_off_intervals": channel_off,
        "global_off_candidates": candidates,
        "global_off_intervals": global_off,
        "off_count_edges": edges,          # counts[i] holds on [edges[i], edges[i+1])
        "off_counts": counts,
        "nrem_windows_sec": (np.asarray(nrem_windows).reshape(-1, 2)
                             if nrem_windows is not None else None),
        "summary": {
            "n_channel_off": int(sum(len(v) for v in channel_off.values())),
            "n_global_candidates": int(len(candidates)),
            "n_global_off": int(len(global_off)),
            "global_off_per_min": float(len(global_off) / (analysed_sec / 60.0))
                                  if analysed_sec > 0 else float("nan"),
            "mean_off_ms": float(np.mean(off_durations) * 1000) if off_durations.size else float("nan"),
            "median_off_ms": float(np.median(off_durations) * 1000) if off_durations.size else float("nan"),
            "off_time_fraction": float(np.sum(off_durations) / analysed_sec)
                                 if off_durations.size and analysed_sec > 0 else 0.0,
        },
        "params": {
            "isi_sec": float(isi_sec),
            "min_on_sec": float(min_on_sec),
            "min_channel_off_sec": float(min_channel_off_sec),
            "global_fraction": None if global_min_channels is not None else float(global_fraction),
            "global_min_channels": int(min_channels),
            "global_min_sec": float(global_min_sec),
            "global_max_sec": float(global_max_sec),
            "min_channel_rate_hz": float(min_channel_rate_hz),
            "flank_mode": flank_mode,
            "nrem_restricted": bool(nrem_windows is not None and len(nrem_windows)),
        },
        "mua_params": mua.get("params", {}),
    }


# =====================================================
# LFP FOR THE FIGURE
# =====================================================

def load_lfp_window(path, mua, start_sec, window_sec, channel_id=None, depth_um=None):
    """Load one LFP channel over a window expressed in MUA-epoch time.

    The LFP file starts at ``sleep_start_sample`` and the MUA epoch at
    ``epoch_start_sample``; both are whole-day indices at ``original_fs``, so
    their difference aligns the two clocks.
    """
    with np.load(path, allow_pickle=False) as data:
        traces = data["traces"]
        fs = float(data["sampling_rate"])
        lfp_ids = [_as_python_scalar(value) for value in data["channel_ids"]]
        ycoord = np.asarray(data["ycoord"], dtype=float) if "ycoord" in data else None
        lfp_start = int(data["sleep_start_sample"]) if "sleep_start_sample" in data else None
        original_fs = float(data["original_fs"]) if "original_fs" in data else None

        if channel_id is not None:
            matches = [i for i, cid in enumerate(lfp_ids) if str(cid) == str(channel_id)]
            if not matches:
                raise KeyError(f"LFP channel {channel_id!r} not present; have {lfp_ids}")
            index = matches[0]
        elif ycoord is not None and ycoord.size == len(lfp_ids):
            target = float(depth_um) if depth_um is not None else float(np.median(ycoord))
            index = int(np.argmin(np.abs(ycoord - target)))
        else:
            index = len(lfp_ids) // 2

        offset_sec = 0.0
        if lfp_start is not None and original_fs:
            offset_sec = (lfp_start - int(mua.get("epoch_start_sample", 0))) / original_fs

        first = int(math.floor((start_sec - offset_sec) * fs))
        last = int(math.ceil((start_sec + window_sec - offset_sec) * fs))
        clipped = first < 0 or last > traces.shape[0]
        first = max(0, first)
        last = min(traces.shape[0], last)
        if first >= last:
            raise ValueError(
                f"MUA-relative window {start_sec:g}-{start_sec + window_sec:g} s "
                f"does not overlap the LFP file"
            )
        trace = np.asarray(traces[first:last, index], dtype=np.float64)

    time = np.arange(first, last, dtype=np.float64) / fs + offset_sec
    depth = float(ycoord[index]) if ycoord is not None and ycoord.size == len(lfp_ids) else float("nan")
    return time, trace, lfp_ids[index], depth, clipped


# =====================================================
# FIGURE
# =====================================================

ORANGE = "#E68624"
PURPLE = "#6F2DA8"


def stamp_figure(figure, text):
    """Embed a reproducibility line (what made this figure, from what, when).

    Wrapped before drawing: matplotlib does not wrap ``figure.text``, and with
    ``bbox_inches="tight"`` a single long line silently widens the saved canvas
    to fit it, which is what stretched earlier versions of this figure.
    """
    body = f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by detect_off_states.py  |  {text}"
    figure.text(0.005, 0.005, textwrap.fill(body, width=175),
                fontsize=5, color="0.4", ha="left", va="top", family="monospace")


def plot_example(result, mua, *, start_sec, window_sec, output_path, mua_pkl,
                 lfp_path=None, lfp_channel=None, lfp_depth_um=None,
                 lfp_scalebar_uv=1000.0, dpi=300):
    """Paper-style figure: LFP trace above a per-channel MUA raster."""
    end_sec = start_sec + window_sec
    channel_ids = [_as_python_scalar(c) for c in result["channel_ids"]]
    n_channels = len(channel_ids)

    figure = plt.figure(figsize=(10.0, 6.4))
    grid = figure.add_gridspec(2, 1, height_ratios=(1.0, 3.2), hspace=0.05)
    ax_lfp = figure.add_subplot(grid[0])
    ax_raster = figure.add_subplot(grid[1], sharex=ax_lfp)

    lfp_note = "no LFP"
    if lfp_path is not None:
        try:
            lfp_t, lfp, used_channel, used_depth, clipped = load_lfp_window(
                lfp_path, mua, start_sec, window_sec, lfp_channel, lfp_depth_um)
            ax_lfp.plot(lfp_t - start_sec, lfp, color="black", linewidth=0.55)
            finite = lfp[np.isfinite(lfp)]
            center = float(np.median(finite)) if finite.size else 0.0
            # Bar and its label sit just inside the axes, with the label to the
            # RIGHT of the bar so neither collides with the y-axis text.
            bar_x = 0.015 * window_sec
            ax_lfp.plot([bar_x, bar_x], [center, center + lfp_scalebar_uv],
                        color="black", linewidth=1.8, clip_on=False)
            ax_lfp.text(bar_x + 0.012 * window_sec, center + lfp_scalebar_uv / 2,
                        f"{lfp_scalebar_uv / 1000:g} mV",
                        ha="left", va="center", fontsize=7)
            ax_lfp.set_ylabel(f"LFP  ch {used_channel}\n{used_depth:.0f} um",
                              rotation=0, ha="right", va="center", fontsize=8,
                              labelpad=12)
            lfp_note = f"lfp=ch{used_channel}@{used_depth:.0f}um" + (" CLIPPED" if clipped else "")
        except (KeyError, ValueError, OSError) as exc:
            ax_lfp.text(0.5, 0.5, f"LFP unavailable: {exc}", transform=ax_lfp.transAxes,
                        ha="center", va="center", color="0.4", fontsize=8)
    else:
        ax_lfp.text(0.5, 0.5, "LFP not supplied", transform=ax_lfp.transAxes,
                    ha="center", va="center", color="0.4")
    ax_lfp.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax_lfp.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

    spike_map = mua["channel_spike_times"]
    contributing = {_as_python_scalar(c) for c in result["contributing_channels"]}
    for row, channel_id in enumerate(channel_ids):
        for off_start, off_end in result["channel_off_intervals"][channel_id]:
            left, right = max(off_start, start_sec), min(off_end, end_sec)
            if right > left:
                ax_raster.add_patch(Rectangle(
                    (left - start_sec, row - 0.38), right - left, 0.76,
                    facecolor=ORANGE, edgecolor="none", alpha=0.45, zorder=1))
        spikes = np.asarray(_lookup_channel(spike_map, channel_id), dtype=float)
        selected = spikes[(spikes >= start_sec) & (spikes <= end_sec)] - start_sec
        ax_raster.scatter(selected, np.full(selected.shape, row), s=3.5,
                          marker="o", color="black", linewidths=0, zorder=3)
        if channel_id not in contributing:
            ax_raster.text(-0.004 * window_sec, row, "x", color="0.6", fontsize=6,
                           ha="right", va="center")

    for off_start, off_end in result["global_off_intervals"]:
        left, right = max(off_start, start_sec), min(off_end, end_sec)
        if right > left:
            ax_raster.add_patch(Rectangle(
                (left - start_sec, -0.55), right - left, n_channels - 0.9,
                fill=False, edgecolor=PURPLE, linewidth=1.6, zorder=5))

    ax_raster.set_xlim(0.0, window_sec)
    ax_raster.set_ylim(-0.8, n_channels - 0.2)
    ax_raster.invert_yaxis()
    ax_raster.set_xticks(np.arange(0, math.floor(window_sec) + 1, 1.0))
    ax_raster.set_xlabel("Time (seconds)")
    ax_raster.set_ylabel(f"MUA channels (n={n_channels}, superficial -> deep)")
    ax_raster.set_yticks([])
    ax_raster.spines[["top", "right", "left"]].set_visible(False)

    params = result["params"]
    summary = result["summary"]
    n_shown = int(np.sum((result["global_off_intervals"][:, 0] < end_sec)
                         & (result["global_off_intervals"][:, 1] > start_sec))
                  if len(result["global_off_intervals"]) else 0)
    figure.suptitle(
        f"{result['session']} {result['epoch']} shank {result['shank']}  |  "
        f"{start_sec:.2f}-{end_sec:.2f} s  |  global OFF >= "
        f"{result['global_min_channels']}/{result['n_contributing_channels']} ch, "
        f"{n_shown} shown",
        fontsize=10)
    figure.text(0.5, 0.925,
                "orange = channel OFF   |   purple = global OFF (50-400 ms)",
                ha="center", fontsize=8, color="0.35")

    stamp_figure(figure, (
        f"src={mua_pkl}  |  {lfp_note}  |  "
        f"isi={params['isi_sec'] * 1000:g}ms min_on={params['min_on_sec'] * 1000:g}ms "
        f"min_off={params['min_channel_off_sec'] * 1000:g}ms flank={params['flank_mode']}  |  "
        f"global={params['global_min_channels']}ch "
        f"({'explicit' if params['global_fraction'] is None else format(params['global_fraction'], '.2f')}) "
        f"{params['global_min_sec'] * 1000:g}-{params['global_max_sec'] * 1000:g}ms  |  "
        f"nrem_restricted={params['nrem_restricted']} analysed={result['analysed_sec']:.0f}s  |  "
        f"detect_channel_radius={result['mua_params'].get('detect_channel_radius')} "
        f"threshold={result['mua_params'].get('detect_threshold')} "
        f"scale={result['mua_params'].get('scale_mode')}  |  "
        f"n_global_off={summary['n_global_off']} ({summary['global_off_per_min']:.2f}/min)  |  "
        f"reproduce: python detect_off_states.py --shanks {result['shank']} "
        f"--epochs {result['suffix']} --start-sec {start_sec:g} --window-sec {window_sec:g}"
    ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


# =====================================================
# CLI
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mua_pkl", nargs="?", type=Path, default=None,
                        help="explicit *_mua_events.pkl (default: discover from the "
                             "registered session)")
    parser.add_argument("--session-folder", type=Path, default=None,
                        help="search this session folder before the registered ones")
    parser.add_argument("--session-name", default=None,
                        help="base name for input/output files (default: from config)")
    parser.add_argument("--epochs", nargs="+", default=None,
                        help="epoch suffixes to process (default: pre post)")
    parser.add_argument("--shanks", type=int, nargs="+", default=None,
                        help="shanks to process (default: every shank in the pkl)")
    parser.add_argument("--list", action="store_true",
                        help="report what is discoverable and exit")

    parser.add_argument("--isi-ms", type=float, default=DEFAULT_ISI_SEC * 1000)
    parser.add_argument("--min-on-ms", type=float, default=DEFAULT_MIN_ON_SEC * 1000)
    parser.add_argument("--min-channel-off-ms", type=float, default=DEFAULT_MIN_OFF_SEC * 1000)
    parser.add_argument("--global-fraction", type=float, default=GLOBAL_OFF_FRACTION,
                        help="fraction of contributing channels that must be OFF "
                             f"(default {GLOBAL_OFF_FRACTION}, the paper's 12/16)")
    parser.add_argument("--global-min-channels", type=int, default=None,
                        help="absolute channel count, overriding --global-fraction")
    parser.add_argument("--global-min-ms", type=float, default=DEFAULT_GLOBAL_MIN_SEC * 1000)
    parser.add_argument("--global-max-ms", type=float, default=DEFAULT_GLOBAL_MAX_SEC * 1000)
    parser.add_argument("--min-channel-rate-hz", type=float, default=MIN_CHANNEL_RATE_HZ)
    parser.add_argument("--flank-mode", choices=("bridge", "strict"),
                        default=DEFAULT_FLANK_MODE)

    parser.add_argument("--nrem-only", action="store_true",
                        help="restrict to NREM bouts from the sleep-periods pkl")
    parser.add_argument("--restrict-sec", type=float, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="restrict to this window (seconds from the epoch start). "
                             "Use when NREM scoring has not been run but you know "
                             "which stretch is sleep; overrides --nrem-only.")
    parser.add_argument("--start-sec", type=float, default=None,
                        help="figure start, seconds from the MUA epoch start "
                             "(default: the densest global-OFF stretch)")
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--lfp-channel", default=None)
    parser.add_argument("--lfp-depth-um", type=float, default=None)
    parser.add_argument("--lfp-scalebar-uv", type=float, default=1000.0,
                        help="scale-bar height in microvolts (traces are uV)")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def pick_example_window(result, window_sec):
    """Start time (s) of the window holding the most global OFF periods."""
    intervals = result["global_off_intervals"]
    if not len(intervals):
        windows = result.get("nrem_windows_sec")
        if windows is not None and len(windows):
            return float(windows[0][0])
        return 0.0
    starts = intervals[:, 0]
    # Each candidate window begins one OFF period earlier than it ends, so the
    # best start is always aligned to some event's onset.
    counts = [(np.sum((starts >= s) & (starts < s + window_sec)), s) for s in starts]
    best = max(counts, key=lambda item: (item[0], -item[1]))
    return float(max(0.0, best[1] - 0.15 * window_sec))


def main():
    args = parse_args()

    folders = candidate_session_folders(args.session_folder)
    names = candidate_session_names(args.session_name)
    epochs = args.epochs or ["pre", "post"]

    print("=" * 72)
    print("OFF-state detection from MUA events")
    print(f"  session folders searched : {[str(f) for f in folders]}")
    print(f"  session names tried      : {names}")
    print(f"  epochs                   : {epochs}")

    if args.mua_pkl is not None:
        located = {args.mua_pkl.stem.split("_mua_events")[0].split("_")[-1]: args.mua_pkl}
    else:
        located = {}
        for suffix in epochs:
            found = resolve_mua_pkl(suffix, folders, names)
            if found is None:
                print(f"  {suffix:>4}: no MUA pkl found")
            else:
                located[suffix] = found
                print(f"  {suffix:>4}: {found}")

    if args.list:
        for suffix, path in located.items():
            data = load_epoch_pkl(path)
            print(f"\n[{suffix}] {path}")
            print(f"  session={data.get('session')} epoch={data.get('epoch')} "
                  f"duration={data.get('duration_sec', float('nan')):.1f}s "
                  f"partial={data.get('partial')}")
            print(f"  detection params: {data.get('params')}")
            for shank in sorted(data.get("shanks", {})):
                entry = data["shanks"][shank]
                print(f"    shank {shank}: {len(entry['channel_ids'])} channels, "
                      f"{entry['population_rate_hz']:.1f} Hz population rate")
            lfp = resolve_lfp_npz(suffix, sorted(data.get("shanks", {}))[0], folders, names)
            print(f"  example LFP: {lfp}")
            print(f"  sleep periods: {resolve_sleep_periods(suffix, folders, names)}")
        return

    if not located:
        raise FileNotFoundError(
            "No MUA pkl found. Check ACTIVE_ANIMAL/ACTIVE_DATE in "
            "sleep_pipeline_config.py, or pass --session-folder / an explicit path."
        )

    summary_rows = []
    for suffix, mua_pkl in located.items():
        print("\n" + "#" * 72)
        print(f"EPOCH {suffix}: {mua_pkl}")
        data = load_epoch_pkl(mua_pkl)
        available = sorted(data.get("shanks", {}))
        shanks = args.shanks if args.shanks is not None else available
        missing = [s for s in shanks if s not in available]
        if missing:
            print(f"  WARNING: shanks {missing} are not in this pkl (present: {available})")
        shanks = [s for s in shanks if s in available]

        radius = (data.get("params") or {}).get("detect_channel_radius")
        if radius:
            print(f"  WARNING: this MUA was detected with detect_channel_radius="
                  f"{radius}, so each spike was kept on only ONE channel. The paper "
                  f"thresholds every channel independently; cross-channel "
                  f"suppression makes neighbouring channels artificially silent "
                  f"together. Re-detect with detect_channel_radius=0.0 for a "
                  f"faithful reproduction.")

        nrem_windows = None
        if args.restrict_sec is not None:
            nrem_windows = np.asarray([args.restrict_sec], dtype=np.float64)
            print(f"  restricted to {nrem_windows[0][0]:.1f}-{nrem_windows[0][1]:.1f} s "
                  f"(--restrict-sec)")
        elif args.nrem_only:
            nrem_pkl = resolve_sleep_periods(suffix, folders, names)
            if nrem_pkl is None:
                print("  --nrem-only requested but no *_sleep_periods.pkl found; "
                      "run score_nrem_epochs.py first. Analysing the whole epoch.")
            else:
                probe = merge_shank(data, shanks[0])
                nrem_windows, _ = nrem_windows_from_pkl(nrem_pkl, probe)
                total = float(np.sum(nrem_windows[:, 1] - nrem_windows[:, 0]))
                print(f"  NREM restriction: {len(nrem_windows)} bouts, {total:.0f} s "
                      f"from {nrem_pkl.name}")
        else:
            print("  NOTE: analysing the WHOLE epoch. This block contains wake as "
                  "well as sleep; pass --nrem-only once score_nrem_epochs.py has run.")

        out_dir = resolve_output_dir(mua_pkl)
        for shank in shanks:
            print(f"\n  --- shank {shank} ---")
            mua = merge_shank(data, shank)
            result = detect_off_states(
                mua,
                isi_sec=args.isi_ms / 1000.0,
                min_on_sec=args.min_on_ms / 1000.0,
                min_channel_off_sec=args.min_channel_off_ms / 1000.0,
                global_fraction=args.global_fraction,
                global_min_channels=args.global_min_channels,
                global_min_sec=args.global_min_ms / 1000.0,
                global_max_sec=args.global_max_ms / 1000.0,
                min_channel_rate_hz=args.min_channel_rate_hz,
                flank_mode=args.flank_mode,
                nrem_windows=nrem_windows,
            )
            result["source_mua_pkl"] = str(mua_pkl)

            info = result["summary"]
            print(f"    channel OFF intervals : {info['n_channel_off']}")
            print(f"    global candidates     : {info['n_global_candidates']}")
            print(f"    accepted global OFF   : {info['n_global_off']} "
                  f"({info['global_off_per_min']:.2f}/min, "
                  f"median {info['median_off_ms']:.0f} ms, "
                  f"{info['off_time_fraction']:.1%} of analysed time)")

            tag = (f"f{int(round(args.global_fraction * 100))}"
                   if args.global_min_channels is None else f"n{args.global_min_channels}")
            tag += f"_{args.flank_mode}"
            if args.restrict_sec is not None:
                tag += f"_win{args.restrict_sec[0]:.0f}-{args.restrict_sec[1]:.0f}"
            elif result["params"]["nrem_restricted"]:
                tag += "_nrem"
            stem = f"{result['session']}_{suffix}_sh{shank}_off_states_{tag}"

            result_path = out_dir / f"{stem}.pkl"
            with result_path.open("wb") as file:
                pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"    saved: {result_path.name}")

            if not args.no_plot:
                start_sec = (args.start_sec if args.start_sec is not None
                             else pick_example_window(result, args.window_sec))
                start_sec = float(np.clip(start_sec, 0.0,
                                          max(0.0, result["duration_sec"] - args.window_sec)))
                lfp_path = resolve_lfp_npz(suffix, shank, folders, names)
                figure_path = out_dir / f"{stem}_{start_sec:.0f}s.png"
                plot_example(
                    result, mua, start_sec=start_sec, window_sec=args.window_sec,
                    output_path=figure_path, mua_pkl=mua_pkl, lfp_path=lfp_path,
                    lfp_channel=args.lfp_channel, lfp_depth_um=args.lfp_depth_um,
                    lfp_scalebar_uv=args.lfp_scalebar_uv, dpi=args.dpi,
                )
                print(f"    figure: {figure_path.name}"
                      + ("" if lfp_path else "  (no LFP found for this shank)"))

            summary_rows.append({
                "epoch": suffix, "shank": shank,
                "n_channels": result["n_channels"],
                "n_contributing": result["n_contributing_channels"],
                "global_min_channels": result["global_min_channels"],
                "analysed_sec": result["analysed_sec"],
                **info,
            })

    if summary_rows:
        out_dir = resolve_output_dir(next(iter(located.values())))
        summary_path = out_dir / "off_states_summary.json"
        summary_path.write_text(json.dumps(summary_rows, indent=2))
        print("\n" + "=" * 72)
        print(f"{'epoch':>6} {'shank':>6} {'ch':>4} {'contrib':>8} {'thresh':>7} "
              f"{'nOFF':>6} {'per min':>8} {'median ms':>10}")
        for row in summary_rows:
            print(f"{row['epoch']:>6} {row['shank']:>6} {row['n_channels']:>4} "
                  f"{row['n_contributing']:>8} {row['global_min_channels']:>7} "
                  f"{row['n_global_off']:>6} {row['global_off_per_min']:>8.2f} "
                  f"{row['median_off_ms']:>10.0f}")
        print(f"\nSummary: {summary_path}")
        print("=" * 72)


if __name__ == "__main__":
    main()
