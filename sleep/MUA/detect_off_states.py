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
8. Supplementary rule, independent of 2-4: any stretch with literally zero
   spikes on EVERY analysed channel -- no flanking-ON requirement -- is also
   accepted once it is 50-400 ms. This recovers population silences that the
   flanking requirement drops, e.g. when the spike group bordering a gap is
   too short to count as a valid ON run even though the whole probe was
   silent. The final global_off_intervals is the union of the fraction-based
   (5-7) and zero-spike-based (8) periods; each is tagged in
   ``global_off_source`` so the two can be told apart.
9. A global ON period is the complement: a stretch strictly BETWEEN two
   global OFF periods (fewer than the threshold fraction of channels OFF
   throughout), kept when it lasts 50-2000 ms. Because only the gap between
   two OFF periods is considered -- never the stretch before the first or
   after the last -- a global ON period is by construction always flanked by
   a global OFF period on both sides, the same way a channel OFF interval
   must be flanked by valid ON runs (criterion 4).

Global ON windows, whole-day sample index
------------------------------------------
Per-shank results carry ``global_on_intervals`` (epoch-relative seconds, for
reuse against that same epoch's own MUA/LFP files) and
``global_on_intervals_daysample`` (whole-day acquisition sample index --
``epoch_start_sample + round(epoch_sec * sampling_frequency)`` -- for reuse
against the day's raw/NWB data once the epoch pkl's own relative clock is no
longer at hand). Per epoch (pre/post), every shank's ON windows are also
collected into ``<session>_<suffix>_global_on_windows_<tag>.pkl`` under
``MUA/off_states/`` (a dict with a ``rows`` list, one row per ON window) for
easy pickup by downstream analysis.

Channel count is NOT assumed
---------------------------
The paper used 16 channels and a threshold of 12. These probes carry a different
number of channels per shank (32 here), and a shank can lose channels, so the
threshold is stored as a FRACTION (``GLOBAL_OFF_FRACTION``, default 0.75) and
resolved per shank against the channels that actually contribute. Channels
firing below ``MIN_CHANNEL_RATE_HZ`` (1 Hz over the analysed window) are dropped
from the criterion entirely -- they are neither counted as OFF nor counted in
the denominator, since a channel too quiet to form a valid ON run can never be
scored OFF but would still make the criterion harder to reach. Pass an absolute
count with ``--global-min-channels`` to override the fraction, or
``--min-channel-rate-hz`` to move the cutoff.

Bad channels
------------
Channels listed in the session's ``bad_channels.txt`` are dropped before any
criterion is applied -- not scored, not counted OFF, not in the denominator.
That file numbers channels in the SpikeGadgets space (one 0-255 index across the
whole probe), while the MUA pickles use per-shank NWB numbering (0-31), so the
ids are converted through the probe's channel map. Which map applies follows
from the animal -- one probe per animal, recorded in ``device_types.json`` --
and the map is then verified against the geometry stored in the pickle before
anything is excluded; see the BAD CHANNELS section below. Pass
``--no-exclude-bad`` to keep them.

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
DEFAULT_GLOBAL_ON_MIN_SEC = 0.050    # global ON duration window, inclusive
DEFAULT_GLOBAL_ON_MAX_SEC = 2.000

#: Fraction of contributing channels that must be simultaneously OFF. Expressed
#: as a fraction so it transfers to any channel count: the paper's 12-of-16 is
#: 0.75, which on a 32-channel shank means 24.
#:
#: Raising it tightens the criterion fast: 0.85 means 28 of 32, i.e. near-total
#: probe silence. Override per run with --global-fraction / --global-min-channels
#: rather than editing this, so the value used stays recorded in the figure
#: stamp and the output filename.
GLOBAL_OFF_FRACTION = 0.75

#: Channels firing below this (Hz) over the ANALYSED time are dropped from the
#: global OFF criterion entirely -- neither counted as OFF nor counted in the
#: denominator. Two separate reasons to exclude them:
#:
#: 1. A channel too quiet to sustain a valid ON run can never be scored OFF, so
#:    leaving it in the denominator quietly raises the bar for every global OFF
#:    period without it ever being able to contribute.
#: 2. A near-silent channel is OFF by the criterion almost all of the time, so
#:    including it would manufacture global OFF periods out of a dead site.
#:
#: Raised from 0.05 to 1.0 Hz so that a channel must fire at least once per
#: second on average to have a vote. Note this rate is measured over the
#: analysed window, so under --nrem-only it is the NREM rate, not the
#: whole-epoch rate, that decides. The value is recorded in the figure stamp and
#: the output filename; override per run with --min-channel-rate-hz.
MIN_CHANNEL_RATE_HZ = 1.0

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
#:   'strict' - a gap is kept only when the groups immediately either side are
#:              both valid ON runs. The reported interval then contains no
#:              spikes at all, apart from the two that define its endpoints.
#:              A single stray spike deletes the whole silence.
#:   'bridge' - the silence stays ONE OFF period spanning any groups in between
#:              that failed the valid-ON test. Follows the criterion's stated
#:              purpose ("prevents isolated spikes from artificially splitting a
#:              long silent period into multiple OFF periods"), but the reported
#:              interval then contains those spikes.
#:
#: 'strict' is the default because 'bridge' does not survive contact with these
#: data. Measured on CnL46 260727 post shank 5 inside NREM, 'bridge' put at
#: least one spike inside 57% of all OFF intervals (287728 spikes in total) and
#: stretched one channel's median OFF period from 107 ms to 293 ms with a
#: maximum of 5.6 s -- covering ~7x more time than 'strict'. The reason is that
#: at ~13 Hz per channel a large share of spike groups fail the valid-ON test,
#: so 'bridge' merges across real activity rather than across the occasional
#: isolated spike it was meant for. The paper's primary definition ("an interval
#: containing no detected MUA spikes for at least 50 ms") is what 'strict'
#: implements, so that is what is applied unless asked otherwise.
DEFAULT_FLANK_MODE = "strict"


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


def resolve_bad_channels_file(folders, explicit=None):
    """Locate ``bad_channels.txt`` at a session-folder root, or None.

    Unlike the other inputs this file carries no session-name prefix -- it sits
    beside the .rec files as ``bad_channels.txt``.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"--bad-channels-file {path} does not exist")
        return path
    return _first_existing([folder / "bad_channels.txt" for folder in folders])


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


def _sorted_channels(channel_ids):
    """Channel ids in numeric order, falling back to string order.

    Ids arrive as numpy scalars or bare numeric strings depending on how the
    recording was read, and plain sorting would put ch10 before ch2.
    """
    def key(value):
        text = str(_as_python_scalar(value))
        return (0, float(text), "") if text.lstrip("-").isdigit() else (1, 0.0, text)

    return sorted(channel_ids, key=key)


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
# BAD CHANNELS (SpikeGadgets ids -> per-shank NWB ids)
# =====================================================
# ``bad_channels.txt`` is written by the SpikeSorting repo's screen_bad_ch.py,
# which numbers channels in the SpikeGadgets space: a single 0-255 index across
# the whole 8-shank probe. The MUA pickles inherit the per-shank NWB numbering
# instead, 0-31 within each shank. The two are related only through the probe's
# channel map -- and NOT by any simple offset, because a shank's SpikeGadgets
# ids are interleaved with its neighbours' (shank 0 spans 23-63, shank 1 spans
# 0-38 on this probe).
#
# The map lives in the SpikeSorting repo as
# ``rec2nwb/mapping/<device_type>.csv`` with columns
# ``spikegadget,xcoord,ycoord,sh``. Row order within a shank IS the NWB channel
# order, so the i-th row with ``sh == S`` is NWB channel i on shank S.
#
# Which device type applies is settled by the animal: one probe per animal, and
# ``rec2nwb/device_types.json`` records the pairing (CnL46 -> 8shank32). The
# animal id comes off the session name in the pickle, so nothing has to be
# passed by hand for a registered animal.
#
# The row-order claim is the whole conversion, so it is verified rather than
# trusted: before any channel is excluded, the map's per-shank (x, y) sequence
# is compared against the ``channel_locations`` stored in the pickle. A mismatch
# raises instead of silently excluding the wrong sites -- which also catches a
# stale or wrong device_types.json entry.

#: Last-resort probe maps, tried in order when the animal is not registered in
#: device_types.json. The first whose geometry matches the pickle wins.
CHANNEL_MAP_CANDIDATES = ("8shank32", "4shank32", "4shank16", "1shank128")

MAP_COLUMNS = ("spikegadget", "xcoord", "ycoord", "sh")


def spikesorting_repo():
    """The SpikeSorting repo root, or None."""
    try:
        from mua_detect import find_spikesorting_repo
    except ImportError:
        return None
    repo = find_spikesorting_repo()
    return Path(repo) if repo is not None else None


def mapping_folder():
    """``<SpikeSorting>/rec2nwb/mapping``, or None if the repo is not found."""
    repo = spikesorting_repo()
    if repo is None:
        return None
    folder = repo / "rec2nwb" / "mapping"
    return folder if folder.is_dir() else None


def animal_id_candidates(session_name):
    """Animal ids to try for ``CnL46_20260727``: the full name, then ``CnL46``.

    Progressive rather than a plain ``split('_')[0]`` because device_types.json
    registers per-implant ids as well (``CnL46_1``, ``CnL45_2``), and those must
    win over the bare animal when a session is named after one.

    Each prefix is also tried with an ``SG`` suffix, and first: animals recorded
    on both rigs are registered twice (``CnL22`` is the Intan probe,
    ``CnL22SG`` the SpikeGadgets one), and everything this script reads is
    SpikeGadgets-derived. A wrong guess is caught by the geometry check.
    """
    parts = str(session_name).split("_")
    candidates = []
    for n in range(len(parts), 0, -1):
        prefix = "_".join(parts[:n])
        candidates += [f"{prefix}SG", prefix]
    return candidates


def device_type_for_session(session_name):
    """Device type registered for this animal, or None.

    Reads ``rec2nwb/device_types.json`` directly rather than calling
    ``get_or_set_device_type``: that helper pops up a chooser when the animal is
    missing, which would hang a batch run.
    """
    repo = spikesorting_repo()
    if repo is None:
        return None, "SpikeSorting repo not found"
    path = repo / "rec2nwb" / "device_types.json"
    if not path.is_file():
        return None, f"{path} not found"
    try:
        registry = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"could not read {path.name} ({exc})"

    for animal in animal_id_candidates(session_name):
        if animal in registry:
            return registry[animal], f"{animal} -> {registry[animal]} ({path.name})"
    return None, (f"no entry for {session_name!r} in {path.name} (tried "
                  f"{animal_id_candidates(session_name)}); register the animal "
                  f"there or pass --channel-map")


def load_channel_map(name_or_path):
    """Read a channel-map CSV by device-type name or explicit path."""
    import pandas as pd

    path = Path(name_or_path)
    if not path.is_file():
        folder = mapping_folder()
        if folder is None:
            raise FileNotFoundError(
                f"Cannot resolve channel map {name_or_path!r}: the SpikeSorting "
                f"repo was not found. Pass --channel-map with a full path, or set "
                f"the SPIKESORTING_REPO environment variable."
            )
        path = folder / f"{name_or_path}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"No channel map at {path}")

    table = pd.read_csv(path)
    missing = [c for c in MAP_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}; a channel map "
                         f"needs {list(MAP_COLUMNS)}")
    return table, path


def map_matches_shank(table, shank, mua):
    """True when the map's per-shank geometry reproduces the pickle's exactly."""
    locations = np.asarray(mua.get("channel_locations", []), dtype=float)
    if locations.ndim != 2 or locations.shape[1] < 2:
        return False
    sub = table[table["sh"] == shank]
    if len(sub) != locations.shape[0]:
        return False
    mapped = sub[["xcoord", "ycoord"]].to_numpy(dtype=float)
    return bool(np.allclose(mapped, locations[:, :2]))


def resolve_channel_map(data, shanks, explicit=None, verbose=True):
    """Pick the channel map for this recording and verify it against the pickle.

    Resolution order:

    1. ``explicit`` (``--channel-map``), a device-type name or a full path
    2. the device type registered for this animal in ``device_types.json`` --
       one probe per animal, so the animal id settles it
    3. :data:`CHANNEL_MAP_CANDIDATES` by geometry, for an unregistered animal

    Whichever route is taken, the map's geometry must reproduce the pickle's
    ``channel_locations`` exactly or a ValueError is raised: an unverified map
    would quietly exclude the wrong sites.
    """
    probes = [(shank, merge_shank(data, shank)) for shank in shanks]

    if explicit is not None:
        candidates, source = [explicit], "--channel-map"
    else:
        device_type, note = device_type_for_session(data.get("session"))
        if device_type is not None:
            candidates, source = [device_type], f"animal registry: {note}"
        else:
            candidates, source = list(CHANNEL_MAP_CANDIDATES), f"geometry scan ({note})"
    if verbose:
        print(f"    map source: {source}")

    tried = []
    for candidate in candidates:
        try:
            table, path = load_channel_map(candidate)
        except (FileNotFoundError, ValueError) as exc:
            tried.append(f"{candidate}: {exc}")
            continue
        mismatched = [shank for shank, mua in probes
                      if not map_matches_shank(table, shank, mua)]
        if not mismatched:
            return table, path
        tried.append(f"{path.name}: geometry differs on shank(s) {mismatched}")

    detail = "\n    ".join(tried) if tried else "no candidate maps available"
    raise ValueError(
        f"Could not find a channel map matching this recording's geometry "
        f"({source}), so SpikeGadgets ids cannot be converted safely.\n    "
        + detail
        + "\n  Fix the animal's entry in device_types.json, pass --channel-map "
          "<name-or-path>, or --no-exclude-bad to skip the exclusion."
    )


def load_bad_channel_ids(path):
    """SpikeGadgets channel ids from bad_channels.txt (one per line)."""
    ids = []
    for token in Path(path).read_text().split():
        token = token.strip()
        if not token or token.startswith("#"):
            continue
        try:
            ids.append(int(token))
        except ValueError:
            raise ValueError(f"{path}: {token!r} is not an integer channel id")
    return sorted(set(ids))


def bad_channels_on_shank(table, shank, bad_spikegadget_ids):
    """Convert SpikeGadgets ids to this shank's NWB ids.

    Returns ``[(nwb_id, spikegadget_id, ycoord), ...]`` in NWB order. Ids
    belonging to other shanks are simply absent, which is how one whole-probe
    list serves every shank.
    """
    sub = table[table["sh"] == shank].reset_index(drop=True)
    wanted = set(int(v) for v in bad_spikegadget_ids)
    return [(index, int(sub.loc[index, "spikegadget"]), float(sub.loc[index, "ycoord"]))
            for index in range(len(sub))
            if int(sub.loc[index, "spikegadget"]) in wanted]


# =====================================================
# NREM RESTRICTION (optional)
# =====================================================

def clock_offset_sec(periods, mua):
    """Seconds to ADD to a time on the sleep-periods clock to reach MUA-epoch time.

    The NREM stage works on the LFP file's own clock, which starts at
    ``sleep_start_sample``; MUA times start at ``epoch_start_sample``. Both are
    whole-day sample indices at the acquisition rate, so the difference converts
    one to the other. Shared by :func:`nrem_windows_from_pkl` (bout intervals)
    and anything that also needs to align a continuous per-epoch score (e.g.
    ``sw_index``) from the same pkl onto the MUA clock.
    """
    lfp_start = periods.get("sleep_start_sample")
    if lfp_start is None:
        # Not recorded by the NREM stage; assume the two clocks already agree.
        return 0.0
    original_fs = float(mua.get("sampling_frequency", 30000.0))
    return (int(lfp_start) - int(mua.get("epoch_start_sample", 0))) / original_fs


def nrem_windows_from_pkl(path, mua):
    """NREM bouts as (start, end) seconds relative to the MUA epoch start.

    See :func:`clock_offset_sec` for how the two clocks are aligned.
    """
    with Path(path).open("rb") as file:
        periods = pickle.load(file)

    bouts = periods.get("bout_intervals_s")
    if not bouts:
        return np.empty((0, 2), dtype=np.float64), periods

    offset_sec = clock_offset_sec(periods, mua)
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


def zero_spike_intervals(spikes_by_channel, duration_sec):
    """Every gap with literally no spike on ANY of ``spikes_by_channel``.

    Pools all channels into one spike train and returns the gaps between
    consecutive spikes, plus the gaps to 0 and to ``duration_sec`` at the
    ends. Unlike :func:`channel_on_off`, there is no flanking-ON requirement
    and no minimum-duration filter here -- both are applied by the caller
    (clip to NREM windows, then the same 50-400 ms bound used for the
    fraction-based criterion) so the two sources combine on equal footing.
    """
    if spikes_by_channel:
        pooled = np.concatenate(
            [np.asarray(s, dtype=np.float64) for s in spikes_by_channel])
    else:
        pooled = np.empty(0, dtype=np.float64)
    pooled = np.unique(pooled[np.isfinite(pooled)])
    pooled = pooled[(pooled >= 0.0) & (pooled <= duration_sec)]

    edges = np.concatenate(([0.0], pooled, [duration_sec]))
    return np.stack([edges[:-1], edges[1:]], axis=1)


def merge_intervals(intervals):
    """Union of overlapping/touching ``(start, end)`` intervals, sorted."""
    intervals = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    if len(intervals) == 0:
        return intervals
    order = np.argsort(intervals[:, 0], kind="stable")
    intervals = intervals[order]
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + COMPARE_TOL_SEC:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return np.asarray(merged, dtype=np.float64)


def to_day_sample_intervals(intervals_sec, epoch_start_sample, sampling_frequency):
    """Convert epoch-relative-second intervals to whole-day acquisition samples.

    MUA spike times are ``sample_index / sampling_frequency`` computed from
    ``epoch_start_sample``, the whole-day sample this epoch's time-zero sits
    at (see the PATH RESOLUTION notes on ``epoch_start_sample`` vs
    ``sleep_start_sample``). This inverts that back to a day-sample index, so
    a window survives being reloaded once the epoch pkl's own relative clock
    is no longer at hand.
    """
    intervals = np.asarray(intervals_sec, dtype=np.float64).reshape(-1, 2)
    if not len(intervals) or not np.isfinite(sampling_frequency):
        return np.empty((0, 2), dtype=np.int64)
    return (np.rint(intervals * sampling_frequency).astype(np.int64)
            + int(epoch_start_sample))


def global_on_intervals(global_off, nrem_windows, min_sec, max_sec):
    """Gaps strictly between two consecutive global OFF periods, 50-2000 ms.

    An ON period is the complement of OFF: only the stretch BETWEEN two OFF
    periods is considered, so it is flanked by OFF on both sides by
    construction -- the stretch before the first OFF period or after the last
    is dropped, exactly as a channel OFF interval requires a valid ON run
    flanking it on both sides (see ``DEFAULT_FLANK_MODE``). ``global_off``
    must already be sorted and non-overlapping, which is how
    ``detect_off_states`` produces it.
    """
    global_off = np.asarray(global_off, dtype=np.float64).reshape(-1, 2)
    if len(global_off) < 2:
        return np.empty((0, 2), dtype=np.float64)

    starts = global_off[:-1, 1]
    ends = global_off[1:, 0]
    keep = ends > starts
    candidates = np.stack([starts[keep], ends[keep]], axis=1) if np.any(keep) \
        else np.empty((0, 2), dtype=np.float64)
    candidates = clip_intervals(candidates, nrem_windows)
    if not len(candidates):
        return candidates

    durations = candidates[:, 1] - candidates[:, 0]
    keep = ((durations >= min_sec - COMPARE_TOL_SEC)
            & (durations <= max_sec + COMPARE_TOL_SEC))
    return candidates[keep]


def detect_off_states(mua, *, isi_sec=DEFAULT_ISI_SEC,
                      min_on_sec=DEFAULT_MIN_ON_SEC,
                      min_channel_off_sec=DEFAULT_MIN_OFF_SEC,
                      global_fraction=GLOBAL_OFF_FRACTION,
                      global_min_channels=None,
                      global_min_sec=DEFAULT_GLOBAL_MIN_SEC,
                      global_max_sec=DEFAULT_GLOBAL_MAX_SEC,
                      global_on_min_sec=DEFAULT_GLOBAL_ON_MIN_SEC,
                      global_on_max_sec=DEFAULT_GLOBAL_ON_MAX_SEC,
                      min_channel_rate_hz=MIN_CHANNEL_RATE_HZ,
                      flank_mode=DEFAULT_FLANK_MODE,
                      exclude_channels=(),
                      nrem_windows=None, verbose=True):
    """Apply the channel and population OFF criteria to one shank.

    ``exclude_channels`` are NWB channel ids dropped before anything is
    computed -- they are not scored, not counted OFF, and not part of the
    denominator, exactly as if the shank had fewer sites. Use it for channels
    known bad from screening; see :func:`bad_channels_on_shank`.
    """
    channel_ids, depths = depth_sorted_channels(mua)
    duration = float(mua["duration_sec"])
    spike_map = mua["channel_spike_times"]

    excluded_bad = []
    if len(exclude_channels):
        drop = {str(_as_python_scalar(c)) for c in exclude_channels}
        keep = [i for i, cid in enumerate(channel_ids)
                if str(_as_python_scalar(cid)) not in drop]
        excluded_bad = [cid for cid in channel_ids
                        if str(_as_python_scalar(cid)) in drop]
        channel_ids = [channel_ids[i] for i in keep]
        depths = depths[keep]
        if verbose:
            print(f"    bad channels excluded: {len(excluded_bad)} "
                  f"({', '.join(str(c) for c in _sorted_channels(excluded_bad))})")
        if not channel_ids:
            raise ValueError(
                f"Every channel on shank {mua.get('shank')} is in the bad-channel "
                f"list; nothing left to analyse."
            )

    if nrem_windows is not None and len(nrem_windows):
        analysed_sec = float(np.sum(nrem_windows[:, 1] - nrem_windows[:, 0]))
    else:
        analysed_sec = duration

    channel_on, channel_off = {}, {}
    channel_rate, channel_off_fraction = {}, {}
    contributing = []
    pooled_spikes = []
    for channel_id in channel_ids:
        spikes = mask_spikes(_lookup_channel(spike_map, channel_id), nrem_windows)
        pooled_spikes.append(spikes)
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
        # Split the exclusions by cause: a rate cutoff and "no OFF interval at
        # all" mean very different things about a channel.
        too_quiet = [c for c in channel_ids
                     if channel_rate[c] < min_channel_rate_hz]
        no_off = [c for c in channel_ids
                  if channel_rate[c] >= min_channel_rate_hz and not len(channel_off[c])]
        print(f"    channels: {len(channel_ids)} total, {n_contributing} contributing")
        print(f"      rate range {min(channel_rate.values()):.2f}-"
              f"{max(channel_rate.values()):.2f} Hz over {analysed_sec:.0f} s analysed")
        if too_quiet:
            print(f"      excluded, below {min_channel_rate_hz:g} Hz: "
                  + ", ".join(f"ch{c} ({channel_rate[c]:.2f})" for c in too_quiet))
        else:
            print(f"      excluded, below {min_channel_rate_hz:g} Hz: none")
        if no_off:
            print(f"      excluded, no OFF interval: {no_off}")
        print(f"    global OFF threshold: >= {min_channels} channels"
              + (f" ({global_fraction:.0%} of contributing)"
                 if global_min_channels is None else " (explicit)"))

    counted = {cid: channel_off[cid] for cid in contributing}
    candidates, edges, counts = simultaneous_off_intervals(counted, min_channels)

    # Supplementary rule (criterion 8): population-wide silence needs no
    # flanking-ON on either side, so it is computed independently of `counted`
    # / `candidates` and only combined afterward -- otherwise a gap next to an
    # invalid (too-short) ON run would still be dropped, exactly the case this
    # rule exists to recover.
    #
    # `zero_spike_intervals` returns a full partition of [0, duration] into
    # every raw inter-spike gap, most of them far shorter than min_off_sec --
    # a channel firing at tens of Hz has thousands of sub-ms gaps. Those MUST
    # be dropped here, before merging: consecutive raw gaps are contiguous by
    # construction (each one's end is the next one's start), so handing the
    # unfiltered set to merge_intervals would glue the entire recording into
    # one interval instead of isolating the genuine >= 50 ms silences.
    zero_spike_raw = zero_spike_intervals(pooled_spikes, duration)
    zero_spike_raw = zero_spike_raw[
        (zero_spike_raw[:, 1] - zero_spike_raw[:, 0]) >= global_min_sec - COMPARE_TOL_SEC]
    zero_spike_candidates = clip_intervals(zero_spike_raw, nrem_windows)

    if len(candidates) or len(zero_spike_candidates):
        combined = merge_intervals(np.concatenate([candidates, zero_spike_candidates], axis=0))
    else:
        combined = np.empty((0, 2), dtype=np.float64)

    if len(combined):
        durations = combined[:, 1] - combined[:, 0]
        keep = ((durations >= global_min_sec - COMPARE_TOL_SEC)
                & (durations <= global_max_sec + COMPARE_TOL_SEC))
        global_off = combined[keep]
    else:
        global_off = combined

    global_off_source = np.full(len(global_off), "zero_spike_only", dtype=object)
    for index, (start, end) in enumerate(global_off):
        if len(candidates) and np.any((candidates[:, 0] < end - COMPARE_TOL_SEC)
                                       & (candidates[:, 1] > start + COMPARE_TOL_SEC)):
            global_off_source[index] = "fraction"

    off_durations = (global_off[:, 1] - global_off[:, 0]) if len(global_off) else np.array([])

    # Criterion 9: ON is the complement of the FINAL, accepted OFF timeline
    # (fraction- and zero-spike-based OFF periods combined) -- never the
    # unfiltered `combined` candidates -- so ON and OFF partition the analysed
    # time consistently with what is actually reported as OFF elsewhere.
    global_on = global_on_intervals(global_off, nrem_windows, global_on_min_sec, global_on_max_sec)
    on_durations = (global_on[:, 1] - global_on[:, 0]) if len(global_on) else np.array([])

    epoch_start_sample = int(mua.get("epoch_start_sample", 0))
    sampling_frequency = float(mua.get("sampling_frequency", np.nan))
    global_on_daysample = to_day_sample_intervals(
        global_on, epoch_start_sample, sampling_frequency)

    return {
        "session": mua.get("session"),
        "epoch": mua.get("epoch"),
        "suffix": mua.get("suffix"),
        "shank": int(mua["shank"]),
        "duration_sec": duration,
        "analysed_sec": analysed_sec,
        "sampling_frequency": sampling_frequency,
        "epoch_start_sample": epoch_start_sample,
        "epoch_end_sample": int(mua.get("epoch_end_sample", 0)),
        "channel_ids": np.asarray(channel_ids),
        "channel_depths_um": depths,
        "channel_rate_hz": channel_rate,
        "channel_off_fraction": channel_off_fraction,
        "contributing_channels": np.asarray(contributing),
        "excluded_bad_channels": np.asarray(excluded_bad),
        "n_channels": len(channel_ids),
        "n_contributing_channels": n_contributing,
        "global_min_channels": min_channels,
        "channel_on_intervals": channel_on,
        "channel_off_intervals": channel_off,
        "global_off_candidates": combined,
        "global_off_candidates_fraction": candidates,
        "zero_spike_candidates": zero_spike_candidates,
        "global_off_intervals": global_off,
        "global_off_source": global_off_source,   # "fraction" or "zero_spike_only" per row
        "global_on_intervals": global_on,
        "global_on_intervals_daysample": global_on_daysample,  # whole-day sample index
        "off_count_edges": edges,          # counts[i] holds on [edges[i], edges[i+1])
        "off_counts": counts,
        "nrem_windows_sec": (np.asarray(nrem_windows).reshape(-1, 2)
                             if nrem_windows is not None else None),
        "summary": {
            "n_channel_off": int(sum(len(v) for v in channel_off.values())),
            "n_global_candidates": int(len(combined)),
            "n_global_off": int(len(global_off)),
            "n_global_off_fraction": int(np.sum(global_off_source == "fraction")),
            "n_global_off_zero_spike_only": int(np.sum(global_off_source == "zero_spike_only")),
            "global_off_per_min": float(len(global_off) / (analysed_sec / 60.0))
                                  if analysed_sec > 0 else float("nan"),
            "mean_off_ms": float(np.mean(off_durations) * 1000) if off_durations.size else float("nan"),
            "median_off_ms": float(np.median(off_durations) * 1000) if off_durations.size else float("nan"),
            "off_time_fraction": float(np.sum(off_durations) / analysed_sec)
                                 if off_durations.size and analysed_sec > 0 else 0.0,
            "n_global_on": int(len(global_on)),
            "global_on_per_min": float(len(global_on) / (analysed_sec / 60.0))
                                 if analysed_sec > 0 else float("nan"),
            "mean_on_ms": float(np.mean(on_durations) * 1000) if on_durations.size else float("nan"),
            "median_on_ms": float(np.median(on_durations) * 1000) if on_durations.size else float("nan"),
            "on_time_fraction": float(np.sum(on_durations) / analysed_sec)
                                if on_durations.size and analysed_sec > 0 else 0.0,
        },
        "params": {
            "isi_sec": float(isi_sec),
            "min_on_sec": float(min_on_sec),
            "min_channel_off_sec": float(min_channel_off_sec),
            "global_fraction": None if global_min_channels is not None else float(global_fraction),
            "global_min_channels": int(min_channels),
            "global_min_sec": float(global_min_sec),
            "global_max_sec": float(global_max_sec),
            "global_on_min_sec": float(global_on_min_sec),
            "global_on_max_sec": float(global_on_max_sec),
            "min_channel_rate_hz": float(min_channel_rate_hz),
            "n_bad_channels_excluded": len(excluded_bad),
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
TEAL = "#1B9E77"
STEELBLUE = "#3B7EA1"


def stamp_figure(figure, text, script=None):
    """Embed a reproducibility line (what made this figure, from what, when).

    ``script`` names the caller, so a sibling detector reusing this helper is
    credited for its own figures rather than for this module's.

    Wrapped before drawing: matplotlib does not wrap ``figure.text``, and with
    ``bbox_inches="tight"`` a single long line silently widens the saved canvas
    to fit it, which is what stretched earlier versions of this figure.
    """
    script = script or Path(__file__).name
    body = f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by {script}  |  {text}"
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

    for (off_start, off_end), source in zip(result["global_off_intervals"],
                                             result["global_off_source"]):
        left, right = max(off_start, start_sec), min(off_end, end_sec)
        if right > left:
            color = TEAL if source == "zero_spike_only" else PURPLE
            ax_raster.add_patch(Rectangle(
                (left - start_sec, -0.55), right - left, n_channels + 0.1,
                fill=False, edgecolor=color, linewidth=1.6, zorder=5))

    for on_start, on_end in result.get("global_on_intervals", []):
        left, right = max(on_start, start_sec), min(on_end, end_sec)
        if right > left:
            ax_raster.add_patch(Rectangle(
                (left - start_sec, -0.55), right - left, n_channels + 0.1,
                fill=False, edgecolor=STEELBLUE, linewidth=1.2, linestyle=(0, (4, 2)),
                zorder=4))

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
                "orange = channel OFF   |   purple = global OFF (fraction rule)   |   "
                "teal = global OFF (zero-spike rule only)   |   "
                "dashed blue = global ON",
                ha="center", fontsize=8, color="0.35")

    stamp_figure(figure, (
        f"src={mua_pkl}  |  {lfp_note}  |  "
        f"isi={params['isi_sec'] * 1000:g}ms min_on={params['min_on_sec'] * 1000:g}ms "
        f"min_off={params['min_channel_off_sec'] * 1000:g}ms flank={params['flank_mode']}  |  "
        f"min_channel_rate={params['min_channel_rate_hz']:g}Hz "
        f"({result['n_contributing_channels']}/{result['n_channels']} ch vote, "
        f"{params.get('n_bad_channels_excluded', 0)} bad excluded)  |  "
        f"global={params['global_min_channels']}ch "
        f"({'explicit' if params['global_fraction'] is None else format(params['global_fraction'], '.2f')}) "
        f"{params['global_min_sec'] * 1000:g}-{params['global_max_sec'] * 1000:g}ms  |  "
        f"nrem_restricted={params['nrem_restricted']} analysed={result['analysed_sec']:.0f}s  |  "
        f"detect_channel_radius={result['mua_params'].get('detect_channel_radius')} "
        f"threshold={result['mua_params'].get('detect_threshold')} "
        f"scale={result['mua_params'].get('scale_mode')}  |  "
        f"n_global_off={summary['n_global_off']} ({summary['global_off_per_min']:.2f}/min, "
        f"{summary['n_global_off_fraction']} fraction + "
        f"{summary['n_global_off_zero_spike_only']} zero-spike-only)  |  "
        f"n_global_on={summary['n_global_on']} ({summary['global_on_per_min']:.2f}/min) "
        f"{params['global_on_min_sec'] * 1000:g}-{params['global_on_max_sec'] * 1000:g}ms  |  "
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
                             f"(default {GLOBAL_OFF_FRACTION}; the paper's 12-of-16 "
                             f"is 0.75)")
    parser.add_argument("--global-min-channels", type=int, default=None,
                        help="absolute channel count, overriding --global-fraction")
    parser.add_argument("--global-min-ms", type=float, default=DEFAULT_GLOBAL_MIN_SEC * 1000)
    parser.add_argument("--global-max-ms", type=float, default=DEFAULT_GLOBAL_MAX_SEC * 1000)
    parser.add_argument("--global-on-min-ms", type=float, default=DEFAULT_GLOBAL_ON_MIN_SEC * 1000,
                        help="minimum duration of a global ON period (gap between "
                             "two global OFF periods)")
    parser.add_argument("--global-on-max-ms", type=float, default=DEFAULT_GLOBAL_ON_MAX_SEC * 1000,
                        help="maximum duration of a global ON period (the paper's "
                             "2000 ms)")
    parser.add_argument("--min-channel-rate-hz", type=float, default=MIN_CHANNEL_RATE_HZ)
    parser.add_argument("--flank-mode", choices=("bridge", "strict"),
                        default=DEFAULT_FLANK_MODE)

    parser.add_argument("--bad-channels-file", type=Path, default=None,
                        help="bad_channels.txt in SpikeGadgets id space (default: "
                             "discovered at the session-folder root)")
    parser.add_argument("--channel-map", default=None,
                        help="device type (e.g. 8shank32) or a full path to the "
                             "mapping CSV used to convert SpikeGadgets ids to "
                             "per-shank NWB ids. Default: the device type "
                             "registered for this animal in device_types.json")
    parser.add_argument("--no-exclude-bad", action="store_true",
                        help="keep the channels listed in bad_channels.txt")

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

        # Bad channels: one whole-probe list in SpikeGadgets ids, converted per
        # shank through the channel map. Resolved once per epoch, since the map
        # verification needs the shanks that will actually be processed.
        bad_by_shank = {shank: [] for shank in shanks}
        if not args.no_exclude_bad:
            bad_file = resolve_bad_channels_file(folders, args.bad_channels_file)
            if bad_file is None:
                print("  no bad_channels.txt found — no channels excluded "
                      "(pass --bad-channels-file, or --no-exclude-bad to silence)")
            else:
                bad_sg = load_bad_channel_ids(bad_file)
                table, map_path = resolve_channel_map(data, shanks, args.channel_map)
                print(f"  bad channels: {len(bad_sg)} SpikeGadgets ids from "
                      f"{bad_file}")
                print(f"    channel map: {map_path.name} (geometry verified "
                      f"against the pkl on every shank)")
                for shank in shanks:
                    hits = bad_channels_on_shank(table, shank, bad_sg)
                    bad_by_shank[shank] = [nwb for nwb, _, _ in hits]
                    if hits:
                        detail = ", ".join(f"ch{nwb}(sg{sg}@{y:.0f}um)"
                                           for nwb, sg, y in hits)
                        print(f"    shank {shank}: {len(hits)} -> {detail}")
                mapped = sum(len(v) for v in bad_by_shank.values())
                if mapped != len(bad_sg):
                    print(f"    NOTE: {len(bad_sg) - mapped} listed id(s) fall on "
                          f"shanks not being processed")

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
        on_window_rows = []
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
                global_on_min_sec=args.global_on_min_ms / 1000.0,
                global_on_max_sec=args.global_on_max_ms / 1000.0,
                min_channel_rate_hz=args.min_channel_rate_hz,
                flank_mode=args.flank_mode,
                exclude_channels=bad_by_shank.get(shank, []),
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
            print(f"      via fraction rule   : {info['n_global_off_fraction']}")
            print(f"      via zero-spike rule : {info['n_global_off_zero_spike_only']} "
                  f"(would have been missed without it)")
            print(f"    accepted global ON    : {info['n_global_on']} "
                  f"({info['global_on_per_min']:.2f}/min, "
                  f"median {info['median_on_ms']:.0f} ms, "
                  f"{info['on_time_fraction']:.1%} of analysed time)")

            tag = (f"f{int(round(args.global_fraction * 100))}"
                   if args.global_min_channels is None else f"n{args.global_min_channels}")
            tag += f"_{args.flank_mode}"
            # The rate cutoff changes which channels vote, so it belongs in the
            # name: without it a run at a new cutoff silently overwrites an old
            # one that is no longer comparable.
            tag += f"_r{args.min_channel_rate_hz:g}hz"
            if not args.no_exclude_bad:
                tag += "_nobad"
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
                "n_bad_excluded": int(result["params"]["n_bad_channels_excluded"]),
                "n_channels": result["n_channels"],
                "n_contributing": result["n_contributing_channels"],
                "global_min_channels": result["global_min_channels"],
                "analysed_sec": result["analysed_sec"],
                **info,
            })

            for (day_start, day_end), (epoch_start, epoch_end) in zip(
                    result["global_on_intervals_daysample"], result["global_on_intervals"]):
                on_window_rows.append({
                    "shank": shank,
                    "on_start_daysample": int(day_start),
                    "on_end_daysample": int(day_end),
                    "on_start_epoch_sec": float(epoch_start),
                    "on_end_epoch_sec": float(epoch_end),
                    "duration_ms": float(epoch_end - epoch_start) * 1000.0,
                })

        if on_window_rows:
            windows_path = out_dir / f"{result['session']}_{suffix}_global_on_windows_{tag}.pkl"
            windows_doc = {
                "session": result["session"],
                "epoch": suffix,
                "sampling_frequency": result["sampling_frequency"],
                "epoch_start_sample": result["epoch_start_sample"],
                "params_tag": tag,
                "note": ("on_start_daysample/on_end_daysample = whole-day acquisition "
                         "sample index (epoch_start_sample + round(epoch_sec * "
                         "sampling_frequency)); on_start_epoch_sec/on_end_epoch_sec = "
                         "seconds relative to this epoch's own MUA/LFP files"),
                "generated": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                "source_script": Path(__file__).name,
                "reproduce": f"python detect_off_states.py --epochs {suffix}",
                "rows": on_window_rows,
            }
            with windows_path.open("wb") as file:
                pickle.dump(windows_doc, file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\n  ON windows ({suffix}): {len(on_window_rows)} -> {windows_path.name}")

    if summary_rows:
        out_dir = resolve_output_dir(next(iter(located.values())))
        summary_path = out_dir / "off_states_summary.json"
        summary_path.write_text(json.dumps(summary_rows, indent=2))
        print("\n" + "=" * 72)
        print(f"{'epoch':>6} {'shank':>6} {'bad':>4} {'ch':>4} {'contrib':>8} "
              f"{'thresh':>7} {'nOFF':>6} {'per min':>8} {'median ms':>10}")
        for row in summary_rows:
            print(f"{row['epoch']:>6} {row['shank']:>6} {row['n_bad_excluded']:>4} "
                  f"{row['n_channels']:>4} "
                  f"{row['n_contributing']:>8} {row['global_min_channels']:>7} "
                  f"{row['n_global_off']:>6} {row['global_off_per_min']:>8.2f} "
                  f"{row['median_off_ms']:>10.0f}")
        print(f"\nSummary: {summary_path}")
        print("=" * 72)


if __name__ == "__main__":
    main()
