r"""Candidate population burst events (PBEs) inside NREM ON periods.

Population activity = all sorted units' spikes pooled into one train (``--source units``,
the default), or every channel's MUA threshold crossings from find_sleep_mua.py
(``--source mua``; each channel then counts as one 'unit' for participation).

Definition (defaults; every value is a CLI flag)
------------------------------------------------
1. Pool every unit's spikes and bin at 5 ms over the whole sleep block.
2. Smooth with a Gaussian, sigma = 15 ms.
3. Z-score using ONLY the bins that fall in NREM AND in a MUA ON window
   (mean and SD of those bins), so the baseline is ON activity, not the
   silent OFF periods.
4. Candidate = run of NREM bins with z > 1 (onset/offset at z = 1).
   ``--state nrem`` (default): z baseline and candidates come from ALL NREM bins;
   each event is tagged ``peak_in_on`` / ``frac_in_on`` against the MUA ON windows.
   ``--state nrem_on``: the older ON-gated version - baseline and candidates only in
   NREM-ON bins (output files get an ``_ongated`` suffix).
5. Merge candidates separated by gaps < 20 ms.
6. Keep a candidate when peak z > 3, duration is 50-400 ms, and enough units
   fire at least one spike in it: n_active >= max(8, ceil(10% of units))
   by default (``--participation-rule min`` uses the looser of the two).

Inputs (per animal-day and pre/post block, all already on the lab share)
-------------------------------------------------------------------------
- Sorted sleep spikes: ``sortout/<animal>/<session>/sleep_spikes_<date>_sleep_<block>.pkl``
  (extract_sleep_session_spikes_multishank.py).
- NREM bouts: ``<session>/low_freq/sleep_segmentation/<session>_<block>_sleep_periods.pkl``.
- MUA ON windows: ``<session>/MUA/off_states/<session>_<block>_global_on_windows_<tag>.pkl``
  (detect_off_states.py), combined across shanks by
  ``sleep_states.load_mua_on_intervals`` (an instant is ON when >= half the shanks are ON).

Outputs, in ``<session>/PBE/``: an events CSV, a pkl with the traces and
parameters, and a per-block figure. A cross-day summary CSV/PNG goes in
``experiment_data/<animal>/PBE_summary/``.

Usage
-----
    python sleep/PBE/detect_pbe.py --days 260729 260730 260731
    python sleep/PBE/detect_pbe.py --days 260731 --blocks post --participation-rule min
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sleep_paths  # noqa: F401,E402

import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from reactivation.sleep_apply.sleep_states import (  # noqa: E402
    SleepBlock, interval_mask, load_mua_on_intervals, load_nrem_intervals,
    normalize_intervals,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = REPO_ROOT / "sleep" / "sleep_day_configs.json"
DEFAULT_ON_TAG = "f75_strict_r1hz_nobad"


# ---------------------------------------------------------------- pickles
def install_pickle_compat():
    """Let numpy-1 Windows read pickles written on the Mac (numpy 2, PosixPath)."""
    pathlib.PosixPath = pathlib.WindowsPath if sys.platform == "win32" else pathlib.PosixPath
    try:
        importlib.import_module("numpy._core.numeric")
        return  # numpy 2 (or a 1.x whose _core stub is complete enough)
    except ImportError:
        pass
    # numpy 1.26 ships a partial numpy._core stub, so alias every submodule explicitly.
    sys.modules["numpy._core"] = np.core
    for name in ("numeric", "multiarray", "umath", "_multiarray_umath",
                 "fromnumeric", "shape_base", "_dtype", "_methods"):
        try:
            sys.modules[f"numpy._core.{name}"] = importlib.import_module(f"numpy.core.{name}")
        except ImportError:
            pass


# ---------------------------------------------------------------- paths
def block_paths(animal, date, block, on_tag=DEFAULT_ON_TAG):
    registry = json.loads(REGISTRY.read_text())
    day = registry[f"{animal}_{date}"]
    rec_folder = Path(day["rec_folder"])
    session = rec_folder.name
    # experiment_data/<animal>/<date>/<session>  ->  sortout/<animal>/<session>
    sortout = Path(str(rec_folder.parents[2]).replace("experiment_data", "sortout")) / animal / session
    on_matches = sorted(
        p for p in (rec_folder / "MUA" / "off_states").glob(
            f"{session}_{block}_global_on_windows_{on_tag}.pkl")
        if not p.name.startswith("._"))
    return {
        "session": session,
        "rec_folder": rec_folder,
        "spikes": sortout / f"sleep_spikes_{date}_sleep_{block}.pkl",
        "nrem": rec_folder / "low_freq" / "sleep_segmentation" / f"{session}_{block}_sleep_periods.pkl",
        "on": on_matches[0] if on_matches else None,
        "mua": rec_folder / "MUA" / f"{session}_{block}_mua_events.pkl",
        "out_dir": rec_folder / "PBE",
    }


def registry_block(animal, date, block, label, fs=30000.0):
    """SleepBlock from the registry's start/end samples (same space as the MUA epoch)."""
    day = json.loads(REGISTRY.read_text())[f"{animal}_{date}"][block]
    start, end = int(day["start_sample"]), int(day["end_sample"])
    return SleepBlock(label=label, sampling_frequency_hz=fs, day_start_sample=start,
                      day_end_sample=end, duration_sec=(end - start) / fs,
                      source_path=str(REGISTRY))


def load_mua_trains(path, block, shanks=None):
    """Per-channel MUA threshold crossings (find_sleep_mua.py) in block-local seconds.

    Each channel is one 'unit' for pooling and participation. Spikes are detected
    independently per channel (channel radius 0), so one physical spike can count
    on several neighbouring channels.
    """
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    if data.get("partial"):
        raise ValueError(f"{Path(path).name} is a partial (--limit-sec) run")
    fs = float(data["sampling_frequency"])
    if not np.isclose(fs, block.sampling_frequency_hz):
        raise ValueError(f"MUA fs {fs} != spike fs {block.sampling_frequency_hz}")
    offset = (int(data["epoch_start_sample"]) - block.day_start_sample) / fs
    trains = {}
    for shank, entry in sorted(data["shanks"].items()):
        if shanks is not None and int(shank) not in shanks:
            continue
        for ch, t in entry["channel_spike_times"].items():
            t = np.asarray(t, dtype=float) + offset
            trains[f"shank{shank}_ch{int(ch):02d}"] = t[(t >= 0) & (t < block.duration_sec)]
    if offset != 0:
        print(f"    MUA clock offset vs sleep block: {offset:+.4f} s (applied)")
    return trains


def shank_tag(shanks):
    """'' for all shanks, '_sh0-3' for a contiguous subset, '_sh0_2_5' otherwise."""
    if not shanks:
        return ""
    s = sorted(int(v) for v in shanks)
    if s == list(range(s[0], s[-1] + 1)) and len(s) > 1:
        return f"_sh{s[0]}-{s[-1]}"
    return "_sh" + "_".join(map(str, s))


def state_tag(state):
    """Default NREM-only output keeps the plain names; the ON-gated version is suffixed."""
    return "" if state == "nrem" else "_ongated"


# ---------------------------------------------------------------- core
def intersect_intervals(a, b):
    """Intersection of two sorted, non-overlapping (n, 2) interval sets."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if hi > lo:
            out.append((lo, hi))
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return np.asarray(out, dtype=float).reshape(-1, 2)


def runs_of_true(mask):
    """[start, stop) index pairs of each run of True."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    diff = np.diff(padded)
    return np.column_stack((np.flatnonzero(diff == 1), np.flatnonzero(diff == -1)))


def population_trace(spike_trains, duration_sec, bin_sec, sigma_sec):
    edges = np.arange(0.0, duration_sec + bin_sec, bin_sec)
    counts = np.zeros(len(edges) - 1)
    for train in spike_trains:
        counts += np.histogram(train, bins=edges)[0]
    rate_hz = gaussian_filter1d(counts, sigma=sigma_sec / bin_sec, mode="constant") / bin_sec
    return edges, counts, rate_hz


def detect_pbes(spike_trains, duration_sec, state_intervals, *, on_intervals=None,
                bin_sec=0.005, sigma_sec=0.015,
                onset_z=1.0, peak_z=3.0, min_dur_sec=0.05, max_dur_sec=0.4,
                merge_gap_sec=0.02, participation_frac=0.10, participation_min_units=8,
                participation_rule="max"):
    """Return (events, trace, funnel). Times are block-relative seconds.

    ``state_intervals`` is where the z-score baseline comes from AND where candidates may
    occur (all NREM, or NREM-ON). ``on_intervals`` (NREM-ON) is only used to tag each
    event with whether it sits in an ON window; defaults to ``state_intervals``.
    """
    edges, counts, rate = population_trace(spike_trains, duration_sec, bin_sec, sigma_sec)
    centers = 0.5 * (edges[:-1] + edges[1:])
    in_state = interval_mask(centers, state_intervals)
    in_on = interval_mask(centers, state_intervals if on_intervals is None else on_intervals)
    if in_state.sum() < 10:
        raise ValueError("fewer than 10 bins in the detection state; nothing to z-score against")
    mu, sd = rate[in_state].mean(), rate[in_state].std()
    z = (rate - mu) / sd

    runs = runs_of_true((z > onset_z) & in_state)
    funnel = {"crossings_z>onset": int(len(runs))}
    # All length tests are done in whole bins: in float seconds 10 x 5 ms comes out as
    # 49.999... ms and a 50 ms event would fail a ">= 50 ms" rule.
    merge_gap_bins = int(round(merge_gap_sec / bin_sec))
    min_bins = int(round(min_dur_sec / bin_sec))
    max_bins = int(round(max_dur_sec / bin_sec))
    # merge runs whose gap is shorter than merge_gap
    merged = []
    for start, stop in runs:
        if merged and (start - merged[-1][1]) < merge_gap_bins:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])
    merged = np.asarray(merged, dtype=int).reshape(-1, 2)
    funnel["after_merge"] = int(len(merged))

    n_units = len(spike_trains)
    frac_units = int(np.ceil(participation_frac * n_units))
    need = (max if participation_rule == "max" else min)(frac_units, participation_min_units)
    sorted_trains = [np.sort(t) for t in spike_trains]

    rows = []
    n_peak = n_dur = 0
    for start, stop in merged:
        t0, t1 = edges[start], edges[stop]
        seg = z[start:stop]
        if seg.max() <= peak_z:
            continue
        n_peak += 1
        dur = (stop - start) * bin_sec
        if not (min_bins <= stop - start <= max_bins):
            continue
        n_dur += 1
        active = np.array([np.searchsorted(t, t1, "left") > np.searchsorted(t, t0, "left")
                           for t in sorted_trains])
        n_active = int(active.sum())
        peak_idx = start + int(np.argmax(seg))
        rows.append({
            "start_sec": t0, "end_sec": t1, "duration_ms": dur * 1e3,
            "peak_time_sec": centers[peak_idx], "peak_z": float(seg.max()),
            "mean_z": float(seg.mean()), "n_spikes": int(counts[start:stop].sum()),
            "n_active_units": n_active, "frac_active_units": n_active / n_units,
            "passes_participation": n_active >= need,
            "peak_in_on": bool(in_on[peak_idx]),
            "frac_in_on": float(in_on[start:stop].mean()),
        })
    funnel["peak_z>thr"] = n_peak
    funnel["duration_ok"] = n_dur
    funnel["participation_ok"] = int(sum(r["passes_participation"] for r in rows))

    trace = {"edges": edges, "counts": counts, "rate_hz": rate, "z": z,
             "in_nrem_on": in_on, "in_state": in_state, "mu_hz": mu, "sd_hz": sd,
             "n_units": n_units, "participation_need": need}
    return rows, trace, funnel


def describe_on_windows(on, nrem, nrem_on, trace, bin_sec):
    """How the ON periods look inside NREM."""
    on_in_nrem = intersect_intervals(on, nrem)
    durs = (on_in_nrem[:, 1] - on_in_nrem[:, 0]) * 1e3 if len(on_in_nrem) else np.array([])
    nrem_sec = float((nrem[:, 1] - nrem[:, 0]).sum()) if len(nrem) else 0.0
    centers = 0.5 * (trace["edges"][:-1] + trace["edges"][1:])
    in_nrem = interval_mask(centers, nrem)
    off_bins = in_nrem & ~trace["in_nrem_on"]
    return {
        "nrem_sec": nrem_sec,
        "n_on_windows_in_nrem": int(len(on_in_nrem)),
        "on_sec_in_nrem": float(durs.sum() / 1e3),
        "on_fraction_of_nrem": float(durs.sum() / 1e3 / nrem_sec) if nrem_sec else np.nan,
        "on_per_min_nrem": float(len(on_in_nrem) / nrem_sec * 60) if nrem_sec else np.nan,
        "on_dur_ms_median": float(np.median(durs)) if len(durs) else np.nan,
        "on_dur_ms_q25": float(np.percentile(durs, 25)) if len(durs) else np.nan,
        "on_dur_ms_q75": float(np.percentile(durs, 75)) if len(durs) else np.nan,
        "on_dur_ms_max": float(durs.max()) if len(durs) else np.nan,
        "pop_rate_hz_nrem_on": float(trace["rate_hz"][trace["in_nrem_on"]].mean()),
        "pop_rate_hz_nrem_not_on": float(trace["rate_hz"][off_bins].mean()) if off_bins.any() else np.nan,
        "durations_ms": durs,
        "on_in_nrem": on_in_nrem,
    }


# ---------------------------------------------------------------- plotting
def plot_block(path_png, label, events, trace, on_desc, funnel, spike_trains, unit_ids,
               params, sources):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges, z = trace["edges"], trace["z"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_sec = params["bin_sec"]
    kept = [e for e in events if e["passes_participation"]]
    on_in_nrem = on_desc["on_in_nrem"]

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.4, 0.8, 1, 1], hspace=0.55, wspace=0.35)

    # example window: densest PBE stretch, else first ON window in NREM
    if kept:
        times = np.array([e["peak_time_sec"] for e in kept])
        counts = np.array([np.sum(np.abs(times - t) < 2.5) for t in times])
        t_mid = times[int(np.argmax(counts))]
    elif len(on_in_nrem):
        t_mid = on_in_nrem[0, 0] + 2.5
    else:
        t_mid = 2.5
    w0, w1 = t_mid - 2.5, t_mid + 2.5

    ax_r = fig.add_subplot(gs[0, :])
    order = sorted(range(len(unit_ids)), key=lambda i: unit_ids[i])
    for row, i in enumerate(order):
        t = spike_trains[i]
        t = t[(t >= w0) & (t < w1)]
        ax_r.vlines(t, row, row + 0.8, lw=0.4, color="0.15")
    for s, e in on_in_nrem:
        if e > w0 and s < w1:
            ax_r.axvspan(max(s, w0), min(e, w1), color="tab:green", alpha=0.12, lw=0)
    for ev in events:
        if ev["end_sec"] > w0 and ev["start_sec"] < w1:
            ax_r.axvspan(ev["start_sec"], ev["end_sec"],
                         color="tab:red" if ev["passes_participation"] else "tab:orange",
                         alpha=0.25, lw=0)
    ax_r.set_xlim(w0, w1)
    ax_r.set_ylim(0, len(order))
    ax_r.set_ylabel(f"{params['source']} train (by shank)")
    ax_r.set_title(f"{label}: raster  |  green = NREM-ON window, red = PBE, "
                   f"orange = failed participation only", fontsize=10)

    ax_z = fig.add_subplot(gs[1, :], sharex=ax_r)
    sel = (centers >= w0) & (centers < w1)
    ax_z.plot(centers[sel], z[sel], lw=0.8, color="k")
    ax_z.axhline(params["onset_z"], ls="--", lw=0.7, color="tab:blue")
    ax_z.axhline(params["peak_z"], ls="--", lw=0.7, color="tab:red")
    ax_z.set_ylabel("pop. MUA (z)")
    ax_z.set_xlabel("block time (s)")

    ax = fig.add_subplot(gs[2, 0])
    d = on_desc["durations_ms"]
    if len(d):
        ax.hist(d, bins=np.linspace(0, 2000, 41), color="tab:green")
    ax.set_xlabel("ON window in NREM (ms)")
    ax.set_ylabel("count")
    ax.set_title(f"n={on_desc['n_on_windows_in_nrem']}, "
                 f"{on_desc['on_fraction_of_nrem']:.0%} of NREM", fontsize=9)

    ax = fig.add_subplot(gs[2, 1])
    zin = z[trace["in_state"]]
    ax.hist(zin, bins=np.linspace(-3, 8, 111), color="0.4", log=True)
    ax.axvline(params["onset_z"], ls="--", color="tab:blue")
    ax.axvline(params["peak_z"], ls="--", color="tab:red")
    ax.set_xlabel(f"z ({params['state']} bins)")
    ax.set_title(f"mean {trace['mu_hz']:.0f} Hz, SD {trace['sd_hz']:.0f} Hz", fontsize=9)

    # population rate aligned to ON onset / offset
    lag = np.arange(-0.3, 0.6 + bin_sec, bin_sec)
    for col, (key, title) in enumerate(((0, "ON onset"), (1, "ON offset"))):
        ax = fig.add_subplot(gs[2, 2 + col])
        if len(on_in_nrem):
            idx = np.searchsorted(edges, on_in_nrem[:, key][:, None] + lag[None, :]) - 1
            ok = (idx >= 0) & (idx < len(z))
            mat = np.where(ok, z[np.clip(idx, 0, len(z) - 1)], np.nan)
            m = np.nanmean(mat, 0)
            se = np.nanstd(mat, 0) / np.sqrt(np.sum(ok, 0).clip(1))
            ax.plot(lag * 1e3, m, color="k")
            ax.fill_between(lag * 1e3, m - se, m + se, color="0.7")
        ax.axvline(0, ls=":", color="tab:green")
        ax.set_xlabel(f"time from {title} (ms)")
        ax.set_ylabel("pop. MUA (z)")

    ax = fig.add_subplot(gs[3, 0])
    if events:
        ax.hist([e["duration_ms"] for e in kept], bins=np.arange(50, 425, 25), color="tab:red")
    ax.set_xlabel("PBE duration (ms)")
    ax.set_title(f"{len(kept)} PBEs", fontsize=9)

    ax = fig.add_subplot(gs[3, 1])
    if events:
        ax.hist([e["n_active_units"] for e in events], bins=40, color="0.4")
    ax.axvline(trace["participation_need"], ls="--", color="tab:red")
    ax.set_xlabel(f"{params['source']} trains active in candidate")
    ax.set_title(f"need >= {trace['participation_need']} of {trace['n_units']}", fontsize=9)

    ax = fig.add_subplot(gs[3, 2])
    if kept and len(on_in_nrem):
        starts = on_in_nrem[:, 0]
        k = np.searchsorted(starts, [e["peak_time_sec"] for e in kept]) - 1
        k = k[k >= 0]
        rel = np.array([e["peak_time_sec"] for e in kept])[-len(k):] - starts[k] if len(k) else []
        ax.hist(np.asarray(rel) * 1e3, bins=np.linspace(0, 2000, 41), color="tab:red")
    ax.set_xlabel("PBE peak - ON onset (ms)")

    ax = fig.add_subplot(gs[3, 3])
    names = list(funnel)
    ax.barh(range(len(names)), [funnel[n] for n in names], color="0.5")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    for i, n in enumerate(names):
        ax.text(funnel[n], i, f" {funnel[n]}", va="center", fontsize=8)
    ax.set_title("candidate funnel", fontsize=9)

    stamp = (f"{datetime.now():%Y-%m-%d %H:%M}  |  sleep/PBE/detect_pbe.py  |  {label}  |  "
             f"bin {params['bin_sec']*1e3:g} ms, sigma {params['sigma_sec']*1e3:g} ms, z from + detection in {params['state']} bins, "
             f"onset z>{params['onset_z']:g}, peak z>{params['peak_z']:g}, "
             f"{params['min_dur_sec']*1e3:g}-{params['max_dur_sec']*1e3:g} ms, merge <{params['merge_gap_sec']*1e3:g} ms, "
             f"participation {params['participation_rule']}({params['participation_frac']:.0%}, "
             f"{params['participation_min_units']}) = {trace['participation_need']} units, "
             f"ON shank fraction {params['min_shank_fraction']:g}, source {params['source']}\n"
             f"spikes: {sources['spikes']}\nNREM: {sources['nrem']}\nON: {sources['on']}")
    fig.text(0.01, 0.005, stamp, fontsize=6, family="monospace", va="bottom")
    fig.subplots_adjust(bottom=0.1, top=0.96, left=0.05, right=0.98)
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- driver
def run_block(animal, date, block, args):
    paths = block_paths(animal, date, block, args.on_tag)
    label = f"{paths['session']}_{block}"
    # sorted spikes are only required for --source units; MUA can take the
    # sleep window from the registry instead (days with no sleep_spikes pkl)
    need = ("spikes", "nrem") if args.source == "units" else ("nrem", "mua")
    missing = [k for k in need if not paths[k].is_file()]
    if paths["on"] is None:
        missing.append(f"on ({args.on_tag})")
    if missing:
        print(f"[{label}] SKIP - missing: {missing}")
        return None

    if paths["spikes"].is_file():
        with paths["spikes"].open("rb") as f:
            spikes = pickle.load(f)
        window = spikes["window"]
        fs = float(spikes["metadata"]["sampling_frequency"])
        block_obj = SleepBlock(label=label, sampling_frequency_hz=fs,
                               day_start_sample=int(window["sleep_start_sample"]),
                               day_end_sample=int(window["sleep_end_sample"]),
                               duration_sec=float(window["window_duration_sec"]),
                               source_path=str(paths["spikes"]))
    else:
        block_obj = registry_block(animal, date, block, label)
        print(f"[{label}] no sleep_spikes pkl - sleep window from the registry "
              f"({block_obj.day_start_sample}-{block_obj.day_end_sample})")
    if args.source == "units":
        raw = {uid: rec["spike_times_sec"] for uid, rec in spikes["spike_data"].items()}
        n_total = len(raw)
    else:
        if not paths["mua"].is_file():
            print(f"[{label}] SKIP - missing MUA events: {paths['mua']}")
            return None
        raw = load_mua_trains(paths["mua"], block_obj, args.mua_shanks)
        n_total = len(raw)
    unit_ids, trains = [], []
    for uid, t in raw.items():
        t = np.asarray(t, dtype=float)
        if len(t) / block_obj.duration_sec < args.min_unit_rate_hz:
            continue
        unit_ids.append(uid)
        trains.append(t)
    del raw

    nrem = load_nrem_intervals(paths["nrem"], duration_sec=block_obj.duration_sec)
    # ON windows come from the same shanks the population is pooled from (all shanks if unset)
    on_shanks = args.on_shanks if args.on_shanks is not None else args.mua_shanks
    on, on_meta = load_mua_on_intervals(paths["on"], block_obj,
                                        min_shank_fraction=args.min_shank_fraction,
                                        shanks=on_shanks)
    on_meta["on_shanks"] = on_shanks
    nrem_on = normalize_intervals(intersect_intervals(on, nrem))

    params = {k: getattr(args, k) for k in (
        "bin_sec", "sigma_sec", "onset_z", "peak_z", "min_dur_sec", "max_dur_sec",
        "merge_gap_sec", "participation_frac", "participation_min_units",
        "participation_rule", "min_shank_fraction", "min_unit_rate_hz", "on_tag",
        "source", "mua_shanks", "on_shanks", "state")}
    params["on_shanks"] = on_shanks
    detect_kwargs = {k: params[k] for k in (
        "bin_sec", "sigma_sec", "onset_z", "peak_z", "min_dur_sec", "max_dur_sec",
        "merge_gap_sec", "participation_frac", "participation_min_units", "participation_rule")}
    state_iv = nrem if args.state == "nrem" else nrem_on
    events, trace, funnel = detect_pbes(trains, block_obj.duration_sec, state_iv,
                                        on_intervals=nrem_on, **detect_kwargs)
    on_desc = describe_on_windows(on, nrem, nrem_on, trace, args.bin_sec)
    kept = [e for e in events if e["passes_participation"]]
    nrem_on_sec = float((nrem_on[:, 1] - nrem_on[:, 0]).sum())

    summary = {
        "animal": animal, "date": date, "block": block, "session": paths["session"],
        "source": args.source, "n_units": len(trains), "n_units_total": n_total,
        "nrem_sec": on_desc["nrem_sec"], "nrem_on_sec": nrem_on_sec,
        **{k: v for k, v in on_desc.items() if k not in ("durations_ms", "on_in_nrem")},
        "state": args.state,
        "n_pbe": len(kept),
        "n_pbe_peak_in_on": int(sum(e["peak_in_on"] for e in kept)),
        "frac_pbe_peak_in_on": float(np.mean([e["peak_in_on"] for e in kept])) if kept else np.nan,
        "pbe_per_min_nrem": len(kept) / on_desc["nrem_sec"] * 60 if on_desc["nrem_sec"] else np.nan,
        "pbe_per_sec_on": len(kept) / nrem_on_sec if nrem_on_sec else np.nan,
        "frac_on_windows_with_pbe": (
            float(np.mean([np.any([(e["peak_time_sec"] >= s) & (e["peak_time_sec"] < t) for e in kept])
                           for s, t in on_desc["on_in_nrem"]])) if kept and len(on_desc["on_in_nrem"]) else 0.0),
        "pbe_dur_ms_median": float(np.median([e["duration_ms"] for e in kept])) if kept else np.nan,
        "pbe_frac_active_median": float(np.median([e["frac_active_units"] for e in kept])) if kept else np.nan,
        "participation_need": trace["participation_need"],
        **{f"funnel_{k}": v for k, v in funnel.items()},
    }
    print(f"[{label}|{args.source}] {'channels' if args.source == 'mua' else 'units'} {len(trains)}  NREM {on_desc['nrem_sec']:.0f} s  "
          f"NREM-ON {nrem_on_sec:.0f} s ({on_desc['n_on_windows_in_nrem']} windows, median "
          f"{on_desc['on_dur_ms_median']:.0f} ms)  -> {len(kept)} PBEs in {args.state} "
          f"({summary['pbe_per_min_nrem']:.1f}/min NREM, {summary['n_pbe_peak_in_on']} peak in ON)  funnel {funnel}")

    if args.dry_run:
        return summary
    from server_fallback import resolve_output_folder
    out_dir = Path(resolve_output_folder(paths["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{label}_pbe_{args.source}{shank_tag(args.mua_shanks)}{state_tag(args.state)}"
    sources = {k: str(paths[k]) for k in ("spikes", "nrem", "on")}
    if args.source == "mua":
        sources["spikes"] = str(paths["mua"])

    import csv
    with (out_dir / f"{stem}_events.csv").open("w", newline="") as f:
        cols = list(events[0]) if events else ["start_sec", "end_sec"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(events)
    with (out_dir / f"{stem}.pkl").open("wb") as f:
        pickle.dump({
            "events": events, "summary": summary, "params": params, "sources": sources,
            "sleep_block": block_obj.to_dict(), "unit_ids": unit_ids,
            "nrem_intervals_sec": nrem, "on_intervals_sec": on, "on_metadata": on_meta,
            "nrem_on_intervals_sec": nrem_on,
            "trace": {k: trace[k] for k in ("edges", "z", "rate_hz", "in_nrem_on", "in_state", "mu_hz", "sd_hz")},
            "generated": datetime.now().isoformat(),
            "reproduce": "python sleep/PBE/detect_pbe.py " + " ".join(sys.argv[1:]),
            "note": "times are seconds from the sleep block start (sleep_block.day_start_sample); "
                    "events with passes_participation=False are candidates that met every other criterion",
        }, f, protocol=4)
    if not args.no_plot:
        plot_block(out_dir / f"{stem}.png", label, events, trace, on_desc, funnel,
                   trains, unit_ids, params, sources)
    print(f"    -> {out_dir}")
    summary["out_dir"] = str(out_dir)
    return summary


def plot_summary(rows, path_png, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = [("pbe_per_min_nrem", "PBEs / min NREM"), ("pbe_per_sec_on", "PBEs / s ON"),
               ("on_fraction_of_nrem", "ON fraction of NREM"), ("pbe_dur_ms_median", "median PBE ms")]
    days = sorted({r["date"] for r in rows})
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.6))
    for ax, (key, title) in zip(axes, metrics):
        for b, color, dx in (("pre", "tab:blue", -0.18), ("post", "tab:orange", 0.18)):
            vals = [next((r[key] for r in rows if r["date"] == d and r["block"] == b), np.nan) for d in days]
            ax.bar(np.arange(len(days)) + dx, vals, width=0.36, color=color, label=b)
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels(days)
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.text(0.01, 0.01, f"{datetime.now():%Y-%m-%d %H:%M}  |  sleep/PBE/detect_pbe.py "
             + " ".join(sys.argv[1:]), fontsize=6, family="monospace")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path_png, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--animal", default="CnL46")
    p.add_argument("--days", nargs="+", default=["260729", "260730", "260731"])
    p.add_argument("--blocks", nargs="+", default=["pre", "post"])
    p.add_argument("--bin-sec", type=float, default=0.005)
    p.add_argument("--sigma-sec", type=float, default=0.015)
    p.add_argument("--onset-z", type=float, default=1.0)
    p.add_argument("--peak-z", type=float, default=3.0)
    p.add_argument("--min-dur-sec", type=float, default=0.05)
    p.add_argument("--max-dur-sec", type=float, default=0.4)
    p.add_argument("--merge-gap-sec", type=float, default=0.02)
    p.add_argument("--participation-frac", type=float, default=0.10)
    p.add_argument("--participation-min-units", type=int, default=8)
    p.add_argument("--participation-rule", choices=("max", "min"), default="max",
                   help="max: need both 10%% and 8 units (stricter); min: either")
    p.add_argument("--min-shank-fraction", type=float, default=0.5,
                   help="fraction of shanks that must be ON for a probe-wide ON window")
    p.add_argument("--min-unit-rate-hz", type=float, default=0.0,
                   help="drop units below this whole-block rate")
    p.add_argument("--on-tag", default=DEFAULT_ON_TAG)
    p.add_argument("--source", choices=("units", "mua"), default="units",
                   help="units: sorted sleep spikes; mua: per-channel MUA threshold crossings "
                        "from find_sleep_mua.py (each channel counts as one unit)")
    p.add_argument("--mua-shanks", type=int, nargs="+", default=None,
                   help="with --source mua: only pool these shanks (default all), e.g. one "
                        "probe; output files get a _sh<first>-<last> suffix")
    p.add_argument("--state", choices=("nrem", "nrem_on"), default="nrem",
                   help="nrem: z baseline and candidates from all NREM (default); "
                        "nrem_on: only NREM inside MUA ON windows (writes *_ongated files)")
    p.add_argument("--on-shanks", type=int, nargs="+", default=None,
                   help="shanks whose ON windows define ON (default: --mua-shanks, else all)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="detect and print, write nothing")
    args = p.parse_args(argv)

    install_pickle_compat()
    rows = [s for d in args.days for b in args.blocks
            if (s := run_block(args.animal, d, b, args)) is not None]
    if not rows or args.dry_run:
        return 0
    from server_fallback import resolve_output_folder
    rec = Path(block_paths(args.animal, args.days[0], args.blocks[0])["rec_folder"])
    out = Path(resolve_output_folder(rec.parents[1] / "PBE_summary"))
    out.mkdir(parents=True, exist_ok=True)
    stem = (f"{args.animal}_{'_'.join(args.days)}_pbe_{args.source}{shank_tag(args.mua_shanks)}"
            f"{state_tag(args.state)}_summary")
    import csv
    with (out / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_summary(rows, out / f"{stem}.png", args)
    print(f"summary -> {out / stem}.csv/.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
