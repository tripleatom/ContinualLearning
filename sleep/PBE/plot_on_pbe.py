r"""Figures for MUA ON periods and MUA-based candidate PBEs in one sleep block.

Reads detect_pbe.py's ``--source mua`` output, the per-channel MUA events, the
per-shank ON windows, the sleep-state scoring and one shank's LFP.

Fig 1  fig1_example_<n>_<t>.png  One example per file, shown at four time
       resolutions side by side (60 s, 10 s, 2 s, 0.5 s, each centred on the
       same PBE; a box marks the next zoom). Rows: LFP, per-shank and probe ON,
       channel MUA (binned rate image at 60/10 s, raster at 2/0.5 s), pooled z
       with PBE spans. Examples = the densest-PBE moment + random PBEs.
Fig 2  fig2_on_heatmap.png       Every NREM-ON window as a row (sorted by length),
       aligned at ON onset, coloured by pooled MUA z, PBE peaks overlaid;
       time-normalised ON profiles; PBEs per window.
Fig 3  fig3_onset_control.png    Are PBEs just the onset burst? PBE rate vs time
       since ON onset (only windows still running) and PBE position within its
       window, each against a within-window shuffle.

Usage
-----
    python sleep/PBE/plot_on_pbe.py --day 260731 --block post
    python sleep/PBE/plot_on_pbe.py --day 260729 --block pre --example-times 1234.5 2000
"""

from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_pbe as dp  # noqa: E402  (also bootstraps sleep_paths)

import numpy as np  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from reactivation.sleep_apply.sleep_states import SleepBlock  # noqa: E402

STATE_COLORS = {"NREM": "#4c72b0", "REM": "#c44e52", "WAKE": "#dd8452", "QUIET": "#8c8c8c"}
PBE_COLOR = "#2b83ba"
ON_COLOR = "#2ca02c"
# (span s, MUA image bin s or None for a raster) per zoom level
ZOOMS = ((10.0, 0.01), (2.0, None), (0.5, None))
# two probes per animal, PBEs detected on each separately; CnL43 has no shanks 3 and 5
PROBES_BY_ANIMAL = {"CnL43": {"P1": (0, 1, 2), "P2": (4, 6, 7)}}
PROBES = {"P1": (0, 1, 2, 3), "P2": (4, 5, 6, 7)}
CAT_COLORS = {"P1": "#e66101", "P2": "#5e3c99", "both": "#1b9e77"}
CAT_LABELS = {"P1": "P1 PBE", "P2": "P2 PBE", "both": "both PBE"}
Z_COLORS = {"P1": "#e66101", "P2": "#5e3c99", "all": "#1b9e77"}


# ---------------------------------------------------------------- loading
def load_inputs(animal, day, block, lfp_shank, on_tag=dp.DEFAULT_ON_TAG):
    paths = dp.block_paths(animal, day, block, on_tag)
    label = f"{paths['session']}_{block}"
    with (Path(paths["out_dir"]) / f"{label}_pbe_mua.pkl").open("rb") as stream:
        pbe = pickle.load(stream)
    blk = SleepBlock(**pbe["sleep_block"])
    probe_pbe = {}
    for name, shanks in PROBES.items():
        f = Path(paths["out_dir"]) / f"{label}_pbe_mua{dp.shank_tag(shanks)}.pkl"
        if not f.is_file():
            raise FileNotFoundError(f"{f.name} missing - run detect_pbe.py --source mua --mua-shanks "
                                    + " ".join(map(str, shanks)))
        with f.open("rb") as stream:
            probe_pbe[name] = pickle.load(stream)

    with paths["mua"].open("rb") as stream:
        mua = pickle.load(stream)
    offset = (int(mua["epoch_start_sample"]) - blk.day_start_sample) / blk.sampling_frequency_hz
    trains, shank_of = [], []
    for shank, entry in sorted(mua["shanks"].items()):
        ids = list(np.asarray(entry["channel_ids"]).tolist())
        y = np.asarray(entry["channel_locations"])[:, 1]
        for ch in sorted(entry["channel_spike_times"], key=lambda c: -y[ids.index(c)]):
            trains.append(np.asarray(entry["channel_spike_times"][ch]) + offset)
            shank_of.append(int(shank))
    del mua

    with Path(paths["on"]).open("rb") as stream:
        on_doc = pickle.load(stream)
    per_shank_on = {}
    for row in on_doc["rows"]:
        s, e = blk.day_sample_to_local_sec(np.array([row["on_start_daysample"],
                                                      row["on_end_daysample"]]))
        per_shank_on.setdefault(int(row["shank"]), []).append((s, e))
    per_shank_on = {k: np.asarray(v) for k, v in sorted(per_shank_on.items())}

    with paths["nrem"].open("rb") as stream:
        states = pickle.load(stream)

    lfp_path = Path(paths["rec_folder"]) / "low_freq" / f"{label}_sh{lfp_shank}_lfp_traces.npz"
    z = np.load(lfp_path, allow_pickle=True)
    fs_lfp = float(z["sampling_rate"])
    lfp_offset = (int(z["sleep_start_sample"]) - blk.day_start_sample) / float(z["original_fs"])
    traces = z["traces"]
    # channel with the most 0.5-4 Hz power inside NREM: the clearest slow oscillation
    nrem = pbe["nrem_intervals_sec"]
    if len(nrem):
        idx = np.concatenate([np.arange(int((s - lfp_offset) * fs_lfp), int((e - lfp_offset) * fs_lfp))
                              for s, e in nrem[:3]])
    else:
        idx = np.arange(min(len(traces), 60 * int(fs_lfp)))
    idx = idx[(idx >= 0) & (idx < len(traces))]
    sos = butter(3, [0.5, 4], btype="band", fs=fs_lfp, output="sos")
    sample = sosfiltfilt(sos, traces[idx[0]:idx[-1] + 1:5].astype(float), axis=0)
    ch = int(np.argmax(sample.var(0)))
    lfp = traces[:, ch].astype(float)
    lfp_label = (f"sh{lfp_shank} ch{int(np.asarray(z['channel_ids'])[ch])} "
                 f"(y={float(np.asarray(z['ycoord'])[ch]):.0f} um)")
    del traces, z

    return {"paths": paths, "label": label, "block": blk, "pbe": pbe, "probe_pbe": probe_pbe,
            "trains": trains, "shank_of": np.array(shank_of),
            "per_shank_on": per_shank_on, "states": states,
            "lfp": lfp, "fs_lfp": fs_lfp, "lfp_offset": lfp_offset,
            "lfp_label": lfp_label, "lfp_path": str(lfp_path)}


def kept_events(doc):
    ev = [e for e in doc["events"] if e["passes_participation"]]
    return np.array([[e["start_sec"], e["end_sec"], e["peak_time_sec"]] for e in ev]).reshape(-1, 3)


def probe_categories(d):
    """Three independent PBE detections, each its own population:

    P1   = probe 1 MUA alone (shanks 0-3),
    P2   = probe 2 MUA alone (shanks 4-7),
    both = both probes' MUA pooled as if one probe (shanks 0-7), detected on its own
           z curve - NOT built from P1/P2 overlaps. The three sets can overlap in time.
    Rows are (start, end, peak) seconds.
    """
    return {"P1": kept_events(d["probe_pbe"]["P1"]), "P2": kept_events(d["probe_pbe"]["P2"]),
            "both": kept_events(d["pbe"])}


def overlaps(a, b):
    """For each event in a: does any event in b overlap it in time?"""
    if not len(b):
        return np.zeros(len(a), bool)
    return np.array([np.any((b[:, 0] < e) & (b[:, 1] > s)) for s, e, _ in a], bool)


def stamp(fig, d, extra=""):
    p = d["pbe"]["params"]
    text = (f"{datetime.now():%Y-%m-%d %H:%M}  |  sleep/PBE/plot_on_pbe.py {' '.join(sys.argv[1:])}  |  {d['label']}  |  "
            f"MUA PBE: bin {p['bin_sec']*1e3:g} ms, sigma {p['sigma_sec']*1e3:g} ms, z + detection in {p.get('state', 'nrem_on')} bins, "
            f"onset z>{p['onset_z']:g}, peak z>{p['peak_z']:g}, {p['min_dur_sec']*1e3:g}-{p['max_dur_sec']*1e3:g} ms, "
            f"merge <{p['merge_gap_sec']*1e3:g} ms, participation {p['participation_rule']}"
            f"({p['participation_frac']:.0%},{p['participation_min_units']}) channels  |  "
            f"ON: {p['on_tag']}, probe ON >= {p['min_shank_fraction']:g} of shanks{extra}\n"
            f"MUA: {d['paths']['mua']}\nON: {d['paths']['on']}\n"
            f"NREM: {d['paths']['nrem']}  |  LFP: {d['lfp_path']} [{d['lfp_label']}]")
    fig.text(0.005, 0.003, text, fontsize=5.5, family="monospace", va="bottom")


# ---------------------------------------------------------------- fig 1
def shade(ax, intervals, t0, t1, color, alpha, y=None):
    for s, e in intervals:
        if e > t0 and s < t1:
            if y is None:
                ax.axvspan(max(s, t0), min(e, t1), color=color, alpha=alpha, lw=0)
            else:
                ax.fill_between([max(s, t0), min(e, t1)], y[0], y[1], color=color, alpha=alpha, lw=0)


def shank_ticks(ax, groups):
    bounds = np.flatnonzero(np.diff(groups)) + 1
    for b in bounds:
        ax.axhline(b, color="0.6", lw=0.4)
    centers = [(a + b) / 2 for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(groups)])]
    ax.set_yticks(centers)
    ax.set_yticklabels([f"sh{g}" for g in groups[np.r_[0, bounds]]], fontsize=6)
    ax.set_ylim(len(groups), 0)


def mua_panel(ax, d, t0, t1, bin_sec):
    if bin_sec is None:
        lw = 0.6 if t1 - t0 <= 0.6 else 0.35
        for row, train in enumerate(d["trains"]):
            t = train[(train >= t0) & (train < t1)]
            if len(t):
                ax.vlines(t, row, row + 0.9, lw=lw, color="0.1")
    else:
        edges = np.arange(t0, t1 + bin_sec / 2, bin_sec)
        img = np.array([np.histogram(t[(t >= t0) & (t < t1)], edges)[0] for t in d["trains"]], float) / bin_sec
        # each channel scaled by its own rate in this window so quiet and busy channels are comparable
        img /= np.maximum(img.mean(1, keepdims=True), 1e-9)
        ax.imshow(img, aspect="auto", cmap="Greys", vmin=0, vmax=4, interpolation="nearest",
                  extent=(t0, t1, len(d["trains"]), 0))
    shank_ticks(ax, d["shank_of"])


def lfp_seg(d, a, b, band=None):
    lfp, fs, off = d["lfp"], d["fs_lfp"], d["lfp_offset"]
    i0, i1 = max(int((a - off) * fs), 0), min(int((b - off) * fs), len(lfp))
    if band is None or i1 - i0 < 50:
        y = lfp[i0:i1]
    else:
        pad = int(3 * fs)
        j0, j1 = max(i0 - pad, 0), min(i1 + pad, len(lfp))
        y = sosfiltfilt(butter(3, band, btype="band", fs=fs, output="sos"), lfp[j0:j1])[i0 - j0:i1 - j0]
    return off + np.arange(i0, i0 + len(y)) / fs, y


def plot_example(d, events, t_center, out_png, title):
    import matplotlib.pyplot as plt
    blk = d["block"]
    cats = probe_categories(d)
    probe_on = {k: v["on_intervals_sec"] for k, v in d["probe_pbe"].items()}
    traces = {"P1": d["probe_pbe"]["P1"]["trace"], "P2": d["probe_pbe"]["P2"]["trace"],
              "all": d["pbe"]["trace"]}
    p = d["pbe"]["params"]
    n_sh = len(d["per_shank_on"])
    n_rows = n_sh + len(probe_on) + len(CAT_COLORS)   # shank ON, probe ON, then PBE tracks

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(5, len(ZOOMS), height_ratios=[1.0, 0.13 * n_rows * 3, 4.0, 1.7, 0.0001],
                          hspace=0.25, wspace=0.12, top=0.835, bottom=0.11, left=0.04, right=0.99)
    ax_h = fig.add_axes([0.04, 0.905, 0.95, 0.04])
    for name, ivs in d["states"]["state_intervals_s"].items():
        for s, e in np.asarray(ivs).reshape(-1, 2):
            ax_h.axvspan(s, e, color=STATE_COLORS.get(name, "0.5"), lw=0)
    span0 = ZOOMS[0][0]
    ax_h.axvspan(t_center - span0 / 2, t_center + span0 / 2, facecolor="none", edgecolor="k", lw=1.5)
    ax_h.set_xlim(0, blk.duration_sec)
    ax_h.set_yticks([])
    ax_h.tick_params(labelsize=7)
    ax_h.set_title(f"{d['label']}  |  {title}  |  hypnogram: "
                   + ", ".join(f"{k}" for k in STATE_COLORS) + " (blue, red, orange, grey)  |  box = 10 s view",
                   fontsize=10, loc="left")

    for col, (span, bin_sec) in enumerate(ZOOMS):
        t0, t1 = t_center - span / 2, t_center + span / 2
        nxt = ZOOMS[col + 1][0] if col + 1 < len(ZOOMS) else None
        ax_l = fig.add_subplot(gs[0, col])
        ax_on = fig.add_subplot(gs[1, col], sharex=ax_l)
        ax_m = fig.add_subplot(gs[2, col], sharex=ax_l)
        ax_z = fig.add_subplot(gs[3, col], sharex=ax_l)

        # LFP: slow band at 60 s, raw + slow band below that
        if span >= 30:
            t, y = lfp_seg(d, t0, t1, band=(0.5, 4))
            ax_l.plot(t, y, lw=0.5, color="tab:blue")
        else:
            t, y = lfp_seg(d, t0, t1)
            ax_l.plot(t, y, lw=0.5, color="k")
            t, yd = lfp_seg(d, t0, t1, band=(0.5, 4))
            ax_l.plot(t, yd, lw=1.0, color="tab:blue", alpha=0.8)
        ax_l.set_title(f"{span:g} s" + (f"  (MUA image, {bin_sec*1e3:g} ms bins)" if bin_sec else "  (MUA raster)"),
                       fontsize=10)

        for i, ivs in enumerate(d["per_shank_on"].values()):
            shade(ax_on, ivs, t0, t1, ON_COLOR, 0.75, y=(i + 0.1, i + 0.9))
        for j, (name, ivs) in enumerate(probe_on.items()):
            shade(ax_on, ivs, t0, t1, Z_COLORS[name], 0.45, y=(n_sh + j + 0.1, n_sh + j + 0.9))
        # one track per independent detection, so overlapping sets stay readable
        base = n_sh + len(probe_on)
        ax_on.axhline(base, color="0.3", lw=0.8)
        for j, (name, color) in enumerate(CAT_COLORS.items()):
            shade(ax_on, cats[name][:, :2], t0, t1, color, 0.95, y=(base + j + 0.1, base + j + 0.9))
        ax_on.set_yticks(np.arange(n_rows) + 0.5)
        ax_on.set_yticklabels([f"sh{s}" for s in d["per_shank_on"]]
                              + [f"{k} ON" for k in probe_on]
                              + [CAT_LABELS[k] for k in CAT_COLORS], fontsize=6)
        for lab, k in zip(ax_on.get_yticklabels()[base:], CAT_COLORS):
            lab.set_color(CAT_COLORS[k])
            lab.set_fontweight("bold")
        ax_on.set_ylim(n_rows, 0)

        mua_panel(ax_m, d, t0, t1, bin_sec)
        for name, color in CAT_COLORS.items():
            shade(ax_m, cats[name][:, :2], t0, t1, color, 0.13 if bin_sec is None else 0.08)
            shade(ax_z, cats[name][:, :2], t0, t1, color, 0.18)
        for name, tr in traces.items():
            c = 0.5 * (tr["edges"][:-1] + tr["edges"][1:])
            sel = (c >= t0) & (c < t1)
            ax_z.plot(c[sel], tr["z"][sel], color=Z_COLORS[name], lw=1.4 if name == "all" else 1.0,
                      label={"P1": "probe 1 (sh" + "/".join(map(str, PROBES["P1"])) + ")",
                             "P2": "probe 2 (sh" + "/".join(map(str, PROBES["P2"])) + ")",
                             "all": "both probes"}[name])
        ax_z.axhline(p["onset_z"], ls="--", lw=0.6, color="0.4")
        ax_z.axhline(p["peak_z"], ls="--", lw=0.6, color="tab:red")
        if col == 0:
            ax_z.legend(fontsize=7, loc="upper left", ncol=3)
        ax_z.set_xlabel("block time (s)", fontsize=8)

        if nxt is not None:
            for ax in (ax_l, ax_on, ax_m, ax_z):
                ax.axvspan(t_center - nxt / 2, t_center + nxt / 2, facecolor="none",
                           edgecolor="tab:red", lw=1.0, zorder=5)
        for ax in (ax_l, ax_on, ax_m, ax_z):
            ax.set_xlim(t0, t1)
            ax.tick_params(labelsize=7)
            ax.xaxis.get_major_formatter().set_useOffset(False)
        for ax in (ax_l, ax_on, ax_m):
            plt.setp(ax.get_xticklabels(), visible=False)
        if col == 0:
            ax_l.set_ylabel("LFP (uV)", fontsize=8)
            ax_on.set_ylabel("ON windows / PBE tracks", fontsize=8)
            ax_m.set_ylabel("channel MUA (shank, deep->sup.)", fontsize=8)
            ax_z.set_ylabel("pooled MUA z\n(PBE curves)", fontsize=8)

    fig.text(0.04, 0.868,
             f"LFP {d['lfp_label']}: raw (black), 0.5-4 Hz (blue)  |  ON: green = shank ON, "
             f"orange / purple = probe 1 / probe 2 ON  |  PBE = 3 separate detections: orange = probe 1 MUA, "
             f"purple = probe 2 MUA, teal = both probes' MUA pooled as one  |  MUA image (10 s): rate per channel / its mean  |  "
             f"dashed = z {p['onset_z']:g} / {p['peak_z']:g}  |  red box = next zoom", fontsize=8)
    stamp(fig, d, extra=f"  |  centred at {t_center:.3f} s")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def plot_heatmap(d, events, out_png):
    import matplotlib.pyplot as plt
    win = d["pbe"]["nrem_on_intervals_sec"]
    dur = win[:, 1] - win[:, 0]
    order = np.argsort(dur)
    win, dur = win[order], dur[order]
    tr = d["pbe"]["trace"]
    edges, z = tr["edges"], tr["z"]
    lag = np.arange(-0.2, 2.0, d["pbe"]["params"]["bin_sec"])

    by_start = np.argsort(win[:, 0])
    peaks = events[:, 2]
    ks = np.searchsorted(win[by_start, 0], peaks, side="right") - 1
    row = by_start[np.clip(ks, 0, None)]
    inside = (ks >= 0) & (peaks < win[row, 1] + 0.05)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1.4, 0.8], hspace=0.3, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    idx = np.searchsorted(edges, win[:, 0][:, None] + lag[None, :]) - 1
    mat = np.where((idx >= 0) & (idx < len(z)), z[np.clip(idx, 0, len(z) - 1)], np.nan)
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=-2, vmax=4, interpolation="nearest",
                   extent=(lag[0] * 1e3, lag[-1] * 1e3, len(win), 0))
    ax.plot(dur * 1e3, np.arange(len(win)) + 0.5, color="w", lw=0.6, label="ON end")
    ax.scatter((peaks - win[row, 0])[inside] * 1e3, row[inside] + 0.5, s=4, color="cyan", lw=0,
               label="PBE peak")
    ax.axvline(0, color="w", ls=":", lw=0.6)
    ax.set_xlabel("time from ON onset (ms)")
    ax.set_ylabel("NREM-ON window (sorted by length)")
    ax.set_title(f"pooled MUA z  |  {len(win)} NREM-ON windows, {inside.sum()} PBEs", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("z")

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(dur * 1e3, bins=np.linspace(0, 2000, 41), color=ON_COLOR, orientation="horizontal")
    ax.set_ylabel("ON window length (ms)")
    ax.set_xlabel("count")
    ax.set_title(f"ON window lengths (median {np.median(dur)*1e3:.0f} ms)", fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    frac = np.linspace(-0.2, 1.2, 141)
    q = np.quantile(dur, [0, 0.25, 0.5, 0.75, 1.0])
    for i, color in enumerate(plt.cm.viridis(np.linspace(0, 0.9, 4))):
        sel = (dur >= q[i]) & (dur <= q[i + 1])
        times = win[sel, 0][:, None] + frac[None, :] * dur[sel][:, None]
        k = np.clip(np.searchsorted(edges, times) - 1, 0, len(z) - 1)
        ax.plot(frac, z[k].mean(0), color=color, label=f"{q[i]*1e3:.0f}-{q[i+1]*1e3:.0f} ms (n={sel.sum()})")
    ax.axvline(0, ls=":", color="0.5")
    ax.axvline(1, ls=":", color="0.5")
    ax.set_xlabel("time / ON window length (0 = onset, 1 = end)")
    ax.set_ylabel("mean MUA z")
    ax.legend(fontsize=7, title="ON length quartile", title_fontsize=7)

    ax = fig.add_subplot(gs[1, 1])
    counts = np.bincount(row[inside], minlength=len(win))
    h = np.bincount(np.minimum(counts, 4), minlength=5) / len(win)
    ax.bar(range(5), h, color=PBE_COLOR)
    for i, v in enumerate(h):
        ax.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["0", "1", "2", "3", "4+"])
    ax.set_xlabel("PBEs per ON window")
    ax.set_ylabel("fraction of windows")
    stamp(fig, d)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.13)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def plot_onset_control(d, events, out_png, n_shuffle=500, seed=0):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    win = d["pbe"]["nrem_on_intervals_sec"]
    win = win[np.argsort(win[:, 0])]
    dur = win[:, 1] - win[:, 0]
    k = np.searchsorted(win[:, 0], events[:, 2], side="right") - 1
    ok = (k >= 0) & (events[:, 2] < win[np.clip(k, 0, None), 1])
    k, ev = k[ok], events[ok]
    rel_peak, rel_s, rel_e = ev[:, 2] - win[k, 0], ev[:, 0] - win[k, 0], ev[:, 1] - win[k, 0]
    d_k = dur[k]
    # shuffle: slide each PBE (length and peak offset kept) to a uniform start inside its own window
    room = np.clip(d_k - (rel_e - rel_s), 0, None)
    sh_peak = rng.uniform(0, 1, (n_shuffle, len(k))) * room[None, :] + (rel_peak - rel_s)[None, :]

    tbin = 0.05
    tb = np.arange(0, 2.0 + tbin, tbin)
    at_risk = np.array([(dur > t).sum() for t in tb[:-1]])

    def hazard(x):
        return np.histogram(x, tb)[0] / np.maximum(at_risk, 1) / tbin

    h_real, h_null = hazard(rel_peak), np.array([hazard(s) for s in sh_peak])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"wspace": 0.3})
    x = tb[:-1] * 1e3 + tbin * 500
    good = at_risk >= 30
    ax = axes[0]
    ax.fill_between(x[good], *np.percentile(h_null, [2.5, 97.5], axis=0)[:, good], color="0.8",
                    label="within-window shuffle 95%")
    ax.plot(x[good], h_real[good], "o-", ms=3, color=PBE_COLOR, label="observed")
    ax.set_xlabel("time since ON onset (ms)")
    ax.set_ylabel("PBE peaks per s of ON still running")
    ax.set_title(f"PBE rate vs time since onset ({tbin*1e3:g} ms bins, >= 30 windows)", fontsize=10)
    ax.legend(fontsize=8)

    fbins = np.linspace(0, 1, 21)
    c_real = np.histogram(rel_peak / d_k, fbins)[0]
    c_null = np.array([np.histogram(s / d_k, fbins)[0] for s in sh_peak])
    ax = axes[1]
    ax.fill_between(fbins[:-1] + 0.025, *np.percentile(c_null, [2.5, 97.5], axis=0), color="0.8",
                    label="shuffle 95%")
    ax.plot(fbins[:-1] + 0.025, c_real, "o-", color=PBE_COLOR, ms=3, label="observed")
    ax.set_xlabel("PBE peak position in ON window (0 = onset, 1 = end)")
    ax.set_ylabel("PBE count")
    ax.set_title(f"median position {np.median(rel_peak / d_k):.2f}", fontsize=10)
    ax.legend(fontsize=8)

    first = np.mean(rel_peak < 0.15)
    p_first = np.mean(sh_peak < 0.15, axis=1)
    pval = np.mean(p_first >= first)
    ax = axes[2]
    ax.hist(p_first, bins=30, color="0.7", label="shuffle")
    ax.axvline(first, color=PBE_COLOR, lw=2, label="observed")
    ax.set_xlabel("fraction of PBEs peaking < 150 ms after ON onset")
    ax.set_title(f"observed {first:.0%} vs shuffle {p_first.mean():.0%} (p = {pval:.3f})", fontsize=10)
    ax.legend(fontsize=8)
    stamp(fig, d, extra=f"  |  {len(k)} PBEs inside NREM-ON windows; shuffle n={n_shuffle}, seed={seed}")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.27)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return len(k), first, p_first.mean(), pval, float(np.median(rel_peak / d_k))


# ---------------------------------------------------------------- driver
def pick_examples(events, n_random, nrem_on=None, n_on=0, seed=0):
    """Densest-PBE moment, randomly drawn PBEs, and random NREM-ON moments (none hand-picked).

    Random ON moments are centred on the middle of a randomly drawn NREM-ON window, so
    they show typical ON activity whether or not a PBE happens there.
    """
    rng = np.random.default_rng(seed)
    times = []
    if len(events):
        peaks = np.sort(events[:, 2])
        dens = np.searchsorted(peaks, peaks + 1.0) - np.searchsorted(peaks, peaks - 1.0)
        best = int(np.argmax(dens))
        times.append(("densest PBE", float(peaks[best])))
        pool = np.setdiff1d(np.arange(len(peaks)), [best])
        for i in rng.choice(pool, size=min(n_random, len(pool)), replace=False):
            times.append(("random PBE", float(peaks[i])))
    if n_on and nrem_on is not None and len(nrem_on):
        for i in rng.choice(len(nrem_on), size=min(n_on, len(nrem_on)), replace=False):
            times.append(("random ON window", float(nrem_on[i].mean())))
    return sorted(times, key=lambda kv: kv[1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--animal", default="CnL46")
    p.add_argument("--day", default="260731")
    p.add_argument("--block", default="post", choices=("pre", "post"))
    p.add_argument("--lfp-shank", type=int, default=5)
    p.add_argument("--on-tag", default=dp.DEFAULT_ON_TAG,
                   help="tag of the ON-window pkl (f75_strict_r1hz for runs made with --no-exclude-bad)")
    p.add_argument("--example-times", type=float, nargs="*", default=None,
                   help="centre times (s) for Fig 1; default = densest-PBE moment + random PBEs")
    p.add_argument("--n-random-examples", type=int, default=5,
                   help="random PBEs to show in Fig 1")
    p.add_argument("--n-random-on", type=int, default=0,
                   help="random NREM-ON windows (PBE or not) to show in Fig 1")
    p.add_argument("--n-shuffle", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    chosen = dict(PROBES_BY_ANIMAL.get(args.animal, PROBES))   # copy before clearing
    PROBES.clear()
    PROBES.update(chosen)

    import matplotlib
    matplotlib.use("Agg")
    dp.install_pickle_compat()
    d = load_inputs(args.animal, args.day, args.block, args.lfp_shank, args.on_tag)
    events = kept_events(d["pbe"])
    cats = probe_categories(d)
    probe_events = np.vstack([cats["P1"], cats["P2"], cats["both"]])
    print("PBEs (3 separate detections): " + ", ".join(f"{k} {len(v)}" for k, v in cats.items()))
    ob = {k: overlaps(cats["both"], cats[k]).mean() for k in ("P1", "P2")}
    print(f"  'both' (pooled) PBEs that overlap a P1 PBE {ob['P1']:.0%}, a P2 PBE {ob['P2']:.0%}; "
          f"P1 PBEs overlapping a 'both' PBE {overlaps(cats['P1'], cats['both']).mean():.0%}, "
          f"P2 {overlaps(cats['P2'], cats['both']).mean():.0%}")
    out = Path(d["paths"]["out_dir"]) / "figures"
    out.mkdir(parents=True, exist_ok=True)

    examples = ([("chosen", t) for t in args.example_times] if args.example_times
                else pick_examples(probe_events, args.n_random_examples, d["pbe"]["nrem_on_intervals_sec"],
                                   args.n_random_on, args.seed))
    for old in out.glob(f"{d['label']}_fig1_*.png"):
        try:
            old.unlink()
        except PermissionError:   # open in a viewer; it gets overwritten if the name recurs
            print(f"  could not remove {old.name} (file in use) - left in place")
    def category_at(t):
        hit = [k for k in ("P1", "P2", "both")
               if np.any((cats[k][:, 0] <= t + 0.05) & (cats[k][:, 1] >= t - 0.05))]
        return "+".join(hit) if hit else "no PBE"
    examples = [(f"{k}, {category_at(t)}" if "PBE" in k else k, t) for k, t in examples]
    for n, (kind, t) in enumerate(examples, 1):
        tag = kind.replace(" ", "_").replace(",", "")
        f = out / f"{d['label']}_fig1_{n:02d}_{t:07.1f}s_{tag}.png"
        plot_example(d, events, t, f, title=f"example {n}/{len(examples)} ({kind}) at {t:.2f} s")
        print(f"fig1 -> {f}")
    f = out / f"{d['label']}_fig2_on_heatmap.png"
    plot_heatmap(d, events, f)
    print(f"fig2 -> {f}")
    f = out / f"{d['label']}_fig3_onset_control.png"
    n, obs, null, pval, med = plot_onset_control(d, events, f, args.n_shuffle, args.seed)
    print(f"fig3 -> {f}\n  {n} PBEs in ON windows | peak <150 ms after onset: observed {obs:.0%}, "
          f"shuffle {null:.0%}, p={pval:.3f} | median position {med:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
