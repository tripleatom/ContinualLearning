"""Regenerate the sleep decoding-confidence figure with an aligned LFP band.

Combines, on a shared *real recording-time* axis:
    A  decoder confidence  (+1 green, -1 blue, ITI grey)
    B  predicted class
    C  mean firing rate (spike count / bin)   -- black
    D  LFP 0.5-4 Hz (delta) band power, remapped from concatenated to real time

Candidate reactivation events (decoder confidence peaks above threshold) are
marked as vertical lines on every panel, +1 green and -1 blue.

The decoder results come from apply_merged_decoder_to_sleep_original.py
(saved under reactivation/sleep_merged_decoder_<session>/<block>/).

Why remap the delta band
------------------------
The decoder runs in the sleep pkl's spike-time frame, which is the *real*
recording time (post block: 0..7015 s, matching the LFP file's
realtime.lfp_time_s span of 22..7015 s).  The delta band trace in the LFP pkl
(`traces['delta']`, z-scored 0.5-4 Hz power) is stored in *concatenated*
(artifact-removed) time.  The mapping back to real time is exact: that trace is
sampled on `trace_time_s` (the concatenated kept axis), which is paired 1:1
with the kept real-time samples `realtime.lfp_time_s[~realtime.artifact_mask_lfp]`.
Plotting the trace against those real times re-aligns it with the decoder; the
artifact seams become NaN gaps so removed spans read as blanks.

Inputs are the saved decoding-results pkl plus the sleep spike pkl (for the
firing rate) and the LFP trace pkl.  Nothing is retrained.

Usage:
    python plot_decoding_with_lfp.py                # post block, default paths
    python plot_decoding_with_lfp.py --results ... --lfp ... --out ...
"""
from pathlib import Path
from datetime import datetime
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

# ---------------------------------------------------------------- #
#  Default paths (CnL42SG_20260313, post block)                     #
# ---------------------------------------------------------------- #
SESSION_FOLDER = r"\\10.129.151.108\xieluanlabs\xl_cl\sortout\CnL42SG\CnL42SG_20260313"
DEFAULT_RESULTS = (rf"{SESSION_FOLDER}\reactivation"
                   r"\sleep_merged_decoder_CnL42SG_20260313\post"
                   r"\sleep_decoding_results.pkl")
DEFAULT_LFP = (r"\\10.129.151.108\xieluanlabs\xl_cl\experiment_data\CnL42\260313"
               r"\CnL42SG_20260313\low_freq\spectrogram"
               r"\CnL42SG_20260313_sh7_ch009_trace_data_concat.pkl")

# Colours per request: +1 green, -1 blue, ITI grey.
COL_POS = "#2ca02c"   # +1  green
COL_NEG = "#1f77b4"   # -1  blue
COL_ITI = "0.55"      # ITI grey
COL_RATE = "black"    # mean firing rate
COL_DELTA = "black"   # delta band trace
DELTA_KEY = "delta"   # 0.5-4 Hz band in the LFP trace pkl
DELTA_DECIMATE = 10   # plot every Nth delta sample (500 Hz -> 50 Hz; band is <4 Hz)
GAP_FACTOR = 5.0      # real-time jump > GAP_FACTOR * median dt => artifact seam

# Sleep-period detection: bins with mean spike count < SLEEP_RATE_THR AND
# delta z-power > SLEEP_DELTA_THR, sustained for >= SLEEP_MIN_DUR_SEC.
SLEEP_RATE_THR = 4
SLEEP_DELTA_THR = -0.5
SLEEP_MIN_DUR_SEC = 5.0
COL_SLEEP = "#8856a7"  # sleep ribbon colour (purple, distinct from event marks)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=DEFAULT_RESULTS,
                   help="sleep_decoding_results.pkl from apply_merged_decoder_to_sleep.py")
    p.add_argument("--lfp", default=DEFAULT_LFP,
                   help="*_trace_data_concat.pkl from spectrogram_plot.py")
    p.add_argument("--out", default=None, help="output PNG path (default: next to results pkl)")
    p.add_argument("--tmin", type=float, default=None,
                   help="start of the real-time window to show (s); default = full block")
    p.add_argument("--tmax", type=float, default=None,
                   help="end of the real-time window to show (s); default = full block")
    return p.parse_args()


def mean_firing_rate(sleep_pkl, common_units, centers):
    """Reproduce X_sleep.mean(axis=1): mean spike count per bin across the
    decoder's common units, on the same uniform bin edges as `centers`."""
    bw = float(np.median(np.diff(centers)))
    edges = np.concatenate([centers - bw / 2.0, [centers[-1] + bw / 2.0]])
    with open(sleep_pkl, "rb") as f:
        spike_data = pickle.load(f)["spike_data"]
    counts = np.zeros((len(centers), len(common_units)), dtype=float)
    for j, u in enumerate(common_units):
        st = np.asarray(spike_data[u]["spike_times_sec"], dtype=float)
        counts[:, j], _ = np.histogram(st, bins=edges)
    return counts.mean(axis=1)


def delta_band_real_time(lfp_pkl, centers=None):
    """Return the z-scored 0.5-4 Hz (delta) band trace on the real recording
    time axis, with NaN inserted at artifact seams so removed spans read as gaps.
    If `centers` is given, also return the per-decoder-bin mean delta.

    `traces['delta']` is sampled on `trace_time_s` (concatenated kept axis),
    which is paired 1:1 with the kept real-time samples
    `realtime.lfp_time_s[~realtime.artifact_mask_lfp]`; plotting the trace
    against those real times re-aligns it with the decoder."""
    with open(lfp_pkl, "rb") as f:
        d = pickle.load(f)
    if DELTA_KEY not in d["traces"]:
        raise KeyError(f"'{DELTA_KEY}' not in traces {list(d['traces'])}")
    delta = np.asarray(d["traces"][DELTA_KEY], dtype=float)            # z-scored, concat order
    mask = np.asarray(d["realtime"]["artifact_mask_lfp"], dtype=bool)  # True = removed
    real_kept = np.asarray(d["realtime"]["lfp_time_s"], dtype=float)[~mask]
    if len(delta) != len(real_kept):
        raise RuntimeError("delta trace and kept lfp_time_s lengths differ; cannot map.")

    step = max(1, int(DELTA_DECIMATE))
    x = real_kept[::step]
    y = delta[::step]

    # Break the line at artifact seams (large real-time jumps) with NaN.
    dx = np.diff(x)
    med = np.median(dx)
    gaps = np.where(dx > GAP_FACTOR * med)[0]
    if gaps.size:
        y = y.copy()
        y[gaps] = np.nan

    # Per-decoder-bin mean delta (for sleep detection), on the `centers` grid.
    per_bin = None
    if centers is not None:
        bw = float(np.median(np.diff(centers)))
        edges = np.concatenate([centers - bw / 2.0, [centers[-1] + bw / 2.0]])
        idx = np.searchsorted(edges, real_kept, side="right") - 1
        ok = (idx >= 0) & (idx < len(centers))
        sums = np.zeros(len(centers)); counts = np.zeros(len(centers))
        np.add.at(sums, idx[ok], delta[ok])
        np.add.at(counts, idx[ok], 1.0)
        per_bin = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)

    return {
        "x": x, "y": y, "per_bin": per_bin,
        "channel": d["channel"], "shank": d["shank"], "session": d["session"],
        "ylim": tuple(d.get("band_ylim", (-4, 4))),
    }


def detect_sleep(centers, pop_rate, delta_per_bin,
                 rate_thr=SLEEP_RATE_THR, delta_thr=SLEEP_DELTA_THR,
                 min_dur_sec=SLEEP_MIN_DUR_SEC):
    """Return [(t_start, t_end), ...] for runs where mean spike count < rate_thr
    AND delta z-power > delta_thr, sustained at least min_dur_sec.  Bins without
    delta coverage (artifact gaps) are treated as non-sleep, which also splits
    runs across removed spans."""
    bw = float(np.median(np.diff(centers)))
    sleep = (pop_rate < rate_thr) & (delta_per_bin > delta_thr) & np.isfinite(delta_per_bin)
    min_bins = max(1, int(round(min_dur_sec / bw)))

    intervals = []
    n = len(sleep)
    i = 0
    while i < n:
        if sleep[i]:
            j = i
            while j + 1 < n and sleep[j + 1]:
                j += 1
            if (j - i + 1) >= min_bins:
                intervals.append((centers[i] - bw / 2.0, centers[j] + bw / 2.0))
            i = j + 1
        else:
            i += 1
    return intervals


def main():
    args = parse_args()
    results_path = Path(args.results)
    lfp_path = Path(args.lfp)

    with open(results_path, "rb") as f:
        r = pickle.load(f)
    centers = np.asarray(r["sleep_centers_sec"], dtype=float)
    proba = np.asarray(r["sleep_proba"], dtype=float)
    classes = list(np.asarray(r["classes"], dtype=int))
    events = r["events"]
    best = r["best"]
    thr = r["event_threshold"]
    c_pos, c_neg, c_iti = classes.index(1), classes.index(-1), classes.index(0)

    pop_rate = mean_firing_rate(r["sleep_pkl"], r["common_units"], centers)
    delta = delta_band_real_time(lfp_path, centers=centers)
    sleep_intervals = detect_sleep(centers, pop_rate, delta["per_bin"])
    total_sleep = sum(b - a for a, b in sleep_intervals)
    print(f"Sleep periods (rate<{SLEEP_RATE_THR} & delta>{SLEEP_DELTA_THR}, "
          f">={SLEEP_MIN_DUR_SEC}s): {len(sleep_intervals)} "
          f"covering {total_sleep:.0f}s of {centers[-1]-centers[0]:.0f}s")

    x_lo = centers[0] if args.tmin is None else args.tmin
    x_hi = centers[-1] if args.tmax is None else args.tmax

    # --- Figure (confidence / mean spike+sleep / delta) -------------------
    fig, axes = plt.subplots(
        3, 1, figsize=(15, 6.8), sharex=True,
        gridspec_kw={"height_ratios": [1.3, 0.9, 1.1], "hspace": 0.18},
    )

    # A: decoder confidence
    axes[0].plot(centers, proba[:, c_pos], color=COL_POS, lw=1.0)
    axes[0].plot(centers, proba[:, c_neg], color=COL_NEG, lw=1.0)
    axes[0].plot(centers, proba[:, c_iti], color=COL_ITI, lw=0.9)
    axes[0].axhline(thr, color="black", linestyle=":", lw=1.0)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Decoder\nconfidence")
    axes[0].set_title(
        f"Sleep decoding + LFP: {best['classifier']} @ {best['bin_ms']} ms "
        f"(merged CV acc={best['mean_acc']:.3f}, {r['sleep_label']} block)"
    )

    # B: mean firing rate -- black, with sleep ribbon on top
    ax_rate = axes[1]
    ax_rate.plot(centers, pop_rate, color=COL_RATE, lw=0.8)
    ax_rate.set_ylabel("Mean spike\ncount / bin")
    span = blended_transform_factory(ax_rate.transData, ax_rate.transAxes)
    for a, b in sleep_intervals:
        a, b = max(a, x_lo), min(b, x_hi)   # clip to window (bars are clip_on=False)
        if b <= a:
            continue
        ax_rate.add_patch(Rectangle((a, 1.02), b - a, 0.06, transform=span,
                                    facecolor=COL_SLEEP, edgecolor="none",
                                    clip_on=False, zorder=6))
    if sleep_intervals:
        ax_rate.text(0.0, 1.05, "sleep ", transform=ax_rate.transAxes,
                     ha="right", va="center", fontsize=7, color=COL_SLEEP)

    # C: LFP 0.5-4 Hz (delta) band power (real time)
    axd = axes[2]
    axd.plot(delta["x"], delta["y"], color=COL_DELTA, lw=0.4)
    axd.set_ylim(*delta["ylim"])
    axd.set_ylabel(f"Delta 0.5-4 Hz\nz-power (sh{delta['shank']} ch{delta['channel']})")
    axd.set_xlabel("Real recording time (s)")

    # Candidate reactivation marks on every panel: +1 green, -1 blue.
    for label, color in [(1, COL_POS), (-1, COL_NEG)]:
        for e in events[label]:
            for ax in axes:
                ax.axvline(e["time_sec"], color=color, alpha=0.5, lw=1.0, zorder=4)

    axes[0].set_xlim(x_lo, x_hi)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    windowed = args.tmin is not None or args.tmax is not None
    win_txt = f" | window {x_lo:g}-{x_hi:g}s" if windowed else ""

    # --- Reproducibility stamp -------------------------------------------
    n_pos, n_neg = len(events[1]), len(events[-1])
    repro_win = (f" --tmin {x_lo:g} --tmax {x_hi:g}") if windowed else ""
    stamp = (
        f"Generated {datetime.now():%Y-%m-%d %H:%M:%S} by plot_decoding_with_lfp.py  |  "
        f"decoder={best['classifier']} @ {best['bin_ms']}ms acc={best['mean_acc']:.3f}  |  "
        f"threshold={thr} events: +1={n_pos} -1={n_neg}{win_txt}  |  "
        f"sleep ribbon: rate<{SLEEP_RATE_THR} & delta>{SLEEP_DELTA_THR} for >={SLEEP_MIN_DUR_SEC}s "
        f"({len(sleep_intervals)} periods, {total_sleep:.0f}s)\n"
        f"delta 0.5-4 Hz remapped concat->real time (sh{delta['shank']} ch{delta['channel']}, "
        f"decimate={DELTA_DECIMATE})  |  "
        f"results={results_path.name}  lfp={lfp_path.name}  |  "
        f"reproduce: python plot_decoding_with_lfp.py "
        f"--results \"{results_path}\" --lfp \"{lfp_path}\"{repro_win}"
    )
    fig.text(0.005, 0.001, stamp, fontsize=5.5, color="0.4", ha="left", va="bottom")

    default_name = (f"sleep_decoding_with_delta_{x_lo:g}-{x_hi:g}s.png"
                    if windowed else "sleep_decoding_with_delta.png")
    out = Path(args.out) if args.out else results_path.with_name(default_name)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
