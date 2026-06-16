"""
Schematic figure: what the visual-stimulus decoder reads.

Renders a continuous population raster (all task units) across a short stretch of
the task recording that contains a -1 stimulus epoch, an inter-trial interval
(ITI), and a +1 stimulus epoch. Epochs are shaded by class, units are sorted by
their +1-vs--1 firing preference, and a per-bin "what we decode" track shows the
population vector the classifier sees in each time bin.

Run:
    python plot_decoding_schematic.py
"""

import sys
import pickle
from datetime import datetime
from pathlib import Path

code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from params import (
    task_pkl,
    class_pos, class_neg, TASK_COL_MAP,
    prep_default_bin_ms,
)

# ----------------------------------------------------------------- #
#  Style (matches apply_merged_decoder_to_sleep_original.py)         #
# ----------------------------------------------------------------- #
FIG_DPI = 300
FIG_EXPORT_FORMATS = ("png", "pdf")
PUB_COLORS = {-1: "#2CA02C", 0: "#6E6E6E", 1: "#0072B2", "rate": "#3B4CC0"}
PUB_CLASS_NAMES = {-1: "VStim 1", 0: "ITI (No VStim)", 1: "VStim 2"}
BIN_SIZE_SEC = prep_default_bin_ms / 1000.0   # decoder bin width (default 50 ms)
PAD_SEC = 1.5                                  # context shown before/after the pair

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "savefig.dpi": FIG_DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _match(tp, class_dict, col_map, tol=1e-6):
    mask = np.ones(len(tp), dtype=bool)
    for canon_key, val in class_dict.items():
        col = col_map[canon_key]
        vals = np.array([t[col] for t in tp], dtype=float)
        mask &= np.abs(vals - float(val)) <= tol
    return mask


def _unit_preference(spike_data, unit_order, on, off, labels):
    """Mean firing rate (spikes/s) in +1 epochs minus -1 epochs, per unit.

    Used only to sort raster rows so the population difference the decoder
    exploits is visually apparent. Returns a (n_units,) array aligned to
    unit_order.
    """
    pos_epochs = [(on[i], off[i]) for i in np.where(labels == 1)[0]]
    neg_epochs = [(on[i], off[i]) for i in np.where(labels == -1)[0]]
    pos_dur = sum(b - a for a, b in pos_epochs)
    neg_dur = sum(b - a for a, b in neg_epochs)

    pref = np.zeros(len(unit_order))
    for k, u in enumerate(unit_order):
        st = np.asarray(spike_data[u]["spike_times_sec"])
        pos_n = sum(np.sum((st >= a) & (st < b)) for a, b in pos_epochs)
        neg_n = sum(np.sum((st >= a) & (st < b)) for a, b in neg_epochs)
        pref[k] = pos_n / max(pos_dur, 1e-9) - neg_n / max(neg_dur, 1e-9)
    return pref


def _find_schematic_window(on, off, labels):
    """Pick the most compact adjacent trial pair of opposite class (so the
    window holds one -1 epoch, one ITI, and one +1 epoch). Returns the
    (t_start, t_end, neg_epoch, pos_epoch, iti_epoch)."""
    best = None
    for i in range(len(on) - 1):
        if labels[i] != 0 and labels[i + 1] != 0 and labels[i] != labels[i + 1]:
            gap = on[i + 1] - off[i]
            if best is None or gap < best[0]:
                best = (gap, i)
    if best is None:
        raise RuntimeError("No adjacent +1/-1 trial pair found.")
    i = best[1]
    a, b = i, i + 1
    neg_i = a if labels[a] == -1 else b
    pos_i = a if labels[a] == 1 else b
    iti = (off[i], on[i + 1])
    t_start = on[i] - PAD_SEC
    t_end = off[i + 1] + PAD_SEC
    return t_start, t_end, (on[neg_i], off[neg_i]), (on[pos_i], off[pos_i]), iti


def _bin_label(t_center, neg_ep, pos_ep):
    if neg_ep[0] <= t_center < neg_ep[1]:
        return -1
    if pos_ep[0] <= t_center < pos_ep[1]:
        return 1
    return 0


def main():
    with open(task_pkl, "rb") as f:
        data = pickle.load(f)
    w = data["window"]
    on = np.asarray(w["trial_onsets_sec"])
    off = np.asarray(w["trial_offsets_sec"])
    tp = data["trial_params"]
    spike_data = data["spike_data"]

    is_pos = _match(tp, class_pos, TASK_COL_MAP)
    is_neg = _match(tp, class_neg, TASK_COL_MAP)
    labels = np.zeros(len(tp), dtype=int)
    labels[is_pos] = 1
    labels[is_neg] = -1

    unit_order = sorted(spike_data.keys())
    pref = _unit_preference(spike_data, unit_order, on, off, labels)
    order = np.argsort(pref)            # -1 preferring at bottom, +1 at top
    unit_order = [unit_order[i] for i in order]
    n_units = len(unit_order)

    t0, t1, neg_ep, pos_ep, iti_ep = _find_schematic_window(on, off, labels)

    # ---- raster spike data for the window -------------------------------- #
    raster = []
    for u in unit_order:
        st = np.asarray(spike_data[u]["spike_times_sec"])
        raster.append(st[(st >= t0) & (st <= t1)])

    # ---- decoder bins over the window ------------------------------------ #
    edges = np.arange(t0, t1 + BIN_SIZE_SEC, BIN_SIZE_SEC)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_labels = np.array([_bin_label(c, neg_ep, pos_ep) for c in centers])
    # population spike count per bin (the decoder's per-bin magnitude)
    pop_counts = np.zeros(len(centers))
    for sp in raster:
        pop_counts += np.histogram(sp, bins=edges)[0]

    # ===================================================================== #
    #  Figure                                                               #
    # ===================================================================== #
    fig, axes = plt.subplots(
        3, 1, figsize=(8.2, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [0.32, 3.0, 0.7], "hspace": 0.12},
    )
    ax_lab, ax_rast, ax_rate = axes

    def shade(ax):
        ax.axvspan(neg_ep[0], neg_ep[1], color=PUB_COLORS[-1], alpha=0.16, lw=0)
        ax.axvspan(pos_ep[0], pos_ep[1], color=PUB_COLORS[1], alpha=0.16, lw=0)

    # --- top: per-bin class label track (what we decode) ------------------ #
    for c in centers:
        lab = _bin_label(c, neg_ep, pos_ep)
        ax_lab.axvspan(c - BIN_SIZE_SEC / 2, c + BIN_SIZE_SEC / 2,
                       color=PUB_COLORS[lab], alpha=0.85, lw=0)
    for e in edges:
        ax_lab.axvline(e, color="white", lw=0.3)
    ax_lab.set_yticks([])
    ax_lab.set_ylabel("decode\nper bin", rotation=0, ha="right", va="center", fontsize=8)
    ax_lab.set_title(
        "What the decoder reads: population spikes → one label per "
        f"{int(prep_default_bin_ms)} ms bin  (VStim 1 / ITI / VStim 2)",
        fontsize=11,
    )
    ax_lab.spines[["top", "right", "left"]].set_visible(False)

    # --- middle: population raster ---------------------------------------- #
    shade(ax_rast)
    ax_rast.eventplot(raster, colors="0.1", lineoffsets=np.arange(n_units),
                      linelengths=0.85, linewidths=0.5)
    ax_rast.set_ylim(-1, n_units)
    ax_rast.set_ylabel(f"Units (n={n_units}, sorted by VStim 2 vs VStim 1 preference)")
    ax_rast.set_yticks([0, n_units - 1])
    ax_rast.set_yticklabels(["VStim 1 preferring", "VStim 2 preferring"], fontsize=8)
    ax_rast.spines[["top", "right"]].set_visible(False)

    # epoch annotations
    ymax = n_units
    for ep, lab in [(neg_ep, -1), (pos_ep, 1)]:
        ax_rast.text(np.mean(ep), ymax * 1.01, PUB_CLASS_NAMES[lab],
                     ha="center", va="bottom", color=PUB_COLORS[lab],
                     fontsize=9, fontweight="bold")
    ax_rast.text(np.mean(iti_ep), ymax * 1.01, PUB_CLASS_NAMES[0],
                 ha="center", va="bottom", color=PUB_COLORS[0],
                 fontsize=9, fontweight="bold")

    # --- bottom: population spike count per bin --------------------------- #
    shade(ax_rate)
    ax_rate.bar(centers, pop_counts, width=BIN_SIZE_SEC * 0.9,
                color=PUB_COLORS["rate"], linewidth=0)
    ax_rate.set_ylabel("Pop. spikes\nper bin", fontsize=8)
    ax_rate.set_xlabel("Time in task recording (s)")
    ax_rate.set_xlim(t0, t1)
    ax_rate.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        Patch(facecolor=PUB_COLORS[-1], alpha=0.85, label=PUB_CLASS_NAMES[-1]
              + f"  (left ori={class_neg['orientation']}, SF={class_neg['spatial_freq']})"),
        Patch(facecolor=PUB_COLORS[0], alpha=0.85, label=PUB_CLASS_NAMES[0]),
        Patch(facecolor=PUB_COLORS[1], alpha=0.85, label=PUB_CLASS_NAMES[1]
              + f"  (left ori={class_pos['orientation']}, SF={class_pos['spatial_freq']})"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.075))

    # ---- reproducibility stamp ------------------------------------------- #
    ts = datetime.now().astimezone()
    stamp = (
        f"Generated {ts.strftime('%Y-%m-%d %H:%M:%S %Z')} | script={Path(__file__).name}\n"
        f"task={Path(task_pkl).name} | window=[{t0:.2f}, {t1:.2f}] s | "
        f"bin={int(prep_default_bin_ms)} ms | n_units={n_units}\n"
        f"classes: +1={class_pos}, -1={class_neg}, 0=ITI | "
        f"neg_epoch=[{neg_ep[0]:.2f},{neg_ep[1]:.2f}] pos_epoch=[{pos_ep[0]:.2f},{pos_ep[1]:.2f}] "
        f"iti=[{iti_ep[0]:.2f},{iti_ep[1]:.2f}] | units sorted by +1 minus -1 mean rate"
    )
    fig.text(0.01, 0.005, stamp, ha="left", va="bottom", fontsize=5.5, color="0.35")

    fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.17)

    session = Path(task_pkl).parent.name
    out_dir = Path(task_pkl).parent / "reactivation" / f"decoding_schematic_{session}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"decoding_schematic_{ts.strftime('%Y%m%d_%H%M%S')}"
    saved = []
    for ext in FIG_EXPORT_FORMATS:
        fp = out_dir / f"{stem}.{ext}"
        fig.savefig(fp, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
        saved.append(fp)
    plt.close(fig)
    for fp in saved:
        print(f"Saved -> {fp}")


if __name__ == "__main__":
    main()
