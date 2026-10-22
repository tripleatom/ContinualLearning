"""
Learning curves for the freely-moving discrimination task (CnL45 / CnL46 cohort).

Reads every per-session task JSON under EXP_DIR/<animal>/<YYMMDD>/:
  <animal>_<date>_SessionNNN_Data.json     -> spatial_discrimination (no grating)
  <animal>_<date>_SessionNNN_Grating.json  -> grating_discrimination
Each trial is classified by its stimulus, not by the file:
  A  = orientation  (rewarded 45 deg vs 315 deg, both SF 0.08)
  B  = spatial freq (ori 0 deg, SF 0.04 vs 0.16)
so interleaved A+B sessions are split into their A and B trials.
Sessions from the same day are pooled per task. Trials with correct=None are dropped.

Panels per animal: % correct per day, P(choose left), trials/day.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_DIR = "//10.129.151.88/xieluanlabs2/xl_cl/experiment_data"
OUT_DIR = "//10.129.151.88/xieluanlabs2/xl_cl/sortout/behavior_learning_curves"
ANIMALS = ["CnL45", "CnL46"]
CRITERION = 0.75
MIN_TRIALS_PLOT = 10          # days with fewer trials of a task are drawn hollow
HIDE_FROM_FIGURE = {          # (animal, date, task) kept in the CSV but not plotted
    ("CnL45", "2026-09-02", "spatial"),   # spatial task re-run after grating training
    ("CnL46", "2026-07-19", "spatial"),
}

TASKS = {                      # key: (label, color)
    "spatial": ("Spatial pretraining (no grating)", "#8a8984"),
    "A": ("A: orientation 45° vs 315° (SF 0.08)", "#2a78d6"),
    "B": ("B: spatial freq 0.04 vs 0.16 (ori 0°)", "#eb6834"),
}


def classify(trial, session_label):
    if session_label == "spatial_discrimination":
        return "spatial"
    if trial["leftSpatialFreq"] != trial["rightSpatialFreq"]:
        return "B"
    if trial["rewardedOrientation"] != trial["nonRewardedOrientation"]:
        return "A"
    return "other"


def rewarded_sf(trial):
    return trial["leftSpatialFreq"] if trial["rewardedOnLeft"] else trial["rightSpatialFreq"]


def load_animal(animal):
    """Return {date: {task: dict(n, k, n_left, files)}} and the set of rewarded SFs for B."""
    days = defaultdict(lambda: defaultdict(lambda: dict(n=0, k=0, n_left=0, files=set())))
    b_rewarded_sf = set()
    pat = f"{EXP_DIR}/{animal}/*/{animal}_*_Session*_*.json"
    for f in sorted(glob.glob(pat)):
        base = os.path.basename(f)
        if base.startswith("._") or not re.search(r"_(Data|Grating)\.json$", base):
            continue
        with open(f) as fh:
            d = json.load(fh)
        date = d["experimentDate"]
        for t in d["trials"]:
            if t.get("correct") is None or t.get("choice") not in ("left", "right"):
                continue
            task = classify(t, d["sessionLabel"])
            if task == "B":
                b_rewarded_sf.add(round(rewarded_sf(t), 3))
            s = days[date][task]
            s["n"] += 1
            s["k"] += bool(t["correct"])
            s["n_left"] += t["choice"] == "left"
            s["files"].add(base)
    return days, b_rewarded_sf


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {a: load_animal(a) for a in ANIMALS}

    fig, axes = plt.subplots(3, len(ANIMALS), figsize=(7.5 * len(ANIMALS), 9),
                             sharex="col", gridspec_kw=dict(height_ratios=[3, 1.2, 1.2]))
    rows = []
    for col, animal in enumerate(ANIMALS):
        all_days, b_sf = data[animal]
        for d in sorted(all_days):
            for task, s in all_days[d].items():
                rows.append((animal, d, task, s["n"], s["k"], s["k"] / s["n"],
                             s["n_left"] / s["n"], "+".join(sorted(s["files"]))))
        days = {d: {t: s for t, s in ts.items() if (animal, d, t) not in HIDE_FROM_FIGURE}
                for d, ts in all_days.items()}
        days = {d: ts for d, ts in days.items() if ts}
        dates = sorted(days)
        x_of = {d: i for i, d in enumerate(dates)}
        # spread tasks that share a day so markers/bars don't overlap
        order = list(TASKS)
        slot = {(d, t): j - (len(days[d]) - 1) / 2
                for d in dates for j, t in enumerate(sorted(days[d], key=order.index))}
        ax_acc, ax_bias, ax_n = axes[:, col]

        for task, (label, color) in TASKS.items():
            tdates = [d for d in dates if task in days[d]]
            if not tdates:
                continue
            xs = [x_of[d] for d in tdates]
            sl = np.array([slot[(d, task)] for d in tdates])
            st = [days[d][task] for d in tdates]
            n = np.array([s["n"] for s in st])
            p = np.array([s["k"] for s in st]) / n
            pl = np.array([s["n_left"] for s in st]) / n
            if task == "B" and b_sf:
                label += f"; rewarded SF {sorted(b_sf)}"

            # break the line where the task was not run on consecutive plotted days
            xs = np.array(xs)
            seg = np.split(np.arange(len(xs)), np.where(np.diff(xs) > 1)[0] + 1)
            xj = xs + 0.15 * sl
            for j, idx in enumerate(seg):
                ax_acc.plot(xj[idx], p[idx], "-", color=color, lw=2,
                            label=label if j == 0 else None, zorder=3)
                ax_bias.plot(xj[idx], pl[idx], "-", color=color, lw=2, zorder=3)
            solid = n >= MIN_TRIALS_PLOT
            for ax, y in ((ax_acc, p), (ax_bias, pl)):
                ax.scatter(xj[solid], y[solid], s=40, color=color, edgecolor="white",
                           linewidth=1.5, zorder=4)
                ax.scatter(xj[~solid], y[~solid], s=40, facecolor="white", edgecolor=color,
                           linewidth=1.5, zorder=4)
            ax_n.bar(xs + 0.3 * sl, n, width=[0.3 if len(days[d]) > 1 else 0.6 for d in tdates],
                     color=color, edgecolor="white", linewidth=1)

        ax_acc.axhline(0.5, color="#8a8984", lw=1, ls=":", zorder=1)
        ax_acc.axhline(CRITERION, color="#52514e", lw=1, ls="--", zorder=1)
        ax_acc.text(-0.6, CRITERION + 0.01, f"{CRITERION:.0%}", ha="left",
                    va="bottom", fontsize=8, color="#52514e")
        ax_acc.set_ylim(0.3, 1.02)
        ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax_acc.set_title(animal, fontsize=13, fontweight="bold", loc="left")
        ax_acc.set_ylabel("% correct")

        ax_bias.axhline(0.5, color="#8a8984", lw=1, ls=":")
        ax_bias.set_ylim(0, 1)
        ax_bias.set_ylabel("P(choose left)")
        ax_n.set_ylabel("trials / day")

        ax_n.set_xticks(range(len(dates)))
        ax_n.set_xticklabels([datetime.strptime(d, "%Y-%m-%d").strftime("%m-%d") for d in dates],
                             rotation=60, ha="right", fontsize=8)
        ax_n.set_xlabel("recording day")
        for ax in (ax_acc, ax_bias, ax_n):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#e4e3dd", lw=0.6)
            ax.set_axisbelow(True)

    fig.suptitle("Discrimination-task learning curves", fontsize=14, fontweight="bold", x=0.01,
                 ha="left")
    stamp = (f"{os.path.abspath(__file__)} | {datetime.now():%Y-%m-%d %H:%M} | "
             f"src {EXP_DIR}/<animal>/*/*_(Data|Grating).json | "
             f"same-day sessions pooled per task; task by stimulus; "
             f"hollow = n<{MIN_TRIALS_PLOT}; not plotted: {', '.join(' '.join(h) for h in sorted(HIDE_FROM_FIGURE))}")
    fig.text(0.01, 0.003, stamp, fontsize=6, color="#52514e", ha="left", va="bottom")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.995, 0.995))
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    out_png = f"{OUT_DIR}/learning_curve_{'_'.join(ANIMALS)}.png"
    fig.savefig(out_png, dpi=200, metadata={"Description": stamp})
    fig.savefig(out_png.replace(".png", ".pdf"), metadata={"Subject": stamp})

    out_csv = out_png.replace(".png", ".csv")
    with open(out_csv, "w") as fh:
        fh.write("animal,date,task,n_trials,n_correct,frac_correct,frac_choose_left,files\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.4f},{r[6]:.4f},{r[7]}\n")
    print(f"Saved {out_png}\n      {out_csv}")


if __name__ == "__main__":
    sys.exit(main())
