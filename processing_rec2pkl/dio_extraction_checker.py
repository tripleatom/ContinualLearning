# ================================================================
# quick_visualize_dio_before_after_splitaxes.py
# Load DIO from .rec, clean short pulses, and plot
# before vs after on separate aligned axes.
# ================================================================

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys, os

# -------------------------------------------------
# 🔧 USER EDIT SECTION
# -------------------------------------------------
rec_folder = Path(r"Z:\xl_cl\Albert\20251031_psv;hf;sphr;2sides\CnL39SG\CnL39SG_20251031_085159.rec")
pd_channel = 3          # photodiode DIN channel
fs = 30000              # sampling rate (Hz)
t0, t1 = 295.0, 310.0   # time window to view (s)

min_high_s = 0.20       # drop highs shorter than this (ON duration)
min_low_s  = 0.2       # delete LOW gaps shorter than this (blips)
# -------------------------------------------------

# Make the 'process_func' package importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from process_func.DIO import get_dio_folders, concatenate_din_data  # noqa: E402


def edges_from_state(time_s: np.ndarray, state: np.ndarray):
    """Return raw rising/falling edges from a 0/1 state trace."""
    ds = np.diff(state, prepend=state[0])
    r_raw = time_s[ds == 1]
    f_raw = time_s[ds == -1]

    # align so the first edge is rising
    if r_raw.size and f_raw.size and f_raw[0] < r_raw[0]:
        f_raw = f_raw[1:]
    m = min(len(r_raw), len(f_raw))
    return r_raw[:m], f_raw[:m]


def clean_dio_simple(r_times, f_times, min_high_s: float, min_low_s: float):
    """
    Simplest possible cleaning:
      1) Drop highs whose ON duration (fall - rise) < min_high_s
      2) For any LOW gap between consecutive highs less than min_low_s,
         delete the *falling* edge of the first high and the *rising* edge of the next high.
    Returns:
      r_clean, f_clean, log (dict with details)
    """
    r = np.asarray(r_times, float).copy()
    f = np.asarray(f_times, float).copy()

    # --- ensure paired & aligned lengths ---
    n0 = min(len(r), len(f))
    r, f = r[:n0], f[:n0]
    if n0 == 0:
        return r, f, {
            "raw_highs": 0,
            "dropped_short_high_idxs": [],
            "removed_low_blips_pairs": [],
            "clean_highs": 0,
        }

    # If first fall precedes first rise, drop that first fall
    if f[0] < r[0]:
        f = f[1:]
        n0 = min(len(r), len(f))
        r, f = r[:n0], f[:n0]

    # 1) Drop short highs
    dur = f - r
    keep_high = dur >= min_high_s
    dropped_short_highs = np.where(~keep_high)[0].tolist()
    r1 = r[keep_high]
    f1 = f[keep_high]

    # 2) Remove short-LOW blips by deleting the two edges that define the blip
    removed_pairs = []  # list of (fall_idx_in_r1f1, next_rise_idx_in_r1f1)
    if len(r1) > 1:
        low = r1[1:] - f1[:-1]
        to_remove = np.where(low < min_low_s)[0]  # remove f1[i] and r1[i+1]

        keep_r = np.ones(len(r1), dtype=bool)
        keep_f = np.ones(len(f1), dtype=bool)
        for i in to_remove:
            keep_f[i] = False     # remove fall of high i
            keep_r[i + 1] = False # remove rise of high i+1
            removed_pairs.append((int(i), int(i + 1)))

        r2 = r1[keep_r]
        f2 = f1[keep_f]
    else:
        r2, f2 = r1, f1

    log = {
        "raw_highs": int(len(r)),
        "dropped_short_high_idxs": dropped_short_highs,
        "removed_low_blips_pairs": removed_pairs,
        "clean_highs": int(len(r2)),
    }
    return r2, f2, log


# --- Load DIO from .rec ---
dio_folders = sorted(get_dio_folders(rec_folder), key=lambda p: p.name)
pd_time, pd_state = concatenate_din_data(dio_folders, pd_channel)
pd_time_s = np.asarray(pd_time, float) / fs
pd_state = np.asarray(pd_state, int)

# --- Raw edges ---
r_raw, f_raw = edges_from_state(pd_time_s, pd_state)

# --- Cleaned edges (simple drop/remove) ---
r_clean, f_clean, log = clean_dio_simple(r_raw, f_raw, min_high_s=min_high_s, min_low_s=min_low_s)

# --- Console summary ---
print(f"[clean] Raw highs:  {log['raw_highs']}")
print(f"[clean] Dropped short highs (<{min_high_s:.3f}s): {log['dropped_short_high_idxs']}")
if log["removed_low_blips_pairs"]:
    pairs = ", ".join([f"(fall of #{i}, rise of #{j})" for i, j in log["removed_low_blips_pairs"]])
    print(f"[clean] Removed short-LOW blips (<{min_low_s:.3f}s) by deleting edge pairs: {pairs}")
    print(f"[clean] Blip count: {len(log['removed_low_blips_pairs'])} | edges removed: {2*len(log['removed_low_blips_pairs'])}")
else:
    print(f"[clean] No short-LOW blips (<{min_low_s:.3f}s) found.")
print(f"[clean] Clean highs: {log['clean_highs']}  | total highs removed: {log['raw_highs'] - log['clean_highs']}")

# --- Filter to view window ---
def in_window(x): return (x >= t0) & (x <= t1)
r_raw_w   = r_raw[in_window(r_raw)]
f_raw_w   = f_raw[in_window(f_raw)]
r_clean_w = r_clean[in_window(r_clean)]
f_clean_w = f_clean[in_window(f_clean)]

# --- Plot (two aligned axes) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
fig.suptitle(f"{rec_folder.name} | {t0:.2f}-{t1:.2f} s | fs={fs/1000:.1f} kHz\n"
             f"min_high={min_high_s:.3f}s, min_low={min_low_s:.3f}s")

# Top: Raw
ax1.set_title("Raw DIO edges (before cleaning)", fontsize=10)
for t in r_raw_w:
    ax1.axvline(t, color="tab:green", lw=1.2)
for t in f_raw_w:
    ax1.axvline(t, color="tab:red", lw=1.0)
ax1.set_xlim(t0, t1); ax1.set_ylim(0, 1); ax1.set_yticks([])
ax1.legend(["rising", "falling"], loc="upper right", fontsize=8)
ax1.grid(axis="x", alpha=0.3)

# Bottom: Cleaned
ax2.set_title("Cleaned DIO edges (after simple drop/remove)", fontsize=10)
for t in r_clean_w:
    ax2.axvline(t, color="tab:green", lw=1.8)
for t in f_clean_w:
    ax2.axvline(t, color="tab:red", lw=1.8)
ax2.set_xlim(t0, t1); ax2.set_ylim(0, 1); ax2.set_yticks([])
ax2.set_xlabel("Time (s)")
ax2.legend(["rising", "falling"], loc="upper right", fontsize=8)
ax2.grid(axis="x", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
