# ================================================================
# Compare PsychoPy CSV timing vs. ephys DIO (.rec) to verify sync.
# Saves cleaned DIO edges as "<csv_name>_DIO_cleaned.npz" (includes fs).
# ================================================================

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import os, sys

# --- Make local package importable (process_func/...) ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from process_func.DIO import get_dio_folders, concatenate_din_data

# =========================
# User config
# =========================
rec_folder = Path(r"L:\xl_cl\Albert_passive_grating\CnL42SG_20251022_160759.rec")
csv_path   = Path(r"L:\xl_cl\Albert\20251022_logs\CnL42_0_90_two_grating_passive_static_20251022_160842.csv")
save_dir   = Path(r"L:\xl_cl\Albert\20251022_dio")  # or None to use csv folder

fs = 30000        # Hz
pd_channel = 3

# Debounce thresholds (seconds)
MIN_HIGH_S = 0.20
MIN_LOW_S  = 0.05

# Alignment / comparison
MAX_ALIGN_GAP_S = 0.50
ITI_TOL_S       = 0.050

# If ON duration is constant and you want a QA check; else None
EXPECTED_ON_S   = None

# =========================
# Load CSV schedule
# =========================
csv_on, csv_off = [], []
with open(csv_path, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        csv_on.append(float(row["stim_on_s"]))
        csv_off.append(float(row["stim_off_s"]))

csv_on = np.asarray(csv_on, float)
csv_off = np.asarray(csv_off, float)

# Between-trial ITI = current ON - previous OFF
csv_off_prev = np.r_[np.nan, csv_off[:-1]]
csv_iti = csv_on - csv_off_prev

# drop first NaN for comparisons/plots
valid = ~np.isnan(csv_iti)
csv_on_v   = csv_on[valid]
csv_iti_v  = csv_iti[valid]
print(f"CSV trials: {len(csv_on)} | mean ITI={np.nanmean(csv_iti):.3f}s (on[i] - off[i-1])")

# =========================
# Load DIO data and extract edges
# =========================
dio_folders = sorted(get_dio_folders(rec_folder), key=lambda p: p.name)
pd_time, pd_state = concatenate_din_data(dio_folders, pd_channel)
pd_time  = np.asarray(pd_time, float) / fs
pd_state = np.asarray(pd_state)

# Event stream (0/1 per row) vs dense edges
uniq = np.unique(pd_state)
is_event_stream = set(uniq.tolist()).issubset({0, 1})

if is_event_stream:
    r_idx = np.where(pd_state == 1)[0]
    f_idx = np.where(pd_state == 0)[0]
    rising_s  = pd_time[r_idx]
    falling_s = pd_time[f_idx]
else:
    s  = pd_state.astype(np.int8)
    ds = np.diff(s, prepend=s[0])
    r_idx = np.where(ds == 1)[0]
    f_idx = np.where(ds == -1)[0]
    rising_s  = pd_time[r_idx]
    falling_s = pd_time[f_idx]

# =========================
# Debounce (pair, drop short highs, merge short lows)
# =========================
r = np.asarray(rising_s, float)
f = np.asarray(falling_s, float)
if r.size == 0 or f.size == 0:
    raise RuntimeError("No DIO edges found on the specified channel.")

# ensure first is rising
if f[0] < r[0]:
    f = f[1:]
n = min(r.size, f.size)
r, f = r[:n], f[:n]

# drop short highs
keep = (f - r) >= MIN_HIGH_S
r, f = r[keep], f[keep]

# merge short lows
if r.size > 1:
    low = r[1:] - f[:-1]
    to_merge = np.where(low < MIN_LOW_S)[0]
    if to_merge.size:
        r_new = [r[0]]
        f_new = []
        i = 0
        while i < len(r) - 1:
            if i in set(to_merge.tolist()):
                j = i
                while j < len(low) and low[j] < MIN_LOW_S:
                    j += 1
                f_new.append(f[j])
                if j + 1 < len(r):
                    r_new.append(r[j + 1])
                i = j + 1
            else:
                f_new.append(f[i])
                r_new.append(r[i + 1])
                i += 1
        r = np.array(r_new[:len(f_new)], float)
        f = np.array(f_new, float)

rising_s, falling_s = r, f
print(f"DIO highs (cleaned): {len(rising_s)}")

# =========================
# Align DIO edges to CSV ON times
# =========================
aligned_r, aligned_f = [], []
for t in csv_on:  # align to all; we’ll filter with `valid` below
    j = np.searchsorted(rising_s, t)
    candidates = []
    if j > 0: candidates.append(j - 1)
    if j < rising_s.size: candidates.append(j)
    if not candidates:
        continue
    best = min(candidates, key=lambda k: abs(rising_s[k] - t))
    if abs(rising_s[best] - t) <= MAX_ALIGN_GAP_S:
        aligned_r.append(rising_s[best])
        aligned_f.append(falling_s[best])

aligned_r = np.asarray(aligned_r, float)
aligned_f = np.asarray(aligned_f, float)
print(f"Aligned highs: {len(aligned_r)} (of {len(csv_on)} CSV trials)")

# Filter aligned series to the same valid range as csv_iti_v (drop first)
if aligned_r.size >= 2:
    dio_iti = aligned_r[1:] - aligned_f[:-1]
    n = min(len(dio_iti), len(csv_iti_v))
    diffs = dio_iti[:n] - csv_iti_v[:n]
    print(f"ITI diffs (DIO - CSV): mean={np.mean(diffs):.4f}s, std={np.std(diffs):.4f}s, "
          f"max|diff|={np.max(np.abs(diffs)):.4f}s")
else:
    print("Not enough aligned highs to compute ITI diffs.")
    dio_iti = np.array([]); n = 0

# =========================
# Optional ON-duration QA
# =========================
if EXPECTED_ON_S is not None and aligned_r.size > 0:
    dio_on = aligned_f - aligned_r
    plt.figure()
    plt.plot(dio_on, label="DIO ON (s)")
    plt.hlines(EXPECTED_ON_S, 0, len(dio_on) - 1, linestyles='dashed', label="Expected ON")
    plt.title("Stimulus ON durations (aligned DIO)")
    plt.xlabel("Trial"); plt.ylabel("Seconds"); plt.legend(); plt.tight_layout(); plt.show()
else:
    print("ON-duration comparison skipped (EXPECTED_ON_S=None).")

# =========================
# Visualize ITI sequence
# =========================
if n > 0:
    plt.figure()
    plt.plot(dio_iti[:n], label="DIO ITI (s)")
    plt.plot(csv_iti_v[:n], label="CSV ITI (s)", alpha=0.7)
    plt.title("ITI durations: DIO (aligned) vs CSV")
    plt.xlabel("Between-trial index"); plt.ylabel("Seconds")
    plt.legend(); plt.tight_layout(); plt.show()

# =========================
# Save cleaned edges + fs
# =========================
out_dir = save_dir if save_dir else csv_path.parent
out_dir.mkdir(parents=True, exist_ok=True)
save_path = out_dir / f"{csv_path.stem}_DIO_cleaned.npz"

np.savez_compressed(
    save_path,
    rising_times=rising_s,
    falling_times=falling_s,
    fs=fs  # include sampling rate for downstream use
)

print(f"Saved cleaned edges to {save_path}")
print(f"Including sampling rate fs={fs} Hz")
