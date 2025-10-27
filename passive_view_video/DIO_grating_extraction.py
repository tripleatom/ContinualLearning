# dio_csv_compare.py
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from process_func.DIO import get_dio_folders, concatenate_din_data

# =========================
# User config
# =========================
rec_folder = Path(r"D:\cl\ephys\CnL42SG_20251022_160759.rec") 
csv_path   = Path(r"C:\Users\Windows\Documents\GitHub\ContinualLearning\passive_view_video\logs\CnL42_0_90_two_grating_passive_static_20251022_160842.csv")
fs = 30000        # Hz; used to convert pd_time to seconds (force conversion below)
pd_channel = 3

# Debounce thresholds (seconds)
MIN_HIGH_S = 0.20   # "real" screen-on should exceed this; raise to ~0.9 if ON≈1s
MIN_LOW_S  = 0.05   # low gaps shorter than this are jitter; set << true ITI

# Alignment and comparison
MAX_ALIGN_GAP_S = 0.50   # max |DIO_rise - CSV_on| to accept a match
ITI_TOL_S       = 0.050  # ± tolerance for ITI diffs

# If ON is constant and you want to check it, set this; otherwise leave None
EXPECTED_ON_S   = None   # e.g., 1.0 or 5.0

# =========================
# CSV schedule
# =========================
def read_schedule_from_csv(path: Path):
    on, off_prev = [], []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            on.append(float(row["stim_on_s"]))
            off_prev.append(float(row["stim_off_s"]))
    on = np.asarray(on, float)
    off_prev = np.asarray(off_prev, float)
    iti = on - off_prev
    return on, off_prev, iti

# =========================
# Edge extraction helpers
# =========================
def detect_event_stream(pd_state):
    s = np.asarray(pd_state)
    uniq = np.unique(s)
    return set(uniq.tolist()).issubset({0,1})

def edges_from_dense_state(pd_state):
    s  = pd_state.astype(np.int8)
    ds = np.diff(s, prepend=s[0])
    r_idx = np.where(ds == 1)[0]
    f_idx = np.where(ds == -1)[0]
    return r_idx, f_idx

def edges_from_event_stream(pd_state):
    s = pd_state.astype(np.int8)
    r_rows = np.where(s == 1)[0]
    f_rows = np.where(s == 0)[0]
    return r_rows, f_rows

def pair_and_debounce(rising_s, falling_s, min_high_s, min_low_s):
    """Pair edges, drop short highs, and merge short lows."""
    r = np.asarray(rising_s, float)
    f = np.asarray(falling_s, float)
    if r.size == 0 or f.size == 0:
        return r[:0], f[:0]

    # ensure rising first
    if f[0] < r[0]:
        f = f[1:]
    n = min(r.size, f.size)
    r, f = r[:n], f[:n]

    # drop short highs
    high = f - r
    keep = high >= min_high_s
    r, f = r[keep], f[keep]
    if r.size == 0:
        return r, f

    # merge short lows
    low = r[1:] - f[:-1]
    to_merge = np.where(low < min_low_s)[0]
    if to_merge.size == 0:
        return r, f

    r_new = [r[0]]
    f_new = []
    short = set(to_merge.tolist())
    i = 0
    while i < len(r) - 1:
        if i in short:
            j = i
            while j < len(low) and low[j] < min_low_s:
                j += 1
            f_new.append(f[j])             # extend high across the chain
            if j + 1 < len(r):
                r_new.append(r[j+1])
            i = j + 1
        else:
            f_new.append(f[i])
            r_new.append(r[i+1])
            i += 1

    r = np.array(r_new[:len(f_new)], float)
    f = np.array(f_new, float)
    return r, f

# =========================
# Alignment (one high per CSV trial)
# =========================
def align_risings_to_csv(csv_on_s, rising_s, falling_s, max_gap_s):
    """
    For each CSV on-time, choose the nearest DIO rising within max_gap_s.
    Returns aligned rising/falling (same length as number of matches).
    """
    rising_s = np.asarray(rising_s, float)
    falling_s = np.asarray(falling_s, float)
    if rising_s.size == 0:
        return rising_s, falling_s

    # searchsorted on rising_s
    sort_idx = np.argsort(rising_s)
    rs = rising_s[sort_idx]
    fs = falling_s[sort_idx]

    aligned_r = []
    aligned_f = []

    for t in csv_on_s:
        j = np.searchsorted(rs, t)
        candidates = []
        if j > 0: candidates.append(j-1)
        if j < rs.size: candidates.append(j)
        # pick nearest
        if not candidates:
            continue
        best = min(candidates, key=lambda k: abs(rs[k]-t))
        if abs(rs[best] - t) <= max_gap_s:
            aligned_r.append(rs[best])
            aligned_f.append(fs[best])

    return np.asarray(aligned_r), np.asarray(aligned_f)

# =========================
# Comparison helper
# =========================
def report_diffs(name, diffs, tol):
    if diffs.size == 0:
        print(f"{name}: no points to compare.")
        return
    bad = np.where(np.abs(diffs) > tol)[0]
    print(f"{name}: mean={np.mean(diffs):.4f}s, std={np.std(diffs):.4f}s, max|diff|={np.max(np.abs(diffs)):.4f}s")
    if bad.size:
        print(f"{name}: {bad.size}/{diffs.size} outside ±{tol:.3f}s (first idx: {bad[:10]})")
    else:
        print(f"{name}: all within ±{tol:.3f}s")

# =========================
# Main
# =========================
if __name__ == "__main__":
    animal_id = rec_folder.name.split('.')[0].split('_')[0]
    session_id = rec_folder.name.split('.')[0]
    print(f"Processing {animal_id}/{session_id}")

    # 1) CSV schedule
    csv_on_s, csv_off_prev_s, csv_iti_s = read_schedule_from_csv(csv_path)
    print(f"CSV trials: {len(csv_on_s)} | mean ITI={np.mean(csv_iti_s):.3f}s  (on - off_prev)")

    # 2) DIO (force seconds)
    dio_folders = sorted(get_dio_folders(rec_folder), key=lambda x: x.name)
    pd_time, pd_state = concatenate_din_data(dio_folders, pd_channel)
    pd_time = np.asarray(pd_time, float) / fs      # <-- FORCE seconds
    pd_state = np.asarray(pd_state)

    # 3) Edge extraction (event stream vs dense)
    if detect_event_stream(pd_state):
        r_rows, f_rows = edges_from_event_stream(pd_state)
        rising_s  = pd_time[r_rows]
        falling_s = pd_time[f_rows]
    else:
        r_idx, f_idx = edges_from_dense_state(pd_state)
        rising_s  = pd_time[r_idx]
        falling_s = pd_time[f_idx]

    # 4) Debounce
    rising_s, falling_s = pair_and_debounce(rising_s, falling_s, MIN_HIGH_S, MIN_LOW_S)
    print(f"DIO highs (cleaned): {len(rising_s)}")

    # 5) Align 1 DIO high per CSV trial
    ar, af = align_risings_to_csv(csv_on_s, rising_s, falling_s, MAX_ALIGN_GAP_S)
    print(f"Aligned highs: {len(ar)} (of {len(csv_on_s)} CSV trials)")

    # 6) Build ITIs from aligned highs and compare to CSV
    if len(ar) >= 2:
        dio_iti_s = ar[1:] - af[:-1]
        n = min(len(dio_iti_s), len(csv_iti_s))
        report_diffs("ITI diffs (DIO - CSV)", dio_iti_s[:n] - csv_iti_s[:n], ITI_TOL_S)
    else:
        print("Not enough aligned highs to compute ITI diffs.")

    # 7) Optional ON-duration QA against constant
    if EXPECTED_ON_S is not None and len(ar) > 0:
        dio_on_s = af - ar
        report_diffs("ON diffs (DIO - EXPECTED)", dio_on_s - EXPECTED_ON_S, 0.050)
        plt.figure()
        plt.plot(dio_on_s, label="DIO ON (s)")
        plt.hlines(EXPECTED_ON_S, 0, len(dio_on_s)-1, linestyles='dashed', label="Expected ON")
        plt.title("Stimulus ON durations (aligned DIO)")
        plt.xlabel("Trial"); plt.ylabel("Seconds"); plt.legend(); plt.tight_layout(); plt.show()
    elif EXPECTED_ON_S is None:
        print("ON-duration comparison skipped (EXPECTED_ON_S=None).")

    # (Optional) visualize ITI series
    if len(ar) >= 2:
        plt.figure()
        plt.plot(dio_iti_s[:n], label="DIO ITI (s)")
        plt.plot(csv_iti_s[:n], label="CSV ITI (s)", alpha=0.7)
        plt.title("ITI durations: DIO (aligned) vs CSV")
        plt.xlabel("Between-trial index"); plt.ylabel("Seconds")
        plt.legend(); plt.tight_layout(); plt.show()

    # 8) Save cleaned (unaligned) edges
    save_path = csv_path.parent / f"{csv_path.stem}_DIO_cleaned.npz"
    np.savez_compressed(save_path, rising_times=rising_s, falling_times=falling_s)
    print(f"Saved cleaned edges to {save_path}")
