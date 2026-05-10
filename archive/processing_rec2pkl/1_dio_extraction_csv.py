# ================================================================
# dio_split_by_gap.py
# Cleans DIO from .rec, finds long gaps to split into experiments,
# (NEW) deglitches: removes short highs and merges short lows,
# aligns CSVs in given order (if any), and saves each segment as NPZ.
# Each NPZ includes orientation, spatial_freq, contrast (NaN if absent).
# ================================================================

from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import your DIO reading helpers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from process_func.DIO import get_dio_folders, concatenate_din_data  # noqa: E402

# ============================
# 🔧 USER INPUT SECTION
# ============================
save_dir   = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\Albert\20251031_psv;hf;sphr;2sides\CnL39SG")
rec_folder = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\Albert\20251031_psv;hf;sphr;2sides\CnL39SG\CnL39SG_20251031_085159.rec")

csv_list = [
    Path(r"\\10.129.151.108\xieluanlabs\xl_cl\Albert\20251022_psv;hf;2grtings\20251022_logs\CnL39_0_90_two_grating_passive_static_20251022_152535.csv"),
    Path(r"\\10.129.151.108\xieluanlabs\xl_cl\Albert\20251022_psv;hf;2grtings\20251022_logs\CnL39_45_135_two_grating_passive_static_20251022_154202.csv"),
]

fs            = 30000       # sampling rate (Hz)
pd_channel    = 3           # photodiode channel
gap_threshold = 10.0        # seconds between highs defining a new segment
min_high_s    = 0.20        # any HIGH shorter than this is considered a glitch → remove/merge
min_low_s     = 0.05        # any LOW shorter than this is considered a glitch → merge highs

# -----------------------
# Helper: robust DIO cleaning (remove short highs, merge short lows)
# -----------------------
def _deglitch_edges(r_s: np.ndarray, f_s: np.ndarray,
                    min_high_s: float, min_low_s: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Given alternating rising/falling times (seconds), repeatedly:
      - remove short highs: drop any (r_i, f_i) with (f_i - r_i) < min_high_s
      - merge short lows: fuse neighboring highs when low gap < min_low_s
    until no changes.

    Returns new (r, f) and a report dict.
    """
    report = {"removed_short_highs": 0, "merged_short_lows": 0, "iterations": 0}

    r = r_s.copy()
    f = f_s.copy()
    if r.size == 0 or f.size == 0:
        return r, f, report

    # ensure alternating and equal length
    if f[0] < r[0]:
        f = f[1:]
    n = min(r.size, f.size)
    r, f = r[:n], f[:n]

    changed = True
    while changed:
        changed = False
        report["iterations"] += 1

        # --- 1) remove short highs ---
        dur = f - r
        short_idx = np.where(dur < min_high_s)[0]
        if short_idx.size:
            mask = np.ones(len(r), dtype=bool)
            mask[short_idx] = False  # drop those high pulses entirely
            r = r[mask]
            f = f[mask]
            report["removed_short_highs"] += int(short_idx.size)
            changed = True
            if r.size == 0 or f.size == 0:
                break

        # --- 2) merge short lows ---
        if r.size > 1:
            low = r[1:] - f[:-1]
            to_merge = np.where(low < min_low_s)[0]  # positions between i and i+1
            if to_merge.size:
                # We'll build new arrays, fusing runs of consecutive short lows
                keep_r = [r[0]]
                keep_f = []
                i = 0
                while i < len(r) - 1:
                    if (r[i+1] - f[i]) < min_low_s:
                        # fuse i .. j where all intermediate lows are short
                        j = i + 1
                        while j < len(r) - 1 and (r[j+1] - f[j]) < min_low_s:
                            j += 1
                        # fused high spans from r[i] to f[j]
                        keep_f.append(f[j])
                        # if there's a next high after j, start it
                        if j + 1 < len(r):
                            keep_r.append(r[j+1])
                        i = j + 1
                        report["merged_short_lows"] += 1
                        changed = True
                    else:
                        # keep high i as-is
                        keep_f.append(f[i])
                        keep_r.append(r[i+1])
                        i += 1
                # trim equal lengths
                if len(keep_f) > len(keep_r):
                    keep_f = keep_f[:len(keep_r)]
                r = np.asarray(keep_r[:len(keep_f)], dtype=float)
                f = np.asarray(keep_f, dtype=float)

    return r, f, report


# -----------------------
# Helper: extract raw DIO from .rec
# -----------------------
def extract_raw_dio_from_rec(rec_path: Path, fs: float, pd_channel: int):
    dio_folders = sorted(get_dio_folders(rec_path), key=lambda p: p.name)
    pd_time, pd_state = concatenate_din_data(dio_folders, pd_channel)
    pd_time = np.asarray(pd_time, float) / fs
    pd_state = np.asarray(pd_state).astype(np.int8)
    ds = np.diff(pd_state, prepend=pd_state[0])
    r_idx = np.where(ds == 1)[0]
    f_idx = np.where(ds == -1)[0]
    r_s = pd_time[r_idx]
    f_s = pd_time[f_idx]
    return r_s, f_s


# -----------------------
# QA gap plot
# -----------------------
def plot_segment(r_s, gap_threshold_s, out_path):
    diffs = np.diff(r_s) if len(r_s) > 1 else np.array([])
    plt.figure(figsize=(7, 3))
    if diffs.size:
        plt.plot(diffs, label="Δ between highs (s)")
        plt.axhline(gap_threshold_s, color="r", linestyle="--", label="Gap threshold")
        plt.xlabel("Trial #")
        plt.ylabel("Δt (s)")
        plt.title("DIO gaps / segment detection")
        plt.legend()
    else:
        plt.title("DIO gaps / segment detection (single trial)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# -----------------------
# Main logic
# -----------------------
def main():
    # ---- Load raw edges ----
    r_raw, f_raw = extract_raw_dio_from_rec(rec_folder, fs, pd_channel)
    if r_raw.size == 0 or f_raw.size == 0:
        raise RuntimeError("No DIO edges found on this channel.")
    print(f"[DIO] Raw highs={min(len(r_raw), len(f_raw))}")

    # ---- Deglitch: remove short highs, merge short lows (repeat until stable) ----
    rising_s, falling_s, rep = _deglitch_edges(r_raw, f_raw, min_high_s, min_low_s)
    print(f"[DIO] Cleaned highs={len(rising_s)}  | removed_short_highs={rep['removed_short_highs']} "
          f"| merged_short_lows={rep['merged_short_lows']} | iters={rep['iterations']}")

    # ---- Detect segments by long gaps ----
    gap_idx = np.where(np.diff(rising_s) > gap_threshold)[0]
    segment_indices = np.split(np.arange(len(rising_s)), gap_idx + 1)
    n_segments = len(segment_indices)
    print(f"[DIO] Found {n_segments} segment(s) with gap > {gap_threshold}s")

    save_dir.mkdir(parents=True, exist_ok=True)
    plot_segment(rising_s, gap_threshold, save_dir / "dio_gap_detection.png")

    # ---- Loop through each segment ----
    for seg_idx, seg_range in enumerate(segment_indices, 1):
        r_seg = rising_s[seg_range]
        f_seg = falling_s[seg_range]
        n_trials = len(r_seg)
        print(f"[Segment {seg_idx}] Trials={n_trials}, start={r_seg[0]:.2f}s")

        # Per-segment CSV metadata (optional)
        csv_path = csv_list[seg_idx - 1] if seg_idx <= len(csv_list) else None
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"  [CSV] Loaded {csv_path.name} | Columns: {list(df.columns)}")
            orientations = (df["left_ori"].to_numpy(float) if "left_ori" in df.columns
                            else np.full(n_trials, np.nan))
            spatial_freq = (df["spatial_freq"].to_numpy(float) if "spatial_freq" in df.columns
                            else np.full(n_trials, np.nan))
            contrast = (df["contrast"].to_numpy(float) if "contrast" in df.columns
                        else np.full(n_trials, np.nan))
            name = f"{csv_path.stem}_DIO_segment.npz"
        else:
            print(f"  [No CSV] Assigning NaN metadata for segment {seg_idx}")
            orientations = np.full(n_trials, np.nan)
            spatial_freq = np.full(n_trials, np.nan)
            contrast     = np.full(n_trials, np.nan)
            name = f"segment_{seg_idx:02d}.npz"

        # Save NPZ
        npz_path = save_dir / name
        np.savez_compressed(
            npz_path,
            rising_times=r_seg,
            falling_times=f_seg,
            fs=fs,
            dio_start_index=int(seg_range[0]),
            segment_index=seg_idx,
            n_trials=n_trials,
            orientations=orientations,
            spatial_freq=spatial_freq,
            contrast=contrast,
            deglitch_report=rep,
            params=dict(min_high_s=float(min_high_s),
                        min_low_s=float(min_low_s),
                        gap_threshold=float(gap_threshold)),
        )
        print(f"  Saved {npz_path.name}")

        # quick per-segment Δt plot
        if n_trials > 1:
            plt.figure(figsize=(6, 3))
            plt.plot(np.diff(r_seg), marker="o")
            plt.title(f"Segment {seg_idx} ({n_trials} trials)")
            plt.xlabel("Trial #")
            plt.ylabel("Δt (s)")
            plt.tight_layout()
            plt.savefig(save_dir / f"segment_{seg_idx:02d}_gap.png", dpi=120)
            plt.close()

    print(f"\n[Done] Saved {n_segments} segment NPZ files in {save_dir}")


if __name__ == "__main__":
    main()
