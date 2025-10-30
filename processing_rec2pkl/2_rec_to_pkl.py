# ================================================================
# Unifies ephys .rec + PsychoPy CSV (+ optional *_DIO_cleaned.npz and *_taskmeta.npz)
# and attaches spikes from Phy or SortingAnalyzer results, producing a single .pkl
# ready for downstream analysis / embeddings.
# Includes robust CSV↔DIO subsequence alignment + verification plot.
# ================================================================

from pathlib import Path
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pickle

# plotting for alignment verification
import matplotlib.pyplot as plt

# SpikeInterface
from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor


# -----------------------
# Alignment helpers
# -----------------------
def _best_csv_dio_alignment(csv_on_s, dio_rising_s, tol_s=0.02):
    """
    Find where the CSV run occurs inside a longer DIO sequence.
    Slide a window of length len(csv_on_s) across dio_rising_s and
    compare relative onset patterns (offset-invariant). Returns:
      best_j: best start index in DIO,
      diffs: per-trial |Δ| (seconds) for the chosen match
    """
    csv_on_s = np.asarray(csv_on_s, float)
    dio_rising_s = np.asarray(dio_rising_s, float)

    m = len(csv_on_s)
    n = len(dio_rising_s)
    if m == 0 or n == 0 or n < m:
        return None, None

    csv_rel = csv_on_s - csv_on_s[0]
    best_j, best_err = None, np.inf

    for j in range(0, n - m + 1):
        dio_block = dio_rising_s[j:j + m]
        dio_rel = dio_block - dio_block[0]
        diffs = np.abs(csv_rel - dio_rel)
        err = np.max(diffs)  # strict criterion; use mean if you prefer
        if err < best_err:
            best_err = err
            best_j = j

    if best_j is None:
        return None, None

    # Recompute diffs for the selected block and enforce tolerance
    dio_block = dio_rising_s[best_j:best_j + m]
    dio_rel = dio_block - dio_block[0]
    diffs = np.abs((csv_on_s - csv_on_s[0]) - dio_rel)
    if np.any(diffs > tol_s):
        return None, diffs
    return best_j, diffs


def select_trials_for_this_csv(csv_on_s, csv_off_s, rising_samples, falling_samples, fs, tol_s=0.02):
    """
    Locate the CSV run inside DIO edges and return the exact subset:
      - trial_windows in *samples* (start=rising, end=falling)
      - keep_idx: indices of the matched trials in the DIO sequence
      - diffs: per-trial |Δ| (sec) between CSV and matched DIO onsets
      - start_j: starting index into DIO rising sequence
    """
    dio_rising_s = np.asarray(rising_samples, float) / float(fs)
    dio_falling_s = np.asarray(falling_samples, float) / float(fs)

    start_j, diffs = _best_csv_dio_alignment(csv_on_s, dio_rising_s, tol_s=tol_s)
    if start_j is None:
        raise RuntimeError(
            f"Could not align CSV run inside DIO within ±{tol_s:.3f}s. "
            f"Try increasing tol_s or verify clocks."
        )

    m = len(csv_on_s)
    dio_slice = slice(start_j, start_j + m)

    chosen_rising = rising_samples[dio_slice].astype(float)
    chosen_falling = falling_samples[dio_slice].astype(float)

    trial_windows = [(float(chosen_rising[i]), float(chosen_falling[i])) for i in range(m)]
    keep_idx = np.arange(start_j, start_j + m, dtype=int)

    return trial_windows, keep_idx, diffs, start_j


# -----------------------
# Spike attachment helper
# -----------------------
def attach_phy_spikes(
    data: dict,
    sorting_root: Path,
    window_pre_s: float = 0.2,
    window_post_s: float = 2.0,
) -> None:
    """
    Recursively find Phy / sorting_analyzer results under `sorting_root`,
    extract unit spike trains, and align to trial windows in `data`.

    Populates:
      data["spike_data"][unique_unit_id] -> list of per-trial dicts
      data["unit_info"][unique_unit_id]  -> unit metadata
      data["metadata"]["fs"]             -> sampling rate (Hz), if missing
    """
    trial_windows = data["trial_info"]["trial_windows"]
    fs = data["metadata"].get("fs", None)

    # Heuristic: if numbers are very large (>> seconds) we treat as samples.
    have_samples = isinstance(trial_windows[0][0], (int, np.integer)) or (
        fs is not None and np.nanmax(np.asarray(trial_windows, float)) > 1e4
    )

    # Gather candidate sorter folders
    sorting_result_dirs = []
    for root, dirs, files in os.walk(sorting_root):
        for d in dirs:
            if d == "phy" or d.startswith("sorting_analyzer"):
                sorting_result_dirs.append(Path(root) / d)

    if not sorting_result_dirs:
        print(f"[attach_phy_spikes] No Phy/sorting_analyzer folders found under {sorting_root}")
        return

    data.setdefault("spike_data", {})
    data.setdefault("unit_info", {})

    unit_counter = 0

    for p in sorting_result_dirs:
        sorter = None
        sorter_fs = None
        try:
            if p.name == "phy":
                sorter = PhySortingExtractor(p)
                sorter_fs = sorter.sampling_frequency
                print(f"[attach_phy_spikes] Loaded Phy at {p} | fs={sorter_fs} Hz | units={len(sorter.unit_ids)}")
            else:
                sa = load_sorting_analyzer(p)  # p is the sorting_analyzer dir
                sorter = sa.sorting
                sorter_fs = sorter.sampling_frequency
                print(f"[attach_phy_spikes] Loaded SortingAnalyzer at {p} | fs={sorter_fs} Hz | units={len(sorter.unit_ids)}")
        except Exception as e:
            print(f"[attach_phy_spikes] Skipping {p}: {e}")
            continue

        # Backfill fs if needed
        if data["metadata"].get("fs") in (None, "None", "") and sorter_fs is not None:
            data["metadata"]["fs"] = float(sorter_fs)

        try:
            unit_ids = sorter.unit_ids
        except Exception:
            print(f"[attach_phy_spikes] Could not read unit_ids in {p}")
            continue

        # Best-effort quality labels
        try:
            if hasattr(sorter, "get_property"):
                qualities = sorter.get_property("quality")
                if qualities is None or len(qualities) != len(unit_ids):
                    qualities = ["good"] * len(unit_ids)
            else:
                qualities = ["good"] * len(unit_ids)
        except Exception:
            qualities = ["good"] * len(unit_ids)

        # Prefer the explicit orientations array if present
        orientations = data["trial_info"].get("orientations", None)
        left_seq = data["trial_info"].get("left_ori_seq", [])
        right_seq = data["trial_info"].get("right_ori_seq", [])

        for idx, uid in enumerate(unit_ids):
            qual = qualities[idx] if idx < len(qualities) else "unknown"
            if str(qual).lower() == "noise":
                continue

            try:
                spike_samples = sorter.get_unit_spike_train(uid)  # sample indices
            except Exception as e:
                print(f"[attach_phy_spikes] Unit {uid} spike train error: {e}")
                continue

            if spike_samples is None or len(spike_samples) == 0:
                continue

            parent_tag = p.parent.name
            unique_id = f"{parent_tag}_{p.name}_unit{uid}"

            trials_out = []
            n_trials = len(trial_windows)

            fs_eff = data["metadata"].get("fs", None) or sorter_fs
            if not have_samples and (fs_eff is None or fs_eff == 0):
                print(f"[attach_phy_spikes] Missing fs to align seconds; skipping {unique_id}")
                continue

            spike_times_s = spike_samples / float(fs_eff)

            for t_idx in range(n_trials):
                start, end = trial_windows[t_idx]

                if have_samples:
                    start_samp = int(start)
                    pre_samp = int(window_pre_s * fs_eff)
                    post_samp = int(window_post_s * fs_eff)
                    mask = (spike_samples >= start_samp - pre_samp) & (spike_samples < start_samp + post_samp)
                    spikes_rel_s = (spike_samples[mask] - start_samp) / float(fs_eff)
                else:
                    start_sec = float(start)
                    mask = (spike_times_s >= start_sec - window_pre_s) & (spike_times_s < start_sec + window_post_s)
                    spikes_rel_s = spike_times_s[mask] - start_sec

                # Orientation per trial: prefer explicit 'orientations', then left/right seq
                if isinstance(orientations, list) and t_idx < len(orientations):
                    ori_val = orientations[t_idx]
                elif isinstance(left_seq, list) and t_idx < len(left_seq):
                    ori_val = left_seq[t_idx]
                elif isinstance(right_seq, list) and t_idx < len(right_seq):
                    ori_val = right_seq[t_idx]
                else:
                    ori_val = None

                trials_out.append({
                    "trial_index": t_idx,
                    "orientation": float(ori_val) if ori_val is not None else None,
                    "spike_times": spikes_rel_s.astype(float).tolist(),
                    "spike_count": int(spikes_rel_s.size),
                    "trial_start": start,
                    "trial_end": end
                })

            data["spike_data"][unique_id] = trials_out
            data["unit_info"][unique_id] = {
                "original_unit_id": int(uid) if isinstance(uid, (int, np.integer)) else str(uid),
                "quality": str(qual),
                "sorting_path": str(p),
                "n_spikes_total": int(len(spike_samples)),
                "unit_index": int(unit_counter),
            }
            unit_counter += 1

    data.setdefault("extraction_params", {})
    data["extraction_params"].update({
        "window_pre": float(window_pre_s),
        "window_post": float(window_post_s),
        "total_units": int(unit_counter),
    })
    print(f"[attach_phy_spikes] Done. Attached {unit_counter} units.")


# -----------------------
# Main build + save
# -----------------------
if __name__ == "__main__":
    # ==== 1) User-defined paths ====
    rec_folder        = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\Albert_passive_grating\CnL42SG_20251022_160759.rec")
    csv_path          = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\Albert\20251022_logs\CnL42_0_90_two_grating_passive_static_20251022_160842.csv")
    dio_npz_path      = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\Albert\20251022_dio\CnL42_0_90_two_grating_passive_static_20251022_160842_DIO_cleaned.npz")
    taskmeta_npz_path = None   # or Path(...)
    sorting_root      = Path(r"\\10.129.151.108\xieluanlabs\xl_cl\code\sortout\CnL42SG\CnL42SG_20251022_160759")
    output_dir        = Path(r"C:\Users\Windows\Desktop\Albert\251022\CnL42SG")

    # ==== 2) Parse session IDs ====
    rec_name   = rec_folder.name.replace(".rec", "")
    parts      = rec_name.split("_")
    animal_id  = parts[0]
    session_id = rec_name
    print(f"\n[Info] Processing session: {animal_id}/{session_id}")

    # ==== 3) Load CSV ====
    df = pd.read_csv(csv_path)
    required_cols = {"stim_on_s", "stim_off_s"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns {required_cols}")

    # --- Orientation from CSV (left_ori) ---
    if "left_ori" in df.columns:
        orientations = df["left_ori"].to_numpy(float)
    else:
        print("[Warning] No 'left_ori' column found in CSV — assigning zeros.")
        orientations = np.zeros(len(df), dtype=float)

    csv_on  = df["stim_on_s"].to_numpy(float)
    csv_off = df["stim_off_s"].to_numpy(float)

    csv_off_prev = np.r_[np.nan, csv_off[:-1]]
    csv_iti = csv_on - csv_off_prev
    mean_iti = np.nanmean(csv_iti)
    print(f"[Info] Loaded {len(csv_on)} CSV trials | mean ITI = {mean_iti:.3f}s")

    # ==== 4) Load DIO ====
    dio_guess = Path(dio_npz_path) if dio_npz_path is not None else csv_path.with_name(f"{csv_path.stem}_DIO_cleaned.npz")
    if dio_guess.exists():
        dio = np.load(dio_guess, allow_pickle=True)
        rising_times  = dio["rising_times"]
        falling_times = dio["falling_times"]
        fs = dio["fs"] if "fs" in dio.files else None
        print(f"[Info] Loaded DIO edges: {len(rising_times)} rising | fs={fs} Hz" if fs else f"[Info] Loaded DIO edges: {len(rising_times)} rising")
    else:
        print("[Warning] DIO NPZ not found — proceeding without it.")
        rising_times = falling_times = np.array([])
        fs = None

    # ==== 5) Load optional task meta ====
    taskmeta_guess = Path(taskmeta_npz_path) if taskmeta_npz_path is not None else csv_path.with_name(f"{csv_path.stem}_taskmeta.npz")
    meta = {}
    if taskmeta_guess.exists():
        task_npz = np.load(taskmeta_guess, allow_pickle=True)
        print(f"[Info] Loaded task metadata: {taskmeta_guess.name}")
        meta = {k: task_npz[k].item() if getattr(task_npz[k], "ndim", 0) == 0 else task_npz[k] for k in task_npz.files}
    else:
        print("[Info] No taskmeta NPZ found — using defaults.")

    stimulus_duration = float(meta.get("stimulus_duration_s", 1.0))
    iti_duration      = float(meta.get("iti_duration_s", 0.5))
    n_trials_meta     = int(meta.get("n_trials", len(csv_on)))
    orientation_pairs = meta.get("orientation_pairs", np.array([[0, 90]]))
    right_seq         = meta.get("right_ori_seq", np.zeros(len(csv_on)))

    # ==== 6) Build trial windows (prefer DIO if present with alignment) ====
    tol_s = 0.02  # tolerance in seconds for CSV↔DIO matching

    if len(rising_times) and len(falling_times) and fs:
        # Align THIS CSV to its segment inside the long DIO stream
        trial_windows, keep_idx, diffs, start_j = select_trials_for_this_csv(
            csv_on_s=csv_on,
            csv_off_s=csv_off,                 # kept for potential future checks
            rising_samples=rising_times,
            falling_samples=falling_times,
            fs=float(fs),
            tol_s=tol_s
        )

        # Slice per-trial arrays to the matched trials only (length m)
        m = len(trial_windows)
        orientations = orientations[:m]

        # ------- Verification: print summary -------
        print(f"[Align] Matched CSV run at DIO index start: {int(start_j)}")
        print(f"[Align] Max |Δonset| = {float(np.max(diffs)):.4f}s | Mean = {float(np.mean(diffs)):.4f}s")
        n_bad = int(np.sum(diffs > tol_s))
        if n_bad:
            print(f"[Align][WARN] {n_bad} trials exceed tolerance ±{tol_s:.3f}s")

        # ------- Verification: quick plot saved to output_dir -------
        try:
            dio_rising_s = np.asarray(rising_times, float) / float(fs)
            csv_rel = csv_on - csv_on[0]
            dio_block = dio_rising_s[start_j:start_j + m]
            dio_rel = dio_block - dio_block[0]

            fig = plt.figure(figsize=(8, 6))

            # Overlay of relative onsets
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(csv_rel, marker='o', linestyle='-', label='CSV (rel)')
            ax1.plot(dio_rel, marker='x', linestyle='--', label='DIO (rel)')
            ax1.set_ylabel('Time (s, relative)')
            ax1.set_title('CSV vs DIO relative onsets')
            ax1.legend(loc='best')

            # Per-trial absolute diffs
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(diffs, marker='o')
            ax2.axhline(tol_s, linestyle='--')
            ax2.set_ylabel('|Δonset| (s)')
            ax2.set_xlabel('Trial # (matched)')
            ax2.set_title('Per-trial onset differences')

            output_dir.mkdir(parents=True, exist_ok=True)
            align_png = output_dir / "alignment_check.png"
            fig.tight_layout()
            fig.savefig(align_png, dpi=150)
            plt.close(fig)
            print(f"[Align] Saved verification plot to: {align_png}")
        except Exception as e:
            print(f"[Align] Plot skipped: {e}")

        alignment_meta = {
            "mode": "DIO_csv_subsequence",
            "tol_s": float(tol_s),
            "dio_start_index": int(start_j),
            "max_abs_diff_s": float(np.max(diffs)),
            "mean_abs_diff_s": float(np.mean(diffs)),
        }

    else:
        # Fallback: no DIO -> use CSV directly (seconds)
        trial_windows = [(float(csv_on[i]), float(csv_off[i])) for i in range(len(csv_on))]
        alignment_meta = {
            "mode": "csv_only",
            "tol_s": float(tol_s),
            "dio_start_index": None,
            "max_abs_diff_s": None,
            "mean_abs_diff_s": None,
        }

    # ==== 7) Combine into unified dict ====
    data = {
        "metadata": {
            "animal_id": animal_id,
            "session_id": session_id,
            "recording_folder": str(rec_folder),
            "csv_path": str(csv_path),
            "dio_npz_path": str(dio_guess) if 'dio_guess' in locals() and dio_guess.exists() else None,
            "taskmeta_npz_path": str(taskmeta_guess) if 'taskmeta_guess' in locals() and taskmeta_guess.exists() else None,
            "extraction_date": datetime.now().isoformat(),
            "fs": float(fs) if fs is not None else None,
            "n_trials": len(trial_windows),
            "source": "CSV+DIO" if len(rising_times) else "CSV",
            "alignment": alignment_meta,
        },
        "experiment_parameters": {
            "stimulus_duration_s": stimulus_duration,
            "iti_duration_s": iti_duration,
            "orientation_pairs": orientation_pairs.tolist() if isinstance(orientation_pairs, np.ndarray) else orientation_pairs,
        },
        "trial_info": {
            "trial_windows": trial_windows,                           # list of (start, end) in samples if DIO, else seconds
            "csv_iti": csv_iti.tolist(),
            "orientations": orientations.tolist(),                    # canonical per-trial orientation from CSV
            "left_ori_seq": orientations.tolist(),                    # kept for backward-compat
            "right_ori_seq": right_seq.tolist() if isinstance(right_seq, np.ndarray) else list(right_seq),
            "csv_df": df.to_dict("records"),
        },
        "dio_edges": {
            "rising_times": rising_times.tolist() if len(rising_times) else [],
            "falling_times": falling_times.tolist() if len(falling_times) else [],
        },
        "spike_data": {},
        "unit_info": {},
        "extraction_params": {},
    }

    # ==== 8) Attach spikes from Phy/SortingAnalyzer ====
    attach_phy_spikes(
        data=data,
        sorting_root=sorting_root,
        window_pre_s=0.2,
        window_post_s=2.0,
    )

    # ==== 9) Save pickle ====
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / f"{animal_id}_{session_id}_grating_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Success] Saved combined data (with spikes) to {pkl_path}")

    # ---- Quick summary ----
    print(f"- Trials: {data['metadata']['n_trials']}")
    print(f"- FS: {data['metadata'].get('fs')}")
    print(f"- Units attached: {data['extraction_params'].get('total_units', 0)}")
    # Optional: orientation distribution
    u, c = np.unique(np.array(data["trial_info"]["orientations"]), return_counts=True)
    print(f"- Orientation counts: {dict(zip(u.tolist(), c.tolist()))}")
