# ================================================================
# Unifies ephys .rec + PsychoPy CSV (+ optional *_DIO_cleaned.npz and *_taskmeta.npz)
# and attaches spikes from Phy or SortingAnalyzer results, producing a single .pkl
# ready for downstream analysis / embeddings.
# ================================================================

from pathlib import Path
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import pickle
import h5py  # only if you later want HDF5 I/O

# SpikeInterface
from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor


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
                # p is the sorting_analyzer dir
                sa = load_sorting_analyzer(p)
                sorter = sa.sorting
                sorter_fs = sorter.sampling_frequency
                print(f"[attach_phy_spikes] Loaded SortingAnalyzer at {p} | fs={sorter_fs} Hz | units={len(sorter.unit_ids)}")
        except Exception as e:
            print(f"[attach_phy_spikes] Skipping {p}: {e}")
            continue

        if data["metadata"].get("fs") in (None, "None", "") and sorter_fs is not None:
            data["metadata"]["fs"] = float(sorter_fs)
            fs = float(sorter_fs)

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

            if not have_samples and (fs is None or fs == 0):
                print(f"[attach_phy_spikes] Missing fs to align seconds; skipping {unique_id}")
                continue

            spike_times_s = spike_samples / float(fs if fs else sorter_fs)

            for t_idx in range(n_trials):
                start, end = trial_windows[t_idx]

                if have_samples:
                    start_samp = int(start)
                    pre_samp = int(window_pre_s * (fs if fs else sorter_fs))
                    post_samp = int(window_post_s * (fs if fs else sorter_fs))
                    mask = (spike_samples >= start_samp - pre_samp) & (spike_samples < start_samp + post_samp)
                    spikes_rel_s = (spike_samples[mask] - start_samp) / float(fs if fs else sorter_fs)
                else:
                    start_sec = float(start)
                    mask = (spike_times_s >= start_sec - window_pre_s) & (spike_times_s < start_sec + window_post_s)
                    spikes_rel_s = spike_times_s[mask] - start_sec

                # Orientation per trial (best-effort)
                left_seq = data["trial_info"].get("left_ori_seq", [])
                right_seq = data["trial_info"].get("right_ori_seq", [])
                ori_val = None
                if isinstance(left_seq, list) and t_idx < len(left_seq):
                    ori_val = left_seq[t_idx]
                elif isinstance(right_seq, list) and t_idx < len(right_seq):
                    ori_val = right_seq[t_idx]

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
    if not {"stim_on_s", "stim_off_s"}.issubset(df.columns):
        raise ValueError("CSV missing required columns 'stim_on_s' and 'stim_off_s'")

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
    left_seq  = meta.get("left_ori_seq", np.zeros(len(csv_on)))
    right_seq = meta.get("right_ori_seq", np.zeros(len(csv_on)))

    # ==== 6) Build trial windows (prefer DIO if present) ====
    if len(rising_times) and len(falling_times):
        n = min(len(csv_on), len(rising_times), len(falling_times))
        trial_windows = [(float(rising_times[i]), float(falling_times[i])) for i in range(n)]
    else:
        trial_windows = [(float(csv_on[i]), float(csv_off[i])) for i in range(len(csv_on))]

    # ==== 7) Combine into unified dict ====
    data = {
        "metadata": {
            "animal_id": animal_id,
            "session_id": session_id,
            "recording_folder": str(rec_folder),
            "csv_path": str(csv_path),
            "dio_npz_path": str(dio_guess) if dio_guess.exists() else None,
            "taskmeta_npz_path": str(taskmeta_guess) if taskmeta_guess.exists() else None,
            "extraction_date": datetime.now().isoformat(),
            "fs": float(fs) if fs is not None else None,
            "n_trials": len(trial_windows),
            "source": "CSV+DIO" if len(rising_times) else "CSV",
        },
        "experiment_parameters": {
            "stimulus_duration_s": stimulus_duration,
            "iti_duration_s": iti_duration,
            "orientation_pairs": orientation_pairs.tolist() if isinstance(orientation_pairs, np.ndarray) else orientation_pairs,
        },
        "trial_info": {
            "trial_windows": trial_windows,
            "csv_iti": csv_iti.tolist(),
            "left_ori_seq": left_seq.tolist() if isinstance(left_seq, np.ndarray) else list(left_seq),
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
