# ================================================================
# extract_grating_neural_data_from_csv_pkl_adapted.py
# Structure/behavior cloned from the "working" extractor, but reads
# a CSV with columns like: left_ori, right_ori, stim_on_s, stim_off_s
# Uses DIO rising edges (samples) for alignment; saves PKL only.
# ================================================================

import os
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor

# ---------------------------
# USER INPUTS
# ---------------------------
rec_folder      = Path(r"Z:\xl_cl\Albert\20251022_psv;hf;2grtings\CnL39SG\Cnl39SG_20251022_152514.rec")
csv_trial_table = Path(r"Z:\xl_cl\Albert\20251022_psv;hf;2grtings\CnL39SG\CnL39_45_135_two_grating_passive_static_20251022_154202.csv")
dio_npz_path    = Path(r"Z:\xl_cl\Albert\20251022_psv;hf;2grtings\CnL39SG\Cnl39SG_20251022_152514_DIO_samples_segment02.npz")
sortout_root    = Path(r"ZZ:\xl_cl\code\sortout\CnL39SG\CnL39SG_20251031_085159")
output_pkl_path = Path(r"Z:\xl_cl\Albert\20251022_psv;hf;2grtings\CnL39SG\CnL39SG_45_135_grating_data_2.pkl")

# Analysis window parameters
window_pre  = 0.2   # seconds before onset
window_post = 1.0   # seconds after onset
default_fs  = 30000

# Which side’s orientation to drive the primary orientation vector
orientation_side = "left"    # "left" or "right"

override_stimulus_duration_s = 1.0   # e.g., 2.0 Set to None if not using
override_iti_duration_s      = 0.5    # ← set your desired ITI here

# ---------------------------
# HELPERS
# ---------------------------
def load_dio_as_samples(dio_npz_path: Path, default_fs=30000):
    """Return rising/falling edges as int64 samples; tolerate *_times or *_samples."""
    dio = np.load(dio_npz_path, allow_pickle=True)
    fs = int(dio["fs"]) if "fs" in dio.files else default_fs

    def as_samples(arr):
        arr = np.asarray(arr)
        if arr.dtype.kind in "iu" or np.nanmax(arr) > 1e6:
            return arr.astype(np.int64)
        return np.round(arr * fs).astype(np.int64)

    if "rising_samples" in dio.files:
        rising = dio["rising_samples"].astype(np.int64)
    elif "rising_times" in dio.files:
        rising = as_samples(dio["rising_times"])
    else:
        raise KeyError("DIO NPZ missing rising edges.")

    falling = None
    if "falling_samples" in dio.files:
        falling = dio["falling_samples"].astype(np.int64)
    elif "falling_times" in dio.files:
        falling = as_samples(dio["falling_times"])

    if falling is not None and falling.size and rising.size and falling[0] < rising[0]:
        falling = falling[1:]
    if falling is not None:
        m = min(rising.size, falling.size)
        rising, falling = rising[:m], falling[:m]
    return rising, falling, fs

def pick_orientation(df: pd.DataFrame, side: str) -> np.ndarray:
    side = side.lower().strip()
    if side == "right":
        if "right_ori" in df.columns:
            return df["right_ori"].to_numpy(float)
        elif "right_orientation" in df.columns:
            return df["right_orientation"].to_numpy(float)
    # default to left
    if "left_ori" in df.columns:
        return df["left_ori"].to_numpy(float)
    elif "left_orientation" in df.columns:
        return df["left_orientation"].to_numpy(float)
    # last resort: a column that contains "ori"
    for c in df.columns:
        if "ori" in c.lower():
            return df[c].to_numpy(float)
    # nothing found → zeros
    return np.zeros(len(df), dtype=float)

def save_to_pickle(data, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving PKL → {filepath}")
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# ---------------------------
# MAIN
# ---------------------------
def extract_grating_neural_data_from_csv(
    rec_folder: Path,
    csv_trial_table: Path,
    dio_npz_path: Path,
    sortout_root: Path,
    output_pkl_path: Path,
    window_pre: float = 0.2,
    window_post: float = 2.0,
    default_fs: int = 30000,
    orientation_side: str = "left",
):
    rec_folder = Path(rec_folder)
    csv_trial_table = Path(csv_trial_table)
    dio_npz_path = Path(dio_npz_path)
    sortout_root = Path(sortout_root)

    rec_name = rec_folder.name.replace(".rec", "")
    animal_id = rec_name.split("_")[0]
    session_id = rec_name

    print(f"\n=== Extracting (CSV) for {animal_id}/{session_id} ===")
    df = pd.read_csv(csv_trial_table)

    # Orientation (primary) + keep full row metadata
    orientations = pick_orientation(df, orientation_side)
    unique_orientations = np.unique(orientations)

    # If your CSV doesn’t carry durations, we set defaults here
    # (you can add columns named 'stimulus_duration' and 'iti_duration' to override)
    stimulus_duration = float(df.get("stimulus_duration", pd.Series([2.0])).iloc[0])
    iti_duration      = float(df.get("iti_duration", pd.Series([1.0])).iloc[0])

    # 2) manual overrides (take precedence if not None)
    if override_stimulus_duration_s is not None:
        stimulus_duration = float(override_stimulus_duration_s)
    if override_iti_duration_s is not None:
        iti_duration = float(override_iti_duration_s)

    n_repeats         = int(df.shape[0])
    trial_duration    = stimulus_duration + iti_duration

    print(f"Stimulus: {stimulus_duration}s | ITI: {iti_duration}s | Trials (CSV rows): {n_repeats}")

    # DIO edges (samples), used for alignment (we ignore CSV on/off times for slicing)
    rising_samples, falling_samples, fs_dio = load_dio_as_samples(dio_npz_path, default_fs)
    if falling_samples is None:
        falling_samples = rising_samples + int(round(trial_duration * fs_dio))

    # Mismatch handling
    n_trials = len(df)
    if len(rising_samples) < n_trials or len(falling_samples) < n_trials:
        print(f"⚠ Mismatch: Trials {n_trials}, Rising {len(rising_samples)}, Falling {len(falling_samples)}")
        n_trials = min(n_trials, len(rising_samples), len(falling_samples))
        df = df.iloc[:n_trials]
        orientations = orientations[:n_trials]

    trial_windows = [(int(rising_samples[i]), int(falling_samples[i])) for i in range(n_trials)]

    # Data structure identical to the working extractor
    neural_data = {
        "metadata": {
            "animal_id": animal_id,
            "session_id": session_id,
            "recording_folder": str(rec_folder),
            "task_file": str(csv_trial_table),  # point to CSV
            "extraction_date": datetime.now().isoformat(),
            "n_trials": n_trials,
            "experiment_type": "grating",
        },
        "experiment_parameters": {
            "stimulus_duration": stimulus_duration,
            "iti_duration": iti_duration,
            "trial_duration": trial_duration,
            "total_trials": n_repeats,
        },
        "trial_info": {
            "orientations": orientations.tolist(),
            "unique_orientations": unique_orientations.tolist(),
            "trial_windows": trial_windows,                  # (start_sample, end_sample)
            "all_trial_parameters": df.to_dict("records"),   # keep entire CSV row
        },
        "spike_data": {},
        "unit_info": {},
    }

    # Extract spikes per trial (keep ALL units)
    pre_samp  = int(round(window_pre * fs_dio))
    post_samp = int(round(window_post * fs_dio))
    total_units = 0

    for shank_dir in sorted(sortout_root.glob("shank*")):
        print(f"\nProcessing {shank_dir.name}")
        sorting_paths = []
        for root, dirs, _ in os.walk(shank_dir):
            for d in dirs:
                if d.startswith("sorting_results_"):
                    sorting_paths.append(Path(root) / d)
        if not sorting_paths:
            print(f"No sorting results found in {shank_dir}")
            continue

        for sort_dir in sorting_paths:
            phy_dir = sort_dir / "phy"
            sa_dir  = sort_dir / "sorting_analyzer"
            sorting = None
            fs_spike = None

            if phy_dir.exists():
                print(f"Loading Phy sorting from {phy_dir}")
                sorting  = PhySortingExtractor(phy_dir)
                fs_spike = sorting.sampling_frequency
            elif sa_dir.exists():
                print(f"Loading SortingAnalyzer from {sa_dir}")
                sa = load_sorting_analyzer(sa_dir)
                sorting  = sa.sorting
                fs_spike = sorting.sampling_frequency
            else:
                print(f"No valid sorting in {sort_dir}")
                continue

            unit_ids = sorting.unit_ids
            print(f"  Found {len(unit_ids)} units")

            try:
                qualities = sorting.get_property("quality")
            except Exception:
                qualities = ["unknown"] * len(unit_ids)
            if qualities is None or len(qualities) != len(unit_ids):
                qualities = ["unknown"] * len(unit_ids)

            for idx, unit_id in enumerate(unit_ids):
                q = str(qualities[idx]) if idx < len(qualities) else "unknown"
                unique_unit_id = f"{shank_dir.name}_sorting{sort_dir.name}_unit{unit_id}"
                spikes = np.asarray(sorting.get_unit_spike_train(unit_id), dtype=np.int64)

                neural_data["unit_info"][unique_unit_id] = {
                    "original_unit_id": int(unit_id),
                    "shank": shank_dir.name.replace("shank", ""),
                    "quality": q,
                    "sorting_folder": str(sort_dir),
                    "n_spikes_total": len(spikes),
                    "unit_index": total_units,
                }

                # Slice [on−pre, on+post) in SAMPLES; store relative seconds (negatives included)
                trial_spike_data = []
                for t_idx, (on_samp, off_samp) in enumerate(trial_windows):
                    lo = on_samp - pre_samp
                    hi = on_samp + post_samp
                    rel = (spikes[(spikes >= lo) & (spikes < hi)] - on_samp) / fs_spike
                    trial_spike_data.append({
                        "trial_index": t_idx,
                        "orientation": float(orientations[t_idx]) if t_idx < len(orientations) else None,
                        "spike_times": rel.tolist(),
                        "spike_count": int(rel.size),
                        "trial_start": int(on_samp),
                        "trial_end": int(off_samp),
                    })
                neural_data["spike_data"][unique_unit_id] = trial_spike_data
                total_units += 1

    neural_data["extraction_params"] = {
        "window_pre": float(window_pre),
        "window_post": float(window_post),
        "total_units": int(total_units),
        "fs_dio": int(default_fs),
    }

    print(f"\n✅ Extraction complete: {total_units} units | {n_trials} trials")
    save_to_pickle(neural_data, output_pkl_path)
    return neural_data

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    extract_grating_neural_data_from_csv(
        rec_folder=rec_folder,
        csv_trial_table=csv_trial_table,
        dio_npz_path=dio_npz_path,
        sortout_root=sortout_root,
        output_pkl_path=output_pkl_path,
        window_pre=window_pre,
        window_post=window_post,
        default_fs=default_fs,
        orientation_side=orientation_side,
    )
