# ================================================================
# extract_grating_neural_data_explicit_sortout_pkl.py
# Behavior cloned from working extractor but with:
#   - explicit sortout_root path
#   - selectable orientation column (default "L_Orient")
#   - all units kept
#   - PKL-only save
# ================================================================

import os
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from spikeinterface import load_sorting_analyzer
from spikeinterface.extractors import PhySortingExtractor
from rf_recon.rf_grid.parse_grating_experiment import parse_grating_experiment

# ------------------------------------------------
# USER INPUTS
# ------------------------------------------------
rec_folder      = Path(r"Z:\xl_cl\Albert\20251031_psv;hf;sphr;2sides\CnL39SG\CnL39SG_20251031_085159.rec")
task_file_path  = Path(r"Z:\xl_cl\Albert\20251031_psv;hf;sphr;2sides\CnL39SG\CnL39_drifting_grating_exp_20251031_085247.txt")
dio_npz_path    = Path(r"Z:\xl_cl\Albert\20251031_psv;hf;sphr;2sides\CnL39SG\CnL39SG_20251031_085159_DIO_samples_segment01.npz")
sortout_root    = Path(r"Z:\xl_cl\code\sortout\CnL39SG\CnL39SG_20251031_085159")
output_pkl_path = Path(r"Z:\xl_cl\Albert\CnL39SG_grating_data.pkl")

# Analysis window parameters
window_pre  = 0.2   # seconds before onset
window_post = 2.0   # seconds after onset
default_fs  = 30000

# Choose which column in the task (TXT) to use for orientation/condition
# e.g., "L_Orient", "R_Orient", or any other available column name
orientation_column = "R_Orient"

# ------------------------------------------------
# HELPERS
# ------------------------------------------------
def load_dio_as_samples(dio_npz_path: Path, default_fs=30000):
    """Load DIO NPZ and return (rising_samples, falling_samples, fs) as int64 samples."""
    dio = np.load(dio_npz_path, allow_pickle=True)
    fs = int(dio["fs"]) if "fs" in dio.files else default_fs

    def as_samples(arr):
        arr = np.asarray(arr)
        if arr.dtype.kind in "iu" or (arr.size and np.nanmax(arr) > 1e6):
            return arr.astype(np.int64)
        return np.round(arr * fs).astype(np.int64)

    if "rising_samples" in dio.files:
        rising = dio["rising_samples"].astype(np.int64)
    elif "rising_times" in dio.files:
        rising = as_samples(dio["rising_times"])
    else:
        raise KeyError("Missing rising edges in DIO NPZ.")

    falling = None
    if "falling_samples" in dio.files:
        falling = dio["falling_samples"].astype(np.int64)
    elif "falling_times" in dio.files:
        falling = as_samples(dio["falling_times"])

    # Pair/align
    if falling is not None and falling.size and rising.size and falling[0] < rising[0]:
        falling = falling[1:]
    if falling is not None:
        m = min(rising.size, falling.size)
        rising, falling = rising[:m], falling[:m]

    return rising, falling, fs


def choose_orientation_column(df: pd.DataFrame, colname: str = "L_Orient") -> np.ndarray:
    """
    Return the numeric orientation/condition vector from a specified column name.
    Fallbacks: 'L_Orient' → any column containing 'orient' (case-insensitive) → zeros.
    """
    # 1) If the specified column exists, use it directly
    if colname in df.columns:
        return df[colname].to_numpy(dtype=float)

    # 2) Fall back to 'L_Orient' if present
    if "L_Orient" in df.columns:
        return df["L_Orient"].to_numpy(dtype=float)

    # 3) Try any other orientation-like column
    for c in df.columns:
        if "orient" in c.lower():
            return df[c].to_numpy(dtype=float)

    # 4) Nothing found → zero vector
    print(f"[WARN] Orientation column '{colname}' not found. Using zeros.")
    return np.zeros(len(df), dtype=float)


def save_to_pickle(data, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving PKL → {filepath}")
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# ------------------------------------------------
# MAIN EXTRACTION FUNCTION
# ------------------------------------------------
def extract_grating_neural_data_for_embedding(
    rec_folder: Path,
    task_file_path: Path,
    dio_npz_path: Path,
    sortout_root: Path,
    output_pkl_path: Path,
    window_pre: float = 0.2,
    window_post: float = 2.0,
    default_fs: int = 30000,
    orientation_column: str = "L_Orient",
):
    rec_folder     = Path(rec_folder)
    task_file_path = Path(task_file_path)
    dio_npz_path   = Path(dio_npz_path)
    sortout_root   = Path(sortout_root)

    rec_name   = rec_folder.name.replace(".rec", "")
    animal_id  = rec_name.split("_")[0]
    session_id = rec_name

    print(f"\n=== Extracting grating data for {animal_id}/{session_id} ===")

    # ---- Parse experiment TXT ----
    task_file = parse_grating_experiment(task_file_path)
    df = task_file["trial_data"]

    # timing from TXT (strings like '1.0s' or numeric)
    def as_sec(x): 
        s = str(x)
        return float(s[:-1]) if s.endswith("s") else float(s)

    stimulus_duration = as_sec(task_file["parameters"]["stimulus_duration"])
    iti_duration      = as_sec(task_file["parameters"]["iti_duration"])
    n_repeats         = int(task_file["parameters"]["total_trials"])
    trial_duration    = stimulus_duration + iti_duration

    print(f"Stimulus: {stimulus_duration}s | ITI: {iti_duration}s | Total trials (TXT): {n_repeats}")

    # ---- Load DIO ----
    rising_samples, falling_samples, fs_dio = load_dio_as_samples(dio_npz_path, default_fs)
    print(f"DIO loaded ({len(rising_samples)} rising edges, fs={fs_dio}Hz)")

    # ---- Labels / orientations ----
    orientations = choose_orientation_column(df, orientation_column)
    unique_orientations = np.unique(orientations)

    # ---- Trial windows from DIO samples ----
    if falling_samples is None:
        falling_samples = rising_samples + int(round(trial_duration * fs_dio))

    n_trials_txt = len(df)
    if len(rising_samples) < n_trials_txt or len(falling_samples) < n_trials_txt:
        print(f"⚠ Mismatch detected: Trials {n_trials_txt}, Rising {len(rising_samples)}, Falling {len(falling_samples)}")
        n_trials = min(n_trials_txt, len(rising_samples), len(falling_samples))
        df = df.iloc[:n_trials]
        orientations = orientations[:n_trials]
    else:
        n_trials = n_trials_txt

    trial_windows = [(int(rising_samples[i]), int(falling_samples[i])) for i in range(n_trials)]

    # ---- Build data structure (matches working code) ----
    neural_data = {
        "metadata": {
            "animal_id": animal_id,
            "session_id": session_id,
            "recording_folder": str(rec_folder),
            "task_file": str(task_file_path),
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
            "orientation_column": orientation_column,        # record which column was used
            "orientations": orientations.tolist(),
            "unique_orientations": unique_orientations.tolist(),
            "trial_windows": trial_windows,                  # (start_sample, end_sample)
            "all_trial_parameters": df.to_dict("records"),   # entire TXT row
        },
        "spike_data": {},
        "unit_info": {},
    }

    # ---- Iterate shanks and keep ALL units ----
    pre_samp  = int(round(window_pre  * fs_dio))
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
                print(f"No valid sorting found in {sort_dir}")
                continue

            unit_ids = sorting.unit_ids
            print(f"  Found {len(unit_ids)} units")

            # optional quality property
            try:
                qualities = sorting.get_property("quality")
            except Exception:
                qualities = ["unknown"] * len(unit_ids)
            if qualities is None or len(qualities) != len(unit_ids):
                qualities = ["unknown"] * len(unit_ids)

            for idx, unit_id in enumerate(unit_ids):
                quality = str(qualities[idx]) if idx < len(qualities) else "unknown"
                unique_unit_id = f"{shank_dir.name}_sorting{sort_dir.name}_unit{unit_id}"
                spikes = np.asarray(sorting.get_unit_spike_train(unit_id), dtype=np.int64)

                neural_data["unit_info"][unique_unit_id] = {
                    "original_unit_id": int(unit_id),
                    "shank": shank_dir.name.replace("shank", ""),
                    "quality": quality,
                    "sorting_folder": str(sort_dir),
                    "n_spikes_total": int(spikes.size),
                    "unit_index": total_units,
                }

                # Slice in SAMPLES; store relative spike times in seconds (negatives allowed)
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

    # ---- Final params ----
    neural_data["extraction_params"] = {
        "window_pre": float(window_pre),
        "window_post": float(window_post),
        "total_units": int(total_units),
        "fs_dio": int(fs_dio),
    }

    print(f"\n✅ Extraction complete: {total_units} units | {n_trials} trials")
    save_to_pickle(neural_data, output_pkl_path)
    return neural_data

# ------------------------------------------------
# RUN SCRIPT
# ------------------------------------------------
if __name__ == "__main__":
    extract_grating_neural_data_for_embedding(
        rec_folder=rec_folder,
        task_file_path=task_file_path,
        dio_npz_path=dio_npz_path,
        sortout_root=sortout_root,
        output_pkl_path=output_pkl_path,
        window_pre=window_pre,
        window_post=window_post,
        default_fs=default_fs,
        orientation_column=orientation_column,  # ← pick your column here
    )
